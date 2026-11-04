/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// bid_elementwise_sub_20261101 · Tiling 常量与结构体（host / kernel 共用）
// ============================================================================
// 本文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 Ascend C 关键字，可被 host 与 kernel 共用。
// 分核采用「常规分块 + 单缓冲」（DOUBLE_BUFFER=1），tile 长度按 N 运行时二选一（N<=65536 用 512，否则 8192；不做 opMode/alpha/Core-Shrink 分支）。
// ============================================================================

#pragma once

#include <cstdint>

// [CONFIG] tile 长度运行时二选一（元素）：N <= SMALL_TILE_MAX_LENGTH(65536) 用小 tile 512，
//   否则用大 tile 8192；两者均 32B 对齐（% 8 == 0），GM 偏移恒 32B 对齐。
//   TILE_LENGTH(8192) 同时作为 UB buffer 分配上限：单块字节 = 8192 * 4 = 32768 <= 65535
//   （DataCopyPad 非 Ext blockLen 为 uint16，不越界）。
constexpr uint32_t TILE_LENGTH = 8192;

// [CONFIG] 小 tile 长度（元素）：N <= SMALL_TILE_MAX_LENGTH 时使用，512 % 8 == 0，GM 偏移恒 32B 对齐。
constexpr uint32_t SMALL_TILE_LENGTH = 512;

// [CONFIG] 选小 tile 的 N 上界（元素）：N <= 65536 用 SMALL_TILE_LENGTH，否则回落到 TILE_LENGTH。
constexpr uint64_t SMALL_TILE_MAX_LENGTH = 65536;

// [CONFIG] 32B 对齐粒度（float32）：8 元素 = 32 字节。
constexpr uint32_t VEC_ALIGN_LENGTH = 8;

// [CONFIG] 单缓冲：搬运/计算三队列各只开 1 个 buffer。
constexpr uint32_t DOUBLE_BUFFER = 1;

// [CONFIG] 910B1（dav-2201）AIV 逻辑核静态参考；运行时以 aclrtGetDeviceInfo 返回为准，无效时以此兜底。
constexpr uint32_t DEFAULT_CORE_NUM = 48;

// [CONFIG] 规格 v3 的 N 上下界（与 cases/bid_elementwise_sub_20261101/cases.json 一致）。
constexpr int64_t N_MIN = 1;
constexpr int64_t N_MAX = 1048576;

// 分核模型（常规分块单缓冲）：
//   核 idx 处理区间 [idx*numPerCore, idx*numPerCore + total)，
//     total = (idx < blockNum-1) ? numPerCore : tailNumLastCore；
//   块内第 i 个 tile 的 GM 偏移 = i * tileLength。
struct BidSubTilingData {
    uint32_t blockNum;        // 实际使用的 block 数（恒 <= 物理核数）
    uint32_t tileLength;      // 运行时给定：N <= SMALL_TILE_MAX_LENGTH(65536) 用 SMALL_TILE_LENGTH(512)，否则 TILE_LENGTH(8192)
    uint64_t totalLength;     // N
    uint64_t numPerCore;      // 非尾核处理元素数 = tilesPerCore * tileLength
    uint64_t tailNumLastCore; // 尾核处理元素数 = N - numPerCore*(blockNum-1) ∈ [1, numPerCore]
};

// 分核逻辑（可执行文件与共享库共用同一份，避免两份漂移）：
//   tileLength     = (N <= SMALL_TILE_MAX_LENGTH) ? SMALL_TILE_LENGTH : TILE_LENGTH   // 运行时二选一
//   totalTiles    = ceil(N / tileLength)
//   tilesPerCore  = ceil(totalTiles / coreNum)
//   blockNum      = ceil(totalTiles / tilesPerCore)   → 恒 <= coreNum
//   numPerCore    = tilesPerCore * tileLength          // = ceil(ceil(N/tileLength)/coreNum) * tileLength
//   tailNumLastCore = N - numPerCore*(blockNum-1)     → 恒 ∈ [1, numPerCore]
static inline void BuildTiling(uint64_t totalLength, uint32_t coreNum, BidSubTilingData *tiling)
{
    if (coreNum == 0) {
        coreNum = DEFAULT_CORE_NUM;
    }
    if (totalLength == 0) {
        totalLength = 1;  // 调用方已校验 N∈[1,N_MAX]；此处仅防御误用
    }

    uint32_t tileLen = (totalLength <= SMALL_TILE_MAX_LENGTH) ? SMALL_TILE_LENGTH : TILE_LENGTH;
    uint64_t totalTiles = (totalLength + tileLen - 1) / tileLen;
    uint64_t tilesPerCore = (totalTiles + coreNum - 1) / coreNum;
    if (tilesPerCore == 0) {
        tilesPerCore = 1;
    }
    uint64_t blockNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    if (blockNum == 0) {
        blockNum = 1;
    }
    uint64_t numPerCore = tilesPerCore * tileLen;
    uint64_t tailNumLastCore = totalLength - numPerCore * (blockNum - 1);

    tiling->blockNum = static_cast<uint32_t>(blockNum);
    tiling->tileLength = tileLen;
    tiling->totalLength = totalLength;
    tiling->numPerCore = numPerCore;
    tiling->tailNumLastCore = tailNumLastCore;
}