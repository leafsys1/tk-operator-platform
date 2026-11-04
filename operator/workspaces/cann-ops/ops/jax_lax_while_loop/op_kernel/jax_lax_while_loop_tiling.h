/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [CONFIG] jax_lax_while_loop Tiling 常量与结构体 —— kernel / host / pyapi 三方共用。
// 纯 C/C++ 语法，不含 __aicore__ / __gm__ 等 ASC 关键字（与 scan/cond/sub 的 tiling 头同约定）。
//
// 语义（严格对齐 JAX 0.8.1 lax.while_loop 的设备实现）：
//   * cond 在设备侧求值：`leaf < leaf|const` 或 `all(leaf > const)` 或恒真；
//   * body 在设备侧求值：每个 carry 叶子是 1~2 个线性项的仿射组合，项可以是逐元素
//     或整叶求和（用于 `a + mean(b)`）；常量项单独携带；
//   * 每次 host 入口只推进“一次 cond”或“一次 body + 一次 cond”（host_step 路由），
//     body_hits / cond_hits / limit_flag / token 全部写在设备内存，由 host D2H 读回。
//
// 计划来自 Python 侧对 body_fun/cond_fun 的 jaxpr 线性化（不是按 case 预置结果）；
// 全部数值在本 kernel 内按实际输入计算（编码规则 A1/A2/A5）。

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 叶子 dtype 码（与 python/jax_lax_while_loop.py 的 DTYPE 表一致）
// ---------------------------------------------------------------------------
constexpr int32_t WHILE_DTYPE_F32 = 0;
constexpr int32_t WHILE_DTYPE_I32 = 1;

// ---------------------------------------------------------------------------
// 条件算子码（与 python/jax_lax_while_loop.py 的 COND_* 一致）
// ---------------------------------------------------------------------------
constexpr int32_t WHILE_CMP_LT = 0;
constexpr int32_t WHILE_CMP_LE = 1;
constexpr int32_t WHILE_CMP_GT = 2;
constexpr int32_t WHILE_CMP_GE = 3;
constexpr int32_t WHILE_CMP_EQ = 4;
constexpr int32_t WHILE_CMP_NE = 5;
constexpr int32_t WHILE_CMP_ALL_GT = 6;
constexpr int32_t WHILE_CMP_ALL_GE = 7;
constexpr int32_t WHILE_CMP_ALL_LT = 8;
constexpr int32_t WHILE_CMP_ALL_LE = 9;
constexpr int32_t WHILE_CMP_ALWAYS_TRUE = 10;

// ---------------------------------------------------------------------------
// 执行阶段码
// ---------------------------------------------------------------------------
constexpr int32_t WHILE_STAGE_COND_ONLY = 0;   // 只求值一次 cond（不计 body）
constexpr int32_t WHILE_STAGE_BODY_COND = 1;   // 先执行一次 body，再求值 cond

// ---------------------------------------------------------------------------
// 证据缓冲区槽位（设备 int32[8]，seq_begin 时清零）
//   [0]=body_hits [1]=cond_hits [2]=limit_flag [3]=token [4]=cond_result
//   [5]=iterations(=body_hits) [6]=reserved [7]=reserved
// ---------------------------------------------------------------------------
constexpr int32_t WHILE_COUNTER_SLOTS = 8;
constexpr int32_t WHILE_SLOT_BODY_HITS = 0;
constexpr int32_t WHILE_SLOT_COND_HITS = 1;
constexpr int32_t WHILE_SLOT_LIMIT_FLAG = 2;
constexpr int32_t WHILE_SLOT_TOKEN = 3;
constexpr int32_t WHILE_SLOT_COND_RESULT = 4;
constexpr int32_t WHILE_SLOT_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// 上限
//   * 一次调用最多 8 个 carry 叶子（pytree 展开后）；
//   * tile = 8192 float（32KB）；UB 中 in1/in2/out 各一个 tile + 1KB 标量区，
//     合计约 97KB，远小于 192KB；
//   * 归约项（mean）只在单核把整叶搬进标量区后做标量求和，源叶子元素数受
//     WHILE_SCAN_MAX 限制；host 侧超限直接参数错误，不静默截断。
// ---------------------------------------------------------------------------
constexpr int32_t WHILE_MAX_LEAVES = 8;
constexpr int64_t WHILE_TILE_ELEMS = 8192;
constexpr int64_t WHILE_SCAN_MAX = 256;

// ---------------------------------------------------------------------------
// 运行时参数
// ---------------------------------------------------------------------------
struct WhileTilingData {
    int32_t stage;                 // WHILE_STAGE_*
    int32_t nLeaves;               // carry 叶子数（输入=输出）
    int32_t hasTangent;            // 1 = 同时计算切线（JVP）
    int32_t condOp;                // WHILE_CMP_*
    int32_t condLeaf;              // 条件左侧叶子下标
    int32_t condRhsLeaf;           // 条件右侧叶子下标；-1 表示常量
    int32_t condIsInt;             // 1 = int32 比较
    int32_t safetyLimit;           // 安全迭代上限
    uint32_t planSeed;             // plan 指纹低 32 位，用于 runtime_token
    float condConst;               // 条件右侧常量
    int64_t inNumel[WHILE_MAX_LEAVES];
    int64_t outNumel[WHILE_MAX_LEAVES];
    int32_t outDtype[WHILE_MAX_LEAVES];
    int32_t nTerms[WHILE_MAX_LEAVES];       // 1 或 2
    int32_t outSrc1[WHILE_MAX_LEAVES];      // 项 1 源叶子
    float outScale1[WHILE_MAX_LEAVES];      // 项 1 系数
    int32_t outRed1[WHILE_MAX_LEAVES];      // 项 1 是否整叶求和（1=求和后广播）
    int32_t outSrc2[WHILE_MAX_LEAVES];      // 项 2 源叶子（nTerms<2 时忽略）
    float outScale2[WHILE_MAX_LEAVES];
    int32_t outRed2[WHILE_MAX_LEAVES];
    float outConst[WHILE_MAX_LEAVES];       // 常量项
    int32_t tanOut[WHILE_MAX_LEAVES];       // 1 = 该叶子的切线也要计算
};
