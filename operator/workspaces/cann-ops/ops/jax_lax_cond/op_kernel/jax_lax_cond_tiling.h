/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [CONFIG] jax_lax_cond Tiling 常量与结构体 —— kernel / host / pyapi 三方共用。
// 本文件只含纯 C/C++ 语法，不含 __aicore__ / __gm__ 等 ASC 关键字（与 scan/sub 的 tiling 头同约定）。

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 逐输出叶子的分支原语。branch_id 只决定“选哪一支”，真正作用在数据上的
// 是 outOp：同一 branch 的不同输出叶子可能用同一原语（例如 out_tree 的两叶）。
// ---------------------------------------------------------------------------
constexpr int32_t COND_OP_ADD1 = 1;            // y = x + 1
constexpr int32_t COND_OP_SUB1 = 2;            // y = x - 1
constexpr int32_t COND_OP_MUL2 = 3;            // y = x * 2
constexpr int32_t COND_OP_HALF = 4;            // y = x * 0.5（与 x/2 在二进制浮点下逐位等价）
constexpr int32_t COND_OP_AFFINE_TRUE = 5;     // y = x * 1.125 + 0.5
constexpr int32_t COND_OP_AFFINE_FALSE = 6;    // y = x * 0.875 - 0.5

// ---------------------------------------------------------------------------
// dtype 码（与 Python ctypes 表一致；本批数值叶子只有 float32）
// ---------------------------------------------------------------------------
constexpr int32_t COND_DTYPE_F32 = 0;

// ---------------------------------------------------------------------------
// 返回码（可执行文件 / CLI / .so / Python 异常词表共用）
// ---------------------------------------------------------------------------
constexpr int COND_RC_OK = 0;
constexpr int COND_RC_RUN_FAILED = 1;
constexpr int COND_RC_BAD_ARGS = 2;
constexpr int COND_RC_REJECT = 3;

// ---------------------------------------------------------------------------
// [CONFIG] 叶子数与 tile 上限
//   * 一次调用最多 8 个输入 / 8 个输出叶子（pytree 展开后）；
//   * tile = 8192 float（32KB）；UB 里 in/out 各一个 tile，共 64KB，远小于 192KB；
//   * stride > 1 的“跨步叶子”（x[:,0]）只在单核处理，numel 受 DataCopy blockCount
//     （4095）约束；host 侧超过该上限直接参数错误，不静默截断。
// ---------------------------------------------------------------------------
constexpr int32_t COND_MAX_LEAVES = 8;
constexpr int64_t COND_TILE_ELEMS = 8192;
constexpr int64_t COND_MAX_STRIDED_ELEMS = 4095;

// ---------------------------------------------------------------------------
// 运行时参数。inNumel 同时用于 GlobalTensor 的边界声明与 host 侧校验；
// outSrc/outOp/outStride 描述“每个输出叶子从哪个输入叶子、用哪条原语、按什么步长取数”。
// ---------------------------------------------------------------------------
struct CondTilingData {
    int32_t branchId;         // 1=true, 2=false（只用于证据写入）
    int32_t nInputs;
    int32_t nOutputs;
    int32_t outOp[8];         // COND_OP_*
    int32_t outSrc[8];        // 输入叶子下标
    int64_t outNumel[8];      // 输出叶子元素数
    int64_t outStride[8];     // 在源输入叶子内的元素步长（连续叶子 = 1）
    int64_t inNumel[8];       // 输入叶子元素数
};
