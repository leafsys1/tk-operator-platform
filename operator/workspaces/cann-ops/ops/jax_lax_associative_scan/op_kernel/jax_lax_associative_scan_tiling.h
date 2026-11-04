/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [CONFIG] jax_lax_associative_scan Tiling 常量与结构体 —— kernel / host / pyapi 三方共用。
// 本文件只含纯 C/C++ 语法，不含 __aicore__ / __gm__ 等 ASC 关键字（与 jax_lax_scan_tiling.h 同约定）。

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 注册 combine（与 ops/jax_lax_associative_scan/scripts/fixtures.py 的 COMBINE_IDS 一一对应）
// ---------------------------------------------------------------------------
constexpr int32_t ASSOC_COMBINE_SUM = 1;       // a + b
constexpr int32_t ASSOC_COMBINE_MAX = 2;       // maximum(a, b)
constexpr int32_t ASSOC_COMBINE_TREE_ADD = 3;  // pytree 逐叶子 a + b

// ---------------------------------------------------------------------------
// 返回码（可执行文件 / CLI / .so / Python 异常词表共用）
// ---------------------------------------------------------------------------
constexpr int ASSOC_RC_OK = 0;
constexpr int ASSOC_RC_RUN_FAILED = 1;
constexpr int ASSOC_RC_BAD_ARGS = 2;
constexpr int ASSOC_RC_REJECT = 3;       // 保留：拒绝在 Python 预校验期产生，不进入 .so
constexpr int ASSOC_RC_UNSUPPORTED = 4;  // 保留：AD 不支持分类在 Python trace 路径产生

// ---------------------------------------------------------------------------
// 几何：任一叶子按 axis 展成 [outer, length, inner] 后，扫描是 outer*inner 条独立链。
//   outer = prod(shape[:axis])、length = shape[axis]、inner = prod(shape[axis+1:])
// ---------------------------------------------------------------------------
constexpr int32_t ASSOC_MAX_LEAVES = 8;
// [CONFIG] 单次向量迭代覆盖的 inner 元素上限；更大的 inner 按 chunk 串行切分，
// 因此 inner 不设硬上限（UB 只按该上限分配：3 × 8KB = 24KB）。
constexpr int64_t ASSOC_MAX_INNER_CHUNK = 2048;
// [CONFIG] UB 对齐粒度：AscendC 要求 UB buffer 32B 对齐（8 个 float）。
constexpr int64_t ASSOC_UB_ALIGN_ELEMS = 8;

struct AssocTilingData {
    int32_t combineId;   // 1/2/3
    int32_t reverse;     // 0/1
    int32_t nLeaves;     // 1..8（whole_tree 一次 launch 覆盖全部叶子）
    int32_t reserved;    // 对齐占位，必须写 0
    int64_t outer[ASSOC_MAX_LEAVES];
    int64_t length[ASSOC_MAX_LEAVES];
    int64_t inner[ASSOC_MAX_LEAVES];
    int64_t maxInner;    // 本次 launch 所有叶子的 inner 最大值（UB 分配依据）
};
