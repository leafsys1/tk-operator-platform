/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [CONFIG] jax_lax_scan Tiling 常量与结构体 —— kernel / host / pyapi 三方共用。
// 本文件只含纯 C/C++ 语法，不含 __aicore__ / __gm__ 等 ASC 关键字（与 elementwise_sub_tiling.h 同约定）。

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 注册 body（与 ops/jax_lax_scan/scripts/fixtures.py 的 BODIES 一一对应）
// 4/5/6 是“非法改变 carry”的拒绝用例，在 Python 层 trace 期拒绝，永不进入 kernel。
// ---------------------------------------------------------------------------
constexpr int32_t SCAN_BODY_ADD = 1;         // new = carry + x;            y = new
constexpr int32_t SCAN_BODY_PYTREE = 2;      // new = (c+1, a+x);           y = a+x
constexpr int32_t SCAN_BODY_QUAD = 3;        // new = 0.5*carry + x;        y = new*new
constexpr int32_t SCAN_BODY_PYTREE_XS = 7;   // new = carry + add*scale;    y = new
constexpr int32_t SCAN_BODY_XS_NONE = 8;     // new = carry + 1.0;          y = new

// ---------------------------------------------------------------------------
// dtype 码（逐叶子，不做全局推断：pytree carry 的 int32 counter 与 float32 accumulator 共存）
// ---------------------------------------------------------------------------
constexpr int32_t SCAN_DTYPE_F32 = 0;
constexpr int32_t SCAN_DTYPE_I32 = 1;

// ---------------------------------------------------------------------------
// 返回码（可执行文件 / CLI / .so / Python 异常词表共用）
// ---------------------------------------------------------------------------
constexpr int SCAN_RC_OK = 0;
constexpr int SCAN_RC_RUN_FAILED = 1;
constexpr int SCAN_RC_BAD_ARGS = 2;
constexpr int SCAN_RC_CARRY_SHAPE_MISMATCH = 3;
constexpr int SCAN_RC_UNSUPPORTED = 4;

// ---------------------------------------------------------------------------
// [CONFIG] UB 对齐粒度：AscendC 要求 UB buffer 32B 对齐（8 个 float）。
// carry 宽度 W 可能只有 4，必须向上取整到 32B 再 InitBuffer。
// ---------------------------------------------------------------------------
constexpr int64_t SCAN_UB_ALIGN_ELEMS = 8;
constexpr int64_t SCAN_MAX_WIDTH = 4096;

// ---------------------------------------------------------------------------
// 运行时参数：一次 launch 执行 [start, start+count) 这一段逻辑步。
//   整循环（compiled_whole_loop）：start=0, count=length，一次 kernel 完成 L 步；
//   逐步（stepwise）            ：start=t,  count=1，Python 层循环 L 次、carry 缓冲轮换。
// ys 的步轴长度恒为 length（逐步模式也要知道总长度才能算写回偏移）。
// ---------------------------------------------------------------------------
struct ScanTilingData {
    int32_t bodyId;
    int32_t reverse;      // 0 / 1
    int32_t nCarry;       // 1 或 2
    int32_t nXs;          // 0 / 1 / 2
    int64_t length;       // 总逻辑步数 L（ys 的步轴长度 / xs 前导轴）
    int64_t width;        // 每个浮点 carry 叶子的元素数 W
    int64_t start;        // 本次执行的第一个逻辑步 t
    int64_t count;        // 本次执行步数
};
