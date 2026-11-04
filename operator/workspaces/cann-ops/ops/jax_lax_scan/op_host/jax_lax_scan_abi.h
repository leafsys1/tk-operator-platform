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
// jax_lax_scan · Python 调用接口的 C ABI 结构体（libjax_lax_scan_host.so）
// ============================================================================
// 与 ops/jax_lax_scan/python/jax_lax_scan.py 的 ctypes.Structure 逐字段对应；
// 布局基线见 0-黑盒测试设计.md §1.3.3，末尾 4 个字段是 ABI v2 的**追加**字段
// （不改变 v1 既有字段语义；Python 侧断言语义版本 = 2）。
//
// 入参不含 PyObject：callable 只在 Python 层解析成 body_id，设备地址由 Python 层持有。
// ============================================================================

#pragma once

#include <cstdint>

constexpr int32_t JAX_LAX_SCAN_ABI_VERSION = 2;

typedef struct {
    /* ---- v1 字段（设计 §1.3.3，语义不可变）---- */
    int32_t body_id;        /* 注册 body（1/2/3/7/8 进 kernel；4/5/6 在 Python 层拒绝） */
    int32_t reverse;        /* 0/1 */
    int64_t length;         /* 静态步数 */
    int32_t n_carry;        /* carry 叶子数（1 或 2） */
    int32_t n_xs;           /* xs 叶子数（0/1/2） */
    int32_t carry_dtype[8]; /* 0=float32, 1=int32；逐叶子 */
    int32_t xs_dtype[8];
    void *carry_in[8];      /* 设备输入地址，按叶子序 */
    void *carry_out[8];     /* 设备输出地址，与 carry_in 分离 */
    int64_t carry_numel[8]; /* 每叶子元素数 */
    void *xs[8];            /* 设备 xs 地址；xs=None 时 n_xs=0 */
    int64_t xs_numel[8];
    void *ys_out;           /* 设备地址，shape=(length,)+y.shape */
    int64_t ys_numel;
    int32_t ys_dtype;       /* 本批 0=float32 */
    int32_t device_id;
    void *err;              /* 调用方提供，>= 256 字节；失败写 "code: detail" */
    int32_t err_len;

    /* ---- v2 追加字段（ABI 版本标记 + 逐段执行区间）---- */
    int32_t abi_version;    /* 恒为 JAX_LAX_SCAN_ABI_VERSION */
    int64_t total_length;   /* ys 的步轴长度；<=0 时视为等于 length */
    int64_t step_start;     /* 本次 launch 的第一个逻辑步 t（整循环 = 0，逐步 = t） */
    int64_t step_count;     /* 本次 launch 的步数（整循环 = length，逐步 = 1）；<=0 时视为 length */
} jax_lax_scan_args_t;

/* extern "C" 导出（pyapi.asc 实现）：
 *   int         jax_lax_scan_run(const jax_lax_scan_args_t *args);
 *   const char *jax_lax_scan_abi_version(void);
 *   int         jax_lax_scan_acl_init(void);
 *   int         jax_lax_scan_malloc(void **dev_ptr, int64_t bytes);
 *   int         jax_lax_scan_free(void *dev_ptr);
 *   int         jax_lax_scan_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
 *   int         jax_lax_scan_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
 */
