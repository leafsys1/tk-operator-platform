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
// jax_lax_associative_scan · Python 调用接口的 C ABI 结构体（libjax_lax_associative_scan_host.so）
// ============================================================================
// 与 ops/jax_lax_associative_scan/python/jax_lax_associative_scan.py 的 ctypes.Structure
// 逐字段对应；布局版本 1（456 字节）。字段可追加，不可改变已用字段语义。
//
// 入参不含 PyObject：combine 只在 Python 层解析成 combine_id，设备地址由 Python 层持有。
// ============================================================================

#pragma once

#include <cstdint>

constexpr int32_t JAX_LAX_ASSOC_SCAN_ABI_VERSION = 1;

typedef struct {
    int32_t combine_id;     /* 1=sum, 2=max, 3=tree_add */
    int32_t reverse;        /* 0/1 */
    int32_t n_leaves;       /* 1..8 */
    int32_t dtype[8];       /* 0=float32（逐叶子） */
    int64_t outer[8];       /* 逐叶子 outer = prod(shape[:axis]) */
    int64_t length[8];      /* 逐叶子 length = shape[axis] */
    int64_t inner[8];       /* 逐叶子 inner = prod(shape[axis+1:]) */
    void *in[8];            /* 设备输入地址，按叶子序 */
    void *out[8];           /* 设备输出地址，与输入分离 */
    int64_t numel[8];       /* 每叶子元素数，必须等于 outer*length*inner */
    int32_t device_id;
    char *err;              /* 调用方提供，>= 256 字节；失败写 "code: detail" */
    int32_t err_len;
    int32_t abi_version;    /* 恒为 JAX_LAX_ASSOC_SCAN_ABI_VERSION */
} jax_lax_associative_scan_args_t;

/* extern "C" 导出（pyapi.asc 实现）：
 *   int         jax_lax_associative_scan_run(const jax_lax_associative_scan_args_t *args);
 *   const char *jax_lax_associative_scan_abi_version(void);
 *   int         jax_lax_associative_scan_acl_init(void);
 *   int         jax_lax_associative_scan_malloc(void **dev_ptr, int64_t bytes);
 *   int         jax_lax_associative_scan_free(void *dev_ptr);
 *   int         jax_lax_associative_scan_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
 *   int         jax_lax_associative_scan_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
 */
