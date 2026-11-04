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
// jax_lax_cond · Python 调用接口的 C ABI 结构体（libjax_lax_cond_host.so）
// ============================================================================
// 与 ops/jax_lax_cond/python/jax_lax_cond.py 的 ctypes.Structure 逐字段对应。
// 0-黑盒测试设计.md §1.3.4 只把该结构定为“非绑定基线”，允许节点 2 按验证结果调整；
// 调整点（只追加字段，不改已有语义）：
//   1) 增加 out_src / out_op / out_stride，用来表达“一个输出叶子从哪个输入叶子、
//      用哪条已注册原语、按什么元素步长取数”。没有这组字段就无法在不做任何
//      host 侧数据重排的前提下表达 c06 的 aux = x[:,0] ± 1（跨步叶子）。
//   2) token / counters 是 **host 侧** int32 缓冲区；kernel 写设备内存后由本库
//      aclrtMemcpy(D2H) 读回填入。证据仍由 kernel 产生，Python 不自报。
//   3) 入参不含 PyObject：callable 在 Python 层解析成 branch_id，设备地址由 Python 持有。
//
// 返回码：0 成功 / 1 运行失败 / 2 参数错误 / 3 拒绝（本层不产生 3，拒绝在 Python 层
// 于任何 device 操作前完成）。
// ============================================================================

#pragma once

#include <cstdint>

constexpr int32_t JAX_LAX_COND_ABI_VERSION = 1;

typedef struct {
    /* ---- 分支与叶子形态 ---- */
    int32_t branch_id;          /* 1=true, 2=false；由 Python 层的 pred 决定 */
    int32_t n_outputs;          /* 1..8 */
    int32_t n_inputs;           /* 1..8 */
    int32_t dtype[8];           /* 逐输入叶子：0=float32 */

    /* ---- 输入叶子（设备地址，按 pytree 稳定顺序）---- */
    int64_t in_numel[8];
    void *in[8];

    /* ---- 输出叶子 ---- */
    void *out[8];               /* 设备地址，与输入分离 */
    int64_t out_numel[8];
    int32_t out_src[8];         /* 该输出叶子取自哪个输入叶子 */
    int32_t out_op[8];          /* COND_OP_*，见 op_kernel/jax_lax_cond_tiling.h */
    int64_t out_stride[8];      /* 源输入叶子内的元素步长；连续为 1 */

    /* ---- 运行参数与证据回读（host 侧缓冲区）---- */
    int32_t device_id;
    void *token;                /* host int32*：kernel 写入 1/2，本库 D2H 读回 */
    void *counters;             /* host int32[2]：kernel 写 [true,false] 执行计数 */
    void *err;                  /* 调用方提供 >= 256 字节；失败写 "code: detail" */
    int32_t err_len;
    int32_t abi_version;        /* 恒为 JAX_LAX_COND_ABI_VERSION */
} jax_lax_cond_args_t;

/* extern "C" 导出（pyapi.asc 实现）：
 *   int         jax_lax_cond_run(const jax_lax_cond_args_t *args);
 *   const char *jax_lax_cond_abi_version(void);
 *   int         jax_lax_cond_acl_init(void);
 *   int         jax_lax_cond_malloc(void **dev_ptr, int64_t bytes);
 *   int         jax_lax_cond_free(void *dev_ptr);
 *   int         jax_lax_cond_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
 *   int         jax_lax_cond_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
 */
