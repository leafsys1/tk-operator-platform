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
// jax_lax_while_loop · Python 调用接口的 C ABI（libjax_lax_while_loop_host.so）
// ============================================================================
// 与 python/jax_lax_while_loop.py 的 ctypes.Structure 逐字段对应。
//
// 分工（编码规则 A1/A2）：
//   * Python 层负责 pytree 结构校验、cond/body 的 jaxpr 线性化（生成计算计划）、
//     循环控制（host_step 路由）、弱类型还原与证据封装；
//   * 本层只做参数校验 + Tiling + kernel launch + 证据 D2H，**不做数值计算**；
//   * 全部数值（body、cond、JVP 切向量）由 op_kernel/jax_lax_while_loop_kernel.asc
//     在 NPU 上按实际 carry 计算。
//
// host_step 路由：一次 step = 一次 cond 求值（stage=0）或一次 body + 一次 cond
// （stage=1）。设备计数（body_hits/cond_hits/limit_flag/token/cond_result）由 kernel
// 写在设备内存，本层在 stream 同步后 D2H 读回；`abi_calls` 由本层按实际 launch 次数
// 累加并回传（原始整数，不是测试侧推算）。
//
// 返回码：0 成功 / 1 运行失败 / 2 参数错误；拒绝（3）在 Python 层于任何 device
// 操作之前或设备安全阀处产生，本层不产生 3。
// ============================================================================

#pragma once

#include <cstdint>

constexpr int32_t JAX_LAX_WHILE_LOOP_ABI_VERSION = 1;
constexpr int32_t JAX_LAX_WHILE_LOOP_MAX_LEAVES = 8;
constexpr int32_t JAX_LAX_WHILE_LOOP_COUNTER_SLOTS = 8;

typedef struct {
    /* ---- 形状与执行阶段 ---- */
    int32_t n_leaves;           /* 1..8，carry 叶子数（pytree 展开后） */
    int32_t has_tangent;        /* 1 = JVP，同时计算切线 */
    int32_t device_id;          /* 容器内逻辑设备号（本轮 0） */
    int32_t abi_version;        /* 恒为 JAX_LAX_WHILE_LOOP_ABI_VERSION */
    int32_t stage;              /* 0 = cond-only，1 = body+cond */

    /* ---- carry 叶子（按 JAX tree flatten 顺序）---- */
    void *in[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int64_t in_numel[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t dtype[JAX_LAX_WHILE_LOOP_MAX_LEAVES];   /* 0=float32, 1=int32 */
    void *out[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int64_t out_numel[JAX_LAX_WHILE_LOOP_MAX_LEAVES];

    /* ---- JVP 切向量（float32 叶子；int32 叶为 nullptr → 输出 float0）---- */
    void *tan_in[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    void *tan_out[JAX_LAX_WHILE_LOOP_MAX_LEAVES];

    /* ---- body 计划：out = const + Σ scale_i * (red_i ? sum(src_i) : src_i) ---- */
    int32_t out_dtype[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t n_terms[JAX_LAX_WHILE_LOOP_MAX_LEAVES];  /* 1 或 2 */
    int32_t out_src1[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    float out_scale1[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t out_red1[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t out_src2[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    float out_scale2[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t out_red2[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    float out_const[JAX_LAX_WHILE_LOOP_MAX_LEAVES];
    int32_t tan_out_flag[JAX_LAX_WHILE_LOOP_MAX_LEAVES];

    /* ---- cond 计划 ---- */
    int32_t cond_op;            /* WHILE_CMP_*，见 op_kernel/jax_lax_while_loop_tiling.h */
    int32_t cond_leaf;
    int32_t cond_rhs_leaf;      /* -1 → 使用 cond_const */
    int32_t cond_is_int;
    float cond_const;
    int32_t safety_limit;
    uint32_t plan_seed;

    /* ---- 序列句柄与证据回读 ---- */
    void *seq;                  /* seq_begin 返回；每次 host_step 调用的计数载体 */
    int32_t *counters_host;     /* 调用方提供 int32[8]；step 后由本层填入 */
    void *err;                  /* 调用方提供 >= 256 字节；失败写 "code: detail" */
    int32_t err_len;
} jax_lax_while_loop_args_t;

/* extern "C" 导出（pyapi.asc 实现）：
 *   int jax_lax_while_loop_seq_begin(void **handle, int32_t device_id);
 *   int jax_lax_while_loop_step(const jax_lax_while_loop_args_t *args);
 *   int jax_lax_while_loop_seq_end(void *handle, int32_t *counters_out, int64_t *abi_calls_out);
 *   int jax_lax_while_loop_seq_free(void *handle);
 *   const char *jax_lax_while_loop_abi_version(void);
 *   int jax_lax_while_loop_acl_init(void);
 *   int jax_lax_while_loop_malloc(void **dev_ptr, int64_t bytes);
 *   int jax_lax_while_loop_free(void *dev_ptr);
 *   int jax_lax_while_loop_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
 *   int jax_lax_while_loop_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
 */
