# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop 白盒场景（节点 3）。

白盒用例的可执行语义：每个 scenario 提供 ``(init, cond, body, tangent)`` 与树描述。
用例清单/分支映射/期望计数在 ``tests/jax_lax_while_loop_whitebox_cases.json``；
本模块只放**函数**（cond/body 无法写进 JSON），供 pytest 主进程调度、worker（NPU）
与 golden 子进程（CPU JAX 0.8.1）共同 import，保证两侧执行同一份语义。

约定：
* ``init``/``tangent`` 是 numpy pytree（确定性、无 RNG），由 :func:`tree_from_value`
  生成稳定叶子路径；候选与 golden 各自把同一份数组重建为 JAX pytree。
* ``make(jax, jnp, variant)`` 内不读文件、不碰设备、不修改全局状态。
* 所有 shape/dtype 都在源码支持范围内（float32 / int32；归约源与 ``all`` 叶子
  numel ≤ WHILE_SCAN_MAX=256；pytree 叶子数 ≤ WHILE_MAX_LEAVES=8）。
"""

from __future__ import annotations

import numpy as np

#: 稳定叶子路径描述：leaf/tuple/dict，与 fixtures.tree_spec 同一结构。
def _leaf(path: str, shape, dtype: str) -> dict:
    return {"kind": "leaf", "path": path, "shape": [int(x) for x in shape], "dtype": dtype}


def _tuple(children: list[dict]) -> dict:
    return {"kind": "tuple", "children": children}


def tree_from_value(value, path: str = "carry") -> dict:
    """把 numpy pytree 转成稳定路径的树描述（tuple 用序号、dict 用 key）。"""
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        return {"kind": "dict", "keys": keys,
                "children": {str(key): tree_from_value(value[key], f"{path}/{key}")
                             for key in value}}
    if isinstance(value, (tuple, list)):
        return _tuple([tree_from_value(child, f"{path}/{index}")
                       for index, child in enumerate(value)])
    arr = np.asarray(value)
    dtype = "float0" if "float0" in str(arr.dtype) else str(arr.dtype)
    return _leaf(path, arr.shape, dtype)


def _f32(*shape, low=-1.0, high=1.0) -> np.ndarray:
    count = int(np.prod(shape)) if shape else 1
    return np.linspace(low, high, count, dtype=np.float32).reshape(shape)


def _i32(*shape) -> np.ndarray:
    count = int(np.prod(shape)) if shape else 1
    return np.arange(count, dtype=np.int32).reshape(shape)


def _float0():
    return np.zeros((), dtype=np.dtype([("float0", "V")]))


def _tangent_tree(init, tangent) -> dict | None:
    if tangent is None:
        return None
    return tree_from_value(tangent, "tangent")


# ---------------------------------------------------------------------------
# 每个 scenario：init(tuple of np arrays) + cond/body + 可选 tangent
# ---------------------------------------------------------------------------
def _scn_tile(width: int):
    def make(jax, jnp, variant):
        init = (np.int32(0), _f32(width))
        return init, (lambda c: c[0] < 2), (lambda c: (c[0] + 1, c[1] * 0.5 + 0.25)), None
    return make


def _make_two_elem(jax, jnp, variant):
    init = (np.int32(0), _f32(8, 4), _f32(8, 4))
    return init, (lambda c: c[0] < 1), (
        lambda c: (c[0] + 1, c[1] * 0.5 + c[2] * 0.25 + 0.125, c[2] * 0.75)), None


def _make_reduce_only(jax, jnp, variant):
    init = (np.int32(0), _f32(8), np.float32(0.0))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] * 0.5, jnp.sum(c[1]) * 0.001 + 0.5)), None


def _make_reduce_two(jax, jnp, variant):
    init = (np.int32(0), _f32(8), _f32(8), np.float32(0.0))
    return init, (lambda c: c[0] < 1), (
        lambda c: (c[0] + 1, c[1] * 0.5, c[2] * 0.25,
                   jnp.sum(c[1]) * 0.001 + jnp.sum(c[2]) * 0.002 + 0.5)), None


def _make_reduce_256(jax, jnp, variant):
    init = (np.int32(0), _f32(256), np.float32(0.0))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] * 0.5, jnp.sum(c[1]) * 0.001)), None


def _make_reduce_257_reject(jax, jnp, variant):
    """归约源 numel=257：host ABI H05 的 ``> 256`` 拒绝侧，不进入 kernel。"""
    init = (np.int32(0), _f32(257), np.float32(0.0))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] * 0.5, jnp.sum(c[1]) * 0.001)), None


def _make_int_sum(jax, jnp, variant):
    init = (np.int32(0), np.arange(1, 9, dtype=np.int32), np.int32(0))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] + 1, jnp.sum(c[1]))), None


def _make_int_two_reduce(jax, jnp, variant):
    init = (np.int32(0), np.arange(1, 9, dtype=np.int32),
            np.arange(2, 10, dtype=np.int32), np.int32(0))
    return init, (lambda c: c[0] < 1), (
        lambda c: (c[0] + 1, c[1] + 1, c[2] + 1, jnp.sum(c[1]) + jnp.sum(c[2]))), None


def _make_int_neg(jax, jnp, variant):
    init = (np.int32(0), np.int32(0))
    return init, (lambda c: c[0] < 3), (lambda c: (c[0] + 1, c[1] * -1 - 5)), None


def _make_int_two_elem(jax, jnp, variant):
    """int32 叶 = 2*a + 3*b：命中 ComputeIntLeaf 的**第二个逐元素源**（outRed2=0）。

    该叶 n_terms=2、n_elem=2、n_red=0；b+1 叶命中源 1 逐元素。预计触发已定位的
    int32 张量截断算子缺陷，保留 FAIL 并归属，不改 golden/阈值。
    """
    init = (np.int32(0), np.arange(1, 9, dtype=np.int32),
            np.arange(2, 10, dtype=np.int32))
    return init, (lambda c: c[0] < 1), (
        lambda c: (c[0] + 1, c[1] * 2 + c[2] * 3, c[2] + 1)), None


def _make_int_const(jax, jnp, variant):
    """int32 叶输出纯常量 7：计划 n_terms=0，命中 ComputeIntLeaf 的 const 子路径。"""
    init = (np.int32(0), np.int32(0))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, jnp.int32(7))), None


def _make_eight(jax, jnp, variant):
    init = tuple(np.int32(i) for i in range(8))
    return init, (lambda c: c[0] < 2), (lambda c: tuple(x + 1 for x in c)), None


_COND_INT = {
    "le": (np.int32(0), lambda c: c[0] <= 3, lambda c: (c[0] + 1,)),
    # 直接命中 WHILE_CMP_GT / WHILE_CMP_GE（而不是经 const>leaf 交换成 lt/le）。
    # 初值 4、body 递减，保证 true 至少一次并自然终止（3 / 4 次）。
    "gt": (np.int32(4), lambda c: c[0] > 1, lambda c: (c[0] - 1,)),
    "ge": (np.int32(4), lambda c: c[0] >= 1, lambda c: (c[0] - 1,)),
    "gt_swap": (np.int32(0), lambda c: np.int32(3) > c[0], lambda c: (c[0] + 1,)),
    "ge_swap": (np.int32(0), lambda c: np.int32(3) >= c[0], lambda c: (c[0] + 1,)),
    "eq": (np.int32(3), lambda c: c[0] == 3, lambda c: (c[0] + 1,)),
    "ne": (np.int32(0), lambda c: c[0] != 3, lambda c: (c[0] + 1,)),
}


def _make_cond_int(jax, jnp, variant):
    if variant == "rhs_leaf":
        # int leaf vs leaf：命中 condRhsLeaf>=0 的 ReadScalarInt(rhs) 侧。
        # 0<3 起，body 只递增左叶，3 次后 3<3 为假并自然终止。
        return ((np.int32(0), np.int32(3)),
                (lambda c: c[0] < c[1]),
                (lambda c: (c[0] + 1, c[1])), None)
    start, cond, body = _COND_INT[variant]
    return (start,), cond, body, None


#: 每个 variant 用与比较方向一致的初值/步进，保证命中 body 路径（0/1/多次都有）
_COND_FLOAT = {
    "le": ((np.float32(0.0625),), lambda c: c[1] <= 0.5,
           lambda c: (c[0] + 1, c[1] * 2.0)),
    # leaf 在左、常量在右：直接命中 WHILE_CMP_LT（gt_swap 会被规范化为 gt）。
    "lt": ((np.float32(0.0625),), lambda c: c[1] < 0.5,
           lambda c: (c[0] + 1, c[1] * 2.0)),
    "gt": ((np.float32(4.0),), lambda c: c[1] > 0.5,
           lambda c: (c[0] + 1, c[1] * 0.5)),
    "ge": ((np.float32(4.0),), lambda c: c[1] >= 0.5,
           lambda c: (c[0] + 1, c[1] * 0.5)),
    "eq": ((np.float32(1.0),), lambda c: c[1] == 1.0,
           lambda c: (c[0] + 1, c[1] * 0.5)),
    "ne": ((np.float32(4.0),), lambda c: c[1] != 0.25,
           lambda c: (c[0] + 1, c[1] * 0.5)),
    "gt_swap": ((np.float32(4.0),), lambda c: np.float32(0.5) < c[1],
                lambda c: (c[0] + 1, c[1] * 0.5)),
    # 浮点 leaf vs leaf：覆盖 condRhsLeaf>=0 的 ReadScalarFloat 分支
    "rhs_leaf": ((np.float32(4.0), np.float32(0.5)), lambda c: c[1] > c[2],
                 lambda c: (c[0] + 1, c[1] * 0.5, c[2])),
}


def _make_cond_float(jax, jnp, variant):
    tail, cond, body = _COND_FLOAT[variant]
    return (np.int32(0),) + tail, cond, body, None


def _make_cond_all(jax, jnp, variant):
    table = {
        "ge": ([4.0, 2.0, 1.0, 0.5], lambda c: jnp.all(c[0] >= 0.125),
               lambda c: (c[0] * 0.5,)),
        "lt": ([0.01, 0.02, 0.03, 0.04], lambda c: jnp.all(c[0] < 0.5),
               lambda c: (c[0] * 2.0,)),
        "le": ([0.01, 0.02, 0.03, 0.04], lambda c: jnp.all(c[0] <= 0.64),
               lambda c: (c[0] * 2.0,)),
        "gt_swap": ([4.0, 2.0, 1.0, 0.5],
                    lambda c: jnp.all(np.float32(0.125) < c[0]),
                    lambda c: (c[0] * 0.5,)),
    }
    values, cond, body = table[variant]
    return (np.asarray(values, dtype=np.float32),), cond, body, None


def _make_all_256(jax, jnp, variant):
    values = np.asarray([2.0 - 0.01 * i for i in range(256)], dtype=np.float32)
    init = (values,)
    return init, (lambda c: jnp.all(c[0] > 0.0)), (lambda c: (c[0] * 0.5,)), None


def _make_all_257_reject(jax, jnp, variant):
    """all() 条件叶 numel=257：host ABI H10 的 ``> 256`` 拒绝侧，不进入 kernel。"""
    values = np.asarray([2.0 - 0.001 * i for i in range(257)], dtype=np.float32)
    init = (values,)
    return init, (lambda c: jnp.all(c[0] > 0.0)), (lambda c: (c[0] * 0.5,)), None


def _make_jvp_two(jax, jnp, variant):
    init = (np.int32(0), _f32(4, 8), _f32(4, 8))
    # int 叶子的切向量是 float0；两个浮点叶子各一份确定性切向量
    tangent = (_float0(), _f32(4, 8, low=-0.5, high=0.5),
               _f32(4, 8, low=-0.25, high=0.25))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] * 1.5 + c[2] * 0.25 + 0.5, c[2] * 0.75)), tangent


def _make_jvp_reduce(jax, jnp, variant):
    init = (np.int32(0), _f32(8), _f32(8))
    tangent = (_float0(), _f32(8, low=-0.5, high=0.5), _f32(8, low=-0.25, high=0.25))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, c[1] * 0.5 + jnp.mean(c[2]) + 0.25, c[2] * 0.75)), tangent


def _make_stop_gradient(jax, jnp, variant):
    init = (np.int32(0), _f32(8))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, jax.lax.stop_gradient(c[1]) * 0.5 + 0.25)), None


def _make_copy(jax, jnp, variant):
    init = (np.int32(0), _f32(8))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, jnp.array(c[1]) * 0.5 + 0.25)), None


def _make_reshape(jax, jnp, variant):
    init = (np.int32(0), _f32(4, 4))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1,
                   jnp.reshape(jnp.reshape(c[1], (2, 8)), (4, 4)) * 0.5 + 0.25)), None


def _make_expand_squeeze(jax, jnp, variant):
    init = (np.int32(0), _f32(4, 4))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1,
                   jnp.squeeze(jnp.expand_dims(c[1], 0)) * 0.5 + 0.25)), None


def _make_sub_div_neg(jax, jnp, variant):
    init = (np.int32(0), _f32(8))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1, np.float32(0.5) - c[1] / 4.0 + (-c[1]) * 0.125)), None


def _make_const_left(jax, jnp, variant):
    init = (np.int32(0), _f32(8))
    return init, (lambda c: c[0] < 2), (
        lambda c: (c[0] + 1,
                   np.float32(0.5) * c[1] + (np.float32(0.25) + c[1] * np.float32(0.125)))), None


#: scenario 名 → make(jax, jnp, variant) → (init, cond, body, tangent)
SCENARIOS = {
    "tile_exact_8192": _scn_tile(8192),
    "tile_tail_8193": _scn_tile(8193),
    "two_elem_terms": _make_two_elem,
    "reduce_only_scalar": _make_reduce_only,
    "reduce_two_terms": _make_reduce_two,
    "reduce_scan_max_256": _make_reduce_256,
    "reduce_scan_over_257": _make_reduce_257_reject,
    "int_sum_reduce": _make_int_sum,
    "int_two_reduce": _make_int_two_reduce,
    "int_two_elem_terms": _make_int_two_elem,
    "int_const_leaf": _make_int_const,
    "int_negative_round": _make_int_neg,
    "eight_leaves": _make_eight,
    "cond_int_ops": _make_cond_int,
    "cond_float_ops": _make_cond_float,
    "cond_all_ops": _make_cond_all,
    "all_scan_max_256": _make_all_256,
    "all_scan_over_257": _make_all_257_reject,
    "jvp_two_terms": _make_jvp_two,
    "jvp_reduce_term": _make_jvp_reduce,
    "stop_gradient": _make_stop_gradient,
    "copy": _make_copy,
    "reshape_roundtrip": _make_reshape,
    "expand_squeeze": _make_expand_squeeze,
    "sub_div_neg": _make_sub_div_neg,
    "const_left_add": _make_const_left,
}


def build(name: str, jax, jnp, variant: str = "") -> dict:
    """构造一个 scenario 的 init/cond/body/tangent 与树描述。"""
    if name not in SCENARIOS:
        raise KeyError(f"未知白盒 scenario：{name!r}（已注册 {sorted(SCENARIOS)}）")
    init, cond, body, tangent = SCENARIOS[name](jax, jnp, variant or "")
    tree = tree_from_value(init, "carry")
    tangent_tree = _tangent_tree(init, tangent)
    return {
        "scenario": name, "variant": variant or "",
        "init": init, "cond": cond, "body": body, "tangent": tangent,
        "tree": tree, "tangent_tree": tangent_tree,
    }


__all__ = ["SCENARIOS", "build", "tree_from_value", "_leaf", "_tuple", "_float0"]
