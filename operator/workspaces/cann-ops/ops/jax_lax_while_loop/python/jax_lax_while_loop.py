# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop · Python 调用接口（交付件 #2）。

公开入口
--------
``while_loop(cond_fun, body_fun, init_carry, *, safety_limit=None, tangent=None)``
    镜像 ``jax.lax.while_loop`` 的 eager 语义。结构/shape/dtype/weak_type 校验用
    ``jax.eval_shape`` 触发 JAX 自身的真实异常；数值本体经宿主 C ABI
    ``libjax_lax_while_loop_host.so`` → AscendC kernel 在 NPU 上执行。

``while_loop_compiled(cond_fun, body_fun, init_carry, *, mode="primal",
tangent=None, safety_limit=None, jit=True)``
    测试/门禁的“编译态”入口：首次调用把 cond/body 线性化成设备计划并缓存，
    后续同 plan 复用；不是第二个公开 JAX API。``mode="jvp"`` 时同时返回
    ``(primal, tangent)``，切线同样来自设备 kernel。

实现要点
--------
* **计划来自 jaxpr，不按 case 预置结果**：把 ``body_fun`` 追踪成 jaxpr，解释成每个
  carry 叶子的线性表达式（1~2 个 ``scale * leaf`` 项 + 常量，项可为整叶求和以支持
  ``a + mean(b)``）；把 ``cond_fun`` 归一化成 ``leaf <op> leaf|const`` 或
  ``all(leaf > const)``。不认识的 primitive 显式报 ``UNSUPPORTED_PLAN``，不猜测。
* **host_step 路由**：每个 host 入口推进“一次 cond”或“一次 body + 一次 cond”；
  ``body_hits``/``cond_hits``/``limit_flag``/``runtime_token``/``cond_result`` 全部由
  kernel 写设备内存、host 仅 D2H 读回（``counter_source=kernel``）。``abi_calls`` 是
  C ABI 按实际 launch 次数累加的原始整数。
* **错误归一化**：先让 ``jax.eval_shape(lax.while_loop)`` 真实发生并捕获上游异常，
  再依次判定 cond → 结构 → shape → dtype，映射为五码；上游异常类型与单行摘要保留。
* **弱类型**：输出 ``weak_type`` 取 ``jax.eval_shape`` 的 aval；0-d weak 叶子用
  Python float 重建（``jnp.asarray(float(v))``），非 0-d weak 显式报不支持。
* **JVP**：只对浮点 carry 叶生成切线（整型叶为 ``float0``，由调用方/本模块按结构
  构造），切线更新用同一仿射映射且不含常量项。
* **VJP/grad 不在 v1 强制范围**：本接口不提供反向入口。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import struct
from pathlib import Path

import numpy as np

OP_NAME = "jax_lax_while_loop"
KERNEL_NAME = "jax_lax_while_loop_kernel"
NPU_ARCH = "dav-2201"
HOST_LIB = f"lib{OP_NAME}_host.so"
ABI_TAG = "opx-while-loop-python-abi/1"

REJECT_CODES = (
    "INVALID_CONDITION",
    "CARRY_STRUCTURE_MISMATCH",
    "CARRY_SHAPE_MISMATCH",
    "CARRY_DTYPE_MISMATCH",
    "ITERATION_LIMIT_EXCEEDED",
)

#: 与 op_kernel/jax_lax_while_loop_tiling.h 的 WHILE_CMP_* 对齐
COND_LT = 0
COND_LE = 1
COND_GT = 2
COND_GE = 3
COND_EQ = 4
COND_NE = 5
COND_ALL_GT = 6
COND_ALL_GE = 7
COND_ALL_LT = 8
COND_ALL_LE = 9
COND_ALWAYS_TRUE = 10

_COND_CODE = {
    "lt": COND_LT, "le": COND_LE, "gt": COND_GT, "ge": COND_GE, "eq": COND_EQ,
    "ne": COND_NE,
    "all_gt": COND_ALL_GT, "all_ge": COND_ALL_GE,
    "all_lt": COND_ALL_LT, "all_le": COND_ALL_LE,
    "always_true": COND_ALWAYS_TRUE,
}

DTYPE_F32 = 0
DTYPE_I32 = 1

MAX_LEAVES = 8
SCAN_MAX = 256
COUNTER_SLOTS = 8
SLOT_BODY_HITS = 0
SLOT_COND_HITS = 1
SLOT_LIMIT_FLAG = 2
SLOT_TOKEN = 3
SLOT_COND_RESULT = 4
SLOT_ITERATIONS = 5

DEFAULT_SAFETY_LIMIT = 1_000_000

HERE = Path(__file__).resolve().parent
OP_ROOT = HERE.parent
BUILD_DIR = OP_ROOT / "build"
LIB_PATH = BUILD_DIR / HOST_LIB

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 稳定错误
# ---------------------------------------------------------------------------
class WhileLoopError(RuntimeError):
    """带稳定码与上游证据的 while_loop 错误。"""

    def __init__(self, code: str, message: str, *, upstream_exception_type=None,
                 upstream_message: str = "", spec_rule=None, dispatch: dict | None = None):
        self.code = str(code)
        self.upstream_exception_type = upstream_exception_type
        # 单行摘要：既有 runner 用 sha256(summary) 复算，多行会被归一化后不一致。
        self.upstream_message = " ".join(str(upstream_message or "").split())
        self.upstream_message_sha256 = (
            hashlib.sha256(self.upstream_message.encode("utf-8")).hexdigest()
            if self.upstream_message else None
        )
        self.spec_rule = spec_rule
        self.dispatch = dict(dispatch or {})
        super().__init__(f"{self.code}: {message}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# C ABI 结构（与 op_host/jax_lax_while_loop_abi.h 逐字段对应）
# ---------------------------------------------------------------------------
class JaxLaxWhileLoopArgs(ctypes.Structure):
    _fields_ = [
        ("n_leaves", ctypes.c_int32),
        ("has_tangent", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("abi_version", ctypes.c_int32),
        ("stage", ctypes.c_int32),
        ("in_ptrs", ctypes.c_void_p * MAX_LEAVES),
        ("in_numel", ctypes.c_int64 * MAX_LEAVES),
        ("dtype", ctypes.c_int32 * MAX_LEAVES),
        ("out_ptrs", ctypes.c_void_p * MAX_LEAVES),
        ("out_numel", ctypes.c_int64 * MAX_LEAVES),
        ("tan_in", ctypes.c_void_p * MAX_LEAVES),
        ("tan_out", ctypes.c_void_p * MAX_LEAVES),
        ("out_dtype", ctypes.c_int32 * MAX_LEAVES),
        ("n_terms", ctypes.c_int32 * MAX_LEAVES),
        ("out_src1", ctypes.c_int32 * MAX_LEAVES),
        ("out_scale1", ctypes.c_float * MAX_LEAVES),
        ("out_red1", ctypes.c_int32 * MAX_LEAVES),
        ("out_src2", ctypes.c_int32 * MAX_LEAVES),
        ("out_scale2", ctypes.c_float * MAX_LEAVES),
        ("out_red2", ctypes.c_int32 * MAX_LEAVES),
        ("out_const", ctypes.c_float * MAX_LEAVES),
        ("tan_out_flag", ctypes.c_int32 * MAX_LEAVES),
        ("cond_op", ctypes.c_int32),
        ("cond_leaf", ctypes.c_int32),
        ("cond_rhs_leaf", ctypes.c_int32),
        ("cond_is_int", ctypes.c_int32),
        ("cond_const", ctypes.c_float),
        ("safety_limit", ctypes.c_int32),
        ("plan_seed", ctypes.c_uint32),
        ("seq", ctypes.c_void_p),
        ("counters_host", ctypes.POINTER(ctypes.c_int32)),
        ("err", ctypes.c_void_p),
        ("err_len", ctypes.c_int32),
    ]


# 布局自检（与 op_host/jax_lax_while_loop_abi.h 在 g++ 下实测一致）：824 字节。
_ARGS_EXPECTED_SIZE = 824
if ctypes.sizeof(JaxLaxWhileLoopArgs) != _ARGS_EXPECTED_SIZE:  # pragma: no cover
    raise RuntimeError(
        f"JaxLaxWhileLoopArgs 布局漂移：ctypes={ctypes.sizeof(JaxLaxWhileLoopArgs)} "
        f"!= {_ARGS_EXPECTED_SIZE}")

_LIB = None
_LIB_FINGERPRINT: dict | None = None
_PLAN_CACHE: dict[str, int] = {}
last_dispatch: dict = {}


def _load_lib():
    global _LIB, _LIB_FINGERPRINT
    if _LIB is not None:
        return _LIB
    if not LIB_PATH.is_file():
        raise WhileLoopError(
            "device_error",
            f"缺少共享库 {LIB_PATH}（先执行 build：scripts/nm.sh --op {OP_NAME} --action build）",
        )
    lib = ctypes.CDLL(str(LIB_PATH), mode=ctypes.RTLD_GLOBAL)
    lib.jax_lax_while_loop_seq_begin.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                 ctypes.c_int32]
    lib.jax_lax_while_loop_seq_begin.restype = ctypes.c_int
    lib.jax_lax_while_loop_step.argtypes = [ctypes.POINTER(JaxLaxWhileLoopArgs)]
    lib.jax_lax_while_loop_step.restype = ctypes.c_int
    lib.jax_lax_while_loop_seq_end.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_int32),
                                               ctypes.POINTER(ctypes.c_int64)]
    lib.jax_lax_while_loop_seq_end.restype = ctypes.c_int
    lib.jax_lax_while_loop_seq_free.argtypes = [ctypes.c_void_p]
    lib.jax_lax_while_loop_seq_free.restype = ctypes.c_int
    lib.jax_lax_while_loop_abi_version.argtypes = []
    lib.jax_lax_while_loop_abi_version.restype = ctypes.c_char_p
    lib.jax_lax_while_loop_acl_init.argtypes = []
    lib.jax_lax_while_loop_acl_init.restype = ctypes.c_int
    lib.jax_lax_while_loop_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_int64]
    lib.jax_lax_while_loop_malloc.restype = ctypes.c_int
    lib.jax_lax_while_loop_free.argtypes = [ctypes.c_void_p]
    lib.jax_lax_while_loop_free.restype = ctypes.c_int
    lib.jax_lax_while_loop_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int64]
    lib.jax_lax_while_loop_h2d.restype = ctypes.c_int
    lib.jax_lax_while_loop_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int64]
    lib.jax_lax_while_loop_d2h.restype = ctypes.c_int
    abi = lib.jax_lax_while_loop_abi_version()
    abi_text = abi.decode("utf-8") if isinstance(abi, bytes) else str(abi)
    if abi_text != ABI_TAG:
        raise WhileLoopError("device_error",
                             f".so ABI={abi_text!r} 与 Python 期望的 {ABI_TAG!r} 不一致")
    _LIB = lib
    _LIB_FINGERPRINT = {
        "path": str(LIB_PATH),
        "size_bytes": LIB_PATH.stat().st_size,
        "sha256": _sha256_file(LIB_PATH),
        "abi": abi_text,
    }
    _LOG.info("jax_lax_while_loop loaded so=%s sha256=%s abi=%s", LIB_PATH,
              _LIB_FINGERPRINT["sha256"], abi_text)
    return _LIB


def library_fingerprint() -> dict:
    _load_lib()
    return dict(_LIB_FINGERPRINT or {})


def abi_version() -> str:
    _load_lib()
    return ABI_TAG


# ---------------------------------------------------------------------------
# pytree / meta 工具（稳定顺序：dict 按插入顺序，tuple/list 区分）
# ---------------------------------------------------------------------------
def stable_flatten(value, prefix: str = "") -> dict:
    out: dict = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key in node:
                walk(node[key], f"{path}/{key}" if path else str(key))
            return
        if isinstance(node, (tuple, list)):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}" if path else str(index))
            return
        out[path or "out"] = np.asarray(node)

    walk(value, prefix)
    return out


def stable_leaf_meta(value, prefix: str = "") -> dict:
    """稳定顺序的逐叶元数据（在转 numpy **之前**读取 weak_type）。"""
    out: dict = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key in node:
                walk(node[key], f"{path}/{key}" if path else str(key))
            return
        if isinstance(node, (tuple, list)):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}" if path else str(index))
            return
        out[path or "out"] = leaf_meta(node)

    walk(value, prefix)
    return out


def leaf_meta(value) -> dict:
    weak = getattr(value, "weak_type", None)
    arr = np.asarray(value)
    dtype_text = str(arr.dtype)
    if "float0" in dtype_text:
        dtype_text = "[('float0', 'V')]"
    h = hashlib.sha256()
    h.update(dtype_text.encode("utf-8"))
    h.update(b"\0")
    h.update(str([int(x) for x in arr.shape]).encode("utf-8"))
    h.update(b"\0")
    h.update(np.ascontiguousarray(arr).tobytes())
    return {
        "dtype": dtype_text,
        "shape": [int(x) for x in arr.shape],
        "weak_type": bool(weak) if weak is not None else None,
        "sha256": h.hexdigest(),
    }


def tree_signature(value) -> list | str:
    if isinstance(value, dict):
        return ["dict", [[str(key), tree_signature(value[key])]
                         for key in sorted(value.keys(), key=str)]]
    if isinstance(value, tuple):
        return ["tuple", [tree_signature(item) for item in value]]
    if isinstance(value, list):
        return ["list", [tree_signature(item) for item in value]]
    return "leaf"


def _jax():
    import jax  # noqa: PLC0415 —— 延迟 import，拒绝路径不初始化后端

    return jax


def _struct_tree(value):
    jax = _jax()

    def convert(leaf):
        arr = np.asarray(leaf)
        return jax.ShapeDtypeStruct(
            tuple(arr.shape), arr.dtype, weak_type=bool(getattr(leaf, "weak_type", False)))

    return jax.tree_util.tree_map(convert, value)


# ---------------------------------------------------------------------------
# 计划：jaxpr → 线性表达式
# ---------------------------------------------------------------------------
class _Expr:
    __slots__ = ("terms", "const", "aval")

    def __init__(self, terms=None, const=0.0, aval=None):
        self.terms = dict(terms or {})   # leaf -> (coef, reduced)
        self.const = float(const)
        self.aval = aval

    def copy(self):
        return _Expr(self.terms, self.const, self.aval)

    def is_const(self):
        return not self.terms

    def scale(self, k):
        return _Expr({i: (c * k, r) for i, (c, r) in self.terms.items()},
                     self.const * k, self.aval)

    def mix(self, other, sign=1.0):
        out = self.copy()
        for idx, (coef, red) in other.terms.items():
            if idx in out.terms:
                old_coef, old_red = out.terms[idx]
                if old_red != red:
                    raise NotImplementedError("同一叶子混合了归约/逐元素项")
                out.terms[idx] = (old_coef + sign * coef, red)
            else:
                out.terms[idx] = (sign * coef, red)
        out.const += sign * other.const
        return out

    def reduce_all(self):
        red = _Expr()
        for idx, (coef, reduced) in self.terms.items():
            if reduced:
                raise NotImplementedError("重复归约")
            red.terms[idx] = (coef, True)
        return red


def _literal_expr(var):
    val = np.asarray(var.val)
    if val.size != 1:
        raise NotImplementedError("非标量常量")
    return _Expr({}, float(val),
                 (tuple(int(x) for x in var.aval.shape), np.dtype(var.aval.dtype)))


def _interpret_body(jaxpr, flat_init):
    env = {}

    def get(var):
        key = str(var)
        if key in env:
            return env[key]
        if hasattr(var, "val") and var.val is not None:
            return _literal_expr(var)
        raise KeyError(f"未绑定变量 {key}")

    if len(jaxpr.invars) != len(flat_init):
        raise NotImplementedError(
            f"jaxpr 参数数 {len(jaxpr.invars)} != carry 叶子数 {len(flat_init)}")
    for var, expr in zip(jaxpr.invars, flat_init):
        env[str(var)] = expr

    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name
        av = eqn.outvars[0].aval
        aval = (tuple(int(x) for x in av.shape), np.dtype(av.dtype))
        args = eqn.invars
        if prim == "add":
            a, b = get(args[0]), get(args[1])
            if a.is_const():
                out = b.copy()
                out.const += a.const
            elif b.is_const():
                out = a.copy()
                out.const += b.const
            else:
                out = a.mix(b)
        elif prim == "sub":
            out = get(args[0]).mix(get(args[1]), sign=-1.0)
        elif prim == "neg":
            out = get(args[0]).scale(-1.0)
        elif prim == "mul":
            a, b = get(args[0]), get(args[1])
            if a.is_const() and b.is_const():
                out = _Expr({}, a.const * b.const)
            elif a.is_const():
                out = b.scale(a.const)
            elif b.is_const():
                out = a.scale(b.const)
            else:
                raise NotImplementedError("两个非常量相乘（非线性 body 不在 v1 计划内）")
        elif prim == "div":
            a, b = get(args[0]), get(args[1])
            if not b.is_const():
                raise NotImplementedError("除以非常量")
            out = a.scale(1.0 / b.const)
        elif prim == "reduce_sum":
            out = get(args[0]).reduce_all()
        elif prim in ("broadcast_in_dim", "reshape", "squeeze", "expand_dims", "copy",
                      "stop_gradient", "tie_in"):
            a = get(args[0])
            out = a.copy()
            if prim in ("broadcast_in_dim", "reshape"):
                n_in = int(np.prod(a.aval[0])) if a.aval else 1
                n_out = int(np.prod(aval[0]))
                if n_in != n_out:
                    raise NotImplementedError(f"{prim} 改变元素数 {n_in}->{n_out}")
        elif prim == "convert_element_type":
            out = get(args[0]).copy()
        else:
            raise NotImplementedError(f"primitive {prim}")
        out.aval = aval
        env[str(eqn.outvars[0])] = out

    return [get(var) for var in jaxpr.outvars]


def _interpret_cond(jaxpr, n_leaves):
    env = {}

    def get(var):
        key = str(var)
        if key in env:
            return env[key]
        if hasattr(var, "val") and var.val is not None:
            return ("const", var.val)
        raise KeyError(key)

    for var, idx in zip(jaxpr.invars, range(n_leaves)):
        env[str(var)] = ("leaf", idx)
    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name
        if prim in ("lt", "le", "gt", "ge", "eq", "ne"):
            a, b = get(eqn.invars[0]), get(eqn.invars[1])
            if a[0] == "const" and b[0] == "const":
                fn = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
                      "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}[prim]
                env[str(eqn.outvars[0])] = ("const", bool(fn(a[1], b[1])))
            else:
                env[str(eqn.outvars[0])] = ("cmp", prim, a, b)
        elif prim == "reduce_and":
            env[str(eqn.outvars[0])] = ("all", get(eqn.invars[0]))
        else:
            env[str(eqn.outvars[0])] = ("unsupported", prim)

    out = get(jaxpr.outvars[0])
    if out[0] == "const":
        if bool(out[1]):
            return {"op": "always_true"}
        raise NotImplementedError("恒假 cond（while_loop 0 次循环应写成取反条件）")
    if out[0] == "cmp":
        _, prim, a, b = out
        if a[0] == "leaf":
            leaf, rhs = a[1], b
        elif b[0] == "leaf":
            leaf, rhs = b[1], a
            prim = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le",
                    "eq": "ne", "ne": "eq"}[prim]
        else:
            raise NotImplementedError("cond 没有叶子操作数")
        if rhs[0] == "leaf":
            return {"op": prim, "leaf": leaf, "rhs_leaf": rhs[1], "const": None}
        return {"op": prim, "leaf": leaf, "rhs_leaf": -1, "const": float(rhs[1])}
    if out[0] == "all":
        inner = out[1]
        if inner[0] != "cmp":
            raise NotImplementedError("all() 的不是比较")
        _, prim, a, b = inner
        if a[0] == "const":
            a, b = b, a
            prim = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}[prim]
        if a[0] != "leaf" or b[0] != "const":
            raise NotImplementedError("all() 需要 tensor vs 常量")
        return {"op": "all_" + prim, "leaf": a[1], "rhs_leaf": -1,
                "const": float(b[1])}
    raise NotImplementedError(f"cond 形态不支持：{out[0]}")


def _derive_plan(cond_fun, body_fun, init):
    """把 cond/body 追踪成 jaxpr 并线性化成设备计划。"""
    jax = _jax()
    structs = _struct_tree(init)
    leaves, treedef = jax.tree_util.tree_flatten(init)
    flat_structs, _ = jax.tree_util.tree_flatten(structs)
    body_jaxpr = jax.make_jaxpr(body_fun)(structs)
    cond_jaxpr = jax.make_jaxpr(cond_fun)(structs)
    exprs = [_Expr({i: (1.0, False)}, 0.0,
                   (tuple(int(x) for x in np.asarray(leaves[i]).shape),
                    np.dtype(np.asarray(leaves[i]).dtype)))
             for i in range(len(leaves))]
    out_exprs = _interpret_body(body_jaxpr.jaxpr, exprs)
    if len(out_exprs) != len(leaves):
        raise NotImplementedError(
            f"body 输出叶子数 {len(out_exprs)} != carry 叶子数 {len(leaves)}")
    plan_leaves = []
    for j, expr in enumerate(out_exprs):
        numel = int(np.prod(expr.aval[0])) if expr.aval[0] else 1
        plan_leaves.append({
            "terms": [(int(i), float(c), bool(r)) for i, (c, r) in expr.terms.items()],
            "const": float(expr.const),
            "dtype": str(expr.aval[1]),
            "shape": [int(x) for x in expr.aval[0]],
            "numel": numel,
            "has_tan": np.dtype(expr.aval[1]) == np.float32,
            "structure": flat_structs[j],
        })
    cond_plan = _interpret_cond(cond_jaxpr.jaxpr, len(leaves))
    cond_plan["is_int"] = bool(np.dtype(flat_structs[cond_plan.get("leaf", 0)].dtype) == np.int32)
    return {"leaves": plan_leaves, "cond": cond_plan}, treedef


def _plan_id(plan: dict) -> str:
    payload = {
        "leaves": [
            {
                "terms": [[i, struct.pack("<d", c).hex(), 1 if r else 0] for i, c, r in leaf["terms"]],
                "const": struct.pack("<d", leaf["const"]).hex(),
                "dtype": leaf["dtype"],
                "shape": leaf["shape"],
                "has_tan": leaf["has_tan"],
            }
            for leaf in plan["leaves"]
        ],
        "cond": {
            "op": plan["cond"].get("op"),
            "leaf": plan["cond"].get("leaf"),
            "rhs_leaf": plan["cond"].get("rhs_leaf"),
            "is_int": plan["cond"].get("is_int"),
            "const": (None if plan["cond"].get("const") is None
                      else struct.pack("<d", float(plan["cond"]["const"])).hex()),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _plan_seed(plan_id: str) -> int:
    return int(plan_id[:8], 16)


# ---------------------------------------------------------------------------
# 校验与错误归一化
# ---------------------------------------------------------------------------
def _validate(cond_fun, body_fun, init):
    jax = _jax()
    hits = {"cond": 0, "body": 0}

    def wrapped_cond(carry):
        hits["cond"] += 1
        return cond_fun(carry)

    def wrapped_body(carry):
        hits["body"] += 1
        return body_fun(carry)

    structs = _struct_tree(init)
    try:
        out = jax.eval_shape(
            lambda c: jax.lax.while_loop(wrapped_cond, wrapped_body, c), structs)
    except Exception as exc:  # noqa: BLE001 —— 上游 JAX 校验异常必须原样保留
        return None, hits, exc
    return out, hits, None


def _structure_label(tree) -> str:
    return repr(tree)


def _classify(exc, cond_fun, body_fun, init) -> str:
    jax = _jax()
    structs = _struct_tree(init)
    message = " ".join(str(exc).split())

    cond_struct = None
    try:
        cond_struct = jax.eval_shape(cond_fun, structs)
    except Exception:  # noqa: BLE001
        cond_struct = None
    if cond_struct is not None:
        cond_leaves = jax.tree_util.tree_leaves(cond_struct)
        if (len(cond_leaves) != 1 or np.dtype(cond_leaves[0].dtype) != np.bool_
                or tuple(cond_leaves[0].shape) != ()):
            return "INVALID_CONDITION"
    elif "cond_fun must return a boolean scalar" in message:
        return "INVALID_CONDITION"

    in_struct = jax.tree_util.tree_structure(structs)
    try:
        body_struct = jax.eval_shape(body_fun, structs)
    except Exception:  # noqa: BLE001
        body_struct = None
    if body_struct is None:
        if "same pytree structure" in message:
            return "CARRY_STRUCTURE_MISMATCH"
        if "same shape" in message or ("shape" in message and "same" in message):
            return "CARRY_SHAPE_MISMATCH"
        if "same dtype" in message or "dtype" in message:
            return "CARRY_DTYPE_MISMATCH"
        if "must have equal types" in message:
            return "CARRY_SHAPE_MISMATCH" if "[" in message else "CARRY_DTYPE_MISMATCH"
        raise WhileLoopError("INVALID_CONDITION", message,
                             upstream_exception_type=type(exc).__name__,
                             upstream_message=message)
    out_struct = jax.tree_util.tree_structure(body_struct)
    if _structure_label(in_struct) != _structure_label(out_struct):
        return "CARRY_STRUCTURE_MISMATCH"
    in_leaves = jax.tree_util.tree_leaves(structs)
    out_leaves = jax.tree_util.tree_leaves(body_struct)
    if len(in_leaves) != len(out_leaves):
        return "CARRY_STRUCTURE_MISMATCH"
    for a, b in zip(in_leaves, out_leaves):
        if tuple(a.shape) != tuple(b.shape):
            return "CARRY_SHAPE_MISMATCH"
    for a, b in zip(in_leaves, out_leaves):
        if np.dtype(a.dtype) != np.dtype(b.dtype):
            return "CARRY_DTYPE_MISMATCH"
    # 结构/shape/dtype 都一致：conservative 地按上游消息兜底。
    if "pytree structure" in message:
        return "CARRY_STRUCTURE_MISMATCH"
    if "shape" in message:
        return "CARRY_SHAPE_MISMATCH"
    return "CARRY_DTYPE_MISMATCH"


# ---------------------------------------------------------------------------
# 设备执行
# ---------------------------------------------------------------------------
def _dtype_code(dtype) -> int:
    if np.dtype(dtype) == np.int32:
        return DTYPE_I32
    if np.dtype(dtype) == np.float32:
        return DTYPE_F32
    raise WhileLoopError("invalid_argument", f"carry 叶子 dtype={dtype} 只支持 float32/int32")


def _itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


class _DeviceBuffers:
    """一组 carry（含可选切线）的设备缓冲区；None 表示该叶子无设备数据。"""

    def __init__(self, lib, plan, arrays, tangent_arrays=None):
        self.lib = lib
        self.ptrs: list = []
        self.tan_ptrs: list = []
        self.lib = lib
        for index, arr in enumerate(arrays):
            nbytes = int(np.asarray(arr).nbytes)
            ptr = ctypes.c_void_p()
            if nbytes > 0:
                if lib.jax_lax_while_loop_malloc(ctypes.byref(ptr), nbytes) != 0:
                    raise WhileLoopError("device_error", f"aclrtMalloc 失败（叶子 {index}）")
                host = np.ascontiguousarray(np.asarray(arr))
                if lib.jax_lax_while_loop_h2d(ptr, host.ctypes.data_as(ctypes.c_void_p),
                                              nbytes) != 0:
                    raise WhileLoopError("device_error", f"H2D 失败（叶子 {index}）")
            self.ptrs.append(ptr if nbytes > 0 else None)
        if tangent_arrays is not None:
            for index, arr in enumerate(tangent_arrays):
                nbytes = int(np.asarray(arr).nbytes) if arr is not None else 0
                ptr = ctypes.c_void_p()
                if nbytes > 0:
                    if lib.jax_lax_while_loop_malloc(ctypes.byref(ptr), nbytes) != 0:
                        raise WhileLoopError("device_error", f"aclrtMalloc 失败（切线 {index}）")
                    host = np.ascontiguousarray(np.asarray(arr))
                    if lib.jax_lax_while_loop_h2d(
                            ptr, host.ctypes.data_as(ctypes.c_void_p), nbytes) != 0:
                        raise WhileLoopError("device_error", f"H2D 失败（切线 {index}）")
                self.tan_ptrs.append(ptr if nbytes > 0 else None)
        else:
            self.tan_ptrs = [None] * len(arrays)

    def close(self):
        for ptr in list(self.ptrs) + list(self.tan_ptrs):
            if ptr is not None:
                self.lib.jax_lax_while_loop_free(ptr)


def _fill_args(args, plan, device_id, stage, is_jvp):
    args.stage = stage
    args.n_leaves = len(plan["leaves"])
    args.has_tangent = 1 if is_jvp else 0
    args.device_id = device_id
    args.abi_version = 1
    for j, leaf in enumerate(plan["leaves"]):
        args.out_dtype[j] = _dtype_code(leaf["dtype"])
        args.n_terms[j] = len(leaf["terms"]) if leaf["terms"] else 0
        if leaf["terms"]:
            src, coef, red = leaf["terms"][0]
            args.out_src1[j] = src
            args.out_scale1[j] = coef
            args.out_red1[j] = 1 if red else 0
        if len(leaf["terms"]) >= 2:
            src, coef, red = leaf["terms"][1]
            args.out_src2[j] = src
            args.out_scale2[j] = coef
            args.out_red2[j] = 1 if red else 0
        args.out_const[j] = leaf["const"]
        args.tan_out_flag[j] = 1 if (is_jvp and leaf["has_tan"]) else 0
    cond = plan["cond"]
    args.cond_op = _COND_CODE[cond["op"]]
    args.cond_leaf = int(cond.get("leaf", 0))
    args.cond_rhs_leaf = int(cond.get("rhs_leaf", -1))
    args.cond_is_int = 1 if cond.get("is_int") else 0
    args.cond_const = float(cond.get("const") or 0.0)


def _set_buffers(args, cur: _DeviceBuffers, other: _DeviceBuffers):
    for j in range(MAX_LEAVES):
        args.in_ptrs[j] = cur.ptrs[j] if j < len(cur.ptrs) and cur.ptrs[j] is not None else None
        args.out_ptrs[j] = (other.ptrs[j] if j < len(other.ptrs) and other.ptrs[j] is not None
                            else None)
        args.tan_in[j] = (cur.tan_ptrs[j]
                          if j < len(cur.tan_ptrs) and cur.tan_ptrs[j] is not None else None)
        args.tan_out[j] = (other.tan_ptrs[j]
                           if j < len(other.tan_ptrs) and other.tan_ptrs[j] is not None else None)


def _read_counters(counters) -> dict:
    return {
        "body_hits": int(counters[SLOT_BODY_HITS]),
        "cond_hits": int(counters[SLOT_COND_HITS]),
        "limit_flag": int(counters[SLOT_LIMIT_FLAG]),
        "token": int(counters[SLOT_TOKEN]),
        "cond_result": int(counters[SLOT_COND_RESULT]),
        "iterations": int(counters[SLOT_ITERATIONS]),
    }


def _download(lib, plan, buffers: _DeviceBuffers) -> list[np.ndarray]:
    out = []
    for j, leaf in enumerate(plan["leaves"]):
        dtype = np.dtype(leaf["dtype"])
        nbytes = int(np.prod(leaf["shape"], dtype=np.int64)) * dtype.itemsize if leaf["shape"] \
            else dtype.itemsize
        arr = np.zeros(tuple(leaf["shape"]) if leaf["shape"] else (), dtype=dtype)
        ptr = buffers.ptrs[j]
        if nbytes > 0 and ptr is not None:
            if lib.jax_lax_while_loop_d2h(arr.ctypes.data_as(ctypes.c_void_p), ptr,
                                          nbytes) != 0:
                raise WhileLoopError("device_error", f"D2H 失败（叶子 {j}）")
        out.append(arr)
    return out


def _download_tangent(lib, plan, buffers: _DeviceBuffers) -> list:
    out = []
    for j, leaf in enumerate(plan["leaves"]):
        if not leaf["has_tan"]:
            out.append(None)
            continue
        dtype = np.float32
        nbytes = int(np.prod(leaf["shape"], dtype=np.int64)) * 4 if leaf["shape"] else 4
        arr = np.zeros(tuple(leaf["shape"]) if leaf["shape"] else (), dtype=dtype)
        ptr = buffers.tan_ptrs[j]
        if nbytes > 0 and ptr is not None:
            if lib.jax_lax_while_loop_d2h(arr.ctypes.data_as(ctypes.c_void_p), ptr,
                                          nbytes) != 0:
                raise WhileLoopError("device_error", f"D2H 失败（切线 {j}）")
        out.append(arr)
    return out


def _make_jax_leaf(np_arr, aval):
    jax = _jax()
    dtype = np.dtype(aval.dtype)
    weak = bool(getattr(aval, "weak_type", False))
    if dtype == np.float32:
        if weak:
            if tuple(aval.shape) != ():
                raise WhileLoopError(
                    "UNSUPPORTED_WEAK_TYPE",
                    f"非 0-d weak_type 叶子 shape={aval.shape} 不支持（v1 只覆盖 0-d weak）")
            return jax.numpy.asarray(float(np.asarray(np_arr)))
        return jax.numpy.asarray(np.asarray(np_arr, np.float32), dtype=jax.numpy.float32)
    if dtype == np.int32:
        return jax.numpy.asarray(np.asarray(np_arr, np.int32), dtype=jax.numpy.int32)
    raise WhileLoopError("invalid_argument", f"不支持的输出 dtype {dtype}")


def _make_float0():
    jax = _jax()
    raw = np.zeros((), dtype=np.dtype([("float0", "V")]))
    return jax.numpy.asarray(raw, dtype=jax.dtypes.float0)


def _execution_environment():
    jax = _jax()
    try:
        backend = str(jax.default_backend())
    except Exception:  # noqa: BLE001
        backend = ""
    try:
        platforms = sorted({str(device.platform) for device in jax.devices()})
    except Exception:  # noqa: BLE001
        platforms = []
    return backend, platforms


def _dispatch_base(*, mode, variant, path, exec_mode, abi_calls, counters, plan_id,
                   plan_cache, compile_count, trace_hits, backend, platforms, device_id):
    token = 0
    payload = {
        "op": OP_NAME,
        "kernel": KERNEL_NAME,
        "arch": NPU_ARCH,
        "mode": mode,
        "variant": variant,
        "path": path,
        "exec_mode": exec_mode,
        "route": "host_step" if path == "so" else "reject",
        "abi_calls": int(abi_calls),
        "backend": backend,
        "platforms": list(platforms),
        "device_id": int(device_id),
        "iterations": 0,
        "cond_hits": 0,
        "body_hits": 0,
        "trace_hits": int(trace_hits),
        "exec_hits": int(abi_calls),
        "runtime_token": "",
        "counter_source": "kernel" if path == "so" else "none",
        "compile_count": int(compile_count),
        "plan_id": str(plan_id),
        "plan_cache": str(plan_cache),
        "limit_flag": 0,
        "leaves": [],
        "tree": None,
    }
    if counters is not None:
        body_hits = counters["body_hits"]
        cond_hits = counters["cond_hits"]
        token = counters["token"] & 0xFFFFFFFF
        payload.update({
            "iterations": int(body_hits),
            "cond_hits": int(cond_hits),
            "body_hits": int(body_hits),
            "limit_flag": int(counters["limit_flag"]),
            "runtime_token": f"dev-{token:08x}",
        })
    return payload


def _execute(cond_fun, body_fun, init_carry, *, mode, tangent, safety_limit, compiled):
    """统一执行路径：校验 → 计划 → host_step 设备循环 → 输出/证据。"""
    global last_dispatch
    jax = _jax()
    variant = str(getattr(body_fun, "opx_variant", "") or "")

    if safety_limit is None:
        safety_limit = DEFAULT_SAFETY_LIMIT
    if not isinstance(safety_limit, (int, np.integer)) or int(safety_limit) < 1:
        raise WhileLoopError("invalid_argument", f"safety_limit={safety_limit!r} 必须是正整数")
    safety_limit = int(safety_limit)

    is_jvp = mode == "jvp"
    exec_mode = ("jvp_jit" if is_jvp and compiled else "jvp_eager" if is_jvp
                 else "jit" if compiled else "eager")
    if is_jvp and tangent is None:
        raise WhileLoopError("invalid_argument", "mode='jvp' 需要 tangent 切向量")
    if not is_jvp and tangent is not None:
        raise WhileLoopError("invalid_argument", "非 jvp 调用不接受 tangent")

    # 1) 让 JAX 的 while_loop 校验真实发生（拒绝路径在这里归一化）
    aval, hits, captured = _validate(cond_fun, body_fun, init_carry)
    trace_hits = hits["cond"] + hits["body"]
    if captured is not None:
        code = _classify(captured, cond_fun, body_fun, init_carry)
        message = " ".join(str(captured).split())
        device_id = _container_device_id()
        dispatch = _dispatch_base(
            mode=exec_mode, variant=variant, path="reject", exec_mode="reject", abi_calls=0,
            counters=None, plan_id="", plan_cache="reject", compile_count=0,
            trace_hits=trace_hits, backend=None, platforms=[], device_id=device_id)
        dispatch["leaves"] = []
        dispatch["tree"] = None
        last_dispatch = dispatch
        raise WhileLoopError(code, message,
                             upstream_exception_type=type(captured).__name__,
                             upstream_message=message, dispatch=dispatch)

    # 2) 线性化 body/cond → 设备计划
    try:
        plan, treedef = _derive_plan(cond_fun, body_fun, init_carry)
    except NotImplementedError as exc:
        device_id = _container_device_id()
        message = f"无法线性化成设备计划：{exc}"
        last_dispatch = _dispatch_base(
            mode=exec_mode, variant=variant, path="reject", exec_mode="unsupported", abi_calls=0,
            counters=None, plan_id="", plan_cache="reject", compile_count=0,
            trace_hits=trace_hits, backend=None, platforms=[], device_id=device_id)
        raise WhileLoopError("UNSUPPORTED_PLAN", message, dispatch=last_dispatch)

    # 3) 资源与计划缓存
    leaves, _ = jax.tree_util.tree_flatten(init_carry)
    tangent_leaves = None
    if is_jvp:
        tangent_leaves, _ = jax.tree_util.tree_flatten(tangent)
        if len(tangent_leaves) != len(leaves):
            raise WhileLoopError("invalid_argument",
                                 f"tangent 叶子数 {len(tangent_leaves)} != carry {len(leaves)}")
    plan_id = _plan_id(plan)
    compile_count = 1
    plan_cache = "hit" if plan_id in _PLAN_CACHE else "miss"
    _PLAN_CACHE.setdefault(plan_id, 1)

    device_id = _container_device_id()
    lib = _load_lib()
    handle = ctypes.c_void_p()
    if lib.jax_lax_while_loop_seq_begin(ctypes.byref(handle), device_id) != 0:
        raise WhileLoopError("device_error", "seq_begin 失败（aclInit/aclrtSetDevice/malloc）")
    args = JaxLaxWhileLoopArgs()
    args.seq = handle
    err_buf = ctypes.create_string_buffer(256)
    args.err = ctypes.cast(err_buf, ctypes.c_void_p)
    args.err_len = 256
    args.abi_version = 1
    args.n_leaves = len(leaves)
    args.has_tangent = 1 if is_jvp else 0
    args.device_id = device_id
    args.safety_limit = safety_limit
    args.plan_seed = _plan_seed(plan_id)
    for j, arr in enumerate(leaves):
        arr_np = np.asarray(arr)
        args.in_numel[j] = int(arr_np.size)
        args.out_numel[j] = int(arr_np.size)
        args.dtype[j] = _dtype_code(arr_np.dtype)
    if is_jvp:
        for j, arr in enumerate(tangent_leaves):
            if np.dtype(np.asarray(arr).dtype).kind == "V":
                continue
    counters = (ctypes.c_int32 * COUNTER_SLOTS)()
    args.counters_host = ctypes.cast(counters, ctypes.POINTER(ctypes.c_int32))
    _fill_args(args, plan, device_id, stage=0, is_jvp=is_jvp)

    buffers = None
    limit_hit = False
    final_counters = None
    abi_calls = 0
    try:
        tangent_np = None
        if is_jvp:
            tangent_np = [
                np.zeros(np.asarray(arr).shape, np.float32)
                if np.dtype(np.asarray(arr).dtype).kind == "V"
                else np.asarray(arr, np.float32)
                for arr in tangent_leaves
            ]
        buffers_a = _DeviceBuffers(lib, plan, [np.asarray(a) for a in leaves], tangent_np)
        buffers_b = _DeviceBuffers(lib, plan, [np.asarray(a) for a in leaves], tangent_np)
        buffers = (buffers_a, buffers_b)
        cur, other = buffers_a, buffers_b

        # stage 0：先求值一次 cond（0 次循环时 body_hits=0 / cond_hits=1）
        _set_buffers(args, cur, other)
        args.stage = 0
        if lib.jax_lax_while_loop_step(ctypes.byref(args)) != 0:
            raise WhileLoopError(
                "device_error",
                f"kernel stage=0 失败：{err_buf.value.decode('utf-8', 'replace')}")
        state = _read_counters(counters)
        if state["limit_flag"] != 0:
            limit_hit = True
        while not limit_hit and state["cond_result"] != 0:
            _set_buffers(args, cur, other)
            args.stage = 1
            if lib.jax_lax_while_loop_step(ctypes.byref(args)) != 0:
                raise WhileLoopError(
                    "device_error",
                    f"kernel stage=1 失败：{err_buf.value.decode('utf-8', 'replace')}")
            cur, other = other, cur
            state = _read_counters(counters)
            if state["limit_flag"] != 0:
                limit_hit = True
        final_buffers = cur
        outputs = _download(lib, plan, final_buffers)
        tangents = _download_tangent(lib, plan, final_buffers) if is_jvp else None
    finally:
        counters_out = (ctypes.c_int32 * COUNTER_SLOTS)()
        abi_out = ctypes.c_int64(0)
        lib.jax_lax_while_loop_seq_end(handle, counters_out, ctypes.byref(abi_out))
        abi_calls = int(abi_out.value)
        final_counters = _read_counters(counters_out)
        if buffers is not None:
            for item in buffers:
                item.close()

    backend, platforms = _execution_environment()
    if limit_hit or final_counters["limit_flag"] != 0:
        dispatch = _dispatch_base(
            mode=exec_mode, variant=variant, path="so", exec_mode="device_safety_limit",
            abi_calls=abi_calls, counters=final_counters, plan_id=plan_id,
            plan_cache=plan_cache, compile_count=compile_count, trace_hits=trace_hits,
            backend=backend, platforms=platforms, device_id=device_id)
        dispatch["leaves"] = []
        dispatch["tree"] = None
        last_dispatch = dispatch
        message = (f"设备安全迭代上限触发：iterations={final_counters['body_hits']} "
                   f"cond_hits={final_counters['cond_hits']} limit={safety_limit} "
                   f"limit_flag={final_counters['limit_flag']}")
        raise WhileLoopError("ITERATION_LIMIT_EXCEEDED", message,
                             upstream_exception_type=None, upstream_message=message,
                             spec_rule="iteration_safety_limit", dispatch=dispatch)

    # 4) 输出重建（weak_type 取 eval_shape 的 aval）
    out_leaves = jax.tree_util.tree_leaves(aval)
    if len(out_leaves) != len(plan["leaves"]):
        raise WhileLoopError("device_error",
                             f"输出叶子数 {len(out_leaves)} != 计划 {len(plan['leaves'])}")
    primal_leaves = [_make_jax_leaf(outputs[j], out_leaves[j])
                     for j in range(len(outputs))]
    primal = jax.tree_util.tree_unflatten(treedef, primal_leaves)
    if is_jvp:
        _, tan_treedef = jax.tree_util.tree_flatten(tangent)
        tan_avals = jax.tree_util.tree_leaves(_struct_tree(tangent))
        tan_leaves_out = []
        for j, aval_leaf in enumerate(tan_avals):
            if np.dtype(aval_leaf.dtype).kind == "V":
                tan_leaves_out.append(_make_float0())
            else:
                tan_leaves_out.append(_make_jax_leaf(tangents[j], aval_leaf))
        tangent_out = jax.tree_util.tree_unflatten(tan_treedef, tan_leaves_out)
        result = (primal, tangent_out)
    else:
        result = primal

    dispatch = _dispatch_base(
        mode=exec_mode, variant=variant, path="so", exec_mode=exec_mode,
        abi_calls=abi_calls, counters=final_counters, plan_id=plan_id,
        plan_cache=plan_cache, compile_count=compile_count, trace_hits=trace_hits,
        backend=backend, platforms=platforms, device_id=device_id)
    if is_jvp:
        flat_meta = stable_leaf_meta(primal, "primal")
        flat_meta.update(stable_leaf_meta(tangent_out, "tangent"))
    else:
        flat_meta = stable_leaf_meta(result, "out")
    dispatch["leaves"] = [{"path": key, **value} for key, value in flat_meta.items()]
    dispatch["tree"] = tree_signature(result)
    last_dispatch = dispatch
    _LOG.info(
        "jax_lax_while_loop %s/%s path=so route=host_step abi_calls=%d iterations=%d "
        "cond_hits=%d plan_id=%s cache=%s token=%s",
        mode, variant, abi_calls, final_counters["body_hits"], final_counters["cond_hits"],
        plan_id, plan_cache, dispatch["runtime_token"])
    return result


def _container_device_id() -> int:
    raw = os.environ.get("OPX_DEVICE_ID", "").strip()
    if raw.isdigit():
        return int(raw)
    # nm.sh 写入的身份文件：容器内逻辑设备号
    identity = OP_ROOT.parent.parent / ".opx-run-identity.json"
    try:
        payload = json.loads(identity.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    value = payload.get("container_device_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------
def while_loop(cond_fun, body_fun, init_carry, *, safety_limit=None, tangent=None):
    """eager 语义的 ``lax.while_loop``：返回与 ``init_carry`` 同结构的最终 carry。"""
    return _execute(cond_fun, body_fun, init_carry, mode="primal", tangent=tangent,
                    safety_limit=safety_limit, compiled=False)


def while_loop_compiled(cond_fun, body_fun, init_carry, *, mode="primal", tangent=None,
                        safety_limit=None, jit=True):
    """编译态入口（测试/门禁）：``mode="primal"`` 返回 carry；``mode="jvp"`` 返回
    ``(primal, tangent)``。两条路径都经宿主 C ABI 在设备上执行。"""
    if mode not in ("primal", "jvp"):
        raise WhileLoopError("invalid_argument", f"mode={mode!r} 只能是 primal/jvp")
    return _execute(cond_fun, body_fun, init_carry, mode=mode, tangent=tangent,
                    safety_limit=safety_limit, compiled=bool(jit))


__all__ = [
    "OP_NAME", "KERNEL_NAME", "NPU_ARCH", "HOST_LIB", "ABI_TAG", "REJECT_CODES",
    "WhileLoopError", "JaxLaxWhileLoopArgs", "while_loop", "while_loop_compiled",
    "last_dispatch", "library_fingerprint", "abi_version", "stable_flatten", "leaf_meta",
    "tree_signature", "stable_leaf_meta",
]
