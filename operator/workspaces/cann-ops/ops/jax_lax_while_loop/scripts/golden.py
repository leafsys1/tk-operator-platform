# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop 的独立参考实现（oracle = 同版本 JAX 0.8.1 CPU）。

**本模块不修改任何环境变量**，也不 import 候选模块 / ``torch`` / ``torch_npu``；CPU 后端断言与
``JAX_PLATFORMS=cpu`` 由父进程在**独立子进程**环境里设置（见 ``golden_cli.py`` 与设计 §2.1）。
候选（NPU）与 golden（CPU）各自持有正确的 backend。

调用契约（设计 §2.1 / §5.5）：

``compute_golden(case, variant, mode, arrays)``
    返回 ``{输出键: numpy 数组}``；普通模式键为 ``out/...``，JVP 为 ``primal/...`` 与
    ``tangent/...``。``mode ∈ eager/jit/jvp_eager/jvp_jit``。

``compute_reject(case, variant, arrays)``
    让 JAX 在 CPU 上**真实执行**同一非法输入，捕获上游异常并保留类型/消息/sha256；
    唯一例外是 c16 ``ITERATION_LIMIT_EXCEEDED``：CPU JAX 没有安全迭代上限参数，
    **不在 CPU 执行 always_true**，只按已确认 spec 记录 ``spec_rule_rejects=True``，
    不伪造上游异常（设计 §2.1 / §5.3）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

import fixtures

EXPECTED_JAX_VERSION = "0.8.1"
JVP_MODES = ("jvp_eager", "jvp_jit")
FLOAT0_DTYPE = np.dtype(jax.dtypes.float0)


def _check_versions() -> None:
    import jaxlib  # noqa: PLC0415

    jax_version = getattr(jax, "__version__", None)
    jaxlib_version = getattr(jaxlib, "__version__", None)
    if jax_version != EXPECTED_JAX_VERSION or jaxlib_version != EXPECTED_JAX_VERSION:
        raise RuntimeError(
            f"GOLDEN_VERSION_MISMATCH jax={jax_version} jaxlib={jaxlib_version}"
            f"（已确认基线 {EXPECTED_JAX_VERSION}）")


def _dtype_text(dtype) -> str:
    text = str(dtype)
    if "float0" in text:
        return "[('float0', 'V')]"
    return text


def _sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(str(list(arr.shape)).encode("utf-8"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


def leaf_meta(value) -> dict:
    """jax.Array / numpy 叶子的结构元数据（weak_type 在转 numpy 前读取）。"""
    weak_type = getattr(value, "weak_type", None)
    arr = np.asarray(value)
    return {
        "dtype": _dtype_text(arr.dtype),
        "shape": [int(x) for x in arr.shape],
        "weak_type": bool(weak_type) if weak_type is not None else None,
        "sha256": _sha256_array(arr),
    }


def flatten(value, prefix: str) -> dict[str, np.ndarray]:
    """稳定叶子路径：dict 按插入顺序、tuple/list 按位置、裸叶子为 prefix。"""
    out: dict[str, np.ndarray] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, Mapping):
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


def flatten_meta(value, prefix: str) -> dict[str, dict]:
    out: dict[str, dict] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, Mapping):
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


# ---------------------------------------------------------------------------
# 树重建
# ---------------------------------------------------------------------------


def build_value(spec: dict, arrays: dict, *, case: dict, variant: str):
    """按树描述从输入 npz 重建 JAX pytree；S1 的 weak/strong 在此精确复现。"""
    kind = spec["kind"]
    if kind == "leaf":
        raw = arrays[spec["path"]]
        dtype = spec["dtype"]
        if "float0" in dtype:
            return jnp.asarray(np.asarray(raw), dtype=jax.dtypes.float0)
        if dtype == "int32":
            return jnp.asarray(raw, dtype=jnp.int32)
        if str(case.get("case_id")) == "while_sup_weak_type":
            if variant == "weak":
                return jnp.asarray(float(np.asarray(raw)))
            return jnp.asarray(raw, dtype=jnp.float32)
        return jnp.asarray(raw, dtype=jnp.float32)
    if kind == "tuple":
        return tuple(build_value(child, arrays, case=case, variant=variant)
                     for child in spec["children"])
    if kind == "dict":
        return {key: build_value(spec["children"][key], arrays, case=case, variant=variant)
                for key in spec["keys"]}
    raise ValueError(f"未知树节点：{kind!r}")


def build_init(case: dict, variant: str, arrays: dict):
    return build_value(fixtures.tree_spec(case, variant), arrays, case=case, variant=variant)


def build_tangent(case: dict, variant: str, arrays: dict):
    spec = fixtures.tangent_spec(case, variant)
    if spec is None:
        raise ValueError(f"{case.get('case_id')}: 没有 JVP 切向量")
    return build_value(spec, arrays, case=case, variant=variant)


# ---------------------------------------------------------------------------
# cond/body 的确定性绑定（与设计 §5.5 一一对应）
# ---------------------------------------------------------------------------


def cond_body(case: dict, variant: str):
    case_id = str(case.get("case_id"))
    if case_id in ("while_zero_iterations", "while_one_iteration"):
        limit = 0 if case_id == "while_zero_iterations" else 1
        if case_id == "while_zero_iterations":
            return (lambda c: c[0] < 0), (lambda c: (c[0] + 1, c[1] + 1.0))
        return (lambda c: c[0] < limit), (lambda c: (c[0] + 1, c[1] * 1.25 + 0.5))
    if case_id == "while_dynamic_multi_iterations":
        return (lambda c: c[0] < c[1]), \
               (lambda c: (c[0] + 1, c[1], c[2] * 1.03125 + 0.125))
    if case_id == "while_scalar_carry":
        return (lambda c: c < 9), (lambda c: c + 2)
    if case_id == "while_tuple_carry":
        return (lambda c: c[0] < 5), \
               (lambda c: (c[0] + 1, c[1] + jnp.mean(c[2]), c[2] * 0.875))
    if case_id == "while_nested_dict_carry":
        return (lambda c: c["step"] < 4), \
               (lambda c: {"step": c["step"] + 1,
                           "state": (c["state"][0] * 0.875 + 0.125,
                                     {"aux": c["state"][1]["aux"] * 0.5 + 0.25})})
    if case_id in ("while_jit_runtime_limit", "while_iteration_token_and_hits"):
        return (lambda c: c[0] < c[1]), \
               (lambda c: (c[0] + 1, c[1], c[2] * 1.03125 + 0.125))
    if case_id == "while_jvp_multi_iteration":
        return (lambda c: c[0] < 6), (lambda c: (c[0] + 1, c[1] * 1.125 + 0.25))
    if case_id == "while_jvp_zero_iteration":
        return (lambda c: c[0] < 0), (lambda c: (c[0] + 1, c[1] * 1.125 + 0.25))
    if case_id == "while_unaligned_tail_widths":
        return (lambda c: c[0] < 5), (lambda c: (c[0] + 1, c[1] * 0.5 + 0.25))
    if case_id == "while_large_tensor":
        return (lambda c: c[0] < 12), \
               (lambda c: (c[0] + 1, c[1] * 1.0009765625 + 0.03125))
    if case_id in ("while_short_small_carry",):
        return (lambda c: c[0] < 4), (lambda c: (c[0] + 1, c[1] * 1.03125 + 0.125))
    if case_id in ("while_long_large_tensor",):
        return (lambda c: c[0] < 64), \
               (lambda c: (c[0] + 1, c[1] * 1.0009765625 + 0.03125))
    if case_id == "while_sup_weak_type":
        return (lambda c: c < 3), (lambda c: c + 1.0)
    if case_id == "while_sup_nonfinite":
        return (lambda c: c[0] < 2), (lambda c: (c[0] + 1, c[1] * 1.0))
    if case_id == "while_sup_tensor_carry_data_cond":
        return (lambda c: jnp.all(c > 0.125)), (lambda c: c * 0.5)
    if case_id == "while_invalid_condition":
        table = {
            "bool1": lambda c: jnp.zeros((1,), dtype=bool),
            "bool2": lambda c: jnp.zeros((2,), dtype=bool),
            "int_scalar": lambda c: jnp.int32(0),
            "float_scalar": lambda c: jnp.float32(0.0),
        }
        return table[variant], (lambda c: c)
    if case_id == "while_carry_structure_mismatch":
        return (lambda c: jnp.bool_(True)), \
               (lambda c: {"a": c[0], "b": c[1]})
    if case_id == "while_carry_shape_or_dtype_mismatch":
        if variant == "shape":
            return (lambda c: jnp.bool_(True)), \
                   (lambda c: jnp.concatenate([c, c[:1]]))
        return (lambda c: jnp.bool_(True)), (lambda c: c.astype(jnp.int32))
    if case_id == "while_iteration_limit_exceeded":
        return (lambda c: jnp.bool_(True)), (lambda c: (c[0] + 1, c[1] + 1.0))
    raise ValueError(f"{case_id}: 没有 cond/body 绑定")


# ---------------------------------------------------------------------------
# 数值 golden
# ---------------------------------------------------------------------------


def _evaluate(case: dict, variant: str, mode: str, arrays: dict):
    """执行一次 CPU JAX 语义，返回**未 flatten** 的真实 pytree（弱类型/容器类型都还在）。"""
    _check_versions()
    init = build_init(case, variant, arrays)
    cond, body = cond_body(case, variant)
    case_id = str(case.get("case_id"))

    def loop(carry):
        return jax.lax.while_loop(cond, body, carry)

    if mode in JVP_MODES:
        tangent = build_tangent(case, variant, arrays)
        if mode == "jvp_jit":
            primal, tang = jax.jit(lambda c: jax.jvp(loop, (c,), (tangent,)))(init)
        else:
            primal, tang = jax.jvp(loop, (init,), (tangent,))
        return primal, tang
    if mode == "jit":
        return jax.jit(loop)(init)
    if mode == "eager":
        return loop(init)
    raise ValueError(f"{case_id}: 未知识别 mode {mode!r}")


def _flatten_result(result, mode: str) -> dict[str, np.ndarray]:
    if mode in JVP_MODES:
        primal, tang = result
        out = flatten(primal, "primal")
        out.update(flatten(tang, "tangent"))
        return out
    return flatten(result, "out")


def _meta_result(result, mode: str) -> dict[str, dict]:
    if mode in JVP_MODES:
        primal, tang = result
        meta = flatten_meta(primal, "primal")
        meta.update(flatten_meta(tang, "tangent"))
        return meta
    return flatten_meta(result, "out")


def compute_payload(case: dict, variant: str, mode: str, arrays: dict) -> dict:
    """一次执行同时产出 ``arrays``/``leaves``/``tree``（weak_type 在转 numpy 前读取）。

    ``tree`` 是 flatten 前的容器指纹（``fixtures.tree_signature``），用于区分
    tuple 与 list；golden_cli 把它写进 ``golden.json`` 供 ``judge_outputs`` 精确比较。
    """
    result = _evaluate(case, variant, mode, arrays)
    return {
        "arrays": _flatten_result(result, mode),
        "leaves": _meta_result(result, mode),
        "tree": fixtures.tree_signature(result),
    }


def compute_golden(case: dict, variant: str, mode: str, arrays: dict) -> dict[str, np.ndarray]:
    return compute_payload(case, variant, mode, arrays)["arrays"]


def compute_meta(case: dict, variant: str, mode: str, arrays: dict) -> dict:
    """兼容入口：只返回叶子元数据（含 weak_type）。"""
    return compute_payload(case, variant, mode, arrays)["leaves"]


def compute_tree(case: dict, variant: str, mode: str, arrays: dict):
    """兼容入口：只返回 flatten 前的容器指纹。"""
    return compute_payload(case, variant, mode, arrays)["tree"]


# ---------------------------------------------------------------------------
# 拒绝证据
# ---------------------------------------------------------------------------


def _capture_exception(case: dict, variant: str, arrays: dict) -> dict:
    init = build_init(case, variant, arrays)
    cond, body = cond_body(case, variant)
    try:
        jax.lax.while_loop(cond, body, init)
    except BaseException as exc:  # noqa: BLE001 —— 上游异常类型/摘要必须原样保留
        message = str(exc)
        return {
            "cpu_rejects": True,
            "upstream_exception_type": type(exc).__name__,
            "upstream_message": message,
            "upstream_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest()
            if message else None,
            "spec_rule_rejects": False,
            "spec_rule": None,
        }
    return {
        "cpu_rejects": False,
        "upstream_exception_type": None,
        "upstream_message": "",
        "upstream_message_sha256": None,
        "spec_rule_rejects": False,
        "spec_rule": None,
    }


def compute_reject(case: dict, variant: str, arrays: dict) -> dict:
    """返回拒绝证据；c16 不执行 CPU（会死循环），只记已确认规格规则。"""
    _check_versions()
    case_id = str(case.get("case_id"))
    expected_code = fixtures.expected_reject_code(case, variant)
    if case_id == "while_iteration_limit_exceeded":
        return {
            "reject_code": expected_code,
            "cpu_rejects": False,
            "upstream_exception_type": None,
            "upstream_message": "",
            "upstream_message_sha256": None,
            "spec_rule_rejects": True,
            "spec_rule": "iteration_safety_limit",
            "reject_basis": "spec_rule",
            "runtime_evidence_required": {"path": "so", "iterations": 64,
                                          "cond_hits": 64, "body_hits": 64,
                                          "limit_flag": 1},
            "note": "JAX lax.while_loop 无安全迭代上限；CPU golden 不执行 always_true",
        }
    evidence = _capture_exception(case, variant, arrays)
    if not evidence["cpu_rejects"]:
        # CPU 0.8.1 对非标量/非 bool cond 及 carry 元数据漂移均抛 TypeError；
        # 若未来版本改为接受，只有 INVALID_CONDITION 可按已确认 spec 的 scalar-bool
        # 规则继续拒绝，不伪造上游异常。
        if expected_code == "INVALID_CONDITION":
            evidence["spec_rule_rejects"] = True
            evidence["spec_rule"] = "scalar_bool_cond"
            evidence["reject_basis"] = "spec_rule"
        else:
            evidence["reject_basis"] = None
    else:
        evidence["reject_basis"] = "cpu_exception"
    evidence["reject_code"] = expected_code
    evidence["runtime_evidence_required"] = None
    return evidence


__all__ = [
    "EXPECTED_JAX_VERSION", "JVP_MODES", "FLOAT0_DTYPE",
    "leaf_meta", "flatten", "flatten_meta",
    "build_value", "build_init", "build_tangent", "cond_body",
    "compute_payload", "compute_golden", "compute_meta", "compute_tree", "compute_reject",
]
