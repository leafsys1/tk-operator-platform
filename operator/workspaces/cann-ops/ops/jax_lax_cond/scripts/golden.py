# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond 的独立参考实现（oracle = 同版本 JAX 0.8.1 CPU）。

**本模块不修改任何环境变量**，也不 import 候选模块 / ``torch`` / ``torch_npu``；CPU 后端断言与
``JAX_PLATFORMS=cpu`` 由父进程在**独立子进程**环境里设置（见 ``golden_cli.py`` 与设计 §2.1.1）。
候选（NPU）与 golden（CPU）各自持有正确的 backend。

调用契约（设计 §2.1 / §2.1.3）：

``compute_golden(case, fixture)``
    按 ``fixture["mode"]`` 走 eager / jit（含动态谓词），返回
    ``{输出路径: 数组}``（与候选 ``outputs.npz`` 同键名）。

``compute_golden_grad(case, fixture)``
    仅 ``grad_eager`` / ``grad_jit``：``grad_x sum(cond(pred, true, false, x))``；返回 ``{"grad": ...}``。

``expect_reject(case, fixture)``
    让 JAX 在 CPU 上**真实执行**同一输入，捕获上游异常并保留类型/消息/sha256；
    bool 非标量、结构/shape/dtype 不一致必须由 JAX 抛错。整型 pred 在 JAX 0.8.1 上会被
    按真值接受（不抛错），本函数如实记录 ``cpu_rejects=False``；但已确认 spec 的
    scalar-bool 规则仍要求 ``INVALID_PREDICATE``，故同时记 ``spec_rule_rejects=True``，
    由 CLI 按“规格规则拒绝”成功返回，拒绝依据与 CPU 原始行为分开留证。
    **不修改 cases/期望，也不伪造上游异常**（设计 §1.3.7、§7.5）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

import fixtures

LEGACY_SIGNATURE_MESSAGE = (
    "jax_lax_cond golden 无 (a,b) 形式；请用 compute_golden(case, fixture)。"
    "共享 runner 的 elementwise 分支与 cond 分支必须按 op 显式分派。"
)
EXPECTED_JAX_VERSION = "0.8.1"

GRAD_MODES = ("grad_eager", "grad_jit")
JIT_MODES = ("jit", "jit_true", "jit_false")


def _require_mapping(case, fixture) -> None:
    if not isinstance(case, Mapping) or not isinstance(fixture, Mapping):
        raise TypeError(LEGACY_SIGNATURE_MESSAGE)


def _check_versions() -> None:
    import jaxlib  # noqa: PLC0415

    if getattr(jax, "__version__", None) != EXPECTED_JAX_VERSION or \
            getattr(jaxlib, "__version__", None) != EXPECTED_JAX_VERSION:
        raise RuntimeError(
            f"GOLDEN_VERSION_MISMATCH jax={getattr(jax, '__version__', None)} "
            f"jaxlib={getattr(jaxlib, '__version__', None)}（已确认基线 {EXPECTED_JAX_VERSION}）")


def build_pred(fixture: Mapping):
    """按 fixture 的 pred 对象重建；jax family 的 0-d bool 转成 jax 标量。"""
    pred = fixture["pred"]
    if fixture.get("family") == "jax":
        return jnp.asarray(pred)
    return pred


def _convert_leaf(fixture: Mapping, value):
    if fixture.get("family") != "jax":
        return jnp.asarray(value, dtype=jnp.float32)
    variant = str(fixture.get("variant") or "")
    if variant == "weak":
        # Python float -> JAX weak type（设计 §2.1.3 S1）
        return jnp.asarray(float(np.asarray(value)))
    return jnp.asarray(value, dtype=jnp.float32)


def build_operands(fixture: Mapping):
    return fixtures.build_operands(fixture, convert=lambda value: _convert_leaf(fixture, value))


def compute_golden(case: Mapping, fixture: Mapping) -> dict:
    """CPU ``lax.cond``：返回按输出路径命名的数组字典。"""
    _require_mapping(case, fixture)
    _check_versions()
    pred = build_pred(fixture)
    operands = (build_operands(fixture),)
    true_fn = fixtures.branch_fn(fixture["true_name"])
    false_fn = fixtures.branch_fn(fixture["false_name"])
    mode = str(fixture["mode"])
    if mode in GRAD_MODES:
        raise ValueError(f"{mode} 必须走 compute_golden_grad，不能复用数值路径")

    def body(p, *args):
        return jax.lax.cond(p, true_fn, false_fn, *args)

    if mode in JIT_MODES and not isinstance(pred, (bool, np.bool_)):
        result = jax.jit(body)(pred, *operands)
    else:
        result = body(pred, *operands)

    flat = dict(fixtures.flatten_with_paths(result, "out"))
    expected = dict(fixture["output_shapes"])
    if set(flat) != set(expected):
        raise RuntimeError(f"golden 输出键 {sorted(flat)} != 期望 {sorted(expected)}")
    return flat


def compute_golden_grad(case: Mapping, fixture: Mapping) -> dict:
    """CPU AD（仅 float32 单叶子 operand；pred 是静态 bool，不参与求导）。"""
    _require_mapping(case, fixture)
    _check_versions()
    if len(fixture["leaves"]) != 1:
        raise ValueError("AD 只定义在单叶子用例上")
    pred = build_pred(fixture)
    x = _convert_leaf(fixture, fixture["arrays"][fixture["leaves"][0]["path"]])
    true_fn = fixtures.branch_fn(fixture["true_name"])
    false_fn = fixtures.branch_fn(fixture["false_name"])

    def loss(value):
        return jnp.sum(jax.lax.cond(pred, true_fn, false_fn, value))

    mode = str(fixture["mode"])
    if mode == "grad_jit":
        grad = jax.jit(jax.grad(loss))(x)
    elif mode == "grad_eager":
        grad = jax.grad(loss)(x)
    else:
        raise ValueError(f"{mode} 不是 grad mode")
    return {"grad": grad}


def _message_sha256(message: str) -> str:
    return hashlib.sha256((message or "").encode("utf-8")).hexdigest()


def _message_summary(message: str, limit: int = 200) -> str:
    first = (message or "").strip().splitlines()[0] if (message or "").strip() else ""
    return first.strip()[:limit]


#: 每个拒绝码的诊断关键词（仅诊断，不作 PASS/FAIL 依据；设计 §1.3.7）
KEYWORDS = {
    "INVALID_PREDICATE": ("scalar", "bool", "predicate", "pred"),
    "BRANCH_STRUCTURE_MISMATCH": ("structure", "tree", "pytree", "identical", "types"),
    "BRANCH_SHAPE_MISMATCH": ("shape", "identical", "types"),
    "BRANCH_DTYPE_MISMATCH": ("dtype", "type", "identical", "types"),
}


def _is_int32_pred(fixture: Mapping) -> bool:
    """c13 的 int32(0/1) 变体：pred_kind 由 fixtures 冻结为 ``int32_scalar``。"""
    return str(fixture.get("pred_kind", "")).startswith("int32")


def expect_reject(case: Mapping, fixture: Mapping) -> dict:
    """在 CPU 上真实执行同输入，记录上游异常与规格规则；不把“JAX 接受”伪造成拒绝。

    两类拒绝必须分开记录：

    * **CPU 可确认**（bool[1]/bool[2]、结构/shape/dtype 不一致）：JAX 0.8.1 CPU 抛
      ``TypeError``，``cpu_rejects=True``，上游类型/消息/sha256 逐项保留。
    * **规格规则拒绝**（仅 c13 的 ``int32(0/1)``）：JAX 0.8.1 CPU 会按真值接受数值 pred
      （``cpu_rejects=False``，没有异常可保留）；但已确认 spec 明确规定整型 0/1 不做隐式
      提升、归一化为 ``INVALID_PREDICATE``。此处把“CPU 原始行为”和“规格规则”同时写进
      证据，``spec_rule_rejects=True`` 表示拒绝依据是规格而非 CPU 异常。本函数不修改
      ``cases``/期望，也不伪造上游异常。
    """
    _require_mapping(case, fixture)
    _check_versions()
    spec_code = str(case.get("expected_error") or "")
    if spec_code not in KEYWORDS:
        raise ValueError(f"未知 expected_error：{spec_code!r}")
    pred = build_pred(fixture)
    operands = (build_operands(fixture),)
    true_fn = fixtures.branch_fn(fixture["true_name"])
    false_fn = fixtures.branch_fn(fixture["false_name"])
    int_spec_rule = _is_int32_pred(fixture) and spec_code == "INVALID_PREDICATE"
    evidence = {
        "spec_code": spec_code,
        "pred_kind": fixture.get("pred_kind"),
        "family": fixture.get("family"),
        "cpu_rejects": False,
        "cpu_exception_type": None,
        "cpu_message": "",
        "cpu_message_sha256": None,
        "cpu_message_summary": "",
        "upstream_keyword_hit": False,
        # 规格规则拒绝：CPU 没有异常，但 spec v1 仍要求拒绝（仅 int32 pred 的
        # INVALID_PREDICATE）。非 int 用例必须由 CPU 真实异常确认。
        "spec_rule_rejects": int_spec_rule,
        "reject_basis": "spec_rule" if int_spec_rule else "cpu_exception",
    }
    try:
        jax.lax.cond(pred, true_fn, false_fn, *operands)
    except Exception as exc:  # noqa: BLE001 —— 这里要的就是上游真实异常
        message = str(exc)
        evidence.update(
            cpu_rejects=True,
            cpu_exception_type=type(exc).__name__,
            cpu_message=message,
            cpu_message_sha256=_message_sha256(message),
            cpu_message_summary=_message_summary(message),
            # 即使 int32 pred 出现异常，也以真实 CPU 异常为准。
            spec_rule_rejects=False,
            reject_basis="cpu_exception",
        )
        lowered = message.lower()
        evidence["upstream_keyword_hit"] = any(
            token in lowered for token in KEYWORDS[spec_code])
    else:
        if int_spec_rule:
            evidence.update(
                spec_rule_rejects=True,
                reject_basis="spec_rule",
                spec_rule="scalar_bool_pred",
                cpu_raw_evidence="JAX 0.8.1 CPU accepts int32 pred by truthiness; "
                                 "no exception is raised, so cpu_rejects=false.",
                note=("JAX 0.8.1 CPU 未拒绝 int32 pred（按真值接受）；"
                      "spec v1 的 scalar-bool 规则仍要求 INVALID_PREDICATE。"
                      "此处保留 CPU 无异常的原始证据，拒绝依据记为 spec_rule，"
                      "不修改 cases/期望，也不伪造上游异常。"),
            )
        else:
            evidence.update(
                spec_rule_rejects=False,
                reject_basis="unconfirmed",
                note=("JAX 0.8.1 CPU 未拒绝该输入；该用例不是已确认的规格规则拒绝，"
                      "缺少 CPU 上游异常证据，必须记 ERROR。"),
            )
    return evidence


def reject_evidence(case: Mapping, fixture: Mapping) -> dict:
    """拒绝用例的完整证据（写进 golden.json；CLI 据 ``cpu_rejects``/``spec_rule_rejects`` 决定退出码）。"""
    evidence = expect_reject(case, fixture)
    evidence["spec_rule"] = evidence.get("spec_rule") or {
        "INVALID_PREDICATE": "spec_scalar_bool_pred",
        "BRANCH_STRUCTURE_MISMATCH": "spec_branch_tree_structure",
        "BRANCH_SHAPE_MISMATCH": "spec_branch_leaf_shape",
        "BRANCH_DTYPE_MISMATCH": "spec_branch_leaf_dtype",
    }[evidence["spec_code"]]
    return evidence


if __name__ == "__main__":  # 自检：只跑 CPU（不设置环境变量）
    demo = {"case_id": "cond_true_eager", "pred": True, "operand": "float32[8,16]",
            "branches": ["x + 1", "x - 1"], "mode": ["eager"], "expected": "match_cpu"}
    fixture = fixtures.make_fixture(demo, "", "eager")
    out = compute_golden(demo, fixture)
    print("backend:", jax.default_backend())
    print("out:", np.asarray(out["out"]).shape, np.asarray(out["out"])[:2, :2])
