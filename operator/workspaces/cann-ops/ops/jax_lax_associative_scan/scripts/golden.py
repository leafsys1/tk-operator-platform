# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan 的独立参考实现（oracle = 同版本 JAX CPU）。

**本模块不修改任何环境变量**，也不 import 候选模块 / ``torch`` / ``torch_npu``；CPU 后端的
断言与 ``JAX_PLATFORMS=cpu`` 由父进程在**独立子进程**环境里设置（见 ``golden_cli.py`` 与
设计 §2.1.1）。这样候选（NPU）与 golden（CPU）各自持有正确的 backend，不会互相污染。

调用契约（设计 §2.1 / §2.1.3）：

``compute_golden(case, fixture)``
    ``jax.lax.associative_scan(combine, elems, reverse=..., axis=...)``；返回
    ``{"out": ...}`` 或 ``{"out/a": ..., "out/b": ...}``（与候选 ``outputs.npz`` 同键名）。

``compute_golden_autodiff(case, fixture)``
    单叶子 F7：``f_ad(e) = lax.associative_scan(jnp.add, e, reverse, axis)`` 的 JVP/VJP；
    返回 ``{"jvp_primal", "jvp_tangent", "vjp_primal", "vjp_cotangent"}``。

``expect_reject(case, fixture)``
    证明用例**按已确认规格**必须被拒绝：dtype 不是 float32 → ``dtype_mismatch``；
    axis 非整数/越界 → ``invalid_axis``。dtype 规则来自 ``spec.json:17,21``（JAX 自身
    支持 float16，候选仍必须按规格拒绝，因此该条不能靠“JAX 也报错”证明）；axis 规则
    来自 ``spec.json:19``，另外用 CPU JAX 交叉核对它同样报错（证据写在 ``reject_evidence``）。
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

import fixtures

LEGACY_SIGNATURE_MESSAGE = (
    "jax_lax_associative_scan golden 无 (a,b) 形式；请用 compute_golden(case, fixture)。"
    "共享 runner 的 elementwise 分支与 assoc 分支必须按 op 显式分派。"
)
EXPECTED_JAX_VERSION = "0.8.1"


def _require_mapping(case, fixture) -> None:
    if not isinstance(case, Mapping) or not isinstance(fixture, Mapping):
        raise TypeError(LEGACY_SIGNATURE_MESSAGE)


def _to_jax(value):
    arr = np.asarray(value)
    if arr.dtype == np.float32:
        return jnp.asarray(arr, dtype=jnp.float32)
    if arr.dtype == np.float16:
        return jnp.asarray(arr, dtype=jnp.float16)
    raise TypeError(f"golden 只支持 float32/float16 叶子，收到 {arr.dtype}")


def build_elems(fixture: dict, convert=_to_jax):
    """按 ``elems_tree`` 还原输入 pytree（单叶子或 dict）。"""
    return fixtures.tree_from_spec(fixture["elems_tree"], fixture["arrays"], convert=convert)


def _check_versions() -> None:
    import jaxlib  # noqa: PLC0415

    if getattr(jax, "__version__", None) != EXPECTED_JAX_VERSION or \
            getattr(jaxlib, "__version__", None) != EXPECTED_JAX_VERSION:
        raise RuntimeError(
            f"GOLDEN_VERSION_MISMATCH jax={getattr(jax, '__version__', None)} "
            f"jaxlib={getattr(jaxlib, '__version__', None)}（已确认基线 {EXPECTED_JAX_VERSION}）")


def compute_golden(case: Mapping, fixture: Mapping) -> dict:
    """CPU ``lax.associative_scan``：返回按输出路径命名的数组字典。"""
    _require_mapping(case, fixture)
    _check_versions()
    fn = fixtures.combine_fn_for(fixture["combine_name"])
    elems = build_elems(fixture)
    result = jax.lax.associative_scan(
        fn, elems, reverse=bool(fixture["reverse"]), axis=int(fixture["axis"]))
    flat = dict(fixtures.flatten_with_paths(result, fixtures.OUT_PATH))
    expected = set(fixtures.output_paths_of(fixture))
    if set(flat) != expected:
        raise RuntimeError(f"golden 输出键 {sorted(flat)} != 期望 {sorted(expected)}")
    return flat


def compute_golden_autodiff(case: Mapping, fixture: Mapping) -> dict:
    """CPU JVP/VJP（仅 F7 单叶子 sum；公式见设计 §2.1.3）。"""
    _require_mapping(case, fixture)
    _check_versions()
    if len(fixture["leaves"]) != 1:
        raise ValueError("AD 只定义在单叶子用例上")
    if fixture["combine_name"] != "sum":
        raise ValueError(f"AD 用例的 op_kind 必须是 sum，收到 {fixture['combine_name']!r}")
    x = build_elems(fixture)
    axis = int(fixture["axis"])
    reverse = bool(fixture["reverse"])

    def f_ad(e):
        return jax.lax.associative_scan(jnp.add, e, reverse=reverse, axis=axis)

    tangent_seed = jnp.ones_like(x)
    jvp_primal, jvp_tangent = jax.jvp(f_ad, (x,), (tangent_seed,))
    vjp_primal, vjp_fn = jax.vjp(f_ad, x)
    cotangent_seed = jnp.ones_like(vjp_primal)
    vjp_cotangent = vjp_fn(cotangent_seed)[0]
    return {
        "jvp_primal": jvp_primal,
        "jvp_tangent": jvp_tangent,
        "vjp_primal": vjp_primal,
        "vjp_cotangent": vjp_cotangent,
    }


def expect_reject(case: Mapping, fixture: Mapping) -> str:
    """按已确认规格确认该用例必须被拒绝，返回稳定拒绝码。

    只覆盖本批 L2 已确认的两类：dtype 非 float32、axis 越界。其他情形显式报错，
    避免把“用例写错”当成“候选该拒绝”。
    """
    _require_mapping(case, fixture)
    dtype = str(fixture["dtype"])
    if dtype != "float32":
        return "dtype_mismatch"
    if not fixture.get("valid_axis", True):
        return "invalid_axis"
    raise RuntimeError(
        f"golden 未发现 {case.get('case_id')!r} 违反已确认规格（dtype={dtype}, "
        f"axis={fixture.get('axis')!r}, valid_axis={fixture.get('valid_axis')!r}）")


def reject_evidence(case: Mapping, fixture: Mapping) -> dict:
    """拒绝用例的补充证据（写进 golden.json；不改变 ``expect_reject`` 的返回值）。"""
    evidence: dict = {"dtype": fixture["dtype"], "axis": fixture.get("axis"),
                      "valid_axis": fixture.get("valid_axis")}
    code = expect_reject(case, fixture)
    evidence["rule"] = "spec_dtype_float32" if code == "dtype_mismatch" else "spec_axis_range"
    if code == "invalid_axis":
        try:
            elems = build_elems(fixture)
            jax.lax.associative_scan(
                fixtures.combine_fn_for(fixture["combine_name"]), elems,
                reverse=bool(fixture["reverse"]), axis=int(fixture["axis"]))
        except Exception as exc:  # noqa: BLE001 —— 这里只要“JAX 也拒绝”的证据
            evidence["jax_error"] = f"{type(exc).__name__}: {exc}"
            evidence["jax_rejects"] = True
        else:
            evidence["jax_rejects"] = False
    else:
        # JAX 本身支持 float16；拒绝依据是已确认规格的 dtype 限制，不是 oracle 报错。
        evidence["jax_rejects"] = None
        evidence["note"] = "JAX 支持 float16；本用例按已确认规格上限 float32 由候选拒绝"
    evidence["reject_code"] = code
    return evidence


if __name__ == "__main__":  # 自检：只跑 CPU（不设置环境变量）
    demo = {"case_id": "assoc_sum_len8_axis0", "shape": [8, 16],
            "axis": 0, "reverse": False, "dtype": "float32", "seed": 20261026}
    fixture = fixtures.make_fixture(demo)
    out = compute_golden(demo, fixture)
    print("backend:", jax.default_backend())
    print("out:", np.asarray(out["out"]).shape, np.asarray(out["out"])[:2, :2])
