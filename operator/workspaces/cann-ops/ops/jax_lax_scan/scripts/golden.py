# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan 的独立参考实现（oracle = 同版本 JAX CPU）。

**本模块不修改任何环境变量**，也不 import 候选模块 / ``torch_npu``；CPU 后端的断言与
``JAX_PLATFORMS=cpu`` 由父进程在**独立子进程**环境里设置（见 ``golden_cli.py`` 与设计 §2.1.1）。
这样候选（NPU）与 golden（CPU）各自持有正确的 backend，不会互相污染。

调用契约（设计 §2.1.2，必须与共享 runner 的 elementwise 分支区分开）：

``compute_golden(case, fixture)``
    scan 语义：``case: dict`` + ``fixture: dict`` → ``{"final_carry": pytree, "ys": Array}``。
    两个位置参数都必须是 Mapping；``compute_golden(a, b)`` 的两输入形式**故意不支持**，
    分派错误时立即 ``TypeError``，避免把 scan 当成 elementwise 静默跑出伪结果。

``compute_golden_autodiff(case, fixture)``
    ``f_ad(xs) = lax.scan(B4_quad, zeros, xs)[0]`` 的 JVP/VJP：
    ``{"jvp_primal", "jvp_tangent", "vjp_primal", "vjp_cotangent"}``。

``compute_golden_expect_reject(case, fixture)``
    证明用例本身非法：CPU ``lax.scan`` 也必须抛 carry 不变式错误，返回稳定码
    ``carry_shape_mismatch``；若 golden 没拒绝则抛 ``RuntimeError``（用例写错）。
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

import fixtures

LEGACY_SIGNATURE_MESSAGE = (
    "jax_lax_scan golden 无 (a,b) 形式；请用 compute_golden(case, fixture)。"
    "共享 runner 的 elementwise 分支与 scan 分支必须按 op 显式分派。"
)


def _require_mapping(case, fixture) -> None:
    if not isinstance(case, Mapping) or not isinstance(fixture, Mapping):
        raise TypeError(LEGACY_SIGNATURE_MESSAGE)


def _to_jax(value):
    arr = np.asarray(value)
    if arr.dtype == np.int32:
        return jnp.asarray(arr, dtype=jnp.int32)
    if arr.dtype == np.float32:
        return jnp.asarray(arr, dtype=jnp.float32)
    raise TypeError(f"golden 只支持 float32/int32 叶子，收到 {arr.dtype}")


def _build_inputs(fixture: dict):
    arrays = fixture["arrays"]
    init = fixtures.tree_from_spec(fixture["carry_tree"], arrays, convert=_to_jax)
    if fixture["xs_tree"] is None:
        xs = None
    else:
        xs = fixtures.tree_from_spec(fixture["xs_tree"], arrays, convert=_to_jax)
    return init, xs


def compute_golden(case: Mapping, fixture: Mapping) -> dict:
    """CPU ``lax.scan``：返回 final_carry（pytree）与 stacked ys。"""
    _require_mapping(case, fixture)
    body = fixtures.get_body(fixture["body_id"])
    init, xs = _build_inputs(fixture)
    final_carry, ys = jax.lax.scan(
        body,
        init,
        xs,
        length=int(fixture["length"]),
        reverse=bool(fixture["reverse"]),
    )
    return {"final_carry": final_carry, "ys": ys}


def compute_golden_autodiff(case: Mapping, fixture: Mapping) -> dict:
    """CPU JVP/VJP（对最终 carry 求导；B4_quad 是冻结的 AD body）。"""
    _require_mapping(case, fixture)
    if fixture["xs_tree"] is None:
        raise ValueError("自动微分用例需要非 None 的 xs")
    body = fixtures.get_body(fixture["body_id"])
    if fixture["body_id"] != 3:
        raise ValueError(f"AD 只接受 B4_quad(body_id=3)，收到 body_id={fixture['body_id']}")
    init, xs = _build_inputs(fixture)

    def f_ad(x):
        final_carry, _ys = jax.lax.scan(
            body, init, x, length=int(fixture["length"]), reverse=bool(fixture["reverse"])
        )
        return final_carry

    tangent_seed = jax.tree_util.tree_map(jnp.ones_like, xs)
    jvp_primal, jvp_tangent = jax.jvp(f_ad, (xs,), (tangent_seed,))
    vjp_primal, vjp_fn = jax.vjp(f_ad, xs)
    cotangent_seed = jax.tree_util.tree_map(jnp.ones_like, vjp_primal)
    vjp_cotangent = vjp_fn(cotangent_seed)[0]
    return {
        "jvp_primal": jvp_primal,
        "jvp_tangent": jvp_tangent,
        "vjp_primal": vjp_primal,
        "vjp_cotangent": vjp_cotangent,
    }


def compute_golden_expect_reject(case: Mapping, fixture: Mapping) -> str:
    """确认 CPU golden 也拒绝非法 carry 变化；返回稳定拒绝码。"""
    _require_mapping(case, fixture)
    try:
        compute_golden(case, fixture)
    except Exception as exc:  # noqa: BLE001 —— 要把 JAX 的 trace 期 TypeError 归一成稳定码
        message = str(exc)
        lower = message.lower()
        carry_related = "carry" in lower and (
            "same pytree structure" in lower
            or "equal types" in lower
            or "must have the same" in lower
            or "must have equal" in lower
        )
        if carry_related:
            return "carry_shape_mismatch"
        raise RuntimeError(
            f"golden 拒绝了，但不是 carry 不变式错误，不能算 {case.get('case_id')} 通过："
            f"{type(exc).__name__}: {message}"
        ) from exc
    raise RuntimeError(
        f"golden 没有拒绝 {case.get('case_id')}：该用例不是合法的 carry 不变式反例"
    )


if __name__ == "__main__":  # 自检：只跑 CPU（不设置环境变量）
    demo = {"case_id": "scan_sum_len8_f32", "shape": [8, 16], "dtype": "float32", "seed": 20261026}
    fixture = fixtures.make_fixture(demo)
    out = compute_golden(demo, fixture)
    print("backend:", jax.default_backend())
    print("carry:", np.asarray(out["final_carry"])[:4])
    print("ys:", np.asarray(out["ys"]).shape)
