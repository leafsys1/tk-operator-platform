# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop · host driver（四动作可执行入口 `build/jax_lax_while_loop` 的协议实现）。

调用契约（0.1-黑盒测试设计.md §2.2 / 0-测试工程说明.md §5.5）：

    jax_lax_while_loop_cli.py --case-json <case.json> --inputs <inputs.npz> \
                              --mode <eager|jit|jvp_eager|jvp_jit> [--variant <name>] --outdir <dir>

* 成功：rc=0；``<outdir>/outputs.npz`` + ``<outdir>/meta.json``；stdout 含 ``EXEC ...``。
* 参数校验拒绝（c13–c15）：rc=3；``REJECT code=<五码之一>`` +
  ``UPSTREAM type=<类型> sha256=<摘要> summary=<单行摘要>``；
  ``EXEC ... path=reject exec_mode=reject abi_calls=0``；**不写** ``outputs.npz``。
* 设备安全阀拒绝（c16）：rc=3；``path=so`` + 设备计数（64/64/64）+ 原始整数 ``abi_calls``。
* 其它非 0：``ERROR``（不算拒绝通过）；本任务没有合法 ``UNSUPPORTED``/rc=4。

数值路径：全部经 ``ops/jax_lax_while_loop/python/jax_lax_while_loop.py`` →
``libjax_lax_while_loop_host.so`` → AscendC kernel 在 NPU 上执行；本文件只做
case.json/inputs.npz → pytree 重建、协议输出与证据落盘，不做数值计算。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax_lax_while_loop as iface  # noqa: E402

OP_NAME = iface.OP_NAME
KERNEL_NAME = iface.KERNEL_NAME
NPU_ARCH = iface.NPU_ARCH

BASE_MODES = ("eager", "jit", "jvp_eager", "jvp_jit")

EXEC_KEYS = (
    "op", "kernel", "arch", "mode", "variant", "path", "exec_mode", "abi_calls",
    "backend", "platforms", "device_id", "iterations", "cond_hits", "body_hits",
    "trace_hits", "exec_hits", "runtime_token", "counter_source", "route",
    "compile_count", "plan_id", "plan_cache", "limit_flag",
)

_LOG = logging.getLogger("jax_lax_while_loop_cli")


# ---------------------------------------------------------------------------
# case.json + inputs.npz → pytree（与 golden.build_value 同一份绑定）
# ---------------------------------------------------------------------------
def _build_value(spec: dict, arrays: dict, case: dict, variant: str):
    """按 case.json 的 tree 描述与 inputs.npz 重建 pytree（与 golden 同一份绑定）。"""
    import jax
    import jax.numpy as jnp

    kind = spec["kind"]
    if kind == "leaf":
        raw = np.asarray(arrays[spec["path"]])
        dtype = str(spec["dtype"])
        if "float0" in dtype:
            return jnp.asarray(np.asarray(raw), dtype=jax.dtypes.float0)
        if dtype == "int32":
            return jnp.asarray(raw, dtype=jnp.int32)
        if str(case.get("case_id")) == "while_sup_weak_type" and variant == "weak":
            return jnp.asarray(float(raw))
        return jnp.asarray(raw, dtype=jnp.float32)
    if kind == "tuple":
        return tuple(_build_value(child, arrays, case, variant)
                     for child in spec["children"])
    if kind == "dict":
        return {key: _build_value(spec["children"][key], arrays, case, variant)
                for key in spec["keys"]}
    raise ValueError(f"未知树节点：{kind!r}")


# ---------------------------------------------------------------------------
# cond/body 绑定（设计 §5.3 / §5.5；与 golden.cond_body 同数学语义）
# ---------------------------------------------------------------------------
def _cond_body(case: dict, variant: str):
    import jax.numpy as jnp

    case_id = str(case.get("case_id"))
    if case_id == "while_zero_iterations":
        return (lambda c: c[0] < 0), (lambda c: (c[0] + 1, c[1] + 1.0))
    if case_id == "while_one_iteration":
        return (lambda c: c[0] < 1), (lambda c: (c[0] + 1, c[1] * 1.25 + 0.5))
    if case_id == "while_dynamic_multi_iterations":
        return (lambda c: c[0] < c[1]), (lambda c: (c[0] + 1, c[1], c[2] * 1.03125 + 0.125))
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
        return (lambda c: c[0] < c[1]), (lambda c: (c[0] + 1, c[1], c[2] * 1.03125 + 0.125))
    if case_id == "while_jvp_multi_iteration":
        return (lambda c: c[0] < 6), (lambda c: (c[0] + 1, c[1] * 1.125 + 0.25))
    if case_id == "while_jvp_zero_iteration":
        return (lambda c: c[0] < 0), (lambda c: (c[0] + 1, c[1] * 1.125 + 0.25))
    if case_id == "while_unaligned_tail_widths":
        return (lambda c: c[0] < 5), (lambda c: (c[0] + 1, c[1] * 0.5 + 0.25))
    if case_id == "while_large_tensor":
        return (lambda c: c[0] < 12), (lambda c: (c[0] + 1, c[1] * 1.0009765625 + 0.03125))
    if case_id == "while_short_small_carry":
        return (lambda c: c[0] < 4), (lambda c: (c[0] + 1, c[1] * 1.03125 + 0.125))
    if case_id == "while_long_large_tensor":
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
        return (lambda c: jnp.bool_(True)), (lambda c: {"a": c[0], "b": c[1]})
    if case_id == "while_carry_shape_or_dtype_mismatch":
        if variant == "shape":
            return (lambda c: jnp.bool_(True)), (lambda c: jnp.concatenate([c, c[:1]]))
        return (lambda c: jnp.bool_(True)), (lambda c: c.astype(jnp.int32))
    if case_id == "while_iteration_limit_exceeded":
        return (lambda c: jnp.bool_(True)), (lambda c: (c[0] + 1, c[1] + 1.0))
    raise ValueError(f"{case_id}: 没有 cond/body 绑定")


def _is_reject(case: dict) -> bool:
    return "expected_error" in case or str(case.get("case_id")) in (
        "while_invalid_condition", "while_carry_structure_mismatch",
        "while_carry_shape_or_dtype_mismatch", "while_iteration_limit_exceeded")


# ---------------------------------------------------------------------------
# 证据输出
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _exec_line(meta: dict) -> str:
    parts = [f"{key}={_fmt(meta.get(key))}" for key in EXEC_KEYS]
    return "EXEC " + " ".join(parts)


def _write_outputs(outdir: Path, result, is_jvp: bool, mode: str) -> None:
    if is_jvp:
        primal, tangent = result
        flat = iface.stable_flatten(primal, "primal")
        flat.update(iface.stable_flatten(tangent, "tangent"))
    else:
        flat = iface.stable_flatten(result, "out")
    np.savez(outdir / "outputs.npz", **flat)


def _write_meta(outdir: Path, meta: dict) -> None:
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _reject_stdout(meta: dict, exc: iface.WhileLoopError) -> str:
    sha = exc.upstream_message_sha256 or "-"
    upstream_type = exc.upstream_exception_type if exc.upstream_exception_type else "null"
    summary = exc.upstream_message or ""
    lines = [
        f"REJECT code={exc.code}",
        f"UPSTREAM type={upstream_type} sha256={sha} summary={summary}",
        _exec_line(meta),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[jax_lax_while_loop] %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="jax_lax_while_loop host driver")
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", required=True, choices=list(BASE_MODES))
    parser.add_argument("--variant", default="")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    inputs = dict(np.load(args.inputs))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    variant = args.variant or str(case.get("variant") or "")
    case_id = str(case.get("case_id"))

    init = _build_value(case["tree"], inputs, case, variant)
    cond, body = _cond_body(case, variant)
    cond.opx_variant = variant
    body.opx_variant = variant
    body.opx_case = case
    tangent = None
    if mode in ("jvp_eager", "jvp_jit"):
        tangent = _build_value(case["tangent"], inputs, case, variant)
    safety_limit = case.get("safety_iteration_limit")

    try:
        if _is_reject(case):
            if mode == "eager":
                iface.while_loop(cond, body, init, safety_limit=safety_limit)
            else:
                iface.while_loop_compiled(cond, body, init, mode="primal",
                                          safety_limit=safety_limit)
        elif mode == "eager":
            result = iface.while_loop(cond, body, init, safety_limit=safety_limit)
        elif mode == "jit":
            result = iface.while_loop_compiled(cond, body, init, mode="primal",
                                               safety_limit=safety_limit)
        else:
            result = iface.while_loop_compiled(cond, body, init, mode="jvp",
                                               tangent=tangent,
                                               jit=(mode == "jvp_jit"),
                                               safety_limit=safety_limit)
    except iface.WhileLoopError as exc:
        meta = dict(exc.dispatch or iface.last_dispatch or {})
        meta.update({"case_id": case_id, "variant": variant, "mode": mode})
        meta.setdefault("platforms", [])
        _write_meta(outdir, meta)
        if exc.code in iface.REJECT_CODES:
            print(_reject_stdout(meta, exc))
            return 3
        print(f"ERROR code={exc.code} detail={exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 —— 非拒绝异常必须非 0，不冒充通过
        import traceback
        traceback.print_exc()
        print(f"ERROR unhandled {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    is_jvp = mode in ("jvp_eager", "jvp_jit")
    _write_outputs(outdir, result, is_jvp, mode)
    meta = dict(iface.last_dispatch)
    meta.update({"case_id": case_id, "variant": variant, "mode": mode,
                 "is_jvp": is_jvp})
    _write_meta(outdir, meta)
    print(_exec_line(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
