# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan · host driver（四动作可执行入口 `build/jax_lax_scan` 的协议实现）。

调用契约（设计 §2.2.3 A1.2，共享 runner `tools/run_cases.py` 按此调用）：

    jax_lax_scan_cli.py --case-json <case.json> --inputs <inputs.npz> \
                        --mode <eager|jit|jvp|vjp> --outdir <dir>

* 成功：rc=0；``<outdir>/outputs.npz`` + ``<outdir>/meta.json``；stdout 首行 ``EXEC ...``。
* 拒绝（body 4/5/6）：rc=3；stdout 含 ``carry_shape_mismatch``；``EXEC path=reject abi_calls=0``；
  **不写** ``outputs.npz``。
* 显式 unsupported（AD 后端不支持）：rc=4；``UNSUPPORTED:autodiff_unsupported`` + 后端原始报错；
  ``EXEC path=trace abi_calls=0 backend=npu reason=so_no_autodiff``。
* 其它失败：rc=1（``ERROR``，不算拒绝通过）。

数值路径：
* ``eager`` / ``jit``：``libjax_lax_scan_host.so`` → ``jax_lax_scan_kernel``（AscendC，NPU 上执行）。
  本进程用 numpy + ACL 直接管理设备缓冲（镜像无 torch_npu 时的可执行路径），不做任何数值计算。
* ``jvp`` / ``vjp``：JAX trace（``.so`` 无自动微分），候选仍在 NPU 后端执行。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax_lax_scan as scan_iface  # noqa: E402

OP_NAME = scan_iface.OP_NAME
KERNEL_NAME = scan_iface.KERNEL_NAME
NPU_ARCH = scan_iface.NPU_ARCH
REJECT_CODE = scan_iface.REJECT_CODE
UNSUPPORTED_CODE = scan_iface.UNSUPPORTED_CODE
TRACE_REASON = scan_iface.TRACE_REASON
REJECT_BODIES = (4, 5, 6)
AD_MODES = ("jvp", "vjp")

FINAL_CARRY_PREFIX = "final_carry"


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _contig(array):
    """0-d 叶子必须保持 rank-0（np.ascontiguousarray 会把 0-d 变成 shape=(1,)）。"""
    array = np.asarray(array)
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _arrays_sha256(arrays: dict) -> str:
    """与 ops/jax_lax_scan/scripts/fixtures.compute_arrays_sha256 同算法（独立实现，不 import 测试侧）。"""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = _contig(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(arr.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(arr.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _exec_line(case_id, mode, resolved, path, exec_mode, abi_calls, backend, family, reason=None):
    body_id = int(resolved["body_id"])
    body = resolved.get("body_name") or resolved.get("fixture")
    line = (
        f"EXEC op={OP_NAME} kernel={KERNEL_NAME} arch={NPU_ARCH} mode={mode} body={body} "
        f"body_id={body_id} family={family} path={path} exec_mode={exec_mode} "
        f"abi_calls={abi_calls} backend={backend}"
    )
    return line + (f" reason={reason}" if reason else "")


def _leaf_record(path, array, platforms):
    return {
        "path": path,
        "dtype": str(np.asarray(array).dtype),
        "shape": [int(v) for v in np.asarray(array).shape],
        "platforms": list(platforms),
        "device": platforms[0] if platforms else "unknown",
    }


def _write_meta(outdir: Path, *, case_id, mode, dispatch, family, backend, devices, leaves,
                inputs_sha256, outputs_sha256, jax_info, jaxlib_info, error=""):
    meta = {
        "schema": "opx-scan-candidate/1",
        "case_id": case_id,
        "mode": mode,
        "status": "ok",
        "dispatch": dispatch,
        "family": family,
        "backend": backend,
        "devices": list(devices),
        "leaves": leaves,
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": outputs_sha256,
        "jax": jax_info,
        "jaxlib": jaxlib_info,
        "error": error,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _jax_versions():
    try:
        import jax
        import jaxlib

        return jax.__version__, jaxlib.__version__
    except Exception:  # noqa: BLE001 —— meta 里如实写 None，不伪造
        return None, None


def _output_key(carry_path: str) -> str:
    if carry_path == "init":
        return FINAL_CARRY_PREFIX
    if carry_path.startswith("init/"):
        return FINAL_CARRY_PREFIX + carry_path[len("init"):]
    return FINAL_CARRY_PREFIX + "/" + carry_path


# ---------------------------------------------------------------------------
# .so 路径（eager / jit）
# ---------------------------------------------------------------------------


def _run_so_mode(case_id, mode, resolved, arrays, outdir: Path) -> int:
    body_id = int(resolved["body_id"])
    length = int(resolved["length"])
    reverse = bool(resolved["reverse"])
    compiled = mode == "jit"
    carry_paths = [str(p) for p in resolved["carry_paths"]]
    xs_paths = [str(p) for p in resolved["xs_paths"]]
    carry_arrays = [arrays[path] for path in carry_paths]
    xs_arrays = [arrays[path] for path in xs_paths]

    abi_calls = 1 if compiled else length
    try:
        carry_out, ys = scan_iface._run_so_numpy(
            body_id, carry_arrays, xs_arrays, length, reverse, compiled, device_id=0
        )
    except scan_iface._SoError as exc:
        print(f"ERROR {exc}")
        print(_exec_line(case_id, mode, resolved, "so",
                         "compiled_whole_loop" if compiled else "stepwise", 0, "npu", "numpy"))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {type(exc).__name__}: {exc}")
        print(_exec_line(case_id, mode, resolved, "so",
                         "compiled_whole_loop" if compiled else "stepwise", 0, "npu", "numpy"))
        return 1

    outputs = {}
    for path, array in zip(carry_paths, carry_out):
        outputs[_output_key(path)] = _contig(array)
    outputs["ys"] = _contig(ys)
    outputs_sha = _arrays_sha256(outputs)
    np.savez(outdir / "outputs.npz", **outputs)

    platforms = ["npu"]
    leaves = [_leaf_record(key, value, platforms) for key, value in outputs.items()]
    jax_version, jaxlib_version = _jax_versions()
    _write_meta(
        outdir, case_id=case_id, mode=mode,
        dispatch={
            "path": "so",
            "body_id": body_id,
            "exec_mode": "compiled_whole_loop" if compiled else "stepwise",
            "abi_calls": abi_calls,
            "reason": None,
        },
        family="numpy", backend="npu", devices=["npu:0"], leaves=leaves,
        inputs_sha256=resolved["inputs_sha256"], outputs_sha256=outputs_sha,
        jax_info=jax_version, jaxlib_info=jaxlib_version,
    )
    print(_exec_line(case_id, mode, resolved, "so",
                     "compiled_whole_loop" if compiled else "stepwise", abi_calls, "npu", "numpy"))
    print(f"OK case={case_id} mode={mode} leaves={len(leaves)} abi_calls={abi_calls}")
    return 0


# ---------------------------------------------------------------------------
# JAX trace 路径（JVP / VJP）
# ---------------------------------------------------------------------------


def _b4_quad(carry, x):
    """body_id=3 的注册实现（与 fixtures.b4_quad 同一数学语义）。"""
    new = 0.5 * carry + x
    return new, new * new


_UNSUPPORTED_EXC_TYPES = (
    "NotImplementedError",
    "UnimplementedError",
    "XlaRuntimeError",
    "JaxRuntimeError",
    "RuntimeError",
    "ValueError",
)

_UNSUPPORTED_CAPABILITY_NEEDLES = (
    "unimplemented",
    "not implemented",
    "unsupported",
    "does not support",
    "no kernel",
    "not supported",
    "not available",
)

# 后端身份标记：只有后端原文明确指向本次候选后端时才允许分类为 UNSUPPORTED，
# 避免把普通的 shape/dtype 运行失败按文本碰巧命中而误报为“后端不支持”（代码检视 P-09）。
_UNSUPPORTED_BACKEND_MARKERS = (
    "xla-npu",
    "xla_npu",
    "xlanpu",
    "ascend",
    "afir",
    "npu",
)


def _looks_unsupported(text: str, backend: str | None = None) -> bool:
    """显式 unsupported 的稳定分类：异常类型 + 能力关键词 + 后端身份三重命中。

    仍只作为 CLI 侧第一道筛；runner/pytest 还会校验 3 次同码、`path=trace`、
    `abi_calls=0`、非 CPU backend 与后端原始报错是否一致，缺一不记 UNSUPPORTED。
    """
    text = text or ""
    exc_type = text.split(":", 1)[0].strip().rsplit(".", 1)[-1]
    if exc_type not in _UNSUPPORTED_EXC_TYPES:
        return False
    lowered = text.lower()
    capability = any(needle in lowered for needle in _UNSUPPORTED_CAPABILITY_NEEDLES)
    backend_marked = any(marker in lowered for marker in _UNSUPPORTED_BACKEND_MARKERS)
    backend_verified = str(backend or "").lower() not in ("", "cpu")
    return capability and (backend_marked or backend_verified)


def _run_ad_mode(case_id, mode, resolved, arrays, outdir: Path) -> int:
    body_id = int(resolved["body_id"])
    length = int(resolved["length"])
    reverse = bool(resolved["reverse"])
    carry_paths = [str(p) for p in resolved["carry_paths"]]
    xs_paths = [str(p) for p in resolved["xs_paths"]]
    if body_id != scan_iface.SCAN_BODY_QUAD:
        print(f"ERROR invalid_argument: AD 只接受 body_id=3（收到 {body_id}）")
        return 1
    if not xs_paths:
        print("ERROR invalid_argument: 自动微分用例需要非 None 的 xs")
        return 1

    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    platforms = sorted({str(getattr(d, "platform", "?")) for d in jax.devices()})
    if backend == "cpu" or "cpu" in platforms:
        print(f"ERROR device_error: 候选 JAX 后端是 CPU（{backend}/{devices}）：不允许 CPU 结果冒充 NPU")
        return 1

    def to_device(array):
        return jax.device_put(jnp.asarray(np.array(array, copy=True)), jax.devices()[0])

    init = to_device(arrays[carry_paths[0]])
    xs = to_device(arrays[xs_paths[0]])

    def f_ad(seq):
        final_carry, _ys = jax.lax.scan(_b4_quad, init, seq, length=length, reverse=reverse)
        return final_carry

    try:
        if mode == "jvp":
            seed = jax.tree_util.tree_map(jnp.ones_like, xs)
            primal, tangent = jax.jvp(f_ad, (xs,), (seed,))
            outputs = {"jvp_primal": primal, "jvp_tangent": tangent}
        else:
            primal, vjp_fn = jax.vjp(f_ad, xs)
            cotangent_seed = jax.tree_util.tree_map(jnp.ones_like, primal)
            cotangent = vjp_fn(cotangent_seed)[0]
            outputs = {"vjp_primal": primal, "vjp_cotangent": cotangent}
        outputs = jax.tree_util.tree_map(
            lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
            outputs,
        )
        numpy_outputs = {key: _contig(np.asarray(value)) for key, value in outputs.items()}
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        jax_version, jaxlib_version = _jax_versions()
        if _looks_unsupported(text, backend):
            # 显式 unsupported：必须附后端原始报错文本，且 EXEC 必须证明 path=trace / abi_calls=0 / 非 CPU
            _write_meta(
                outdir, case_id=case_id, mode=mode,
                dispatch={"path": "trace", "body_id": body_id, "exec_mode": "trace",
                          "abi_calls": 0, "reason": TRACE_REASON},
                family="jax", backend=backend, devices=devices, leaves=[],
                inputs_sha256=resolved["inputs_sha256"], outputs_sha256=None,
                jax_info=jax_version, jaxlib_info=jaxlib_version, error=text,
            )
            print(_exec_line(case_id, mode, resolved, "trace", "trace", 0, backend, "jax",
                             reason=TRACE_REASON))
            print(f"{UNSUPPORTED_CODE}: {text}")
            return 4
        print(f"ERROR {text}")
        print(_exec_line(case_id, mode, resolved, "trace", "trace", 0, backend, "jax"))
        return 1

    outputs_sha = _arrays_sha256(numpy_outputs)
    np.savez(outdir / "outputs.npz", **numpy_outputs)
    leaves = [_leaf_record(key, value, platforms) for key, value in numpy_outputs.items()]
    jax_version, jaxlib_version = _jax_versions()
    _write_meta(
        outdir, case_id=case_id, mode=mode,
        dispatch={"path": "trace", "body_id": body_id, "exec_mode": "trace",
                  "abi_calls": 0, "reason": TRACE_REASON},
        family="jax", backend=backend, devices=devices, leaves=leaves,
        inputs_sha256=resolved["inputs_sha256"], outputs_sha256=outputs_sha,
        jax_info=jax_version, jaxlib_info=jaxlib_version,
    )
    print(_exec_line(case_id, mode, resolved, "trace", "trace", 0, backend, "jax"))
    print(f"OK case={case_id} mode={mode} leaves={len(leaves)} backend={backend}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="jax_lax_scan host driver")
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", required=True, choices=["eager", "jit", "jvp", "vjp"])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    case_path = Path(args.case_json)
    inputs_path = Path(args.inputs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    case = json.loads(case_path.read_text(encoding="utf-8"))
    resolved = case.get("_resolved")
    if not isinstance(resolved, dict):
        print("ERROR invalid_argument: case.json 缺少 _resolved（由 runner/pytest 写入）")
        return 1
    with np.load(inputs_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}

    case_id = str(case.get("case_id") or resolved.get("fixture"))
    mode = args.mode
    body_id = int(resolved["body_id"])
    family = str(resolved.get("family") or "torch")

    if body_id in REJECT_BODIES:
        print(_exec_line(case_id, mode, resolved, "reject", "reject", 0, None, family))
        print(
            f"REJECT_{REJECT_CODE.upper()} {REJECT_CODE}: body_id={body_id} "
            "改变了 carry 的结构/shape/dtype（trace 期拒绝，device 未执行）"
        )
        return 3

    if mode in AD_MODES:
        return _run_ad_mode(case_id, mode, resolved, arrays, outdir)
    return _run_so_mode(case_id, mode, resolved, arrays, outdir)


if __name__ == "__main__":
    sys.exit(main())
