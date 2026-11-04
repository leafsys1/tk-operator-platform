# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan · host driver（门禁 CLI 的 Python 侧）。

协议（``0-黑盒测试设计.md`` §1.3.3 / ``1-测试工程说明.md`` §5.1）：

    jax_lax_associative_scan --case-json <case.json> --inputs <inputs.npz>
                             --mode <eager|jit|jvp|vjp> --outdir <dir>

* 成功：rc=0，写 ``outputs.npz`` + ``meta.json``，stdout 首行 ``EXEC ...``；
* 拒绝：rc=3，写 ``status="reject"`` 的 ``meta.json``（无 ``outputs.npz``），
  stdout/stderr 含稳定拒绝码；
* 显式 unsupported：rc=4，输出 ``UNSUPPORTED:autodiff_unsupported`` + 后端原文；
* 参数/IO 错误：rc=2；其他失败：rc=1。

数值计算全部在 ``ops/jax_lax_associative_scan/python/jax_lax_associative_scan.py`` →
``libjax_lax_associative_scan_host.so`` → AscendC kernel 内；本文件只做 I/O、路由参数的
解析与证据落盘，不做扫描计算（编码规则 A1）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP_ROOT = HERE.parent
CANDIDATE_PATH = HERE / "jax_lax_associative_scan.py"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ARGS = 2
EXIT_REJECT = 3
EXIT_UNSUPPORTED = 4

ELEM_KEY = "elems"
OUT_PATH = "out"

_UNSUPPORTED_MARKER = "UNSUPPORTED:autodiff_unsupported"


# ---------------------------------------------------------------------------
# combine 注册表（只用于把 CLI 的 fn 解析成 combine_id；实际计算在 kernel 内）
# ---------------------------------------------------------------------------


def _combine_sum(a, b):
    return a + b


def _combine_max(a, b):
    return a if a >= b else b


def _combine_tree_add(a, b):
    return a + b


for _fn, _name, _cid in ((_combine_sum, "sum", 1), (_combine_max, "max", 2),
                         (_combine_tree_add, "pytree_add", 3)):
    _fn.combine_name = _name
    _fn.combine_id = _cid

_COMBINE_FUNCS = {"sum": _combine_sum, "max": _combine_max, "pytree_add": _combine_tree_add}


# ---------------------------------------------------------------------------
# 通用 I/O
# ---------------------------------------------------------------------------


def _load_candidate():
    spec = importlib.util.spec_from_file_location("jax_lax_associative_scan_candidate",
                                                  CANDIDATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载候选模块：{CANDIDATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flatten(value, prefix: str = ""):
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(_flatten(item, f"{prefix}/{key}" if prefix else str(key)))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for index, item in enumerate(value):
            out.extend(_flatten(item, f"{prefix}/{index}" if prefix else str(index)))
        return out
    return [(prefix, value)]


def _tree_from_spec(spec: dict, arrays: dict, convert):
    kind = spec["kind"]
    if kind == "leaf":
        return convert(arrays[spec["path"]])
    if kind == "tuple":
        return tuple(_tree_from_spec(child, arrays, convert) for child in spec["children"])
    if kind == "list":
        return [_tree_from_spec(child, arrays, convert) for child in spec["children"]]
    if kind == "dict":
        return {key: _tree_from_spec(spec["children"][key], arrays, convert)
                for key in spec["keys"]}
    raise ValueError(f"未知树节点类型：{kind}")


def _arrays_sha256(arrays: dict) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(arr.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(arr.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _leaf_sha256(array) -> str:
    return _arrays_sha256({"leaf": array})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_exec(**fields) -> None:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    print("EXEC " + " ".join(parts))


def _dispatch_view(module, mode: str) -> dict:
    dispatch = getattr(module, "_last_dispatch", None)
    if not isinstance(dispatch, dict):
        dispatch = {}
    view = {
        "path": dispatch.get("path"),
        "exec_mode": dispatch.get("exec_mode"),
        "abi_calls": int(dispatch.get("abi_calls") or 0),
        "combine_id": dispatch.get("combine_id"),
        "family": dispatch.get("family"),
        "backend": dispatch.get("backend"),
        "device_id": dispatch.get("device_id"),
        "plan_id": dispatch.get("plan_id"),
        "plan_cache": dispatch.get("plan_cache"),
        "reason": dispatch.get("reason"),
    }
    view["mode"] = mode
    return view


def _exec_common(**fields) -> dict:
    base = {
        "op": "jax_lax_associative_scan",
        "kernel": "jax_lax_associative_scan_kernel",
        "arch": "dav-2201",
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="jax_lax_associative_scan host driver")
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", required=True, choices=("eager", "jit", "jvp", "vjp"))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    import numpy as np

    case_path = Path(args.case_json)
    inputs_path = Path(args.inputs)
    outdir = Path(args.outdir)
    mode = args.mode

    if not case_path.is_file():
        print(f"ARGS_ERROR case.json 不存在：{case_path}", file=sys.stderr)
        return EXIT_ARGS
    if not inputs_path.is_file():
        print(f"ARGS_ERROR inputs.npz 不存在：{inputs_path}", file=sys.stderr)
        return EXIT_ARGS
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        resolved = case["_resolved"]
    except Exception as exc:  # noqa: BLE001
        print(f"ARGS_ERROR case.json 解析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ARGS
    try:
        with np.load(inputs_path, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    except Exception as exc:  # noqa: BLE001
        print(f"ARGS_ERROR inputs.npz 解析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ARGS

    combine_name = str(resolved["combine_name"])
    combine_id = int(resolved["combine_id"])
    fn = _COMBINE_FUNCS[combine_name]
    axis = int(resolved["axis"])
    reverse = bool(resolved["reverse"])
    family = "numpy"

    try:
        elems = _tree_from_spec(resolved["elems_tree"], arrays, lambda value: np.array(value, copy=True))
    except Exception as exc:  # noqa: BLE001
        print(f"ARGS_ERROR elems 还原失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ARGS

    try:
        module = _load_candidate()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR 候选模块加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        if mode in ("eager", "jit"):
            call = module.associative_scan_compiled if mode == "jit" else module.associative_scan
            result = call(fn, elems, reverse=reverse, axis=axis)
            outputs = {key: np.asarray(value) for key, value in _flatten(result, OUT_PATH)}
            expected_paths = list(resolved["output_paths"])
            ordered = {}
            for path, (key, value) in zip(expected_paths, outputs.items()):
                ordered[path] = value
            outputs = ordered
        else:
            result = module._autodiff_kernel(fn, elems, reverse=reverse, axis=axis, mode=mode)
            keep = ("jvp_primal", "jvp_tangent") if mode == "jvp" else (
                "vjp_primal", "vjp_cotangent")
            outputs = {key: np.asarray(result[key]) for key in keep}
    except module._AssocReject as exc:
        dispatch = _dispatch_view(module, mode)
        _print_exec(**_exec_common(mode=mode, combine=combine_name, combine_id=combine_id,
                                   family=family, path="reject", exec_mode="reject",
                                   abi_calls=0, backend="npu"))
        print(f"REJECT code={exc.code} {exc.detail}")
        meta = {
            "case_id": case.get("case_id"),
            "mode": mode,
            "status": "reject",
            "reject_code": exc.code,
            "inputs_sha256": _arrays_sha256(arrays),
            "dispatch": dispatch,
            "leaves": [],
        }
        _write_json(outdir / "meta.json", meta)
        return EXIT_REJECT
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        if _UNSUPPORTED_MARKER in text:
            dispatch = _dispatch_view(module, mode)
            backend = dispatch.get("backend") or "npu"
            _print_exec(**_exec_common(mode=mode, combine=combine_name, combine_id=combine_id,
                                       family=family, path="trace", exec_mode="trace",
                                       abi_calls=0, backend=backend))
            print(text)
            meta = {
                "case_id": case.get("case_id"),
                "mode": mode,
                "status": "unsupported",
                "inputs_sha256": _arrays_sha256(arrays),
                "dispatch": dispatch,
                "backend_error": text,
                "leaves": [],
            }
            _write_json(outdir / "meta.json", meta)
            return EXIT_UNSUPPORTED
        print(f"ERROR {text}", file=sys.stderr)
        return EXIT_ERROR

    dispatch = _dispatch_view(module, mode)
    dispatch["mode"] = mode
    outputs = {key: np.ascontiguousarray(value) for key, value in outputs.items()}
    np.savez(outdir / "outputs.npz", **outputs)

    leaves = [
        {"path": key, "dtype": str(np.asarray(value).dtype),
         "shape": list(np.asarray(value).shape), "sha256": _leaf_sha256(np.asarray(value))}
        for key, value in sorted(outputs.items())
    ]
    meta = {
        "case_id": case.get("case_id"),
        "mode": mode,
        "status": "ok",
        "inputs_sha256": _arrays_sha256(arrays),
        "dispatch": dispatch,
        "leaves": leaves,
        "so_fingerprint": module.library_fingerprint(),
    }
    _write_json(outdir / "meta.json", meta)

    _print_exec(**_exec_common(
        mode=mode,
        combine=combine_name,
        combine_id=combine_id,
        family=family,
        path=dispatch.get("path"),
        exec_mode=dispatch.get("exec_mode"),
        abi_calls=dispatch.get("abi_calls"),
        backend=dispatch.get("backend"),
        device_id=dispatch.get("device_id"),
        plan_id=dispatch.get("plan_id"),
        plan_cache=dispatch.get("plan_cache"),
    ))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
