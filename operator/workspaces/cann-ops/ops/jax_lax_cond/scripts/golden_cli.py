# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""golden 子进程入口：读 case.json + inputs.npz，写 golden.npz + golden.json。

父进程（共享 runner / pytest）负责在**子进程环境**里设 ``JAX_PLATFORMS=cpu``；本 CLI
import 后只**断言**、不覆盖环境变量，便于 ``test_selfcheck_golden_env_guard_negative``
用 ``--expect-backend npu`` 做失败注入（设计 §2.1.1）。

退出码（设计 §2.1.1）：

====  ==========================================================
0     成功；``--mode reject`` 表示 CPU 抛错确认，或已确认的规格规则拒绝
      （仅 c13 的 int32(0/1) pred：``cpu_rejects=false`` + ``spec_rule_rejects=true``）
2     参数/IO/输入缺失
3     后端/版本不是已确认基线（``GOLDEN_ENV_MISMATCH``）
4     golden 计算异常
5     ``--mode reject`` 时 CPU 未抛错，且不属于已确认的规格规则拒绝
====  ==========================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

EXPECTED_JAX_VERSION = "0.8.1"


def _sha256_arrays(arrays: dict) -> str:
    import numpy as np  # noqa: PLC0415

    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(list(arr.shape)).encode("utf-8"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def _module_leaks() -> list[str]:
    leaks = []
    for name in sys.modules:
        short = name.split(".")[-1]
        if short in ("jax_lax_cond", "jax_lax_cond_cli"):
            leaks.append(name)
        if name in ("torch", "torch_npu") or name.startswith("torch.") or \
                name.startswith("torch_npu."):
            leaks.append(name)
    return leaks


def _barrier(message: str) -> int:
    print(f"GOLDEN_ENV_MISMATCH {message}", file=sys.stderr)
    return 3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="jax_lax_cond CPU JAX golden CLI")
    ap.add_argument("--case", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--mode", required=True,
                    choices=("eager", "jit", "grad_eager", "grad_jit", "reject"))
    ap.add_argument("--variant", default="")
    ap.add_argument("--expect-backend", default="cpu", choices=("cpu", "npu"))
    args = ap.parse_args(argv)

    try:
        import numpy as np  # noqa: PLC0415
        import jax  # noqa: PLC0415
        import jaxlib  # noqa: PLC0415

        import fixtures  # noqa: PLC0415
        import golden  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"GOLDEN_IMPORT_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    try:
        backend = jax.default_backend()
        platforms = sorted({str(device.platform) for device in jax.devices()})
        enable_x64 = bool(jax.config.jax_enable_x64)
    except Exception as exc:  # noqa: BLE001 —— 请求了不可用的后端时同样必须拦住
        return _barrier(f"后端初始化失败：{type(exc).__name__}: {exc}")

    if args.expect_backend == "cpu":
        if backend != "cpu" or platforms != ["cpu"]:
            return _barrier(f"backend={backend} devices={platforms}（golden 只允许 CPU）")
    else:
        if backend == "cpu" or platforms == ["cpu"]:
            return _barrier(f"期望非 CPU 后端，实际 backend={backend} devices={platforms}")

    versions = (getattr(jax, "__version__", None), getattr(jaxlib, "__version__", None))
    if versions != (EXPECTED_JAX_VERSION, EXPECTED_JAX_VERSION):
        return _barrier(f"jax={versions[0]} jaxlib={versions[1]}（已确认基线 {EXPECTED_JAX_VERSION}）")
    if enable_x64:
        return _barrier("jax_enable_x64=True（已确认基线 False）")
    leaks = _module_leaks()
    if leaks:
        return _barrier(f"候选/设备模块已被导入：{sorted(set(leaks))}（golden 子进程必须隔离）")

    case_path = Path(args.case)
    inputs_path = Path(args.inputs)
    meta_path = Path(args.meta)
    npz_path = Path(args.npz) if args.npz else None
    if not case_path.is_file():
        print(f"GOLDEN_ARGS_ERROR case.json 不存在：{case_path}", file=sys.stderr)
        return 2
    if not inputs_path.is_file():
        print(f"GOLDEN_ARGS_ERROR inputs.npz 不存在：{inputs_path}", file=sys.stderr)
        return 2

    try:
        case_obj = json.loads(case_path.read_text(encoding="utf-8"))
        arrays = fixtures.load_inputs_npz(inputs_path)
        case_obj.setdefault("variant", args.variant)
        if args.variant:
            case_obj["variant"] = args.variant
        fixture = fixtures.fixture_from_case(case_obj, arrays)
        if args.mode != "reject" and str(fixture["base_mode"]) != args.mode:
            print(f"GOLDEN_ARGS_ERROR case.json base_mode={fixture['base_mode']!r} "
                  f"!= --mode={args.mode!r}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"GOLDEN_ARGS_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    base_meta = {
        "schema": "opx-jax-cond-golden/1",
        "case_id": case_obj.get("case_id"),
        "variant": fixture["variant"],
        "mode": ("reject" if args.mode == "reject" else fixture["mode"]),
        "base_mode": args.mode,
        "family": fixture["family"],
        "kind": fixture["kind"],
        "pred": fixture["pred_descriptor"],
        "pred_kind": fixture["pred_kind"],
        "selected": fixture["selected"],
        "branch_ids": [fixture["true_id"], fixture["false_id"]],
        "branch_exprs": [fixture["true_name"], fixture["false_name"]],
        "operands": [{"path": leaf["path"], "shape": leaf["shape"], "dtype": leaf["dtype"]}
                     for leaf in fixture["leaves"]],
        "expected": case_obj.get("expected"),
        "expected_error": case_obj.get("expected_error"),
        "input_sha256": fixture["inputs_sha256"],
        "jax": getattr(jax, "__version__", None),
        "jaxlib": getattr(jaxlib, "__version__", None),
        "backend": backend,
        "devices": platforms,
        "x64": enable_x64,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
    }

    if args.mode == "reject":
        try:
            evidence = golden.reject_evidence(case_obj, fixture)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"GOLDEN_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return 4
        upstream = {
            "type": evidence.get("cpu_exception_type"),
            "message": evidence.get("cpu_message"),
            "sha256": evidence.get("cpu_message_sha256"),
            "summary": evidence.get("cpu_message_summary"),
            "keyword_hit": evidence.get("upstream_keyword_hit"),
            # 把“CPU 原始行为”与“规格规则拒绝”分开落盘：int32 pred 的 CPU 行为是
            # 不抛错（cpu_rejects=false），拒绝依据是 spec v1 的 scalar-bool 规则。
            "cpu_rejects": bool(evidence.get("cpu_rejects")),
            "spec_rule_rejects": bool(evidence.get("spec_rule_rejects")),
            "spec_rule": evidence.get("spec_rule"),
            "reject_basis": evidence.get("reject_basis"),
        }
        cpu_rejects = bool(evidence.get("cpu_rejects"))
        spec_rule_rejects = bool(evidence.get("spec_rule_rejects"))
        confirmed = cpu_rejects or spec_rule_rejects
        base_meta.update(status="reject" if confirmed else "reject_unconfirmed",
                         reject_code=evidence["spec_code"], upstream=upstream,
                         reject_evidence=evidence, leaves=[])
        _write_json(meta_path, base_meta)
        if not confirmed:
            print(
                f"GOLDEN_REJECT_UNCONFIRMED code={evidence['spec_code']} "
                f"case={base_meta['case_id']} variant={base_meta['variant']} "
                f"pred_kind={base_meta['pred_kind']}：JAX 0.8.1 CPU 未抛错，"
                f"且不属于已确认的规格规则拒绝（不修改期望、不伪造上游证据）",
                file=sys.stderr)
            return 5
        print(
            f"GOLDEN_EXEC backend=cpu jax={base_meta['jax']} jaxlib={base_meta['jaxlib']} "
            f"pid={base_meta['pid']} case={base_meta['case_id']} variant={base_meta['variant']} "
            f"mode=reject code={evidence['spec_code']} "
            f"upstream={evidence.get('cpu_exception_type')} "
            f"cpu_rejects={cpu_rejects} spec_rule_rejects={spec_rule_rejects}")
        return 0

    out_arrays: dict = {}
    weak_types: dict[str, object] = {}
    try:
        if fixture["base_mode"] in ("grad_eager", "grad_jit"):
            result = golden.compute_golden_grad(case_obj, fixture)
        else:
            result = golden.compute_golden(case_obj, fixture)
        if not isinstance(result, dict) or not result:
            raise RuntimeError("golden 输出为空或不是 dict")
        jax_family = (fixture["family"] == "jax"
                      or fixture["base_mode"] in ("grad_eager", "grad_jit"))
        for key, value in result.items():
            weak_types[key] = bool(getattr(value, "weak_type", False)) if jax_family else None
            out_arrays[key] = np.asarray(value)
        expected_keys = set(fixture["output_shapes"])
        if set(out_arrays) != expected_keys:
            raise RuntimeError(f"golden 输出键 {sorted(out_arrays)} != 期望 {sorted(expected_keys)}")
        if npz_path is None:
            print("GOLDEN_ARGS_ERROR 非 reject 模式必须给 --npz", file=sys.stderr)
            return 2
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **out_arrays)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"GOLDEN_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    leaves = []
    for key in sorted(out_arrays):
        arr = np.ascontiguousarray(out_arrays[key])
        shape = list(arr.shape)
        leaf = fixtures.leaf_bytes(shape, str(arr.dtype))
        leaf.update(path=key, dtype=str(arr.dtype), shape=shape,
                    weak_type=weak_types.get(key),
                    sha256=_sha256_arrays({key: arr}))
        leaves.append(leaf)
    base_meta.update(status="ok", output_sha256=_sha256_arrays(out_arrays), leaves=leaves)
    _write_json(meta_path, base_meta)
    print(
        f"GOLDEN_EXEC backend=cpu jax={base_meta['jax']} jaxlib={base_meta['jaxlib']} "
        f"pid={base_meta['pid']} case={base_meta['case_id']} variant={base_meta['variant']} "
        f"mode={args.mode} leaves={len(leaves)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
