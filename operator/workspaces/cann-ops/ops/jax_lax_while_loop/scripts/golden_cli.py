# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""golden 子进程入口：读 case.json + inputs.npz，写 golden.npz + golden.json。

父进程（共享 runner / pytest）负责在**子进程环境**里设 ``JAX_PLATFORMS=cpu``；本 CLI
import 后只**断言**、不覆盖环境变量，便于 ``--expect-backend npu`` 做失败注入（设计 §2.1）。

退出码：

====  ==========================================================
0     成功；``--mode reject`` 表示 CPU 抛错确认，或已确认的规格规则拒绝
      （c16 的 ``ITERATION_LIMIT_EXCEEDED``：``cpu_rejects=false`` +
      ``spec_rule_rejects=true``，不在 CPU 执行 always_true）
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
import signal
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

EXPECTED_JAX_VERSION = "0.8.1"
REJECT_MODES = ("eager", "jit")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        if short in ("jax_lax_while_loop", "jax_lax_while_loop_cli"):
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


def _timeout_handler(_signum, _frame):
    raise TimeoutError("golden 计算超时（30s）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="jax_lax_while_loop CPU JAX golden CLI")
    ap.add_argument("--case", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--mode", required=True,
                    choices=("eager", "jit", "jvp_eager", "jvp_jit", "reject"))
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
        return _barrier(f"jax/jaxlib={versions}（已确认基线 {EXPECTED_JAX_VERSION}）")
    leaks = _module_leaks()
    if leaks:
        return _barrier(f"golden 子进程 import 了候选/torch 模块：{leaks}")

    case_path = Path(args.case)
    inputs_path = Path(args.inputs)
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        arrays = fixtures.load_inputs(inputs_path)
    except Exception as exc:  # noqa: BLE001
        print(f"GOLDEN_INPUT_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    variant = str(args.variant if args.variant else case.get("variant") or "")
    base = {
        "schema": "opx-jax-while-loop-golden/1",
        "op": fixtures.OP_NAME,
        "case_id": str(case.get("case_id")),
        "variant": variant,
        "mode": args.mode,
        "backend": backend,
        "platforms": platforms,
        "jax": versions[0],
        "jaxlib": versions[1],
        "x64": enable_x64,
        "case_sha256": _sha256_file(case_path),
        "inputs_sha256": _sha256_file(inputs_path),
        "expected_iterations": case.get("expected_iterations"),
        "expected_counts": case.get("expected_counts"),
    }

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    try:
        if args.mode == "reject":
            # c16 的 compute_reject 内部已拦截，不在 CPU 执行 always_true。
            evidence = golden.compute_reject(case, variant, arrays)
            signal.alarm(0)
            confirmed = bool(evidence.get("cpu_rejects") or evidence.get("spec_rule_rejects"))
            base["reject_evidence"] = evidence
            base["status"] = "reject" if confirmed else "unconfirmed"
            _write_json(Path(args.meta), base)
            if confirmed:
                print(f"GOLDEN_REJECT_CONFIRMED code={evidence.get('reject_code')} "
                      f"basis={evidence.get('reject_basis')}")
                return 0
            print(f"GOLDEN_REJECT_UNCONFIRMED code={evidence.get('reject_code')}", file=sys.stderr)
            return 5

        payload = golden.compute_payload(case, variant, args.mode, arrays)
        out = payload["arrays"]
        meta = payload["leaves"]
        tree = payload["tree"]
        signal.alarm(0)
    except TimeoutError as exc:
        signal.alarm(0)
        print(f"GOLDEN_TIMEOUT {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        signal.alarm(0)
        print(f"GOLDEN_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4
    finally:
        signal.alarm(0)

    npz_path = Path(args.npz) if args.npz else Path(args.meta).with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **out)
    base.update({
        "status": "ok",
        "npz": str(npz_path),
        "outputs": sorted(out),
        "outputs_sha256": _sha256_arrays(out),
        "leaves": meta,
        "tree": tree,
    })
    _write_json(Path(args.meta), base)
    print(f"GOLDEN_OK outputs={len(out)} sha256={base['outputs_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
