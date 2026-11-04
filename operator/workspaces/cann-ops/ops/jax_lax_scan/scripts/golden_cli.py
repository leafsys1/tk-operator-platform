# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""golden 子进程入口：读 case.json + inputs.npz，写 golden.npz + golden.json。

父进程（共享 runner / pytest）负责在**子进程环境**里设 ``JAX_PLATFORMS=cpu``；本 CLI
import 后只**断言**、不覆盖环境变量，便于 ``test_golden_env_guard_negative`` 做失败注入。

退出码（设计 §2.1.1，父进程把非 0 一律判 ERROR，``--mode reject`` 的 0 表示 golden 也确认拒绝）：

====  ==========================================================
0     成功
2     参数/IO/输入缺失
3     后端不是 CPU（``GOLDEN_ENV_MISMATCH``）
4     golden 计算异常
5     ``--mode reject`` 时未按预期抛出 ``carry_shape_mismatch``
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
    # `python -I` 也会把脚本目录从 sys.path 去掉；这里显式加回，便于 import fixtures/golden。
    sys.path.insert(0, str(_HERE))


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
        if name == "torch_npu" or name == "jax_lax_scan" or name.endswith(".jax_lax_scan"):
            leaks.append(name)
    return leaks


def _barrier(message: str) -> int:
    print(f"GOLDEN_ENV_MISMATCH {message}", file=sys.stderr)
    return 3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="jax_lax_scan CPU JAX golden CLI")
    ap.add_argument("--case", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--mode", required=True,
                    choices=("eager", "jit", "jvp", "vjp", "reject"))
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
        platforms = sorted({str(d.platform) for d in jax.devices()})
    except Exception as exc:  # noqa: BLE001 —— 请求了不可用的非 CPU 后端时同样必须拦住
        return _barrier(f"CPU 后端初始化失败：{type(exc).__name__}: {exc}")
    if backend != "cpu" or platforms != ["cpu"]:
        return _barrier(f"backend={backend} devices={platforms}（golden 只允许 CPU）")
    leaks = _module_leaks()
    if leaks:
        return _barrier(f"候选/设备模块已被导入：{leaks}（golden 子进程必须隔离）")

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
        fixture = fixtures.fixture_from_case(case_obj, arrays)
    except Exception as exc:  # noqa: BLE001
        print(f"GOLDEN_ARGS_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    mode = args.mode
    base_meta = {
        "schema": "opx-scan-golden/1",
        "case_id": case_obj.get("case_id"),
        "mode": mode,
        "fixture": fixture["fixture"],
        "body_id": fixture["body_id"],
        "length": fixture["length"],
        "width": fixture["width"],
        "reverse": fixture["reverse"],
        "seed": fixture["seed"],
        "dtype": fixture["dtype"],
        "family": fixture["family"],
        "carry_tree": fixture["carry_tree"],
        "xs_tree": fixture["xs_tree"],
        "input_sha256": fixture["inputs_sha256"],
        "expected": case_obj.get("expected"),
        "jax": getattr(jax, "__version__", None),
        "jaxlib": getattr(jaxlib, "__version__", None),
        "backend": backend,
        "devices": platforms,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
    }

    try:
        if mode == "reject":
            code = golden.compute_golden_expect_reject(case_obj, fixture)
            expected_code = case_obj.get("expect_reject_code", "carry_shape_mismatch")
            if code != expected_code:
                print(
                    f"GOLDEN_REJECT_MISMATCH golden 拒绝码 {code!r} != 期望 {expected_code!r}",
                    file=sys.stderr,
                )
                return 5
            base_meta.update(status="reject", reject_code=code)
            _write_json(meta_path, base_meta)
            print(
                f"GOLDEN_EXEC backend=cpu jax={base_meta['jax']} jaxlib={base_meta['jaxlib']} "
                f"pid={base_meta['pid']} case={base_meta['case_id']} mode=reject code={code}"
            )
            return 0

        if mode in ("jvp", "vjp"):
            result = golden.compute_golden_autodiff(case_obj, fixture)
            # 只落该 mode 需要的两个叶子，与候选 outputs.npz 的输出合同一致（设计 A1.2）。
            keep = ("jvp_primal", "jvp_tangent") if mode == "jvp" else ("vjp_primal", "vjp_cotangent")
            result = {key: result[key] for key in keep}
        else:
            result = golden.compute_golden(case_obj, fixture)
        flat = dict(fixtures.flatten_with_paths(result))
        if not flat:
            raise RuntimeError("golden 输出为空")
        out_arrays = {key: np.asarray(value) for key, value in flat.items()}
        if npz_path is None:
            print("GOLDEN_ARGS_ERROR 非 reject 模式必须给 --npz", file=sys.stderr)
            return 2
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **out_arrays)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"GOLDEN_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    leaves = [
        {"path": key, "dtype": str(arr.dtype), "shape": list(arr.shape)}
        for key, arr in sorted(out_arrays.items())
    ]
    base_meta.update(
        status="ok",
        output_sha256=_sha256_arrays(out_arrays),
        leaves=leaves,
    )
    _write_json(meta_path, base_meta)
    print(
        f"GOLDEN_EXEC backend=cpu jax={base_meta['jax']} jaxlib={base_meta['jaxlib']} "
        f"pid={base_meta['pid']} case={base_meta['case_id']} mode={mode} "
        f"leaves={len(leaves)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
