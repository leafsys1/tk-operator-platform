# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan 的 scan 分支执行核心（被 `tools/run_cases.py` / `tools/smoke_case.py` 复用）。

为什么单独成模块：scan 用例的字段模型（body / length / reverse / xs pytree / mode /
`match_cpu_or_explicit_unsupported`）与 elementwise 的 `n_a/n_b` 完全不同，但**统一入口与判据**
必须保持只有一个。这里把 scan 特有的输入、候选调用、golden 子进程、逐叶子判定与报告组装集中实现，
共享 runner 只按 `op` 显式分派进来；elementwise 分支原样保留。

契约（与设计 §2.2.3 A1.2 一致）
--------------------------------
候选可执行文件：

``ops/jax_lax_scan/build/jax_lax_scan --case-json <case.json> --inputs <inputs.npz> --mode <m> --outdir <dir>``

* 成功：rc=0；``<outdir>/outputs.npz``（普通模式 ``final_carry...``/``ys``；jvp/vjp 模式
  ``jvp_primal``+``jvp_tangent`` / ``vjp_primal``+``vjp_cotangent``）+ ``<outdir>/meta.json``；
  stdout 首行 ``EXEC ...``。
* 拒绝：rc=3；stdout/stderr 含 ``carry_shape_mismatch``；``EXEC ... path=reject exec_mode=reject abi_calls=0``；
  不得写 ``outputs.npz``。
* 显式 unsupported：rc=4；输出含 ``UNSUPPORTED:autodiff_unsupported`` + 后端原始报错。
* 其他非 0：ERROR（不得因“别的原因失败”算拒绝通过）。

golden 一律走独立 CPU 子进程 ``golden_cli.py``；本模块**不在 runner 进程 import jax**，避免候选侧
backend 被 golden 的 CPU 环境限制（设计 §2.1.1 / §2.2.3 A1.2.8）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fixtures  # noqa: E402

SCAN_OP = "jax_lax_scan"
CANDIDATE_EXIT_OK = 0
CANDIDATE_EXIT_REJECT = 3
CANDIDATE_EXIT_UNSUPPORTED = 4
REJECT_CODE = "carry_shape_mismatch"
UNSUPPORTED_CODE = "UNSUPPORTED:autodiff_unsupported"
GOLDEN_TIMEOUT_S = 180
CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN", "UNSUPPORTED")
AD_MODES = ("jvp", "vjp")
EXEC_REQUIRED_KEYS = ("op", "kernel", "arch", "mode", "body", "body_id", "family",
                      "path", "exec_mode", "abi_calls", "backend")


def is_scan_operator(op: str, suite: dict | None = None) -> bool:
    """显式分派判据：只认 op 名（不用“有没有 mode 字段”这种会漂移的隐式条件）。"""
    if op != SCAN_OP:
        return False
    return suite is None or suite.get("op") in (None, SCAN_OP)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(ws: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ws.resolve()))
    except (ValueError, OSError):
        return str(path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"模块加载失败：{path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _tail(text: str, limit: int = 400) -> str:
    return (text or "").strip()[-limit:]


# ---------------------------------------------------------------------------
# 候选 / golden 调用
# ---------------------------------------------------------------------------


def _candidate_cmd(exe: Path, case_json: Path, inputs: Path, mode: str, outdir: Path) -> list[str]:
    return [str(exe), "--case-json", str(case_json), "--inputs", str(inputs),
            "--mode", mode, "--outdir", str(outdir)]


def _parse_exec(stdout: str) -> dict | None:
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("EXEC"):
            continue
        marker = stripped[len("EXEC"):].strip()
        fields: dict[str, str] = {}
        for token in marker.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key.strip()] = value.strip()
        fields["_raw"] = stripped
        return fields
    return None


def _exec_once(exe: Path, case_json: Path, inputs: Path, mode: str, outdir: Path,
               timeout: int) -> dict:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = _candidate_cmd(exe, case_json, inputs, mode, outdir)
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(outdir.parent))
        rc = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        timed_out = True
    return {
        "rc": rc,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cmd": " ".join(cmd),
        "outdir": outdir,
        "exec": _parse_exec(stdout),
    }


def _has_marker(run: dict, marker: str) -> bool:
    return marker in ((run.get("stdout") or "") + (run.get("stderr") or ""))


def strip_unsupported_marker(text: str) -> str:
    """去掉稳定分类码，只留后端原始报错文本，供“附后端原文”的硬断言使用。"""
    return (text or "").replace(UNSUPPORTED_CODE, "").strip().strip(":;,，。 \t\r\n")


def unsupported_attempt_problems(dispatch: dict | None, backend_error: str, *,
                                 require_reason: bool = True, require_op: bool = False,
                                 expected_mode: str | None = None) -> list[str]:
    """校验一次 ``rc=4`` 尝试是否满足“显式 unsupported”的硬证据。

    设计 §1.3.4/§5.3：AD 必须走 trace、``abi_calls=0``、backend 非 CPU；3 次同码后
    ``dispatch.reason`` 必须是 ``so_no_autodiff``，且必须附后端原始报错文本。
    ``require_op``/``expected_mode`` 同时用于 runner 的 EXEC 与 pytest 的 ``_last_dispatch``
    （`op=jax_lax_scan`、mode 正确）。
    返回问题列表；调用方必须判 ``INVALID``/``FAIL``，**不能**因此记 ``UNSUPPORTED``。
    """
    dispatch = dispatch or {}
    problems: list[str] = []
    if require_op and dispatch.get("op") != SCAN_OP:
        problems.append(f"UNSUPPORTED 的 EXEC op 必须是 {SCAN_OP}，收到 {dispatch.get('op')!r}")
    if expected_mode is not None and dispatch.get("mode") != expected_mode:
        problems.append(f"UNSUPPORTED 的 mode 必须是 {expected_mode}，收到 {dispatch.get('mode')!r}")
    if dispatch.get("path") != "trace":
        problems.append(f"UNSUPPORTED 的 dispatch.path 必须是 trace，收到 {dispatch.get('path')!r}")
    abi_calls = _as_int(dispatch.get("abi_calls"))
    if abi_calls != 0:
        problems.append(f"UNSUPPORTED 的 abi_calls 必须是 0，收到 {dispatch.get('abi_calls')!r}")
    backend = str(dispatch.get("backend"))
    if backend in ("", "None", "NoneType", "cpu"):
        problems.append(f"UNSUPPORTED 的 backend 必须是 NPU 后端，收到 {dispatch.get('backend')!r}")
    if require_reason and str(dispatch.get("reason")) != "so_no_autodiff":
        problems.append(
            f"UNSUPPORTED 的 dispatch.reason 必须是 so_no_autodiff，收到 {dispatch.get('reason')!r}")
    if not strip_unsupported_marker(backend_error):
        problems.append("UNSUPPORTED 必须附后端原始报错文本（去掉稳定码后为空）")
    return problems


def _load_candidate_meta(outdir: Path) -> tuple[dict | None, str]:
    path = outdir / "meta.json"
    if not path.is_file():
        return None, f"候选没有写 meta.json（{path}）"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"meta.json 读不了：{type(exc).__name__}: {exc}"


def _golden_env() -> dict:
    """父进程给**子进程**准备的 CPU 环境；绝不回写 os.environ。"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    return env


def _run_golden(ws: Path, case_json: Path, inputs: Path, mode: str, cdir: Path,
                timeout: int = GOLDEN_TIMEOUT_S) -> dict:
    cli = ws / "ops" / SCAN_OP / "scripts" / "golden_cli.py"
    npz = cdir / f"golden_{mode}.npz"
    meta_path = cdir / f"golden_{mode}.json"
    cmd = [sys.executable, "-I", str(cli), "--case", str(case_json), "--inputs", str(inputs),
           "--npz", str(npz), "--meta", str(meta_path), "--mode", mode]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=_golden_env(), cwd=str(cdir))
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "detail": f"golden 超时（{timeout}s）", "cmd": " ".join(cmd)}
    rc = proc.returncode
    if rc != 0:
        return {
            "ok": False, "rc": rc, "cmd": " ".join(cmd),
            "detail": f"golden 非 0 退出 rc={rc}；stderr={_tail(proc.stderr or '')}",
        }
    if not meta_path.is_file():
        return {"ok": False, "rc": rc, "detail": "golden 返回 0 但没有 meta.json", "cmd": " ".join(cmd)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = fixtures.load_inputs_npz(npz) if npz.is_file() else {}
    return {"ok": True, "rc": rc, "meta": meta, "arrays": arrays,
            "npz": str(npz), "cmd": " ".join(cmd)}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def _expected_golden_keys(mode: str, fixture: dict) -> list[str]:
    if mode == "jvp":
        return ["jvp_primal", "jvp_tangent"]
    if mode == "vjp":
        return ["vjp_primal", "vjp_cotangent"]
    keys = ["final_carry" if path == "init" else path.replace("init/", "final_carry/", 1)
            for path in fixture["carry_paths"]]
    keys.append(fixtures.Y_PATH)
    return keys


def _meta_leaf_map(meta: dict) -> dict[str, dict]:
    out = {}
    for leaf in meta.get("leaves") or []:
        if isinstance(leaf, dict) and "path" in leaf:
            out[str(leaf["path"])] = leaf
    return out


def _validate_candidate_meta(meta: dict, *, case_id: str, mode: str, fixture: dict,
                             outputs: dict) -> tuple[list[str], dict[str, dict]]:
    problems: list[str] = []
    if str(meta.get("schema", "")).split("@")[0] != "opx-scan-candidate/1":
        problems.append(f"meta.schema 不是 opx-scan-candidate/1：{meta.get('schema')!r}")
    if str(meta.get("case_id")) != str(case_id):
        problems.append(f"meta.case_id={meta.get('case_id')!r} != {case_id!r}")
    if str(meta.get("mode")) != mode:
        problems.append(f"meta.mode={meta.get('mode')!r} != {mode!r}")
    if meta.get("status") != "ok":
        problems.append(f"meta.status={meta.get('status')!r}（成功路径必须是 ok）")
    if str(meta.get("backend")) in ("", "None", "cpu"):
        problems.append(f"meta.backend={meta.get('backend')!r}（候选必须在非 CPU 后端执行）")
    devices = meta.get("devices")
    if not isinstance(devices, list) or not devices:
        problems.append(f"meta.devices 必须是非空列表：{devices!r}")
    elif any(str(d).lower().startswith("cpu") for d in devices):
        problems.append(f"meta.devices 不是 NPU：{devices}")
    if meta.get("inputs_sha256") != fixture["inputs_sha256"]:
        problems.append(
            f"meta.inputs_sha256 与本次 inputs.npz 不一致：{meta.get('inputs_sha256')} != "
            f"{fixture['inputs_sha256']}（候选读到的不是这条用例的数据）"
        )
    leaf_map = _meta_leaf_map(meta)
    missing = [key for key in outputs if key not in leaf_map]
    if missing:
        problems.append(f"meta.leaves 缺输出叶子：{missing}")
    extra = [key for key in leaf_map if key not in outputs]
    if extra:
        problems.append(f"meta.leaves 多出 outputs.npz 没有的叶子：{extra}")
    for key, leaf in leaf_map.items():
        platforms = [str(p) for p in (leaf.get("platforms") or [])]
        if not platforms:
            problems.append(f"{key}: meta 没有 platforms，无法证明设备")
        elif any(p == "cpu" for p in platforms):
            problems.append(f"{key}: platforms 含 cpu：{platforms}（不允许 CPU 结果冒充 NPU）")
    return problems, leaf_map


def _judge_arrays(actual: dict, golden: dict, expected_keys: list[str], criteria,
                  leaf_map: dict[str, dict], fixture: dict, mode: str) -> tuple[list[dict], list[str]]:
    leaves: list[dict] = []
    problems: list[str] = []
    actual_keys = set(actual)
    expected = set(expected_keys)
    if actual_keys != expected:
        problems.append(f"输出叶子集合不一致：候选 {sorted(actual_keys)} != 期望 {sorted(expected)}")
    for key in expected_keys:
        if key not in actual or key not in golden:
            continue
        got = np.asarray(actual[key])
        want = np.asarray(golden[key])
        rec = {
            "path": key,
            "dtype": str(got.dtype),
            "shape": list(got.shape),
            "golden_dtype": str(want.dtype),
            "golden_shape": list(want.shape),
            "is_pass": False,
        }
        meta_leaf = leaf_map.get(key) or {}
        rec["kind"] = meta_leaf.get("kind")
        rec["platforms"] = sorted(str(p) for p in (meta_leaf.get("platforms") or []))
        rec["device"] = meta_leaf.get("device")
        if got.dtype != want.dtype:
            problems.append(f"{key}: dtype {got.dtype} != golden {want.dtype}")
            leaves.append(rec)
            continue
        if got.shape != want.shape:
            problems.append(f"{key}: shape {got.shape} != golden {want.shape}")
            leaves.append(rec)
            continue
        if got.dtype == np.int32:
            ok = bool(np.array_equal(got, want))
            rec.update(is_pass=ok, matched_ratio=1.0 if ok else 0.0, max_abs_error=0.0 if ok else None,
                       nonfinite={"count": 0, "match": True, "finite_kind_mismatch": 0})
            if not ok:
                problems.append(f"{key}: int32 必须逐元素精确相等")
            leaves.append(rec)
            continue
        try:
            verdict = criteria.check(np.ascontiguousarray(got, dtype=np.float32),
                                     np.ascontiguousarray(want, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key}: 判据机判失败：{type(exc).__name__}: {exc}")
            leaves.append(rec)
            continue
        rec.update(
            is_pass=bool(verdict["is_pass"]),
            matched_ratio=verdict["matched_ratio"],
            max_abs_error=verdict["max_abs_error"],
            max_abs_error_limit=verdict["max_abs_error_limit"],
            required_matched_ratio=verdict["required_matched_ratio"],
            nonfinite=verdict["nonfinite"],
        )
        if not verdict["is_pass"]:
            problems.append(f"{key}: " + "；".join(verdict.get("failure_reasons") or ["数值未通过"]))
        leaves.append(rec)

    if mode not in AD_MODES and fixtures.Y_PATH in actual:
        ys = np.asarray(actual[fixtures.Y_PATH])
        if ys.ndim < 1 or ys.shape[0] != int(fixture["length"]):
            problems.append(f"ys 前导轴 {ys.shape} 与 length={fixture['length']} 不一致")
    return leaves, problems


def _dispatch_problems(mode: str, exec_fields: dict, meta: dict, fixture: dict) -> list[str]:
    problems: list[str] = []
    dispatch = meta.get("dispatch") or {}
    for key in ("path", "exec_mode", "body_id"):
        if key not in dispatch:
            problems.append(f"meta.dispatch 缺 {key}")
    try:
        abi_calls = int(exec_fields.get("abi_calls"))
    except (TypeError, ValueError):
        abi_calls = None
    if abi_calls is None:
        problems.append(f"EXEC abi_calls 不是整数：{exec_fields.get('abi_calls')!r}")

    if mode in AD_MODES:
        if exec_fields.get("path") != "trace":
            problems.append(f"AD 必须走 trace，EXEC path={exec_fields.get('path')!r}")
        if str(dispatch.get("reason")) != "so_no_autodiff":
            problems.append(f"AD 的 dispatch.reason 必须是 so_no_autodiff，收到 {dispatch.get('reason')!r}")
        if abi_calls != 0:
            problems.append(f"AD 走 trace 时 .so 调用次数必须为 0，收到 abi_calls={abi_calls}")
        return problems

    if exec_fields.get("path") != "so":
        problems.append(f"{mode} 已注册 body 必须走 .so，EXEC path={exec_fields.get('path')!r}")
    expected_exec_mode = "stepwise" if mode == "eager" else "compiled_whole_loop"
    if exec_fields.get("exec_mode") != expected_exec_mode:
        problems.append(f"{mode} 的 exec_mode 必须是 {expected_exec_mode}，收到 {exec_fields.get('exec_mode')!r}")
    if str(dispatch.get("exec_mode")) != expected_exec_mode:
        problems.append(f"{mode} 的 meta.dispatch.exec_mode 必须是 {expected_exec_mode}，收到 {dispatch.get('exec_mode')!r}")
    try:
        if int(dispatch.get("body_id")) != int(fixture["body_id"]):
            problems.append(
                f"meta.dispatch.body_id={dispatch.get('body_id')!r} != fixture.body_id={fixture['body_id']}")
    except (TypeError, ValueError):
        problems.append(f"meta.dispatch.body_id 不是整数：{dispatch.get('body_id')!r}")
    expected_calls = int(fixture["length"]) if mode == "eager" else 1
    if abi_calls != expected_calls:
        problems.append(f"{mode} 的 abi_calls 必须是 {expected_calls}，收到 {abi_calls}")
    if int(dispatch.get("abi_calls", -1)) != expected_calls:
        problems.append(f"{mode} 的 meta.dispatch.abi_calls 必须是 {expected_calls}，收到 {dispatch.get('abi_calls')!r}")
    return problems


def _exec_problems(exec_fields: dict | None, mode: str) -> list[str]:
    if exec_fields is None:
        return ["候选没有打印 EXEC 标记行 → 无法证明实际执行的是候选算子路径"]
    problems = []
    missing = [key for key in EXEC_REQUIRED_KEYS if key not in exec_fields]
    if missing:
        problems.append(f"EXEC 缺字段：{missing}")
    if exec_fields.get("op") != SCAN_OP:
        problems.append(f"EXEC op={exec_fields.get('op')!r} != {SCAN_OP!r}")
    if exec_fields.get("mode") != mode:
        problems.append(f"EXEC mode={exec_fields.get('mode')!r} != {mode!r}")
    if str(exec_fields.get("backend")) in ("", "None", "cpu"):
        problems.append(f"EXEC backend={exec_fields.get('backend')!r}（候选必须证明非 CPU）")
    return problems


# ---------------------------------------------------------------------------
# 单条用例 / 单 mode
# ---------------------------------------------------------------------------


class ScanRun:
    """一次 scan 用例执行的上下文（版本字段、路径、判据）。"""

    def __init__(self, *, ws: Path, op: str, exe: Path, criteria, workdir: Path,
                 evidence_dir: Path, timeout: int):
        self.ws = ws
        self.op = op
        self.exe = exe
        self.criteria = criteria
        self.workdir = workdir
        self.evidence_dir = evidence_dir
        self.timeout = timeout


def _prepare_case(ctx: ScanRun, case: dict) -> tuple[dict, Path, Path]:
    case_id = case["case_id"]
    fixture = fixtures.make_fixture(case)
    cdir = ctx.workdir / str(case_id)
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    return fixture, case_json, inputs


def _run_mode(ctx: ScanRun, case: dict, fixture: dict, case_json: Path, inputs: Path,
              mode: str) -> dict:
    case_id = case["case_id"]
    cdir = case_json.parent
    expected = case.get("expected", "match_cpu")
    rec: dict = {
        "mode": mode,
        "status": "NOT_RUN",
        "expected": expected,
        "leaves": [],
        "dispatch": {},
        "attempts": [],
    }

    if expected == "reject":
        return _run_reject_mode(ctx, case, fixture, case_json, inputs, mode, rec)

    # ---- 普通 / AD：最多 3 次 only for explicit unsupported 的稳定分类 ----
    attempt_records: list[dict] = []
    usable_run: dict | None = None
    max_attempts = 3 if (mode in AD_MODES and expected == "match_cpu_or_explicit_unsupported") else 1
    for index in range(max_attempts):
        run = _exec_once(ctx.exe, case_json, inputs, mode, cdir / "out", ctx.timeout)
        exec_fields = run["exec"] or {}
        attempt_records.append({
            "attempt": index + 1,
            "rc": run["rc"],
            "timed_out": run["timed_out"],
            "unsupported_marker": _has_marker(run, UNSUPPORTED_CODE),
            "duration_ms": run["duration_ms"],
            "stderr_tail": _tail(run["stderr"], 200),
            "backend_error": _tail(
                strip_unsupported_marker((run["stderr"] or "") + "\n" + (run["stdout"] or "")), 400),
            "exec": {
                "op": exec_fields.get("op"),
                "mode": exec_fields.get("mode"),
                "path": exec_fields.get("path"),
                "exec_mode": exec_fields.get("exec_mode"),
                "abi_calls": _as_int(exec_fields.get("abi_calls")),
                "backend": exec_fields.get("backend"),
                "reason": exec_fields.get("reason"),
            },
        })
        if run["rc"] == CANDIDATE_EXIT_OK:
            usable_run = run
            if any(a["unsupported_marker"] for a in attempt_records[:-1]):
                attempt_records.append({"attempt": index + 1, "rc": run["rc"],
                                        "unstable": "UNSUPPORTED 后又成功"})
            break
        if run["rc"] == CANDIDATE_EXIT_UNSUPPORTED and _has_marker(run, UNSUPPORTED_CODE):
            continue
        usable_run = run
        break
    rec["attempts"] = attempt_records
    rec["duration_ms"] = sum(a.get("duration_ms", 0) for a in attempt_records)

    if (usable_run is None and max_attempts == 3
            and len(attempt_records) == 3
            and all(a["rc"] == CANDIDATE_EXIT_UNSUPPORTED and a["unsupported_marker"]
                    for a in attempt_records)):
        # 设计 §1.3.4/§5.3：显式 unsupported 是“有硬证据的分类”，不是“只打印 marker 就算”。
        # 每次尝试都必须有 EXEC（op/mode/path=trace/abi_calls=0/backend 非 CPU）；
        # 3 次同码后再要求 dispatch.reason=so_no_autodiff + 后端原始报错文本。
        problems: list[str] = []
        for attempt in attempt_records:
            problems.extend(unsupported_attempt_problems(
                attempt.get("exec"), attempt.get("backend_error") or "",
                require_reason=False, require_op=True, expected_mode=mode))
        last_exec = attempt_records[-1].get("exec") or {}
        if str(last_exec.get("reason")) != "so_no_autodiff":
            problems.append(
                f"3 次同码后 dispatch.reason 必须是 so_no_autodiff，收到 {last_exec.get('reason')!r}")
        problems = list(dict.fromkeys(problems))  # 3 次同码时会重复同一问题，去重便于定位
        rec["backend_errors"] = [a["stderr_tail"] for a in attempt_records]
        rec["dispatch"] = dict(last_exec)
        if problems:
            rec.update(status="INVALID",
                       detail="显式 unsupported 分类缺少硬证据，不能记 UNSUPPORTED：" + "；".join(problems))
            return rec
        rec.update(status="UNSUPPORTED",
                   detail="3 次均显式 UNSUPPORTED:autodiff_unsupported（path=trace/abi_calls=0/"
                          "reason=so_no_autodiff/backend 非 CPU，附后端原文）")
        return rec

    run = usable_run or _exec_once(ctx.exe, case_json, inputs, mode, cdir / "out", ctx.timeout)
    if any(a.get("unstable") for a in attempt_records):
        rec.update(status="FAIL",
                   detail="AD 分类不稳定：先出现 UNSUPPORTED，随后又成功（必须 3 次同码）")
        return rec
    rec["cmd"] = run["cmd"]
    rec["exec_marker"] = (run["exec"] or {}).get("_raw")
    rec["stdout_tail"] = _tail(run["stdout"])
    rec["stderr_tail"] = _tail(run["stderr"])

    if run["timed_out"]:
        rec.update(status="ERROR", detail=f"候选实现超时（{ctx.timeout}s）")
        return rec
    if run["rc"] == CANDIDATE_EXIT_UNSUPPORTED:
        if _has_marker(run, UNSUPPORTED_CODE) and mode in AD_MODES:
            rec.update(status="FAIL",
                       detail="AD 分类不稳定：出现 UNSUPPORTED 后又有不同结果（必须 3 次同码）")
            rec["attempts"] = attempt_records
        elif _has_marker(run, UNSUPPORTED_CODE):
            rec.update(status="FAIL",
                       detail=f"mode={mode} / expected={expected} 不允许 UNSUPPORTED 分类")
        else:
            rec.update(status="ERROR",
                       detail=f"候选 rc=4 但没有 {UNSUPPORTED_CODE} 稳定码（不能算 unsupported）")
        return rec
    if run["rc"] != CANDIDATE_EXIT_OK:
        rec.update(status="ERROR",
                   detail=f"候选非 0 退出 rc={run['rc']}；stderr={_tail(run['stderr'], 300)}")
        return rec

    exec_problems = _exec_problems(run["exec"], mode)
    if exec_problems:
        rec.update(status="INVALID", detail="；".join(exec_problems))
        return rec
    rec["dispatch"] = {
        "path": run["exec"].get("path"),
        "exec_mode": run["exec"].get("exec_mode"),
        "body_id": _as_int(run["exec"].get("body_id")),
        "abi_calls": _as_int(run["exec"].get("abi_calls")),
        "family": run["exec"].get("family"),
    }

    meta, meta_err = _load_candidate_meta(cdir / "out")
    if meta is None:
        rec.update(status="INVALID", detail=meta_err)
        return rec
    rec["meta_status"] = meta.get("status")

    outputs_path = cdir / "out" / "outputs.npz"
    if not outputs_path.is_file():
        rec.update(status="ERROR", detail="候选返回 0 但没有写 outputs.npz（静默失败）")
        return rec
    outputs = fixtures.load_inputs_npz(outputs_path)

    meta_problems, leaf_map = _validate_candidate_meta(
        meta, case_id=case_id, mode=mode, fixture=fixture, outputs=outputs)
    meta_problems.extend(_dispatch_problems(mode, run["exec"], meta, fixture))
    if meta_problems:
        rec.update(status="INVALID", detail="；".join(meta_problems))
        rec["leaves"] = [
            {"path": key, "dtype": str(np.asarray(value).dtype),
             "shape": list(np.asarray(value).shape),
             "platforms": (leaf_map.get(key) or {}).get("platforms")}
            for key, value in sorted(outputs.items())
        ]
        return rec

    golden = _run_golden(ctx.ws, case_json, inputs, mode, cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"), "cmd": golden.get("cmd")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    golden_meta = golden["meta"]
    rec["golden"].update({
        "backend": golden_meta.get("backend"),
        "jax": golden_meta.get("jax"),
        "jaxlib": golden_meta.get("jaxlib"),
        "devices": golden_meta.get("devices"),
        "pid": golden_meta.get("pid"),
    })

    expected_keys = _expected_golden_keys(mode, fixture)
    leaves, judge_problems = _judge_arrays(outputs, golden["arrays"], expected_keys,
                                           ctx.criteria, leaf_map, fixture, mode)
    rec["leaves"] = leaves
    rec["matched_ratio"] = min((leaf.get("matched_ratio", 1.0) or 0.0) for leaf in leaves) if leaves else 0.0
    rec["max_abs_err"] = max((leaf.get("max_abs_error") or 0.0) for leaf in leaves) if leaves else None
    rec["nonfinite"] = {
        "count": sum(int((leaf.get("nonfinite") or {}).get("count", 0)) for leaf in leaves),
        "match": all(bool((leaf.get("nonfinite") or {}).get("match", True)) for leaf in leaves),
    }
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
    return rec


def _run_reject_mode(ctx: ScanRun, case: dict, fixture: dict, case_json: Path, inputs: Path,
                     mode: str, rec: dict) -> dict:
    cdir = case_json.parent
    run = _exec_once(ctx.exe, case_json, inputs, mode, cdir / "out", ctx.timeout)
    rec["attempts"] = [{"attempt": 1, "rc": run["rc"], "timed_out": run["timed_out"],
                        "duration_ms": run["duration_ms"]}]
    rec["duration_ms"] = run["duration_ms"]
    rec["cmd"] = run["cmd"]
    rec["exec_marker"] = (run["exec"] or {}).get("_raw")
    rec["stdout_tail"] = _tail(run["stdout"])
    rec["stderr_tail"] = _tail(run["stderr"])

    expected_code = case.get("expect_reject_code", REJECT_CODE)
    code_hit = expected_code in ((run["stdout"] or "") + (run["stderr"] or ""))
    exec_fields = run["exec"] or {}
    abi_calls = _as_int(exec_fields.get("abi_calls"))
    rec["dispatch"] = {
        "path": exec_fields.get("path"),
        "exec_mode": exec_fields.get("exec_mode"),
        "body_id": _as_int(exec_fields.get("body_id")),
        "abi_calls": abi_calls,
        "family": exec_fields.get("family"),
    }

    golden = _run_golden(ctx.ws, case_json, inputs, "reject", cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"), "cmd": golden.get("cmd")}
    if not golden["ok"]:
        rec.update(status="ERROR",
                   detail=f"golden 未能确认该用例应被拒绝：{golden.get('detail')}")
        return rec
    rec["golden"].update({
        "backend": (golden.get("meta") or {}).get("backend"),
        "jax": (golden.get("meta") or {}).get("jax"),
        "jaxlib": (golden.get("meta") or {}).get("jaxlib"),
        "reject_code": (golden.get("meta") or {}).get("reject_code"),
    })

    if run["timed_out"]:
        rec.update(status="ERROR", detail=f"拒绝用例候选超时（{ctx.timeout}s）")
        return rec
    if run["rc"] == CANDIDATE_EXIT_OK:
        rec.update(status="FAIL", detail=f"该拒绝却成功返回（rc=0）；期望拒绝码 {expected_code}")
        return rec
    if run["rc"] != CANDIDATE_EXIT_REJECT:
        rec.update(status="ERROR",
                   detail=f"拒绝了但不是约定退出码：rc={run['rc']}（期望 {CANDIDATE_EXIT_REJECT}）")
        return rec
    if not code_hit:
        rec.update(status="FAIL", detail=f"rc=3 但输出不含约定拒绝码 {expected_code}")
        return rec
    if abi_calls != 0:
        rec.update(status="FAIL", detail=f"拒绝路径 abi_calls={abi_calls}（必须先于 device 执行，期望 0）")
        return rec
    if exec_fields.get("op") != SCAN_OP or exec_fields.get("mode") != mode:
        rec.update(status="FAIL",
                   detail=f"EXEC 拒绝标记不对：op={exec_fields.get('op')!r} mode={exec_fields.get('mode')!r}")
        return rec
    if exec_fields.get("path") != "reject" or exec_fields.get("exec_mode") != "reject":
        rec.update(status="FAIL",
                   detail=f"EXEC 拒绝标记不对：path={exec_fields.get('path')!r} exec_mode={exec_fields.get('exec_mode')!r}")
        return rec
    if (cdir / "out" / "outputs.npz").is_file():
        rec.update(status="FAIL", detail="拒绝用例不得写 outputs.npz")
        return rec
    rec.update(status="PASS", detail=f"按预期拒绝：rc=3 {expected_code}；abi_calls=0")
    return rec


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_case(ctx: ScanRun, case: dict, modes_override: list[str] | None = None) -> dict:
    """单条用例执行；把执行器自身的意外异常收敛成该用例 ERROR，不拖垮整轮报告。"""
    try:
        return _run_case_impl(ctx, case, modes_override=modes_override)
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        rec = {
            "case_id": case.get("case_id"),
            "group": case.get("group"),
            "required": bool(case.get("required", True)),
            "dtype": case.get("dtype"),
            "expected": case.get("expected", "match_cpu"),
            "note": case.get("note"),
            "status": "ERROR",
            "detail": f"执行器异常：{type(exc).__name__}: {exc}",
            "modes": [],
            "dispatch": {},
        }
        try:
            ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
            _dump_json(ctx.evidence_dir / str(rec["case_id"]) / "meta.json", rec)
        except Exception:  # noqa: BLE001
            pass
        return rec


def _run_case_impl(ctx: ScanRun, case: dict, modes_override: list[str] | None = None) -> dict:
    case_id = case["case_id"]
    fixture, case_json, inputs = _prepare_case(ctx, case)

    modes = list(modes_override) if modes_override is not None else list(case.get("mode") or ["jit"])
    mode_records = []
    for mode in modes:
        mode_records.append(_run_mode(ctx, case, fixture, case_json, inputs, mode))
        # 同一用例的 eager/jit 共用一个 outdir；此刻立刻快照，供稍后的 eager×jit 一致性互比。
        _snapshot_outputs(ctx, case_id, mode, ctx.workdir / str(case_id) / "out")

    statuses = [m["status"] for m in mode_records]
    overall = "NOT_RUN"
    detail = ""
    if mode_records:
        if all(s == "PASS" for s in statuses):
            overall, detail = "PASS", "全部 mode 通过"
        elif all(s == "UNSUPPORTED" for s in statuses):
            overall = "UNSUPPORTED"
            detail = "全部 mode 为显式 UNSUPPORTED（不是数值 PASS）"
        elif "UNSUPPORTED" in statuses:
            overall = "FAIL"
            detail = "mode 分类混合（UNSUPPORTED 与其它状态并存）"
        else:
            for label in ("ERROR", "INVALID", "FAIL", "NOT_RUN"):
                if label in statuses:
                    overall = label
                    break
            detail = "；".join(f"{m['mode']}:{m['status']}:{m.get('detail', '')[:160]}"
                               for m in mode_records if m["status"] != "PASS")
    else:
        detail = "用例没有 mode"

    # eager × jit 一致性：同一条用例两个 mode 都 PASS 时再互比一次（设计 §5.3）。
    consistency_problems = _consistency_check(ctx, case, fixture, mode_records)
    if consistency_problems and overall == "PASS":
        overall, detail = "FAIL", "；".join(consistency_problems)

    rec = {
        "case_id": case_id,
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "shape": case.get("shape") or [fixture["length"], fixture["width"]],
        "dtype": case.get("dtype", fixture["dtype"]),
        "seed": case.get("seed"),
        "expected": case.get("expected", "match_cpu"),
        "note": case.get("note"),
        "status": overall,
        "detail": detail,
        "modes": mode_records,
        "dispatch": mode_records[0]["dispatch"] if mode_records else {},
        "matched_ratio": min((m.get("matched_ratio", 1.0) or 0.0)
                            for m in mode_records if m.get("status") == "PASS") if overall == "PASS" else None,
        "max_abs_err": max((m.get("max_abs_err") or 0.0) for m in mode_records) if mode_records else None,
    }
    _dump_case_evidence(ctx, rec, fixture, case_json, inputs)
    return rec


def _consistency_check(ctx: ScanRun, case: dict, fixture: dict, modes: list[dict]) -> list[str]:
    by_mode = {m["mode"]: m for m in modes}
    if not ("eager" in by_mode and "jit" in by_mode):
        return []
    if any(by_mode[m]["status"] != "PASS" for m in ("eager", "jit")):
        return []
    # 同一 outdir 会被后一个 mode 覆盖，因此读各自 mode 保存的证据快照。
    eager = _snapshot_arrays(ctx, case["case_id"], "eager")
    jit = _snapshot_arrays(ctx, case["case_id"], "jit")
    if eager is None or jit is None:
        return []
    problems: list[str] = []
    if set(eager) != set(jit):
        return [f"eager/jit 输出叶子集合不同：{sorted(eager)} vs {sorted(jit)}"]
    for key in eager:
        a, b = np.asarray(eager[key]), np.asarray(jit[key])
        if a.dtype != b.dtype or a.shape != b.shape:
            problems.append(f"{key}: eager/jit dtype/shape 不同")
            continue
        if a.dtype == np.int32:
            if not np.array_equal(a, b):
                problems.append(f"{key}: eager/jit int32 不一致")
            continue
        verdict = ctx.criteria.check(np.ascontiguousarray(a, dtype=np.float32),
                                     np.ascontiguousarray(b, dtype=np.float32))
        if not verdict["is_pass"]:
            problems.append(f"{key}: eager 与 jit 输出不一致（" + "；".join(
                verdict.get("failure_reasons") or ["数值不一致"]) + "）")
    return problems


def _snapshot_dir(ctx: ScanRun, case_id: str, mode: str) -> Path:
    return ctx.workdir / str(case_id) / "snapshots" / mode


def _snapshot_arrays(ctx: ScanRun, case_id: str, mode: str):
    path = _snapshot_dir(ctx, case_id, mode) / "outputs.npz"
    if not path.is_file():
        return None
    return fixtures.load_inputs_npz(path)


def _snapshot_outputs(ctx: ScanRun, case_id: str, mode: str, outdir: Path) -> None:
    dst = _snapshot_dir(ctx, case_id, mode)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("outputs.npz", "meta.json"):
        src = outdir / name
        if src.is_file():
            shutil.copy2(src, dst / name)


def _dump_case_evidence(ctx: ScanRun, rec: dict, fixture: dict, case_json: Path, inputs: Path) -> None:
    dst = ctx.evidence_dir / str(rec["case_id"])
    dst.mkdir(parents=True, exist_ok=True)
    _dump_json(dst / "meta.json", rec)
    if rec["status"] != "PASS":
        for src in (case_json, inputs):
            if src.is_file():
                shutil.copy2(src, dst / src.name)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _summary(results: list[dict]) -> dict:
    summary = {status: sum(1 for r in results if r["status"] == status) for status in CASE_STATUSES}
    summary["TOTAL"] = len(results)
    summary["mode_executions"] = sum(len(r.get("modes") or []) for r in results)
    summary["UNSUPPORTED_modes"] = sum(
        1 for r in results for m in (r.get("modes") or []) if m.get("status") == "UNSUPPORTED")
    return summary


def _render_md(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"# 正确性报告 · {report['op']}（scan 分支）",
        "",
        f"- 用例集：`{report['case_source']}`（sha256 `{report['case_source_sha256'][:16]}…`）",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}`",
        f"- 代码快照：`{report['code_snapshot_sha256'][:16]}…` ｜ 候选 sha256：`{report['candidate']['sha256'][:16]}…`",
        f"- 汇总：{summary}",
        "",
        "| 用例 | 组 | 必需 | 期望 | 状态 | mode 状态 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["cases"]:
        modes = ", ".join(f"{m['mode']}={m['status']}" for m in (r.get("modes") or []))
        lines.append(
            f"| {r['case_id']} | {r.get('group', '')} | {'是' if r.get('required', True) else '否'} | "
            f"{r.get('expected')} | **{r['status']}** | {modes} | "
            f"{(r.get('detail') or r.get('note') or '').replace('|', '/')[:160]} |"
        )
    lines += [
        "",
        "> 判定口径：`criteria/check.py`（ops-precision-standard@1）逐叶子；int32 精确；非有限位逐元素精确。",
        "> 候选必须在非 CPU 后端执行；golden 一律独立 CPU 子进程。`UNSUPPORTED` 不是 PASS。",
        "> 本文件是给人看的伴生品；机判只认同目录的 `correctness.json`。",
    ]
    return "\n".join(lines) + "\n"


def _finish_report(ctx: ScanRun, *, op: str, cases_path: Path, out: Path, md: Path,
                   results: list[dict], version: dict) -> int:
    summary = _summary(results)
    report = {
        "schema": "opx-correctness/1",
        "op": op,
        "action": "test",
        "run_id": version.get("run_id") or None,
        "spec_version_id": int(version["spec_version_id"]),
        "criteria_ref": version["criteria_ref"],
        "code_snapshot_sha256": version["code_snapshot_sha256"],
        "suite_ref": version["suite_ref"],
        "candidate": {
            "exe": _rel(ctx.ws, ctx.exe),
            "sha256": _sha256_file(ctx.exe),
            "exec_path_marker_required": True,
        },
        "case_source": _rel(ctx.ws, cases_path),
        "case_source_sha256": _sha256_file(cases_path),
        "cases": results,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(out, report)
    cases_copy = out.parent / "cases" / f"{op}-cases.json"
    cases_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cases_path, cases_copy)
    md.write_text(_render_md(report), encoding="utf-8")

    print(f"[run_cases] {summary}  → {out}")
    bad = [r["case_id"] for r in results
           if r["status"] in ("FAIL", "ERROR", "INVALID", "NOT_RUN")]
    if bad:
        print(f"[run_cases] 未通过：{bad}", file=sys.stderr)
    # required 用例的 UNSUPPORTED 仅在 expected 允许时不算失败；UNSUPPORTED 本身不记 PASS。
    unsupported_bad = [
        r["case_id"] for r in results
        if r["status"] == "UNSUPPORTED"
        and r.get("expected") != "match_cpu_or_explicit_unsupported"
    ]
    if unsupported_bad:
        print(f"[run_cases] 不允许的 UNSUPPORTED：{unsupported_bad}", file=sys.stderr)
    ok = not bad and not unsupported_bad
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 对外入口：test / smoke
# ---------------------------------------------------------------------------


def run_suite(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if not cases_path.is_file():
        print(f"[run_cases] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_scan_operator(op, suite):
        print(f"[run_cases] scan 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = ScanRun(ws=ws, op=op, exe=exe, criteria=criteria,
                  workdir=workdir, evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    version = {
        "spec_version_id": spec_version_id,
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "run_id": run_id,
    }
    results = [_run_case(ctx, case) for case in (suite.get("cases") or [])]
    return _finish_report(ctx, op=op, cases_path=cases_path, out=out, md=md,
                          results=results, version=version)


def pick_smoke_case(cases: list[dict]) -> dict:
    """smoke 选例：required + expected=match_cpu + mode 含 jit，按 (length, width, case_id) 升序。

    必须排除 reject 与 match_cpu_or_explicit_unsupported（设计 §2.2.3 A1.3 / V-007）。
    """
    candidates = []
    for case in cases:
        if not case.get("required", True):
            continue
        if case.get("expected") != "match_cpu":
            continue
        if "jit" not in (case.get("mode") or []):
            continue
        geometry = fixtures.case_geometry(case)
        candidates.append((geometry["length"], geometry["width"], str(case.get("case_id")), case))
    if not candidates:
        raise SystemExit(
            "cases 里没有可用的 required match_cpu + jit 用例"
            "（smoke 需要至少一条能比对的非拒绝用例）"
        )
    return min(candidates, key=lambda item: item[:3])[3]


def run_smoke(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if not cases_path.is_file():
        print(f"[smoke] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_scan_operator(op, suite):
        print(f"[smoke] scan 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = ScanRun(ws=ws, op=op, exe=exe, criteria=criteria,
                  workdir=workdir, evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    case = pick_smoke_case(suite.get("cases") or [])
    modes = ["jit"] if "jit" in (case.get("mode") or []) else list(case.get("mode") or ["eager"])[:1]
    print(f"[smoke] 最小用例：{case['case_id']} modes={modes}（scan 分支）")
    rec = _run_case(ctx, case, modes_override=modes)
    passed = rec.get("status") == "PASS"
    report = {
        "schema": "opx-smoke/1",
        "op": op,
        "action": "smoke",
        "run_id": run_id or None,
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "candidate": {
            "exe": _rel(ws, exe),
            "sha256": _sha256_file(exe),
            "exec_path_marker_required": True,
        },
        "case_source": _rel(ws, cases_path),
        "case_source_sha256": _sha256_file(cases_path),
        "case": rec,
        "verdict": "PASS" if passed else "FAIL",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(out, report)
    lines = [
        f"# 冒烟结果：{op}（scan 分支）",
        "",
        f"- 用例：`{rec.get('case_id')}`（mode={','.join(modes)}，length={fixtures.case_geometry(case)['length']}）",
        f"- 结论：**{report['verdict']}**",
        f"- 候选：`{report['candidate']['exe']}` sha256 `{report['candidate']['sha256'][:16]}…`",
        f"- 判据档：`{criteria_ref}`；用例集：`{suite_ref}`",
    ]
    if not passed:
        lines += ["", f"- 失败原因：{rec.get('detail') or rec.get('status')}"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[smoke] SMOKE_VERDICT {report['verdict']}  → {out}")
    return 0 if passed else 1
