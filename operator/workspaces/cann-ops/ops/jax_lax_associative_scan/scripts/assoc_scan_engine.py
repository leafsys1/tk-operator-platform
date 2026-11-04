# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan 的 assoc-scan 分支执行核心。

被 ``tools/run_cases.py``（test）与 ``tools/smoke_case.py``（smoke）按 ``op`` 显式分派复用。
assoc-scan 用例字段模型（``op_kind/axis/reverse/shape|elems/mode/expected``）与 elementwise 的
``n_a/n_b``、与 ``jax_lax_scan`` 的 carry 模型都不同，但**统一入口与判据只有一个**：本模块把
assoc-scan 特有的输入、候选调用、golden 子进程、逐叶子判定与报告组装集中实现，共享 runner
只做显式分派；elementwise / jax_lax_scan 旧分支原样保留。

候选可执行文件契约（设计 §1.3.3）
---------------------------------

``ops/jax_lax_associative_scan/build/jax_lax_associative_scan \\
  --case-json <case.json> --inputs <inputs.npz> --mode <eager|jit|jvp|vjp> --outdir <dir>``

* 成功：rc=0；``<outdir>/outputs.npz``（单叶子 ``out``；pytree ``out/a``、``out/b``；
  AD ``jvp_primal``+``jvp_tangent`` / ``vjp_primal``+``vjp_cotangent``）+ ``<outdir>/meta.json``；
  stdout 首行 ``EXEC op=jax_lax_associative_scan kernel=... arch=dav-2201 mode=... ...``。
* 拒绝：rc=3；输出含 ``REJECT code=<稳定码>``；``EXEC ... path=reject exec_mode=reject abi_calls=0``；
  不得写 ``outputs.npz``。
* 显式 unsupported：rc=4；输出含 ``UNSUPPORTED:autodiff_unsupported`` + 后端原始报错。
* 其他非 0：ERROR（不得因“别的原因失败”算拒绝通过）。

golden 一律走独立 CPU 子进程 ``golden_cli.py``；本模块**不在 runner 进程 import jax**，避免候选侧
backend 被 golden 的 CPU 环境限制（设计 §2.1.1）。
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

ASSOC_OP = "jax_lax_associative_scan"
KERNEL_NAME = "jax_lax_associative_scan_kernel"
ARCH = "dav-2201"
HOST_LIB = f"lib{ASSOC_OP}_host.so"
CANDIDATE_EXIT_OK = 0
CANDIDATE_EXIT_REJECT = 3
CANDIDATE_EXIT_UNSUPPORTED = 4
UNSUPPORTED_CODE = "UNSUPPORTED:autodiff_unsupported"
GOLDEN_TIMEOUT_S = 180
CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN", "UNSUPPORTED")
AD_MODES = ("jvp", "vjp")
EXEC_REQUIRED_KEYS = ("op", "kernel", "arch", "mode", "combine", "family",
                      "path", "exec_mode", "abi_calls", "backend")
META_DISPATCH_KEYS = ("path", "exec_mode", "abi_calls", "combine_id", "family",
                      "backend", "plan_id", "plan_cache", "reason")


def is_assoc_operator(op: str, suite: dict | None = None) -> bool:
    """显式分派判据：只认 op 名（不用“有没有 mode 字段”这种会漂移的隐式条件）。"""
    if op != ASSOC_OP:
        return False
    return suite is None or suite.get("op") in (None, ASSOC_OP)


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


def expected_host_so(ws: Path, op: str = ASSOC_OP) -> Path:
    """候选 Python 接口实际 dlopen 的 host 共享库（构建产物）。"""
    return Path(ws) / "ops" / op / "build" / f"lib{op}_host.so"


def ensure_host_so(ws: Path, op: str = ASSOC_OP, log_dir: Path | None = None) -> tuple[bool, str]:
    """``.so`` preflight（设计 §2.2.3）：缺失时按同一份源码补编一次，仍缺则失败。

    ``test.sh`` 对 test 动作自身也做同样的检查；smoke/benchmark 由本函数承担
    （``benchmark.sh`` 不改，保持原行为）。补编显式设置 ``OPX_OP``/``OPX_WS``，
    不依赖 ``build.sh`` 的默认 ``elementwise_add``。
    """
    so = expected_host_so(ws, op)
    if so.is_file():
        return True, str(so)
    build = Path(ws) / "build.sh"
    print(f"[assoc_scan] 缺少 {so} → 按同一份源码补编一次（OPX_OP={op}）", file=sys.stderr)
    if not build.is_file():
        print(f"MISSING_SO {so}（且找不到 {build}）", file=sys.stderr)
        return False, str(so)
    env = os.environ.copy()
    env["OPX_OP"] = op
    env["OPX_WS"] = str(ws)
    try:
        proc = subprocess.run(["bash", str(build)], cwd=str(ws), env=env,
                              capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"MISSING_SO {so}（补编超时）", file=sys.stderr)
        return False, str(so)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "build_for_so.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if not so.is_file():
        print(f"MISSING_SO {so}（补编 rc={proc.returncode}）", file=sys.stderr)
        print(_tail((proc.stdout or "") + (proc.stderr or ""), 800), file=sys.stderr)
        return False, str(so)
    print(f"[assoc_scan] 补编后已产出 {so}", file=sys.stderr)
    return True, str(so)


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
                                 require_op: bool = True,
                                 expected_mode: str | None = None) -> list[str]:
    """校验一次 ``rc=4`` 尝试是否满足“显式 unsupported”的硬证据（设计 §1.3.6）。

    要求：``path=trace``、``abi_calls=0``、``backend`` 非 CPU、附后端原始报错；
    ``op``/``mode`` 与当前用例一致。返回问题列表；调用方必须判 ``INVALID``，
    **不能**因此记 ``UNSUPPORTED``。``UNSUPPORTED`` 本身也绝不记 PASS。
    """
    dispatch = dispatch or {}
    problems: list[str] = []
    if require_op and dispatch.get("op") != ASSOC_OP:
        problems.append(f"UNSUPPORTED 的 EXEC op 必须是 {ASSOC_OP}，收到 {dispatch.get('op')!r}")
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
    cli = ws / "ops" / ASSOC_OP / "scripts" / "golden_cli.py"
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


def _combine_tokens(fixture: dict) -> set[str]:
    name = str(fixture["combine_name"])
    tokens = {name, str(fixture["combine_id"])}
    if name == "pytree_add":
        tokens |= {"tree_add", "combine_tree_add"}
    else:
        tokens.add(f"combine_{name}")
    return tokens


def _meta_leaf_map(meta: dict) -> dict[str, dict]:
    out = {}
    for leaf in meta.get("leaves") or []:
        if isinstance(leaf, dict) and "path" in leaf:
            out[str(leaf["path"])] = leaf
    return out


def _validate_so_fingerprint(meta: dict, ctx: "AssocRun") -> list[str]:
    fingerprint = meta.get("so_fingerprint")
    if not isinstance(fingerprint, dict):
        return ["meta.so_fingerprint 缺失（必须由 Python 接口实际 dlopen 的库计算）"]
    problems = []
    for key in ("path", "size_bytes", "sha256", "abi"):
        if fingerprint.get(key) in (None, ""):
            problems.append(f"meta.so_fingerprint 缺 {key}")
    expected = expected_host_so(ctx.ws, ctx.op)
    if problems:
        return problems
    if not expected.is_file():
        return [f"无法核对 so_fingerprint：构建产物不存在 {expected}"]
    actual_size = expected.stat().st_size
    actual_sha = _sha256_file(expected)
    try:
        reported_size = int(fingerprint["size_bytes"])
    except (TypeError, ValueError):
        reported_size = -1
    if reported_size != actual_size:
        problems.append(f"so_fingerprint.size_bytes={reported_size} != 实际 {actual_size}")
    if str(fingerprint["sha256"]) != actual_sha:
        problems.append(
            f"so_fingerprint.sha256={str(fingerprint['sha256'])[:16]}… != 实际 {actual_sha[:16]}…")
    if not str(fingerprint.get("path", "")).endswith(HOST_LIB):
        problems.append(f"so_fingerprint.path 不是 {HOST_LIB}：{fingerprint.get('path')!r}")
    return problems


def _validate_candidate_meta(meta: dict, *, ctx: "AssocRun", case_id: str, mode: str,
                             fixture: dict, outputs: dict) -> tuple[list[str], dict[str, dict]]:
    problems: list[str] = []
    if str(meta.get("case_id")) != str(case_id):
        problems.append(f"meta.case_id={meta.get('case_id')!r} != {case_id!r}")
    if str(meta.get("mode")) != mode:
        problems.append(f"meta.mode={meta.get('mode')!r} != {mode!r}")
    if meta.get("status") != "ok":
        problems.append(f"meta.status={meta.get('status')!r}（成功路径必须是 ok）")
    if meta.get("inputs_sha256") != fixture["inputs_sha256"]:
        problems.append(
            f"meta.inputs_sha256 与本次 inputs.npz 不一致：{meta.get('inputs_sha256')} != "
            f"{fixture['inputs_sha256']}（候选读到的不是这条用例的数据）")
    dispatch = meta.get("dispatch")
    if not isinstance(dispatch, dict):
        problems.append("meta.dispatch 缺失")
        dispatch = {}
    missing = [key for key in META_DISPATCH_KEYS if key not in dispatch]
    if missing:
        problems.append(f"meta.dispatch 缺字段：{missing}")
    if _as_int(dispatch.get("combine_id")) != int(fixture["combine_id"]):
        problems.append(
            f"meta.dispatch.combine_id={dispatch.get('combine_id')!r} != {fixture['combine_id']}")
    if str(dispatch.get("family")) != "numpy":
        problems.append(f"meta.dispatch.family={dispatch.get('family')!r}（门禁 family 必须是 numpy）")
    if str(dispatch.get("backend")) in ("", "None", "cpu"):
        problems.append(f"meta.dispatch.backend={dispatch.get('backend')!r}（候选必须在非 CPU 后端执行）")
    if _as_int(dispatch.get("device_id")) != 0:
        problems.append(f"meta.dispatch.device_id={dispatch.get('device_id')!r}（必须是 0）")

    expected_keys = set(fixtures.output_shapes_of(fixture, mode))
    actual_keys = set(outputs)
    if actual_keys != expected_keys:
        problems.append(f"输出叶子集合不一致：候选 {sorted(actual_keys)} != 期望 {sorted(expected_keys)}")
    leaf_map = _meta_leaf_map(meta)
    missing_leaves = [key for key in expected_keys if key not in leaf_map]
    if missing_leaves:
        problems.append(f"meta.leaves 缺输出叶子：{missing_leaves}")
    extra = [key for key in leaf_map if key not in expected_keys]
    if extra:
        problems.append(f"meta.leaves 多出 outputs.npz 没有的叶子：{extra}")
    expected_shapes = fixtures.output_shapes_of(fixture, mode)
    expected_dtype = "float32"
    for key in sorted(expected_keys):
        leaf = leaf_map.get(key) or {}
        if leaf.get("dtype") != expected_dtype:
            problems.append(f"{key}: meta.leaves.dtype={leaf.get('dtype')!r} != {expected_dtype}")
        if list(leaf.get("shape") or []) != list(expected_shapes[key]):
            problems.append(f"{key}: meta.leaves.shape={leaf.get('shape')} != {expected_shapes[key]}")
        if not leaf.get("sha256"):
            problems.append(f"{key}: meta.leaves 缺 sha256")
        if key in outputs:
            arr = np.asarray(outputs[key])
            if str(arr.dtype) != expected_dtype:
                problems.append(f"{key}: outputs.npz dtype={arr.dtype} != {expected_dtype}")
            if list(arr.shape) != list(expected_shapes[key]):
                problems.append(f"{key}: outputs.npz shape={list(arr.shape)} != {expected_shapes[key]}")
    problems.extend(_validate_so_fingerprint(meta, ctx))
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
        rec["meta_sha256"] = meta_leaf.get("sha256")
        if got.dtype != want.dtype:
            problems.append(f"{key}: dtype {got.dtype} != golden {want.dtype}")
            leaves.append(rec)
            continue
        if got.shape != want.shape:
            problems.append(f"{key}: shape {got.shape} != golden {want.shape}")
            leaves.append(rec)
            continue
        if got.dtype != np.float32:
            problems.append(f"{key}: 本批判据只定义 float32，收到 {got.dtype}")
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
            exact_equal=bool(np.array_equal(got, want)),
        )
        if not verdict["is_pass"]:
            problems.append(f"{key}: " + "；".join(verdict.get("failure_reasons") or ["数值未通过"]))
        leaves.append(rec)
    return leaves, problems


def _exec_problems(exec_fields: dict | None, mode: str, fixture: dict) -> list[str]:
    if exec_fields is None:
        return ["候选没有打印 EXEC 标记行 → 无法证明实际执行的是候选算子路径"]
    problems = []
    missing = [key for key in EXEC_REQUIRED_KEYS if key not in exec_fields]
    if missing:
        problems.append(f"EXEC 缺字段：{missing}")
        return problems
    if exec_fields.get("op") != ASSOC_OP:
        problems.append(f"EXEC op={exec_fields.get('op')!r} != {ASSOC_OP!r}")
    if exec_fields.get("kernel") != KERNEL_NAME:
        problems.append(f"EXEC kernel={exec_fields.get('kernel')!r} != {KERNEL_NAME!r}")
    if exec_fields.get("arch") != ARCH:
        problems.append(f"EXEC arch={exec_fields.get('arch')!r} != {ARCH!r}")
    if exec_fields.get("mode") != mode:
        problems.append(f"EXEC mode={exec_fields.get('mode')!r} != {mode!r}")
    if str(exec_fields.get("combine")) not in _combine_tokens(fixture):
        problems.append(
            f"EXEC combine={exec_fields.get('combine')!r}（允许 {sorted(_combine_tokens(fixture))}）")
    if str(exec_fields.get("family")) != "numpy":
        problems.append(f"EXEC family={exec_fields.get('family')!r}（门禁 family 必须是 numpy）")
    if str(exec_fields.get("backend")) in ("", "None", "cpu"):
        problems.append(f"EXEC backend={exec_fields.get('backend')!r}（候选必须证明非 CPU）")
    return problems


def _dispatch_problems(mode: str, exec_fields: dict, meta: dict, fixture: dict) -> list[str]:
    problems: list[str] = []
    dispatch = meta.get("dispatch") or {}
    expected_exec_mode = {"eager": "per_leaf", "jit": "whole_tree",
                          "jvp": ("jvp", "trace"), "vjp": ("vjp", "trace")}.get(mode)
    if mode in AD_MODES:
        path = dispatch.get("path")
        if path not in ("so", "trace"):
            problems.append(f"AD 的 dispatch.path 必须是 so 或 trace，收到 {path!r}")
        if path == "so":
            if _as_int(dispatch.get("abi_calls")) is None or int(dispatch["abi_calls"]) < 1:
                problems.append(f"AD kernel 路径 abi_calls 必须 >=1，收到 {dispatch.get('abi_calls')!r}")
            if str(dispatch.get("exec_mode")) not in expected_exec_mode:
                problems.append(
                    f"AD kernel 路径 exec_mode 必须是 jvp/vjp，收到 {dispatch.get('exec_mode')!r}")
        else:
            if _as_int(dispatch.get("abi_calls")) != 0:
                problems.append(f"AD trace 路径 abi_calls 必须为 0，收到 {dispatch.get('abi_calls')!r}")
            if str(dispatch.get("exec_mode")) != "trace":
                problems.append(f"AD trace 路径 exec_mode 必须是 trace，收到 {dispatch.get('exec_mode')!r}")
        return problems

    if dispatch.get("path") != "so":
        problems.append(f"{mode} 数值路径必须走 .so，dispatch.path={dispatch.get('path')!r}")
    if str(dispatch.get("exec_mode")) != expected_exec_mode:
        problems.append(
            f"{mode} 的 dispatch.exec_mode 必须是 {expected_exec_mode}，收到 {dispatch.get('exec_mode')!r}")
    if exec_fields.get("exec_mode") != expected_exec_mode:
        problems.append(
            f"{mode} 的 EXEC exec_mode 必须是 {expected_exec_mode}，收到 {exec_fields.get('exec_mode')!r}")
    expected_calls = len(fixture["leaves"]) if mode == "eager" else 1
    if _as_int(dispatch.get("abi_calls")) != expected_calls:
        problems.append(f"{mode} 的 dispatch.abi_calls 必须是 {expected_calls}，收到 {dispatch.get('abi_calls')!r}")
    if _as_int(exec_fields.get("abi_calls")) != expected_calls:
        problems.append(f"{mode} 的 EXEC abi_calls 必须是 {expected_calls}，收到 {exec_fields.get('abi_calls')!r}")
    if mode == "eager":
        if dispatch.get("plan_cache") is not None:
            problems.append(f"eager 的 plan_cache 必须是 null，收到 {dispatch.get('plan_cache')!r}")
    else:
        if not str(dispatch.get("plan_id") or "").strip():
            problems.append("jit 的 dispatch.plan_id 不能为空（编译计划稳定标识）")
    return problems


# ---------------------------------------------------------------------------
# 单条用例 / 单 mode
# ---------------------------------------------------------------------------


class AssocRun:
    """一次 assoc-scan 用例执行的上下文（版本字段、路径、判据）。"""

    def __init__(self, *, ws: Path, op: str, exe: Path, criteria, workdir: Path,
                 evidence_dir: Path, timeout: int):
        self.ws = ws
        self.op = op
        self.exe = exe
        self.criteria = criteria
        self.workdir = workdir
        self.evidence_dir = evidence_dir
        self.timeout = timeout


def _prepare_case(ctx: AssocRun, case: dict) -> tuple[dict, Path, Path]:
    case_id = case["case_id"]
    fixture = fixtures.make_fixture(case)
    cdir = ctx.workdir / str(case_id)
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    return fixture, case_json, inputs


def _run_mode(ctx: AssocRun, case: dict, fixture: dict, case_json: Path, inputs: Path,
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

    allow_unsupported = (mode in AD_MODES
                         and expected == "match_cpu_or_explicit_unsupported")
    max_attempts = 3 if allow_unsupported else 1
    attempt_records: list[dict] = []
    usable_run: dict | None = None
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
            if any(item["unsupported_marker"] for item in attempt_records[:-1]):
                attempt_records.append({"attempt": index + 1, "rc": run["rc"],
                                        "unstable": "UNSUPPORTED 后又成功"})
            break
        if run["rc"] == CANDIDATE_EXIT_UNSUPPORTED and _has_marker(run, UNSUPPORTED_CODE):
            continue
        usable_run = run
        break
    rec["attempts"] = attempt_records
    rec["duration_ms"] = sum(item.get("duration_ms", 0) for item in attempt_records)

    if (usable_run is None and max_attempts == 3
            and len(attempt_records) == 3
            and all(item["rc"] == CANDIDATE_EXIT_UNSUPPORTED and item["unsupported_marker"]
                    for item in attempt_records)):
        problems: list[str] = []
        for attempt in attempt_records:
            problems.extend(unsupported_attempt_problems(
                attempt.get("exec"), attempt.get("backend_error") or "",
                require_op=True, expected_mode=mode))
        problems = list(dict.fromkeys(problems))  # 3 次同码会重复同一问题，去重便于定位
        rec["backend_errors"] = [item["stderr_tail"] for item in attempt_records]
        rec["dispatch"] = dict(attempt_records[-1].get("exec") or {})
        if problems:
            rec.update(status="INVALID",
                       detail="显式 unsupported 分类缺少硬证据，不能记 UNSUPPORTED：" + "；".join(problems))
            return rec
        rec.update(status="UNSUPPORTED",
                   detail="3 次均显式 UNSUPPORTED:autodiff_unsupported（path=trace/abi_calls=0/"
                          "backend 非 CPU，附后端原文）")
        return rec

    run = usable_run or _exec_once(ctx.exe, case_json, inputs, mode, cdir / "out", ctx.timeout)
    if any(item.get("unstable") for item in attempt_records):
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

    exec_problems = _exec_problems(run["exec"], mode, fixture)
    if exec_problems:
        rec.update(status="INVALID", detail="；".join(exec_problems))
        return rec
    rec["dispatch"] = {
        "path": run["exec"].get("path"),
        "exec_mode": run["exec"].get("exec_mode"),
        "abi_calls": _as_int(run["exec"].get("abi_calls")),
        "combine_id": _as_int(run["exec"].get("combine_id")),
        "family": run["exec"].get("family"),
        "backend": run["exec"].get("backend"),
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
        meta, ctx=ctx, case_id=case_id, mode=mode, fixture=fixture, outputs=outputs)
    meta_problems.extend(_dispatch_problems(mode, run["exec"], meta, fixture))
    if meta_problems:
        rec.update(status="INVALID", detail="；".join(meta_problems))
        rec["leaves"] = [
            {"path": key, "dtype": str(np.asarray(value).dtype),
             "shape": list(np.asarray(value).shape)}
            for key, value in sorted(outputs.items())
        ]
        return rec
    # 报告里的 dispatch 以候选 meta 为准（EXEC 只承载可观测的路径/次数，不含
    # device_id/plan_id/plan_cache；设计 §1.3.3 要求 meta.dispatch 给出这些字段）。
    rec["dispatch"] = dict(meta.get("dispatch") or {})
    rec["dispatch"]["abi_calls"] = _as_int(rec["dispatch"].get("abi_calls"))
    rec["dispatch"]["combine_id"] = _as_int(rec["dispatch"].get("combine_id"))
    rec["dispatch"]["device_id"] = _as_int(rec["dispatch"].get("device_id"))
    rec["so_fingerprint"] = meta.get("so_fingerprint")

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

    expected_keys = [key for key in fixtures.output_shapes_of(fixture, mode)]
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


def _run_reject_mode(ctx: AssocRun, case: dict, fixture: dict, case_json: Path, inputs: Path,
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

    expected_code = case.get("expect_reject_code", "invalid_argument")
    text = (run["stdout"] or "") + (run["stderr"] or "")
    code_hit = expected_code in text
    exec_fields = run["exec"] or {}
    abi_calls = _as_int(exec_fields.get("abi_calls"))
    rec["dispatch"] = {
        "path": exec_fields.get("path"),
        "exec_mode": exec_fields.get("exec_mode"),
        "abi_calls": abi_calls,
        "combine_id": _as_int(exec_fields.get("combine_id")),
        "family": exec_fields.get("family"),
        "backend": exec_fields.get("backend"),
    }

    golden = _run_golden(ctx.ws, case_json, inputs, "reject", cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"), "cmd": golden.get("cmd")}
    if not golden["ok"]:
        rec.update(status="ERROR",
                   detail=f"golden 未能确认该用例应被拒绝：{golden.get('detail')}")
        return rec
    golden_meta = golden.get("meta") or {}
    rec["golden"].update({
        "backend": golden_meta.get("backend"),
        "jax": golden_meta.get("jax"),
        "jaxlib": golden_meta.get("jaxlib"),
        "reject_code": golden_meta.get("reject_code"),
        "reject_evidence": golden_meta.get("reject_evidence"),
    })
    rec["reject_evidence"] = golden_meta.get("reject_evidence")

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
    if exec_fields.get("op") != ASSOC_OP or exec_fields.get("mode") != mode:
        rec.update(status="FAIL",
                   detail=f"EXEC 拒绝标记不对：op={exec_fields.get('op')!r} mode={exec_fields.get('mode')!r}")
        return rec
    if exec_fields.get("path") != "reject" or exec_fields.get("exec_mode") != "reject":
        rec.update(status="FAIL",
                   detail=f"EXEC 拒绝标记不对：path={exec_fields.get('path')!r} "
                          f"exec_mode={exec_fields.get('exec_mode')!r}")
        return rec
    if (cdir / "out" / "outputs.npz").is_file():
        rec.update(status="FAIL", detail="拒绝用例不得写 outputs.npz")
        return rec

    meta, meta_err = _load_candidate_meta(cdir / "out")
    if meta is None:
        rec.update(status="INVALID", detail=meta_err)
        return rec
    meta_dispatch = meta.get("dispatch") or {}
    meta_problems = []
    if meta.get("status") != "reject":
        meta_problems.append(f"拒绝时 meta.status 必须是 reject，收到 {meta.get('status')!r}")
    if meta_dispatch.get("path") != "reject":
        meta_problems.append(f"拒绝时 meta.dispatch.path 必须是 reject，收到 {meta_dispatch.get('path')!r}")
    if _as_int(meta_dispatch.get("abi_calls")) != 0:
        meta_problems.append(f"拒绝时 meta.dispatch.abi_calls 必须是 0，收到 {meta_dispatch.get('abi_calls')!r}")
    if meta.get("leaves"):
        meta_problems.append("拒绝时 meta.leaves 必须为空")
    rec["meta"] = {"status": meta.get("status"), "dispatch": meta_dispatch}
    if meta_problems:
        rec.update(status="INVALID", detail="；".join(meta_problems))
        return rec
    rec.update(status="PASS", detail=f"按预期拒绝：rc=3 {expected_code}；abi_calls=0；无输出")
    return rec


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_case(ctx: AssocRun, case: dict, modes_override: list[str] | None = None) -> dict:
    """单条用例执行；执行器自身异常收敛成该用例 ERROR，不拖垮整轮报告。"""
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


def _run_case_impl(ctx: AssocRun, case: dict, modes_override: list[str] | None = None) -> dict:
    case_id = case["case_id"]
    fixture, case_json, inputs = _prepare_case(ctx, case)

    modes = list(modes_override) if modes_override is not None else list(case.get("mode") or ["jit"])
    mode_records = []
    for mode in modes:
        mode_records.append(_run_mode(ctx, case, fixture, case_json, inputs, mode))
        # 同一用例的 eager/jit 共用一个 outdir；此刻立刻快照，供稍后的 eager×jit 一致性互比。
        _snapshot_outputs(ctx, case_id, mode, ctx.workdir / str(case_id) / "out")

    statuses = [item["status"] for item in mode_records]
    overall = "NOT_RUN"
    detail = ""
    if mode_records:
        if all(status == "PASS" for status in statuses):
            overall, detail = "PASS", "全部 mode 通过"
        elif all(status == "UNSUPPORTED" for status in statuses):
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
            detail = "；".join(f"{item['mode']}:{item['status']}:{item.get('detail', '')[:160]}"
                               for item in mode_records if item["status"] != "PASS")
    else:
        detail = "用例没有 mode"

    consistency_problems = _consistency_check(ctx, case, fixture, mode_records)
    if consistency_problems and overall == "PASS":
        overall, detail = "FAIL", "；".join(consistency_problems)

    rec = {
        "case_id": case_id,
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "shape": case.get("shape") or (
            [int(case["length"]), int(case["width"])] if case.get("length") is not None else None),
        "geometry": fixture["geometry"],
        "dtype": case.get("dtype", fixture["dtype"]),
        "seed": case.get("seed"),
        "expected": case.get("expected", "match_cpu"),
        "note": case.get("note"),
        "status": overall,
        "detail": detail,
        "modes": mode_records,
        "dispatch": mode_records[0]["dispatch"] if mode_records else {},
        "matched_ratio": min((item.get("matched_ratio", 1.0) or 0.0)
                             for item in mode_records if item.get("status") == "PASS")
        if overall == "PASS" else None,
        "max_abs_err": max((item.get("max_abs_err") or 0.0) for item in mode_records) if mode_records else None,
    }
    _dump_case_evidence(ctx, rec, fixture, case_json, inputs)
    return rec


def _consistency_check(ctx: AssocRun, case: dict, fixture: dict, modes: list[dict]) -> list[str]:
    by_mode = {item["mode"]: item for item in modes}
    if not ("eager" in by_mode and "jit" in by_mode):
        return []
    if any(by_mode[mode]["status"] != "PASS" for mode in ("eager", "jit")):
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
        verdict = ctx.criteria.check(np.ascontiguousarray(a, dtype=np.float32),
                                     np.ascontiguousarray(b, dtype=np.float32))
        if not verdict["is_pass"]:
            problems.append(f"{key}: eager 与 jit 输出不一致（" + "；".join(
                verdict.get("failure_reasons") or ["数值不一致"]) + "）")
    return problems


def _snapshot_dir(ctx: AssocRun, case_id: str, mode: str) -> Path:
    return ctx.workdir / str(case_id) / "snapshots" / mode


def _snapshot_arrays(ctx: AssocRun, case_id: str, mode: str):
    path = _snapshot_dir(ctx, case_id, mode) / "outputs.npz"
    if not path.is_file():
        return None
    return fixtures.load_inputs_npz(path)


def _snapshot_outputs(ctx: AssocRun, case_id: str, mode: str, outdir: Path) -> None:
    dst = _snapshot_dir(ctx, case_id, mode)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("outputs.npz", "meta.json"):
        src = outdir / name
        if src.is_file():
            shutil.copy2(src, dst / name)


def _dump_case_evidence(ctx: AssocRun, rec: dict, fixture: dict, case_json: Path, inputs: Path) -> None:
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
    summary = {status: sum(1 for item in results if item["status"] == status) for status in CASE_STATUSES}
    summary["TOTAL"] = len(results)
    summary["mode_executions"] = sum(len(item.get("modes") or []) for item in results)
    summary["UNSUPPORTED_modes"] = sum(
        1 for item in results for mode in (item.get("modes") or [])
        if mode.get("status") == "UNSUPPORTED")
    return summary


def _render_md(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"# 正确性报告 · {report['op']}（assoc-scan 分支）",
        "",
        f"- 用例集：`{report['case_source']}`（sha256 `{report['case_source_sha256'][:16]}…`）",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 规格版本：v{report['spec_version_id']} ｜ "
        f"判据档：`{report['criteria_ref']}`",
        f"- 代码快照：`{report['code_snapshot_sha256'][:16]}…` ｜ "
        f"候选 sha256：`{report['candidate']['sha256'][:16]}…`",
        f"- 汇总：{summary}",
        "",
        "| 用例 | 组 | 必需 | 期望 | 状态 | mode 状态 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["cases"]:
        modes = ", ".join(f"{mode['mode']}={mode['status']}" for mode in (item.get("modes") or []))
        lines.append(
            f"| {item['case_id']} | {item.get('group', '')} | "
            f"{'是' if item.get('required', True) else '否'} | {item.get('expected')} | "
            f"**{item['status']}** | {modes} | "
            f"{(item.get('detail') or item.get('note') or '').replace('|', '/')[:160]} |")
    lines += [
        "",
        "> 判定口径：`criteria/check.py`（ops-precision-standard@1）逐叶子；结构/shape/dtype 精确；",
        "> 非有限位逐元素精确。候选必须在非 CPU 后端执行；golden 一律独立 CPU 子进程。",
        "> `UNSUPPORTED` 不是 PASS。本文件是给人看的伴生品；机判只认同目录的 `correctness.json`。",
    ]
    return "\n".join(lines) + "\n"


def _finish_report(ctx: AssocRun, *, op: str, cases_path: Path, out: Path, md: Path,
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
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render_md(report), encoding="utf-8")

    print(f"[run_cases] {summary}  → {out}")
    bad = [item["case_id"] for item in results
           if item["status"] in ("FAIL", "ERROR", "INVALID", "NOT_RUN")]
    if bad:
        print(f"[run_cases] 未通过：{bad}", file=sys.stderr)
    # required 用例的 UNSUPPORTED 仅在 expected 允许时不算失败；UNSUPPORTED 本身不记 PASS。
    unsupported_bad = [
        item["case_id"] for item in results
        if item["status"] == "UNSUPPORTED"
        and item.get("expected") != "match_cpu_or_explicit_unsupported"
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
    if not is_assoc_operator(op, suite):
        print(f"[run_cases] assoc 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = AssocRun(ws=ws, op=op, exe=exe, criteria=criteria,
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
    """smoke 选例（设计 §2.2.3）：required + expected=match_cpu + mode 含 jit，
    按 ``(numel, case_id)`` 升序；当前应选 ``assoc_sum_len1``（16 元素）。
    """
    candidates = []
    for case in cases:
        if not case.get("required", True):
            continue
        if case.get("expected") != "match_cpu":
            continue
        if "jit" not in (case.get("mode") or []):
            continue
        leaves, _kind = fixtures.case_leaves(case)
        numel = int(sum(int(np.prod(leaf["shape"], dtype=np.int64)) for leaf in leaves))
        candidates.append((numel, str(case.get("case_id")), case))
    if not candidates:
        raise SystemExit(
            "cases 里没有可用的 required match_cpu + jit 用例"
            "（smoke 需要至少一条能比对的非拒绝用例）")
    return min(candidates, key=lambda item: item[:2])[2]


def run_smoke(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if not cases_path.is_file():
        print(f"[smoke] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_assoc_operator(op, suite):
        print(f"[smoke] assoc 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}", file=sys.stderr)
        return 2
    ok, detail = ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[smoke] MISSING_SO {detail}（不在缺共享库时降级为通过）", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = AssocRun(ws=ws, op=op, exe=exe, criteria=criteria,
                   workdir=workdir, evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    case = pick_smoke_case(suite.get("cases") or [])
    modes = ["jit"] if "jit" in (case.get("mode") or []) else list(case.get("mode") or ["eager"])[:1]
    print(f"[smoke] 最小用例：{case['case_id']} modes={modes}（assoc-scan 分支）")
    rec = _run_case(ctx, case, modes_override=modes)
    passed = rec.get("status") == "PASS"
    leaves, _kind = fixtures.case_leaves(case)
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
        f"# 冒烟结果：{op}（assoc-scan 分支）",
        "",
        f"- 用例：`{rec.get('case_id')}`（mode={','.join(modes)}，"
        f"shape={[leaf['shape'] for leaf in leaves]}）",
        f"- 结论：**{report['verdict']}**",
        f"- 候选：`{report['candidate']['exe']}` sha256 `{report['candidate']['sha256'][:16]}…`",
        f"- 判据档：`{criteria_ref}`；用例集：`{suite_ref}`",
    ]
    if not passed:
        lines += ["", f"- 失败原因：{rec.get('detail') or rec.get('status')}"]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[smoke] SMOKE_VERDICT {report['verdict']}  → {out}")
    return 0 if passed else 1
