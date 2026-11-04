# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond 的 cond 分支执行核心（共享 runner 的唯一门禁实现）。

被 ``tools/run_cases.py``（test）与 ``tools/smoke_case.py``（smoke）按 ``op`` 显式分派复用。
cond 的字段模型（``pred/branch_ids/mode/variant/pytree``）与 elementwise 的 ``n_a/n_b``、
``jax_lax_scan`` 的 carry、``jax_lax_associative_scan`` 的 combine 都不同，但**统一入口与判据
只有一个**：本模块把 cond 特有的输入、候选调用、golden 子进程、逐叶判定、分支证据与报告组装
集中实现；elementwise / scan / assoc-scan 旧分支原样保留。

候选可执行文件契约（设计 §1.3.3）
--------------------------------

``ops/jax_lax_cond/build/jax_lax_cond \\
  --case-json <case.json> --inputs <inputs.npz> --mode <eager|jit|grad_eager|grad_jit> \\
  [--variant <name>] --outdir <dir>``

* 成功：rc=0；``<outdir>/outputs.npz`` + ``<outdir>/meta.json``；stdout 首行 ``EXEC ...``。
* 拒绝：rc=3；输出含 ``REJECT code=<稳定码>`` 与 ``UPSTREAM type=<类型> sha256=<摘要>
  summary=<截断消息>``；``EXEC ... path=reject exec_mode=reject abi_calls=0``；不得写
  ``outputs.npz``。CPU 可确认的拒绝要求 ``upstream.type == cpu_exception_type``；
  c13 的 ``int32(0/1)`` 是已确认的规格规则拒绝，CPU 不抛错，要求候选给
  ``type=None + spec_rule=scalar_bool_pred + cpu_probe.rejects=false``，缺项即 FAIL。
* 其他非 0：ERROR（不得因“别的原因失败”算拒绝通过）。本任务没有合法 ``UNSUPPORTED``。

golden 一律走独立 CPU 子进程 ``golden_cli.py``；本模块**不在 runner 进程 import jax**，
避免候选侧 backend 被 golden 的 CPU 环境限制（设计 §2.1.1）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
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

COND_OP = "jax_lax_cond"
KERNEL_NAME = "jax_lax_cond_kernel"
ARCH = "dav-2201"
HOST_LIB = f"lib{COND_OP}_host.so"
CANDIDATE_EXIT_OK = 0
CANDIDATE_EXIT_REJECT = 3
GOLDEN_TIMEOUT_S = 180
CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN")
REJECT_CODES = ("INVALID_PREDICATE", "BRANCH_STRUCTURE_MISMATCH",
                "BRANCH_SHAPE_MISMATCH", "BRANCH_DTYPE_MISMATCH")
GRAD_MODES = ("grad_eager", "grad_jit")
JIT_MODES = ("jit", "jit_true", "jit_false")

EXEC_REQUIRED_KEYS = (
    "op", "kernel", "arch", "mode", "variant", "pred_kind", "selected", "path",
    "exec_mode", "abi_calls", "backend", "device_id", "trace_true", "trace_false",
    "exec_true", "exec_false", "runtime_token", "compile_count", "plan_id", "plan_cache",
)
META_DISPATCH_KEYS = ("path", "exec_mode", "abi_calls", "selected", "branch_id",
                      "compile_count", "plan_id", "plan_cache", "backend", "device_id")
META_EVIDENCE_KEYS = ("trace_hits", "exec_hits", "runtime_token", "evidence_level")
REJECT_RE = re.compile(r"REJECT\s+code=([A-Za-z0-9_]+)")
UPSTREAM_RE = re.compile(
    r"UPSTREAM\s+type=(\S*)\s+sha256=([0-9a-fA-F]*)\s+summary=(.*)")


def is_cond_operator(op: str, suite: dict | None = None) -> bool:
    """显式分派判据：只认 op 名（不用“有没有 mode 字段”这种会漂移的隐式条件）。"""
    if op != COND_OP:
        return False
    return suite is None or suite.get("op") in (None, COND_OP)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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


def _json_default(value):
    """把 numpy 标量/数组转成 JSON 可写类型（证据里的 matched_ratio 等可能是 np.float64）。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _tail(text: str, limit: int = 400) -> str:
    return (text or "").strip()[-limit:]


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def expected_host_so(ws: Path, op: str = COND_OP) -> Path:
    return Path(ws) / "ops" / op / "build" / f"lib{op}_host.so"


def ensure_host_so(ws: Path, op: str = COND_OP,
                   log_dir: Path | None = None) -> tuple[bool, str]:
    """``.so`` preflight（设计 §2.2.3）：缺失时按同一份源码补编一次，仍缺则失败。"""
    so = expected_host_so(ws, op)
    if so.is_file():
        return True, str(so)
    build = Path(ws) / "build.sh"
    print(f"[cond] 缺少 {so} → 按同一份源码补编一次（OPX_OP={op}）", file=sys.stderr)
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
    print(f"[cond] 补编后已产出 {so}", file=sys.stderr)
    return True, str(so)


# ---------------------------------------------------------------------------
# 用例发现（供 runner / pytest / 自检共用）
# ---------------------------------------------------------------------------


def discover(cases_path: Path) -> dict:
    """读一份 ``opx-cases/1`` 文件并给出身份/变体/执行次数；不改源文件。"""
    suite = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    cases = suite.get("cases") or []
    if suite.get("op") != COND_OP:
        raise ValueError(f"suite.op={suite.get('op')!r} != {COND_OP!r}：{cases_path}")
    summary = fixtures.count_summary(cases)
    summary.update(source=str(cases_path),
                   case_ids=[case.get("case_id") for case in cases],
                   sha256=_sha256_file(Path(cases_path)))
    return summary


# ---------------------------------------------------------------------------
# 候选 / golden 调用
# ---------------------------------------------------------------------------


def _candidate_cmd(exe: Path, case_json: Path, inputs: Path, mode: str,
                   variant: str, outdir: Path) -> list[str]:
    cmd = [str(exe), "--case-json", str(case_json), "--inputs", str(inputs),
           "--mode", mode]
    if variant:
        cmd += ["--variant", variant]
    cmd += ["--outdir", str(outdir)]
    return cmd


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


def _parse_reject(text: str) -> dict:
    code_match = REJECT_RE.search(text or "")
    upstream_match = UPSTREAM_RE.search(text or "")
    return {
        "code": code_match.group(1) if code_match else None,
        "upstream_type": upstream_match.group(1) if upstream_match else None,
        "upstream_sha256": upstream_match.group(2) if upstream_match else None,
        "upstream_summary": upstream_match.group(3).strip() if upstream_match else None,
    }


def _exec_once(exe: Path, case_json: Path, inputs: Path, mode: str, variant: str,
               outdir: Path, timeout: int) -> dict:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = _candidate_cmd(exe, case_json, inputs, mode, variant, outdir)
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(outdir.parent))
        rc = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else \
            (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else \
            (exc.stderr or b"").decode("utf-8", "replace")
        timed_out = True
    return {
        "rc": rc, "stdout": stdout, "stderr": stderr, "timed_out": timed_out,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cmd": " ".join(cmd), "outdir": outdir, "exec": _parse_exec(stdout),
        "reject": _parse_reject(stdout + "\n" + stderr),
    }


def _golden_env() -> dict:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    return env


def _run_golden(ws: Path, case_json: Path, inputs: Path, mode: str, variant: str,
                cdir: Path, timeout: int = GOLDEN_TIMEOUT_S) -> dict:
    cli = ws / "ops" / COND_OP / "scripts" / "golden_cli.py"
    npz = cdir / f"golden_{mode}.npz"
    meta_path = cdir / f"golden_{mode}.json"
    cmd = [sys.executable, "-I", str(cli), "--case", str(case_json),
           "--inputs", str(inputs), "--npz", str(npz), "--meta", str(meta_path),
           "--mode", mode]
    if variant:
        cmd += ["--variant", variant]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=_golden_env(), cwd=str(cdir))
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "detail": f"golden 超时（{timeout}s）",
                "cmd": " ".join(cmd)}
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "cmd": " ".join(cmd),
                "detail": f"golden 非 0 退出 rc={proc.returncode}；"
                          f"stderr={_tail(proc.stderr or '')}",
                "stderr": _tail(proc.stderr or "", 800)}
    if not meta_path.is_file():
        return {"ok": False, "rc": proc.returncode,
                "detail": "golden 返回 0 但没有 meta.json", "cmd": " ".join(cmd)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = fixtures.load_inputs_npz(npz) if npz.is_file() else {}
    return {"ok": True, "rc": proc.returncode, "meta": meta, "arrays": arrays,
            "npz": str(npz), "cmd": " ".join(cmd)}


# ---------------------------------------------------------------------------
# 校验：EXEC / meta / dispatch / evidence
# ---------------------------------------------------------------------------


def _expected_family(fixture: dict) -> str:
    return "jax" if (fixture["family"] == "jax" or fixture["base_mode"] in GRAD_MODES) else "numpy"


def reject_meta_problems(meta: dict) -> list[str]:
    """拒绝用例的 CLI ``meta.json`` 契约（与 pytest 接口层断言分开）。

    pytest 只拿到 Python 接口的 ``_last_dispatch``/``CondError``；``meta.status``、
    ``meta.leaves``、``meta.upstream`` 是候选 CLI 的落盘证据，由本函数在门禁引擎里统一
    校验。``test_selfcheck_engine_dispatch_negative`` 对该函数做正/负向自检，避免出现
    “pytest 过了但 meta 契约没查”的缺口。
    """
    problems: list[str] = []
    dispatch = meta.get("dispatch") or {}
    if meta.get("status") != "reject":
        problems.append(f"拒绝时 meta.status 必须是 reject，收到 {meta.get('status')!r}")
    if str(dispatch.get("path")) != "reject":
        problems.append(f"拒绝时 meta.dispatch.path 必须是 reject，"
                        f"收到 {dispatch.get('path')!r}")
    if _as_int(dispatch.get("abi_calls")) != 0:
        problems.append(f"拒绝时 meta.dispatch.abi_calls 必须是 0，"
                        f"收到 {dispatch.get('abi_calls')!r}")
    if meta.get("leaves"):
        problems.append("拒绝时 meta.leaves 必须为空")
    if not meta.get("upstream"):
        problems.append("拒绝时 meta.upstream 不能为空（必须保留上游证据/规格规则说明）")
    return problems


def _exec_problems(exec_fields: dict | None, fixture: dict) -> list[str]:
    if not exec_fields:
        return ["候选没有打印 EXEC 标记行（无法证明执行的是候选路径）"]
    problems: list[str] = []
    missing = [key for key in EXEC_REQUIRED_KEYS if key not in exec_fields]
    if missing:
        return [f"EXEC 缺必需键：{missing}"]
    if exec_fields.get("op") != COND_OP:
        problems.append(f"EXEC op={exec_fields.get('op')!r} != {COND_OP!r}")
    if exec_fields.get("kernel") != KERNEL_NAME:
        problems.append(f"EXEC kernel={exec_fields.get('kernel')!r} != {KERNEL_NAME!r}")
    if exec_fields.get("arch") != ARCH:
        problems.append(f"EXEC arch={exec_fields.get('arch')!r} != {ARCH!r}")
    exec_variant = str(exec_fields.get("variant"))
    allowed_variants = {str(fixture["variant"])} | ({"default"} if not fixture["variant"] else set())
    if exec_variant not in allowed_variants:
        problems.append(f"EXEC variant={exec_variant!r} 不属于 {sorted(allowed_variants)}")
    base_mode = fixture["base_mode"]
    if exec_fields.get("mode") not in (base_mode, fixture["mode"]):
        problems.append(f"EXEC mode={exec_fields.get('mode')!r} 不是 {base_mode!r}/{fixture['mode']!r}")
    if not str(exec_fields.get("pred_kind") or "").strip():
        problems.append("EXEC pred_kind 为空")
    if fixture["case_id"] == "cond_sup_pred_scalar_array" and \
            exec_fields.get("pred_kind") != "scalar_bool_array":
        problems.append(f"EXEC pred_kind={exec_fields.get('pred_kind')!r} "
                        "（S2 冻结为 scalar_bool_array）")
    selected = "true" if fixture["selected"] else "false"
    if str(exec_fields.get("selected")) != selected:
        problems.append(f"EXEC selected={exec_fields.get('selected')!r} != {selected!r}")
    trace_true = _as_int(exec_fields.get("trace_true"))
    trace_false = _as_int(exec_fields.get("trace_false"))
    if trace_true is None or trace_true < 1 or trace_false is None or trace_false < 1:
        problems.append(f"EXEC trace_true/trace_false 必须 >=1，收到 "
                        f"{exec_fields.get('trace_true')!r}/{exec_fields.get('trace_false')!r}")
    runtime_token = _as_int(exec_fields.get("runtime_token"))
    want_token = 1 if fixture["selected"] else 2
    if runtime_token != want_token:
        problems.append(f"EXEC runtime_token={exec_fields.get('runtime_token')!r} != {want_token}")
    if _as_int(exec_fields.get("compile_count")) is None:
        problems.append(f"EXEC compile_count 不是整数：{exec_fields.get('compile_count')!r}")
    if not str(exec_fields.get("plan_cache") or "").strip():
        problems.append("EXEC plan_cache 为空")
    if (fixture["base_mode"] in JIT_MODES and _expected_family(fixture) == "numpy"
            and not str(exec_fields.get("plan_id") or "").strip()):
        problems.append("numpy family jit 模式 EXEC plan_id 为空")

    family = _expected_family(fixture)
    path = str(exec_fields.get("path"))
    if family == "numpy":
        if path != "so":
            problems.append(f"numpy family 的 EXEC path={path!r}（必须是 so，CPU/trace 冒充不算 PASS）")
        if str(exec_fields.get("exec_mode")) != "selected_only":
            problems.append(f"numpy family 的 exec_mode={exec_fields.get('exec_mode')!r} "
                            "（必须是 selected_only）")
        if _as_int(exec_fields.get("abi_calls")) != 1:
            problems.append(f"numpy family 的 abi_calls={exec_fields.get('abi_calls')!r}（必须是 1）")
        if str(exec_fields.get("backend")) != "npu":
            problems.append(f"numpy family 的 backend={exec_fields.get('backend')!r}（必须是 npu）")
        if _as_int(exec_fields.get("device_id")) != 0:
            problems.append(f"numpy family 的 device_id={exec_fields.get('device_id')!r}（必须是 0）")
        exec_true = _as_int(exec_fields.get("exec_true"))
        exec_false = _as_int(exec_fields.get("exec_false"))
        want_true, want_false = (1, 0) if fixture["selected"] else (0, 1)
        if (exec_true, exec_false) != (want_true, want_false):
            problems.append(f"EXEC exec_true/exec_false={exec_true!r}/{exec_false!r} "
                            f"!= {want_true}/{want_false}（运行期只执行选中分支）")
        if fixture["base_mode"] in JIT_MODES:
            if _as_int(exec_fields.get("compile_count")) != 1:
                problems.append(f"jit 的 compile_count={exec_fields.get('compile_count')!r}（必须是 1）")
    else:
        if path != "trace":
            problems.append(f"jax family 的 EXEC path={path!r}（必须是 trace）")
        if _as_int(exec_fields.get("abi_calls")) != 0:
            problems.append(f"jax family 的 abi_calls={exec_fields.get('abi_calls')!r}（必须是 0）")
        backend = str(exec_fields.get("backend"))
        if backend in ("", "None", "cpu"):
            problems.append(f"jax family 的 backend={backend!r}（必须是非 CPU 后端）")
    return problems


def _meta_leaf_map(meta: dict) -> dict[str, dict]:
    out = {}
    for leaf in meta.get("leaves") or []:
        if isinstance(leaf, dict) and "path" in leaf:
            out[str(leaf["path"])] = leaf
    return out


def _validate_so_fingerprint(meta: dict, ctx: "CondRun") -> list[str]:
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
    if _as_int(fingerprint["size_bytes"]) != actual_size:
        problems.append(f"so_fingerprint.size_bytes={fingerprint['size_bytes']!r} != 实际 {actual_size}")
    if str(fingerprint["sha256"]) != actual_sha:
        problems.append(f"so_fingerprint.sha256={str(fingerprint['sha256'])[:16]}… != "
                        f"实际 {actual_sha[:16]}…")
    if not str(fingerprint.get("path", "")).endswith(HOST_LIB):
        problems.append(f"so_fingerprint.path 不是 {HOST_LIB}：{fingerprint.get('path')!r}")
    return problems


def _validate_candidate_meta(meta: dict, *, ctx: "CondRun", fixture: dict,
                             outputs: dict) -> tuple[list[str], dict[str, dict]]:
    problems: list[str] = []
    if str(meta.get("case_id")) != str(fixture["case_id"]):
        problems.append(f"meta.case_id={meta.get('case_id')!r} != {fixture['case_id']!r}")
    meta_variant = str(meta.get("variant", ""))
    allowed_variants = {str(fixture["variant"])} | ({"default"} if not fixture["variant"] else set())
    if meta_variant not in allowed_variants:
        problems.append(f"meta.variant={meta_variant!r} 不属于 {sorted(allowed_variants)}")
    if str(meta.get("mode")) not in (fixture["base_mode"], fixture["mode"]):
        problems.append(f"meta.mode={meta.get('mode')!r} 不是 "
                        f"{fixture['base_mode']!r}/{fixture['mode']!r}")
    if meta.get("status") != "ok":
        problems.append(f"meta.status={meta.get('status')!r}（成功路径必须是 ok）")
    if meta.get("inputs_sha256") != fixture["inputs_sha256"]:
        problems.append(f"meta.inputs_sha256 与本次 inputs.npz 不一致："
                        f"{meta.get('inputs_sha256')} != {fixture['inputs_sha256']}")
    dispatch = meta.get("dispatch")
    if not isinstance(dispatch, dict):
        problems.append("meta.dispatch 缺失")
        dispatch = {}
    missing = [key for key in META_DISPATCH_KEYS if key not in dispatch]
    if missing:
        problems.append(f"meta.dispatch 缺字段：{missing}")
    if _as_int(dispatch.get("abi_calls")) is None:
        problems.append(f"meta.dispatch.abi_calls 不是整数：{dispatch.get('abi_calls')!r}")
    if _as_int(dispatch.get("compile_count")) is None:
        problems.append("meta.dispatch.compile_count 不是整数")
    if not str(dispatch.get("plan_cache") or "").strip():
        problems.append("meta.dispatch.plan_cache 为空")
    if (fixture["base_mode"] in JIT_MODES and _expected_family(fixture) == "numpy"
            and not str(dispatch.get("plan_id") or "").strip()):
        problems.append("numpy family jit 模式 meta.dispatch.plan_id 为空")

    evidence = meta.get("evidence")
    if not isinstance(evidence, dict):
        problems.append("meta.evidence 缺失")
        evidence = {}
    missing = [key for key in META_EVIDENCE_KEYS if key not in evidence]
    if missing:
        problems.append(f"meta.evidence 缺字段：{missing}")
    trace_hits = evidence.get("trace_hits") or {}
    for side in ("true", "false"):
        value = _as_int(trace_hits.get(side)) if isinstance(trace_hits, dict) else None
        if value is None or value < 1:
            problems.append(f"meta.evidence.trace_hits.{side}={value!r}（必须 >=1）")
    exec_hits = evidence.get("exec_hits") or {}
    if _expected_family(fixture) == "numpy":
        want = {"true": 1, "false": 0} if fixture["selected"] else {"true": 0, "false": 1}
        got = {side: _as_int(exec_hits.get(side)) if isinstance(exec_hits, dict) else None
               for side in ("true", "false")}
        if got != want:
            problems.append(f"meta.evidence.exec_hits={got!r} != {want!r}"
                            "（必须由 kernel 写、host 读回）")
    want_token = 1 if fixture["selected"] else 2
    if _as_int(evidence.get("runtime_token")) != want_token:
        problems.append(f"meta.evidence.runtime_token={evidence.get('runtime_token')!r} "
                        f"!= {want_token}")
    level = str(evidence.get("evidence_level") or "")
    if _expected_family(fixture) == "numpy":
        if level != "device_counters":
            problems.append(f"numpy family 的 evidence_level={level!r}（必须是 device_counters）")
    elif level not in ("jaxpr_runtime_token", "jaxpr+runtime_token"):
        problems.append(f"jax family 的 evidence_level={level!r}"
                        "（必须是 jaxpr_runtime_token / jaxpr+runtime_token）")

    expected_keys = set(fixture["output_shapes"])
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
    expected_shapes = fixture["output_shapes"]
    for key in sorted(expected_keys):
        leaf = leaf_map.get(key) or {}
        if leaf.get("dtype") != "float32":
            problems.append(f"{key}: meta.leaves.dtype={leaf.get('dtype')!r} != float32")
        if list(leaf.get("shape") or []) != list(expected_shapes[key]):
            problems.append(f"{key}: meta.leaves.shape={leaf.get('shape')} != {expected_shapes[key]}")
        if not leaf.get("sha256"):
            problems.append(f"{key}: meta.leaves 缺 sha256")
        arr = np.asarray(outputs.get(key))
        if str(arr.dtype) != "float32":
            problems.append(f"{key}: outputs.npz dtype={arr.dtype} != float32")
        if list(arr.shape) != list(expected_shapes[key]):
            problems.append(f"{key}: outputs.npz shape={list(arr.shape)} != {expected_shapes[key]}")
        if _expected_family(fixture) == "jax" and "weak_type" not in leaf:
            problems.append(f"{key}: jax family meta.leaves 缺 weak_type")
    if _expected_family(fixture) == "numpy":
        problems.extend(_validate_so_fingerprint(meta, ctx))
    return problems, leaf_map


def _dispatch_problems(meta: dict, fixture: dict) -> list[str]:
    dispatch = meta.get("dispatch") or {}
    family = _expected_family(fixture)
    problems = []
    if family == "numpy":
        if str(dispatch.get("path")) != "so":
            problems.append(f"meta.dispatch.path={dispatch.get('path')!r}（必须是 so）")
        if str(dispatch.get("exec_mode")) != "selected_only":
            problems.append(f"meta.dispatch.exec_mode={dispatch.get('exec_mode')!r}"
                            "（必须是 selected_only）")
        if _as_int(dispatch.get("abi_calls")) != 1:
            problems.append(f"meta.dispatch.abi_calls={dispatch.get('abi_calls')!r}（必须是 1）")
        if str(dispatch.get("backend")) != "npu":
            problems.append(f"meta.dispatch.backend={dispatch.get('backend')!r}（必须是 npu）")
        if _as_int(dispatch.get("device_id")) != 0:
            problems.append(f"meta.dispatch.device_id={dispatch.get('device_id')!r}（必须是 0）")
    else:
        if str(dispatch.get("path")) != "trace":
            problems.append(f"meta.dispatch.path={dispatch.get('path')!r}（必须是 trace）")
        if _as_int(dispatch.get("abi_calls")) != 0:
            problems.append(f"meta.dispatch.abi_calls={dispatch.get('abi_calls')!r}（必须是 0）")
        if str(dispatch.get("backend")) in ("", "None", "cpu"):
            problems.append(f"meta.dispatch.backend={dispatch.get('backend')!r}（必须非 CPU）")
    selected = "true" if fixture["selected"] else "false"
    if str(dispatch.get("selected")) != selected:
        problems.append(f"meta.dispatch.selected={dispatch.get('selected')!r} != {selected!r}")
    branch_id = _as_int(dispatch.get("branch_id"))
    want_id = fixture["true_id"] if fixture["selected"] else fixture["false_id"]
    if branch_id not in (want_id, 1 if fixture["selected"] else 2):
        problems.append(f"meta.dispatch.branch_id={dispatch.get('branch_id')!r} "
                        f"不是 {want_id}（或 1/2 语义）")
    return problems


# ---------------------------------------------------------------------------
# 逐叶判定
# ---------------------------------------------------------------------------


def _judge_leaves(outputs: dict, golden: dict, golden_meta: dict, fixture: dict,
                  criteria) -> tuple[list[dict], list[str]]:
    leaves: list[dict] = []
    problems: list[str] = []
    expected_keys = list(fixture["output_shapes"])
    golden_leaves = _meta_leaf_map(golden_meta)
    family = _expected_family(fixture)
    for key in expected_keys:
        got = np.asarray(outputs.get(key))
        want = np.asarray(golden.get(key))
        info = fixtures.leaf_bytes(list(want.shape), str(want.dtype))
        leaf = {"path": key, "dtype": str(want.dtype), "shape": list(want.shape),
                **info, "weak_type": None, "is_pass": False}
        if got.shape != want.shape:
            problems.append(f"{key}: shape {got.shape} != {want.shape}")
            leaf["failure_reasons"] = ["shape 不一致"]
            leaves.append(leaf)
            continue
        if got.dtype != want.dtype:
            problems.append(f"{key}: dtype {got.dtype} != {want.dtype}")
            leaf["failure_reasons"] = ["dtype 不一致"]
            leaves.append(leaf)
            continue
        verdict = criteria.check(np.ascontiguousarray(got, dtype=np.float32),
                                 np.ascontiguousarray(want, dtype=np.float32))
        leaf.update({
            "is_pass": bool(verdict["is_pass"]),
            "matched_ratio": verdict["matched_ratio"],
            "max_abs_error": verdict["max_abs_error"],
            "nonfinite": verdict["nonfinite"],
            "failure_reasons": verdict.get("failure_reasons") or [],
        })
        if not verdict["is_pass"]:
            problems.append(f"{key}: " + "；".join(verdict.get("failure_reasons") or ["数值未通过"]))
        if family == "jax":
            cand_weak = None
            golden_leaf = golden_leaves.get(key) or {}
            leaf["golden_weak_type"] = golden_leaf.get("weak_type")
            leaf["weak_type"] = None  # 由调用方用候选 meta 填入
        leaves.append(leaf)
    return leaves, problems


def _apply_weak_types(leaves: list[dict], meta: dict) -> list[str]:
    problems: list[str] = []
    leaf_map = _meta_leaf_map(meta)
    for leaf in leaves:
        key = leaf["path"]
        candidate = leaf_map.get(key) or {}
        if "weak_type" not in candidate:
            problems.append(f"{key}: 候选 meta.leaves 缺 weak_type")
            continue
        cand = bool(candidate["weak_type"])
        want = leaf.get("golden_weak_type")
        leaf["weak_type"] = cand
        if want is None:
            problems.append(f"{key}: golden 未记录 weak_type，无法精确判定")
        elif cand != bool(want):
            problems.append(f"{key}: weak_type={cand} != golden {bool(want)}")
    return problems


# ---------------------------------------------------------------------------
# 单条用例 / 单 mode
# ---------------------------------------------------------------------------


class CondRun:
    """一次 cond 用例执行的上下文（版本字段、路径、判据）。"""

    def __init__(self, *, ws: Path, op: str, exe: Path, criteria, workdir: Path,
                 evidence_dir: Path, timeout: int):
        self.ws = ws
        self.op = op
        self.exe = exe
        self.criteria = criteria
        self.workdir = workdir
        self.evidence_dir = evidence_dir
        self.timeout = timeout


def _case_dir(ctx: CondRun, case_id: str, variant: str, mode: str) -> Path:
    variant_dir = variant or "default"
    return ctx.workdir / str(case_id) / variant_dir / str(mode)


def _prepare_case(ctx: CondRun, case: dict, variant: str, mode: str) -> tuple[dict, Path, Path]:
    fixture = fixtures.make_fixture(case, variant, mode)
    cdir = _case_dir(ctx, case["case_id"], variant, mode)
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    return fixture, case_json, inputs


def _run_mode(ctx: CondRun, case: dict, variant: str, mode: str) -> dict:
    fixture, case_json, inputs = _prepare_case(ctx, case, variant, mode)
    rec: dict = {
        "case_id": case["case_id"], "variant": variant, "mode": mode,
        "expected": case.get("expected", "match_cpu"),
        "expected_error": case.get("expected_error"),
        "status": "NOT_RUN", "leaves": [], "dispatch": {}, "evidence": {},
    }
    if case.get("expected_error"):
        return _run_reject_mode(ctx, case, fixture, case_json, inputs, rec)

    cdir = case_json.parent
    run = _exec_once(ctx.exe, case_json, inputs, fixture["base_mode"], variant,
                     cdir / "out", ctx.timeout)
    rec["cmd"] = run["cmd"]
    rec["exec_marker"] = (run["exec"] or {}).get("_raw")
    rec["stdout_tail"] = _tail(run["stdout"])
    rec["stderr_tail"] = _tail(run["stderr"])
    rec["duration_ms"] = run["duration_ms"]
    if run["timed_out"]:
        rec.update(status="ERROR", detail=f"候选实现超时（{ctx.timeout}s）")
        return rec
    if run["rc"] != CANDIDATE_EXIT_OK:
        rec.update(status="ERROR",
                   detail=f"候选非 0 退出 rc={run['rc']}；stderr={_tail(run['stderr'], 300)}")
        return rec
    exec_problems = _exec_problems(run["exec"], fixture)
    if exec_problems:
        rec.update(status="INVALID", detail="；".join(exec_problems))
        return rec
    rec["exec"] = {key: (run["exec"] or {}).get(key) for key in EXEC_REQUIRED_KEYS}
    rec["dispatch"] = {
        "path": run["exec"].get("path"),
        "exec_mode": run["exec"].get("exec_mode"),
        "abi_calls": _as_int(run["exec"].get("abi_calls")),
        "selected": run["exec"].get("selected"),
        "branch_id": run["exec"].get("branch_id"),
        "compile_count": _as_int(run["exec"].get("compile_count")),
        "plan_id": run["exec"].get("plan_id"),
        "plan_cache": run["exec"].get("plan_cache"),
        "backend": run["exec"].get("backend"),
        "device_id": _as_int(run["exec"].get("device_id")),
    }

    meta_path = cdir / "out" / "meta.json"
    if not meta_path.is_file():
        rec.update(status="INVALID", detail=f"候选没有写 meta.json（{meta_path}）")
        return rec
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="INVALID", detail=f"meta.json 读不了：{type(exc).__name__}: {exc}")
        return rec
    rec["meta_status"] = meta.get("status")

    outputs_path = cdir / "out" / "outputs.npz"
    if not outputs_path.is_file():
        rec.update(status="ERROR", detail="候选返回 0 但没有写 outputs.npz（静默失败）")
        return rec
    outputs = fixtures.load_inputs_npz(outputs_path)

    meta_problems, _leaf_map = _validate_candidate_meta(
        meta, ctx=ctx, fixture=fixture, outputs=outputs)
    meta_problems.extend(_dispatch_problems(meta, fixture))
    if meta_problems:
        rec.update(status="INVALID", detail="；".join(meta_problems))
        rec["leaves"] = [
            {"path": key, "dtype": str(np.asarray(value).dtype),
             "shape": list(np.asarray(value).shape)}
            for key, value in sorted(outputs.items())
        ]
        return rec
    rec["dispatch"] = dict(meta.get("dispatch") or {})
    rec["dispatch"]["abi_calls"] = _as_int(rec["dispatch"].get("abi_calls"))
    rec["dispatch"]["device_id"] = _as_int(rec["dispatch"].get("device_id"))
    rec["dispatch"]["compile_count"] = _as_int(rec["dispatch"].get("compile_count"))
    rec["evidence"] = dict(meta.get("evidence") or {})
    rec["so_fingerprint"] = meta.get("so_fingerprint")

    golden = _run_golden(ctx.ws, case_json, inputs, fixture["base_mode"], variant, cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"),
                     "cmd": golden.get("cmd")}
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

    leaves, judge_problems = _judge_leaves(outputs, golden["arrays"], golden_meta,
                                           fixture, ctx.criteria)
    if _expected_family(fixture) == "jax":
        judge_problems.extend(_apply_weak_types(leaves, meta))
    rec["leaves"] = leaves
    rec["matched_ratio"] = min((leaf.get("matched_ratio", 1.0) or 0.0)
                               for leaf in leaves) if leaves else 0.0
    rec["max_abs_err"] = max((leaf.get("max_abs_error") or 0.0) for leaf in leaves) \
        if leaves else None
    rec["nonfinite"] = {
        "count": sum(int((leaf.get("nonfinite") or {}).get("count", 0)) for leaf in leaves),
        "match": all(bool((leaf.get("nonfinite") or {}).get("match", True)) for leaf in leaves),
    }
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
    return rec


def _run_reject_mode(ctx: CondRun, case: dict, fixture: dict, case_json: Path,
                     inputs: Path, rec: dict) -> dict:
    cdir = case_json.parent
    expected_code = str(case.get("expected_error"))
    if expected_code not in REJECT_CODES:
        rec.update(status="ERROR", detail=f"未知 expected_error：{expected_code!r}")
        return rec
    base_mode = fixtures.base_mode_of(rec["mode"])
    run = _exec_once(ctx.exe, case_json, inputs, base_mode, rec["variant"],
                     cdir / "out", ctx.timeout)
    rec["cmd"] = run["cmd"]
    rec["exec_marker"] = (run["exec"] or {}).get("_raw")
    rec["stdout_tail"] = _tail(run["stdout"])
    rec["stderr_tail"] = _tail(run["stderr"])
    rec["duration_ms"] = run["duration_ms"]
    rec["dispatch"] = {
        "path": (run["exec"] or {}).get("path"),
        "exec_mode": (run["exec"] or {}).get("exec_mode"),
        "abi_calls": _as_int((run["exec"] or {}).get("abi_calls")),
        "selected": (run["exec"] or {}).get("selected"),
        "backend": (run["exec"] or {}).get("backend"),
    }
    reject = run["reject"]

    golden = _run_golden(ctx.ws, case_json, inputs, "reject", rec["variant"], cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"),
                     "cmd": golden.get("cmd")}
    if not golden["ok"]:
        reason = "CPU 未确认该输入应被拒绝（既无 CPU 异常，也不属于已确认规格规则拒绝）" \
            if golden.get("rc") == 5 else "golden 执行失败"
        rec.update(status="ERROR",
                   detail=f"{reason}：{golden.get('detail')}")
        rec["golden"]["stderr"] = golden.get("stderr")
        return rec
    golden_meta = golden.get("meta") or {}
    reject_evidence = golden_meta.get("reject_evidence") or {}
    cpu_rejects = bool(reject_evidence.get("cpu_rejects"))
    spec_rule_rejects = bool(reject_evidence.get("spec_rule_rejects"))
    int_spec_rule = (str(fixture.get("pred_kind", "")).startswith("int32")
                     and expected_code == "INVALID_PREDICATE")
    rec["golden"].update({
        "backend": golden_meta.get("backend"),
        "jax": golden_meta.get("jax"),
        "jaxlib": golden_meta.get("jaxlib"),
        "reject_code": golden_meta.get("reject_code"),
        "upstream": golden_meta.get("upstream"),
        "reject_evidence": reject_evidence,
        "cpu_rejects": cpu_rejects,
        "spec_rule_rejects": spec_rule_rejects,
        "reject_basis": reject_evidence.get("reject_basis"),
    })
    rec["reject_evidence"] = reject_evidence
    cpu_upstream = golden_meta.get("upstream") or {}
    if not cpu_rejects and not (spec_rule_rejects and int_spec_rule):
        rec.update(
            status="ERROR",
            detail=("CPU 未抛错，且该变体不是已确认的规格规则拒绝（只有 c13 的 "
                    "int32(0/1) 允许 reject_basis=spec_rule；不修改期望、不伪造上游异常）"))
        return rec

    if run["timed_out"]:
        rec.update(status="ERROR", detail=f"拒绝用例候选超时（{ctx.timeout}s）")
        return rec
    if run["rc"] == CANDIDATE_EXIT_OK:
        rec.update(status="FAIL", detail=f"该拒绝却成功返回（rc=0）；期望 {expected_code}")
        return rec
    if run["rc"] != CANDIDATE_EXIT_REJECT:
        rec.update(status="ERROR",
                   detail=f"拒绝了但不是约定退出码：rc={run['rc']}（期望 {CANDIDATE_EXIT_REJECT}）")
        return rec
    if reject.get("code") != expected_code:
        rec.update(status="FAIL",
                   detail=f"拒绝码 {reject.get('code')!r} != {expected_code!r}")
        return rec
    exec_fields = run["exec"] or {}
    if str(exec_fields.get("path")) != "reject" or \
            str(exec_fields.get("exec_mode")) != "reject":
        rec.update(status="FAIL",
                   detail=f"EXEC 拒绝标记不对：path={exec_fields.get('path')!r} "
                          f"exec_mode={exec_fields.get('exec_mode')!r}")
        return rec
    if _as_int(exec_fields.get("abi_calls")) != 0:
        rec.update(status="FAIL",
                   detail=f"拒绝路径 abi_calls={exec_fields.get('abi_calls')!r}（期望 0）")
        return rec
    if (cdir / "out" / "outputs.npz").is_file():
        rec.update(status="FAIL", detail="拒绝用例不得写 outputs.npz")
        return rec

    meta_path = cdir / "out" / "meta.json"
    if not meta_path.is_file():
        rec.update(status="INVALID", detail="拒绝用例没有写 meta.json")
        return rec
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="INVALID", detail=f"meta.json 读不了：{type(exc).__name__}: {exc}")
        return rec
    meta_problems = reject_meta_problems(meta)
    meta_dispatch = meta.get("dispatch") or {}
    rec["meta"] = {"status": meta.get("status"), "dispatch": meta_dispatch,
                   "leaves": meta.get("leaves"), "upstream": meta.get("upstream")}
    if meta_problems:
        rec.update(status="INVALID", detail="；".join(meta_problems))
        return rec

    upstream = meta.get("upstream") or {}
    cand_type = upstream.get("type") or upstream.get("upstream_exception_type")
    cand_msg = upstream.get("message") or upstream.get("upstream_message") or ""
    cand_sha = upstream.get("sha256") or upstream.get("upstream_message_sha256") or ""
    cpu_type = cpu_upstream.get("type")
    cpu_sha = cpu_upstream.get("sha256")
    if cpu_rejects:
        if str(cand_type) != str(cpu_type):
            rec.update(status="FAIL",
                       detail=f"upstream_exception_type={cand_type!r} != CPU {cpu_type!r}")
            return rec
    else:
        # 规格规则拒绝：CPU 没有异常，候选必须以 type=None 明确“无可保留的上游异常”，
        # 同时给出 spec_rule 与 cpu_probe，证明拒绝依据不是伪造的 CPU 异常。
        if not int_spec_rule:
            rec.update(status="FAIL",
                       detail="CPU 未抛错且变体不是 c13 int32(0/1) 的规格规则拒绝")
            return rec
        if cand_type not in (None, "", "None", "none", "null"):
            rec.update(status="FAIL",
                       detail=f"规格规则拒绝的 upstream.type 必须是 None，收到 {cand_type!r}")
            return rec
        if str(upstream.get("spec_rule") or "") != "scalar_bool_pred":
            rec.update(status="FAIL",
                       detail="规格规则拒绝缺少 upstream.spec_rule=scalar_bool_pred")
            return rec
        probe = upstream.get("cpu_probe")
        if not isinstance(probe, dict):
            rec.update(status="FAIL",
                       detail="规格规则拒绝缺少 upstream.cpu_probe（CPU 原始行为证据）")
            return rec
        if probe.get("rejects") is not False:
            rec.update(status="FAIL",
                       detail=f"规格规则拒绝的 cpu_probe.rejects 必须是 false，"
                              f"收到 {probe.get('rejects')!r}")
            return rec
        if probe.get("type") is not None:
            rec.update(status="FAIL",
                       detail=f"规格规则拒绝的 cpu_probe.type 必须是 null，"
                              f"收到 {probe.get('type')!r}")
            return rec
    if not str(cand_msg).strip():
        rec.update(status="FAIL", detail="upstream_message 为空（必须保留上游消息/规格规则说明）")
        return rec
    if str(cand_sha) != _sha256_text(str(cand_msg)):
        rec.update(status="FAIL",
                   detail="upstream_message_sha256 不可复算（必须等于 sha256(message)）")
        return rec
    if not re.fullmatch(r"[0-9a-f]{64}", str(cand_sha)):
        rec.update(status="FAIL", detail="upstream_message_sha256 不是 64 位小写十六进制")
        return rec
    if reject.get("upstream_type") in (None, ""):
        rec.update(status="FAIL",
                   detail="拒绝路径缺少 UPSTREAM type=<类型> 标记（设计 §1.3.3 必需）")
        return rec
    if cpu_rejects:
        if str(reject.get("upstream_type")) != str(cand_type):
            rec.update(status="FAIL",
                       detail=f"UPSTREAM type={reject.get('upstream_type')!r} 与 meta.upstream "
                              f"type={cand_type!r} 不一致")
            return rec
    else:
        if str(reject.get("upstream_type")).strip().lower() not in ("none", "null"):
            rec.update(status="FAIL",
                       detail=f"规格规则拒绝的 UPSTREAM type 必须是 None，"
                              f"收到 {reject.get('upstream_type')!r}")
            return rec
    if reject.get("upstream_sha256") and reject["upstream_sha256"] != str(cand_sha):
        rec.update(status="FAIL",
                   detail=f"UPSTREAM sha256={reject['upstream_sha256'][:16]}… 与 "
                          f"meta.upstream sha256={str(cand_sha)[:16]}… 不一致")
        return rec
    rec["upstream"] = {"type": cand_type, "message": cand_msg, "sha256": cand_sha,
                       "summary": (upstream.get("summary") or
                                   str(cand_msg).strip().splitlines()[0][:200]),
                       "cpu_type": cpu_type, "cpu_sha256": cpu_sha,
                       "cpu_rejects": cpu_rejects,
                       "spec_rule_rejects": spec_rule_rejects,
                       "spec_rule": upstream.get("spec_rule"),
                       "cpu_probe": upstream.get("cpu_probe"),
                       "keyword_hit": (upstream.get("keyword_hit") or
                                       reject_evidence.get("upstream_keyword_hit"))}
    if cpu_rejects:
        detail = (f"按预期拒绝：rc=3 {expected_code}；abi_calls=0；上游 "
                  f"{cand_type}（sha256 可复算）")
    else:
        detail = (f"按已确认规格规则拒绝：rc=3 {expected_code}；abi_calls=0；"
                  f"CPU 0.8.1 无异常（cpu_rejects=false），拒绝依据 scalar_bool_pred")
    rec.update(status="PASS", detail=detail)
    return rec


def _run_case(ctx: CondRun, case: dict, modes_override: list[str] | None = None) -> dict:
    try:
        return _run_case_impl(ctx, case, modes_override=modes_override)
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        rec = {
            "case_id": case.get("case_id"), "group": case.get("group"),
            "required": bool(case.get("required", True)),
            "expected": case.get("expected", "match_cpu"),
            "expected_error": case.get("expected_error"),
            "note": case.get("note"), "status": "ERROR",
            "detail": f"执行器异常：{type(exc).__name__}: {exc}",
            "variants": [], "modes": [], "dispatch": {},
        }
        try:
            ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
            _dump_json(ctx.evidence_dir / str(rec["case_id"]) / "meta.json", rec)
        except Exception:  # noqa: BLE001
            pass
        return rec


def _run_case_impl(ctx: CondRun, case: dict, modes_override: list[str] | None = None) -> dict:
    case_id = str(case["case_id"])
    variants = fixtures.variants_for(case)
    default_modes = fixtures.modes_for(case)
    mode_records: list[dict] = []
    for variant in variants:
        modes = list(modes_override) if modes_override is not None else default_modes
        for mode in modes:
            rec = _run_mode(ctx, case, variant, mode)
            mode_records.append(rec)
            _snapshot_outputs(ctx, case_id, variant, mode,
                              _case_dir(ctx, case_id, variant, mode) / "out")

    statuses = [item["status"] for item in mode_records]
    overall, detail = "NOT_RUN", ""
    if mode_records:
        if all(status == "PASS" for status in statuses):
            overall, detail = "PASS", "全部 variant/mode 通过"
        else:
            for label in ("ERROR", "INVALID", "FAIL", "NOT_RUN"):
                if label in statuses:
                    overall = label
                    break
            detail = "；".join(
                f"{item['variant'] or 'default'}/{item['mode']}:{item['status']}:"
                f"{str(item.get('detail', ''))[:160]}"
                for item in mode_records if item["status"] != "PASS")
    else:
        detail = "用例没有 mode"

    consistency = _consistency_check(ctx, case, variants, mode_records)
    if consistency and overall == "PASS":
        overall, detail = "FAIL", "；".join(consistency)

    rec = {
        "case_id": case_id,
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "expected_error": case.get("expected_error"),
        "note": case.get("note"),
        "status": overall,
        "detail": detail,
        "variants": variants,
        "modes": mode_records,
        "dispatch": mode_records[0]["dispatch"] if mode_records else {},
        "matched_ratio": min((item.get("matched_ratio", 1.0) or 0.0)
                             for item in mode_records if item.get("status") == "PASS")
        if overall == "PASS" else None,
        "max_abs_err": max((item.get("max_abs_err") or 0.0) for item in mode_records) \
            if mode_records else None,
    }
    _dump_case_evidence(ctx, rec, case)
    return rec


def _consistency_check(ctx: CondRun, case: dict, variants: list[str],
                       mode_records: list[dict]) -> list[str]:
    problems: list[str] = []
    by_key = {(item["variant"], item["mode"]): item for item in mode_records}
    pairs = (("eager", "jit"), ("eager_true", "jit_true"),
             ("eager_false", "jit_false"))
    for variant in variants:
        for eager_mode, jit_mode in pairs:
            eager = by_key.get((variant, eager_mode))
            jit = by_key.get((variant, jit_mode))
            if not eager or not jit:
                continue
            if eager["status"] != "PASS" or jit["status"] != "PASS":
                continue
            eager_arrays = _snapshot_arrays(ctx, case["case_id"], variant, eager_mode)
            jit_arrays = _snapshot_arrays(ctx, case["case_id"], variant, jit_mode)
            if eager_arrays is None or jit_arrays is None:
                continue
            if set(eager_arrays) != set(jit_arrays):
                problems.append(f"{variant or 'default'}: eager/jit 输出叶子集合不同")
                continue
            for key in eager_arrays:
                a = np.asarray(eager_arrays[key], dtype=np.float32)
                b = np.asarray(jit_arrays[key], dtype=np.float32)
                if a.shape != b.shape or a.dtype != b.dtype:
                    problems.append(f"{variant or 'default'}/{key}: eager/jit shape/dtype 不同")
                    continue
                verdict = ctx.criteria.check(np.ascontiguousarray(a), np.ascontiguousarray(b))
                if not verdict["is_pass"]:
                    problems.append(f"{variant or 'default'}/{key}: eager 与 jit 输出不一致")
        # 动态谓词两种取值必须共享同一 plan（pred 是运行期参数，不进入 plan key）。
        jit_true = by_key.get((variant, "jit_true"))
        jit_false = by_key.get((variant, "jit_false"))
        if jit_true and jit_false and jit_true["status"] == "PASS" and \
                jit_false["status"] == "PASS":
            id_true = (jit_true.get("dispatch") or {}).get("plan_id")
            id_false = (jit_false.get("dispatch") or {}).get("plan_id")
            if not id_true or not id_false:
                problems.append(f"{variant or 'default'}: jit_true/jit_false 缺 plan_id")
            elif str(id_true) != str(id_false):
                problems.append(f"{variant or 'default'}: jit_true plan_id={id_true!r} != "
                                f"jit_false plan_id={id_false!r}（pred 被编进了 plan）")
    return problems


def _snapshot_dir(ctx: CondRun, case_id: str, variant: str, mode: str) -> Path:
    return ctx.workdir / str(case_id) / (variant or "default") / "snapshots" / str(mode)


def _snapshot_arrays(ctx: CondRun, case_id: str, variant: str, mode: str):
    path = _snapshot_dir(ctx, case_id, variant, mode) / "outputs.npz"
    if not path.is_file():
        return None
    return fixtures.load_inputs_npz(path)


def _snapshot_outputs(ctx: CondRun, case_id: str, variant: str, mode: str,
                      outdir: Path) -> None:
    dst = _snapshot_dir(ctx, case_id, variant, mode)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("outputs.npz", "meta.json"):
        src = outdir / name
        if src.is_file():
            shutil.copy2(src, dst / name)


def _dump_case_evidence(ctx: CondRun, rec: dict, case: dict) -> None:
    dst = ctx.evidence_dir / str(rec["case_id"])
    dst.mkdir(parents=True, exist_ok=True)
    _dump_json(dst / "meta.json", rec)
    for item in rec.get("modes") or []:
        if item.get("status") != "PASS":
            cdir = _case_dir(ctx, rec["case_id"], item["variant"], item["mode"])
            for name in ("case.json", "inputs.npz", "stdout.txt", "stderr.txt"):
                src = cdir / name
                if src.is_file():
                    shutil.copy2(src, dst / f"{item['variant'] or 'default'}-{item['mode']}-{name}")
            if item.get("stdout_tail"):
                (dst / f"{item['variant'] or 'default'}-{item['mode']}-stdout_tail.txt") \
                    .write_text(item["stdout_tail"], encoding="utf-8")
            if item.get("stderr_tail"):
                (dst / f"{item['variant'] or 'default'}-{item['mode']}-stderr_tail.txt") \
                    .write_text(item["stderr_tail"], encoding="utf-8")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _summary(results: list[dict]) -> dict:
    summary = {status: sum(1 for item in results if item["status"] == status)
               for status in CASE_STATUSES}
    summary["TOTAL"] = len(results)
    summary["mode_executions"] = sum(len(item.get("modes") or []) for item in results)
    summary["variant_executions"] = len({
        (item["case_id"], mode.get("variant")) for item in results
        for mode in (item.get("modes") or [])})
    return summary


def _render_md(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"# 正确性报告 · {report['op']}（cond 分支）",
        "",
        f"- 用例集：`{report['case_source']}`（sha256 `{report['case_source_sha256'][:16]}…`）",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 规格版本：v{report['spec_version_id']} ｜ "
        f"判据档：`{report['criteria_ref']}`",
        f"- 代码快照：`{report['code_snapshot_sha256'][:16]}…` ｜ "
        f"候选 sha256：`{report['candidate']['sha256'][:16]}…`",
        f"- 汇总：{summary}",
        "",
        "| 用例 | 组 | 必需 | variant | mode | 状态 | selected | path | trace(t/f) | exec(t/f) | token | 说明 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["cases"]:
        for mode in item.get("modes") or []:
            dispatch = mode.get("dispatch") or {}
            evidence = mode.get("evidence") or {}
            trace = evidence.get("trace_hits") or {}
            exec_hits = evidence.get("exec_hits") or {}
            lines.append(
                f"| {item['case_id']} | {item.get('group', '')} | "
                f"{'是' if item.get('required', True) else '否'} | "
                f"{mode.get('variant') or '—'} | {mode.get('mode')} | **{mode['status']}** | "
                f"{dispatch.get('selected', '—')} | {dispatch.get('path', '—')} | "
                f"{trace.get('true', '—')}/{trace.get('false', '—')} | "
                f"{exec_hits.get('true', '—')}/{exec_hits.get('false', '—')} | "
                f"{evidence.get('runtime_token', '—')} | "
                f"{(mode.get('detail') or '').replace('|', '/')[:120]} |")
    lines += [
        "",
        "> 判定口径：`criteria/check.py`（ops-precision-standard@1）逐叶子；结构/shape/dtype "
        "精确；jax family 的 weak_type 逐叶精确；非有限位逐元素精确。",
        "> 候选必须在非 CPU 后端执行；golden 一律独立 CPU 子进程；REFUSED 四码必须带可复算的上游证据。",
        "> 本文件是给人看的伴生品；机判只认同目录的 `correctness.json`。",
    ]
    return "\n".join(lines) + "\n"


def _finish_report(ctx: CondRun, *, op: str, cases_path: Path, out: Path, md: Path,
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
    return 0 if not bad else 1


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
    if not is_cond_operator(op, suite):
        print(f"[run_cases] cond 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}",
              file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = CondRun(ws=ws, op=op, exe=exe, criteria=criteria,
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
    """smoke 选例（设计 §2.2.3 冻结）：固定 ``cond_true_eager``；缺失/不匹配则失败。"""
    for case in cases:
        if case.get("case_id") == "cond_true_eager":
            if not bool(case.get("required", True)):
                raise SystemExit("cond_true_eager 不是 required，smoke 选例规则失效")
            if case.get("expected") != "match_cpu":
                raise SystemExit("cond_true_eager 的 expected 不是 match_cpu，smoke 选例规则失效")
            if "eager" not in (case.get("mode") or []):
                raise SystemExit("cond_true_eager 没有 eager mode，smoke 选例规则失效")
            return case
    raise SystemExit(
        "cases 里没有 cond_true_eager（smoke 固定选例缺失；不得退化为拒绝/证据用例）")


def run_smoke(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if not cases_path.is_file():
        print(f"[smoke] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_cond_operator(op, suite):
        print(f"[smoke] cond 分支拒绝执行：suite.op={suite.get('op')!r} != {op!r}",
              file=sys.stderr)
        return 2
    ok, detail = ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[smoke] MISSING_SO {detail}（不在缺共享库时降级为通过）", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = CondRun(ws=ws, op=op, exe=exe, criteria=criteria,
                  workdir=workdir, evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    case = pick_smoke_case(suite.get("cases") or [])
    print(f"[smoke] 最小用例：{case['case_id']} mode=eager（cond 分支）")
    rec = _run_case(ctx, case, modes_override=["eager"])
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
        f"# 冒烟结果：{op}（cond 分支）",
        "",
        f"- 用例：`{rec.get('case_id')}`（mode=eager）",
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
