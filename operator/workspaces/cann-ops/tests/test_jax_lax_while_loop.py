# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop · pytest 交付件（真机；候选 = Python 接口，参考 = 同版本 JAX CPU 子进程）。

结构（设计 §2.2 / §5 / §7.2）：

* 冻结 ``cases/jax_lax_while_loop/cases.json``（16 条 / 51 次）
  + 基准复用 ``benchmark.json``（2 条 / 2 次，仅判正确性）
  + 补充 ``tests/jax_lax_while_loop_supplement_cases.json``（3 条 / 8 次）
  = **21 条黑盒身份 / 61 个 ``(case_id, variant, mode)`` 参数化项**；
  + 白盒 ``tests/jax_lax_while_loop_whitebox_cases.json``（节点 2 创建，27 条 / 59 次，
    同一 JSON→expander→pytest 入口；场景函数在 ``tests/while_whitebox_scenarios.py``）
  = 21+27 条身份 / 61+59 个参数化项（黑盒/白盒计数单列）。

候选：``ops/jax_lax_while_loop/python/jax_lax_while_loop.py`` 的
``while_loop(cond_fun, body_fun, init_carry, *, safety_limit=None, tangent=None)`` 与
``while_loop_compiled(cond_fun, body_fun, init_carry, *, mode="primal", tangent=None,
safety_limit=None, jit=True)``；两条入口都经宿主 C ABI（``libjax_lax_while_loop_host.so``）
在设备上执行，并把本次调用的证据写入模块级 ``last_dispatch``（path/route/abi_calls/
iterations/cond_hits/body_hits/trace_hits/exec_hits/runtime_token/counter_source/
plan_id/compile_count/plan_cache/backend/platforms/device_id/leaves/tree）。

          ``leaves`` 每项为 ``{path,dtype,shape,weak_type,sha256}``，``tree`` 为 flatten 前的
          容器指纹（``fixtures.tree_signature``；JVP 顶层为 ``(primal, tangent)`` 二元组）。

参考：``golden_cli.py`` 独立 CPU 子进程（``JAX_PLATFORMS=cpu``）；pytest 主进程只调度，
**不 import jax**。每个 ``(case_id, variant, mode)`` 的候选执行在独立 fresh worker
（``python -I <本文件> --worker <payload>``）中完成，避免 AFIR 进程内编译缓存跨 case 复用。

判定（设计 §2.3/§2.4）：逐叶先设备守卫（worker 真实 ``backend!=cpu``、devices 含 npu、
jax/jaxlib=0.8.1、``path=so``、platforms 含 npu、原始整数 ``abi_calls``、
``counter_source=kernel``、合法 route、容器 device 0、runtime_token 非空，并叠加独立
后端/设备节点探测与 run 身份 image #145/host 5）；再比较树容器类型（tuple≠list）与
dtype/shape 精确；jax family 的 ``weak_type`` 与 golden 逐叶精确；浮点走
``criteria/check.py``；float0 要求 nbytes=0；非有限位逐元素精确；拒绝按五码 + 上游类型/摘要，
c16 与数值用例同样要求完整设备证据。本任务没有合法 ``xfail``/``unsupported``：任何非 PASS
都 fail，不计入通过数。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
OP = "jax_lax_while_loop"
for extra in (WS, WS / "ops" / OP / "python", WS / "ops" / OP / "scripts", WS / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# numpy 是交付件运行必需依赖；缺失时让收集阶段显式失败，不能 importorskip 成“通过”。
import numpy as np  # noqa: E402

import fixtures  # noqa: E402
import while_engine as engine  # noqa: E402
import while_whitebox_scenarios as wb  # noqa: E402

CANDIDATE_PATH = WS / "ops" / OP / "python" / f"{OP}.py"
GOLDEN_DIR = WS / "ops" / OP / "scripts"
GOLDEN_CLI = GOLDEN_DIR / "golden_cli.py"
FROZEN_CASES_PATH = WS / "cases" / OP / "cases.json"
BENCH_PATH = WS / "cases" / OP / "benchmark.json"
SUPPLEMENT_PATH = WS / "tests" / f"{OP}_supplement_cases.json"
WHITEBOX_PATH = WS / "tests" / f"{OP}_whitebox_cases.json"

EXPECTED_GATE_CASES = 16
EXPECTED_GATE_MODES = 51
EXPECTED_BENCH_CASES = 2
EXPECTED_BENCH_MODES = 2
EXPECTED_SUPPLEMENT_CASES = 3
EXPECTED_SUPPLEMENT_MODES = 8
EXPECTED_BLACKBOX_CASES = 21
EXPECTED_BLACKBOX_MODES = 61
#: 节点 2 白盒用例：27 条身份 / 59 个 (case_id, variant, mode) 参数化项。
EXPECTED_WHITEBOX_CASES = 27
EXPECTED_WHITEBOX_MODES = 59
EXPECTED_SELFCHECKS = 18

CASES_SHA256 = "89e5376f7d31f34e6594e6284fc572b93b67d14b5b0b604996f0557bcd0b0f81"
BENCH_SHA256 = "e645bd81b9d614ba90917ed511ad893673e4603738613be0518691a5dd0805ca"
OUT_DIR = Path(os.environ.get("OPX_OUT") or (WS / "tests" / "_while_out"))
RESULT_JSON = OUT_DIR / "while_loop_correctness.json"
RESULT_MD = OUT_DIR / "while_loop_correctness.md"
WORK_DIR = Path(os.environ.get("OPX_WHILE_WORKDIR") or tempfile.gettempdir()) / "opx-while-pytest"

_RESULTS: list[dict] = []
_SELFCHECK_RECORDS: list[dict] = []


# ---------------------------------------------------------------------------
# 用例装载
# ---------------------------------------------------------------------------


def _load_whitebox(path: Path | None = None) -> dict:
    """节点 3 白盒用例装载：只读 tests/ 下的 JSON，不参与 cases/suite_ref。

    * 文件缺失是正常状态（返回空表），便于在实现前先跑黑盒；
    * 每个 case 打 ``_origin="whitebox"``，``test_case_matrix`` 据它分派到白盒执行器；
    * 结构与 fixture 用例保持同一 ``cases`` 形状（variants/mode/expected_error 语义相同），
      因此 ``fixtures.expand_items``/``count_summary`` 可原样复用。
    """
    path = Path(path or WHITEBOX_PATH)
    if not path.is_file():
        return {"schema": "opx-whitebox-cases/1", "op": OP, "cases": []}
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("op") != OP:
        raise ValueError(f"whitebox.op={suite.get('op')!r} != {OP!r}（{path}）")
    cases = []
    seen = set()
    for raw in suite.get("cases") or []:
        case = dict(raw)
        case_id = str(case.get("case_id"))
        if not case_id or case_id in seen:
            raise ValueError(f"whitebox case_id 缺失或重复：{case_id!r}")
        seen.add(case_id)
        case["_origin"] = "whitebox"
        cases.append(case)
    return {**suite, "cases": cases}


def _load_all() -> dict:
    gate = fixtures.load_gate(FROZEN_CASES_PATH)
    bench = fixtures.load_bench(BENCH_PATH)
    supplement = fixtures.load_supplement(SUPPLEMENT_PATH)
    whitebox = _load_whitebox()
    return {"gate": gate, "bench": bench, "supplement": supplement, "whitebox": whitebox}


def _items_for(suite: dict) -> list[dict]:
    return fixtures.expand_items(suite["cases"])


def _blackbox_items() -> list[dict]:
    data = _load_all()
    items: list[dict] = []
    for key in ("gate", "bench", "supplement"):
        items.extend(_items_for(data[key]))
    return items


def _all_items() -> list[dict]:
    data = _load_all()
    items: list[dict] = []
    for key in ("gate", "bench", "supplement", "whitebox"):
        items.extend(_items_for(data[key]))
    return items


def _item_id(item: dict) -> str:
    return f"{item['case_id']}::{item['variant'] or 'single'}::{item['display_mode']}"


# ---------------------------------------------------------------------------
# worker 侧：候选调用（只在子进程 import jax）
# ---------------------------------------------------------------------------


def _load_candidate(path=None):
    target = Path(path or CANDIDATE_PATH)
    spec = importlib.util.spec_from_file_location("opx_while_candidate", target)
    if not spec or not spec.loader:
        raise RuntimeError(f"候选 Python 接口不存在或不可加载：{target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attach_ids(case: dict, variant: str):
    sys.path.insert(0, str(GOLDEN_DIR))
    import golden  # noqa: PLC0415

    cond, body = golden.cond_body(case, variant)
    cond.__name__ = f"cond_{fixtures.cond_id(case, variant)}"
    body.__name__ = f"body_{fixtures.body_id(case, variant)}"
    cond.opx_cond_id = fixtures.cond_id(case, variant)
    body.opx_body_id = fixtures.body_id(case, variant)
    body.opx_case = case
    body.opx_variant = variant
    body.opx_expected_counts = fixtures.expected_counts(case, variant)
    return cond, body


def _worker_main(payload: dict) -> int:
    """fresh worker：候选 + 设备证据；不跑 golden（golden 由父进程 CPU 子进程负责）。"""
    import jax  # noqa: PLC0415
    import golden  # noqa: PLC0415

    case = payload["case"]
    variant = payload["variant"]
    mode = payload["mode"]
    arrays = fixtures.load_inputs(Path(payload["inputs"]))
    init = golden.build_init(case, variant, arrays)
    cond, body = _attach_ids(case, variant)
    tangent = golden.build_tangent(case, variant, arrays) if mode in golden.JVP_MODES else None
    safety_limit = case.get("safety_iteration_limit")
    module = _load_candidate(payload.get("candidate_path"))
    result: dict = {
        "worker_pid": os.getpid(),
        "worker_backend": jax.default_backend(),
        "worker_devices": sorted({str(device.platform) for device in jax.devices()}),
        "worker_jax": getattr(jax, "__version__", None),
        "worker_jaxlib": getattr(__import__("jaxlib"), "__version__", None),
        "worker_backend_after": None,
        "rc": 0,
        "reject": None,
        "dispatch": {},
        "outputs_npz": None,
        "tree": None,
        "leaves": [],
    }
    try:
        if fixtures.is_reject(case):
            if mode == "eager":
                module.while_loop(cond, body, init, safety_limit=safety_limit)
            else:
                module.while_loop_compiled(cond, body, init, mode="primal",
                                           safety_limit=safety_limit)
            result["rc"] = 0
        elif mode == "eager":
            out = module.while_loop(cond, body, init, safety_limit=safety_limit)
            result["tree"] = fixtures.tree_signature(out)
            result["outputs_npz"] = _save_flat(out, "out", payload["outdir"])
            meta = golden.flatten_meta(out, "out")
            result["leaves"] = [{"path": key, **value} for key, value in meta.items()]
        elif mode == "jit":
            out = module.while_loop_compiled(cond, body, init, mode="primal",
                                             safety_limit=safety_limit)
            result["tree"] = fixtures.tree_signature(out)
            result["outputs_npz"] = _save_flat(out, "out", payload["outdir"])
            meta = golden.flatten_meta(out, "out")
            result["leaves"] = [{"path": key, **value} for key, value in meta.items()]
        else:
            out = module.while_loop_compiled(cond, body, init, mode="jvp", tangent=tangent,
                                             jit=(mode == "jvp_jit"),
                                             safety_limit=safety_limit)
            result["tree"] = fixtures.tree_signature(out)
            primal, tang = out
            flat = golden.flatten(primal, "primal")
            flat.update(golden.flatten(tang, "tangent"))
            npz = Path(payload["outdir"]) / "outputs.npz"
            np.savez(npz, **flat)
            result["outputs_npz"] = str(npz)
            meta = golden.flatten_meta(primal, "primal")
            meta.update(golden.flatten_meta(tang, "tangent"))
            result["leaves"] = [{"path": key, **value} for key, value in meta.items()]
        result["dispatch"] = _dispatch_of(module)
    except BaseException as exc:  # noqa: BLE001 —— 拒绝/异常证据都原样返回父进程
        code = getattr(exc, "code", None)
        if not code:
            text = str(exc)
            code = text.split(":", 1)[0] if ":" in text else type(exc).__name__
        message = getattr(exc, "upstream_message", "") or str(exc)
        upstream_type = getattr(exc, "upstream_exception_type", None)
        result.update({
            "rc": 3 if code in fixtures.REJECT_CODES else 1,
            "reject": {
                "code": code,
                "upstream_type": upstream_type,
                "upstream_message": message,
                "upstream_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest()
                if message else None,
                "upstream_summary": (message or "").replace("\n", " ").strip(),
                "spec_rule": getattr(exc, "spec_rule", None),
            },
            "dispatch": _dispatch_of(module) or getattr(exc, "dispatch", {}) or {},
            "error": f"{type(exc).__name__}: {exc}",
        })
    try:
        # 调用后再次读取：候选/驱动初始化 NPU 插件后 backend 可能从 cpu 切到 npu。
        result["worker_backend_after"] = jax.default_backend()
        result["worker_devices_after"] = sorted({str(device.platform) for device in jax.devices()})
    except Exception as exc:  # noqa: BLE001
        result["worker_backend_after_error"] = f"{type(exc).__name__}: {exc}"
    print("WHILE_WORKER_RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


def _save_flat(value, prefix: str, outdir: str) -> str:
    import golden  # noqa: PLC0415

    flat = golden.flatten(value, prefix)
    npz = Path(outdir) / "outputs.npz"
    np.savez(npz, **flat)
    return str(npz)


def _dispatch_of(module) -> dict:
    dispatch = getattr(module, "last_dispatch", None)
    return dict(dispatch) if isinstance(dispatch, dict) else {}


def _run_worker(payload: dict, timeout: int = 600, attempts: int = 2) -> dict:
    """fresh worker；仅对“NPU 后端初始化失败”做一次可审计重试，不重试候选错误。"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / "worker_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(payload_path)]
    attempt_logs: list[dict] = []
    proc = None
    parsed = None
    for attempt in range(1, max(1, attempts) + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        parsed = None
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("WHILE_WORKER_RESULT "):
                parsed = json.loads(line[len("WHILE_WORKER_RESULT "):])
                break
        text = (proc.stderr or "") + (proc.stdout or "")
        transient = (parsed is None and proc.returncode != 0
                     and ("Unable to initialize backend 'npu'" in text
                          or "no supported devices found for platform npu" in text))
        attempt_logs.append({"attempt": attempt, "rc": proc.returncode,
                             "transient_npu_init": transient,
                             "stderr_tail": engine._tail(proc.stderr, 1200)})
        if parsed is not None or not transient or attempt >= attempts:
            break
        time.sleep(3)
    return {
        "cmd": cmd, "rc": proc.returncode if proc else -1,
        "stdout": proc.stdout if proc else "", "stderr": proc.stderr if proc else "",
        "result": parsed, "attempts": len(attempt_logs),
        "attempt_logs": attempt_logs,
    }


# ---------------------------------------------------------------------------
# 父进程：判据
# ---------------------------------------------------------------------------


def _run_golden(case_json: Path, inputs: Path, mode: str, variant: str, cdir: Path) -> dict:
    return engine._run_golden(WS, case_json, inputs, mode, variant, cdir)


def _prepare_item(item: dict) -> dict:
    case, variant, mode = item["case"], item["variant"], item["mode"]
    cdir = WORK_DIR / f"{item['case_id']}__{variant or 'single'}__{mode}"
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(case, variant, mode, case_json)
    inputs_sha = fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
    return {
        "case": case, "variant": variant, "mode": mode,
        "case_json": case_json, "inputs": inputs, "inputs_sha256": inputs_sha,
        "cdir": cdir,
    }


def _build_reject_run(worker: dict, dispatch: dict, outputs_npz: str | None) -> dict:
    result = worker.get("result") or {}
    return {
        "rc": result.get("rc"),
        "reject": result.get("reject") or {},
        "outputs": outputs_npz,
        "meta": dispatch,
        "exec": {},
        "timed_out": False,
    }


def _worker_view(result: dict) -> dict:
    """把 worker 的**真实**后端证据整理成 engine.device_evidence 的入参。

    候选自报的 ``dispatch.backend`` 只作线索；这里用同一 worker 进程调用前后
    ``jax.default_backend()``/``jax.devices()`` 与 jax/jaxlib 版本做独立断言。
    """
    backend_after = result.get("worker_backend_after")
    devices_after = result.get("worker_devices_after")
    return {
        "backend": backend_after or result.get("worker_backend"),
        "backend_before": result.get("worker_backend"),
        "platforms": devices_after or result.get("worker_devices") or [],
        "jax": result.get("worker_jax"),
        "jaxlib": result.get("worker_jaxlib"),
        "pid": result.get("worker_pid"),
    }


def _run_case_item(item: dict) -> dict:
    case, variant, mode = item["case"], item["variant"], item["mode"]
    rec = {
        "case_id": item["case_id"], "variant": variant, "mode": item["display_mode"],
        "requested_mode": mode, "required": bool(case.get("required", True)),
        "status": "ERROR", "detail": "",
        "bytes": fixtures.bytes_info(case, variant),
        "expected_iterations": fixtures.expected_iterations(case, variant),
        "expected_counts": fixtures.expected_counts(case, variant),
    }
    try:
        prepared = _prepare_item(item)
    except Exception as exc:  # noqa: BLE001
        rec["detail"] = f"输入生成失败：{type(exc).__name__}: {exc}"
        return rec
    rec["inputs_sha256"] = prepared["inputs_sha256"]
    rec["case_json"] = str(prepared["case_json"].relative_to(WS)) \
        if str(prepared["case_json"]).startswith(str(WS)) else str(prepared["case_json"])
    rec["inputs"] = str(prepared["inputs"].relative_to(WS)) \
        if str(prepared["inputs"]).startswith(str(WS)) else str(prepared["inputs"])
    payload = {
        "case": case, "variant": variant, "mode": mode,
        "case_json": str(prepared["case_json"]), "inputs": str(prepared["inputs"]),
        "outdir": str(prepared["cdir"] / "out"),
        "candidate_path": str(CANDIDATE_PATH),
    }
    try:
        worker = _run_worker(payload)
    except subprocess.TimeoutExpired:
        rec["detail"] = "worker 超时（候选可能死循环）"
        return rec
    result = worker.get("result")
    worker_info = _worker_view(result) if isinstance(result, dict) else {}
    rec["worker"] = {
        "rc": worker["rc"], "pid": worker_info.get("pid"),
        "attempts": worker.get("attempts"),
        "attempt_logs": worker.get("attempt_logs"),
        "backend": worker_info.get("backend"),
        "backend_before": worker_info.get("backend_before"),
        "platforms": worker_info.get("platforms"),
        "jax": worker_info.get("jax"),
        "jaxlib": worker_info.get("jaxlib"),
        "stderr_tail": engine._tail(worker["stderr"], 2000),
        "stdout_tail": engine._tail(worker["stdout"], 2000),
    }
    if worker["rc"] != 0 or not isinstance(result, dict):
        rec.update(status="ERROR",
                   detail=f"worker 失败 rc={worker['rc']}：{engine._tail(worker['stderr'], 800)}")
        return rec
    dispatch = dict(result.get("dispatch") or {})
    golden = _run_golden(prepared["case_json"], prepared["inputs"],
                         "reject" if fixtures.is_reject(case) else mode,
                         variant, prepared["cdir"])
    rec["golden"] = {"ok": golden["ok"], "rc": golden["rc"],
                     "backend": (golden["meta"] or {}).get("backend"),
                     "jax": (golden["meta"] or {}).get("jax"),
                     "jaxlib": (golden["meta"] or {}).get("jaxlib")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{engine._tail(golden['stderr'], 800)}")
        return rec

    if fixtures.is_reject(case):
        c16 = str(case.get("case_id")) == "while_iteration_limit_exceeded"
        dispatch.setdefault("_meta", dict(dispatch))
        run = _build_reject_run(worker, dispatch, result.get("outputs_npz"))
        problems = engine.reject_meta_problems(
            case=case, variant=variant, run=run, golden=golden, dispatch=dispatch,
            worker=worker_info if c16 else None)
        evidence = dispatch.get("_device_evidence") or {}
        rec["device_evidence"] = evidence
        rec["identity"] = evidence.get("identity")
        rec["dispatch"] = {key: dispatch.get(key) for key in engine.EXEC_REQUIRED_KEYS}
        rec["reject"] = result.get("reject")
        device_problems = list(evidence.get("problems") or []) if c16 else []
        if device_problems:
            rec.update(status="INVALID",
                       detail="c16 设备门禁未通过：" + "；".join(device_problems))
        elif problems:
            status = "FAIL" if result.get("rc") == engine.CANDIDATE_EXIT_REJECT else "ERROR"
            rec.update(status=status, detail="；".join(problems))
        else:
            rec.update(status="PASS",
                       detail=f"{fixtures.expected_reject_code(case, variant)} 拒绝证据完整")
        return rec

    if result.get("rc") != 0:
        rec.update(status="FAIL", detail="数值用例被候选拒绝/抛错")
        return rec
    outputs_npz = result.get("outputs_npz")
    if not outputs_npz or not Path(outputs_npz).is_file():
        rec.update(status="INVALID", detail="worker 未产出 outputs.npz")
        return rec
    run = {"rc": 0, "meta": dispatch, "exec": {}, "timed_out": False}
    validated, problems = engine._validate_dispatch(case=case, variant=variant, mode=mode,
                                                    run=run, worker=worker_info)
    rec["device_evidence"] = validated.get("_device_evidence") or {}
    rec["identity"] = (rec["device_evidence"] or {}).get("identity")
    rec["dispatch"] = {key: validated.get(key) for key in engine.EXEC_REQUIRED_KEYS}
    if problems:
        rec.update(status="INVALID", detail="；".join(problems))
        return rec
    criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
    leaves, judge_problems = engine.judge_outputs(
        case=case, variant=variant, mode=mode,
        candidate_npz=Path(outputs_npz), golden_npz=golden["npz"],
        candidate_meta={"leaves": result.get("leaves") or [],
                        "tree": result.get("tree")},
        golden_meta=golden["meta"], criteria=criteria)
    rec["leaves"] = leaves
    rec["tree"] = result.get("tree")
    rec["golden_tree"] = (golden["meta"] or {}).get("tree")
    rec["outputs_sha256"] = engine._sha256_file(Path(outputs_npz))
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
    return rec


# ---------------------------------------------------------------------------
# 白盒执行器（节点 3；只写 tests/，不改 ops/、cases/、criteria/、golden）
# ---------------------------------------------------------------------------
# 白盒用例的 cond/body 无法写进 JSON，函数在 tests/while_whitebox_scenarios.py；
# 候选（NPU worker）与 golden（JAX 0.8.1 CPU 子进程）import 同一份场景，保证语义一致。
# 判据、阈值、树/weak_type/float0 规则与黑盒完全相同（criteria/check.py + 同一比较口径）。


def _wb_counts(case: dict, variant: str) -> dict:
    if case.get("variants"):
        table = case.get("variant_expected") or {}
        if variant not in table:
            raise ValueError(f"{case.get('case_id')}: 缺少 variant {variant!r} 的期望计数")
        return dict(table[variant])
    if not case.get("expected_counts"):
        raise ValueError(f"{case.get('case_id')}: 缺少 expected_counts")
    return dict(case["expected_counts"])


def _wb_key(path: str, prefix: str) -> str:
    return prefix if "/" not in path else f"{prefix}/{path.split('/', 1)[1]}"


def _wb_tree(case: dict, variant: str) -> dict:
    table = case.get("variant_tree") or {}
    return table.get(variant) if variant in table else case["tree"]


def _wb_bytes(case: dict, variant: str) -> dict:
    table = case.get("variant_bytes") or {}
    return table.get(variant) if variant in table else (case.get("bytes") or {})


def _wb_leaf_specs(case: dict, variant: str, mode: str) -> dict[str, dict]:
    jvp = mode in ("jvp_eager", "jvp_jit")
    specs: dict[str, dict] = {}
    for leaf in fixtures.leaf_paths(_wb_tree(case, variant)):
        specs[_wb_key(leaf["path"], "primal" if jvp else "out")] = dict(leaf)
    if jvp:
        tangent_tree = case.get("tangent_tree")
        if tangent_tree is None:
            raise ValueError(f"{case.get('case_id')}: {mode} 缺少 tangent_tree")
        for leaf in fixtures.leaf_paths(tangent_tree):
            specs[_wb_key(leaf["path"], "tangent")] = dict(leaf)
    return specs


def _wb_expected_tree(case: dict, variant: str, mode: str):
    base = fixtures.spec_tree(_wb_tree(case, variant))
    if mode in ("jvp_eager", "jvp_jit"):
        return ["tuple", [base, fixtures.spec_tree(case["tangent_tree"])]]
    return base


def _wb_expected_plan(case: dict, variant: str) -> dict | None:
    table = case.get("variant_plan") or {}
    return table.get(variant) if variant in table else case.get("expected_plan")


def _plan_view(plan: dict) -> dict:
    """规范化候选计划，供 manifest 与 CPU 计划自检比对（分支映射的机器可读依据）。"""
    cond = plan.get("cond") or {}
    leaves = []
    for leaf in plan.get("leaves") or []:
        terms = list(leaf.get("terms") or [])
        n_red = sum(1 for term in terms if bool(term[2]))
        leaves.append({
            "dtype": str(leaf.get("dtype")),
            "numel": int(leaf.get("numel") or 0),
            "n_terms": len(terms),
            "n_red": n_red,
            "n_elem": len(terms) - n_red,
        })
    return {
        "cond_op": str(cond.get("op")),
        "cond_is_int": bool(cond.get("is_int")),
        "cond_rhs_leaf": int(cond.get("rhs_leaf")),
        "n_leaves": len(leaves),
        "leaves": leaves,
    }


def _plan_problems(plan: dict, expected: dict | None) -> list[str]:
    if not expected:
        return [f"{plan.get('plan_id', '')}: 缺少 expected_plan/variant_plan"]
    actual = _plan_view(plan)
    problems: list[str] = []
    for key in ("cond_op", "cond_is_int", "cond_rhs_leaf", "n_leaves"):
        if actual[key] != expected.get(key):
            problems.append(f"{key}={actual[key]!r} != {expected.get(key)!r}")
    want_leaves = expected.get("leaves") or []
    if len(want_leaves) != len(actual["leaves"]):
        problems.append(f"leaves={len(actual['leaves'])} != {len(want_leaves)}")
    for index, (got, want) in enumerate(zip(actual["leaves"], want_leaves)):
        for key in ("dtype", "numel", "n_terms", "n_red", "n_elem"):
            if got[key] != want.get(key):
                problems.append(f"leaf[{index}].{key}={got[key]!r} != {want.get(key)!r}")
    return problems


_SIX_CMP = {"lt", "le", "gt", "ge", "eq", "ne"}
_ALL_CMP = {"all_gt", "all_ge", "all_lt", "all_le"}


def _branch_plan_problems(case_id: str, variant: str, expected: dict,
                          branches: set[str], modes: list[str]) -> list[str]:
    """校验 source_branches 与 expected_plan/条件一致（分支 ID 之外的机器可读约束）。"""
    problems: list[str] = []

    def need(ok: bool, text: str):
        if not ok:
            problems.append(f"{case_id}/{variant or 'single'}: {text}")

    cond_op = str(expected.get("cond_op"))
    cond_is_int = bool(expected.get("cond_is_int"))
    cond_leaf = int(expected.get("cond_leaf", -1))
    leaves = expected.get("leaves") or []
    jvp = any(str(mode).startswith("jvp") for mode in modes)
    if "K10" in branches:
        need(cond_is_int and cond_op in _SIX_CMP,
             f"K10 要求 int 标量比较，plan 为 {cond_op!r}/is_int={cond_is_int}")
    if "K12" in branches:
        need((not cond_is_int) and cond_op in _SIX_CMP,
             f"K12 要求 float 标量比较，plan 为 {cond_op!r}/is_int={cond_is_int}")
    if "K11" in branches:
        need(cond_is_int and cond_op in _SIX_CMP, "K11 要求 int 标量 cond")
    if "K13" in branches:
        need((not cond_is_int) and cond_op in _SIX_CMP, "K13 要求 float 标量 cond")
    if "K09" in branches:
        need(cond_op in _ALL_CMP, f"K09 要求 all_* cond，plan 为 {cond_op!r}")
    if "K08" in branches:
        need(cond_op == "always_true", "K08 要求 always_true cond")
    if "K16" in branches:
        need(any(int(leaf.get("n_red") or 0) >= 1 for leaf in leaves),
             "K16 要求至少一个归约项")
    if "K17" in branches:
        need(any(int(leaf.get("n_red") or 0) >= 2 for leaf in leaves),
             "K17 要求同一叶子至少两个归约项")
    if "K19" in branches:
        need(any(leaf.get("dtype") == "float32" and int(leaf.get("n_elem") or 0) == 0
                 for leaf in leaves), "K19 要求 float 叶子只有常量/归约项")
    if "K20" in branches:
        need(any(leaf.get("dtype") == "float32" and int(leaf.get("n_elem") or 0) > 0
                 for leaf in leaves), "K20 要求 float 逐元素项")
    if "K18" in branches:
        need(any(leaf.get("dtype") == "float32" and int(leaf.get("n_elem") or 0) > 0
                 for leaf in leaves), "K18 要求 float 逐元素项")
    if "K21" in branches:
        need(any(leaf.get("dtype") == "float32" for leaf in leaves),
             "K21 要求至少一个 float 叶子")
    if "K22" in branches:
        need(any(leaf.get("dtype") == "float32" and int(leaf.get("n_red") or 0) >= 1
                 for leaf in leaves), "K22 要求 float 归约")
    if "K23" in branches:
        need(jvp and any(leaf.get("dtype") == "float32"
                         and int(leaf.get("n_red") or 0) >= 1 for leaf in leaves),
             "K23 要求 JVP 浮点归约切线")
    if "K24" in branches:
        need(any(leaf.get("dtype") == "int32" for leaf in leaves),
             "K24 要求 int32 叶子")
    if "K26" in branches:
        need(any(leaf.get("dtype") == "int32" and int(leaf.get("n_red") or 0) >= 1
                 for leaf in leaves), "K26 要求 int32 归约")
    if "K25" in branches:
        need(case_id == "while_wb_int_negative_round", "K25 只由 int 负向取整用例命中")
    if "K15" in branches or "K27" in branches:
        need(jvp, "K15/K27 要求 JVP mode")
    if "H05" in branches:
        need(max(expected.get("reduce_src_numels") or [0]) >= 256,
             "H05 要求归约源 numel >= 256")
    if "H10" in branches:
        need(cond_op in _ALL_CMP and 0 <= cond_leaf < len(leaves)
             and int(leaves[cond_leaf].get("numel") or 0) >= 256,
             "H10 要求 all() 条件叶 numel >= 256")
    return problems


def _branch_suite_problems(cases: list[dict]) -> list[str]:
    """跨 variant 的分支子条件（问题 10）：六算子、rhs 两侧、K24 五种组合。

    逐 variant 的 ``_branch_plan_problems`` 只能验证“这条 variant 自身”的合法性；
    本函数把同一 case 的多个 variant 聚合起来，缺少任一子条件即让
    ``test_selfcheck_whitebox_manifest`` 失败，避免 18/18 PASS 掩盖覆盖缺口。
    """
    problems: list[str] = []
    int_ops: set[str] = set()
    float_ops: set[str] = set()
    int_rhs_sides: set[bool] = set()
    float_rhs_sides: set[bool] = set()
    int_leaf_shapes: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id"))
        variants = fixtures.variants_for(case) or [""]
        for variant in variants:
            want = _wb_expected_plan(case, variant)
            if not want:
                continue
            table = case.get("variant_branches") or {}
            branches = set(table.get(variant) if variant in table
                           else case.get("source_branches") or [])
            cond_op = str(want.get("cond_op"))
            is_int = bool(want.get("cond_is_int"))
            rhs_leaf = int(want.get("cond_rhs_leaf", -1))
            if "K10" in branches and is_int:
                int_ops.add(cond_op)
            if "K12" in branches and not is_int:
                float_ops.add(cond_op)
            if "K11" in branches and is_int:
                int_rhs_sides.add(rhs_leaf >= 0)
            if "K13" in branches and not is_int:
                float_rhs_sides.add(rhs_leaf >= 0)
            if "K24" in branches:
                for leaf in want.get("leaves") or []:
                    if str(leaf.get("dtype")) != "int32":
                        continue
                    n_terms = int(leaf.get("n_terms") or 0)
                    n_red = int(leaf.get("n_red") or 0)
                    if n_terms == 0:
                        int_leaf_shapes.add("const")
                    if n_terms == 1 and n_red == 0:
                        int_leaf_shapes.add("elem1")
                    if n_terms >= 2 and n_red == 0:
                        int_leaf_shapes.add("elem2")
                    if n_red == 1:
                        int_leaf_shapes.add("red1")
                    if n_red >= 2:
                        int_leaf_shapes.add("red2")
    missing_int = sorted(_SIX_CMP - int_ops)
    if missing_int:
        problems.append(f"K10 int 比较算子未覆盖全集：missing={missing_int}")
    missing_float = sorted(_SIX_CMP - float_ops)
    if missing_float:
        problems.append(f"K12 float 比较算子未覆盖全集：missing={missing_float}")
    if int_rhs_sides != {False, True}:
        problems.append(f"K11 int rhs 常量/叶两侧未同时覆盖：{sorted(int_rhs_sides)}")
    if float_rhs_sides != {False, True}:
        problems.append(f"K13 float rhs 常量/叶两侧未同时覆盖：{sorted(float_rhs_sides)}")
    want_shapes = {"const", "elem1", "elem2", "red1", "red2"}
    missing_shapes = sorted(want_shapes - int_leaf_shapes)
    if missing_shapes:
        problems.append(f"K24 int 叶组合未覆盖：missing={missing_shapes}")
    return problems


def _worker_main_whitebox(payload: dict) -> int:
    """NPU worker：按 scenario 重建 carry，调用候选 Python 接口，落 outputs.npz + 证据。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import golden  # noqa: PLC0415

    case = payload["case"]
    variant = payload.get("variant") or ""
    mode = payload["mode"]
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    built = wb.build(case["scenario"], jax, jnp, variant)
    init = jax.tree_util.tree_map(jnp.asarray, built["init"])
    cond, body = built["cond"], built["body"]
    # 与 CLI/golden 同一绑定：dispatch.variant 记录测试变体，便于逐用例证据回指。
    cond.opx_variant = variant
    body.opx_variant = variant
    body.opx_case = case
    tangent = None
    if mode in golden.JVP_MODES:
        if built["tangent"] is None:
            raise RuntimeError(f"{case.get('case_id')}: {mode} 场景缺少切向量")
        tangent = jax.tree_util.tree_map(jnp.asarray, built["tangent"])
    module = _load_candidate(payload.get("candidate_path"))
    result: dict = {
        "worker_pid": os.getpid(),
        "worker_backend": jax.default_backend(),
        "worker_devices": sorted({str(device.platform) for device in jax.devices()}),
        "worker_jax": getattr(jax, "__version__", None),
        "worker_jaxlib": getattr(__import__("jaxlib"), "__version__", None),
        "worker_backend_after": None,
        "rc": 0,
        "reject": None,
        "dispatch": {},
        "outputs_npz": None,
        "tree": None,
        "leaves": [],
        "inputs_sha256": None,
        "plan": None,
    }
    # 记录本次调用的线性化计划（分支归属的原始证据；与候选内部 _execute 同源）。
    try:
        plan, _ = module._derive_plan(cond, body, init)
        result["plan"] = {
            "leaves": [{key: leaf[key] for key in
                        ("terms", "const", "dtype", "shape", "numel", "has_tan")}
                       for leaf in plan["leaves"]],
            "cond": plan["cond"],
            "plan_id": module._plan_id(plan),
        }
    except BaseException as exc:  # noqa: BLE001
        result["plan"] = {"error": f"{type(exc).__name__}: {exc}"}
    # 输入快照（numpy）落盘，供证据复核；候选与 golden 都从同一 scenario 确定性重建。
    inputs = golden.flatten(built["init"], "carry")
    if built["tangent"] is not None:
        inputs.update(golden.flatten(built["tangent"], "tangent"))
    result["inputs_sha256"] = fixtures.write_inputs_npz(inputs, outdir / "inputs.npz")
    repeat = max(1, int(case.get("repeat") or 1))
    dispatches: list[dict] = []
    out = None
    try:
        for call_index in range(repeat):
            if mode == "eager":
                out = module.while_loop(cond, body, init)
            elif mode == "jit":
                out = module.while_loop_compiled(cond, body, init, mode="primal")
            else:
                out = module.while_loop_compiled(cond, body, init, mode="jvp",
                                                 tangent=tangent, jit=(mode == "jvp_jit"))
            dispatches.append(dict(_dispatch_of(module)))
        if mode in golden.JVP_MODES:
            primal, tang = out
            result["tree"] = fixtures.tree_signature(out)
            flat = golden.flatten(primal, "primal")
            flat.update(golden.flatten(tang, "tangent"))
            meta = golden.flatten_meta(primal, "primal")
            meta.update(golden.flatten_meta(tang, "tangent"))
        else:
            result["tree"] = fixtures.tree_signature(out)
            flat = golden.flatten(out, "out")
            meta = golden.flatten_meta(out, "out")
        np.savez(outdir / "outputs.npz", **flat)
        result["outputs_npz"] = str(outdir / "outputs.npz")
        result["leaves"] = [{"path": key, **value} for key, value in meta.items()]
        result["repeat"] = repeat
        result["dispatches"] = dispatches
        result["dispatch"] = dispatches[-1] if dispatches else _dispatch_of(module)
    except BaseException as exc:  # noqa: BLE001 —— 拒绝/异常证据原样返回父进程
        code = getattr(exc, "code", None)
        if not code:
            text = str(exc)
            code = text.split(":", 1)[0] if ":" in text else type(exc).__name__
        message = getattr(exc, "upstream_message", "") or str(exc)
        result.update({
            "rc": 3 if code in fixtures.REJECT_CODES else 1,
            "reject": {
                "code": code,
                "upstream_type": getattr(exc, "upstream_exception_type", None),
                "upstream_message": message,
                "upstream_summary": (message or "").replace("\n", " ").strip(),
                "spec_rule": getattr(exc, "spec_rule", None),
            },
            "dispatch": _dispatch_of(module) or getattr(exc, "dispatch", {}) or {},
            "error": f"{type(exc).__name__}: {exc}",
        })
    try:
        result["worker_backend_after"] = jax.default_backend()
        result["worker_devices_after"] = sorted({str(device.platform)
                                                 for device in jax.devices()})
    except Exception as exc:  # noqa: BLE001
        result["worker_backend_after_error"] = f"{type(exc).__name__}: {exc}"
    print("WHILE_WORKER_RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


def _golden_main_whitebox(payload: dict) -> int:
    """CPU golden 子进程：同一 scenario 在 JAX 0.8.1 CPU 上跑 while_loop / jvp。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import golden  # noqa: PLC0415

    case = payload["case"]
    variant = payload.get("variant") or ""
    mode = payload["mode"]
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "golden.json"
    meta: dict = {"ok": False, "backend": None, "leaves": {}, "tree": None}
    try:
        built = wb.build(case["scenario"], jax, jnp, variant)
        init = jax.tree_util.tree_map(jnp.asarray, built["init"])
        cond, body = built["cond"], built["body"]

        def loop(c):
            return jax.lax.while_loop(cond, body, c)

        if mode in golden.JVP_MODES:
            tangent = jax.tree_util.tree_map(jnp.asarray, built["tangent"])
            if mode == "jvp_jit":
                primal, tang = jax.jit(lambda c: jax.jvp(loop, (c,), (tangent,)))(init)
            else:
                primal, tang = jax.jvp(loop, (init,), (tangent,))
            result = (primal, tang)
            flat = golden.flatten(primal, "primal")
            flat.update(golden.flatten(tang, "tangent"))
            leaves = golden.flatten_meta(primal, "primal")
            leaves.update(golden.flatten_meta(tang, "tangent"))
        else:
            result = jax.jit(loop)(init) if mode == "jit" else loop(init)
            flat = golden.flatten(result, "out")
            leaves = golden.flatten_meta(result, "out")
        np.savez(outdir / "golden.npz", **flat)
        meta = {
            "ok": True,
            "backend": jax.default_backend(),
            "jax": getattr(jax, "__version__", None),
            "jaxlib": getattr(__import__("jaxlib"), "__version__", None),
            "leaves": leaves,
            "tree": fixtures.tree_signature(result),
        }
    except BaseException as exc:  # noqa: BLE001
        meta = {"ok": False, "backend": None, "leaves": {}, "tree": None,
                "error": f"{type(exc).__name__}: {exc}"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WHILE_GOLDEN_RESULT " + json.dumps({"ok": meta.get("ok"),
                                              "error": meta.get("error")},
                                             ensure_ascii=False))
    return 0


def _run_whitebox_worker(payload: dict, timeout: int = 600, attempts: int = 2) -> dict:
    """fresh worker；仅对“NPU 后端初始化失败”做一次可审计重试，不重试候选错误。"""
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / "worker_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--whitebox-worker",
           str(payload_path)]
    attempt_logs: list[dict] = []
    proc = None
    parsed = None
    for attempt in range(1, max(1, attempts) + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        parsed = None
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("WHILE_WORKER_RESULT "):
                parsed = json.loads(line[len("WHILE_WORKER_RESULT "):])
                break
        text = (proc.stderr or "") + (proc.stdout or "")
        transient = (parsed is None and proc.returncode != 0
                     and ("Unable to initialize backend 'npu'" in text
                          or "no supported devices found for platform npu" in text))
        attempt_logs.append({"attempt": attempt, "rc": proc.returncode,
                             "transient_npu_init": transient,
                             "stderr_tail": engine._tail(proc.stderr, 1200)})
        if parsed is not None or not transient or attempt >= attempts:
            break
        time.sleep(3)
    return {"cmd": cmd, "rc": proc.returncode if proc else -1,
            "stdout": proc.stdout if proc else "",
            "stderr": proc.stderr if proc else "",
            "result": parsed, "attempts": len(attempt_logs),
            "attempt_logs": attempt_logs}


def _run_whitebox_golden(payload: dict, timeout: int = 600) -> dict:
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / "golden_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--whitebox-golden",
           str(payload_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    meta_path = outdir / "golden.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            meta = {"ok": False, "error": f"golden.json 解析失败：{type(exc).__name__}: {exc}"}
    return {"cmd": cmd, "rc": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "npz": outdir / "golden.npz", "meta": meta,
            "ok": proc.returncode == 0 and bool(meta.get("ok"))}


def _judge_whitebox(*, case: dict, variant: str, mode: str, candidate_npz: Path,
                    golden_npz: Path, candidate_meta: dict, golden_meta: dict,
                    criteria) -> tuple[list[dict], list[str]]:
    """与 engine.judge_outputs 同口径的逐叶判定（树/结构/weak_type/float0/判据）。"""
    problems: list[str] = []
    specs = _wb_leaf_specs(case, variant, mode)
    expected = list(specs)
    try:
        candidate = engine._iter_npz(candidate_npz)
    except Exception as exc:  # noqa: BLE001
        return [], [f"候选 outputs.npz 读取失败：{type(exc).__name__}: {exc}"]
    try:
        gold = engine._iter_npz(golden_npz)
    except Exception as exc:  # noqa: BLE001
        return [], [f"golden.npz 读取失败：{type(exc).__name__}: {exc}"]
    if sorted(candidate) != sorted(expected):
        problems.append(f"候选输出键 {sorted(candidate)} != 期望 {sorted(expected)}")
    if sorted(gold) != sorted(expected):
        problems.append(f"golden 输出键 {sorted(gold)} != 期望 {sorted(expected)}")
    expected_tree = _wb_expected_tree(case, variant, mode)
    problems.extend(engine._tree_problems("候选", expected_tree, candidate_meta.get("tree")))
    problems.extend(engine._tree_problems("golden", expected_tree, golden_meta.get("tree")))
    candidate_leaves = engine._leaves_map(candidate_meta.get("leaves"))
    gold_leaves = engine._leaves_map(golden_meta.get("leaves"))
    if not candidate_leaves:
        problems.append("候选 meta.leaves 缺失（无法核对逐叶结构/weak_type）")
    if not gold_leaves:
        problems.append("golden meta.leaves 缺失（无法核对逐叶 weak_type）")
    leaves: list[dict] = []
    for key in expected:
        spec = specs[key]
        item = candidate_leaves.get(key) or {}
        gold_item = gold_leaves.get(key) or {}
        record = {
            "path": key, "dtype": spec["dtype"], "shape": list(spec["shape"]),
            "weak_type": item.get("weak_type"),
            "candidate_weak_type": item.get("weak_type"),
            "golden_weak_type": gold_item.get("weak_type"),
            "weak_type_compared": False, "is_pass": False, "detail": "",
        }
        if not item:
            record["detail"] = "候选 meta.leaves 缺少该叶子"
            leaves.append(record)
            continue
        if not gold_item:
            problems.append(f"golden meta.leaves 缺少 {key}")
        dtype = engine._dtype_text(item.get("dtype"))
        if not engine._dtype_matches(spec["dtype"], dtype):
            record["dtype"] = dtype
            record["detail"] = f"dtype={dtype} != {spec['dtype']}"
            leaves.append(record)
            continue
        shape = [int(x) for x in (item.get("shape") or [])]
        if shape != [int(x) for x in spec["shape"]]:
            record["shape"] = shape
            record["detail"] = f"shape={shape} != {spec['shape']}"
            leaves.append(record)
            continue
        gold_wt = gold_item.get("weak_type")
        cand_wt = item.get("weak_type")
        if gold_wt is not None:
            if cand_wt is None:
                record["detail"] = "无法观测 weak_type（候选未提供 jax.Array weak_type）"
                problems.append(f"{key} 无法观测 weak_type（golden={gold_wt}）")
                leaves.append(record)
                continue
            record["weak_type_compared"] = True
            if bool(cand_wt) != bool(gold_wt):
                record["detail"] = f"weak_type={cand_wt} != golden {gold_wt}"
                problems.append(f"{key} weak_type={cand_wt} != golden {gold_wt}")
                leaves.append(record)
                continue
        if key not in candidate or key not in gold:
            record["detail"] = "输出键缺失"
            leaves.append(record)
            continue
        actual, reference = candidate[key], gold[key]
        record["dtype"] = engine._dtype_text(actual.dtype)
        record["shape"] = [int(x) for x in actual.shape]
        if "float0" in spec["dtype"]:
            if actual.nbytes != 0 or reference.nbytes != 0:
                record["detail"] = (f"float0 尺寸非 0：candidate.nbytes={actual.nbytes} "
                                    f"golden.nbytes={reference.nbytes}")
            else:
                record["is_pass"] = True
                record["detail"] = "float0 结构精确（dtype/shape/nbytes=0）"
            leaves.append(record)
            continue
        if spec["dtype"] == "int32":
            ok = bool(np.array_equal(actual, reference))
            record["is_pass"] = ok
            record["detail"] = "int32 逐元素精确相等" if ok else "int32 存在不等元素"
            leaves.append(record)
            continue
        try:
            judged = criteria.check(actual, reference)
            record.update({
                "is_pass": bool(judged["is_pass"]),
                "matched_ratio": judged.get("matched_ratio"),
                "max_abs_error": judged.get("max_abs_error"),
                "nonfinite": judged.get("nonfinite"),
                "failure_reasons": judged.get("failure_reasons") or [],
            })
            record["detail"] = "混合容差通过" if judged["is_pass"] else "混合容差未通过"
        except Exception as exc:  # noqa: BLE001
            record["detail"] = f"判据异常：{type(exc).__name__}: {exc}"
        leaves.append(record)
    for leaf in leaves:
        if not leaf["is_pass"]:
            problems.append(f"{leaf['path']} 未通过：{leaf['detail']}")
    return leaves, problems


def _run_whitebox_item(item: dict) -> dict:
    """白盒用例：NPU worker + CPU golden + 设备证据 + 判据；预期错误用例只判拒绝码/摘要。"""
    case, variant, mode = item["case"], item["variant"], item["mode"]
    expected_error = str(case.get("expected_error") or "")
    counts = {} if expected_error else _wb_counts(case, variant)
    rec = {
        "case_id": item["case_id"], "variant": variant, "mode": item["display_mode"],
        "requested_mode": mode, "required": True, "origin": "whitebox",
        "status": "ERROR", "detail": "",
        "bytes": _wb_bytes(case, variant),
        "expected_iterations": counts.get("iterations"),
        "expected_counts": counts,
        "expected_error": expected_error or None,
        "source_branches": list(case.get("source_branches") or []),
    }
    cdir = WORK_DIR / f"wb_{item['case_id']}__{variant or 'single'}__{mode}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    payload = {
        "case": {key: value for key, value in case.items() if key != "_origin"},
        "variant": variant, "mode": mode,
        "outdir": str(cdir / "out"),
        "candidate_path": str(CANDIDATE_PATH),
    }
    try:
        worker = _run_whitebox_worker(payload)
    except subprocess.TimeoutExpired:
        rec["detail"] = "白盒 worker 超时（候选可能死循环）"
        return rec
    result = worker.get("result")
    if not isinstance(result, dict):
        rec.update(status="ERROR",
                   detail=f"白盒 worker 失败 rc={worker['rc']}："
                          f"{engine._tail(worker['stderr'], 800)}")
        return rec
    worker_info = _worker_view(result)
    rec["worker"] = {
        "rc": worker["rc"], "pid": worker_info.get("pid"),
        "attempts": worker.get("attempts"),
        "attempt_logs": worker.get("attempt_logs"),
        "backend": worker_info.get("backend"),
        "backend_before": worker_info.get("backend_before"),
        "platforms": worker_info.get("platforms"),
        "jax": worker_info.get("jax"), "jaxlib": worker_info.get("jaxlib"),
        "stderr_tail": engine._tail(worker["stderr"], 2000),
        "stdout_tail": engine._tail(worker["stdout"], 2000),
    }
    rec["inputs_sha256"] = result.get("inputs_sha256")
    rec["plan"] = result.get("plan")
    rec["inputs"] = str((cdir / "out" / "inputs.npz").relative_to(WS)) \
        if str(cdir).startswith(str(WS)) else str(cdir / "out" / "inputs.npz")
    if expected_error:
        reject = dict(result.get("reject") or {})
        code = str(reject.get("code") or "")
        message = str(reject.get("upstream_message") or result.get("error") or "")
        problems = []
        if code != expected_error:
            problems.append(f"拒绝码 {code!r} != 期望 {expected_error!r}：{message}")
        for needle in case.get("expected_error_contains") or []:
            if str(needle) not in message:
                problems.append(f"上游摘要缺少 {needle!r}：{message}")
        plan = result.get("plan") or {}
        if not plan or plan.get("error"):
            problems.append(f"计划未生成：{plan.get('error') if plan else '缺失'}")
        backend = str(worker_info.get("backend") or "")
        platforms = [str(value) for value in (worker_info.get("platforms") or [])]
        if backend != "npu" or "npu" not in platforms:
            problems.append(f"预期错误仍须真实 NPU worker：backend={backend!r} "
                            f"platforms={platforms}")
        rec.update(status="PASS" if not problems else "FAIL",
                   detail=("预期错误命中：" + (message or code)) if not problems
                          else "；".join(problems),
                   reject=reject, plan=plan, dispatch={},
                   route="host_validate_reject",
                   worker_backend=backend, worker_platforms=platforms,
                   worker_pid=worker_info.get("pid"))
        return rec
    if worker["rc"] != 0 or result.get("rc") != 0:
        rec.update(status="FAIL",
                   detail=f"候选非 0：{result.get('reject') or result.get('error')}")
        return rec
    golden_payload = dict(payload, outdir=str(cdir / "golden"))
    try:
        golden = _run_whitebox_golden(golden_payload)
    except subprocess.TimeoutExpired:
        rec["detail"] = "白盒 golden 超时"
        return rec
    rec["golden"] = {"ok": golden["ok"], "rc": golden["rc"],
                     "backend": (golden["meta"] or {}).get("backend"),
                     "jax": (golden["meta"] or {}).get("jax"),
                     "jaxlib": (golden["meta"] or {}).get("jaxlib"),
                     "error": (golden["meta"] or {}).get("error")}
    if not golden["ok"]:
        rec.update(status="ERROR",
                   detail=f"golden 失败：{(golden['meta'] or {}).get('error') or golden['stderr'][-400:]}")
        return rec
    dispatch = dict(result.get("dispatch") or {})
    evidence, evidence_problems = engine.device_evidence(
        dispatch=dispatch, worker=worker_info, require_npu=True)
    rec["device_evidence"] = evidence
    rec["identity"] = evidence.get("identity")
    rec["dispatch"] = {key: dispatch.get(key) for key in engine.EXEC_REQUIRED_KEYS}
    missing = [key for key in engine.EXEC_REQUIRED_KEYS
               if key not in dispatch or (key != "variant" and dispatch.get(key) in (None, ""))]
    problems = list(evidence_problems)
    if missing:
        problems.append(f"dispatch 缺少字段：{missing}")
    for key in ("iterations", "cond_hits", "body_hits", "limit_flag"):
        actual = engine._as_int(dispatch.get(key))
        expected = counts.get(key)
        if actual is None:
            problems.append(f"{key} 不是整数")
        elif expected is not None and actual != expected:
            problems.append(f"{key}={actual} != {expected}")
    abi = engine._as_int(dispatch.get("abi_calls"))
    expected_abi = 1 + int(counts.get("iterations") or 0)
    if abi is None:
        problems.append(f"abi_calls={dispatch.get('abi_calls')!r} 不是整数")
    elif abi != expected_abi:
        problems.append(f"abi_calls={abi} != 1+iterations={expected_abi}")
    if mode in ("jit", "jvp_jit"):
        if not str(dispatch.get("plan_id") or ""):
            problems.append("jit/jvp_jit 缺少 plan_id")
        if engine._as_int(dispatch.get("compile_count")) != 1:
            problems.append(f"compile_count={dispatch.get('compile_count')!r}（fresh worker 首调应为 1）")
        if str(dispatch.get("plan_cache") or "") not in ("miss", "hit"):
            problems.append(f"plan_cache={dispatch.get('plan_cache')!r}")
    expected_cache = str(case.get("expected_plan_cache") or "")
    if expected_cache and str(dispatch.get("plan_cache") or "") != expected_cache:
        problems.append(f"plan_cache={dispatch.get('plan_cache')!r} != {expected_cache!r}")
    repeat = int(case.get("repeat") or 1)
    if repeat > 1:
        sequence = [str(item.get("plan_cache") or "")
                    for item in (result.get("dispatches") or [])]
        expect_sequence = ["miss"] + ["hit"] * (repeat - 1)
        if sequence != expect_sequence:
            problems.append(f"plan_cache 序列 {sequence} != {expect_sequence}")
    if problems:
        rec.update(status="INVALID", detail="；".join(problems))
        return rec
    criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
    outputs_npz = Path(result.get("outputs_npz") or "")
    if not outputs_npz.is_file():
        rec.update(status="INVALID", detail="候选未产出 outputs.npz")
        return rec
    leaves, judge_problems = _judge_whitebox(
        case=case, variant=variant, mode=mode,
        candidate_npz=outputs_npz, golden_npz=Path(golden["npz"]),
        candidate_meta={"leaves": result.get("leaves") or [], "tree": result.get("tree")},
        golden_meta=golden["meta"], criteria=criteria)
    rec["leaves"] = leaves
    rec["tree"] = result.get("tree")
    rec["golden_tree"] = (golden["meta"] or {}).get("tree")
    rec["outputs_sha256"] = engine._sha256_file(outputs_npz)
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS",
                   detail=(f"{mode} 通过（{len(leaves)} 个叶子；iterations="
                           f"{counts.get('iterations')}；abi_calls={abi}；"
                           f"branches={','.join(rec['source_branches'])}）"))
    rec["repeat"] = int(case.get("repeat") or 1)
    rec["plan_cache_sequence"] = [str(item.get("plan_cache") or "")
                                  for item in (result.get("dispatches") or [])]
    return rec


# ---------------------------------------------------------------------------
# pytest 入口
# ---------------------------------------------------------------------------


_ALL_ITEMS = _all_items()


@pytest.mark.opx_case
@pytest.mark.parametrize("item", _ALL_ITEMS, ids=[_item_id(item) for item in _ALL_ITEMS])
def test_case_matrix(item):
    if item["case"].get("_origin") == "whitebox":
        rec = _run_whitebox_item(item)
    else:
        rec = _run_case_item(item)
    _RESULTS.append(rec)
    assert rec["status"] == "PASS", f"{rec['case_id']}/{rec['variant']}/{rec['mode']}: {rec['detail']}"


# ---------------------------------------------------------------------------
# 自检（CPU；不执行候选，不计入 61 个门禁项）
# ---------------------------------------------------------------------------


def _selfcheck(name: str, fn):
    try:
        detail = fn() or ""
        _SELFCHECK_RECORDS.append({"name": name, "status": "PASS", "detail": detail})
    except Exception as exc:  # noqa: BLE001
        _SELFCHECK_RECORDS.append({"name": name, "status": "FAIL",
                                   "detail": f"{type(exc).__name__}: {exc}"})
        raise


def test_selfcheck_discovery_counts():
    def check():
        data = _load_all()
        gate = fixtures.count_summary(data["gate"]["cases"])
        bench = fixtures.count_summary(data["bench"]["cases"])
        supp = fixtures.count_summary(data["supplement"]["cases"])
        whitebox = fixtures.count_summary(data["whitebox"]["cases"])
        assert (gate["identities"], gate["mode_executions"]) == (16, 51), gate
        assert (bench["identities"], bench["mode_executions"]) == (2, 2), bench
        assert (supp["identities"], supp["mode_executions"]) == (3, 8), supp
        total_cases = 16 + 2 + 3 + whitebox["identities"]
        total_modes = 51 + 2 + 8 + whitebox["mode_executions"]
        assert total_cases == EXPECTED_BLACKBOX_CASES + whitebox["identities"]
        assert total_modes == EXPECTED_BLACKBOX_MODES + whitebox["mode_executions"]
        assert len(_ALL_ITEMS) == total_modes
        expected_by_mode = {"eager": 18, "jit": 23, "jvp_eager": 2, "jvp_jit": 2,
                            "reject": 16}
        for item in _items_for(data["whitebox"]):
            mode = item["display_mode"]
            expected_by_mode[mode] = expected_by_mode.get(mode, 0) + 1
        by_mode = {}
        for item in _ALL_ITEMS:
            by_mode[item["display_mode"]] = by_mode.get(item["display_mode"], 0) + 1
        assert by_mode == expected_by_mode, (by_mode, expected_by_mode)
        return (f"gate=16/51 bench=2/2 supplement=3/8 whitebox="
                f"{whitebox['identities']}/{whitebox['mode_executions']} "
                f"pytest={total_cases}/{total_modes} by_mode={by_mode}")
    _selfcheck("discovery_counts", check)


def test_selfcheck_frozen_hashes():
    def check():
        import hashlib

        def sha(path):
            h = hashlib.sha256()
            h.update(Path(path).read_bytes())
            return h.hexdigest()

        assert sha(FROZEN_CASES_PATH) == CASES_SHA256, sha(FROZEN_CASES_PATH)
        assert sha(BENCH_PATH) == BENCH_SHA256, sha(BENCH_PATH)
        return "cases.json/benchmark.json 与冻结哈希一致"
    _selfcheck("frozen_hashes", check)


def test_selfcheck_data_generation_and_bytes():
    def check():
        data = _load_all()
        by_id = {case["case_id"]: case for case in data["gate"]["cases"]}
        # 同一 case/variant 两次生成完全一致；eager/jit 共用同一份输入。
        first = fixtures.input_arrays(by_id["while_large_tensor"], "")
        second = fixtures.input_arrays(by_id["while_large_tensor"], "")
        for key in first:
            assert np.array_equal(first[key], second[key]), key
        expected = {
            ("while_zero_iterations", ""): ("carry/1", 8, 32, 0),
            ("while_unaligned_tail_widths", "w17"): ("carry/1", 17, 68, 4),
            ("while_unaligned_tail_widths", "w33"): ("carry/1", 33, 132, 4),
            ("while_tuple_carry", ""): ("carry/2", 12, 48, 16),
            ("while_nested_dict_carry", ""): ("carry/state/0", 15, 60, 28),
            ("while_nested_dict_carry", ""): ("carry/state/1/aux", 5, 20, 20),
            ("while_large_tensor", ""): ("carry/1", 262144, 1048576, 0),
        }
        for (case_id, variant), (path, elements, total, mod32) in expected.items():
            info = fixtures.bytes_info(by_id[case_id], variant)[path]
            assert (info["elements"], info["bytes"], info["bytes_mod_32"]) == \
                (elements, total, mod32), (case_id, path, info)
        s1 = {case["case_id"]: case for case in data["supplement"]["cases"]}["while_sup_weak_type"]
        info = fixtures.bytes_info(s1, "weak")["carry"]
        assert (info["elements"], info["bytes"], info["bytes_mod_32"]) == (1, 4, 4), info
        special = fixtures.input_arrays(
            {case["case_id"]: case for case in data["supplement"]["cases"]}
            ["while_sup_nonfinite"], "")["carry/1"]
        assert np.isnan(special[0]) and np.isinf(special[1]) and special[1] > 0
        return "确定性输入 + 8 组字节/对齐实算 + S2 特殊值"
    _selfcheck("data_generation_and_bytes", check)


def test_selfcheck_smoke_case():
    def check():
        data = _load_all()
        case = engine.pick_smoke_case(data["gate"]["cases"])
        assert case["case_id"] == "while_one_iteration"
        assert fixtures.modes_for(case) == ["eager", "jit"]
        assert fixtures.expected_counts(case, "") == {
            "iterations": 1, "body_hits": 1, "cond_hits": 2, "limit_flag": 0}
        return "smoke 固定 while_one_iteration/eager"
    _selfcheck("smoke_case", check)


def test_selfcheck_thresholds():
    def check():
        criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
        sample = np.asarray([1.0, np.inf, np.nan], dtype=np.float32)
        judged = criteria.check(sample, sample.copy())
        assert judged["rtol"] == 2 ** -10, judged
        assert judged["atol"] == 2 ** -16, judged
        assert judged["required_matched_ratio"] == 0.99, judged
        assert judged["max_abs_error_limit"] == 1e-2, judged
        assert judged["is_pass"]
        return "ops-precision-standard@1 阈值逐项一致"
    _selfcheck("thresholds", check)


def test_selfcheck_golden_representative():
    def check():
        data = _load_all()
        gate = {case["case_id"]: case for case in data["gate"]["cases"]}
        supp = {case["case_id"]: case for case in data["supplement"]["cases"]}
        cdir = WORK_DIR / "selfcheck_golden_rep"
        cdir.mkdir(parents=True, exist_ok=True)
        records = []

        def run(case, variant, mode):
            case_json = cdir / f"{case['case_id']}_{variant or 'single'}_{mode}_case.json"
            inputs = cdir / f"{case['case_id']}_{variant or 'single'}_{mode}_inputs.npz"
            fixtures.write_case_json(case, variant, mode, case_json)
            fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
            golden = engine._run_golden(WS, case_json, inputs, mode, variant, cdir)
            assert golden["ok"], f"{case['case_id']} golden 失败：{golden['stderr']}"
            records.append((case["case_id"], variant, mode, sorted(golden["meta"]["outputs"])))
            return golden

        run(gate["while_zero_iterations"], "", "eager")
        run(gate["while_scalar_carry"], "", "eager")
        run(gate["while_jvp_multi_iteration"], "", "jvp_eager")
        weak = run(supp["while_sup_weak_type"], "weak", "eager")
        strong = run(supp["while_sup_weak_type"], "strong", "eager")
        assert weak["meta"]["leaves"]["out"]["weak_type"] is True
        assert strong["meta"]["leaves"]["out"]["weak_type"] is False
        jvp = run(gate["while_jvp_multi_iteration"], "", "jvp_eager")
        assert jvp["meta"]["leaves"]["tangent/0"]["dtype"] == "[('float0', 'V')]"
        return f"代表 golden 绑定 {records}"
    _selfcheck("golden_representative", check)


def test_selfcheck_golden_all_items():
    def check():
        data = _load_all()
        cdir = WORK_DIR / "selfcheck_golden_all"
        cdir.mkdir(parents=True, exist_ok=True)
        counts = {"numeric": 0, "reject": 0, "spec_rule": 0}
        for item in _blackbox_items():
            case, variant, mode = item["case"], item["variant"], item["mode"]
            tag = f"{item['case_id']}_{variant or 'single'}_{mode}"
            case_json = cdir / f"{tag}_case.json"
            inputs = cdir / f"{tag}_inputs.npz"
            fixtures.write_case_json(case, variant, mode, case_json)
            fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
            golden_mode = "reject" if fixtures.is_reject(case) else mode
            result = engine._run_golden(WS, case_json, inputs, golden_mode, variant, cdir)
            assert result["ok"], f"{tag} golden rc={result['rc']}: {result['stderr'][-400:]}"
            if golden_mode == "reject":
                evidence = result["meta"]["reject_evidence"]
                counts["reject"] += 1
                if evidence.get("spec_rule_rejects"):
                    counts["spec_rule"] += 1
            else:
                counts["numeric"] += 1
        assert counts["numeric"] == 45, counts
        assert counts["reject"] == 16, counts
        assert counts["spec_rule"] == 2, counts  # c16 eager/jit
        return f"{counts}（45 数值 + 16 拒绝，其中 c16 两条规格规则）"
    _selfcheck("golden_all_items", check)


def test_selfcheck_reject_rules():
    def check():
        data = _load_all()
        gate = {case["case_id"]: case for case in data["gate"]["cases"]}
        cdir = WORK_DIR / "selfcheck_reject"
        cdir.mkdir(parents=True, exist_ok=True)
        expected = {
            ("while_invalid_condition", "bool1"): "INVALID_CONDITION",
            ("while_invalid_condition", "int_scalar"): "INVALID_CONDITION",
            ("while_carry_structure_mismatch", ""): "CARRY_STRUCTURE_MISMATCH",
            ("while_carry_shape_or_dtype_mismatch", "shape"): "CARRY_SHAPE_MISMATCH",
            ("while_carry_shape_or_dtype_mismatch", "dtype"): "CARRY_DTYPE_MISMATCH",
            ("while_iteration_limit_exceeded", ""): "ITERATION_LIMIT_EXCEEDED",
        }
        for (case_id, variant), code in expected.items():
            case = gate[case_id]
            tag = f"{case_id}_{variant or 'single'}"
            case_json = cdir / f"{tag}_case.json"
            inputs = cdir / f"{tag}_inputs.npz"
            fixtures.write_case_json(case, variant, "eager", case_json)
            fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
            result = engine._run_golden(WS, case_json, inputs, "reject", variant, cdir)
            assert result["ok"], f"{tag} golden 未确认：{result['stderr'][-400:]}"
            evidence = result["meta"]["reject_evidence"]
            assert evidence["reject_code"] == code, (tag, evidence)
            if case_id == "while_iteration_limit_exceeded":
                assert evidence["cpu_rejects"] is False
                assert evidence["spec_rule_rejects"] is True
                assert evidence["spec_rule"] == "iteration_safety_limit"
                assert evidence["runtime_evidence_required"]["iterations"] == 64
            else:
                assert evidence["cpu_rejects"] is True
                assert evidence["upstream_exception_type"] == "TypeError"
                assert evidence["upstream_message"]
        return "c13-c16 五码 + c16 规格规则（CPU 不执行 always_true）"
    _selfcheck("reject_rules", check)


def test_selfcheck_report_generation():
    def check():
        synthetic = [{
            "case_id": "while_one_iteration", "variant": "", "mode": "eager",
            "requested_mode": "eager", "required": True, "status": "PASS",
            "detail": "SYNTHETIC schema selfcheck", "dispatch": {
                "path": "so", "backend": "npu", "abi_calls": 1, "route": "device_loop",
                "iterations": 1, "cond_hits": 2, "body_hits": 1, "limit_flag": 0,
                "plan_id": "", "compile_count": 0, "plan_cache": "", "runtime_token": "t",
                "counter_source": "kernel", "trace_hits": 0, "exec_hits": 1,
            }, "bytes": {},
        }]
        report = {
            "schema": "opx-jax-while-loop-correctness/1", "op": OP, "action": "test",
            "run_id": "SYNTHETIC", "spec_version_id": 7,
            "criteria_ref": "ops-precision-standard@1",
            "code_snapshot_sha256": "SYNTHETIC",
            "suite_ref": "cases@288876910a5e2843",
            "candidate": {"exe": "SYNTHETIC", "sha256": "0" * 64},
            "cases": synthetic, "summary": {"PASS": 1, "FAIL": 0, "ERROR": 0,
                                            "INVALID": 0, "NOT_RUN": 0, "TOTAL": 1},
            "by_mode": engine._by_mode(synthetic),
            "authenticity": engine._authenticity(synthetic),
            "cross_checks": engine._cross_checks(synthetic),
        }
        out = OUT_DIR / "selfcheck_report.json"
        md = OUT_DIR / "selfcheck_report.md"
        engine.write_report(out=out, md=md, report=report)
        assert out.is_file() and out.stat().st_size > 0
        assert md.is_file() and md.stat().st_size > 0
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["schema"] == "opx-jax-while-loop-correctness/1"
        assert loaded["by_mode"]["eager"]["total"] == 1
        return "JSON/MD 报告可生成（合成条目，非设备证据）"
    _selfcheck("report_generation", check)


def _golden_bundle(case: dict, variant: str, mode: str, tag: str):
    cdir = WORK_DIR / "selfcheck_negative"
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / f"{tag}_case.json"
    inputs = cdir / f"{tag}_inputs.npz"
    fixtures.write_case_json(case, variant, mode, case_json)
    fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
    result = engine._run_golden(WS, case_json, inputs, mode, variant, cdir)
    assert result["ok"], f"{tag} golden 失败：{result['stderr'][-400:]}"
    assert result["meta"].get("tree") is not None
    leaves = [{"path": key, **value} for key, value in result["meta"]["leaves"].items()]
    return result, {"leaves": leaves, "tree": result["meta"]["tree"]}


def test_selfcheck_weak_type_negative():
    """P2 负向自检：S1 四项（weak/strong × eager/jit）weak_type 翻转都必须 FAIL。"""
    def check():
        data = _load_all()
        supp = {case["case_id"]: case for case in data["supplement"]["cases"]}
        criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
        checked = []
        for variant in ("weak", "strong"):
            for mode in ("eager", "jit"):
                tag = f"s1_{variant}_{mode}"
                golden, candidate_meta = _golden_bundle(
                    supp["while_sup_weak_type"], variant, mode, tag)
                leaves, problems = engine.judge_outputs(
                    case=supp["while_sup_weak_type"], variant=variant, mode=mode,
                    candidate_npz=golden["npz"], golden_npz=golden["npz"],
                    candidate_meta=candidate_meta, golden_meta=golden["meta"],
                    criteria=criteria)
                assert not problems, problems
                assert leaves[0]["weak_type_compared"] is True, leaves[0]
                assert leaves[0]["candidate_weak_type"] == leaves[0]["golden_weak_type"]
                wrong = json.loads(json.dumps(candidate_meta))
                wrong["leaves"][0]["weak_type"] = not bool(
                    wrong["leaves"][0]["weak_type"])
                _, bad_problems = engine.judge_outputs(
                    case=supp["while_sup_weak_type"], variant=variant, mode=mode,
                    candidate_npz=golden["npz"], golden_npz=golden["npz"],
                    candidate_meta=wrong, golden_meta=golden["meta"],
                    criteria=criteria)
                assert any("weak_type" in item for item in bad_problems), bad_problems
                checked.append((variant, mode, candidate_meta["leaves"][0]["weak_type"]))
        return f"S1 四项 weak/strong×eager/jit 翻转均 FAIL；正确候选 PASS：{checked}"
    _selfcheck("weak_type_negative", check)


def _flip_first_tuple(node) -> bool:
    """把容器指纹里第一个 ``tuple`` 改成 ``list``（模拟候选返回错容器类型）。"""
    if not isinstance(node, list):
        return False
    if node and node[0] == "tuple":
        node[0] = "list"
        return True
    for child in node:
        if _flip_first_tuple(child):
            return True
    return False


def test_selfcheck_tree_container_negative():
    """P4 负向自检：c05/c06 多叶用例的 tuple→list 容器漂移必须 FAIL。"""
    def check():
        data = _load_all()
        gate = {case["case_id"]: case for case in data["gate"]["cases"]}
        criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
        checked = []
        for case_id in ("while_tuple_carry", "while_nested_dict_carry"):
            case = gate[case_id]
            tag = f"{case_id}_tree"
            golden, candidate_meta = _golden_bundle(case, "", "eager", tag)
            expected_tree = fixtures.expected_output_tree(case, "", "eager")
            assert candidate_meta["tree"] == expected_tree, candidate_meta["tree"]
            _, problems = engine.judge_outputs(
                case=case, variant="", mode="eager",
                candidate_npz=golden["npz"], golden_npz=golden["npz"],
                candidate_meta=candidate_meta, golden_meta=golden["meta"],
                criteria=criteria)
            assert not problems, problems
            wrong = json.loads(json.dumps(candidate_meta))
            assert _flip_first_tuple(wrong["tree"]), wrong["tree"]
            _, bad_problems = engine.judge_outputs(
                case=case, variant="", mode="eager",
                candidate_npz=golden["npz"], golden_npz=golden["npz"],
                candidate_meta=wrong, golden_meta=golden["meta"], criteria=criteria)
            assert any("tree" in item for item in bad_problems), bad_problems
            checked.append(case_id)
        return f"{checked} 的 tuple→list 均被拒；正确容器 PASS"
    _selfcheck("tree_container_negative", check)


def test_selfcheck_device_identity_negative():
    """P1/P3/P5 负向自检：CPU/缺 platforms/字符串 abi_calls/身份缺失必须 INVALID。"""
    def check():
        identity = {"image_id": "145", "host_device_id": "5",
                    "container_device_id": "0", "source": {}}
        expected = dict(identity)
        probe_ok = {"backend": "npu", "platforms": ["npu"], "jax": "0.8.1",
                    "jaxlib": "0.8.1", "problems": []}
        worker_ok = {"backend": "npu", "platforms": ["npu"], "jax": "0.8.1",
                     "jaxlib": "0.8.1"}
        good = {"path": "so", "backend": "npu", "platforms": ["npu"], "abi_calls": 1,
                "counter_source": "kernel", "route": "device_loop", "device_id": 0,
                "runtime_token": "tok", "_meta": {}}
        _, problems = engine.device_evidence(
            dispatch=good, worker=worker_ok, identity=identity, expected=expected,
            probe=probe_ok)
        assert not problems, problems
        cases = {
            "cpu_backend": dict(good, backend="cpu"),
            "missing_platforms": {k: v for k, v in good.items() if k != "platforms"},
            "string_abi_calls": dict(good, abi_calls="1"),
            "wrong_device_id": dict(good, device_id=1),
            "bad_route": dict(good, route="cpu"),
        }
        for name, dispatch in cases.items():
            _, bad = engine.device_evidence(
                dispatch=dispatch, worker=worker_ok, identity=identity,
                expected=expected, probe=probe_ok)
            assert bad, name
        _, bad_worker = engine.device_evidence(
            dispatch=good, worker={"backend": "cpu", "platforms": ["cpu"],
                                   "jax": "0.8.1", "jaxlib": "0.8.1"},
            identity=identity, expected=expected, probe=probe_ok)
        assert any("worker" in item for item in bad_worker), bad_worker
        _, bad_identity = engine.device_evidence(
            dispatch=good, worker=worker_ok,
            identity=dict(identity, image_id=None), expected=expected, probe=probe_ok)
        assert any("image_id" in item for item in bad_identity), bad_identity
        _, bad_probe = engine.device_evidence(
            dispatch=good, worker=worker_ok, identity=identity, expected=expected,
            probe={"backend": "cpu", "platforms": ["cpu"], "problems": ["cpu"]})
        assert any("cpu" in item for item in bad_probe), bad_probe
        return "CPU/缺 platforms/字符串 abi_calls/错误 device_id/身份缺失/CPU probe 均被拒"
    _selfcheck("device_identity_negative", check)


def test_selfcheck_whitebox_loader():
    def check():
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = WORK_DIR / "selfcheck_whitebox.json"
        payload = {
            "schema": "opx-whitebox-cases/1", "op": OP,
            "cases": [{
                "case_id": "while_wb_selfcheck", "group": "白盒自检",
                "scenario": "tile_exact_8192", "mode": ["eager"],
                "expected_counts": {"iterations": 2, "body_hits": 2, "cond_hits": 3,
                                    "limit_flag": 0},
            }],
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        loaded = _load_whitebox(tmp)
        assert loaded["cases"][0]["_origin"] == "whitebox"
        summary = fixtures.count_summary(loaded["cases"])
        assert summary["identities"] == 1 and summary["mode_executions"] == 1
        bad = WORK_DIR / "selfcheck_whitebox_bad.json"
        bad.write_text(json.dumps({"op": "other", "cases": []}), encoding="utf-8")
        try:
            _load_whitebox(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("未知 op 的白盒文件未被拒绝")
        return "白盒 loader：origin 标记 + fixtures.expand_items 复用 + op 校验"
    _selfcheck("whitebox_loader", check)


def test_selfcheck_whitebox_manifest():
    def check():
        suite = _load_whitebox()
        cases = suite["cases"]
        assert cases, "白盒用例文件为空"
        assert len(cases) == EXPECTED_WHITEBOX_CASES, len(cases)
        ids = [case["case_id"] for case in cases]
        assert len(ids) == len(set(ids)), ids
        registry = set(suite.get("branch_registry") or [])
        assert registry, "白盒 branch_registry 为空"
        notes = suite.get("branch_notes") or {}
        assert set(notes) <= registry, sorted(set(notes) - registry)
        for branch, note in notes.items():
            assert note.get("status") in ("hit", "partial", "unreachable",
                                          "uncovered", "not_applicable"), branch
            assert str(note.get("evidence") or "").strip(), branch
        used: set[str] = set()
        executions = 0
        for case in cases:
            case_id = case["case_id"]
            assert case.get("scenario") in wb.SCENARIOS, case.get("scenario")
            branches = set(case.get("source_branches") or [])
            assert branches, case_id
            assert branches <= registry, (case_id, sorted(branches - registry))
            used |= branches
            modes = fixtures.modes_for(case)
            variants = fixtures.variants_for(case)
            executions += len(modes) * len(variants)
            expected_error = str(case.get("expected_error") or "")
            if case.get("variants"):
                assert set(variants) == set(case.get("variant_expected") or {}), case_id
                assert set(variants) == set(case.get("variant_branches") or {}), case_id
                union: set[str] = set()
                for variant in variants:
                    want = _wb_expected_plan(case, variant)
                    assert want, (case_id, variant)
                    variant_branches = set((case.get("variant_branches") or {})[variant])
                    assert variant_branches <= branches, (case_id, variant)
                    union |= variant_branches
                    problems = _branch_plan_problems(case_id, variant, want,
                                                     variant_branches, modes)
                    assert not problems, problems
                    counts = _wb_counts(case, variant)
                    assert counts["iterations"] == counts["body_hits"], case_id
                    assert counts["cond_hits"] == counts["iterations"] + 1, case_id
                    assert counts["limit_flag"] == 0, case_id
                assert union == branches, (case_id, sorted(branches - union))
            else:
                want = _wb_expected_plan(case, "")
                assert want, case_id
                problems = _branch_plan_problems(case_id, "", want, branches, modes)
                assert not problems, problems
                if expected_error:
                    assert case.get("expected_error_contains"), case_id
                else:
                    counts = _wb_counts(case, "")
                    assert counts["iterations"] == case.get("expected_iterations"), case_id
                    assert counts["cond_hits"] == counts["iterations"] + 1, case_id
            for variant in variants:
                assert _wb_tree(case, variant), case_id
                for path, info in _wb_bytes(case, variant).items():
                    assert info["bytes"] == info["elements"] * info["dtype_size"], \
                        (case_id, variant, path)
                    assert info["bytes_mod_32"] == info["bytes"] % 32, \
                        (case_id, variant, path)
            assert (case.get("tangent_tree") is not None) == \
                any(mode.startswith("jvp") for mode in modes), case_id
        assert executions == EXPECTED_WHITEBOX_MODES, executions
        suite_problems = _branch_suite_problems(cases)
        assert not suite_problems, suite_problems
        assert suite.get("counts", {}).get("identities") == len(cases)
        assert suite.get("counts", {}).get("mode_executions") == executions
        unclassified = registry - used - set(notes)
        assert not unclassified, f"分支未分类（既无白盒用例也无 note）：{sorted(unclassified)}"
        by_mode = {}
        for item in _items_for(suite):
            by_mode[item["display_mode"]] = by_mode.get(item["display_mode"], 0) + 1
        status = {}
        for note in notes.values():
            status[note["status"]] = status.get(note["status"], 0) + 1
        return (f"白盒 manifest {len(cases)} 条 / {executions} 次 by_mode={by_mode}；"
                f"分支 {len(registry)} 个（用例覆盖 {len(used)}，note {status}）")
    _selfcheck("whitebox_manifest", check)


def test_selfcheck_whitebox_cpu_plans():
    """CPU 自检：每个白盒 scenario 都能线性化 + CPU golden 迭代次数与 manifest 一致。"""
    def check():
        suite = _load_whitebox()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cases": [{key: value for key, value in case.items() if key != "_origin"}
                      for case in suite["cases"]],
            "out": str(WORK_DIR / "whitebox_cpu_check.json"),
        }
        payload_path = WORK_DIR / "whitebox_cpu_check_payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_ENABLE_X64"] = "0"
        env.pop("XLA_FLAGS", None)
        env.pop("JAX_COMPILATION_CACHE_DIR", None)
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()),
               "--whitebox-cpu-check", str(payload_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        assert proc.returncode == 0, f"CPU 自检子进程 rc={proc.returncode}: {proc.stderr[-800:]}"
        out = Path(payload["out"])
        assert out.is_file(), proc.stderr[-800:]
        report = json.loads(out.read_text(encoding="utf-8"))
        bad = [item for item in report.get("records") or [] if not item.get("ok")]
        assert not bad, bad[:5]
        return (f"{len(report['records'])} 个 scenario/variant 计划可线性化且 CPU 迭代次数一致；"
                f"backend={report.get('backend')} jax={report.get('jax')}")
    _selfcheck("whitebox_cpu_plans", check)


def _whitebox_cpu_check_main(payload: dict) -> int:
    """CPU 子进程：验证白盒 scenario 的 jaxpr 可线性化、CPU 迭代次数与 manifest 一致。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    module = _load_candidate()
    records = []
    for case in payload["cases"]:
        modes = fixtures.modes_for(case)
        variants = fixtures.variants_for(case)
        expected_error = str(case.get("expected_error") or "")
        for variant in variants:
            counts = {} if expected_error else _wb_counts(case, variant)
            record = {"case_id": case["case_id"], "variant": variant,
                      "scenario": case["scenario"], "ok": False, "detail": ""}
            try:
                built = wb.build(case["scenario"], jax, jnp, variant)
                init = jax.tree_util.tree_map(jnp.asarray, built["init"])
                cond, body = built["cond"], built["body"]
                plan, _ = module._derive_plan(cond, body, init)
                tree_leaves = len(fixtures.leaf_paths(_wb_tree(case, variant)))
                assert len(plan["leaves"]) == tree_leaves, \
                    f"plan leaves={len(plan['leaves'])} != tree {tree_leaves}"
                plan_problems = _plan_problems(plan, _wb_expected_plan(case, variant))
                assert not plan_problems, f"plan 与 manifest 不一致：{plan_problems}"
                if expected_error:
                    record.update(plan_id=module._plan_id(plan),
                                  plan_leaves=len(plan["leaves"]), ok=True,
                                  detail=f"plan 与 expected_plan 一致；预期错误 {expected_error}")
                    records.append(record)
                    continue
                carry = init
                n = 0
                while n < 1000 and bool(cond(carry)):
                    carry = body(carry)
                    n += 1
                assert n == counts["iterations"], f"iterations={n} != {counts['iterations']}"
                record.update(plan_id=module._plan_id(plan), iterations=n,
                              plan_leaves=len(plan["leaves"]), ok=True,
                              detail="plan 与 expected_plan 一致 + CPU 迭代一致")
                for mode in modes:
                    if not mode.startswith("jvp"):
                        continue
                    tangent = jax.tree_util.tree_map(jnp.asarray, built["tangent"])
                    _, tang = jax.jvp(
                        lambda c: jax.lax.while_loop(cond, body, c), (init,), (tangent,))
                    tan_leaves = jax.tree_util.tree_leaves(tang)
                    expected = jax.tree_util.tree_leaves(built["tangent"])
                    for got, want in zip(tan_leaves, expected):
                        assert np.asarray(got).dtype == np.asarray(want).dtype, \
                            f"{mode} tangent dtype {np.asarray(got).dtype} != " \
                            f"{np.asarray(want).dtype}"
                    record[f"{mode}_ok"] = True
            except Exception as exc:  # noqa: BLE001
                record["detail"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    report = {
        "backend": jax.default_backend(),
        "jax": getattr(jax, "__version__", None),
        "jaxlib": getattr(__import__("jaxlib"), "__version__", None),
        "records": records,
    }
    Path(payload["out"]).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"WHILE_CPU_CHECK records={len(records)} ok={sum(1 for r in records if r['ok'])}")
    return 0


def _whitebox_negative_main(payload: dict) -> int:
    """CPU 子进程：白盒负向分支（错误归一化 / 不支持计划 / 校验顺序），不触设备。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    module = _load_candidate()
    records: list[dict] = []

    def expect(name: str, code: str, fn):
        record = {"name": name, "expected": code, "ok": False, "detail": ""}
        try:
            fn()
            record["detail"] = "未抛异常"
        except module.WhileLoopError as exc:
            record.update(code=exc.code, upstream_type=exc.upstream_exception_type,
                          upstream_message=exc.upstream_message,
                          ok=exc.code == code,
                          detail=f"code={exc.code}")
        except BaseException as exc:  # noqa: BLE001
            record["detail"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    int_carry = (jnp.int32(0),)
    float_carry = (np.float32(1.0), np.float32(2.0))
    cond_lt = lambda c: c[0] < 2
    body_inc = lambda c: (c[0] + 1,)

    expect("mode_vjp", "invalid_argument",
           lambda: module.while_loop_compiled(cond_lt, body_inc, int_carry, mode="vjp"))
    expect("safety_limit_zero", "invalid_argument",
           lambda: module.while_loop(cond_lt, body_inc, int_carry, safety_limit=0))
    expect("safety_limit_str", "invalid_argument",
           lambda: module.while_loop(cond_lt, body_inc, int_carry, safety_limit="8"))
    expect("jvp_without_tangent", "invalid_argument",
           lambda: module.while_loop_compiled(cond_lt, body_inc, int_carry, mode="jvp"))
    expect("primal_with_tangent", "invalid_argument",
           lambda: module.while_loop(cond_lt, body_inc, int_carry, tangent=int_carry))
    expect("unsupported_primitive", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: c[0] < 2,
                                     lambda c: (c[0] + 1, jnp.sin(c[1])),
                                     (jnp.int32(0), jnp.float32(1.0))))
    expect("nonlinear_mul", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: c[0] < 2,
                                     lambda c: (c[0] * c[1], c[1]), float_carry))
    expect("div_by_nonconst", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: c[0] < 2,
                                     lambda c: (c[0], c[1] / c[0]), float_carry))
    expect("always_false_cond", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: jnp.bool_(False), body_inc, int_carry))
    expect("all_tensor_vs_tensor", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: jnp.all(c[0] > c[1]),
                                     lambda c: (c[0] * 0.5, c[1]), float_carry))
    expect("same_leaf_reduce_mix", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: c[0] < 2,
                                     lambda c: (c[0] * 0.5 + jnp.sum(c[0]),), float_carry[:1]))
    expect("unsupported_cond_primitive", "UNSUPPORTED_PLAN",
           lambda: module.while_loop(lambda c: jnp.sin(c[0]) > 0.0,
                                     lambda c: (c[0] * 0.5,), float_carry[:1]))
    expect("non_bool_cond", "INVALID_CONDITION",
           lambda: module.while_loop(lambda c: jnp.int32(c[0] > 0),
                                     lambda c: (c[0] * 0.5,), float_carry[:1]))
    # body 在 eval_shape 阶段抛非 JAX 异常 → _classify 无法按消息归类时的兜底
    expect("unclassifiable_body", "INVALID_CONDITION",
           lambda: module.while_loop(lambda c: c[0] < 2,
                                     lambda c: (_ for _ in ()).throw(ZeroDivisionError("boom")),
                                     int_carry))
    expect("non_0d_weak_output", "UNSUPPORTED_WEAK_TYPE",
           lambda: module._make_jax_leaf(
               np.zeros(3, np.float32),
               jax.ShapeDtypeStruct((3,), np.float32, weak_type=True)))
    expect("unsupported_output_dtype", "invalid_argument",
           lambda: module._make_jax_leaf(
               np.zeros(3, np.int32),
               jax.ShapeDtypeStruct((3,), np.int64)))
    # _classify 的 body eval_shape 失败消息归类（结构/形状/dtype/equal-types 四种）
    for name, message, code in (
            ("body_msg_structure", "body must have same pytree structure",
             "CARRY_STRUCTURE_MISMATCH"),
            ("body_msg_shape", "body must have same shape at leaves", "CARRY_SHAPE_MISMATCH"),
            ("body_msg_dtype", "body must have same dtype at leaves", "CARRY_DTYPE_MISMATCH"),
            ("body_msg_equal_types_shape", "body must have equal types [int32]", "CARRY_SHAPE_MISMATCH"),
            ("body_msg_equal_types_dtype", "body must have equal types", "CARRY_DTYPE_MISMATCH")):
        def make_raiser(text):
            def raiser(c):
                raise TypeError(text)
            return raiser
        expect(name, code,
               lambda r=make_raiser(message): module.while_loop(
                   lambda c: jnp.bool_(True), r, int_carry))
    expect("cond_eval_message", "INVALID_CONDITION",
           lambda: module.while_loop(
               lambda c: (_ for _ in ()).throw(
                   TypeError("cond_fun must return a boolean scalar: boom")),
               body_inc, int_carry))
    # dtype 分派（纯函数）；bool/float64 不在支持范围
    for name, dtype in (("dtype_float64", np.float64), ("dtype_bool", np.bool_)):
        record = {"name": name, "expected": "invalid_argument", "ok": False, "detail": ""}
        try:
            module._dtype_code(np.dtype(dtype))
            record["detail"] = "未抛异常"
        except module.WhileLoopError as exc:
            record.update(code=exc.code, ok=exc.code == "invalid_argument",
                          detail=f"code={exc.code}")
        records.append(record)
    # 设备号来源：环境变量 > 身份文件 > 缺省 0
    record = {"name": "container_device_id", "expected": "3/2/0", "ok": False, "detail": ""}
    try:
        old_env = os.environ.get("OPX_DEVICE_ID")
        old_root = module.OP_ROOT
        os.environ["OPX_DEVICE_ID"] = "3"
        got_env = module._container_device_id()
        os.environ.pop("OPX_DEVICE_ID", None)
        tmp_root = Path(payload["tmp"]) / "op_root"
        identity = tmp_root.parent.parent / ".opx-run-identity.json"
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text(json.dumps({"container_device_id": "2"}), encoding="utf-8")
        module.OP_ROOT = tmp_root
        got_file = module._container_device_id()
        identity.unlink()
        got_missing = module._container_device_id()
        module.OP_ROOT = old_root
        if old_env is not None:
            os.environ["OPX_DEVICE_ID"] = old_env
        record.update(values=[got_env, got_file, got_missing],
                      ok=(got_env, got_file, got_missing) == (3, 2, 0),
                      detail=f"{got_env}/{got_file}/{got_missing}")
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    # 缺失 .so：device_error，且不初始化设备
    record = {"name": "missing_host_so", "expected": "device_error", "ok": False, "detail": ""}
    try:
        old_path, old_lib = module.LIB_PATH, module._LIB
        module.LIB_PATH = Path(payload["tmp"]) / "missing_host.so"
        module._LIB = None
        try:
            module._load_lib()
            record["detail"] = "未抛异常"
        except module.WhileLoopError as exc:
            record.update(code=exc.code, ok=exc.code == "device_error",
                          detail=f"code={exc.code}")
        finally:
            module.LIB_PATH, module._LIB = old_path, old_lib
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    # 共享库 ABI 漂移（P49 的第二侧）：伪造 CDLL，必须 device_error 且不进入设备
    record = {"name": "host_so_abi_mismatch", "expected": "device_error/ABI",
              "ok": False, "detail": ""}
    try:
        fake_path = Path(payload["tmp"]) / "fake_abi.so"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.write_bytes(b"\x7fELF-fake")
        old_path, old_lib = module.LIB_PATH, module._LIB
        old_cdll = module.ctypes.CDLL

        class _FakeFunc:
            def __init__(self, ret=None):
                self.argtypes = None
                self.restype = None
                self._ret = ret

            def __call__(self, *args, **kwargs):
                return self._ret

        class _FakeLib:
            def __getattr__(self, name):
                if name == "jax_lax_while_loop_abi_version":
                    return _FakeFunc(b"wrong-abi/9")
                return _FakeFunc()

        module.LIB_PATH, module._LIB = fake_path, None
        module.ctypes.CDLL = lambda *args, **kwargs: _FakeLib()
        try:
            module._load_lib()
            record["detail"] = "未抛异常"
        except module.WhileLoopError as exc:
            ok = exc.code == "device_error" and "ABI" in str(exc)
            record.update(code=exc.code, ok=ok, detail=f"code={exc.code}")
        finally:
            module.LIB_PATH, module._LIB = old_path, old_lib
            module.ctypes.CDLL = old_cdll
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    # H01 的 9 叶上界：公开 Python 接口在 _fill_args 的 ctypes[8] 越界（已知限制）。
    # 8 叶边界必须可填；9 叶必须留下可执行复现，证明 host H01 的 n_leaves>8 侧
    # 在 Python 接口下不可达，而不是“已覆盖”。
    record = {"name": "h01_nine_leaves_public_api", "expected": "8 ok / 9 IndexError",
              "ok": False, "detail": "", "known_limit": True}
    try:
        def build_plan(count):
            init = tuple(jnp.int32(index) for index in range(count))
            cond = lambda c: c[0] < 2
            body = lambda c: tuple(value + 1 for value in c)
            plan, _ = module._derive_plan(cond, body, init)
            return plan

        plan8 = build_plan(8)
        args8 = module.JaxLaxWhileLoopArgs()
        module._fill_args(args8, plan8, 0, stage=0, is_jvp=False)
        plan9 = build_plan(9)
        args9 = module.JaxLaxWhileLoopArgs()
        try:
            module._fill_args(args9, plan9, 0, stage=0, is_jvp=False)
            record["detail"] = "9 叶未抛异常（应显式拒绝或声明上限）"
        except IndexError as exc:
            record.update(ok=True, n_leaves=9,
                          detail=f"8 叶可填；9 叶 IndexError（已知限制，非 H01 命中）: {exc}")
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    # P35（len(in_leaves)!=len(out_leaves)）不可达证明：结构标签相同 ⇒ treedef 相同
    # ⇒ 叶子数必然相同；结构不同时 P34 先返回。
    record = {"name": "p35_same_structure_same_leaf_count", "expected": "unreachable",
              "ok": False, "detail": ""}
    try:
        structs = module._struct_tree((jnp.int32(0), jnp.int32(1)))
        same = jax.eval_shape(lambda c: (c[0] + 1, c[1] + 1), structs)
        other = jax.eval_shape(lambda c: {"a": c[0], "b": c[1]}, structs)
        in_leaves = jax.tree_util.tree_leaves(structs)
        same_leaves = jax.tree_util.tree_leaves(same)
        other_leaves = jax.tree_util.tree_leaves(other)
        in_label = module._structure_label(jax.tree_util.tree_structure(structs))
        same_label = module._structure_label(jax.tree_util.tree_structure(same))
        other_label = module._structure_label(jax.tree_util.tree_structure(other))
        same_struct_same_count = (in_label == same_label and len(in_leaves) == len(same_leaves))
        diff_struct_same_count = (in_label != other_label
                                  and len(in_leaves) == len(other_leaves))
        record.update(ok=bool(same_struct_same_count and diff_struct_same_count),
                      detail=(f"同结构 {len(in_leaves)}=={len(same_leaves)} 叶；"
                              f"异结构 {len(in_leaves)}=={len(other_leaves)} 叶但 P34 先返回"))
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    # CLI C03（其它 WhileLoopError → rc=1）与 C04（未处理异常 → rc=1）：
    # 用真实 CLI main 的可注入 case/cond_body 走错误出口，不触设备。
    try:
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415
        import jax_lax_while_loop_cli as cli  # noqa: PLC0415

        cli_dir = Path(payload["tmp"]) / "cli_branches"
        cli_dir.mkdir(parents=True, exist_ok=True)
        case_path = cli_dir / "case.json"
        case_path.write_text(json.dumps({
            "case_id": "while_scalar_carry",
            "tree": {"kind": "leaf", "path": "carry", "shape": [], "dtype": "int32"},
        }), encoding="utf-8")
        inputs_path = cli_dir / "inputs.npz"
        np.savez(inputs_path, carry=np.int32(0))

        record = {"name": "cli_other_whileloop_error", "expected": "rc=1/UNSUPPORTED_PLAN",
                  "ok": False, "detail": ""}
        original = cli._cond_body
        cli._cond_body = lambda case, variant: (lambda c: c < 100, lambda c: c * c)
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["--case-json", str(case_path), "--inputs", str(inputs_path),
                               "--mode", "eager", "--outdir", str(cli_dir / "c03")])
            text = err.getvalue()
            record.update(ok=(rc == 1 and "code=UNSUPPORTED_PLAN" in text),
                          rc=rc, detail=text.strip().splitlines()[-1] if text.strip() else "")
        finally:
            cli._cond_body = original
        records.append(record)

        record = {"name": "cli_unhandled_exception", "expected": "rc=1/RuntimeError",
                  "ok": False, "detail": ""}
        original_loop = cli.iface.while_loop

        def _boom(*args, **kwargs):
            raise RuntimeError("injected non-WhileLoopError")

        cli.iface.while_loop = _boom
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["--case-json", str(case_path), "--inputs", str(inputs_path),
                               "--mode", "eager", "--outdir", str(cli_dir / "c04")])
            text = err.getvalue()
            record.update(ok=(rc == 1 and "ERROR unhandled RuntimeError" in text),
                          rc=rc, detail=text.strip().splitlines()[-1] if text.strip() else "")
        finally:
            cli.iface.while_loop = original_loop
        records.append(record)
    except BaseException as exc:  # noqa: BLE001
        records.append({"name": "cli_error_paths", "expected": "C03/C04",
                        "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    # 纯工具：容器指纹区分 tuple/list、leaf_meta 归一化 float0、plan_id 稳定
    record = {"name": "pytree_helpers", "expected": "tree/float0/plan_id", "ok": False,
              "detail": ""}
    try:
        tree_tuple = module.tree_signature((np.zeros(1, np.float32),))
        tree_list = module.tree_signature([np.zeros(1, np.float32)])
        float0 = module.leaf_meta(np.zeros((), dtype=np.dtype([("float0", "V")])))
        plan_a, _ = module._derive_plan(cond_lt, body_inc, int_carry)
        plan_b, _ = module._derive_plan(cond_lt, body_inc, int_carry)
        ok = (tree_tuple != tree_list
              and float0["dtype"] == "[('float0', 'V')]" and float0["shape"] == []
              and module._plan_id(plan_a) == module._plan_id(plan_b)
              and len(module._plan_id(plan_a)) == 16)
        record.update(ok=bool(ok), detail=f"float0={float0['dtype']} plan_id="
                                          f"{module._plan_id(plan_a)}")
    except BaseException as exc:  # noqa: BLE001
        record["detail"] = f"{type(exc).__name__}: {exc}"
    records.append(record)
    report = {"backend": jax.default_backend(),
              "jax": getattr(jax, "__version__", None),
              "jaxlib": getattr(__import__("jaxlib"), "__version__", None),
              "records": records}
    Path(payload["out"]).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"WHILE_NEGATIVE records={len(records)} "
          f"ok={sum(1 for r in records if r['ok'])}")
    return 0


def test_selfcheck_whitebox_negative_paths():
    def check():
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        out = WORK_DIR / "whitebox_negative.json"
        payload = {"out": str(out), "tmp": str(WORK_DIR / "whitebox_negative_tmp")}
        payload_path = WORK_DIR / "whitebox_negative_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_ENABLE_X64"] = "0"
        env.pop("XLA_FLAGS", None)
        env.pop("JAX_COMPILATION_CACHE_DIR", None)
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()),
               "--whitebox-negative", str(payload_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        assert proc.returncode == 0, f"负向自检 rc={proc.returncode}: {proc.stderr[-800:]}"
        report = json.loads(out.read_text(encoding="utf-8"))
        bad = [item for item in report["records"] if not item.get("ok")]
        assert not bad, bad
        codes = sorted({item.get("code") for item in report["records"] if item.get("code")})
        return f"{len(report['records'])} 个负向分支均按预期归一化：{codes}"
    _selfcheck("whitebox_negative_paths", check)


def test_selfcheck_whitebox_judge_negative():
    """白盒判据负向：weak_type/树/数值翻转必须 FAIL（防止判据恒过）。"""
    def check():
        case = next(c for c in _load_whitebox()["cases"]
                    if c["case_id"] == "while_wb_reduce_only_scalar")
        ratio_dir = WORK_DIR / "whitebox_judge_negative"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        # golden 必须走 CPU 子进程：父进程不 import jax（NPU 主机上会初始化设备后端）
        golden = _run_whitebox_golden({
            "case": {k: v for k, v in case.items() if k != "_origin"},
            "variant": "", "mode": "eager", "outdir": str(ratio_dir)})
        assert golden["ok"], golden.get("meta")
        meta = golden["meta"]
        golden_npz = Path(golden["npz"])
        criteria = engine._load_module(WS / "criteria" / "check.py", "opx_criteria_check")
        _, problems = _judge_whitebox(
            case=case, variant="", mode="eager",
            candidate_npz=golden_npz, golden_npz=golden_npz,
            candidate_meta=json.loads(json.dumps(meta)), golden_meta=meta, criteria=criteria)
        assert not problems, problems
        # 1) tree 翻转
        bad_tree = json.loads(json.dumps(meta))
        bad_tree["tree"][0] = "list"
        _, tree_problems = _judge_whitebox(
            case=case, variant="", mode="eager",
            candidate_npz=golden_npz, golden_npz=golden_npz,
            candidate_meta=bad_tree, golden_meta=meta, criteria=criteria)
        assert tree_problems and any("tree" in item for item in tree_problems), tree_problems
        # 2) weak_type 翻转（至少一个叶子可与 golden 比较）
        bad_weak = json.loads(json.dumps(meta))
        key = next(iter(bad_weak["leaves"]))
        if bad_weak["leaves"][key].get("weak_type") is None:
            bad_weak["leaves"][key]["weak_type"] = False
        bad_weak["leaves"][key]["weak_type"] = not bool(
            bad_weak["leaves"][key]["weak_type"])
        _, weak_problems = _judge_whitebox(
            case=case, variant="", mode="eager",
            candidate_npz=golden_npz, golden_npz=golden_npz,
            candidate_meta=bad_weak, golden_meta=meta, criteria=criteria)
        # 若该叶 golden weak_type=None（numpy），判据按 n/a；这里场景是 jax.Array，必须翻转成功
        assert weak_problems and any("weak_type" in item for item in weak_problems), weak_problems
        # 3) 数值翻转
        bad_npz = ratio_dir / "bad.npz"
        arrays = dict(np.load(golden_npz))
        arrays[key] = arrays[key] + np.float32(10.0)
        np.savez(bad_npz, **arrays)
        _, value_problems = _judge_whitebox(
            case=case, variant="", mode="eager",
            candidate_npz=bad_npz, golden_npz=golden_npz,
            candidate_meta=json.loads(json.dumps(meta)), golden_meta=meta, criteria=criteria)
        assert value_problems, value_problems
        return "白盒判据：正确输出 PASS；tree/weak_type/数值翻转均 FAIL"
    _selfcheck("whitebox_judge_negative", check)


def test_selfcheck_shared_runner_dispatch():
    def check():
        sys.path.insert(0, str(WS / "tools"))
        import run_cases  # noqa: PLC0415
        import smoke_case  # noqa: PLC0415
        import run_benchmark  # noqa: PLC0415

        assert hasattr(run_cases, "_load_while_engine")
        assert hasattr(smoke_case, "main")
        assert hasattr(run_benchmark, "main")
        source = Path(run_cases.__file__).read_text(encoding="utf-8")
        assert 'op == "jax_lax_while_loop"' in source
        assert "jax_lax_scan" in source and "jax_lax_cond" in source
        engine_mod = run_cases._load_while_engine(WS)
        assert engine_mod.WHILE_OP == OP
        return "run_cases/smoke_case/run_benchmark 显式分派；旧 op 分支保留"
    _selfcheck("shared_runner_dispatch", check)


def _write_session_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {status: sum(1 for rec in _RESULTS if rec["status"] == status)
               for status in engine.CASE_STATUSES}
    summary["TOTAL"] = len(_RESULTS)
    report = {
        "schema": "opx-jax-while-loop-correctness/1",
        "op": OP,
        "action": "pytest",
        "run_id": os.environ.get("OPX_RUN_ID"),
        "spec_version_id": int(os.environ.get("OPX_SPEC_VERSION_ID") or 7),
        "criteria_ref": os.environ.get("OPX_CRITERIA_REF") or "ops-precision-standard@1",
        "code_snapshot_sha256": os.environ.get("OPX_CODE_SNAPSHOT_SHA256") or "",
        "suite_ref": os.environ.get("OPX_SUITE_REF") or "cases@288876910a5e2843",
        "candidate": {
            "python_interface": str(CANDIDATE_PATH.relative_to(WS)),
            "exists": CANDIDATE_PATH.is_file(),
            "sha256": hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
            if CANDIDATE_PATH.is_file() else None,
        },
        "environment": {
            **{key: value for key, value in engine.observed_identity(WS).items()
               if key != "source"},
            "identity_source": engine.observed_identity(WS).get("source"),
            "expected_identity": engine.expected_identity(),
        },
        "expected": {
            "identities": EXPECTED_BLACKBOX_CASES,
            "mode_executions": EXPECTED_BLACKBOX_MODES,
            "whitebox_cases": EXPECTED_WHITEBOX_CASES,
            "whitebox_modes": EXPECTED_WHITEBOX_MODES,
            "pytest_cases": EXPECTED_BLACKBOX_CASES + EXPECTED_WHITEBOX_CASES,
            "pytest_modes": EXPECTED_BLACKBOX_MODES + EXPECTED_WHITEBOX_MODES,
            "selfchecks": EXPECTED_SELFCHECKS,
        },
        "cases": _RESULTS,
        "summary": summary,
        "by_mode": engine._by_mode(_RESULTS),
        "by_suite": {
            "blackbox": sum(1 for rec in _RESULTS if rec.get("origin") != "whitebox"),
            "whitebox": sum(1 for rec in _RESULTS if rec.get("origin") == "whitebox"),
        },
        "authenticity": engine._authenticity(_RESULTS),
        "cross_checks": engine._cross_checks(_RESULTS),
        "selfchecks": _SELFCHECK_RECORDS,
        "execution": {
            "isolation": "fresh_worker_per_case",
            "worker_results": sum(1 for rec in _RESULTS if (rec.get("worker") or {}).get("pid")),
            "device_gated": sum(1 for rec in _RESULTS if rec.get("device_evidence")),
            "device_evidence_ok": sum(1 for rec in _RESULTS
                                      if (rec.get("device_evidence") or {}).get("ok")),
        },
    }
    engine.write_report(out=RESULT_JSON, md=RESULT_MD, report=report)


@pytest.fixture(scope="session", autouse=True)
def _write_report_after_session():
    """pytest 的 session 级 teardown 写报告；测试模块里的 pytest_sessionfinish 不会被注册。"""
    yield
    try:
        _write_session_report()
    except Exception as exc:  # noqa: BLE001 —— 报告失败必须让交付可见
        print(f"WHILE_REPORT_WRITE_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# worker 入口
# ---------------------------------------------------------------------------


def _worker_entry(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: test_jax_lax_while_loop.py "
              "<--worker|--whitebox-worker|--whitebox-golden|--whitebox-cpu-check> "
              "<payload.json>", file=sys.stderr)
        return 2
    mode, payload_path = argv
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if mode == "--worker":
        return _worker_main(payload)
    if mode == "--whitebox-worker":
        return _worker_main_whitebox(payload)
    if mode == "--whitebox-golden":
        return _golden_main_whitebox(payload)
    if mode == "--whitebox-cpu-check":
        return _whitebox_cpu_check_main(payload)
    if mode == "--whitebox-negative":
        return _whitebox_negative_main(payload)
    print(f"未知入口：{mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_worker_entry(sys.argv[1:]))
