"""测试管理（P3 读侧 / 方案 §7「测试管理」）：用例矩阵 + 逐例历史 + 失败点。

数据真相源两处：
- **用例矩阵** = 工作区 `cases/<op>/cases.json`（这次跑的就是它；也可退回最近一次 test 报告里的
  `cases/` 快照）。
- **逐例结果** = 每次 `test` run 的 `correctness.json`（机读报告，`gate.artifact_paths` 解出路径）。

本模块只读、聚合，不改任何东西；判据/门禁口径仍归 `gate.py`。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Run, Task
from . import gate


def _load_json(path: Path) -> dict | None:
    """只接受顶层 JSON 对象；数组/标量/坏文件一律返回 None（页面按「没有」处理，不 500）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _safe_child(base: Path, *parts: str) -> Path | None:
    """`base/<parts...>` 且每段都是单段名（拒绝 `..`/`/`/`\\`/空）；越界 → None。"""
    for part in parts:
        if (not isinstance(part, str) or not part or part in (".", "..")
                or "/" in part or "\\" in part
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in part)):
            return None
    root = base.resolve()
    p = base.joinpath(*parts).resolve()
    if p != root and root not in p.parents:
        return None
    return p


def _norm_from_matrix(op_cases: dict) -> list[dict]:
    """工作区用例矩阵 → 统一展示结构。"""
    out = []
    if not isinstance(op_cases, dict):
        return out
    for c in _as_list(op_cases.get("cases")):
        if not isinstance(c, dict) or not c.get("case_id"):
            continue
        out.append({
            "case_id": c.get("case_id"),
            "group": c.get("group"),
            "required": bool(c.get("required")),
            "shape": [c.get("n_a")] if c.get("n_a") is not None else None,
            "shape_b": [c.get("n_b")] if c.get("n_b") is not None else None,
            "dtype": c.get("dtype"),
            "seed": c.get("seed"),
            "expected": c.get("expected"),
            "note": c.get("note"),
        })
    return out


def _norm_from_report(cases: list) -> list[dict]:
    """correctness.json 的用例（含结果）→ 展示结构（只取矩阵字段）。"""
    out = []
    for c in _as_list(cases):
        if not isinstance(c, dict) or not c.get("case_id"):
            continue
        out.append({
            "case_id": c.get("case_id"),
            "group": c.get("group"),
            "required": bool(c.get("required")),
            "shape": c.get("shape"),
            "shape_b": c.get("shape_b"),
            "dtype": c.get("dtype"),
            "seed": c.get("seed"),
            "expected": c.get("expected"),
            "note": c.get("note"),
        })
    return out


def _run_view(run: Run, report: dict | None, report_dir: Path | None) -> dict:
    """一次 test run 的概览（含可下载的产物 id）。"""
    dl = []
    for a in run.artifacts:
        if a.type in ("correctness", "report") and str(a.path).endswith((".json", ".md")):
            name = Path(a.path).name
            if a.type == "correctness" or name.startswith("correctness"):
                dl.append({"id": a.id, "type": a.type, "name": name})
    return {
        "run_id": run.id,
        "status": run.status,
        "loop_id": run.loop_id,
        "suite_ref": run.suite_ref,
        "criteria_ref": run.criteria_ref,
        "spec_version_id": run.spec_version_id,
        "code_snapshot_sha256": run.code_snapshot_sha256,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "exit_code": run.exit_code,
        "report_available": report is not None,
        "summary": (report or {}).get("summary"),
        "case_source": (report or {}).get("case_source"),
        "case_source_sha256": (report or {}).get("case_source_sha256"),
        "downloads": dl,
    }


def _norm_perf_matrix(bench: dict | None) -> list[dict]:
    """benchmark.json → 性能用例矩阵（任务生命周期 v2：测试管理页要并列显示精度/性能两类）。"""
    out = []
    if not isinstance(bench, dict):
        return out
    primary = bench.get("primary_case")
    for c in _as_list(bench.get("cases")):
        if not isinstance(c, dict) or not c.get("case_id"):
            continue
        out.append({
            "case_id": c.get("case_id"),
            "n": c.get("n"),
            "dtype": c.get("dtype"),
            "required": bool(c.get("required")),
            "primary": c.get("case_id") == primary,
            "note": c.get("note"),
        })
    return out


def _perf_latest(report: dict | None) -> dict[str, dict]:
    """performance.json → {case_id: 最近一次逐用例结果}。"""
    out: dict[str, dict] = {}
    for c in _as_list((report or {}).get("cases")):
        if not isinstance(c, dict) or not c.get("case_id"):
            continue
        cand = c.get("candidate") or {}
        base = c.get("baseline") or {}
        derived = c.get("derived") or {}
        cand_us = (cand.get("device_us") or {}).get("median_us")
        base_us = (base.get("device_us") or {}).get("median_us")
        out[c["case_id"]] = {
            "candidate_median_us": cand_us,
            "baseline_median_us": base_us,
            "speedup": derived.get("speedup_baseline_over_candidate"),
            "candidate_bw_gbps": derived.get("candidate_achieved_bw_gbps"),
            "baseline_bw_gbps": derived.get("baseline_achieved_bw_gbps"),
            "candidate_std_us": (cand.get("device_us") or {}).get("std_us"),
            "baseline_std_us": (base.get("device_us") or {}).get("std_us"),
            "n": c.get("n"),
            "dtype": c.get("dtype"),
            "required": bool(c.get("required")),
            "primary": bool(c.get("primary")),
        }
    return out


def _performance(db_runs: list[Run], workspace: Path | None, op: str) -> dict:
    """性能用例段：矩阵来自 benchmark.json，实测来自各次 benchmark run 的 performance.json。

    只读；口径与 `perf_diff` 一致（设备侧中位耗时 + 基线对照）。页面按「只读表格」展示。
    """
    bench: dict | None = None
    source = None
    if workspace and op:
        main = _safe_child(workspace, "cases", op, "benchmark.json")
        bench = _load_json(main) if main is not None else None
        if bench:
            source = f"cases/{op}/benchmark.json"
    cases = _norm_perf_matrix(bench)
    reports: dict[int, dict] = {}
    for r in db_runs:
        paths = gate.artifact_paths(r)
        art = paths.get("performance")
        if art and art.suffix == ".json":
            data = _load_json(art)
            if data is not None:
                reports[r.id] = data
    latest_id = max(reports) if reports else None
    latest = _perf_latest(reports.get(latest_id)) if latest_id is not None else {}
    views = []
    for m in cases:
        hit = latest.get(m["case_id"]) or {}
        views.append({**m, "latest": ({**hit, "run_id": latest_id} if hit else None)})
    runs = []
    for r in db_runs:
        try:
            verdict = gate.evaluate(r).get("verdict")
        except Exception:  # noqa: BLE001 —— 门禁判不了按「未定」展示
            verdict = None
        runs.append({
            "run_id": r.id,
            "status": r.status,
            "gate_verdict": verdict,
            "suite_ref": r.suite_ref,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "primary_case": (reports.get(r.id) or {}).get("summary", {}).get("primary_case")
            if isinstance((reports.get(r.id) or {}).get("summary"), dict) else None,
        })
    return {
        "source": source,
        "primary_case": (bench or {}).get("primary_case") if isinstance(bench, dict) else None,
        "cases": views,
        "runs": runs,
        "latest_run_id": latest_id,
    }


def build(task: Task, test_runs: list[Run], workspace: Path | None,
          criteria_ref: str | None = None, benchmark_runs: list[Run] | None = None) -> dict:
    """聚合一个任务的用例矩阵与逐例结果。`test_runs` 应按 id 升序（旧→新）。"""
    op = task.op_name or ""

    # 逐 run 读机读报告
    reports: dict[int, dict] = {}
    for r in test_runs:
        paths = gate.artifact_paths(r)
        art = paths.get("correctness")
        if art and art.suffix == ".json":
            data = _load_json(art)
            if data is not None:
                reports[r.id] = data

    # 用例矩阵：工作区 cases/<op>/cases.json 优先，退回最近一份报告
    source = ""
    case_source_sha256 = None
    matrix_raw: list[dict] = []
    if workspace and op:
        # op 必须单段且路径落在工作区内：非法/越界 → 空矩阵，绝不读库外文件
        main = _safe_child(workspace, "cases", op, "cases.json")
        data = _load_json(main) if main is not None else None
        if data:
            matrix_raw = _norm_from_matrix(data)
            source = f"cases/{op}/cases.json"
    if not matrix_raw and reports:
        last = reports[max(reports)]
        matrix_raw = _norm_from_report(last.get("cases"))
        source = last.get("case_source") or "（来自最近一次 test 报告）"
    # 报告里的 case_source 指纹更精确，优先展示它
    if reports:
        last = reports[max(reports)]
        case_source_sha256 = last.get("case_source_sha256")

    # 逐例 × 逐 run 状态矩阵 + 失败点
    by_case: dict[str, dict[int, dict]] = {}
    failures: list[dict] = []
    for rid in sorted(reports):
        for c in _as_list(reports[rid].get("cases")):
            if not isinstance(c, dict):
                continue
            cid = c.get("case_id")
            if not cid:
                continue
            by_case.setdefault(cid, {})[rid] = c
            if c.get("status") != "PASS":
                failures.append({
                    "case_id": cid,
                    "run_id": rid,
                    "status": c.get("status"),
                    "expected": c.get("expected"),
                    "max_abs_err": c.get("max_abs_err"),
                    "matched_ratio": c.get("matched_ratio"),
                    "note": c.get("note"),
                    "detail": c.get("detail"),
                })

    cases = []
    for m in matrix_raw:
        cid = m["case_id"]
        results = by_case.get(cid, {})
        latest = results[max(results)] if results else None
        cases.append({
            **m,
            "results": {str(rid): results[rid].get("status") for rid in sorted(results)},
            "latest_status": (latest or {}).get("status"),
            "latest_run_id": max(results) if results else None,
            "max_abs_err": (latest or {}).get("max_abs_err"),
            "matched_ratio": (latest or {}).get("matched_ratio"),
        })

    runs = [_run_view(r, reports.get(r.id) or None, None) for r in test_runs]
    return {
        "task_id": task.id,
        "op_name": op,
        "case_source": source,
        "case_source_sha256": case_source_sha256,
        "criteria_ref": criteria_ref or task.criteria_ref,
        "case_groups": sorted({c["group"] for c in cases if c.get("group")}),
        "total_cases": len(cases),
        "required_cases": sum(1 for c in cases if c.get("required")),
        "cases": cases,
        "runs": runs,
        "failures": failures,
        # 任务生命周期 v2：测试管理页并列显示「精度 / 性能」两类用例（性能段只读）
        "performance": _performance(list(benchmark_runs or []), workspace, op),
        "cases_confirmed_at": task.cases_confirmed_at.isoformat() if task.cases_confirmed_at else None,
        "cases_confirmed_sha256": task.cases_confirmed_sha256,
        "notes": [
            "用例矩阵来自工作区 cases/<op>/cases.json；逐例结果来自各次 test run 的 correctness.json。",
            "状态 PASS = 达到判据档（含 expected=reject 的用例「按预期拒绝」）。",
            "性能用例矩阵来自 cases/<op>/benchmark.json；实测来自各次 benchmark run 的 performance.json。",
        ],
    }
