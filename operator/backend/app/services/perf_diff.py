"""自动优化闭环的对比报告与门禁（方案 v1.1 §13-A5 / §4.6；需求 C6）。

一个 `loops.kind == "perf"` 的优化 Loop 把「优化前 / 优化后」两组真实执行串起来：
每一侧至少要有一次 `test`（正确性）与一次 `benchmark`（性能）。本模块：

1. **配对**：按 `runs.code_snapshot_sha256` 分成两侧（出现早的 = 优化前，晚的 = 优化后）；
2. **可比性检查**：规格版本 / 判据档 / 用例集 ref / 镜像 digest / 设备必须一致，否则结论不成立；
3. **逐用例对比**：候选设备侧耗时（benchmark 的主值）的前后差 + 测量不确定性 → 单例结论；
4. **精度不退化**：optimized 侧的正确性报告不得出现 FAIL/ERROR，且逐用例状态不得比 before 差；
5. **总结论** `improved` / `no_significant_change` / `inconclusive` / `regressed`；
6. 产出 `performance-diff.json`（机读）。

**口径（需求调整已定）**：自动优化 = 闭环 + 对比报告，**不设加速比硬线** —— 性能回退只给 warn，
不判 fail；但**精度退化是硬拦**（`E_PRECISION_REGRESSION`）。

**目标用例口径**（`loop.objective`，建轮次页写入）：

- `type in {perf, hardware}` + `target_dimension=cases`：以 `target_cases`（benchmark case_id）驱动；
- `type in {perf, hardware}` + `target_dimension=all`：对**全部可配对用例**保守聚合（任一 regressed
  即 regressed），与 `optimization._perf_summary` 的报告层结论复用同一聚合函数；
- `type=precision`（`target_cases` 是 correctness/test 用例 id，不是 benchmark case_id）：benchmark
  侧回到旧 Loop 口径（benchmark primary），目标用例匹配由 `optimization.report` 的 precision 段负责；
- 没有 objective 的旧 Loop：同样按 benchmark primary（兼容口径，不改历史结论）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..models import Loop, Run
from . import gate

#: 允许的总结论（方案 §13-A5）
CONCLUSIONS = ("improved", "no_significant_change", "inconclusive", "regressed")

#: 逐用例状态里「算过」的
_PASSING = "PASS"

#: 判定「有显著差异」的最小效应量（相对），滤掉极小但稳定地越过的噪声
MIN_EFFECT = 0.02

#: benchmark 侧目标用例门禁只对「性能 / 硬件」目标生效。precision 轮次的 `target_cases` 是
#: correctness/test 用例 id（见 `optimization._normalize_objective` 按 test 池校验），不能拿去和
#: benchmark 的 case_id 配对 —— 否则每个精度轮都会 E_TARGET_CASE_MISSING / 门禁 fail（H2）。
_BENCH_OBJECTIVE_TYPES = ("perf", "hardware")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _run_dir(run: Run) -> Path | None:
    if not run.manifest_path:
        return None
    return Path(str(run.manifest_path)).expanduser().parent


def _artifact_paths(run: Run) -> dict[str, Path]:
    """类型 → 文件路径（与门禁同一口径，见 `gate.artifact_paths`）。"""
    return gate.artifact_paths(run)


def _read_typed(run: Run, *types: str) -> dict | None:
    paths = _artifact_paths(run)
    for t in types:
        p = paths.get(t)
        if p and p.suffix == ".json" and p.is_file():
            data = _load_json(p)
            if data is not None:
                return data
    return None


class _Report:
    def __init__(self, loop: Loop) -> None:
        self.loop = loop
        self.findings: list[dict] = []

    def fail(self, code: str, message: str, **ev) -> None:
        self.findings.append({"code": code, "level": "fail", "message": message, "evidence": ev})

    def warn(self, code: str, message: str, **ev) -> None:
        self.findings.append({"code": code, "level": "warn", "message": message, "evidence": ev})

    @property
    def failed(self) -> list[dict]:
        return [f for f in self.findings if f["level"] == "fail"]


# ---------------------------------------------------------------- 配对


def _side_runs(runs: list[Run]) -> tuple[dict | None, dict | None]:
    """把 Loop 里的 runs 按代码快照分成两侧：出现早的是 before，晚的是 after。

    每侧取 `test`（正确性）与 `benchmark`（性能）各一条（同动作多条时取最新一条；同一快照重跑
    属于「同侧复现」，不改变结论）。
    """
    usable = [r for r in runs if r.status == "ok" and r.code_snapshot_sha256]
    if not usable:
        return None, None
    # 每个快照首次出现的 run id 作为排序键
    first_seen: dict[str, int] = {}
    for r in sorted(usable, key=lambda x: x.id):
        first_seen.setdefault(str(r.code_snapshot_sha256), r.id)
    order = sorted(first_seen.items(), key=lambda kv: kv[1])   # [(snapshot, first_id)]，早→晚
    if len(order) < 2:
        return None, None

    def side_for(snapshot: str) -> dict:
        rs = [r for r in usable if str(r.code_snapshot_sha256) == snapshot]
        by_action: dict[str, Run] = {}
        for r in sorted(rs, key=lambda x: x.id):
            by_action[r.action] = r          # 同动作取最新
        return {"snapshot": snapshot, "runs": rs, "by_action": by_action}

    before = side_for(order[0][0])
    after = side_for(order[-1][0])
    return before, after


# ---------------------------------------------------------------- 逐例对比


def _case_index(correctness: dict | None) -> dict[str, str]:
    if not correctness:
        return {}
    return {str(c.get("case_id")): str(c.get("status")) for c in (correctness.get("cases") or [])}


def _rel_uncertainty(agg: dict) -> float:
    """相对不确定性：样本 std / 中位数（两个数都没有就按 0 处理）。"""
    med = float(agg.get("median_us") or 0)
    std = float(agg.get("std_us") or 0)
    return (std / med) if med > 0 else 0.0


def _perf_verdict(delta_pct: float, noise: float) -> str:
    """单用例结论：delta_pct 为 after 相对 before 的变化率（负 = 变快）。"""
    threshold = max(noise, MIN_EFFECT)
    if delta_pct <= -threshold:
        return "improved"
    if delta_pct >= threshold:
        return "regressed"
    if noise > 0.20:
        return "inconclusive"
    return "no_significant_change"


def aggregate_case_verdicts(verdicts: list[str] | tuple[str, ...]) -> str:
    """把逐用例结论保守聚合成本 Loop 的总结论（**门禁与报告共用同一口径**，避免两份结论打架）。

    优先级（与 H3 修复口径一致）：任一 `regressed` → regressed；有 `inconclusive` 且无 regressed
    → inconclusive；**全部** improved → improved；其余（含全 no_significant_change / 混合）→
    no_significant_change。「任一目标回退即 regressed」是多目标场景的保守约定。
    """
    items = [str(v) for v in verdicts]
    if "regressed" in items:
        return "regressed"
    if "inconclusive" in items:
        return "inconclusive"
    if items and all(v == "improved" for v in items):
        return "improved"
    return "no_significant_change"


def _case_fingerprint(perf: dict) -> list:
    """基准用例集指纹（用例 / 形状 / dtype / 是否主用例）：两侧必须一致才叫「同条件」。"""
    return [[str(c.get("case_id")), c.get("n"), c.get("dtype"), bool(c.get("primary"))]
            for c in (perf.get("cases") or [])]


def _objective_target_cases(loop: Loop) -> list[str]:
    """benchmark 侧目标用例（只对性能 / 硬件目标生效；precision 轮的目标用例不在 benchmark 报告里）。"""
    objective = loop.objective
    if not isinstance(objective, dict):
        return []
    if str(objective.get("type") or "") not in _BENCH_OBJECTIVE_TYPES:
        return []
    if objective.get("target_dimension") != "cases":
        return []
    return [str(case_id) for case_id in (objective.get("target_cases") or []) if str(case_id)]


def _objective_all_cases(loop: Loop) -> bool:
    """目标是否「全部用例」（dimension=all 的性能 / 硬件轮次）：是则保守聚合全部可配对用例。

    没有 objective 的旧 Loop 返回 False → 保持 benchmark primary 的兼容口径（不改历史结论）。
    """
    objective = loop.objective
    if not isinstance(objective, dict):
        return False
    if str(objective.get("type") or "") not in _BENCH_OBJECTIVE_TYPES:
        return False
    return str(objective.get("target_dimension") or "all") == "all"


def _diff_cases(before_perf: dict, after_perf: dict, *,
                target_case_ids: list[str] | None = None,
                aggregate_all: bool = False) -> tuple[list[dict], str]:
    """逐用例耗时对比 + 总结论。

    - `target_case_ids`：dimension=cases 的目标用例（benchmark case_id）；
    - `aggregate_all`：dimension=all 时对全部可配对用例保守聚合；
    - 两者都没有（旧 Loop / precision 轮）：沿用 benchmark primary 的兼容口径。
    """
    b_cases = {c["case_id"]: c for c in (before_perf.get("cases") or [])}
    a_cases = {c["case_id"]: c for c in (after_perf.get("cases") or [])}
    rows: list[dict] = []
    for cid, a in a_cases.items():
        b = b_cases.get(cid)
        if not b:
            continue
        b_dev = (b.get("candidate") or {}).get("device_us") or {}
        a_dev = (a.get("candidate") or {}).get("device_us") or {}
        b_med, a_med = float(b_dev.get("median_us") or 0), float(a_dev.get("median_us") or 0)
        noise = _rel_uncertainty(b_dev) + _rel_uncertainty(a_dev)
        # 缺有效中位数（after 漏采 / 为 0）时不能按 0 做分母算「-100% improved」（M12）：
        # 该用例记为不可比，delta 记 null，绝不驱动 improved/regressed。
        missing = b_med <= 0 or a_med <= 0
        delta = None if missing else (a_med - b_med) / b_med
        rows.append({
            "case_id": cid,
            "n": a.get("n") or b.get("n"),
            "before_us": b_med, "after_us": a_med,
            "delta_pct": None if delta is None else round(delta * 100, 3),
            "uncertainty_pct": round(noise * 100, 3),
            "verdict": "inconclusive" if delta is None else _perf_verdict(delta, noise),
            "missing_data": missing,
            "before_baseline_us": float(((b.get("baseline") or {}).get("device_us") or {}).get("median_us") or 0),
            "after_baseline_us": float(((a.get("baseline") or {}).get("device_us") or {}).get("median_us") or 0),
        })
    primary_id = (after_perf.get("summary") or {}).get("primary_case")
    overall = "inconclusive"   # 没有可配对用例时判不了，不给 no_significant_change 的假结论
    if rows:
        selected: list[dict] = []
        if target_case_ids:
            wanted = set(target_case_ids)
            selected = [r for r in rows if r["case_id"] in wanted]
        if not selected and aggregate_all:
            # dimension=all：目标就是「全部用例」，任一用例回退都不能被主用例遮住（H3）。
            selected = list(rows)
        if not selected:
            # 兼容旧 Loop / precision 轮：沿用 benchmark 声明的 primary case（缺失时退回第一个）。
            primary = next((r for r in rows if r["case_id"] == primary_id), None)
            selected = [primary or rows[0]]
        overall = aggregate_case_verdicts([r["verdict"] for r in selected])
    return rows, overall


def _precision(before_c: dict | None, after_c: dict | None) -> dict:
    b = _case_index(before_c)
    a = _case_index(after_c)
    cases = []
    regressed = []
    for cid, a_stat in a.items():
        b_stat = b.get(cid)
        worse = a_stat not in (_PASSING,) or (b_stat == _PASSING and a_stat != _PASSING)
        cases.append({"case_id": cid, "before": b_stat, "after": a_stat, "regressed": bool(worse)})
        if worse:
            regressed.append(cid)
    return {"cases": cases, "regressed_cases": regressed,
            "non_regression": not regressed and bool(a)}


def _metric_evidence(perf: dict | None, ids: tuple[str, ...]) -> list[str]:
    if not perf:
        return []
    out = []
    for m in perf.get("metrics") or []:
        if m.get("id") in ids:
            v = m.get("value")
            out.append(f"{m['id']}={v if not isinstance(v, float) else round(v, 3)}")
    return out


def _status_for(conclusion: str) -> str:
    return {"improved": "validated", "regressed": "rejected",
            "no_significant_change": "inconclusive", "inconclusive": "inconclusive"}.get(
        conclusion, "inconclusive")


def _delta_text(delta: float | None) -> str:
    """逐用例变化率文案：缺有效测量（`delta_pct is None`）时如实写「缺有效测量」，不伪造 0%。"""
    return f"{delta:+.1f}%" if isinstance(delta, (int, float)) else "缺有效测量"


def _recommendations(loop: Loop, *, before: dict, after: dict, perf_after: dict | None,
                     rows: list[dict], conclusion: str) -> list[dict]:
    """把「本轮做了什么 + 应该做什么」落成**结构化建议**（方案 §4.6 的 R1，证据引用 + 验证方法）。

    第 1 条 = 本轮实际尝试的改动（状态由 A/B 结论定）；若有用例回退，再给第 2 条**面向下一步**的
    建议（proposed）。证据一律引用 run id 与具体指标，不空口。
    """
    recs: list[dict] = []
    target_ids = _objective_target_cases(loop)
    primary_id = target_ids[0] if target_ids else (perf_after or {}).get("summary", {}).get("primary_case")
    primary = next((r for r in rows if r["case_id"] == primary_id), rows[0] if rows else None)
    after_ids = [r.id for r in after["runs"]]
    after_bench = after["by_action"].get("benchmark")
    recs.append({
        "id": f"R{loop.id}-1",
        "title": "本轮优化尝试",
        "affected_cases": [primary["case_id"]] if primary else [],
        "observed_evidence": [
            f"after runs {after_ids}（benchmark #{after_bench.id if after_bench else '?'}）",
            *_metric_evidence(perf_after, ("op_time_us", "candidate_achieved_bw_gbps", "baseline_op_time_us")),
        ],
        "bottleneck_hypothesis": loop.hypothesis or "（未记假设）",
        "confidence": "medium",
        "proposed_change": loop.scope or "（未记改动）",
        "expected_observation": loop.judge_condition or "（未记判定条件）",
        "constraints_and_risks": "仅在本镜像/本卡/本 shape 上验证；换条件需重测",
        "validation_method": "同条件 A/B（本报告 cases；设备独占、镜像/判据/用例集一致）",
        "status": _status_for(conclusion),
    })

    regressed = [r for r in rows if r["verdict"] == "regressed"]
    if regressed:
        recs.append({
            "id": f"R{loop.id}-2",
            "title": "小 shape 回退的后续方向",
            "affected_cases": [r["case_id"] for r in regressed],
            "observed_evidence": [f"{r['case_id']} {r['before_us']:.3f}→{r['after_us']:.3f}us"
                                  f"（{_delta_text(r.get('delta_pct'))}）"
                                  for r in regressed],
            "bottleneck_hypothesis": "小 shape 下核启动/同步开销占比高（或分块过大导致每块固定开销摊不薄）",
            "confidence": "low",
            "proposed_change": "按最佳实践做小 shape 核数裁剪（core-shrink）：coreNum = min(满核, ceil(totalRows / min_rows_per_core))，min_rows_per_core 逐 shape 扫描；核数选择收敛到 Tiling 侧",
            "expected_observation": "小 shape 的 aiv_time 下降，主用例不退化",
            "constraints_and_risks": "仅对小 shape 有效；核数太多/太少都可能更差，要逐 shape 校准；与分块/双缓冲叠加需组合验证",
            "validation_method": "新开一轮 perf Loop，同条件 A/B（本报告作为基线一侧）",
            "status": "proposed",
            "skill": "common/core_shrink_design.md",
        })
    return recs


# ---------------------------------------------------------------- 对外入口


def evaluate_loop(loop: Loop, runs: list[Run], *, write: bool = False) -> dict:
    """对一个 per 优化 Loop 出对比报告与判定（不抛异常；判不了本身就是结论）。"""
    rep = _Report(loop)
    if loop.kind != "perf":
        rep.fail("E_NOT_PERF_LOOP", f"Loop #{loop.id} 的 kind={loop.kind!r}，不是优化（perf）Loop")

    before, after = _side_runs(runs)
    report: dict = {
        "schema": "opx-perf-diff/1",
        "loop_id": loop.id,
        "task_id": loop.task_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": loop.hypothesis,
        "judge_condition": loop.judge_condition,
        "min_effect_pct": MIN_EFFECT * 100,
    }
    if before is None or after is None:
        rep.fail("E_PERF_LOOP_INCOMPLETE",
                 "优化 Loop 需要两个不同代码快照（优化前/后）；当前不足两个可用的快照",
                 snapshots=sorted({str(r.code_snapshot_sha256) for r in runs if r.code_snapshot_sha256}))
        report.update({"verdict": "fail", "conclusion": "inconclusive",
                       "findings": rep.findings, "sides": None, "cases": [], "precision": None})
        return report

    b_bench, a_bench = before["by_action"].get("benchmark"), after["by_action"].get("benchmark")
    b_test, a_test = before["by_action"].get("test"), after["by_action"].get("test")

    report["sides"] = {
        "before": {"code_snapshot_sha256": before["snapshot"],
                   "test_run_id": b_test.id if b_test else None,
                   "benchmark_run_id": b_bench.id if b_bench else None,
                   "run_ids": [r.id for r in before["runs"]]},
        "after": {"code_snapshot_sha256": after["snapshot"],
                  "test_run_id": a_test.id if a_test else None,
                  "benchmark_run_id": a_bench.id if a_bench else None,
                  "run_ids": [r.id for r in after["runs"]]},
    }

    if b_bench is None or a_bench is None:
        rep.fail("E_PERF_DIFF_INCOMPLETE", "优化前/后都必须各有一次 benchmark 才能对比耗时")
    if b_test is None or a_test is None:
        rep.fail("E_PERF_DIFF_INCOMPLETE", "优化前/后都必须各有一次 test 才能判断精度不退化")

    # ---- 可比性：规格 / 判据 / 镜像 / 设备必须一致 ----
    # 注：`suite_ref` 是 cases/ 目录的内容哈希，会随「基准配置等非正确性文件」变化，
    # 所以**不拿它做硬条件**（只记录）；真正该锁的是**基准用例集本身**（下面按 case 指纹比）。
    checks: dict = {}
    for key, label in (("spec_version_id", "规格版本"), ("criteria_ref", "判据档"),
                       ("image_digest", "镜像 digest")):
        b_val = getattr(b_bench or b_test, key, None)
        a_val = getattr(a_bench or a_test, key, None)
        same = str(b_val) == str(a_val)
        checks[key] = {"before": b_val, "after": a_val, "same": same, "hard": True}
        if not same:
            rep.fail("E_PERF_DIFF_NOT_COMPARABLE",
                     f"前后 {label} 不一致（{b_val} ≠ {a_val}）—— 同条件对比不成立",
                     key=key, before=b_val, after=a_val)
    # 设备**不再作为硬条件**（2026-10-28 用户口径：去掉「必须同卡」的要求）：
    # 执行侧在指定卡被占时会自动切到空闲卡，前后落到不同卡是允许的；设备号仍记录进报告，
    # 供人工判断测量条件（不同卡之间的绝对耗时差异由报告使用方自行留意）。
    b_dev = getattr(b_bench or b_test, "device_id", None)
    a_dev = getattr(a_bench or a_test, "device_id", None)
    checks["device_id"] = {
        "before": b_dev, "after": a_dev, "same": str(b_dev) == str(a_dev), "hard": False,
        "note": "设备只记录不硬判（允许自动切卡）；跨卡绝对耗时可比性交由使用方判断",
    }
    checks["suite_ref"] = {
        "before": getattr(b_bench or b_test, "suite_ref", None),
        "after": getattr(a_bench or a_test, "suite_ref", None),
        "same": str(getattr(b_bench or b_test, "suite_ref", None)) == str(getattr(a_bench or a_test, "suite_ref", None)),
        "hard": False,
        "note": "suite_ref 只记录不硬判（会随 cases/ 里的非正确性文件一起变）",
    }

    before_perf = _read_typed(b_bench, "performance") if b_bench else None
    after_perf = _read_typed(a_bench, "performance") if a_bench else None
    before_corr = _read_typed(b_test, "correctness") if b_test else None
    after_corr = _read_typed(a_test, "correctness") if a_test else None

    # ---- 可比性（续）：基准用例集必须一致（用例 / 形状 / dtype）----
    if before_perf and after_perf:
        b_fp, a_fp = _case_fingerprint(before_perf), _case_fingerprint(after_perf)
        checks["bench_cases"] = {"before": b_fp, "after": a_fp, "same": b_fp == a_fp, "hard": True}
        if b_fp != a_fp:
            rep.fail("E_PERF_DIFF_NOT_COMPARABLE",
                     "前后基准用例集不一致（用例/形状/dtype 有变）—— 对比的不是同一组条件",
                     before=b_fp, after=a_fp)
    report["comparability"] = checks

    # ---- 耗时对比 ----
    rows: list[dict] = []
    conclusion = "inconclusive"
    # 目标口径在 perf_diff 内只认「benchmark 侧」：precision 轮与无 objective 旧 Loop 都回到
    # benchmark primary 兼容口径（precision 的目标用例匹配由 optimization.report 的 precision 段负责）。
    target_case_ids = _objective_target_cases(loop)
    aggregate_all = _objective_all_cases(loop)
    if before_perf and after_perf:
        rows, conclusion = _diff_cases(before_perf, after_perf,
                                       target_case_ids=target_case_ids, aggregate_all=aggregate_all)
        if not rows:
            rep.fail("E_PERF_DIFF_NO_CASES", "前后基准报告没有可配对的用例（case_id 对不上）")
        missing_data = [row["case_id"] for row in rows if row.get("missing_data")]
        if missing_data:
            rep.warn("W_PERF_MISSING_DATA",
                     "这些用例缺 before/after 有效耗时中位数（缺失或 median_us<=0），不参与提升/回退判定",
                     case_ids=missing_data)
        # 只有「用例 id 在报告里根本不存在」才算缺用例；存在但没测到有效 median 的走上面的
        # W_PERF_MISSING_DATA/inconclusive（口径见 M12 修复）。
        missing_targets = [case_id for case_id in target_case_ids
                           if not any(row["case_id"] == case_id for row in rows)]
        if missing_targets:
            rep.fail("E_TARGET_CASE_MISSING", "目标用例在前后基准报告里不存在，无法对比",
                     case_ids=missing_targets)
            conclusion = "inconclusive"  # 别让「目标对不上」与 improved/regressed 的结论自相矛盾（H2）
    else:
        rep.fail("E_REPORT_UNREADABLE", "读不到前后任一侧的 performance.json（benchmark 产物）")

    # ---- 精度不退化（硬拦）----
    prec = None
    if before_corr and after_corr:
        prec = _precision(before_corr, after_corr)
        if prec["regressed_cases"]:
            rep.fail("E_PRECISION_REGRESSION",
                     f"优化后精度退化：{prec['regressed_cases'][:10]}（这些用例从 PASS 变差）",
                     case_ids=prec["regressed_cases"][:50])
        elif not prec["cases"]:
            rep.fail("E_PRECISION_UNCHECKED", "优化后的正确性报告里没有用例，精度不退化判不了")
    else:
        rep.fail("E_REPORT_UNREADABLE", "读不到前后任一侧的 correctness.json（test 产物）")

    # ---- 证据输入一致性：参考实现 / 判据不得被改 ----
    # 旧报告的 correctness.json 没有 oracle 字段：只在两侧都有时硬判，避免历史数据误伤；
    # 新报告（tools/run_cases.py 2026-10-26 起）两侧都带 golden/criteria 内容哈希。
    before_oracle = before_corr.get("oracle") if isinstance(before_corr, dict) else None
    after_oracle = after_corr.get("oracle") if isinstance(after_corr, dict) else None
    oracle_check = {"before": before_oracle, "after": after_oracle}
    if isinstance(before_oracle, dict) and isinstance(after_oracle, dict):
        for key, label in (("golden_sha256", "参考实现 golden"),
                           ("criteria_sha256", "判据档")):
            b_val, a_val = before_oracle.get(key), after_oracle.get(key)
            if b_val and a_val and str(b_val) != str(a_val):
                rep.fail("E_ORACLE_CHANGED",
                         f"{label}在优化前后不一致（{b_val} ≠ {a_val}）—— 精度对比不成立",
                         field=key, before=b_val, after=a_val)

    objective = loop.objective if isinstance(loop.objective, dict) else {}
    constraints = objective.get("constraints") if isinstance(objective.get("constraints"), dict) else {}
    perf_non_regression = bool(constraints.get("perf_non_regression"))
    # 旧 Loop 仍保持“性能回退只 warn”；优化目标明确要求性能不退化时，目标/全部用例回退是硬拦。
    if conclusion == "regressed":
        affected = [r["case_id"] for r in rows if r["verdict"] == "regressed"]
        if perf_non_regression:
            rep.fail("E_TARGET_PERF_REGRESSION",
                     "优化目标要求性能不退化，但目标/全部对比用例中有显著变慢",
                     case_ids=affected)
        else:
            rep.warn("W_PERF_REGRESSED", "优化后有对比用例耗时变慢（超过测量不确定性）；如实记录",
                     cases=affected)
    elif conclusion == "inconclusive":
        rep.warn("W_PERF_INCONCLUSIVE", "耗时差异落在测量不确定性内（或缺少有效测量），无法确认提升")
    elif conclusion == "no_significant_change":
        rep.warn("W_PERF_NO_SIGNIFICANT_CHANGE", "优化后耗时没有显著变化（提升不显著如实报告）")

    failed = rep.failed
    precision_regressed = bool(prec and prec["regressed_cases"])
    if target_case_ids:
        decision_basis = "target_cases"
    elif aggregate_all:
        decision_basis = "all_cases"
    else:
        decision_basis = "benchmark_primary"
    report.update({
        "verdict": "pass" if not failed else "fail",
        # 精度退化时整体结论按「优化失败/变差」记；否则用性能结论
        "conclusion": "regressed" if precision_regressed else conclusion,
        "cases": rows,
        "precision": prec,
        "oracle": oracle_check,
        "recommendations": _recommendations(loop, before=before, after=after, perf_after=after_perf,
                                            rows=rows, conclusion=conclusion),
        "summary": {"cases": len(rows), "fail": len(failed),
                    "warn": len([f for f in rep.findings if f["level"] == "warn"]),
                    "primary_verdict": conclusion,
                    "decision_basis": decision_basis,
                    "target_case_ids": target_case_ids},
        "findings": rep.findings,
        "grouping": {"before": "优化前（出现较早的代码快照）", "after": "优化后（出现较晚的代码快照）"},
        "notes": [
            "自动优化验收 = 闭环 + 对比报告；性能提升幅度不设硬线（需求调整已定）。",
            "精度退化是硬拦（E_PRECISION_REGRESSION）：优化不得让正确性用例变差。",
            "耗时主值 = benchmark 的候选设备侧中位数；不确定性 = 前后样本 std/中位数之和。",
            "目标用例口径：cases → target_cases；all → 全部可配对用例保守聚合（任一 regressed 即 regressed）；"
            "旧 Loop / precision 轮按 benchmark primary。",
            "缺有效耗时中位数（缺失或 median_us<=0）的用例记 inconclusive/missing_data，不参与提升/回退判定。",
        ],
    })

    if write:
        target = None
        if a_bench is not None:
            d = _run_dir(a_bench)
            if d is not None:
                target = d / "performance-diff.json"
        if target is not None:
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            report["written_to"] = str(target)
    return report
