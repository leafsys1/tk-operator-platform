"""优化轮次 v1（P1）：目标解析 / 建轮次 / 采纳 / 否决 / 可复现报告。

契约见 `docs/plan-optimize-rounds-v1.md`。一句话模型：
**轮次（Loop）是优化单位**，存盘点（savepoint）通过 `loop_id` 归属轮次：

- `baseline`：建轮次时对当前工作区打的基线存点（代码 = 本轮起点）；
- `candidate`：benchmark 成功时自动落的候选存点（`action_runner._finalize`）；
- `accept` / `rejected`：人工采纳 / 否决后把 candidate 行原地升级。

数据真相源仍是 `runs`（真实执行证据）与 `savepoints`（代码内容）；本模块只做编排与报告，
不跑真机、不写工作区（回滚走 `savepoints.restore`）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (ActionJob, AgentJob, Loop, LoopConfirmation, LoopRunLink, OptimizationIdea,
                      Project, Run, Savepoint, Task)
from . import gate, loop_state, optimize, perf_diff, savepoints, workspace_lease

logger = logging.getLogger(__name__)

OBJECTIVE_TYPES = ("perf", "precision", "hardware")
#: cases = 指定 benchmark 用例；all = 全部可配对用例（保守聚合）；text = 只有文字描述的优化点，
#: 不指定目标用例（perf-diff/报告按 benchmark primary 兜底，如思路池 -> 轮次）。
TARGET_DIMENSIONS = ("cases", "all", "text")
OBJECTIVE_SCHEMA = "opx-optimization-objective/1"
REPORT_SCHEMA = "opx-optimization-report/1"

#: 缺省假设 / 预算（objective 里没给时补齐；预算由 `loop_budget` 强制真实执行次数）
DEFAULT_HYPOTHESIS = "基于目标硬件画像与现有证据，寻找可回退的最小改动"
#: 每动作 3 次（2026-10-23 由 2 上调）：基线 + 候选刚好 2 次，留 1 次给失败/环境类重试；
#: 失败尝试仍占额度（硬限制语义不变）。
DEFAULT_BUDGET = {"diagnose_rounds": 3, "verify_rounds": 3, "runs_per_action": 3}

#: 曾属于本轮结果、需要进报告的存点类型（candidate 采纳/否决后原地升级）
_RESULT_KINDS = ("candidate", "accept", "rejected")


class OptimizationError(ValueError):
    """优化轮次业务错误；`status` 供路由层映射 HTTP 状态码（默认 400）。"""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- 通用小工具


def _iso(dt: datetime | None) -> str | None:
    """UTC ISO（SQLite 读回来的 naive datetime 按 UTC 处理）。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _artifact_json(run: Run | None, *types: str) -> dict | None:
    """按产物类型读 run 的机读报告（`gate.artifact_paths` 口径，只认 .json）。"""
    if run is None:
        return None
    paths = gate.artifact_paths(run)
    for atype in types:
        path = paths.get(atype)
        if path is not None and path.suffix == ".json":
            data = _load_json(path)
            if data is not None:
                return data
    return None


def _workspace_of(db: Session, task: Task) -> Path | None:
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        return None
    return Path(project.workspace_path).expanduser()


def _safe_current_sha(workspace: Path | None, sp: Savepoint | None = None) -> str | None:
    if workspace is None or not workspace.is_dir():
        return None
    try:
        return (savepoints.current_evidence_sha(workspace, sp) if sp is not None
                else savepoints.current_sha(workspace))
    except savepoints.SavepointError:
        return None


def loop_view(loop: Loop) -> dict:
    """`LoopOut` 的 dict 形态（与 Pydantic 序列化字段一致，时间用 UTC ISO）。"""
    return {
        "id": loop.id,
        "task_id": loop.task_id,
        "kind": loop.kind,
        "scope": loop.scope,
        "hypothesis": loop.hypothesis,
        "judge_condition": loop.judge_condition,
        "conclusion": loop.conclusion,
        "status": loop.status,
        "optimize": loop.optimize,
        "objective": loop.objective,
        # Step 1 状态机：页面/对话都从这里渲染，不再靠 runs/jobs 拓扑反推。
        "phase": loop.phase or "idle",
        "state_version": int(loop.state_version or 0),
        "active_job_id": loop.active_job_id,
        "phase_error": loop.phase_error,
        "phase_updated_at": _iso(loop.phase_updated_at),
        "created_at": _iso(loop.created_at),
        "ended_at": _iso(loop.ended_at),
    }


def _latest_savepoint(db: Session, loop_id: int, kinds: tuple[str, ...]) -> Savepoint | None:
    return db.scalar(
        select(Savepoint)
        .where(Savepoint.loop_id == loop_id, Savepoint.kind.in_(kinds))
        .order_by(Savepoint.created_at.desc(), Savepoint.id.desc())
        .limit(1)
    )


def _require_task(db: Session, loop: Loop) -> Task:
    task = db.get(Task, loop.task_id)
    if task is None:
        raise OptimizationError("任务不存在", status=404)
    return task


def _has_accepted(db: Session, loop: Loop) -> bool:
    if (loop.conclusion or "").startswith("accepted"):
        return True
    return db.scalar(
        select(Savepoint.id)
        .where(Savepoint.loop_id == loop.id, Savepoint.kind == "accept")
        .limit(1)
    ) is not None


# ---------------------------------------------------------------- 门禁（docs/plan-optimize-auto-v1.md §3）
#
# 这一节只做「证据是否成立」的判定，不改工作区、不跑真机。口径：
#   - Run.status=ok 只代表脚本跑完（exit 0），**不等于**业务通过；
#   - 任何“就绪/通过”都必须有 gate verdict=pass；
#   - before/after 由 baseline/candidate 的 sha 显式绑定，不按 run 出现顺序推断。

#: 基线两侧必须一致的执行环境键（None 视为不满足）。
#: 设备不在此列（2026-10-28 用户口径：去掉「必须同卡」的要求）——执行侧被占会自动切卡，
#: 两侧设备号仅记录进 `env.device_id` / `env.device_consistent`，不作为 baseline 门禁条件。
_RUN_ENV_KEYS = ("spec_version_id", "criteria_ref", "image_digest")


def _loop_run_condition(loop_id: int):
    """轮次可见的 run 集合：直接归属（`runs.loop_id`）∪ 复用关联（`loop_run_links`）。

    Step 1b 起「复用建轮次前已存在的基线 run」不再改写 `runs.loop_id`，而是写 link；
    读取侧统一走这个并集，历史数据（loop_id 已被改写）照旧可读。
    """
    return or_(
        Run.loop_id == loop_id,
        Run.id.in_(select(LoopRunLink.run_id).where(LoopRunLink.loop_id == loop_id)),
    )


def loop_runs(db: Session, loop_id: int) -> list[Run]:
    return list(db.scalars(
        select(Run).where(_loop_run_condition(loop_id)).order_by(Run.id)
    ))


def _linked_run_ids(db: Session, loop_id: int) -> set[int]:
    return set(db.scalars(select(LoopRunLink.run_id).where(LoopRunLink.loop_id == loop_id)))


def _latest_run(db: Session, loop_id: int, action: str, *, sha: str | None = None,
                status: str | None = "ok") -> Run | None:
    stmt = select(Run).where(_loop_run_condition(loop_id), Run.action == action)
    if sha is not None:
        stmt = stmt.where(Run.code_snapshot_sha256 == sha)
    if status is not None:
        stmt = stmt.where(Run.status == status)
    return db.scalar(stmt.order_by(Run.id.desc()).limit(1))


def _run_gate(run: Run) -> dict:
    """门禁结论（异常也按 fail 处理：判不了不能当通过）。"""
    try:
        report = gate.evaluate(run)
    except Exception as e:  # noqa: BLE001 —— 门禁异常不是“通过”
        return {"verdict": "fail", "reason": f"门禁判定异常：{type(e).__name__}: {e}"}
    failed = [f for f in (report.get("findings") or []) if f.get("level") == "fail"]
    reason = "；".join(f"{f.get('code')}: {f.get('message')}" for f in failed[:5])
    return {"verdict": str(report.get("verdict") or "fail"), "reason": reason or None}


def _run_evidence(run: Run | None) -> dict | None:
    if run is None:
        return None
    verdict = _run_gate(run)
    return {
        "run_id": run.id,
        "action": run.action,
        "status": run.status,
        "gate": verdict["verdict"],
        "gate_reason": verdict["reason"],
        "sha256": run.code_snapshot_sha256,
        "spec_version_id": run.spec_version_id,
        "criteria_ref": run.criteria_ref,
        "image_digest": run.image_digest,
        "device_id": run.device_id,
        "host_id": run.host_id,
        "started_at": _iso(run.started_at),
        "ended_at": _iso(run.ended_at),
    }


def _running_jobs(db: Session, task_id: int) -> list[dict]:
    """占用任务工作区/设备的运行中作业（agent 与动作两套执行器都要看）。"""
    rows: list[dict] = []
    for model, kind in ((AgentJob, "agent"), (ActionJob, "action")):
        row = db.scalar(select(model.id).where(model.task_id == task_id,
                                                model.status == "running").limit(1))
        if row is not None:
            rows.append({"kind": kind, "job_id": row})
    return rows


def _baseline_issues(db: Session, loop: Loop, task: Task, workspace: Path | None,
                     baseline: Savepoint | None, out: dict) -> None:
    """把基线问题写进 out（内部用，统一错误码文案）。"""
    def issue(code: str, message: str) -> None:
        out["issues"].append({"code": code, "message": message})

    if not isinstance(loop.objective, dict):
        issue("E_BASELINE_NO_OBJECTIVE", "轮次没有 objective，旧 Loop 只读")
    if loop.status != "open":
        issue("E_BASELINE_LOOP_CLOSED", f"轮次状态是 {loop.status!r}，不能启动自动优化")
    running = _running_jobs(db, task.id)
    if running:
        issue("E_WORKSPACE_BUSY", "任务还有运行中的作业：" + "、".join(
            f"{r['kind']}#{r['job_id']}" for r in running))
    if baseline is None:
        issue("E_BASELINE_MISSING", "本轮还没有 baseline 存点（重新建优化轮次）")
        return
    out["baseline_savepoint_id"] = baseline.id
    out["baseline_sha256"] = baseline.code_snapshot_sha256
    workspace_sha = _safe_current_sha(workspace, baseline)
    out["workspace_sha256"] = workspace_sha
    if workspace_sha is None:
        issue("E_WORKSPACE_UNREADABLE", "读不到工作区源码快照（工作区不存在或被截断）")
    elif workspace_sha != baseline.code_snapshot_sha256:
        issue(
            "E_BASELINE_SHA_MISMATCH",
            "当前工作区代码与基线快照不一致（建轮次后代码被改过）；先回到基线或重新建轮次",
        )


def _env_issues(before: dict[str, Run | None], out: dict) -> None:
    """基线 test / benchmark 的关键执行环境必须一致；设备号只记录（允许自动切卡）。"""
    def issue(code: str, message: str) -> None:
        out["issues"].append({"code": code, "message": message})

    b_test, b_bench = before.get("test"), before.get("benchmark")
    if b_test is None or b_bench is None:
        return
    env: dict = {}
    for key in _RUN_ENV_KEYS:
        bv, av = getattr(b_test, key, None), getattr(b_bench, key, None)
        env[key] = bv
        if bv is None or av is None:
            issue("E_BASELINE_ENV_MISSING", f"基线两侧 {key} 缺失（{bv!r} / {av!r}），不可比")
        elif str(bv) != str(av):
            issue("E_BASELINE_ENV_MISMATCH", f"基线 test/benchmark 的 {key} 不一致：{bv} ≠ {av}")
    # 设备（2026-10-28）：不再作为硬条件——执行侧被占会自动换空闲卡，两侧可能不同卡。
    # 仍把两侧设备号记进 evidence，便于人工判断测量条件。
    env["device_id"] = b_bench.device_id or b_test.device_id
    if b_test.device_id is not None and b_bench.device_id is not None:
        env["device_consistent"] = str(b_test.device_id) == str(b_bench.device_id)
    else:
        env["device_consistent"] = None
    # host_id 有值时必须一致；两边都为空按“历史数据未记录”处理（Step 1 起强制）。
    if b_test.host_id is not None and b_bench.host_id is not None \
            and b_test.host_id != b_bench.host_id:
        issue("E_BASELINE_ENV_MISMATCH",
              f"基线 test/benchmark 的执行机不一致：{b_test.host_id} ≠ {b_bench.host_id}")
    env["host_id"] = b_bench.host_id
    perf = perf_diff._read_typed(b_bench, "performance")
    fingerprint = perf_diff._case_fingerprint(perf) if perf else None
    env["bench_cases_fingerprint"] = fingerprint
    out["env"] = env
    if not fingerprint:
        issue("E_BASELINE_NO_BENCH_CASES", "基线 benchmark 没有可用 case，无法做性能对比")


def baseline_evidence(db: Session, loop: Loop) -> dict:
    """G1：基线是否就绪（docs/plan-optimize-auto-v1.md §3.1）。"""
    task = _require_task(db, loop)
    workspace = _workspace_of(db, task)
    out: dict = {
        "schema": "opx-baseline-evidence/1",
        "loop_id": loop.id,
        "ready": False,
        "issues": [],
        "baseline_savepoint_id": None,
        "baseline_sha256": None,
        "workspace_sha256": None,
        "before": {"test": None, "benchmark": None},
        "env": {},
    }
    baseline = _latest_savepoint(db, loop.id, ("baseline",))
    _baseline_issues(db, loop, task, workspace, baseline, out)
    if baseline is None:
        return out

    before: dict[str, Run | None] = {}
    for action in ("test", "benchmark"):
        run = _latest_run(db, loop.id, action, sha=baseline.code_snapshot_sha256, status="ok")
        if run is None:
            run_any = _latest_run(db, loop.id, action, sha=baseline.code_snapshot_sha256,
                                  status=None)
            out["issues"].append({
                "code": "E_BASELINE_RUN_MISSING",
                "message": (f"基线缺少 {action} 的成功 run"
                            + (f"（最新 run #{run_any.id} 状态 {run_any.status}）"
                               if run_any else "")),
            })
            before[action] = None
            out["before"][action] = _run_evidence(run_any) if run_any else None
            continue
        before[action] = run
        evidence = _run_evidence(run)
        out["before"][action] = evidence
        if evidence and evidence["gate"] != "pass":
            out["issues"].append({
                "code": "E_BASELINE_GATE_FAIL",
                "message": f"基线 {action} run #{run.id} 门禁未通过："
                           f"{evidence.get('gate_reason') or '见 gate 报告'}",
            })
    _env_issues(before, out)
    out["ready"] = not out["issues"]
    return out


def require_baseline(db: Session, loop: Loop) -> dict:
    """启动改码前调用；不就绪一律 409（不得静默放行）。"""
    evidence = baseline_evidence(db, loop)
    if not evidence["ready"]:
        detail = "；".join(f"{i['code']}: {i['message']}" for i in evidence["issues"])
        raise OptimizationError(f"基线未就绪，禁止启动自动优化：{detail}", status=409)
    return evidence


def accept_evidence(db: Session, loop: Loop) -> dict:
    """G7：采纳前的证据门禁（docs/plan-optimize-auto-v1.md §3.2）。"""
    task = _require_task(db, loop)
    workspace = _workspace_of(db, task)
    out: dict = {
        "schema": "opx-accept-evidence/1",
        "loop_id": loop.id,
        "ready": False,
        "issues": [],
        "candidate_savepoint_id": None,
        "candidate_sha256": None,
    }

    def issue(code: str, message: str) -> None:
        out["issues"].append({"code": code, "message": message})

    if loop.status != "open":
        issue("E_ACCEPT_LOOP_CLOSED", f"轮次状态是 {loop.status!r}，不能采纳")
    running = _running_jobs(db, task.id)
    if running:
        issue("E_ACCEPT_BUSY", "任务还有运行中的作业：" + "、".join(
            f"{r['kind']}#{r['job_id']}" for r in running))
    # 自动验证链在跑时也不能采纳：它还会继续改 phase/落报告，并发会互相覆盖。
    from . import loop_orchestrator  # 延迟 import：避免与 orchestrator→optimization 的循环
    auto = loop_orchestrator.latest(db, loop.id)
    if auto is not None and auto.status == "running":
        issue("E_ACCEPT_BUSY", f"自动验证链 #{auto.id} 正在跑，等它结束再采纳")

    baseline = _latest_savepoint(db, loop.id, ("baseline",))
    candidate = _latest_savepoint(db, loop.id, ("candidate",))
    if candidate is None:
        issue("E_ACCEPT_NO_CANDIDATE", "先跑一轮候选验证（还没有 candidate 存点）")
        return out
    out["candidate_savepoint_id"] = candidate.id
    out["candidate_sha256"] = candidate.code_snapshot_sha256
    if baseline is None:
        issue("E_ACCEPT_NO_BASELINE", "轮次没有 baseline 存点，无法证明优化前状态")
        return out

    runs = loop_runs(db, loop.id)
    sides = {
        "before": {
            action: _latest_run(db, loop.id, action, sha=baseline.code_snapshot_sha256,
                                status="ok")
            for action in ("test", "benchmark")
        },
        "after": {
            action: _latest_run(db, loop.id, action, sha=candidate.code_snapshot_sha256,
                                status="ok")
            for action in ("test", "benchmark")
        },
    }
    for side in ("before", "after"):
        for action in ("test", "benchmark"):
            run = sides[side][action]
            if run is None:
                want_sha = (baseline if side == "before" else candidate).code_snapshot_sha256
                issue("E_ACCEPT_RUN_MISSING",
                      f"{side} 侧缺少 {action} 的 ok run（sha={want_sha}）")
                continue
            verdict = _run_gate(run)
            if verdict["verdict"] != "pass":
                issue("E_ACCEPT_RUN_GATE_FAIL",
                      f"{side} 侧 {action} run #{run.id} 门禁未通过："
                      f"{verdict.get('reason') or '见 gate 报告'}")

    perf = perf_diff.evaluate_loop(loop, runs)
    perf_sides = perf.get("sides") if isinstance(perf.get("sides"), dict) else {}
    for side, want in (("before", baseline.code_snapshot_sha256),
                       ("after", candidate.code_snapshot_sha256)):
        got = ((perf_sides.get(side) or {}).get("code_snapshot_sha256"))
        if got != want:
            issue("E_ACCEPT_SIDE_MISMATCH",
                  f"perf-diff 的 {side} 侧绑定到 {got!r}，期望 {want!r}"
                  "（存在多个代码快照/未显式绑定）")
    if perf.get("verdict") != "pass":
        fails = [f for f in (perf.get("findings") or []) if f.get("level") == "fail"]
        issue("E_ACCEPT_PERF_DIFF_FAIL",
              "性能/精度对比未通过：" + "；".join(
                  f"{f.get('code')}: {f.get('message')}" for f in fails[:3]))
    precision = perf.get("precision") if isinstance(perf.get("precision"), dict) else {}
    if precision.get("non_regression") is not True:
        issue("E_ACCEPT_PRECISION_REGRESSION",
              "精度未证明不退化（regressions=" + str(precision.get("regressed_cases")) + "）")
    if perf.get("conclusion") == "regressed":
        issue("E_ACCEPT_PERF_REGRESSED", "优化后性能回退，不能采纳")

    workspace_sha = _safe_current_sha(workspace, candidate)
    if workspace_sha != candidate.code_snapshot_sha256:
        issue("E_ACCEPT_WORKSPACE_MISMATCH",
              "当前工作区代码与候选快照不一致（采纳前先回滚/重新验证）")

    if not out["issues"]:
        # Step 3：采纳要求「append-only 归档报告存在且逐个文件 sha 深度校验通过」，
        # 不接受「刚刚写成功但目录已被改过/raw 缺失」的报告。
        try:
            from . import optimization_reports  # 延迟 import，避免循环依赖
            latest = optimization_reports.latest(loop.id)
            stored = None
            if latest is None:
                stored = optimization_reports.write(db, loop)
                latest = optimization_reports.latest(loop.id)
                if latest is None:
                    raise OptimizationError("报告归档后读不到归档目录", status=409)
            # 只认**最新**归档：深度校验 report.json/files/raw。旧归档有效不能掩盖最新归档被改。
            verified = optimization_reports.verify(loop.id, latest["report_sha256"])
            if verified is None or not verified["valid"]:
                raise OptimizationError(
                    "最新归档报告校验未通过（可能被改动/raw 副本缺失），请重新生成归档后再采纳",
                    status=409)
            # 归档必须与「当前要采纳的候选」绑定：candidate 变了 / 两侧 run 变了要重新归档，
            # 不能拿一份旧的（仍然 valid 的）报告当本次采纳证据。
            archived = optimization_reports.load(loop.id, latest["report_sha256"])
            archived_report = (archived or {}).get("report") or {}
            want_candidate = candidate.code_snapshot_sha256
            got_candidate = ((archived_report.get("code") or {}).get("candidate_sha256")
                             if isinstance(archived_report.get("code"), dict) else None)
            archived_runs = archived_report.get("runs") if isinstance(archived_report.get("runs"), dict) else {}
            want_runs = {
                "before": {a: (sides["before"][a].id if sides["before"][a] is not None else None)
                           for a in ("test", "benchmark")},
                "after": {a: (sides["after"][a].id if sides["after"][a] is not None else None)
                          for a in ("test", "benchmark")},
            }
            if got_candidate != want_candidate or archived_runs != want_runs:
                # 旧归档对应的是上一版候选/另一组 run：重新归档（append 新 seq）后再校验。
                stored = optimization_reports.write(db, loop)
                latest = optimization_reports.latest(loop.id)
                verified = (optimization_reports.verify(loop.id, latest["report_sha256"])
                            if latest is not None else None)
                if verified is None or not verified["valid"]:
                    raise OptimizationError("重新归档后校验未通过", status=409)
                latest = latest or {}
                stored = stored or {}
            out["report"] = {
                "written_to": str(Path(latest["dir"]) / "report.json"),
                "report_sha256": latest.get("report_sha256"),
                "report_dir": latest.get("dir"),
                "verdict": latest.get("verdict"),
                "valid": True,
            }
        except OptimizationError as e:
            issue("E_ACCEPT_REPORT_UNWRITTEN", f"优化报告归档/校验失败：{e}")
        except Exception as e:  # noqa: BLE001 —— 报告异常不能变成“采纳成功”
            issue("E_ACCEPT_REPORT_UNWRITTEN", f"优化报告归档/校验失败：{type(e).__name__}: {e}")
    out["ready"] = not out["issues"]
    return out


def require_accept(db: Session, loop: Loop) -> dict:
    evidence = accept_evidence(db, loop)
    if not evidence["ready"]:
        detail = "；".join(f"{i['code']}: {i['message']}" for i in evidence["issues"])
        raise OptimizationError(f"证据不足，不能采纳本轮候选：{detail}", status=409)
    return evidence


# ---------------------------------------------------------------- 确认草案 / 一次性 token（Step 1b）
#
# 目的（docs/plan-optimize-auto-v1.md §15.1）：把「对话里确认的方案」落成服务端记录，
# 启动只能消费一次；确认后目标/基线被改，启动必须失败，而不是拿旧口令跑新目标。

CONFIRM_SCHEMA = "opx-loop-confirm-draft/1"
#: 确认口令有效期：够用户读完草案再按回车，又不至于长期有效被误用。
CONFIRM_TTL = timedelta(minutes=10)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _objective_sha(loop: Loop) -> str:
    """objective 的稳定指纹（确认后目标被改 → 启动 409）。"""
    raw = loop.objective_json or ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _require_workspace_idle(db: Session, task: Task) -> None:
    """确认/启动前先判「工作区是否被作业占用」，报清楚是哪个作业。

    不能只靠 `require_baseline` 的 `E_WORKSPACE_BUSY`：它被包在「基线未就绪」文案里，
    页面看到的往往是「基线不对」而不是「有个 job 在跑」，用户不知道下一步该干什么。
    """
    running = _running_jobs(db, task.id)
    if running:
        detail = "、".join(f"{r['kind']}#{r['job_id']}" for r in running)
        raise OptimizationError(
            f"E_CONFIRM_BUSY: 任务还有运行中的作业（{detail}），等它结束或去任务页取消后再确认",
            status=409)


def confirm_draft(db: Session, loop: Loop, *, instruction: str | None = None,
                  reference_loop_id: int | None = None, note: str | None = None,
                  actor: str = "page", ttl: timedelta = CONFIRM_TTL) -> dict:
    """确认优化草案：冻结目标/基线/指令 + 发一次性 token（只 flush，不 commit 由调用方定）。"""
    if not isinstance(loop.objective, dict):
        raise OptimizationError("旧 Loop（无 objective）已改为只读；请新建优化轮次后再启动",
                                status=409)
    task = _require_task(db, loop)
    if loop.status != "open":
        raise OptimizationError(f"轮次状态是 {loop.status!r}，不能确认启动", status=409)
    if reference_loop_id is not None:
        ref = db.get(Loop, int(reference_loop_id))
        if ref is None or ref.task_id != task.id:
            raise OptimizationError(f"参考轮次 #{reference_loop_id} 不存在或不属于该任务")
    _require_workspace_idle(db, task)
    evidence = require_baseline(db, loop)  # 未就绪抛 409（含 E_BASELINE_* issues）
    now = datetime.now(timezone.utc)
    expires_at = now + ttl
    token = secrets.token_urlsafe(32)
    draft = {
        "schema": CONFIRM_SCHEMA,
        "loop_id": loop.id,
        "objective_sha256": _objective_sha(loop),
        "baseline_sha256": evidence.get("baseline_sha256"),
        "workspace_sha256": evidence.get("workspace_sha256"),
        "instruction": (instruction or "").strip() or None,
        "reference_loop_id": int(reference_loop_id) if reference_loop_id is not None else None,
        "note": (note or "").strip() or None,
        "state_version": int(loop.state_version or 0),
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    row = LoopConfirmation(
        loop_id=loop.id,
        token_hash=_hash_token(token),
        draft_json=json.dumps(draft, ensure_ascii=False),
        objective_sha256=draft["objective_sha256"],
        baseline_sha256=draft["baseline_sha256"],
        workspace_sha256=draft["workspace_sha256"],
        state_version=draft["state_version"],
        created_at=now,
        expires_at=expires_at,
    )
    db.add(row)
    db.flush()
    loop_state.add_event(db, loop.id, "confirm", phase=loop_state.phase_of(loop), actor=actor,
                         payload={"confirmation_id": row.id,
                                  "baseline_sha256": draft["baseline_sha256"],
                                  "has_instruction": bool(draft["instruction"]),
                                  "reference_loop_id": draft["reference_loop_id"],
                                  "expires_at": draft["expires_at"]})
    return {"confirmation_id": row.id, "loop_id": loop.id, "token": token,
            "expires_at": expires_at.isoformat(), "draft": draft, "row": row}


def validate_confirmation(db: Session, loop: Loop, token: str) -> tuple[LoopConfirmation, dict]:
    """校验 token（存在/未过期/未使用/目标未变）；不做任何写操作。失败抛 409。"""
    raw = (token or "").strip()
    if not raw:
        raise OptimizationError("E_CONFIRM_TOKEN: 缺少确认 token，请先确认优化方案", status=409)
    row = db.scalar(
        select(LoopConfirmation)
        .where(LoopConfirmation.token_hash == _hash_token(raw),
               LoopConfirmation.loop_id == loop.id)
        .order_by(LoopConfirmation.id.desc()).limit(1)
    )
    if row is None:
        raise OptimizationError("E_CONFIRM_TOKEN: 确认 token 无效或不属于该轮次", status=409)
    if row.used_at is not None:
        raise OptimizationError(
            "E_CONFIRM_TOKEN: 该确认已被使用（一次性）；请重新确认优化方案后再启动", status=409)
    expires_at = _utc(row.expires_at)
    if expires_at is None or expires_at <= datetime.now(timezone.utc):
        raise OptimizationError("E_CONFIRM_TOKEN: 确认已过期，请重新确认优化方案后再启动",
                                status=409)
    if _objective_sha(loop) != row.objective_sha256:
        raise OptimizationError(
            "E_CONFIRM_OBJECTIVE_CHANGED: 确认后轮次目标被修改，请重新确认优化方案", status=409)
    if not isinstance(loop.objective, dict) or loop.status != "open":
        raise OptimizationError("E_CONFIRM_TOKEN: 轮次已不可启动", status=409)
    return row, json.loads(row.draft_json or "{}")


def require_confirmation_baseline(db: Session, loop: Loop, draft: dict) -> dict:
    """启动前重跑 G1，并确认「确认时冻结的基线」仍是当前基线。失败抛 409。"""
    _require_workspace_idle(db, _require_task(db, loop))
    try:
        evidence = require_baseline(db, loop)
    except OptimizationError as e:
        raise OptimizationError(f"E_CONFIRM_BASELINE_CHANGED: {e}", status=409) from e
    if draft.get("baseline_sha256") != evidence.get("baseline_sha256"):
        raise OptimizationError(
            "E_CONFIRM_BASELINE_CHANGED: 确认后基线存点已变化，请重新确认优化方案", status=409)
    if draft.get("workspace_sha256") != evidence.get("workspace_sha256"):
        raise OptimizationError(
            "E_CONFIRM_BASELINE_CHANGED: 确认后工作区代码已变化，请重新确认优化方案", status=409)
    return evidence


def consume_confirmation(db: Session, row: LoopConfirmation, job_id: int | None = None) -> None:
    """CAS 消费 token：`used_at` 只能写一次。多请求并发时只有一个成功（不改 commit）。"""
    result = db.execute(
        update(LoopConfirmation)
        .where(LoopConfirmation.id == row.id, LoopConfirmation.used_at.is_(None))
        .values(used_at=datetime.now(timezone.utc), used_agent_job_id=job_id)
    )
    if (result.rowcount or 0) != 1:
        raise OptimizationError(
            "E_CONFIRM_TOKEN: 该确认已被其他请求使用；请重新确认优化方案后再启动", status=409)


# ---------------------------------------------------------------- 目标用例（数据驱动）


def targets(db: Session, task: Task) -> dict:
    """解析目标用例：最近一次 ok 的 benchmark/test 报告（`performance.json` / `correctness.json`）。

    字段一律有默认（不同算子的 case 语义不同，UI 不假设 `n`）：`n`/`dtype`/`note` 可空，
    `primary`/`required` 缺失按 false。
    """
    benchmark = db.scalar(
        select(Run)
        .where(Run.task_id == task.id, Run.action == "benchmark", Run.status == "ok")
        .order_by(Run.id.desc())
        .limit(1)
    )
    test = db.scalar(
        select(Run)
        .where(Run.task_id == task.id, Run.action == "test", Run.status == "ok")
        .order_by(Run.id.desc())
        .limit(1)
    )
    perf = _artifact_json(benchmark, "performance")
    correctness = _artifact_json(test, "correctness")

    cases: list[dict] = []
    primary_case: str | None = None
    if perf:
        summary = perf.get("summary") if isinstance(perf.get("summary"), dict) else {}
        for item in perf.get("cases") or []:
            if not isinstance(item, dict) or not item.get("case_id"):
                continue
            cases.append({
                "case_id": str(item["case_id"]),
                "n": item.get("n"),
                "dtype": item.get("dtype"),
                "primary": bool(item.get("primary")),
                "required": bool(item.get("required")),
                "note": item.get("note"),
            })
        if summary.get("primary_case"):
            primary_case = str(summary["primary_case"])
        elif cases:
            primary_case = cases[0]["case_id"]

    test_cases: list[dict] = []
    failing_cases: list[dict] = []
    for item in (correctness or {}).get("cases") or []:
        if not isinstance(item, dict) or not item.get("case_id"):
            continue
        row = {"case_id": str(item["case_id"]), "status": item.get("status")}
        test_cases.append(row)
        if row["status"] != "PASS":
            failing_cases.append(dict(row))

    hint = ""
    if benchmark is None or perf is None:
        hint = "缺少基准报告：先跑一次「基准测试」"
    elif test is None or correctness is None:
        hint = "缺少正确性报告：先跑一次「正确性测试」"

    return {
        "task_id": task.id,
        "benchmark_run_id": benchmark.id if benchmark is not None else None,
        "test_run_id": test.id if test is not None else None,
        "suite_ref": (benchmark.suite_ref if benchmark is not None else None)
        or (test.suite_ref if test is not None else None),
        "primary_case": primary_case,
        "cases": cases,
        "failing_cases": failing_cases,
        "test_cases": test_cases,
        "hint": hint,
    }


# ---------------------------------------------------------------- 建轮次


def _normalize_objective(task: Task, raw: dict, targets_data: dict) -> dict:
    """校验 + 规范化 objective（补 schema/约束/预算默认值 + 写入 targets_snapshot）。"""
    if not isinstance(raw, dict):
        raise OptimizationError("objective 必须是对象")
    objective = dict(raw)
    otype = str(objective.get("type") or "").strip()
    if otype not in OBJECTIVE_TYPES:
        raise OptimizationError(
            f"objective.type 只能是 {' / '.join(OBJECTIVE_TYPES)}（当前 {otype!r}）")
    dimension = str(objective.get("target_dimension") or "all").strip() or "all"
    if dimension not in TARGET_DIMENSIONS:
        raise OptimizationError(
            f"objective.target_dimension 只能是 {' / '.join(TARGET_DIMENSIONS)}（当前 {dimension!r}）")

    raw_cases = objective.get("target_cases")
    if raw_cases is None:
        raw_cases = []
    if not isinstance(raw_cases, list):
        raise OptimizationError("objective.target_cases 必须是数组")
    target_cases = [str(c).strip() for c in raw_cases if str(c).strip()]
    focus = str(objective.get("focus") or "").strip()
    if dimension == "text":
        # 文字优化点：由思路池/用户描述「需要优化的点」，不指定目标用例。
        # perf-diff/报告按 benchmark primary 兜底；focus 会进 scope/judge 与 Agent prompt。
        if not focus:
            raise OptimizationError("target_dimension=text 时必须给 focus（文字描述需要优化的点）")
        if len(focus) > 2000:
            raise OptimizationError("focus 过长（最多 2000 字）")
        target_cases = []
    elif dimension == "cases":
        if not target_cases:
            raise OptimizationError("target_dimension=cases 时必须给非空 target_cases")
        pool = ([c["case_id"] for c in targets_data["test_cases"]] if otype == "precision"
                else [c["case_id"] for c in targets_data["cases"]])
        missing = [c for c in target_cases if c not in pool]
        if missing:
            what = "test 用例" if otype == "precision" else "benchmark 用例"
            raise OptimizationError(
                f"目标用例不在最新{what}快照里：{missing}（先跑一次"
                f"{'「正确性测试」' if otype == 'precision' else '「基准测试」'}）")
    # all 维度允许目标用例留空（表示「全部用例」）

    constraints = objective.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    constraints = dict(constraints)
    constraints.setdefault("precision_non_regression", True)
    constraints.setdefault("perf_non_regression", otype == "precision")
    if not task.op_name:
        raise OptimizationError("任务未配置 op_name，不能创建自动优化轮次")
    allowed_root = f"ops/{task.op_name}/"
    # 默认只放开算子实现目录（kernel/host）；参考实现 `scripts/`、接口/用例/判据/执行器不在默认范围。
    # 即使调用方显式传 `ops/<op>/`，`optimize._finish` 仍会对受保护路径硬拦（E_PROTECTED_PATH）。
    default_allowed = [f"{allowed_root}op_host/", f"{allowed_root}op_kernel/"]
    raw_allowed = constraints.get("allowed_paths")
    if raw_allowed is None:
        raw_allowed = default_allowed
    if not isinstance(raw_allowed, list):
        raise OptimizationError("constraints.allowed_paths 必须是数组")
    allowed_paths: list[str] = []
    for value in raw_allowed:
        text = str(value).strip().replace("\\", "/")
        if not text:
            continue
        while text.startswith("./"):
            text = text[2:]
        normalized = posixpath.normpath(text).rstrip("/") + "/"
        if normalized.startswith("/") or normalized == "../" or normalized.startswith("../"):
            raise OptimizationError(f"允许改动路径必须位于 {allowed_root}：{value!r}")
        if not normalized.startswith(allowed_root):
            raise OptimizationError(f"允许改动路径不能越过 {allowed_root}：{value!r}")
        if normalized not in allowed_paths:
            allowed_paths.append(normalized)
    # 页面留空也只能回退到默认实现目录，不能用空数组关闭 optimize._finish 的越界门禁。
    constraints["allowed_paths"] = allowed_paths or default_allowed
    budget = objective.get("budget")
    if not isinstance(budget, dict):
        budget = dict(DEFAULT_BUDGET)
    else:
        merged = dict(DEFAULT_BUDGET)
        merged.update(budget)
        budget = merged

    snapshot = {
        "benchmark_run_id": targets_data["benchmark_run_id"],
        "test_run_id": targets_data["test_run_id"],
        "primary_case": targets_data["primary_case"],
        "suite_ref": targets_data["suite_ref"],
        "cases": targets_data["cases"],
        "failing_cases": targets_data["failing_cases"],
    }
    objective.update({
        "schema": objective.get("schema") or OBJECTIVE_SCHEMA,
        "type": otype,
        "target_dimension": dimension,
        "target_cases": target_cases,
        "focus": focus or None,
        "constraints": constraints,
        "budget": budget,
        "user_text": str(objective.get("user_text") or ""),
        "targets_snapshot": snapshot,
    })
    return objective


def _judge_condition(target_cases: list[str], dimension: str,
                     focus: str | None = None) -> str:
    if dimension == "text":
        text = (focus or "").strip()
        summary = text if len(text) <= 200 else text[:200] + "…"
        return f"优化点（文字描述）：{summary}；精度不退化；同设备/镜像/规格/用例集同条件 A/B"
    cases_text = "、".join(target_cases) if dimension == "cases" and target_cases else "全部"
    return f"目标用例 {cases_text}；精度不退化；同设备/镜像/规格/用例集同条件 A/B"


def _existing_round_payload(db: Session, task: Task, loop: Loop, targets_data: dict) -> dict:
    """幂等重放：返回已存在轮次的同一形态响应（不重复建 Loop/基线存档）。"""
    workspace = _workspace_of(db, task)
    baseline = _latest_savepoint(db, loop.id, ("baseline",))
    linked = _linked_run_ids(db, loop.id)
    attached = [
        {"id": run.id, "action": run.action,
         "role": "baseline" if run.id in linked else "owned"}
        for run in loop_runs(db, loop.id)
        if run.action in ("test", "benchmark")
    ]
    return {
        "loop": loop_view(loop),
        "baseline_savepoint": (savepoints.out(db, baseline, workspace=workspace)
                               if baseline is not None else None),
        "attached_runs": attached,
        "targets": targets_data,
    }


def create_round(db: Session, task: Task, *, objective: dict, label: str | None = None,
                 hypothesis: str | None = None, attach_baseline_runs: bool = True,
                 idempotency_key: str | None = None) -> dict:
    """建一个优化轮次：Loop + 基线存档（+ 尽量复用已有基线 run）。

    `idempotency_key`：同任务同 key 只建一个轮次（双击/请求重放返回已有轮次）。
    """
    targets_data = targets(db, task)
    key = (idempotency_key or "").strip() or None
    if key:
        existing = db.scalar(select(Loop).where(Loop.task_id == task.id,
                                                Loop.idempotency_key == key))
        if existing is not None:
            return _existing_round_payload(db, task, existing, targets_data)
    normalized = _normalize_objective(task, objective, targets_data)

    workspace = _workspace_of(db, task)
    if workspace is None:
        raise OptimizationError("项目工作区未配置，无法生成基线存档", status=409)
    if not workspace.is_dir():
        raise OptimizationError(f"工作区不可用：{workspace}", status=409)

    title = (label or "").strip() or str(normalized.get("label") or "").strip()
    otype = normalized["type"]
    dimension = normalized["target_dimension"]
    case_list = normalized["target_cases"]
    focus = str(normalized.get("focus") or "").strip()
    if dimension == "text":
        fallback = (focus[:60] + "…") if len(focus) > 60 else focus
        scope = title or f"{otype} 优化：{fallback or '文字优化点'}"
    else:
        cases_text = "、".join(case_list) if dimension == "cases" and case_list else "全部"
        scope = title or f"{otype} 优化：{cases_text}"
    judge = _judge_condition(case_list, dimension, focus)
    loop = Loop(
        task_id=task.id,
        kind="perf",  # 三类目标都用 perf Loop（perf-diff 需要成对 run；类型在 objective 里区分）
        scope=scope,
        hypothesis=(hypothesis or "").strip() or DEFAULT_HYPOTHESIS,
        judge_condition=judge,
        objective_json=json.dumps(normalized, ensure_ascii=False),
        status="open",
        phase="baseline_pending",
        idempotency_key=key,
    )
    db.add(loop)
    db.flush()
    loop_state.add_event(db, loop.id, "created", phase="baseline_pending",
                         payload={"idempotency_key": key, "type": otype})

    try:
        baseline = savepoints.create(db, task, label=f"基线 · {title or scope}",
                                     kind="baseline", loop_id=loop.id,
                                     evidence_op_name=(task.op_name
                                         if task.cases_confirmed_scope == "operator-v1" else None))
    except savepoints.SavepointError as e:
        raise OptimizationError(f"基线存档失败：{e}", status=409) from e

    attached: list[dict] = []
    if attach_baseline_runs:
        for action in ("test", "benchmark"):
            run = db.scalar(
                select(Run)
                .where(Run.task_id == task.id, Run.loop_id.is_(None), Run.action == action,
                       Run.status == "ok",
                       Run.code_snapshot_sha256 == baseline.code_snapshot_sha256)
                # 同一份同 sha 基线 run 允许被多个轮次复用（文档 TC-010 的推荐路径）。
                # Link 的唯一性只在 (loop_id, run_id)：删轮次只删自己的 link，不改 run 归属。
                .order_by(Run.id.desc())
                .limit(1)
            )
            if run is not None:
                # Step 1b：复用「建轮次前已存在的基线 run」只写显式 link，不改 run 的原归属
                # （否则 run 到底属于哪一轮不可审计，删轮次时也会误伤其它轮次的证据）。
                db.add(LoopRunLink(loop_id=loop.id, run_id=run.id, role="baseline",
                                   created_by="api"))
                attached.append({"id": run.id, "action": run.action, "role": "baseline"})

    try:
        db.commit()
    except IntegrityError as e:  # 并发双击：唯一索引兜底，返回先建的那一轮
        db.rollback()
        if key:
            existing = db.scalar(select(Loop).where(Loop.task_id == task.id,
                                                    Loop.idempotency_key == key))
            if existing is not None:
                return _existing_round_payload(db, task, existing, targets_data)
        raise OptimizationError("轮次创建冲突，请重试", status=409) from e
    db.refresh(loop)
    db.refresh(baseline)
    # 基线能复用就直接推进到 baseline_ready（派生失败不影响建轮次本身）。
    try:
        loop_state.sync_phase(db, loop, event="baseline_synced", commit=True)
    except Exception:  # noqa: BLE001
        logger.exception("建轮次后同步 phase 失败（loop=%s）", loop.id)
    return {
        "loop": loop_view(loop),
        "baseline_savepoint": savepoints.out(db, baseline, workspace=workspace),
        "attached_runs": attached,
        "targets": targets_data,
    }


# ---------------------------------------------------------------- 采纳 / 否决


def accept(db: Session, loop: Loop, *, label: str | None = None,
           note: str | None = None) -> dict:
    """把该 Loop 最新 candidate 原地升级为版本（accept）并关闭轮次。"""
    task = _require_task(db, loop)
    if _has_accepted(db, loop):
        raise OptimizationError("这一轮已经采纳过，不能重复采纳", status=409)
    if loop.status == "closed":
        raise OptimizationError(
            f"轮次已关闭（{loop.conclusion or 'closed'}），不能采纳", status=409)
    # G7 证据门禁：候选存点不等于证据成立；四条 run 必须 gate pass、perf-diff pass、
    # 精度不退化、报告落盘、工作区仍是候选代码。任一不满足 409。
    evidence = require_accept(db, loop)
    candidate = db.get(Savepoint, evidence["candidate_savepoint_id"])
    if candidate is None:
        raise OptimizationError("先跑一轮候选验证（还没有候选存档）", status=409)

    version_number = int(db.scalar(
        select(func.count()).select_from(Savepoint)
        .where(Savepoint.task_id == task.id, Savepoint.kind == "accept")
    ) or 0) + 1
    version_label = (label or "").strip() or f"c{version_number} · {loop.scope or f'Loop #{loop.id}'}"
    candidate.kind = "accept"
    candidate.label = version_label[:200]
    if note is not None:
        candidate.note = note
    loop.status = "closed"
    loop.conclusion = "accepted"
    loop.ended_at = datetime.now(timezone.utc)
    idea = db.scalar(select(OptimizationIdea).where(OptimizationIdea.loop_id == loop.id).limit(1))
    if idea is not None:
        idea.status = "verified"
        idea.outcome_json = json.dumps({"conclusion": "accepted", "loop_id": loop.id}, ensure_ascii=False)
        idea.updated_at = datetime.now(timezone.utc)
    # 任务生命周期 v2：采纳 = 小版本 +1（幂等）。候选 run 的快照就是当前工作区，因此
    # 候选的 test/benchmark 证据直接属于新版本，不需要重跑。
    from . import versions  # 延迟导入：避免服务间循环
    versions.accept_optimization(db, task, loop, note=note)
    # 状态机与版本升级同事务（Step 1）：失败回滚时不能留下“phase=accepted 但版本没升”。
    # CAS 失败 = 另一请求已改过状态（并发采纳/取消）：必须报错让路由 rollback，不能空提交成 200。
    if not loop_state.transition(db, loop, "accepted", event="accept", force=True,
                                 active_job_id=None, error=None,
                                 payload={"version_number": version_number}, commit=False):
        db.rollback()
        raise OptimizationError("轮次状态已被其他请求改变（可能已采纳/否决/取消），请刷新后重试",
                                status=409)
    db.commit()
    db.refresh(candidate)
    db.refresh(loop)

    workspace = _workspace_of(db, task)
    workspace_matches = _safe_current_sha(workspace, candidate) == candidate.code_snapshot_sha256
    return {
        "loop": loop_view(loop),
        "version": savepoints.out(db, candidate, workspace=workspace),
        "version_number": version_number,
        "workspace_matches": workspace_matches,
    }


def reject(db: Session, loop: Loop, *, reason: str | None = None,
           revert: bool = False) -> dict:
    """否决本轮：不要求有候选存档（失败的轮次也能关闭/给思路标记无效）。

    - 最新 candidate（若有）→ rejected；
    - 关闭 Loop，关联思路 → rejected（负样本，后续搜索会看到「已试过」）；
    - `revert=True` 时**尽力**把工作区回退到本轮改前：优先用 `loops.optimize` 的改前快照，
      没有自动优化证据则回退到建轮次时的 baseline 存点；回退失败不影响否决本身，只写进返回值。
    """
    task = _require_task(db, loop)
    if _has_accepted(db, loop):
        raise OptimizationError("这一轮已经采纳过，不能再否决", status=409)
    if loop.status == "closed":
        raise OptimizationError(
            f"轮次已关闭（{loop.conclusion or 'closed'}），不能重复否决", status=409)
    # 有作业还在写工作区时不能关轮次：否则 attempt 被错标 succeeded，agent 收尾还会把
    # closed Loop 的 phase 改回来。先取消/等结束再否决。
    running = _running_jobs(db, task.id)
    if running:
        detail = "、".join(f"{r['kind']}#{r['job_id']}" for r in running)
        raise OptimizationError(
            f"E_REJECT_BUSY: 任务还有运行中的作业（{detail}），先取消或等结束再否决", status=409)
    from . import loop_orchestrator  # 延迟 import（同上）
    auto = loop_orchestrator.latest(db, loop.id)
    if auto is not None and auto.status == "running":
        raise OptimizationError(
            f"E_REJECT_BUSY: 自动验证链 #{auto.id} 正在跑，先取消或等结束再否决", status=409)

    reason_text = (reason or "").strip()
    candidate = _latest_savepoint(db, loop.id, ("candidate",))
    if candidate is not None:
        candidate.kind = "rejected"
        if reason_text:
            candidate.note = reason_text
    loop.status = "closed"
    loop.conclusion = f"rejected：{reason_text}" if reason_text else "rejected"
    loop.ended_at = datetime.now(timezone.utc)
    idea = db.scalar(select(OptimizationIdea).where(OptimizationIdea.loop_id == loop.id).limit(1))
    if idea is not None:
        idea.status = "rejected"
        idea.outcome_json = json.dumps(
            {"conclusion": "rejected", "loop_id": loop.id, "reason": reason_text or None},
            ensure_ascii=False,
        )
        idea.updated_at = datetime.now(timezone.utc)
    if not loop_state.transition(db, loop, "rejected", event="reject", force=True,
                                 active_job_id=None, payload={"reason": reason_text or None},
                                 commit=False):
        db.rollback()
        raise OptimizationError("轮次状态已被其他请求改变（可能已取消/否决/采纳），请刷新后重试",
                                status=409)
    db.commit()
    db.refresh(loop)
    workspace = _workspace_of(db, task)
    revert_info: dict | None = None
    if revert:
        if workspace is None:
            revert_info = {"ok": False, "reason": "项目工作区未配置，未回退"}
        else:
            evidence = loop.optimize if isinstance(loop.optimize, dict) else None
            try:
                if evidence and evidence.get("revert"):
                    # 自动优化记录里有「改前」快照：回退到 agent 动手之前（= 本轮基线代码）。
                    # 回退会写工作区，必须持项目级租约（与 savepoints.restore 同口径）。
                    with workspace_lease.hold(db, task, kind="restore",
                                              note=f"否决轮次 #{loop.id} 回退改前"):
                        ev = optimize.revert(db, loop, workspace)
                    revert_info = {
                        "ok": bool(ev.get("reverted_clean")),
                        "mode": "optimize_before",
                        "reverted_clean": bool(ev.get("reverted_clean")),
                        "revert_failed": ev.get("revert_failed") or [],
                    }
                else:
                    baseline = _latest_savepoint(db, loop.id, ("baseline",))
                    if baseline is None:
                        revert_info = {"ok": False,
                                       "reason": "本轮没有改前快照/基线存档，无法回退"}
                    else:
                        restored = savepoints.restore(
                            db, baseline, task, scope="task",
                            note=f"否决轮次 #{loop.id} 时自动回退原版本")
                        revert_info = {"ok": True, "mode": "baseline_savepoint",
                                       "savepoint_id": baseline.id,
                                       "restore": restored}
            except Exception as exc:  # noqa: BLE001 —— 否决必须完成；回退失败只记录原因
                revert_info = {"ok": False, "mode": "error",
                               "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "loop": loop_view(loop),
        "savepoint": (savepoints.out(db, candidate, workspace=workspace)
                      if candidate is not None else None),
        "reason": reason_text or None,
        "reverted": revert_info,
    }


# ---------------------------------------------------------------- 可复现报告


def _case_status(run: Run | None) -> dict[str, str]:
    data = _artifact_json(run, "correctness")
    out: dict[str, str] = {}
    for item in (data or {}).get("cases") or []:
        if isinstance(item, dict) and item.get("case_id"):
            out[str(item["case_id"])] = str(item.get("status"))
    return out


def _precision_block(before_map: dict[str, str], after_map: dict[str, str],
                     target_ids: list[str]) -> dict:
    rows: list[dict] = []
    newly_passed: list[str] = []
    regressions: list[str] = []
    for case_id in target_ids:
        before = before_map.get(case_id)
        after = after_map.get(case_id)
        gained = before is not None and before != "PASS" and after == "PASS"
        lost = before == "PASS" and after is not None and after != "PASS"
        rows.append({
            "case_id": case_id,
            "before": before,
            "after": after,
            "newly_passed": gained,
            "regressed": lost,
        })
        if gained:
            newly_passed.append(case_id)
        if lost:
            regressions.append(case_id)
    return {
        "before": before_map,
        "after": after_map,
        "target_cases": rows,
        "newly_passed": len(newly_passed),
        "regressions": regressions,
        "non_regression": bool(after_map) and not regressions,
    }


def _precision_summary(precision: dict) -> dict:
    regressions = precision.get("regressions") or []
    newly_passed = int(precision.get("newly_passed") or 0)
    if regressions:
        verdict, conclusion = "fail", "regressed"
    elif newly_passed > 0:
        verdict, conclusion = "pass", "improved"
    elif not precision.get("after"):
        verdict, conclusion = "inconclusive", "inconclusive"
    else:
        verdict, conclusion = "pass", "no_significant_change"
    headline = f"目标用例新通过 {newly_passed} 个，回退 {len(regressions)} 个"
    return {"type": "precision", "verdict": verdict, "conclusion": conclusion,
            "headline": headline}


def _chip_summary(workspace: Path | None, chip: str | None) -> dict:
    """报告里的芯片画像摘要：选中的 `ops/chips/*.yaml` + 关键事实（硬件适配轮次用）。"""
    info = {"target_chip": chip, "profile": None, "summary": None}
    if workspace is None or not (workspace / "ops" / "chips").is_dir():
        return info
    selected = optimize._chip_profile_for(workspace, chip)
    if selected is None:
        return info
    path, _data = selected
    info["profile"] = (path.relative_to(workspace).as_posix()
                       if path.is_relative_to(workspace) else str(path))
    brief = optimize.hardware_brief(workspace, chip)
    info["summary"] = brief or None
    return info


def _summary_case_text(case: dict | None, label: str) -> str:
    """单用例变化率文案：缺有效测量（perf_diff 记 `delta_pct=None`）时如实写「缺有效测量」。"""
    if case is None:
        return f"{label} 缺少可配对数据"
    delta = case.get("delta_pct")
    if not isinstance(delta, (int, float)):
        return f"{label} 缺有效测量"
    return f"{label} {delta:+.1f}%"


def _perf_summary(perf_report: dict, precision: dict, otype: str, *, task: Task,
                  workspace: Path | None, primary_case: str | None,
                  target_dimension: str, target_cases: list[str],
                  constraints: dict, focus: str | None = None) -> dict:
    rows = perf_report.get("cases") or []
    if target_dimension == "cases" and target_cases:
        wanted = set(target_cases)
        selected = [r for r in rows if str(r.get("case_id")) in wanted]
        missing = [case_id for case_id in target_cases
                   if not any(str(r.get("case_id")) == case_id for r in selected)]
    elif target_dimension == "all":
        selected = list(rows)
        missing = []
    else:
        # text（文字优化点）不指定目标用例：按 benchmark primary 兜底，与 legacy 口径一致。
        primary = next((r for r in rows if str(r.get("case_id")) == primary_case),
                       rows[0] if rows else None)
        selected = [primary] if primary is not None else []
        missing = []

    # 与 perf_diff 复用同一保守聚合口径（regressed > inconclusive > improved > no_significant_change），
    # 否则同一个 Loop 的 perf-diff 与 objective_summary 会给出互相矛盾的结论（H3）。
    if missing or not selected:
        conclusion = "inconclusive"
    else:
        conclusion = perf_diff.aggregate_case_verdicts(
            [str(row.get("verdict")) for row in selected])

    # 标题必须跟本轮 objective 对齐，不能永远展示 benchmark 文件声明的 primary case。
    # 否则“目标 64K 回退、1M 持平”会被误报成 pass / no significant change。
    if target_dimension == "cases" and len(target_cases) == 1:
        case_id = target_cases[0]
        case = next((r for r in selected if str(r.get("case_id")) == case_id), None)
        label = "主用例" if case_id == primary_case else f"目标用例 {case_id}"
        perf_text = _summary_case_text(case, label)
    elif target_dimension == "all" and len(selected) > 1:
        # dimension=all 的结论按全部用例聚合，标题也不能只报主用例（H3）。
        regressed = sum(1 for r in selected if str(r.get("verdict")) == "regressed")
        improved = sum(1 for r in selected if str(r.get("verdict")) == "improved")
        unmeasured = sum(1 for r in selected
                         if r.get("missing_data") and str(r.get("verdict")) == "inconclusive")
        detail = [f"回退 {regressed}", f"改善 {improved}"]
        if unmeasured:
            detail.append(f"缺测量 {unmeasured}")
        perf_text = f"全部用例聚合 {len(selected)} 个（{' / '.join(detail)}）"
    elif target_dimension == "text":
        case = selected[0] if selected else None
        perf_text = _summary_case_text(case, "benchmark 主用例") if case else "无可配对的基准用例"
    else:
        case = next((r for r in selected if r.get("case_id") == primary_case),
                    selected[0] if selected else None)
        perf_text = _summary_case_text(case, "主用例") if case else "无可配对的基准用例"

    precision_text = "精度不退化" if precision.get("non_regression") else "精度未确认/有回退"
    perf_non_regression = bool(constraints.get("perf_non_regression", False))
    precision_non_regression = bool(constraints.get("precision_non_regression", True))
    if perf_report.get("verdict") == "fail":
        verdict = "fail"
    elif precision_non_regression and not precision.get("non_regression"):
        verdict = "fail"
    elif conclusion == "regressed":
        # 性能回退不是硬线；但也不能自报 pass —— 如实给 warn（与 perf_diff 的 W_PERF_REGRESSED 同口径）。
        verdict = "fail" if perf_non_regression else "warn"
    elif conclusion == "inconclusive":
        verdict = "inconclusive"
    else:
        verdict = "pass"
    summary = {
        "type": otype,
        "verdict": verdict,
        "conclusion": conclusion,
        "headline": f"{perf_text}，{precision_text}",
        "target_cases": selected,
        "missing_target_cases": missing,
        "target_dimension": target_dimension,
        "focus": focus or None,
        "decision_basis": ("target_cases" if target_dimension == "cases" and target_cases
                           else "all_cases" if target_dimension == "all"
                           else "benchmark_primary"),
    }
    if otype == "hardware":
        summary["chip"] = _chip_summary(workspace, task.target_chip)
    return summary


def report(db: Session, loop: Loop, *, write: bool = False) -> dict:
    """可复现报告：代码（基线/候选/版本存点）+ 两侧 run + perf-diff + 精度 + 复现命令。"""
    task = _require_task(db, loop)
    objective = loop.objective
    if objective is None:
        raise OptimizationError("该轮次不是优化轮次（没有 objective）")
    workspace = _workspace_of(db, task)

    baseline = _latest_savepoint(db, loop.id, ("baseline",))
    version = _latest_savepoint(db, loop.id, ("accept",))
    result = _latest_savepoint(db, loop.id, _RESULT_KINDS)

    runs = loop_runs(db, loop.id)
    perf_report = perf_diff.evaluate_loop(loop, runs, write=False)
    sides = perf_report.get("sides") if isinstance(perf_report.get("sides"), dict) else None

    def _side_sha(kind: str) -> str | None:
        """优先取 perf-diff 配对出的真实 run 快照；存点 sha 只作兜底。

        原因：历史数据里「基线存档 sha」可能与「run 快照 sha」不相等（快照口径漂移），
        若直接拿存点 sha 找 run 会得到 before=None，报告就丢证据。
        """
        side = (sides or {}).get(kind) or {}
        if side.get("code_snapshot_sha256"):
            return side["code_snapshot_sha256"]
        if kind == "before" and baseline is not None:
            return baseline.code_snapshot_sha256
        if kind == "after" and result is not None:
            return result.code_snapshot_sha256
        return None

    before_sha = _side_sha("before")
    after_sha = _side_sha("after")

    def _runs_for(sha: str | None) -> dict:
        if not sha:
            return {"test": None, "benchmark": None}
        by_action: dict[str, Run] = {}
        for run in runs:
            if run.code_snapshot_sha256 == sha:
                by_action[run.action] = run
        return {
            "test": by_action["test"].id if "test" in by_action else None,
            "benchmark": by_action["benchmark"].id if "benchmark" in by_action else None,
        }

    runs_block = {"before": _runs_for(before_sha), "after": _runs_for(after_sha)}

    def _run_of(side: str, action: str) -> Run | None:
        rid = runs_block[side].get(action)
        return db.get(Run, rid) if rid else None

    before_test, after_test = _run_of("before", "test"), _run_of("after", "test")
    before_bench, after_bench = _run_of("before", "benchmark"), _run_of("after", "benchmark")
    before_map, after_map = _case_status(before_test), _case_status(after_test)

    snapshot = objective.get("targets_snapshot") if isinstance(
        objective.get("targets_snapshot"), dict) else {}
    dimension = str(objective.get("target_dimension") or "all")
    target_cases = [str(c) for c in (objective.get("target_cases") or [])]
    focus = str(objective.get("focus") or "").strip()
    otype = str(objective.get("type") or "perf")
    # 精度轮次：逐「目标用例」看 before/after；性能/硬件轮次：精度不退化是全局约束，
    # 覆盖全部正确性用例（目标用例是基准 case，不在这份报告里）。
    if otype == "precision" and dimension == "cases" and target_cases:
        precision_targets = target_cases
    else:
        precision_targets = sorted(set(before_map) | set(after_map))
    precision = _precision_block(before_map, after_map, precision_targets)
    if otype == "precision":
        objective_summary = _precision_summary(precision)
    else:
        constraints = (objective.get("constraints")
                       if isinstance(objective.get("constraints"), dict) else {})
        objective_summary = _perf_summary(
            perf_report, precision, otype, task=task, workspace=workspace,
            primary_case=snapshot.get("primary_case"), target_dimension=dimension,
            target_cases=target_cases, constraints=constraints, focus=focus or None)

    env_run = after_bench or before_bench or after_test or before_test
    report_data = {
        "schema": REPORT_SCHEMA,
        "task_id": task.id,
        "loop_id": loop.id,
        "label": objective.get("label") or loop.scope or f"Loop #{loop.id}",
        "objective": objective,
        "targets": {
            "benchmark_run_id": snapshot.get("benchmark_run_id"),
            "test_run_id": snapshot.get("test_run_id"),
            "primary_case": snapshot.get("primary_case"),
            "target_cases": target_cases,
            "target_dimension": dimension,
            "focus": focus or None,
            "suite_ref": snapshot.get("suite_ref"),
        },
        "code": {
            "baseline_savepoint_id": baseline.id if baseline is not None else None,
            "candidate_savepoint_id": result.id if result is not None else None,
            "version_savepoint_id": version.id if version is not None else None,
            "baseline_sha256": before_sha,
            "candidate_sha256": after_sha,
        },
        "runs": runs_block,
        "perf_diff": perf_report,
        "precision": precision,
        "objective_summary": objective_summary,
        "repro": {
            "before": {
                "test_command": before_test.command if before_test is not None else None,
                "benchmark_command": before_bench.command if before_bench is not None else None,
            },
            "after": {
                "test_command": after_test.command if after_test is not None else None,
                "benchmark_command": after_bench.command if after_bench is not None else None,
            },
            "env": {
                "spec_version_id": env_run.spec_version_id if env_run is not None else None,
                "criteria_ref": env_run.criteria_ref if env_run is not None else None,
                "image_digest": env_run.image_digest if env_run is not None else None,
                "device_id": env_run.device_id if env_run is not None else None,
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        # Step 3：报告改写 append-only 归档仓库（manifest + sha + raw 副本 + 复现命令），
        # 不再写进会被重跑覆盖的 run 目录。落不了盘就明确报错，不能静默返回“看起来有报告”。
        if after_bench is None or not after_bench.manifest_path:
            raise OptimizationError(
                "报告无法落盘：找不到优化后的 benchmark run 目录（先跑候选 benchmark）",
                status=409,
            )
        from . import optimization_reports  # 延迟 import：reports 依赖本模块的 report(write=False)
        return optimization_reports.write(db, loop)
    return report_data
