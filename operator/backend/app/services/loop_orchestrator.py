"""自动验证链 orchestrator（自动优化 Step 2）。

契约见 `docs/plan-optimize-auto-v1.md` §16。一句话：agent 改完代码（`awaiting_verification`）
后，由服务端把 **build → smoke → test → benchmark → report** 串成一条可恢复、可取消、证据绑定的链，
替代页面上手动逐个点动作。

设计取舍：
- **驱动在进程内 asyncio task**（动作执行器本身就是 asyncio 子进程），但进度持久化在
  `loop_auto_runs`；进程重启后 `reconcile_startup` 把 running → blocked，不假装还在跑。
- 每步用**短事务**：起作业/写进度各用独立 `SessionLocal`，await 期间不持有 Session
  （SQLite + 长事务会被页面请求饿死）；`build_plan`（可能 ssh 查设备）/ 远端清理 / 回退
  放线程里跑，不堵事件循环。
- 所有“下一步”前先确认上一步的**证据 + 门禁 + 远端清理**，不靠 exit code。

边界（v2.1）：链内参数固定用任务/项目默认（device/timeout/criteria）；不自动采纳；
从 `/confirm`+`/start` 直接串起（auto_run flag）与多 attempt 编排属 Step 2b。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..models import (ActionJob, AgentJob, Loop, LoopAttempt, LoopAutoRun, Project,
                      Savepoint, Task)
from . import (action_runner, exec_actions, loop_budget, loop_state, optimization, optimize,
               remote_cancel, savepoints, workspace_lease)

logger = logging.getLogger(__name__)

#: 会话工厂：生产就是 app.db.SessionLocal；测试用临时库 sessionmaker 覆盖它，
#: 保证驱动任务与请求（dependency override）看到同一个库。
_session_factory = SessionLocal


def _session() -> Session:
    return _session_factory()


#: 自动链固定步骤顺序（report 不是动作，落盘报告）。
STEPS = ("build", "smoke", "test", "benchmark", "report")
ACTION_STEPS = ("build", "smoke", "test", "benchmark")
#: 步骤 → 进入的 phase（build/smoke 快速失败门禁；test/benchmark 全量验证）。
STEP_PHASE = {"build": "smoke_running", "smoke": "smoke_running",
              "test": "full_running", "benchmark": "full_running",
              "report": "awaiting_adoption"}
#: 轮询间隔（秒）：动作子进程本来就要几十秒到几分钟，1s 足够且不压库。
POLL_INTERVAL = 1.0
#: 步骤失败时回退租约 TTL（与 savepoints.restore 同口径）。
RESTORE_TTL = timedelta(minutes=15)
#: 允许启动/重试自动链的 phase（重试时候选/基线/预算由 preflight 再判）。
_START_PHASES = ("awaiting_verification", "blocked", "failed")


class AutoRunError(ValueError):
    """自动链前置/操作错误；`status` 供路由映射 HTTP 状态码。"""

    def __init__(self, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- 小工具


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _steps() -> list[dict]:
    return [{
        "step": step, "status": "pending", "job_id": None, "run_id": None,
        "gate": None, "error": None, "started_at": None, "ended_at": None,
        "cleanup": None,
    } for step in STEPS]


def _load_steps(run: LoopAutoRun) -> list[dict]:
    try:
        data = json.loads(run.steps_json or "[]")
    except (TypeError, ValueError):
        data = []
    if not isinstance(data, list) or not data:
        return _steps()
    known = {s.get("step") for s in data if isinstance(s, dict)}
    for step in STEPS:
        if step not in known:
            data.append({"step": step, "status": "pending", "job_id": None, "run_id": None,
                         "gate": None, "error": None, "started_at": None,
                         "ended_at": None, "cleanup": None})
    return data


def _set_step(run: LoopAutoRun, step: str, **fields) -> dict:
    # 时间列转 ISO（steps_json 要能 JSON 序列化；对外也统一用 ISO 字符串）。
    fields = {key: (_iso(value) if isinstance(value, datetime) else value)
              for key, value in fields.items()}
    steps = _load_steps(run)
    row = next(s for s in steps if s.get("step") == step)
    row.update(fields)
    run.steps_json = json.dumps(steps, ensure_ascii=False)
    run.updated_at = _now()
    return row


def view(db: Session, run: LoopAutoRun) -> dict:
    try:
        budget = json.loads(run.budget_json or "{}")
    except (TypeError, ValueError):
        budget = {}
    return {
        "id": run.id,
        "loop_id": run.loop_id,
        "status": run.status,
        "current_step": run.current_step,
        "current_action_job_id": run.current_action_job_id,
        "attempt_id": run.attempt_id,
        "baseline_sha256": run.baseline_sha256,
        "candidate_sha256": run.candidate_sha256,
        "steps": _load_steps(run),
        "budget": budget if isinstance(budget, dict) else {},
        "conclusion": next((s.get("conclusion") for s in _load_steps(run)
                            if s.get("step") == "report" and s.get("conclusion")), None),
        "error": run.error,
        "started_at": _iso(run.started_at),
        "ended_at": _iso(run.ended_at),
        "updated_at": _iso(run.updated_at),
    }


def latest(db: Session, loop_id: int) -> LoopAutoRun | None:
    return db.scalar(
        select(LoopAutoRun).where(LoopAutoRun.loop_id == loop_id)
        .order_by(LoopAutoRun.id.desc()).limit(1)
    )


def running_for_task(db: Session, task_id: int) -> LoopAutoRun | None:
    """该任务是否有运行中的自动链（回滚/手动动作/对话改码入口用它做互斥门禁）。"""
    return db.scalar(
        select(LoopAutoRun).where(
            LoopAutoRun.status == "running",
            LoopAutoRun.loop_id.in_(select(Loop.id).where(Loop.task_id == task_id)),
        ).order_by(LoopAutoRun.id.desc()).limit(1)
    )


def latest_view(db: Session, loop_id: int) -> dict | None:
    run = latest(db, loop_id)
    return view(db, run) if run is not None else None


# ---------------------------------------------------------------- 前置预检


def _task_of(db: Session, loop: Loop) -> Task:
    task = db.get(Task, loop.task_id)
    if task is None:
        raise AutoRunError("任务不存在", status=404)
    return task


def _workspace_of(db: Session, task: Task) -> Path | None:
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        return None
    return Path(project.workspace_path).expanduser()


def _issue(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def preflight(db: Session, loop: Loop) -> dict:
    """自动链启动前的一次性预检（不写库）。失败抛 `AutoRunError`（全部 409）。"""
    issues: list[dict] = []
    task = _task_of(db, loop)
    if not isinstance(loop.objective, dict):
        raise AutoRunError("E_AUTO_PHASE: 旧 Loop（无 objective）只读，不能跑自动链", status=409)
    if loop.status != "open":
        raise AutoRunError(f"E_AUTO_PHASE: 轮次状态是 {loop.status!r}，不能跑自动链", status=409)
    phase = loop_state.phase_of(loop)
    # awaiting_verification = 首次；blocked/failed = 取消/清理未确认/回滚后重试（候选是否还在
    # 由下面的候选 sha 检查判定；失败回滚过的工作区会明确报 E_AUTO_NO_CANDIDATE）。
    if phase not in _START_PHASES:
        raise AutoRunError(
            f"E_AUTO_PHASE: 当前 phase={phase}；自动链只能从「待验证」或可重试的"
            f" blocked/failed 启动",
            status=409,
        )
    running_auto = latest(db, loop.id)
    if running_auto is not None and running_auto.status == "running":
        raise AutoRunError("E_AUTO_BUSY: 本轮已有一条自动验证链在跑", status=409)
    agent_busy = db.scalar(
        select(AgentJob.id).where(AgentJob.task_id == task.id, AgentJob.status == "running")
        .limit(1)
    )
    action_busy = action_runner.running_for_task(task.id)
    if agent_busy is not None or action_busy:
        detail = []
        if agent_busy is not None:
            detail.append(f"agent#{agent_busy}")
        if action_busy:
            detail.append(f"action#{action_busy[0]}")
        raise AutoRunError(
            "E_AUTO_BUSY: 工作区还有运行中的作业（" + "、".join(detail) + "），等它结束再跑自动链",
            status=409,
        )

    # 候选必须明确：证据 after_sha（或最新成功 attempt 的候选 sha）存在且 = 当前工作区。
    evidence = loop.optimize if isinstance(loop.optimize, dict) else {}
    if not evidence or not evidence.get("changed") or evidence.get("guard_reverted"):
        raise AutoRunError(
            "E_AUTO_NO_CANDIDATE: 本轮还没有有效候选（agent 未改码 / 改动被 guard 回退），不能跑自动链",
            status=409,
        )
    candidate_sha = str(evidence.get("after_sha256") or "").strip()
    attempt = db.scalar(
        select(LoopAttempt).where(LoopAttempt.loop_id == loop.id,
                                  LoopAttempt.status == "succeeded")
        .order_by(LoopAttempt.id.desc()).limit(1)
    )
    if attempt is not None and attempt.candidate_sha256:
        candidate_sha = candidate_sha or attempt.candidate_sha256
    if not candidate_sha:
        raise AutoRunError("E_AUTO_NO_CANDIDATE: 候选 sha 缺失，无法绑定自动链证据", status=409)
    workspace = _workspace_of(db, task)
    if workspace is None or not workspace.is_dir():
        issues.append(_issue("E_AUTO_NO_CANDIDATE", "项目工作区不存在/未配置"))
    else:
        try:
            current_sha = savepoints.current_sha(
                workspace, op_name=task.op_name if evidence.get("evidence_scope") == "operator-v1" else None)
        except savepoints.SavepointError as e:
            current_sha = None
            issues.append(_issue("E_AUTO_NO_CANDIDATE", f"读不到工作区源码快照：{e}"))
        if current_sha is not None and current_sha != candidate_sha:
            issues.append(_issue(
                "E_AUTO_NO_CANDIDATE",
                "当前工作区代码不是 agent 产出的候选（工作区被改过）；先回到候选或重新跑一轮 agent",
            ))

    # G1：before 侧证据仍成立（否则 after 侧没有可信对照）。
    baseline_sha = None
    try:
        base = optimization.baseline_evidence(db, loop)
        baseline_sha = base.get("baseline_sha256")
        # G1 里的「工作区仍是基线」在 agent 改码后**必然**不成立（工作区=候选），
        # 那不是缺陷；这里只保留 before run / 环境 / 门禁类问题。
        g1_issues = [i for i in (base.get("issues") or [])
                     if i.get("code") != "E_BASELINE_SHA_MISMATCH"]
        if g1_issues:
            detail = "；".join(f"{i['code']}: {i['message']}" for i in g1_issues)
            issues.append(_issue("E_AUTO_BASELINE", f"基线证据未就绪：{detail}"))
    except Exception as e:  # noqa: BLE001 —— 判不了不能当成通过
        issues.append(_issue("E_AUTO_BASELINE", f"基线证据判定失败：{type(e).__name__}: {e}"))

    # 预算一次性预检：四个动作各还要有剩余次数（链中每步仍会走 allowed 审计）。
    limit = loop_budget.limit_for(loop)
    budget: dict[str, dict] = {}
    for action in ACTION_STEPS:
        used = loop_budget.used_for(db, loop.id, action)
        budget[action] = {"limit": limit, "used": used}
        if used >= limit:
            issues.append(_issue(
                "E_AUTO_BUDGET",
                f"{action} 预算已用完（{used}/{limit}）；自动链需要每个动作至少 1 次",
            ))

    # 四动作 plan 一次性预检（不查设备占用：设备在真正起动作时再查，避免启动阶段多次 ssh）。
    # smoke 硬前置延后：链严格按 build→smoke→test→benchmark 顺序跑，启动时还没有 smoke run，
    # 无条件要求会让自动链永远 409；真正到 test/benchmark 步骤时 `_prepare_step` 会再判。
    for action in ACTION_STEPS:
        try:
            plan = exec_actions.build_plan(db, task, action, check_device=False,
                                           defer_smoke=True)
        except ValueError as e:
            issues.append(_issue("E_AUTO_PLAN", f"{action} plan 失败：{e}"))
            continue
        if not plan.ready:
            issues.append(_issue("E_AUTO_PLAN",
                                 f"{action} 前置不齐：" + "；".join(plan.missing)))

    if issues:
        detail = "；".join(f"{i['code']}: {i['message']}" for i in issues)
        raise AutoRunError(f"自动链前置不满足，未启动：{detail}", status=409)
    return {
        "task": task,
        "attempt": attempt,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "budget": budget,
    }


# ---------------------------------------------------------------- 启动 / 取消 / 重启对账


def start(db: Session, loop: Loop) -> LoopAutoRun:
    """预检 → 建 auto_run 行 → phase CAS 进 smoke_running；由路由把它 schedule 起来。"""
    info = preflight(db, loop)
    run = LoopAutoRun(
        loop_id=loop.id,
        status="running",
        current_step="build",
        attempt_id=info["attempt"].id if info["attempt"] is not None else None,
        baseline_sha256=info["baseline_sha256"],
        candidate_sha256=info["candidate_sha256"],
        steps_json=json.dumps(_steps(), ensure_ascii=False),
        budget_json=json.dumps(info["budget"], ensure_ascii=False),
        started_at=_now(),
        updated_at=_now(),
    )
    db.add(run)
    db.flush()
    # phase CAS（awaiting_verification → smoke_running）+ active_job_id 空闲：并发/重复点击只有一个赢。
    if not loop_state.transition(db, loop, "smoke_running", event="auto_run_start",
                                 expect=_START_PHASES, require_idle=True,
                                 error=None, payload={"auto_run_id": run.id}):
        db.rollback()
        raise AutoRunError(
            "E_AUTO_BUSY: 轮次状态刚被其他请求改变（可能已有自动链在跑），请刷新后重试", status=409)
    db.commit()
    db.refresh(run)
    return run


#: 驱动任务的强引用（asyncio 只保弱引用，官方建议自己存一份，避免任务被 GC）。
_DRIVE_TASKS: set[asyncio.Task] = set()


def start_loop(loop_id: int) -> dict:
    """路由用：在线程里跑 preflight + 建行 + phase CAS（preflight 会算全量工作区快照，不能堵事件循环）。"""
    with _session() as db:
        loop = db.get(Loop, loop_id)
        if loop is None:
            raise AutoRunError("轮次不存在", status=404)
        run = start(db, loop)
        return view(db, run)


def schedule(run_id: int) -> None:
    """在当前事件循环里起驱动任务（路由 async def 里调用；进程重启即消失，靠 reconcile）。"""
    task = asyncio.create_task(_drive(run_id), name=f"loop-auto-run-{run_id}")
    _DRIVE_TASKS.add(task)
    task.add_done_callback(_DRIVE_TASKS.discard)


def cancel(db: Session, loop: Loop, *, reason: str | None = None) -> dict:
    """取消当前自动链：杀当前动作作业 + 远端清理 + status=cancelled / phase=blocked。"""
    run = latest(db, loop.id)
    if run is None or run.status != "running":
        raise AutoRunError("E_AUTO_BUSY: 本轮没有运行中的自动链", status=409)
    job = db.get(ActionJob, run.current_action_job_id) if run.current_action_job_id else None
    cancelled: dict = {}
    if job is not None:
        cancelled["action_job_id"] = job.id
        if job.status == "running":
            cancelled["local"] = action_runner.cancel(job.id)
        task = db.get(Task, job.task_id)
        if task is not None:
            cancelled["remote"] = remote_cancel.cancel_container(db, task, job.run_ref)
    run.status = "cancelled"
    run.error = (reason or "").strip() or "用户取消自动验证链（现场保留，需回退请走「否决并回退」）"
    run.ended_at = _now()
    run.updated_at = _now()
    # 页面一致性：把还在 running/pending 的当前步骤标 skipped 并清空作业指针
    # （否则会看到「已取消但某步 running」并指向已结束作业）。
    steps = _load_steps(run)
    for row in steps:
        if row.get("status") == "running":
            row.update({"status": "skipped", "ended_at": _iso(_now()),
                        "error": "用户取消自动验证链"})
    run.steps_json = json.dumps(steps, ensure_ascii=False)
    run.current_action_job_id = None
    if loop_state.phase_of(loop) in loop_state.RUNNING_PHASES:
        loop_state.transition(db, loop, "blocked", event="auto_run_cancel", force=True,
                              active_job_id=None, error=run.error,
                              payload={"auto_run_id": run.id}, commit=False)
    db.commit()
    db.refresh(run)
    return {"auto_run": view(db, run), "cancelled": cancelled}


def cancel_if_running(db: Session, loop: Loop, *, reason: str | None = None) -> dict | None:
    """loop 级取消（POST /loops/{id}/cancel）复用它：有自动链在跑就一并停掉。"""
    run = latest(db, loop.id)
    if run is None or run.status != "running":
        return None
    try:
        return cancel(db, loop, reason=reason)
    except AutoRunError:
        return None


def reconcile_startup(db: Session) -> int:
    """控制台重启：本地驱动任务已不在，running → blocked（别让页面以为还在自动跑）。"""
    rows = list(db.scalars(select(LoopAutoRun).where(LoopAutoRun.status == "running")))
    for run in rows:
        run.status = "blocked"
        run.error = "控制台重启/进程中断：自动链已停止；确认远端清理后可重新发起"
        run.ended_at = _now()
        run.updated_at = _now()
        steps = _load_steps(run)
        for row in steps:
            if row.get("status") == "running":
                row.update({"status": "skipped", "ended_at": _iso(_now()),
                            "error": "控制台重启/进程中断，自动链已停止"})
        run.steps_json = json.dumps(steps, ensure_ascii=False)
        run.current_action_job_id = None
    if rows:
        db.commit()
    return len(rows)


# ---------------------------------------------------------------- 驱动


async def _drive(run_id: int) -> None:
    """自动链主循环。异常一律收敛成 failed/blocked，不把状态机留在 running。"""
    status = "failed"
    error: str | None = None
    try:
        error, status = await _drive_steps(run_id)
    except asyncio.CancelledError:  # 进程退出/测试收尾：状态交给 reconcile_startup
        raise
    except Exception as exc:  # noqa: BLE001 —— 驱动异常不能静默
        logger.exception("自动链驱动异常（auto_run=%s）", run_id)
        error, status = f"自动链内部错误：{type(exc).__name__}: {exc}", "failed"
        await _finish_failed_phase(run_id, error)
    _finish(run_id, status=status, error=error)


def _finish(run_id: int, *, status: str, error: str | None) -> None:
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None:
            return
        if run.status != "running":
            return  # 已被取消/重启对账收尾，不要覆盖
        run.status = status
        run.error = error
        run.ended_at = _now()
        run.updated_at = _now()
        db.commit()


async def _finish_failed_phase(run_id: int, error: str) -> None:
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None:
            return
        loop = db.get(Loop, run.loop_id)
        if loop is None:
            return
        # 只从 running phase 收口；取消/否决已把 phase 改成 blocked/cancelled 时不要覆盖。
        loop_state.transition(db, loop, "failed", event="auto_run_error", force=False,
                              expect=("awaiting_verification", "smoke_running", "full_running"),
                              active_job_id=None, error=error[:2000],
                              payload={"auto_run_id": run_id}, commit=True)


def _set_phase(loop_id: int, to: str, *, event: str, error: str | None = None,
               force: bool = False, payload: dict | None = None,
               expect: tuple[str, ...] | None = None) -> bool:
    with _session() as db:
        loop = db.get(Loop, loop_id)
        if loop is None:
            return False
        if loop_state.phase_of(loop) == to:
            return True
        return loop_state.transition(db, loop, to, event=event, force=force,
                                     expect=expect, active_job_id=None, error=error,
                                     payload=payload)


async def _drive_steps(run_id: int) -> tuple[str | None, str]:
    """返回 `(error, status)`；status ∈ done/failed/blocked/cancelled。"""
    for step in ACTION_STEPS:
        state = _run_state(run_id)
        if state is None or state["status"] != "running":
            return None, "cancelled"
        entered = _enter_phase(run_id, step)
        if not entered:
            return (await _fail(run_id, step,
                                "E_AUTO_PHASE: 轮次 phase 在自动链执行中被改变（可能被取消/否决）",
                                status="blocked"), "blocked")
        outcome = await _run_action_step(run_id, step)
        if outcome is not None:
            return outcome
    state = _run_state(run_id)
    if state is None or state["status"] != "running":
        return None, "cancelled"
    return await _run_report_step(run_id)


def _run_state(run_id: int) -> dict | None:
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None:
            return None
        loop = db.get(Loop, run.loop_id)
        return {
            "run_id": run.id,
            "status": run.status,
            "loop_id": run.loop_id,
            "candidate_sha256": run.candidate_sha256,
            "loop_phase": loop_state.phase_of(loop) if loop is not None else None,
        }


def _enter_phase(run_id: int, step: str) -> bool:
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            return False
        loop = db.get(Loop, run.loop_id)
        if loop is None:
            return False
        want = STEP_PHASE[step]
        current = loop_state.phase_of(loop)
        if current == want:
            return True
        # 只允许在「本链会经过的 phase」之间移动；失败/终态一律停（交给阻塞分支）。
        if current not in ("awaiting_verification", "smoke_running", "full_running"):
            return False
        run.current_step = step
        run.updated_at = _now()
        db.commit()
        return loop_state.transition(db, loop, want, event=f"auto_run_{step}",
                                     force=False, active_job_id=None,
                                     payload={"auto_run_id": run.id})


def _prepare_step(task_id: int, loop_id: int, step: str) -> dict:
    """线程里做「起动作前」的重活：plan（可能 ssh 查设备）+ busy 复核 + 预算预留。

    这里失败都属于「资源/前置」类（不是动作跑完不通过），统一 blocked、不回退候选：
    修好设备/镜像/预算后可原地重试。
    """
    with _session() as db:
        task = db.get(Task, task_id)
        loop = db.get(Loop, loop_id)
        if task is None or loop is None:
            return {"ok": False, "status": "blocked", "error": "E_AUTO_PLAN: 任务/轮次不存在"}
        busy = action_runner.running_for_task(task.id)
        if busy:
            return {"ok": False, "status": "blocked",
                    "error": f"E_AUTO_BUSY: 任务已有动作作业 #{busy[0]} 在跑，自动链中止（可重试）"}
        try:
            plan = exec_actions.build_plan(db, task, step)
        except ValueError as e:
            return {"ok": False, "status": "blocked",
                    "error": f"E_AUTO_PLAN: {step} plan 失败：{e}"}
        if not plan.ready:
            return {"ok": False, "status": "blocked",
                    "error": f"E_AUTO_PLAN: {step} 前置不齐：" + "；".join(plan.missing)}
        try:
            budget = loop_budget.check_and_audit(db, task, loop, step)
        except loop_budget.BudgetExceeded as e:
            return {"ok": False, "status": "blocked", "error": f"E_AUTO_BUDGET: {e}"}
        return {"ok": True, "plan": plan, "audit_id": budget.get("audit_id"),
                # 比 action_runner 自身的子进程 timeout 多留 60s：先等它自己收尾，
                # 避免 orchestrator 先超时去回退（此时容器/租约可能还没清）。
                "timeout_s": plan.timeout_s + 180}


def _start_job(run_id: int, step: str, task_id: int, loop_id: int, prep: dict) -> dict:
    """起一次动作作业（在事件循环里同步调用，action_runner.start 需要 running loop）。

    **起完必须复查 run.status**：取消可能发生在 start 的阻塞窗口里（那时 cancel 看不到
    job id），这里补杀 + 回收预算，避免「已取消却真起了一个没人管的作业」。
    """
    with _session() as db:
        task = db.get(Task, task_id)
        loop = db.get(Loop, loop_id)
        run = db.get(LoopAutoRun, run_id)
        if task is None or loop is None or run is None or run.status != "running":
            loop_budget.void_audit(db, prep.get("audit_id"))
            return {"kind": "cancelled", "job_id": None}
        try:
            job = action_runner.start(db, task, prep["plan"], loop_id=loop.id,
                                      budget_audit_id=prep.get("audit_id"))
        except ValueError as e:
            loop_budget.void_audit(db, prep.get("audit_id"))
            return {"kind": "blocked", "error": f"E_AUTO_BUSY: {step} 启动前置不满足：{e}"}
        except Exception as e:  # noqa: BLE001
            loop_budget.void_audit(db, prep.get("audit_id"))
            return {"kind": "blocked",
                    "error": f"E_AUTO_PLAN: {step} 启动异常：{type(e).__name__}: {e}"}
        db.refresh(run)
        if run.status != "running":
            action_runner.cancel(job.id)
            loop_budget.void_audit(db, prep.get("audit_id"))
            db.commit()
            return {"kind": "cancelled", "job_id": job.id, "run_ref": job.run_ref}
        run.current_step = step
        run.current_action_job_id = job.id
        _set_step(run, step, status="running", job_id=job.id, started_at=_now())
        db.commit()
        return {"kind": "ok", "job_id": job.id, "timeout_s": prep["timeout_s"]}


async def _cancel_started_job(job_id: int, task_id: int) -> None:
    """取消一个已起但没人等的动作作业 + 远端清理（best-effort）。"""
    try:
        action_runner.cancel(job_id)
    except Exception:  # noqa: BLE001
        logger.exception("取消动作作业失败（job=%s）", job_id)
    run_ref = None
    with _session() as db:
        job = db.get(ActionJob, job_id)
        run_ref = job.run_ref if job is not None else None
    await _remote_cleanup(task_id, run_ref)


async def _wait_terminal(job_id: int, timeout_s: float) -> bool:
    """等动作作业到终态（取消/超时后给 _finalize 收尾的时间）。"""
    deadline = time.monotonic() + max(5.0, timeout_s)
    while time.monotonic() < deadline:
        with _session() as db:
            job = db.get(ActionJob, job_id)
            if job is None or job.status != "running":
                return True
        await asyncio.sleep(1.0)
    return False


async def _run_action_step(run_id: int, step: str) -> tuple[str | None, str] | None:
    """跑一个动作步骤；成功返回 None，失败返回 `(error, status)`（已做清理/回退/收尾）。"""
    task_id: int | None = None
    loop_id: int | None = None
    project_id: int | None = None
    state_error: str | None = None
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            return None, "cancelled"
        loop = db.get(Loop, run.loop_id)
        task = db.get(Task, loop.task_id) if loop is not None else None
        if loop is None or task is None:
            state_error = "E_AUTO_PLAN: 任务/轮次不存在"
        else:
            task_id, loop_id, project_id = task.id, loop.id, task.project_id
            if loop_state.phase_of(loop) not in ("awaiting_verification", "smoke_running",
                                                 "full_running"):
                state_error = "E_AUTO_PHASE: 轮次 phase 在自动链执行中被改变（可能被取消/否决）"
    if state_error is not None:
        return await _fail(run_id, step, state_error, status="blocked"), "blocked"
    if task_id is None or loop_id is None:
        return await _fail(run_id, step, "E_AUTO_PLAN: 任务/轮次不存在", status="blocked"), "blocked"

    prep = await asyncio.to_thread(_prepare_step, task_id, loop_id, step)
    if not prep.get("ok"):
        status = prep.get("status", "blocked")
        return await _fail(run_id, step, prep["error"], status=status), status

    started = _start_job(run_id, step, task_id, loop_id, prep)
    if started.get("kind") == "cancelled":
        if started.get("job_id"):
            await _cancel_started_job(started["job_id"], task_id)
        return None, "cancelled"
    if started.get("kind") == "blocked":
        return await _fail(run_id, step, started["error"], status="blocked"), "blocked"
    job_id, timeout_s = started["job_id"], started["timeout_s"]

    wait = await _wait_action(run_id, job_id, timeout_s)
    if wait == "cancelled":
        return None, "cancelled"
    if wait == "blocked":
        await _cancel_started_job(job_id, task_id)
        return await _fail(run_id, step, "E_AUTO_PLAN: 动作作业记录丢失，已中止",
                           status="blocked"), "blocked"
    timeout_hit = wait == "timeout"
    if timeout_hit:
        # orchestrator 的超时比子进程自身 timeout 多 60s；这里再等它收尾，收不了就 blocked，
        # 不改工作区（远端可能还有东西在跑）。
        terminal = await _wait_terminal(job_id, 60)
        if not terminal:
            await _cancel_started_job(job_id, task_id)
            return await _fail(run_id, step,
                               f"E_AUTO_TIMEOUT: {step} 超时且作业未收尾；已发取消信号，"
                               "确认远端清理后重试", status="blocked"), "blocked"

    # 真实 `_finalize` 还要落 run/门禁/candidate/释放租约；等租约释放再读结果与下一步。
    if project_id is not None and not await _wait_action_lease_clear(project_id, job_id):
        return await _fail(run_id, step,
                           "E_AUTO_CLEANUP: 动作作业收尾未完成（工作区租约未释放），已中止",
                           status="blocked"), "blocked"

    info: dict | None = None
    run_cancelled = False
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            run_cancelled = True
        else:
            job = db.get(ActionJob, job_id)
            if job is not None:
                info = {"run_id": job.run_id, "gate": job.gate_verdict,
                        "status": job.status, "error": job.error,
                        "task_id": job.task_id, "run_ref": job.run_ref}
    if run_cancelled:
        return None, "cancelled"
    if info is None:
        return await _fail(run_id, step, "E_AUTO_PLAN: 动作作业记录丢失",
                           status="blocked"), "blocked"

    # 每步都确认远端清理（失败/超时也先清，再决定是否回退 —— 容器还在跑时不能改本地代码）。
    cleanup = await _remote_cleanup(info["task_id"], info["run_ref"])
    if not cleanup.get("ok"):
        return await _fail(run_id, step,
                           f"E_AUTO_CLEANUP: {step} 远端清理未确认：" +
                           str(cleanup.get("why") or cleanup.get("output") or ""),
                           status="blocked"), "blocked"
    if timeout_hit:
        return await _fail(run_id, step,
                           f"E_AUTO_TIMEOUT: {step} 超时（{timeout_s}s）；已取消并确认清理",
                           job_id=job_id, run_id_val=info["run_id"],
                           gate=info["gate"]), "failed"
    ok = info["status"] == "ok" and info["gate"] == "pass"
    if not ok:
        why = info["error"] or ("作业状态 " + str(info["status"]) +
                                ("；门禁 " + str(info["gate"]) if info["gate"] else ""))
        return await _fail(run_id, step, f"{step} 未通过：{why}", job_id=job_id,
                           run_id_val=info["run_id"], gate=info["gate"]), "failed"

    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            return None, "cancelled"
        _set_step(run, step, status="ok", gate=info["gate"], run_id=info["run_id"],
                  ended_at=_now(), cleanup=cleanup)
        run.current_action_job_id = None
        db.commit()
    return None


async def _wait_action(run_id: int, job_id: int, timeout_s: int) -> str:
    """轮询作业终态；返回 `done` / `cancelled` / `blocked` / `timeout`。"""
    deadline = time.monotonic() + max(30, timeout_s)
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        with _session() as db:
            run = db.get(LoopAutoRun, run_id)
            if run is None or run.status != "running":
                return "cancelled"
            job = db.get(ActionJob, job_id)
            if job is None:
                return "blocked"
            if job.status != "running":
                return "done"
        if time.monotonic() > deadline:
            action_runner.cancel(job_id)
            return "timeout"


async def _wait_action_lease_clear(project_id: int, job_id: int, *,
                                   timeout_s: float = 20.0) -> bool:
    """等这次动作作业的租约被 `_finalize` 释放（= 作业真正收尾完成）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with _session() as db:
            lease = workspace_lease.current(db, project_id)
            if lease is None or lease.job_id != job_id:
                return True
        await asyncio.sleep(0.2)
    return False


async def _remote_cleanup(task_id: int, run_ref: str | None) -> dict:
    """远端清理确认（ssh docker rm -f；脚本正常结束时容器本应已删，这里只做确认）。

    放到线程里跑：`remote_cancel` 是同步 ssh subprocess（最长 25s），不能堵事件循环。
    """

    def work() -> dict:
        with _session() as db:
            task = db.get(Task, task_id)
            if task is None:
                return {"ok": False, "why": "任务不存在"}
            return remote_cancel.cancel_container(db, task, run_ref)

    try:
        return await asyncio.to_thread(work)
    except Exception as e:  # noqa: BLE001 —— 清理确认失败 = 不能进下一步
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}


async def _fail(run_id: int, step: str, error: str, *, job_id: int | None = None,
                run_id_val: int | None = None, gate: str | None = None,
                status: str = "failed") -> str:
    """步骤失败收尾：落 failed_candidate → 回退到 agent 改前 → phase failed/blocked。

    - 只有「动作真实失败」（status=failed）才回退；资源类/清理未确认/phase 冲突（blocked）
      保留现场，避免远端可能还在跑时改本地代码。
    - 取消可能与这里并发：进入不可逆段之前、写 phase 之前都复查 run.status；被取消就只清
      本步状态，不回退、不覆盖 phase（取消不静默回滚工作区）。
    """
    loop_id: int | None = None
    task_id: int | None = None
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None:
            return error
        loop = db.get(Loop, run.loop_id)
        loop_id = run.loop_id
        if loop is not None:
            task = db.get(Task, loop.task_id)
            task_id = task.id if task is not None else None
        if run.status != "running":
            # 已被取消/重启对账：把还在 running 的步骤标 skipped，不写失败/phase。
            steps = _load_steps(run)
            row = next((r for r in steps if r.get("step") == step), None)
            if row is not None and row.get("status") in ("running", "pending"):
                row.update({"status": "skipped", "ended_at": _iso(_now()),
                            "error": "已取消/停止，未执行完"})
                run.steps_json = json.dumps(steps, ensure_ascii=False)
                db.commit()
            return error
        _set_step(run, step, status="failed", error=error, gate=gate,
                  job_id=job_id, run_id=run_id_val, ended_at=_now())
        db.commit()

    if status == "failed" and loop_id is not None and task_id is not None:
        await asyncio.to_thread(_capture_failed_candidate, loop_id, task_id, step)
        if await asyncio.to_thread(_run_status, run_id) != "running":
            return error
        revert = await asyncio.to_thread(_rollback_to_before, loop_id, task_id)
        if await asyncio.to_thread(_run_status, run_id) != "running":
            return error
        if not revert.get("ok"):
            status = "blocked"
            error = error + "；回退未完成，现场可能不干净：" + str(
                revert.get("reason") or revert.get("revert_failed") or "")
    if loop_id is not None:
        # 不用 force：phase 已被取消/否决改成 blocked/cancelled 时 CAS 失败即停手。
        _set_phase(loop_id, "blocked" if status == "blocked" else "failed",
                   event="auto_run_failed", error=error, force=False,
                   expect=("awaiting_verification", "smoke_running", "full_running"),
                   payload={"auto_run_id": run_id, "step": step})

    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is not None and run.status == "running":
            run.status = status
            run.error = error
            run.ended_at = _now()
            run.updated_at = _now()
            run.current_action_job_id = None
            db.commit()
    return error


def _run_status(run_id: int) -> str | None:
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        return run.status if run is not None else None


def _capture_failed_candidate(loop_id: int, task_id: int, step: str) -> None:
    """失败候选存点（best-effort）：不落的话现场只剩 optimize_json 的 diff 文本。"""
    try:
        with _session() as db:
            task = db.get(Task, task_id)
            loop = db.get(Loop, loop_id)
            if task is None or loop is None:
                return
            project = db.get(Project, task.project_id)
            workspace = (Path(project.workspace_path).expanduser()
                         if project is not None and project.workspace_path else None)
            if workspace is None or not workspace.is_dir():
                return
            evidence = loop.optimize or {}
            sha = savepoints.current_sha(
                workspace, op_name=task.op_name if evidence.get("evidence_scope") == "operator-v1" else None)
            if sha is None:
                return
            baseline = db.scalar(
                select(Savepoint).where(Savepoint.loop_id == loop_id,
                                        Savepoint.kind == "baseline")
                .order_by(Savepoint.id.desc()).limit(1)
            )
            if baseline is not None and baseline.code_snapshot_sha256 == sha:
                return  # 已经回到基线，没有失败的候选可留
            savepoints.create_for_sha(db, task, sha, kind="failed_candidate",
                                      loop_id=loop_id,
                                      label=f"失败候选 · {step} · Loop #{loop_id}")
            db.commit()
    except Exception:  # noqa: BLE001 —— 留档失败不影响失败收尾
        logger.exception("失败候选存点失败（loop=%s, step=%s）", loop_id, step)


def _rollback_to_before(loop_id: int, task_id: int) -> dict:
    """把工作区回退到 agent 改前（loops.optimize 的 revert 快照），持 restore 租约。"""
    try:
        with _session() as db:
            task = db.get(Task, task_id)
            loop = db.get(Loop, loop_id)
            if task is None or loop is None:
                return {"ok": False, "reason": "任务/轮次不存在"}
            project = db.get(Project, task.project_id)
            workspace = (Path(project.workspace_path).expanduser()
                         if project is not None and project.workspace_path else None)
            if workspace is None or not workspace.is_dir():
                return {"ok": False, "reason": "工作区不存在"}
            with workspace_lease.hold(db, task, kind="restore", ttl=RESTORE_TTL,
                                      note=f"自动链失败回退 · Loop #{loop_id}"):
                ev = optimize.revert(db, loop, workspace)
            return {"ok": bool(ev.get("reverted_clean")),
                    "reverted_clean": bool(ev.get("reverted_clean")),
                    "revert_failed": ev.get("revert_failed") or []}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def _run_report(loop_id: int) -> dict:
    """线程里落盘报告（report 会读盘/perf-diff，不能堵事件循环）。"""
    with _session() as db:
        loop = db.get(Loop, loop_id)
        if loop is None:
            raise ValueError("轮次不存在")
        return optimization.report(db, loop, write=True)


def _candidate_savepoint_exists(loop_id: int, sha: str | None) -> bool:
    if not sha:
        return False
    with _session() as db:
        return db.scalar(
            select(Savepoint.id).where(Savepoint.loop_id == loop_id,
                                       Savepoint.kind == "candidate",
                                       Savepoint.code_snapshot_sha256 == sha).limit(1)
        ) is not None


async def _run_report_step(run_id: int) -> tuple[str | None, str]:
    """report：校验候选存点 → 落盘报告 → phase awaiting_adoption。

    候选存点由 `action_runner._maybe_candidate_savepoint` 在 benchmark 收尾时落（best-effort）；
    没落就不能报 done —— 否则 G7 采纳会永久 409 E_ACCEPT_NO_CANDIDATE，属于假成功。
    """
    loop_id: int | None = None
    candidate_sha: str | None = None
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            return None, "cancelled"
        loop = db.get(Loop, run.loop_id)
        if loop is None:
            return "E_AUTO_PHASE: 轮次不存在", "failed"
        loop_id = loop.id
        candidate_sha = run.candidate_sha256
        run.current_step = "report"
        _set_step(run, "report", status="running", started_at=_now())
        db.commit()

    if not await asyncio.to_thread(_candidate_savepoint_exists, loop_id, candidate_sha):
        error = ("E_AUTO_CANDIDATE: benchmark 成功但没有候选存点（sha="
                 f"{(candidate_sha or '')[:12]}…）；不能进入采纳，请重跑候选验证")
        with _session() as db:
            run = db.get(LoopAutoRun, run_id)
            if run is not None and run.status == "running":
                _set_step(run, "report", status="failed", error=error, ended_at=_now())
                run.status = "failed"
                run.error = error
                run.ended_at = _now()
                run.updated_at = _now()
                run.current_action_job_id = None
                db.commit()
        _set_phase(loop_id, "failed", event="auto_run_failed", error=error, force=False,
                   expect=("full_running", "smoke_running"),
                   payload={"auto_run_id": run_id, "step": "report"})
        return error, "failed"

    try:
        report = await asyncio.to_thread(_run_report, loop_id)
    except Exception as e:  # noqa: BLE001 —— 报告失败不能伪装成通过
        error = f"E_AUTO_REPORT: 报告落盘失败：{type(e).__name__}: {e}"
        with _session() as db:
            run = db.get(LoopAutoRun, run_id)
            if run is not None and run.status == "running":
                _set_step(run, "report", status="failed", error=error, ended_at=_now())
                run.status = "failed"
                run.error = error
                run.ended_at = _now()
                run.updated_at = _now()
                run.current_action_job_id = None
                db.commit()
        # 计划口径：报告失败停在 awaiting_adoption（禁止 accept 由 G7 兜底）。
        _set_phase(loop_id, "awaiting_adoption", event="auto_run_report_failed",
                   error=error, force=False, expect=("full_running", "smoke_running"),
                   payload={"auto_run_id": run_id})
        return error, "failed"

    written = report.get("written_to")
    summary = report.get("objective_summary") if isinstance(report.get("objective_summary"), dict) else {}
    perf_report = report.get("perf_diff") if isinstance(report.get("perf_diff"), dict) else {}
    conclusion = str(perf_report.get("conclusion") or summary.get("conclusion") or "")
    if not written:
        error = "E_AUTO_REPORT: 报告未落盘（write=true 未返回 written_to）"
        with _session() as db:
            run = db.get(LoopAutoRun, run_id)
            if run is not None and run.status == "running":
                _set_step(run, "report", status="failed", error=error, ended_at=_now())
                run.status = "failed"
                run.error = error
                run.ended_at = _now()
                run.updated_at = _now()
                run.current_action_job_id = None
                db.commit()
        _set_phase(loop_id, "awaiting_adoption", event="auto_run_report_failed",
                   error=error, force=False, expect=("full_running", "smoke_running"),
                   payload={"auto_run_id": run_id})
        return error, "failed"

    note = None
    if conclusion == "regressed":
        note = "性能对比结论 regressed：现场保留，可人工否决/回退（不能采纳）"
    with _session() as db:
        run = db.get(LoopAutoRun, run_id)
        if run is None or run.status != "running":
            return None, "cancelled"
        _set_step(run, "report", status="ok", gate=summary.get("verdict"),
                  conclusion=conclusion or None, ended_at=_now(), error=note)
        run.current_step = "report"
        run.current_action_job_id = None
        run.updated_at = _now()
        db.commit()
    # phase 用 CAS：取消/否决已把它改成 blocked/cancelled 时不要覆盖。
    _set_phase(loop_id, "awaiting_adoption", event="auto_run_done", force=False,
               expect=("full_running", "smoke_running"), payload={"auto_run_id": run_id})
    return None, "done"
