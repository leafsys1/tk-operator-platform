"""自动优化轮次状态机 + 事件（Step 1）。

契约见 `docs/plan-optimize-auto-v1.md` §5。核心目标：**不再用 runs/jobs 拓扑反推轮次状态**，
而是持久化 `loops.phase`，所有转移走 `state_version` CAS，并写 append-only `loop_events`。

用法约定：
- 读路径可调 `derive_phase()` 给历史数据兜底，但**不该在 GET 里写库**；
- 写路径用 `transition/start_agent/finish_agent`，`expect=` 不满足或 CAS 失败时返回 `None`，
  由调用方映射 409（并发/重复启动）；
- 状态机不跑真机、不改工作区，只写 `loops.phase*` 与 `loop_events`。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from ..models import Loop, LoopAttempt, LoopEvent, LoopRunLink, Run, Savepoint, utcnow

#: 非优化 Loop（设计/修复/历史数据）不参与自动优化状态机。
PHASES = (
    "idle",
    "baseline_pending",        # 已建轮次，基线未就绪
    "baseline_ready",          # before test+benchmark gate pass，允许改码
    "agent_running",           # agent 正在改码
    "awaiting_verification",   # 改码完成，等待 smoke/候选验证
    "smoke_running",           # 二期自动链：单例 smoke
    "full_running",            # 二期自动链：全量 test/benchmark
    "awaiting_adoption",       # 证据齐，等人工采纳/否决
    "accepted",
    "rejected",
    "failed",
    "blocked",                 # 重启/远端未清理/需要人工处理，禁止自动继续
    "cancelled",
)

RUNNING_PHASES = ("agent_running", "smoke_running", "full_running")
TERMINAL_PHASES = ("accepted", "rejected", "cancelled")

#: 普通转移的允许表；accept/reject 走 `force=True`（证据门禁已在别处判过）。
ALLOWED: dict[str, set[str]] = {
    "idle": {"baseline_pending", "failed", "blocked"},
    "baseline_pending": {"baseline_ready", "failed", "blocked", "accepted", "rejected"},
    "baseline_ready": {"agent_running", "awaiting_adoption", "failed", "blocked",
                       "accepted", "rejected"},
    "agent_running": {"awaiting_verification", "failed", "blocked", "accepted", "rejected"},
    "awaiting_verification": {"awaiting_adoption", "smoke_running", "full_running",
                              "failed", "blocked", "accepted", "rejected"},
    "smoke_running": {"full_running", "awaiting_verification", "failed", "blocked"},
    "full_running": {"awaiting_adoption", "failed", "blocked", "accepted", "rejected"},
    "awaiting_adoption": {"accepted", "rejected", "failed", "blocked"},
    "failed": {"baseline_pending", "baseline_ready", "agent_running", "smoke_running",
               "idle", "accepted", "rejected"},
    # blocked / failed 允许重试「自动验证链」（候选仍在工作区时）；具体能不能跑由
    # orchestrator preflight 的候选/基线/预算检查判定，状态机只负责不重复进入。
    "blocked": {"baseline_pending", "baseline_ready", "agent_running", "smoke_running",
                "failed", "accepted", "rejected"},
    "accepted": set(),
    "rejected": set(),
    "cancelled": set(),
}


#: running phase 退出时的 attempt 结论映射（attempt 是一次 agent 改码尝试）。
_ATTEMPT_DONE = {
    "awaiting_verification": "succeeded",
    "awaiting_adoption": "succeeded",
    "accepted": "succeeded",
    "rejected": "succeeded",
    "failed": "failed",
    "blocked": "blocked",
    "cancelled": "cancelled",
}


class StateConflict(ValueError):
    """状态转移冲突（并发/重复启动/当前 phase 不允许）。"""


_UNSET = object()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def phase_of(loop: Loop) -> str:
    """当前 phase（历史数据空值按 idle 处理）。"""
    value = str(loop.phase or "").strip()
    return value if value in PHASES else "idle"


def add_event(db: Session, loop: Loop | int, kind: str, *, phase: str | None = None,
              actor: str = "page", payload: dict | list | str | None = None) -> LoopEvent:
    """追加一条事件（只 flush，不 commit；由调用方决定事务边界）。"""
    loop_id = loop.id if isinstance(loop, Loop) else int(loop)
    body = None
    if payload is not None:
        body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    row = LoopEvent(loop_id=loop_id, kind=str(kind)[:40], phase=phase, actor=str(actor)[:40],
                    payload_json=body, created_at=_now())
    db.add(row)
    return row


def _latest_baseline(db: Session, loop_id: int) -> Savepoint | None:
    return db.scalar(
        select(Savepoint)
        .where(Savepoint.loop_id == loop_id, Savepoint.kind == "baseline")
        .order_by(Savepoint.id.desc()).limit(1)
    )


def open_attempt(db: Session, loop: Loop, job_id: int, *, conversation_id: int | None = None,
                 actor: str = "page") -> LoopAttempt:
    """改码开始：给这次 agent job 开一条 attempt（只 flush，不 commit）。"""
    seq = int(db.scalar(
        select(func.count()).select_from(LoopAttempt)
        .where(LoopAttempt.loop_id == loop.id)
    ) or 0) + 1
    baseline = _latest_baseline(db, loop.id)
    row = LoopAttempt(
        loop_id=loop.id,
        seq=seq,
        agent_job_id=job_id,
        conversation_id=conversation_id,
        baseline_savepoint_id=baseline.id if baseline is not None else None,
        baseline_sha256=baseline.code_snapshot_sha256 if baseline is not None else None,
        status="running",
        started_at=_now(),
    )
    db.add(row)
    add_event(db, loop.id, "attempt_start", phase="agent_running", actor=actor,
              payload={"attempt_seq": seq, "job_id": job_id,
                       "conversation_id": conversation_id})
    return row


def _close_attempt(db: Session, loop_id: int, status: str, *, error: str | None = None,
                   candidate_sha256: str | None = None,
                   payload: dict | None = None) -> int:
    """把该 Loop 还在 running 的 attempt 收尾（幂等；返回收尾行数）。"""
    rows = list(db.scalars(select(LoopAttempt).where(
        LoopAttempt.loop_id == loop_id, LoopAttempt.status == "running")))
    for row in rows:
        row.status = status
        row.error = (error or row.error or None)
        if candidate_sha256 is not None:
            row.candidate_sha256 = candidate_sha256
        if payload:
            row.summary_json = json.dumps(payload, ensure_ascii=False)
        row.ended_at = _now()
    return len(rows)


def attempts_view(db: Session, loop_id: int, *, limit: int = 20) -> list[dict]:
    rows = list(db.scalars(
        select(LoopAttempt).where(LoopAttempt.loop_id == loop_id)
        .order_by(LoopAttempt.seq.desc()).limit(max(1, min(limit, 100)))
    ))
    return [{
        "id": row.id,
        "seq": row.seq,
        "agent_job_id": row.agent_job_id,
        "conversation_id": row.conversation_id,
        "baseline_savepoint_id": row.baseline_savepoint_id,
        "baseline_sha256": row.baseline_sha256,
        "candidate_sha256": row.candidate_sha256,
        "status": row.status,
        "error": row.error,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
    } for row in rows]


def _apply(db: Session, loop_id: int, *, to: str,
           expect: tuple[str, ...] | None = None,
           version: int | None = None, require_idle: bool = False,
           active_job_id: int | None | object = _UNSET,
           error: str | None | object = _UNSET) -> bool:
    """CAS 更新 phase；rowcount==1 才算成功。"""
    values: dict = {"phase": to, "phase_updated_at": _now(),
                    "state_version": Loop.state_version + 1}
    if active_job_id is not _UNSET:
        values["active_job_id"] = active_job_id
    if error is not _UNSET:
        values["phase_error"] = error
    stmt = update(Loop).where(Loop.id == loop_id)
    if expect:
        stmt = stmt.where(Loop.phase.in_(list(expect)))
    if version is not None:
        stmt = stmt.where(Loop.state_version == version)
    if require_idle:
        stmt = stmt.where(Loop.active_job_id.is_(None))
    result = db.execute(stmt.values(**values))
    return (result.rowcount or 0) == 1


def transition(db: Session, loop: Loop, to: str, *, event: str = "phase",
               expect: tuple[str, ...] | None = None, force: bool = False,
               require_idle: bool = False, active_job_id: int | None | object = _UNSET,
               error: str | None | object = _UNSET, actor: str = "page",
               payload: dict | list | str | None = None, commit: bool = True) -> bool:
    """转移到一个新 phase；成功返回 True，并发/非法转移返回 False。

    - 默认按 `ALLOWED` 校验当前 phase；证据门禁已判过的情况（accept/reject）用 `force=True`；
    - CAS 以 `state_version` 为条件，避免两个请求同时从 baseline_ready 进入 agent_running。
    """
    if to not in PHASES:
        raise StateConflict(f"未知 phase：{to}")
    current = phase_of(loop)
    # 终态不可再迁移（accepted/rejected/cancelled）：否则 agent 收尾可能把已否决的轮次改回
    # awaiting_verification，或重复取消再覆盖状态。同态迁移（如 cancel 收尾）仍然允许。
    if current in TERMINAL_PHASES and to != current:
        return False
    if not force and to not in ALLOWED.get(current, set()):
        return False
    ok = _apply(db, loop.id, to=to, expect=expect or (current,), version=loop.state_version,
                require_idle=require_idle, active_job_id=active_job_id, error=error)
    if not ok:
        # 注意：**不要在这里 rollback**。transition 跑在调用方的事务里，CAS 失败本身没有写入，
        # 但 db.rollback() 会把调用方此前挂起的修改（loop.optimize_json / job 状态 / 版本升级）
        # 一起丢掉，造成「接口 200 但什么都没落库」。由调用方决定 rollback + 报 409。
        db.expire(loop)
        return False
    add_event(db, loop.id, event, phase=to, actor=actor, payload=payload)
    # 离开 running phase = 这次 attempt 结束（无论成功/失败/取消/重启对账），
    # 统一在状态机里收尾，避免每个调用方各写一遍（漏一处 attempt 就永远 running）。
    if current in RUNNING_PHASES and to not in RUNNING_PHASES:
        resolved_error = None if error is _UNSET else error
        attempt_status = _ATTEMPT_DONE.get(to, "failed")
        _close_attempt(
            db, loop.id, attempt_status,
            error=resolved_error if isinstance(resolved_error, str) else None,
            candidate_sha256=(payload.get("after_sha256")
                              if isinstance(payload, dict) else None),
            payload=payload if isinstance(payload, dict) else None,
        )
    if commit:
        db.commit()
        db.refresh(loop)
    return True


def start_agent(db: Session, loop: Loop, job_id: int, *, conversation_id: int | None = None,
                actor: str = "page", commit: bool = False) -> bool:
    """改码开始：baseline_ready/等可重试 phase → agent_running，并占用 active_job_id。

    `conversation_id` 只用于 attempt 留痕（对话化启动时记录是哪个会话）。
    """
    allowed = ("idle", "baseline_pending", "baseline_ready", "failed", "blocked",
               "awaiting_verification", "awaiting_adoption")
    # force=True：allowed 是显式白名单（含 idle/历史 phase），不走 ALLOWED 的“当前状态”表；
    # 并发安全仍由 state_version CAS + active_job_id IS NULL 保证。
    ok = transition(db, loop, "agent_running", event="start", expect=allowed, force=True,
                    require_idle=True, active_job_id=job_id, error=None,
                    actor=actor, payload={"job_id": job_id}, commit=False)
    if not ok:
        return False
    open_attempt(db, loop, job_id, conversation_id=conversation_id, actor=actor)
    if commit:
        db.commit()
        db.refresh(loop)
    return True


def finish_agent(db: Session, loop_id: int, evidence: dict, *, commit: bool = True) -> str:
    """改码结束：有有效候选 → awaiting_verification；否则 failed。返回新 phase。"""
    loop = db.get(Loop, loop_id)
    if loop is None:
        return "failed"
    ok = bool(evidence.get("status") == "done" and evidence.get("changed")
              and not evidence.get("guard_reverted"))
    to = "awaiting_verification" if ok else "failed"
    error = None if ok else str(evidence.get("guard_reason") or evidence.get("error")
                                or "agent 未产出有效候选")
    moved = transition(db, loop, to, event="agent_done", force=True, active_job_id=None,
                       error=error, payload={
                           "changed": bool(evidence.get("changed")),
                           "guard_reverted": bool(evidence.get("guard_reverted")),
                           "after_sha256": evidence.get("after_sha256"),
                       }, commit=commit)
    if not moved:
        # 并发取消/否决已把轮次推进到终态：不要覆盖它，返回真实 phase 让上层按现状处理。
        db.expire(loop)
        return phase_of(loop)
    return to


def derive_phase(db: Session, loop: Loop) -> str:
    """给历史/旧数据兜底：从现有证据推导一个 phase（不写库）。"""
    if not isinstance(loop.objective, dict):
        if loop.status == "closed":
            return "rejected" if (loop.conclusion or "").startswith("rejected") else "accepted"
        return "idle"
    if loop.status == "closed":
        return "rejected" if (loop.conclusion or "").startswith("rejected") else "accepted"
    current = phase_of(loop)
    if current in RUNNING_PHASES and loop.active_job_id is not None:
        return current
    evidence = loop.optimize if isinstance(loop.optimize, dict) else None
    if evidence:
        if evidence.get("status") == "failed":
            return "failed"
        if evidence.get("changed") and not evidence.get("guard_reverted"):
            candidate = db.scalar(
                select(Savepoint.id)
                .where(Savepoint.loop_id == loop.id, Savepoint.kind == "candidate")
                .limit(1)
            )
            if candidate is not None:
                after_sha = str(evidence.get("after_sha256") or "")
                has_pair = db.scalar(
                    select(Run.id)
                    .where(Run.status == "ok", Run.code_snapshot_sha256 == after_sha,
                           Run.action.in_(("test", "benchmark")),
                           or_(Run.loop_id == loop.id,
                               Run.id.in_(select(LoopRunLink.run_id).where(
                                   LoopRunLink.loop_id == loop.id))))
                    .limit(1)
                )
                return "awaiting_adoption" if has_pair else "awaiting_verification"
            return "awaiting_verification"
    # 基线就绪用 Step 0 的门禁判定（本地 import 避免循环依赖）
    try:
        from . import optimization
        if optimization.baseline_evidence(db, loop).get("ready"):
            return "baseline_ready"
    except Exception:  # noqa: BLE001 —— 派生失败按未就绪处理
        pass
    return "baseline_pending"


def sync_phase(db: Session, loop: Loop, *, event: str = "sync", commit: bool = True) -> str:
    """把“证据推导出来的 phase”写回库（只在变化时写事件）。"""
    derived = derive_phase(db, loop)
    current = phase_of(loop)
    if derived == current:
        return current
    if current in RUNNING_PHASES and loop.active_job_id is not None:
        return current
    # blocked 是「需要人工处理」的粘性状态（取消自动链/远端未清理/重启中断）：动作作业收尾时的
    # 自动 sync 不得把它改回 awaiting_verification/failed（真实路径：cancel 后被杀作业的
    # _finalize 仍会 sync_phase）。解除 blocked 只能走显式操作（重试/取消/否决等）。
    if current == "blocked":
        return current
    if not transition(db, loop, derived, event=event, force=True, commit=commit):
        # CAS 冲突（别的请求刚改过）：不回滚调用方事务，保持现状返回。
        db.expire(loop)
        return phase_of(loop)
    return derived


def reconcile_startup(db: Session) -> int:
    """控制台重启：所有 running phase → blocked（本地进程已不在，禁止自动继续）。"""
    rows = list(db.scalars(select(Loop).where(Loop.phase.in_(RUNNING_PHASES))))
    for loop in rows:
        transition(db, loop, "blocked", event="recover", force=True, active_job_id=None,
                   error="控制台重启/进程中断，自动链停在 blocked；确认远端清理后可重试",
                   payload={"from": loop.phase}, commit=False)
    # 兜底：phase 已不是 running 但 attempt 仍 running（崩溃在两次写之间）也要收尾。
    stale = list(db.scalars(select(LoopAttempt).where(LoopAttempt.status == "running")))
    for row in stale:
        row.status = "blocked"
        row.error = "控制台重启/进程中断，本次尝试不再继续"
        row.ended_at = _now()
    if rows or stale:
        db.commit()
    return len(rows)


def view(db: Session, loop: Loop, *, event_limit: int = 20) -> dict:
    events = list(db.scalars(
        select(LoopEvent).where(LoopEvent.loop_id == loop.id)
        .order_by(LoopEvent.id.desc()).limit(max(1, min(event_limit, 100)))
    ))
    return {
        "loop_id": loop.id,
        "phase": phase_of(loop),
        "derived_phase": derive_phase(db, loop),
        "state_version": int(loop.state_version or 0),
        "active_job_id": loop.active_job_id,
        "error": loop.phase_error,
        "updated_at": loop.phase_updated_at.isoformat() if loop.phase_updated_at else None,
        "events": [
            {"id": e.id, "kind": e.kind, "phase": e.phase, "actor": e.actor,
             "payload": json.loads(e.payload_json) if e.payload_json else None,
             "created_at": e.created_at.isoformat() if e.created_at else None}
            for e in events
        ],
    }
