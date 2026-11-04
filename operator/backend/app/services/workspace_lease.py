"""DB 级任务工作区租约（自动优化 Step 1）。

为什么需要：agent 作业（对话/优化）与动作作业（test/benchmark）各自只查自己的表，
benchmark 跑着时仍能在对话里改代码；进程内锁也覆盖不了重启/多 worker。这里用
`workspace_leases.project_id` 唯一索引做单飞，调用方必须在**真正动手前** acquire、
作业结束/失败时 release。

约定：
- `acquire()` 会在失败时抛 `LeaseBusy`，调用方映射 409；
- `acquire()` 只 `flush`，不 commit —— 与建 job/写消息放在同一个事务里；
- 过期或对应 job 已经不在 running 的租约视为 stale，可被抢占；
- 进程重启时本地作业必然不在，`reconcile_startup()` 清空租约（远端容器清理另属 Step 2）。
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import ActionJob, AgentJob, Task, WorkspaceLease, utcnow

#: 租约 TTL：作业超时上限远小于 12h；过期即可被下一个请求抢占。
LEASE_TTL = timedelta(hours=12)


class LeaseBusy(ValueError):
    """工作区已被其他作业占用。"""

    def __init__(self, message: str, *, holder: WorkspaceLease | None = None) -> None:
        super().__init__(message)
        self.holder = holder


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _job_running(db: Session, kind: str, job_id: int | None) -> bool:
    if job_id is None:
        return False
    model = {"agent": AgentJob, "action": ActionJob}.get(kind)
    if model is None:
        return False
    row = db.get(model, job_id)
    return row is not None and row.status == "running"


def _is_stale(db: Session, lease: WorkspaceLease) -> bool:
    expires = _as_utc(lease.expires_at)
    if expires is not None and expires <= _now():
        return True
    if lease.job_id is None:
        return False
    return not _job_running(db, lease.kind, lease.job_id)


def acquire(db: Session, task: Task, *, kind: str, job_id: int | None = None,
            ttl: timedelta = LEASE_TTL, note: str | None = None) -> WorkspaceLease:
    """占用任务工作区；返回租约行。失败抛 `LeaseBusy`。只 flush，不 commit。"""
    if kind not in ("agent", "action", "restore"):
        raise ValueError(f"未知租约类型：{kind}")
    now = _now()
    existing = db.scalar(select(WorkspaceLease).where(
        WorkspaceLease.project_id == task.project_id))
    if existing is not None:
        if (job_id is not None and existing.job_id == job_id and existing.kind == kind):
            return existing  # 同一作业重入
        if _is_stale(db, existing):
            # 就地接管（不 delete+insert）：同一 project_id 的行复用，避免 PK 复用/身份冲突。
            existing.task_id = task.id
            existing.kind = kind
            existing.job_id = job_id
            existing.acquired_at = now
            existing.expires_at = now + ttl
            existing.note = note
            db.flush()
            return existing
        raise LeaseBusy(
            f"项目工作区正被 {existing.kind} 作业 #{existing.job_id} 占用"
            f"（task #{existing.task_id}），等它结束再发",
            holder=existing,
        )
    row = WorkspaceLease(project_id=task.project_id, task_id=task.id, kind=kind,
                         job_id=job_id, acquired_at=now, expires_at=now + ttl, note=note)
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError as e:  # 两个请求同时抢唯一 project_id
        raise LeaseBusy("项目工作区刚被其他作业占用，请重试") from e
    return row


def release(db: Session, *, project_id: int | None = None, task_id: int | None = None,
            job_id: int | None = None, kind: str | None = None,
            commit: bool = True) -> int:
    """释放租约（按条件删除）；返回删除行数。"""
    stmt = delete(WorkspaceLease)
    if project_id is not None:
        stmt = stmt.where(WorkspaceLease.project_id == project_id)
    if task_id is not None:
        stmt = stmt.where(WorkspaceLease.task_id == task_id)
    if job_id is not None:
        stmt = stmt.where(WorkspaceLease.job_id == job_id)
    if kind is not None:
        stmt = stmt.where(WorkspaceLease.kind == kind)
    result = db.execute(stmt)
    # 注意：这里**不要** expire_all —— 调用方常在同一事务里还挂着 optimize_json 等
    # pending 修改，expire 会把它们丢掉（Step 1 集成时踩过）。身份冲突只是 SAWarning。
    if commit:
        db.commit()
    return int(result.rowcount or 0)


def release_for_task(db: Session, task_id: int, *, commit: bool = True) -> int:
    return release(db, task_id=task_id, commit=commit)


def current(db: Session, project_id: int) -> WorkspaceLease | None:
    return db.scalar(select(WorkspaceLease).where(WorkspaceLease.project_id == project_id))


@contextlib.contextmanager
def hold(db: Session, task: Task, *, kind: str, note: str | None = None,
         ttl: timedelta = LEASE_TTL) -> Iterator[WorkspaceLease]:
    """占用 → 用 → 无论成功失败都释放（写工作区的一次性操作专用，如回滚/回退）。

    失败时只删自己这条租约（按 project + kind），不会误删同时段别人的 agent/action 租约。
    """
    lease = acquire(db, task, kind=kind, note=note, ttl=ttl)
    try:
        yield lease
    finally:
        release(db, project_id=task.project_id, kind=kind)


def reconcile_startup(db: Session) -> int:
    """控制台重启：本地作业都不在了，清空全部租约；返回清理数量。"""
    result = db.execute(delete(WorkspaceLease))
    db.commit()
    return int(result.rowcount or 0)
