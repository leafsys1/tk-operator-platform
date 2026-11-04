"""任务版本 API（任务生命周期 v2）：列表 / 手工小版本 / 证据豁免。

契约见 `docs/impl-contract-v2.md` §3.4；语义见 `docs/plan-task-lifecycle-v2.md` §4。
版本证据由 `services/versions.evidence` 统一判定（交付门禁复用同一份口径）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task, TaskVersion
from ..services import versions

router = APIRouter(prefix="/api", tags=["task-lifecycle-v2"])


class VersionCreate(BaseModel):
    label: str | None = None
    note: str | None = None


class WaiverCreate(BaseModel):
    actions: list[str]
    reason: str
    job_id: int | None = None


def _view(row: TaskVersion | None, current_id: int | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row.id,
        "task_id": row.task_id,
        "major": row.major,
        "minor": row.minor,
        "label": f"v{row.major}.{row.minor}",
        "source": row.source,
        "spec_version_id": row.spec_version_id,
        "code_snapshot_sha256": row.code_snapshot_sha256,
        "suite_ref": row.suite_ref,
        "loop_id": row.loop_id,
        "note": row.note,
        "label_text": row.label,
        "delivered_at": row.delivered_at.isoformat() if row.delivered_at else None,
        "is_current": row.id == current_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _payload(db: Session, task: Task) -> dict:
    cur = versions.current(db, task)
    return {
        "task_id": task.id,
        "current": _view(cur, cur.id if cur else None),
        "history": [_view(row, cur.id if cur else None) for row in versions.history(db, task.id)],
        "evidence": versions.evidence(db, task),
    }


@router.get("/tasks/{task_id}/versions")
def list_versions(task_id: int, db: Session = Depends(get_db)) -> dict:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return _payload(db, task)


@router.post("/tasks/{task_id}/versions", status_code=201)
def create_version(task_id: int, payload: VersionCreate, db: Session = Depends(get_db)) -> dict:
    """手工小版本（交付前非优化的代码变更）：小版本 +1，证据随之失效、需重跑。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task.current_spec_version_id is None:
        raise HTTPException(409, "任务还没有规格版本，先定版")
    versions.bump_minor(db, task, source="manual",
                        label=(payload.label or "").strip() or None,
                        note=(payload.note or "").strip() or None)
    db.commit()
    return _payload(db, task)


@router.post("/tasks/{task_id}/versions/waiver")
def create_waiver(task_id: int, payload: WaiverCreate, db: Session = Depends(get_db)) -> dict:
    """对话确认的证据豁免：把某几个动作记为「用户确认无需重跑」（写当前版本 waivers_json）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    reason = (payload.reason or "").strip()
    if not reason:
        raise HTTPException(400, "reason 不能为空（对话确认必须留原因）")
    if not payload.actions:
        raise HTTPException(400, "actions 不能为空")
    try:
        out = versions.add_waiver(db, task, actions=payload.actions, reason=reason, job_id=payload.job_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    db.commit()
    return {**out, "evidence": versions.evidence(db, task)}
