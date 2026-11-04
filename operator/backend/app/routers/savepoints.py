"""版本存档（savepoints v1）API：列表 / 手动存档 / 详情 / 对比 / 回滚 / 下载 / 删除。

契约见 `docs/plan-savepoints-v1.md` §4（字段名与错误语义逐字对齐）。归档本体在
`services/savepoints.py`；本文件只做存在性校验与错误码映射。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import update
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Project, Savepoint, Task
from ..schemas import SavepointCreate, SavepointRestore
from ..services import savepoints

router = APIRouter(prefix="/api", tags=["savepoints"])


def _get_task(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return task


def _get_savepoint(db: Session, savepoint_id: int) -> Savepoint:
    sp = db.get(Savepoint, savepoint_id)
    if sp is None:
        raise HTTPException(404, "存档不存在")
    return sp


def _task_of(db: Session, sp: Savepoint) -> Task:
    task = db.get(Task, sp.task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return task


def _workspace(db: Session, task: Task) -> Path | None:
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        return None
    return Path(project.workspace_path).expanduser()


def _create_error(e: savepoints.SavepointError) -> HTTPException:
    """建/对比/回滚的错误码映射：忙碌、归档损坏 → 409；其余（参数/工作区/过大）→ 400。"""
    status = 409 if isinstance(e, (savepoints.SavepointBusyError,
                                   savepoints.SavepointArchiveError)) else 400
    return HTTPException(status, str(e))


@router.get("/tasks/{task_id}/savepoints")
def list_savepoints(task_id: int, limit: int = Query(50, ge=1, le=200),
                    loop_id: int | None = None,
                    db: Session = Depends(get_db)) -> list[dict]:
    """任务的存档列表（`created_at desc, id desc`；可按优化轮次 `loop_id` 过滤）。

    `in_sync` 全量口径，一次请求只算一次。
    """
    task = _get_task(db, task_id)
    return savepoints.list_for_task(db, task, limit=limit, loop_id=loop_id)


@router.post("/tasks/{task_id}/savepoints", status_code=201)
def create_savepoint(task_id: int, payload: SavepointCreate,
                     db: Session = Depends(get_db)) -> dict:
    """对当前工作区打包并建一个 `kind=manual` 存档。"""
    task = _get_task(db, task_id)
    try:
        sp = savepoints.create(db, task, label=payload.label, note=payload.note)
        db.commit()
    except savepoints.SavepointError as e:
        db.rollback()
        raise _create_error(e) from e
    db.refresh(sp)
    return savepoints.out(db, sp, workspace=_workspace(db, task))


@router.get("/savepoints/{savepoint_id}")
def get_savepoint(savepoint_id: int, db: Session = Depends(get_db)) -> dict:
    sp = _get_savepoint(db, savepoint_id)
    task = _task_of(db, sp)
    return savepoints.out(db, sp, workspace=_workspace(db, task))


@router.get("/savepoints/{savepoint_id}/diff")
def savepoint_diff(savepoint_id: int, db: Session = Depends(get_db)) -> dict:
    """对比存档代码与当前工作区（任务/共享范围、unified diff、全量 in_sync）。"""
    sp = _get_savepoint(db, savepoint_id)
    task = _task_of(db, sp)
    try:
        return savepoints.diff(db, sp, task)
    except savepoints.SavepointError as e:
        raise _create_error(e) from e


@router.post("/savepoints/{savepoint_id}/restore")
def restore_savepoint(savepoint_id: int, payload: SavepointRestore,
                      db: Session = Depends(get_db)) -> dict:
    """回滚到该存档：`scope=task` 只回滚任务范围；`full` 全工作区回滚。"""
    sp = _get_savepoint(db, savepoint_id)
    task = _task_of(db, sp)
    try:
        return savepoints.restore(db, sp, task, scope=payload.scope, note=payload.note)
    except savepoints.SavepointError as e:
        db.rollback()
        raise _create_error(e) from e


@router.get("/savepoints/{savepoint_id}/archive")
def download_savepoint_archive(savepoint_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """下载存档的源码归档（路径必须在归档根下，防 DB 被篡改后任意读文件）。"""
    sp = _get_savepoint(db, savepoint_id)
    try:
        path = savepoints.get_archive_path(sp)
    except savepoints.SavepointError as e:
        raise HTTPException(404, str(e)) from e
    if not path.is_file():
        raise HTTPException(404, f"存档 #{sp.id} 的归档文件缺失")
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=f"savepoint-{sp.id}-{(sp.code_snapshot_sha256 or '')[:12]}.tar.gz",
    )


@router.delete("/savepoints/{savepoint_id}", status_code=204)
def delete_savepoint(savepoint_id: int, db: Session = Depends(get_db)) -> Response:
    """只删行，**不删归档**（内容寻址、可能被别的行/任务引用；GC 放 v2）。

    该存档作为「急救目标」被 rescue 行引用时，只断引用（`parent_id=NULL`）—— 不连坐删
    急救存档，否则「回滚回去」的证据链会断（SQLite 默认没开外键，必须显式做）。
    """
    sp = _get_savepoint(db, savepoint_id)
    db.execute(update(Savepoint).where(Savepoint.parent_id == sp.id).values(parent_id=None))
    db.delete(sp)
    db.commit()
    return Response(status_code=204)
