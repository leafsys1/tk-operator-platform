"""P0-2 对话页：会话 / 消息（引擎复用 `services/agent_runner`，见 `services/conversations.py`）。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AgentJob, Attachment, Message, Task
from ..schemas import ConversationCreate, MessageIn
from ..services import conversations as svc

router = APIRouter(prefix="/api", tags=["conversations"])


def _view(db: Session, conv, *, with_messages: bool = False) -> dict:
    data = {
        "id": conv.id,
        "project_id": conv.project_id,
        "task_id": conv.task_id,
        "task_title": (db.get(Task, conv.task_id).title if conv.task_id and db.get(Task, conv.task_id) else None),
        "dsh_session_id": conv.dsh_session_id,
        "title": conv.title,
        "mode": conv.mode,
        "message_count": conv.message_count or 0,
        "tokens": conv.tokens or 0,
        "cost": conv.cost,
        "created_at": conv.created_at,
        "last_active_at": conv.last_active_at,
    }
    # 前端刷新/切页后据此接回运行中的作业（配合 GET /api/agent-jobs/{id}/events）。
    # 不按 kind 过滤：自动优化作业（kind=优化）也挂在会话上，刷新后同样要能接回/显示运行中。
    data["running_job"] = db.scalar(
        select(AgentJob.id).where(AgentJob.conversation_id == conv.id,
                                  AgentJob.status == "running")
        .order_by(AgentJob.id.desc()).limit(1)
    )
    data["attachments"] = [
        {"id": a.id, "filename": a.filename, "path": a.path, "size_bytes": a.size_bytes, "sha256": a.sha256}
        for a in db.query(Attachment).filter(Attachment.conversation_id == conv.id).order_by(Attachment.id)
    ]
    if with_messages:
        rows = []
        for m in sorted(conv.messages, key=lambda x: x.id):
            changes = None
            if m.changes_json:
                try:
                    changes = json.loads(m.changes_json)
                except (ValueError, TypeError):
                    changes = None
            rows.append({"id": m.id, "role": m.role, "text": m.text, "job_id": m.job_id,
                         "created_at": m.created_at, "changes": changes})
        data["messages"] = rows
    return data


@router.get("/conversations")
def list_conversations(task_id: int | None = None, project_id: int | None = None,
                       limit: int = 50, db: Session = Depends(get_db)) -> list[dict]:
    return [_view(db, c) for c in svc.list_for(db, task_id=task_id, project_id=project_id, limit=limit)]


@router.post("/conversations", status_code=201)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)) -> dict:
    try:
        conv = svc.create(db, payload.task_id, title=payload.title, mode=payload.mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _view(db, conv)


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: int, db: Session = Depends(get_db)) -> dict:
    conv = svc.load(db, conv_id, with_messages=True)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    return _view(db, conv, with_messages=True)


@router.post("/conversations/{conv_id}/messages", status_code=201)
async def send_message(conv_id: int, payload: MessageIn, db: Session = Depends(get_db)) -> dict:
    """发一条消息：立刻返回 `job_id`；过程订阅 `GET /api/agent-jobs/{job_id}/events`（SSE）。"""
    conv = svc.load(db, conv_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    try:
        job = svc.send(db, conv, payload.text)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {"conversation_id": conv.id, "job_id": job.id, "dsh_session_id": conv.dsh_session_id}


@router.post("/conversations/{conv_id}/attachments", status_code=201)
async def upload_attachment(conv_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    """上传附件（落到工作区 `.opx-chat/conv-<id>/attachments/`，发送时自动带进 prompt）。"""
    conv = svc.load(db, conv_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    # 分块读取并边读边计数：超限立刻拒绝，避免整个请求体先全量进内存。
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > svc.ATTACH_MAX_BYTES:
            raise HTTPException(413, "附件超过 20MB 上限")
        chunks.append(chunk)
    try:
        row = svc.save_attachment(db, conv, svc.workspace_of(db, conv), file.filename or "upload.bin",
                                  b"".join(chunks))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": row.id, "filename": row.filename, "path": row.path,
            "size_bytes": row.size_bytes, "sha256": row.sha256}


@router.delete("/conversations/{conv_id}/attachments/{attachment_id}", status_code=204)
def delete_attachment(conv_id: int, attachment_id: int, db: Session = Depends(get_db)) -> None:
    """删除一个附件（DB 行 + 磁盘文件；文件已不存在也成功）。"""
    conv = svc.load(db, conv_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    try:
        workspace = svc.workspace_of(db, conv)
    except ValueError:
        workspace = None  # 工作区没了也要能删掉 DB 行
    if not svc.delete_attachment(db, conv, workspace, attachment_id):
        raise HTTPException(404, "附件不存在")


@router.post("/conversations/{conv_id}/messages/{message_id}/revert")
def revert_message(conv_id: int, message_id: int, db: Session = Depends(get_db)) -> dict:
    """回退某一轮的改动（按该条助手消息的 changes_json）。"""
    conv = svc.load(db, conv_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    msg = db.get(Message, message_id)
    if msg is None or msg.conversation_id != conv.id:
        raise HTTPException(404, "消息不存在")
    try:
        return svc.revert_message(db, conv, msg, svc.workspace_of(db, conv))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: int, db: Session = Depends(get_db)) -> None:
    conv = svc.load(db, conv_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    try:
        svc.delete(db, conv)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
