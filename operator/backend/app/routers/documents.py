"""任务文档 + 设计文档模板 + 用例组确认 API（任务生命周期 v2 §3.1/§3.2/§3.3）。

本 router 收敛三类契约接口（都挂在 `/api` 下）：
- `GET /api/templates/operator-design-doc?format=docx|md` —— 模板下载（FileResponse）；
- `POST/GET/DELETE /api/tasks/{task_id}/documents...` —— 文档上传 / 列表 / 下载 / 删除；
- `POST /api/tasks/{task_id}/cases/confirm` + `GET .../cases/confirmation` —— 用例组确认。

业务逻辑在 `services/documents.py` 与 `services/case_confirm.py`；本文件只做存在性校验、
HTTP 状态码映射与响应结构（错误风格与既有 router 一致：`HTTPException(4xx, 中文说明)`）。
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Task, TaskDocument
from ..services import case_confirm, documents

router = APIRouter(prefix="/api", tags=["documents"])

#: 模板源文件（`<repo_root>/docs/templates/`，契约 §3.1）
TEMPLATE_DIR = "docs/templates"
TEMPLATE_STEM = "算子设计文档模板"
TEMPLATE_FORMATS = {
    "docx": (f"{TEMPLATE_STEM}.docx",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "md": (f"{TEMPLATE_STEM}.md", "text/markdown; charset=utf-8"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CaseConfirmIn(BaseModel):
    """`POST /api/tasks/{task_id}/cases/confirm` 请求体（契约 §3.3）。"""

    cases_sha256: str
    benchmark_sha256: str
    by: str | None = None


def _get_task(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return task


def _get_document(db: Session, task: Task, doc_id: int) -> TaskDocument:
    row = documents.get(db, task, doc_id)
    if row is None:
        raise HTTPException(404, "文档不存在")
    return row


def _workspace_or_error(db: Session, task: Task) -> Path:
    try:
        return documents.workspace_of(db, task)
    except documents.DocumentError as e:
        raise HTTPException(e.status_code, str(e)) from e


# ---------------------------------------------------------------------------
# 模板下载
# ---------------------------------------------------------------------------

@router.get("/templates/operator-design-doc")
def download_operator_design_doc(format: str = Query("docx")) -> FileResponse:
    """《算子设计文档》模板下载：docx / md 两种（设计 §3.2）。"""
    fmt = str(format or "").strip().lower()
    if fmt not in TEMPLATE_FORMATS:
        raise HTTPException(400, "format 只能是 docx 或 md")
    filename, media_type = TEMPLATE_FORMATS[fmt]
    path = Path(settings.repo_root).expanduser() / TEMPLATE_DIR / filename
    if not path.is_file():
        raise HTTPException(404, f"模板文件不存在：{TEMPLATE_DIR}/{filename}")
    return FileResponse(str(path), filename=filename, media_type=media_type)


# ---------------------------------------------------------------------------
# 任务文档
# ---------------------------------------------------------------------------

@router.post("/tasks/{task_id}/documents", status_code=201)
async def upload_document(task_id: int, file: UploadFile = File(...),
                          db: Session = Depends(get_db)) -> dict:
    """上传设计文档（multipart `file`）：仅 .docx/.md，≤20MB；同名文件覆盖。

    分块读取并边读边计数：超限立刻拒绝，避免整个请求体先全量进内存（与对话附件同一做法）。
    解析失败 → 400 且不落库、不留文件（解析在 `documents.save` 的任何写入之前完成）。
    """
    task = _get_task(db, task_id)
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > documents.MAX_BYTES:
            raise HTTPException(413, "文档超过 20MB 上限")
        chunks.append(chunk)
    try:
        row = documents.save(db, task, file.filename or "", b"".join(chunks))
    except documents.DocumentError as e:
        db.rollback()
        raise HTTPException(e.status_code, str(e)) from e
    return documents.as_dict(row)


@router.get("/tasks/{task_id}/documents")
def list_documents(task_id: int, db: Session = Depends(get_db)) -> list[dict]:
    """任务的文档列表（新→旧；`path` / `extracted_path` 相对工作区）。"""
    task = _get_task(db, task_id)
    return [documents.as_dict(row) for row in documents.list_for_task(db, task)]


@router.get("/tasks/{task_id}/documents/{doc_id}/download")
def download_document(task_id: int, doc_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """下载文档原文（交付前核对/复现用）。"""
    task = _get_task(db, task_id)
    row = _get_document(db, task, doc_id)
    workspace = _workspace_or_error(db, task)
    try:
        path = documents.absolute_path(workspace, row)
    except documents.DocumentError as e:
        raise HTTPException(e.status_code, str(e)) from e
    if not path.is_file():
        raise HTTPException(404, f"文档文件不在磁盘上（可能已被删除）：{row.path}")
    return FileResponse(str(path), filename=row.filename,
                        media_type=documents.media_type(row.filename))


@router.delete("/tasks/{task_id}/documents/{doc_id}", status_code=204)
def delete_document(task_id: int, doc_id: int, db: Session = Depends(get_db)) -> None:
    """删除文档（DB 行 + 磁盘文件；定版后也允许，契约 §3.2）。"""
    task = _get_task(db, task_id)
    if not documents.delete(db, task, doc_id):
        raise HTTPException(404, "文档不存在")


# ---------------------------------------------------------------------------
# 用例组确认
# ---------------------------------------------------------------------------

@router.post("/tasks/{task_id}/cases/confirm")
def confirm_cases(task_id: int, payload: CaseConfirmIn,
                  db: Session = Depends(get_db)) -> dict:
    """整组确认精度/性能用例（契约 §3.3）。

    校验 `cases/<op>/cases.json` 与 `benchmark.json` 的存在性与 sha256；不匹配 → 409。
    成功后写 `tasks.cases_confirmed_*` + StageEvent(reason="用例组确认")。
    """
    task = _get_task(db, task_id)
    normalized = {}
    for name in ("cases_sha256", "benchmark_sha256"):
        value = str(getattr(payload, name) or "").strip().lower()
        if not _SHA256_RE.match(value):
            raise HTTPException(400, f"{name} 必须是 64 位十六进制 sha256")
        normalized[name] = value
    try:
        data = case_confirm.confirm(db, task, cases_sha256=normalized["cases_sha256"],
                                    benchmark_sha256=normalized["benchmark_sha256"],
                                    by=payload.by)
    except case_confirm.CaseConfirmError as e:
        db.rollback()
        raise HTTPException(409, str(e)) from e
    suite_sha = data["confirmed_sha256"] or data["current_suite_sha256"]
    return {
        "confirmed_at": data["confirmed_at"],
        "cases_sha256": normalized["cases_sha256"],
        "cases_suite_sha256": suite_sha,
        "suite_ref": case_confirm.suite_ref_from_sha256(suite_sha),
    }


@router.get("/tasks/{task_id}/cases/confirmation")
def cases_confirmation(task_id: int, db: Session = Depends(get_db)) -> dict:
    """用例确认状态：`{confirmed_at, confirmed_sha256, current_suite_sha256, in_sync}`。"""
    task = _get_task(db, task_id)
    return case_confirm.confirmation(db, task)
