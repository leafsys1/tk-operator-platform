"""案例库（M2）：外置 bundle 列表 + 一键发起任务。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..services import case_library

router = APIRouter(prefix="/api", tags=["case-library"])


class StartCase(BaseModel):
    bundle: str
    op: str
    project_id: int | None = None


@router.get("/case-library")
def get_case_library() -> dict:
    """已导入的案例库索引（来自 `~/.local/share/op-agent/case-library/index.json`）。"""
    return case_library.index()


@router.post("/case-library/start", status_code=201)
def start_case(payload: StartCase, db: Session = Depends(get_db)) -> dict:
    """从案例建任务：复制 cases.json（可选 workspace/ops 子树）→ 建 Task。"""
    try:
        task = case_library.start(db, payload.bundle, payload.op, project_id=payload.project_id)
    except case_library.CaseError as e:
        raise HTTPException(409, str(e)) from e
    return {"task_id": task.id, "title": task.title, "op_name": task.op_name}
