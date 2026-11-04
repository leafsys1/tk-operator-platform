"""目标硬件画像接口（方案 §13-A11）：给「极限 / Roofline 对比」提供峰值带宽/算力。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Project
from ..services import chips

router = APIRouter(prefix="/api", tags=["chips"])


def _workspace(db: Session) -> Path | None:
    proj = db.scalar(select(Project).order_by(Project.id))
    if proj and proj.workspace_path:
        return Path(proj.workspace_path).expanduser()
    return None


@router.get("/chips")
def list_chips(db: Session = Depends(get_db)) -> list[dict]:
    """列出任务工作区里 `ops/chips/*.yaml` 的硬件画像摘要。"""
    ws = _workspace(db)
    return chips.list_chips(ws) if ws else []


@router.get("/chips/{name}")
def get_chip(name: str, db: Session = Depends(get_db)) -> dict:
    ws = _workspace(db)
    doc = chips.load_chip(ws, name) if ws else None
    if not doc:
        raise HTTPException(404, f"硬件画像 ops/chips/{name}.yaml 不存在")
    return doc
