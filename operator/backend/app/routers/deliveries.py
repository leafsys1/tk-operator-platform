"""交付 API（任务生命周期 v2）：预检 / 生成交付包 / 历史 / 下载。

交付是**本地动作**（不连远端容器、不需要设备）：门禁 → 打包 zip → 落 run/artifacts →
当前版本标记 delivered（之后改代码必须新小版本 + 重跑 + 重新交付）。契约见 `docs/impl-contract-v2.md` §3.5。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..services import delivery

router = APIRouter(prefix="/api", tags=["task-lifecycle-v2"])


@router.get("/tasks/{task_id}/deliveries/preview")
def preview_delivery(task_id: int, db: Session = Depends(get_db)) -> dict:
    """交付门禁预检：缺什么、证据是否有效（页面据此显示「去跑哪个动作」）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    try:
        result = delivery.check(db, task)
    except delivery.DeliveryError as e:
        return {"ok": False, "missing": e.missing,
                "codes": e.codes, "error": str(e)}
    return result


@router.post("/tasks/{task_id}/deliveries", status_code=201)
def create_delivery(task_id: int, db: Session = Depends(get_db)) -> dict:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    try:
        return delivery.package(db, task)
    except delivery.DeliveryError as e:
        raise HTTPException(409, {"detail": str(e), "codes": e.codes, "missing": e.missing}) from e


@router.get("/tasks/{task_id}/deliveries")
def list_deliveries(task_id: int, db: Session = Depends(get_db)) -> dict:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return {"task_id": task.id, "deliveries": delivery.view(db, task)}


@router.get("/deliveries/{run_id}/download")
def download_delivery(run_id: int, db: Session = Depends(get_db)) -> FileResponse:
    try:
        _run, path = delivery.delivery_zip_path(db, run_id)
    except delivery.DeliveryError as e:
        raise HTTPException(404, str(e)) from e
    return FileResponse(str(path), filename=path.name)
