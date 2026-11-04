"""消耗情况（P4）：从 `agent_jobs` 聚合「调用数 / token / 费用」。

口径（方案 §7 边界③、§3.1；2026-10-21 补费用）：
- **只算运行态的真实调用**：`agent_jobs` 是「一次智能体调用」，聚合**必须排除** `is_demo` 的 P0 示例行
  （那些 usage 是假的，见 `models.AgentJob.is_demo`）。
- **费用**：按这次调用所属端点的价目实时算（`services/pricing.py`，人民币元 / 每百万 token）。
  端点没配 `price_in` + `price_out` → 该项如实标未定价（`cost_known=false`），不编金额；
  早期调用没有端点快照时会回退「当前端点」估算，计入 `cost_estimated` 并在 notes 里说清楚。
  费用不落库：**改了「API 管理」里的单价，这里立刻跟着变**。
- token 一律取真实回填值（`tokens_in/out`、`cache_read/write`）。

聚合维度：总计 / 按任务 / 按调用类型 / 按天（UTC）。数据量小，直接在 Python 里聚合。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AgentJob, Endpoint, Task
from ..services import pricing

router = APIRouter(prefix="/api", tags=["usage"])


def _blank() -> dict:
    return {"jobs": 0, "ok": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0,
            "cache_read": 0, "cache_write": 0, "cost": 0.0, "cost_known": True,
            "priced_jobs": 0, "unpriced_jobs": 0, "cost_estimated": 0}


def _add(acc: dict, job: AgentJob, endpoints: dict[int, Endpoint],
         active: Endpoint | None) -> None:
    acc["jobs"] += 1
    if job.status == "ok":
        acc["ok"] += 1
    elif job.status in ("failed", "interrupted"):
        acc["failed"] += 1
    acc["tokens_in"] += int(job.tokens_in or 0)
    acc["tokens_out"] += int(job.tokens_out or 0)
    acc["cache_read"] += int(job.cache_read or 0)
    acc["cache_write"] += int(job.cache_write or 0)
    cost, estimated = pricing.job_cost(job, endpoints, active)
    if cost is None:
        acc["cost_known"] = False  # 有未定价的调用，别把 sum 当成完整金额
        acc["unpriced_jobs"] += 1
    else:
        acc["cost"] += cost
        acc["priced_jobs"] += 1
        if estimated:
            acc["cost_estimated"] += 1


@router.get("/usage")
def usage(db: Session = Depends(get_db)) -> dict:
    """智能体调用消耗聚合（排除 is_demo 示例行）。"""
    rows = list(db.scalars(select(AgentJob).where(AgentJob.is_demo.is_(False))))
    titles = {t.id: t.title for t in db.scalars(select(Task))}
    endpoints = pricing.endpoints_by_id(db)
    active = pricing.active_endpoint(db)

    total = _blank()
    by_task: dict[int, dict] = {}
    by_kind: dict[str, dict] = {}
    by_day: dict[str, dict] = {}
    for j in rows:
        _add(total, j, endpoints, active)
        t = by_task.setdefault(j.task_id, _blank() | {"task_id": j.task_id,
                                                      "title": titles.get(j.task_id, f"task {j.task_id}")})
        _add(t, j, endpoints, active)
        _add(by_kind.setdefault(j.kind, _blank() | {"kind": j.kind}), j, endpoints, active)
        day = (j.started_at or j.ended_at)
        key = day.strftime("%Y-%m-%d") if day else "未知"
        _add(by_day.setdefault(key, _blank() | {"day": key}), j, endpoints, active)

    def sorted_vals(d: dict, key: str, reverse: bool = True) -> list[dict]:
        return sorted(d.values(), key=lambda x: x[key], reverse=reverse)

    demo_rows = int(
        db.scalar(
            select(func.count()).select_from(AgentJob).where(AgentJob.is_demo.is_(True))
        )
        or 0
    )
    notes = [
        "只统计真实调用（agent_jobs）；P0 示例行（is_demo）已排除。",
        "费用 = token × 端点单价（人民币元 / 每百万 token），改「API 管理」里的单价后这里实时更新。",
    ]
    if not pricing.is_priced(active):
        notes.append("当前端点还没配单价 —— 到「API 管理」页给端点填输入/输出单价后，费用才会显示金额。")
    if total["cost_estimated"]:
        notes.append(f"{total['cost_estimated']} 次早期调用没有端点快照，按当前端点价目估算（未冒充当时账单）。")
    if total["unpriced_jobs"]:
        notes.append(f"{total['unpriced_jobs']} 次调用没有可用的端点价目（或金额溢出），费用按「未定价」如实展示。")

    return {
        "totals": total,
        "by_task": sorted_vals(by_task, "tokens_in"),
        "by_kind": sorted_vals(by_kind, "jobs"),
        "by_day": sorted(by_day.values(), key=lambda x: x["day"]),
        "excluded_demo_rows": demo_rows,
        "pricing": {
            "currency": "CNY",
            "unit": "per_1m_tokens",
            "endpoint_id": active.id if active else None,
            "endpoint": active.name if active else None,
            "priced": pricing.is_priced(active),
        },
        "notes": notes,
    }
