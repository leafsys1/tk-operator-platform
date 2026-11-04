"""端点价目 → 一次智能体调用的费用（解 D2：单价口径）。

口径（2026-10-21 补）：
- 单价存 `endpoints.price_in` / `price_out` / `price_cache_read` / `price_cache_write`，
  **人民币元 / 每百万 token**；cache 两项留空 = 不单独计价（按 0 计）。
- `price_in` + `price_out` 都配了才算「已定价」；只配一半按未定价处理（别拿半个价算钱）。
- 费用**不落库**（`agent_jobs.cost` 只留给 P0 示例/手工行）：读消耗/任务详情时按
  token × 当前价目实时算。好处是改单价后历史消耗立刻跟着更新，不会出现「配了价还挂着旧金额」。
- 归属：`agent_jobs.endpoint_id` 记调用时的端点（调用创建时快照）；早期行没有 → 回退
  「当前端点」，此时金额标 `estimated=True`（按当前价目估算，不冒充当时的账单）。
"""

from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AgentJob, Endpoint
from . import llm_config

#: 价目单位：每百万 token
PER_TOKENS = 1_000_000.0


def is_priced(endpoint: Endpoint | None) -> bool:
    """端点是否已定价：in/out 两项必须都有（cache 两项可选，缺省按 0 计）。"""
    return endpoint is not None and endpoint.price_in is not None and endpoint.price_out is not None


def cost_of(endpoint: Endpoint | None, *,
            tokens_in: int = 0, tokens_out: int = 0,
            cache_read: int = 0, cache_write: int = 0) -> float | None:
    """按端点价目算金额（人民币元）；端点未定价 → None（调用方如实标「未定价」）。"""
    if not is_priced(endpoint):
        return None
    assert endpoint is not None  # is_priced 已排除 None，这里只为类型收窄
    amount = (
        int(tokens_in or 0) * float(endpoint.price_in or 0.0)
        + int(tokens_out or 0) * float(endpoint.price_out or 0.0)
        + int(cache_read or 0) * float(endpoint.price_cache_read or 0.0)
        + int(cache_write or 0) * float(endpoint.price_cache_write or 0.0)
    ) / PER_TOKENS
    # 极端单价 × 大 token 可能溢出成 inf（审查 H3）：不把非有限金额带进 JSON/前端，
    # 按「未定价」返回 None（调用方会累成 cost_known=false，而不是 null.toFixed 崩溃）。
    if not math.isfinite(amount):
        return None
    # 金额到 1e-6 元足够（token 单价换算后的展示精度）；避免浮点尾巴污染前端 toFixed。
    return round(amount, 6)


def endpoints_by_id(db: Session) -> dict[int, Endpoint]:
    """一次读出**所有**端点（含软删），供批量聚合时按 id 查。

    含软删是刻意的（审查 H2）：历史 `agent_jobs.endpoint_id` 必须仍能解析到当时端点与价目；
    列表/当前端点查询才过滤 `deleted_at`。
    """
    return {r.id: r for r in db.scalars(select(Endpoint))}


def active_endpoint(db: Session) -> Endpoint | None:
    """当前端点（没显式 active 时取第一条，与 `llm_config` 同口径）。"""
    return llm_config.active_endpoint(db)


def job_cost(job: AgentJob, endpoints: dict[int, Endpoint],
             active: Endpoint | None) -> tuple[float | None, bool]:
    """一条调用的金额：`(amount, estimated)`。

    优先级：
    1. `job.cost` 有值（P0 示例/手工行）→ 直接用它，不算估算；
    2. 全零 token → 0 元（没调用就没花钱，不因未定价而标未知）；
    3. `job.endpoint_id` 对应的端点 → 按它的价目算；
    4. 没记端点（早期行）/端点已删 → 回退当前端点，金额标 estimated=True。
    端点未定价 → (None, False)。
    """
    if job.cost is not None:
        try:
            stored = float(job.cost)
        except (TypeError, ValueError):
            stored = None
        # 历史坏行可能存了 inf（旧校验缺口）：不要把它带进聚合/JSON，退回按 token 重算。
        if stored is not None and math.isfinite(stored):
            return stored, False
    if not any((job.tokens_in, job.tokens_out, job.cache_read, job.cache_write)):
        return 0.0, False

    endpoint = endpoints.get(job.endpoint_id) if job.endpoint_id else None
    estimated = False
    if endpoint is None:
        endpoint = active
        estimated = endpoint is not None
    amount = cost_of(
        endpoint,
        tokens_in=job.tokens_in or 0,
        tokens_out=job.tokens_out or 0,
        cache_read=job.cache_read or 0,
        cache_write=job.cache_write or 0,
    )
    if amount is None:
        return None, False
    return amount, estimated


def attach_endpoint(db: Session, job: AgentJob) -> Endpoint | None:
    """调用创建时快照端点/模型，作为后续计价的归属依据。"""
    row = active_endpoint(db)
    if row is not None:
        job.endpoint_id = row.id
        job.model = row.model
    return row
