"""端点价目 → 费用（D2 费用口径）。

规则：人民币元 / 每百万 token；`price_in` + `price_out` 都有才算已定价；cache 两项留空按 0 计；
费用不落库，读时按 token × 当前价目算（早期调用没端点快照时回退当前端点，标 estimated）。
"""

from __future__ import annotations

import pytest

from app.models import AgentJob, Endpoint
from app.services import pricing


def _ep(**kw) -> Endpoint:
    return Endpoint(name="e", provider="dsh-llm-deepseek", **kw)


def test_cost_of_per_million_and_optional_cache():
    ep = _ep(price_in=4.0, price_out=16.0)
    # 1M 输入 + 0.5M 输出 = 4 + 8
    assert pricing.cost_of(ep, tokens_in=1_000_000, tokens_out=500_000) == pytest.approx(12.0)
    # cache 没配 = 不单独计价（按 0），不是「未定价」
    assert pricing.cost_of(ep, tokens_in=1_000_000, cache_read=1_000_000) == pytest.approx(4.0)

    ep2 = _ep(price_in=4.0, price_out=16.0, price_cache_read=0.4, price_cache_write=4.0)
    # 1M 输入 + 0.5M 输出 + 1M cache 读 + 1M cache 写 = 4 + 8 + 0.4 + 4
    assert pricing.cost_of(ep2, tokens_in=1_000_000, tokens_out=500_000,
                           cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(16.4)


def test_only_in_and_out_together_means_priced():
    assert pricing.cost_of(_ep(price_in=1.0), tokens_in=10) is None
    assert pricing.cost_of(_ep(price_out=1.0), tokens_out=10) is None
    assert pricing.cost_of(None) is None
    assert pricing.is_priced(_ep(price_in=0, price_out=0)) is True  # 0 元价也是「已定价」


def test_job_cost_prefers_stored_then_endpoint_then_estimates():
    ep = _ep(price_in=4.0, price_out=16.0)
    endpoints = {7: ep}

    job = AgentJob(task_id=1, kind="开发", tokens_in=1_000_000, endpoint_id=7)
    assert pricing.job_cost(job, endpoints, ep) == (pytest.approx(4.0), False)

    # 早期行没记端点 → 回退当前端点，金额标估算
    old = AgentJob(task_id=1, kind="开发", tokens_in=1_000_000)
    amount, estimated = pricing.job_cost(old, endpoints, ep)
    assert amount == pytest.approx(4.0) and estimated is True

    # 已有手工金额（P0 示例/人工补录）优先，不算估算
    manual = AgentJob(task_id=1, kind="开发", tokens_in=999, cost=1.23)
    assert pricing.job_cost(manual, endpoints, ep) == (1.23, False)

    # 没有 token = 0 元，不因未定价而标未知
    empty = AgentJob(task_id=1, kind="开发")
    assert pricing.job_cost(empty, {}, None) == (0.0, False)

    # 端点未定价 → None（调用方标「未定价」）
    unpriced = _ep(price_in=1.0)
    assert pricing.job_cost(old, {8: unpriced}, unpriced) == (None, False)


def test_cost_of_rejects_non_finite_overflow():
    """极端单价 × 大 token 会溢出 inf；cost_of 返回 None，不把 inf 带进 JSON（审查 H3）。"""
    ep = _ep(price_in=1e308, price_out=1.0)
    assert pricing.cost_of(ep, tokens_in=1_000_000_000) is None
    inf_ep = _ep(price_in=float("inf"), price_out=1.0)
    assert pricing.cost_of(inf_ep, tokens_in=100) is None


def test_job_cost_ignores_non_finite_stored_cost():
    """历史坏行 `job.cost=inf` 不能直接返回（会让 /api/usage cost 变 null）；退回按价目重算。"""
    ep = _ep(price_in=4.0, price_out=16.0)
    bad = AgentJob(task_id=1, kind="开发", tokens_in=1_000_000, cost=float("inf"))
    # 没有 endpoint_id 快照 → 回退当前端点重算，标 estimated（不是直接用坏值）
    assert pricing.job_cost(bad, {7: ep}, ep) == (pytest.approx(4.0), True)

    # 没有可用端点/价目 → 未定价（None），而不是 inf
    assert pricing.job_cost(bad, {}, None) == (None, False)
