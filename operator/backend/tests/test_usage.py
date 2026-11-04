"""消耗情况（P4）：聚合口径 —— 真实调用才算，P0 示例行（is_demo）必须排除。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models import AgentJob, Endpoint, Project, Task


def test_usage_aggregates_and_excludes_demo(client, workspace):
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.flush()
        now = datetime(2026, 10, 19, 3, 0, tzinfo=timezone.utc)
        db.add_all([
            AgentJob(task_id=t.id, kind="开发", status="ok", started_at=now,
                     tokens_in=1000, tokens_out=200, cache_read=50, cache_write=10),
            AgentJob(task_id=t.id, kind="需求澄清", status="ok", started_at=now,
                     tokens_in=500, tokens_out=100, cache_read=20, cache_write=0),
            AgentJob(task_id=t.id, kind="开发", status="failed", started_at=now,
                     tokens_in=300, tokens_out=0),
            AgentJob(task_id=t.id, kind="开发", status="ok", started_at=now, is_demo=True,
                     tokens_in=9_999_999, tokens_out=9_999_999),
            AgentJob(task_id=t.id, kind="开发", status="ok", started_at=now, is_demo=True,
                     tokens_in=1, tokens_out=1),
        ])
        db.commit()

    body = c.get("/api/usage").json()
    tot = body["totals"]
    assert tot["jobs"] == 3  # demo 行排除
    assert tot["ok"] == 2 and tot["failed"] == 1
    assert tot["tokens_in"] == 1800 and tot["tokens_out"] == 300
    assert tot["cache_read"] == 70 and tot["cache_write"] == 10
    assert tot["cost_known"] is False  # 没有定价，别编金额
    assert body["excluded_demo_rows"] == 2  # 真实条数，不是「是否存在」

    assert body["by_task"][0]["tokens_in"] == 1800
    kinds = {k["kind"]: k for k in body["by_kind"]}
    assert kinds["开发"]["jobs"] == 2 and kinds["需求澄清"]["jobs"] == 1
    assert body["by_day"][0]["day"] == "2026-10-19"


def test_usage_empty_db(client):
    c, _ = client
    body = c.get("/api/usage").json()
    assert body["totals"]["jobs"] == 0
    assert body["by_task"] == [] and body["by_kind"] == []


def test_usage_costs_by_endpoint_price(client, workspace):
    """端点配了单价后：按 token 算金额；早期没记端点的调用回退当前端点并标估算。"""
    c, Session = client
    with Session() as db:
        ep = Endpoint(name="deepseek", provider="dsh-llm-deepseek",
                      price_in=2.0, price_out=8.0, price_cache_read=0.2)
        db.add(ep)
        db.flush()
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.flush()
        now = datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc)
        db.add_all([
            # 记了端点：1M in + 1M out + 1M cache 读 = 2 + 8 + 0.2
            AgentJob(task_id=t.id, kind="开发", status="ok", started_at=now, endpoint_id=ep.id,
                     tokens_in=1_000_000, tokens_out=1_000_000, cache_read=1_000_000),
            # 早期行没有端点快照：0.5M in → 按当前端点价目估算 = 1.0
            AgentJob(task_id=t.id, kind="需求澄清", status="ok", started_at=now,
                     tokens_in=500_000),
        ])
        db.commit()

    body = c.get("/api/usage").json()
    tot = body["totals"]
    assert tot["cost_known"] is True
    assert tot["cost"] == pytest.approx(11.2)
    assert tot["priced_jobs"] == 2
    assert tot["cost_estimated"] == 1
    assert tot["unpriced_jobs"] == 0
    assert body["pricing"]["priced"] is True and body["pricing"]["currency"] == "CNY"
    assert any("估算" in n for n in body["notes"])


def test_usage_unpriced_when_endpoint_has_no_price(client, workspace):
    """端点没配单价：如实「未定价」，但不影响 token 聚合。"""
    c, Session = client
    with Session() as db:
        ep = Endpoint(name="gw", provider="dsh-llm-pi-ai", price_in=2.0)  # 只配 in = 未定价
        db.add(ep)
        db.flush()
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="t", op_name="t")
        db.add(t)
        db.flush()
        db.add(AgentJob(task_id=t.id, kind="开发", status="ok", endpoint_id=ep.id,
                        tokens_in=1000, tokens_out=100))
        db.commit()

    tot = c.get("/api/usage").json()["totals"]
    assert tot["cost_known"] is False
    assert tot["unpriced_jobs"] == 1 and tot["cost"] == 0.0
    assert tot["tokens_in"] == 1000


def test_usage_overflow_price_is_unpriced_not_null(client, workspace):
    """极端单价 × 大 token 溢出 inf：按未定价处理，不能返回 cost=null（前端会崩）。"""
    c, Session = client
    with Session() as db:
        ep = Endpoint(name="huge", provider="dsh-llm-deepseek", price_in=1e308, price_out=1.0)
        db.add(ep)
        db.flush()
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="t", op_name="t")
        db.add(t)
        db.flush()
        db.add(AgentJob(task_id=t.id, kind="开发", status="ok", endpoint_id=ep.id,
                        tokens_in=1_000_000_000, tokens_out=1))
        db.commit()

    body = c.get("/api/usage").json()
    tot = body["totals"]
    assert tot["cost_known"] is False
    assert tot["unpriced_jobs"] == 1
    assert tot["cost"] == 0.0  # 不是 inf/null
