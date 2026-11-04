"""P2/P3：优化点搜索池、一思路一 Loop、测量预算硬门禁。"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.models import (ActionJob, AgentJob, Conversation, IdeaSearch, Loop, LoopBudgetAudit,
                        Message, OptimizationIdea, Project, Run, SpecVersion, Task)
from app.services import exec_actions, idea_search, loop_budget


def _seed_api(Session, workspace) -> int:
    with Session() as db:
        project = Project(name="ideas", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name="elementwise_add",
                    target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return task.id


def test_user_idea_dedup_and_one_idea_one_loop(client, workspace):
    c, Session = client
    task_id = _seed_api(Session, workspace)
    payload = {
        "title": "缩小小 shape 的 tile",
        "hypothesis": "核数不足",
        "change_scope": "ops/elementwise_add/",
        "expected": "64K 延迟下降",
    }
    first = c.post(f"/api/tasks/{task_id}/optimization-ideas", json=payload)
    assert first.status_code == 201, first.text
    assert first.json()["deduplicated"] is False
    idea_id = first.json()["idea"]["id"]

    again = c.post(f"/api/tasks/{task_id}/optimization-ideas", json=payload)
    assert again.status_code == 201
    assert again.json()["deduplicated"] is True
    assert again.json()["idea"]["id"] == idea_id
    assert len(c.get(f"/api/tasks/{task_id}/optimization-ideas").json()) == 1

    created = c.post(f"/api/optimization-ideas/{idea_id}/round", json={})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["loop"]["objective"]["type"] == "perf"
    # 思路 -> 轮次用「文字优化点」目标（不指定目标用例）；focus 由思路字段拼出
    assert body["loop"]["objective"]["target_dimension"] == "text"
    assert "优化点" in body["loop"]["objective"]["focus"]
    assert body["loop"]["objective"]["target_cases"] == []
    assert body["idea"]["status"] == "validating"
    assert body["idea"]["loop_id"] == body["loop"]["id"]
    assert c.post(f"/api/optimization-ideas/{idea_id}/round", json={}).status_code == 409

    second_idea = c.post(f"/api/tasks/{task_id}/optimization-ideas", json={
        "title": "减少重复 DataCopy", "change_scope": "ops/elementwise_add/",
    }).json()["idea"]
    blocked = c.post(f"/api/optimization-ideas/{second_idea['id']}/round", json={})
    assert blocked.status_code == 409
    assert "串行验证" in blocked.text

    rejected = c.post(f"/api/loops/{body['loop']['id']}/reject", json={"reason": "无收益"})
    assert rejected.status_code == 200, rejected.text
    rows = c.get(f"/api/tasks/{task_id}/optimization-ideas").json()
    first_row = next(row for row in rows if row["id"] == idea_id)
    assert first_row["status"] == "rejected"
    assert first_row["outcome"]["reason"] == "无收益"
    assert c.post(f"/api/optimization-ideas/{second_idea['id']}/round", json={}).status_code == 201


def test_conversation_direction_is_recorded_and_waits_for_baseline(client, workspace):
    c, Session = client
    task_id = _seed_api(Session, workspace)
    blank = c.post(f"/api/tasks/{task_id}/optimization-chat", json={"direction": "   \n"})
    assert blank.status_code == 422
    direction = "重点优化 64K 小 shape 的并行度，保持 1M 路径不变"

    response = c.post(f"/api/tasks/{task_id}/optimization-chat", json={"direction": direction})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "waiting_baseline"
    assert body["job_id"] is None
    assert body["idea"]["source"] == "conversation"
    assert body["idea"]["loop_id"] == body["loop"]["id"]
    assert direction in body["loop"]["objective"]["focus"]
    assert {row["action"] for row in body["attached_runs"]} == set()

    with Session() as db:
        conv = db.get(Conversation, body["conversation_id"])
        assert conv is not None and conv.mode == "task"
        messages = list(db.scalars(select(Message).where(
            Message.conversation_id == conv.id).order_by(Message.id)))
        assert [(m.role, m.text) for m in messages] == [
            ("user", direction),
            ("assistant", body["reason"]),
        ]
        assert db.scalar(select(AgentJob).where(AgentJob.loop_id == body["loop"]["id"])) is None

    queued = c.post(f"/api/tasks/{task_id}/optimization-chat", json={
        "direction": "第二个方向：减少重复 DataCopy",
    })
    assert queued.status_code == 201, queued.text
    assert queued.json()["status"] == "waiting_previous"
    assert queued.json()["loop"] is None
    assert queued.json()["idea"]["status"] == "proposed"
    assert len(c.get(f"/api/tasks/{task_id}/optimization-ideas").json()) == 2


def test_conversation_direction_starts_agent_when_same_snapshot_baseline_exists(
    client, workspace, monkeypatch,
):
    c, Session = client
    task_id = _seed_api(Session, workspace)
    sha = exec_actions.snapshot(workspace).sha256
    with Session() as db:
        for action in ("test", "benchmark"):
            db.add(Run(task_id=task_id, action=action, status="ok", code_snapshot_sha256=sha))
        db.commit()

    monkeypatch.setenv("OPX_LLM_KEY", "test-only")
    from app.routers import runs as runs_router
    from app.services import optimization as optimization_service

    # 本用例只验「对话方向 → idea/Loop → 复用消息启动 Agent」这条入口；
    # 本地 Step 1b 的 G1 基线证据（env 字段 / benchmark case 指纹）由
    # test_step1b.py / test_auto_run.py 单独覆盖，这里把 G1 假掉。
    monkeypatch.setattr(optimization_service, "require_baseline", lambda *a, **k: None)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/tmp/dsh")
    started = {}

    def fake_start(job_id, prompt, **kwargs):
        started.update(job_id=job_id, prompt=prompt, kwargs=kwargs)

    monkeypatch.setattr(runs_router.optimize, "start", fake_start)
    direction = "减少小 shape 的空闲核，只改变核间分配"
    response = c.post(f"/api/tasks/{task_id}/optimization-chat", json={"direction": direction})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "started"
    assert {row["action"] for row in body["attached_runs"]} == {"test", "benchmark"}
    assert body["job_id"] == started["job_id"]
    assert started["kwargs"]["conversation_id"] == body["conversation_id"]
    assert direction in started["prompt"]

    with Session() as db:
        job = db.get(AgentJob, body["job_id"])
        assert job is not None and job.kind == "优化" and job.loop_id == body["loop"]["id"]
        messages = list(db.scalars(select(Message).where(
            Message.conversation_id == body["conversation_id"]).order_by(Message.id)))
        # 入口已经落过用户消息，启动 helper 必须复用它，不能在会话里重复一遍。
        assert [(m.role, m.text) for m in messages] == [("user", direction)]


def test_agent_ingest_deduplicates_and_updates(db_session, seeded):
    task = seeded["task"]
    existing, duplicate = idea_search.add_user(db_session, task, {
        "title": "小 shape tile", "change_scope": "ops/elementwise_add/",
    })
    assert duplicate is False
    search = IdeaSearch(task_id=task.id, status="running")
    db_session.add(search)
    db_session.flush()
    result = idea_search._ingest(db_session, search, {
        "ideas": [
            {"title": "小 shape tile", "change_scope": "ops/elementwise_add/"},
            {"title": "减少 DataCopy 次数", "change_scope": "ops/elementwise_add/",
             "evidence": {"run_id": 8}, "confidence": "medium"},
        ],
        "updates": [{"idea_id": existing.id, "status": "rejected",
                     "outcome": {"reason": "已验证无收益"}}],
    })
    db_session.commit()
    assert result["new"] == 1 and result["duplicates"] == 1 and result["updated"] == 1
    rows = list(db_session.scalars(select(OptimizationIdea).where(
        OptimizationIdea.task_id == task.id).order_by(OptimizationIdea.id)))
    assert len(rows) == 2
    assert rows[0].status == "rejected"
    assert json.loads(rows[0].outcome_json)["reason"] == "已验证无收益"
    assert rows[1].search_id == search.id and rows[1].confidence == "medium"


def test_start_diagnostic_job_is_read_only(client, workspace, monkeypatch):
    c, Session = client
    task_id = _seed_api(Session, workspace)
    monkeypatch.setenv("OPX_LLM_KEY", "test-only")
    from app.routers import optimization as optimization_router

    monkeypatch.setattr(optimization_router.shutil, "which", lambda _name: "/tmp/dsh")
    started = {}

    def fake_start(search, job, prompt, workspace, dsh_home):
        started.update(search=search.id, job=job.id, prompt=prompt,
                       workspace=str(workspace), dsh_home=str(dsh_home))

    monkeypatch.setattr(optimization_router.idea_search, "start", fake_start)
    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={"extra": "重点看访存"})
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "running"
    assert "禁止修改任何文件" in started["prompt"]
    assert "禁止调用 scripts/nm.sh" in started["prompt"]
    with Session() as db:
        job = db.get(AgentJob, started["job"])
        assert job.kind == "诊断" and job.status == "running"


def test_loop_budget_hard_limit_and_audit(db_session, seeded):
    task = seeded["task"]
    loop = Loop(
        task_id=task.id, kind="perf", status="open",
        objective_json=json.dumps({"type": "perf", "budget": {"runs_per_action": 2}}),
    )
    db_session.add(loop)
    db_session.commit()

    first = loop_budget.check_and_audit(db_session, task, loop, "benchmark")
    assert {k: first[k] for k in ("enforced", "limit", "used", "remaining_after_start")} == {
        "enforced": True, "limit": 2, "used": 0, "remaining_after_start": 1,
    }
    # 页面路径要拿 audit_id 做服务端一次性预留（脚本执行前用它 claim，审查 M4）。
    assert isinstance(first["audit_id"], int)
    # 页面路径会在 allowed 审计后创建 ActionJob；同一次尝试不能被算两遍。
    db_session.add(ActionJob(task_id=task.id, loop_id=loop.id, action="benchmark",
                             status="failed", run_ref="r1"))
    db_session.commit()
    second = loop_budget.check_and_audit(db_session, task, loop, "benchmark")
    assert second["used"] == 1 and second["remaining_after_start"] == 0
    assert second["audit_id"] != first["audit_id"]
    with pytest.raises(loop_budget.BudgetExceeded, match="预算已用完"):
        loop_budget.check_and_audit(db_session, task, loop, "benchmark")

    audits = list(db_session.scalars(select(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == loop.id).order_by(LoopBudgetAudit.id)))
    assert [a.decision for a in audits] == ["allowed", "allowed", "rejected"]
    # 旧 Loop 没 objective：只读兼容，不启用新预算。
    legacy = Loop(task_id=task.id, kind="perf", status="open")
    db_session.add(legacy)
    db_session.commit()
    assert loop_budget.check_and_audit(db_session, task, legacy, "benchmark") == {"enforced": False}
