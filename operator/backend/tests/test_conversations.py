"""P0-2 对话页：会话 / 消息 / 两种模式 / 并发保护 / 收尾落库。

不跑 dsh：`agent_runner.start` 被替换成记录参数；收尾回调直接调用 `_finish`。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.models import AgentJob, Conversation, Message
from app.services import agent_runner, conversations as svc


@pytest.fixture()
def conv_db(db_session, seeded, monkeypatch):
    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(svc, "SessionLocal", Session)
    return db_session


def test_create_list_and_detail(conv_db, seeded):
    task = seeded["task"]
    conv = svc.create(conv_db, task.id, title="聊 elementwise_add", mode="task")
    assert conv.task_id == task.id and conv.mode == "task"
    rows = svc.list_for(conv_db, task_id=task.id)
    assert [c.id for c in rows] == [conv.id]
    loaded = svc.load(conv_db, conv.id, with_messages=True)
    assert loaded is not None and loaded.messages == []


def test_task_mode_uses_workspace_and_skills_prompt(conv_db, seeded, monkeypatch):
    task = seeded["task"]
    conv = svc.create(conv_db, task.id, mode="task")
    captured: dict = {}

    def fake_start(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(agent_runner, "start", fake_start)
    job = svc.send(conv_db, conv, "这个算子现在什么状态？")
    assert job.kind == "对话" and job.conversation_id == conv.id
    assert captured["cwd"] == Path(seeded["project"].workspace_path)
    assert "skills" in captured["cwd"].as_posix() or captured["cwd"] == Path(seeded["project"].workspace_path)
    # 自由聊天走 plain_stream：回答原样上屏，不按「prose + JSON 尾」两段式切分
    assert captured["plain_stream"] is True and captured["stream"] is True
    assert "elementwise_add" in job.prompt and "规格要点" in job.prompt
    # 控制台默认策略：不要静默 + 上机入口（否则 agent 会按 cannbot 模板「静默开发」）
    assert "不要静默" in job.prompt and "scripts/nm.sh" in job.prompt
    # 用户消息已落库
    msgs = list(conv_db.query(Message).filter_by(conversation_id=conv.id))
    assert len(msgs) == 1 and msgs[0].role == "user"


def test_light_mode_uses_scratch_and_no_tools_prompt(conv_db, seeded, monkeypatch):
    task = seeded["task"]
    conv = svc.create(conv_db, task.id, mode="light")
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: captured.update(kw))
    job = svc.send(conv_db, conv, "解释一下 tiling 是什么")
    assert "轻对话模式" in job.prompt and "不要调用工具" in job.prompt
    assert captured["cwd"].name == f"conv-{conv.id}"
    assert captured["cwd"].parent.name == ".opx-chat"


def test_same_task_concurrent_chat_rejected(conv_db, seeded, monkeypatch):
    task = seeded["task"]
    conv = svc.create(conv_db, task.id)
    conv_db.add(AgentJob(task_id=task.id, kind="对话", status="running"))
    conv_db.commit()
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="已有一个对话作业"):
        svc.send(conv_db, conv, "在吗")


def test_finish_appends_assistant_and_updates_session(conv_db, seeded):
    task = seeded["task"]
    conv = svc.create(conv_db, task.id)
    job = AgentJob(task_id=task.id, conversation_id=conv.id, kind="对话", status="running")
    conv_db.add(job)
    conv_db.commit()

    state = SimpleNamespace(prose="好的，已完成。", final="", session_id="session-x",
                            usage={"inputTokens": 10, "outputTokens": 5}, error=None)
    svc._finish(job.id, conv.id, state)

    conv_db.expire_all()
    row = conv_db.get(Conversation, conv.id)
    assert row.dsh_session_id == "session-x"
    assert row.message_count == 2 and row.tokens == 15
    msgs = list(conv_db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id))
    assert msgs[-1].role == "assistant" and msgs[-1].text == "好的，已完成。" and msgs[-1].job_id == job.id


def test_endpoints_crud_and_send(client, workspace: Path, monkeypatch):
    from app.models import Project, Task

    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.commit()
        tid = t.id

    conv = c.post("/api/conversations", json={"task_id": tid, "mode": "task"}).json()
    assert conv["task_id"] == tid and conv["mode"] == "task"

    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: None)
    r = c.post(f"/api/conversations/{conv['id']}/messages", json={"text": "你好"})
    assert r.status_code == 201, r.text
    assert r.json()["job_id"] > 0

    detail = c.get(f"/api/conversations/{conv['id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user"]
    # 会话视图暴露运行中的对话作业（前端刷新后据此接回 SSE）
    assert detail["running_job"] == r.json()["job_id"]
    assert c.get(f"/api/conversations?task_id={tid}").json()[0]["id"] == conv["id"]

    # 有 running 作业时禁止删除（不产生幽灵 agent）；把它标结束后才能删
    assert c.delete(f"/api/conversations/{conv['id']}").status_code == 409
    with Session() as db:
        job = db.get(AgentJob, r.json()["job_id"])
        job.status = "ok"
        db.commit()
    assert c.delete(f"/api/conversations/{conv['id']}").status_code == 204
    assert c.get(f"/api/conversations/{conv['id']}").status_code == 404


def test_finish_prefers_final_over_streamed_prose(conv_db, seeded, monkeypatch):
    """最终答复优先于流式叙述：多步 agent 的 prose 是所有中间 assistant 消息的拼接，
    直接落库会变成「回复全在一起了」（用户 2026-10-21 实测反馈）。"""
    task = seeded["task"]
    conv = svc.create(conv_db, task.id)
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: None)
    job = svc.send(conv_db, conv, "继续开发")
    state = SimpleNamespace(
        prose="我先加载技能。\n接下来检查工作区。\n再看看契约。",
        final="需求已对齐，即将开始静默开发；完成后或需要你决定时再通知你。",
        session_id="session-x", usage={"inputTokens": 1, "outputTokens": 2}, error=None,
    )
    svc._finish(job.id, conv.id, state)
    conv_db.expire_all()
    msgs = list(conv_db.query(Message).filter_by(conversation_id=conv.id, role="assistant"))
    assert len(msgs) == 1
    assert msgs[0].text == state.final
    assert "我先加载技能" not in msgs[0].text


@pytest.mark.parametrize("fallback", [False, True])
def test_chat_case_identity_matches_operator_scope(conv_db, seeded, monkeypatch, fallback):
    """Other operators must not change the ref injected into an already confirmed task."""
    from app.services import case_confirm

    task = seeded["task"]
    ws = Path(seeded["project"].workspace_path)
    own = ws / "cases" / task.op_name
    own.mkdir()
    (own / "cases.json").write_text('{"cases": []}')
    (own / "benchmark.json").write_text('{"benchmarks": []}')
    task.cases_confirmed_scope = "operator-v1"
    task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(ws, task.op_name)
    conv_db.commit()
    if fallback:
        monkeypatch.setattr(svc, "_case_confirm_api", lambda: None)
    expected = case_confirm.cases_ref(ws, task.op_name)
    assert expected != case_confirm.cases_ref(ws)
    state = svc._cases_confirmation(conv_db, task, ws, task.op_name)
    assert state["state"] == "confirmed" and state["current_ref"] == expected
    foreign = ws / "cases" / "other_op"
    foreign.mkdir()
    (foreign / "cases.json").write_text('{"unrelated": true}')
    prompt = svc._cases_prompt(conv_db, task, ws, "http://example.test", "test")
    assert expected in prompt and case_confirm.cases_ref(ws) not in prompt
    (own / "cases.json").write_text('{"changed": true}')
    changed = svc._cases_confirmation(conv_db, task, ws, task.op_name)
    assert changed["state"] == "drift"
    assert changed["current_ref"] == case_confirm.cases_ref(ws, task.op_name)
    assert changed["current_ref"] != expected
