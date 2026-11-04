"""智能体调用（agent_runner）生命周期：起进程失败 / 超时 / 取消都要落库，REGISTRY 不泄漏。

不跑真 dsh：把 `build_command` 换成「不存在的可执行文件」/ `sleep` / 立刻退出的 shell，
专测 `_consume` 的兜底路径与 `cancel()`（对应审查 #1 / #10）。

隔离：`app.db.SessionLocal` 被换成临时库的 sessionmaker —— `_consume` / `cancel` 内部都是
延迟 `from ..db import SessionLocal`，所以能拿到替换后的值，不碰 backend/opx.db。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models import AgentJob, Project, Task
from app.services import agent_runner


@pytest.fixture()
def store(tmp_path: Path, monkeypatch):
    """临时 SQLite + 把 app.db.SessionLocal 指过去（不经过 HTTP client）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.db as db_module
    from app.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'agent.db'}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_module, "SessionLocal", Session)
    return Session


def _seed_job(Session, workspace: Path) -> tuple[int, int]:
    """项目 + 任务 + 一条 running 的 AgentJob，返回 (task_id, job_id)。"""
    with Session() as db:
        project = Project(name="p-agent-runner", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="t", op_name="t")
        db.add(task)
        db.flush()
        job = AgentJob(task_id=task.id, kind="需求澄清", status="running", prompt="p",
                       started_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        return task.id, job.id


def test_consume_start_failure_marks_failed_and_cleans_registry(store, tmp_path: Path, monkeypatch):
    """dsh 不在 PATH（起进程抛 FileNotFoundError）也要把 job 收死，不能留 running 僵尸。"""
    Session = store
    _tid, jid = _seed_job(Session, tmp_path)
    monkeypatch.setattr(agent_runner, "build_command",
                        lambda *a, **kw: ["/nonexistent/dsh-opx-test", "--profile", "headless"])
    called: list = []
    state = agent_runner.JobState(jid, "p")
    agent_runner.REGISTRY[jid] = state

    asyncio.run(agent_runner._consume(state, tmp_path, tmp_path / ".dsh", {},
                                      on_done=lambda s: called.append(s)))

    assert state.done is True
    assert state.error and "FileNotFoundError" in state.error
    assert state.exit_code == -1
    with Session() as db:
        row = db.get(AgentJob, jid)
        assert row is not None
        assert row.status == "failed"
        assert row.ended_at is not None
        assert row.exit_code == -1
    assert agent_runner.running(jid) is None  # 不再是 running（REGISTRY 里没有僵尸）
    # 有界保留最近结束的状态：`/live` 跑完仍能报 running=false + 错误原因
    assert agent_runner.REGISTRY.get(jid) is state and state.done
    from app.routers.runs import job_live

    live = job_live(jid)
    assert live["running"] is False
    assert "FileNotFoundError" in (live["error"] or "")
    assert called == [state]  # 失败也回调：对话页要落一条错误消息，而不是干等


def test_consume_timeout_kills_and_marks_failed(store, tmp_path: Path, monkeypatch):
    """总超时（settings.agent_timeout）到点杀进程并落 failed。"""
    Session = store
    _tid, jid = _seed_job(Session, tmp_path)
    monkeypatch.setattr(agent_runner, "build_command", lambda *a, **kw: ["bash", "-c", "sleep 30"])
    monkeypatch.setattr(agent_runner.settings, "agent_timeout", 1)
    state = agent_runner.JobState(jid, "p")
    agent_runner.REGISTRY[jid] = state

    started = time.monotonic()
    asyncio.run(agent_runner._consume(state, tmp_path, tmp_path / ".dsh", {}))
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"_consume 没有按 agent_timeout 退出（耗时 {elapsed:.1f}s）"
    assert state.error and "超时" in state.error
    assert state.exit_code not in (None, 0)
    with Session() as db:
        row = db.get(AgentJob, jid)
        assert row.status == "failed"
        assert row.ended_at is not None
    assert agent_runner.running(jid) is None
    assert agent_runner.REGISTRY.get(jid) is state  # 有界保留，/live 仍可读错误


def test_cancel_kills_and_marks_interrupted(store, tmp_path: Path, monkeypatch):
    """cancel() → 杀 dsh 进程组 + DB 写 interrupted（B 线取消端点依赖这个契约）。"""
    Session = store
    _tid, jid = _seed_job(Session, tmp_path)
    monkeypatch.setattr(agent_runner, "build_command", lambda *a, **kw: ["bash", "-c", "sleep 30"])

    async def scenario() -> None:
        state = agent_runner.start(jid, "p", cwd=tmp_path, dsh_home=tmp_path / ".dsh")
        for _ in range(200):  # 等进程真的起来
            if state.proc is not None:
                break
            await asyncio.sleep(0.01)
        assert agent_runner.running(jid) is state
        message = agent_runner.cancel(jid)
        assert "已发送终止信号" in message
        for _ in range(200):
            if state.done:
                break
            await asyncio.sleep(0.01)
        assert state.done and state.cancelled

    asyncio.run(scenario())

    with Session() as db:
        row = db.get(AgentJob, jid)
        assert row.status == "interrupted"  # 不是 failed
        assert row.ended_at is not None
    assert agent_runner.running(jid) is None  # 已不是 running；结束状态有界保留给 /live 读
    assert agent_runner.REGISTRY.get(jid) is not None
    assert "不在运行" in agent_runner.cancel(jid)  # 已收尾再取消：明确返回，不抛


def test_consume_success_persists_ok_and_keeps_state_for_replay(store, tmp_path: Path, monkeypatch):
    """正常路径回归：退出码 0 → ok；成功状态有界保留（晚订阅 /events 还能重放）。"""
    Session = store
    _tid, jid = _seed_job(Session, tmp_path)
    monkeypatch.setattr(
        agent_runner, "build_command",
        lambda *a, **kw: ["bash", "-c", 'echo \'{"type": "final", "text": "done"}\''])
    state = agent_runner.JobState(jid, "p")
    agent_runner.REGISTRY[jid] = state

    asyncio.run(agent_runner._consume(state, tmp_path, tmp_path / ".dsh", {}))

    assert state.exit_code == 0 and state.final == "done"
    assert agent_runner.running(jid) is None  # 不再是 running
    assert agent_runner.REGISTRY.get(jid) is state  # 保留给 /events 重放
    with Session() as db:
        row = db.get(AgentJob, jid)
        assert row.status == "ok"
        assert row.exit_code == 0 and row.ended_at is not None


def test_registry_bounded_after_finished_jobs():
    """成功状态有界保留：超过 RECENT_LIMIT 淘汰最旧的；还在跑的状态不受影响。"""
    agent_runner.REGISTRY.clear()
    try:
        finished = []
        for jid in range(1000, 1000 + agent_runner.RECENT_LIMIT + 5):
            st = agent_runner.JobState(jid, "p")
            st.done = True
            st.exit_code = 0
            agent_runner.REGISTRY[jid] = st
            finished.append(st)
        live = agent_runner.JobState(9999, "live")
        agent_runner.REGISTRY[9999] = live

        agent_runner._retire_state(finished[-1])

        done_ids = [jid for jid, st in agent_runner.REGISTRY.items() if st.done]
        assert len(done_ids) == agent_runner.RECENT_LIMIT
        assert 1000 not in agent_runner.REGISTRY  # 最旧的被淘汰
        assert agent_runner.REGISTRY[9999] is live and not live.done
    finally:
        agent_runner.REGISTRY.clear()

