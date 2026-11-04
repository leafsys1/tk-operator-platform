"""自动优化 Step 1 验收：phase 状态机 / 事件 / 幂等建轮 / 工作区租约 / 重启恢复。

契约见 `docs/plan-optimize-auto-v1.md` §5、§6、§8。全部用临时库/临时工作区，不跑 dsh/真机。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models import (ActionJob, AgentJob, Loop, LoopAttempt, LoopEvent, Project,
                        SpecVersion, Task, WorkspaceLease)
from app.services import loop_state, workspace_lease


def _seed(Session, workspace: Path) -> tuple[int, int]:
    with Session() as db:
        project = Project(name="p", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add",
                    op_name="elementwise_add", target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return project.id, task.id


def test_create_round_phase_events_and_idempotency(client, workspace: Path):
    """建轮次写 phase/事件；同 idempotency_key 双击只建一个 Loop。"""
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    body = {
        "objective": {"type": "perf", "target_dimension": "all"},
        "idempotency_key": "confirm-001",
    }
    r1 = c.post(f"/api/tasks/{tid}/optimization-rounds", json=body)
    assert r1.status_code == 201, r1.text
    loop1 = r1.json()["loop"]
    assert loop1["phase"] == "baseline_pending"
    assert loop1["state_version"] >= 0

    r2 = c.post(f"/api/tasks/{tid}/optimization-rounds", json=body)
    assert r2.status_code == 201, r2.text
    assert r2.json()["loop"]["id"] == loop1["id"]

    with Session() as db:
        assert db.scalar(select(func.count()).select_from(Loop).where(Loop.task_id == tid)) == 1
        events = list(db.scalars(select(LoopEvent).where(LoopEvent.loop_id == loop1["id"])))
    assert any(e.kind == "created" for e in events)

    phase = c.get(f"/api/loops/{loop1['id']}/phase").json()
    assert phase["phase"] == "baseline_pending"
    assert phase["events"] and phase["events"][0]["kind"] in ("created", "baseline_synced")


def test_state_machine_cas_and_events(db_session, seeded):
    """start_agent 用 state_version + active_job_id CAS；重复启动失败；结束推进 phase。"""
    task = seeded["task"]
    loop = Loop(task_id=task.id, kind="perf", scope="s", hypothesis="h",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.commit()
    loop_id = loop.id

    assert loop_state.start_agent(db_session, loop, 101, commit=True) is True
    db_session.expire_all()
    loop = db_session.get(Loop, loop_id)
    assert loop.phase == "agent_running" and loop.active_job_id == 101

    # 第二个并发启动：active_job_id 非空 → CAS 失败
    assert loop_state.start_agent(db_session, loop, 102) is False
    db_session.expire_all()
    loop = db_session.get(Loop, loop_id)
    assert loop.active_job_id == 101

    loop_state.finish_agent(db_session, loop_id, {
        "status": "done", "changed": True, "guard_reverted": False,
        "after_sha256": "b" * 64,
    })
    db_session.expire_all()
    loop = db_session.get(Loop, loop_id)
    assert loop.phase == "awaiting_verification" and loop.active_job_id is None
    kinds = [e.kind for e in db_session.scalars(
        select(LoopEvent).where(LoopEvent.loop_id == loop_id).order_by(LoopEvent.id))]
    # Step 1b：start 之后会补一条 attempt_start（一次 agent 改码尝试一行）
    assert kinds[:3] == ["start", "attempt_start", "agent_done"]
    attempt = db_session.scalar(select(LoopAttempt).where(LoopAttempt.loop_id == loop_id))
    assert attempt is not None
    assert attempt.status == "succeeded" and attempt.agent_job_id == 101
    assert attempt.candidate_sha256 == "b" * 64 and attempt.ended_at is not None


def test_finish_agent_without_candidate_goes_failed(db_session, seeded):
    task = seeded["task"]
    loop = Loop(task_id=task.id, kind="perf", objective_json="{}", phase="agent_running",
                active_job_id=7)
    db_session.add(loop)
    db_session.commit()
    loop_state.finish_agent(db_session, loop.id, {
        "status": "failed", "changed": False, "guard_reverted": True,
        "guard_reason": "E_PROTECTED_PATH",
    })
    db_session.expire_all()
    loop = db_session.get(Loop, loop.id)
    assert loop.phase == "failed" and loop.active_job_id is None
    assert "E_PROTECTED_PATH" in (loop.phase_error or "")


@pytest.mark.filterwarnings("ignore:Identity map already had an identity")
def test_workspace_lease_conflict_release_and_stale(client, workspace: Path):
    """同一 project 只能有一个租约；release 后可再占；过期/作业不在跑可被抢占。"""
    c, Session = client
    pid, tid = _seed(Session, workspace)
    with Session() as db:
        task = db.get(Task, tid)
        agent_job = AgentJob(task_id=tid, kind="优化", status="running")
        db.add(agent_job)
        db.flush()
        lease = workspace_lease.acquire(db, task, kind="agent", job_id=agent_job.id)
        db.commit()
        assert lease.project_id == pid

        # 同 project 的另一个任务（或 action）不能抢
        other = Task(project_id=pid, title="other", op_name="elementwise_add")
        db.add(other)
        db.commit()
        with pytest.raises(workspace_lease.LeaseBusy):
            workspace_lease.acquire(db, other, kind="action", job_id=999)
        db.rollback()

        # 释放后可以再占（用一个真实 running 的 ActionJob）
        action_job = ActionJob(task_id=tid, action="test", status="running", run_ref="r-lease")
        db.add(action_job)
        db.flush()
        workspace_lease.release(db, task_id=tid, job_id=agent_job.id)
        lease2 = workspace_lease.acquire(db, other, kind="action", job_id=action_job.id)
        db.commit()
        assert lease2.kind == "action" and lease2.job_id == action_job.id

        # 作业已不在 running → stale，可被抢占
        action_job.status = "ok"
        db.commit()
        lease3 = workspace_lease.acquire(db, task, kind="agent", job_id=999)
        db.commit()
        assert lease3.task_id == tid and lease3.job_id == 999

        # 过期租约也可被抢占
        row = workspace_lease.current(db, pid)
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        lease4 = workspace_lease.acquire(db, other, kind="action", job_id=1000)
        db.commit()
        assert lease4.job_id == 1000


def test_reconcile_startup_blocks_running_loops_and_clears_leases(client, workspace: Path):
    """重启：running phase → blocked，active_job 清空；租约清空（本地进程已不在）。"""
    c, Session = client
    pid, tid = _seed(Session, workspace)
    with Session() as db:
        loop = Loop(task_id=tid, kind="perf", objective_json="{}", phase="agent_running",
                    active_job_id=9)
        db.add(loop)
        task = db.get(Task, tid)
        workspace_lease.acquire(db, task, kind="agent", job_id=9)
        db.commit()
        loop_id = loop.id

        assert loop_state.reconcile_startup(db) == 1
        assert workspace_lease.reconcile_startup(db) == 1

        loop = db.get(Loop, loop_id)
        assert loop.phase == "blocked" and loop.active_job_id is None
        assert "重启" in (loop.phase_error or "")
        assert db.scalar(select(func.count()).select_from(WorkspaceLease)) == 0


def test_cancel_round_marks_cancelled_and_releases_lease(client, workspace: Path):
    """取消：杀 agent 进程（若有）+ phase=cancelled + 释放租约；不静默回滚工作区。"""
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    loop = c.post(f"/api/tasks/{tid}/optimization-rounds",
                  json={"objective": {"type": "perf", "target_dimension": "all"}}).json()["loop"]
    with Session() as db:
        task = db.get(Task, tid)
        job = AgentJob(task_id=tid, kind="优化", status="running")
        db.add(job)
        db.flush()
        row = db.get(Loop, loop["id"])
        row.phase = "agent_running"
        row.active_job_id = job.id
        db.commit()
        workspace_lease.acquire(db, task, kind="agent", job_id=job.id)
        db.commit()
        job_id = job.id

    r = c.post(f"/api/loops/{loop['id']}/cancel", json={"reason": "用户取消"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loop"]["phase"] == "cancelled"
    assert body["loop"]["active_job_id"] is None
    with Session() as db:
        assert db.get(AgentJob, job_id).status == "running"  # 无真实进程，取消函数不动 DB 状态
        assert db.scalar(select(func.count()).select_from(WorkspaceLease)) == 0


def test_action_lease_blocks_agent_side(client, workspace: Path):
    """动作作业占着租约时，agent 侧（对话/优化）必须被同一把租约挡住。"""
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    with Session() as db:
        task = db.get(Task, tid)
        job = ActionJob(task_id=tid, action="benchmark", status="running", run_ref="r-busy")
        db.add(job)
        db.flush()
        workspace_lease.acquire(db, task, kind="action", job_id=job.id)
        db.commit()
        from app.routers import runs as runs_router
        with pytest.raises(runs_router.HTTPException) as exc:
            runs_router._acquire_agent_lease(db, task)
        assert exc.value.status_code == 409
