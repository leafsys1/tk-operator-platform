"""自动优化 Step 1b 验收：确认草案/一次性 token / loop_run_links / attempt 历史 / 回滚租约。

契约见 `docs/plan-optimize-auto-v1.md` §15。全部用临时库/临时工作区/临时归档根，
dsh 与模型都假掉（monkeypatch `agent_runner.start`），不跑真机、不花 token。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import settings
from app.models import (AgentJob, Conversation, Loop, LoopAttempt, LoopConfirmation,
                        LoopEvent, LoopRunLink, Message, Project, Run, Savepoint, SpecVersion,
                        Task)
from app.routers import runs as runs_router
from app.services import agent_runner, loop_state, optimize, savepoints, workspace_lease

OP = "elementwise_add"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@deadbeef"
DIGEST = "sha256:cafe"


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path: Path, monkeypatch):
    """每个用例独立归档根；清掉 agent 自动优化的模块级运行态。"""
    monkeypatch.setattr(settings, "savepoint_root", str(tmp_path / "archives"))
    optimize.PENDING.clear()
    yield
    optimize.PENDING.clear()


# ---------------------------------------------------------------- 造数据小工具


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_dir(root: Path, name: str, *, snapshot: str,
             perf_cases: list[dict] | None = None,
             corr_cases: list[dict] | None = None, device: str = "1") -> Path:
    """造一个 gate 可判 pass 的 run 产物目录（manifest + 机读报告 + log）。"""
    d = root / name
    (d / "out").mkdir(parents=True, exist_ok=True)
    arts: list[dict] = []
    common = {"spec_version_id": 1, "code_snapshot_sha256": snapshot,
              "criteria_ref": CRITERIA, "suite_ref": SUITE}
    if perf_cases is not None:
        perf = {**common, "cases": perf_cases,
                "summary": {"primary_case": perf_cases[0]["case_id"] if perf_cases else None}}
        body = json.dumps(perf, ensure_ascii=False).encode()
        (d / "out" / "performance.json").write_bytes(body)
        arts.append({"type": "performance", "path": "out/performance.json", "sha256": _sha(body)})
    if corr_cases is not None:
        corr = {**common, "cases": corr_cases}
        body = json.dumps(corr, ensure_ascii=False).encode()
        (d / "out" / "correctness.json").write_bytes(body)
        arts.append({"type": "correctness", "path": "out/correctness.json", "sha256": _sha(body)})
    log_body = b"ok\n"
    (d / "out" / "run.log").write_bytes(log_body)
    arts.append({"type": "log", "path": "out/run.log", "sha256": _sha(log_body)})
    manifest = {
        "schema": "opx-manifest/1", "run_id": name, "exit_code": 0, "image_digest": DIGEST,
        "code_snapshot_sha256": snapshot, "spec_version_id": 1, "criteria_ref": CRITERIA,
        "suite_ref": SUITE, "device": device, "artifacts": arts,
    }
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _perf_case(case_id: str, med_us: float) -> dict:
    return {"case_id": case_id, "n": 1048576,
            "candidate": {"device_us": {"median_us": med_us, "std_us": 0.02}},
            "baseline": {"device_us": {"median_us": med_us + 0.3}}}


def _corr(case_id: str, status: str = "PASS") -> dict:
    return {"case_id": case_id, "status": status, "required": True}


def _add_run(db, task: Task, *, action: str, manifest: Path, snapshot: str,
             loop_id: int | None = None) -> Run:
    run = Run(task_id=task.id, loop_id=loop_id, action=action, status="ok", exit_code=0,
              manifest_path=str(manifest), code_snapshot_sha256=snapshot,
              criteria_ref=CRITERIA, suite_ref=SUITE, image_digest=DIGEST,
              device_id="1", spec_version_id=1,
              command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action {action}")
    db.add(run)
    db.flush()
    return run


def _seed_api(Session, workspace: Path, *, name: str = "p") -> tuple[int, int]:
    with Session() as db:
        project = Project(name=name, workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=OP, op_name=OP, target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return project.id, task.id


def _write_kernel(workspace: Path, text: str) -> None:
    path = workspace / "ops" / OP / "op_host" / "add.asc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_round(c, task_id: int) -> dict:
    r = c.post(f"/api/tasks/{task_id}/optimization-rounds",
               json={"objective": {"type": "perf", "target_dimension": "all"}})
    assert r.status_code == 201, r.text
    return r.json()


def _seed_ready_round(c, Session, workspace: Path, tmp_path: Path) -> tuple[int, int, str]:
    """建一个「基线已就绪」的轮次：before test/bench 用 link 复用（不改 runs.loop_id）。"""
    _pid, tid = _seed_api(Session, workspace)
    _write_kernel(workspace, "// kernel v1\n")
    sha_a = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "bt-bench", snapshot=sha_a, perf_cases=[_perf_case("bench_1m", 10.0)])
    mt = _run_dir(tmp_path, "bt-test", snapshot=sha_a, corr_cases=[_corr("case_001")])
    with Session() as db:
        task = db.get(Task, tid)
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha_a)
        _add_run(db, task, action="test", manifest=mt, snapshot=sha_a)
        db.commit()
    body = _create_round(c, tid)
    loop_id = body["loop"]["id"]
    assert body["loop"]["phase"] == "baseline_ready", body["loop"]
    return tid, loop_id, sha_a


def _fake_agent(monkeypatch) -> dict:
    """假 dsh：只记 start 的参数，返回一个 JobState；不跑模型。"""
    captured: dict = {}

    def fake_start(job_id: int, prompt: str, **kwargs):
        captured.update(kwargs)
        captured["job_id"] = job_id
        return agent_runner.JobState(job_id, prompt)

    monkeypatch.setattr(agent_runner, "start", fake_start)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")
    return captured


# ---------------------------------------------------------------- confirm / start


def test_confirm_requires_baseline_ready(client, workspace: Path):
    """G1 未就绪 → confirm 409（E_BASELINE_*）；旧 Loop（无 objective）也只读。"""
    c, Session = client
    _pid, tid = _seed_api(Session, workspace)
    _write_kernel(workspace, "// kernel v1\n")
    loop_id = _create_round(c, tid)["loop"]["id"]
    assert c.get(f"/api/loops/{loop_id}/phase").json()["phase"] == "baseline_pending"

    r = c.post(f"/api/loops/{loop_id}/confirm", json={"instruction": "x"})
    assert r.status_code == 409, r.text
    assert "E_BASELINE_" in r.text
    with Session() as db:
        assert db.scalar(select(LoopConfirmation)) is None
        assert any(e.kind == "created" for e in db.scalars(
            select(LoopEvent).where(LoopEvent.loop_id == loop_id)))


def test_confirm_start_consumes_token_once_and_records_attempt(client, workspace: Path,
                                                               tmp_path: Path, monkeypatch):
    """confirm 冻结草案 → start 消费 token 一次 → 重复 409 → attempt 历史 running/succeeded。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    captured = _fake_agent(monkeypatch)

    r = c.post(f"/api/loops/{loop_id}/confirm",
               json={"instruction": "保守调分块", "note": "人工确认"})
    assert r.status_code == 201, r.text
    conf = r.json()
    token = conf["token"]
    assert conf["loop_id"] == loop_id
    assert conf["draft"]["baseline_sha256"] == sha_a
    assert conf["draft"]["instruction"] == "保守调分块"
    assert conf["draft"]["workspace_sha256"] == sha_a
    assert conf["draft"]["note"] == "人工确认"
    # 明文 token 只在响应里出现，库里存 sha256；确认事件已留痕
    with Session() as db:
        row = db.scalar(select(LoopConfirmation).where(LoopConfirmation.loop_id == loop_id))
        assert row is not None and row.used_at is None
        assert row.token_hash == hashlib.sha256(token.encode()).hexdigest()
        assert token not in (row.draft_json or "")
        assert any(e.kind == "confirm" for e in db.scalars(
            select(LoopEvent).where(LoopEvent.loop_id == loop_id)))

    r = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r.status_code == 201, r.text
    started = r.json()
    job_id, conv_id = started["job_id"], started["conversation_id"]
    assert started["loop_id"] == loop_id and started["message_id"]
    assert captured["job_id"] == job_id
    # 指令来自冻结草案（而不是请求体）
    assert "保守调分块" in started["prompt"]

    # 同一 token 重放：一次性 → 409，且不会起第二个 job
    r2 = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r2.status_code == 409, r2.text
    assert "E_CONFIRM_TOKEN" in r2.text

    with Session() as db:
        row = db.scalar(select(LoopConfirmation).where(LoopConfirmation.loop_id == loop_id))
        assert row.used_at is not None and row.used_agent_job_id == job_id
        loop = db.get(Loop, loop_id)
        assert loop.phase == "agent_running" and loop.active_job_id == job_id
        attempt = db.scalar(select(LoopAttempt).where(LoopAttempt.loop_id == loop_id))
        assert attempt is not None
        assert attempt.status == "running" and attempt.seq == 1
        assert attempt.agent_job_id == job_id and attempt.conversation_id == conv_id
        assert attempt.baseline_sha256 == sha_a
        conv = db.get(Conversation, conv_id)
        assert conv is not None and conv.task_id == tid
        msg = db.scalar(select(Message).where(Message.conversation_id == conv_id,
                                              Message.role == "user"))
        assert msg is not None and "保守调分块" in msg.text

    data = c.get(f"/api/loops/{loop_id}/attempts").json()
    assert data["loop_id"] == loop_id
    assert [a["seq"] for a in data["attempts"]] == [1]
    assert data["attempts"][0]["status"] == "running"

    # agent 跑完（有候选）：attempt → succeeded + 候选 sha
    with Session() as db:
        loop_state.finish_agent(db, loop_id, {
            "status": "done", "changed": True, "guard_reverted": False,
            "after_sha256": "b" * 64,
        })
    attempt = c.get(f"/api/loops/{loop_id}/attempts").json()["attempts"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["candidate_sha256"] == "b" * 64
    assert attempt["ended_at"] is not None


def test_start_rejects_objective_change_after_confirm(client, workspace: Path, tmp_path: Path,
                                                      monkeypatch):
    """确认后目标被改 → start 409（E_CONFIRM_OBJECTIVE_CHANGED），token 不被消耗。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _fake_agent(monkeypatch)
    token = c.post(f"/api/loops/{loop_id}/confirm", json={}).json()["token"]

    with Session() as db:
        loop = db.get(Loop, loop_id)
        objective = dict(loop.objective or {})
        objective["focus"] = "确认后临时改的目标"
        loop.objective_json = json.dumps(objective, ensure_ascii=False)
        db.commit()

    r = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r.status_code == 409, r.text
    assert "E_CONFIRM_OBJECTIVE_CHANGED" in r.text
    with Session() as db:
        row = db.scalar(select(LoopConfirmation).where(LoopConfirmation.loop_id == loop_id))
        assert row.used_at is None
        loop = db.get(Loop, loop_id)
        assert loop.phase == "baseline_ready"


def test_start_rejects_workspace_change_after_confirm(client, workspace: Path, tmp_path: Path,
                                                      monkeypatch):
    """确认后工作区代码被改 → start 409（E_CONFIRM_BASELINE_CHANGED），token 不被消耗。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _fake_agent(monkeypatch)
    token = c.post(f"/api/loops/{loop_id}/confirm", json={}).json()["token"]
    _write_kernel(workspace, "// 手改过，没重新跑基线\n")

    r = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r.status_code == 409, r.text
    assert "E_CONFIRM_BASELINE_CHANGED" in r.text
    with Session() as db:
        row = db.scalar(select(LoopConfirmation).where(LoopConfirmation.loop_id == loop_id))
        assert row.used_at is None


def test_start_rejects_expired_or_foreign_token(client, workspace: Path, tmp_path: Path,
                                                monkeypatch):
    """过期 token 与「别人的 token」都 409；缺 token 也 409。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _fake_agent(monkeypatch)
    assert c.post(f"/api/loops/{loop_id}/start", json={"token": ""}).status_code == 409
    token = c.post(f"/api/loops/{loop_id}/confirm", json={}).json()["token"]

    with Session() as db:
        row = db.scalar(select(LoopConfirmation).where(LoopConfirmation.loop_id == loop_id))
        row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
    r = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r.status_code == 409 and "E_CONFIRM_TOKEN" in r.text

    # 换一个轮次：token 属于上一个 loop，不能跨轮次用
    other = _create_round(c, _tid)
    r = c.post(f"/api/loops/{other['loop']['id']}/start", json={"token": token})
    assert r.status_code == 409 and "E_CONFIRM_TOKEN" in r.text


def test_legacy_chat_still_works_and_marks_auto_confirm(client, workspace: Path, tmp_path: Path,
                                                        monkeypatch):
    """兼容：/optimize/chat 不传 token 仍能启动，并写 confirm_auto 事件 + attempt。"""
    c, Session = client
    tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _fake_agent(monkeypatch)
    r = c.post(f"/api/loops/{loop_id}/optimize/chat", json={"instruction": "legacy 路径"})
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    with Session() as db:
        kinds = [e.kind for e in db.scalars(
            select(LoopEvent).where(LoopEvent.loop_id == loop_id).order_by(LoopEvent.id))]
        assert "confirm_auto" in kinds
        attempt = db.scalar(select(LoopAttempt).where(LoopAttempt.loop_id == loop_id))
        assert attempt is not None and attempt.agent_job_id == job_id and attempt.status == "running"


# ---------------------------------------------------------------- loop_run_links 证据并集


def test_accept_works_when_before_runs_are_only_linked(client, workspace: Path, tmp_path: Path):
    """复用来的 before run 只有 link（runs.loop_id 为空）时，G7 采纳仍能配对两侧证据。"""
    c, Session = client
    _tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    with Session() as db:
        task = db.get(Task, db.get(Loop, loop_id).task_id)
        # agent 改码后的候选侧：显式归属本轮（loop_id），并落 candidate 存点
        _write_kernel(workspace, "// kernel v2\n")
        sha_b = savepoints.current_sha(workspace)
        mb = _run_dir(tmp_path, "after-bench", snapshot=sha_b,
                      perf_cases=[_perf_case("bench_1m", 8.0)])
        mt = _run_dir(tmp_path, "after-test", snapshot=sha_b, corr_cases=[_corr("case_001")])
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha_b, loop_id=loop_id)
        _add_run(db, task, action="test", manifest=mt, snapshot=sha_b, loop_id=loop_id)
        savepoints.create_for_sha(db, task, sha_b, label="候选 · 分块", kind="candidate",
                                  loop_id=loop_id)
        # 复用来的 before run 只有 link
        linked = {l.run_id for l in db.scalars(
            select(LoopRunLink).where(LoopRunLink.loop_id == loop_id))}
        before = {r.id for r in db.scalars(select(Run).where(Run.code_snapshot_sha256 == sha_a))}
        assert linked == before
        assert all(db.get(Run, rid).loop_id is None for rid in linked)
        db.commit()

    r = c.post(f"/api/loops/{loop_id}/accept", json={"label": "c1"})
    assert r.status_code == 200, r.text
    assert r.json()["loop"]["status"] == "closed"
    assert r.json()["loop"]["phase"] == "accepted"


# ---------------------------------------------------------------- savepoint restore 持租约


def test_restore_acquires_and_releases_workspace_lease(db_session, seeded, workspace: Path):
    """回滚前必须抢到项目级租约（DB 级单飞）；回滚结束/失败都释放。"""
    task = seeded["task"]
    sp = savepoints.create(db_session, task, label="存档", kind="manual")
    db_session.commit()

    held = workspace_lease.acquire(db_session, task, kind="agent")
    db_session.commit()
    assert held is not None
    with pytest.raises(savepoints.SavepointBusyError):
        savepoints.restore(db_session, sp, task)
    # 租约仍是别人的（busy 失败不能把别人的租约删掉）
    assert workspace_lease.current(db_session, task.project_id) is not None

    workspace_lease.release(db_session, project_id=task.project_id, kind="agent")
    out = savepoints.restore(db_session, sp, task)
    assert out["in_sync"] is True and out["restored"] is False
    assert workspace_lease.current(db_session, task.project_id) is None


# ---------------------------------------------------------------- 审查修复回归（A1–A3 / B1–B3）


def test_start_endpoint_must_be_async():
    """A1：/start 必须 async —— 同步端点在线程池里没有 event loop，agent_runner.start 必 500。"""
    import inspect

    assert inspect.iscoroutinefunction(runs_router.start_optimization_round)
    assert inspect.iscoroutinefunction(runs_router.start_loop_optimize_chat)


def test_confirm_reports_busy_with_job_id(client, workspace: Path, tmp_path: Path):
    """工作区有作业在跑时，confirm 给 E_CONFIRM_BUSY（不是「基线未就绪」）。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    with Session() as db:
        loop = db.get(Loop, loop_id)
        db.add(AgentJob(task_id=loop.task_id, kind="优化", status="running", prompt="x"))
        db.commit()
    r = c.post(f"/api/loops/{loop_id}/confirm", json={})
    assert r.status_code == 409
    assert "E_CONFIRM_BUSY" in r.text and "agent#" in r.text


def test_start_launch_failure_cleans_pending_and_marks_failed(client, workspace: Path,
                                                              tmp_path: Path, monkeypatch):
    """A1/A3：起 agent 抛异常 → 500，但 PENDING/租约/job/attempt 都要收尾，不能永久 409。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _n: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    def boom(*_a, **_kw):
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(agent_runner, "start", boom)
    monkeypatch.setattr(runs_router.optimize, "start", lambda *a, **kw: boom())
    token = c.post(f"/api/loops/{loop_id}/confirm", json={}).json()["token"]
    r = c.post(f"/api/loops/{loop_id}/start", json={"token": token})
    assert r.status_code == 500, r.text
    assert optimize.PENDING == {}
    with Session() as db:
        loop = db.get(Loop, loop_id)
        assert loop.phase == "failed" and loop.active_job_id is None
        job = db.scalar(select(AgentJob).where(AgentJob.loop_id == loop_id))
        assert job is not None and job.status == "failed"
        attempt = db.scalar(select(LoopAttempt).where(LoopAttempt.loop_id == loop_id))
        assert attempt is not None and attempt.status == "failed"
        assert workspace_lease.current(db, loop.task_id and db.get(Task, loop.task_id).project_id) is None


def test_candidate_savepoint_from_link_only_baseline(db_session, seeded, workspace: Path):
    """A2：before run 只有 link 时，benchmark 收尾仍要生成 candidate（否则 accept 永久 409）。"""
    from app.models import ActionJob

    task = seeded["task"]
    loop = Loop(task_id=task.id, kind="perf", scope="s",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.flush()
    sha_a = savepoints.current_sha(workspace)
    baseline = savepoints.create(db_session, task, label="基线", kind="baseline", loop_id=loop.id)
    assert baseline.code_snapshot_sha256 == sha_a
    # before 侧两个 run 只挂 link（Step 1b 语义：不改 runs.loop_id）
    before_test = Run(task_id=task.id, action="test", status="ok",
                      code_snapshot_sha256=sha_a)
    before_bench = Run(task_id=task.id, action="benchmark", status="ok",
                       code_snapshot_sha256=sha_a)
    db_session.add_all([before_test, before_bench])
    db_session.flush()
    db_session.add_all([
        LoopRunLink(loop_id=loop.id, run_id=before_test.id, role="baseline"),
        LoopRunLink(loop_id=loop.id, run_id=before_bench.id, role="baseline"),
    ])
    # agent 改码后的候选侧：同 sha 的 test + benchmark（显式归属本轮）
    _write_kernel(workspace, "// kernel v2\n")
    sha_b = savepoints.current_sha(workspace)
    cand_test = Run(task_id=task.id, loop_id=loop.id, action="test", status="ok",
                    code_snapshot_sha256=sha_b)
    cand_bench = Run(task_id=task.id, loop_id=loop.id, action="benchmark", status="ok",
                     code_snapshot_sha256=sha_b)
    db_session.add_all([cand_test, cand_bench])
    job = ActionJob(task_id=task.id, loop_id=loop.id, action="benchmark", status="ok",
                    run_ref="probe-run")
    db_session.add(job)
    db_session.flush()

    from app.services import action_runner
    action_runner._maybe_candidate_savepoint(db_session, job, cand_bench)

    cand = db_session.scalar(select(Savepoint).where(
        Savepoint.loop_id == loop.id, Savepoint.kind == "candidate"))
    assert cand is not None and cand.code_snapshot_sha256 == sha_b


def test_cancel_does_not_release_other_leases(client, workspace: Path, tmp_path: Path):
    """B1：取消一个没有 active job 的轮次，不能删掉同任务别人的 agent/action 租约。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    with Session() as db:
        loop = db.get(Loop, loop_id)
        task = db.get(Task, loop.task_id)
        lease = workspace_lease.acquire(db, task, kind="action", note="测试用外部租约")
        lease.job_id = 777
        db.commit()
        project_id = task.project_id

    r = c.post(f"/api/loops/{loop_id}/cancel", json={"reason": "test"})
    assert r.status_code == 200, r.text
    assert r.json()["loop"]["phase"] == "cancelled"
    with Session() as db:
        still = workspace_lease.current(db, project_id)
        assert still is not None and still.kind == "action" and still.job_id == 777


def test_reject_refuses_while_job_running(client, workspace: Path, tmp_path: Path):
    """B3：有作业在写工作区时否决 → 409（避免 attempt 错标 + closed 轮次被收尾改回）。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    with Session() as db:
        loop = db.get(Loop, loop_id)
        db.add(AgentJob(task_id=loop.task_id, kind="优化", status="running", prompt="x"))
        db.commit()
    r = c.post(f"/api/loops/{loop_id}/reject", json={"reason": "x"})
    assert r.status_code == 409
    assert "E_REJECT_BUSY" in r.text
    with Session() as db:
        assert db.get(Loop, loop_id).status == "open"


def test_start_without_token_returns_409(client, workspace: Path, tmp_path: Path):
    """B13：缺 token 走业务 409 + E_CONFIRM_TOKEN，而不是 FastAPI 422。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    r = c.post(f"/api/loops/{loop_id}/start", json={})
    assert r.status_code == 409 and "E_CONFIRM_TOKEN" in r.text


def test_legacy_chat_with_confirm_token_uses_frozen_draft(client, workspace: Path, tmp_path: Path,
                                                          monkeypatch):
    """B13：/optimize/chat 带 confirm_token 时，指令取冻结 draft，请求体不能覆盖。"""
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _fake_agent(monkeypatch)
    token = c.post(f"/api/loops/{loop_id}/confirm",
                   json={"instruction": "冻结的指令 A"}).json()["token"]
    r = c.post(f"/api/loops/{loop_id}/optimize/chat",
               json={"instruction": "请求体里的指令 B", "confirm_token": token})
    assert r.status_code == 201, r.text
    assert "冻结的指令 A" in r.json()["prompt"]
    assert "请求体里的指令 B" not in r.json()["prompt"]
