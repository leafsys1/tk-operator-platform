"""自动验证链（Step 2 orchestrator）验收：预检 / 快乐路径 / 失败回退 / 取消 / 重启对账。

契约见 `docs/plan-optimize-auto-v1.md` §16。动作执行器与远端清理全部假掉（monkeypatch
`action_runner.start` / `remote_cancel.cancel_container`），不跑 dsh/真机。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import settings
from app.models import (ActionJob, Host, HostImage, Loop, LoopAttempt, LoopAutoRun, LoopRunLink,
                        Project, Run, Savepoint, SpecVersion, Task, utcnow)
from app.services import (action_runner, case_confirm, exec_actions, loop_orchestrator,
                        optimize, remote_cancel, savepoints)

OP = "elementwise_add"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@deadbeef"
DIGEST = "sha256:cafe"


@pytest.fixture(autouse=True)
def _orchestrator_uses_test_db(client):
    """orchestrator 的驱动任务走可注入 session 工厂：测试里必须指到 client 的临时库。"""
    _c, Session = client
    old = loop_orchestrator._session_factory
    loop_orchestrator._session_factory = Session
    yield
    loop_orchestrator._session_factory = old


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "savepoint_root", str(tmp_path / "archives"))
    optimize.PENDING.clear()
    yield
    optimize.PENDING.clear()


# ---------------------------------------------------------------- 造数据（与 test_step1b 同口径）


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_dir(root: Path, name: str, *, snapshot: str, action: str = "test") -> Path:
    """造一个 gate 可判 pass 的最小 run 产物（test 要 correctness.json，benchmark 要 performance.json）。"""
    d = root / name
    (d / "out").mkdir(parents=True, exist_ok=True)
    common = {"spec_version_id": 1, "code_snapshot_sha256": snapshot,
              "criteria_ref": CRITERIA, "suite_ref": SUITE}
    arts: list[dict] = []
    if action == "benchmark":
        body = json.dumps({**common, "cases": [{
            "case_id": "bench_1m", "n": 1048576, "dtype": "float32",
            "candidate": {"device_us": {"median_us": 10.0, "std_us": 0.1}},
            "baseline": {"device_us": {"median_us": 10.2}},
        }], "summary": {"primary_case": "bench_1m"}}, ensure_ascii=False).encode()
        (d / "out" / "performance.json").write_bytes(body)
        arts.append({"type": "performance", "path": "out/performance.json", "sha256": _sha(body)})
    else:
        body = json.dumps({**common, "cases": [_corr("case_001")]}, ensure_ascii=False).encode()
        (d / "out" / "correctness.json").write_bytes(body)
        arts.append({"type": "correctness", "path": "out/correctness.json", "sha256": _sha(body)})
    log_body = b"ok\n"
    (d / "out" / "run.log").write_bytes(log_body)
    arts.append({"type": "log", "path": "out/run.log", "sha256": _sha(log_body)})
    manifest = {
        "schema": "opx-manifest/1", "run_id": name, "exit_code": 0, "image_digest": DIGEST,
        **common, "device": "1", "artifacts": arts,
    }
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _corr(case_id: str) -> dict:
    return {"case_id": case_id, "status": "PASS", "required": True}


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


def _seed_api(Session, workspace: Path) -> tuple[int, int]:
    """项目 + 执行机 + 算子开发镜像 + 任务 + 规格 + 已确认用例组 + 四动作入口脚本。"""
    for name in ("build.sh", "smoke_check.sh", "test.sh", "benchmark.sh", "profile.sh"):
        path = workspace / name
        if not path.exists():
            path.write_text("#!/usr/bin/env bash\necho stub\n", encoding="utf-8")
    (workspace / "cases").mkdir(exist_ok=True)
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, env_recipe="docker run --privileged ...",
                          torch_version="2.10.0+cpu / torch_npu 2.10.0.post4",
                          digest=DIGEST)
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id,
                          image_id=image.id, criteria_ref=CRITERIA)
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=OP, op_name=OP, target_chip="910B1",
                    device_id="0", cases_confirmed_at=utcnow(),
                    # origin/main 起：确认记录必须带当前 cases 内容哈希，否则 plan 会拦。
                    cases_confirmed_sha256=case_confirm.cases_suite_sha256(workspace),
                    cases_confirmed_by="test")
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
    _pid, tid = _seed_api(Session, workspace)
    _write_kernel(workspace, "// kernel v1\n")
    sha_a = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "bt-bench", snapshot=sha_a, action="benchmark")
    mt = _run_dir(tmp_path, "bt-test", snapshot=sha_a, action="test")
    with Session() as db:
        task = db.get(Task, tid)
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha_a)
        _add_run(db, task, action="test", manifest=mt, snapshot=sha_a)
        db.commit()
    body = _create_round(c, tid)
    loop_id = body["loop"]["id"]
    assert body["loop"]["phase"] == "baseline_ready", body["loop"]
    return tid, loop_id, sha_a


def _make_candidate(Session, workspace: Path, tid: int, loop_id: int,
                    sha_a: str) -> str:
    """agent 改完码：写新代码 + 写 optimize 证据 + phase=awaiting_verification。"""
    _write_kernel(workspace, "// kernel v2 (candidate)\n")
    sha_b = savepoints.current_sha(workspace)
    assert sha_b != sha_a
    with Session() as db:
        loop = db.get(Loop, loop_id)
        loop.optimize_json = json.dumps({
            "status": "done", "changed": True, "guard_reverted": False,
            "after_sha256": sha_b, "agent_summary": "改 kernel v2",
            "revert": {"before_sha256": sha_a, "after_sha256": sha_b},
        }, ensure_ascii=False)
        loop.phase = "awaiting_verification"
        loop.state_version = int(loop.state_version or 0) + 1
        # 让 preflight 的 G1 仍成立：baseline 存点 + before run 都没变（工作区变了但基线 sha 是历史值）
        db.commit()
    return sha_b


# ---------------------------------------------------------------- 假动作执行器


class _FakeActions:
    """按 action 决定作业结论；异步把 ActionJob 从 running 置终态（模拟 _finalize）。"""

    def __init__(self, *, fail_on: str | None = None, delay: float = 0.05,
                 create_candidate: bool = True):
        self.fail_on = fail_on
        self.delay = delay
        self.create_candidate = create_candidate
        self.started: list[str] = []

    def install(self, monkeypatch, Session):
        self.Session = Session
        monkeypatch.setattr(loop_orchestrator, "_session_factory", Session)
        monkeypatch.setattr(action_runner, "start", self.start)
        monkeypatch.setattr(remote_cancel, "cancel_container",
                            lambda db, task, run_ref: {"ok": True, "container": f"opx-{run_ref}"})
        monkeypatch.setattr(optimize, "revert", self.revert)
        # 假动作只落 ActionJob、不产 run/产物；smoke 硬前置由 test_smoke.py / test_plan.py 单独覆盖，
        # 这里只测 orchestrator 的顺序/门禁/回退，故把 smoke 前置判据假掉。
        monkeypatch.setattr(exec_actions, "smoke_gate_reason", lambda *a, **k: None)
        self.reverted: list[int] = []

    def revert(self, db, loop, workspace):
        self.reverted.append(loop.id)
        return {"reverted_clean": True, "revert_failed": []}

    def start(self, db, task, plan, *, loop_id=None, budget_audit_id=None, **_kw):
        self.started.append(plan.action)
        project = db.get(Project, task.project_id)
        snapshot = (savepoints.current_sha(Path(project.workspace_path).expanduser())
                    if project is not None and project.workspace_path else None)
        job = ActionJob(task_id=task.id, loop_id=loop_id, action=plan.action,
                        status="running", run_ref=f"auto-{task.id}-{plan.action}",
                        device_id=plan.device_id, command="fake")
        db.add(job)
        db.commit()
        db.refresh(job)
        fail = plan.action == self.fail_on

        async def complete():
            await asyncio.sleep(self.delay)
            with self.Session() as d:
                row = d.get(ActionJob, job.id)
                if row is not None and row.status == "running":
                    row.status = "failed" if fail else "ok"
                    row.gate_verdict = "fail" if fail else "pass"
                    row.exit_code = 1 if fail else 0
                    row.error = f"{plan.action} 模拟失败" if fail else None
                    row.ended_at = utcnow()
                    if (not fail and self.create_candidate and plan.action == "benchmark"
                            and loop_id is not None and snapshot):
                        task2 = d.get(Task, task.id)
                        if task2 is not None:
                            savepoints.create_for_sha(d, task2, snapshot, kind="candidate",
                                                      loop_id=loop_id, label="候选 · fake")
                    d.commit()

        asyncio.get_running_loop().create_task(complete())
        return job


def _fake_report(monkeypatch, *, written: bool = True):
    def report(db, loop, *, write=False):
        assert write is True
        return {
            "schema": "opx-optimization-report/1", "loop_id": loop.id,
            "written_to": f"/tmp/loop-{loop.id}/optimization-report.json" if written else None,
            "objective_summary": {"verdict": "pass", "conclusion": "improved"},
            "perf_diff": {"verdict": "pass", "conclusion": "improved"},
        }

    monkeypatch.setattr(loop_orchestrator.optimization, "report", report)


def _poll(c, loop_id: int, *, until=("done", "failed", "blocked", "cancelled"),
          timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = c.get(f"/api/loops/{loop_id}/auto-run").json()
        run = body.get("auto_run")
        if run and run.get("status") in until:
            return run
        time.sleep(0.1)
    raise AssertionError(f"自动链未在 {timeout}s 内到终态：{body}")


# ---------------------------------------------------------------- 预检


def test_auto_run_requires_awaiting_verification(client, workspace: Path, tmp_path: Path):
    c, Session = client
    _tid, loop_id, _sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    r = c.post(f"/api/loops/{loop_id}/auto-run")
    assert r.status_code == 409, r.text
    assert "E_AUTO_PHASE" in r.text


def test_auto_run_rejects_workspace_not_candidate(client, workspace: Path, tmp_path: Path,
                                                  monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    _write_kernel(workspace, "// 手改过，不是 agent 的候选\n")
    _FakeActions().install(monkeypatch, Session)
    r = c.post(f"/api/loops/{loop_id}/auto-run")
    assert r.status_code == 409, r.text
    assert "E_AUTO_NO_CANDIDATE" in r.text


def test_auto_run_budget_preflight(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    _FakeActions().install(monkeypatch, Session)
    with Session() as db:
        task = db.get(Task, tid)
        loop = db.get(Loop, loop_id)
        loop.objective_json = json.dumps({"type": "perf", "target_dimension": "all",
                                          "budget": {"runs_per_action": 1}})
        # 把 build 那一格用完
        for _ in range(1):
            db.add(ActionJob(task_id=task.id, loop_id=loop.id, action="build",
                             status="ok", run_ref="x"))
        db.commit()
    r = c.post(f"/api/loops/{loop_id}/auto-run")
    assert r.status_code == 409, r.text
    assert "E_AUTO_BUDGET" in r.text and "build" in r.text


# ---------------------------------------------------------------- 快乐路径 / 失败 / 取消


def test_auto_run_happy_path_marks_steps_and_phase(client, workspace: Path, tmp_path: Path,
                                                   monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    sha_b = _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions()
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch)

    r = c.post(f"/api/loops/{loop_id}/auto-run")
    assert r.status_code == 201, r.text
    started = r.json()["auto_run"]
    assert started["status"] == "running"
    assert [s["step"] for s in started["steps"]] == ["build", "smoke", "test", "benchmark", "report"]
    assert started["candidate_sha256"] == sha_b

    run = _poll(c, loop_id)
    assert run["status"] == "done", run
    assert run["current_step"] == "report"
    assert fake.started == ["build", "smoke", "test", "benchmark"]
    assert [s["status"] for s in run["steps"]] == ["ok"] * 5
    assert all(s["gate"] == "pass" for s in run["steps"][:4])
    assert run["steps"][0]["cleanup"]["ok"] is True
    with Session() as db:
        loop = db.get(Loop, loop_id)
        assert loop.phase == "awaiting_adoption"
        audits = db.scalars(select(LoopAutoRun)).all()
        assert audits and audits[-1].status == "done"
        attempt = db.scalar(select(LoopAttempt).where(LoopAttempt.loop_id == loop_id))
        assert attempt is None or run["attempt_id"] in (None, attempt.id)


def test_auto_run_failure_rolls_back_and_keeps_failed_candidate(client, workspace: Path,
                                                                tmp_path: Path, monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(fail_on="smoke")
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch)

    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    run = _poll(c, loop_id)
    assert run["status"] == "failed", run
    assert run["current_step"] == "smoke"
    smoke = next(s for s in run["steps"] if s["step"] == "smoke")
    assert smoke["status"] == "failed" and "smoke 未通过" in (smoke["error"] or "")
    assert fake.reverted == [loop_id]  # 回退到 agent 改前（optimize.revert 被调用）
    with Session() as db:
        loop = db.get(Loop, loop_id)
        assert loop.phase == "failed"
        failed = db.scalar(select(Savepoint).where(Savepoint.loop_id == loop_id,
                                                   Savepoint.kind == "failed_candidate"))
        assert failed is not None  # 现场留档，便于人工排查


def test_auto_run_cancel_blocks_chain(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(delay=5.0)
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch)

    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    time.sleep(0.2)
    r = c.post(f"/api/loops/{loop_id}/auto-run/cancel")
    assert r.status_code == 200, r.text
    assert r.json()["auto_run"]["status"] == "cancelled"
    with Session() as db:
        assert db.get(Loop, loop_id).phase == "blocked"
    # 再等一秒：驱动不能再把 cancelled 改回 failed/done
    time.sleep(1.0)
    run = c.get(f"/api/loops/{loop_id}/auto-run").json()["auto_run"]
    assert run["status"] == "cancelled"


def test_auto_run_report_failure_stays_awaiting_adoption(client, workspace: Path,
                                                         tmp_path: Path, monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions()
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch, written=False)

    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    run = _poll(c, loop_id)
    assert run["status"] == "failed", run
    assert "E_AUTO_REPORT" in (run["error"] or "")
    with Session() as db:
        loop = db.get(Loop, loop_id)
        assert loop.phase == "awaiting_adoption"


def test_auto_run_reconcile_marks_blocked(client, workspace: Path, tmp_path: Path,
                                          monkeypatch):
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(delay=5.0)
    fake.install(monkeypatch, Session)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    time.sleep(0.2)
    with Session() as db:
        n = loop_orchestrator.reconcile_startup(db)
        assert n == 1
        run = db.scalar(select(LoopAutoRun).where(LoopAutoRun.loop_id == loop_id))
        assert run.status == "blocked"
    time.sleep(0.5)
    body = c.get(f"/api/loops/{loop_id}/auto-run").json()["auto_run"]
    assert body["status"] == "blocked"


def test_auto_run_can_retry_from_blocked(client, workspace: Path, tmp_path: Path, monkeypatch):
    """取消后 phase=blocked（现场保留）→ 允许再发起一次自动链（候选仍在工作区）。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    slow = _FakeActions(delay=5.0)
    slow.install(monkeypatch, Session)
    _fake_report(monkeypatch)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    time.sleep(0.2)
    assert c.post(f"/api/loops/{loop_id}/auto-run/cancel").status_code == 200
    with Session() as db:
        assert db.get(Loop, loop_id).phase == "blocked"
        # 模拟 _finalize 收尾：被杀的动作作业标 interrupted（真实路径里 cancel->子进程退出即收尾）
        for job in db.scalars(select(ActionJob).where(ActionJob.status == "running")):
            job.status = "interrupted"
            job.ended_at = utcnow()
        db.commit()

    quick = _FakeActions()
    quick.install(monkeypatch, Session)
    _fake_report(monkeypatch)
    r = c.post(f"/api/loops/{loop_id}/auto-run")
    assert r.status_code == 201, r.text
    run = _poll(c, loop_id)
    assert run["status"] == "done", run
    with Session() as db:
        assert db.get(Loop, loop_id).phase == "awaiting_adoption"


def test_auto_run_missing_candidate_is_failed_not_done(client, workspace: Path, tmp_path: Path,
                                                       monkeypatch):
    """benchmark 成功但 candidate 存点没落 → 不能报 done（否则 G7 永久 409）。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(create_candidate=False)
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    run = _poll(c, loop_id)
    assert run["status"] == "failed", run
    assert "E_AUTO_CANDIDATE" in (run["error"] or "")
    with Session() as db:
        assert db.get(Loop, loop_id).phase == "failed"


def test_auto_run_resource_failure_blocks_without_rollback(client, workspace: Path,
                                                           tmp_path: Path, monkeypatch):
    """资源/前置类失败（plan 不齐/设备被占）→ blocked，不回退候选（可原地重试）。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    sha_b = _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions()
    fake.install(monkeypatch, Session)
    _fake_report(monkeypatch)
    orig_prepare = loop_orchestrator._prepare_step

    def blocked_smoke(task_id, loop_id_, step):
        if step == "smoke":
            return {"ok": False, "status": "blocked",
                    "error": "E_AUTO_PLAN: smoke 前置不齐：模拟设备被占"}
        return orig_prepare(task_id, loop_id_, step)

    monkeypatch.setattr(loop_orchestrator, "_prepare_step", blocked_smoke)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    run = _poll(c, loop_id)
    assert run["status"] == "blocked", run
    assert "E_AUTO_PLAN" in (run["error"] or "")
    assert fake.reverted == []           # 不回退工作区
    with Session() as db:
        loop = db.get(Loop, loop_id)
        assert loop.phase == "blocked"
    assert savepoints.current_sha(workspace) == sha_b  # 候选仍在，可直接重试


def test_auto_run_cancel_cleans_step_state(client, workspace: Path, tmp_path: Path,
                                           monkeypatch):
    """取消后 steps 不应残留 running、current_action_job_id 必须清空。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(delay=5.0)
    fake.install(monkeypatch, Session)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    time.sleep(0.3)
    assert c.post(f"/api/loops/{loop_id}/auto-run/cancel").status_code == 200
    run = c.get(f"/api/loops/{loop_id}/auto-run").json()["auto_run"]
    assert run["status"] == "cancelled"
    assert run["current_action_job_id"] is None
    assert not any(s["status"] == "running" for s in run["steps"])


def test_manual_write_entries_blocked_while_auto_run(client, workspace: Path, tmp_path: Path,
                                                     monkeypatch):
    """自动链运行中：手动动作 / 一键回退 / savepoint restore 都要被拦（工作区必须稳定）。"""
    c, Session = client
    tid, loop_id, sha_a = _seed_ready_round(c, Session, workspace, tmp_path)
    _make_candidate(Session, workspace, tid, loop_id, sha_a)
    fake = _FakeActions(delay=5.0)
    fake.install(monkeypatch, Session)
    assert c.post(f"/api/loops/{loop_id}/auto-run").status_code == 201
    time.sleep(0.4)

    r = c.post(f"/api/tasks/{tid}/actions", json={"action": "test", "dry_run": False,
                                                  "loop_id": loop_id})
    assert r.status_code == 409 and "自动验证链" in r.text
    r = c.post(f"/api/loops/{loop_id}/optimize/revert")
    assert r.status_code == 409 and "自动验证链" in r.text
    with Session() as db:
        task = db.get(Task, tid)
        sp = db.scalar(select(Savepoint).where(Savepoint.loop_id == loop_id,
                                               Savepoint.kind == "baseline"))
        assert sp is not None
        with pytest.raises(savepoints.SavepointBusyError):
            savepoints.restore(db, sp, task)
    assert c.post(f"/api/loops/{loop_id}/auto-run/cancel").status_code == 200
