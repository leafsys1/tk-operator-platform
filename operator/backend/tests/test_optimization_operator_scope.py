"""Regression for real operator-v1 runs joined to optimization savepoints.

Legacy full-workspace tests remain unchanged. These tests use the production
operator fingerprint while archives retain a separate complete-workspace hash.
"""
import json

import pytest
from sqlalchemy import select

from app.models import Loop, LoopRunLink, Savepoint, Task
from app.services import agent_runner, exec_actions, optimization, optimize, savepoints
from app.services.prose_relay import SPEC_SEP
from test_optimization_rounds import (_add_run, _corr, _perf_case, _run_dir,
                                     _seed_api, _write_kernel)
from test_optimize import _DoneState, _fake_start_recorder


def test_operator_baseline_candidate_accept_and_scoped_restore(client, workspace, tmp_path, monkeypatch):
    c, Session = client
    monkeypatch.setattr(savepoints.settings, "savepoint_root", str(tmp_path / "archives"))
    _, tid, _ = _seed_api(Session, workspace)
    kernel = _write_kernel(workspace, "// v1\n")
    sha_a = exec_actions.operator_snapshot(workspace, "elementwise_add").sha256
    with Session() as db:
        task = db.get(Task, tid)
        task.cases_confirmed_scope = "operator-v1"
        before = []
        for action in ("test", "benchmark"):
            manifest = _run_dir(tmp_path, action, snapshot=sha_a,
                                corr_cases=[_corr("case_001")] if action == "test" else None,
                                perf_cases=[_perf_case("bench_1m", 10)] if action == "benchmark" else None)
            run = _add_run(db, task, action=action, manifest=manifest, snapshot=sha_a)
            run.evidence_scope = "operator-v1"
            before.append(run.id)
        db.commit()
        result = optimization.create_round(db, task, objective={"type": "perf", "target_dimension": "all"})
        lid = result["loop"]["id"]
        loop = db.get(Loop, lid)
        baseline = db.scalar(select(Savepoint).where(Savepoint.loop_id == lid))
        assert baseline.code_snapshot_sha256 == sha_a
        assert baseline.archive_snapshot_sha256 == savepoints.current_sha(workspace) != sha_a
        assert baseline.evidence_scope == "operator-v1"
        assert sorted(db.scalars(select(LoopRunLink.run_id).where(LoopRunLink.loop_id == lid))) == before
        assert optimization.require_baseline(db, loop)["ready"] is True
        detail = c.get(f"/api/tasks/{tid}").json()
        assert detail["loop_run_links"] == [
            {"loop_id": lid, "run_id": rid, "role": "baseline"} for rid in before]
        assert all(run["loop_id"] is None for run in detail["runs"])

        # Unrelated operator and Markdown/build changes do not invalidate code evidence.
        other = workspace / "ops" / "other" / "kernel.cpp"
        other.parent.mkdir(parents=True)
        other.write_text("other task must survive rollback\n")
        (kernel.parent / "README.md").write_text("notes\n")
        assert optimization.require_baseline(db, loop)["ready"] is True
        assert savepoints.out(db, baseline, workspace=workspace)["in_sync"] is True
        kernel.write_text("// candidate v2\n")
        sha_b = exec_actions.operator_snapshot(workspace, task.op_name).sha256
        assert sha_a != sha_b
        assert not optimization.baseline_evidence(db, loop)["ready"]
        for action in ("test", "benchmark"):
            manifest = _run_dir(tmp_path, "candidate-" + action, snapshot=sha_b,
                                corr_cases=[_corr("case_001")] if action == "test" else None,
                                perf_cases=[_perf_case("bench_1m", 8)] if action == "benchmark" else None)
            run = _add_run(db, task, action=action, manifest=manifest, snapshot=sha_b, loop_id=lid)
            run.evidence_scope = "operator-v1"
        candidate = savepoints.create_for_sha(db, task, sha_b, label="candidate", kind="candidate", loop_id=lid)
        db.commit()
        assert candidate.evidence_scope == "operator-v1"
        accepted = optimization.accept(db, loop)
        assert accepted["workspace_matches"] is True
        assert accepted["version"]["code_snapshot_sha256"] == sha_b
        restored = savepoints.restore(db, baseline, task, scope="task")
        assert restored["current_sha256"] == sha_a and restored["in_sync"]
        assert other.read_text() == "other task must survive rollback\n"
        assert kernel.read_text() == "// v1\n"

        # Complete archive integrity is still checked, including unrelated contents.
        baseline.archive_snapshot_sha256 = "f" * 64
        with pytest.raises(savepoints.SavepointArchiveError):
            savepoints.diff(db, baseline, task)


@pytest.mark.parametrize("truncated", [False, True])
def test_optimize_callback_uses_baseline_scope(client, workspace, tmp_path, monkeypatch, truncated):
    _, Session = client
    monkeypatch.setattr(savepoints.settings, "savepoint_root", str(tmp_path / "archives"))
    _, tid, _ = _seed_api(Session, workspace)
    kernel = _write_kernel(workspace, "// v1\n")
    with Session() as db:
        task = db.get(Task, tid)
        loop = Loop(task_id=tid, kind="perf", status="open", phase="agent_running",
                    objective_json=json.dumps({"constraints": {"allowed_paths": ["ops/elementwise_add/op_host/"]}}))
        db.add(loop)
        db.flush()
        lid = loop.id
        baseline = savepoints.create(db, task, kind="baseline", loop_id=lid, evidence_op_name=task.op_name)
        sha_a = baseline.code_snapshot_sha256
        db.commit()
    captured = {}
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    optimize.start(9999, "test", loop_id=lid, workspace=workspace, dsh_home=tmp_path)
    kernel.write_text("// candidate v2\n")
    final = "候选\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "candidate", "changed": True, "verification": "pass", "pipeline": {"verifier": "PASS"}})
    captured["on_done"](_DoneState(final[:-3] if truncated else final))
    with Session() as db:
        loop = db.get(Loop, lid)
        ev = json.loads(loop.optimize_json)
        assert ev["evidence_scope"] == "operator-v1"
        assert ev["before_sha256"] == sha_a
        assert ev["after_sha256"] == exec_actions.operator_snapshot(workspace, "elementwise_add").sha256
        assert ev["after_sha256"] != ev["after_workspace_sha256"]
        if truncated:
            assert ev["status"] == "failed" and ev["guard_reverted"] and not ev["changed"]
            assert ev["reverted_clean"] and ev["after_sha256"] == sha_a
            assert "结构化输出解析失败" in ev["error"]
            assert kernel.read_text() == "// v1\n"
            assert ev["raw_output"] == final[:-3]
            message = optimize._conversation_result_text(ev)
            assert "本轮没有可验证的候选" in message and SPEC_SEP not in message
        else:
            assert ev["status"] == "done" and ev["changed"]
