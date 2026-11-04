"""Step 3 验收：append-only 报告归档 / manifest+sha 校验 / raw 副本 / 独立目录复现脚本。

契约见 `docs/plan-optimize-auto-v1.md` §17。全部临时库/临时工作区/临时报告根。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import settings
from app.models import Loop, Project, Run, Savepoint, SpecVersion, Task
from app.services import optimization, optimization_reports, savepoints

OP = "elementwise_add"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@deadbeef"
DIGEST = "sha256:cafe"
BACKEND = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND.parent / "scripts" / "repro_report.py"


# ---------------------------------------------------------------- 造数据


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_dir(root: Path, name: str, *, snapshot: str, action: str) -> Path:
    d = root / name
    (d / "out").mkdir(parents=True, exist_ok=True)
    common = {"spec_version_id": 1, "code_snapshot_sha256": snapshot,
              "criteria_ref": CRITERIA, "suite_ref": SUITE}
    arts: list[dict] = []
    if action == "benchmark":
        body = json.dumps({**common, "cases": [{
            "case_id": "bench_1m", "n": 1048576, "dtype": "float32",
            "candidate": {"device_us": {"median_us": 8.0, "std_us": 0.1}},
            "baseline": {"device_us": {"median_us": 10.0}},
        }], "summary": {"primary_case": "bench_1m"}}, ensure_ascii=False).encode()
        (d / "out" / "performance.json").write_bytes(body)
        arts.append({"type": "performance", "path": "out/performance.json", "sha256": _sha(body)})
    else:
        body = json.dumps({**common, "cases": [
            {"case_id": "case_001", "status": "PASS", "required": True},
        ], "oracle": {"golden_path": f"ops/{OP}/scripts/golden.py",
                      "golden_sha256": "a" * 64,
                      "criteria_path": "criteria/check.py",
                      "criteria_sha256": "b" * 64}},
            ensure_ascii=False).encode()
        (d / "out" / "correctness.json").write_bytes(body)
        arts.append({"type": "correctness", "path": "out/correctness.json", "sha256": _sha(body)})
    log_body = b"ok\n"
    (d / "out" / "run.log").write_bytes(log_body)
    arts.append({"type": "log", "path": "out/run.log", "sha256": _sha(log_body)})
    manifest = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0,
                "image_digest": DIGEST, **common, "device": "1", "artifacts": arts}
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _add_run(db, task: Task, *, action: str, manifest: Path, snapshot: str,
             loop_id: int) -> Run:
    run = Run(task_id=task.id, loop_id=loop_id, action=action, status="ok", exit_code=0,
              manifest_path=str(manifest), code_snapshot_sha256=snapshot,
              criteria_ref=CRITERIA, suite_ref=SUITE, image_digest=DIGEST,
              device_id="1", spec_version_id=1,
              command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action {action}")
    db.add(run)
    db.flush()
    return run


def _seed_pair(db, workspace: Path, tmp_path: Path) -> int:
    """建一个可出报告的完整轮次（baseline/candidate + 两侧 test/benchmark）。"""
    workspace_sha_a = savepoints.current_sha(workspace)
    if True:
        project = Project(name="p", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=OP, op_name=OP, target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        loop = Loop(task_id=task.id, kind="perf", scope="1M 带宽优化", hypothesis="搬运受限",
                    judge_condition="精度不退化 + 同条件 A/B",
                    objective_json=json.dumps({"type": "perf", "target_dimension": "all",
                                               "label": "1M 带宽优化"}, ensure_ascii=False))
        db.add(loop)
        db.flush()
        baseline = savepoints.create(db, task, label="基线", kind="baseline", loop_id=loop.id)
        assert baseline.code_snapshot_sha256 == workspace_sha_a
        mb = _run_dir(tmp_path, "before-bench", snapshot=workspace_sha_a, action="benchmark")
        mt = _run_dir(tmp_path, "before-test", snapshot=workspace_sha_a, action="test")
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=workspace_sha_a,
                 loop_id=loop.id)
        _add_run(db, task, action="test", manifest=mt, snapshot=workspace_sha_a,
                 loop_id=loop.id)
        (workspace / "ops" / OP / "op_host").mkdir(parents=True, exist_ok=True)
        (workspace / "ops" / OP / "op_host" / "add.asc").write_text("// v2\n", encoding="utf-8")
        sha_b = savepoints.current_sha(workspace)
        assert sha_b != workspace_sha_a
        mb2 = _run_dir(tmp_path, "after-bench", snapshot=sha_b, action="benchmark")
        mt2 = _run_dir(tmp_path, "after-test", snapshot=sha_b, action="test")
        _add_run(db, task, action="benchmark", manifest=mb2, snapshot=sha_b, loop_id=loop.id)
        _add_run(db, task, action="test", manifest=mt2, snapshot=sha_b, loop_id=loop.id)
        savepoints.create_for_sha(db, task, sha_b, label="候选", kind="candidate",
                                  loop_id=loop.id)
        db.commit()
        return loop.id


def _write_report(db, loop_id: int) -> dict:
    loop = db.get(Loop, loop_id)
    return optimization_reports.write(db, loop)


# ---------------------------------------------------------------- 归档布局 / 校验


def test_report_archive_layout_manifest_and_verify(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    stored = _write_report(db_session, loop_id)
    assert stored["stored"] is True and stored["seq"] == 1
    report_dir = Path(stored["report_dir"])
    assert report_dir.is_dir() and report_dir.name == f"0001-{stored['report_sha256'][:12]}"
    for name in ("report.json", "manifest.json", "repro.md"):
        assert (report_dir / name).is_file(), name
    assert (report_dir / "raw").is_dir()
    assert (report_dir / "report.json").read_bytes() == optimization_reports._canonical(
        json.loads((report_dir / "report.json").read_text(encoding="utf-8")))
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "opx-report-manifest/1"
    assert manifest["loop_id"] == loop_id and manifest["seq"] == 1
    assert manifest["report_sha256"] == stored["report_sha256"]
    assert {f["path"] for f in manifest["files"]} == {"report.json", "repro.md"}
    assert all(f["sha256"] for f in manifest["files"])
    copied = [r for r in manifest["raw"] if r["copied"]]
    assert copied and all((report_dir / r["copy"]).is_file() for r in copied)
    data = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert data["evidence"]["schema"] == "opx-report-evidence/1"
    assert data["evidence"]["oracle"]["before"]["golden_sha256"] == "a" * 64
    assert len(data["evidence"]["runs"]) == 4

    verified = optimization_reports.verify(loop_id, stored["report_sha256"])
    assert verified is not None and verified["valid"] is True
    assert verified["errors"] == []

    # 改一个字节 → 深度校验必须失败
    (report_dir / "report.json").write_text('{"tampered": true}\n', encoding="utf-8")
    bad = optimization_reports.verify(loop_id, stored["report_sha256"])
    assert bad is not None and bad["valid"] is False and bad["errors"]


def test_report_archive_is_append_only_and_idempotent(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    first = _write_report(db_session, loop_id)
    again = _write_report(db_session, loop_id)
    assert again["seq"] == first["seq"] and again["report_sha256"] == first["report_sha256"]
    assert again["stored"] is False
    assert len(optimization_reports.list_reports(loop_id)) == 1

    # 内容变化（改 scope/label）→ 新 seq，旧目录不动
    loop = db_session.get(Loop, loop_id)
    loop.scope = "第二轮：不同目标"
    obj = dict(loop.objective or {})
    obj["label"] = "第二轮：不同目标"
    loop.objective_json = json.dumps(obj, ensure_ascii=False)
    db_session.commit()
    second = _write_report(db_session, loop_id)
    assert second["seq"] == 2 and second["stored"] is True
    assert second["report_sha256"] != first["report_sha256"]
    rows = optimization_reports.list_reports(loop_id)
    assert [r["seq"] for r in rows] == [2, 1]
    assert Path(first["report_dir"]).is_dir() and Path(second["report_dir"]).is_dir()
    assert optimization_reports.latest(loop_id)["seq"] == 2


def test_accept_gate_requires_valid_archive(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    loop = db_session.get(Loop, loop_id)
    ev = optimization.accept_evidence(db_session, loop)
    assert ev["ready"] is True, ev
    assert ev["report"]["valid"] is True and ev["report"]["report_sha256"]
    latest = optimization_reports.latest(loop_id)
    # 篡改归档 → 采纳门禁必须 409（不能偷偷重生成掩盖）
    (Path(latest["dir"]) / "report.json").write_text('{"tampered": true}\n', encoding="utf-8")
    ev2 = optimization.accept_evidence(db_session, loop)
    assert ev2["ready"] is False
    assert any(i["code"] == "E_ACCEPT_REPORT_UNWRITTEN" for i in ev2["issues"])


# ---------------------------------------------------------------- API + 独立目录复现


def test_archive_api_list_get_verify(client, workspace: Path, tmp_path: Path):
    c, Session = client
    with Session() as db:
        loop_id = _seed_pair(db, workspace, tmp_path)
        stored = _write_report(db, loop_id)

    rows = c.get(f"/api/loops/{loop_id}/optimization-reports").json()["reports"]
    assert len(rows) == 1 and rows[0]["report_sha256"] == stored["report_sha256"]
    assert rows[0]["seq"] == 1 and rows[0]["valid"] is True

    got = c.get(f"/api/loops/{loop_id}/optimization-reports/{stored['report_sha256']}").json()
    assert got["manifest"]["report_sha256"] == stored["report_sha256"]
    assert got["report"]["schema"] == "opx-optimization-report/1" and got["valid"] is True

    verified = c.post(
        f"/api/loops/{loop_id}/optimization-reports/{stored['report_sha256']}/verify").json()
    assert verified["valid"] is True and verified["errors"] == []
    assert c.get(f"/api/loops/{loop_id}/optimization-reports/deadbeefdead").status_code == 404

    # POST 归档接口（同一内容 → 幂等复用）
    posted = c.post(f"/api/loops/{loop_id}/optimization-reports")
    assert posted.status_code == 201, posted.text
    assert posted.json()["seq"] == 1 and posted.json()["stored"] is False


def test_repro_script_verifies_independent_copy(client, workspace: Path, tmp_path: Path):
    c, Session = client
    with Session() as db:
        loop_id = _seed_pair(db, workspace, tmp_path)
        stored = _write_report(db, loop_id)
    src_dir = Path(stored["report_dir"])
    copy_dir = tmp_path / "repro-copy"
    shutil.copytree(src_dir, copy_dir)

    r = subprocess.run([sys.executable, str(SCRIPT), str(copy_dir), "--json"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr + r.stdout
    out = json.loads(r.stdout)
    assert out["valid"] is True and out["report_sha256"] == stored["report_sha256"]
    assert out["repro_commands"]

    # 在副本里改一个 raw 字节 → 校验失败（不依赖任何 DB/控制台）
    raw = next((copy_dir / item["copy"]) for item in
               json.loads((copy_dir / "manifest.json").read_text(encoding="utf-8"))["raw"]
               if item["copied"])
    raw.write_bytes(b"tampered\n")
    r2 = subprocess.run([sys.executable, str(SCRIPT), str(copy_dir), "--json"],
                        capture_output=True, text=True, timeout=60)
    assert r2.returncode == 1
    assert json.loads(r2.stdout)["valid"] is False


def test_archive_requires_after_benchmark(db_session, workspace: Path, tmp_path: Path):
    """没有「优化后 benchmark run」的轮次不能归档（报告没有 after 侧可复算证据）。"""
    from app.models import Project, SpecVersion, Task

    with db_session as db:
        project = Project(name="p2", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=OP, op_name=OP, target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        loop = Loop(task_id=task.id, kind="perf", scope="no-after",
                    objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
        db.add(loop)
        db.flush()
        savepoints.create(db, task, label="基线", kind="baseline", loop_id=loop.id)
        db.commit()
        with pytest.raises(optimization.OptimizationError) as ei:
            optimization_reports.write(db, loop)
        assert ei.value.status == 409
        assert "benchmark" in str(ei.value)


def test_verify_rejects_manifest_path_escape(db_session, workspace: Path, tmp_path: Path):
    """被篡改的 manifest 里 `../` 路径不能逃逸报告目录（只报错，不读目录外文件）。"""
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    stored = _write_report(db_session, loop_id)
    report_dir = Path(stored["report_dir"])
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied = next(r for r in manifest["raw"] if r["copied"])
    copied["copy"] = "../../../etc/passwd"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    result = optimization_reports.verify(loop_id, stored["report_sha256"])
    assert result is not None and result["valid"] is False
    assert any("raw" in err for err in result["errors"])


# ---------------------------------------------------------------- 审查修复回归


def test_accept_rearchives_when_candidate_changes(db_session, workspace: Path, tmp_path: Path):
    """归档对应旧 candidate 时，采纳必须重新归档（不能拿旧的有效报告当本次证据）。"""
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    first = _write_report(db_session, loop_id)
    assert first["report_valid"] is True

    loop = db_session.get(Loop, loop_id)
    task = db_session.get(Task, loop.task_id)
    # 再改一版代码 → 新 candidate + 新的 after 侧 run
    (workspace / "ops" / OP / "op_host" / "add.asc").write_text("// v3\n", encoding="utf-8")
    sha_c = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "after2-bench", snapshot=sha_c, action="benchmark")
    mt = _run_dir(tmp_path, "after2-test", snapshot=sha_c, action="test")
    _add_run(db_session, task, action="benchmark", manifest=mb, snapshot=sha_c, loop_id=loop_id)
    _add_run(db_session, task, action="test", manifest=mt, snapshot=sha_c, loop_id=loop_id)
    savepoints.create_for_sha(db_session, task, sha_c, label="候选2", kind="candidate",
                              loop_id=loop_id)
    db_session.commit()

    ev = optimization.accept_evidence(db_session, loop)
    assert ev["ready"] is True, ev
    latest = optimization_reports.latest(loop_id)
    assert latest["seq"] == 2  # append 了新归档
    archived = optimization_reports.load(loop_id, latest["report_sha256"])["report"]
    assert archived["code"]["candidate_sha256"] == sha_c
    assert archived["runs"]["after"]["benchmark"] == mb_run_id(db_session, mb)


def mb_run_id(db, manifest: Path) -> int:
    run = db.scalar(select(Run).where(Run.manifest_path == str(manifest)))
    return run.id


def test_tmp_dirs_are_not_treated_as_published(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    stored = _write_report(db_session, loop_id)
    base = optimization_reports.loop_root(loop_id)
    tmp = base / ".tmp-999-1"
    shutil.copytree(stored["report_dir"], tmp)
    manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
    manifest["seq"] = 99
    (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert [r["seq"] for r in optimization_reports.list_reports(loop_id)] == [1]
    assert optimization_reports.latest(loop_id)["seq"] == 1
    assert optimization_reports._find(loop_id, "deadbeefdead") is None


def test_broken_manifest_is_visible_and_blocks_accept(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    stored = _write_report(db_session, loop_id)
    (Path(stored["report_dir"]) / "manifest.json").write_text("{not json", encoding="utf-8")
    latest = optimization_reports.latest(loop_id)
    assert latest is not None and latest["broken"] is True and latest["valid"] is False
    loop = db_session.get(Loop, loop_id)
    ev = optimization.accept_evidence(db_session, loop)
    assert ev["ready"] is False
    assert any(i["code"] == "E_ACCEPT_REPORT_UNWRITTEN" for i in ev["issues"])


def test_raw_artifact_path_escape_is_rejected(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    # 往 after-benchmark 的 manifest 里塞一个 .. 产物路径（远端拉回的 manifest 不可信）
    with db_session as db:
        loop = db.get(Loop, loop_id)
        after_bench = optimization.report(db, loop)["runs"]["after"]["benchmark"]
        run = db.get(Run, after_bench)
        run_dir = Path(str(run.manifest_path)).parent
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"] = list(manifest.get("artifacts") or []) + [
            {"type": "evil", "path": "../../evil.txt"}]
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        db.commit()
    stored = _write_report(db_session, loop_id)
    report_dir = Path(stored["report_dir"])
    evil_entries = [r for r in stored["manifest"]["raw"]
                    if str(r.get("path") or "").endswith("evil.txt")
                    or "evil.txt" in str(r.get("source") or "")]
    assert evil_entries and all(r["copied"] is False for r in evil_entries)
    assert not (report_dir.parent / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()
    assert optimization_reports.verify(loop_id, stored["report_sha256"])["valid"] is True


def test_raw_change_forces_new_archive(db_session, workspace: Path, tmp_path: Path):
    loop_id = _seed_pair(db_session, workspace, tmp_path)
    first = _write_report(db_session, loop_id)
    manifest = first["manifest"]
    copied = next(r for r in manifest["raw"] if r["copied"] and r["path"].endswith("run.log"))
    Path(str(copied["source"])).write_bytes(b"changed log\n")

    second = _write_report(db_session, loop_id)
    assert second["seq"] == 2 and second["stored"] is True
    assert second["report_sha256"] != first["report_sha256"]


def test_download_endpoint_returns_original_bytes(client, workspace: Path, tmp_path: Path):
    c, Session = client
    with Session() as db:
        loop_id = _seed_pair(db, workspace, tmp_path)
        stored = _write_report(db, loop_id)
    r = c.get(f"/api/loops/{loop_id}/optimization-reports/{stored['report_sha256']}/download")
    assert r.status_code == 200
    body = r.content
    assert hashlib.sha256(body).hexdigest() == stored["report_sha256"]
    assert body == (Path(stored["report_dir"]) / "report.json").read_bytes()
