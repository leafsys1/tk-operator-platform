"""优化轮次 v1（P1）验收：目标解析 / 建轮次 / create_for_sha / 自动候选 / 采纳否决 /
可复现报告 / savepoints loop_id 过滤 / 迁移往返。

契约见 `docs/plan-optimize-rounds-v1.md` §10。全部用临时库 / 临时归档根 / 临时 run 目录，
不跑 dsh、不碰真机、不写真实仓库 `archives/`。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import (ActionJob, Loop, LoopRunLink, Project, Run, Savepoint, SpecVersion,
                        Task)
from app.services import action_runner, optimize, optimization, savepoints

BACKEND = Path(__file__).resolve().parents[1]
OP = "elementwise_add"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@deadbeef"
DIGEST = "sha256:cafe"


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path: Path, monkeypatch):
    """每个用例独立归档根；清掉 agent 自动优化的模块级运行态（假 start 不会自己收尾）。"""
    monkeypatch.setattr(settings, "savepoint_root", str(tmp_path / "archives"))
    optimize.PENDING.clear()
    yield
    optimize.PENDING.clear()


# ---------------------------------------------------------------- 造数据小工具


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_dir(root: Path, name: str, *, snapshot: str, perf_cases: list[dict] | None = None,
             corr_cases: list[dict] | None = None, device: str = "1") -> Path:
    """造一个 run 产物目录（manifest + performance.json / correctness.json）。"""
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


def _perf_case(case_id: str, med_us: float, *, std: float = 0.02, n: int = 1048576) -> dict:
    return {"case_id": case_id, "n": n,
            "candidate": {"device_us": {"median_us": med_us, "std_us": std}},
            "baseline": {"device_us": {"median_us": med_us + 0.3}}}


def _corr(case_id: str, status: str = "PASS") -> dict:
    return {"case_id": case_id, "status": status, "required": True}


def _add_run(db, task: Task, *, action: str, manifest: Path, snapshot: str,
             loop_id: int | None = None, status: str = "ok", device: str = "1",
             command: str | None = None) -> Run:
    run = Run(task_id=task.id, loop_id=loop_id, action=action, status=status, exit_code=0,
              manifest_path=str(manifest), code_snapshot_sha256=snapshot,
              criteria_ref=CRITERIA, suite_ref=SUITE, image_digest=DIGEST,
              device_id=device, spec_version_id=1, command=command)
    db.add(run)
    db.flush()
    return run


def _seed_api(Session, workspace: Path, *, name: str = "p") -> tuple[int, int, int]:
    with Session() as db:
        project = Project(name=name, workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name=OP,
                    target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return project.id, task.id, spec.id


def _create_round(c, task_id: int, objective: dict | None = None, **payload) -> dict:
    body = {"objective": objective or {"type": "perf", "target_dimension": "all"}}
    body.update(payload)
    r = c.post(f"/api/tasks/{task_id}/optimization-rounds", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _write_kernel(workspace: Path, text: str) -> Path:
    path = workspace / "ops" / OP / "op_host" / "add.asc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_candidate(Session, workspace: Path, task_id: int, loop_id: int, *,
                    text: str) -> int:
    """改一下工作区再建一个 candidate 存点（模拟 agent 改完 + benchmark 成功）。"""
    _write_kernel(workspace, text)
    with Session() as db:
        task = db.get(Task, task_id)
        sp = savepoints.create(db, task, label="候选 · 测试", kind="candidate", loop_id=loop_id)
        db.commit()
        return sp.id


def _seed_round_pair(db, seeded, workspace: Path, tmp_path: Path, *, objective: dict,
                     before_med: float = 10.0, after_med: float = 8.0,
                     perf_before: list[dict] | None = None,
                     perf_after: list[dict] | None = None,
                     corr_before: list[dict] | None = None,
                     corr_after: list[dict] | None = None,
                     tag: str = "r") -> tuple[Loop, dict]:
    """造一个完整的优化轮次：基线存点 + before/after 两侧 test+benchmark + candidate。"""
    task = seeded["task"]
    corr_before = corr_before if corr_before is not None else [_corr("case_001")]
    corr_after = corr_after if corr_after is not None else [_corr("case_001")]
    sha_a = savepoints.current_sha(workspace)
    loop = Loop(task_id=task.id, kind="perf", scope="增大分块", hypothesis="MTE2 搬运受限",
                judge_condition="精度不退化 + 同条件 A/B",
                objective_json=json.dumps(objective, ensure_ascii=False))
    db.add(loop)
    db.flush()
    baseline = savepoints.create(db, task, label="基线 · 分块", kind="baseline", loop_id=loop.id)
    assert baseline.code_snapshot_sha256 == sha_a

    mb_test = _run_dir(tmp_path, f"{tag}-bt", snapshot=sha_a, corr_cases=corr_before)
    mb_bench = _run_dir(tmp_path, f"{tag}-bb", snapshot=sha_a,
                        perf_cases=(perf_before if perf_before is not None
                                    else [_perf_case("bench_1m", before_med)]))
    b_test = _add_run(db, task, action="test", manifest=mb_test, snapshot=sha_a, loop_id=loop.id,
                      command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action test")
    b_bench = _add_run(db, task, action="benchmark", manifest=mb_bench, snapshot=sha_a,
                       loop_id=loop.id,
                       command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action benchmark")

    _write_kernel(workspace, f"// kernel v2 {tag}\n")
    sha_b = savepoints.current_sha(workspace)
    ma_test = _run_dir(tmp_path, f"{tag}-at", snapshot=sha_b, corr_cases=corr_after)
    ma_bench = _run_dir(tmp_path, f"{tag}-ab", snapshot=sha_b,
                        perf_cases=(perf_after if perf_after is not None
                                    else [_perf_case("bench_1m", after_med)]))
    a_test = _add_run(db, task, action="test", manifest=ma_test, snapshot=sha_b, loop_id=loop.id,
                      command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action test")
    a_bench = _add_run(db, task, action="benchmark", manifest=ma_bench, snapshot=sha_b,
                       loop_id=loop.id,
                       command=f"OPX_TASK_ID={task.id} bash scripts/nm_test.sh --action benchmark")
    candidate = savepoints.create_for_sha(db, task, sha_b, label="候选 · 增大分块",
                                          kind="candidate", loop_id=loop.id)
    db.commit()
    return loop, {
        "baseline": baseline, "candidate": candidate,
        "before": {"test": b_test, "benchmark": b_bench},
        "after": {"test": a_test, "benchmark": a_bench},
        "sha_a": sha_a, "sha_b": sha_b,
    }


# ---------------------------------------------------------------- §10 targets


def test_targets_without_reports_is_empty_with_hint(db_session, seeded):
    data = optimization.targets(db_session, seeded["task"])
    assert data["task_id"] == seeded["task"].id
    assert data["benchmark_run_id"] is None and data["test_run_id"] is None
    assert data["primary_case"] is None and data["suite_ref"] is None
    assert data["cases"] == [] and data["test_cases"] == [] and data["failing_cases"] == []
    assert "缺少基准报告" in data["hint"]


def test_targets_parses_reports_and_defaults_missing_fields(db_session, seeded, tmp_path: Path):
    task = seeded["task"]
    mb = _run_dir(tmp_path, "b", snapshot="a" * 64, perf_cases=[
        {"case_id": "bench_1m", "n": 1048576, "dtype": "float32", "primary": True},
        {"case_id": "bench_64k", "n": 65536},  # 缺 dtype / primary / required / note
    ])
    mt = _run_dir(tmp_path, "t", snapshot="a" * 64, corr_cases=[
        {"case_id": "case_001", "status": "FAIL", "required": True},
        {"case_id": "case_002", "status": "PASS"},
    ])
    bench = _add_run(db_session, task, action="benchmark", manifest=mb, snapshot="a" * 64)
    test = _add_run(db_session, task, action="test", manifest=mt, snapshot="a" * 64)
    # 更新的 failed benchmark 不应被选中（只取最近一次 ok）
    bench_failed = _add_run(db_session, task, action="benchmark", manifest=mb, snapshot="b" * 64,
                            status="failed")
    db_session.commit()

    data = optimization.targets(db_session, task)
    assert data["benchmark_run_id"] == bench.id != bench_failed.id
    assert data["test_run_id"] == test.id
    assert data["suite_ref"] == SUITE
    assert data["primary_case"] == "bench_1m"
    assert data["hint"] == ""
    assert [c["case_id"] for c in data["cases"]] == ["bench_1m", "bench_64k"]
    assert data["cases"][0]["primary"] is True and data["cases"][0]["required"] is False
    assert data["cases"][1] == {"case_id": "bench_64k", "n": 65536, "dtype": None,
                                "primary": False, "required": False, "note": None}
    assert data["failing_cases"] == [{"case_id": "case_001", "status": "FAIL"}]
    assert data["test_cases"] == [{"case_id": "case_001", "status": "FAIL"},
                                  {"case_id": "case_002", "status": "PASS"}]


# ---------------------------------------------------------------- §10 建轮次


def test_create_round_validates_objective(client, workspace: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    assert c.get("/api/tasks/999999/optimization-targets").status_code == 404
    assert c.post("/api/tasks/999999/optimization-rounds",
                  json={"objective": {"type": "perf"}}).status_code == 404

    cases = [
        ({"type": "bogus", "target_dimension": "all"}, "objective.type"),
        ({"type": "perf", "target_dimension": "bogus"}, "target_dimension"),
        ({"type": "perf", "target_dimension": "cases"}, "target_cases"),
        ({"type": "perf", "target_dimension": "cases", "target_cases": ["bench_1m"]}, "不在"),
        ({"type": "precision", "target_dimension": "cases", "target_cases": ["case_001"]}, "不在"),
    ]
    for objective, needle in cases:
        r = c.post(f"/api/tasks/{tid}/optimization-rounds", json={"objective": objective})
        assert r.status_code == 400, (objective, r.text)
        assert needle in r.text


def test_create_round_builds_loop_baseline_and_attaches_runs(client, workspace: Path,
                                                             tmp_path: Path):
    c, Session = client
    _pid, tid, spec_id = _seed_api(Session, workspace)
    _write_kernel(workspace, "// kernel v1\n")
    sha = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "b", snapshot=sha, perf_cases=[_perf_case("bench_1m", 10.0)])
    mt = _run_dir(tmp_path, "t", snapshot=sha, corr_cases=[_corr("case_001", "FAIL")])
    with Session() as db:
        task = db.get(Task, tid)
        bench = _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha)
        test = _add_run(db, task, action="test", manifest=mt, snapshot=sha)
        # 已挂到别的轮次的 run 不允许被复用
        other = Loop(task_id=tid, kind="perf", scope="别的轮次")
        db.add(other)
        db.flush()
        attached_elsewhere = _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha,
                                      loop_id=other.id)
        bench_id, test_id = bench.id, test.id
        elsewhere_id = attached_elsewhere.id
        db.commit()

    targets = c.get(f"/api/tasks/{tid}/optimization-targets").json()
    assert targets["primary_case"] == "bench_1m"

    body = _create_round(c, tid, {"type": "perf", "target_dimension": "cases",
                                  "target_cases": ["bench_1m"], "label": "1M 带宽优化"},
                         label="1M 带宽优化")
    assert set(body) == {"loop", "baseline_savepoint", "attached_runs", "targets"}
    loop = body["loop"]
    assert loop["kind"] == "perf" and loop["status"] == "open"
    assert loop["scope"] == "1M 带宽优化"
    assert loop["hypothesis"] == optimization.DEFAULT_HYPOTHESIS
    assert "同设备/镜像/规格/用例集同条件 A/B" in loop["judge_condition"]
    obj = loop["objective"]
    assert obj["schema"] == optimization.OBJECTIVE_SCHEMA
    assert obj["type"] == "perf" and obj["target_cases"] == ["bench_1m"]
    assert obj["constraints"]["precision_non_regression"] is True
    # 默认只放开算子实现目录（kernel/host）；参考实现/用例/判据不在默认范围。
    assert obj["constraints"]["allowed_paths"] == [
        f"ops/{OP}/op_host/", f"ops/{OP}/op_kernel/"]
    # 目标快照取「最近一次 ok benchmark」（这里就是已挂到别的轮次那条；挂载逻辑不受影响）
    assert obj["targets_snapshot"]["benchmark_run_id"] == targets["benchmark_run_id"]
    assert obj["targets_snapshot"]["primary_case"] == "bench_1m"

    base = body["baseline_savepoint"]
    assert base["kind"] == "baseline" and base["loop_id"] == loop["id"]
    assert base["code_snapshot_sha256"] == sha
    assert base["label"].startswith("基线 · ")
    assert base["spec_version_id"] == spec_id

    # Step 1b：复用 run 不再改写 runs.loop_id，改写成显式 loop_run_links（role=baseline）
    assert body["attached_runs"] == [{"id": test_id, "action": "test", "role": "baseline"},
                                     {"id": bench_id, "action": "benchmark", "role": "baseline"}]
    assert body["targets"]["primary_case"] == "bench_1m"

    with Session() as db:
        runs = {r.id: r for r in db.scalars(select(Run).where(Run.task_id == tid))}
        assert runs[test_id].loop_id is None
        assert runs[bench_id].loop_id is None
        links = {(l.run_id, l.role) for l in db.scalars(
            select(LoopRunLink).where(LoopRunLink.loop_id == loop["id"]))}
        assert links == {(test_id, "baseline"), (bench_id, "baseline")}
        assert runs[elsewhere_id].loop_id != loop["id"]
    # 任务详情的 loops 也要带 objective（LoopOut.objective）
    detail = c.get(f"/api/tasks/{tid}").json()
    detail_loop = next(item for item in detail["loops"] if item["id"] == loop["id"])
    assert detail_loop["objective"]["type"] == "perf"


def test_second_round_reuses_linked_baseline_runs(client, workspace: Path, tmp_path: Path):
    """同一份同 sha 基线 run 允许被多个轮次复用（文档 TC-010 推荐路径）。

    回归：旧逻辑用 `Run.id.not_in(select(LoopRunLink.run_id))` 排除已 link 的 run，
    第二个轮次会变成 baseline_pending / E_BASELINE_RUN_MISSING。
    """
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace, name="baseline-reuse")
    _write_kernel(workspace, "// kernel v1\n")
    sha = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "reuse-b", snapshot=sha, perf_cases=[_perf_case("bench_1m", 10.0)])
    mt = _run_dir(tmp_path, "reuse-t", snapshot=sha, corr_cases=[_corr("case_001", "PASS")])
    with Session() as db:
        task = db.get(Task, tid)
        bench = _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha)
        test = _add_run(db, task, action="test", manifest=mt, snapshot=sha)
        bench_id, test_id = bench.id, test.id
        db.commit()

    first = _create_round(c, tid, {"type": "perf", "target_dimension": "all"})
    assert {x["id"] for x in first["attached_runs"]} == {bench_id, test_id}
    second = _create_round(c, tid, {"type": "perf", "target_dimension": "all"})
    assert {x["id"] for x in second["attached_runs"]} == {bench_id, test_id}
    assert second["loop"]["phase"] == "baseline_ready"
    ev = c.get(f"/api/loops/{second['loop']['id']}/baseline-evidence").json()
    assert ev["ready"] is True, ev.get("issues")


def test_round_allowed_paths_cannot_escape_operator_root(client, workspace: Path):
    """API 直调也不能绕过“Agent 只改 ops/<op>/”的硬边界。"""
    c, Session = client
    _, task_id, _ = _seed_api(Session, workspace, name="allowed-paths")
    _write_kernel(workspace, "baseline\n")

    outside = c.post(f"/api/tasks/{task_id}/optimization-rounds", json={
        "objective": {
            "type": "perf", "target_dimension": "all",
            "constraints": {"allowed_paths": ["cases/elementwise_add/"]},
        },
    })
    assert outside.status_code == 400
    assert f"不能越过 ops/{OP}/" in outside.text

    traversal = c.post(f"/api/tasks/{task_id}/optimization-rounds", json={
        "objective": {
            "type": "perf", "target_dimension": "all",
            "constraints": {"allowed_paths": [f"ops/{OP}/../other/"]},
        },
    })
    assert traversal.status_code == 400

    # 页面把输入清空时会发 []；后端安全回退到算子实现目录，而不是关闭门禁。
    created = _create_round(c, task_id, {
        "type": "perf", "target_dimension": "all",
        "constraints": {"allowed_paths": []},
    })
    assert created["loop"]["objective"]["constraints"]["allowed_paths"] == [
        f"ops/{OP}/op_host/", f"ops/{OP}/op_kernel/"]


def test_create_text_target_round_and_default_budget(client, workspace: Path):
    """思路池 -> 轮次：目标用文字描述的「优化点」，不指定目标用例；默认每动作 3 次。"""
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace, name="p-text-round")
    _write_kernel(workspace, "baseline\n")
    body = _create_round(c, tid, {
        "type": "perf", "target_dimension": "text",
        "focus": "优化 64K 小 shape 的搬运与并行度",
    }, label="64K 搬运优化")
    obj = body["loop"]["objective"]
    assert obj["target_dimension"] == "text"
    assert obj["focus"] == "优化 64K 小 shape 的搬运与并行度"
    assert obj["target_cases"] == []
    assert "优化点（文字描述）" in body["loop"]["judge_condition"]
    assert "64K" in body["loop"]["judge_condition"]
    # 未显式给预算 → 默认每动作 3 次（基线+候选 2 次，留 1 次失败重试）
    assert obj["budget"]["runs_per_action"] == 3

    report = c.get(f"/api/loops/{body['loop']['id']}/optimization-report").json()
    assert report["targets"]["focus"] == "优化 64K 小 shape 的搬运与并行度"
    assert report["targets"]["target_dimension"] == "text"
    assert report["objective_summary"]["focus"] == "优化 64K 小 shape 的搬运与并行度"
    # 文字目标不产生 E_TARGET_CASE_MISSING（不拿 benchmark case_id 去配对）
    assert "E_TARGET_CASE_MISSING" not in {f["code"] for f in report["perf_diff"]["findings"]}


def test_create_text_round_requires_focus(client, workspace: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace, name="p-text-focus")
    empty = c.post(f"/api/tasks/{tid}/optimization-rounds", json={
        "objective": {"type": "perf", "target_dimension": "text", "focus": "   "}})
    assert empty.status_code == 400 and "focus" in empty.text
    too_long = c.post(f"/api/tasks/{tid}/optimization-rounds", json={
        "objective": {"type": "perf", "target_dimension": "text", "focus": "x" * 2001}})
    assert too_long.status_code == 400 and "focus" in too_long.text


def test_reject_open_round_with_revert_restores_baseline(client, workspace: Path):
    """否决不要求有候选；revert=true 时用 baseline 存点把工作区回退到本轮改前。"""
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace, name="p-reject-revert")
    kernel = _write_kernel(workspace, "baseline\n")
    body = _create_round(c, tid, {"type": "perf", "target_dimension": "all"})
    lid = body["loop"]["id"]
    # 模拟 agent 改了工作区但还没跑候选验证（没有候选存档 / 没有 optimize 证据）
    kernel.write_text("candidate changed\n", encoding="utf-8")
    r = c.post(f"/api/loops/{lid}/reject", json={"reason": "无收益", "revert": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["loop"]["status"] == "closed"
    assert data["reverted"]["ok"] is True
    assert data["reverted"]["mode"] == "baseline_savepoint"
    assert kernel.read_text(encoding="utf-8") == "baseline\n"
    assert c.post(f"/api/loops/{lid}/reject", json={"reason": "x"}).status_code == 409


def test_create_round_auto_attach_can_be_disabled(client, workspace: Path, tmp_path: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    sha = savepoints.current_sha(workspace)
    mb = _run_dir(tmp_path, "b", snapshot=sha, perf_cases=[_perf_case("bench_1m", 10.0)])
    with Session() as db:
        task = db.get(Task, tid)
        run = _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha)
        run_id = run.id
        db.commit()
    body = _create_round(c, tid, {"type": "perf", "target_dimension": "all"},
                         attach_baseline_runs=False)
    assert body["attached_runs"] == []
    with Session() as db:
        assert db.get(Run, run_id).loop_id is None


def test_create_round_workspace_unavailable_409(client, tmp_path: Path):
    c, Session = client
    with Session() as db:
        project = Project(name="p", workspace_path=str(tmp_path / "missing-ws"))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="t", op_name=OP)
        db.add(task)
        db.commit()
        tid = task.id
    r = c.post(f"/api/tasks/{tid}/optimization-rounds",
               json={"objective": {"type": "perf", "target_dimension": "all"}})
    assert r.status_code == 409 and "工作区" in r.text


# ---------------------------------------------------------------- §10 create_for_sha


def test_create_for_sha_reuses_archive_packs_or_rejects(db_session, seeded, workspace: Path):
    task = seeded["task"]
    _write_kernel(workspace, "// kernel v1\n")
    sha = savepoints.current_sha(workspace)
    archive, _size, _sha = savepoints.ensure_archive(workspace)

    # 1) 归档已存在 → 直接复用（不重新打包）
    sp = savepoints.create_for_sha(db_session, task, sha, label="候选", kind="candidate")
    sp_id = sp.id
    assert sp.code_snapshot_sha256 == sha and Path(sp.archive_path) == archive
    db_session.rollback()  # 只 flush 不 commit：调用方回滚后行应消失
    db_session.expire_all()
    assert db_session.get(Savepoint, sp_id) is None

    # 2) 归档不存在、但当前工作区就是该 sha → 现打包
    for stale in (savepoints.archive_root() / "sources").glob("*.tar.gz"):
        stale.unlink()
    sp2 = savepoints.create_for_sha(db_session, task, sha, label="候选", kind="candidate")
    db_session.commit()
    assert Path(sp2.archive_path).is_file()
    assert Path(sp2.archive_path).name == f"{sha}.tar.gz"

    # 3) 归档不存在且工作区已不是该 sha → 明确报错（不能拿当前代码冒充历史快照）
    _write_kernel(workspace, "// kernel v3\n")
    for stale in (savepoints.archive_root() / "sources").glob("*.tar.gz"):
        stale.unlink()
    with pytest.raises(savepoints.SavepointError, match="不是该快照"):
        savepoints.create_for_sha(db_session, task, sha, label="候选", kind="candidate")
    with pytest.raises(savepoints.SavepointError, match="缺少代码快照"):
        savepoints.create_for_sha(db_session, task, "", label="候选", kind="candidate")


# ---------------------------------------------------------------- §10 自动 candidate


class _DoneState:
    """`_finalize` 只需要 `text()` / `cancelled`。"""

    cancelled = False

    def __init__(self, text: str = "") -> None:
        self._text = text

    def text(self) -> str:
        return self._text


def test_finalize_auto_candidate_skips_baseline_generates_and_dedupes(
        db_session, seeded, workspace: Path, monkeypatch):
    task = seeded["task"]
    sha_base = savepoints.current_sha(workspace)
    loop = Loop(task_id=task.id, kind="perf", scope="增大分块", hypothesis="h",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.flush()
    baseline = savepoints.create(db_session, task, label="基线 · 分块", kind="baseline",
                                 loop_id=loop.id)
    db_session.commit()
    assert baseline.code_snapshot_sha256 == sha_base

    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(action_runner, "SessionLocal", Session)

    def _job_with_run(action: str, run_ref: str, sha: str) -> int:
        run = Run(task_id=task.id, loop_id=loop.id, action=action, status="ok", exit_code=0,
                  remote_workdir=run_ref, code_snapshot_sha256=sha)
        job = ActionJob(task_id=task.id, loop_id=loop.id, action=action, status="running",
                        run_ref=run_ref)
        db_session.add_all([run, job])
        db_session.commit()
        return job.id

    def _candidates() -> list[Savepoint]:
        return list(db_session.scalars(
            select(Savepoint).where(Savepoint.loop_id == loop.id,
                                    Savepoint.kind == "candidate").order_by(Savepoint.id)))

    # 基线轮：run sha == 基线存档 sha → 跳过，不生成候选
    base_job = _job_with_run("benchmark", "rBASE", sha_base)
    action_runner._finalize(base_job, 0, None, _DoneState())
    db_session.expire_all()
    assert _candidates() == []

    # 候选轮：新 sha（归档尚不存在，但当前工作区就是它）→ 自动生成 candidate。
    # 候选必须有同 sha 的 test 成功 run；benchmark 单独成功不生成候选。
    _write_kernel(workspace, "// kernel v2\n")
    sha_c = savepoints.current_sha(workspace)
    _job_with_run("test", "rCANDT", sha_c)
    cand_job = _job_with_run("benchmark", "rCAND", sha_c)
    action_runner._finalize(cand_job, 0, None, _DoneState())
    db_session.expire_all()
    cands = _candidates()
    assert len(cands) == 1
    assert cands[0].code_snapshot_sha256 == sha_c
    assert cands[0].label.startswith("候选 · ")
    assert Path(cands[0].archive_path).is_file()

    # 同 sha 重复收尾 → 去重（仍 1 条）
    dup_job = _job_with_run("benchmark", "rCAND2", sha_c)
    action_runner._finalize(dup_job, 0, None, _DoneState())
    db_session.expire_all()
    assert len(_candidates()) == 1

    # best-effort：归档不存在且工作区不匹配 → 只 log，不收尾抛错、不加候选行
    bad_job = _job_with_run("benchmark", "rBAD", "d" * 64)
    action_runner._finalize(bad_job, 0, None, _DoneState())
    db_session.expire_all()
    assert len(_candidates()) == 1


def test_finalize_baseline_side_by_first_run_when_savepoint_sha_drifted(
        db_session, seeded, workspace: Path, monkeypatch):
    """基线识别以「本轮最早 ok run」为准：基线存档 sha 与 run sha 不相等（快照口径漂移）
    时，基线 benchmark 不能被误判成候选；候选轮仍应正常生成。"""
    task = seeded["task"]
    older = "a" * 64
    loop = Loop(task_id=task.id, kind="perf", scope="漂移用例", hypothesis="h",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.flush()
    savepoints.create(db_session, task, label="基线", kind="baseline", loop_id=loop.id)
    db_session.commit()

    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(action_runner, "SessionLocal", Session)

    def _add(action: str, run_ref: str, sha: str) -> int:
        db_session.add(Run(task_id=task.id, loop_id=loop.id, action=action, status="ok",
                           exit_code=0, remote_workdir=run_ref, code_snapshot_sha256=sha))
        job = ActionJob(task_id=task.id, loop_id=loop.id, action=action, status="running",
                        run_ref=run_ref)
        db_session.add(job)
        db_session.commit()
        return job.id

    def _candidates() -> list[Savepoint]:
        return list(db_session.scalars(
            select(Savepoint).where(Savepoint.loop_id == loop.id,
                                    Savepoint.kind == "candidate").order_by(Savepoint.id)))

    # 基线 test + benchmark 都是 older；benchmark 收尾时 first run sha == older → 跳过
    _add("test", "rT-base", older)
    action_runner._finalize(_add("benchmark", "rB-base", older), 0, None, _DoneState())
    db_session.expire_all()
    assert _candidates() == []

    # 候选轮：新代码 → 新 sha 的 test + benchmark → 正常生成 candidate
    _write_kernel(workspace, "// kernel v2\n")
    sha_c = savepoints.current_sha(workspace)
    _add("test", "rT-cand", sha_c)
    action_runner._finalize(_add("benchmark", "rB-cand", sha_c), 0, None, _DoneState())
    db_session.expire_all()
    cands = _candidates()
    assert len(cands) == 1 and cands[0].code_snapshot_sha256 == sha_c


# ---------------------------------------------------------------- §10 采纳 / 否决


def test_accept_version_numbering_and_guards(client, workspace: Path, tmp_path: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)

    # 没有候选 → 409
    empty_loop = _create_round(c, tid)["loop"]
    r = c.post(f"/api/loops/{empty_loop['id']}/accept", json={})
    assert r.status_code == 409 and "候选" in r.text

    with Session() as db:
        task = db.get(Task, tid)
        seeded = {"task": task}
        # G7：采纳必须有 before/after 成对 test+benchmark、gate pass、perf-diff pass、
        # 精度不退化、报告落盘、工作区仍是候选代码；候选存点本身不够。
        loop1, _ = _seed_round_pair(
            db, seeded, workspace, tmp_path,
            objective={"type": "perf", "target_dimension": "all"}, tag="a")
        loop1_id = loop1.id

    r = c.post(f"/api/loops/{loop1_id}/accept",
               json={"label": "v1 优化", "note": "采纳"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_number"] == 1
    assert body["version"]["kind"] == "accept" and body["version"]["loop_id"] == loop1_id
    assert body["version"]["label"] == "v1 优化" and body["version"]["note"] == "采纳"
    assert body["workspace_matches"] is True
    assert body["loop"]["status"] == "closed" and body["loop"]["conclusion"] == "accepted"
    # 重复采纳 → 409
    assert c.post(f"/api/loops/{loop1_id}/accept", json={}).status_code == 409

    # 第二轮：版本编号 +1，默认 label 用 c<n> · scope
    with Session() as db:
        task = db.get(Task, tid)
        seeded = {"task": task}
        loop2, _ = _seed_round_pair(
            db, seeded, workspace, tmp_path,
            objective={"type": "perf", "target_dimension": "all"}, tag="b")
        loop2_id = loop2.id
    r = c.post(f"/api/loops/{loop2_id}/accept", json={})
    assert r.status_code == 200, r.text
    assert r.json()["version_number"] == 2
    assert r.json()["version"]["label"].startswith("c2 · ")


def test_reject_marks_candidate_and_closes_loop(client, workspace: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    loop = _create_round(c, tid)["loop"]
    cand_id = _make_candidate(Session, workspace, tid, loop["id"], text="// kernel c\n")

    r = c.post(f"/api/loops/{loop['id']}/reject", json={"reason": "性能回退，不采纳"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["savepoint"]["id"] == cand_id
    assert body["savepoint"]["kind"] == "rejected"
    assert body["savepoint"]["note"] == "性能回退，不采纳"
    assert body["loop"]["status"] == "closed"
    assert body["loop"]["conclusion"] == "rejected：性能回退，不采纳"
    # 否决后不能再采纳 / 重复否决
    assert c.post(f"/api/loops/{loop['id']}/accept", json={}).status_code == 409
    assert c.post(f"/api/loops/{loop['id']}/reject", json={}).status_code == 409


# ---------------------------------------------------------------- §10 可复现报告


def test_report_perf_objective_summary_and_write(db_session, seeded, workspace: Path,
                                                 tmp_path: Path):
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "perf", "label": "1M 带宽优化",
        "target_dimension": "cases", "target_cases": ["bench_1m"],
        "targets_snapshot": {"benchmark_run_id": None, "test_run_id": None,
                             "primary_case": "bench_1m", "suite_ref": SUITE},
    }
    loop, ctx = _seed_round_pair(db_session, seeded, workspace, tmp_path, objective=objective,
                                 before_med=10.0, after_med=8.0)
    report = optimization.report(db_session, loop)
    assert report["schema"] == "opx-optimization-report/1"
    assert report["task_id"] == seeded["task"].id and report["loop_id"] == loop.id
    assert report["label"] == "1M 带宽优化"
    assert report["objective"]["type"] == "perf"
    assert report["targets"]["primary_case"] == "bench_1m"
    assert report["targets"]["target_cases"] == ["bench_1m"]
    assert report["code"]["baseline_savepoint_id"] == ctx["baseline"].id
    assert report["code"]["candidate_savepoint_id"] == ctx["candidate"].id
    assert report["code"]["baseline_sha256"] == ctx["sha_a"]
    assert report["code"]["candidate_sha256"] == ctx["sha_b"]
    assert report["runs"] == {
        "before": {"test": ctx["before"]["test"].id,
                   "benchmark": ctx["before"]["benchmark"].id},
        "after": {"test": ctx["after"]["test"].id,
                  "benchmark": ctx["after"]["benchmark"].id},
    }
    assert report["perf_diff"]["schema"] == "opx-perf-diff/1"
    assert report["perf_diff"]["conclusion"] == "improved"
    assert report["precision"]["non_regression"] is True
    summary = report["objective_summary"]
    assert summary["type"] == "perf" and summary["verdict"] == "pass"
    assert summary["conclusion"] == "improved"
    assert summary["headline"] == "主用例 -20.0%，精度不退化"
    assert report["repro"]["before"]["test_command"].endswith("--action test")
    assert report["repro"]["before"]["benchmark_command"].endswith("--action benchmark")
    assert report["repro"]["after"]["benchmark_command"].endswith("--action benchmark")
    assert report["repro"]["env"]["criteria_ref"] == CRITERIA
    assert report["repro"]["env"]["image_digest"] == DIGEST
    assert report["repro"]["env"]["device_id"] == "1"
    assert report["generated_at"]
    assert "written_to" not in report

    # write=true：Step 3 起写 append-only 归档目录（report.json + manifest.json + raw/ + repro.md）
    written = optimization.report(db_session, loop, write=True)
    target = Path(written["written_to"])
    report_dir = Path(written["report_dir"])
    assert target.name == "report.json" and target.parent == report_dir
    assert report_dir.name.startswith(f"{written['seq']:04d}-")
    assert report_dir.parent.name == str(loop.id)
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "opx-report-manifest/1"
    assert manifest["loop_id"] == loop.id and manifest["report_sha256"] == written["report_sha256"]
    assert any(item.get("copied") for item in manifest["raw"])
    assert (report_dir / "repro.md").is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema"] == "opx-optimization-report/1"
    assert data["objective_summary"]["conclusion"] == "improved"
    assert data["evidence"]["schema"] == "opx-report-evidence/1"
    assert data["evidence"]["runs"] and data["evidence"]["runs"][0]["run_id"]


def test_report_perf_summary_uses_non_primary_objective_target(
        db_session, seeded, workspace: Path, tmp_path: Path):
    """目标是 64K 时不能被持平的 1M 主用例遮住；性能不退化约束应把目标回退判 fail。"""
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA,
        "type": "perf",
        "target_dimension": "cases",
        "target_cases": ["bench_64k"],
        "constraints": {"precision_non_regression": True, "perf_non_regression": True},
        "targets_snapshot": {"primary_case": "bench_1m", "suite_ref": SUITE},
    }
    loop, _ctx = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=objective, tag="target-64k",
        perf_before=[_perf_case("bench_1m", 10.0),
                     _perf_case("bench_64k", 5.0, n=65536)],
        perf_after=[_perf_case("bench_1m", 10.0),
                    _perf_case("bench_64k", 7.0, n=65536)],
    )

    report = optimization.report(db_session, loop)
    # perf-diff 与 objective_summary 都必须按本轮显式目标，不能被 benchmark primary 遮住。
    assert report["perf_diff"]["conclusion"] == "regressed"
    assert report["perf_diff"]["verdict"] == "fail"
    assert report["perf_diff"]["summary"]["decision_basis"] == "target_cases"
    summary = report["objective_summary"]
    assert summary["verdict"] == "fail"
    assert summary["conclusion"] == "regressed"
    assert summary["headline"] == "目标用例 bench_64k +40.0%，精度不退化"
    assert [row["case_id"] for row in summary["target_cases"]] == ["bench_64k"]
    assert summary["missing_target_cases"] == []


def test_report_precision_round_perf_diff_stays_benchmark_legacy(
        db_session, seeded, workspace: Path, tmp_path: Path):
    """H2 回归：精度轮的 target_cases（test 用例 id）不能让 perf-diff 误判 fail。

    修复前：perf-diff 把 case_001 当 benchmark case_id 配对 → E_TARGET_CASE_MISSING / fail，
    而 precision 段的 objective_summary 却是 pass/improved（同一份报告两份矛盾结论）。
    修复后：perf-diff 回到 benchmark primary，精度目标匹配只在 precision 段生效。
    """
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "precision",
        "target_dimension": "cases", "target_cases": ["case_001"],
        "constraints": {"precision_non_regression": True, "perf_non_regression": True},
        "targets_snapshot": {"primary_case": "bench_1m", "suite_ref": SUITE},
    }
    loop, _ctx = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=objective, tag="precision-perf",
        before_med=10.0, after_med=8.0,
        corr_before=[_corr("case_001", "FAIL")], corr_after=[_corr("case_001", "PASS")])

    report = optimization.report(db_session, loop)
    perf = report["perf_diff"]
    assert perf["verdict"] == "pass", perf["findings"]
    assert "E_TARGET_CASE_MISSING" not in {f["code"] for f in perf["findings"]}
    assert perf["summary"]["decision_basis"] == "benchmark_primary"
    assert perf["summary"]["target_case_ids"] == []
    # 精度目标匹配仍由 precision 段给出（FAIL→PASS = improved）
    summary = report["objective_summary"]
    assert summary["type"] == "precision"
    assert summary["verdict"] == "pass" and summary["conclusion"] == "improved"
    assert report["precision"]["newly_passed"] == 1


def test_report_dimension_all_regression_consistent_and_not_pass(
        db_session, seeded, workspace: Path, tmp_path: Path):
    """H3 回归：dimension=all 时报告层与 perf-diff 用同一保守聚合；回退不再自报 pass。

    perf_non_regression=true → 两层都 fail/regressed；
    perf_non_regression=false → 不硬拦，但 objective_summary 如实给 warn（不是 pass）。
    """
    perf_before = [_perf_case("bench_1m", 10.0), _perf_case("bench_64k", 5.0, n=65536)]
    perf_after = [_perf_case("bench_1m", 9.0), _perf_case("bench_64k", 9.25, n=65536)]
    base_objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "perf", "target_dimension": "all",
        "targets_snapshot": {"primary_case": "bench_1m", "suite_ref": SUITE},
    }
    hard = dict(base_objective, constraints={
        "precision_non_regression": True, "perf_non_regression": True})
    loop, _ctx = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=hard, tag="all-hard",
        perf_before=perf_before, perf_after=perf_after)

    report = optimization.report(db_session, loop)
    assert report["perf_diff"]["conclusion"] == "regressed"
    assert report["perf_diff"]["verdict"] == "fail"
    assert report["perf_diff"]["summary"]["decision_basis"] == "all_cases"
    summary = report["objective_summary"]
    assert summary["conclusion"] == "regressed" and summary["verdict"] == "fail"
    assert "全部用例聚合 2 个" in summary["headline"]
    assert "回退 1" in summary["headline"]

    soft = dict(base_objective, constraints={
        "precision_non_regression": True, "perf_non_regression": False})
    loop_soft, _ctx_soft = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=soft, tag="all-soft",
        perf_before=perf_before, perf_after=perf_after)
    report_soft = optimization.report(db_session, loop_soft)
    assert report_soft["perf_diff"]["conclusion"] == "regressed"
    assert report_soft["perf_diff"]["verdict"] == "pass"   # 不设加速比硬线
    summary_soft = report_soft["objective_summary"]
    assert summary_soft["conclusion"] == "regressed"
    assert summary_soft["verdict"] == "warn"              # 新口径：不能再自报 pass


def test_report_missing_after_median_is_inconclusive(db_session, seeded, workspace: Path,
                                                     tmp_path: Path):
    """M12 回归（报告层）：after 缺有效 median → inconclusive；headline 不再崩溃/不再显示提升。"""
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "perf", "target_dimension": "all",
        "targets_snapshot": {"primary_case": "bench_1m", "suite_ref": SUITE},
    }
    missing_after = [{"case_id": "bench_1m", "n": 1048576,
                      "candidate": {"device_us": {}},
                      "baseline": {"device_us": {"median_us": 10.3}}}]
    loop, _ctx = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=objective, tag="missing-after",
        perf_before=[_perf_case("bench_1m", 10.0)], perf_after=missing_after)

    report = optimization.report(db_session, loop)
    assert report["perf_diff"]["verdict"] == "pass", report["perf_diff"]["findings"]
    assert report["perf_diff"]["conclusion"] == "inconclusive"
    row = report["perf_diff"]["cases"][0]
    assert row["verdict"] == "inconclusive" and row["missing_data"] is True
    assert row["delta_pct"] is None
    summary = report["objective_summary"]
    assert summary["conclusion"] == "inconclusive" and summary["verdict"] == "inconclusive"
    assert "缺有效测量" in summary["headline"]


def test_report_precision_objective_summary(db_session, seeded, workspace: Path,
                                            tmp_path: Path):
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "precision",
        "target_dimension": "cases", "target_cases": ["case_001"],
    }
    loop, _ctx = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=objective, tag="fixed",
        corr_before=[_corr("case_001", "FAIL")], corr_after=[_corr("case_001", "PASS")],
    )
    report = optimization.report(db_session, loop)
    summary = report["objective_summary"]
    assert summary["type"] == "precision"
    assert summary["verdict"] == "pass" and summary["conclusion"] == "improved"
    assert report["precision"]["newly_passed"] == 1
    assert report["precision"]["regressions"] == []
    assert report["precision"]["non_regression"] is True
    row = next(r for r in report["precision"]["target_cases"] if r["case_id"] == "case_001")
    assert row == {"case_id": "case_001", "before": "FAIL", "after": "PASS",
                   "newly_passed": True, "regressed": False}

    # 精度回退：已 PASS 变 FAIL → verdict fail / conclusion regressed
    objective_bad = dict(objective)
    loop_bad, _ = _seed_round_pair(
        db_session, seeded, workspace, tmp_path, objective=objective_bad, tag="regress",
        corr_before=[_corr("case_001", "PASS")], corr_after=[_corr("case_001", "FAIL")],
    )
    report_bad = optimization.report(db_session, loop_bad)
    summary_bad = report_bad["objective_summary"]
    assert summary_bad["verdict"] == "fail" and summary_bad["conclusion"] == "regressed"
    assert report_bad["precision"]["regressions"] == ["case_001"]
    assert report_bad["precision"]["non_regression"] is False


def test_report_hardware_includes_chip_summary_and_brief_selects_by_chip(
        db_session, seeded, workspace: Path, tmp_path: Path):
    chips = workspace / "ops" / "chips"
    chips.mkdir(parents=True, exist_ok=True)
    (chips / "dav-2201.yaml").write_text(
        "schema: opx-chip-profile/1\n"
        "models: [Ascend910B1]\n"
        "facts:\n"
        "  - {key: ub_size_bytes, value: 196608, confidence: high}\n"
        "  - {key: l1_size_bytes, value: 524288, confidence: high}\n"
        "model_variable:\n"
        "  vector_core_count: {value: 48, confidence: high}\n",
        encoding="utf-8")
    (chips / "other.yaml").write_text(
        "schema: opx-chip-profile/1\n"
        "models: [Ascend910B9]\n"
        "facts:\n"
        "  - {key: ub_size_bytes, value: 999999, confidence: high}\n",
        encoding="utf-8")

    # hardware_brief 按 target_chip 选画像（命中 / 未命中退回第一份）
    assert "196608" in optimize.hardware_brief(workspace, "Ascend910B1")
    assert "999999" not in optimize.hardware_brief(workspace, "Ascend910B1")
    assert "999999" in optimize.hardware_brief(workspace, "Ascend910B9")

    seeded["task"].target_chip = "Ascend910B1"
    db_session.flush()
    objective = {
        "schema": optimization.OBJECTIVE_SCHEMA, "type": "hardware",
        "target_dimension": "all",
    }
    loop, _ctx = _seed_round_pair(db_session, seeded, workspace, tmp_path, objective=objective,
                                  tag="hw")
    report = optimization.report(db_session, loop)
    summary = report["objective_summary"]
    assert summary["type"] == "hardware"
    assert summary["verdict"] == "pass" and summary["conclusion"] == "improved"
    assert summary["chip"]["target_chip"] == "Ascend910B1"
    assert summary["chip"]["profile"] == "ops/chips/dav-2201.yaml"
    assert "196608" in summary["chip"]["summary"]


# ---------------------------------------------------------------- §10 savepoints loop_id


def test_savepoints_loop_id_filter_and_out(client, workspace: Path):
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    loop1 = _create_round(c, tid)["loop"]
    loop2 = _create_round(c, tid)["loop"]
    cand1 = _make_candidate(Session, workspace, tid, loop1["id"], text="// kernel c1\n")
    cand2 = _make_candidate(Session, workspace, tid, loop2["id"], text="// kernel c2\n")

    rows1 = c.get(f"/api/tasks/{tid}/savepoints", params={"loop_id": loop1["id"]}).json()
    assert rows1 and {r["loop_id"] for r in rows1} == {loop1["id"]}
    assert cand1 in {r["id"] for r in rows1} and cand2 not in {r["id"] for r in rows1}
    rows2 = c.get(f"/api/tasks/{tid}/savepoints", params={"loop_id": loop2["id"]}).json()
    assert {r["loop_id"] for r in rows2} == {loop2["id"]}
    # 详情也带 loop_id
    detail = c.get(f"/api/savepoints/{cand1}").json()
    assert detail["loop_id"] == loop1["id"]


# ---------------------------------------------------------------- Step 0 门禁（自动优化 v1）


def test_baseline_gate_ready_then_sha_mismatch(client, workspace: Path, tmp_path: Path):
    """G1：同 sha、gate pass 的 before test+benchmark 才叫基线就绪；工作区一变就拒绝。"""
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    _write_kernel(workspace, "baseline\n")
    sha = savepoints.current_sha(workspace)
    with Session() as db:
        task = db.get(Task, tid)
        mt = _run_dir(tmp_path, "bt", snapshot=sha, corr_cases=[_corr("case_001")])
        mb = _run_dir(tmp_path, "bb", snapshot=sha, perf_cases=[_perf_case("bench_1m", 10.0)])
        _add_run(db, task, action="test", manifest=mt, snapshot=sha)
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha)
        db.commit()

    loop = _create_round(c, tid, {"type": "perf", "target_dimension": "all"})["loop"]
    with Session() as db:
        ev = optimization.baseline_evidence(db, db.get(Loop, loop["id"]))
    assert ev["ready"] is True, ev
    assert ev["before"]["test"]["gate"] == "pass"
    assert ev["before"]["benchmark"]["gate"] == "pass"

    _write_kernel(workspace, "changed after baseline\n")
    with Session() as db:
        ev = optimization.baseline_evidence(db, db.get(Loop, loop["id"]))
    assert ev["ready"] is False
    assert any(i["code"] == "E_BASELINE_SHA_MISMATCH" for i in ev["issues"])
    # 启动改码必须被后端 409（不是靠页面按钮 disabled）
    with Session() as db:
        loop_row = db.get(Loop, loop["id"])
        with pytest.raises(optimization.OptimizationError) as exc:
            optimization.require_baseline(db, loop_row)
    assert exc.value.status == 409 and "E_BASELINE_SHA_MISMATCH" in str(exc.value)


def test_baseline_gate_blocks_gate_fail_but_tolerates_device_change(client, workspace: Path,
                                                                    tmp_path: Path):
    """G1：Run.status=ok 不够——gate fail 不能放行；但设备不同不再拦（允许自动切卡）。"""
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)
    _write_kernel(workspace, "baseline\n")
    sha = savepoints.current_sha(workspace)
    with Session() as db:
        task = db.get(Task, tid)
        mt = _run_dir(tmp_path, "bt", snapshot=sha, corr_cases=[_corr("case_001", "FAIL")])
        mb = _run_dir(tmp_path, "bb", snapshot=sha, perf_cases=[_perf_case("bench_1m", 10.0)],
                      device="2")
        _add_run(db, task, action="test", manifest=mt, snapshot=sha)
        _add_run(db, task, action="benchmark", manifest=mb, snapshot=sha, device="2")
        db.commit()
    loop = _create_round(c, tid, {"type": "perf", "target_dimension": "all"})["loop"]
    with Session() as db:
        ev = optimization.baseline_evidence(db, db.get(Loop, loop["id"]))
    codes = {i["code"] for i in ev["issues"]}
    assert ev["ready"] is False
    assert "E_BASELINE_GATE_FAIL" in codes
    # 2026-10-28：去掉「必须同卡」的要求，设备不一致只记录、不再拦 baseline 门禁
    assert "E_BASELINE_ENV_MISMATCH" not in codes
    assert ev["env"]["device_consistent"] is False


def test_accept_gate_blocks_regressed_and_workspace_mismatch(client, workspace: Path,
                                                            tmp_path: Path):
    """G7：性能回退/工作区不是候选代码都不能采纳（后端 409，不做“先采纳再说”）。"""
    c, Session = client
    _pid, tid, _sid = _seed_api(Session, workspace)

    # 性能回退 + 目标要求 perf 不退化 → perf-diff fail → 采纳 409
    with Session() as db:
        loop, _ = _seed_round_pair(
            db, {"task": db.get(Task, tid)}, workspace, tmp_path,
            objective={"type": "perf", "target_dimension": "all",
                       "constraints": {"precision_non_regression": True,
                                       "perf_non_regression": True}},
            before_med=10.0, after_med=12.0, tag="reg")
        loop_reg = loop.id
    r = c.post(f"/api/loops/{loop_reg}/accept", json={})
    assert r.status_code == 409 and "E_ACCEPT_PERF" in r.text

    # 证据都成立，但工作区在候选验证后又被改过 → 409
    with Session() as db:
        loop, _ = _seed_round_pair(
            db, {"task": db.get(Task, tid)}, workspace, tmp_path,
            objective={"type": "perf", "target_dimension": "all"}, tag="wm")
        loop_ok = loop.id
    _write_kernel(workspace, "// edited after candidate\n")
    r = c.post(f"/api/loops/{loop_ok}/accept", json={})
    assert r.status_code == 409 and "E_ACCEPT_WORKSPACE_MISMATCH" in r.text


def test_finalize_benchmark_without_test_does_not_create_candidate(
        db_session, seeded, workspace: Path, monkeypatch):
    """候选必须有同 sha 的 test 成功 run；只跑 benchmark 不生成 candidate（否则采纳没有精度证据）。"""
    task = seeded["task"]
    sha_base = savepoints.current_sha(workspace)
    loop = Loop(task_id=task.id, kind="perf", scope="只跑 benchmark", hypothesis="h",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.flush()
    savepoints.create(db_session, task, label="基线", kind="baseline", loop_id=loop.id)
    db_session.commit()

    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(action_runner, "SessionLocal", Session)
    _write_kernel(workspace, "// kernel v2\n")
    sha_c = savepoints.current_sha(workspace)
    run = Run(task_id=task.id, loop_id=loop.id, action="benchmark", status="ok", exit_code=0,
              remote_workdir="rONLY", code_snapshot_sha256=sha_c)
    job = ActionJob(task_id=task.id, loop_id=loop.id, action="benchmark", status="running",
                    run_ref="rONLY")
    db_session.add_all([run, job])
    db_session.commit()
    action_runner._finalize(job.id, 0, None, _DoneState())
    db_session.expire_all()
    assert list(db_session.scalars(select(Savepoint).where(
        Savepoint.loop_id == loop.id, Savepoint.kind == "candidate"))) == []


def test_report_write_is_not_silent_when_no_after_benchmark(db_session, seeded,
                                                            workspace: Path):
    """报告是采纳证据：write=true 落不了盘必须 409，不能静默返回。"""
    task = seeded["task"]
    loop = Loop(task_id=task.id, kind="perf", scope="无候选", hypothesis="h",
                objective_json=json.dumps({"type": "perf", "target_dimension": "all"}))
    db_session.add(loop)
    db_session.flush()
    savepoints.create(db_session, task, label="基线", kind="baseline", loop_id=loop.id)
    db_session.commit()
    with pytest.raises(optimization.OptimizationError) as exc:
        optimization.report(db_session, loop, write=True)
    assert exc.value.status == 409 and "报告无法落盘" in str(exc.value)


# ---------------------------------------------------------------- §10 迁移往返


def test_migration_adds_columns_and_roundtrips(tmp_path: Path, monkeypatch):
    db = tmp_path / "mig.db"
    monkeypatch.setattr(settings, "db_url", f"sqlite:///{db}")
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def _cols(table: str) -> set[str]:
        con = sqlite3.connect(db)
        try:
            return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        finally:
            con.close()

    def _indexes(table: str) -> set[str]:
        con = sqlite3.connect(db)
        try:
            return {r[1] for r in con.execute(f"PRAGMA index_list({table})")}
        finally:
            con.close()

    command.upgrade(cfg, "head")
    assert "objective_json" in _cols("loops")
    assert "loop_id" in _cols("savepoints")
    assert "ix_savepoints_loop_id" in _indexes("savepoints")

    command.downgrade(cfg, "a9b8c7d6e5f4")
    assert "objective_json" not in _cols("loops")
    assert "loop_id" not in _cols("savepoints")

    command.upgrade(cfg, "head")
    assert "objective_json" in _cols("loops") and "loop_id" in _cols("savepoints")
