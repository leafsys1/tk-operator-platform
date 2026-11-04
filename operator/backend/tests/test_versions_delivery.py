"""任务生命周期 v2：版本模型 / 证据判定 / 交付门禁（HTTP + service 级）。

覆盖：
- 定版创建 v1.0；落新规格 = 大版本 +1、小版本归 0、用例确认清空；
- 手工小版本 / 采纳优化小版本 +1；
- 证据按 (spec_version_id, code_snapshot_sha256, suite_ref) 与 run 匹配；
- 用例未确认、代码快照变化、用例集变化 → 交付门禁给出对应缺项；
- 交付包生成 + 下载 + 版本 delivered_at 冻结。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import settings
from app.models import Host, HostImage, Loop, Project, Run, SpecVersion, Task
from app.services import case_confirm, exec_actions

CRITERIA = "ops-precision-standard@1"
DIGEST = "sha256:deadbeef"


def _seed(Session, workspace: Path):
    """项目 + 执行机/镜像 + 任务 + 规格 v1（不建版本行；版本行由 API 流程产生）。"""
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, env_recipe="docker run ...", torch_version="2.10")
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id,
                          image_id=image.id, criteria_ref=CRITERIA)
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name="elementwise_add",
                    target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, op_name="elementwise_add", spec_json="{}",
                           sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        task.stage = "需求与设计"
        db.commit()
        return {"task_id": task.id, "spec_id": spec.id, "project_id": project.id}


def _write_deliverables(workspace: Path, op: str = "elementwise_add") -> None:
    case_dir = workspace / "cases" / op
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "cases.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
    (case_dir / "benchmark.json").write_text(json.dumps({
        "primary_case": "bench_1m",
        "cases": [{"case_id": "bench_1m", "n": 1048576, "dtype": "float32",
                   "required": True, "note": "带宽主场景"}],
    }), encoding="utf-8")
    (workspace / "ops" / op / "python").mkdir(parents=True, exist_ok=True)
    (workspace / "ops" / op / "python" / f"{op}.py").write_text("# python api\n", encoding="utf-8")
    (workspace / "ops" / op / "README.md").write_text("# readme\n", encoding="utf-8")
    (workspace / "ops" / op / "op_host").mkdir(parents=True, exist_ok=True)
    (workspace / "ops" / op / "op_host" / f"{op}.asc").write_text("// kernel\n", encoding="utf-8")
    (workspace / "tests").mkdir(exist_ok=True)
    (workspace / "tests" / f"test_{op}.py").write_text("# pytest\n", encoding="utf-8")
    (workspace / "docs" / op).mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / op / "design.md").write_text("# design\n", encoding="utf-8")
    (workspace / "examples").mkdir(exist_ok=True)
    (workspace / "examples" / f"{op}_demo.py").write_text("# demo\n", encoding="utf-8")


def _confirm_cases(c, task_id: int, workspace: Path) -> dict:
    op = "elementwise_add"
    cases = (workspace / "cases" / op / "cases.json").read_bytes()
    bench = (workspace / "cases" / op / "benchmark.json").read_bytes()
    r = c.post(f"/api/tasks/{task_id}/cases/confirm", json={
        "cases_sha256": hashlib.sha256(cases).hexdigest(),
        "benchmark_sha256": hashlib.sha256(bench).hexdigest(),
        "by": "test",
    })
    assert r.status_code == 200, r.text
    return r.json()


def _mk_run(tmp_path: Path, name: str, *, action: str, spec_id: int, snapshot: str,
            suite_ref: str) -> Path:
    rd = tmp_path / name
    (rd / "out").mkdir(parents=True)
    arts: list[dict] = []

    def add(rel: str, body: bytes, atype: str) -> None:
        p = rd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        arts.append({"type": atype, "path": rel, "sha256": hashlib.sha256(body).hexdigest(),
                     "size_bytes": len(body)})

    add("out/run.log", b"ok\n", "log")
    if action == "smoke":
        add("out/smoke.json", json.dumps({
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite_ref,
            "verdict": "PASS", "case": {"case_id": "c1", "status": "PASS"},
        }).encode(), "smoke")
    if action == "test":
        add("out/correctness.json", json.dumps({
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite_ref,
            "cases": [{"case_id": "c1", "status": "PASS", "required": True}],
        }).encode(), "correctness")
    if action == "benchmark":
        add("out/performance.json", json.dumps({
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite_ref,
            "summary": {"primary_case": "bench_1m"},
            "cases": [{"case_id": "bench_1m", "n": 1048576, "dtype": "float32",
                       "primary": True, "required": True,
                       "baseline": {"device_us": {"median_us": 9.0}},
                       "candidate": {"device_us": {"median_us": 8.0, "std_us": 0.1}},
                       "derived": {"speedup_baseline_over_candidate": 1.125,
                                   "candidate_achieved_bw_gbps": 1500.0,
                                   "baseline_achieved_bw_gbps": 1333.0}}],
        }).encode(), "performance")
    man = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0,
           "code_snapshot_sha256": snapshot, "spec_version_id": spec_id,
           "criteria_ref": CRITERIA, "suite_ref": suite_ref,
           "image_digest": DIGEST, "image": "x", "device": "0", "lock_mode": "exclusive",
           "snapshot": {"files": 1, "truncated": False}, "artifacts": arts}
    (rd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return rd / "manifest.json"


def _add_run(Session, task_id: int, action: str, man: Path, *, spec_id: int,
             snapshot: str, suite_ref: str) -> int:
    with Session() as db:
        run = Run(task_id=task_id, action=action, status="ok", manifest_path=str(man),
                  spec_version_id=spec_id, code_snapshot_sha256=snapshot,
                  suite_ref=suite_ref, criteria_ref=CRITERIA, image_digest=DIGEST,
                  device_id="0", exit_code=0)
        db.add(run)
        db.commit()
        return run.id


def _prepare(client, workspace: Path, tmp_path: Path, monkeypatch, *, with_runs: bool,
             confirm: bool = True) -> dict:
    """建任务 + 产物 + 可选三条证据 run；返回 ids/snapshot/suite_ref。"""
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)
    _write_deliverables(workspace)
    # 先落 v1.0（定版），再确认用例 —— 与真实流程一致（确认不会被版本创建清掉）
    with Session() as db:
        from app.services import versions
        task = db.get(Task, ids["task_id"])
        spec = db.get(SpecVersion, ids["spec_id"])
        versions.create_spec_version(db, task, spec)
        db.commit()
    if confirm:
        _confirm_cases(c, ids["task_id"], workspace)
    snap = exec_actions.operator_snapshot(workspace, "elementwise_add").sha256
    suite = case_confirm.cases_ref(workspace, "elementwise_add")
    if with_runs:
        for action in ("build", "smoke", "test", "benchmark"):
            man = _mk_run(tmp_path, f"{action}-1", action=action, spec_id=ids["spec_id"],
                          snapshot=snap, suite_ref=suite)
            _add_run(Session, ids["task_id"], action, man, spec_id=ids["spec_id"],
                     snapshot=snap, suite_ref=suite)
    return {**ids, "snapshot": snap, "suite_ref": suite}


def test_version_flow_and_delivery(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)

    # 证据齐全 → 交付门禁通过
    body = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert body["current"]["label"] == "v1.0"
    assert body["evidence"]["deliverable"] is True, body["evidence"]["missing"]
    assert body["evidence"]["cases"]["in_sync"] is True

    # 交付包
    r = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert r.status_code == 201, r.text
    pack = r.json()
    assert Path(pack["zip_path"]).is_file()
    with __import__("zipfile").ZipFile(pack["zip_path"]) as zf:
        names = zf.namelist()
        assert "ops/elementwise_add/op_host/elementwise_add.asc" in names
        assert "tests/test_elementwise_add.py" in names
        assert "docs/算子说明文档.md" in names
        assert "manifest.json" in names
    dl = c.get(pack["download_url"])
    assert dl.status_code == 200 and dl.content[:2] == b"PK"

    # 交付后版本标记 delivered；再次交付仍可（产物齐备）
    body = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert body["evidence"]["delivered_at"] is not None
    hist = c.get(f"/api/tasks/{ids['task_id']}/deliveries").json()["deliveries"]
    assert hist and hist[0]["gate_verdict"] == "pass"

    # 同一版本重复交付必须保留两份不可变产物，不能让新包覆盖旧 Artifact 指向的文件。
    again = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert again.status_code == 201, again.text
    second = again.json()
    assert second["zip_path"] != pack["zip_path"]
    assert Path(pack["zip_path"]).is_file() and Path(second["zip_path"]).is_file()
    assert c.get(pack["download_url"]).content[:2] == b"PK"


def test_delivery_backfills_and_freezes_legacy_version_row(client, workspace: Path,
                                                           tmp_path: Path, monkeypatch):
    """历史任务无 task_versions 行时，交付应补录当前 vM.m，而不是永久显示“版本行未落库”。"""
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)
    _write_deliverables(workspace)
    _confirm_cases(c, ids["task_id"], workspace)
    snap = exec_actions.operator_snapshot(workspace, "elementwise_add").sha256
    suite = case_confirm.cases_ref(workspace, "elementwise_add")
    for action in ("build", "smoke", "test", "benchmark"):
        man = _mk_run(tmp_path, f"legacy-{action}", action=action, spec_id=ids["spec_id"],
                      snapshot=snap, suite_ref=suite)
        _add_run(Session, ids["task_id"], action, man, spec_id=ids["spec_id"],
                 snapshot=snap, suite_ref=suite)

    before = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert before["current"] is None
    assert before["evidence"]["deliverable"] is True

    created = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert created.status_code == 201, created.text
    after = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert after["current"]["label"] == "v1.0"
    assert after["evidence"]["delivered_at"] is not None
    assert after["evidence"]["current_version_id"] is not None


def test_delivery_blocked_when_cases_unconfirmed(client, workspace: Path, tmp_path: Path,
                                                 monkeypatch):
    c, Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=False, confirm=False)
    r = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    codes = detail["codes"]
    assert "E_CASES_UNCONFIRMED" in codes
    assert any(m["action"] == "cases" for m in detail["missing"])


def test_delivery_blocked_after_code_change(client, workspace: Path, tmp_path: Path,
                                           monkeypatch):
    """交付前改代码（工作区快照变）→ 证据失效，必须重跑 build/smoke/test/benchmark。"""
    c, Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)
    (workspace / "ops/elementwise_add/op_host/elementwise_add.asc").write_text("// changed\n",
                                                                              encoding="utf-8")
    ev = c.get(f"/api/tasks/{ids['task_id']}/versions").json()["evidence"]
    assert ev["deliverable"] is False
    assert {m["action"] for m in ev["missing"]} == {"build", "smoke", "test", "benchmark"}
    r = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert r.status_code == 409


def test_delivery_ignores_other_operator_changes(client, workspace: Path, tmp_path: Path,
                                                 monkeypatch):
    """#11 的代码/用例变更不应使 #12 的已通过证据失效。"""
    c, _Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)
    other_op = workspace / "ops" / "other_op"
    other_cases = workspace / "cases" / "other_op"
    other_op.mkdir(parents=True)
    other_cases.mkdir(parents=True)
    (other_op / "kernel.asc").write_text("// other", encoding="utf-8")
    (other_cases / "cases.json").write_text('{"cases": []}', encoding="utf-8")
    (other_cases / "benchmark.json").write_text('{"cases": []}', encoding="utf-8")
    ev = c.get(f"/api/tasks/{ids['task_id']}/versions").json()["evidence"]
    assert ev["deliverable"] is True, ev["missing"]


def test_delivery_blocked_when_minor_version_workspace_changes(client, workspace: Path,
                                                               tmp_path: Path, monkeypatch):
    """小版本已固化快照后再改代码，旧 run 不能继续证明当前待打包工作区。"""
    c, _Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)
    created = c.post(f"/api/tasks/{ids['task_id']}/versions", json={"label": "人工版本"})
    assert created.status_code == 201
    assert created.json()["evidence"]["deliverable"] is True

    (workspace / "ops/elementwise_add/op_host/elementwise_add.asc").write_text(
        "// changed after version\n", encoding="utf-8")
    ev = c.get(f"/api/tasks/{ids['task_id']}/versions").json()["evidence"]
    assert ev["deliverable"] is False
    assert any(m["action"] == "version" for m in ev["missing"])
    assert c.post(f"/api/tasks/{ids['task_id']}/deliveries").status_code == 409


def test_delivery_rejects_workspace_symlink(client, workspace: Path, tmp_path: Path,
                                            monkeypatch):
    """交付包不能跟随 ops 下的符号链接读取工作区外文件。"""
    c, _Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "ops/elementwise_add/leak.txt").symlink_to(outside)

    response = c.post(f"/api/tasks/{ids['task_id']}/deliveries")
    assert response.status_code == 409, response.text
    assert "E_DELIVERY_PATH_UNSAFE" in response.json()["detail"]["codes"]


def test_waiver_rejects_unknown_actions(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, _Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=False)
    response = c.post(f"/api/tasks/{ids['task_id']}/versions/waiver", json={
        "actions": ["delivery", "made-up"], "reason": "不能写空豁免",
    })
    assert response.status_code == 400


def test_manual_minor_and_spec_major(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=False)

    r = c.post(f"/api/tasks/{ids['task_id']}/versions", json={"label": "修 bug", "note": "对话确认"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["current"]["major"] == 1 and body["current"]["minor"] == 1

    # 落新规格 → 大版本 +1、小版本归 0、用例确认清空
    r = c.post(f"/api/tasks/{ids['task_id']}/specs", json={
        "spec_json": "{\"op_name\":\"elementwise_add\",\"math_def\":\"a+b\"}",
        "set_current": True, "change_reason": "对话确认的实质变更",
    })
    assert r.status_code == 201, r.text
    body = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert body["current"]["major"] == 2 and body["current"]["minor"] == 0
    assert body["evidence"]["cases"]["confirmed_at"] is None


def test_optimization_accept_bumps_minor(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=False)
    with Session() as db:
        task = db.get(Task, ids["task_id"])
        loop = Loop(task_id=task.id, kind="perf", status="open", scope="tile 优化")
        db.add(loop)
        db.flush()
        db.commit()
        loop_id = loop.id

    body = c.post(f"/api/loops/{loop_id}/close?conclusion=accepted").json()
    assert body["conclusion"] == "accepted"
    v = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert v["current"]["major"] == 1 and v["current"]["minor"] == 1
    assert v["current"]["source"] == "optimization"
    assert v["current"]["loop_id"] == loop_id

    # 幂等：第二次 close 不应再 +1（loop 已 closed 时接口仍会 accept? close 是通用关闭口，允许重复调）
    c.post(f"/api/loops/{loop_id}/close?conclusion=accepted")
    v = c.get(f"/api/tasks/{ids['task_id']}/versions").json()
    assert v["current"]["minor"] == 1


def test_cases_include_performance_section(client, workspace: Path, tmp_path: Path, monkeypatch):
    """测试管理页数据源：GET /cases 同时给精度矩阵 + 性能用例矩阵（含最近实测）。"""
    ids = _prepare(client, workspace, tmp_path, monkeypatch, with_runs=True)
    body = client[0].get(f"/api/tasks/{ids['task_id']}/cases").json()
    perf = body["performance"]
    assert perf["source"] == "cases/elementwise_add/benchmark.json"
    assert perf["primary_case"] == "bench_1m"
    case = perf["cases"][0]
    assert case["case_id"] == "bench_1m" and case["primary"] is True
    latest = case["latest"]
    assert latest is not None
    assert latest["candidate_median_us"] == 8.0
    assert latest["baseline_median_us"] == 9.0
    assert latest["speedup"] == 1.125
    assert perf["runs"] and perf["runs"][0]["gate_verdict"] == "pass"
