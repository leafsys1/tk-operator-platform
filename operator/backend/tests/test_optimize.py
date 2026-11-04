"""M3 余项：agent 自动改代码（接进 perf Loop）。

dsh 与模型都假掉（monkeypatch `agent_runner.start`）：不花 token、不依赖 CLI。
只测「契约与证据」——起 job / 跑完落改前改后快照 + diff + 摘要 / 一键回退 / 并发保护 / 空态。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import AgentJob, Conversation, Loop, Project, Run, Task
from app.routers import runs as runs_router
from app.services import agent_runner, optimize, savepoints
from app.services.prose_relay import SPEC_SEP


@pytest.fixture(autouse=True)
def _clean_optimize_pending():
    """模块级 PENDING 是运行态：假 start 不会自己收尾，别把下一个用例的 Loop 卡成 409。"""
    optimize.PENDING.clear()
    yield
    optimize.PENDING.clear()


_CRITERIA = "ops-precision-standard@1"
_SUITE = "cases@deadbeef"
_DIGEST = "sha256:test"


def _synth_run_dir(root: Path, action: str, sha: str) -> Path:
    """造一个 gate 能判 pass 的最小 run 产物（manifest + 机读报告 + log）。"""
    d = root / action
    (d / "out").mkdir(parents=True, exist_ok=True)
    common = {"spec_version_id": 1, "code_snapshot_sha256": sha,
              "criteria_ref": _CRITERIA, "suite_ref": _SUITE}
    arts: list[dict] = []
    if action == "test":
        body = json.dumps({**common, "cases": [
            {"case_id": "case_001", "status": "PASS", "required": True},
        ]}, ensure_ascii=False).encode()
        (d / "out" / "correctness.json").write_bytes(body)
        arts.append({"type": "correctness", "path": "out/correctness.json"})
    else:
        body = json.dumps({**common, "cases": [{
            "case_id": "bench_1m", "n": 1048576, "dtype": "float32",
            "candidate": {"device_us": {"median_us": 10.0, "std_us": 0.1}},
            "baseline": {"device_us": {"median_us": 10.2}},
        }], "summary": {"primary_case": "bench_1m"}}, ensure_ascii=False).encode()
        (d / "out" / "performance.json").write_bytes(body)
        arts.append({"type": "performance", "path": "out/performance.json"})
    (d / "out" / "run.log").write_bytes(b"ok\n")
    arts.append({"type": "log", "path": "out/run.log"})
    manifest = {
        "schema": "opx-manifest/1", "run_id": f"synth-{action}", "exit_code": 0,
        "image_digest": _DIGEST, **common, "device": "1", "artifacts": arts,
    }
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _seed(db, workspace: Path, *, hypothesis: str = "增大分块以减少搬运循环",
          objective: bool = True) -> tuple[int, int]:
    p = Project(name="p", workspace_path=str(workspace), criteria_ref="ops-precision-standard@1")
    db.add(p)
    db.flush()
    t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add", target_chip="910B1")
    db.add(t)
    db.flush()
    # 自动优化只允许 objective 轮次：测试默认建一个带 allowed_paths=src/ 的 objective Loop。
    objective_json = None
    if objective:
        objective_json = json.dumps({
            "schema": "opx-optimization-objective/1", "type": "perf",
            "target_dimension": "all", "target_cases": [],
            "constraints": {"precision_non_regression": True, "perf_non_regression": False,
                            "allowed_paths": ["src/"]},
            "budget": {"diagnose_rounds": 3, "verify_rounds": 3, "runs_per_action": 2},
        }, ensure_ascii=False)
    loop = Loop(task_id=t.id, kind="perf", scope="分块/搬运", hypothesis=hypothesis,
                judge_condition="精度不退化；耗时同条件 A/B", status="open",
                objective_json=objective_json)
    db.add(loop)
    db.flush()
    if objective:
        # G1 基线门禁：objective 轮次启动改码前必须有同 sha、gate pass 的 before test+benchmark。
        # 这里造最小可判 pass 的 run 产物，后续用例专心测 agent 改码证据/回退。
        sha = savepoints.current_sha(workspace)
        savepoints.create(db, t, label="基线 · 测试", kind="baseline", loop_id=loop.id)
        root = workspace.parent / f"synth-runs-{loop.id}"
        for action in ("test", "benchmark"):
            manifest = _synth_run_dir(root, action, sha)
            db.add(Run(
                task_id=t.id, loop_id=loop.id, action=action, status="ok", exit_code=0,
                manifest_path=str(manifest), code_snapshot_sha256=sha,
                criteria_ref=_CRITERIA, suite_ref=_SUITE, image_digest=_DIGEST,
                device_id="1", spec_version_id=1,
            ))
    db.commit()
    return t.id, loop.id


def _seed_via(Session, workspace: Path, **kw) -> tuple[int, int]:
    """给端点测试用：每次开一个新 session 建数据。"""
    with Session() as db:
        return _seed(db, workspace, **kw)


# ---------------------------------------------------------------- 纯函数


def test_capture_diff_revert_roundtrip(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.h").write_text("x = 1\n", encoding="utf-8")
    before = optimize.capture(ws)

    (ws / "a.h").write_text("x = 2\n", encoding="utf-8")
    (ws / "b.cpp").write_text("// new\n", encoding="utf-8")
    after = optimize.capture(ws)

    changed, diff = optimize.diff_captures(before, after)
    assert {c["path"] for c in changed} == {"a.h", "b.cpp"}
    by = {c["path"]: c for c in changed}
    assert by["a.h"]["status"] == "modified" and by["a.h"]["additions"] == 1 and by["a.h"]["deletions"] == 1
    assert by["b.cpp"]["status"] == "added"
    assert "-x = 1" in diff and "+x = 2" in diff
    assert before.sha256 != after.sha256

    failed = optimize.apply_revert(ws, optimize._revert_payload(before, changed))
    assert failed == []
    assert (ws / "a.h").read_text(encoding="utf-8") == "x = 1\n"
    assert not (ws / "b.cpp").exists()
    assert optimize.capture(ws).sha256 == before.sha256  # 还原后快照回到改前


def test_apply_revert_reports_oversize_file(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "big.bin").write_bytes(b"a" * (optimize.MAX_DIFF_FILE_BYTES + 1))
    before = optimize.capture(ws)
    (ws / "big.bin").write_bytes(b"b" * (optimize.MAX_DIFF_FILE_BYTES + 2))
    changed, _ = optimize.diff_captures(before, optimize.capture(ws))
    payload = optimize._revert_payload(before, changed)
    assert payload["big.bin"]["restorable"] is False
    assert optimize.apply_revert(ws, payload) == ["big.bin"]  # 超限文件明确报「回退不了」


def test_build_prompt_embeds_loop_and_reference(workspace: Path, db_session):
    tid, lid = _seed(db_session, workspace)  # noqa: F841
    loop = db_session.get(Loop, lid)
    task = db_session.get(Task, tid)
    prompt = optimize.build_prompt(db=db_session, task=task, loop=loop, workspace=workspace,
                                   reference="[参考：Loop #2 的建议]\n- 待验证建议 R2-2：core-shrink\n",
                                   extra="只动 tiling")
    assert "增大分块以减少搬运循环" in prompt
    assert "core-shrink" in prompt
    assert "只动 tiling" in prompt
    assert "必须使用 workflow 工具" in prompt
    assert "ops-direct-invoke-architect" in prompt
    assert "ops-direct-invoke-developer" in prompt
    assert "ops-direct-invoke-verifier" in prompt
    assert "单变量归因" in prompt
    assert "不得要求必须先有该候选本身的 A/B" in prompt
    assert SPEC_SEP in prompt  # 两段式契约：模型要按分隔符吐 JSON


def test_build_prompt_inlines_hardware_facts(workspace: Path, db_session):
    """A11：硬件画像的关键事实要直接进 prompt（不是只给个目录名）。"""
    tid, lid = _seed(db_session, workspace)
    chips = workspace / "ops" / "chips"
    chips.mkdir(parents=True)
    (chips / "dav-2201.yaml").write_text(
        "facts:\n"
        "  - {key: ub_size_bytes, value: 196608, confidence: high}\n"
        "  - {key: l1_size_bytes, value: 524288, confidence: high}\n"
        "model_variable:\n"
        "  vector_core_count: {value: 48, confidence: high}\n"
        "  hbm_bandwidth_gbps: {value: 1600, confidence: medium}\n"
        "alignment: {datacopy_min_align_bytes: 32}\n"
        "vector_width: {fp32_elements_per_repeat: 64}\n",
        encoding="utf-8",
    )
    loop = db_session.get(Loop, lid)
    task = db_session.get(Task, tid)
    prompt = optimize.build_prompt(db=db_session, task=task, loop=loop, workspace=workspace,
                                   reference=None)
    assert "UB=196608B(high)" in prompt
    assert "L1=524288B(high)" in prompt
    assert "向量核=48(high)" in prompt
    assert "HBM GB/s=1600(medium)" in prompt
    assert "DataCopy对齐=32B" in prompt
    assert "PlatformAscendC" in prompt  # 运行时为准，不许硬编码


def test_evidence_empty_state(db_session, workspace: Path):
    _tid, lid = _seed(db_session, workspace)
    loop = db_session.get(Loop, lid)
    ev = optimize.evidence(loop)
    assert ev["status"] == "none" and ev["changed"] is False and ev["changed_files"] == []


# ---------------------------------------------------------------- 端点（dsh 假掉）


def _fake_start_recorder(captured: dict):
    def fake_start(job_id, prompt, cwd, dsh_home, *, on_done=None, stream=False, prose_of=None,
                   session_id=None):
        captured["on_done"] = on_done
        captured["prompt"] = prompt
        captured["stream"] = stream
        captured["session_id"] = session_id
        return agent_runner.JobState(job_id, prompt)
    return fake_start


class _DoneState:
    def __init__(self, final: str):
        self.final = final
        self.session_id = "ses-opt"
        self.exit_code = 0
        self.error = None
        self.usage = {}


def test_optimize_endpoint_records_evidence_and_revert(client, workspace: Path, monkeypatch):
    c, Session = client
    tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    r = c.post(f"/api/loops/{lid}/optimize", json={"extra": "保守一点"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert isinstance(body["job_id"], int)
    assert "增大分块以减少搬运循环" in body["prompt"] and "保守一点" in body["prompt"]
    assert captured["stream"] is True  # 逐 token 流式（前端打字机）

    detail = c.get(f"/api/tasks/{tid}").json()
    job = [j for j in detail["agent_jobs"] if j["kind"] == "优化"][-1]
    assert job["loop_id"] == lid and job["status"] == "running"
    # 并发保护：这一轮已有正在跑的自动优化
    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 409

    # 模拟 agent 跑完：改了 tiling，并给出结构化摘要
    final = ("我按最小改动调大了分块。\n" + SPEC_SEP + "\n"
             + json.dumps({"summary": "TILE_LENGTH 4096→8192", "changed_files": ["src/add.cpp"],
                           "hypothesis_check": "搬运循环次数减少", "expected": "主用例耗时下降",
                           "risks": "小 shape 可能回退", "changed": True, "verification": "pass",
                           "pipeline": {"architect": "通过", "developer": "已实施", "verifier": "PASS"}},
                          ensure_ascii=False))
    (workspace / "src" / "add.cpp").write_text("// kernel TILE=8192\n", encoding="utf-8")
    captured["on_done"](_DoneState(final))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "done" and ev["changed"] is True
    assert ev["agent_summary"] == "TILE_LENGTH 4096→8192"
    assert ev["verification"] == "pass" and ev["pipeline"]["verifier"] == "PASS"
    assert ev["changed_files"][0]["path"] == "src/add.cpp"
    assert ev["before_sha256"] != ev["after_sha256"] and "+// kernel TILE=8192" in ev["diff"]
    assert ev["reverted"] is False

    # 一键回退：工作区还原到改前
    rr = c.post(f"/api/loops/{lid}/optimize/revert")
    assert rr.status_code == 200, rr.text
    assert rr.json()["reverted"] is True and rr.json()["reverted_clean"] is True
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == "// kernel\n"


def test_optimize_no_change_is_recorded(client, workspace: Path, monkeypatch):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    captured["on_done"](_DoneState("证据不足，本次不动代码。\n" + SPEC_SEP + "\n"
                                   + json.dumps({"summary": "不动", "changed_files": [], "changed": False,
                                                 "verification": "pass"})))
    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["changed"] is False and ev["changed_files"] == []
    assert ev["agent_summary"] == "不动"


def test_optimize_guard_reverts_unverified_change(client, workspace: Path, monkeypatch):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    original = (workspace / "src" / "add.cpp").read_text(encoding="utf-8")
    (workspace / "src" / "add.cpp").write_text("// unverified\n", encoding="utf-8")
    captured["on_done"](_DoneState("验证失败。\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "候选未通过静态验证", "changed_files": ["src/add.cpp"], "changed": True,
        "verification": "fail", "pipeline": {"verifier": "FAIL"},
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    assert ev["changed"] is False and ev["reverted_clean"] is True
    assert "verifier 未明确 PASS" in ev["error"]
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == original


def test_optimize_guard_rejects_incomplete_no_change_workflow(client, workspace: Path, monkeypatch):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    captured["on_done"](_DoneState("架构师仍在运行，稍后继续。"))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["changed"] is False
    assert ev["verification"] is None and ev["guard_reverted"] is True
    assert "结构化输出解析失败" in ev["error"]


def test_optimize_guard_requires_verifier_evidence(client, workspace: Path, monkeypatch):
    """verification=pass 但 pipeline.verifier 为空 → 视为无独立复核证据，自动回退。"""
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    original = (workspace / "src" / "add.cpp").read_text(encoding="utf-8")
    (workspace / "src" / "add.cpp").write_text("// no verifier evidence\n", encoding="utf-8")
    captured["on_done"](_DoneState("改完了。\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "改 tile", "changed_files": ["src/add.cpp"], "changed": True,
        "verification": "pass",
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    assert "pipeline.verifier" in ev["error"]
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == original


def test_optimize_guard_empty_allowed_paths_fails_closed(client, workspace: Path, monkeypatch):
    """objective 没有 allowed_paths 时不能关闭越界门禁：有改动一律回退。"""
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    with Session() as db:
        loop = db.get(Loop, lid)
        loop.objective_json = json.dumps({"type": "perf", "target_dimension": "all",
                                          "constraints": {}}, ensure_ascii=False)
        db.commit()
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    original = (workspace / "src" / "add.cpp").read_text(encoding="utf-8")
    (workspace / "src" / "add.cpp").write_text("// changed without allowed paths\n", encoding="utf-8")
    captured["on_done"](_DoneState("x\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "改 tile", "changed_files": ["src/add.cpp"], "changed": True,
        "verification": "pass", "pipeline": {"verifier": "PASS"},
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    assert "allowed_paths" in ev["error"]
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == original


def test_optimize_guard_reverts_change_outside_allowed_path(client, workspace: Path, monkeypatch):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    case = workspace / "cases" / "case_001.json"
    original = case.read_text(encoding="utf-8")
    case.write_text('{"case_id": "tampered"}', encoding="utf-8")
    captured["on_done"](_DoneState("x\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "改了用例", "changed_files": ["cases/case_001.json"], "changed": True,
        "verification": "pass", "pipeline": {"verifier": "PASS"},
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    # 用例属于受保护证据输入：优先报 E_PROTECTED_PATH（比“越过 allowed_paths”更准确）
    assert "E_PROTECTED_PATH" in ev["error"] and "cases/case_001.json" in ev["error"]
    assert ev["protected_paths"] == ["cases/case_001.json"]
    assert case.read_text(encoding="utf-8") == original


def test_optimize_guard_reverts_protected_golden(client, workspace: Path, monkeypatch):
    """参考实现 golden 属于证据输入：即使 allowed_paths 放得很宽，改了也必须回滚。"""
    golden = workspace / "ops" / "elementwise_add" / "scripts" / "golden.py"
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text("def compute_golden(a, b):\n    return a + b\n", encoding="utf-8")
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    golden.write_text("def compute_golden(a, b):\n    return b\n", encoding="utf-8")
    captured["on_done"](_DoneState("改 oracle\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "改参考实现", "changed_files": ["ops/elementwise_add/scripts/golden.py"],
        "changed": True, "verification": "pass", "pipeline": {"verifier": "PASS"},
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    assert "E_PROTECTED_PATH" in ev["error"]
    assert ev["protected_paths"] == ["ops/elementwise_add/scripts/golden.py"]
    assert "return a + b" in golden.read_text(encoding="utf-8")


def test_optimize_guard_reverts_external_symlink(client, workspace: Path, monkeypatch, tmp_path: Path):
    """允许路径下新建指向工作区外的符号链接 → 拒绝并删除链接（不能写穿外部文件）。"""
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    external = tmp_path / "outside.txt"
    external.write_text("outside\n", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 201
    original = (workspace / "src" / "add.cpp").read_text(encoding="utf-8")
    (workspace / "src" / "evil").symlink_to(external)
    (workspace / "src" / "add.cpp").write_text("// symlink escape\n", encoding="utf-8")
    captured["on_done"](_DoneState("x\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "加链接", "changed_files": ["src/add.cpp", "src/evil"], "changed": True,
        "verification": "pass", "pipeline": {"verifier": "PASS"},
    }, ensure_ascii=False)))

    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "failed" and ev["guard_reverted"] is True
    assert "符号链接" in ev["error"]
    assert not (workspace / "src" / "evil").is_symlink()
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == original


def test_optimize_endpoint_rejects_legacy_loop_without_objective(client, workspace: Path, monkeypatch):
    """旧 Loop（无 objective）已改为只读：API 也不能起自动优化。"""
    c, Session = client
    _tid, lid = _seed_via(Session, workspace, objective=False)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")
    r = c.post(f"/api/loops/{lid}/optimize", json={})
    assert r.status_code == 409, r.text
    assert "objective" in r.text or "只读" in r.text


def test_capture_and_revert_symlink(tmp_path: Path):
    """capture 不跟随链接读目标；回退按链接本身还原（含断链删除）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    external = tmp_path / "outside.txt"
    external.write_text("outside\n", encoding="utf-8")
    (ws / "target.txt").write_text("inside\n", encoding="utf-8")
    (ws / "link").symlink_to(external)
    before = optimize.capture(ws)
    assert before.files["link"]["symlink"] == str(external)
    assert before.files["link"]["external"] is True

    # 链接被改指向工作区内 + 新建一个断链
    (ws / "link").unlink()
    (ws / "link").symlink_to(ws / "target.txt")
    (ws / "broken").symlink_to(ws / "missing.txt")
    changed, diff = optimize.diff_captures(before, optimize.capture(ws))
    paths = {c["path"] for c in changed}
    assert {"link", "broken"} <= paths
    assert "符号链接" in diff

    failed = optimize.apply_revert(ws, optimize._revert_payload(before, changed))
    assert failed == []
    assert (ws / "link").is_symlink() and str((ws / "link").readlink()) == str(external)
    assert not (ws / "broken").is_symlink()


def test_optimize_revert_without_evidence_409(client, workspace: Path):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    assert c.post(f"/api/loops/{lid}/optimize/revert").status_code == 409


def test_optimize_requires_dsh_and_key(client, workspace: Path, monkeypatch):
    c, Session = client
    _tid, lid = _seed_via(Session, workspace)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: None)
    monkeypatch.setenv("OPX_LLM_KEY", "x")
    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 400
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.delenv("OPX_LLM_KEY", raising=False)
    assert c.post(f"/api/loops/{lid}/optimize", json={}).status_code == 400


def test_loop_detail_exposes_optimize(client, workspace: Path, monkeypatch):
    c, Session = client
    tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")
    c.post(f"/api/loops/{lid}/optimize", json={})
    captured["on_done"](_DoneState("x\n" + SPEC_SEP + "\n" + json.dumps({"summary": "s", "changed": False})))
    detail = c.get(f"/api/tasks/{tid}").json()
    loop = next(l for l in detail["loops"] if l["id"] == lid)
    assert loop["optimize"]["agent_summary"] == "s"


# ---------------------------------------------------------------- 对话化自动优化


def test_chat_optimize_creates_conversation_and_writes_back(client, workspace: Path, monkeypatch):
    """自动优化跳对话：建会话/写用户消息 → 起优化 job → 跑完回写助手消息与 session。"""
    c, Session = client
    tid, lid = _seed_via(Session, workspace)
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder(captured))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    r = c.post(f"/api/loops/{lid}/optimize/chat", json={"instruction": "优先 64K，别动 1M"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["loop_id"] == lid and body["conversation_id"] and body["message_id"]
    assert "优先 64K" in captured["prompt"]
    assert captured["stream"] is True

    conv = c.get(f"/api/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user"]
    assert "优先 64K" in conv["messages"][0]["text"]
    job = next(j for j in c.get(f"/api/tasks/{tid}").json()["agent_jobs"]
               if j["id"] == body["job_id"])
    assert job["kind"] == "优化" and job["loop_id"] == lid
    assert job["conversation_id"] == body["conversation_id"]
    assert c.get(f"/api/loops/{lid}/conversation").json()["conversation_id"] == body["conversation_id"]

    original = (workspace / "src" / "add.cpp").read_text(encoding="utf-8")
    (workspace / "src" / "add.cpp").write_text("// chat optimized\n", encoding="utf-8")
    captured["on_done"](_DoneState("改好了。\n" + SPEC_SEP + "\n" + json.dumps({
        "summary": "调整 64K tile", "changed_files": ["src/add.cpp"], "changed": True,
        "verification": "pass", "pipeline": {"verifier": "PASS"},
    }, ensure_ascii=False)))

    conv2 = c.get(f"/api/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in conv2["messages"]] == ["user", "assistant"]
    text = conv2["messages"][-1]["text"]
    assert "调整 64K tile" in text and "下一步" in text and "候选验证" in text
    assert conv2["dsh_session_id"] == "ses-opt" and conv2["message_count"] == 2
    ev = c.get(f"/api/loops/{lid}/optimize").json()
    assert ev["status"] == "done" and ev["changed"] is True
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") != original

    # 会话复用：上一轮结束后可以在同一会话里再起一轮（fake start 不落 job 状态，手动收尾）。
    # 但 G1 要求工作区仍是基线代码：先模拟用户把工作区恢复到基线，再起第二轮。
    with Session() as db:
        db.get(AgentJob, body["job_id"]).status = "ok"
        db.commit()
    (workspace / "src" / "add.cpp").write_text(original, encoding="utf-8")
    r2 = c.post(f"/api/loops/{lid}/optimize/chat",
                json={"conversation_id": body["conversation_id"], "instruction": "再来一轮"})
    assert r2.status_code == 201, r2.text
    assert r2.json()["conversation_id"] == body["conversation_id"]


def test_chat_optimize_rejects_busy_and_foreign_conversation(client, workspace: Path, monkeypatch):
    c, Session = client
    tid, lid = _seed_via(Session, workspace)
    monkeypatch.setattr(agent_runner, "start", _fake_start_recorder({}))
    monkeypatch.setattr(optimize, "SessionLocal", Session)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")

    with Session() as db:
        busy = AgentJob(task_id=tid, kind="对话", status="running")
        db.add(busy)
        db.commit()
        busy_id = busy.id
    r = c.post(f"/api/loops/{lid}/optimize/chat", json={})
    # 基线门禁会先看到“工作区忙”（E_WORKSPACE_BUSY），同样是 409 且不应启动。
    assert r.status_code == 409 and ("已有" in r.text or "E_WORKSPACE_BUSY" in r.text)
    with Session() as db:
        db.get(AgentJob, busy_id).status = "ok"
        db.commit()

    with Session() as db:
        project = Project(name="other", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        other = Task(project_id=project.id, title="other", op_name="elementwise_add")
        db.add(other)
        db.flush()
        conv = Conversation(task_id=other.id, mode="task")
        db.add(conv)
        db.commit()
        foreign_id = conv.id
    r2 = c.post(f"/api/loops/{lid}/optimize/chat", json={"conversation_id": foreign_id})
    assert r2.status_code == 400 and "不属于" in r2.text
