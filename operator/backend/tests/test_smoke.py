"""smoke 动作（开发内冒烟：编译 + 最小用例跑通）的契约与门禁。

背景（2026-10-23 用户要求）：算子开发完不能直接等全套 test/benchmark —— 先编译 + 一条最小用例
证明能跑；smoke 失败要硬拦，且阶段「开发」必须在 smoke 通过后才算完成。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.models import Run, Task
from app.services import exec_actions, gate

CRITERIA = "ops-precision-standard@1"
DIGEST = "sha256:deadbeef"
SNAP = "a" * 64
SUITE = "cases@0123456789abcdef"
VER = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA, "suite_ref": SUITE}


def _mk_smoke_run(tmp_path: Path, name: str, *, smoke: dict | None,
                  digest: str = DIGEST) -> Path:
    rd = tmp_path / name
    (rd / "out").mkdir(parents=True)
    arts = []
    log = b"ok\n"
    (rd / "out" / "run.log").write_bytes(log)
    arts.append({"type": "log", "path": "out/run.log",
                 "sha256": hashlib.sha256(log).hexdigest(), "size_bytes": len(log)})
    if smoke is not None:
        body = json.dumps(smoke).encode()
        (rd / "out" / "smoke.json").write_bytes(body)
        arts.append({"type": "smoke", "path": "out/smoke.json",
                     "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)})
    man = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0,
           "code_snapshot_sha256": SNAP, "spec_version_id": 1, "criteria_ref": CRITERIA,
           "suite_ref": SUITE, "image_digest": digest, "image": "x", "device": "0",
           "lock_mode": "exclusive", "snapshot": {"files": 1, "truncated": False},
           "artifacts": arts}
    (rd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return rd / "manifest.json"


def _run(db, task: Task, man: Path, *, digest: str = DIGEST) -> Run:
    r = Run(task_id=task.id, action="smoke", status="ok", manifest_path=str(man),
            spec_version_id=1, code_snapshot_sha256=SNAP, suite_ref=SUITE,
            criteria_ref=CRITERIA, image_digest=digest, device_id="0", exit_code=0)
    db.add(r)
    db.commit()
    return r


def test_smoke_action_contract():
    spec = exec_actions.ACTION_SPECS["smoke"]
    assert spec["entry"] == "smoke_check.sh"  # 别和根目录的环境自检 smoke.sh 混
    assert spec["needs_device"] is True
    assert "smoke" in spec["artifact_types"]
    assert "cases_confirmed" in spec["requires"]
    assert "smoke_pass" in exec_actions.ACTION_SPECS["test"]["requires"]
    assert "smoke_pass" in exec_actions.ACTION_SPECS["benchmark"]["requires"]


def test_gate_smoke_pass(db_session, seeded, tmp_path: Path):
    man = _mk_smoke_run(tmp_path, "s-pass", smoke={
        **VER, "schema": "opx-smoke/1", "verdict": "PASS",
        "case": {"case_id": "add_edge_001", "status": "PASS", "n": 1},
    })
    run = _run(db_session, seeded["task"], man)
    rep = gate.evaluate(run)
    assert rep["verdict"] == "pass", rep["findings"]
    assert exec_actions.smoke_gate_reason(
        db_session, seeded["task"], spec_version_id=1,
        code_snapshot_sha256=SNAP, suite_ref=SUITE, criteria_ref=CRITERIA,
        image_digest=DIGEST,
    ) is None
    assert "没有门禁通过" in exec_actions.smoke_gate_reason(
        db_session, seeded["task"], spec_version_id=1,
        code_snapshot_sha256="b" * 64, suite_ref=SUITE, criteria_ref=CRITERIA,
        image_digest=DIGEST,
    )


def test_image_digests_match_tolerates_truncated_metadata():
    """镜像元数据可能只记短 digest（控制台手工登记）；同一镜像的短/全 digest 必须认作一致。"""
    full = "local-id:sha256:313b20aa34b4a7cf405e291a7e5ed1a8b63551522abeaf9317da396f0f90cf8d"
    assert len(full.split("sha256:")[-1]) == 64
    assert exec_actions.image_digests_match(full, "local-id:sha256:313b20aa34b4")
    assert exec_actions.image_digests_match("local-id:sha256:313b20aa34b4", full)
    assert exec_actions.image_digests_match("sha256:deadbeef", "sha256:deadbeef")
    assert not exec_actions.image_digests_match(full, "local-id:sha256:" + "b" * 64)
    assert not exec_actions.image_digests_match(full, "local-id:sha256:deadbeef")  # <12 hex 不认
    assert not exec_actions.image_digests_match(None, full)


def test_smoke_gate_accepts_run_with_full_digest_when_metadata_is_short(db_session, seeded,
                                                                       tmp_path: Path):
    """回归（2026-10-28 实测）：焦点镜像表里 digest 记短了，不能把已通过的 smoke 判成「没有」。"""
    full = "local-id:sha256:313b20aa34b4a7cf405e291a7e5ed1a8b63551522abeaf9317da396f0f90cf8d"
    short = "local-id:sha256:313b20aa34b4"
    man = _mk_smoke_run(tmp_path, "s-full-digest", smoke={
        **VER, "schema": "opx-smoke/1", "verdict": "PASS",
        "case": {"case_id": "sub_norm_001", "status": "PASS", "n": 1},
    }, digest=full)
    _run(db_session, seeded["task"], man, digest=full)
    assert exec_actions.smoke_gate_reason(
        db_session, seeded["task"], spec_version_id=1,
        code_snapshot_sha256=SNAP, suite_ref=SUITE, criteria_ref=CRITERIA,
        image_digest=short,
    ) is None
    # 真正换成另一个镜像仍要被拦
    assert "没有门禁通过" in exec_actions.smoke_gate_reason(
        db_session, seeded["task"], spec_version_id=1,
        code_snapshot_sha256=SNAP, suite_ref=SUITE, criteria_ref=CRITERIA,
        image_digest="local-id:sha256:" + "c" * 64,
    )


def test_gate_smoke_fail_is_hard_fail(db_session, seeded, tmp_path: Path):
    man = _mk_smoke_run(tmp_path, "s-fail", smoke={
        **VER, "verdict": "FAIL",
        "case": {"case_id": "add_edge_001", "status": "FAIL", "detail": "max_abs_err 超限"},
    })
    run = _run(db_session, seeded["task"], man)
    rep = gate.evaluate(run)
    assert rep["verdict"] == "fail"
    assert "E_SMOKE_FAILED" in {f["code"] for f in rep["findings"]}


def test_gate_smoke_missing_report(db_session, seeded, tmp_path: Path):
    man = _mk_smoke_run(tmp_path, "s-missing", smoke=None)
    run = _run(db_session, seeded["task"], man)
    rep = gate.evaluate(run)
    assert rep["verdict"] == "fail"
    assert "E_SMOKE_REPORT_MISSING" in {f["code"] for f in rep["findings"]}


def test_register_smoke_run_via_api(client, workspace: Path):
    """nm_test.sh 会用 action=smoke 回填台账；/api/actions 的允许列表必须包含它。"""
    c, Session = client
    from app.models import Host, HostImage, Project, SpecVersion

    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="add", op_name="elementwise_add")
        db.add(t)
        db.flush()
        sv = SpecVersion(task_id=t.id, version=1, spec_json="{}")
        db.add(sv)
        db.flush()
        t.current_spec_version_id = sv.id
        db.commit()
        tid = t.id

    man = _mk_smoke_run(workspace / "runs" / "smoke-man", "smoke-man", smoke={
        **VER, "verdict": "PASS", "case": {"case_id": "c", "status": "PASS"},
    })
    r = c.post(f"/api/tasks/{tid}/runs", json={
        "action": "smoke", "status": "ok", "manifest_path": str(man), "exit_code": 0,
        "spec_version_id": 1, "code_snapshot_sha256": SNAP, "suite_ref": SUITE,
        "criteria_ref": CRITERIA, "image_digest": DIGEST, "device_id": "0",
        "remote_workdir": "smoke-man",
    })
    assert r.status_code == 201, r.text
    assert r.json()["action"] == "smoke"
