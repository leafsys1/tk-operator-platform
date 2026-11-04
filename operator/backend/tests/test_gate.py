"""门禁校验器 v1 的 fail code 行为（每个 code 都要能被造出来）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.models import Run
from app.services import gate

SNAP = "a" * 64
DIGEST = "sha256:deadbeef"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@0123456789abcdef"


def _write_run_dir(tmp_path: Path, artifacts: dict[str, bytes], *,
                   extra_manifest: dict | None = None,
                   correctness: dict | None = None, metrics: dict | None = None,
                   override_sha: bool = False) -> tuple[Path, dict]:
    run_dir = tmp_path / "r20261015-000000"
    (run_dir / "out").mkdir(parents=True)
    arts = []
    for rel, body in artifacts.items():
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        # 与 scripts/nm_test.sh 的生产者口径保持一致：**只有 .json 才认机读类型**，
        # 其余（.md/.txt/...）一律 report；统一执行入口的 run.log 算 log。
        stem = Path(rel).name.rsplit(".", 1)[0]
        ext = Path(rel).name.rsplit(".", 1)[-1].lower()
        if stem == "run":
            atype = "log"
        elif ext == "json":
            atype = stem if stem in ("correctness", "performance", "metrics") else "report"
        else:
            atype = "report"
        arts.append({"type": atype, "path": rel, "sha256": hashlib.sha256(body).hexdigest(),
                     "size_bytes": len(body)})
    man = {
        "schema": "opx-manifest/1",
        "run_id": "r20261015-000000",
        "exit_code": 0,
        "code_snapshot_sha256": SNAP,
        "spec_version_id": 1,
        "criteria_ref": CRITERIA,
        "suite_ref": SUITE,
        "image_digest": DIGEST,
        "image": "lfx-op-dev-910b:torch210-cann910",
        "device": "0",
        "lock_mode": "exclusive",
        "snapshot": {"files": 8, "truncated": False},
        "artifacts": arts,
    }
    if correctness is not None:
        cpath = run_dir / "out" / "correctness.json"
        cpath.write_text(json.dumps(correctness, ensure_ascii=False), encoding="utf-8")
        man["artifacts"] = [a for a in man["artifacts"] if a["path"] != "out/correctness.json"]
        man["artifacts"].append({"type": "correctness", "path": "out/correctness.json",
                                 "sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(),
                                 "size_bytes": cpath.stat().st_size})
    if metrics is not None:
        mpath = run_dir / "out" / "metrics.json"
        mpath.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
        man["artifacts"].append({"type": "metrics", "path": "out/metrics.json",
                                 "sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
                                 "size_bytes": mpath.stat().st_size})
    if extra_manifest:
        man.update(extra_manifest)
    if override_sha:
        man["artifacts"][0]["sha256"] = "0" * 64
    man_path = run_dir / "manifest.json"
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, man


def _run(db_session, task, man_path: Path | None, **kw) -> Run:
    row = Run(task_id=task.id, action=kw.pop("action", "test"), status=kw.pop("status", "ok"),
              manifest_path=str(man_path) if man_path else None,
              spec_version_id=kw.pop("spec_version_id", 1),
              code_snapshot_sha256=kw.pop("code_snapshot_sha256", SNAP),
              suite_ref=kw.pop("suite_ref", SUITE),
              criteria_ref=kw.pop("criteria_ref", CRITERIA),
              image_digest=kw.pop("image_digest", DIGEST),
              device_id=kw.pop("device_id", "0"),
              exit_code=kw.pop("exit_code", 0), **kw)
    db_session.add(row)
    db_session.commit()
    return row


def _codes(report: dict) -> set[str]:
    return {f["code"] for f in report["findings"]}


def test_pass_when_everything_aligned(db_session, seeded, tmp_path: Path):
    correctness = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA,
                   "suite_ref": SUITE,
                   "cases": [{"case_id": "c1", "group": "常规", "status": "PASS", "required": True}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    report = gate.evaluate(row, run_dir=run_dir)
    assert report["verdict"] == "pass", report["findings"]


def test_no_manifest(db_session, seeded):
    row = _run(db_session, seeded["task"], None)
    assert "E_NO_MANIFEST" in _codes(gate.evaluate(row))


def test_missing_artifact_and_sha_mismatch(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, override_sha=True)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    assert "E_ARTIFACT_SHA_MISMATCH" in _codes(gate.evaluate(row, run_dir=run_dir))
    (run_dir / "out" / "run.log").unlink()
    assert "E_ARTIFACT_MISSING" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_version_mismatch_between_row_and_manifest(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", code_snapshot_sha256="b" * 64)
    assert "E_SNAPSHOT_MISMATCH" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_missing_digest_is_fail(db_session, seeded, tmp_path: Path):
    """三硬规则：镜像 digest 必须留（靠 tag 判不了两次跑的是不是同一镜像）。"""
    man = {"image_digest": None}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, extra_manifest=man)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", image_digest=None)
    assert "E_IMAGE_DIGEST_MISSING" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_criteria_without_version_is_fail(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"},
                                extra_manifest={"criteria_ref": "官方档"})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", criteria_ref="官方档")
    assert "E_CRITERIA_VERSION_MISSING" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_required_case_not_run(db_session, seeded, tmp_path: Path):
    correctness = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA,
                   "suite_ref": SUITE,
                   "cases": [{"case_id": "c1", "status": "PASS", "required": True},
                             {"case_id": "c2", "status": "NOT_RUN", "required": True},
                             {"case_id": "c3", "status": "NOT_RUN", "required": False}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    assert "E_CASE_NOT_RUN" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_case_failed(db_session, seeded, tmp_path: Path):
    correctness = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA,
                   "suite_ref": SUITE,
                   "cases": [{"case_id": "c1", "status": "FAIL", "required": True}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    assert "E_CASE_FAILED" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_explicit_unsupported_is_legal_but_not_reported_as_pass(db_session, seeded, tmp_path: Path):
    correctness = {
        "spec_version_id": 1,
        "code_snapshot_sha256": SNAP,
        "criteria_ref": CRITERIA,
        "suite_ref": SUITE,
        "cases": [{
            "case_id": "ad_boundary",
            "status": "UNSUPPORTED",
            "required": True,
            "expected": "match_cpu_or_explicit_unsupported",
        }],
    }
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    report = gate.evaluate(row, run_dir=run_dir)
    assert report["verdict"] == "pass", report["findings"]
    assert "W_CASE_EXPLICIT_UNSUPPORTED" in _codes(report)
    assert "E_CASE_STATUS_INVALID" not in _codes(report)


def test_unsupported_without_explicit_contract_is_fail(db_session, seeded, tmp_path: Path):
    correctness = {
        "spec_version_id": 1,
        "code_snapshot_sha256": SNAP,
        "criteria_ref": CRITERIA,
        "suite_ref": SUITE,
        "cases": [{"case_id": "ordinary", "status": "UNSUPPORTED", "required": True}],
    }
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    report = gate.evaluate(row, run_dir=run_dir)
    assert report["verdict"] == "fail"
    assert "E_CASE_UNSUPPORTED_UNEXPECTED" in _codes(report)


def test_report_from_another_version(db_session, seeded, tmp_path: Path):
    """报告是上一版代码跑出来的 —— 「看起来过了」的经典骗法。"""
    correctness = {"spec_version_id": 1, "code_snapshot_sha256": "c" * 64, "criteria_ref": CRITERIA,
                   "suite_ref": SUITE,
                   "cases": [{"case_id": "c1", "status": "PASS", "required": True}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    assert "E_REPORT_VERSION_MISMATCH" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_shared_device_warns(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"},
                                extra_manifest={"lock_mode": "shared"})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="benchmark")
    report = gate.evaluate(row, run_dir=run_dir)
    assert "W_DEVICE_SHARED" in _codes(report)


def test_required_metric_failed(db_session, seeded, tmp_path: Path):
    metrics = {"metrics": [{"id": "aicore_util", "status": "failed", "required": True, "source": "msprof"}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, metrics=metrics)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="profile")
    assert "E_METRIC_REQUIRED_FAILED" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_nonzero_exit_is_fail(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"boom\n"}, extra_manifest={"exit_code": 2})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", exit_code=2, status="failed")
    assert "E_EXIT_NONZERO" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_write_report_lands_next_to_manifest(db_session, seeded, tmp_path: Path):
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    report = gate.evaluate(row, run_dir=run_dir, write=True)
    assert Path(report["written_to"]).is_file()


def test_required_artifact_types_per_action():
    assert gate.REQUIRED_ARTIFACTS["test"] == ("log", "correctness")
    assert gate.REQUIRED_ARTIFACTS["benchmark"] == ("log", "performance")
    assert gate.REQUIRED_ARTIFACTS["profile"] == ("log", "metrics")
    with pytest.raises(KeyError):
        gate.REQUIRED_ARTIFACTS["nope"]


def test_profile_without_metrics_report_is_fail(db_session, seeded, tmp_path: Path):
    """D9 已定（四类指标可采）后 profile 必须交 metrics.json —— 缺了不能只给 warn。"""
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="profile")
    codes = _codes(gate.evaluate(row, run_dir=run_dir))
    assert "E_METRICS_REPORT_MISSING" in codes
    assert "E_REQUIRED_ARTIFACT_MISSING" in codes


def test_metrics_report_from_another_version(db_session, seeded, tmp_path: Path):
    """性能报告也要能对齐版本，否则「报告不是这一版跑的」在性能侧没有防线。"""
    metrics = {"spec_version_id": 1, "code_snapshot_sha256": "d" * 64, "criteria_ref": CRITERIA,
               "suite_ref": SUITE,
               "metrics": [{"id": "op_time_us", "status": "measured", "required": True, "source": "msopprof"}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, metrics=metrics)
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="profile")
    assert "E_REPORT_VERSION_MISMATCH" in _codes(gate.evaluate(row, run_dir=run_dir))


def test_benchmark_performance_report_passes(db_session, seeded, tmp_path: Path):
    """benchmark 动作：performance.json 带对齐版本 + measured 指标 → pass。"""
    perf = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA, "suite_ref": SUITE,
            "metrics": [{"id": "op_time_us", "kind": "duration", "status": "measured", "required": True,
                         "source": "candidate exe (launch+sync, median)", "value": 8.1}]}
    run_dir, _ = _write_run_dir(tmp_path, {"out/run.log": b"ok\n",
                                           "out/performance.json": json.dumps(perf).encode()})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="benchmark")
    report = gate.evaluate(row, run_dir=run_dir)
    assert report["verdict"] == "pass", report["findings"]


def _append_artifact(man: dict, run_dir: Path, rel: str, atype: str) -> None:
    """按给定类型往 manifest 里补一条产物（模拟生产者的行为）。"""
    body = (run_dir / rel).read_bytes()
    man["artifacts"].append({"type": atype, "path": rel,
                             "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)})
    (run_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")


def test_correctness_md_same_type_must_not_hide_the_json(db_session, seeded, tmp_path: Path):
    """同类产物撞车的回归：`.md` 也被打成 correctness 类型、且排在 .json 后面时，
    门禁必须读到 `.json` —— 否则「报告根本没被解析」却什么都不报，用例 FAIL 会被判漏。

    2026-10-15 实测踩到：run#4（16/18 的那个版本）只报出 E_EXIT_NONZERO，
    逐用例 FAIL 因为 `{a["type"]: a}` 取到了 correctness.md 而整段被静默跳过。
    """
    correctness = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA,
                   "suite_ref": SUITE,
                   "cases": [{"case_id": "c1", "group": "常规", "status": "FAIL", "required": True}]}
    run_dir, man = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"}, correctness=correctness)
    (run_dir / "out" / "correctness.md").write_text("# 人看的伴生报告\n", encoding="utf-8")
    _append_artifact(man, run_dir, "out/correctness.md", "correctness")   # 排在 .json 之后
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    report = gate.evaluate(row, run_dir=run_dir)
    assert "E_CASE_FAILED" in _codes(report), report["findings"]


def test_artifact_paths_prefers_json(db_session, seeded, tmp_path: Path):
    """结果视图/门禁共用：同类多文件取 .json，且给绝对路径。"""
    run_dir, man = _write_run_dir(tmp_path, {"out/run.log": b"ok\n",
                                             "out/performance.json": b'{"metrics": []}'})
    row = _run(db_session, seeded["task"], run_dir / "manifest.json", action="benchmark")
    paths = gate.artifact_paths(row)
    assert paths["performance"].name == "performance.json"
    assert paths["performance"].is_file()
    assert "log" in paths


def test_non_json_report_is_an_error_not_a_silent_skip(db_session, seeded, tmp_path: Path):
    """某类型只有非机读文件时必须报错：静默跳过等于门禁放行。"""
    run_dir, man = _write_run_dir(tmp_path, {"out/run.log": b"ok\n"})
    (run_dir / "out" / "correctness.md").write_text("# 只有伴生报告\n", encoding="utf-8")
    _append_artifact(man, run_dir, "out/correctness.md", "correctness")
    row = _run(db_session, seeded["task"], run_dir / "manifest.json")
    assert "E_REPORT_UNREADABLE" in _codes(gate.evaluate(row, run_dir=run_dir))
