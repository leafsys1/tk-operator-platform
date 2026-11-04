"""测试管理（P3 读侧）：用例矩阵 + 逐例结果合并 + 失败点。"""

from __future__ import annotations

import json
from pathlib import Path

from app.models import Artifact, Project, Run, Task


def _seed(Session, workspace: Path) -> tuple[int, int]:
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add",
                 criteria_ref="ops-precision-standard@1")
        db.add(t)
        db.commit()
        return p.id, t.id


def _write_matrix(workspace: Path) -> None:
    d = workspace / "cases" / "elementwise_add"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cases.json").write_text(json.dumps({
        "schema": "opx-cases/1", "op": "elementwise_add",
        "cases": [
            {"case_id": "add_norm_001", "group": "常规", "required": True, "n_a": 4096, "n_b": 4096,
             "dtype": "float32", "seed": 1, "expected": "match", "note": "典型"},
            {"case_id": "add_bad_001", "group": "异常", "required": True, "n_a": 0, "n_b": 0,
             "dtype": "float32", "seed": 2, "expected": "reject", "note": "空输入"},
        ],
    }, ensure_ascii=False), encoding="utf-8")


def _run_dir(tmp_path: Path, workspace: Path, name: str, correctness: dict | None) -> Path:
    rd = Path(str(workspace)) / "runs" / name
    (rd / "out").mkdir(parents=True, exist_ok=True)
    arts = []
    if correctness is not None:
        (rd / "out" / "correctness.json").write_text(json.dumps(correctness, ensure_ascii=False), encoding="utf-8")
        arts.append({"type": "correctness", "path": "out/correctness.json"})
    (rd / "manifest.json").write_text(json.dumps({"artifacts": arts}), encoding="utf-8")
    return rd


def test_matrix_only_when_no_runs(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    _write_matrix(workspace)

    body = c.get(f"/api/tasks/{tid}/cases").json()
    assert body["op_name"] == "elementwise_add"
    assert body["case_source"] == "cases/elementwise_add/cases.json"
    assert body["total_cases"] == 2 and body["required_cases"] == 2
    assert body["runs"] == [] and body["failures"] == []
    c1 = next(x for x in body["cases"] if x["case_id"] == "add_norm_001")
    assert c1["shape"] == [4096] and c1["expected"] == "match" and c1["latest_status"] is None
    assert body["case_groups"] == ["异常", "常规"] or set(body["case_groups"]) == {"常规", "异常"}


def test_merges_results_and_collects_failures(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    _write_matrix(workspace)

    correctness = {
        "schema": "opx-correctness/1", "op": "elementwise_add",
        "case_source": "cases/elementwise_add/cases.json", "case_source_sha256": "abc123",
        "summary": {"PASS": 1, "FAIL": 1, "ERROR": 0, "INVALID": 0, "NOT_RUN": 0, "TOTAL": 2},
        "cases": [
            {"case_id": "add_norm_001", "group": "常规", "required": True, "shape": [4096], "shape_b": [4096],
             "dtype": "float32", "expected": "match", "status": "PASS", "max_abs_err": 0.0, "matched_ratio": 1.0},
            {"case_id": "add_bad_001", "group": "异常", "required": True, "shape": [0], "shape_b": [0],
             "dtype": "float32", "expected": "match", "status": "FAIL", "max_abs_err": 0.5, "matched_ratio": 0.1},
        ],
    }
    rd = _run_dir(Path(str(workspace)), workspace, "r-test-1", correctness)
    with Session() as db:
        run = Run(task_id=tid, action="test", status="ok", manifest_path=str(rd / "manifest.json"),
                  code_snapshot_sha256="deadbeef", suite_ref="cases@x", criteria_ref="ops-precision-standard@1")
        db.add(run)
        db.flush()
        db.add(Artifact(run_id=run.id, type="correctness", path="out/correctness.json"))
        db.add(Artifact(run_id=run.id, type="report", path="out/correctness.md"))
        db.commit()
        rid = run.id

    body = c.get(f"/api/tasks/{tid}/cases").json()
    assert body["case_source_sha256"] == "abc123"
    assert len(body["runs"]) == 1 and body["runs"][0]["report_available"] is True
    assert body["runs"][0]["run_id"] == rid
    assert body["runs"][0]["summary"]["TOTAL"] == 2
    assert body["runs"][0]["downloads"], "应暴露 correctness 产物可下载"

    by_id = {x["case_id"]: x for x in body["cases"]}
    assert by_id["add_bad_001"]["results"][str(rid)] == "FAIL"
    assert by_id["add_norm_001"]["latest_status"] == "PASS"

    assert len(body["failures"]) == 1
    f = body["failures"][0]
    assert f["case_id"] == "add_bad_001" and f["run_id"] == rid
    assert f["max_abs_err"] == 0.5 and f["status"] == "FAIL"


def test_missing_report_flagged_not_crash(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed(Session, workspace)
    _write_matrix(workspace)
    rd = _run_dir(Path(str(workspace)), workspace, "r-noreport", None)
    with Session() as db:
        db.add(Run(task_id=tid, action="test", status="failed", manifest_path=str(rd / "manifest.json")))
        db.commit()

    body = c.get(f"/api/tasks/{tid}/cases").json()
    assert body["runs"][0]["report_available"] is False
    assert body["failures"] == []
    # 矩阵仍来自工作区
    assert body["total_cases"] == 2
