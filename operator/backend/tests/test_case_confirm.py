"""用例组确认（契约 §3.3）+ 用例集哈希（契约 §5.1，必须与 scripts/nm_test.sh 一致）。

覆盖点：
- `confirm` 成功：校验两个文件 sha256 → 写 `tasks.cases_confirmed_*` + StageEvent；
- 哈希不匹配 / 文件缺失 → 409，且确认状态不被写入；
- `confirmation` 的 in_sync：用例改动后自动变 False，cases/ 删掉后 current=None；
- `cases_suite_sha256` 的遍历顺序与 `scripts/nm_test.sh` 里的实现逐字节一致（直接抽脚本片段跑）；
- 非法入参 / 任务不存在 4xx。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models import Project, StageEvent, Task
from app.services import case_confirm

REPO = Path(__file__).resolve().parents[2]
NM_TEST = REPO / "scripts" / "nm_test.sh"
#: nm_test.sh 里「用例集 ref」那段 Python 的锚点（脚本改动时这里会先炸，提醒同步）
SUITE_SNIPPET_MARK = 'OPX_SUITE_REF="cases@$(/usr/bin/python3 - "$OPX_WORKSPACE/cases/$OPX_OP"'


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _seed_task(Session, workspace: Path, *, op: str = "add", title: str = "elementwise_add") -> int:
    with Session() as db:
        project = Project(name=f"p-{title}", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=title, op_name=op)
        db.add(task)
        db.commit()
        return task.id


def _write_cases(workspace: Path, op: str = "add") -> tuple[Path, Path]:
    """落 cases/<op>/cases.json + benchmark.json（模拟 agent 确认前写好的文件）。"""
    base = workspace / "cases" / op
    base.mkdir(parents=True, exist_ok=True)
    cases_json = base / "cases.json"
    benchmark_json = base / "benchmark.json"
    cases_json.write_text(json.dumps(
        {"case_groups": ["常规", "边界"], "cases": [{"case_id": "add_1", "group": "常规"}]},
        ensure_ascii=False), encoding="utf-8")
    benchmark_json.write_text(json.dumps(
        {"primary_case": "bench_1m", "cases": [{"case_id": "bench_1m", "n": 1024}]},
        ensure_ascii=False), encoding="utf-8")
    return cases_json, benchmark_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirm_body(cases_json: Path, benchmark_json: Path, **overrides) -> dict:
    body = {"cases_sha256": _sha(cases_json), "benchmark_sha256": _sha(benchmark_json),
            "by": "chat:job#7"}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 哈希算法（契约 §5.1）
# ---------------------------------------------------------------------------

def _nm_test_suite_snippet() -> str:
    text = NM_TEST.read_text(encoding="utf-8")
    start = text.index(SUITE_SNIPPET_MARK)
    body = text[start:].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert "hashlib" in body and "os.walk" in body
    return body


def test_suite_hash_is_whole_cases_dir_in_nm_test_order(tmp_path: Path):
    """手算期望值：先根目录文件，再按 dirnames.sort() 深度优先进子目录。"""
    cases = tmp_path / "cases"
    (cases / "add").mkdir(parents=True)
    (cases / "zz").mkdir()
    files = {
        "top.json": b'{"top": 1}',
        "add/benchmark.json": b'{"bench": 2}',
        "add/cases.json": b'{"cases": 3}',
        "zz/x.json": b'{"x": 4}',
    }
    for rel, data in files.items():
        (cases / rel).write_bytes(data)

    expected = hashlib.sha256()
    # os.walk 深度优先：根目录文件在前（filenames 排序），子目录按 dirnames.sort() 依次进入。
    for rel in ("top.json", "add/benchmark.json", "add/cases.json", "zz/x.json"):
        expected.update(rel.encode())
        expected.update(b"\0")
        expected.update(files[rel])
        expected.update(b"\0")

    assert case_confirm.cases_suite_sha256(tmp_path) == expected.hexdigest()
    assert case_confirm.cases_ref(tmp_path) == "cases@" + expected.hexdigest()[:16]


def test_suite_hash_matches_nm_test_script(tmp_path: Path):
    """与 scripts/nm_test.sh 的用例集 ref 片段跑同一目录，结果必须逐字节一致。"""
    cases = tmp_path / "cases"
    (cases / "elementwise_add").mkdir(parents=True)
    (cases / "elementwise_add" / "cases.json").write_text('{"cases": []}', encoding="utf-8")
    (cases / "elementwise_add" / "benchmark.json").write_text('{"cases": []}', encoding="utf-8")
    (cases / "README.md").write_text("suite\n", encoding="utf-8")

    op_root = cases / "elementwise_add"
    proc = subprocess.run([sys.executable, "-c", _nm_test_suite_snippet(), str(op_root)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    ref_hex16 = proc.stdout.strip()
    full = case_confirm.cases_suite_sha256(tmp_path, "elementwise_add")
    assert full is not None and len(full) == 64
    assert full[:16] == ref_hex16, "后端哈希与 nm_test.sh 的 suite_ref 漂移了"
    assert case_confirm.cases_ref(tmp_path, "elementwise_add") == f"cases@{ref_hex16}"


def test_operator_suite_hash_ignores_other_operator(tmp_path: Path):
    _write_cases(tmp_path, "op_a")
    _write_cases(tmp_path, "op_b")
    before = case_confirm.cases_suite_sha256(tmp_path, "op_a")
    (tmp_path / "cases/op_b/cases.json").write_text('{"changed": true}', encoding="utf-8")
    assert case_confirm.cases_suite_sha256(tmp_path, "op_a") == before
    (tmp_path / "cases/op_a/cases.json").write_text('{"changed": true}', encoding="utf-8")
    assert case_confirm.cases_suite_sha256(tmp_path, "op_a") != before


def test_suite_hash_none_without_cases_dir(tmp_path: Path):
    assert case_confirm.cases_suite_sha256(tmp_path) is None
    assert case_confirm.cases_ref(tmp_path) is None


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello world")
    assert case_confirm.sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


# ---------------------------------------------------------------------------
# confirm / confirmation
# ---------------------------------------------------------------------------

def test_confirm_success_writes_task_and_stage_event(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    body = _confirm_body(cases_json, benchmark_json)

    resp = c.post(f"/api/tasks/{task_id}/cases/confirm", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    suite_sha = case_confirm.cases_suite_sha256(workspace, "add")
    assert suite_sha is not None
    assert data["confirmed_at"]
    assert data["cases_sha256"] == body["cases_sha256"]
    assert data["cases_suite_sha256"] == suite_sha
    assert data["suite_ref"] == f"cases@{suite_sha[:16]}"

    with Session() as db:
        task = db.get(Task, task_id)
        assert task.cases_confirmed_at is not None
        assert task.cases_confirmed_sha256 == suite_sha
        assert task.cases_confirmed_by == "chat:job#7"
        events = list(db.scalars(select(StageEvent).where(StageEvent.task_id == task_id)))
        confirmed = [e for e in events if str(e.reason or "").startswith("用例组确认")]
        assert len(confirmed) == 1 and confirmed[0].actor == "chat:job#7"

    confirmation = c.get(f"/api/tasks/{task_id}/cases/confirmation").json()
    assert confirmation["confirmed_at"]
    assert confirmation["confirmed_sha256"] == suite_sha
    assert confirmation["current_suite_sha256"] == suite_sha
    assert confirmation["in_sync"] is True


def test_confirm_default_by_is_api(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    body = _confirm_body(cases_json, benchmark_json)
    body.pop("by")
    assert c.post(f"/api/tasks/{task_id}/cases/confirm", json=body).status_code == 200
    with Session() as db:
        assert db.get(Task, task_id).cases_confirmed_by == "api"


def test_confirmation_in_sync_becomes_false_after_cases_change(client, workspace: Path):
    """用例集变了 ≠ 已确认的那一版：旧证据失效（设计 §4.2），in_sync 必须变 False。"""
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    assert c.post(f"/api/tasks/{task_id}/cases/confirm",
                  json=_confirm_body(cases_json, benchmark_json)).status_code == 200
    confirmed = c.get(f"/api/tasks/{task_id}/cases/confirmation").json()
    assert confirmed["in_sync"] is True

    cases_json.write_text(json.dumps({"cases": [{"case_id": "add_2"}]}, ensure_ascii=False),
                          encoding="utf-8")
    changed = c.get(f"/api/tasks/{task_id}/cases/confirmation").json()
    assert changed["in_sync"] is False
    assert changed["current_suite_sha256"] != changed["confirmed_sha256"]
    assert changed["confirmed_sha256"] == confirmed["confirmed_sha256"]  # 确认记录没被改写


def test_confirmation_in_sync_false_without_cases_dir(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    import shutil

    shutil.rmtree(workspace / "cases")
    confirmation = c.get(f"/api/tasks/{task_id}/cases/confirmation").json()
    assert confirmation["confirmed_at"] is None
    assert confirmation["confirmed_sha256"] is None
    assert confirmation["current_suite_sha256"] is None
    assert confirmation["in_sync"] is False


# ---------------------------------------------------------------------------
# 校验失败 → 409，且不写确认状态
# ---------------------------------------------------------------------------

def _assert_not_confirmed(c, Session, task_id: int) -> None:
    with Session() as db:
        task = db.get(Task, task_id)
        assert task.cases_confirmed_at is None
        assert task.cases_confirmed_sha256 is None
        assert db.scalars(select(StageEvent).where(StageEvent.task_id == task_id)).first() is None
    assert c.get(f"/api/tasks/{task_id}/cases/confirmation").json()["in_sync"] is False


def test_confirm_hash_mismatch_is_409(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    body = _confirm_body(cases_json, benchmark_json, benchmark_sha256="f" * 64)

    resp = c.post(f"/api/tasks/{task_id}/cases/confirm", json=body)
    assert resp.status_code == 409
    assert "benchmark.json" in resp.json()["detail"]
    _assert_not_confirmed(c, Session, task_id)


def test_confirm_missing_benchmark_is_409(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    expected = _sha(benchmark_json)
    benchmark_json.unlink()
    cases_json_hash = _sha(cases_json)

    resp = c.post(f"/api/tasks/{task_id}/cases/confirm",
                  json={"cases_sha256": cases_json_hash, "benchmark_sha256": expected})
    assert resp.status_code == 409
    assert "benchmark.json" in resp.json()["detail"]
    _assert_not_confirmed(c, Session, task_id)


def test_confirm_missing_op_name_is_409(client, workspace: Path):
    c, Session = client
    with Session() as db:
        project = Project(name="p-noop", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="noop")  # op_name 为空
        db.add(task)
        db.commit()
        task_id = task.id
    _write_cases(workspace)
    resp = c.post(f"/api/tasks/{task_id}/cases/confirm",
                  json={"cases_sha256": "0" * 64, "benchmark_sha256": "0" * 64})
    assert resp.status_code == 409
    assert "op_name" in resp.json()["detail"]


@pytest.mark.parametrize("field,value", [
    ("cases_sha256", "not-a-hash"),
    ("benchmark_sha256", "1234"),
])
def test_confirm_rejects_non_hex_sha_is_400(client, workspace: Path, field: str, value: str):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    cases_json, benchmark_json = _write_cases(workspace)
    body = _confirm_body(cases_json, benchmark_json)
    body[field] = value
    resp = c.post(f"/api/tasks/{task_id}/cases/confirm", json=body)
    assert resp.status_code == 400
    assert field in resp.json()["detail"]
    _assert_not_confirmed(c, Session, task_id)


def test_confirm_task_not_found(client):
    c, _ = client
    resp = c.post("/api/tasks/999999/cases/confirm",
                  json={"cases_sha256": "0" * 64, "benchmark_sha256": "0" * 64})
    assert resp.status_code == 404
