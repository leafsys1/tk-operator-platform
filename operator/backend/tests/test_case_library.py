"""案例库（M2）：外置 bundle 索引 + 一键发起（复制 cases/workspace/ops）。"""

from __future__ import annotations

import json
from pathlib import Path

from app.models import Project, Task


def _lib(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    case = lib / "demo" / "cases" / "relu"
    (case / "workspace" / "ops").mkdir(parents=True)
    (case / "ops").mkdir(parents=True)
    (case / "cases.json").write_text(json.dumps({"cases": [{"case_id": "c1"}]}), encoding="utf-8")
    (case / "workspace" / "README-case.md").write_text("case workspace", encoding="utf-8")
    (case / "ops" / "impl.c").write_text("// impl", encoding="utf-8")
    (lib / "index.json").write_text(json.dumps({
        "schema": "opx-case-library-index/1",
        "dest": str(lib),
        "bundles": [{"name": "demo", "version": "1", "cases": [
            {"op": "relu", "scenario": "激活", "path": "cases/relu"}
        ]}],
    }), encoding="utf-8")
    return lib


def _project(client, workspace: Path) -> int:
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.commit()
        return p.id


def test_case_library_list_and_start(client, workspace: Path, tmp_path: Path, monkeypatch):
    from app.config import settings

    lib = _lib(tmp_path)
    monkeypatch.setattr(settings, "case_library", str(lib))
    c, Session = client
    pid = _project(client, workspace)

    body = c.get("/api/case-library").json()
    assert body["imported"] is True
    assert body["bundles"][0]["name"] == "demo"
    case = body["bundles"][0]["cases"][0]
    assert case["op"] == "relu" and case["has_workspace"] and case["has_ops"]

    r = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r.status_code == 201, r.text
    tid = r.json()["task_id"]
    with Session() as db:
        task = db.get(Task, tid)
        assert task.op_name == "relu" and task.input_form == "案例库"
    assert (workspace / "cases" / "relu" / "cases.json").is_file()
    assert (workspace / "README-case.md").read_text() == "case workspace"
    assert (workspace / "ops" / "relu" / "impl.c").read_text() == "// impl"

    # 目标已存在 → 409，不覆盖
    r2 = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r2.status_code == 409
    assert "已存在" in r2.json()["detail"]


def test_case_library_missing_case(client, workspace: Path, tmp_path: Path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "case_library", str(_lib(tmp_path)))
    _project(client, workspace)
    c, _Session = client
    r = c.post("/api/case-library/start", json={"bundle": "demo", "op": "nope"})
    assert r.status_code == 409
