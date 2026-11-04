"""工作区只读浏览 + 产物文本预览（「实现与接口」分区）。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from app.models import Artifact, Project, Run, Task


def _seed(client, workspace: Path) -> int:
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.commit()
        return t.id


def test_workspace_list_and_file(client, workspace: Path):
    c, _ = client
    tid = _seed(client, workspace)
    (workspace / ".dsh" / "skills").mkdir(parents=True)
    (workspace / ".dsh" / "skills" / "x.md").write_text("hidden")
    (workspace / "runs" / "r1").mkdir(parents=True)
    (workspace / "runs" / "r1" / "out.log").write_text("hidden")

    body = c.get(f"/api/tasks/{tid}/workspace").json()
    paths = {f["path"] for f in body["files"]}
    assert "build.sh" in paths and "src/add.cpp" in paths
    assert not any(p.startswith(".dsh/") or p.startswith("runs/") for p in paths)

    f = c.get(f"/api/tasks/{tid}/workspace/file", params={"path": "src/add.cpp"}).json()
    assert f["binary"] is False and "kernel" in f["text"]

    assert c.get(f"/api/tasks/{tid}/workspace/file", params={"path": "../secret"}).status_code == 400
    assert c.get(f"/api/tasks/{tid}/workspace/file", params={"path": "nope.txt"}).status_code == 400


def test_workspace_diff_non_repo(client, workspace: Path):
    c, _ = client
    tid = _seed(client, workspace)
    body = c.get(f"/api/tasks/{tid}/workspace/diff").json()
    assert body["is_repo"] is False


def test_artifact_text_preview(client, workspace: Path):
    c, Session = client
    tid = _seed(client, workspace)
    rd = workspace / "runs" / "r9"
    rd.mkdir(parents=True)
    (rd / "out").mkdir()
    log = rd / "out" / "run.log"
    log.write_text("BUILD_OK sha256=abc\n", encoding="utf-8")
    man = rd / "manifest.json"
    man.write_text(json.dumps({"run_id": "r9"}), encoding="utf-8")
    with Session() as db:
        run = Run(task_id=tid, action="build", status="ok", manifest_path=str(man))
        db.add(run)
        db.flush()
        art = Artifact(run_id=run.id, type="log", path="out/run.log", size_bytes=log.stat().st_size)
        db.add(art)
        db.commit()
        aid = art.id

    body = c.get(f"/api/artifacts/{aid}/text").json()
    assert body["binary"] is False and "BUILD_OK" in body["text"]
    assert c.get("/api/artifacts/99999/text").status_code == 404


def test_list_files_prunes_skip_dirs_and_stops_at_limit(client, workspace: Path, monkeypatch):
    """遍历时就剪掉 .git/node_modules（不进入），到达 limit 立即停（审查 #11）。"""
    from app.services import workspace_view

    c, _ = client
    tid = _seed(client, workspace)
    (workspace / ".git" / "objects").mkdir(parents=True)
    (workspace / ".git" / "objects" / "big").write_text("x", encoding="utf-8")
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")

    visited: list[str] = []
    real_walk = workspace_view.os.walk

    def spy(path, *args, **kwargs):
        visited.append(Path(path).name)
        return real_walk(path, *args, **kwargs)

    monkeypatch.setattr(workspace_view.os, "walk", spy)
    body = workspace_view.list_files(workspace, limit=2)
    assert body["truncated"] is True and body["count"] == 2
    assert ".git" not in visited and "node_modules" not in visited
    assert not any(p.startswith((".git/", "node_modules/")) for p in
                   [row["path"] for row in body["files"]])

    full = workspace_view.list_files(workspace)
    paths = {row["path"] for row in full["files"]}
    assert "build.sh" in paths and "src/add.cpp" in paths
    assert not any(p.startswith((".git/", "node_modules/", "runs/")) for p in paths)


def test_workspace_diff_git_timeout_is_readable_error(client, workspace: Path, monkeypatch):
    """git 子进程超时不能 500：返回可读 error，页面仍能渲染（审查 #11）。"""
    c, _ = client
    tid = _seed(client, workspace)
    from app.services import workspace_view

    def fake_run(args, **_kw):
        if "--show-toplevel" in args:
            return SimpleNamespace(returncode=0, stdout=f"{workspace}\n", stderr="")
        if "status" in args:
            raise subprocess.TimeoutExpired(args, 15)
        return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

    monkeypatch.setattr(workspace_view.subprocess, "run", fake_run)
    r = c.get(f"/api/tasks/{tid}/workspace/diff")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_repo"] is True and "TimeoutExpired" in body["error"]
    assert body["dirty"] is False and body["diff"] == ""
