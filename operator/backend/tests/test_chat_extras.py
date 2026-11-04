"""P0-2 增强：附件 / 每轮改动审阅回退 / 命令执行 / 远端取消。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.models import Conversation, Message, Project, Task
from app.services import agent_runner, conversations as svc, optimize, remote_cancel


def _seed(client, workspace: Path) -> tuple[int, int]:
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.commit()
        tid = t.id
    conv = c.post("/api/conversations", json={"task_id": tid, "mode": "task"}).json()
    return tid, conv["id"]


def test_attachment_upload_and_prompt(client, workspace: Path, monkeypatch):
    c, _ = client
    tid, cid = _seed(client, workspace)

    r = c.post(f"/api/conversations/{cid}/attachments",
               files={"file": ("note.txt", b"hello attachment", "text/plain")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["filename"] == "note.txt" and body["size_bytes"] == 16
    assert (workspace / body["path"]).read_text() == "hello attachment"

    detail = c.get(f"/api/conversations/{cid}").json()
    assert [a["filename"] for a in detail["attachments"]] == ["note.txt"]

    captured: dict = {}
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: captured.update(kw))
    r2 = c.post(f"/api/conversations/{cid}/messages", json={"text": "看看附件"})
    assert r2.status_code == 201
    # prompt 里带附件相对路径
    from app.models import AgentJob
    _, Session = client
    with Session() as db:
        job = db.get(AgentJob, r2.json()["job_id"])
        assert ".opx-chat/conv-" in job.prompt and "note.txt" in job.prompt


def test_message_revert_restores_files(client, workspace: Path):
    c, Session = client
    tid, cid = _seed(client, workspace)
    target = workspace / "src" / "add.cpp"
    target.write_text("old\n", encoding="utf-8")
    before = optimize.capture(workspace)
    target.write_text("new\n", encoding="utf-8")
    after = optimize.capture(workspace)
    payload = svc._changes_payload(before, after)
    assert payload and payload["changed_files"]

    with Session() as db:
        conv = db.get(Conversation, cid)
        msg = Message(conversation_id=conv.id, role="assistant", text="改了",
                      changes_json=json.dumps(payload))
        db.add(msg)
        db.commit()
        mid = msg.id

    r = c.post(f"/api/conversations/{cid}/messages/{mid}/revert")
    assert r.status_code == 200, r.text
    assert r.json()["files"]["src/add.cpp"] == "restored"
    assert target.read_text() == "old\n"


def test_shell_runs_in_workspace(client, workspace: Path):
    c, _ = client
    tid, _cid = _seed(client, workspace)
    r = c.post(f"/api/tasks/{tid}/shell", json={"command": "pwd && echo hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exit_code"] == 0
    assert str(workspace) in body["stdout"] and "hi" in body["stdout"]


def test_remote_cancel_builds_docker_command(seeded, db_session, monkeypatch):
    task = seeded["task"]
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(remote_cancel.subprocess, "run", fake_run)
    out = remote_cancel.cancel_container(db_session, task, "rTEST-9")
    assert out["ok"] is True
    assert "docker rm -f opx-rTEST-9" in seen[0][-1]
    assert seen[0][0] == "ssh"
