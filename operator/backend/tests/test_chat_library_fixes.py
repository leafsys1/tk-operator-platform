"""对话 / 案例库审查修复的回归测试（`runs/review/pages-chat-library.md`）。

覆盖：
- 发送并发互斥、`开发/优化` 作业在跑时 409、`plain_stream=True`；
- 失败/取消消息可见 + cache token 计入；
- 删除会话清附件行/文件、running 时 409、id 复用不继承旧附件；
- 附件分块上传超限 413、每会话数量上限、附件删除接口；
- 轻对话附件 prompt 路径；
- 回退冲突 409 且不改文件 / 无冲突正常回退 / 旧记录兼容；
- 案例一键发起冲突无半截、可重试、`target_chip` 落 Task；
- 导入脚本 manifest.name 穿越、`--force` 不删库外、zip 限制、index 原子写；
- 畸形 cases.json / correctness.json → 200，`op_name` 越界读不到库外。

全部使用临时库/临时目录；不跑 dsh、不调 LLM。
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.models import AgentJob, Artifact, Attachment, Conversation, Message, Project, Run, Task
from app.services import agent_runner, case_library, conversations as svc, optimize

# ---------------------------------------------------------------- 通用夹具


def _seed_project_task(Session, workspace: Path, *, op: str = "elementwise_add",
                       title: str = "elementwise_add") -> tuple[int, int]:
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title=title, op_name=op)
        db.add(t)
        db.commit()
        return p.id, t.id


def _conv_session(db_session, monkeypatch):
    """让 `_finish` 用测试库的 sessionmaker（同 test_conversations 的做法）。"""
    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(svc, "SessionLocal", Session)
    return Session


# ---------------------------------------------------------------- 1. 发送并发 / 契约


def test_concurrent_send_starts_only_one_job(db_session, seeded, monkeypatch):
    task = seeded["task"]
    project = seeded["project"]
    conv = svc.create(db_session, task.id)
    Session = sessionmaker(bind=db_session.bind)
    # 线程里只用普通标量：ORM 对象属于主线程的 session，跨线程访问会触发 expire/refresh
    conv_id, task_id, project_id = conv.id, task.id, project.id

    started: list[int] = []
    monkeypatch.setattr(agent_runner, "start", lambda job_id, *a, **kw: started.append(job_id))
    real_capture = svc.optimize.capture

    def slow_capture(cwd):  # 放大「检查 running」与「插入 job」之间的窗口
        import time

        time.sleep(0.1)
        return real_capture(cwd)

    monkeypatch.setattr(svc.optimize, "capture", slow_capture)

    barrier = threading.Barrier(2)
    ok: list[int] = []
    errors: list[str] = []

    def worker() -> None:
        with Session() as db:
            fake_conv = SimpleNamespace(id=conv_id, task_id=task_id, project_id=project_id,
                                        mode="task", dsh_session_id=None, last_active_at=None)
            barrier.wait(timeout=5)
            try:
                job = svc.send(db, fake_conv, "并发发送")
                ok.append(job.id)
            except ValueError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001 —— 别让线程静默死掉把另一个卡在 barrier 上
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), f"并发发送线程未结束：{errors}"

    assert len(ok) == 1, f"并发发送应只成功一个：ok={ok} errors={errors}"
    assert len(errors) == 1 and "已有一个" in errors[0]
    assert len(started) == 1
    with Session() as db:
        running = list(db.scalars(select(AgentJob.id).where(
            AgentJob.task_id == task_id, AgentJob.kind == "对话", AgentJob.status == "running")))
        assert len(running) == 1


@pytest.mark.parametrize("kind", ["开发", "优化"])
def test_send_rejected_when_workspace_job_running(client, workspace: Path, kind: str):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    with Session() as db:
        db.add(AgentJob(task_id=tid, kind=kind, status="running"))
        db.commit()
    r = c.post(f"/api/conversations/{conv['id']}/messages", json={"text": "在吗"})
    assert r.status_code == 409, r.text
    assert kind in r.json()["detail"]


# ---------------------------------------------------------------- 2. 失败/中断可见 + token 口径


def test_finish_marks_interruption_and_counts_cache_tokens(db_session, seeded, monkeypatch):
    _conv_session(db_session, monkeypatch)
    task = seeded["task"]

    conv = svc.create(db_session, task.id)
    state = SimpleNamespace(prose="半截回答：我先说前一半……", final="", session_id="s-1",
                            usage={"inputTokens": 10, "outputTokens": 5,
                                   "cacheReadTokens": 7, "cacheWriteTokens": 3},
                            error="智能体调用超时（1800s），已终止 dsh 进程")
    svc._finish(999, conv.id, state)

    db_session.expire_all()
    row = db_session.get(Conversation, conv.id)
    assert row.tokens == 10 + 5 + 7 + 3
    msg = db_session.scalar(select(Message).where(Message.conversation_id == conv.id)
                            .order_by(Message.id.desc()).limit(1))
    assert msg.text.startswith("半截回答：我先说前一半……")
    assert "本轮中断" in msg.text and "超时" in msg.text


def test_finish_cancelled_without_prose_keeps_note(db_session, seeded, monkeypatch):
    _conv_session(db_session, monkeypatch)
    conv = svc.create(db_session, seeded["task"].id)
    state = SimpleNamespace(prose="", final="", session_id=None, usage={},
                            error=None, cancelled=True)
    svc._finish(1000, conv.id, state)
    db_session.expire_all()
    msg = db_session.scalar(select(Message).where(Message.conversation_id == conv.id)
                            .order_by(Message.id.desc()).limit(1))
    assert msg.text == "（本轮中断：已取消）"


# ---------------------------------------------------------------- 3. 删除会话 / 附件清理


def test_delete_conversation_clears_attachments_and_files(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    cid = conv["id"]

    up = c.post(f"/api/conversations/{cid}/attachments",
                files={"file": ("a.txt", b"attachment", "text/plain")})
    assert up.status_code == 201, up.text
    att_path = workspace / up.json()["path"]
    assert att_path.is_file()
    conv_dir = workspace / ".opx-chat" / f"conv-{cid}"
    assert conv_dir.is_dir()

    assert c.delete(f"/api/conversations/{cid}").status_code == 204
    with Session() as db:
        assert db.scalar(select(func.count(Attachment.id))) == 0
        assert db.scalar(select(func.count(Message.id))) == 0
        assert db.get(Conversation, cid) is None
    assert not att_path.exists()
    assert not conv_dir.exists()

    # 会话 id 被复用时不能继承旧附件
    conv2 = c.post("/api/conversations", json={"task_id": tid}).json()
    assert conv2["id"] == cid, "SQLite 未开 AUTOINCREMENT 时会复用最大 id（本测试前提）"
    detail = c.get(f"/api/conversations/{cid}").json()
    assert detail["attachments"] == []


def test_delete_conversation_with_running_job_409(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    with Session() as db:
        job = AgentJob(task_id=tid, conversation_id=conv["id"], kind="对话", status="running")
        db.add(job)
        db.commit()
        jid = job.id
    r = c.delete(f"/api/conversations/{conv['id']}")
    assert r.status_code == 409
    with Session() as db:
        assert db.get(AgentJob, jid).status == "running"
        assert db.get(Conversation, conv["id"]) is not None


def test_delete_conversation_allows_stale_running_job(client, workspace: Path):
    """DB 标 running 但进程已崩且超时 → 视为僵尸，可删并把 job 标 interrupted。"""
    from datetime import datetime, timedelta, timezone

    from app.config import settings

    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    started = datetime.now(timezone.utc) - timedelta(seconds=int(settings.agent_timeout) + 60)
    with Session() as db:
        job = AgentJob(task_id=tid, conversation_id=conv["id"], kind="对话", status="running",
                       started_at=started)
        db.add(job)
        db.commit()
        jid = job.id

    r = c.delete(f"/api/conversations/{conv['id']}")
    assert r.status_code == 204, r.text
    with Session() as db:
        assert db.get(Conversation, conv["id"]) is None
        assert db.get(AgentJob, jid).status == "interrupted"


def test_attachment_delete_route_and_count_limit(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    cid = conv["id"]

    up = c.post(f"/api/conversations/{cid}/attachments",
                files={"file": ("a.txt", b"hi", "text/plain")})
    assert up.status_code == 201
    aid = up.json()["id"]
    att_path = workspace / up.json()["path"]
    assert att_path.is_file()

    assert c.delete(f"/api/conversations/{cid}/attachments/{aid}").status_code == 204
    assert not att_path.exists()
    with Session() as db:
        assert db.get(Attachment, aid) is None
    # 重复删除 → 404（已不存在）
    assert c.delete(f"/api/conversations/{cid}/attachments/{aid}").status_code == 404

    # 每会话附件数上限
    with Session() as db:
        row = db.get(Conversation, cid)
        for i in range(svc.ATTACH_MAX_COUNT):
            svc.save_attachment(db, row, workspace, f"f{i}.txt", b"x")
        with pytest.raises(ValueError, match="最多"):
            svc.save_attachment(db, row, workspace, "over.txt", b"x")


def test_upload_over_limit_returns_413(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()
    big = b"x" * (svc.ATTACH_MAX_BYTES + 1)
    r = c.post(f"/api/conversations/{conv['id']}/attachments",
               files={"file": ("big.bin", big, "application/octet-stream")})
    assert r.status_code == 413, r.text


# ---------------------------------------------------------------- 4. 轻对话附件路径


def test_light_prompt_attachment_path_matches_cwd(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid, "mode": "light"}).json()
    up = c.post(f"/api/conversations/{conv['id']}/attachments",
                files={"file": ("note.txt", b"hello", "text/plain")})
    assert up.status_code == 201

    with Session() as db:
        conv_row = db.get(Conversation, conv["id"])
        task = db.get(Task, tid)
        cwd, _home = svc.conversation_cwd(db, conv_row)
        prompt = svc.build_prompt(db, task, conv_row, cwd, "解释一下")
    assert "attachments/note.txt" in prompt
    assert (cwd / "attachments" / "note.txt").is_file()


# ---------------------------------------------------------------- 5. 回退冲突


def _changes_for(workspace: Path, rel: str, before_text: str, after_text: str) -> tuple[dict, Path]:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(before_text, encoding="utf-8")
    before = optimize.capture(workspace)
    target.write_text(after_text, encoding="utf-8")
    after = optimize.capture(workspace)
    payload = svc._changes_payload(before, after)
    assert payload is not None and rel in payload["revert"]
    return payload, target


def _message_with_changes(db_session, conv_id: int, payload: dict) -> Message:
    msg = Message(conversation_id=conv_id, role="assistant", text="改了",
                  changes_json=json.dumps(payload, ensure_ascii=False))
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return msg


def test_revert_conflict_409_leaves_files_untouched(db_session, seeded, workspace: Path):
    payload, target = _changes_for(workspace, "src/add.cpp", "// kernel\n", "v1\n")
    assert payload["revert"]["src/add.cpp"]["after_sha256"] == hashlib.sha256(b"v1\n").hexdigest()
    conv = svc.create(db_session, seeded["task"].id)
    msg = _message_with_changes(db_session, conv.id, payload)

    target.write_text("v2\n", encoding="utf-8")  # 后续轮的改动
    with pytest.raises(ValueError, match="文件已被后续改动"):
        svc.revert_message(db_session, conv, msg, workspace)
    assert target.read_text(encoding="utf-8") == "v2\n", "冲突时一个文件都不应改"

    target.write_text("v1\n", encoding="utf-8")  # 回到该轮结束时状态 → 正常回退
    out = svc.revert_message(db_session, conv, msg, workspace)
    assert out["files"]["src/add.cpp"] == "restored"
    assert target.read_text(encoding="utf-8") == "// kernel\n"


def test_revert_added_file_conflict_and_success(db_session, seeded, workspace: Path):
    target = workspace / "src" / "new.txt"
    assert not target.exists()
    before = optimize.capture(workspace)
    target.write_text("brand new\n", encoding="utf-8")
    after = optimize.capture(workspace)
    payload = svc._changes_payload(before, after)
    assert payload and payload["revert"]["src/new.txt"]["existed"] is False
    conv = svc.create(db_session, seeded["task"].id)
    msg = _message_with_changes(db_session, conv.id, payload)

    target.write_text("later edit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="文件已被后续改动"):
        svc.revert_message(db_session, conv, msg, workspace)
    assert target.read_text(encoding="utf-8") == "later edit\n"

    target.write_text("brand new\n", encoding="utf-8")
    out = svc.revert_message(db_session, conv, msg, workspace)
    assert out["files"]["src/new.txt"] == "removed"
    assert not target.exists()


def test_revert_old_record_without_after_hash_is_compatible(db_session, seeded, workspace: Path):
    target = workspace / "src" / "add.cpp"
    target.write_text("newer\n", encoding="utf-8")
    payload = {
        "revert": {
            "src/add.cpp": {
                "existed": True,
                "restorable": True,
                "content_b64": base64.b64encode(b"// kernel\n").decode("ascii"),
                # 旧记录没有 after_sha256 → 跳过冲突校验
            }
        }
    }
    conv = svc.create(db_session, seeded["task"].id)
    msg = _message_with_changes(db_session, conv.id, payload)
    out = svc.revert_message(db_session, conv, msg, workspace)
    assert out["files"]["src/add.cpp"] == "restored"
    assert target.read_text(encoding="utf-8") == "// kernel\n"


def test_revert_conflict_http_409(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace)
    conv = c.post("/api/conversations", json={"task_id": tid}).json()

    target = workspace / "src" / "add.cpp"
    target.write_text("v0\n", encoding="utf-8")
    before = optimize.capture(workspace)
    target.write_text("v1\n", encoding="utf-8")
    after = optimize.capture(workspace)
    payload = svc._changes_payload(before, after)
    with Session() as db:
        msg = Message(conversation_id=conv["id"], role="assistant", text="改了",
                      changes_json=json.dumps(payload))
        db.add(msg)
        db.commit()
        mid = msg.id

    target.write_text("v2\n", encoding="utf-8")  # 后续轮改动 → 回退必须 409 且不改文件
    r = c.post(f"/api/conversations/{conv['id']}/messages/{mid}/revert")
    assert r.status_code == 409, r.text
    assert "文件已被后续改动" in r.json()["detail"]
    assert target.read_text(encoding="utf-8") == "v2\n"


# ---------------------------------------------------------------- 6. 案例一键发起


def _case_lib(tmp_path: Path, *, target_chip: str | None = None,
              extra_cases: list[dict] | None = None) -> Path:
    lib = tmp_path / "lib"
    case = lib / "demo" / "cases" / "relu"
    (case / "workspace" / "src").mkdir(parents=True)
    (case / "ops").mkdir(parents=True)
    (case / "cases.json").write_text(json.dumps({"cases": [{"case_id": "c1"}]}), encoding="utf-8")
    (case / "workspace" / "README.md").write_text("case workspace", encoding="utf-8")
    (case / "workspace" / "src" / "impl.cpp").write_text("// case impl", encoding="utf-8")
    (case / "ops" / "impl.c").write_text("// impl", encoding="utf-8")
    case_row = {"op": "relu", "scenario": "激活", "path": "cases/relu"}
    if target_chip:
        case_row["target_chip"] = target_chip
    cases = [case_row, *(extra_cases or [])]
    (lib / "index.json").write_text(json.dumps({
        "schema": "opx-case-library-index/1",
        "dest": str(lib),
        "bundles": [{"name": "demo", "version": "1", "cases": cases}],
    }), encoding="utf-8")
    return lib


def test_case_start_conflict_leaves_no_half_state_then_retry_ok(client, workspace: Path,
                                                               tmp_path: Path, monkeypatch):
    from app.config import settings

    lib = _case_lib(tmp_path, target_chip="910B2")
    monkeypatch.setattr(settings, "case_library", str(lib))
    c, Session = client
    pid, _tid = _seed_project_task(Session, workspace)
    (workspace / "README.md").write_text("existing", encoding="utf-8")  # 与案例 workspace 冲突

    r = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r.status_code == 409, r.text
    # 冲突时一个文件都不写（旧实现会留下半截 cases.json 挡住重试）
    assert not (workspace / "cases" / "relu").exists()
    assert not (workspace / "ops" / "relu").exists()
    with Session() as db:
        assert db.scalar(select(func.count(Task.id)).where(Task.input_form == "案例库")) == 0

    (workspace / "README.md").unlink()
    r2 = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r2.status_code == 201, r2.text
    tid = r2.json()["task_id"]
    with Session() as db:
        task = db.get(Task, tid)
        assert task.input_form == "案例库" and task.target_chip == "910B2"
    assert (workspace / "cases" / "relu" / "cases.json").is_file()
    assert (workspace / "README.md").read_text(encoding="utf-8") == "case workspace"
    assert (workspace / "ops" / "relu" / "impl.c").read_text(encoding="utf-8") == "// impl"

    # 二次发起仍被目标已存在挡住（409），且不会破坏第一次的结果
    r3 = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r3.status_code == 409
    assert (workspace / "ops" / "relu" / "impl.c").is_file()


def test_case_start_rejects_unsafe_op(client, workspace: Path, tmp_path: Path, monkeypatch):
    from app.config import settings

    lib = _case_lib(tmp_path, extra_cases=[
        {"op": "../../evil", "scenario": None, "path": "cases/relu"},
    ])
    monkeypatch.setattr(settings, "case_library", str(lib))
    c, Session = client
    pid, _tid = _seed_project_task(Session, workspace)
    r = c.post("/api/case-library/start",
               json={"bundle": "demo", "op": "../../evil", "project_id": pid})
    assert r.status_code == 409
    assert "非法算子名" in r.json()["detail"]
    assert not (tmp_path / "evil").exists()


def test_case_start_rolls_back_files_when_task_commit_fails(client, workspace: Path,
                                                           tmp_path: Path, monkeypatch):
    from app.config import settings

    lib = _case_lib(tmp_path)
    monkeypatch.setattr(settings, "case_library", str(lib))
    c, Session = client
    pid, _tid = _seed_project_task(Session, workspace)
    assert not (workspace / "ops").exists()

    with Session() as db:
        def boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(db, "commit", boom)
        with pytest.raises(RuntimeError):
            case_library.start(db, "demo", "relu", project_id=pid)

    # Task commit 失败 → 已复制的文件/目录全部回滚，可安全重试
    assert not (workspace / "cases" / "relu").exists()
    assert not (workspace / "ops").exists()
    r = c.post("/api/case-library/start", json={"bundle": "demo", "op": "relu", "project_id": pid})
    assert r.status_code == 201, r.text


def test_case_index_passes_target_chip(tmp_path: Path, monkeypatch):
    from app.config import settings

    lib = _case_lib(tmp_path, target_chip="910B3")
    monkeypatch.setattr(settings, "case_library", str(lib))
    body = case_library.index()
    case = body["bundles"][0]["cases"][0]
    assert case["target_chip"] == "910B3"


# ---------------------------------------------------------------- 7. 导入脚本


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_cases.py"
_spec = importlib.util.spec_from_file_location("import_cases_fixes", SCRIPT)
import_cases = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(import_cases)


def _zip(path: Path, files: dict[str, str]) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return path


def _manifest(cases=None, name="demo", **extra) -> str:
    man = {"schema": "opx-case-library/1", "name": name, "version": "2026.10", **extra}
    if cases is not None:
        man["cases"] = cases
    return json.dumps(man)


def _bundle_zip(tmp_path: Path, name: str = "demo",
                cases: list[dict] | None = None, extra_files: dict[str, str] | None = None) -> Path:
    files = {
        "manifest.json": _manifest(cases or [{"op": "relu", "path": "cases/relu"}], name=name),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "c1"}]}),
    }
    files.update(extra_files or {})
    return _zip(tmp_path / "bundle.zip", files)


@pytest.mark.parametrize("bad_name", ["../escaped", "..", ".", "a/b", "a\\b", "x" * 121,
                                      "bad\x01name"])
def test_import_rejects_unsafe_manifest_name(tmp_path: Path, bad_name: str):
    z = _bundle_zip(tmp_path, name=bad_name)
    with pytest.raises(import_cases.BundleError):
        import_cases.import_bundle(z, tmp_path / "lib")
    assert not (tmp_path / "escaped").exists()


def test_import_force_does_not_delete_outside(tmp_path: Path):
    dest = tmp_path / "lib"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "IMPORTANT.txt").write_text("keep me", encoding="utf-8")

    z = _bundle_zip(tmp_path, name="../victim")
    with pytest.raises(import_cases.BundleError):
        import_cases.import_bundle(z, dest, force=True)
    assert (victim / "IMPORTANT.txt").read_text(encoding="utf-8") == "keep me"

    # 合法 --force：只替换库目录内同名 bundle，库内其它文件不动
    ok = _bundle_zip(tmp_path, name="demo")
    dest.mkdir(exist_ok=True)
    (dest / "keepme.txt").write_text("library keeper", encoding="utf-8")
    first = import_cases.import_bundle(ok, dest)
    assert (dest / "demo").is_dir()
    (dest / "demo" / "stale.txt").write_text("old", encoding="utf-8")
    again = _zip(tmp_path / "bundle2.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "v2"}]}),
    })
    import_cases.import_bundle(again, dest, force=True)
    assert first["name"] == "demo"
    assert not (dest / "demo" / "stale.txt").exists()
    assert (dest / "keepme.txt").read_text(encoding="utf-8") == "library keeper"
    assert json.loads((dest / "demo" / "cases" / "relu" / "cases.json").read_text())["cases"][0]["case_id"] == "v2"


def test_import_rejects_zip_limits(tmp_path: Path, monkeypatch):
    # 条目数
    z = _zip(tmp_path / "many.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": []}),
        "cases/relu/a.txt": "a", "cases/relu/b.txt": "b", "cases/relu/c.txt": "c",
    })
    monkeypatch.setattr(import_cases, "MAX_FILES", 3)
    with pytest.raises(import_cases.BundleError, match="条目数"):
        import_cases.import_bundle(z, tmp_path / "lib")

    # 压缩比（1MB 0 字节压缩后极小）
    z2 = _zip(tmp_path / "bomb.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": []}),
        "cases/relu/zeros.bin": "0" * (1 << 20),
    })
    with pytest.raises(import_cases.BundleError, match="压缩比"):
        import_cases.import_bundle(z2, tmp_path / "lib2")

    # 单文件解压上限
    z3 = _bundle_zip(tmp_path)
    monkeypatch.setattr(import_cases, "MAX_FILE_BYTES", 20)
    with pytest.raises(import_cases.BundleError, match="单文件"):
        import_cases.import_bundle(z3, tmp_path / "lib3")
    monkeypatch.setattr(import_cases, "MAX_FILE_BYTES", 100 * 1024 * 1024)

    # 总解压上限
    monkeypatch.setattr(import_cases, "MAX_TOTAL_BYTES", 20)
    with pytest.raises(import_cases.BundleError, match="总量"):
        import_cases.import_bundle(z3, tmp_path / "lib4")


def test_import_index_write_is_atomic(tmp_path: Path, monkeypatch):
    dest = tmp_path / "lib"
    z1 = _bundle_zip(tmp_path)
    import_cases.import_bundle(z1, dest)
    index_before = (dest / "index.json").read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(import_cases.os, "replace", boom)
    z2 = _zip(tmp_path / "bundle2.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "v2"}]}),
    })
    with pytest.raises(OSError):
        import_cases.import_bundle(z2, dest, force=True)
    # 旧 index 完好，且没有留下临时文件
    assert (dest / "index.json").read_text(encoding="utf-8") == index_before
    assert list(dest.glob(".*.tmp")) == []


def test_import_keeps_target_chip_in_rows_and_index(tmp_path: Path):
    z = _zip(tmp_path / "chip.zip", {
        "manifest.json": _manifest([
            {"op": "relu", "path": "cases/relu", "target_chip": "910B2"},
            {"op": "softmax", "path": "cases/softmax"},
        ]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "c1"}]}),
        # 没在 manifest 条目里写 → 用 cases.json 顶层 target_chip
        "cases/softmax/cases.json": json.dumps({"cases": [{"case_id": "c2"}], "target_chip": "910B3"}),
    })
    res = import_cases.import_bundle(z, tmp_path / "lib")
    chips = {c["op"]: c["target_chip"] for c in res["cases"]}
    assert chips == {"relu": "910B2", "softmax": "910B3"}
    idx = json.loads((tmp_path / "lib" / "index.json").read_text(encoding="utf-8"))
    idx_chips = {c["op"]: c["target_chip"] for c in idx["bundles"][0]["cases"]}
    assert idx_chips == {"relu": "910B2", "softmax": "910B3"}


# ---------------------------------------------------------------- 8. 案例报告健壮性


def _write_run_with_correctness(workspace: Path, Session, tid: int, body: object) -> None:
    rd = workspace / "runs" / "r-bad"
    (rd / "out").mkdir(parents=True, exist_ok=True)
    (rd / "out" / "correctness.json").write_text(json.dumps(body), encoding="utf-8")
    (rd / "manifest.json").write_text(json.dumps({
        "artifacts": [{"type": "correctness", "path": "out/correctness.json"}],
    }), encoding="utf-8")
    with Session() as db:
        run = Run(task_id=tid, action="test", status="ok", manifest_path=str(rd / "manifest.json"))
        db.add(run)
        db.flush()
        db.add(Artifact(run_id=run.id, type="correctness", path="out/correctness.json"))
        db.commit()


def test_cases_endpoint_tolerates_non_dict_cases_json(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace, op="op1", title="op1")
    d = workspace / "cases" / "op1"
    d.mkdir(parents=True)
    (d / "cases.json").write_text("[1, 2, 3]", encoding="utf-8")
    r = c.get(f"/api/tasks/{tid}/cases")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_cases"] == 0 and body["case_source"] == ""


def test_cases_endpoint_tolerates_non_dict_correctness(client, workspace: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace, op="op1", title="op1")
    _write_run_with_correctness(workspace, Session, tid, [{"case_id": "x", "status": "FAIL"}])
    r = c.get(f"/api/tasks/{tid}/cases")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["failures"] == []
    assert body["runs"][0]["report_available"] is False


def test_cases_endpoint_does_not_read_outside_workspace(client, workspace: Path, tmp_path: Path):
    c, Session = client
    _pid, tid = _seed_project_task(Session, workspace, op="op1", title="op1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "cases.json").write_text(json.dumps({
        "cases": [{"case_id": "LEAKED-OUTSIDE"}],
    }), encoding="utf-8")

    for op in ("../../outside", str(outside)):
        with Session() as db:
            db.get(Task, tid).op_name = op
            db.commit()
        r = c.get(f"/api/tasks/{tid}/cases")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "LEAKED-OUTSIDE" not in r.text
        assert body["total_cases"] == 0 and body["case_source"] == ""
