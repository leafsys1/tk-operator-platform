"""P0-3 第二步：对话式需求澄清（走 dsh 会话）+ 收尾（定版 / 上下文空间）。

模型与 dsh 都假掉（monkeypatch `chat.converse`）：不花 token、不依赖网络/CLI，
只测「契约与落库」——回复/规格草稿/待确认项如何进 spec_draft / chat_log / agent_job，
以及定版门禁与上下文空间文件。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.models import AgentJob, Project, StageEvent, Task
from app.routers import tasks as tasks_router
from app.services import agent_runner, chat


def _draft(Session, workspace: Path, *, ready: bool, fields: dict) -> int:
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace), criteria_ref="ops-precision-standard@1")
        db.add(p)
        db.flush()
        t = Task(
            project_id=p.id, title="rms_norm", op_name="rms_norm", is_draft=True,
            criteria_ref="ops-precision-standard@1",
            spec_draft=json.dumps({
                "fields": fields,
                "questions": [] if ready else [{"id": "q1", "text": "支持哪些 dtype？"}],
                "ready": ready,
            }, ensure_ascii=False),
            chat_log=json.dumps([{"role": "user", "text": "开发 rms_norm"}], ensure_ascii=False),
        )
        db.add(t)
        db.commit()
        return t.id


def test_chat_merges_spec_and_persists(client, workspace: Path, monkeypatch):
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})

    async def fake_converse(*, prompt, cwd, dsh_home, session_id, timeout):
        assert "rms_norm" in prompt and "dtype" in prompt  # 固定层（当前草稿/待确认项）确实拼进了 prompt
        return chat.TurnResult(
            ok=True, prompt=prompt, session_id="ses-abc",
            parsed={"reply": "还差三点：1) dtype？2) eps？3) 归约维？",
                    "spec": {"op_name": "rms_norm", "math_def": "x / sqrt(mean(x^2)+eps)", "oracle": "torch CPU"},
                    "questions": [{"id": "q2", "text": "支持哪些 dtype？"}], "ready": False},
            usage={"inputTokens": 111, "outputTokens": 42, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            exit_code=0,
        )

    monkeypatch.setattr(chat, "converse", fake_converse)

    r = c.post(f"/api/tasks/{tid}/chat", json={"message": "dtype 用 fp32+fp16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "还差三点" in body["reply"]
    assert body["spec"]["math_def"].startswith("x / sqrt")
    assert body["questions"][0]["id"] == "q2"
    assert [m["role"] for m in body["messages"]] == ["user", "user", "assistant"]

    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["spec_draft"]["fields"]["oracle"] == "torch CPU"
    assert detail["dsh_session_id"] == "ses-abc"  # 会话 id 落到任务上（续会话用）
    jobs = [j for j in detail["agent_jobs"] if j["kind"] == "需求澄清"]
    assert jobs and jobs[-1]["status"] == "ok" and jobs[-1]["dsh_session_id"] == "ses-abc"
    assert jobs[-1]["tokens_in"] == 111 and jobs[-1]["tokens_out"] == 42


def test_chat_failure_returns_502_and_keeps_record(client, workspace: Path, monkeypatch):
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})

    async def boom(*, prompt, cwd, dsh_home, session_id, timeout):
        return chat.TurnResult(ok=False, prompt=prompt, session_id=None, error="dsh 超时（180s）")

    monkeypatch.setattr(chat, "converse", boom)
    r = c.post(f"/api/tasks/{tid}/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert "dsh 超时" in r.json()["detail"]
    detail = c.get(f"/api/tasks/{tid}").json()
    assert any(j["kind"] == "需求澄清" and j["status"] == "failed" for j in detail["agent_jobs"])


def test_finalize_creates_spec_and_context_space(client, workspace: Path):
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"op_name": "rms_norm", "math_def": "x / sqrt(mean(x^2)+eps)", "oracle": "torch CPU"})

    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["spec_version"]["version"] == 1
    assert body["spec_version"]["op_name"] == "rms_norm"
    assert body["spec_version"]["sha256"]
    paths = {f["path"] for f in body["files"]}
    assert {"TASK.md", "AGENTS.md", "context.yaml", "spec/spec.json", "spec/spec.md"} <= paths

    space = Path(body["space_dir"])
    assert (space / "TASK.md").is_file() and (space / "context.yaml").is_file()
    spec_json = json.loads((space / "spec" / "spec.json").read_text(encoding="utf-8"))
    assert spec_json["_meta"]["spec_version_id"] == body["spec_version"]["id"]
    assert spec_json["oracle"] == "torch CPU"

    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["is_draft"] is False
    assert detail["current_spec_version_id"] == body["spec_version"]["id"]

    ctx = c.get(f"/api/tasks/{tid}/context").json()
    assert ctx["exists"] is True and any(f["path"] == "TASK.md" for f in ctx["files"])


def test_finalize_blocked_when_not_ready(client, workspace: Path):
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})
    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 409
    assert "定版门禁" in r.json()["detail"]


def test_finalize_requires_valid_op_name_and_remains_recoverable(client, workspace: Path):
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"math_def": "x + 1", "oracle": "torch CPU"})
    with Session() as db:
        task = db.get(Task, tid)
        task.op_name = None
        db.commit()

    blocked = c.post(f"/api/tasks/{tid}/finalize")
    assert blocked.status_code == 409
    assert "缺少算子名" in blocked.json()["detail"]

    patched = c.patch(f"/api/tasks/{tid}", json={"op_name": "elementwise_add"})
    assert patched.status_code == 200, patched.text
    finalized = c.post(f"/api/tasks/{tid}/finalize")
    assert finalized.status_code == 200, finalized.text
    assert finalized.json()["spec_version"]["op_name"] == "elementwise_add"


def test_model_invalid_op_name_does_not_poison_draft(client, workspace: Path, monkeypatch):
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})

    async def fake_converse(*, prompt, cwd, dsh_home, session_id, timeout):
        return chat.TurnResult(
            ok=True, prompt=prompt,
            parsed={"reply": "已完成", "spec": {"op_name": "../evil", "math_def": "x"},
                    "questions": [], "ready": True},
            exit_code=0,
        )

    monkeypatch.setattr(chat, "converse", fake_converse)
    r = c.post(f"/api/tasks/{tid}/chat", json={"message": "继续"})
    assert r.status_code == 200, r.text
    assert r.json()["ready"] is False
    assert any(q.get("id") == "op_name" for q in r.json()["questions"])
    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["op_name"] == "rms_norm"
    assert detail["spec_draft"]["fields"].get("op_name") is None


def test_draft_cannot_bypass_finalize_and_specs_require_nonempty_object(client, workspace: Path):
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"op_name": "rms_norm", "math_def": "x"})
    draft_spec = c.post(f"/api/tasks/{tid}/specs", json={"spec_json": '{"math_def":"x"}'})
    assert draft_spec.status_code == 409
    assert "草稿任务" in draft_spec.json()["detail"]

    assert c.post(f"/api/tasks/{tid}/finalize").status_code == 200
    for bad in (None, "", "[]", "{}"):
        r = c.post(f"/api/tasks/{tid}/specs", json={"spec_json": bad})
        assert r.status_code == 400, (bad, r.status_code, r.text)
    good = c.post(f"/api/tasks/{tid}/specs", json={"spec_json": '{"math_def":"x + 1"}'})
    assert good.status_code == 201, good.text


def test_extract_json_tolerates_fences_and_noise():
    assert chat.extract_json('```json\n{"reply": "a", "ready": false}\n```')["reply"] == "a"
    assert chat.extract_json('好的：{"reply": "x", "ready": true} 完毕')["ready"] is True


def test_chat_stream_starts_job_and_applies_on_done(client, workspace: Path, monkeypatch):
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})
    captured: dict = {}
    monkeypatch.setattr(agent_runner, "available", lambda: True)  # 测试机不一定装 dsh

    def fake_start(job_id, prompt, cwd, dsh_home, session_id=None, on_done=None, **_kw):
        captured["on_done"] = on_done
        return agent_runner.JobState(job_id, prompt)

    monkeypatch.setattr(agent_runner, "start", fake_start)
    monkeypatch.setattr(tasks_router, "SessionLocal", Session)  # 回调改用测试库

    r = c.post(f"/api/tasks/{tid}/chat/stream", json={"message": "dtype 用 fp32"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["job_id"], int)
    assert body["messages"][-1] == {"role": "user", "text": "dtype 用 fp32"}  # 用户这条已先落库
    assert c.get(f"/api/tasks/{tid}").json()["chat_log"][-1]["role"] == "user"

    class FakeState:
        final = ('{"reply": "好的，还差 eps", "spec": {"op_name": "rms_norm", "oracle": "torch CPU"},'
                 ' "questions": [{"id": "q3", "text": "eps?"}], "ready": false}')
        session_id = "ses-stream"
        exit_code = 0
        error = None
        usage = {"inputTokens": 5, "outputTokens": 6}

    captured["on_done"](FakeState())  # 模拟 dsh 跑完触发完成回调
    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["chat_log"][-1] == {"role": "assistant", "text": "好的，还差 eps"}
    assert detail["spec_draft"]["fields"]["oracle"] == "torch CPU"
    assert detail["spec_draft"]["questions"][0]["id"] == "q3"
    assert detail["dsh_session_id"] == "ses-stream"


def test_chat_stream_rejects_when_dsh_missing(client, workspace: Path, monkeypatch):
    """dsh 不在 PATH：明确 503，且不建 job / 不落半条用户消息（审查 #1）。"""
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})
    monkeypatch.setattr(agent_runner, "available", lambda: False)

    r = c.post(f"/api/tasks/{tid}/chat/stream", json={"message": "hi"})
    assert r.status_code == 503
    assert "dsh" in r.json()["detail"]

    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["agent_jobs"] == []  # 没有留下永久 running 的 job
    assert detail["chat_log"] == [{"role": "user", "text": "开发 rms_norm"}]  # 没有重复消费这条消息


def test_chat_stream_rejects_second_message_while_job_running(client, workspace: Path, monkeypatch):
    """同一任务已有在跑的澄清轮：409，不重复消费同一条消息/不开第二条 SSE。"""
    c, Session = client
    tid = _draft(Session, workspace, ready=False, fields={"op_name": "rms_norm"})
    monkeypatch.setattr(agent_runner, "available", lambda: True)
    with Session() as db:
        job = AgentJob(task_id=tid, kind="需求澄清", status="running", prompt="p",
                       started_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        jid = job.id
    agent_runner.REGISTRY[jid] = agent_runner.JobState(jid, "p")  # 本进程确实还在跑
    try:
        r = c.post(f"/api/tasks/{tid}/chat/stream", json={"message": "再来一条"})
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert f"job #{jid}" in detail and "/api/agent-jobs/" in detail and "/cancel" in detail
    finally:
        agent_runner.REGISTRY.pop(jid, None)


def test_finalize_is_idempotent_and_writes_derived_stage(client, workspace: Path):
    """二次定版 409、只落一版 spec；定版后 stage=推导阶段「开发」并写 stage_events（审查 #2/#3）。"""
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"op_name": "rms_norm", "math_def": "x / sqrt(mean(x^2)+eps)"})

    first = c.post(f"/api/tasks/{tid}/finalize")
    assert first.status_code == 200, first.text

    # v2：阶段一还要求「用例组已确认」，所以定版后停在「需求与设计」（等对话确认用例）
    view = c.get(f"/api/tasks/{tid}/stage").json()
    assert view["stored_stage"] == "需求与设计"
    assert view["derived_stage"] == "需求与设计"
    assert view["in_sync"] is True

    # 落用例文件 → 对话确认 → 阶段一达成、推导进入「开发」
    import hashlib as _hashlib
    case_dir = workspace / "cases" / "rms_norm"
    case_dir.mkdir(parents=True)
    cases_bytes = b'{"cases": []}'
    bench_bytes = b'{"cases": []}'
    (case_dir / "cases.json").write_bytes(cases_bytes)
    (case_dir / "benchmark.json").write_bytes(bench_bytes)
    conf = c.post(f"/api/tasks/{tid}/cases/confirm", json={
        "cases_sha256": _hashlib.sha256(cases_bytes).hexdigest(),
        "benchmark_sha256": _hashlib.sha256(bench_bytes).hexdigest(),
        "by": "chat:job#1",
    })
    assert conf.status_code == 200, conf.text
    assert conf.json()["cases_suite_sha256"]
    assert c.get(f"/api/tasks/{tid}/cases/confirmation").json()["in_sync"] is True

    view = c.get(f"/api/tasks/{tid}/stage").json()
    assert view["derived_stage"] == "开发"
    assert view["stored_stage"] == "开发"  # 用例确认时按证据推进阶段并留痕
    assert view["in_sync"] is True

    second = c.post(f"/api/tasks/{tid}/finalize")
    assert second.status_code == 409
    assert "已定版" in second.json()["detail"] and "/specs" in second.json()["detail"]

    assert [s["version"] for s in c.get(f"/api/tasks/{tid}/specs").json()] == [1]

    # v2：定版落 v1.0 版本行；用例确认写 stage_event
    versions = c.get(f"/api/tasks/{tid}/versions").json()
    assert versions["current"]["major"] == 1 and versions["current"]["minor"] == 0
    with Session() as db:
        evs = list(db.scalars(select(StageEvent).where(StageEvent.task_id == tid).order_by(StageEvent.id)))
        assert any(str(ev.reason or "").startswith("用例组确认") for ev in evs)


def test_context_space_follows_owner_marker_after_rename_and_same_slug_delete(client, workspace: Path):
    """上下文空间按 `.opx-task` 归属找人，而不是按可变的 op_name 现算（审查 #5）。"""
    c, Session = client
    with Session() as db:
        project = Project(name="p-ctx", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        ids = []
        for _ in range(2):  # 两个任务同 op_name="add"
            task = Task(
                project_id=project.id, title="add", op_name="add", is_draft=True,
                criteria_ref="ops-precision-standard@1",
                spec_draft=json.dumps({"fields": {"op_name": "add"}, "questions": [], "ready": True},
                                      ensure_ascii=False),
            )
            db.add(task)
            db.flush()
            ids.append(task.id)
        db.commit()
    a, b = ids

    ra = c.post(f"/api/tasks/{a}/finalize")
    rb = c.post(f"/api/tasks/{b}/finalize")
    assert ra.status_code == 200 and rb.status_code == 200
    dir_a, dir_b = Path(ra.json()["space_dir"]), Path(rb.json()["space_dir"])
    assert dir_a.name == "add"
    assert dir_b.name == f"add-{b}"  # 同 slug 冲突 → 退化成 <slug>-<id>

    # 模拟历史遗留漂移：op_name 已被改过（新接口已冻结 op_name，这里直接改库）
    with Session() as db:
        db.get(Task, a).op_name = "renamed"
        db.commit()
    ctx_a = c.get(f"/api/tasks/{a}/context").json()
    assert ctx_a["exists"] is True
    assert Path(ctx_a["space_dir"]) == dir_a  # 仍指向 tasks/add，而不是 tasks/renamed

    # 删掉 A（含文件）：即使实时推导已变成 tasks/renamed，也必须按标记删掉 tasks/add
    deleted = c.delete(f"/api/tasks/{a}?delete_files=true")
    assert deleted.status_code == 200, deleted.text
    assert not dir_a.exists()

    # A 消失后 B 的实时推导会漂回 tasks/add（不存在），标记扫描仍能找到 tasks/add-2
    ctx_b = c.get(f"/api/tasks/{b}/context").json()
    assert ctx_b["exists"] is True
    assert Path(ctx_b["space_dir"]) == dir_b

    deleted_b = c.delete(f"/api/tasks/{b}?delete_files=true")
    assert deleted_b.status_code == 200, deleted_b.text
    assert not dir_b.exists()


def test_chat_after_finalize_only_appends_log(client, workspace: Path, monkeypatch):
    """定版后仍可聊天（追问实现/历史），但只写 chat_log：spec_draft/raw_input/身份字段冻结。

    否则聊天会产出一份永不生效的旁路 spec_draft，与 spec_versions 真相源冲突（审查 #4 残项）。
    """
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"op_name": "rms_norm", "math_def": "x / sqrt(mean(x^2)+eps)", "oracle": "torch CPU"})
    assert c.post(f"/api/tasks/{tid}/finalize").status_code == 200
    before = c.get(f"/api/tasks/{tid}").json()

    async def fake_converse(*, prompt, cwd, dsh_home, session_id, timeout):
        return chat.TurnResult(
            ok=True, prompt=prompt, session_id="ses-after",
            parsed={"reply": "定版后追问的答复",
                    "spec": {"op_name": "renamed", "math_def": "SHOULD NOT LAND"},
                    "questions": [{"id": "q9", "text": "不该出现"}], "ready": False},
            usage={"inputTokens": 1, "outputTokens": 2, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            exit_code=0,
        )

    monkeypatch.setattr(chat, "converse", fake_converse)
    r = c.post(f"/api/tasks/{tid}/chat", json={"message": "实现上有什么坑？"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "定版后追问的答复"
    # 返回的是当前已定版规格，不是模型本轮解析的草稿
    assert body["spec"] == before["spec_draft"]["fields"]
    assert body["questions"] == before["spec_draft"]["questions"]
    assert body["ready"] is True

    after = c.get(f"/api/tasks/{tid}").json()
    assert after["spec_draft"] == before["spec_draft"]
    assert after["raw_input"] == before["raw_input"]
    assert after["op_name"] == "rms_norm"
    assert after["chat_log"][-2:] == [
        {"role": "user", "text": "实现上有什么坑？"},
        {"role": "assistant", "text": "定版后追问的答复"},
    ]


def test_create_spec_cannot_rename_finalized_task(client, workspace: Path):
    """定版后新规格版本不能换算子名（上下文空间目录/OPX_OP 依赖它，冻结以免目录漂移）。"""
    c, Session = client
    tid = _draft(Session, workspace, ready=True,
                 fields={"op_name": "rms_norm", "math_def": "x / sqrt(mean(x^2)+eps)"})
    assert c.post(f"/api/tasks/{tid}/finalize").status_code == 200

    bad = c.post(f"/api/tasks/{tid}/specs", json={
        "spec_json": json.dumps({"op_name": "renamed"}), "op_name": "renamed", "set_current": True,
    })
    assert bad.status_code == 409
    assert "冻结" in bad.json()["detail"]

    ok = c.post(f"/api/tasks/{tid}/specs", json={
        "spec_json": json.dumps({"op_name": "rms_norm", "math_def": "changed"}),
        "op_name": "rms_norm", "set_current": True,
    })
    assert ok.status_code == 201, ok.text
    detail = c.get(f"/api/tasks/{tid}").json()
    assert detail["op_name"] == "rms_norm"
    assert [s["version"] for s in detail["spec_versions"]] == [1, 2]
    # 上下文空间仍指向定版时的目录（marker 归属不变）
    assert c.get(f"/api/tasks/{tid}/context").json()["exists"] is True
