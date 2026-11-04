"""任务删除：预览、级联清理、运行中保护、本机文件边界。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from app.config import settings
from app.models import (
    ActionJob,
    AgentJob,
    Artifact,
    Attachment,
    Conversation,
    Host,
    HostImage,
    IdeaSearch,
    Loop,
    LoopBudgetAudit,
    Message,
    OptimizationIdea,
    Project,
    Run,
    Savepoint,
    SpecVersion,
    StageEvent,
    Task,
)


def _seed(Session, workspace: Path, *, action_status: str = "ok",
          context_marker: bool = True):
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, is_test=True, env_recipe="docker run ...",
                          torch_version="2.10.0", digest="sha256:beef")
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id, image_id=image.id,
                          criteria_ref="ops-precision-standard@1")
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="delete me", op_name="delete_me", stage="开发")
        db.add(task)
        db.flush()

        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id

        loop = Loop(task_id=task.id, kind="design", scope="s", status="closed")
        job = AgentJob(task_id=task.id, kind="开发", status="ok")
        conv = Conversation(task_id=task.id, project_id=project.id, mode="task", title="c")
        db.add_all([loop, job, conv])
        db.flush()
        search = IdeaSearch(task_id=task.id, agent_job_id=job.id, status="ok")
        db.add(search)
        db.flush()
        db.add(OptimizationIdea(task_id=task.id, loop_id=loop.id, search_id=search.id,
                                source="agent", title="减少搬运", status="validating",
                                dedupe_key="d" * 64))
        db.add(LoopBudgetAudit(task_id=task.id, loop_id=loop.id, action="benchmark",
                               decision="allowed", limit_count=2, used_count=0))

        run_dir = Path(settings.repo_root) / "runs" / "r-delete-test"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = run_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        (run_dir / "out").mkdir(exist_ok=True)
        (run_dir / "out" / "run.log").write_text("hi\n", encoding="utf-8")

        run = Run(task_id=task.id, loop_id=loop.id, agent_job_id=job.id, action="test", status="ok",
                  spec_version_id=spec.id, remote_workdir="r-delete-test",
                  manifest_path=str(manifest), code_snapshot_sha256="1" * 64)
        db.add(run)
        db.flush()
        db.add(Artifact(run_id=run.id, type="log", path="out/run.log", size_bytes=3))
        db.add(ActionJob(task_id=task.id, action="test", status=action_status,
                         run_ref="r-delete-test", run_id=run.id))
        db.add(StageEvent(task_id=task.id, to_stage="开发", actor="test"))
        # 版本存档行：只删行（归档文件不随任务删除清理），counts 要能数到它
        db.add(Savepoint(task_id=task.id, project_id=project.id, label="基线",
                         kind="manual", code_snapshot_sha256="2" * 64,
                         archive_path=str(Path(settings.repo_root) / "archives" / "sources"
                                          / f"{'2' * 64}.tar.gz"),
                         archive_bytes=16))
        db.add(Message(conversation_id=conv.id, role="user", text="hello"))
        db.add(Attachment(conversation_id=conv.id, filename="a.txt",
                          path=f".opx-chat/conv-{conv.id}/attachments/a.txt", size_bytes=1))

        context = workspace / "tasks" / "delete_me"
        context.mkdir(parents=True, exist_ok=True)
        if context_marker:
            (context / ".opx-task").write_text(str(task.id), encoding="utf-8")
        (context / "TASK.md").write_text("task\n", encoding="utf-8")
        chat = workspace / ".opx-chat" / f"task-{task.id}"
        chat.mkdir(parents=True, exist_ok=True)
        (chat / "notes.md").write_text("chat\n", encoding="utf-8")
        conv_dir = workspace / ".opx-chat" / f"conv-{conv.id}" / "attachments"
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "a.txt").write_text("a", encoding="utf-8")

        db.commit()
        return {
            "task_id": task.id,
            "conversation_id": conv.id,
            "run_dir": run_dir,
            "context": context,
            "chat": chat,
            "conv_dir": workspace / ".opx-chat" / f"conv-{conv.id}",
        }


def _assert_empty(Session, task_id: int) -> None:
    with Session() as db:
        assert db.get(Task, task_id) is None
        for model, where in [
            (SpecVersion, SpecVersion.task_id == task_id),
            (Loop, Loop.task_id == task_id),
            (OptimizationIdea, OptimizationIdea.task_id == task_id),
            (IdeaSearch, IdeaSearch.task_id == task_id),
            (LoopBudgetAudit, LoopBudgetAudit.task_id == task_id),
            (AgentJob, AgentJob.task_id == task_id),
            (Run, Run.task_id == task_id),
            (ActionJob, ActionJob.task_id == task_id),
            (StageEvent, StageEvent.task_id == task_id),
            (Savepoint, Savepoint.task_id == task_id),
            (Conversation, Conversation.task_id == task_id),
        ]:
            assert int(db.scalar(select(func.count()).select_from(model).where(where)) or 0) == 0, model
        assert int(db.scalar(select(func.count()).select_from(Artifact)) or 0) == 0
        assert int(db.scalar(select(func.count()).select_from(Message)) or 0) == 0
        assert int(db.scalar(select(func.count()).select_from(Attachment)) or 0) == 0


def test_delete_preview_and_files(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)

    preview = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    assert preview["counts"] == {
        "spec_versions": 1, "loops": 1, "optimization_ideas": 1, "idea_searches": 1,
        "loop_budget_audits": 1, "agent_jobs": 1, "runs": 1, "artifacts": 1,
        "action_jobs": 1, "stage_events": 1, "conversations": 1, "messages": 1, "attachments": 1,
        "savepoints": 1,
        # 任务生命周期 v2：版本历史 + 任务级文档（seed 没有这两类记录 → 0）
        "task_versions": 0, "task_documents": 0,
    }
    assert preview["running"] == {"action_jobs": [], "agent_jobs": []}
    kinds = {f["kind"] for f in preview["files"]["items"] if f["removable"]}
    assert kinds == {"context", "chat", "conversation", "run"}
    assert preview["files"]["total_bytes"] > 0

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["deleted"] is True
    assert body["files"]["requested"] is True
    assert len(body["files"]["deleted"]) == 4
    assert body["files"]["errors"] == []

    _assert_empty(Session, ids["task_id"])
    assert not ids["run_dir"].exists()
    assert not ids["context"].exists()
    assert not ids["chat"].exists()
    assert not ids["conv_dir"].exists()
    assert c.get(f"/api/tasks/{ids['task_id']}").status_code == 404


def test_delete_keeps_local_files_by_default(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)

    res = c.delete(f"/api/tasks/{ids['task_id']}")
    assert res.status_code == 200, res.text
    assert res.json()["files"] == {"requested": False, "deleted": [], "skipped": [], "errors": []}
    _assert_empty(Session, ids["task_id"])
    assert ids["run_dir"].exists()
    assert ids["context"].exists()
    assert ids["chat"].exists()
    assert ids["conv_dir"].exists()


def test_delete_blocked_while_running(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace, action_status="running")

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 409
    assert "运行中的作业" in res.text
    with Session() as db:
        assert db.get(Task, ids["task_id"]) is not None
        assert db.scalar(select(ActionJob).where(ActionJob.task_id == ids["task_id"])) is not None
    assert ids["run_dir"].exists()


def test_delete_skips_unsafe_paths(client, workspace: Path, tmp_path: Path, monkeypatch):
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace, context_marker=False)

    # run 目录放到允许根之外：应只跳过文件，不阻止删库
    outside = tmp_path / "outside" / "r-delete-test"
    outside.mkdir(parents=True)
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    with Session() as db:
        run = db.scalar(select(Run).where(Run.task_id == ids["task_id"]))
        run.manifest_path = str(outside / "manifest.json")
        db.commit()

    preview = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    by_kind = {f["kind"]: f for f in preview["files"]["items"] if f["exists"]}
    assert by_kind["context"]["removable"] is False
    assert by_kind["run"]["removable"] is False

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["files"]["errors"] == []
    assert {s["path"] for s in body["files"]["skipped"]} >= {str(ids["context"]), str(outside)}
    assert ids["context"].exists()
    assert outside.exists()
    # 安全的那两个目录仍然被清掉
    assert not ids["chat"].exists()
    assert not ids["conv_dir"].exists()
    _assert_empty(Session, ids["task_id"])


def test_delete_blocks_fresh_agent_job_with_cancel_hint(client, workspace: Path, tmp_path: Path,
                                                       monkeypatch):
    """DB running 且时间没过 agent_timeout（可能另一个进程真在跑）→ 仍阻断，并提示取消接口。"""
    from datetime import datetime, timezone

    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)
    with Session() as db:
        job = AgentJob(task_id=ids["task_id"], kind="需求澄清", status="running",
                       started_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()

    res = c.delete(f"/api/tasks/{ids['task_id']}")
    assert res.status_code == 409
    detail = res.json()["detail"]
    assert "运行中的作业" in detail and "/api/agent-jobs/" in detail and "/cancel" in detail


def test_delete_ignores_stale_agent_job(client, workspace: Path, tmp_path: Path, monkeypatch):
    """僵尸兜底：DB running 但本进程无 state 且已超 agent_timeout → 视为 stale，不阻断删除。"""
    from datetime import datetime, timedelta, timezone

    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    monkeypatch.setattr(settings, "agent_timeout", 1800)
    ids = _seed(Session, workspace)
    with Session() as db:
        job = AgentJob(task_id=ids["task_id"], kind="需求澄清", status="running",
                       started_at=datetime.now(timezone.utc) - timedelta(seconds=7200))
        db.add(job)
        db.commit()
        stale_id = job.id

    preview = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    assert preview["running"]["agent_jobs"] == []  # stale 不阻断

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 200, res.text
    _assert_empty(Session, ids["task_id"])
    with Session() as db:
        assert db.get(AgentJob, stale_id) is None  # 僵尸行随任务一起级联删掉


def test_delete_skips_run_dir_owned_by_other_task(client, workspace: Path, tmp_path: Path,
                                                  monkeypatch):
    """run 目录归属看 manifest.task_id：属于别的任务 → removable=False，不误删（审查 #7/#17）。"""
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)
    ids["run_dir"].joinpath("manifest.json").write_text('{"task_id": 999999}', encoding="utf-8")

    preview = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    run_item = next(f for f in preview["files"]["items"] if f["kind"] == "run")
    assert run_item["removable"] is False
    assert "属于任务 #999999" in run_item["reason"]

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 200, res.text
    body = res.json()
    assert str(ids["run_dir"]) in {s["path"] for s in body["files"]["skipped"]}
    assert ids["run_dir"].exists()  # 别的任务的产物不能被删
    assert not ids["context"].exists()  # 本任务自己的目录照常清理


def test_delete_run_dir_with_matching_manifest_task_id(client, workspace: Path, tmp_path: Path,
                                                       monkeypatch):
    """旧 manifest（无 task_id）按旧判断兼容；manifest.task_id 等于本任务 → 可删。"""
    c, Session = client
    monkeypatch.setattr(settings, "repo_root", str(tmp_path))
    ids = _seed(Session, workspace)  # _seed 写的 manifest 是 "{}"：旧格式，无 task_id

    legacy = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    assert next(f for f in legacy["files"]["items"] if f["kind"] == "run")["removable"] is True

    ids["run_dir"].joinpath("manifest.json").write_text(
        f'{{"task_id": {ids["task_id"]}}}', encoding="utf-8")
    preview = c.get(f"/api/tasks/{ids['task_id']}/delete-preview").json()
    assert next(f for f in preview["files"]["items"] if f["kind"] == "run")["removable"] is True

    res = c.delete(f"/api/tasks/{ids['task_id']}?delete_files=true")
    assert res.status_code == 200, res.text
    assert not ids["run_dir"].exists()
