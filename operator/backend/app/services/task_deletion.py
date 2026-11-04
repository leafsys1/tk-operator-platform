"""任务删除：预览影响面 + 显式级联清理 + 可选的本机文件清理。

为什么不能只 `db.delete(task)`：
- SQLite 默认没开 `PRAGMA foreign_keys=ON`，`ondelete="CASCADE"` 不会生效；
- `stage_events` / `action_jobs` / `conversations` 没有挂到 Task 的 ORM relationship 上；
- 只靠关系级联会留下孤儿行，后续按 task_id 查询会读到不存在任务的记录。

文件清理只碰「明确属于该任务」的目录：
- `tasks/<slug>`（必须 `.opx-task` 标记等于 task.id；改名/同 slug 任务删除后按标记扫描兜底）
- `.opx-chat/task-<task_id>`
- `.opx-chat/conv-<conversation_id>`（该任务会话的附件）
- 本地 `runs/<run_ref>`（必须位于允许的 runs 根目录、目录名匹配 remote_workdir；
  manifest 里带 `task_id` 时还必须等于该任务 —— B 线新 run_ref 起携带归属，旧 manifest 兼容）

远端 run 目录与 dsh 会话 JSONL 不在 v1 清理范围内（避免误删远端证据）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete as sql_delete, func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    ActionJob,
    AgentJob,
    Artifact,
    Attachment,
    Conversation,
    IdeaSearch,
    Loop,
    LoopBudgetAudit,
    Message,
    OptimizationIdea,
    Run,
    Savepoint,
    SpecVersion,
    StageEvent,
    Task,
    TaskDocument,
    TaskVersion,
)
from . import agent_runner, context_space


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _file_item(kind: str, path: Path, *, removable: bool = True,
               reason: str | None = None, **extra) -> dict:
    exists = path.exists()
    return {
        "kind": kind,
        "path": str(path),
        "exists": exists,
        "bytes": _dir_size(path) if exists and path.is_dir() else (path.stat().st_size if exists and path.is_file() else 0),
        "removable": removable and exists,
        "reason": reason,
        **extra,
    }


def _workspace_of(db: Session, task: Task) -> Path | None:
    from ..models import Project

    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        return None
    ws = Path(project.workspace_path).expanduser()
    return ws if ws.is_dir() else None


def _context_space_item(workspace: Path, task: Task) -> dict:
    """只把「确认属于本任务」的上下文空间列为可删。

    用 `find_space_dir` 而不是实时推导：改名/删掉同 slug 任务后，目录归属只看 `.opx-task`。
    """
    base = context_space.find_space_dir(workspace, task)
    if not base.exists():
        return _file_item("context", base)
    marker = base / ".opx-task"
    if not marker.is_file():
        return _file_item("context", base, removable=False,
                          reason="缺少 .opx-task 归属标记，跳过以防误删")
    try:
        owner = marker.read_text(encoding="utf-8").strip()
    except OSError as e:
        return _file_item("context", base, removable=False, reason=f"读不到归属标记：{e}")
    if owner != str(task.id):
        return _file_item("context", base, removable=False,
                          reason=f"目录属于任务 #{owner or '?'}，跳过")
    return _file_item("context", base)


def _manifest_task_id(manifest: Path) -> int | None:
    """读 manifest 里的 `task_id`（B 线起脚本会写）；旧 manifest 没有该字段/读不了 → None。

    None 表示「无从校验归属」，调用方保持旧判断兼容（不能因为老目录没字段就拒绝清理）。
    """
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("task_id") is None:
        return None
    try:
        return int(data["task_id"])
    except (TypeError, ValueError):
        return None


def _run_dir_item(run: Run, workspace: Path | None, task: Task) -> dict | None:
    """本地 run 目录：必须落在允许的 runs/ 根目录内、目录名 = remote_workdir、且归属本任务。

    归属以 manifest 的 `task_id` 为准（run_ref 自 2026-10-21 起含 task id，但 delete_files 仍以
    目录内的 manifest 为硬证据）；旧 manifest 没有 `task_id` 时退回「根路径 + 目录名」兼容判断。
    """
    if not run.manifest_path:
        return None
    manifest = Path(str(run.manifest_path)).expanduser()
    path = manifest.parent
    if not path.exists():
        return None
    roots = [(Path(settings.repo_root) / "runs")]
    if workspace is not None:
        roots.append(workspace / "runs")
    if not any(_under(path, root) for root in roots):
        return _file_item("run", path, removable=False,
                          reason="不在允许的 runs/ 根目录下，跳过", run_id=run.id,
                          run_ref=run.remote_workdir)
    if run.remote_workdir and path.name != run.remote_workdir:
        return _file_item("run", path, removable=False,
                          reason=f"目录名与 run_ref（{run.remote_workdir}）不一致，跳过",
                          run_id=run.id, run_ref=run.remote_workdir)
    owner = _manifest_task_id(manifest)
    if owner is not None and owner != task.id:
        return _file_item("run", path, removable=False,
                          reason=f"目录属于任务 #{owner}，跳过", run_id=run.id,
                          run_ref=run.remote_workdir)
    return _file_item("run", path, run_id=run.id, run_ref=run.remote_workdir)


def _cleanup_items(db: Session, task: Task) -> list[dict]:
    workspace = _workspace_of(db, task)
    if workspace is None:
        return []
    items: list[dict] = [_context_space_item(workspace, task)]

    chat_dir = workspace / ".opx-chat" / f"task-{task.id}"
    items.append(_file_item("chat", chat_dir))

    # 任务生命周期 v2：任务级文档（原文 + 解析稿）
    items.append(_file_item("task_docs", workspace / "task-docs" / f"task-{task.id}"))

    conv_ids = list(db.scalars(select(Conversation.id).where(Conversation.task_id == task.id)))
    for cid in conv_ids:
        items.append(_file_item("conversation", workspace / ".opx-chat" / f"conv-{cid}",
                                conversation_id=cid))

    runs = list(db.scalars(select(Run).where(Run.task_id == task.id).order_by(Run.id)))
    for run in runs:
        item = _run_dir_item(run, workspace, task)
        if item is not None:
            items.append(item)
    return items


def _count(db: Session, model, where) -> int:
    return int(db.scalar(select(func.count()).select_from(model).where(where)) or 0)


def counts(db: Session, task: Task) -> dict[str, int]:
    run_ids = select(Run.id).where(Run.task_id == task.id)
    conv_ids = select(Conversation.id).where(Conversation.task_id == task.id)
    return {
        "spec_versions": _count(db, SpecVersion, SpecVersion.task_id == task.id),
        "loops": _count(db, Loop, Loop.task_id == task.id),
        "optimization_ideas": _count(db, OptimizationIdea, OptimizationIdea.task_id == task.id),
        "idea_searches": _count(db, IdeaSearch, IdeaSearch.task_id == task.id),
        "loop_budget_audits": _count(db, LoopBudgetAudit, LoopBudgetAudit.task_id == task.id),
        "agent_jobs": _count(db, AgentJob, AgentJob.task_id == task.id),
        "runs": _count(db, Run, Run.task_id == task.id),
        "artifacts": _count(db, Artifact, Artifact.run_id.in_(run_ids)),
        "action_jobs": _count(db, ActionJob, ActionJob.task_id == task.id),
        "stage_events": _count(db, StageEvent, StageEvent.task_id == task.id),
        # 版本存档（savepoints v1）：只删行；归档文件内容寻址、可能被其它任务引用，v1 不清（GC 放 v2）
        "savepoints": _count(db, Savepoint, Savepoint.task_id == task.id),
        "conversations": _count(db, Conversation, Conversation.task_id == task.id),
        "messages": _count(db, Message, Message.conversation_id.in_(conv_ids)),
        "attachments": _count(db, Attachment, Attachment.conversation_id.in_(conv_ids)),
        # 任务生命周期 v2：版本历史 + 任务级文档
        "task_versions": _count(db, TaskVersion, TaskVersion.task_id == task.id),
        "task_documents": _count(db, TaskDocument, TaskDocument.task_id == task.id),
    }


def _stale_agent_job(job: AgentJob) -> bool:
    """DB 里标着 running、但本进程已没有它的 state 且已超过总超时 → 僵尸（不阻断删除）。

    僵尸来源：后台 task 被异常杀掉/进程崩溃（lifespan 只在启动时兜重启，运行中崩溃没人兜）。
    """
    if agent_runner.running(job.id) is not None:
        return False  # 本进程确实还在跑
    started = job.started_at
    if started is None:
        return False
    if started.tzinfo is None:  # SQLite 读回来是 naive；写库一律 UTC
        started = started.replace(tzinfo=timezone.utc)
    timeout = max(1, int(settings.agent_timeout or 0))
    return (datetime.now(timezone.utc) - started).total_seconds() > timeout


def running_jobs(db: Session, task: Task) -> dict[str, list[int]]:
    """会阻断删除的运行中作业（动作作业 + 智能体调用）。

    智能体调用做僵尸兜底：DB running 但没有本进程 state 且已超 `agent_timeout` 的，视为 stale，
    不阻断删除（否则后台 task 崩掉后任务永久 409、无法自救）。
    """
    action_jobs = list(db.scalars(
        select(ActionJob.id).where(ActionJob.task_id == task.id, ActionJob.status == "running")
    ))
    agent_jobs = [
        job.id for job in db.scalars(
            select(AgentJob).where(AgentJob.task_id == task.id, AgentJob.status == "running")
        )
        if not _stale_agent_job(job)
    ]
    return {"action_jobs": action_jobs, "agent_jobs": agent_jobs}


def preview(db: Session, task: Task) -> dict:
    items = _cleanup_items(db, task)
    return {
        "task_id": task.id,
        "title": task.title,
        "op_name": task.op_name,
        "is_draft": task.is_draft,
        "stage": task.stage,
        "counts": counts(db, task),
        "running": running_jobs(db, task),
        "files": {
            "items": items,
            "total_bytes": sum(item["bytes"] for item in items if item["removable"]),
        },
    }


def _delete_files(items: list[dict]) -> dict:
    deleted: list[str] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    for item in items:
        path = Path(item["path"])
        if not item.get("removable"):
            if item.get("exists"):
                skipped.append({"path": str(path), "reason": item.get("reason") or "不可清理"})
            continue
        try:
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            deleted.append(str(path))
        except OSError as e:
            errors.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
    return {"deleted": deleted, "skipped": skipped, "errors": errors}


def delete(db: Session, task: Task, *, delete_files: bool = False) -> dict:
    """硬删除任务及所有从属记录；文件清理是 best-effort（失败不回滚库）。"""
    task_id = task.id
    before = counts(db, task)
    file_items = _cleanup_items(db, task) if delete_files else []

    # 先把 current_spec_version_id 摘掉：否则（开启外键时）删 spec_versions 会撞 FK。
    task.current_spec_version_id = None
    db.flush()

    run_ids = select(Run.id).where(Run.task_id == task_id)
    conv_ids = list(db.scalars(select(Conversation.id).where(Conversation.task_id == task_id)))

    db.execute(sql_delete(Artifact).where(Artifact.run_id.in_(run_ids)))
    # P2/P3 记录分别引用 Loop / IdeaSearch / AgentJob，先删从属行再删主体。
    db.execute(sql_delete(OptimizationIdea).where(OptimizationIdea.task_id == task_id))
    db.execute(sql_delete(LoopBudgetAudit).where(LoopBudgetAudit.task_id == task_id))
    db.execute(sql_delete(IdeaSearch).where(IdeaSearch.task_id == task_id))
    # action_jobs.run_id 指向 runs：先删作业，再删 run。
    db.execute(sql_delete(ActionJob).where(ActionJob.task_id == task_id))
    db.execute(sql_delete(Run).where(Run.task_id == task_id))
    if conv_ids:
        db.execute(sql_delete(Message).where(Message.conversation_id.in_(conv_ids)))
        db.execute(sql_delete(Attachment).where(Attachment.conversation_id.in_(conv_ids)))
        db.execute(sql_delete(Conversation).where(Conversation.id.in_(conv_ids)))
    db.execute(sql_delete(AgentJob).where(AgentJob.task_id == task_id))
    db.execute(sql_delete(StageEvent).where(StageEvent.task_id == task_id))
    # savepoints 显式删（SQLite 默认没开 PRAGMA foreign_keys，不能只靠 cascade）；
    # 归档文件不在这里删（内容寻址、可能被别的行/任务引用）。
    db.execute(sql_delete(Savepoint).where(Savepoint.task_id == task_id))
    # 任务生命周期 v2：版本历史引用 spec_versions/loops，必须在它们之前删；文档行直接属任务。
    db.execute(sql_delete(TaskVersion).where(TaskVersion.task_id == task_id))
    db.execute(sql_delete(TaskDocument).where(TaskDocument.task_id == task_id))
    db.execute(sql_delete(SpecVersion).where(SpecVersion.task_id == task_id))
    db.execute(sql_delete(Loop).where(Loop.task_id == task_id))
    db.execute(sql_delete(Task).where(Task.id == task_id))
    db.commit()

    files = {"requested": delete_files, "deleted": [], "skipped": [], "errors": []}
    if delete_files:
        files = {"requested": True, **_delete_files(file_items)}
    return {"task_id": task_id, "deleted": True, "counts": before, "files": files}
