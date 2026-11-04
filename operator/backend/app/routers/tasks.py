"""项目 / 任务 / 规格版本 / 轮次 / 手工登记的真实执行。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..db import SessionLocal, get_db
from ..models import AgentJob, Host, HostImage, Loop, LoopRunLink, Project, Run, SpecVersion, StageEvent, Task
from ..schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamOut,
    FinalizeResponse,
    LoopCreate,
    LoopOut,
    LoopRunLinkOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    SpecVersionCreate,
    SpecVersionOut,
    TaskCreate,
    TaskDetail,
    TaskOut,
    TaskUpdate,
    validate_op_name,
)
from ..services import (agent_runner, case_report, chat, context_space, exec_actions, gate, pricing,
                        stageboard, task_deletion, versions, workspace_view)

router = APIRouter(prefix="/api", tags=["tasks"])


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return list(db.scalars(select(Project).order_by(Project.id)))


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    if db.scalar(select(Project).where(Project.name == payload.name)):
        raise HTTPException(409, f"项目 {payload.name} 已存在")
    if payload.host_id is not None and db.get(Host, payload.host_id) is None:
        raise HTTPException(400, f"执行机 #{payload.host_id} 不存在")
    if payload.image_id is not None:
        image = db.get(HostImage, payload.image_id)
        if image is None:
            raise HTTPException(400, f"镜像 #{payload.image_id} 不存在")
        if payload.host_id is None:
            raise HTTPException(400, "设置默认镜像前必须先设置执行机")
        if image.host_id != payload.host_id:
            raise HTTPException(400, f"镜像 #{image.id} 不在执行机 #{payload.host_id} 上")
        if image.dangling:
            raise HTTPException(400, "悬空镜像不能设为项目默认")
    row = Project(**payload.model_dump())
    db.add(row)
    db.commit()
    return row


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)) -> Project:
    """改项目默认配置（执行机 / 镜像 / 工作区 / 判据档）——「把某镜像设为项目默认」的入口。"""
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "项目不存在")
    data = payload.model_dump(exclude_unset=True)

    if "name" in data and data["name"] != project.name:
        if db.scalar(select(Project).where(Project.name == data["name"], Project.id != project_id)):
            raise HTTPException(409, f"项目 {data['name']} 已存在")

    if "host_id" in data and data["host_id"] is not None:
        from ..models import Host

        if db.get(Host, data["host_id"]) is None:
            raise HTTPException(400, f"执行机 #{data['host_id']} 不存在")
    target_host_id = data.get("host_id", project.host_id)
    # host/image 是一对配置。即使本次只改 host，也必须复核保留下来的旧 image，
    # 否则会得到「执行机 B + 机器 A 的镜像」这种永远无法执行的项目状态。
    target_image_id = data.get("image_id", project.image_id)
    if target_image_id is not None:
        image = db.get(HostImage, target_image_id)
        if image is None:
            raise HTTPException(400, f"镜像 #{target_image_id} 不存在")
        if target_host_id is None:
            raise HTTPException(400, "保留默认镜像时不能清空执行机；请同时清空 image_id")
        if image.host_id != target_host_id:
            raise HTTPException(400, f"镜像 #{image.id} 不在执行机 #{target_host_id} 上")
        if image.dangling:
            raise HTTPException(400, "悬空镜像不能设为项目默认")

    for key, value in data.items():
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(draft: bool | None = None, include_archived: bool = False,
               db: Session = Depends(get_db)) -> list[Task]:
    """任务列表。`?draft=true|false` 只看草稿 / 只看正式任务；不传则两者都返回。

    草稿（`is_draft=1`）不该混进「阶段统计」（UI-task-start-v1 §2.3），但列表里要能看到、能接着编。
    归档任务默认隐藏；`include_archived=true` 可用于历史查询，不删除任何证据。
    """
    q = select(Task).order_by(Task.id.desc())
    if not include_archived:
        q = q.where(Task.is_archived.is_(False))
    if draft is not None:
        q = q.where(Task.is_draft == draft)
    return list(db.scalars(q))


@router.post("/tasks", response_model=TaskDetail, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)) -> Task:
    project = db.get(Project, payload.project_id)
    if not project:
        raise HTTPException(400, "project_id 不存在")
    data = payload.model_dump()
    if not data.get("stage"):
        data.pop("stage", None)
    _check_task_refs(db, data, project=project)
    row = Task(**data)
    db.add(row)
    db.commit()
    return _task_detail(db, row.id)


@router.patch("/tasks/{task_id}", response_model=TaskDetail)
def update_task(task_id: int, payload: TaskUpdate, db: Session = Depends(get_db)) -> Task:
    """存草稿 / 回填 / 转正（`is_draft=false`）。

    只改显式给的字段（`exclude_unset`）：不传的字段保持原值，显式传 `null` 才清空
    （例：把覆盖的执行机清掉 = 回到项目默认）。
    """
    row = db.get(Task, task_id)
    if not row:
        raise HTTPException(404, "任务不存在")
    project = db.get(Project, row.project_id)
    data = payload.model_dump(exclude_unset=True)
    if data.get("stage") is None:
        data.pop("stage", None)
    if data.get("is_draft") is True:
        # 草稿是「还没产生任何证据」的创建态；已有规格版本/执行记录的任务不许打回草稿
        # （否则它会从阶段统计/正式列表里消失，但 spec_versions/runs 还在）。
        has_spec = db.scalar(select(SpecVersion.id).where(SpecVersion.task_id == task_id).limit(1))
        has_run = db.scalar(select(Run.id).where(Run.task_id == task_id).limit(1))
        if row.current_spec_version_id is not None or has_spec is not None or has_run is not None:
            raise HTTPException(
                409, "任务已有规格版本/执行记录，不能打回草稿；改规格请走 POST "
                     f"/api/tasks/{task_id}/specs 落新版本")
    if "op_name" in data and data["op_name"] != row.op_name and row.current_spec_version_id is not None:
        # 算子名是规格版本的身份：定版后改它会让「规格 vN / OPX_OP / 上下文空间目录」三者漂移。
        raise HTTPException(
            409, f"任务已定版（当前规格 #{row.current_spec_version_id}），不能再改算子名；"
                 f"改规格请走 POST /api/tasks/{task_id}/specs 落新版本")
    _check_task_refs(db, data, task=row, project=project)
    for key, value in data.items():
        setattr(row, key, value)
    db.commit()
    return _task_detail(db, row.id)


@router.get("/tasks/{task_id}/delete-preview")
def task_delete_preview(task_id: int, db: Session = Depends(get_db)) -> dict:
    """删除前预览：会删哪些记录、哪些本机目录、有没有运行中的作业。只读。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task_deletion.preview(db, task)


@router.delete("/tasks/{task_id}")
def delete_task(task_id: int, delete_files: bool = False,
                db: Session = Depends(get_db)) -> dict:
    """硬删除任务及其从属记录；`delete_files=true` 时顺带清理任务专属本机目录。

    运行中的作业不删：先让用户取消/等待，避免删库后容器还在远端占卡。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    running = task_deletion.running_jobs(db, task)
    if running["action_jobs"] or running["agent_jobs"]:
        parts = []
        if running["action_jobs"]:
            parts.append("动作作业 " + "、".join(f"#{i}" for i in running["action_jobs"]))
        if running["agent_jobs"]:
            parts.append("智能体调用 " + "、".join(f"#{i}" for i in running["agent_jobs"]))
        raise HTTPException(409, "任务还有运行中的作业（" + "；".join(parts)
                            + "）—— 先调取消接口（POST /api/agent-jobs/{id}/cancel、"
                              "POST /api/action-jobs/{id}/cancel）或等结束再删")
    return task_deletion.delete(db, task, delete_files=delete_files)


def _check_task_refs(db: Session, data: dict, *, task: Task | None = None,
                     project: Project | None = None) -> None:
    """校验任务上的覆盖指针：给了值就得真实存在，且镜像必须与**生效执行机**同机。

    「生效」= 请求显式值 > 任务现值 > 项目默认（与 `exec_actions.resolve_host/resolve_image`
    的优先级一致）。只校验单字段会放过「任务覆盖 host 但沿用项目默认 image（在另一台机）」的组合。
    """
    effective_host = data["host_id"] if "host_id" in data else (task.host_id if task else None)
    effective_image = data["image_id"] if "image_id" in data else (task.image_id if task else None)
    if project is not None:
        if effective_host is None:
            effective_host = project.host_id
        if effective_image is None:
            effective_image = project.image_id

    host = None
    if effective_host is not None:
        host = db.get(Host, effective_host)
        if not host:
            raise HTTPException(400, f"host_id {effective_host} 不存在")
    if effective_image is not None:
        image = db.get(HostImage, effective_image)
        if not image:
            raise HTTPException(400, f"image_id {effective_image} 不存在")
        if host is not None and image.host_id != host.id:
            raise HTTPException(400, f"镜像 #{image.id} 不在执行机 #{host.id} 上"
                                     "（host_id 与 image_id 必须属于同一台机器）")


# ---------------------------------------------------------------- 对话式需求澄清（P0-3 第二步）


def _load_json(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


def _draft_parts(task: Task) -> tuple[dict, list, bool]:
    """当前规格草稿 → `(fields, questions, ready)`（草稿为空就是 `({}, [], False)`）。"""
    draft = _load_json(task.spec_draft, {}) or {}
    if not isinstance(draft, dict):
        draft = {}
    fields = draft.get("fields")
    questions = draft.get("questions")
    return (fields if isinstance(fields, dict) else {},
            questions if isinstance(questions, list) else [],
            bool(draft.get("ready")))


def _design_doc_text(db: Session, task: Task) -> str | None:
    """创建期对话注入：任务上传的《算子设计文档》解析稿（task 生命周期 v2）。

    有文档 → 把 `task-docs/task-<id>/<stem>.extracted.md` 全文交给 `chat.build_prompt`（它负责截断）；
    没有/读不到 → None（保持旧行为）。只读，不改文档；路径越界一律忽略。
    """
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        return None
    ws = Path(project.workspace_path).expanduser()
    try:
        from ..services import documents
        rows = documents.list_for_task(db, task)
    except Exception:  # noqa: BLE001 —— 文档模块异常不能让创建期对话 500
        return None
    for row in rows:
        if (getattr(row, "kind", None) or "design") != "design":
            continue
        rel = getattr(row, "extracted_path", None) or getattr(row, "path", None)
        if not rel:
            continue
        try:
            path = (ws / rel).resolve()
            if path != ws.resolve() and ws.resolve() not in path.parents:
                continue
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


def _apply_parsed(task: Task, history: list, parsed: dict, spec: dict, questions: list,
                  reply: str | None = None, *, draft_ready: bool = False):
    """把模型返回的结果合进任务：规格草稿 / 对话 / 镜像字段 / 需求原文。返回 (reply, spec, questions, ready)。

    `reply` 是**自然语言回复**（新契约里 = 流式上屏的那段 prose）；不传则退回 JSON 里的 `reply` 字段（旧格式）。

    **定版后（`current_spec_version_id` 不为空）只追加过程记录**：`chat_log` 记这一轮，
    `spec_draft` / `raw_input` / `op_name` / `target_chip` 一律不再改写 —— 规格以
    `spec_versions` 为唯一真相源，避免聊天旁路出一份「永不生效」的规格草稿。
    定版后仍允许聊天（问实现、追问历史都合理），只是不产生规格变更。
    """
    reply = (reply if reply is not None else str(parsed.get("reply") or "")).strip()
    finalized = task.current_spec_version_id is not None
    if finalized:
        history.append({"role": "assistant", "text": reply})
        task.chat_log = json.dumps(history[-200:], ensure_ascii=False)
        # 返回当前已定版规格（不是本轮模型解析的草稿），前端不要把它当成当前规格
        return reply, spec, questions, bool(draft_ready)
    new_spec = parsed.get("spec") if isinstance(parsed.get("spec"), dict) else spec
    new_spec = dict(new_spec)
    new_questions = parsed.get("questions") if isinstance(parsed.get("questions"), list) else questions
    new_questions = list(new_questions)
    ready = bool(parsed.get("ready"))
    # 模型输出同样是不可信输入。op_name 会参与 cases/ops/上下文空间路径，必须复用 API
    # 的单段名称校验；非法值不能落库，更不能让 ready=true 把任务定版成不可恢复状态。
    candidate = new_spec.get("op_name") or task.op_name
    try:
        valid_op_name = validate_op_name(candidate)
    except ValueError:
        valid_op_name = None
    if valid_op_name:
        new_spec["op_name"] = valid_op_name
    else:
        new_spec.pop("op_name", None)
        ready = False
        if not any(isinstance(q, dict) and q.get("id") == "op_name" for q in new_questions):
            new_questions.append({"id": "op_name", "text": "请确认合法的算子名（单段名称，不能含路径分隔符）"})
    history.append({"role": "assistant", "text": reply})
    task.spec_draft = json.dumps({"fields": new_spec, "questions": new_questions, "ready": ready},
                                 ensure_ascii=False)
    task.chat_log = json.dumps(history[-200:], ensure_ascii=False)
    if valid_op_name:
        # 算子名是规格版本的身份（OPX_OP 与上下文空间目录都按它推导）：定版后不再被对话改写，
        # 否则 /context 与 delete_files 会指向另一个目录。改算子名只能走新规格版本。
        task.op_name = valid_op_name
        if not task.title or task.title == "未命名任务":
            task.title = task.op_name
    if new_spec.get("target_chip"):
        task.target_chip = str(new_spec["target_chip"])[:80]
    task.raw_input = "\n".join(m["text"] for m in history if m.get("role") == "user")[:20000]
    return reply, new_spec, new_questions, ready


def _chat_paths(project: Project, task_id: int) -> tuple[Path, Path, Path]:
    """(workspace, chat_cwd, dsh_home)。会话目录刻意是空目录、不挂 skills，让澄清轻快。"""
    workspace = Path(project.workspace_path).expanduser()
    chat_cwd = workspace / ".opx-chat" / f"task-{task_id}"
    chat_cwd.mkdir(parents=True, exist_ok=True)
    dsh_home = Path(project.dsh_home or (workspace.parent / ".dsh")).expanduser()
    dsh_home.mkdir(parents=True, exist_ok=True)
    return workspace, chat_cwd, dsh_home


def _stream_on_done(task_id: int):
    """dsh 跑完后的回调：解析 final → 落这一轮（用独立 session，因为请求早已返回）。"""
    def on_done(state) -> None:
        if state.error or state.exit_code != 0:
            return
        try:
            reply, parsed = chat.parse_output(state.final)
        except chat.ChatError:
            return
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            if not task:
                return
            history = _load_json(task.chat_log, []) or []
            spec, questions, _ready = _draft_parts(task)
            _apply_parsed(task, history, parsed, spec, questions, reply=reply or None,
                          draft_ready=_ready)
            if state.session_id and not task.dsh_session_id:
                task.dsh_session_id = state.session_id
            db.commit()
    return on_done


def _running_clarify_job(db: Session, task_id: int) -> int | None:
    """该任务是否已有**真的在跑**的需求澄清作业（DB running 且本进程 REGISTRY 有未完成 state）。

    只看 DB 会把后台 task 崩溃/重启留下的僵尸行也算成「在跑」→ 任务永久不能继续对话；
    只看 REGISTRY 又挡不住另一进程刚建完 job 的窗口。两边都满足才算忙。
    """
    rows = db.scalars(
        select(AgentJob.id).where(AgentJob.task_id == task_id, AgentJob.kind == "需求澄清",
                                  AgentJob.status == "running")
    )
    for jid in rows:
        if agent_runner.running(jid) is not None:
            return jid
    return None


@router.post("/tasks/{task_id}/chat", response_model=ChatResponse)
async def chat_turn(task_id: int, payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """一步对话式需求澄清（**阻塞版**）：走 dsh 会话（`--session-id` 续会话），拿回回复/草稿/待确认项。

    每轮固定层由后端重拼（方案 §8.1）；每轮记一条 agent_job（过程与消耗，P4 用）。
    业务证据（规格草稿 → `tasks.spec_draft`）与过程证据（对话 → `tasks.chat_log`）分开存（方案 §1.1）。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(400, "message 不能为空")
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")

    spec, questions, _ready = _draft_parts(task)
    history = _load_json(task.chat_log, []) or []
    history.append({"role": "user", "text": message})

    prompt = chat.build_prompt(task.target_chip or "910B1", spec, questions, message,
                               doc_text=_design_doc_text(db, task))
    _ws, chat_cwd, dsh_home = _chat_paths(project, task_id)

    job = AgentJob(task_id=task_id, kind="需求澄清", status="running", prompt=prompt[:6000],
                   started_at=datetime.now(timezone.utc))
    pricing.attach_endpoint(db, job)
    db.add(job)
    db.commit()
    db.refresh(job)

    result = await chat.converse(prompt=prompt, cwd=chat_cwd, dsh_home=dsh_home,
                                 session_id=task.dsh_session_id, timeout=settings.chat_timeout)
    usage = result.usage or {}
    job.status = "ok" if result.ok else "failed"
    job.exit_code = result.exit_code
    job.dsh_session_id = result.session_id
    job.tokens_in = usage.get("inputTokens") or None
    job.tokens_out = usage.get("outputTokens") or None
    job.cache_read = usage.get("cacheReadTokens") or None
    job.cache_write = usage.get("cacheWriteTokens") or None
    job.ended_at = datetime.now(timezone.utc)
    if result.session_id and not task.dsh_session_id:
        task.dsh_session_id = result.session_id
    db.commit()

    if not result.ok or not result.parsed:
        raise HTTPException(502, result.error or "对话失败")

    reply, new_spec, new_questions, ready = _apply_parsed(
        task, history, result.parsed, spec, questions, reply=result.reply or None,
        draft_ready=_ready)
    db.commit()
    return ChatResponse(reply=reply, spec=new_spec, questions=new_questions, ready=ready,
                        messages=history, usage=usage or None)


@router.post("/tasks/{task_id}/chat/stream", response_model=ChatStreamOut)
async def chat_stream(task_id: int, payload: ChatRequest, db: Session = Depends(get_db)) -> ChatStreamOut:
    """对话式澄清（**流式版**）：起 dsh 后台任务并立刻返回 `job_id`。

    前端订阅 `GET /api/agent-jobs/{job_id}/events`，一条流上看两样东西：
    ① **逐 token 的文字**（`{"type":"delta"}`，由 `opx-typewriter` 插件回流 → 打字机）；
    ② **过程事件**（thinking / 步骤 / 工具）。
    dsh 跑完后由完成回调把这一轮落进任务。插件没生效时由 `emit_fallback_prose` 兜底补发整段答复。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(400, "message 不能为空")
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")
    if not agent_runner.available():
        # 没有 dsh 就别建 job：否则后台消费者起进程失败，任务页会永久「对话中」、还删不掉。
        raise HTTPException(503, "找不到 dsh 可执行文件（控制台进程的 PATH 里要有）—— "
                                 "无法启动流式对话，请先装好 dsh 或改用阻塞版 /chat")
    busy = _running_clarify_job(db, task_id)
    if busy is not None:
        # 同一任务同时只跑一轮澄清：双击/重试不会重复消费同一条消息，也不会开两条 SSE。
        raise HTTPException(409, f"上一轮需求澄清（job #{busy}）还在跑，等它结束再发；"
                                 f"卡住可调 POST /api/agent-jobs/{busy}/cancel")

    spec, questions, _ready = _draft_parts(task)
    history = _load_json(task.chat_log, []) or []
    history.append({"role": "user", "text": message})
    # 先把用户这条落库（失败也不丢）
    task.chat_log = json.dumps(history[-200:], ensure_ascii=False)
    task.raw_input = "\n".join(m["text"] for m in history if m.get("role") == "user")[:20000]

    prompt = chat.build_prompt(task.target_chip or "910B1", spec, questions, message,
                               doc_text=_design_doc_text(db, task))
    _ws, chat_cwd, dsh_home = _chat_paths(project, task_id)

    job = AgentJob(task_id=task_id, kind="需求澄清", status="running", prompt=prompt[:6000],
                   started_at=datetime.now(timezone.utc))
    pricing.attach_endpoint(db, job)
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        agent_runner.start(job.id, prompt, cwd=chat_cwd, dsh_home=dsh_home,
                           session_id=task.dsh_session_id, on_done=_stream_on_done(task_id),
                           stream=settings.chat_stream, prose_of=chat.prose_only)
    except Exception as e:  # 起后台消费者失败也要把 job 收死，不能留 running 行
        job.status = "failed"
        job.ended_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(503, f"启动 dsh 失败：{type(e).__name__}: {e}") from e
    return ChatStreamOut(job_id=job.id, messages=history)


# ---------------------------------------------------------------- 定版 + 上下文空间（P0-3 收尾）


@router.post("/tasks/{task_id}/finalize", response_model=FinalizeResponse)
def finalize_task(task_id: int, db: Session = Depends(get_db)) -> FinalizeResponse:
    """定版门禁：待澄清清零才放行 → 先落 `spec_versions` v1 → 再写上下文空间 → 草稿转正。

    写盘顺序遵循 §6「先库后盘」：spec_version 先落库拿到 id + 哈希，再回填进文件写盘。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")
    if task.current_spec_version_id is not None:
        # 幂等门禁：重复定版会落重复 spec 版本（同内容 v2），并把阶段/当前规格指针打回旧值。
        raise HTTPException(409, "任务已定版（当前规格版本已落库），不能重复定版；"
                                 f"改规格请走 POST /api/tasks/{task_id}/specs 落新版本")

    spec, questions, ready = _draft_parts(task)
    if not spec:
        raise HTTPException(400, "还没有规格草稿 —— 先在「任务开始」页把需求聊清楚")
    if not ready:
        raise HTTPException(409, f"定版门禁：待澄清项未清零（{len(questions)} 项），先回到对话把它们问清楚")
    try:
        op_name = validate_op_name(task.op_name or spec.get("op_name"))
    except ValueError as e:
        raise HTTPException(409, f"定版门禁：算子名不合法（{e}）；请先在任务草稿中修正") from e
    if not op_name:
        raise HTTPException(409, "定版门禁：缺少算子名；请先在任务草稿中补充 op_name")
    task.op_name = op_name
    spec = dict(spec)
    spec["op_name"] = op_name

    latest = db.scalar(
        select(SpecVersion).where(SpecVersion.task_id == task_id).order_by(SpecVersion.version.desc()).limit(1)
    )
    version = (latest.version + 1) if latest else 1
    spec_json = json.dumps(spec, ensure_ascii=False)
    unknowns = json.dumps([q.get("text") for q in questions if isinstance(q, dict)], ensure_ascii=False)
    assumptions = json.dumps(spec["assumptions"], ensure_ascii=False) if spec.get("assumptions") else None
    spec_row = SpecVersion(
        task_id=task_id, version=version, raw_input=task.raw_input, input_form=task.input_form,
        op_name=task.op_name, spec_json=spec_json, assumptions=assumptions, unknowns=unknowns,
        change_reason="由「任务开始」页对话定版" if not latest else "由「任务开始」页对话定版（新版本）",
        sha256=hashlib.sha256((spec_json + "|" + (assumptions or "")).encode()).hexdigest(),
    )
    db.add(spec_row)
    db.flush()

    criteria_ref, _src = exec_actions.resolve_criteria(task, project)
    workspace = Path(project.workspace_path).expanduser()
    chat_log = _load_json(task.chat_log, []) or []
    base, files = context_space.generate(
        workspace=workspace, task=task, fields=spec, questions=questions,
        spec_version=version, spec_version_id=spec_row.id, spec_sha=spec_row.sha256,
        criteria_ref=criteria_ref, chat_log=chat_log,
    )

    task.current_spec_version_id = spec_row.id
    task.is_draft = False
    # 任务生命周期 v2：创建 v<spec>.0 版本行（之后落新规格 / 采纳优化 / 手工变更再推进版本）。
    versions.create_spec_version(db, task, spec_row)
    # 定版完成 = 「需求与设计」段达成；按 stageboard 同一套证据推导写下一段（通常是「开发」），
    # 并写 stage_events 留痕。不再无条件写回「需求与设计」—— 那会让存值与推导必然不一致（审查 #3）。
    # v2：阶段一还要求用例组已确认，所以定版后通常停在「需求与设计」等用例确认。
    view = stageboard.build_task_view(
        task, _task_runs(db, task_id), perf=stageboard.perf_evidence(db, task), spec_version=version)
    derived = view["derived_stage"]
    if derived != task.stage:
        db.add(StageEvent(task_id=task_id, from_stage=task.stage, to_stage=derived, actor="console",
                          reason="定版：规格版本已落库，进入下一阶段"))
    task.stage = derived
    db.commit()
    db.refresh(spec_row)
    return FinalizeResponse(spec_version=spec_row, space_dir=str(base), files=files)


@router.get("/tasks/{task_id}/context")
def get_context(task_id: int, db: Session = Depends(get_db)) -> dict:
    """读上下文空间的文件清单（不存在就 exists=false）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")
    workspace = Path(project.workspace_path).expanduser()
    # 按 `.opx-task` 归属标记找目录（改过算子名 / 删过同 slug 任务后仍指向本任务的目录）
    base = context_space.find_space_dir(workspace, task)
    if not base.is_dir():
        return {"exists": False, "space_dir": str(base), "files": []}
    files = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != ".opx-task":
            data = p.read_bytes()
            files.append({"path": str(p.relative_to(base)), "sha256": hashlib.sha256(data).hexdigest(),
                          "size_bytes": len(data)})
    return {"exists": True, "space_dir": str(base), "files": files}


@router.get("/tasks/{task_id}", response_model=TaskDetail)
def get_task(task_id: int, db: Session = Depends(get_db)) -> TaskDetail:
    """任务详情（含每条 agent_jobs 的费用）。

    费用不落库：序列化后按端点价目实时算，与 `/api/usage` 同口径（见 `services/pricing.py`）。
    """
    task = _task_detail(db, task_id)
    payload = TaskDetail.model_validate(task)
    payload.loop_run_links = [LoopRunLinkOut.model_validate(link) for link in db.scalars(
        select(LoopRunLink).join(Loop, Loop.id == LoopRunLink.loop_id)
        .join(Run, Run.id == LoopRunLink.run_id)
        .where(Loop.task_id == task_id, Run.task_id == task_id)
        .order_by(LoopRunLink.id)
    )]
    endpoints = pricing.endpoints_by_id(db)
    active = pricing.active_endpoint(db)
    for src, out in zip(task.agent_jobs, payload.agent_jobs):
        out.cost, out.cost_estimated = pricing.job_cost(src, endpoints, active)
    return payload


def _task_detail(db: Session, task_id: int) -> Task:
    """任务详情 = 需求/规格 + 过程（agent_jobs）+ 结果（runs）+ 轮次。"""
    task = db.scalar(
        select(Task)
        .options(
            selectinload(Task.agent_jobs),
            selectinload(Task.runs).selectinload(Run.artifacts),
            selectinload(Task.spec_versions),
            selectinload(Task.loops),
        )
        .where(Task.id == task_id)
    )
    if not task:
        raise HTTPException(404, "任务不存在")
    task.agent_jobs.sort(key=lambda j: j.id)
    task.runs.sort(key=lambda r: r.id)
    task.spec_versions.sort(key=lambda s: s.version)
    task.loops.sort(key=lambda l: l.id)
    return task


# ---------------------------------------------------------------- 规格版本（SpecVersion）


@router.get("/tasks/{task_id}/specs", response_model=list[SpecVersionOut])
def list_specs(task_id: int, db: Session = Depends(get_db)) -> list[SpecVersion]:
    if not db.get(Task, task_id):
        raise HTTPException(404, "任务不存在")
    return list(db.scalars(select(SpecVersion).where(SpecVersion.task_id == task_id).order_by(SpecVersion.version)))


@router.post("/tasks/{task_id}/specs", response_model=SpecVersionOut, status_code=201)
def create_spec(task_id: int, payload: SpecVersionCreate, db: Session = Depends(get_db)) -> SpecVersion:
    """落一版规格。规格是代码生成/测试生成/验收的共同依据（v1.1 §4.3）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task.is_draft:
        raise HTTPException(409, "草稿任务不能直接落规格版本；请先完成创建期对话并调用 finalize 定版")
    # 算子名是「规格版本身份 + OPX_OP + 上下文空间目录名」的共同来源：定版后冻结，
    # 否则新规格版本一落，上下文空间目录名就与磁盘上的既有目录漂移（审查 #5 残项）。
    if (payload.op_name and task.current_spec_version_id is not None
            and payload.op_name != (task.op_name or "")):
        raise HTTPException(
            409, f"任务已定版，算子名已冻结（当前 {task.op_name!r}）；如需改名请新建任务，"
                 "或先手工迁移 tasks/<slug> 目录后再改")
    try:
        parsed_spec = json.loads(payload.spec_json or "")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"spec_json 不是合法 JSON：{e}") from e
    if not isinstance(parsed_spec, dict) or not parsed_spec:
        raise HTTPException(400, "spec_json 必须是非空 JSON 对象")
    latest = db.scalar(
        select(SpecVersion).where(SpecVersion.task_id == task_id).order_by(SpecVersion.version.desc()).limit(1)
    )
    body = (payload.spec_json or "") + "|" + (payload.assumptions or "")
    row = SpecVersion(
        task_id=task_id,
        version=(latest.version + 1) if latest else 1,
        spec_json=payload.spec_json,
        raw_input=payload.raw_input,
        input_form=payload.input_form or task.input_form,
        op_name=payload.op_name or task.op_name,
        assumptions=payload.assumptions,
        unknowns=payload.unknowns,
        change_reason=payload.change_reason or ("首版规格" if not latest else None),
        sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    db.add(row)
    db.flush()
    if payload.set_current:
        task.current_spec_version_id = row.id
        task.op_name = row.op_name or task.op_name
        # 任务生命周期 v2：落新规格 = 大版本 +1、小版本归 0、用例需重新确认
        versions.create_spec_version(db, task, row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------- 阶段看板（P3 读侧 + 按证据同步）


def _task_runs(db: Session, task_id: int) -> list[Run]:
    return list(db.scalars(select(Run).where(Run.task_id == task_id).order_by(Run.id)))


@router.get("/stages")
def stage_board(include_drafts: bool = False, db: Session = Depends(get_db)) -> dict:
    """阶段看板：所有任务按**推导阶段**分组（证据来自各动作最近一次 run 的门禁判定）。"""
    return stageboard.build_board(db, include_drafts=include_drafts)


@router.post("/tasks/{task_id}/stage/sync")
def sync_stage(task_id: int, db: Session = Depends(get_db)) -> dict:
    """按证据把 `tasks.stage` 同步到推导阶段，并写一条 `stage_events`（阶段推进留痕）。

    只改阶段这一件事；不改任何执行数据。已一致时返回 `changed=false`（不写事件）。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    runs = _task_runs(db, task_id)
    view = stageboard.build_task_view(task, runs, perf=stageboard.perf_evidence(db, task),
                                      spec_version=stageboard.spec_version_of(db, task))
    derived = view["derived_stage"]
    if derived == task.stage:
        return {"changed": False, "stage": task.stage, "view": view}
    prev = task.stage
    ok_stages = [s["stage"] for s in view["stages"] if s["ok"]]
    db.add(StageEvent(task_id=task_id, from_stage=prev, to_stage=derived, actor="console",
                      reason="按证据同步（已达成：" + ("、".join(ok_stages) or "无") + "）"))
    task.stage = derived
    db.commit()
    last = db.scalar(select(StageEvent).where(StageEvent.task_id == task_id)
                     .order_by(StageEvent.id.desc()).limit(1))
    runs = _task_runs(db, task_id)
    return {"changed": True, "from": prev, "to": derived,
            "view": stageboard.build_task_view(task, runs, last,
                                              perf=stageboard.perf_evidence(db, task),
                                              spec_version=stageboard.spec_version_of(db, task))}


# ---------------------------------------------------------------- 测试管理（P3 读侧）


def _workspace_of(db: Session, task: Task) -> Path:
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")
    ws = Path(project.workspace_path).expanduser()
    if not ws.is_dir():
        raise HTTPException(400, f"工作区不存在：{ws}")
    return ws


@router.get("/tasks/{task_id}/workspace")
def task_workspace(task_id: int, db: Session = Depends(get_db)) -> dict:
    """工作区文件清单（只读；跳过 .dsh/.cannbot/runs/build 等运行时大件）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    try:
        return workspace_view.list_files(_workspace_of(db, task))
    except workspace_view.WorkspaceError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/tasks/{task_id}/workspace/file")
def task_workspace_file(task_id: int, path: str, db: Session = Depends(get_db)) -> dict:
    """读一个工作区文件（文本；过大截断 / 二进制明确标记）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    try:
        return workspace_view.read_text(_workspace_of(db, task), path)
    except workspace_view.WorkspaceError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/tasks/{task_id}/workspace/diff")
def task_workspace_diff(task_id: int, db: Session = Depends(get_db)) -> dict:
    """工作区 git 状态 + 已跟踪文件 diff（只读；不是仓库就明确返回 is_repo=false）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return workspace_view.git_view(_workspace_of(db, task))


class ShellRequest(BaseModel):
    command: str
    timeout_s: int = 30


@router.post("/tasks/{task_id}/shell")
def task_shell(task_id: int, payload: ShellRequest, db: Session = Depends(get_db)) -> dict:
    """在工作区执行**一条**命令（非交互终端；30s 默认 / 60s 上限，输出截断）。

    安全边界：单人本机使用；环境变量里去掉疑似凭据（KEY/TOKEN/SECRET/PASSWORD），
    不提供 PTY / 常驻 shell。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    ws = _workspace_of(db, task)
    timeout = max(1, min(int(payload.timeout_s or 30), 60))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    started = time.monotonic()
    try:
        proc = subprocess.run(payload.command, shell=True, cwd=str(ws), capture_output=True,
                              text=True, timeout=timeout, env=env)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -1, (e.stdout or ""), (e.stderr or "") + f"\n超时（{timeout}s）"
    except OSError as e:
        raise HTTPException(400, f"命令执行失败：{e}") from e
    return {
        "command": payload.command,
        "exit_code": rc,
        "stdout": (out or "")[:64 * 1024],
        "stderr": (err or "")[:64 * 1024],
        "duration_ms": int((time.monotonic() - started) * 1000),
        "cwd": str(ws),
    }


@router.get("/tasks/{task_id}/stage")
def task_stage(task_id: int, db: Session = Depends(get_db)) -> dict:
    """单任务的阶段视图（与看板同一套推导：门禁 verdict + perf-diff）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    runs = _task_runs(db, task_id)
    return stageboard.build_task_view(task, runs, perf=stageboard.perf_evidence(db, task),
                                      spec_version=stageboard.spec_version_of(db, task))


@router.get("/tasks/{task_id}/gates")
def task_gates(task_id: int, db: Session = Depends(get_db)) -> dict:
    """与阶段看板同口径的当前证据门禁（页面直接展示，避免逐个 run 请求）。

    「交付」页与任务详情用它，替代「看产物齐不齐」的展示判定 —— 退出码不是结论。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    runs = _task_runs(db, task_id)
    view = stageboard.build_task_view(task, runs, perf=stageboard.perf_evidence(db, task),
                                     spec_version=stageboard.spec_version_of(db, task))
    out = []
    for action in ("build", "smoke", "test", "benchmark", "profile"):
        evidence = view["actions"].get(action)
        if evidence is None:
            # 未跑过该动作也要给完整形状：前端按 `fail_codes.length` 渲染，
            # 缺字段会在「没有任何 run」的任务详情上直接白屏。
            out.append({"action": action, "run_id": None, "verdict": None,
                        "fail_codes": [], "warn_codes": []})
            continue
        out.append({"action": action, **evidence})
    return {"task_id": task_id, "gates": out, "perf": view["perf"]}


@router.get("/tasks/{task_id}/cases")
def task_cases(task_id: int, db: Session = Depends(get_db)) -> dict:
    """用例矩阵 + 逐例结果 + 失败点（方案 §7「测试管理」）。

    精度：矩阵来自工作区 `cases/<op>/cases.json`，逐例结果来自各次 `test` run 的 `correctness.json`。
    性能（v2）：矩阵来自 `benchmark.json`，实测来自各次 `benchmark` run 的 `performance.json`。
    只读聚合，不改任何东西。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    project = db.get(Project, task.project_id)
    ws = Path(project.workspace_path).expanduser() if project and project.workspace_path else None
    runs = list(db.scalars(select(Run)
                           .where(Run.task_id == task_id, Run.action == "test")
                           .order_by(Run.id)))
    bench_runs = list(db.scalars(select(Run)
                                 .where(Run.task_id == task_id, Run.action == "benchmark")
                                 .order_by(Run.id)))
    criteria_ref, _src = exec_actions.resolve_criteria(task, project)
    return case_report.build(task, runs, ws, criteria_ref=criteria_ref,
                             benchmark_runs=bench_runs)


# ---------------------------------------------------------------- 轮次（Loop）

@router.get("/tasks/{task_id}/loops", response_model=list[LoopOut])
def list_loops(task_id: int, db: Session = Depends(get_db)) -> list[Loop]:
    if not db.get(Task, task_id):
        raise HTTPException(404, "任务不存在")
    return list(db.scalars(select(Loop).where(Loop.task_id == task_id).order_by(Loop.id)))


@router.post("/tasks/{task_id}/loops", response_model=LoopOut, status_code=201)
def create_loop(task_id: int, payload: LoopCreate, db: Session = Depends(get_db)) -> Loop:
    """开一轮（设计/修复/优化）：范围、假设、判定条件（v1.1 §3.1）。"""
    if not db.get(Task, task_id):
        raise HTTPException(404, "任务不存在")
    row = Loop(task_id=task_id, **payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/loops/{loop_id}", response_model=LoopOut)
def get_loop(loop_id: int, db: Session = Depends(get_db)) -> Loop:
    row = db.get(Loop, loop_id)
    if not row:
        raise HTTPException(404, "轮次不存在")
    return row


@router.post("/loops/{loop_id}/close", response_model=LoopOut)
def close_loop(loop_id: int, conclusion: str, status: str = "closed", db: Session = Depends(get_db)) -> Loop:
    row = db.get(Loop, loop_id)
    if not row:
        raise HTTPException(404, "轮次不存在")
    row.conclusion = conclusion
    row.status = status
    row.ended_at = datetime.now(timezone.utc)
    # 任务生命周期 v2：perf 轮次结论为 accepted → 小版本 +1（幂等）。采纳意味着候选已经
    # 跑过 test/benchmark，候选 run 的快照就是当前工作区 → 无需重跑即成为新版本证据。
    if (row.kind == "perf" and status == "closed"
            and str(conclusion or "").strip().lower().startswith("accepted")):
        task = db.get(Task, row.task_id)
        if task is not None:
            versions.accept_optimization(db, task, row)
    db.commit()
    db.refresh(row)
    return row
