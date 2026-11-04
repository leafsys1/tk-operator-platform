"""智能体调用（agent_jobs）与真实执行（runs）—— 两条线，一套 API。

命名（v1.1）：`agent_jobs` = 一次智能体调用（dsh 会话）；`runs` = 一次真实执行。
事件流对的是 **agent job**（跑的是 dsh 会话），不是真实执行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..db import get_db
from ..models import (ACTIONS, ActionJob, AgentJob, Artifact, Conversation, Loop,
                       LoopConfirmation, Message, Project, Run, SpecVersion, Task)
from ..schemas import (ActionJobOut, ActionRequest, LoopConfirmRequest, LoopConversationOut,
                       LoopStartRequest, OptimizeChatRequest, OptimizeChatStartOut,
                       OptimizationConversationRequest, OptimizeRequest, OptimizeStartOut,
                       RunCreate, RunOut, RunPatch)
from ..services import (action_runner, agent_runner, exec_actions, gate, idea_search,
                        loop_budget, loop_orchestrator, loop_state, optimize, optimization,
                        perf_diff, pricing, remote_cancel, savepoints, workspace_lease)

router = APIRouter(prefix="/api", tags=["runs"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 智能体调用


@router.get("/agent-jobs/{job_id}/events")
async def job_events(job_id: int) -> StreamingResponse:
    state = agent_runner.REGISTRY.get(job_id)
    if state is None:
        raise HTTPException(404, "该 agent job 不在本进程的运行表里（可能已完成或服务重启过）")

    async def gen():
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        buffered = list(state.events)
        state.subscribers.append(q)
        try:
            for ev in buffered:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if state.done:
                yield "event: done\ndata: {}\n\n"
                return
            while True:
                ev = await q.get()
                if ev is None:
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            if q in state.subscribers:
                state.subscribers.remove(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/agent-jobs/{job_id}/delta")
async def job_delta(job_id: int, payload: dict) -> dict:
    """收 dsh 插件（`opx-typewriter`）POST 回来的**逐 token 增量**，并入该 job 的事件流。

    插件是纯只读观察者（hook `agent/assistant-stream`），不干扰 agent loop；它把 text/reason
    增量发到这里，控制台经 `ProseRelay` 切掉 JSON 尾后，作为 `{"type": "delta"}` 事件写给
    正在订阅 `GET /api/agent-jobs/{job_id}/events` 的前端。**插件不发 stdout/stderr**（那被 JSONL 占用）。
    """
    state = agent_runner.REGISTRY.get(job_id)
    if state is None:
        raise HTTPException(404, "该 agent job 不在本进程的运行表里（可能已完成或服务重启过）")
    if state.done:  # 跑完之后的迟到增量：不是错误，丢掉即可（插件不看返回值）
        return {"ok": False, "why": "job finished"}
    kind = str(payload.get("k") or "")
    if kind in ("text", "reason"):
        state.push_delta(kind, str(payload.get("t") or ""))
    return {"ok": True, "text_deltas": state.text_deltas}


@router.get("/agent-jobs/{job_id}/live")
def job_live(job_id: int) -> dict:
    """给前端轮询用的轻量状态（SSE 断了也能看进度）。"""
    state = agent_runner.REGISTRY.get(job_id)
    if state is None:
        return {"running": False}
    return {
        "running": not state.done,
        "exit_code": state.exit_code,
        "session_id": state.session_id,
        "final": state.final,
        "prose": state.prose,
        "text_deltas": state.text_deltas,
        "error": state.error,
        "usage": state.usage,
        "turns": state.turns,
        "tool_calls": state.tool_calls,
        "events": len(state.events),
    }


@router.post("/agent-jobs/{job_id}/cancel")
def cancel_agent_job(job_id: int, db: Session = Depends(get_db)) -> dict:
    """取消一次智能体调用（kill dsh 子进程）。

    这是「任务卡在 对话中/运行中、DELETE 又 409」时的自救入口（审查 #1）：
    真正杀进程在 `agent_runner.cancel()`，这里只做存在性 / 已完成校验并把结果回给前端。
    """
    job = db.get(AgentJob, job_id)
    if not job:
        raise HTTPException(404, "作业不存在")
    if job.status != "running":
        raise HTTPException(409, f"作业 #{job_id} 不在运行（{job.status}），无需取消")
    result = agent_runner.cancel(job_id)
    return {"result": result}


# ---------------------------------------------------------------- 真实执行（四动作契约）


@router.get("/actions")
def list_actions() -> dict:
    """四个动作的契约说明（前端/文档/脚本共用一份）。"""
    return {
        "actions": [
            {
                "action": key,
                "title": spec["title"],
                "entry": spec["entry"],
                "requires": list(spec["requires"]),
                "needs_device": spec["needs_device"],
                "artifact_types": list(spec["artifact_types"]),
                "note": spec["remote_note"],
            }
            for key, spec in exec_actions.ACTION_SPECS.items()
        ],
        "required_artifacts": {k: list(v) for k, v in gate.REQUIRED_ARTIFACTS.items()},
    }


@router.post("/tasks/{task_id}/actions")
async def plan_action(task_id: int, payload: ActionRequest, db: Session = Depends(get_db)) -> dict:
    """算一次真实执行的 plan，并（`dry_run=false` 时）**真的起一次异步执行**。

    plan 包含前置检查的真结论：设备占用（共享机）、镜像能不能跑这个动作、判据档、源码快照。
    `dry_run=true`（默认）只返回 plan、不碰远端；`dry_run=false` 会：
    ⓪ 带 loop 时先做 P3 预算预留（写 allowed 审计，启动失败会回收）→
    ① 再确认一遍前置齐备 → ② 建一条 `action_jobs` → ③ 起子进程跑 `scripts/nm_test.sh` →
    ④ 脚本自己登记 run → ⑤ 跑完回填 run_id + 门禁 verdict。进度看 `GET /api/action-jobs/{id}`。
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    loop: Loop | None = None
    if payload.loop_id is not None:
        loop = db.get(Loop, payload.loop_id)
        if not loop or loop.task_id != task_id:
            raise HTTPException(400, f"loop {payload.loop_id} 不存在或不属于任务 #{task_id}")
    try:
        plan = exec_actions.build_plan(db, task, payload.action, device_id=payload.device_id,
                                       timeout_s=payload.timeout_s)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    out = plan.as_dict()
    if payload.dry_run:
        return out
    if not plan.ready:
        raise HTTPException(409, "前置不齐，先补齐：" + "；".join(plan.missing))
    budget: dict = {}
    if loop is not None:
        # Step 2：自动链运行中禁止手工插动作（否则抢设备/工作区，链会 E_AUTO_BUSY）。
        try:
            auto = loop_orchestrator.running_for_task(db, task_id)
        except Exception:  # noqa: BLE001
            auto = None
        if auto is not None:
            raise HTTPException(409, f"自动验证链 #{auto.id} 正在跑，等它结束再手动执行动作")
        # 并发 busy 必须**先于**预算审计：已经有一个作业在跑时，这次点击根本起不来，
        # 先写 allowed 审计就等于白扣一格额度（审查 M1）。
        busy = action_runner.running_for_task(task_id)
        if busy:
            raise HTTPException(409, f"任务 #{task_id} 已有正在执行的动作（作业 #{busy[0]}），等它结束再发")
        try:
            budget = loop_budget.check_and_audit(db, task, loop, payload.action)
        except loop_budget.BudgetExceeded as e:
            raise HTTPException(409, str(e)) from e
        out["budget"] = budget
    # 预算审计先 commit（= 预留）；启动失败（前置没齐的 race / busy）要回收这格额度，
    # 否则一次无效点击就把预算烧掉、用户无从重试（审查 M1）。
    budget_audit_id = budget.get("audit_id")
    try:
        job = action_runner.start(db, task, plan, loop_id=payload.loop_id,
                                  budget_audit_id=budget_audit_id)
    except ValueError as e:
        loop_budget.void_audit(db, budget_audit_id)
        raise HTTPException(409, str(e)) from e
    return {**out, "action_job_id": job.id, "run_ref": job.run_ref}


@router.get("/action-jobs/{job_id}", response_model=ActionJobOut)
def get_action_job(job_id: int, db: Session = Depends(get_db)) -> dict:
    """读一次动作作业的状态与日志尾部（运行中给实时日志）。"""
    row = db.get(ActionJob, job_id)
    if not row:
        raise HTTPException(404, "作业不存在")
    return action_runner.view(row)


@router.get("/tasks/{task_id}/action-jobs", response_model=list[ActionJobOut])
def list_action_jobs(task_id: int, limit: int = 20, db: Session = Depends(get_db)) -> list[dict]:
    """任务最近的执行作业（倒序）。"""
    if not db.get(Task, task_id):
        raise HTTPException(404, "任务不存在")
    rows = db.scalars(
        select(ActionJob).where(ActionJob.task_id == task_id).order_by(ActionJob.id.desc()).limit(limit)
    )
    return [action_runner.view(r) for r in rows]


@router.post("/action-jobs/{job_id}/cancel")
def cancel_action_job(job_id: int, db: Session = Depends(get_db)) -> dict:
    """取消一次动作作业：先杀本地进程组，再清远端容器（不等脚本 timeout）。"""
    job = db.get(ActionJob, job_id)
    if not job:
        raise HTTPException(404, "作业不存在")
    local = action_runner.cancel(job_id)
    task = db.get(Task, job.task_id)
    remote = remote_cancel.cancel_container(db, task, job.run_ref) if task else {"ok": False, "why": "任务不存在"}
    return {"result": local, "remote": remote}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    """取消一次真实执行：把 `docker rm -f opx-<run_ref>` 送到远端。

    只杀本地会话不够 —— 容器会继续跑并占卡（P1.5-a 实测泄漏过两次）。
    """
    row = db.get(Run, run_id)
    if not row:
        raise HTTPException(404, "run 不存在")
    task = db.get(Task, row.task_id)
    remote = remote_cancel.cancel_container(db, task, row.remote_workdir)
    if remote.get("ok") and row.status == "running":
        row.status = "interrupted"
        row.note = ((row.note + "；") if row.note else "") + "console 取消（远端容器已清理）"
        db.commit()
    return {"run_id": run_id, "remote": remote}


@router.get("/runs/{run_id}/gate")
def gate_report(run_id: int, write: bool = False, db: Session = Depends(get_db)) -> dict:
    """门禁校验器 v1：对这次真实执行给结论（结果有效 + 业务达标 + 版本一致）。

    判据与 fail code 清单见 `docs/gate-report-v1.md`。`write=true` 会把 gate_report.json
    落到该 run 的产物目录（与 manifest 在一起）。**退出码不是结论**。
    """
    row = db.get(Run, run_id)
    if not row:
        raise HTTPException(404, "run 不存在")
    return gate.evaluate(row, write=write)


@router.get("/runs/{run_id}/report")
def run_report(run_id: int, type: str = "correctness", db: Session = Depends(get_db)) -> dict:
    """读回一次真实执行的**机读报告**内容（结果视图画图用）。

    `type` ∈ `correctness` / `performance` / `metrics`（就是 manifest 里的产物类型）。
    只读、不改：内容原样返回（含报告里的版本对齐字段），供前端渲染逐用例/指标表与图。
    """
    row = db.get(Run, run_id)
    if not row:
        raise HTTPException(404, "run 不存在")
    paths = gate.artifact_paths(row)
    art = paths.get(type)
    if art is None:
        raise HTTPException(404, f"run #{run_id} 没有 {type} 产物（该动作不产出它？）")
    if art.suffix != ".json":
        raise HTTPException(409, f"{type} 产物不是机读文件：{art.name}")
    if not art.is_file():
        raise HTTPException(404, f"产物文件不存在：{art}")
    try:
        report = json.loads(art.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(422, f"{type} 报告不是合法 JSON：{e}") from e
    return {"run_id": row.id, "action": row.action, "type": type,
            "path": str(art), "report": report}


@router.get("/artifacts/{artifact_id}/text")
def artifact_text(artifact_id: int, limit: int = 64 * 1024, db: Session = Depends(get_db)) -> dict:
    """读产物文本（构建日志 / 报告预览用；二进制或超大明确标记）。"""
    art = db.get(Artifact, artifact_id)
    if not art:
        raise HTTPException(404, "产物不存在")
    path = _safe_artifact_path(db, art)
    if not path.is_file():
        raise HTTPException(404, f"产物文件不在磁盘上：{path}")
    limit = max(1024, min(int(limit), 512 * 1024))
    size = path.stat().st_size
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"id": art.id, "path": str(path), "binary": True, "text": None, "size_bytes": size}
    return {"id": art.id, "path": str(path), "binary": False,
            "size_bytes": size, "truncated": size > limit, "text": text[:limit]}


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """下载一个产物文件（交付/复现用）。路径不存在时明确 404（产物可能已被清理）。"""
    art = db.get(Artifact, artifact_id)
    if not art:
        raise HTTPException(404, "产物不存在")
    path = _safe_artifact_path(db, art)
    if not path.is_file():
        raise HTTPException(404, f"产物文件不在磁盘上（可能已被清理）：{path}")
    return FileResponse(str(path), filename=path.name)


def _allowed_run_roots(db: Session, task: Task) -> list[Path]:
    """本机 run 目录允许的根：仓库 `runs/` 与项目工作区 `runs/`（与 nm_test.sh 落点一致）。"""
    roots = [(Path(settings.repo_root) / "runs").resolve()]
    project = db.get(Project, task.project_id)
    if project and project.workspace_path:
        roots.append((Path(project.workspace_path).expanduser() / "runs").resolve())
    return roots


def _within_roots(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _safe_artifact_path(db: Session, art: Artifact) -> Path:
    """读取时再做一次 fail-closed：产物只能位于该 run 的 manifest 目录内。

    兼容历史与内部 delivery 行的绝对路径，但 resolve 后必须仍在同一 run
    目录下；因此即使旧 DB 里已有恶意路径，也不能透过预览/下载读宿主文件。
    """
    run = art.run or db.get(Run, art.run_id)
    if run is None or not run.manifest_path:
        raise HTTPException(409, "产物缺少可验证的 run manifest 目录")
    try:
        root = Path(str(run.manifest_path)).expanduser().parent.resolve()
        raw = Path(str(art.path)).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise HTTPException(403, "产物路径越界，已拒绝读取") from None
    return resolved


def _check_artifact_refs(data: dict, arts: list[dict]) -> None:
    """对外 run 登记只接受 manifest 目录下的相对产物路径。"""
    if not arts:
        return
    manifest_path = data.get("manifest_path")
    if not manifest_path:
        raise HTTPException(400, "登记产物时必须提供 manifest_path")
    try:
        root = Path(str(manifest_path)).expanduser().parent.resolve()
    except OSError as exc:
        raise HTTPException(400, f"manifest_path 不可解析：{exc}") from exc
    for item in arts:
        raw = Path(str(item.get("path") or ""))
        if not str(raw) or raw.is_absolute():
            raise HTTPException(400, "artifact.path 必须是 run 目录内的相对路径")
        try:
            (root / raw).resolve().relative_to(root)
        except (OSError, ValueError):
            raise HTTPException(400, f"artifact.path 越界：{raw}") from None


def _check_run_refs(db: Session, task: Task, data: dict) -> None:
    """登记/改归属前的校验：关联必须属于同一个任务，run 目录必须在允许的 runs/ 根下。

    两个风险（审查 #13 / #7）：跨任务的 `loop_id`/`spec_version_id` 会污染任务页的阶段与
    性能证据；任意路径的 `manifest_path` 会让「删除任务时同时删本机产物」误删别的目录。
    """
    loop_id = data.get("loop_id")
    if loop_id is not None:
        loop = db.get(Loop, loop_id)
        if loop is None or loop.task_id != task.id:
            raise HTTPException(400, f"loop {loop_id} 不存在或不属于任务 #{task.id}")
    spec_version_id = data.get("spec_version_id")
    if spec_version_id is not None:
        spec = db.get(SpecVersion, spec_version_id)
        if spec is None or spec.task_id != task.id:
            raise HTTPException(400, f"spec_version {spec_version_id} 不存在或不属于任务 #{task.id}")
    roots = _allowed_run_roots(db, task)
    remote_workdir = data.get("remote_workdir")
    # 正常登记时 remote_workdir 只是 run_id（如 r1-20261021-120000），没有路径分隔符；
    # 带路径的写法（本机 runs/<ref>）才做根目录校验，避免把远端路径误判为越界。
    if remote_workdir:
        text = str(remote_workdir)
        if os.path.isabs(text) or "/" in text or "\\" in text:
            if not _within_roots(Path(text), roots):
                raise HTTPException(400, f"remote_workdir 不在允许的 runs/ 根目录下：{text}")
    manifest_path = data.get("manifest_path")
    if manifest_path:
        if not _within_roots(Path(str(manifest_path)).expanduser().parent, roots):
            raise HTTPException(400, f"manifest_path 不在允许的 runs/ 根目录下：{manifest_path}")


@router.post("/tasks/{task_id}/runs", response_model=RunOut, status_code=201)
def register_run(task_id: int, payload: RunCreate, db: Session = Depends(get_db)) -> Run:
    """手工登记一次真实执行（自动执行接通前的过渡手段，也给脚本回填用）。"""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    # 自定义命令可记入执行台账，但不属于可调度的正式验证动作。
    # 不扩展 ACTIONS，否则它会混入动作入口与阶段门禁。
    record_actions = [*ACTIONS, "custom"]
    if payload.action not in record_actions:
        raise HTTPException(400, f"未知动作 {payload.action}；合法动作：{', '.join(record_actions)}")
    data = payload.model_dump()
    arts = data.pop("artifacts", []) or []
    data["task_id"] = task_id
    _check_run_refs(db, task, data)
    _check_artifact_refs(data, arts)
    row = Run(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    for a in arts:
        db.add(Artifact(run_id=row.id, **{k: v for k, v in a.items() if k in {"type", "path", "sha256", "size_bytes"}}))
    db.commit()
    db.refresh(row)
    return row


@router.patch("/runs/{run_id}", response_model=RunOut)
def patch_run(run_id: int, payload: RunPatch, db: Session = Depends(get_db)) -> Run:
    """只允许把已有 run 挂到某个 Loop 上（M3-a：优化前/后的成对关联）。

    真实执行的数据（产物、版本、退出码）一律不许改 —— 只改「归属哪个 Loop」这一个关联字段。
    """
    row = db.get(Run, run_id)
    if not row:
        raise HTTPException(404, "run 不存在")
    data = payload.model_dump(exclude_unset=True)
    if "loop_id" in data and data["loop_id"] is not None:
        loop = db.get(Loop, data["loop_id"])
        if loop is None or loop.task_id != row.task_id:
            raise HTTPException(400, f"loop {data['loop_id']} 不存在或不属于该 run 的任务 #{row.task_id}")
    if "loop_id" in data:
        row.loop_id = data["loop_id"]
    db.commit()
    db.refresh(row)
    return row


@router.get("/loops/{loop_id}/perf-diff")
def loop_perf_diff(loop_id: int, write: bool = False, db: Session = Depends(get_db)) -> dict:
    """自动优化闭环的对比报告与判定（方案 §13-A5）。

    优化 Loop（`loops.kind=perf`）里优化前/后各有一次 `test` + `benchmark`；本端点配对两侧、
    逐用例对比耗时与精度，出 `performance-diff.json` 与 `verdict`。`write=true` 会把报告落到
    优化后那次 benchmark 的产物目录。口径见 `services/perf_diff.py`。
    """
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "loop 不存在")
    runs = optimization.loop_runs(db, loop_id)
    return perf_diff.evaluate_loop(loop, runs, write=write)


# ---------------------------------------------------------------- 自动优化：agent 改成代码（M3 余项）


def _optimize_workspace(db: Session, loop: Loop) -> tuple[Task, Project, Path]:
    task = db.get(Task, loop.task_id)
    project = db.get(Project, task.project_id) if task else None
    if not project or not project.workspace_path:
        raise HTTPException(400, "项目的 workspace_path 未配置")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise HTTPException(400, f"工作区不存在：{workspace}")
    return task, project, workspace


#: 会占用任务工作区的作业类型（同一任务同时只允许一个）：
#: 对话/开发/优化都会改工作区，不能并发跑（与 conversations.send 的 guard 同口径）。
_WORKSPACE_JOB_KINDS = ("对话", "开发", "优化")


def _prepare_optimize(db: Session, loop_id: int) -> tuple[Loop, Task, Project, Path]:
    """自动优化的公共前置校验：loop/objective、工作区、key、dsh、同 loop 并发。"""
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    # 旧 Loop（无 objective）只读（审查 M5）：没有 objective 就没有 allowed_paths 与预算约束，
    # 自动优化改完代码越界也不会回滚 —— 拒绝，让用户按新建轮次流程重开一轮。
    if not loop.objective:
        raise HTTPException(409, "旧 Loop（无 objective）已改为只读；请新建优化轮次后再自动优化")
    task, project, workspace = _optimize_workspace(db, loop)
    # G1 基线门禁（docs/plan-optimize-auto-v1.md §3.1）：没有同条件 before test+benchmark
    # 就不允许改码；否则后面 A/B 没有可信的“优化前”侧。后端硬拦，不能只靠页面按钮 disabled。
    try:
        optimization.require_baseline(db, loop)
    except optimization.OptimizationError as e:
        raise HTTPException(e.status, str(e)) from e
    if settings.llm_key_env and not os.environ.get(settings.llm_key_env):
        raise HTTPException(400, "缺少凭据：先在「API 管理」页配置 key（或设置环境变量 "
                                 f"{settings.llm_key_env}）")
    if not shutil.which("dsh"):
        raise HTTPException(400, "找不到 dsh 可执行文件（控制台进程的 PATH 里要有）")
    if any(ctx.get("loop_id") == loop_id for ctx in optimize.PENDING.values()):
        raise HTTPException(409, "这一轮已有正在跑的自动优化，先等它结束")
    return loop, task, project, workspace


def _auto_capture_before_optimize(db: Session, task: Task, workspace: Path, loop_id: int) -> None:
    """自动归档：agent 改代码前留一份可回滚的源码（best-effort，失败只 log）。"""
    try:
        savepoints.auto_capture(db, task, workspace, label="自动优化改前自动存档")
    except Exception:  # noqa: BLE001 —— 自动归档失败不能阻断起优化
        logger.exception("自动优化改前自动归档失败（task=%s, loop=%s）", task.id, loop_id)


def _running_workspace_job(db: Session, task_id: int) -> AgentJob | None:
    return db.scalar(
        select(AgentJob).where(AgentJob.task_id == task_id, AgentJob.status == "running",
                               AgentJob.kind.in_(_WORKSPACE_JOB_KINDS)).limit(1)
    )


def _acquire_agent_lease(db: Session, task: Task) -> workspace_lease.WorkspaceLease:
    """改码前占用项目工作区（DB 级单飞，Step 1；agent/action 共用同一租约）。"""
    try:
        return workspace_lease.acquire(db, task, kind="agent")
    except workspace_lease.LeaseBusy as e:
        raise HTTPException(409, str(e)) from e


def _release_agent_lease(db: Session, task: Task, job_id: int | None = None) -> None:
    workspace_lease.release(db, task_id=task.id, kind="agent", job_id=job_id, commit=True)


def _fail_agent_start(db: Session, task: Task, loop: Loop, job: AgentJob, error: str) -> None:
    """起 dsh 失败：job 标 failed、phase 回 failed、释放租约（不留 running 僵尸）。

    同时清掉进程内的 `optimize.PENDING` / `agent_runner.REGISTRY`：它们只在 `_finish` 里 pop，
    启动就失败时不清会把该 Loop 永久判成「已有正在跑的自动优化」（之后任何启动都 409）。
    """
    optimize.PENDING.pop(job.id, None)
    agent_runner.REGISTRY.pop(job.id, None)
    db.refresh(loop)
    job.status = "failed"
    job.ended_at = datetime.now(timezone.utc)
    loop_state.transition(db, loop, "failed", event="start_failed", force=True,
                          active_job_id=None, error=error[:2000], commit=False)
    workspace_lease.release(db, task_id=task.id, kind="agent", job_id=job.id, commit=False)
    db.commit()


@router.post("/loops/{loop_id}/optimize", response_model=OptimizeStartOut, status_code=201)
async def start_loop_optimize(loop_id: int, payload: OptimizeRequest,
                              db: Session = Depends(get_db)) -> dict:
    """让 agent 按本轮 Loop 的假设**自动改算子代码**（M3 余项；「agent 提议、工具判定」）。

    起一次 dsh agent job（`kind=优化`、挂到 Loop），跑在**项目工作区**（故能读 `.dsh/skills`）。
    跑完把它改了什么（改前/改后快照 + 逐文件 diff + 结构化摘要）落进 `loops.optimize_json`。
    **不跑 build/test/benchmark** —— 真机执行仍照 `plan.command` 单独触发（见 `services/optimize.py`）。
    页面已改为走 `POST /loops/{loop_id}/optimize/chat`（对话化），本端点保留兼容/脚本调用。
    """
    loop, task, project, workspace = _prepare_optimize(db, loop_id)
    lease = _acquire_agent_lease(db, task)
    dsh_home = Path(project.dsh_home or (workspace.parent / ".dsh")).expanduser()
    dsh_home.mkdir(parents=True, exist_ok=True)
    try:
        _auto_capture_before_optimize(db, task, workspace, loop.id)
        reference, ref_id = optimize.reference_text(db, loop, payload.reference_loop_id)
        prompt = optimize.build_prompt(db=db, task=task, loop=loop, workspace=workspace,
                                       reference=reference, extra=payload.extra)
        job = AgentJob(task_id=task.id, loop_id=loop_id, kind="优化", status="running",
                       prompt=prompt[:6000])
        pricing.attach_endpoint(db, job)
        db.add(job)
        db.flush()
        # 状态机 CAS：两个并发请求只有一个能从 baseline_ready 进入 agent_running；
        # 同时 active_job_id 非空也挡住重复启动。
        if not loop_state.start_agent(db, loop, job.id, commit=False):
            db.rollback()
            raise HTTPException(409, "轮次已有正在跑的改码作业或状态不允许启动，请刷新后再试")
        lease.job_id = job.id
        db.commit()
        db.refresh(job)
    except HTTPException:
        _release_agent_lease(db, task)
        raise
    except Exception:
        db.rollback()
        _release_agent_lease(db, task)
        raise

    try:
        optimize.start(job.id, prompt, loop_id=loop_id, workspace=workspace, dsh_home=dsh_home,
                       reference_loop_id=ref_id)
    except Exception as exc:  # noqa: BLE001 —— 起不来要收尾 job/phase/租约
        _fail_agent_start(db, task, loop, job, f"自动优化启动失败：{type(exc).__name__}: {exc}")
        raise HTTPException(500, f"自动优化启动失败：{type(exc).__name__}: {exc}") from exc
    return {"job_id": job.id, "loop_id": loop_id, "prompt": prompt[:6000]}


def _start_chat_optimize(db: Session, *, loop_id: int, instruction: str | None,
                         conversation_id: int | None,
                         reference_loop_id: int | None = None,
                         existing_message_id: int | None = None,
                         confirmation: LoopConfirmation | None = None,
                         actor: str = "page") -> dict:
    """对话化自动优化的公共启动实现（/optimize/chat、/start、/tasks/{id}/optimization-chat 共用）。

    - `confirmation` 非空：启动前在同一事务里 CAS 消费一次性 token（失败整体回滚）；
    - 为空：legacy 路径，补一条 `confirm_auto` 事件留痕（页面已改走 confirm → start）；
    - `existing_message_id`：对话入口已先落用户消息时复用，不重复写一条。
    """
    loop, task, project, workspace = _prepare_optimize(db, loop_id)
    # 同一任务的工作区作业不能并发：对话/开发/优化任一在跑都拒绝（本端点自身也是"优化"）。
    running = _running_workspace_job(db, task.id)
    if running is not None:
        raise HTTPException(
            409,
            f"任务 #{task.id} 已有 {running.kind} 作业在跑（job #{running.id}），等它结束再发"
            "（可在任务页取消）",
        )

    lease = _acquire_agent_lease(db, task)
    try:
        conv = None
        if conversation_id is not None:
            conv = db.get(Conversation, conversation_id)
            if conv is None or conv.task_id != task.id:
                raise HTTPException(
                    400, f"会话 #{conversation_id} 不存在或不属于任务 #{task.id}")
            if conv.mode != "task":
                raise HTTPException(400, "自动优化只能挂在任务工作区的会话（mode=task）上")
        if conv is None:
            detail = ""
            if isinstance(loop.objective, dict):
                detail = str(loop.objective.get("label") or loop.objective.get("focus") or "").strip()
            detail = (detail or loop.scope or "")[:60]
            title = f"自动优化 · Loop #{loop.id}" + (f" · {detail}" if detail else "")
            conv = Conversation(task_id=task.id, project_id=project.id, mode="task",
                                title=title[:300], last_active_at=datetime.now(timezone.utc))
            db.add(conv)
            db.flush()

        instruction = (instruction or "").strip()
        message = db.get(Message, existing_message_id) if existing_message_id is not None else None
        if message is not None and message.conversation_id != conv.id:
            raise HTTPException(400, f"消息 #{existing_message_id} 不属于会话 #{conv.id}")
        if message is None:
            message = Message(conversation_id=conv.id, role="user",
                              text=instruction or "（未提供额外参考指令）按本轮目标开始自动优化",
                              created_at=datetime.now(timezone.utc))
            db.add(message)

        dsh_home = Path(project.dsh_home or (workspace.parent / ".dsh")).expanduser()
        dsh_home.mkdir(parents=True, exist_ok=True)
        _auto_capture_before_optimize(db, task, workspace, loop.id)

        reference, ref_id = optimize.reference_text(db, loop, reference_loop_id)
        prompt = optimize.build_prompt(db=db, task=task, loop=loop, workspace=workspace,
                                       reference=reference,
                                       extra=instruction or None)
        job = AgentJob(task_id=task.id, loop_id=loop.id, conversation_id=conv.id,
                       kind="优化", status="running", prompt=prompt[:6000])
        pricing.attach_endpoint(db, job)
        db.add(job)
        db.flush()
        # 与 legacy 端点同一把状态机 CAS：并发/重复启动只有一个能进 agent_running。
        if not loop_state.start_agent(db, loop, job.id, conversation_id=conv.id, actor=actor,
                                      commit=False):
            db.rollback()
            raise HTTPException(409, "轮次已有正在跑的改码作业或状态不允许启动，请刷新后再试")
        if confirmation is not None:
            # token 消费与 job/phase 同一事务：CAS 失败（别的请求先用了）则整体回滚。
            optimization.consume_confirmation(db, confirmation, job.id)
        else:
            loop_state.add_event(db, loop.id, "confirm_auto", phase="agent_running",
                                 actor=actor,
                                 payload={"instruction": bool(instruction),
                                          "reference_loop_id": ref_id})
        lease.job_id = job.id
        conv.last_active_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(job)
        db.refresh(message)
        db.refresh(conv)
    except HTTPException:
        _release_agent_lease(db, task)
        raise
    except optimization.OptimizationError as e:
        # 确认 token 失效/被并发消费：整个事务回滚，不消耗 token、不建 job。
        db.rollback()
        _release_agent_lease(db, task)
        raise HTTPException(e.status, str(e)) from e
    except Exception:
        db.rollback()
        _release_agent_lease(db, task)
        raise

    try:
        optimize.start(job.id, prompt, loop_id=loop.id, workspace=workspace, dsh_home=dsh_home,
                       reference_loop_id=ref_id, conversation_id=conv.id,
                       session_id=conv.dsh_session_id)
    except Exception as exc:  # noqa: BLE001 —— 起不来要收尾 job/会话，不留 running 僵尸
        db.add(Message(conversation_id=conv.id, role="assistant", job_id=job.id,
                       text=f"自动优化启动失败：{type(exc).__name__}: {exc}",
                       created_at=datetime.now(timezone.utc)))
        _fail_agent_start(db, task, loop, job, f"自动优化启动失败：{type(exc).__name__}: {exc}")
        raise HTTPException(500, f"自动优化启动失败：{type(exc).__name__}: {exc}") from exc

    return {"job_id": job.id, "loop_id": loop.id, "conversation_id": conv.id,
            "message_id": message.id, "prompt": prompt[:6000]}


@router.post("/loops/{loop_id}/optimize/chat", response_model=OptimizeChatStartOut,
             status_code=201)
async def start_loop_optimize_chat(loop_id: int, payload: OptimizeChatRequest,
                                   db: Session = Depends(get_db)) -> dict:
    """对话化自动优化（legacy 入口，兼容脚本；页面已改走 /confirm + /start）。

    传了 `confirm_token` 就按一次性确认消费；不传按 legacy 处理并写 `confirm_auto` 事件。
    """
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    confirmation = None
    draft: dict | None = None
    if (payload.confirm_token or "").strip():
        try:
            confirmation, draft = optimization.validate_confirmation(
                db, loop, payload.confirm_token or "")
            optimization.require_confirmation_baseline(db, loop, draft)
        except optimization.OptimizationError as e:
            raise HTTPException(e.status, str(e)) from e
    # 带确认 token 时与 /start 同语义：指令/参考轮次只取 draft，请求体不能覆盖。
    instruction = (payload.instruction or "").strip() or None
    reference_loop_id = None
    if draft is not None:
        instruction = (draft.get("instruction")
                       if isinstance(draft.get("instruction"), str) else None)
        reference_loop_id = (draft.get("reference_loop_id")
                            if isinstance(draft.get("reference_loop_id"), int) else None)
    return _start_chat_optimize(
        db, loop_id=loop_id, instruction=instruction,
        conversation_id=payload.conversation_id, reference_loop_id=reference_loop_id,
        confirmation=confirmation, actor="chat",
    )


@router.post("/loops/{loop_id}/confirm", status_code=201)
def confirm_optimization_round(loop_id: int, payload: LoopConfirmRequest,
                               db: Session = Depends(get_db)) -> dict:
    """确认优化草案（Step 1b）：冻结目标/基线/指令 + 发一次性 token（10 分钟）。

    `POST /api/loops/{id}/start` 必须带这个 token，且只能成功一次；确认后目标/基线被改动，
    start 会 409（`E_CONFIRM_OBJECTIVE_CHANGED` / `E_CONFIRM_BASELINE_CHANGED`）。
    """
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    try:
        result = optimization.confirm_draft(
            db, loop, instruction=payload.instruction,
            reference_loop_id=payload.reference_loop_id, note=payload.note, actor="page")
    except optimization.OptimizationError as e:
        db.rollback()
        raise HTTPException(e.status, str(e)) from e
    result.pop("row", None)
    db.commit()
    return result


@router.post("/loops/{loop_id}/start", response_model=OptimizeChatStartOut, status_code=201)
async def start_optimization_round(loop_id: int, payload: LoopStartRequest,
                                   db: Session = Depends(get_db)) -> dict:
    """按已确认草案启动改码：校验 + 消费一次性 token，然后走与 /optimize/chat 相同的启动链。

    **必须是 async def**：`agent_runner.start` 内部 `asyncio.create_task` 起消费协程，
    同步端点会被 FastAPI 丢进线程池（没有 running loop）→ 500 且白烧 token。
    """
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    try:
        confirmation, draft = optimization.validate_confirmation(db, loop, payload.token or "")
        optimization.require_confirmation_baseline(db, loop, draft)
    except optimization.OptimizationError as e:
        raise HTTPException(e.status, str(e)) from e
    instruction = draft.get("instruction") if isinstance(draft.get("instruction"), str) else None
    return _start_chat_optimize(
        db, loop_id=loop_id, instruction=instruction,
        conversation_id=payload.conversation_id,
        reference_loop_id=draft.get("reference_loop_id"),
        confirmation=confirmation, actor="page",
    )


@router.get("/loops/{loop_id}/attempts")
def loop_attempts(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """该轮次的 agent 改码尝试历史（Step 1b；最新在前）。"""
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    return {"loop_id": loop.id, "attempts": loop_state.attempts_view(db, loop.id)}
def _direction_title(text: str) -> str:
    """把对话首句压成思路标题；完整原文仍保存在 evidence/objective 中。"""
    first = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    first = re.sub(r"^(请|麻烦|帮我|我想|希望|建议)?\s*(针对|重点)?\s*", "", first)
    first = re.sub(r"^(优化一下|优化|改进|提升)\s*[:：,，-]?\s*", "", first).strip()
    return (first or text.strip())[:300]


def _ensure_direction_conversation(db: Session, task: Task, project: Project, *,
                                   conversation_id: int | None, direction: str) -> tuple[Conversation, Message]:
    conv = db.get(Conversation, conversation_id) if conversation_id is not None else None
    if conversation_id is not None and (conv is None or conv.task_id != task.id or conv.mode != "task"):
        raise HTTPException(400, f"会话 #{conversation_id} 不存在、不是任务会话或不属于任务 #{task.id}")
    if conv is None:
        conv = Conversation(task_id=task.id, project_id=project.id, mode="task",
                            title=(f"优化对话 · {_direction_title(direction)}")[:300],
                            last_active_at=datetime.now(timezone.utc))
        db.add(conv)
        db.flush()
    message = Message(conversation_id=conv.id, role="user", text=direction,
                      created_at=datetime.now(timezone.utc))
    db.add(message)
    conv.last_active_at = datetime.now(timezone.utc)
    db.flush()
    return conv, message


@router.post("/tasks/{task_id}/optimization-chat", status_code=201)
async def create_optimization_from_conversation(
    task_id: int, payload: OptimizationConversationRequest, db: Session = Depends(get_db)
) -> dict:
    """自然语言方向 → 思路池 → 独立 Loop →（基线齐全时）自动启动 Agent。

    这里不会自动采纳，也不会在缺少基线时先改代码：没有同快照 test+benchmark 时只把方向和
    Loop 记好，明确停在 ``waiting_baseline``，从而保住同设备/镜像/规格/用例集的 A/B 链路。
    """
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        raise HTTPException(409, "项目工作区未配置，无法发起优化")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise HTTPException(409, f"工作区不存在：{workspace}")
    if _running_workspace_job(db, task.id) is not None:
        running = _running_workspace_job(db, task.id)
        assert running is not None
        raise HTTPException(409, f"任务 #{task.id} 已有 {running.kind} 作业在跑（job #{running.id}）")

    direction = payload.direction.strip()
    if not direction:
        raise HTTPException(422, "优化方向不能只包含空白字符")
    conv, message = _ensure_direction_conversation(
        db, task, project, conversation_id=payload.conversation_id, direction=direction)
    idea, duplicate = idea_search.add_user(db, task, {
        "title": _direction_title(direction),
        "evidence": {
            "kind": "conversation_direction",
            "conversation_id": conv.id,
            "message_id": message.id,
            "user_text": direction,
        },
        "hypothesis": direction,
        "change_scope": f"ops/{task.op_name}/" if task.op_name else None,
    }, source="conversation")
    if duplicate and idea.loop_id is not None:
        loop = db.get(Loop, idea.loop_id)
        db.add(Message(conversation_id=conv.id, role="assistant",
                       text=f"已识别为已有优化方向 #{idea.id}，对应 Loop #{idea.loop_id}，未重复创建。",
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        return {
            "status": "deduplicated", "reason": "同一优化方向已存在",
            "conversation_id": conv.id, "message_id": message.id,
            "job_id": None, "idea": idea_search.idea_view(idea),
            "loop": optimization.loop_view(loop) if loop is not None else None,
        }

    try:
        created = idea_search.create_round(
            db, task, idea, objective=None, attach_baseline_runs=True)
    except idea_search.IdeaSearchError as exc:
        db.rollback()
        # “一思路一 Loop”是串行门禁，不应丢掉用户刚说的下一个方向：留在思路池排队。
        if exc.status == 409 and idea.loop_id is None and idea.status == "proposed" and "串行验证" in str(exc):
            reason = f"优化方向已记录；{exc}"
            db.add(Message(conversation_id=conv.id, role="assistant", text=reason,
                           created_at=datetime.now(timezone.utc)))
            db.commit()
            return {
                "status": "waiting_previous", "reason": reason,
                "conversation_id": conv.id, "message_id": message.id, "job_id": None,
                "idea": idea_search.idea_view(idea), "loop": None,
            }
        raise HTTPException(exc.status, str(exc)) from exc
    except optimization.OptimizationError as exc:
        db.rollback()
        raise HTTPException(getattr(exc, "status", 400), str(exc)) from exc

    attached = {str(row.get("action")) for row in created.get("attached_runs", [])}
    loop = db.get(Loop, int(created["loop"]["id"]))
    assert loop is not None
    if not {"test", "benchmark"}.issubset(attached):
        reason = "优化方向已记录并创建轮次；请先在本轮跑完同快照基线 test + benchmark，再启动 Agent"
        db.add(Message(conversation_id=conv.id, role="assistant", text=reason,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        return {
            **created, "status": "waiting_baseline", "reason": reason,
            "conversation_id": conv.id, "message_id": message.id, "job_id": None,
        }

    try:
        started = _start_chat_optimize(
            db, loop_id=loop.id, instruction=direction,
            conversation_id=conv.id, existing_message_id=message.id, actor="conversation",
        )
    except HTTPException as exc:
        # 方向和 Loop 已经是有价值的持久化结果；dsh/key 暂不可用时保留并明确排队，不回滚它们。
        reason = f"优化方向和基线已就绪；Agent 暂未启动：{exc.detail}"
        db.add(Message(conversation_id=conv.id, role="assistant", text=reason,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        return {
            **created, "status": "waiting_agent", "reason": reason,
            "conversation_id": conv.id, "message_id": message.id, "job_id": None,
        }
    return {
        **created, "status": "started", "reason": "已识别优化方向并启动 Agent",
        "conversation_id": started["conversation_id"], "message_id": started["message_id"],
        "job_id": started["job_id"],
    }


@router.get("/loops/{loop_id}/conversation", response_model=LoopConversationOut)
def loop_conversation(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """某轮次最近一次自动优化对应的会话（「在对话中查看」用；没有则 null）。"""
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    conversation_id = db.scalar(
        select(AgentJob.conversation_id)
        .where(AgentJob.loop_id == loop_id, AgentJob.kind == "优化",
               AgentJob.conversation_id.is_not(None))
        .order_by(AgentJob.id.desc()).limit(1)
    )
    return {"conversation_id": conversation_id}


@router.get("/loops/{loop_id}/optimize")
def loop_optimize_evidence(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """读回这一轮自动优化的证据（改前/改后快照、逐文件 diff、摘要、回退数据）。"""
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    return optimize.evidence(loop)


@router.post("/loops/{loop_id}/optimize/revert")
def loop_optimize_revert(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """把工作区还原到这一轮改前（把「优化不予采纳」变成一键回退）。"""
    loop = db.get(Loop, loop_id)
    if not loop:
        raise HTTPException(404, "轮次不存在")
    task, _project, workspace = _optimize_workspace(db, loop)
    if any(ctx.get("loop_id") == loop_id for ctx in optimize.PENDING.values()):
        raise HTTPException(409, "这一轮正在跑自动优化，先等它结束再回退")
    auto = loop_orchestrator.running_for_task(db, task.id)
    if auto is not None:
        raise HTTPException(409, f"自动验证链 #{auto.id} 正在跑，等它结束再回退改动")
    try:
        # 回退会写工作区：持项目级租约，避免与 agent/action 同时改同一份代码。
        with workspace_lease.hold(db, task, kind="restore",
                                  note=f"轮次 #{loop_id} 一键回退改前"):
            return optimize.revert(db, loop, workspace)
    except workspace_lease.LeaseBusy as e:
        raise HTTPException(409, f"工作区正被占用，等它结束再回退：{e}") from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)) -> Run:
    row = db.scalar(select(Run).options(selectinload(Run.artifacts)).where(Run.id == run_id))
    if not row:
        raise HTTPException(404, "run 不存在")
    return row
