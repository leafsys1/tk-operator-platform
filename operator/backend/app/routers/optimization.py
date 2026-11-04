"""优化轮次 v1（P1）API：目标解析 / 建轮次 / 采纳 / 否决 / 可复现报告。

契约见 `docs/plan-optimize-rounds-v1.md` §2–§7（字段名逐字对齐）。业务编排在
`services/optimization.py`；本文件只做存在性校验与错误码映射。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..db import get_db
from ..models import AgentJob, IdeaSearch, Loop, OptimizationIdea, Project, Task
from ..schemas import (IdeaSearchRequest, LoopAcceptRequest, LoopCancelRequest, LoopRejectRequest,
                       OptimizationIdeaCreate, OptimizationIdeaRoundCreate,
                       OptimizationIdeaStatusPatch, OptimizationRoundCreate)
from ..services import (agent_runner, idea_search, loop_budget, loop_orchestrator, loop_state,
                        optimization, optimization_reports, pricing, workspace_lease)

router = APIRouter(prefix="/api", tags=["optimization"])

#: PATCH 允许用户侧写入的状态；selected/validating/verified 由系统（建轮/采纳/否决）写入。
_USER_IDEA_STATUSES = idea_search.USER_IDEA_STATUSES
_SYSTEM_IDEA_STATUSES = idea_search.SYSTEM_IDEA_STATUSES


def _get_task(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return task


def _get_loop(db: Session, loop_id: int) -> Loop:
    loop = db.get(Loop, loop_id)
    if loop is None:
        raise HTTPException(404, "轮次不存在")
    return loop


def _http(e: optimization.OptimizationError) -> HTTPException:
    return HTTPException(getattr(e, "status", 400), str(e))


@router.get("/tasks/{task_id}/optimization-targets")
def optimization_targets(task_id: int, db: Session = Depends(get_db)) -> dict:
    """目标用例（数据驱动）：最近一次 ok 的 benchmark/test 报告里的 cases。"""
    task = _get_task(db, task_id)
    return optimization.targets(db, task)


@router.post("/tasks/{task_id}/optimization-rounds", status_code=201)
def create_optimization_round(task_id: int, payload: OptimizationRoundCreate,
                              db: Session = Depends(get_db)) -> dict:
    """建一个优化轮次：校验 objective → 建 Loop → 基线存档 → 复用同基线 sha 的 run。"""
    task = _get_task(db, task_id)
    try:
        return optimization.create_round(
            db, task, objective=payload.objective, label=payload.label,
            hypothesis=payload.hypothesis, attach_baseline_runs=payload.attach_baseline_runs,
            idempotency_key=payload.idempotency_key,
        )
    except optimization.OptimizationError as e:
        db.rollback()
        raise _http(e) from e


@router.get("/loops/{loop_id}/phase")
def loop_phase(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """轮次状态机（Step 1）：phase / state_version / active_job / 最近事件。

    页面与对话都从这里渲染，不再各自用 runs/jobs 拓扑推断。
    """
    return loop_state.view(db, _get_loop(db, loop_id))


@router.get("/loops/{loop_id}/baseline-evidence")
def loop_baseline_evidence(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """G1 基线就绪证据（前端展示“为什么不能开始优化”/复用条件）。"""
    return optimization.baseline_evidence(db, _get_loop(db, loop_id))


@router.post("/loops/{loop_id}/auto-run", status_code=201)
async def start_loop_auto_run(loop_id: int) -> dict:
    """启动自动验证链（Step 2）：build → smoke → test → benchmark → report。

    前置见 `docs/plan-optimize-auto-v1.md` §16.1：从 `awaiting_verification`（首次）或
    `blocked/failed`（重试）启动；预算/四动作 plan/基线证据一次性预检，不满足 409（`E_AUTO_*`）。
    驱动在本进程内，进度落 `loop_auto_runs`（重启后 blocked）。

    preflight + 建行 + phase CAS 放到线程里：preflight 会算全量工作区快照（秒级），
    不能堵事件循环（否则页面轮询/取消都卡）。schedule 必须在事件循环里做，故本端点仍 async。
    """
    try:
        data = await asyncio.to_thread(loop_orchestrator.start_loop, loop_id)
    except loop_orchestrator.AutoRunError as e:
        raise HTTPException(e.status, str(e)) from e
    loop_orchestrator.schedule(data["id"])
    return {"auto_run": data}


@router.get("/loops/{loop_id}/auto-run")
def get_loop_auto_run(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """最近一次自动验证链（无 → `{"auto_run": null}`）。"""
    loop = _get_loop(db, loop_id)
    return {"auto_run": loop_orchestrator.latest_view(db, loop.id)}


@router.post("/loops/{loop_id}/auto-run/cancel")
def cancel_loop_auto_run(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """取消自动验证链：杀当前动作作业 + 远端清理；phase=blocked（不静默回退工作区）。"""
    loop = _get_loop(db, loop_id)
    try:
        return loop_orchestrator.cancel(db, loop)
    except loop_orchestrator.AutoRunError as e:
        raise HTTPException(e.status, str(e)) from e


@router.post("/loops/{loop_id}/cancel")
def cancel_optimization_round(loop_id: int, payload: LoopCancelRequest,
                              db: Session = Depends(get_db)) -> dict:
    """取消本轮正在跑的自动优化（Step 1）：杀 agent 进程 + phase=cancelled + 释放租约。

    取消不静默回滚工作区（避免把“已改对的代码”无声撤销）；需要回退请走「否决并回退」。
    """
    loop = _get_loop(db, loop_id)
    if loop.status == "closed":
        raise HTTPException(409, "轮次已关闭，不能取消")
    # 有自动验证链在跑就一并停掉（否则驱动会在 agent 取消后又起动作作业）。
    auto_cancelled = loop_orchestrator.cancel_if_running(
        db, loop, reason=(payload.reason or "").strip() or "随轮次取消停止自动验证链")
    active_job_id = loop.active_job_id
    job = db.get(AgentJob, active_job_id) if active_job_id else None
    if job is not None and job.loop_id != loop.id:
        job = None  # 防御：active_job_id 只应指向本轮自己的改码作业
    result = "没有正在跑的改码作业；已标记取消"
    if job is not None and job.status == "running":
        result = agent_runner.cancel(job.id)
    if not loop_state.transition(db, loop, "cancelled", event="cancel", force=True,
                                 active_job_id=None,
                                 error=(payload.reason or "").strip() or "用户取消",
                                 payload={"revert_requested": payload.revert,
                                          "agent_job_id": job.id if job else None},
                                 commit=False):
        db.rollback()
        raise HTTPException(409, "轮次状态已变化（可能已被取消/否决/收尾），请刷新后再试")
    # 只释放**本轮自己的** agent 租约：按 job_id 精确匹配，不能按 task_id 一刀切
    # （否则会删掉同任务另一轮 agent / 正在跑的 action / restore 的租约）。
    if active_job_id is not None:
        workspace_lease.release(db, task_id=loop.task_id, kind="agent",
                                job_id=active_job_id, commit=False)
    db.commit()
    db.refresh(loop)
    return {"loop": optimization.loop_view(loop), "result": result,
            "revert_requested": bool(payload.revert),
            "auto_run_cancelled": auto_cancelled}


@router.post("/loops/{loop_id}/accept")
def accept_optimization_round(loop_id: int, payload: LoopAcceptRequest,
                              db: Session = Depends(get_db)) -> dict:
    """采纳本轮最新候选为版本：candidate → accept，关闭 Loop。"""
    loop = _get_loop(db, loop_id)
    try:
        return optimization.accept(db, loop, label=payload.label, note=payload.note)
    except optimization.OptimizationError as e:
        db.rollback()
        raise _http(e) from e


@router.post("/loops/{loop_id}/reject")
def reject_optimization_round(loop_id: int, payload: LoopRejectRequest,
                              db: Session = Depends(get_db)) -> dict:
    """否决本轮：关闭 Loop + 思路标记 rejected；可选把工作区回退到本轮改前。

    不要求有候选存档（失败的轮次也应能标记无效）；`revert=true` 时回退失败不影响否决，
    结果写进返回的 `reverted` 字段。
    """
    loop = _get_loop(db, loop_id)
    try:
        return optimization.reject(db, loop, reason=payload.reason, revert=payload.revert)
    except optimization.OptimizationError as e:
        db.rollback()
        raise _http(e) from e


@router.get("/loops/{loop_id}/optimization-report")
def optimization_report(loop_id: int, write: bool = False,
                        db: Session = Depends(get_db)) -> dict:
    """可复现报告（perf/precision/hardware 的 objective_summary + 代码/run/复现命令）。

    `write=true` 生成 append-only 归档报告（report.json + manifest.json + raw/），返回原报告字段
    平铺 + report_sha256/report_dir/written_to/manifest/report_valid。
    """
    loop = _get_loop(db, loop_id)
    try:
        return optimization.report(db, loop, write=write)
    except optimization.OptimizationError as e:
        raise _http(e) from e


@router.get("/loops/{loop_id}/optimization-reports")
def list_optimization_reports(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """append-only 报告归档列表（最新在前，≤50；valid 为快速校验结果）。"""
    loop = _get_loop(db, loop_id)
    return {"reports": optimization_reports.list_reports(loop.id)}


@router.post("/loops/{loop_id}/optimization-reports", status_code=201)
def archive_optimization_report(loop_id: int, db: Session = Depends(get_db)) -> dict:
    """生成并归档一次优化报告（manifest + sha + raw 副本 + repro.md）。"""
    loop = _get_loop(db, loop_id)
    try:
        return optimization_reports.write(db, loop)
    except optimization.OptimizationError as e:
        db.rollback()
        raise _http(e) from e


@router.get("/loops/{loop_id}/optimization-reports/{report_sha256}")
def get_optimization_report_archive(loop_id: int, report_sha256: str,
                                    db: Session = Depends(get_db)) -> dict:
    """按 sha 读回归档报告（从磁盘读并校验；不存在 404）。"""
    loop = _get_loop(db, loop_id)
    got = optimization_reports.load(loop.id, report_sha256)
    if got is None:
        raise HTTPException(404, f"归档报告不存在：{report_sha256}")
    return got


@router.get("/loops/{loop_id}/optimization-reports/{report_sha256}/download")
def download_optimization_report(loop_id: int, report_sha256: str,
                                 db: Session = Depends(get_db)) -> FileResponse:
    """下载**归档原始字节**的 report.json（不是重新序列化的副本，sha256 与 manifest 一致）。"""
    loop = _get_loop(db, loop_id)
    got = optimization_reports.load(loop.id, report_sha256)
    if got is None:
        raise HTTPException(404, f"归档报告不存在：{report_sha256}")
    path = Path(got["dir"]) / "report.json"
    if not path.is_file():
        raise HTTPException(404, "归档报告文件缺失")
    return FileResponse(str(path), media_type="application/json",
                        filename=f"optimization-report-{str(got['manifest'].get('report_sha256'))[:12]}.json")


@router.post("/loops/{loop_id}/optimization-reports/{report_sha256}/verify")
def verify_optimization_report(loop_id: int, report_sha256: str,
                               db: Session = Depends(get_db)) -> dict:
    """深度校验归档完整性（report.json/repro.md/raw 副本逐个 sha）；只读，不重跑真机。"""
    loop = _get_loop(db, loop_id)
    result = optimization_reports.verify(loop.id, report_sha256)
    if result is None:
        raise HTTPException(404, f"归档报告不存在：{report_sha256}")
    return result


# ---------------------------------------------------------------- P2 优化点搜索


@router.get("/tasks/{task_id}/optimization-ideas")
def optimization_ideas(task_id: int, db: Session = Depends(get_db)) -> list[dict]:
    _get_task(db, task_id)
    return idea_search.list_ideas(db, task_id)


@router.post("/tasks/{task_id}/optimization-ideas", status_code=201)
def create_optimization_idea(task_id: int, payload: OptimizationIdeaCreate,
                             db: Session = Depends(get_db)) -> dict:
    task = _get_task(db, task_id)
    try:
        row, duplicate = idea_search.add_user(db, task, payload.model_dump())
        return {"idea": idea_search.idea_view(row), "deduplicated": duplicate}
    except idea_search.IdeaSearchError as e:
        db.rollback()
        raise HTTPException(e.status, str(e)) from e


@router.patch("/optimization-ideas/{idea_id}")
def patch_optimization_idea(idea_id: int, payload: OptimizationIdeaStatusPatch,
                            db: Session = Depends(get_db)) -> dict:
    """用户侧状态修正：只允许 proposed/rejected/archived（审查 M11）。

    系统态 selected/validating/verified 由「建轮次 / 采纳 / 否决」写入；用户直接 PATCH 成
    这些状态既无审批也无对应 Loop，必须拒绝（400）。串行判定也不再依赖 idea.status
    （见 ``idea_search.create_round``），PATCH rejected 绕不过「一思路一 Loop」。
    """
    row = db.get(OptimizationIdea, idea_id)
    if row is None:
        raise HTTPException(404, "优化思路不存在")
    status = str(payload.status or "").strip()
    if status in _SYSTEM_IDEA_STATUSES:
        raise HTTPException(
            400,
            f"status={status} 由系统在建轮次/采纳/否决时写入，用户侧只能改 "
            "proposed/rejected/archived",
        )
    if status not in _USER_IDEA_STATUSES:
        raise HTTPException(400, "status 只能是 proposed/rejected/archived")
    row.status = status
    if payload.outcome is not None:
        row.outcome_json = json.dumps(payload.outcome, ensure_ascii=False)
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return idea_search.idea_view(row)



def _heal_zombie_searches(db: Session, task_id: int) -> int:
    """把本任务「running 但 AgentJob 已不在跑」的搜索置 failed；返回仍存活的 running id。

    不靠「DB 里有 running 行」直接 409（审查 M9）：`start_idea_search` 在 commit 之后抛错、
    进程重启/被杀都可能留下永久 running 的僵尸行，必须能自愈，否则新搜索永远 409。
    返回存活行 id（非 0 表示应拒绝新建）。
    """
    live_id = 0
    changed = False
    rows = list(db.scalars(select(IdeaSearch).where(
        IdeaSearch.task_id == task_id, IdeaSearch.status == "running"
    ).order_by(IdeaSearch.id)))
    for row in rows:
        job = db.get(AgentJob, row.agent_job_id) if row.agent_job_id is not None else None
        if job is not None and job.status == "running":
            live_id = row.id
            continue
        row.status = "failed"
        row.ended_at = datetime.now(timezone.utc)
        row.summary = "诊断搜索状态失联（Agent 进程已不存在），已标记失败，可重新发起搜索"
        row.result_json = json.dumps(
            {"new": 0, "duplicates": 0, "updated": 0, "stale": True}, ensure_ascii=False)
        changed = True
    if changed:
        db.commit()
    return live_id


@router.post("/tasks/{task_id}/idea-searches", status_code=201)
async def start_idea_search(task_id: int, payload: IdeaSearchRequest,
                            db: Session = Depends(get_db)) -> dict:
    task = _get_task(db, task_id)
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        raise HTTPException(409, "项目工作区未配置，无法搜索优化点")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise HTTPException(409, f"工作区不存在：{workspace}")
    if settings.llm_key_env and not os.environ.get(settings.llm_key_env):
        raise HTTPException(400, "缺少凭据：先在「API 管理」页配置 key")
    if not shutil.which("dsh"):
        raise HTTPException(400, "找不到 dsh 可执行文件")

    live_search_id = _heal_zombie_searches(db, task_id)
    if live_search_id:
        raise HTTPException(409, f"优化点搜索 #{live_search_id} 仍在运行")

    # build_prompt 要读最多 16 份 run 报告 + 硬件画像（同步重 IO），放到线程池，
    # 否则会卡住事件循环（同期 SSE / 其它请求，审查 M13）。
    prompt = await run_in_threadpool(idea_search.build_prompt, db, task, workspace, payload.extra)
    search = IdeaSearch(task_id=task.id, user_text=payload.extra, status="running")
    db.add(search)
    db.flush()
    job = AgentJob(task_id=task.id, kind="诊断", status="running", prompt=prompt[:6000])
    pricing.attach_endpoint(db, job)
    db.add(job)
    db.flush()
    search.agent_job_id = job.id
    db.commit()
    db.refresh(search)
    db.refresh(job)
    dsh_home = Path(project.dsh_home or (workspace.parent / ".dsh")).expanduser()
    try:
        dsh_home.mkdir(parents=True, exist_ok=True)
        idea_search.start(search, job, prompt, workspace, dsh_home)
    except Exception as exc:  # noqa: BLE001 —— 起不来必须把 search/job 收尾，不留 running 僵尸
        search = db.get(IdeaSearch, search.id) or search
        search.status = "failed"
        search.ended_at = datetime.now(timezone.utc)
        search.summary = f"诊断 Agent 启动失败：{type(exc).__name__}: {exc}"
        search.result_json = json.dumps(
            {"new": 0, "duplicates": 0, "updated": 0, "start_failed": True},
            ensure_ascii=False,
        )
        failed_job = db.get(AgentJob, job.id)
        if failed_job is not None:
            failed_job.status = "failed"
            failed_job.ended_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(500, search.summary) from exc
    return idea_search.search_view(search)


@router.get("/tasks/{task_id}/idea-searches")
def list_idea_searches(task_id: int, db: Session = Depends(get_db)) -> list[dict]:
    _get_task(db, task_id)
    rows = db.scalars(select(IdeaSearch).where(IdeaSearch.task_id == task_id)
                      .order_by(IdeaSearch.id.desc()).limit(20))
    return [idea_search.search_view(row) for row in rows]


@router.get("/idea-searches/{search_id}")
def get_idea_search(search_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(IdeaSearch, search_id)
    if row is None:
        raise HTTPException(404, "优化点搜索不存在")
    return idea_search.search_view(row)


@router.post("/optimization-ideas/{idea_id}/round", status_code=201)
def create_round_from_idea(idea_id: int, payload: OptimizationIdeaRoundCreate,
                           db: Session = Depends(get_db)) -> dict:
    idea = db.get(OptimizationIdea, idea_id)
    if idea is None:
        raise HTTPException(404, "优化思路不存在")
    task = _get_task(db, idea.task_id)
    try:
        return idea_search.create_round(
            db, task, idea, objective=payload.objective,
            attach_baseline_runs=payload.attach_baseline_runs,
        )
    except (idea_search.IdeaSearchError, optimization.OptimizationError) as e:
        db.rollback()
        raise HTTPException(getattr(e, "status", 400), str(e)) from e


@router.get("/loops/{loop_id}/budget-audits")
def loop_budget_audits(loop_id: int, db: Session = Depends(get_db)) -> list[dict]:
    _get_loop(db, loop_id)
    return loop_budget.list_audits(db, loop_id)


@router.post("/loops/{loop_id}/budget-check")
def loop_budget_check(loop_id: int, payload: dict, db: Session = Depends(get_db)) -> dict:
    """给工作区 ``scripts/nm.sh --loop`` 的执行前硬门禁；dry-run 不调用这里。

    body 契约：``{"action": "build|smoke|test|benchmark|profile", "reservation_id": int|null}``。
    ``reservation_id`` 是上一次 allowed 审计的 id（脚本先预留、真跑前凭 id 复用），透传给
    ``loop_budget.check_and_audit``；预算侧自己校验归属/状态，失败一律 409（fail-closed）。
    smoke 是独立 action（与 test/benchmark 分账），否则二期自动链在 ``--action smoke --loop`` 下
    会被这里 400 拒掉、容器还没起就失败并白扣一格预算。
    """
    loop = _get_loop(db, loop_id)
    task = _get_task(db, loop.task_id)
    action = str(payload.get("action") or "").strip()
    if action not in {"build", "smoke", "test", "benchmark", "profile"}:
        raise HTTPException(400, "action 只能是 build/smoke/test/benchmark/profile")
    reservation_id = payload.get("reservation_id")
    if reservation_id is not None and (
            isinstance(reservation_id, bool) or not isinstance(reservation_id, int)):
        raise HTTPException(400, "reservation_id 必须是整数或 null")
    try:
        return loop_budget.check_and_audit(
            db, task, loop, action, reservation_id=reservation_id)
    except loop_budget.BudgetExceeded as e:
        raise HTTPException(409, str(e)) from e
