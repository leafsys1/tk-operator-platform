import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .models import ActionJob, AgentJob, IdeaSearch
from .routers import (case_library, chips, conversations, deliveries, documents, endpoints, health,
                      hosts, optimization, prototype, runs, savepoints, tasks, usage, versions)
from .services import llm_config, loop_orchestrator, loop_state, workspace_lease, runtime_lock

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Uvicorn calls lifespan before binding the port. A duplicate server must not
    # reconcile the still-running owner's jobs before failing with port-in-use.
    with runtime_lock.sqlite_console_lock(settings.db_url):
        async with _owned_lifespan(_):
            yield


@asynccontextmanager
async def _owned_lifespan(_: FastAPI):
    """进程重启后，之前标着 running 的 agent job 不可能还在跑 → 标记中断，避免页面显示假在跑。

    v1.1 目标：真实执行（runs）要按远端容器/run 目录状态核对，而不是一律标中断（P1.5-a 起逐步）。
    """
    with SessionLocal() as db:
        for row in db.scalars(select(AgentJob).where(AgentJob.status == "running")):
            row.status = "interrupted"
            row.ended_at = datetime.now(timezone.utc)
        # 四动作作业同理：本地子进程随进程退出而没了，别让页面永远显示「执行中」
        for job in db.scalars(select(ActionJob).where(ActionJob.status == "running")):
            job.status = "interrupted"
            job.ended_at = datetime.now(timezone.utc)
            job.error = (job.error + "；" if job.error else "") + "控制台重启，执行进程已不在了（看日志尾部）"
        for search in db.scalars(select(IdeaSearch).where(IdeaSearch.status == "running")):
            search.status = "failed"
            search.ended_at = datetime.now(timezone.utc)
            search.summary = "控制台重启，诊断 Agent 已中断"
        db.commit()
        # Step 1：状态机 running → blocked（本地进程已不在，自动链不得继续）；
        # 租约随重启清空（远端容器清理仍属后续步骤，这里只保证本地不会误放行）。
        try:
            loop_state.reconcile_startup(db)
        except Exception:  # noqa: BLE001 —— 对账失败不能阻塞控制台启动
            logger.exception("轮次状态机重启对账失败")
        try:
            workspace_lease.reconcile_startup(db)
        except Exception:  # noqa: BLE001
            logger.exception("工作区租约重启对账失败")
        try:
            # Step 2：自动链驱动在进程内，重启后不可能还在跑 → running → blocked。
            loop_orchestrator.reconcile_startup(db)
        except Exception:  # noqa: BLE001
            logger.exception("自动验证链重启对账失败")
        # 先记下启动时环境里的 OPX_LLM_KEY（页面 key 清空后的回退基线），再应用当前端点 key。
        llm_config.capture_baseline()
        try:
            # 旧库明文 key → 本机密钥库（Keychain/secret-tool/0600 文件），然后清空 api_key 列。
            report = llm_config.migrate_legacy_secrets(db)
            if report.get("failed"):
                logger.warning("部分端点 key 未能迁移到本机密钥库：%s", report["failed"])
        except Exception:  # noqa: BLE001 —— 迁移失败不能阻塞控制台启动（运行时会回退旧列）
            logger.exception("端点 key 迁移到本机密钥库失败（保留旧库明文列）")
        llm_config.apply_active_key(db)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# 开发期：Vite dev server（5173）直连本 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(hosts.router)
app.include_router(tasks.router)
app.include_router(documents.router)
app.include_router(versions.router)
app.include_router(deliveries.router)
app.include_router(runs.router)
app.include_router(optimization.router)
app.include_router(savepoints.router)
app.include_router(conversations.router)
app.include_router(case_library.router)
app.include_router(chips.router)
app.include_router(usage.router)
app.include_router(endpoints.router)
app.include_router(prototype.router)

# 生产：由 FastAPI 托管前端 build 产物（web/dist），仍是单进程部署
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
