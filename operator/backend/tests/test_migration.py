"""迁移往返：upgrade head → downgrade base → upgrade head，且数据修正真的生效。

为什么值得测：迁移是唯一不可逆的部分（方案 §3），而 SQLite 上的 ALTER/DROP 又是「部分生效」高发区
（2026-10-15 我第一次写这条迁移时，就在 UPDATE 报错后退化成了半应用状态）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture()
def alembic_db(tmp_path: Path):
    """把迁移指向**临时库**。

    注意：`alembic/env.py` 故意用 `app.config.settings.db_url` 覆盖 alembic.ini 的 url
    （保证「代码与迁移看同一个库」）。所以测试必须改这个 settings，而不是只改 Config ——
    否则迁移会打到真实的 opx.db 上（我第一次写这个测试就踩了，幸好断言先失败）。
    """
    from app.config import settings

    db = tmp_path / "mig.db"
    old = settings.db_url
    settings.db_url = f"sqlite:///{db}"
    try:
        yield db
    finally:
        settings.db_url = old


def _insert(con: sqlite3.Connection, table: str, **cols) -> None:
    """插入一行老数据：没给的 NOT NULL 列按类型补个哑值。

    这样写是为了**不再逐列猜**迁移前的表结构（第一次写这个测试时，hosts 的 kind / probe_state /
    created_at 轮着报 NOT NULL，改三次才发现该自适应）。
    """
    info = con.execute(f"PRAGMA table_info({table})").fetchall()
    values = dict(cols)
    for _, name, ctype, notnull, default, pk in info:
        if name in values or not notnull or default is not None or pk:
            continue
        if "INT" in ctype.upper() or "BOOL" in ctype.upper() or "FLOAT" in ctype.upper():
            values[name] = 0
        elif "DATE" in ctype.upper():
            values[name] = "2026-01-01 00:00:00.000000"
        else:
            values[name] = ""
    keys = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    con.execute(f"INSERT INTO {table} ({keys}) VALUES ({marks})", list(values.values()))


def _cfg(db_path: Path) -> Config:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _tables(db: Path) -> set[str]:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_upgrade_downgrade_upgrade(alembic_db: Path):
    db = alembic_db
    cfg = _cfg(db)

    command.upgrade(cfg, "head")
    tables = _tables(db)
    assert {"agent_jobs", "runs", "loops", "spec_versions", "stage_events", "endpoints",
            "action_jobs", "conversations", "messages", "attachments", "optimization_ideas",
            "idea_searches", "loop_budget_audits",
            "loop_events", "workspace_leases",
            "loop_confirmations", "loop_attempts", "loop_run_links",
            "loop_auto_runs"} <= tables

    con = sqlite3.connect(db)
    assert {"evidence_scope", "op_name_snapshot", "archive_snapshot_sha256"} <= {
        r[1] for r in con.execute("PRAGMA table_info(savepoints)")}
    cols_tasks = {r[1] for r in con.execute("PRAGMA table_info(tasks)")}
    cols_runs = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    cols_jobs = {r[1] for r in con.execute("PRAGMA table_info(agent_jobs)")}
    cols_proj = {r[1] for r in con.execute("PRAGMA table_info(projects)")}
    cols_loops = {r[1] for r in con.execute("PRAGMA table_info(loops)")}
    cols_messages = {r[1] for r in con.execute("PRAGMA table_info(messages)")}
    cols_endpoints = {r[1] for r in con.execute("PRAGMA table_info(endpoints)")}
    con.close()
    # 复查修正（F7/F8/F9 + 判据档载体）
    assert "spec_json" not in cols_tasks and "criteria_ref" in cols_tasks
    assert "suite_ref" in cols_runs and "suite_version_id" not in cols_runs
    assert "is_demo" in cols_jobs
    assert "criteria_ref" in cols_proj
    # P0-3 任务草稿与创建期字段
    assert {"is_draft", "raw_input", "host_id", "image_id", "device_id", "timeout_s",
            "fix_rounds_max", "budget_per_action"} <= cols_tasks
    # P0-3 第二步：对话式需求澄清
    assert {"spec_draft", "chat_log", "dsh_session_id"} <= cols_tasks
    # M3 余项：agent 自动改代码的证据列
    assert "optimize_json" in cols_loops
    # 自动优化 Step 1：显式状态机 + 幂等键列
    assert {"phase", "state_version", "active_job_id", "idempotency_key",
            "phase_error", "phase_updated_at"} <= cols_loops
    # Step 1b：确认 token / attempt / run 复用关联
    con = sqlite3.connect(db)
    assert {"token_hash", "draft_json", "used_at", "expires_at"} <= {
        r[1] for r in con.execute("PRAGMA table_info(loop_confirmations)")}
    assert {"seq", "status", "baseline_sha256", "candidate_sha256"} <= {
        r[1] for r in con.execute("PRAGMA table_info(loop_attempts)")}
    assert {"role", "created_by"} <= {
        r[1] for r in con.execute("PRAGMA table_info(loop_run_links)")}
    # Step 2：自动验证链进度
    assert {"status", "current_step", "steps_json", "budget_json", "candidate_sha256"} <= {
        r[1] for r in con.execute("PRAGMA table_info(loop_auto_runs)")}
    con.close()
    # 对话增强：每轮改动审阅证据
    assert "changes_json" in cols_messages
    # D2 费用口径：端点单价 + 调用端点归属
    assert {"price_in", "price_out", "price_cache_read", "price_cache_write"} <= cols_endpoints
    assert {"endpoint_id", "model"} <= cols_jobs
    # 任务外页面审查 H2：端点软删（保留 id 供历史计价）
    assert "deleted_at" in cols_endpoints
    # key 迁移到本机密钥库：DB 只存引用（旧 api_key 列保留兼容）
    assert "secret_ref" in cols_endpoints
    # 合并 origin/main（4751e6e/9e7b2d4）：任务归档 + 算子级证据作用域列
    assert {"is_archived", "cases_confirmed_scope"} <= cols_tasks
    assert {"evidence_scope", "scoped_code_snapshot_sha256", "scoped_suite_ref"} <= cols_runs
    con = sqlite3.connect(db)
    assert "evidence_scope" in {
        r[1] for r in con.execute("PRAGMA table_info(task_versions)")}
    con.close()

    command.downgrade(cfg, "base")
    assert "runs" not in _tables(db)

    command.upgrade(cfg, "head")
    assert "spec_versions" in _tables(db)


def test_data_fix_moves_legacy_spec_and_marks_demo_image(alembic_db: Path):
    """在「旧形态」的库上跑迁移：spec_json 要搬进 spec_versions，不能跑的镜像要摘掉 is_op_dev。"""
    db = alembic_db
    cfg = _cfg(db)
    # 先建到上一个版本，塞进老数据，再升到 head
    command.upgrade(cfg, "b7f2c9a4e1d3")
    con = sqlite3.connect(db)
    _insert(con, "hosts", id=1, name="tk36", ssh_alias="tk36", cards="8×910B1")
    _insert(con, "host_images", id=20, host_id=1,
            image_ref="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann",
            tag="9.2.0-beta.2-910b-ubuntu22.04-py3.12", is_op_dev=1)
    _insert(con, "host_images", id=52, host_id=1, image_ref="lfx-op-dev-910b",
            tag="torch210-cann910", is_op_dev=1)
    _insert(con, "projects", id=1, name="p", host_id=1, image_id=20)
    _insert(con, "tasks", id=1, project_id=1, title="complex64 mm", op_name="complex64_mm",
            spec_json='{"dtype": "complex64"}', stage="开发")
    _insert(con, "agent_jobs", task_id=1, kind="开发", prompt="（P0 示例：手工登记的）",
            status="ok", dsh_session_id="session-demo-0001", tokens_in=1190000)
    con.commit()
    con.close()

    command.upgrade(cfg, "head")
    con = sqlite3.connect(db)
    # 规格搬到 spec_versions 并成为 current
    spec = con.execute("SELECT id, task_id, version, spec_json FROM spec_versions WHERE task_id=1").fetchone()
    assert spec and spec[3] == '{"dtype": "complex64"}'
    assert con.execute("SELECT current_spec_version_id FROM tasks WHERE id=1").fetchone()[0] == spec[0]
    # demo 行被标记（聚合要排除它）
    assert con.execute("SELECT is_demo FROM agent_jobs WHERE dsh_session_id='session-demo-0001'").fetchone()[0] == 1
    # 不能跑算子的镜像摘掉 is_op_dev，能跑的留下并写版本
    assert con.execute("SELECT is_op_dev FROM host_images WHERE id=20").fetchone()[0] == 0
    assert con.execute("SELECT is_op_dev FROM host_images WHERE id=52").fetchone()[0] == 1
    assert con.execute("SELECT torch_version FROM host_images WHERE id=52").fetchone()[0]
    # 项目默认镜像切到能跑的那个
    assert con.execute("SELECT image_id FROM projects WHERE id=1").fetchone()[0] == 52
    con.close()
