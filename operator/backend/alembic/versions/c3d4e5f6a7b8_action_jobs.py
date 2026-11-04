"""四动作「点一下就真执行」：动作作业表

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-10-20

把 `dry_run=false` 从「501 拒绝」变成「起一个异步作业」：后端照 plan 调 `scripts/nm_test.sh`，
脚本自己登记 run，作业行只记过程（日志/退出码/门禁结论）并回填 `run_id`。

- `action_jobs.run_ref`：透传给脚本的 `--run-id`，也是 `runs.remote_workdir` —— 跑完按它找回 run。
- `action_jobs.run_id`：跑完回填的 `runs.id`（门禁判定用）。
"""

from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "action_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id"), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
        sa.Column("run_ref", sa.String(length=120), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=True),
        sa.Column("device_id", sa.String(length=80), nullable=True),
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column("log", sa.Text(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("gate_verdict", sa.String(length=32), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_action_jobs_task_id", "action_jobs", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_action_jobs_task_id", table_name="action_jobs")
    op.drop_table("action_jobs")
