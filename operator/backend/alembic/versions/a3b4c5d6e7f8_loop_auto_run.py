"""自动优化 Step 2：自动验证链 loop_auto_runs

Revision ID: a3b4c5d6e7f8
Revises: e9f0a1b2c3d4
Create Date: 2026-10-27

`loop_auto_runs`：一次自动链（build → smoke → test → benchmark → report）的持久化进度。
驱动任务在进程内（asyncio），重启后由 `loop_orchestrator.reconcile_startup` 标 blocked；
进度 `steps_json` 含每步状态/作业 id/run id/门禁/远端清理结果，供页面轮询。
"""

from alembic import op
import sqlalchemy as sa

revision = "a3b4c5d6e7f8"
down_revision = "e9f0a1b2c3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "loop_auto_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("current_step", sa.String(20), nullable=True),
        sa.Column("current_action_job_id", sa.Integer(), nullable=True),
        sa.Column("attempt_id", sa.Integer(), nullable=True),
        sa.Column("baseline_sha256", sa.String(80), nullable=True),
        sa.Column("candidate_sha256", sa.String(80), nullable=True),
        sa.Column("steps_json", sa.Text(), nullable=True),
        sa.Column("budget_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_loop_auto_runs_loop_id", "loop_auto_runs", ["loop_id"])
    op.create_index("ix_loop_auto_runs_loop_status", "loop_auto_runs", ["loop_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_loop_auto_runs_loop_status", table_name="loop_auto_runs")
    op.drop_index("ix_loop_auto_runs_loop_id", table_name="loop_auto_runs")
    op.drop_table("loop_auto_runs")
