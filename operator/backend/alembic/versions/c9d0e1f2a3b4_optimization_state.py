"""自动优化 Step 1：loops 状态机 + loop_events + workspace_leases

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-10-26

- `loops.phase/state_version/active_job_id/idempotency_key/phase_error/phase_updated_at`：
  显式状态机，替代“用 runs/jobs 拓扑反推”；`(task_id, idempotency_key)` 唯一索引支持幂等确认。
- `loop_events`：append-only 事件（创建/启动/完成/失败/采纳/否决/恢复）。
- `workspace_leases`：DB 级工作区单飞租约（同一 project 唯一）。

SQLite 的 `ALTER TABLE ADD COLUMN` 不能可靠附加外键；`active_job_id` 故意只是整数列，
归属存在性由应用层校验。加列都带 `server_default`，避免存量行非空失败。
"""

from alembic import op
import sqlalchemy as sa

revision = "c9d0e1f2a3b4"
down_revision = "b8c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("loops", sa.Column("phase", sa.String(32), nullable=False,
                                     server_default="idle"))
    op.add_column("loops", sa.Column("state_version", sa.Integer(), nullable=False,
                                     server_default="0"))
    op.add_column("loops", sa.Column("active_job_id", sa.Integer(), nullable=True))
    op.add_column("loops", sa.Column("idempotency_key", sa.String(120), nullable=True))
    op.add_column("loops", sa.Column("phase_error", sa.Text(), nullable=True))
    op.add_column("loops", sa.Column("phase_updated_at", sa.DateTime(timezone=True),
                                     nullable=True))
    op.create_index("uq_loops_task_idempotency", "loops",
                    ["task_id", "idempotency_key"], unique=True)

    op.create_table(
        "loop_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("phase", sa.String(32), nullable=True),
        sa.Column("actor", sa.String(40), nullable=False, server_default="page"),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_loop_events_loop_id", "loop_events", ["loop_id"])

    op.create_table(
        "workspace_leases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.Integer(),
                  sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.create_index("uq_workspace_leases_project", "workspace_leases",
                    ["project_id"], unique=True)

    # 存量回填：非优化 Loop = idle/已关闭；优化轮次按关闭结论；其余等首次读取派生。
    op.execute(
        "UPDATE loops SET phase = CASE "
        "WHEN objective_json IS NULL THEN "
        "  CASE WHEN status = 'closed' THEN 'accepted' ELSE 'idle' END "
        "WHEN status = 'closed' THEN "
        "  CASE WHEN conclusion LIKE 'rejected%' THEN 'rejected' ELSE 'accepted' END "
        "ELSE 'baseline_pending' END"
    )


def downgrade() -> None:
    op.drop_index("uq_workspace_leases_project", table_name="workspace_leases")
    op.drop_table("workspace_leases")
    op.drop_index("ix_loop_events_loop_id", table_name="loop_events")
    op.drop_table("loop_events")
    op.drop_index("uq_loops_task_idempotency", table_name="loops")
    op.drop_column("loops", "phase_updated_at")
    op.drop_column("loops", "phase_error")
    op.drop_column("loops", "idempotency_key")
    op.drop_column("loops", "active_job_id")
    op.drop_column("loops", "state_version")
    op.drop_column("loops", "phase")
