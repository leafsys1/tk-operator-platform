"""P2 优化点搜索 + P3 Loop 测量预算审计

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-10-22
"""

from alembic import op
import sqlalchemy as sa

revision = "c2d3e4f5a6b7"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "idea_searches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_job_id", sa.Integer(), sa.ForeignKey("agent_jobs.id"), nullable=True),
        sa.Column("user_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("agent_job_id", name="uq_idea_searches_agent_job_id"),
    )
    op.create_index("ix_idea_searches_task_id", "idea_searches", ["task_id"])
    op.create_table(
        "optimization_ideas",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id"), nullable=True),
        sa.Column("search_id", sa.Integer(), sa.ForeignKey("idea_searches.id"), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="agent"),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("change_scope", sa.Text(), nullable=True),
        sa.Column("expected", sa.Text(), nullable=True),
        sa.Column("risk", sa.Text(), nullable=True),
        sa.Column("confidence", sa.String(32), nullable=True),
        sa.Column("effort", sa.String(32), nullable=True),
        sa.Column("skill", sa.String(200), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="proposed"),
        sa.Column("dedupe_key", sa.String(80), nullable=False),
        sa.Column("outcome_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_optimization_ideas_task_id", "optimization_ideas", ["task_id"])
    op.create_index("ix_optimization_ideas_loop_id", "optimization_ideas", ["loop_id"])
    op.create_index("ix_optimization_ideas_search_id", "optimization_ideas", ["search_id"])
    op.create_index("ix_optimization_ideas_dedupe_key", "optimization_ideas", ["dedupe_key"])
    op.create_table(
        "loop_budget_audits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("limit_count", sa.Integer(), nullable=False),
        sa.Column("used_count", sa.Integer(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_loop_budget_audits_task_id", "loop_budget_audits", ["task_id"])
    op.create_index("ix_loop_budget_audits_loop_id", "loop_budget_audits", ["loop_id"])


def downgrade() -> None:
    op.drop_index("ix_loop_budget_audits_loop_id", table_name="loop_budget_audits")
    op.drop_index("ix_loop_budget_audits_task_id", table_name="loop_budget_audits")
    op.drop_table("loop_budget_audits")
    op.drop_index("ix_optimization_ideas_dedupe_key", table_name="optimization_ideas")
    op.drop_index("ix_optimization_ideas_search_id", table_name="optimization_ideas")
    op.drop_index("ix_optimization_ideas_loop_id", table_name="optimization_ideas")
    op.drop_index("ix_optimization_ideas_task_id", table_name="optimization_ideas")
    op.drop_table("optimization_ideas")
    op.drop_index("ix_idea_searches_task_id", table_name="idea_searches")
    op.drop_table("idea_searches")
