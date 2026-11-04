"""自动优化 Step 1b：confirm token + attempt 历史 + loop_run_links

Revision ID: e9f0a1b2c3d4
Revises: c9d0e1f2a3b4
Create Date: 2026-10-26

- `loop_confirmations`：确认草案 + 一次性 token（库里只存 sha256，明文只在响应里出现一次）。
- `loop_attempts`：一次 agent 改码尝试一行（running/succeeded/failed/blocked/cancelled）。
- `loop_run_links`：复用 run 不再改 `runs.loop_id`，改为显式 link（唯一 (loop_id, run_id)）。

历史数据不需要回填：旧做法把复用的基线 run 直接写成 `runs.loop_id=loop.id`，证据读取
仍按并集兼容。
"""

from alembic import op
import sqlalchemy as sa

revision = "e9f0a1b2c3d4"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "loop_confirmations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("draft_json", sa.Text(), nullable=False),
        sa.Column("objective_sha256", sa.String(64), nullable=False),
        sa.Column("baseline_sha256", sa.String(80), nullable=True),
        sa.Column("workspace_sha256", sa.String(80), nullable=True),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("used_agent_job_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_loop_confirmations_loop_id", "loop_confirmations", ["loop_id"])
    op.create_index("uq_loop_confirmations_token", "loop_confirmations", ["token_hash"],
                    unique=True)

    op.create_table(
        "loop_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("agent_job_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("baseline_savepoint_id", sa.Integer(), nullable=True),
        sa.Column("baseline_sha256", sa.String(80), nullable=True),
        sa.Column("candidate_sha256", sa.String(80), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("summary_json", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_loop_attempts_loop_id", "loop_attempts", ["loop_id"])
    op.create_index("uq_loop_attempts_loop_seq", "loop_attempts", ["loop_id", "seq"],
                    unique=True)

    op.create_table(
        "loop_run_links",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="baseline"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_by", sa.String(40), nullable=False, server_default="api"),
    )
    op.create_index("ix_loop_run_links_loop_id", "loop_run_links", ["loop_id"])
    op.create_index("ix_loop_run_links_run_id", "loop_run_links", ["run_id"])
    op.create_index("uq_loop_run_links_loop_run", "loop_run_links", ["loop_id", "run_id"],
                    unique=True)


def downgrade() -> None:
    # 降级前把「只有 link」的 run 回填成旧语义的 runs.loop_id（旧版本只认这一列），
    # 否则 1b→1a 后复用来的 before 侧证据在旧页面会「消失」且升级回来也找不回。
    # 同一 run 被多轮 link 时取最小 loop_id（确定性；仅用于降级后的展示）。
    op.execute(
        "UPDATE runs SET loop_id = ("
        "  SELECT MIN(l.loop_id) FROM loop_run_links l WHERE l.run_id = runs.id"
        ") WHERE loop_id IS NULL AND EXISTS ("
        "  SELECT 1 FROM loop_run_links l WHERE l.run_id = runs.id"
        ")"
    )
    op.drop_index("uq_loop_run_links_loop_run", table_name="loop_run_links")
    op.drop_index("ix_loop_run_links_run_id", table_name="loop_run_links")
    op.drop_index("ix_loop_run_links_loop_id", table_name="loop_run_links")
    op.drop_table("loop_run_links")

    op.drop_index("uq_loop_attempts_loop_seq", table_name="loop_attempts")
    op.drop_index("ix_loop_attempts_loop_id", table_name="loop_attempts")
    op.drop_table("loop_attempts")

    op.drop_index("uq_loop_confirmations_token", table_name="loop_confirmations")
    op.drop_index("ix_loop_confirmations_loop_id", table_name="loop_confirmations")
    op.drop_table("loop_confirmations")
