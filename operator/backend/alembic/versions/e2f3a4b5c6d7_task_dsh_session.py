"""P0-3 收尾：需求澄清改用 dsh 会话（记录会话 id）

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-10-15

对话式需求澄清从「直连模型」改成**走 dsh 会话**（`--session-id` 续会话），
所以任务上要记住这个会话 id，跨轮复用。只加一列。
"""

from alembic import op
import sqlalchemy as sa

revision = "e2f3a4b5c6d7"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("dsh_session_id", sa.String(120)))


def downgrade() -> None:
    op.drop_column("tasks", "dsh_session_id")
