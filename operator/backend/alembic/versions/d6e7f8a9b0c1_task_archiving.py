"""任务归档：主视图隐藏，保留 run / Loop / 存点证据。

Revision ID: d6e7f8a9b0c1
Revises: b8c9d0e1f2a3
Create Date: 2026-10-26
"""

from alembic import op
import sqlalchemy as sa


revision = "d6e7f8a9b0c1"
down_revision = "b8c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("tasks", "is_archived")
