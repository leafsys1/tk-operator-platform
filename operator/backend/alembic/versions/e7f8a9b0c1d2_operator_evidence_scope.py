"""任务/算子级证据作用域。

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-10-27
"""

from alembic import op
import sqlalchemy as sa


revision = "e7f8a9b0c1d2"
down_revision = "d6e7f8a9b0c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("cases_confirmed_scope", sa.String(length=32), nullable=True))
    op.add_column("task_versions", sa.Column("evidence_scope", sa.String(length=32), nullable=True))
    op.add_column("runs", sa.Column("evidence_scope", sa.String(length=32), nullable=True))
    op.add_column("runs", sa.Column("scoped_code_snapshot_sha256", sa.String(length=80), nullable=True))
    op.add_column("runs", sa.Column("scoped_suite_ref", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "scoped_suite_ref")
    op.drop_column("runs", "scoped_code_snapshot_sha256")
    op.drop_column("runs", "evidence_scope")
    op.drop_column("task_versions", "evidence_scope")
    op.drop_column("tasks", "cases_confirmed_scope")
