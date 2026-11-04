"""版本存档（savepoints v1）

Revision ID: a9b8c7d6e5f4
Revises: e3f4a5b6c7d8
Create Date: 2026-10-21

新增 `savepoints` 表：存盘点只是「标签 + 环境引用 + 时间 + 类型」，代码本体在内容寻址的
tar.gz 归档里（`<savepoint_root>/sources/<sha256>.tar.gz`），同一 sha 只归档一份。
设计/契约见 `docs/plan-savepoints-v1.md` §3。
"""

from alembic import op
import sqlalchemy as sa

revision = "a9b8c7d6e5f4"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "savepoints",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.Integer(),
                  sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="manual"),
        sa.Column("code_snapshot_sha256", sa.String(length=80), nullable=False),
        sa.Column("archive_path", sa.String(length=600), nullable=False),
        sa.Column("archive_bytes", sa.BigInteger(), nullable=True),
        sa.Column("spec_version_id", sa.Integer(), sa.ForeignKey("spec_versions.id"), nullable=True),
        sa.Column("criteria_ref", sa.String(length=200), nullable=True),
        sa.Column("suite_ref", sa.String(length=200), nullable=True),
        sa.Column("image_digest", sa.String(length=200), nullable=True),
        sa.Column("env_recipe_snapshot", sa.Text(), nullable=True),
        sa.Column("parent_id", sa.Integer(),
                  sa.ForeignKey("savepoints.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=80), nullable=True),
    )
    op.create_index("ix_savepoints_task_id", "savepoints", ["task_id"])
    op.create_index("ix_savepoints_code_snapshot_sha256", "savepoints", ["code_snapshot_sha256"])


def downgrade() -> None:
    op.drop_index("ix_savepoints_code_snapshot_sha256", table_name="savepoints")
    op.drop_index("ix_savepoints_task_id", table_name="savepoints")
    op.drop_table("savepoints")
