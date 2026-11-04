"""对话增强：附件 + 每轮改动审阅证据

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-20

- `attachments`：对话附件索引（文件落工作区 `.opx-chat/conv-<id>/attachments/`）。
- `messages.changes_json`：该轮对话的改动证据（改前/改后快照、diff、回退数据），用于「审阅/回退本轮」。
"""

from alembic import op
import sqlalchemy as sa

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "attachments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(length=300), nullable=False),
        sa.Column("path", sa.String(length=600), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_attachments_conversation_id", "attachments", ["conversation_id"])
    op.add_column("messages", sa.Column("changes_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "changes_json")
    op.drop_index("ix_attachments_conversation_id", table_name="attachments")
    op.drop_table("attachments")
