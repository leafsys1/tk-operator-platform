"""P0-2 对话页：conversations（模型早有、一直没建表）+ messages

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-20

- `conversations`：会话索引（dsh_session_id 第一轮跑完回填；`mode=task|light`）。
- `messages`：用户可见的消息正文；dsh 的完整事件仍在它自己的 JSONL。
- `agent_jobs.conversation_id`：把一次 agent 调用挂到会话上（SQLite 的 ADD COLUMN 不能带 FK，故为普通列）。
"""

from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True),
        sa.Column("dsh_session_id", sa.String(length=120), nullable=True, unique=True),
        sa.Column("title", sa.String(length=300), nullable=True),
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="task"),
        sa.Column("message_count", sa.Integer(), nullable=True),
        sa.Column("tokens", sa.BigInteger(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_conversations_task_id", "conversations", ["task_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.add_column("agent_jobs", sa.Column("conversation_id", sa.Integer(), nullable=True))
    op.create_index("ix_agent_jobs_conversation_id", "agent_jobs", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_jobs_conversation_id", table_name="agent_jobs")
    op.drop_column("agent_jobs", "conversation_id")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_task_id", table_name="conversations")
    op.drop_table("conversations")
