"""P0-3 第二步：对话式需求澄清（规格草稿 + 对话）

Revision ID: d1e2f3a4b5c6
Revises: c7d8e9f0a1b2
Create Date: 2026-10-15

来源 = `docs/UI-task-start-v1.md` §3（对话式解析）+ §1.1（对话过程与业务证据**分开存**）：

- `tasks.spec_draft`：规格草稿（业务证据），JSON `{fields: {...}, questions: [...], ready: bool}`。
  定版时转进 `spec_versions`，本列即清空/停用（不留第二个真相源）。
- `tasks.chat_log`：对话记录（过程证据），JSON `[{role, text}]`。

只加两列，老任务为空。
"""

from alembic import op
import sqlalchemy as sa

revision = "d1e2f3a4b5c6"
down_revision = "c7d8e9f0a1b2"
branch_labels = None
depends_on = None

NEW_COLUMNS = [
    sa.Column("spec_draft", sa.Text()),
    sa.Column("chat_log", sa.Text()),
]


def upgrade() -> None:
    for col in NEW_COLUMNS:
        op.add_column("tasks", col)


def downgrade() -> None:
    for col in reversed(NEW_COLUMNS):
        op.drop_column("tasks", col.name)
