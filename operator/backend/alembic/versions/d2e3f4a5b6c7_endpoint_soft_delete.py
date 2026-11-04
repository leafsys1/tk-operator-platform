"""端点软删（任务外页面审查 H2）

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-10-21

问题：`endpoints.id` 是普通 `INTEGER PRIMARY KEY`（非 AUTOINCREMENT），DELETE 物理删行后
新建端点会复用 id → 历史 `agent_jobs.endpoint_id` 指向新端点，费用被静默重算。

修法：不物理删，加 `deleted_at` 做软删；历史调用仍能按原 id 解析到当时的价目，
列表 / 当前端点 / 创建重名检查过滤已删行。
"""

from alembic import op
import sqlalchemy as sa

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("endpoints", sa.Column("deleted_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("endpoints", "deleted_at")
