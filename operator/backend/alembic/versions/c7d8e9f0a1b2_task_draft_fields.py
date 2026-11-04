"""P0-3「任务开始」页：任务草稿（is_draft）与创建期字段

Revision ID: c7d8e9f0a1b2
Revises: c9a4e2f7b6d1
Create Date: 2026-10-15

来源 = `docs/UI-task-start-v1.md` §2.3（创建流程与草稿生命周期，2026-10-15 定）：

- `tasks.is_draft`：草稿态。在「任务开始」页里尚未定版，**不进阶段统计**、不算「已建任务」。
- `tasks.raw_input`：需求原文（草稿期承载用户原话）。**定版时转进 `spec_versions.raw_input`**，
  不留成第二个真相源（方案 §6「先库后盘」）。
- `tasks.host_id` / `image_id` / `device_id` / `timeout_s`：创建期配置，**空 = 用项目默认**
  （与 `criteria_ref` 同一口径，避免建任务时被迫先把默认值抄一遍）。
- `tasks.fix_rounds_max` / `budget_per_action`：D8 预算（按案例 / 按 Loop 计），空 = 默认 3 / 2。

改动只是加列；老任务 `is_draft` 一律为 0（= 正式任务），数据无需搬运。
SQLite 的 `ALTER TABLE ADD COLUMN` 只在**新列可空或无默认**时才能带 REFERENCES，这里都满足。
"""

from alembic import op
import sqlalchemy as sa

revision = "c7d8e9f0a1b2"
down_revision = "c9a4e2f7b6d1"
branch_labels = None
depends_on = None

NEW_COLUMNS = [
    sa.Column("is_draft", sa.Boolean(), nullable=False, server_default="0"),
    sa.Column("raw_input", sa.Text()),
    # 不设 FK：SQLite 的 ALTER TABLE ADD COLUMN 加不了外键约束（会退化成 ADD CONSTRAINT 报错）。
    # 存在性由应用层校验（tasks 路由的 _check_task_refs）。
    sa.Column("host_id", sa.Integer()),
    sa.Column("image_id", sa.Integer()),
    sa.Column("device_id", sa.String(80)),
    sa.Column("timeout_s", sa.Integer()),
    sa.Column("fix_rounds_max", sa.Integer()),
    sa.Column("budget_per_action", sa.Integer()),
]


def upgrade() -> None:
    for col in NEW_COLUMNS:
        op.add_column("tasks", col)


def downgrade() -> None:
    for col in reversed(NEW_COLUMNS):
        op.drop_column("tasks", col.name)
