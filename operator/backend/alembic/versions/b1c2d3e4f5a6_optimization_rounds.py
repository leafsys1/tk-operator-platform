"""优化轮次 v1（P1）：loops.objective_json + savepoints.loop_id

Revision ID: b1c2d3e4f5a6
Revises: a9b8c7d6e5f4
Create Date: 2026-10-21

- `loops.objective_json`（Text，可空）：本轮优化的目标/约束/预算/用户补充文本/目标快照。
- `savepoints.loop_id`（Integer，可空，索引 `ix_savepoints_loop_id`）：存盘点归属优化轮次。

注意：SQLite 的 `ALTER TABLE ... ADD COLUMN` 不能可靠地附加外键约束（项目既有约定，见
`models.Task.host_id` 的注释），所以这里只加普通列；归属存在性由应用层校验
（`services/optimization.py` / `services/savepoints.py`）。模型里仍声明 FK 供 metadata 使用。
"""

from alembic import op
import sqlalchemy as sa

revision = "b1c2d3e4f5a6"
down_revision = "a9b8c7d6e5f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("loops", sa.Column("objective_json", sa.Text(), nullable=True))
    op.add_column("savepoints", sa.Column("loop_id", sa.Integer(), nullable=True))
    op.create_index("ix_savepoints_loop_id", "savepoints", ["loop_id"])


def downgrade() -> None:
    op.drop_index("ix_savepoints_loop_id", table_name="savepoints")
    op.drop_column("savepoints", "loop_id")
    op.drop_column("loops", "objective_json")
