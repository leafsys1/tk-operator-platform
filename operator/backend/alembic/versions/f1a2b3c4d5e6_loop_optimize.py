"""M3 余项：Loop 的「agent 自动改代码」证据列

Revision ID: f1a2b3c4d5e6
Revises: e2f3a4b5c6d7
Create Date: 2026-10-19

把自动优化里原本手工的「按优化假设改算子代码」交给 agent（见 `services/optimize.py`）。

- `loops.optimize_json`：这一轮自动优化的证据（JSON）—— 改前/改后代码快照哈希、逐文件 unified diff、
  agent 的结构化摘要、回退数据（改动前内容）。空 = 这一轮没跑过自动优化。

只加一列，老数据无需搬运。
"""

from alembic import op
import sqlalchemy as sa

revision = "f1a2b3c4d5e6"
down_revision = "e2f3a4b5c6d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("loops", sa.Column("optimize_json", sa.Text()))


def downgrade() -> None:
    op.drop_column("loops", "optimize_json")
