"""端点单价 + 调用端点归属（D2 费用口径）

Revision ID: c1d2e3f4a5b6
Revises: a1b2c3d4e5f6
Create Date: 2026-10-21

补 P4 欠的「费用」半条链（此前消耗页只能如实标「未定价」）：

- `endpoints.price_in` / `price_out` / `price_cache_read` / `price_cache_write`：
  人民币元 / 每百万 token，nullable；Existence 为空 = 该项不单独计价。
  `price_in` + `price_out` 都有才算「已定价」（见 `services/pricing.py`）。
- `agent_jobs.endpoint_id` / `model`：这次调用用的端点与模型（计价归属，创建时快照）。
  早期行为 NULL —— 消耗页回退「当前端点」估算并标注 `cost_estimated`，不改写历史行。

费用本身**不落库**（`agent_jobs.cost` 只留给 P0 示例/手工行），读时按 token × 当前价目实时算，
所以改单价后历史消耗立刻跟着更新。
"""

from alembic import op
import sqlalchemy as sa

revision = "c1d2e3f4a5b6"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("endpoints", sa.Column("price_in", sa.Float()))
    op.add_column("endpoints", sa.Column("price_out", sa.Float()))
    op.add_column("endpoints", sa.Column("price_cache_read", sa.Float()))
    op.add_column("endpoints", sa.Column("price_cache_write", sa.Float()))
    op.add_column("agent_jobs", sa.Column("endpoint_id", sa.Integer()))
    op.add_column("agent_jobs", sa.Column("model", sa.String(200)))


def downgrade() -> None:
    op.drop_column("agent_jobs", "model")
    op.drop_column("agent_jobs", "endpoint_id")
    op.drop_column("endpoints", "price_cache_write")
    op.drop_column("endpoints", "price_cache_read")
    op.drop_column("endpoints", "price_out")
    op.drop_column("endpoints", "price_in")
