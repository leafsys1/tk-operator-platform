"""预算一次性预留 + 优化点去重唯一索引

Revision ID: a7b8c9d0e1f2
Revises: c2d3e4f5a6b7
Create Date: 2026-10-23

修复 commit `1cf4408` 审查里的两处数据层缺口：

- `loop_budget_audits.claimed_at`：服务端可校验的**一次性预算预留**（审查 M4）。
  页面起动作时先写 allowed 审计（预留），`scripts/nm_test.sh` 执行前带
  `OPX_BUDGET_AUDIT_ID` 回 `/budget-check` claim；已 claim 的审计不能复用。
  替代原来调用方自己设 `OPX_BUDGET_RESERVED=1` 就能跳过的“协议式”门禁。
- `optimization_ideas` 的 `(task_id, dedupe_key)` **唯一索引**：去重从 check-then-act
  收敛到数据库约束，并发也不会重复落 idea（审查 M15，P2 搜索依赖它捕获 IntegrityError）。

升级前先清掉历史重复行（同一 task 下同 dedupe_key 只保留最早 id 一行）—— 否则唯一索引
建不出来。降级只还原结构，**不还原**被清掉的重复行（它们本来就是去重要消除的数据）。
"""

from alembic import op
import sqlalchemy as sa

revision = "a7b8c9d0e1f2"
down_revision = "c2d3e4f5a6b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- 1) 预算审计：一次性预留的消费时间 -------------------------------------
    op.add_column("loop_budget_audits",
                  sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))

    # ---- 2) 优化点去重：历史重复行清理 + 唯一索引 -------------------------------
    # 旧实现（1cf4408）是「先查再插」，并发/重试可能留下重复行；保留最早一条（min id）。
    op.execute(
        "DELETE FROM optimization_ideas WHERE id NOT IN "
        "(SELECT MIN(id) FROM optimization_ideas GROUP BY task_id, dedupe_key)"
    )
    op.create_index("uq_optimization_ideas_task_dedupe", "optimization_ideas",
                    ["task_id", "dedupe_key"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_optimization_ideas_task_dedupe", table_name="optimization_ideas")
    op.drop_column("loop_budget_audits", "claimed_at")
