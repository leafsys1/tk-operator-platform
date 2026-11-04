"""合并两条迁移链：自动优化门禁/自动链/报告归档 ↔ 任务归档/算子级证据作用域。

Revision ID: f2b3c4d5e6a7
Revises: a3b4c5d6e7f8, e7f8a9b0c1d2
Create Date: 2026-10-27

两边都从 `b8c9d0e1f2a3` 分叉：
- 本地：c9d0e1f2a3b4（优化状态机）→ e9f0a1b2c3d4（confirm/attempts/links）→ a3b4c5d6e7f8（loop_auto_runs）
- origin/main：d6e7f8a9b0c1（任务归档）→ e7f8a9b0c1d2（证据作用域列）

merge revision 本身不改结构，只把两条 head 合回单头；down_revision 用元组。
两个分支的各自动作都已在各自迁移里落库，双向 upgrade/downgrade 由 alembic 逐支完成。
"""

from alembic import op  # noqa: F401  (保持与其它迁移一致的导入形态)

revision = "f2b3c4d5e6a7"
down_revision = ("a3b4c5d6e7f8", "e7f8a9b0c1d2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """两条分支的迁移都已各自应用；合并点无额外结构变更。"""


def downgrade() -> None:
    """合并点无结构变更；降级时 alembic 会继续向下走两条分支。"""
