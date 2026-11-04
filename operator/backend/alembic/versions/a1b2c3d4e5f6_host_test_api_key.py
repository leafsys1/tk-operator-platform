"""服务器/镜像/API：当前测试服务器、测试镜像、页面配置 key

Revision ID: a1b2c3d4e5f6
Revises: e5f6a7b8c9d0
Create Date: 2026-10-20

- `hosts.is_test_server`：当前测试服务器（全局最多一个；任务未显式指定时优先用它）
- `host_images.is_test`：测试镜像（任务选镜像时只暴露这些；未配置的默认隐藏）
- `endpoints.api_key`：页面配置的模型 key（本机 SQLite 存储，接口不回显）
- `endpoints.is_active`：当前生效的端点（agent 调用优先用它的 key）

数据回填（尽量让现有环境页面上立即可用）：
- 第一台服务器置为当前测试服务器；
- 项目当前绑定的镜像置为测试镜像；
- 第一个端点置为当前端点（api_key 留空时仍回退环境变量）。
"""

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("hosts", sa.Column("is_test_server", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("host_images", sa.Column("is_test", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("endpoints", sa.Column("api_key", sa.Text()))
    op.add_column("endpoints", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()))

    op.execute("UPDATE hosts SET is_test_server = 1 WHERE id = (SELECT MIN(id) FROM hosts)")

    op.execute(
        "UPDATE host_images SET is_test = 1 "
        "WHERE id = (SELECT image_id FROM projects WHERE image_id IS NOT NULL ORDER BY id LIMIT 1)"
    )
    op.execute(
        "UPDATE endpoints SET is_active = 1 WHERE id = (SELECT MIN(id) FROM endpoints)"
    )
    op.execute(
        "UPDATE endpoints SET note = "
        "'key 可在「API 管理」页配置（本机 SQLite、接口不回显）；"
        "也支持 key_ref 环境变量回退，当前端点的 key 会同步给 dsh' "
        "WHERE name = 'opencode-go / dsh headless'"
    )


def downgrade() -> None:
    op.drop_column("endpoints", "is_active")
    op.drop_column("endpoints", "api_key")
    op.drop_column("host_images", "is_test")
    op.drop_column("hosts", "is_test_server")
