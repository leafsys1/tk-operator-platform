"""端点 key 迁移到本机密钥库（2026-10-21）

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-10-21

页面配置的模型 API key 此前明文存 `endpoints.api_key`。改为：
- key 本体存本机密钥库（macOS Keychain / Linux Secret Service / 0600 文件）；
- DB 只存 `secret_ref`（如 `endpoint-1`），接口仍只回 `has_key`。

本迁移只加引用列，不动数据；旧 `api_key` 由启动时
`llm_config.migrate_legacy_secrets()` 搬到密钥库并清空（迁移脚本里访问 Keychain
依赖运行环境，放在应用启动更可控）。该列暂保留兼容，后续可再删。
"""

from alembic import op
import sqlalchemy as sa

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("endpoints", sa.Column("secret_ref", sa.String(200)))


def downgrade() -> None:
    op.drop_column("endpoints", "secret_ref")
