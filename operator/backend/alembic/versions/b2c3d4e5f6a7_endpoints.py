"""P4 API 管理：建 `endpoints` 表 + 落一条默认路由

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
Create Date: 2026-10-19

`endpoints` 模型（`models.Endpoint`）早就有，但一直没建表 —— P4「API 管理」页要用了才补上。

- **只存凭据引用名**（`key_ref` = 环境变量名），**不存明文 key**；探针从控制台进程的环境变量读取。
- 默认落一条**真实路由**（控制台 + dsh headless 实际在用的 opencode go 网关，见
  `ops/dsh/llm-opencode-go.patch.yml`），免得页面空着。
"""

from alembic import op
import sqlalchemy as sa

revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None

DEFAULT_ENDPOINT = (
    "opencode-go / dsh headless",
    "dsh-llm-pi-ai",
    "https://opencode.ai/inference/go/v1",
    "deepseek-v4.1-flash",
    "OPX_LLM_KEY",
    262144,
    "控制台 + dsh headless 的默认路由（ops/dsh/llm-opencode-go.patch.yml）；"
    "key 只从环境变量 OPX_LLM_KEY 取，不进库/不回显",
)


def upgrade() -> None:
    op.create_table(
        "endpoints",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("provider", sa.String(120), nullable=False),
        sa.Column("base_url", sa.String(400)),
        sa.Column("model", sa.String(200)),
        sa.Column("key_ref", sa.String(200)),
        sa.Column("max_model_len", sa.Integer()),
        sa.Column("probe_state", sa.String(32), nullable=False, server_default="unknown"),
        sa.Column("last_probe_at", sa.DateTime(timezone=True)),
        sa.Column("note", sa.Text()),
    )
    op.execute(
        sa.text(
            "INSERT INTO endpoints (name, provider, base_url, model, key_ref, max_model_len, probe_state, note) "
            "VALUES (:name, :provider, :base_url, :model, :key_ref, :max_model_len, 'unknown', :note)"
        ).bindparams(
            name=DEFAULT_ENDPOINT[0], provider=DEFAULT_ENDPOINT[1], base_url=DEFAULT_ENDPOINT[2],
            model=DEFAULT_ENDPOINT[3], key_ref=DEFAULT_ENDPOINT[4], max_model_len=DEFAULT_ENDPOINT[5],
            note=DEFAULT_ENDPOINT[6],
        )
    )


def downgrade() -> None:
    op.drop_table("endpoints")
