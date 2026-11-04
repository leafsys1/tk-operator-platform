"""任务生命周期 v2：大/小版本 + 用例组确认 + 任务级文档

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-23

依据 `docs/plan-task-lifecycle-v2.md`（用户 2026-10-23 定稿口径）：

- `tasks` 增加：`version_minor`（小版本号）、`cases_confirmed_at / _sha256 / _by`（
  精度 + 性能用例组「整组确认」记录；未确认时四动作 plan 拦「用例组未确认」）。
- 新表 `task_versions`：任务版本历史（大版本 = spec 版本，小版本 = 采纳优化 / 交付前人工变更）。
  证据不按 run 打标，交付门禁按 `(spec_version_id, code_snapshot_sha256, suite_ref)` 与 run 匹配。
- 新表 `task_documents`：任务开始页上传的《算子设计文档》（原文 + 解析稿，均落工作区）。

老数据兼容：`version_minor` 默认 0；版本表为空时由应用层按当前 spec 推导出 v<spec>.0 的只读兜底。
"""

from alembic import op
import sqlalchemy as sa

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- tasks：小版本 + 用例组确认 -------------------------------------------
    op.add_column("tasks", sa.Column("version_minor", sa.Integer(), nullable=False,
                                     server_default="0"))
    op.add_column("tasks", sa.Column("cases_confirmed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("tasks", sa.Column("cases_confirmed_sha256", sa.String(length=80), nullable=True))
    op.add_column("tasks", sa.Column("cases_confirmed_by", sa.String(length=120), nullable=True))

    # ---- task_versions：任务版本历史 ------------------------------------------
    op.create_table(
        "task_versions",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("major", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("minor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="spec"),
        sa.Column("spec_version_id", sa.Integer(), sa.ForeignKey("spec_versions.id"), nullable=True),
        sa.Column("code_snapshot_sha256", sa.String(length=80), nullable=True),
        sa.Column("suite_ref", sa.String(length=200), nullable=True),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id"), nullable=True),
        sa.Column("label", sa.String(length=200), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("waivers_json", sa.Text(), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_task_versions_task_id", "task_versions", ["task_id"])
    op.create_index("ix_task_versions_is_current", "task_versions", ["is_current"])
    op.create_index("uq_task_versions_task_major_minor", "task_versions",
                    ["task_id", "major", "minor"], unique=True)

    # ---- task_documents：任务级文档 -------------------------------------------
    op.create_table(
        "task_documents",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(length=300), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="design"),
        sa.Column("path", sa.String(length=600), nullable=False),
        sa.Column("extracted_path", sa.String(length=600), nullable=True),
        sa.Column("sha256", sa.String(length=80), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_task_documents_task_id", "task_documents", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_task_documents_task_id", table_name="task_documents")
    op.drop_table("task_documents")

    op.drop_index("uq_task_versions_task_major_minor", table_name="task_versions")
    op.drop_index("ix_task_versions_is_current", table_name="task_versions")
    op.drop_index("ix_task_versions_task_id", table_name="task_versions")
    op.drop_table("task_versions")

    op.drop_column("tasks", "cases_confirmed_by")
    op.drop_column("tasks", "cases_confirmed_sha256")
    op.drop_column("tasks", "cases_confirmed_at")
    op.drop_column("tasks", "version_minor")
