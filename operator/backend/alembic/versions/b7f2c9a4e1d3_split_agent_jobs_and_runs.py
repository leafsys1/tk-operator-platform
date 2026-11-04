"""v1.1 契约最小集：拆 AgentJob / Run，新增 Loop 与 SpecVersion

Revision ID: b7f2c9a4e1d3
Revises: 47ca9ff7d1f1
Create Date: 2026-10-14

做法（方案 v1.1 §3.2，已定「单人自用、测试数据可直接重建」）：
- 老 `runs` 里存的其实都是**智能体调用** → 数据搬进新的 `agent_jobs`，老表退场；
- 新建 `runs`，语义改为**一次真实执行**（build/test/benchmark/profile）；
- 老 `artifacts` 行挂在老 runs 上（demo 数据），随老表一起清掉，由 seed 重新挂到真实执行上；
- 新增 `loops` / `spec_versions`；`tasks` 加 current_spec_version_id / next_action；
  顺便给 `host_images` 补 digest（性能可比性的固定条件）。
"""

from alembic import op
import sqlalchemy as sa

revision = "b7f2c9a4e1d3"
down_revision = "47ca9ff7d1f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0) 老 artifacts 全挂在「老 runs」（= 智能体调用）上 —— demo 数据，直接清掉
    op.execute("DELETE FROM artifacts")

    # 1) 新表：规格版本 / 轮次 / 智能体调用
    op.create_table(
        "spec_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("raw_input", sa.Text()),
        sa.Column("input_form", sa.String(80)),
        sa.Column("op_name", sa.String(200)),
        sa.Column("spec_json", sa.Text()),
        sa.Column("assumptions", sa.Text()),
        sa.Column("unknowns", sa.Text()),
        sa.Column("change_reason", sa.Text()),
        sa.Column("sha256", sa.String(80)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_spec_versions_task_id", "spec_versions", ["task_id"])

    op.create_table(
        "loops",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False, server_default="design"),
        sa.Column("scope", sa.Text()),
        sa.Column("hypothesis", sa.Text()),
        sa.Column("judge_condition", sa.Text()),
        sa.Column("conclusion", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_loops_task_id", "loops", ["task_id"])

    op.create_table(
        "agent_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id")),
        sa.Column("kind", sa.String(40), nullable=False, server_default="开发"),
        sa.Column("prompt", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("dsh_session_id", sa.String(120)),
        sa.Column("dsh_jsonl_path", sa.String(600)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("tokens_in", sa.BigInteger()),
        sa.Column("tokens_out", sa.BigInteger()),
        sa.Column("cache_read", sa.BigInteger()),
        sa.Column("cache_write", sa.BigInteger()),
        sa.Column("cost", sa.Float()),
    )
    op.create_index("ix_agent_jobs_task_id", "agent_jobs", ["task_id"])

    # 2) 老 runs 的数据搬进 agent_jobs（它们本来就是智能体调用）
    op.execute(
        """
        INSERT INTO agent_jobs (id, task_id, kind, prompt, status, dsh_session_id, dsh_jsonl_path,
                                started_at, ended_at, exit_code, tokens_in, tokens_out,
                                cache_read, cache_write, cost)
        SELECT id, task_id, kind, prompt, status, dsh_session_id, dsh_jsonl_path,
               started_at, ended_at, exit_code, tokens_in, tokens_out,
               cache_read, cache_write, cost
        FROM runs
        """
    )

    # 3) 老 runs 退场；新建「真实执行」runs
    op.drop_table("runs")
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("loop_id", sa.Integer(), sa.ForeignKey("loops.id")),
        sa.Column("agent_job_id", sa.Integer(), sa.ForeignKey("agent_jobs.id")),
        sa.Column("action", sa.String(32), nullable=False, server_default="test"),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("spec_version_id", sa.Integer(), sa.ForeignKey("spec_versions.id")),
        sa.Column("code_snapshot_sha256", sa.String(80)),
        sa.Column("suite_version_id", sa.Integer()),
        sa.Column("criteria_ref", sa.String(200)),
        sa.Column("host_id", sa.Integer(), sa.ForeignKey("hosts.id")),
        sa.Column("image_id", sa.Integer(), sa.ForeignKey("host_images.id")),
        sa.Column("image_digest", sa.String(200)),
        sa.Column("device_id", sa.String(80)),
        sa.Column("env_recipe_snapshot", sa.Text()),
        sa.Column("command", sa.Text()),
        sa.Column("cwd", sa.String(600)),
        sa.Column("remote_workdir", sa.String(600)),
        sa.Column("manifest_path", sa.String(600)),
        sa.Column("sync_state", sa.String(32)),
        sa.Column("failure_kind", sa.String(40)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("note", sa.Text()),
    )
    op.create_index("ix_runs_task_id", "runs", ["task_id"])
    op.create_index("ix_runs_agent_job_id", "runs", ["agent_job_id"])

    # 4) tasks：当前规格版本 + 下一步
    with op.batch_alter_table("tasks") as batch:
        batch.add_column(sa.Column("current_spec_version_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("next_action", sa.Text(), nullable=True))

    # 5) host_images：镜像 digest（性能和复现要固定它）
    with op.batch_alter_table("host_images") as batch:
        batch.add_column(sa.Column("digest", sa.String(200), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("host_images") as batch:
        batch.drop_column("digest")

    with op.batch_alter_table("tasks") as batch:
        batch.drop_column("next_action")
        batch.drop_column("current_spec_version_id")

    # 真实执行数据（新 runs / artifacts）在降级里丢弃 —— 它们可由契约重新产生
    op.execute("DELETE FROM artifacts")
    op.drop_table("runs")

    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False, server_default="开发"),
        sa.Column("host_id", sa.Integer(), sa.ForeignKey("hosts.id")),
        sa.Column("image_id", sa.Integer(), sa.ForeignKey("host_images.id")),
        sa.Column("dsh_session_id", sa.String(120)),
        sa.Column("dsh_jsonl_path", sa.String(600)),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("tokens_in", sa.BigInteger()),
        sa.Column("tokens_out", sa.BigInteger()),
        sa.Column("cache_read", sa.BigInteger()),
        sa.Column("cache_write", sa.BigInteger()),
        sa.Column("cost", sa.Float()),
        sa.Column("remote_workdir", sa.String(600)),
        sa.Column("sync_state", sa.String(32)),
        sa.Column("prompt", sa.Text()),
    )
    op.execute(
        """
        INSERT INTO runs (id, task_id, kind, status, dsh_session_id, dsh_jsonl_path,
                          started_at, ended_at, exit_code, tokens_in, tokens_out,
                          cache_read, cache_write, cost, prompt)
        SELECT id, task_id, kind, status, dsh_session_id, dsh_jsonl_path,
               started_at, ended_at, exit_code, tokens_in, tokens_out,
               cache_read, cache_write, cost, prompt
        FROM agent_jobs
        """
    )
    op.drop_table("agent_jobs")
    op.drop_table("loops")
    op.drop_table("spec_versions")
