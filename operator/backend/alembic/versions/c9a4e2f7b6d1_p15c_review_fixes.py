"""P1.5-c 前的复查修正：判据档字段、demo 标记、用例集 ref、规格单一真相源、镜像可与不可跑区分

Revision ID: c9a4e2f7b6d1
Revises: b7f2c9a4e1d3
Create Date: 2026-10-15

改动来源 = 2026-10-15 对方案 v1.1 的复查（事实修正 F5/F6/F7/F8/F9 + 判据档载体）：

- `projects.criteria_ref` / `tasks.criteria_ref`：判据档要有载体（`档名@档版本`），
  任务为空则用项目默认；原先 `apply` 时把判据硬编码在代码里（方案 §11.1 要求可自定义）。
- `tasks.spec_json` 删列：v1.1 起规格的唯一真相源是 `spec_versions`（原来两处都有 = 双真相源）。
  删之前先给还带 spec_json 的任务补一版 `spec_versions`，并把它设为 current。
- `agent_jobs.is_demo`：P0 期手工登记的示例行（假 usage/cost）保留为历史但显式标记，
  消耗聚合必须排除（v1.1 的证据口径：不能有「看起来是真的」的假数据混在里面）。
- `runs.suite_version_id` → `runs.suite_ref`：原列是个指向**不存在的表**的半截外键；
  改为存用例集内容哈希 ref，P3 真上用例集表时再迁。
- 镜像数据修正（P1.5-a 实测结论回填）：能跑算子的那个镜像（带 torch/torch_npu）标 is_op_dev 并写版本；
  只有 CANN 工具链、**不带 torch** 的镜像摘掉 is_op_dev（它只适合做编译底座）；
  项目默认镜像从「不能跑的」切到「能跑的」。

数据都是单人自用、可重建的测试数据；降级只还原结构（不还原数据）。
"""

from alembic import op
import sqlalchemy as sa

revision = "c9a4e2f7b6d1"
down_revision = "b7f2c9a4e1d3"
branch_labels = None
depends_on = None

DEFAULT_CRITERIA = "ops-precision-standard@1"

#: 能跑算子的归档镜像的标识（P1.5-a 实测：torch 2.10.0 + torch_npu 2.10.0.post4 + 8×910B1）
OP_DEV_RUNNABLE_IMAGE = "lfx-op-dev-910b"
OP_DEV_RUNNABLE_TAG = "torch210-cann910"
#: 只有工具链、不带 torch 的镜像（只能做编译底座）。注意 `image_ref` 存的是完整仓库名，
#: 会带 registry 前缀（如 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann`）→ 用 LIKE 匹配。
TOOLCHAIN_ONLY_IMAGE_LIKE = "%ascendhub/cann"
TOOLCHAIN_ONLY_TAG_LIKE = "9.2.0-beta.2%"


def upgrade() -> None:
    bind = op.get_bind()

    # ---- 1) 判据档载体 ----------------------------------------------------------
    op.add_column("projects", sa.Column("criteria_ref", sa.String(200)))
    op.add_column("tasks", sa.Column("criteria_ref", sa.String(200)))
    bind.execute(sa.text("UPDATE projects SET criteria_ref = :c WHERE criteria_ref IS NULL"),
                 {"c": DEFAULT_CRITERIA})

    # ---- 2) demo 行显式标记 -----------------------------------------------------
    op.add_column("agent_jobs", sa.Column("is_demo", sa.Boolean(), nullable=False, server_default="0"))
    bind.execute(sa.text(
        "UPDATE agent_jobs SET is_demo = 1 "
        "WHERE dsh_session_id LIKE 'session-demo%' OR prompt LIKE '%P0 示例%'"
    ))

    # ---- 3) 用例集版本：半截外键 → 内容哈希 ref ------------------------------------
    op.add_column("runs", sa.Column("suite_ref", sa.String(200)))
    op.execute("ALTER TABLE runs DROP COLUMN suite_version_id")

    # ---- 4) 规格单一真相源：spec_json → spec_versions，然后删列 ---------------------
    rows = bind.execute(sa.text(
        "SELECT id, spec_json, op_name, input_form, title FROM tasks WHERE spec_json IS NOT NULL AND spec_json <> ''"
    )).mappings().all()
    for r in rows:
        has_spec = bind.execute(
            sa.text("SELECT COUNT(*) FROM spec_versions WHERE task_id = :t"), {"t": r["id"]}
        ).scalar_one()
        if has_spec:
            continue
        import hashlib
        import json
        body = r["spec_json"] + "|"
        bind.execute(sa.text(
            "INSERT INTO spec_versions (task_id, version, raw_input, input_form, op_name, spec_json, "
            "assumptions, unknowns, change_reason, sha256, created_at) "
            "VALUES (:t, 1, :raw, :iform, :op, :sj, :asum, :unk, :reason, :sha, :now)"
        ), {
            "t": r["id"],
            "raw": f"（P0 期手工登记 {r['title']}，2026-10-15 复查时从 tasks.spec_json 迁入）",
            "iform": r["input_form"],
            "op": r["op_name"],
            "sj": r["spec_json"],
            "asum": json.dumps({"shape/dtype 是登记时给定的，未声明容差与特殊值行为": True}, ensure_ascii=False),
            "unk": json.dumps(["容差策略", "特殊值（NaN/Inf/空输入）行为", "性能测量范围"], ensure_ascii=False),
            "reason": "迁移：tasks.spec_json → spec_versions（v1.1 规格单一真相源）",
            "sha": hashlib.sha256(body.encode()).hexdigest(),
            "now": "2026-10-15 00:00:00.000000",
        })
        new_id = bind.execute(sa.text(
            "SELECT id FROM spec_versions WHERE task_id = :t ORDER BY version DESC LIMIT 1"
        ), {"t": r["id"]}).scalar_one()
        bind.execute(sa.text("UPDATE tasks SET current_spec_version_id = :s WHERE id = :t"),
                     {"s": new_id, "t": r["id"]})
    op.execute("ALTER TABLE tasks DROP COLUMN spec_json")

    # ---- 5) 镜像数据修正（P1.5-a 实测结论回填） ------------------------------------
    note_run = " ｜ 2026-10-15 复查：这是**能跑算子**的那个（torch_npu 可用，8×910B1 可见）"
    note_tool = (
        " ｜ 2026-10-15 复查：**不带 torch/torch_npu**，只能做编译 / 工具链底座；"
        f"跑算子请用 {OP_DEV_RUNNABLE_IMAGE}:{OP_DEV_RUNNABLE_TAG}（P1.5-a 实测）"
    )
    bind.execute(sa.text(
        "UPDATE host_images SET is_op_dev = 1, "
        " torch_version = '2.10.0+cpu / torch_npu 2.10.0.post4', "
        " cann_version = COALESCE(cann_version, 'CANN 9.0（容器内，见 env_recipe）'), "
        " note = COALESCE(note, '') || :note "
        "WHERE image_ref = :ref AND tag = :tag"
    ), {"note": note_run, "ref": OP_DEV_RUNNABLE_IMAGE, "tag": OP_DEV_RUNNABLE_TAG})

    bind.execute(sa.text(
        "UPDATE host_images SET is_op_dev = 0, note = COALESCE(note, '') || :note "
        "WHERE image_ref LIKE :tool_ref AND tag LIKE :tool_tag"
    ), {"note": note_tool, "tool_ref": TOOLCHAIN_ONLY_IMAGE_LIKE, "tool_tag": TOOLCHAIN_ONLY_TAG_LIKE})

    # 项目默认镜像：从「起得来但跑不了算子」切到能跑的那个
    runnable_id = bind.execute(sa.text(
        "SELECT id FROM host_images WHERE image_ref = :ref AND tag = :tag ORDER BY id LIMIT 1"
    ), {"ref": OP_DEV_RUNNABLE_IMAGE, "tag": OP_DEV_RUNNABLE_TAG}).scalar_one_or_none()
    if runnable_id is not None:
        bind.execute(sa.text(
            "UPDATE projects SET image_id = :img WHERE image_id IS NULL OR image_id IN ("
            "  SELECT id FROM host_images WHERE COALESCE(is_op_dev, 0) = 0)"
        ), {"img": runnable_id})


def downgrade() -> None:
    bind = op.get_bind()
    op.add_column("tasks", sa.Column("spec_json", sa.Text()))
    op.add_column("runs", sa.Column("suite_version_id", sa.Integer()))
    op.execute("ALTER TABLE runs DROP COLUMN suite_ref")
    op.execute("ALTER TABLE agent_jobs DROP COLUMN is_demo")
    op.execute("ALTER TABLE tasks DROP COLUMN criteria_ref")
    op.execute("ALTER TABLE projects DROP COLUMN criteria_ref")
    bind.execute(sa.text(
        "UPDATE tasks SET spec_json = (SELECT spec_json FROM spec_versions s WHERE s.id = tasks.current_spec_version_id)"
    ))
