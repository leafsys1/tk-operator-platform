"""数据模型 —— 方案 v1.1 §3。

层级：Project → Task → **Loop** → {AgentJob, Run} → Artifact，外加 SpecVersion（规格版本）、
hosts / host_images（资源）、stage_events（阶段留痕）。

三条硬规则（v1.1 §3.1）：
1. **业务通过状态不由 AgentJob 的退出码决定** —— `agent_jobs.exit_code == 0` 只说明 agent 跑完了。
2. 一轮 agent 调用（AgentJob）可以产生多个真实执行（Run）。
3. 版本关联必须留：代码快照哈希 / 规格版本 / 测试集版本 / 判据版本 / 镜像 digest / 环境配方快照。

时间一律存 UTC；事件明细不落库（真相源是 dsh 的 sessions/*.jsonl），这里只存指针与摘要。

`agent_jobs.is_demo`：P0 期手工登记的示例行（假 usage）保留为历史，但**任何消耗/成本聚合都要排除**它。
"""

import json
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


STAGES = ["需求与设计", "开发", "验收", "交付"]

#: 统一执行入口的动作（方案 v1.1 §6 的四个 + 2026-10-23 新增开发内冒烟 smoke）
#: smoke：编译 + 一条最小用例跑通，用于开发对话里的「先证明能跑」；不是正式正确性测试。
ACTIONS = ["build", "smoke", "test", "benchmark", "profile"]

#: 失败分类（方案 v1.1 §4.4）
FAILURE_KINDS = ["spec_unclear", "incomplete_output", "build_failed", "correctness_failed",
                 "measurement_invalid", "env_failed", "timeout"]

#: 判据档（方案 v1.1 §11.1：默认官方档，支持项目自定义档）。写法 = 档名@档版本号，
#: 档版本号是**这套判据本身**的版本，不是 CANN 版本；判定「报告是否与当前判据一致」要用它。
DEFAULT_CRITERIA_REF = "ops-precision-standard@1"


class Host(Base):
    """裸金属服务器（登记 + 探针结果）。"""

    __tablename__ = "hosts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    kind: Mapped[str] = mapped_column(String(32), default="baremetal")
    address: Mapped[str | None] = mapped_column(String(200))
    ssh_alias: Mapped[str | None] = mapped_column(String(120))
    cred_ref: Mapped[str | None] = mapped_column(String(200))  # 只存引用，如 PEM 路径或密钥名
    cards: Mapped[str | None] = mapped_column(String(200))  # 如 "8×910B1"
    cann_version: Mapped[str | None] = mapped_column(String(120))
    docker_ok: Mapped[bool | None] = mapped_column(Boolean)
    npu_state: Mapped[str | None] = mapped_column(Text)  # 探针原始输出（截断），便于排错
    probe_state: Mapped[str] = mapped_column(String(32), default="unknown")  # unknown|ok|fail
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    # 当前测试服务器：全局唯一（应用层保证只置一个）。任务未显式指定执行机时优先用它。
    is_test_server: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    images: Mapped[list["HostImage"]] = relationship(back_populates="host", cascade="all, delete-orphan")


class HostImage(Base):
    """宿主机上的镜像；is_op_dev 标记的即为「算子开发镜像」。"""

    __tablename__ = "host_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    host_id: Mapped[int] = mapped_column(ForeignKey("hosts.id", ondelete="CASCADE"))
    image_ref: Mapped[str] = mapped_column(String(400))  # repository
    tag: Mapped[str | None] = mapped_column(String(200))
    image_id: Mapped[str | None] = mapped_column(String(80))  # docker 的 IMAGE ID 短 id
    digest: Mapped[str | None] = mapped_column(String(200))  # 有 digest 就记（性能可比性的固定条件）
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    created_raw: Mapped[str | None] = mapped_column(String(120))  # docker 的 CREATED 原文
    dangling: Mapped[bool] = mapped_column(Boolean, default=False)  # <none>:<none>
    is_op_dev: Mapped[bool] = mapped_column(Boolean, default=False)
    # 测试镜像：任务选镜像时只暴露这些；未配置的默认隐藏（页面仍可看到并标记）
    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    env_recipe: Mapped[str | None] = mapped_column(Text)  # 怎么把它变成可跑环境
    preflight_cmd: Mapped[str | None] = mapped_column(Text)  # 可选自检命令
    cann_version: Mapped[str | None] = mapped_column(String(120))
    python_version: Mapped[str | None] = mapped_column(String(60))
    torch_version: Mapped[str | None] = mapped_column(String(60))
    scan_state: Mapped[str] = mapped_column(String(32), default="scanned")
    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)

    host: Mapped[Host] = relationship(back_populates="images")


class Project(Base):
    """项目 = 一个持续开发工作区 + 默认执行机/镜像（默认配置，不构成项目身份）。"""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    workspace_path: Mapped[str | None] = mapped_column(String(500))  # dsh 的 cwd / projectRoot
    host_id: Mapped[int | None] = mapped_column(ForeignKey("hosts.id"))
    image_id: Mapped[int | None] = mapped_column(ForeignKey("host_images.id"))
    dsh_profile: Mapped[str | None] = mapped_column(String(120), default="headless")
    dsh_home: Mapped[str | None] = mapped_column(String(500))
    criteria_ref: Mapped[str | None] = mapped_column(String(200))  # 判据档默认值（任务可覆盖）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    tasks: Mapped[list["Task"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Task(Base):
    """一个算子开发需求：目标、当前规格版本、阶段、下一步。"""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(300))
    op_name: Mapped[str | None] = mapped_column(String(200))
    target_chip: Mapped[str | None] = mapped_column(String(80))
    input_form: Mapped[str | None] = mapped_column(String(80))  # 自然语言 / 数学表达式 / 参考代码
    stage: Mapped[str] = mapped_column(String(32), default=STAGES[0])
    current_spec_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("spec_versions.id", use_alter=True, name="fk_tasks_current_spec")
    )
    criteria_ref: Mapped[str | None] = mapped_column(String(200))  # 判据档（空 = 用项目的默认）
    next_action: Mapped[str | None] = mapped_column(Text)  # 「下一步做什么」
    # —— 「任务开始」页（P0-3）创建期字段 ——
    # 草稿：在「任务开始」页里尚未定版的创建态。不进阶段统计、不算「已建任务」（方案 §13-A2 / UI-task-start-v1 §2.3）。
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False)
    # 归档：从主任务列表/阶段看板隐藏，但保留 run、Loop、存点和日志证据。
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    # 需求原文（草稿期承载用户原话；定版时转进 spec_versions.raw_input —— 别留成第二个真相源）
    raw_input: Mapped[str | None] = mapped_column(Text)
    # 以下四项：空 = 用项目默认（与 criteria_ref 同一口径，避免「建任务时被迫先把默认值抄一遍」）
    # 不设 DB 级 FK：SQLite 的 ALTER TABLE ADD COLUMN 加不了外键约束（Alembic 会退化成 ADD CONSTRAINT 报错），
    # 存在性由应用层校验（tasks 路由的 _check_task_refs）。
    host_id: Mapped[int | None] = mapped_column(Integer)  # 覆盖项目默认执行机
    image_id: Mapped[int | None] = mapped_column(Integer)  # 覆盖项目默认镜像
    device_id: Mapped[str | None] = mapped_column(String(80))  # 指定设备；空 = 自动选空卡
    timeout_s: Mapped[int | None] = mapped_column(Integer)  # 单次执行超时；空 = 默认 1800
    # D8 预算（按案例/按 Loop 计）：受控修复轮次上限、每动作执行次数上限；空 = 默认 3 / 2
    fix_rounds_max: Mapped[int | None] = mapped_column(Integer)
    budget_per_action: Mapped[int | None] = mapped_column(Integer)
    # —— 对话式需求澄清（P0-3 第二步）——
    # 业务证据（规格草稿：{fields, questions, ready}）与过程证据（对话）**分开存**（方案 §1.1）。
    spec_draft: Mapped[str | None] = mapped_column(Text)  # JSON {fields: {...}, questions: [...], ready: bool}
    chat_log: Mapped[str | None] = mapped_column(Text)  # JSON [{role, text}]
    dsh_session_id: Mapped[str | None] = mapped_column(String(120))  # 需求澄清用的 dsh 会话（--session-id 续会话）
    # —— 任务生命周期 v2（2026-10-23，docs/plan-task-lifecycle-v2.md）——
    # 大版本 = 当前 spec_versions.version；小版本 = 采纳优化 / 交付前人工变更时 +1。
    version_minor: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # 用例组确认（精度 + 性能整组确认）：未确认时四动作 plan 拦「用例组未确认」。
    cases_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cases_confirmed_sha256: Mapped[str | None] = mapped_column(String(80))
    cases_confirmed_scope: Mapped[str | None] = mapped_column(String(32))
    cases_confirmed_by: Mapped[str | None] = mapped_column(String(120))  # chat:job#123 / api
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="tasks")
    spec_versions: Mapped[list["SpecVersion"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", foreign_keys="SpecVersion.task_id"
    )
    loops: Mapped[list["Loop"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    agent_jobs: Mapped[list["AgentJob"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    runs: Mapped[list["Run"]] = relationship(back_populates="task", cascade="all, delete-orphan")


class SpecVersion(Base):
    """规格的一个版本：代码生成、测试生成、验收的共同依据（方案 v1.1 §4.3）。"""

    __tablename__ = "spec_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    raw_input: Mapped[str | None] = mapped_column(Text)  # 原始描述 / 参考代码路径
    input_form: Mapped[str | None] = mapped_column(String(80))
    op_name: Mapped[str | None] = mapped_column(String(200))
    spec_json: Mapped[str | None] = mapped_column(Text)  # 结构化规格（缺失事实标 UNKNOWN）
    assumptions: Mapped[str | None] = mapped_column(Text)  # 显式假设
    unknowns: Mapped[str | None] = mapped_column(Text)  # 未确定项
    change_reason: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    task: Mapped[Task] = relationship(back_populates="spec_versions", foreign_keys=[task_id])


class TaskDocument(Base):
    """任务级文档（2026-10-23 任务生命周期 v2）：任务开始页上传的《算子设计文档》。

    原文与解析稿都落工作区 `task-docs/task-<id>/`（`path` / `extracted_path` 相对工作区）；
    交付时原文原样进包。草稿任务（is_draft=1）也允许上传。
    """

    __tablename__ = "task_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    kind: Mapped[str] = mapped_column(String(32), default="design")  # design | other
    path: Mapped[str] = mapped_column(String(600))  # 相对工作区
    extracted_path: Mapped[str | None] = mapped_column(String(600))  # 相对工作区
    sha256: Mapped[str] = mapped_column(String(80))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskVersion(Base):
    """任务版本（2026-10-23 任务生命周期 v2）：大版本 = 规格版本，小版本 = 优化/人工变更。

    版本的**证据不按 run 打标**，而是靠 `(spec_version_id, code_snapshot_sha256, suite_ref)` 三键
    与 run 匹配 —— 所以「优化采纳」时候选 run 直接成为新小版本的证据，不需要重跑。
    """

    __tablename__ = "task_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    major: Mapped[int] = mapped_column(Integer, default=1)
    minor: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(32), default="spec")  # spec | optimization | manual
    spec_version_id: Mapped[int | None] = mapped_column(ForeignKey("spec_versions.id"))
    code_snapshot_sha256: Mapped[str | None] = mapped_column(String(80))
    suite_ref: Mapped[str | None] = mapped_column(String(200))
    evidence_scope: Mapped[str | None] = mapped_column(String(32))
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"))
    label: Mapped[str | None] = mapped_column(String(200))
    note: Mapped[str | None] = mapped_column(Text)
    # 对话确认的证据豁免：[{actions, reason, job_id, at}]；交付门禁据此放行并在 manifest 留痕。
    waivers_json: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("uq_task_versions_task_major_minor", "task_id", "major", "minor", unique=True),
    )


class Loop(Base):
    """一轮设计 / 修复 / 诊断 / 优化（方案 v1.1 §3.1）。"""

    __tablename__ = "loops"
    # 幂等确认：同一任务同一 idempotency_key 只能建一个轮次（NULL 不参与唯一约束）。
    __table_args__ = (
        Index("uq_loops_task_idempotency", "task_id", "idempotency_key", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32), default="design")  # design|fix|perf
    scope: Mapped[str | None] = mapped_column(Text)  # 这一轮改什么
    hypothesis: Mapped[str | None] = mapped_column(Text)  # 假设（尤其 fix/perf）
    judge_condition: Mapped[str | None] = mapped_column(Text)  # 判定条件
    conclusion: Mapped[str | None] = mapped_column(Text)  # accepted|rejected|inconclusive + 依据
    status: Mapped[str] = mapped_column(String(32), default="open")  # open|closed
    # 「agent 自动改代码」这一步的证据（M3 余项）：JSON —— 改前/改后快照、逐文件 diff、结构化摘要、
    # 回退数据。空 = 这一轮没跑过自动优化。见 services/optimize.py。
    optimize_json: Mapped[str | None] = mapped_column(Text)
    # 优化轮次的目标/约束/预算/用户补充文本/目标快照（P1，schema 见
    # docs/plan-optimize-rounds-v1.md §1）。空 = 不是「建轮次」页建的优化 Loop。
    objective_json: Mapped[str | None] = mapped_column(Text)
    # ---- 自动优化 Step 1：显式状态机（不再靠 runs/jobs 拓扑反推）----
    # idle = 非优化 Loop / 历史数据；见 docs/plan-optimize-auto-v1.md §5。
    phase: Mapped[str] = mapped_column(String(32), default="idle", server_default="idle")
    state_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    active_job_id: Mapped[int | None] = mapped_column(Integer)  # 无 FK（SQLite 加列限制）
    idempotency_key: Mapped[str | None] = mapped_column(String(120))
    phase_error: Mapped[str | None] = mapped_column(Text)
    phase_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task: Mapped[Task] = relationship(back_populates="loops")
    agent_jobs: Mapped[list["AgentJob"]] = relationship(back_populates="loop")
    runs: Mapped[list["Run"]] = relationship(back_populates="loop")

    @property
    def optimize(self) -> dict | None:
        """`optimize_json` 的解析视图（对外只读；写仍走 optimize_json）。"""
        if not self.optimize_json:
            return None
        try:
            data = json.loads(self.optimize_json)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    @property
    def objective(self) -> dict | None:
        """`objective_json` 的解析视图（对外只读；写仍走 objective_json）。"""
        if not self.objective_json:
            return None
        try:
            data = json.loads(self.objective_json)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None


class LoopEvent(Base):
    """自动优化轮次事件（append-only）：状态转移、启动/完成、失败、取消、恢复都留痕。

    只记指针/摘要；代码证据仍在 `loops.optimize_json`、`runs`、`savepoints`。
    """

    __tablename__ = "loop_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40))  # created|start|phase|agent_done|accept|reject|recover|cancel
    phase: Mapped[str | None] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(40), default="page")
    payload_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorkspaceLease(Base):
    """任务工作区的单飞租约（DB 级）：同一 project 同时只允许一个写/执行作业。

    存在原因（审查）：agent 作业与 action 作业各自只查自己的表，benchmark 跑着时仍可改码；
    进程内锁也不能覆盖重启/多 worker。租约由 `services/workspace_lease.py` 管理。
    """

    __tablename__ = "workspace_leases"
    __table_args__ = (
        Index("uq_workspace_leases_project", "project_id", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20))  # agent|action|restore
    job_id: Mapped[int | None] = mapped_column(Integer)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)


class LoopAttempt(Base):
    """一次「agent 改码尝试」（自动优化 Step 1b）。

    一个 Loop 可以有多次 attempt（失败/回退/重试）；每次 attempt 关联一个 agent job，
    记录它的基线 sha、候选 sha、结论与时间。状态由 `services/loop_state.py` 在
    `start_agent/finish_agent/cancel/reconcile` 时维护，页面只读。
    """

    __tablename__ = "loop_attempts"
    #: 轮内序号唯一：并发启动由 start_agent 的 CAS 兜底，这里再兜一次数据完整性。
    __table_args__ = (
        Index("uq_loop_attempts_loop_seq", "loop_id", "seq", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=1)
    agent_job_id: Mapped[int | None] = mapped_column(Integer)  # 无 FK（与 loops.active_job_id 同口径）
    conversation_id: Mapped[int | None] = mapped_column(Integer)
    baseline_savepoint_id: Mapped[int | None] = mapped_column(Integer)
    baseline_sha256: Mapped[str | None] = mapped_column(String(80))
    candidate_sha256: Mapped[str | None] = mapped_column(String(80))
    # running|succeeded|failed|blocked|cancelled
    status: Mapped[str] = mapped_column(String(20), default="running")
    error: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoopConfirmation(Base):
    """「确认优化草案」的一次性凭证（自动优化 Step 1b）。

    为什么不用 `loops` 上的几列：token 要能过期/重发、要留「确认时冻结了什么」的审计，
    且**同一事务里 CAS 消费**（`used_at` 只能写一次）比改 loop 行更直接。
    明文 token 只在 confirm 响应里出现一次，库里只存 sha256。
    """

    __tablename__ = "loop_confirmations"
    __table_args__ = (
        Index("uq_loop_confirmations_token", "token_hash", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64))
    draft_json: Mapped[str] = mapped_column(Text)
    objective_sha256: Mapped[str] = mapped_column(String(64))
    baseline_sha256: Mapped[str | None] = mapped_column(String(80))
    workspace_sha256: Mapped[str | None] = mapped_column(String(80))
    state_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    used_agent_job_id: Mapped[int | None] = mapped_column(Integer)


class LoopRunLink(Base):
    """Loop ↔ Run 的显式复用关联（自动优化 Step 1b）。

    背景：建轮次时会复用「建轮次前就存在、同 sha、同动作」的基线 run。早期做法是把
    `runs.loop_id` 改成新 Loop（破坏 run 的原归属，复用语义不可审计）；现在改为只写 link，
    读取证据按 `Run.loop_id == loop.id OR run.id IN (links)` 的并集取，历史数据仍兼容。
    """

    __tablename__ = "loop_run_links"
    __table_args__ = (
        Index("uq_loop_run_links_loop_run", "loop_id", "run_id", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="baseline")  # baseline|candidate|manual
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str] = mapped_column(String(40), default="api")


class LoopAutoRun(Base):
    """一轮「自动验证链」（自动优化 Step 2 orchestrator）。

    一次自动链 = 当前 attempt 的 build → smoke → test → benchmark → report；进度落在
    `steps_json`（每步状态/作业 id/门禁/远端清理结果），驱动任务在进程内，重启后由
    `loop_orchestrator.reconcile_startup` 标 blocked（不假装还在跑）。
    """

    __tablename__ = "loop_auto_runs"
    #: 同一个 loop 同时只允许一个 running auto_run（应用层 + phase CAS 双保险；SQLite 无部分索引）。
    __table_args__ = (
        Index("ix_loop_auto_runs_loop_status", "loop_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    current_step: Mapped[str | None] = mapped_column(String(20))
    current_action_job_id: Mapped[int | None] = mapped_column(Integer)
    attempt_id: Mapped[int | None] = mapped_column(Integer)
    baseline_sha256: Mapped[str | None] = mapped_column(String(80))
    candidate_sha256: Mapped[str | None] = mapped_column(String(80))
    steps_json: Mapped[str | None] = mapped_column(Text)
    budget_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OptimizationIdea(Base):
    """P2 优化点池：Agent 诊断与用户补充共享一张可去重、可追踪结果的表。"""

    __tablename__ = "optimization_ideas"
    # 去重是并发安全的关键：只有普通索引时 check-then-act 在并发下会落重复 idea（审查 M15）。
    # 迁移 a7b8c9d0e1f2 建同名唯一索引；写入方捕获 IntegrityError 后按 (task_id, dedupe_key) 重查。
    __table_args__ = (
        Index("uq_optimization_ideas_task_dedupe", "task_id", "dedupe_key", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"), index=True)
    search_id: Mapped[int | None] = mapped_column(ForeignKey("idea_searches.id"), index=True)
    source: Mapped[str] = mapped_column(String(20), default="agent")  # agent|user|conversation
    title: Mapped[str] = mapped_column(String(300))
    evidence_json: Mapped[str | None] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    change_scope: Mapped[str | None] = mapped_column(Text)
    expected: Mapped[str | None] = mapped_column(Text)
    risk: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[str | None] = mapped_column(String(32))
    effort: Mapped[str | None] = mapped_column(String(32))
    skill: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="proposed")
    dedupe_key: Mapped[str] = mapped_column(String(80), index=True)
    outcome_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdeaSearch(Base):
    """一次 P2 只读诊断搜索；具体 Agent 过程仍由 ``agent_jobs`` 承载。"""

    __tablename__ = "idea_searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    agent_job_id: Mapped[int | None] = mapped_column(ForeignKey("agent_jobs.id"), unique=True)
    user_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="running")
    summary: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoopBudgetAudit(Base):
    """P3 测量预算审计：每次允许/拒绝都留痕，拒绝不会产生真实 run。

    `claimed_at`：一次性预留的消费时间（审查 M4）。页面起动作时先写 allowed 审计（预留），
    `scripts/nm_test.sh` 执行前带 `OPX_BUDGET_AUDIT_ID` 回来 claim；claim 过的审计不能复用，
    替代了原来调用方自己就能设的 `OPX_BUDGET_RESERVED=1`。
    """

    __tablename__ = "loop_budget_audits"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    loop_id: Mapped[int] = mapped_column(ForeignKey("loops.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(20))  # allowed|rejected
    limit_count: Mapped[int] = mapped_column(Integer)
    used_count: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentJob(Base):
    """一次智能体调用（原 v1.0 的 runs 就是它）：dsh 会话 + usage 摘要。

    只回答「agent 干了什么、花了多少」，不回答「算子对不对」。
    """

    __tablename__ = "agent_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"))
    conversation_id: Mapped[int | None] = mapped_column(Integer)  # 对话页会话（无 FK，SQLite 加列限制）
    kind: Mapped[str] = mapped_column(String(40), default="开发")
    prompt: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|ok|failed|interrupted
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)  # P0 示例行：usage/cost 是假的，聚合必须排除
    dsh_session_id: Mapped[str | None] = mapped_column(String(120))
    dsh_jsonl_path: Mapped[str | None] = mapped_column(String(600))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    tokens_in: Mapped[int | None] = mapped_column(BigInteger)
    tokens_out: Mapped[int | None] = mapped_column(BigInteger)
    cache_read: Mapped[int | None] = mapped_column(BigInteger)
    cache_write: Mapped[int | None] = mapped_column(BigInteger)
    # 金额：P0 示例行/历史手工行可能预留；真实调用的费用按端点价目**实时算出**
    # （见 services/pricing.py），不在这里落库 —— 改了单价，历史消耗跟着更新。
    cost: Mapped[float | None] = mapped_column()
    # 这次调用用的端点 + 当时的模型名（计价依据/归属；不加 FK：删端点不能连坐删调用记录）
    endpoint_id: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(200))

    task: Mapped[Task] = relationship(back_populates="agent_jobs")
    loop: Mapped[Loop | None] = relationship(back_populates="agent_jobs")
    runs: Mapped[list["Run"]] = relationship(back_populates="agent_job")


class Run(Base):
    """一次**真实执行**：build / test / benchmark / profile（方案 v1.1 §6）。

    只回答「结果是什么、可不可信」，是门禁判定的依据。
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"))
    agent_job_id: Mapped[int | None] = mapped_column(ForeignKey("agent_jobs.id"))
    action: Mapped[str] = mapped_column(String(32), default="test")  # ACTIONS
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|ok|failed|interrupted
    # 版本 / 可比性（可复现的最低要求）
    spec_version_id: Mapped[int | None] = mapped_column(ForeignKey("spec_versions.id"))
    code_snapshot_sha256: Mapped[str | None] = mapped_column(String(80))
    suite_ref: Mapped[str | None] = mapped_column(String(200))  # 用例集版本：先存内容哈希 ref，P3 起用表
    # 新 run 直接用 operator-v1；旧 run 保留原 manifest/全局哈希，用下面两个别名
    # 做可审计迁移，不改写原始证据。
    evidence_scope: Mapped[str | None] = mapped_column(String(32))
    scoped_code_snapshot_sha256: Mapped[str | None] = mapped_column(String(80))
    scoped_suite_ref: Mapped[str | None] = mapped_column(String(200))
    criteria_ref: Mapped[str | None] = mapped_column(String(200))  # 判据档：档名@档版本（官方档 / 自定义档）
    host_id: Mapped[int | None] = mapped_column(ForeignKey("hosts.id"))
    image_id: Mapped[int | None] = mapped_column(ForeignKey("host_images.id"))
    image_digest: Mapped[str | None] = mapped_column(String(200))
    device_id: Mapped[str | None] = mapped_column(String(80))  # 如 device 3（性能执行按设备互斥）
    env_recipe_snapshot: Mapped[str | None] = mapped_column(Text)
    command: Mapped[str | None] = mapped_column(Text)
    cwd: Mapped[str | None] = mapped_column(String(600))
    remote_workdir: Mapped[str | None] = mapped_column(String(600))
    manifest_path: Mapped[str | None] = mapped_column(String(600))
    sync_state: Mapped[str | None] = mapped_column(String(32))  # pushed|pulled|cleaned
    failure_kind: Mapped[str | None] = mapped_column(String(40))  # FAILURE_KINDS
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text)

    task: Mapped[Task] = relationship(back_populates="runs")
    loop: Mapped[Loop | None] = relationship(back_populates="runs")
    agent_job: Mapped[AgentJob | None] = relationship(back_populates="runs")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class ActionJob(Base):
    """一次「点一下就真执行」的四动作作业（2026-10-20）。

    只记录**执行过程**（进程 / 日志 / 退出码 / 门禁结论）；真实结果仍由 `runs` 承载 ——
    `scripts/nm_test.sh` 跑完自己登记 run，本表只把「这次点击 → 那次 run」串起来。
    """

    __tablename__ = "action_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"))
    action: Mapped[str] = mapped_column(String(32))  # ACTIONS
    status: Mapped[str] = mapped_column(String(32), default="running")  # running|ok|failed|interrupted
    run_ref: Mapped[str] = mapped_column(String(120))  # 远端 run_id（= runs.remote_workdir）
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))  # 跑完回填
    device_id: Mapped[str | None] = mapped_column(String(80))
    command: Mapped[str | None] = mapped_column(Text)
    log: Mapped[str | None] = mapped_column(Text)  # 只留尾部（超长截断），过程日志为主
    exit_code: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    gate_verdict: Mapped[str | None] = mapped_column(String(32))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(60))  # spec|code_diff|cases|precision|perf|log|delivery|report|metric
    path: Mapped[str] = mapped_column(String(600))
    sha256: Mapped[str | None] = mapped_column(String(80))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="artifacts")


class StageEvent(Base):
    """阶段推进/打回留痕（谁、从哪到哪、为什么）。"""

    __tablename__ = "stage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    from_stage: Mapped[str | None] = mapped_column(String(32))
    to_stage: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str | None] = mapped_column(String(80))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    """dsh 会话与项目/任务的关联（P0-2 对话页）。

    `task_id` 绑定任务：任务会话（`mode=task`）用任务工作区（挂 skills）；
    轻对话（`mode=light`）用空目录（不挂 skills、不碰工作区）。
    `dsh_session_id` 在**第一轮跑完**后由 dsh 回填（之前为 NULL，用于续会话）。
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"))
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    dsh_session_id: Mapped[str | None] = mapped_column(String(120), unique=True)
    title: Mapped[str | None] = mapped_column(String(300))
    mode: Mapped[str] = mapped_column(String(16), default="task")  # task|light
    message_count: Mapped[int | None] = mapped_column(Integer)
    tokens: Mapped[int | None] = mapped_column(BigInteger)
    cost: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """对话消息（用户可见的正文；dsh 的完整事件仍在它自己的 JSONL 里）。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    text: Mapped[str] = mapped_column(Text)
    job_id: Mapped[int | None] = mapped_column(Integer)  # 对应的 agent_jobs.id（无 FK，避免循环）
    # 本轮改动审阅证据（JSON）：改前/改后快照 + 逐文件 diff + 回退数据（见 services/conversations.py）
    changes_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Attachment(Base):
    """对话附件（保存在工作区 `.opx-chat/conv-<id>/attachments/`，库里只存索引）。"""

    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(600))  # 相对工作区
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Endpoint(Base):
    """模型/API 端点（配置态；只存凭据引用名，不回显明文）。"""

    __tablename__ = "endpoints"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    provider: Mapped[str] = mapped_column(String(120))  # dsh-llm-deepseek | dsh-llm-pi-ai
    base_url: Mapped[str | None] = mapped_column(String(400))
    model: Mapped[str | None] = mapped_column(String(200))
    key_ref: Mapped[str | None] = mapped_column(String(200))  # 兼容旧的「环境变量名」引用
    # 页面配置的 key（本机 SQLite 存储；接口不回显，只返回 has_key）
    # ⚠️ 已废弃（2026-10-21 迁移到本机密钥库）：新数据只写 `secret_ref`，该列仅保留兼容旧库；
    # `llm_config.migrate_legacy_secrets()` 启动时会把旧值搬进 Keychain/文件并把这里清空。
    api_key: Mapped[str | None] = mapped_column(Text)
    # 本机密钥库引用（如 endpoint-1）：key 本体在 macOS Keychain / secret-tool / 0600 文件，
    # DB 只存引用名（见 services/secret_store.py）。
    secret_ref: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # 当前 agent 调用使用哪个端点
    max_model_len: Mapped[int | None] = mapped_column(Integer)
    probe_state: Mapped[str] = mapped_column(String(32), default="unknown")
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    # 计价口径：人民币元 / 每百万 token。cache 两项留空 = 不单独计价（按 0 计）；
    # price_in + price_out 都配了才算「已定价」，否则消耗页如实标「未定价」。
    price_in: Mapped[float | None] = mapped_column()
    price_out: Mapped[float | None] = mapped_column()
    price_cache_read: Mapped[float | None] = mapped_column()
    price_cache_write: Mapped[float | None] = mapped_column()
    # 软删（任务外页面审查 H2）：端点被删除后**保留行与 id**，历史 `agent_jobs.endpoint_id`
    # 仍能解析到当时的价目；SQLite 非 AUTOINCREMENT 主键也不会被新端点复用。
    # 列表/当前端点/创建重名检查都过滤已删行；`pricing.endpoints_by_id` 故意包含它们。
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Savepoint(Base):
    """版本存档（savepoints v1）：一个「标签 + 环境引用 + 时间 + 类型」的行，代码本体在
    内容寻址的 tar.gz 归档里（`archive_path` / `services/savepoints.py`）。

    存盘点**不重复存代码**：同一 `code_snapshot_sha256` 只归档一份（归档根见
    `settings.savepoint_root`）。`kind`：manual（手动）/ auto（自动挂点）/ rescue（回滚前急救；
    `parent_id` 记录「为回滚哪个存档而做」）。归档文件不随任务删除清理（内容寻址、可能被其它
    任务引用；GC 留 v2）。
    """

    __tablename__ = "savepoints"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(200))
    note: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="manual")  # manual|auto|rescue|baseline|candidate|accept|rejected
    code_snapshot_sha256: Mapped[str] = mapped_column(String(80), index=True)
    # 优化证据按算子代码计；完整归档仍独立校验，不能把两种哈希混用。
    evidence_scope: Mapped[str | None] = mapped_column(String(32))
    op_name_snapshot: Mapped[str | None] = mapped_column(String(200))
    archive_snapshot_sha256: Mapped[str | None] = mapped_column(String(80))
    archive_path: Mapped[str] = mapped_column(String(600))
    archive_bytes: Mapped[int | None] = mapped_column(BigInteger)
    # 环境引用：创建时取自快照对应的 run，没有则取 task 当前（可空）
    spec_version_id: Mapped[int | None] = mapped_column(ForeignKey("spec_versions.id"))
    criteria_ref: Mapped[str | None] = mapped_column(String(200))
    suite_ref: Mapped[str | None] = mapped_column(String(200))
    image_digest: Mapped[str | None] = mapped_column(String(200))
    env_recipe_snapshot: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("savepoints.id", ondelete="SET NULL"))
    # 归属优化轮次（P1）：baseline / candidate / accept / rejected 存点都挂在 Loop 上；
    # manual / auto / rescue 留空（它们不属于某一轮优化）。
    loop_id: Mapped[int | None] = mapped_column(ForeignKey("loops.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str] = mapped_column(String(80), default="page")
