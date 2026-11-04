"""对外 Pydantic 模型（读写分离；只暴露必要字段，凭据一律不回显明文）。

v1.1 起：AgentJob（一次智能体调用）与 Run（一次真实执行）分列，Loop / SpecVersion 登场。
"""

from datetime import datetime
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 任务阶段枚举（与 `models.STAGES` 同序同值）：写入口只收这四个，避免「瞎写的阶段」
#: 从阶段看板的存值列里消失（审查 #3）。
TaskStage = Literal["需求与设计", "开发", "验收", "交付"]


def validate_op_name(value: str | None) -> str | None:
    """算子名只允许「单段名称」：不接受 `/`、`\\`、`..`、绝对路径或控制字符。

    它会参与工作区路径拼接（`cases/<op>/cases.json`、`ops/<op>/`），必须挡住路径穿越
    （任务外页面审查 M15：未定版任务可把 op_name 设成 `../..` 读到工作区外）。
    """
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    if len(name) > 120:
        raise ValueError("op_name 过长（>120）")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("op_name 只能是单段名称（不能含 /、\\、.. 或绝对路径）")
    if any(ord(ch) < 32 for ch in name):
        raise ValueError("op_name 不能含控制字符")
    return name


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HostImageOut(ORMModel):
    id: int
    host_id: int
    image_ref: str
    tag: str | None = None
    image_id: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    created_raw: str | None = None
    dangling: bool = False
    is_op_dev: bool = False
    is_test: bool = False
    env_recipe: str | None = None
    preflight_cmd: str | None = None
    cann_version: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    scan_state: str = "scanned"  # scanned | stale | missing（stale=探针失败留的旧值；missing=远端已无）
    last_scanned_at: datetime | None = None
    note: str | None = None


class HostImagePatch(BaseModel):
    """镜像标记 / 元数据编辑。

    `torch_version` / `digest` 等此前只存在于模型列，PATCH schema 不含 → 前端传了被静默丢弃，
    页面标记的新镜像永远缺 `torch_version`，test/benchmark/profile 的 plan 永久缺项
    （任务外页面审查 M10）。这里补齐为可直接编辑的元数据字段。
    """

    is_op_dev: bool | None = None
    is_test: bool | None = None
    env_recipe: str | None = None
    preflight_cmd: str | None = None
    digest: str | None = None
    cann_version: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    note: str | None = None


class HostOut(ORMModel):
    id: int
    name: str
    kind: str
    address: str | None = None
    ssh_alias: str | None = None
    cred_ref: str | None = None  # 只存引用（私钥路径），不是私钥内容
    cards: str | None = None
    cann_version: str | None = None
    docker_ok: bool | None = None
    probe_state: str
    last_probe_at: datetime | None = None
    note: str | None = None
    is_test_server: bool = False
    image_count: int | None = None
    op_dev_image_count: int | None = None
    #: 最近一次探针的非致命错误摘要（从 npu_state 里回读；失败时给页面一个原因）
    probe_error: str | None = None


class HostCreate(BaseModel):
    # 名称可省：页面按「SSH 目标」自动补，后端也做同样的回退。
    name: str | None = None
    address: str | None = None
    ssh_alias: str | None = None
    cred_ref: str | None = None
    note: str | None = None


class HostUpdate(BaseModel):
    """改登记信息（探针结果不在这里改）。字段缺省 = 不改。"""

    name: str | None = None
    address: str | None = None
    ssh_alias: str | None = None
    cred_ref: str | None = None
    note: str | None = None


class ProjectOut(ORMModel):
    id: int
    name: str
    workspace_path: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    dsh_profile: str | None = None
    dsh_home: str | None = None
    criteria_ref: str | None = None  # 判据档默认值（档名@档版本）


class ProjectCreate(BaseModel):
    name: str
    workspace_path: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    dsh_home: str | None = None
    criteria_ref: str | None = None


class ProjectUpdate(BaseModel):
    """项目配置更新（默认执行机/镜像/工作区/判据档）。

    只改显式提交的字段（`exclude_unset`）；`image_id` / `host_id` / `criteria_ref` 显式传 null 表示清空。
    """

    name: str | None = None
    workspace_path: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    dsh_home: str | None = None
    criteria_ref: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _name_not_blank(cls, v):
        if v is None:
            return v
        text = str(v).strip()
        if not text:
            raise ValueError("project name 不能为空")
        return text


class ArtifactOut(ORMModel):
    id: int
    type: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: datetime


class SpecVersionOut(ORMModel):
    id: int
    task_id: int
    version: int
    input_form: str | None = None
    op_name: str | None = None
    raw_input: str | None = None
    spec_json: str | None = None
    assumptions: str | None = None
    unknowns: str | None = None
    change_reason: str | None = None
    sha256: str | None = None
    created_at: datetime


class SpecVersionCreate(BaseModel):
    raw_input: str | None = None
    input_form: str | None = None
    op_name: str | None = None
    spec_json: str | None = None
    assumptions: str | None = None
    unknowns: str | None = None
    change_reason: str | None = None
    set_current: bool = True

    @field_validator("op_name", mode="before")
    @classmethod
    def _op_name_safe(cls, v):
        return validate_op_name(v)


class LoopOut(ORMModel):
    id: int
    task_id: int
    kind: str
    scope: str | None = None
    hypothesis: str | None = None
    judge_condition: str | None = None
    conclusion: str | None = None
    status: str
    optimize: dict | None = None  # 「agent 自动改代码」的证据（M3 余项；见 services/optimize.py）
    objective: dict | None = None  # 优化轮次目标/约束/预算/目标快照（P1；见 services/optimization.py）
    # Step 1 状态机（见 services/loop_state.py）：页面/对话按 phase 渲染。
    phase: str = "idle"
    state_version: int = 0
    active_job_id: int | None = None
    phase_error: str | None = None
    phase_updated_at: datetime | None = None
    created_at: datetime
    ended_at: datetime | None = None


class LoopCreate(BaseModel):
    kind: str = "design"
    scope: str | None = None
    hypothesis: str | None = None
    judge_condition: str | None = None


class OptimizeRequest(BaseModel):
    """让 agent 按本轮 Loop 的假设自动改代码（M3 余项）。"""

    extra: str | None = None  # 用户补充说明（可选）
    reference_loop_id: int | None = None  # 参考哪一轮的对比结论/建议（默认取上一轮 perf Loop）


class OptimizeStartOut(BaseModel):
    """起一轮自动优化的响应：立刻拿 job_id，过程事件走 /api/agent-jobs/{id}/events。"""

    job_id: int
    loop_id: int
    prompt: str


class OptimizeChatRequest(BaseModel):
    """在对话里启动一轮自动优化（自动优化 = 跳对话窗口 + 输入参考指令回车开始）。

    Step 1b 起页面改走 `POST /loops/{id}/confirm` + `/start`；本端点保留兼容，可选传
    `confirm_token`（传了就校验并消费一次性确认，不传按 legacy 处理并写 `confirm_auto` 事件）。
    """

    instruction: str | None = None  # 用户参考指令（可选；空则由本轮 objective 决定）
    conversation_id: int | None = None  # 复用哪个会话；空则新建一个该任务的 task 会话
    confirm_token: str | None = None  # 可选：一次性确认 token（Step 1b）


class OptimizeChatStartOut(BaseModel):
    """对话内自动优化的启动响应（事件流仍走 /api/agent-jobs/{id}/events）。"""

    job_id: int
    loop_id: int
    conversation_id: int
    message_id: int  # 刚落库的用户消息 id
    prompt: str


class OptimizationConversationRequest(BaseModel):
    """从优化页的一句自然语言对话发起完整优化链路。"""

    direction: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None


class LoopConversationOut(BaseModel):
    """某轮次最近一次自动优化对应的会话（找不到返回 null）。"""

    conversation_id: int | None = None


class EndpointOut(ORMModel):
    """模型/API 端点（配置态）。key 只回 `has_key`，**不回显明文**。

    单价口径：人民币元 / 每百万 token；cache 两项留空 = 不单独计价（按 0 计）。
    `priced=True` 表示 price_in + price_out 都配了（消耗页才可能算出金额）。
    """

    id: int
    name: str
    provider: str
    base_url: str | None = None
    model: str | None = None
    key_ref: str | None = None
    has_key: bool = False
    #: key 存在哪个本机后端：keychain | secret-service | file | legacy（旧库明文，待迁移）
    key_backend: str | None = None
    is_active: bool = False
    max_model_len: int | None = None
    probe_state: str = "unknown"
    last_probe_at: datetime | None = None
    note: str | None = None
    price_in: float | None = None
    price_out: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None
    priced: bool = False  # price_in + price_out 都配了 = 已定价


class EndpointCreate(BaseModel):
    name: str
    provider: str = "dsh-llm-pi-ai"
    base_url: str | None = None
    model: str | None = None
    key_ref: str | None = "OPX_LLM_KEY"
    api_key: str | None = None  # 页面配置的 key；不填则回退 key_ref 环境变量
    is_active: bool = False
    max_model_len: int | None = None
    note: str | None = None
    # 单价（人民币元 / 每百万 token）；不填 = 未定价
    price_in: float | None = None
    price_out: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None


class EndpointUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    key_ref: str | None = None
    # 不传 = 不改；传 "" = 清空（清空后回退环境变量）
    api_key: str | None = None
    is_active: bool | None = None
    max_model_len: int | None = None
    note: str | None = None
    # 单价：不传 = 不改；显式传 null/0 = 清掉该项（0 等价于不单独计价）
    price_in: float | None = None
    price_out: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None


class AgentJobOut(ORMModel):
    """一次智能体调用（过程与消耗）。`is_demo=True` 的是 P0 示例行：usage/cost 是假的，聚合要排除。"""

    id: int
    task_id: int
    loop_id: int | None = None
    conversation_id: int | None = None  # 关联的对话（自动优化/对话作业）
    kind: str
    prompt: str | None = None
    status: str
    is_demo: bool = False
    dsh_session_id: str | None = None
    dsh_jsonl_path: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    cost: float | None = None  # 历史/示例行金额；真实调用改为按端点价目实时算
    cost_estimated: bool = False  # True = 金额是按当前端点价目估算的（调用时没留快照）
    endpoint_id: int | None = None
    model: str | None = None


class RunOut(ORMModel):
    """一次真实执行（结果与证据）。"""

    id: int
    task_id: int
    loop_id: int | None = None
    agent_job_id: int | None = None
    action: str
    status: str
    spec_version_id: int | None = None
    code_snapshot_sha256: str | None = None
    suite_ref: str | None = None
    evidence_scope: str | None = None
    scoped_code_snapshot_sha256: str | None = None
    scoped_suite_ref: str | None = None
    criteria_ref: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    image_digest: str | None = None
    device_id: str | None = None
    command: str | None = None
    cwd: str | None = None
    remote_workdir: str | None = None
    manifest_path: str | None = None
    sync_state: str | None = None
    failure_kind: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None
    note: str | None = None
    artifacts: list[ArtifactOut] = []


class RunCreate(BaseModel):
    """手工登记一次真实执行（P1.5 期还没有自动执行时用）。"""

    action: str = "test"
    status: str = "ok"
    host_id: int | None = None
    image_id: int | None = None
    spec_version_id: int | None = None
    loop_id: int | None = None
    agent_job_id: int | None = None
    code_snapshot_sha256: str | None = None
    suite_ref: str | None = None
    evidence_scope: str | None = None
    criteria_ref: str | None = None
    image_digest: str | None = None
    device_id: str | None = None
    env_recipe_snapshot: str | None = None
    command: str | None = None
    sync_state: str | None = None
    cwd: str | None = None
    remote_workdir: str | None = None
    manifest_path: str | None = None
    failure_kind: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = 0
    note: str | None = None
    artifacts: list[dict] = []


class RunPatch(BaseModel):
    """真实执行的**关联**可变部分（M3-a：把已有 run 挂到某个优化 Loop）。

    真实执行的数据（产物 / 版本 / 退出码 / 状态）一律不许改 —— 只改归属。
    """

    loop_id: int | None = None


class ActionRequest(BaseModel):
    """发起一次真实执行（四动作契约，方案 v1.1 §6）。"""

    action: str
    dry_run: bool = True  # 默认只算 plan，不碰远端
    host_id: int | None = None
    device_id: str | None = None
    timeout_s: int | None = None
    loop_id: int | None = None  # 挂到某一轮（perf Loop 的成对 A/B 用）


class ConversationCreate(BaseModel):
    """新建对话（P0-2 首版要求绑定任务）。"""

    task_id: int
    title: str | None = None
    mode: str = "task"  # task（挂 skills）/ light（空目录）


class MessageIn(BaseModel):
    text: str


class ActionJobOut(BaseModel):
    """一次「点一下就真执行」的作业视图（`services/action_runner.py`）。"""

    id: int
    task_id: int
    loop_id: int | None = None
    action: str
    status: str
    running: bool = False
    run_ref: str
    run_id: int | None = None
    device_id: str | None = None
    command: str | None = None
    log: str = ""
    exit_code: int | None = None
    error: str | None = None
    gate_verdict: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class TaskOut(ORMModel):
    id: int
    project_id: int
    title: str
    op_name: str | None = None
    target_chip: str | None = None
    input_form: str | None = None
    stage: str
    is_draft: bool = False  # 「任务开始」页的草稿（未定版；不进阶段统计）
    is_archived: bool = False  # 归档任务默认不进主列表/阶段看板，证据不删
    current_spec_version_id: int | None = None
    criteria_ref: str | None = None  # 判据档（空 = 用项目的默认）
    next_action: str | None = None
    host_id: int | None = None  # 覆盖项目默认（空 = 用项目）
    image_id: int | None = None
    device_id: str | None = None
    timeout_s: int | None = None
    fix_rounds_max: int | None = None
    budget_per_action: int | None = None
    # —— 任务生命周期 v2（2026-10-23）——
    version_minor: int = 0  # 小版本；大版本 = 当前 spec_versions.version
    cases_confirmed_at: datetime | None = None  # 精度/性能用例组确认时间（UTC）
    cases_confirmed_sha256: str | None = None  # 确认时 cases/ 目录哈希
    cases_confirmed_by: str | None = None  # chat:job#123 / api
    created_at: datetime


class LoopRunLinkOut(ORMModel):
    loop_id: int
    run_id: int
    role: str


class TaskDetail(TaskOut):
    """任务详情：需求/规格 + 过程（agent_jobs）+ 结果（runs）分列。"""

    raw_input: str | None = None  # 需求原文（草稿期承载用户原话；定版时转进 spec_versions）
    dsh_session_id: str | None = None  # 需求澄清用的 dsh 会话（--session-id 续会话）
    spec_draft: dict | None = None  # 规格草稿（业务证据）：{fields, questions, ready}
    chat_log: list[dict] = []  # 对话记录（过程证据）：[{role, text}]
    agent_jobs: list[AgentJobOut] = []
    runs: list[RunOut] = []
    spec_versions: list[SpecVersionOut] = []
    loops: list[LoopOut] = []
    loop_run_links: list[LoopRunLinkOut] = []

    @field_validator("spec_draft", mode="before")
    @classmethod
    def _parse_spec_draft(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except ValueError:
                return None
        return v

    @field_validator("chat_log", mode="before")
    @classmethod
    def _parse_chat_log(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except ValueError:
                return []
        return v if v is not None else []


class TaskCreate(BaseModel):
    project_id: int
    title: str
    op_name: str | None = None
    target_chip: str | None = "910B1"
    input_form: str | None = None
    stage: TaskStage | None = None
    criteria_ref: str | None = None
    next_action: str | None = None
    is_draft: bool = False
    is_archived: bool = False
    raw_input: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    device_id: str | None = None
    timeout_s: int | None = None
    fix_rounds_max: int | None = None
    budget_per_action: int | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _title_not_blank(cls, v):
        """title 是 NOT NULL：显式空串也拒（否则任务列表出现无标题行）。"""
        if v is None or not str(v).strip():
            raise ValueError("title 不能为空")
        return str(v).strip()

    @field_validator("op_name", mode="before")
    @classmethod
    def _op_name_safe(cls, v):
        return validate_op_name(v)


class TaskUpdate(BaseModel):
    """PATCH 存草稿 / 回填：只改显式给的字段（`exclude_unset` 区分「没传」与「传 null」）。

    `title` / `is_draft` 是不可空字段：显式传 null 直接 422（而不是写库时 NOT NULL 500）；
    真正可空的覆盖字段（host/image/device/...）仍可用 null 表示「回到项目默认」。
    """

    title: str | None = None
    op_name: str | None = None
    target_chip: str | None = None
    input_form: str | None = None
    stage: TaskStage | None = None
    criteria_ref: str | None = None
    next_action: str | None = None
    is_draft: bool | None = None
    is_archived: bool | None = None
    raw_input: str | None = None
    host_id: int | None = None
    image_id: int | None = None
    device_id: str | None = None
    timeout_s: int | None = None
    fix_rounds_max: int | None = None
    budget_per_action: int | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _title_not_none_or_blank(cls, v):
        if v is None:
            raise ValueError("title 不能为 null（要改标题请传非空字符串）")
        if not str(v).strip():
            raise ValueError("title 不能为空")
        return str(v).strip()

    @field_validator("is_draft", mode="before")
    @classmethod
    def _is_draft_not_none(cls, v):
        if v is None:
            raise ValueError("is_draft 不能为 null（显式传 true / false）")
        return v

    @field_validator("is_archived", mode="before")
    @classmethod
    def _is_archived_not_none(cls, v):
        if v is None:
            raise ValueError("is_archived 不能为 null（显式传 true / false）")
        return v

    @field_validator("op_name", mode="before")
    @classmethod
    def _op_name_safe(cls, v):
        return validate_op_name(v)


class ChatRequest(BaseModel):
    """一步对话式需求澄清：用户这一轮说的话。"""

    message: str


class ChatQuestion(BaseModel):
    id: str | None = None
    text: str
    answer: str | None = None


class ChatResponse(BaseModel):
    """一轮澄清的结果：回复 + 规格草稿 + 待确认项 + 是否已可进入下一步。"""

    reply: str
    spec: dict = {}
    questions: list[ChatQuestion] = []
    ready: bool = False
    messages: list[dict] = []  # 全量对话（供前端整段刷新）
    usage: dict | None = None


class ChatStreamOut(BaseModel):
    """流式对话的启动响应：立刻拿 job_id，过程事件走 /api/agent-jobs/{id}/events。"""

    job_id: int
    messages: list[dict] = []


class ContextFile(BaseModel):
    path: str
    sha256: str | None = None
    size_bytes: int | None = None


class FinalizeResponse(BaseModel):
    """定版结果：spec v1 落库 + 上下文空间落盘。"""

    spec_version: SpecVersionOut
    space_dir: str
    files: list[ContextFile] = []


class SavepointCreate(BaseModel):
    """手动建一个存档（`POST /api/tasks/{task_id}/savepoints`）。

    `label` 缺省 = 「存档 YYYY-MM-DD HH:mm」；`note` 可选。
    """

    label: str | None = None
    note: str | None = None


class SavepointRestore(BaseModel):
    """回滚到某个存档：`scope=task`（默认，只回滚任务范围路径）/ `full`（全工作区）。"""

    scope: Literal["task", "full"] = "task"
    note: str | None = None


class OptimizationRoundCreate(BaseModel):
    """建一个优化轮次（`POST /api/tasks/{task_id}/optimization-rounds`）。

    `objective` 的 schema 见 `docs/plan-optimize-rounds-v1.md` §1；后端会补齐
    `targets_snapshot`（以建轮次时的最新 ok benchmark/test 报告为准）。
    """

    objective: dict
    label: str | None = None
    hypothesis: str | None = None
    attach_baseline_runs: bool = True
    #: 幂等键：同一任务同一 key 双击/重放只建一个轮次（返回已有轮次）。
    idempotency_key: str | None = None


class LoopAcceptRequest(BaseModel):
    """采纳本轮最新候选为版本（`POST /api/loops/{id}/accept`）。"""

    label: str | None = None
    note: str | None = None


class LoopRejectRequest(BaseModel):
    """否决本轮（`POST /api/loops/{id}/reject`）。"""

    reason: str | None = None
    #: 是否同时把工作区回退到本轮改前（优先 automatic-optimize 改前快照，其次 baseline 存点）。
    revert: bool = False


class LoopCancelRequest(BaseModel):
    """取消本轮正在跑的自动优化（Step 1）。

    `revert=true` 时仅记录意图；v1 的自动回滚仍由 agent 的 verifier/allowed_paths guard
    与人工「否决并回退」负责，取消本身不静默改工作区。
    """

    reason: str | None = None
    revert: bool = False


class LoopConfirmRequest(BaseModel):
    """确认优化草案（Step 1b）：冻结目标/基线/参考指令，返回一次性确认 token。

    契约见 `docs/plan-optimize-auto-v1.md` §15.1。`instruction`/`reference_loop_id`
    会被写进 draft；之后的 `/start` 只能按 draft 启动，不能改。
    """

    instruction: str | None = None
    reference_loop_id: int | None = None
    note: str | None = None


class LoopStartRequest(BaseModel):
    """按已确认的草案启动 agent 改码（Step 1b）：token 一次性，启动即消费。

    `token` 允许缺省：由业务层返回 409 + `E_CONFIRM_TOKEN`（契约口径），而不是 FastAPI 422。
    """

    token: str | None = None
    conversation_id: int | None = None


class OptimizationIdeaCreate(BaseModel):
    """用户补充优化思路；其余字段可逐步补全。"""

    title: str
    evidence: dict | list | str | None = None
    hypothesis: str | None = None
    change_scope: str | None = None
    expected: str | None = None
    risk: str | None = None
    confidence: str | None = None
    effort: str | None = None
    skill: str | None = None


class IdeaSearchRequest(BaseModel):
    """起一次 P2 只读优化点搜索；`extra` 会直拼进 prompt，必须有长度上限（审查 M10）。"""

    extra: str | None = Field(default=None, max_length=4000)


class OptimizationIdeaRoundCreate(BaseModel):
    """选择一个思路后建独立 Loop；objective 留空时后端生成性能优化默认值。"""

    objective: dict | None = None
    attach_baseline_runs: bool = True


class OptimizationIdeaStatusPatch(BaseModel):
    status: Literal["proposed", "selected", "validating", "verified", "rejected", "archived"]
    outcome: dict | None = None
