import type { ActionJob, CaseReport, CasesConfirmation, CasesConfirmResult, ChatResult, ChatStreamResult, ChipProfile, ChipSummary, CaseLibrary, ContextSpace, Conversation, DeliveryList, DeliveryRecord, WorkspaceDiff, WorkspaceFileText, WorkspaceInfo, EndpointConf, EndpointProbeResult, ExecutionPlan, FinalizeResult, Host, HostImage, IdeaSearch, Loop, LoopAttempts, LoopAutoRun, LoopAutoRunCancel, LoopBudgetAudit, LoopConfirmResult, LoopConversation, OptimizationAcceptResult, OptimizationConversationResult, OptimizationIdea, OptimizationIdeaRoundCreate, OptimizationRejectResult, OptimizationReport, OptimizationReportDetail, OptimizationReports, OptimizationReportStored, OptimizationReportVerify, OptimizationRoundCreate, OptimizationRoundCreated, OptimizationTargets, OptimizeChatStart, OptimizeEvidence, PerfDiff, Project, Run, RunReport, Savepoint, SavepointDiff, SavepointRestoreResult, ShellResult, SpecVersion, SpecVersionWrite, StageBoard, StageTaskView, Task, TaskDeletePreview, TaskDeleteResult, TaskDetail, TaskDocument, TaskGates, TaskVersionWrite, TaskVersions, UsageReport } from "../types";
async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    // 放宽到 ~4000：409 的 detail 可能是多 issue 人话（confirm 的基线/工作区/忙等），
    // 截太短会变成解析不了的半截 JSON，界面上只剩原始 JSON。
    throw new Error(`${res.status} ${text.slice(0, 4000)}`);
  }
  if (res.status === 204) return undefined as T; // 无内容（如 DELETE）
  return (await res.json()) as T;
}

/** 建草稿 / 建任务的字段（全可选，除 project_id/title）。 */
export type TaskWriteBody = {
  title?: string;
  op_name?: string | null;
  target_chip?: string | null;
  input_form?: string | null;
  criteria_ref?: string | null;
  next_action?: string | null;
  is_draft?: boolean;
  is_archived?: boolean;
  raw_input?: string | null;
  host_id?: number | null;
  image_id?: number | null;
  device_id?: string | null;
  timeout_s?: number | null;
  fix_rounds_max?: number | null;
  budget_per_action?: number | null;
};

export type HostWriteBody = {
  /** 可省：留空时后端取 SSH 目标作为显示名。 */
  name?: string;
  address?: string | null;
  ssh_alias?: string | null;
  /** 私钥路径引用；留空 = 用全局 OPX_SSH_KEY。 */
  cred_ref?: string | null;
  note?: string | null;
};

export type HostUpdateBody = Partial<HostWriteBody>;

/** `POST /api/tasks/{id}/deliveries` 在交付门禁不满足时（409）抛的错误：detail 人话 + codes 机读缺项码。 */
export type DeliveryCreateError = Error & {
  status?: number;
  detail?: string;
  codes?: string[];
  /** 409 时后端附带的缺项（`{code,item,path,reason}`；GET /versions 的 evidence.missing 是另一份更全的） */
  missing?: { code?: string; item?: string; path?: string; reason?: string }[];
};

/** 上传私钥后的返回值：只回本机落点路径和原文件名，不回私钥内容。 */
export type SshKeyUpload = { path: string; name: string };

/** API 端点写操作：key 只写不读（响应只有 has_key）。 */
export type EndpointWriteBody = {
  name?: string;
  provider?: string;
  base_url?: string | null;
  model?: string | null;
  key_ref?: string | null;
  api_key?: string;
  is_active?: boolean;
  max_model_len?: number | null;
  note?: string | null;
  // 单价：人民币元 / 每百万 token；传 null = 清掉该项（按 0 / 未定价处理）
  price_in?: number | null;
  price_out?: number | null;
  price_cache_read?: number | null;
  price_cache_write?: number | null;
};

export const api = {
  hosts: () => req<Host[]>("/api/hosts"),
  createHost: (body: HostWriteBody) =>
    req<Host>("/api/hosts", { method: "POST", body: JSON.stringify(body) }),
  patchHost: (id: number, body: HostUpdateBody) =>
    req<Host>(`/api/hosts/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteHost: (id: number) => req<void>(`/api/hosts/${id}`, { method: "DELETE" }),
  // FormData 不能走 req（它会强制 JSON Content-Type），这里单独 fetch。
  uploadSshKey: async (file: File): Promise<SshKeyUpload> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/hosts/ssh-key", { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 200)}`);
    return (await res.json()) as SshKeyUpload;
  },
  setTestHost: (id: number) => req<Host>(`/api/hosts/${id}/set-test`, { method: "POST" }),
  probe: (id: number) => req<Host>(`/api/hosts/${id}/probe`, { method: "POST" }),
  images: (hostId: number) => req<HostImage[]>(`/api/hosts/${hostId}/images`),
  patchImage: (id: number, body: Partial<HostImage>) =>
    req<HostImage>(`/api/host-images/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  projects: () => req<Project[]>("/api/projects"),
  tasks: (draft?: boolean | null) =>
    req<Task[]>(`/api/tasks${draft === undefined || draft === null ? "" : `?draft=${draft}`}`),
  task: (id: number) => req<TaskDetail>(`/api/tasks/${id}`),
  deletePreview: (id: number) => req<TaskDeletePreview>(`/api/tasks/${id}/delete-preview`),
  deleteTask: (id: number, deleteFiles: boolean) =>
    req<TaskDeleteResult>(`/api/tasks/${id}?delete_files=${deleteFiles}`, { method: "DELETE" }),
  createTask: (projectId: number, body: TaskWriteBody) =>
    req<TaskDetail>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ project_id: projectId, title: "未命名任务", ...body }),
    }),
  patchTask: (id: number, body: TaskWriteBody) =>
    req<TaskDetail>(`/api/tasks/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  chat: (taskId: number, message: string) =>
    req<ChatResult>(`/api/tasks/${taskId}/chat`, { method: "POST", body: JSON.stringify({ message }) }),
  startChatStream: (taskId: number, message: string) =>
    req<ChatStreamResult>(`/api/tasks/${taskId}/chat/stream`, { method: "POST", body: JSON.stringify({ message }) }),
  finalize: (taskId: number) => req<FinalizeResult>(`/api/tasks/${taskId}/finalize`, { method: "POST" }),
  eventsUrl: (jobId: number) => `/api/agent-jobs/${jobId}/events`,
  agentLive: (jobId: number) =>
    req<{ running: boolean; exit_code?: number | null; final?: string; prose?: string; error?: string | null }>(
      `/api/agent-jobs/${jobId}/live`,
    ),
  /** 取消一次智能体调用（任务页「卡住/不想等了」的自救入口）。 */
  cancelAgentJob: (jobId: number) =>
    req<{ result: string }>(`/api/agent-jobs/${jobId}/cancel`, { method: "POST" }),
  protoOpencodeStreamUrl: (prompt: string) =>
    `/api/proto/opencode/stream?prompt=${encodeURIComponent(prompt)}`,
  protoDshStreamUrl: (prompt: string) =>
    `/api/proto/dsh/stream?prompt=${encodeURIComponent(prompt)}`,
  planAction: (taskId: number, action: string, deviceId?: string | null) =>
    req<ExecutionPlan>(`/api/tasks/${taskId}/actions`, {
      method: "POST",
      body: JSON.stringify({ action, dry_run: true, device_id: deviceId ?? null }),
    }),
  // ---- 四动作真执行（异步作业，2026-10-20）----
  runAction: (taskId: number, action: string, deviceId?: string | null, loopId?: number | null) =>
    req<ExecutionPlan & { action_job_id: number; run_ref: string }>(`/api/tasks/${taskId}/actions`, {
      method: "POST",
      body: JSON.stringify({ action, dry_run: false, device_id: deviceId ?? null, loop_id: loopId ?? null }),
    }),
  actionJob: (jobId: number) => req<ActionJob>(`/api/action-jobs/${jobId}`),
  actionJobs: (taskId: number) => req<ActionJob[]>(`/api/tasks/${taskId}/action-jobs`),
  cancelActionJob: (jobId: number) =>
    req<{ result: string }>(`/api/action-jobs/${jobId}/cancel`, { method: "POST" }),
  // ---- 自动优化闭环（M3-a）----
  loops: (taskId: number) => req<Loop[]>(`/api/tasks/${taskId}/loops`),
  createLoop: (taskId: number, body: { kind?: string; scope?: string | null; hypothesis?: string | null; judge_condition?: string | null }) =>
    req<Loop>(`/api/tasks/${taskId}/loops`, { method: "POST", body: JSON.stringify(body) }),
  closeLoop: (loopId: number, conclusion: string) =>
    req<Loop>(`/api/loops/${loopId}/close?conclusion=${encodeURIComponent(conclusion)}`, { method: "POST" }),
  perfDiff: (loopId: number, write = false) =>
    req<PerfDiff>(`/api/loops/${loopId}/perf-diff${write ? "?write=true" : ""}`),
  patchRun: (runId: number, loopId: number | null) =>
    req<Run>(`/api/runs/${runId}`, { method: "PATCH", body: JSON.stringify({ loop_id: loopId }) }),
  // ---- 自动优化：agent 自动改代码（M3 余项）----
  optimizeLoop: (loopId: number, body?: { extra?: string | null; reference_loop_id?: number | null }) =>
    req<{ job_id: number; loop_id: number; prompt: string }>(`/api/loops/${loopId}/optimize`, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
  /** legacy：直接从对话起一次 kind=优化 job（新流程走 confirm → start，两步可审计；本端点保留兼容）。 */
  startOptimizeChat: (loopId: number, body?: { instruction?: string | null; conversation_id?: number | null }) =>
    req<OptimizeChatStart>(`/api/loops/${loopId}/optimize/chat`, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
  /** 两步启动（第一步）：服务端冻结「确认的优化草案」并返回一次性 token（10 分钟有效）。 */
  confirmLoop: (
    loopId: number,
    body?: { instruction?: string | null; reference_loop_id?: number | null; note?: string | null },
  ) =>
    req<LoopConfirmResult>(`/api/loops/${loopId}/confirm`, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
  /** 两步启动（第二步）：消费 confirm 的 token 起 agent job（token 过期/已用/目标或基线变了 → 409）。 */
  startLoop: (loopId: number, body: { token: string; conversation_id?: number | null }) =>
    req<OptimizeChatStart>(`/api/loops/${loopId}/start`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  /** 该轮次的自动优化尝试历史（confirm→start 的可审计记录）。 */
  loopAttempts: (loopId: number) => req<LoopAttempts>(`/api/loops/${loopId}/attempts`),
  /** Step 2 自动验证链：一键串跑 build → smoke → test → benchmark → report（后端异步驱动）。 */
  startAutoRun: (loopId: number) =>
    req<{ auto_run: LoopAutoRun }>(`/api/loops/${loopId}/auto-run`, { method: "POST" }),
  /** 该轮次最近一次自动验证链（没有则 auto_run=null）。 */
  loopAutoRun: (loopId: number) => req<{ auto_run: LoopAutoRun | null }>(`/api/loops/${loopId}/auto-run`),
  /** 取消运行中的自动验证链：后端杀当前动作作业 + 远端清理；工作区保留现状（不回退）。 */
  cancelAutoRun: (loopId: number) =>
    req<{ auto_run: LoopAutoRun; cancelled: LoopAutoRunCancel }>(`/api/loops/${loopId}/auto-run/cancel`, {
      method: "POST",
    }),
  /** 自然语言方向 → 记录思路 → 建 Loop → 基线齐全时自动启动 Agent。 */
  startOptimizationConversation: (taskId: number, direction: string, conversationId?: number | null) =>
    req<OptimizationConversationResult>(`/api/tasks/${taskId}/optimization-chat`, {
      method: "POST",
      body: JSON.stringify({ direction, conversation_id: conversationId ?? null }),
    }),
  /** 该轮次最近一次自动优化对应的会话（没有则 conversation_id=null）。 */
  loopConversation: (loopId: number) => req<LoopConversation>(`/api/loops/${loopId}/conversation`),
  optimizeEvidence: (loopId: number) => req<OptimizeEvidence>(`/api/loops/${loopId}/optimize`),
  revertOptimize: (loopId: number) =>
    req<OptimizeEvidence>(`/api/loops/${loopId}/optimize/revert`, { method: "POST" }),
  // ---- 版本存档（savepoints v1）----
  listSavepoints: (taskId: number, limit = 50, loopId?: number | null) =>
    req<Savepoint[]>(
      `/api/tasks/${taskId}/savepoints?limit=${limit}${loopId != null ? `&loop_id=${loopId}` : ""}`,
    ),
  createSavepoint: (taskId: number, body?: { label?: string; note?: string }) =>
    req<Savepoint>(`/api/tasks/${taskId}/savepoints`, { method: "POST", body: JSON.stringify(body ?? {}) }),
  savepointDiff: (id: number) => req<SavepointDiff>(`/api/savepoints/${id}/diff`),
  restoreSavepoint: (id: number, body?: { scope?: "task" | "full"; note?: string }) =>
    req<SavepointRestoreResult>(`/api/savepoints/${id}/restore`, { method: "POST", body: JSON.stringify(body ?? {}) }),
  deleteSavepoint: (id: number) => req<void>(`/api/savepoints/${id}`, { method: "DELETE" }),
  /** 归档下载（tar.gz）走浏览器直连：给 <a href> 用，不经过 fetch。 */
  savepointArchiveUrl: (id: number) => `/api/savepoints/${id}/archive`,
  // ---- 优化轮次（P1，docs/plan-optimize-rounds-v1.md）----
  optimizationTargets: (taskId: number) =>
    req<OptimizationTargets>(`/api/tasks/${taskId}/optimization-targets`),
  createOptimizationRound: (taskId: number, body: OptimizationRoundCreate) =>
    req<OptimizationRoundCreated>(`/api/tasks/${taskId}/optimization-rounds`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  acceptLoop: (loopId: number, body?: { label?: string | null; note?: string | null }) =>
    req<OptimizationAcceptResult>(`/api/loops/${loopId}/accept`, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
  /** 否决本轮：`reason` 写进结论；`revert=true` 时后端尽力回退工作区并返回 `reverted`。 */
  rejectLoop: (loopId: number, body: { reason?: string; revert?: boolean }) =>
    req<OptimizationRejectResult>(`/api/loops/${loopId}/reject`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  optimizationReport: (loopId: number, write = false) =>
    req<OptimizationReport>(`/api/loops/${loopId}/optimization-report${write ? "?write=true" : ""}`),
  // ---- 报告归档（append-only；Step 3）----
  /** 该轮次的归档报告列表（最新在前；不改归档内容，重跑只追加）。 */
  optimizationReports: (loopId: number) =>
    req<OptimizationReports>(`/api/loops/${loopId}/optimization-reports`),
  /** 生成并归档报告：独立目录写 report.json + manifest.json + raw/ 副本；409/400 的 detail 原样展示。 */
  archiveOptimizationReport: (loopId: number) =>
    req<OptimizationReportStored>(`/api/loops/${loopId}/optimization-reports`, { method: "POST" }),
  /** 校验某个归档：逐文件 sha256 + raw 副本是否齐全/一致。 */
  verifyOptimizationReport: (loopId: number, reportSha256: string) =>
    req<OptimizationReportVerify>(
      `/api/loops/${loopId}/optimization-reports/${encodeURIComponent(reportSha256)}/verify`,
      { method: "POST" },
    ),
  /** 读某个归档的 manifest + report（404 = 不存在）。 */
  getOptimizationReport: (loopId: number, reportSha256: string) =>
    req<OptimizationReportDetail>(
      `/api/loops/${loopId}/optimization-reports/${encodeURIComponent(reportSha256)}`,
    ),
  /** 归档原始 `report.json` 的字节下载地址（同源；sha256 与 report_sha256 完全一致，直接走这个 URL，不要前端重新序列化）。 */
  archivedReportDownloadUrl: (loopId: number, reportSha256: string) =>
    `/api/loops/${loopId}/optimization-reports/${encodeURIComponent(reportSha256)}/download`,
  // ---- P2 优化点搜索 / 一思路一 Loop ----
  optimizationIdeas: (taskId: number) => req<OptimizationIdea[]>(`/api/tasks/${taskId}/optimization-ideas`),
  addOptimizationIdea: (taskId: number, body: { title: string; hypothesis?: string | null; change_scope?: string | null; expected?: string | null }) =>
    req<{ idea: OptimizationIdea; deduplicated: boolean }>(`/api/tasks/${taskId}/optimization-ideas`, {
      method: "POST", body: JSON.stringify(body),
    }),
  startIdeaSearch: (taskId: number, extra?: string | null) =>
    req<IdeaSearch>(`/api/tasks/${taskId}/idea-searches`, {
      method: "POST", body: JSON.stringify({ extra: extra ?? null }),
    }),
  ideaSearches: (taskId: number) => req<IdeaSearch[]>(`/api/tasks/${taskId}/idea-searches`),
  ideaSearch: (searchId: number) => req<IdeaSearch>(`/api/idea-searches/${searchId}`),
  /** 由思路建独立轮次；body 可带 text objective（focus/约束/预算）与复用基线 run 开关。 */
  createRoundFromIdea: (ideaId: number, body?: OptimizationIdeaRoundCreate) =>
    req<OptimizationRoundCreated & { idea: OptimizationIdea }>(`/api/optimization-ideas/${ideaId}/round`, {
      method: "POST", body: JSON.stringify(body ?? {}),
    }),
  patchOptimizationIdea: (ideaId: number, status: string, outcome?: Record<string, unknown>) =>
    req<OptimizationIdea>(`/api/optimization-ideas/${ideaId}`, {
      method: "PATCH", body: JSON.stringify({ status, outcome: outcome ?? null }),
    }),
  loopBudgetAudits: (loopId: number) => req<LoopBudgetAudit[]>(`/api/loops/${loopId}/budget-audits`),
  // ---- 结果视图（P2）----
  runReport: (runId: number, type: "correctness" | "performance" | "metrics") =>
    req<RunReport>(`/api/runs/${runId}/report?type=${type}`),
  // ---- 硬件画像（极限/Roofline，方案 §13-A11）----
  chips: () => req<ChipSummary[]>("/api/chips"),
  chip: (name: string) => req<ChipProfile>(`/api/chips/${name}`),
  // ---- 消耗情况（P4）----
  usage: () => req<UsageReport>("/api/usage"),
  // ---- 测试管理（P3 读侧）----
  cases: (taskId: number) => req<CaseReport>(`/api/tasks/${taskId}/cases`),
  // ---- 阶段 / 门禁（单任务）----
  taskStage: (taskId: number) => req<StageTaskView>(`/api/tasks/${taskId}/stage`),
  taskGates: (taskId: number) => req<TaskGates>(`/api/tasks/${taskId}/gates`),
  // ---- 规格版本（P0-3）----
  specs: (taskId: number) => req<SpecVersion[]>(`/api/tasks/${taskId}/specs`),
  createSpec: (taskId: number, body: SpecVersionWrite) =>
    req<SpecVersion>(`/api/tasks/${taskId}/specs`, { method: "POST", body: JSON.stringify(body) }),
  context: (taskId: number) => req<ContextSpace>(`/api/tasks/${taskId}/context`),
  // ---- 工作区只读浏览（实现与接口）----
  workspace: (taskId: number) => req<WorkspaceInfo>(`/api/tasks/${taskId}/workspace`),
  workspaceFile: (taskId: number, path: string) =>
    req<WorkspaceFileText>(`/api/tasks/${taskId}/workspace/file?path=${encodeURIComponent(path)}`),
  workspaceDiff: (taskId: number) => req<WorkspaceDiff>(`/api/tasks/${taskId}/workspace/diff`),
  artifactText: (artifactId: number) =>
    req<{ id: number; path: string; binary: boolean; size_bytes: number; truncated?: boolean; text: string | null }>(
      `/api/artifacts/${artifactId}/text`,
    ),
  // ---- 阶段看板（P3 读侧 + 按证据同步）----
  stages: (includeDrafts = false) => req<StageBoard>(`/api/stages${includeDrafts ? "?include_drafts=true" : ""}`),
  syncStage: (taskId: number) =>
    req<{ changed: boolean; stage?: string; from?: string; to?: string; view?: StageTaskView }>(
      `/api/tasks/${taskId}/stage/sync`, { method: "POST" }),
  // ---- API 管理（P4 配置态）----
  endpoints: () => req<EndpointConf[]>("/api/endpoints"),
  createEndpoint: (body: EndpointWriteBody) =>
    req<EndpointConf>("/api/endpoints", { method: "POST", body: JSON.stringify(body) }),
  patchEndpoint: (id: number, body: EndpointWriteBody) =>
    req<EndpointConf>(`/api/endpoints/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  activateEndpoint: (id: number) =>
    req<EndpointConf>(`/api/endpoints/${id}/activate`, { method: "POST" }),
  deleteEndpoint: (id: number) => req<void>(`/api/endpoints/${id}`, { method: "DELETE" }),
  probeEndpoint: (id: number) =>
    req<EndpointProbeResult>(`/api/endpoints/${id}/probe`, { method: "POST" }),
  probeAllEndpoints: () =>
    req<{ results: EndpointProbeResult[] }>("/api/endpoints/probe", { method: "POST" }),
  // ---- 对话页（P0-2）----
  conversations: (taskId?: number) =>
    req<Conversation[]>(`/api/conversations${taskId ? `?task_id=${taskId}` : ""}`),
  createConversation: (body: { task_id: number; title?: string | null; mode?: string }) =>
    req<Conversation>("/api/conversations", { method: "POST", body: JSON.stringify(body) }),
  conversation: (id: number) => req<Conversation>(`/api/conversations/${id}`),
  sendMessage: (id: number, text: string) =>
    req<{ conversation_id: number; job_id: number; dsh_session_id: string | null }>(
      `/api/conversations/${id}/messages`,
      { method: "POST", body: JSON.stringify({ text }) },
    ),
  deleteConversation: (id: number) =>
    req<void>(`/api/conversations/${id}`, { method: "DELETE" }),
  revertMessage: (convId: number, messageId: number) =>
    req<{ reverted: boolean; files: Record<string, string> }>(
      `/api/conversations/${convId}/messages/${messageId}/revert`, { method: "POST" }),
  uploadAttachment: async (convId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`/api/conversations/${convId}/attachments`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 200)}`);
    return (await res.json()) as { id: number; filename: string; path: string; size_bytes: number; sha256: string };
  },
  /** 删附件（DB 行 + 工作区文件）；对话页附件 chip 上的 ✕ 用。 */
  deleteAttachment: (convId: number, attachmentId: number) =>
    req<void>(`/api/conversations/${convId}/attachments/${attachmentId}`, { method: "DELETE" }),
  shell: (taskId: number, command: string, timeoutS = 30) =>
    req<ShellResult>(`/api/tasks/${taskId}/shell`, {
      method: "POST",
      body: JSON.stringify({ command, timeout_s: timeoutS }),
    }),
  // ---- 案例库（M2）----
  caseLibrary: () => req<CaseLibrary>("/api/case-library"),
  startCase: (bundle: string, op: string, projectId?: number | null) =>
    req<{ task_id: number; title: string; op_name: string | null }>("/api/case-library/start", {
      method: "POST",
      body: JSON.stringify({ bundle, op, project_id: projectId ?? null }),
    }),
  // ---- 任务生命周期 v2（docs/impl-contract-v2.md §3）----
  /** 设计文档模板下载（浏览器直连：给 <a href> 用；format=docx|md）。 */
  templateUrl: (format: "docx" | "md") => `/api/templates/operator-design-doc?format=${format}`,
  // 任务级设计文档（multipart；草稿任务 is_draft=1 也可上传，同名覆盖）
  documents: (taskId: number) => req<TaskDocument[]>(`/api/tasks/${taskId}/documents`),
  uploadDocument: async (taskId: number, file: File): Promise<TaskDocument> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`/api/tasks/${taskId}/documents`, { method: "POST", body: form });
    // 上传失败要把后端 detail（格式/大小/解析失败原因）带给用户，不能只显示状态码。
    if (!res.ok) {
      const text = await res.text();
      let detail = text;
      try {
        const body = JSON.parse(text) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        /* 原文 */
      }
      throw new Error(`${res.status} ${detail.slice(0, 300)}`);
    }
    return (await res.json()) as TaskDocument;
  },
  deleteDocument: (taskId: number, docId: number) =>
    req<void>(`/api/tasks/${taskId}/documents/${docId}`, { method: "DELETE" }),
  /** 原文下载（浏览器直连）。 */
  downloadDocumentUrl: (taskId: number, docId: number) => `/api/tasks/${taskId}/documents/${docId}/download`,
  // 用例组（精度 + 性能）整组确认：由对话里的 agent 调，页面只读展示状态。
  confirmCases: (taskId: number, body: { cases_sha256: string; benchmark_sha256: string; by?: string | null }) =>
    req<CasesConfirmResult>(`/api/tasks/${taskId}/cases/confirm`, { method: "POST", body: JSON.stringify(body) }),
  casesConfirmation: (taskId: number) => req<CasesConfirmation>(`/api/tasks/${taskId}/cases/confirmation`),
  // 任务版本（大版本 = 规格；小版本 = 优化采纳 / 手工变更）
  versions: (taskId: number) => req<TaskVersions>(`/api/tasks/${taskId}/versions`),
  /** 手工小版本（交付前非优化变更）：后端返回整个版本 payload（current/history/evidence）。 */
  createVersion: (taskId: number, body: TaskVersionWrite = {}) =>
    req<TaskVersions>(`/api/tasks/${taskId}/versions`, { method: "POST", body: JSON.stringify(body) }),
  // 交付（本地动作：打包 + 下载）
  deliveries: (taskId: number) => req<DeliveryList>(`/api/tasks/${taskId}/deliveries`),
  /**
   * 生成交付包。409 是业务门禁不满足（用例未确认 / 证据失效 / 产物缺失），
   * 后端返回 `{detail, codes}`：这里把 detail/codes 挂在错误上，页面直接展示缺项。
   */
  createDelivery: async (taskId: number): Promise<DeliveryRecord> => {
    const res = await fetch(`/api/tasks/${taskId}/deliveries`, { method: "POST" });
    if (!res.ok) {
      const text = await res.text();
      let detail = text.slice(0, 400);
      let codes: string[] = [];
      let missing: DeliveryCreateError["missing"] = [];
      try {
        const body = JSON.parse(text) as { detail?: unknown; codes?: unknown };
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (body.detail && typeof body.detail === "object") {
          // FastAPI `HTTPException(409, {"detail": "...", "codes": [...]})` → body.detail 是对象。
          const nested = body.detail as { detail?: unknown; codes?: unknown; missing?: unknown };
          if (typeof nested.detail === "string") detail = nested.detail;
          else detail = JSON.stringify(nested.detail ?? nested);
          if (Array.isArray(nested.codes)) codes = nested.codes.map(String);
          if (Array.isArray(nested.missing)) {
            missing = (nested.missing as Record<string, unknown>[]).map((m) => ({
              code: typeof m.code === "string" ? m.code : undefined,
              item: typeof m.item === "string" ? m.item : undefined,
              path: typeof m.path === "string" ? m.path : undefined,
              reason: typeof m.reason === "string" ? m.reason : undefined,
            }));
          }
        } else if (body.detail != null) {
          detail = JSON.stringify(body.detail);
        }
        if (!codes.length && Array.isArray(body.codes)) codes = body.codes.map(String);
      } catch {
        /* 不是 JSON 就用原文 */
      }
      const err = new Error(`${res.status} ${detail}`) as DeliveryCreateError;
      err.status = res.status;
      err.detail = detail;
      err.codes = codes;
      err.missing = missing;
      throw err;
    }
    return (await res.json()) as DeliveryRecord;
  },
  downloadDeliveryUrl: (runId: number) => `/api/deliveries/${runId}/download`,
  // ---- 交付：产物下载 ----
  artifactUrl: (artifactId: number) => `/api/artifacts/${artifactId}/download`,
};

export function gb(bytes: number | null | undefined): string {
  if (!bytes) return "—";
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/** 按量级选单位：小文件不再显示成「0.0 GB」。 */
export function size(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

export function num(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

/** 金额展示（人民币元）：小额不显示成 ¥0.00；非有限/空值如实标「未定价」。 */
export function money(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "未定价";
  if (n === 0) return "¥0";
  if (Math.abs(n) < 0.01) return `¥${n.toFixed(4)}`;
  return `¥${n.toFixed(2)}`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleString("zh-CN", { hour12: false });
}

export const STAGES = ["需求与设计", "开发", "验收", "交付"];
