export type HostImage = {
  id: number;
  host_id: number;
  image_ref: string;
  tag: string | null;
  image_id?: string | null;
  digest?: string | null;
  size_bytes: number | null;
  created_raw: string | null;
  dangling: boolean;
  is_op_dev: boolean;
  is_test: boolean;
  env_recipe: string | null;
  preflight_cmd: string | null;
  cann_version?: string | null;
  python_version?: string | null;
  torch_version?: string | null;
  /** scanned | stale（探针失败时的旧值）| missing（远端已无） */
  scan_state?: string;
  last_scanned_at?: string | null;
  note: string | null;
};

export type Host = {
  id: number;
  name: string;
  kind: string;
  address: string | null;
  ssh_alias: string | null;
  cred_ref: string | null;
  cards: string | null;
  cann_version: string | null;
  docker_ok: boolean | null;
  probe_state: string;
  last_probe_at: string | null;
  note: string | null;
  is_test_server: boolean;
  image_count: number | null;
  op_dev_image_count: number | null;
  /** 最近一次探针的失败原因摘要（probe_state=fail 时有值） */
  probe_error?: string | null;
};

export type Artifact = {
  id: number;
  type: string;
  path: string;
  sha256: string | null;
  size_bytes: number | null;
  created_at: string;
};

/** 规格的一个版本（代码生成/测试生成/验收的共同依据） */
export type SpecVersion = {
  id: number;
  task_id: number;
  version: number;
  op_name: string | null;
  input_form: string | null;
  raw_input: string | null;
  spec_json: string | null;
  assumptions: string | null;
  unknowns: string | null;
  change_reason: string | null;
  sha256: string | null;
  created_at: string;
};

/** 一轮设计 / 修复 / 优化 */
export type Loop = {
  id: number;
  task_id: number;
  kind: string;
  scope: string | null;
  hypothesis: string | null;
  judge_condition: string | null;
  conclusion: string | null;
  status: string;
  /** 优化轮次目标（objective_json 解析；只有「优化轮次」建的 Loop 才有） */
  objective?: OptimizationObjective | null;
  optimize?: OptimizeEvidence | null; // 「agent 自动改代码」的证据（M3 余项）
  /** Step 1 状态机（services/loop_state.py）；历史轮次可能为 idle/空 */
  phase?: string | null;
  state_version?: number;
  active_job_id?: number | null;
  phase_error?: string | null;
  phase_updated_at?: string | null;
  created_at: string;
  ended_at: string | null;
};

/** agent 自动改代码的证据（`GET /api/loops/{id}/optimize`） */
export type OptimizeEvidence = {
  schema: string;
  loop_id: number;
  job_id?: number | null;
  status: "none" | "done" | "failed";
  error?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  session_id?: string | null;
  before_sha256?: string | null;
  after_sha256?: string | null;
  current_sha256?: string | null;
  changed: boolean;
  changed_files: { path: string; status: string; additions?: number; deletions?: number }[];
  diff?: string | null;
  agent_summary?: string | null;
  agent_output?: Record<string, unknown> | null;
  reference_loop_id?: number | null;
  /** agent 自报的 verifier 结论（pass/fail/空）；非 pass 会被后端自动回退 */
  verification?: string | null;
  /** CANNBot 三段工作流摘要（有则展示 architect/developer/verifier 行） */
  pipeline?: { architect?: string; developer?: string; verifier?: string } | null;
  /** 因 verifier 未 PASS 或改动越过 allowed_paths 被后端自动回退（reverted 的其中一种） */
  guard_reverted?: boolean;
  /** 自动回退前的「尝试快照」sha（回退后 after_sha256 会回到 before） */
  attempted_after_sha256?: string | null;
  reverted?: boolean;
  reverted_clean?: boolean;
  revert_failed?: string[];
};

/** 一次智能体调用（过程与消耗）—— 原 v1.0 的 run */
export type AgentJob = {
  id: number;
  task_id: number;
  loop_id: number | null;
  kind: string;
  prompt: string | null;
  status: string;
  is_demo?: boolean; // P0 示例行：usage/cost 是假的，消耗聚合要排除
  dsh_session_id: string | null;
  dsh_jsonl_path: string | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  cache_read: number | null;
  cache_write: number | null;
  cost: number | null; // 历史/示例行金额；真实调用按端点价目实时算
  cost_estimated?: boolean; // true = 按当前端点价目估算（早期调用没端点快照）
  endpoint_id?: number | null; // 这次调用用的端点（计价归属）
  model?: string | null;
};

/** 一次真实执行：build / test / benchmark / profile（结果与证据） */
export type Run = {
  id: number;
  task_id: number;
  loop_id: number | null;
  agent_job_id: number | null;
  action: string;
  status: string;
  spec_version_id: number | null;
  code_snapshot_sha256: string | null;
  suite_ref: string | null;
  criteria_ref: string | null;
  host_id: number | null;
  image_id: number | null;
  image_digest: string | null;
  device_id: string | null;
  command: string | null;
  cwd: string | null;
  remote_workdir: string | null;
  manifest_path: string | null;
  sync_state: string | null;
  failure_kind: string | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  note: string | null;
  artifacts: Artifact[];
};

/** 存档匹配到的证据 run（`code_snapshot_sha256` 相等，后端实时匹配，id 升序） */
export type SavepointRun = {
  id: number;
  action: string;
  status: string;
  device_id: string | null;
  loop_id: number | null;
  ended_at: string | null;
};

/** 版本存档（savepoints v1）：按内容哈希寻址的源码归档 + 指向它的存点行 */
export type Savepoint = {
  id: number;
  task_id: number;
  label: string;
  note: string | null;
  /** manual | auto | rescue | baseline | candidate | accept | rejected */
  kind: string;
  code_snapshot_sha256: string;
  archive_bytes: number;
  /** 归属的优化轮次（Loops）；非轮次存点为 null */
  loop_id: number | null;
  spec_version_id: number | null;
  /** 人看的规格版本号 vN（不是 DB id） */
  spec_version: number | null;
  criteria_ref: string | null;
  suite_ref: string | null;
  image_digest: string | null;
  runs: SavepointRun[];
  /** 全量 snapshot(workspace) 是否等于存档 sha；工作区不存在 → false */
  in_sync: boolean;
  /** rescue 行记录「为回滚哪个存档而做」 */
  parent_id: number | null;
  created_at: string;
};

/** 存档 vs 当前工作区的一个变更文件（scope: task | shared） */
export type SavepointChangedFile = {
  path: string;
  status: string;
  additions: number;
  deletions: number;
  scope: string;
};

/** `GET /api/savepoints/{id}/diff`：存档代码对比当前工作区 */
export type SavepointDiff = {
  savepoint: Savepoint;
  saved_sha256: string;
  current_sha256: string | null;
  in_sync: boolean;
  changed: SavepointChangedFile[];
  /** 任务范围外（共享）的差异路径 */
  outside_scope: string[];
  diff: string;
  diff_truncated: boolean;
};

/** `POST /api/savepoints/{id}/restore`：回滚结果（in_sync 为 scope 口径） */
export type SavepointRestoreResult = {
  savepoint: Savepoint;
  rescue: Savepoint | null;
  restored: boolean;
  in_sync: boolean;
  current_sha256: string | null;
  changed: SavepointChangedFile[];
  skipped_outside_scope: string[];
  note: string | null;
};

// ---------------------------------------------------------------- 优化轮次（P1）
// 契约见 docs/plan-optimize-rounds-v1.md §1 / §2 / §3 / §6 / §7。

/** 目标用例（数据驱动：不同算子语义不同，UI 不假设 n/dtype 一定存在） */
export type OptimizationTargetCase = {
  case_id: string;
  n?: number | null;
  dtype?: string | null;
  primary?: boolean | null;
  required?: boolean | null;
  note?: string | null;
};

export type OptimizationTargetCaseStatus = { case_id: string; status: string };

/** `GET /api/tasks/{id}/optimization-targets`：从最近一次 ok 的 test / benchmark 报告解析 */
export type OptimizationTargets = {
  task_id: number;
  benchmark_run_id: number | null;
  test_run_id: number | null;
  suite_ref: string | null;
  primary_case: string | null;
  cases: OptimizationTargetCase[];
  failing_cases: OptimizationTargetCaseStatus[];
  test_cases: OptimizationTargetCaseStatus[];
  hint: string | null;
};

/** objective.targets_snapshot：建轮次时冻结的目标证据（run id / 用例清单） */
export type OptimizationTargetsSnapshot = {
  benchmark_run_id: number | null;
  test_run_id: number | null;
  primary_case: string | null;
  suite_ref: string | null;
  cases: OptimizationTargetCase[];
  failing_cases: OptimizationTargetCaseStatus[];
};

/** objective_json（schema `opx-optimization-objective/1`） */
export type OptimizationObjective = {
  schema?: string;
  label?: string | null;
  /** perf | precision | hardware */
  type?: string;
  /** cases | all | text（text = 文字优化点：用 focus 描述，不指定目标用例；P2 二次修订） */
  target_dimension?: "cases" | "all" | "text";
  target_cases?: string[];
  /** target_dimension=text 时的「需要优化的点」（注入 Agent 提示；必填，≤2000 字） */
  focus?: string | null;
  constraints?: {
    precision_non_regression?: boolean;
    perf_non_regression?: boolean;
    allowed_paths?: string[];
  };
  budget?: {
    diagnose_rounds?: number;
    verify_rounds?: number;
    runs_per_action?: number;
  };
  user_text?: string | null;
  targets_snapshot?: OptimizationTargetsSnapshot | null;
};

/** 建轮次请求体（§3） */
export type OptimizationRoundCreate = {
  objective: OptimizationObjective;
  label?: string | null;
  hypothesis?: string | null;
  attach_baseline_runs?: boolean;
};

/** 建轮次返回（§3） */
export type OptimizationRoundCreated = {
  loop: Loop;
  baseline_savepoint: Savepoint;
  attached_runs: { id: number; action: string }[];
  targets: OptimizationTargets;
};

/** 由优化思路建轮次请求体（P2 二次修订；objective 留空时后端生成性能优化默认值） */
export type OptimizationIdeaRoundCreate = {
  objective?: OptimizationObjective | null;
  attach_baseline_runs?: boolean;
};

/** `POST /api/loops/{id}/reject` 的回退结果（revert=true 时后端尽力回退，失败不影响否决） */
export type OptimizationRevertInfo = {
  ok: boolean;
  /** optimize_before（本轮改前快照）| baseline_savepoint（基线存档）| error */
  mode?: string | null;
  reason?: string | null;
  reverted_clean?: boolean;
  revert_failed?: string[];
  savepoint_id?: number | null;
};

/** `POST /api/loops/{id}/reject` 返回（P2 二次修订：body 支持 reason + revert） */
export type OptimizationRejectResult = {
  loop?: Loop;
  savepoint?: Savepoint | null;
  reason?: string | null;
  reverted?: OptimizationRevertInfo | null;
};

/** `POST /api/loops/{id}/accept` 返回（§6） */
export type OptimizationAcceptResult = {
  loop: Loop;
  version: Savepoint;
  version_number: number;
  /** 当前工作区 sha 是否等于候选 sha（不一致只提示，不改工作区） */
  workspace_matches: boolean;
};

export type OptimizationIdea = {
  id: number;
  task_id: number;
  loop_id: number | null;
  search_id: number | null;
  source: "agent" | "user" | "conversation";
  title: string;
  evidence: Record<string, unknown> | unknown[] | string | null;
  hypothesis: string | null;
  change_scope: string | null;
  expected: string | null;
  risk: string | null;
  confidence: string | null;
  effort: string | null;
  skill: string | null;
  status: string;
  dedupe_key: string;
  outcome: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

/** 优化页自然语言入口：方向落库后，按基线/Agent 可用性返回当前停点。 */
export type OptimizationConversationResult = Partial<OptimizationRoundCreated> & {
  status: "started" | "waiting_previous" | "waiting_baseline" | "waiting_agent" | "deduplicated";
  reason: string;
  conversation_id: number;
  message_id: number;
  job_id: number | null;
  idea: OptimizationIdea;
  loop: Loop | null;
};

export type IdeaSearch = {
  id: number;
  task_id: number;
  agent_job_id: number | null;
  status: "running" | "ok" | "failed";
  summary: string | null;
  result: { new?: number; duplicates?: number; updated?: number };
  created_at: string;
  ended_at: string | null;
};

export type LoopBudgetAudit = {
  id: number;
  task_id: number;
  loop_id: number;
  action: string;
  decision: "allowed" | "rejected";
  limit_count: number;
  used_count: number;
  note: string | null;
  created_at: string;
};

/** `GET /api/loops/{id}/optimization-report`（§7） */
export type OptimizationReport = {
  schema: string;
  task_id: number;
  loop_id: number;
  label: string | null;
  objective: OptimizationObjective | null;
  targets: {
    benchmark_run_id: number | null;
    test_run_id: number | null;
    primary_case: string | null;
    target_cases: string[];
    suite_ref: string | null;
  };
  code: {
    baseline_savepoint_id: number | null;
    candidate_savepoint_id: number | null;
    version_savepoint_id: number | null;
    baseline_sha256: string | null;
    candidate_sha256: string | null;
  };
  runs: {
    before: { test: number | null; benchmark: number | null };
    after: { test: number | null; benchmark: number | null };
  };
  perf_diff: PerfDiff | null;
  precision: {
    before: Record<string, string>;
    after: Record<string, string>;
    target_cases: {
      case_id: string;
      before: string | null;
      after: string | null;
      newly_passed: boolean;
      regressed: boolean;
    }[];
    newly_passed: number;
    regressions: string[];
    non_regression: boolean;
  } | null;
  objective_summary: {
    type: string;
    /** pass | fail | null（精度轮次按回退；性能/硬件取 perf_diff.verdict） */
    verdict: string | null;
    /** improved | no_significant_change | inconclusive | regressed | null */
    conclusion: string | null;
    headline: string;
    /** hardware 类型的芯片画像摘要 */
    chip?: Record<string, unknown> | null;
  };
  repro: {
    before: { test_command: string | null; benchmark_command: string | null };
    after: { test_command: string | null; benchmark_command: string | null };
    env: {
      spec_version_id: number | null;
      criteria_ref: string | null;
      image_digest: string | null;
      device_id: string | null;
    };
  };
  generated_at: string;
};

/** `GET /api/loops/{id}/optimization-reports`：append-only 报告归档列表项（最新在前）。 */
export type OptimizationReportEntry = {
  report_sha256: string;
  seq: number;
  dir: string;
  generated_at: string;
  verdict: string | null;
  conclusion: string | null;
  valid: boolean;
  size_bytes: number;
  raw_count: number;
};

export type OptimizationReports = { reports: OptimizationReportEntry[] };

/** 归档 raw 清单条目（manifest.raw[]）。 */
export type OptimizationReportRaw = {
  kind: string;
  /** 源侧路径：一般为 run 目录内相对路径；run 目录不存在时是绝对路径。 */
  path: string;
  sha256: string | null;
  bytes: number | null;
  copied: boolean;
  /** 归档目录内相对路径（`raw/...`，仅 copied=true 时有）。 */
  copy?: string | null;
  /** 源文件绝对路径（后端诊断用）。 */
  source?: string | null;
  /** copied=false 的原因（如超过单文件/总量上限）。 */
  why?: string | null;
};

/**
 * 归档 manifest：files = 归档目录内文件 sha256；raw = 原始产物副本清单。
 * 后端还会带 raw_copied_count/raw_total（复制成功数 / 总条目数），这里补可选字段。
 */
export type OptimizationReportManifest = {
  schema: string;
  files: { path: string; sha256: string; bytes: number }[];
  raw: OptimizationReportRaw[];
  raw_copied_count?: number;
  raw_total?: number;
  [k: string]: unknown;
};

/**
 * `POST /api/loops/{id}/optimization-reports`（201）：
 * 后端把原 `GET /optimization-report` 的 payload **平铺**返回，再追加归档元数据。
 */
export type OptimizationReportStored = OptimizationReport & {
  report_sha256: string;
  content_sha256: string;
  seq: number;
  report_dir: string;
  written_to: string;
  manifest: OptimizationReportManifest;
  /** manifest 自校验结果（report.json sha 一致 + 文件存在）。 */
  report_valid: boolean;
  /** true = 新写入；false = 同内容复用已有归档（幂等）。 */
  stored: boolean;
};

/** `POST /api/loops/{id}/optimization-reports/{report_sha256}/verify`：逐文件 / 逐 raw 副本完整性校验。 */
export type OptimizationReportVerify = {
  report_sha256: string;
  valid: boolean;
  checked_at: string;
  files: { path: string; ok: boolean; expected: string | null; actual: string | null }[];
  raw: {
    kind: string;
    path: string;
    ok: boolean;
    copied: boolean;
    expected: string | null;
    actual: string | null;
  }[];
  errors: string[];
};

/** `GET /api/loops/{id}/optimization-reports/{report_sha256}`：读归档（404 = 不存在）。 */
export type OptimizationReportDetail = {
  manifest: OptimizationReportManifest;
  report: OptimizationReport;
  valid: boolean;
};

export type ShellResult = {
  command: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
  cwd: string;
};

export type CaseLibraryCase = {
  op: string;
  scenario: string | null;
  target_chip?: string | null;
  path: string | null;
  exists: boolean;
  has_workspace: boolean;
  has_ops: boolean;
};

export type CaseLibrary = {
  dest: string;
  imported: boolean;
  error?: string;
  bundles: { name: string; version: string | null; cases: CaseLibraryCase[] }[];
};

export type WorkspaceFile = { path: string; size_bytes: number };
export type WorkspaceInfo = { root: string; files: WorkspaceFile[]; truncated: boolean; count: number };
export type WorkspaceFileText = {
  path: string;
  size_bytes: number;
  binary?: boolean;
  truncated?: boolean;
  text: string | null;
};
export type WorkspaceDiff = {
  is_repo: boolean;
  error?: string;
  repo_root?: string;
  path?: string;
  branch?: string;
  dirty?: boolean;
  status?: string[];
  diff?: string;
  truncated?: boolean;
};

export type ContextSpace = {
  exists: boolean;
  space_dir: string;
  files: { path: string; sha256?: string | null; size_bytes?: number | null }[];
};

export type SpecVersionWrite = {
  spec_json?: string | null;
  assumptions?: string | null;
  unknowns?: string | null;
  change_reason?: string | null;
  raw_input?: string | null;
  input_form?: string | null;
  op_name?: string | null;
  set_current?: boolean;
};

export type Task = {
  id: number;
  project_id: number;
  title: string;
  op_name: string | null;
  target_chip: string | null;
  input_form: string | null;
  stage: string;
  is_draft: boolean; // 「任务开始」页的草稿（未定版；不进阶段统计）
  is_archived: boolean; // 已归档：默认不进主列表/阶段看板，证据保留
  current_spec_version_id: number | null;
  criteria_ref: string | null; // 判据档（空 = 用项目默认）
  next_action: string | null;
  host_id: number | null; // 覆盖项目默认（空 = 用项目）
  image_id: number | null;
  device_id: string | null;
  timeout_s: number | null;
  fix_rounds_max: number | null;
  budget_per_action: number | null;
  created_at: string;
};

/** 优化 Loop 的对比报告（方案 §13-A5）：performance-diff.json */
export type PerfDiffSide = {
  code_snapshot_sha256: string;
  test_run_id: number | null;
  benchmark_run_id: number | null;
  run_ids: number[];
};

export type PerfDiffCase = {
  case_id: string;
  n: number | null;
  before_us: number;
  after_us: number;
  /** 缺有效测量时为 null（配 missing_data=true，verdict=inconclusive），UI 不要格式化/比较 */
  delta_pct: number | null;
  /** 该用例缺有效前后测量（before/after median 缺失或 <=0） */
  missing_data?: boolean;
  uncertainty_pct: number;
  verdict: string; // improved | no_significant_change | inconclusive | regressed
  before_baseline_us: number;
  after_baseline_us: number;
};

export type PerfFinding = {
  code: string;
  level: "fail" | "warn";
  message: string;
  evidence?: Record<string, unknown>;
};

export type PerfDiff = {
  schema: string;
  loop_id: number;
  task_id: number;
  verdict: "pass" | "fail";
  conclusion: string;
  hypothesis: string | null;
  judge_condition: string | null;
  sides: { before: PerfDiffSide; after: PerfDiffSide } | null;
  cases: PerfDiffCase[];
  comparability?: Record<string, { before: unknown; after: unknown; same: boolean }>;
  precision: {
    cases: { case_id: string; before: string | null; after: string | null; regressed: boolean }[];
    regressed_cases: string[];
    non_regression: boolean;
  } | null;
  summary: { cases: number; fail: number; warn: number; primary_verdict: string };
  findings: PerfFinding[];
  recommendations?: Recommendation[];
  notes?: string[];
  written_to?: string;
};

/** 机读报告读取结果（P2 结果视图） */
export type RunReport = {
  run_id: number;
  action: string;
  type: string;
  path: string;
  report: Record<string, unknown>;
};

export type CorrectnessCase = {
  case_id: string;
  group?: string;
  required?: boolean;
  shape?: number[];
  dtype?: string;
  expected?: string;
  status: string;
  note?: string | null;
  detail?: string;
  max_abs_err?: number;
  matched_ratio?: number;
  duration_ms?: number;
  tol?: { rtol?: number; atol?: number; required_matched_ratio?: number; profile?: string };
};

export type BenchmarkSideAgg = { median_us?: number; mean_us?: number; std_us?: number; min_us?: number; max_us?: number };
export type BenchmarkCase = {
  case_id: string;
  n: number;
  dtype: string;
  primary?: boolean;
  /** same_device_only 报告允许没有可比基线，此时 baseline 为 null。 */
  baseline?: { impl?: string; op_name?: string | null; device_us?: BenchmarkSideAgg } | null;
  candidate?: { impl?: string; op_name?: string | null; device_us?: BenchmarkSideAgg } | null;
  derived?: {
    speedup_baseline_over_candidate?: number | null;
    candidate_achieved_bw_gbps?: number;
    baseline_achieved_bw_gbps?: number;
    moved_bytes?: number;
    read_bytes?: number;
    write_bytes?: number;
    /** 新报告：flop / byte（按 benchmark.json 的 memory/compute 画像算） */
    arithmetic_intensity?: number | null;
    flops?: number | null;
    reads_per_elem?: number | null;
    writes_per_elem?: number | null;
    flops_per_elem?: number | null;
  } | null;
};

export type MetricRow = {
  id: string;
  kind?: string;
  status: string; // measured | estimated | unsupported | failed
  value?: number | null;
  unit?: string;
  source?: string;
  required?: boolean;
  note?: string;
  assumption?: string;
};

/** 结构化优化建议（方案 §4.6 R1） */
export type Recommendation = {
  id: string;
  title?: string;
  affected_cases: string[];
  observed_evidence: string[];
  bottleneck_hypothesis: string;
  confidence: string;
  proposed_change: string;
  expected_observation: string;
  constraints_and_risks: string;
  validation_method: string;
  status: string; // proposed | validated | rejected | inconclusive
  skill?: string;
};

/** 硬件画像摘要（列表用） */
export type ChipSummary = {
  name: string;
  schema?: string;
  arch?: string;
  npu_arch_flag?: string;
  product_family?: string;
  models: string[];
};

/** 硬件画像（极限/Roofline 对比用） */
export type ChipProfile = ChipSummary & {
  model_variable?: Record<string, { value?: number; source?: string; confidence?: string }>;
  compute?: Record<string, { value?: number }>;
};

export type TaskDetail = Task & {
  raw_input: string | null; // 需求原文（草稿期承载用户原话）
  dsh_session_id?: string | null; // 需求澄清用的 dsh 会话
  spec_draft: SpecDraft | null; // 规格草稿（业务证据）：{fields, questions, ready}
  chat_log: ChatMessage[]; // 对话记录（过程证据）
  agent_jobs: AgentJob[];
  runs: Run[];
  spec_versions: SpecVersion[];
  loops: Loop[];
  loop_run_links?: { loop_id: number; run_id: number; role: string }[];
  // —— 任务生命周期 v2（docs/impl-contract-v2.md §6）——
  /** 小版本号；大版本 = 当前 spec_versions.version。标题显示 `v{major}.{minor}`。 */
  version_minor?: number;
  /** 用例组（精度 + 性能）整组确认时间（UTC）；空 = 未确认（四动作 plan 会拦）。 */
  cases_confirmed_at?: string | null;
  /** 确认时用例集套件哈希（cases/ 目录哈希）。 */
  cases_confirmed_sha256?: string | null;
  /** 确认来源（`chat:job#123` / `api`）。 */
  cases_confirmed_by?: string | null;
};

// ---------------------------------------------------------------- 任务生命周期 v2
// 契约：docs/impl-contract-v2.md §2.2 / §3.4 / §3.5 / §3.6。

/** 任务版本来源：规格落版 / 采纳优化 / 手工（交付前非优化变更）。 */
export type TaskVersionSource = "spec" | "optimization" | "manual";

/** `task_versions` 一行：大版本 = 规格版本号，小版本 = 任务内递增。 */
export type TaskVersion = {
  id: number;
  task_id?: number;
  major: number;
  minor: number;
  /** 后端格式化的版本串（如 `v1.2`；缺省由前端按 major.minor 生成） */
  label?: string | null;
  /** 人写的名字（如「规格 v1」「采纳 Loop #3」；后端 `label_text`） */
  label_text?: string | null;
  source: string;
  spec_version_id: number | null;
  code_snapshot_sha256: string | null;
  suite_ref: string | null;
  /** 采纳优化时关联的轮次 */
  loop_id?: number | null;
  note?: string | null;
  /** 对话确认的证据豁免 `[{actions, reason, job_id, at}]`（原样 JSON 字符串） */
  waivers_json?: string | null;
  /** 交付冻结时间；空 = 未交付 */
  delivered_at: string | null;
  is_current?: boolean;
  created_at: string | null;
};

/** `GET /versions` → evidence.runs[action]：当前版本匹配到的最新证据 run。 */
export type VersionEvidenceRun = {
  run_id: number;
  status: string;
  gate_verdict: string | null;
  spec_version_id: number | null;
  code_snapshot_sha256: string | null;
  suite_ref: string | null;
};

/** 交付门禁缺项（`evidence.missing`）。 */
export type VersionMissingItem = {
  /** 动作名；用例确认缺项为 `cases`；交付打包缺项可能没有 action（只有 code/item）。 */
  action?: string;
  /** 人话动作名（后端 `label`） */
  label?: string;
  /** `未跑` / `门禁未过` / `版本快照不一致` 等 */
  reason: string;
  /** 交付打包缺项（`GET /deliveries/preview` / POST 409）的机读码 / 缺项名 / 路径 */
  code?: string;
  item?: string;
  path?: string;
};

/** evidence.checks 单项：某个必需动作的证据是否有效（含豁免）。 */
export type VersionEvidenceCheck = {
  action: string;
  ok: boolean;
  waived: boolean;
  run_id?: number | null;
  reason?: string | null;
};

/** `GET /api/tasks/{id}/versions` 的 evidence 段（交付门禁预览）。 */
export type VersionEvidence = {
  major?: number | null;
  minor?: number | null;
  version?: string;
  required_actions: string[];
  optional_actions?: string[];
  runs: Record<string, VersionEvidenceRun | null>;
  missing: VersionMissingItem[];
  deliverable: boolean;
  checks?: VersionEvidenceCheck[];
  waived_actions?: string[];
  expected_spec_version_id?: number | null;
  /** 当前工作区实时快照 */
  current_snapshot_sha256: string | null;
  /** 当前版本绑定的快照（工作区无版本行时回退实时快照） */
  expected_snapshot_sha256?: string | null;
  current_suite_sha256?: string | null;
  current_suite_ref: string | null;
  cases?: CasesConfirmation | null;
  delivered_at?: string | null;
  history_count?: number;
  current_version_id?: number | null;
};

export type TaskVersions = {
  current: TaskVersion | null;
  history: TaskVersion[];
  evidence: VersionEvidence;
};

export type TaskVersionWrite = {
  note?: string | null;
  label?: string | null;
};

/** `task_documents`：任务开始页上传的《算子设计文档》（原文 + 解析稿）。 */
export type TaskDocument = {
  id: number;
  task_id: number;
  filename: string;
  /** design / other */
  kind: string;
  /** 相对工作区：task-docs/task-<id>/<filename> */
  path: string;
  /** 相对工作区的解析稿（docx 抽段落/表格、md 原样）；空 = 未解析 */
  extracted_path: string | null;
  sha256: string;
  size_bytes: number | null;
  created_at: string;
};

/** `POST /cases/confirm` 返回：整组用例确认结果。 */
export type CasesConfirmResult = {
  confirmed_at: string;
  cases_sha256: string;
  cases_suite_sha256: string;
  suite_ref: string;
};

/** `GET /cases/confirmation` 返回：确认状态与当前用例集是否一致。 */
export type CasesConfirmation = {
  confirmed_at: string | null;
  confirmed_sha256: string | null;
  current_suite_sha256: string | null;
  in_sync: boolean;
};

/** `POST /deliveries` 返回 / `GET /deliveries` 列表项。 */
export type DeliveryRecord = {
  run_id: number;
  /**
   * 版本展示串。注意：POST 返回 `v{major}.{minor}`；GET 列表里后端用 run.note
   * （如「交付包 <op>_v1.0_20261023.zip」），前端原样展示、不要盲目加 v 前缀。
   */
  version: string;
  sha256: string | null;
  size_bytes: number | null;
  /** POST 响应没有 created_at；GET 列表有 */
  created_at?: string | null;
  /** 后端直出下载路径（前端也提供 downloadDeliveryUrl 兜底） */
  download_url?: string | null;
  // GET /deliveries 列表项附加字段
  action?: string;
  status?: string;
  gate_verdict?: string | null;
  spec_version_id?: number | null;
  code_snapshot_sha256?: string | null;
  suite_ref?: string | null;
  zip_path?: string;
};

/** `GET /api/tasks/{id}/deliveries` 返回结构（不是裸数组）。 */
export type DeliveryList = { task_id: number; deliveries: DeliveryRecord[] };

export type TaskDeleteCounts = {
  spec_versions: number;
  loops: number;
  optimization_ideas: number;
  idea_searches: number;
  loop_budget_audits: number;
  agent_jobs: number;
  runs: number;
  artifacts: number;
  action_jobs: number;
  stage_events: number;
  conversations: number;
  messages: number;
  attachments: number;
  savepoints: number;
};

export type TaskDeleteFile = {
  kind: "context" | "chat" | "conversation" | "run";
  path: string;
  exists: boolean;
  bytes: number;
  removable: boolean;
  reason: string | null;
  run_id?: number | null;
  run_ref?: string | null;
  conversation_id?: number | null;
};

export type TaskDeletePreview = {
  task_id: number;
  title: string;
  op_name: string | null;
  is_draft: boolean;
  stage: string;
  counts: TaskDeleteCounts;
  running: { action_jobs: number[]; agent_jobs: number[] };
  files: { items: TaskDeleteFile[]; total_bytes: number };
};

export type TaskDeleteResult = {
  task_id: number;
  deleted: boolean;
  counts: TaskDeleteCounts;
  files: {
    requested: boolean;
    deleted: string[];
    skipped: { path: string; reason: string }[];
    errors: { path: string; error: string }[];
  };
};

/** 定版结果：spec v1 落库 + 上下文空间落盘 */
export type FinalizeResult = {
  spec_version: SpecVersion;
  space_dir: string;
  files: { path: string; sha256?: string | null; size_bytes?: number | null }[];
};

/** 对话式需求澄清的一轮结果 */
export type ChatResult = {
  reply: string;
  spec: Record<string, unknown>;
  questions: ChatQuestion[];
  ready: boolean;
  messages: ChatMessage[];
  usage?: Record<string, unknown> | null;
};

export type ChatStreamResult = { job_id: number; messages: ChatMessage[] };

export type ChatMessage = { role: "user" | "assistant"; text: string };
export type ChatQuestion = { id?: string; text: string; answer?: string };
export type SpecDraft = {
  fields: Record<string, unknown>;
  questions: ChatQuestion[];
  ready: boolean;
};

/** 项目 = 持续开发工作区 + 默认执行机/镜像 */
export type Project = {
  id: number;
  name: string;
  workspace_path: string | null;
  host_id: number | null;
  image_id: number | null;
  dsh_profile: string | null;
  dsh_home: string | null;
  criteria_ref: string | null; // 判据档默认值（档名@档版本）
};

/** 四动作之一的一次执行计划（dry_run 也能算） */
/** 对话页（P0-2）：会话 + 消息；引擎复用后端 agent_runner。 */
export type ChangeFile = { path: string; status: string; additions?: number; deletions?: number };
export type MessageChanges = {
  before_sha256?: string;
  after_sha256?: string;
  changed_files: ChangeFile[];
  diff: string;
  revert?: Record<string, { existed: boolean; restorable: boolean; content_b64?: string | null }>;
};

export type ConversationMessage = {
  id: number;
  role: string; // user | assistant
  text: string;
  job_id: number | null;
  created_at: string | null;
  changes?: MessageChanges | null;
};

export type ChatAttachment = { id: number; filename: string; path: string; size_bytes: number | null; sha256: string | null };

export type Conversation = {
  id: number;
  project_id: number | null;
  task_id: number | null;
  task_title: string | null;
  dsh_session_id: string | null;
  title: string | null;
  mode: string; // task | light
  message_count: number;
  tokens: number;
  cost: number | null;
  created_at: string | null;
  last_active_at: string | null;
  /** 该会话正在跑的对话作业 id（刷新/切页后据此接回 SSE；没有则 null） */
  running_job?: number | null;
  attachments?: ChatAttachment[];
  messages?: ConversationMessage[];
};

/** 「自动优化对话化」：对话页顶部横幅要展示的轮次上下文。 */
export type OptimizeChatSeed = {
  loopId: number;
  label: string;
  /** 文字优化点（text 维度才有） */
  focus?: string | null;
  allowedPaths?: string[];
  /** 每动作执行次数（后端硬限制） */
  budget?: number | null;
  /** 该轮次正在跑的优化 job（兜底接回 SSE；后端 conversation.running_job 现已覆盖优化 job） */
  jobId?: number | null;
};

/** 外部带入「对话」分区的内容：概览/AI 审查文本，或「性能与优化」的自动优化上下文。 */
export type ChatSeed = {
  text: string;
  nonce: number;
  /** 优先打开的会话（自动优化续聊用；不属于本任务时按普通逻辑回退） */
  conversationId?: number;
  /** 有值 = 对话页进入「自动优化」模式：回车即启动优化（允许空指令） */
  optimize?: OptimizeChatSeed;
  /** 对话先行的新优化：下一条用户消息作为优化方向，后端再落 idea/Loop 并安全起 Agent。 */
  newOptimization?: boolean;
};

/**
 * 启动一次对话化自动优化的返回：`POST /api/loops/{id}/start`（confirm→start 新流程）
 * 与 legacy `POST /api/loops/{id}/optimize/chat` 同形。
 */
export type OptimizeChatStart = {
  job_id: number;
  loop_id: number;
  conversation_id: number;
  message_id: number;
  prompt: string;
};

/** `GET /api/loops/{id}/conversation` 返回（该轮次最近一次自动优化对应的会话） */
export type LoopConversation = { conversation_id: number | null };

/** `POST /api/loops/{id}/confirm` 冻结的「确认的优化草案」（审计口径：目标/基线/工作区快照 + 指令）。 */
export type LoopConfirmDraft = {
  loop_id: number;
  objective_sha256: string;
  baseline_sha256: string;
  workspace_sha256: string;
  instruction: string | null;
  reference_loop_id: number | null;
  /** 用户确认时填写的备注（draft 里可能带 schema/note；后端多回字段不影响前端） */
  note?: string | null;
  schema?: string;
  state_version: number;
  created_at: string;
  expires_at: string;
};

/** `POST /api/loops/{id}/confirm` 返回（一次性 token 10 分钟内有效，`start` 时消费）。 */
export type LoopConfirmResult = {
  confirmation_id: number;
  loop_id: number;
  token: string;
  expires_at: string;
  draft: LoopConfirmDraft;
};

/** 一次自动优化尝试（`GET /api/loops/{id}/attempts`，可审计的 confirm→start 历史）。 */
export type LoopAttempt = {
  id: number;
  seq: number;
  agent_job_id: number | null;
  status: "running" | "succeeded" | "failed" | "blocked" | "cancelled";
  baseline_sha256: string | null;
  candidate_sha256: string | null;
  error: string | null;
  started_at: string | null;
  ended_at: string | null;
};

/** `GET /api/loops/{id}/attempts` 返回（后端按 seq 降序，最新在前；前端取最近 N 条）。 */
export type LoopAttempts = { loop_id: number; attempts: LoopAttempt[] };

/** 自动验证链单步状态（Step 2 orchestrator；pending|running|ok|failed|skipped）。 */
export type LoopAutoRunStepStatus = "pending" | "running" | "ok" | "failed" | "skipped";

/**
 * 自动验证链的一步：build/smoke/test/benchmark 是动作作业（job_id/run_id/gate/cleanup），
 * report 是报告落盘（无 job_id，gate/error 记落盘结果）。
 */
export type LoopAutoRunStep = {
  step: "build" | "smoke" | "test" | "benchmark" | "report";
  status: LoopAutoRunStepStatus;
  job_id: number | null;
  run_id: number | null;
  gate: string | null; // "pass" | "fail" | "warn" | null
  error: string | null;
  started_at: string | null;
  ended_at: string | null;
  cleanup: { ok?: boolean; why?: string; container?: string; [k: string]: unknown } | null;
  /** 仅 report 步：性能/精度结论 `improved|no_significant_change|inconclusive|regressed`（regressed 时 status=ok 但 error 有提示）。 */
  conclusion?: string | null;
};

/**
 * 一条自动验证链（`GET/POST /api/loops/{id}/auto-run`，后端 loop_auto_runs 表；GET 返回最近一次）。
 * 五步固定 build → smoke → test → benchmark → report；steps 缺失的步由前端按 pending 兜底。
 */
export type LoopAutoRun = {
  id: number;
  loop_id: number;
  status: "running" | "done" | "failed" | "blocked" | "cancelled";
  current_step: string | null;
  current_action_job_id: number | null;
  attempt_id: number | null;
  baseline_sha256: string | null;
  candidate_sha256: string | null;
  steps: LoopAutoRunStep[];
  /** 四动作预算快照：`{build: {limit, used}, ...}` */
  budget: Record<string, { limit: number; used: number }>;
  /** report 步的结论便捷字段（后端 view 从 `steps[report].conclusion` 提出来；没有则 null）。 */
  conclusion?: string | null;
  error: string | null;
  started_at: string | null;
  ended_at: string | null;
  updated_at: string | null;
};

/** `POST /api/loops/{id}/auto-run/cancel` 返回的取消明细（已终止的作业 / 本地 / 远端清理）。 */
export type LoopAutoRunCancel = {
  action_job_id?: number | null;
  local?: string;
  remote?: { ok?: boolean; why?: string; container?: string; [k: string]: unknown } | null;
};

/** 一次「点一下就真执行」的作业（后端 services/action_runner.py）。 */
export type ActionJob = {
  id: number;
  task_id: number;
  loop_id: number | null;
  action: string;
  status: string; // running | ok | failed | interrupted
  running: boolean;
  run_ref: string;
  run_id: number | null;
  device_id: string | null;
  command: string | null;
  log: string;
  exit_code: number | null;
  error: string | null;
  gate_verdict: string | null;
  started_at: string | null;
  ended_at: string | null;
};

export type ExecutionPlan = {
  task_id: number;
  action: string;
  workspace: string;
  command: string;
  cwd: string;
  remote_dir: string;
  code_snapshot_sha256: string;
  snapshot_files: number;
  snapshot_truncated: boolean;
  spec_version_id: number | null;
  spec_version: number | null;
  criteria_ref: string | null;
  criteria_source: string | null;
  host_id: number | null;
  host_ref: string | null;
  image_id: number | null;
  image_ref: string | null;
  image_digest: string | null;
  env_recipe: string | null;
  device_id: string | null;
  lock_mode: string;
  timeout_s: number;
  artifact_types: string[];
  steps: string[];
  missing: string[];
  warnings: string[];
  ready: boolean;
};

/** dsh headless --json 的事件（实测 8 类 + 插件回流增量） */
export type AgentEvent = {
  type: string;
  text?: string;
  kind?: string; // type=delta 时的增量种类（text）；fallback=true 表示插件没生效、整段补发
  fallback?: boolean;
  phase?: string;
  turn?: number;
  step?: number;
  tool?: string;
  input?: { command?: string } & Record<string, unknown>;
  status?: string;
  result?: string;
  sessionId?: string;
  usage?: { inputTokens: number; outputTokens: number; cacheReadTokens: number; cacheWriteTokens: number; totalTokens: number };
  reason?: { kind: string; error?: { message?: string } };
  command?: string;
  cwd?: string;
  dsh_home?: string;
  workspace?: string;
  /** _meta 事件：这次 dsh 是否挂了逐 token 插件 */
  stream?: boolean;
};

/** 消耗聚合的一个桶（总计 / 按任务 / 按类型 / 按天） */
export type UsageBucket = {
  jobs: number;
  ok: number;
  failed: number;
  tokens_in: number;
  tokens_out: number;
  cache_read: number;
  cache_write: number;
  cost: number;
  cost_known: boolean; // false = 有未定价的调用（别把 cost 当完整金额）
  priced_jobs?: number; // 有价目可算的调用数
  unpriced_jobs?: number; // 没有可用价目的调用数
  cost_estimated?: number; // 其中按「当前端点价目」估算的（早期调用没端点快照）
  task_id?: number;
  title?: string;
  kind?: string;
  day?: string;
};

/** 消耗情况（P4）：`GET /api/usage`（排除 is_demo 示例行） */
export type UsageReport = {
  totals: UsageBucket;
  by_task: UsageBucket[];
  by_kind: UsageBucket[];
  by_day: UsageBucket[];
  excluded_demo_rows: number;
  pricing?: {
    currency: string; // CNY
    unit: string; // per_1m_tokens
    endpoint_id: number | null;
    endpoint: string | null;
    priced: boolean;
  };
  notes: string[];
};

/** 用例矩阵里的一条（含最近结果与逐 run 历史） */
export type CaseResult = {
  case_id: string;
  group?: string | null;
  required?: boolean;
  shape?: number[] | null;
  shape_b?: number[] | null;
  dtype?: string | null;
  seed?: number | null;
  expected?: string | null;
  note?: string | null;
  results: Record<string, string>; // run_id → PASS/FAIL/...
  latest_status: string | null;
  latest_run_id: number | null;
  max_abs_err?: number | null;
  matched_ratio?: number | null;
};

export type CaseRun = {
  run_id: number;
  status: string;
  loop_id: number | null;
  suite_ref: string | null;
  criteria_ref: string | null;
  spec_version_id: number | null;
  code_snapshot_sha256: string | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  report_available: boolean;
  summary?: Record<string, number> | null;
  case_source?: string | null;
  case_source_sha256?: string | null;
  downloads: { id: number; type: string; name: string }[];
};

export type CaseFailure = {
  case_id: string;
  run_id: number;
  status: string;
  expected?: string | null;
  max_abs_err?: number | null;
  matched_ratio?: number | null;
  note?: string | null;
  detail?: string | null;
};

/** 性能用例最近一次测量（来自 benchmark run 的 performance.json；无则 null）。 */
export type PerformanceCaseLatest = {
  run_id: number;
  candidate_median_us: number | null;
  baseline_median_us: number | null;
  /** baseline / candidate；无有效测量时 null */
  speedup: number | null;
  /** improved | no_significant_change | inconclusive | regressed 等（后端可能不给） */
  conclusion?: string | null;
  gate_verdict?: string | null;
  candidate_bw_gbps?: number | null;
  baseline_bw_gbps?: number | null;
  candidate_std_us?: number | null;
  baseline_std_us?: number | null;
};

/** benchmark.json 里的一条性能用例（只读展示；编辑走对话）。 */
export type PerformanceCase = {
  case_id: string;
  n: number | null;
  dtype: string | null;
  required?: boolean | null;
  /** 主用例（性能结论以此为准） */
  primary?: boolean | null;
  note?: string | null;
  latest: PerformanceCaseLatest | null;
};

/** 性能组的历史 benchmark run（契约 §3.6 的 performance.runs）。 */
export type PerformanceRun = {
  run_id: number;
  status: string;
  gate_verdict: string | null;
  suite_ref: string | null;
  started_at: string | null;
  /** 该次 benchmark 报告里的主用例（有则） */
  primary_case?: string | null;
};

/** `GET /api/tasks/{id}/cases` 追加的 performance 段（精度段保持原样，不破坏旧字段）。 */
export type CasePerformance = {
  /** 用例文件来源，如 `cases/<op>/benchmark.json`；空 = 没有性能用例 */
  source: string | null;
  primary_case: string | null;
  cases: PerformanceCase[];
  runs: PerformanceRun[];
  latest_run_id?: number | null;
};

/** 测试管理（P3）：`GET /api/tasks/{id}/cases` */
export type CaseReport = {
  task_id: number;
  op_name: string;
  case_source: string;
  case_source_sha256: string | null;
  criteria_ref: string | null;
  case_groups: string[];
  total_cases: number;
  required_cases: number;
  cases: CaseResult[];
  runs: CaseRun[];
  failures: CaseFailure[];
  notes: string[];
  /** 性能用例段（任务生命周期 v2，契约 §3.6）；老后端没有该字段时为空 */
  performance?: CasePerformance | null;
  /** 用例组确认状态（后端 /cases 也带回） */
  cases_confirmed_at?: string | null;
  cases_confirmed_sha256?: string | null;
};

/** 阶段看板（P3）：`GET /api/stages` */
export type StageGate = {
  stage: string;
  name: string;
  ok: boolean;
  actions: string[];
  hint: string;
  evidence: string[];
};

export type StageAction = {
  run_id: number;
  status: string;
  verdict: string;
  fail_codes: string[];
  warn_codes: string[];
} | null;

export type TaskGate = {
  action: string;
  run_id: number | null;
  status?: string;
  verdict?: string | null;
  fail_codes: string[];
  warn_codes: string[];
  code_snapshot_sha256?: string | null;
  device_id?: string | null;
  loop_id?: number | null;
};

export type TaskGates = {
  task_id: number;
  gates: TaskGate[];
  perf: { loop_id: number; verdict: string | null; conclusion: string | null; fail_codes: string[] } | null;
};

export type StageTaskView = {
  task_id: number;
  title: string;
  op_name: string | null;
  is_draft: boolean;
  stored_stage: string;
  derived_stage: string;
  in_sync: boolean;
  spec_version: number | null;
  next_step: string;
  next_action: string | null;
  stages: StageGate[];
  actions: Record<string, StageAction>;
  run_count: number;
  perf: { loop_id: number; verdict: string | null; conclusion: string | null; fail_codes: string[] } | null;
  last_event: { from: string | null; to: string; reason: string | null; actor: string | null; created_at: string | null } | null;
};

export type StageBoard = {
  stages: string[];
  columns: { stage: string; tasks: StageTaskView[] }[];
  stored_columns: Record<string, number[]>;
  total: number;
  out_of_sync: number[];
  notes: string[];
};

/** API 管理（P4）：模型端点（配置态；key 只存引用名） */
export type EndpointConf = {
  id: number;
  name: string;
  provider: string;
  base_url: string | null;
  model: string | null;
  key_ref: string | null;
  has_key: boolean;
  /** key 存在哪个本机后端：keychain | secret-service | file | legacy（旧库明文） */
  key_backend?: string | null;
  is_active: boolean;
  max_model_len: number | null;
  probe_state: string; // unknown | ok | fail
  last_probe_at: string | null;
  note: string | null;
  // 单价：人民币元 / 每百万 token；cache 两项留空 = 不单独计价（按 0 计）
  price_in: number | null;
  price_out: number | null;
  price_cache_read: number | null;
  price_cache_write: number | null;
  priced: boolean; // price_in + price_out 都配了
};

export type EndpointProbeResult = {
  endpoint_id: number;
  probe_state: string;
  last_probe_at: string | null;
  probe: {
    ok: boolean;
    url?: string;
    status_code?: number;
    latency_ms?: number;
    key_present?: boolean;
    models?: string[];
    model_count?: number;
    error?: string;
  };
};
