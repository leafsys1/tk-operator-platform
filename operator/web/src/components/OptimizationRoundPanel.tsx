import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, size, when } from "../lib/api";
import type {
  ActionJob,
  IdeaSearch,
  Loop,
  LoopAttempt,
  LoopAutoRun,
  LoopAutoRunStep,
  LoopAutoRunStepStatus,
  LoopBudgetAudit,
  OptimizeEvidence,
  OptimizationObjective,
  OptimizationIdea,
  OptimizationReportDetail,
  OptimizationReportEntry,
  OptimizationReportVerify,
  OptimizationRoundCreate,
  OptimizationRoundCreated,
  OptimizationTargets,
  OptimizationTargetsSnapshot,
  PerfDiff,
  Run,
  Savepoint,
  TaskDetail,
} from "../types";
import { downloadJson } from "./ResultsPanel";

type RoundType = "perf" | "precision" | "hardware";
type Dimension = "cases" | "all";
type VerifyTarget = "baseline" | "candidate";
type VerifyStage = "plan" | "test" | "benchmark";

const TYPE_OPTIONS: { key: RoundType; label: string; color: string }[] = [
  { key: "perf", label: "性能", color: "var(--run)" },
  { key: "precision", label: "精度", color: "var(--ok)" },
  { key: "hardware", label: "硬件适配", color: "var(--warn)" },
];
const TYPE_META = Object.fromEntries(TYPE_OPTIONS.map((t) => [t.key, t])) as Record<
  string,
  { label: string; color: string }
>;

const LOOP_STATE: Record<string, { cls: string; label: string }> = {
  open: { cls: "chip chip-run", label: "进行中" },
  accepted: { cls: "chip chip-ok", label: "已采纳" },
  rejected: { cls: "chip chip-bad", label: "已否决" },
  closed: { cls: "chip", label: "已关闭" },
};

/** 自动优化 Step 1 状态机的中文标签；与后端 services/loop_state.py 的 PHASES 对齐。 */
const PHASE_META: Record<string, { cls: string; label: string }> = {
  idle: { cls: "chip !py-0 !text-[10px]", label: "未开始" },
  baseline_pending: { cls: "chip !py-0 !text-[10px]", label: "待基线" },
  baseline_ready: { cls: "chip chip-ok !py-0 !text-[10px]", label: "基线就绪" },
  agent_running: { cls: "chip chip-run !py-0 !text-[10px]", label: "改码中" },
  awaiting_verification: { cls: "chip !py-0 !text-[10px]", label: "待验证" },
  smoke_running: { cls: "chip chip-run !py-0 !text-[10px]", label: "编译 / 冒烟中" },
  full_running: { cls: "chip chip-run !py-0 !text-[10px]", label: "全量验证中" },
  awaiting_adoption: { cls: "chip !py-0 !text-[10px]", label: "待采纳" },
  accepted: { cls: "chip chip-ok !py-0 !text-[10px]", label: "已采纳" },
  rejected: { cls: "chip chip-bad !py-0 !text-[10px]", label: "已否决" },
  failed: { cls: "chip chip-bad !py-0 !text-[10px]", label: "失败" },
  blocked: { cls: "chip chip-bad !py-0 !text-[10px]", label: "阻塞" },
  cancelled: { cls: "chip !py-0 !text-[10px]", label: "已取消" },
};

/** 自动优化 attempts 的 status → 紧凑 chip（`GET /api/loops/{id}/attempts`）。 */
const ATTEMPT_STATUS: Record<string, { cls: string; label: string }> = {
  running: { cls: "chip chip-run", label: "进行中" },
  succeeded: { cls: "chip chip-ok", label: "改码完成（待验证）" },
  failed: { cls: "chip chip-bad", label: "失败" },
  blocked: { cls: "chip chip-bad", label: "阻塞" },
  cancelled: { cls: "chip", label: "已取消" },
};

/** Step 2 自动验证链：固定五步（build/smoke/test/benchmark 是动作，report 是报告落盘）。 */
const AUTO_STEP_ORDER: LoopAutoRunStep["step"][] = ["build", "smoke", "test", "benchmark", "report"];
const AUTO_STEP_LABEL: Record<LoopAutoRunStep["step"], string> = {
  build: "build",
  smoke: "smoke",
  test: "test",
  benchmark: "benchmark",
  report: "报告",
};
/** 自动验证链单步状态 → 紧凑 chip。 */
const AUTO_STEP_STATUS: Record<LoopAutoRunStepStatus, { cls: string; label: string }> = {
  pending: { cls: "chip !py-0 !text-[10px]", label: "待跑" },
  running: { cls: "chip chip-run !py-0 !text-[10px]", label: "运行中" },
  ok: { cls: "chip chip-ok !py-0 !text-[10px]", label: "通过" },
  failed: { cls: "chip chip-bad !py-0 !text-[10px]", label: "失败" },
  skipped: { cls: "chip !py-0 !text-[10px]", label: "跳过" },
};
/** 自动验证链整体状态 → 紧凑 chip。 */
const AUTO_RUN_STATUS: Record<LoopAutoRun["status"], { cls: string; label: string }> = {
  running: { cls: "chip chip-run", label: "运行中" },
  done: { cls: "chip chip-ok", label: "完成" },
  failed: { cls: "chip chip-bad", label: "失败" },
  blocked: { cls: "chip chip-bad", label: "阻塞" },
  cancelled: { cls: "chip", label: "已取消" },
};
/** 自动链运行中：会抢工作区/设备的按钮统一禁用并给这条 tooltip。 */
const AUTO_RUN_BUSY_HINT = "自动验证链运行中（正占用工作区/设备）：等它结束或先取消，避免两边抢同一份代码/同一张卡";
/** 自动链状态读不出来（GET 失败）：不能当成「没有」，保守禁用抢工作区/设备的入口。 */
const AUTO_RUN_UNKNOWN_HINT =
  "无法确认是否有自动链在跑（状态读取失败）：为避免抢工作区/设备，相关按钮已暂时禁用；请点进度块的「重试读取」或刷新页面。";
/**
 * 自动链可（重）试的轮次 phase：后端 preflight 允许从「待验证 / 阻塞 / 失败」启动
 * （候选/基线/预算仍逐项校验）。动作失败后工作区已回退，重试会 409 `E_AUTO_NO_CANDIDATE`，detail 原样展示。
 */
const AUTO_RUN_RETRY_PHASES = new Set(["awaiting_verification", "blocked", "failed"]);

/** API 错误 → 后端 detail 文案（与 TaskDeleteDialog / SavepointPanel 同口径）。 */
function errorText(e: unknown): string {
  const raw = String(e);
  const i = raw.indexOf("{");
  if (i >= 0) {
    try {
      const body = JSON.parse(raw.slice(i)) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
      if (body.detail) return JSON.stringify(body.detail);
    } catch {
      // 保持原文
    }
  }
  return raw;
}

/** 预算类错误（后端返回 409）→ 补充一句人话建议；非预算错误返回 null。 */
function budgetErrorHint(text: string | null | undefined): string | null {
  if (!text) return null;
  return /预算已用完|预算超限|新建下一轮|每动作|budget/i.test(text)
    ? "本轮该动作的执行次数已达后端硬限制（失败/环境类尝试也会占用预算）。可新建下一轮优化，或先处理环境问题后再试。"
    : null;
}

/** agent 自报的 pipeline 字段是任意 JSON：渲染前转成安全文本，避免对象直接塞进 React 子节点。 */
function agentText(v: unknown): string {
  if (typeof v === "string") return v;
  if (v == null) return "";
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/** 日志尾部（作业日志可能很长，只展示末尾）。 */
function logTail(log: string, max = 4000): string {
  const s = log ?? "";
  return s.length > max ? s.slice(-max) : s;
}

/** 由思路字段拼出「需要优化的点」预填文本（用户可编辑；作为 text objective 的 focus）。 */
function defaultFocusForIdea(idea: OptimizationIdea): string {
  const title = (idea.title ?? "").trim();
  const hypothesis = (idea.hypothesis ?? "").trim();
  const expected = (idea.expected ?? "").trim();
  const scope = (idea.change_scope ?? "").trim();
  const lines: string[] = [];
  if (title) lines.push(title);
  // 用户手工补充的思路 hypothesis == title，避免重复一行
  if (hypothesis && hypothesis !== title) lines.push(`假设：${hypothesis}`);
  if (expected) lines.push(`预期：${expected}`);
  if (scope) lines.push(`改动范围：${scope}`);
  const text = lines.join("\n");
  return text.length > 2000 ? text.slice(0, 2000) : text;
}

function conversationIdForIdea(idea: OptimizationIdea): number | null {
  const evidence = idea.evidence;
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) return null;
  const value = evidence.conversation_id;
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

/** 否决时后端回退模式 → 人话（未知模式原样展示）。 */
const REVERT_MODE_LABEL: Record<string, string> = {
  optimize_before: "本轮自动优化改前快照",
  baseline_savepoint: "建轮次时的基线存档",
  error: "回退出错",
};

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** 设备 pin：task.device_id > 最近一次 ok run 的 device > 首个 plan 返回的 device。 */
function initialDevicePin(task: TaskDetail): string | null {
  if (task.device_id) return task.device_id;
  const last = [...task.runs]
    .filter((r) => r.status === "ok" && r.device_id)
    .sort((a, b) => b.id - a.id)[0];
  return last?.device_id ?? null;
}

type RunSide = { sha: string; runs: Run[]; byAction: Record<string, Run> };
type Sides = { before: RunSide | null; after: RunSide | null };

/**
 * 把 Loop 里的 ok run 按代码快照分两侧（与后端 perf_diff 同口径）：
 * 首次出现的 sha = 基线侧；最后一个 sha = 候选侧（只有 1 个 sha 时候选侧还没有）。
 */
function sidesOf(runs: Run[]): Sides {
  const usable = runs.filter((r) => r.status === "ok" && r.code_snapshot_sha256);
  const firstSeen = new Map<string, number>();
  for (const r of [...usable].sort((a, b) => a.id - b.id)) {
    const sha = r.code_snapshot_sha256 as string;
    if (!firstSeen.has(sha)) firstSeen.set(sha, r.id);
  }
  const order = [...firstSeen.entries()].sort((a, b) => a[1] - b[1]).map(([sha]) => sha);
  const sideFor = (sha: string): RunSide => {
    const rs = usable.filter((r) => r.code_snapshot_sha256 === sha);
    const byAction: Record<string, Run> = {};
    for (const r of [...rs].sort((a, b) => a.id - b.id)) byAction[r.action] = r;
    return { sha, runs: rs, byAction };
  };
  return {
    before: order.length >= 1 ? sideFor(order[0]) : null,
    after: order.length >= 2 ? sideFor(order[order.length - 1]) : null,
  };
}

const ACTIONS: ("test" | "benchmark")[] = ["test", "benchmark"];

function missingActions(side: RunSide | null): ("test" | "benchmark")[] {
  return side ? ACTIONS.filter((a) => !side.byAction[a]) : [...ACTIONS];
}

type VerifyState = {
  loopId: number;
  target: VerifyTarget;
  stage: VerifyStage;
  action: "test" | "benchmark" | null;
  jobId: number | null;
  deviceId: string | null;
  log: string;
  /** 本次验证开始时间（ms）；刷新恢复时用作业 started_at，用于展示已运行时长。 */
  startedAt: number;
};

type VerifyResult = {
  loopId: number;
  target: VerifyTarget;
  ok: boolean;
  text: string;
  log: string;
};

/** perf-diff 结论 → 中文与色 */
const CONCLUSION: Record<string, { label: string; color: string }> = {
  improved: { label: "有提升", color: "var(--ok)" },
  no_significant_change: { label: "无显著变化", color: "var(--fg-dim)" },
  inconclusive: { label: "无法确认（不确定性内）", color: "var(--fg-mute)" },
  regressed: { label: "变慢", color: "var(--bad)" },
};

function conclusionChip(c: string | null | undefined) {
  if (!c) return { label: "未定", color: "var(--fg-mute)" };
  return CONCLUSION[c] ?? { label: c, color: "var(--fg-dim)" };
}

/** 归档报告的 verdict（pass/fail）→ 紧凑 chip；没有 verdict 时不渲染。 */
function verdictChip(v: string | null | undefined): { cls: string; label: string } | null {
  if (!v) return null;
  if (v === "pass") return { cls: "chip-ok", label: "verdict pass" };
  if (v === "fail") return { cls: "chip-bad", label: "verdict fail" };
  return { cls: "", label: `verdict ${v}` };
}

/** 报告归档的 per-report UI key（同一 sha 在不同轮次下也可能出现，带上 loopId）。 */
const reportKey = (loopId: number, sha: string) => `${loopId}:${sha}`;

/** sha 展示兜底：后端字段缺失/非字符串时不崩（仅用于展示与文件名）。 */
const shortSha = (sha: unknown, n = 12) => String(sha ?? "").slice(0, n);

/** 归档报告下载文件名：轮次 + sha 前 8。 */
const reportFileName = (loopId: number, sha: unknown) =>
  `optimization-report-loop${loopId}-${shortSha(sha, 8)}.json`;

/** 归档失败文案：后端 detail 原样展示；5xx 且没有 detail 时补一句排查提示（不吞错误）。 */
function archiveErrorText(e: unknown): string {
  const text = errorText(e);
  const raw = String(e);
  if (/^5\d\d\b/.test(raw) && !/"detail"\s*:/.test(raw)) {
    return `${text}（后端内部错误：看控制台日志/磁盘空间）`;
  }
  return text;
}

/** 已运行时长：mm:ss（超过 1 小时显示 h:mm:ss）。 */
function fmtElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * 优化轮次（P1）：把「性能与优化」从手动 A/B 升级为
 * 建轮次（目标/约束/用户思路）→ 采基线 → agent 改代码 → 一键候选验证 → 人工采纳成版本。
 *
 * 契约见 `docs/plan-optimize-rounds-v1.md` §3 / §5 / §6 / §7 / §9。
 * 一键验证由前端顺序调度（P1 不新增后端队列）：plan → test → benchmark，带 loop_id 与设备 pin，
 * 轮询 actionJob 展示阶段与日志尾部；失败即停。
 */
export default function OptimizationRoundPanel({
  task,
  onChanged,
  onOptimizeChat,
  onNewOptimizationChat,
}: {
  task: TaskDetail;
  onChanged: () => void;
  /** 自动优化已对话化：跳到「对话」分区（conversationId 有值 = 打开该轮次已有会话）。 */
  onOptimizeChat?: (loop: Loop, conversationId?: number | null, jobId?: number | null) => void;
  /** 新优化先进任务对话，确认方向后再落 idea/Loop 台账。 */
  onNewOptimizationChat?: () => void;
}) {
  // 轮次 = 带 objective 的 Loop（旧版手动优化 Loop 不在此列）
  const rounds = useMemo(
    () => task.loops.filter((l) => l.objective).sort((a, b) => b.id - a.id),
    [task.loops],
  );
  const oldLoops = useMemo(
    () => task.loops.filter((l) => !l.objective).sort((a, b) => b.id - a.id),
    [task.loops],
  );

  const [savepoints, setSavepoints] = useState<Savepoint[] | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [targets, setTargets] = useState<OptimizationTargets | null>(null);
  const [targetsErr, setTargetsErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busyLoop, setBusyLoop] = useState<number | null>(null);
  const [verify, setVerify] = useState<VerifyState | null>(null);
  const [verifyResult, setVerifyResult] = useState<VerifyResult | null>(null);
  const verifyingRef = useRef(false);
  // 进度区「已运行时长」每秒 tick（只在验证进行中开定时器）
  const [nowTick, setNowTick] = useState(() => Date.now());
  const verifying = verify !== null;

  // agent 按本轮目标改代码：证据按轮次缓存（自动优化过程/SSE 已迁移到「对话」分区）
  const [evidenceByLoop, setEvidenceByLoop] = useState<Record<number, OptimizeEvidence | null>>({});
  // 自动优化尝试历史（confirm→start 可审计记录）：按轮次缓存；无记录不渲染该块
  const [attemptsByLoop, setAttemptsByLoop] = useState<Record<number, LoopAttempt[]>>({});
  const [attemptsLoading, setAttemptsLoading] = useState<Record<number, boolean>>({});
  const [attemptsErr, setAttemptsErr] = useState<Record<number, string | null>>({});
  // 报告归档（append-only）：按轮次缓存归档列表；生成/校验/查看状态按轮次或 `loopId:sha` 记。
  const [reportsByLoop, setReportsByLoop] = useState<Record<number, OptimizationReportEntry[]>>({});
  const [reportsLoading, setReportsLoading] = useState<Record<number, boolean>>({});
  const [reportsErr, setReportsErr] = useState<Record<number, string | null>>({});
  /** per-loop 读取序号：只接受最后发出的列表请求结果（归档后追加的条目不会被早发晚到的旧列表覆盖）。 */
  const reportsSeqRef = useRef<Record<number, number>>({});
  const [archiveBusyLoop, setArchiveBusyLoop] = useState<number | null>(null);
  /** 归档动作结果（成功/失败都留在该轮次小块里；409/400 的 detail 原样展示）。 */
  const [archiveMsg, setArchiveMsg] = useState<Record<number, { ok: boolean; text: string } | null>>({});
  const [verifyBusyKey, setVerifyBusyKey] = useState<string | null>(null);
  /** 深度校验结果（逐文件 sha256）：行 chip 以它为准（`verifiedByKey[key]?.valid ?? rep.valid`）。 */
  const [verifiedByKey, setVerifiedByKey] = useState<Record<string, OptimizationReportVerify>>({});
  const [verifyErrByKey, setVerifyErrByKey] = useState<Record<string, string | null>>({});
  /** 下载归档原始字节：busy/结果都按 key 记（不覆盖归档提示）。 */
  const [downloadBusyKey, setDownloadBusyKey] = useState<string | null>(null);
  const [downloadMsgByKey, setDownloadMsgByKey] = useState<
    Record<string, { ok: boolean; text: string } | null>
  >({});
  /** 查看 manifest 时顺带读到的归档详情（按 key 缓存；校验后失效）。 */
  const [reportDetailByKey, setReportDetailByKey] = useState<Record<string, OptimizationReportDetail>>({});
  const [reportDetailErrByKey, setReportDetailErrByKey] = useState<Record<string, string | null>>({});
  /** manifest 详情读取中的 key（防双击重复 GET）。 */
  const [manifestBusyKey, setManifestBusyKey] = useState<string | null>(null);
  const [manifestOpen, setManifestOpen] = useState<Record<string, boolean>>({});
  /** 归档列表「展开全部」（后端最多 50，默认只显示最近 5 条）。 */
  const [expandReports, setExpandReports] = useState<Record<number, boolean>>({});
  // Step 2 自动验证链（build → smoke → test → benchmark → report）：按轮次缓存最近一次 auto_run。
  // 有 key = 已读到（值可为 null = 后端明确没有）；没有 key = 还没读到（见 autoRunUnknown / loading）。
  const [autoRunByLoop, setAutoRunByLoop] = useState<Record<number, LoopAutoRun | null>>({});
  // 该轮次 auto_run 读取失败（不能当成「没有」：否则会误放开手动验证、漏掉正在跑的自动链）
  const [autoRunUnknown, setAutoRunUnknown] = useState<Record<number, boolean>>({});
  const [autoRunCancelling, setAutoRunCancelling] = useState<number | null>(null);
  /** per-loop 读取/写入序号：只接受最后发出的请求结果，防早发晚到覆盖新状态。 */
  const autoRunSeqRef = useRef<Record<number, number>>({});
  /** 自动链状态镜像：判断「running → 终态」时读它，避免闭包拿到旧 state。 */
  const autoRunStatusRef = useRef<Record<number, string>>({});
  /** 已触发过「终态刷新」的 auto_run id（去重：同一个 auto_run 只 reloadTask 一次）。 */
  const autoRunSettledRef = useRef<Set<number>>(new Set());
  const [showAgentDiff, setShowAgentDiff] = useState<Record<number, boolean>>({});
  // 每次进任务只尝试恢复一次动作作业进度（避免 effect 抖动重复拉 action-jobs）
  const recoveredRef = useRef(false);
  // 自动优化在「对话」分区跑：这里从任务详情推导运行态，用于禁用/提示相关按钮
  const runningOptimizeJob = useMemo(
    () => task.agent_jobs.find((j) => j.kind === "优化" && j.status === "running") ?? null,
    [task.agent_jobs],
  );
  const agentRunning = runningOptimizeJob !== null;
  /** 正在跑自动优化的轮次（用于按钮文案区分「本轮在跑」）。 */
  const agentRunningLoopId = runningOptimizeJob?.loop_id ?? null;

  // P2 优化点池：诊断 job 只读；选点后才建独立 Loop。
  const [ideas, setIdeas] = useState<OptimizationIdea[] | null>(null);
  /** 由思路建轮次的确认/编辑弹窗（选中思路；null = 关闭）。 */
  const [ideaDialog, setIdeaDialog] = useState<OptimizationIdea | null>(null);
  const [ideaSearches, setIdeaSearches] = useState<IdeaSearch[]>([]);
  const [ideaSearchRun, setIdeaSearchRun] = useState<IdeaSearch | null>(null);
  const [ideaExtra, setIdeaExtra] = useState("");
  const [newIdea, setNewIdea] = useState("");
  const [ideaBusy, setIdeaBusy] = useState(false);
  const [ideaCancelBusy, setIdeaCancelBusy] = useState(false);
  const ideaPollRef = useRef(0);
  /** 正在被本面板轮询的搜索 id：mount 恢复与本地启动共用，避免重复轮询（M9）。 */
  const ideaPollSearchRef = useRef<number | null>(null);
  // 运行态以「本地轮询态 ?? 服务端最新 running 搜索」为准：刷新页面后也能看到运行中（M9）。
  const activeIdeaSearch =
    ideaSearchRun ?? ideaSearches.find((s) => s.status === "running") ?? null;
  const ideaRunning = activeIdeaSearch !== null;
  // 最近一次已完成（ok/failed）的搜索：新一轮 running 时也能看到上一次结果（M9）
  const lastFinishedSearch =
    ideaSearches.find((s) => s.status !== "running" && s.summary) ?? null;
  const [budgetAudits, setBudgetAudits] = useState<Record<number, LoopBudgetAudit[]>>({});

  // 自动验证链轮询 key：只包含 status=running 的轮次 id（空串 = 不轮询，避免空转）。
  const autoRunPollKey = useMemo(
    () =>
      rounds
        .filter((r) => autoRunByLoop[r.id]?.status === "running")
        .map((r) => r.id)
        .join(","),
    [rounds, autoRunByLoop],
  );
  /** 是否有轮次的自动链状态读不出来（保守：视为可能在跑）。 */
  const anyAutoRunUnknown = useMemo(
    () => rounds.some((r) => autoRunUnknown[r.id]),
    [rounds, autoRunUnknown],
  );
  /** 是否有轮次的自动链在跑（读不出来的也算，用于禁掉会抢工作区的入口）。 */
  const anyAutoRunRunning = autoRunPollKey !== "" || anyAutoRunUnknown;
  /** 抢工作区入口的禁用提示：确实在跑优先说「运行中」，只有状态未知时才说读不出来。 */
  const autoRunBusyHint = autoRunPollKey !== "" ? AUTO_RUN_BUSY_HINT : AUTO_RUN_UNKNOWN_HINT;

  // 加载函数读 ref 里的最新 task.id / rounds，自身保持稳定引用：
  // 这样重置 effect 只依赖 task.id，父级 task 刷新（onChanged → reloadTask）不会清空面板状态（H1）。
  const taskIdRef = useRef(task.id);
  const dataSeqRef = useRef(0);
  const roundsRef = useRef<Loop[]>([]);
  useEffect(() => {
    taskIdRef.current = task.id;
  }, [task.id]);
  useEffect(() => {
    roundsRef.current = rounds;
  }, [rounds]);

  /** 在途请求是否已过期（切任务/卸载/重新初始化后旧响应不能再写状态）。 */
  const stale = useCallback(
    (taskId: number, seq: number) => seq !== dataSeqRef.current || taskId !== taskIdRef.current,
    [],
  );

  const loadSavepoints = useCallback(async () => {
    const taskId = taskIdRef.current;
    const seq = dataSeqRef.current;
    try {
      const list = await api.listSavepoints(taskId, 200);
      if (stale(taskId, seq)) return;
      setSavepoints(list);
    } catch (e) {
      if (stale(taskId, seq)) return;
      setErr(errorText(e));
    }
  }, [stale]);

  const loadIdeas = useCallback(async () => {
    const taskId = taskIdRef.current;
    const seq = dataSeqRef.current;
    const [pool, searches] = await Promise.all([
      api.optimizationIdeas(taskId),
      api.ideaSearches(taskId),
    ]);
    if (stale(taskId, seq)) return;
    setIdeas(pool);
    setIdeaSearches(searches);
  }, [stale]);

  /** 拉各轮次预算审计；失败向上抛（调用方决定展示方式，不再静默吞）。 */
  const loadBudgetAudits = useCallback(async (targets?: Loop[]) => {
    const taskId = taskIdRef.current;
    const seq = dataSeqRef.current;
    const list = targets ?? roundsRef.current;
    if (list.length === 0) {
      if (!stale(taskId, seq)) setBudgetAudits({});
      return;
    }
    const pairs = await Promise.all(
      list.map(async (round) => [round.id, await api.loopBudgetAudits(round.id)] as const),
    );
    if (stale(taskId, seq)) return;
    setBudgetAudits(Object.fromEntries(pairs));
  }, [stale]);

  /** 写入 mutation 返回的最新自动链快照，并作废该轮次更早发出的 GET（防晚到响应覆盖）。 */
  const applyAutoRun = useCallback((loopId: number, value: LoopAutoRun | null) => {
    autoRunSeqRef.current[loopId] = (autoRunSeqRef.current[loopId] ?? 0) + 1;
    setAutoRunByLoop((prev) => ({ ...prev, [loopId]: value }));
  }, []);

  /**
   * 读某轮次最近一次自动链。失败绝不能当「没有」（会误放开手动验证/漏掉 running 自动链）：
   * 先退避重试一次，仍失败就标记 autoRunUnknown，等「重试读取」或下次轮询/刷新。
   * per-loop 序号保证只接受最后发出的请求结果（早发晚到不覆盖）。
   */
  const readAutoRun = useCallback(async (loopId: number) => {
    const taskId = taskIdRef.current;
    const seq = (autoRunSeqRef.current[loopId] ?? 0) + 1;
    autoRunSeqRef.current[loopId] = seq;
    const current = () => taskId === taskIdRef.current && seq === autoRunSeqRef.current[loopId];
    const apply = (value: LoopAutoRun | null) => {
      if (!current()) return;
      setAutoRunByLoop((prev) => ({ ...prev, [loopId]: value }));
      setAutoRunUnknown((prev) => (prev[loopId] ? { ...prev, [loopId]: false } : prev));
    };
    try {
      const res = await api.loopAutoRun(loopId);
      apply(res.auto_run);
      return;
    } catch {
      /* 退避重试一次（网络抖动/后端重启中） */
    }
    await sleep(800);
    if (!current()) return;
    try {
      const res = await api.loopAutoRun(loopId);
      apply(res.auto_run);
    } catch {
      if (!current()) return;
      setAutoRunUnknown((prev) => ({ ...prev, [loopId]: true }));
    }
  }, []);


  // 切任务 / 卸载：只重置面板状态并首次加载数据，依赖只有 task.id。
  // H1 回归修复：这里绝不能依赖 loadBudgetAudits（依赖 rounds）/loadIdeas 等不稳定引用，
  // 否则父级 task 刷新（onChanged → reloadTask）会被误判为「切任务」，
  // 清空 notice/verifyResult/evidenceByLoop/输入等面板状态。
  useEffect(() => {
    const bootSeq = ++dataSeqRef.current;
    taskIdRef.current = task.id;
    ideaPollRef.current += 1;
    ideaPollSearchRef.current = null;
    setSavepoints(null);
    setNotice(null);
    setErr(null);
    setFormOpen(false);
    setIdeaDialog(null);
    setVerify(null);
    setVerifyResult(null);
    setTargets(null);
    setTargetsErr(null);
    setEvidenceByLoop({});
    setAttemptsByLoop({});
    setAttemptsLoading({});
    setAttemptsErr({});
    setReportsByLoop({});
    setReportsLoading({});
    setReportsErr({});
    setArchiveBusyLoop(null);
    setArchiveMsg({});
    setVerifyBusyKey(null);
    setVerifiedByKey({});
    setVerifyErrByKey({});
    setDownloadBusyKey(null);
    setDownloadMsgByKey({});
    setReportDetailByKey({});
    setReportDetailErrByKey({});
    setManifestBusyKey(null);
    setManifestOpen({});
    setExpandReports({});
    setAutoRunByLoop({});
    setAutoRunUnknown({});
    setAutoRunCancelling(null);
    autoRunSeqRef.current = {};
    autoRunStatusRef.current = {};
    autoRunSettledRef.current = new Set();
    reportsSeqRef.current = {};
    setShowAgentDiff({});
    setIdeas(null);
    setIdeaSearches([]);
    setIdeaSearchRun(null);
    setIdeaExtra("");
    setNewIdea("");
    setIdeaBusy(false);
    setIdeaCancelBusy(false);
    setBusyLoop(null);
    setBudgetAudits({});
    recoveredRef.current = false;
    // 初始数据：失败不再静默吞
    void loadSavepoints();
    void loadIdeas().catch((e) => {
      if (dataSeqRef.current === bootSeq) setErr(errorText(e));
    });
    return () => {
      // 卸载/切任务：作废在途请求与诊断搜索轮询（旧响应不能覆盖新状态，轮询循环下一拍退出）
      dataSeqRef.current += 1;
      ideaPollRef.current += 1;
      ideaPollSearchRef.current = null;
    };
    // load* 是稳定引用 —— 不需要进依赖
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id]);

  // 数据刷新（不重置状态）：轮次集合变化时只重拉预算审计；父级 task 刷新时 rounds id 不变 → 不触发。
  const roundIdsKey = useMemo(() => rounds.map((r) => r.id).join(","), [rounds]);
  // 轮次 id + Step 1 状态机签名：状态迁移（baseline_ready → agent_running → failed/blocked…）
  // 往往伴随 attempts 状态变化，用于重拉尝试历史。
  const roundPhaseSig = useMemo(
    () => rounds.map((r) => `${r.id}:${r.phase ?? ""}:${r.state_version ?? 0}`).join(","),
    [rounds],
  );
  useEffect(() => {
    let cancelled = false;
    void loadBudgetAudits().catch((e) => {
      if (!cancelled) setErr((prev) => prev ?? `预算审计读取失败：${errorText(e)}`);
    });
    return () => {
      cancelled = true;
    };
    // rounds 通过 roundsRef 读取最新值；这里只按轮次 id 集合变化刷新
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roundIdsKey]);

  // 进度区：验证进行中每秒刷新「已运行时长」；同时驱动已运行时长文本。
  useEffect(() => {
    if (!verifying) return;
    setNowTick(Date.now());
    const t = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [verifying]);

  // 别处跑了动作（run 数变化）后，候选存档可能刚出现：顺手刷新存点。
  const runsCountRef = useRef(task.runs.length);
  useEffect(() => {
    if (runsCountRef.current === task.runs.length) return;
    runsCountRef.current = task.runs.length;
    void loadSavepoints();
  }, [task.runs.length, loadSavepoints]);

  const spByLoop = useMemo(() => {
    const m = new Map<number, Savepoint[]>();
    for (const sp of savepoints ?? []) {
      if (sp.loop_id == null) continue;
      const list = m.get(sp.loop_id) ?? [];
      list.push(sp);
      m.set(sp.loop_id, list);
    }
    return m;
  }, [savepoints]);

  const runsByLoop = useMemo(() => {
    const m = new Map<number, Run[]>();
    const byId = new Map(task.runs.map((run) => [run.id, run]));
    for (const r of task.runs) {
      if (r.loop_id == null) continue;
      const list = m.get(r.loop_id) ?? [];
      list.push(r);
      m.set(r.loop_id, list);
    }
    // Reused baseline evidence has explicit links; its original loop_id stays unchanged.
    for (const link of task.loop_run_links ?? []) {
      const run = byId.get(link.run_id);
      if (!run) continue;
      const list = m.get(link.loop_id) ?? [];
      if (!list.some((r) => r.id === run.id)) list.push(run);
      m.set(link.loop_id, list);
    }
    return m;
  }, [task.runs, task.loop_run_links]);

  const roundSps = (loopId: number): Savepoint[] => spByLoop.get(loopId) ?? [];
  const latestOf = (loopId: number, kind: string): Savepoint | null =>
    [...roundSps(loopId)].filter((s) => s.kind === kind).sort((a, b) => b.id - a.id)[0] ?? null;

  // 报告归档：只对「已关闭 / 有 candidate 存点」的轮次拉列表并渲染块。
  // 全新轮次还没候选 benchmark，归档必然 409，不发 GET、也不显示空态（减少每轮一次请求）。
  const archiveLoopIds = useMemo(() => {
    const sps = savepoints ?? [];
    const ids: number[] = [];
    for (const r of rounds) {
      if (r.status === "closed" || sps.some((s) => s.loop_id === r.id && s.kind === "candidate")) {
        ids.push(r.id);
      }
    }
    return ids.sort((a, b) => a - b);
  }, [rounds, savepoints]);
  const archiveKey = archiveLoopIds.join(",");

  // 有 agent 改代码 job 的轮次：任务刷新后按 job 状态签名重拉证据。
  // （自动优化已对话化：job 跑完由 ChatPage 的 onOptimized → 父级 reloadTask 触发这里刷新轮次卡证据）
  const optJobSig = task.agent_jobs
    .filter((j) => j.kind === "优化" && j.loop_id != null)
    .map((j) => `${j.id}:${j.status}`)
    .join(",");
  useEffect(() => {
    let alive = true;
    const ids = new Set<number>();
    for (const j of task.agent_jobs) {
      if (j.kind === "优化" && j.loop_id != null) ids.add(j.loop_id);
    }
    for (const id of ids) {
      api
        .optimizeEvidence(id)
        .then((ev) => {
          if (alive) setEvidenceByLoop((prev) => ({ ...prev, [id]: ev }));
        })
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
    // 只按任务与优化 job 状态集合变化重拉；agent_jobs 数组本身每次 reload 都是新引用
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id, optJobSig]);

  // 自动优化尝试历史（confirm→start 可审计）：进任务 / 轮次集合 / 优化 job 状态变化时拉一次。
  // 轮次卡常驻渲染（无折叠），等价于「卡片出现时加载」；点「刷新证据」也会重拉（见 refreshEvidence）。
  useEffect(() => {
    let alive = true;
    const ids = rounds.map((r) => r.id);
    if (ids.length === 0) {
      setAttemptsByLoop({});
      setAttemptsLoading({});
      setAttemptsErr({});
      return;
    }
    setAttemptsLoading((prev) => {
      const next = { ...prev };
      for (const id of ids) next[id] = true;
      return next;
    });
    for (const id of ids) {
      api
        .loopAttempts(id)
        .then((res) => {
          if (!alive) return;
          setAttemptsByLoop((prev) => ({ ...prev, [id]: res.attempts ?? [] }));
          setAttemptsErr((prev) => (prev[id] == null ? prev : { ...prev, [id]: null }));
        })
        .catch((e) => {
          if (!alive) return;
          setAttemptsErr((prev) => ({ ...prev, [id]: errorText(e) }));
        })
        .finally(() => {
          if (!alive) return;
          setAttemptsLoading((prev) => (prev[id] ? { ...prev, [id]: false } : prev));
        });
    }
    return () => {
      alive = false;
    };
    // rounds 用当前渲染值（roundPhaseSig 已覆盖轮次集合与 phase/state_version 变化）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id, roundPhaseSig, optJobSig]);

  // Step 2 自动验证链：进任务 / 轮次集合 / phase / 优化 job 变化时拉各轮次「最近一次 auto_run」。
  // 读取失败会记 autoRunUnknown（不能当成「没有」；进度块给「重试读取」）。
  useEffect(() => {
    const ids = rounds.map((r) => r.id);
    if (ids.length === 0) {
      setAutoRunByLoop({});
      setAutoRunUnknown({});
      return;
    }
    for (const id of ids) void readAutoRun(id);
    // rounds 用当前渲染值（roundPhaseSig 已覆盖轮次集合与 phase/state_version 变化）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id, roundPhaseSig, optJobSig]);

  // 自动链快照变化：running → 终态时刷新证据/尝试/预算/存点（同一个 auto_run 只触发一次）。
  useEffect(() => {
    const prev = autoRunStatusRef.current;
    const next: Record<number, string> = {};
    const settled: number[] = [];
    for (const [idStr, ar] of Object.entries(autoRunByLoop)) {
      const id = Number(idStr);
      if (!ar) continue;
      next[id] = ar.status;
      if (prev[id] === "running" && ar.status !== "running" && !autoRunSettledRef.current.has(ar.id)) {
        autoRunSettledRef.current.add(ar.id);
        settled.push(id);
      }
    }
    autoRunStatusRef.current = next;
    for (const id of settled) void refreshAutoRunContext(id);
    // refreshAutoRunContext 每次渲染重建，这里只在快照变化时触发，靠 settledRef 去重
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRunByLoop]);

  /**
   * 启动自动优化（已对话化）：本面板不再起 job，直接跳到「对话」分区；
   * 由 ChatPage 的回车入口走 `POST /api/loops/{id}/confirm` → `POST /api/loops/{id}/start`（两步可审计），
   * 过程与续聊都在对话里。
   */
  const openOptimizeChat = (loop: Loop) => {
    if (verifyingRef.current) {
      setErr("真机验证进行中：等当前验证结束后再启动自动优化。");
      return;
    }
    if (anyAutoRunRunning) {
      setErr(autoRunBusyHint);
      return;
    }
    if (!onOptimizeChat) {
      setErr("当前页面未接入对话窗口，无法启动自动优化。");
      return;
    }
    if (agentRunning) {
      setNotice("自动优化正在「对话」分区进行：请到对话窗口查看过程或停止。");
      return;
    }
    onOptimizeChat(loop);
  };

  /** 「在对话中查看」：先查该轮次最近一次自动优化对应的会话，再带 conversationId 跳转。 */
  const openLoopConversation = async (loop: Loop) => {
    if (!onOptimizeChat) {
      setErr("当前页面未接入对话窗口，无法查看优化会话。");
      return;
    }
    const taskId = task.id;
    setBusyLoop(loop.id);
    setErr(null);
    try {
      const r = await api.loopConversation(loop.id);
      // 切任务后旧响应不能再跳到新任务的对话（seed 会带旧 loop）
      if (taskId !== taskIdRef.current) return;
      onOptimizeChat(loop, r.conversation_id ?? undefined);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  /** 轮询一次优化点搜索直到 ok/failed；「本地启动」与「刷新后恢复」共用（token 判重，避免重复轮询）。 */
  const waitIdeaSearch = async (initial: IdeaSearch) => {
    const taskId = taskIdRef.current;
    const token = ++ideaPollRef.current;
    ideaPollSearchRef.current = initial.id;
    setIdeaSearchRun(initial);
    let current = initial;
    try {
      while (current.status === "running" && token === ideaPollRef.current) {
        await sleep(1500);
        if (token !== ideaPollRef.current) return;
        current = await api.ideaSearch(current.id);
        if (token !== ideaPollRef.current) return;
        setIdeaSearchRun(current);
      }
      if (token !== ideaPollRef.current) return;
      await loadIdeas();
      if (token !== ideaPollRef.current || taskId !== taskIdRef.current) return;
      setIdeaSearchRun(null);
      if (current.status === "ok") {
        setNotice(
          `优化点搜索完成：新增 ${current.result.new ?? 0} / 重复 ${current.result.duplicates ?? 0} / 更新 ${current.result.updated ?? 0}`,
        );
      } else if (/取消|cancel|interrupt|终止/i.test(current.summary ?? "")) {
        // 用户主动取消不是故障：用 notice 而不是红色错误
        setNotice(`搜索 #${current.id} 已取消（${current.summary || "用户取消"}）。`);
      } else {
        setErr(current.summary || "优化点搜索失败");
      }
      onChanged();
    } catch (e) {
      // 轮询失败也要退出运行态，避免搜索按钮永远禁用
      if (token !== ideaPollRef.current || taskId !== taskIdRef.current) return;
      setIdeaSearchRun(null);
      setErr(errorText(e));
    } finally {
      if (ideaPollSearchRef.current === initial.id) ideaPollSearchRef.current = null;
    }
  };

  /** 确保某个服务端 running 搜索被本面板轮询；重复调用不会重复轮询（M9）。 */
  const ensureIdeaSearchPolling = (search: IdeaSearch) => {
    if (ideaPollSearchRef.current === search.id) return;
    void waitIdeaSearch(search);
  };

  const startIdeaSearch = async () => {
    if (ideaBusy || ideaRunning) return;
    const taskId = task.id;
    setIdeaBusy(true);
    setErr(null);
    setNotice(null);
    try {
      const search = await api.startIdeaSearch(taskId, ideaExtra.trim() || null);
      if (taskId !== taskIdRef.current) return;
      setIdeaExtra("");
      ensureIdeaSearchPolling(search);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setErr(errorText(e));
      // 409「已有搜索在跑」等场景：重拉列表，让 UI 显示服务端 running 态并自动接管轮询
      try {
        await loadIdeas();
      } catch {
        /* 列表刷新失败不覆盖原始错误 */
      }
    } finally {
      setIdeaBusy(false);
    }
  };

  // M9：页面刷新/重进后，若服务端还有 running 搜索（且有关联 job）→ 自动接管轮询（不重复起搜索）。
  useEffect(() => {
    const running = ideaSearches.find((s) => s.status === "running");
    if (!running || running.agent_job_id == null) return;
    ensureIdeaSearchPolling(running);
    // ensureIdeaSearchPolling/waitIdeaSearch 每次渲染重建，但内部用 ref/token 判重；只在搜索列表变化时接管
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ideaSearches]);

  /** 取消运行中的诊断搜索：cancel 后继续轮询到 failed/ok，再 loadIdeas 刷新思路池。 */
  const cancelIdeaSearch = async () => {
    const search = activeIdeaSearch;
    if (!search || ideaCancelBusy) return;
    const taskId = task.id;
    if (search.agent_job_id == null) {
      setErr(`搜索 #${search.id} 没有关联诊断 job，无法取消；可稍后刷新查看结果`);
      return;
    }
    setIdeaCancelBusy(true);
    setErr(null);
    try {
      const res = await api.cancelAgentJob(search.agent_job_id);
      if (taskId !== taskIdRef.current) return;
      setNotice(`${res.result || `已请求取消搜索 #${search.id}`}；正在等待后端收尾…`);
      // 本地还没接管轮询时补上（已接管则沿用原 token 的循环）
      ensureIdeaSearchPolling(search);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setErr(errorText(e));
    } finally {
      setIdeaCancelBusy(false);
    }
  };

  const addIdea = async () => {
    const title = newIdea.trim();
    if (!title || ideaBusy) return;
    setIdeaBusy(true);
    setErr(null);
    try {
      const result = await api.addOptimizationIdea(task.id, {
        title,
        hypothesis: title,
        change_scope: task.op_name ? `ops/${task.op_name}/` : "ops/",
      });
      setNewIdea("");
      await loadIdeas();
      setNotice(result.deduplicated ? `已有同一思路 #${result.idea.id}，未重复添加` : `已把用户思路 #${result.idea.id} 加入候选池`);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setIdeaBusy(false);
    }
  };

  /**
   * 由思路建轮次（P2 二次修订）：先由弹窗编辑「需要优化的点」等目标/约束，再以 text objective 建独立 Loop。
   * 失败向上抛给弹窗就地展示；成功后与「新建轮次」共用 loadIdeas + handleCreated。
   */
  const createIdeaRound = async (
    idea: OptimizationIdea,
    objective: OptimizationObjective,
    attachBaseline: boolean,
  ) => {
    if (ideaBusy || idea.loop_id != null) return;
    const taskId = task.id;
    setIdeaBusy(true);
    setErr(null);
    setNotice(null);
    try {
      const result = await api.createRoundFromIdea(idea.id, {
        objective,
        attach_baseline_runs: attachBaseline,
      });
      // 切任务/卸载后旧响应不能再写新任务的状态
      if (taskId !== taskIdRef.current) return;
      setIdeaDialog(null);
      try {
        await loadIdeas();
      } catch {
        /* 轮次已建：思路池刷新失败不当作创建失败 */
      }
      if (taskId !== taskIdRef.current) return;
      await handleCreated(result);
    } finally {
      setIdeaBusy(false);
    }
  };

  const rejectIdea = async (idea: OptimizationIdea) => {
    if (
      !window.confirm(
        `不采用思路 #${idea.id}「${idea.title}」？\n\n` +
          "该思路会被标记为 rejected 并保留在候选池（不会删除），之后不能再用它建轮次。",
      )
    ) {
      return;
    }
    setIdeaBusy(true);
    setErr(null);
    try {
      await api.patchOptimizationIdea(idea.id, "rejected", { reason: "用户在候选池中不采用" });
      await loadIdeas();
      setNotice(`已把思路 #${idea.id} 标记为不采用（保留在候选池，可查历史）。`);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setIdeaBusy(false);
    }
  };

  /** 拉一个轮次的尝试历史（best-effort：失败只写进该轮次的小块，不打扰主流程）。 */
  const refreshAttempts = (loopId: number) => {
    const taskId = task.id;
    setAttemptsLoading((prev) => ({ ...prev, [loopId]: true }));
    void api
      .loopAttempts(loopId)
      .then((res) => {
        if (taskId !== taskIdRef.current) return;
        setAttemptsByLoop((prev) => ({ ...prev, [loopId]: res.attempts ?? [] }));
        setAttemptsErr((prev) => (prev[loopId] == null ? prev : { ...prev, [loopId]: null }));
      })
      .catch((e) => {
        if (taskId !== taskIdRef.current) return;
        setAttemptsErr((prev) => ({ ...prev, [loopId]: errorText(e) }));
      })
      .finally(() => {
        if (taskId !== taskIdRef.current) return;
        setAttemptsLoading((prev) => (prev[loopId] ? { ...prev, [loopId]: false } : prev));
      });
  };

  /**
   * 拉某轮次的报告归档列表（append-only，最新在前）。
   * best-effort：失败只写进该轮次小块（不遮挡轮次主流程），成功时顺带清掉旧错误。
   */
  const loadReports = useCallback(
    async (loopId: number) => {
      const taskId = taskIdRef.current;
      const seq = (reportsSeqRef.current[loopId] ?? 0) + 1;
      reportsSeqRef.current[loopId] = seq;
      const current = () => taskId === taskIdRef.current && seq === reportsSeqRef.current[loopId];
      setReportsLoading((prev) => ({ ...prev, [loopId]: true }));
      try {
        const res = await api.optimizationReports(loopId);
        if (!current()) return;
        setReportsByLoop((prev) => ({ ...prev, [loopId]: res.reports ?? [] }));
        setReportsErr((prev) => (prev[loopId] == null ? prev : { ...prev, [loopId]: null }));
      } catch (e) {
        if (!current()) return;
        setReportsErr((prev) => ({ ...prev, [loopId]: errorText(e) }));
      } finally {
        if (current()) {
          setReportsLoading((prev) => (prev[loopId] ? { ...prev, [loopId]: false } : prev));
        }
      }
    },
    [],
  );

  // 归档列表：可归档轮次集合（已关闭 / 有 candidate 存点）或 phase 签名变化时重拉。
  // append-only：重拉只反映新增，不会把旧归档冲掉。
  useEffect(() => {
    if (!archiveKey) return;
    for (const id of archiveKey.split(",").map(Number)) void loadReports(id);
    // loadReports 是稳定引用；这里只按可归档轮次集合 + phase 签名刷新
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archiveKey, roundPhaseSig]);

  /**
   * 自动链起链/结束后刷新关联数据（best-effort）：改码证据、尝试历史、预算审计、存点，并 reloadTask
   * （phase/runs 变化由父级重拉任务详情驱动按钮与对比区）。失败只提示，不遮挡自动链进度。
   */
  const refreshAutoRunContext = async (loopId: number) => {
    const taskId = taskIdRef.current;
    refreshAttempts(loopId);
    try {
      const ev = await api.optimizeEvidence(loopId);
      if (taskId !== taskIdRef.current) return;
      setEvidenceByLoop((prev) => ({ ...prev, [loopId]: ev }));
    } catch {
      /* best-effort：证据刷新失败不挡住预算/任务刷新 */
    }
    try {
      await loadBudgetAudits();
    } catch (e) {
      if (taskId === taskIdRef.current) setErr(`自动链状态已更新，但预算审计刷新失败：${errorText(e)}`);
    }
    if (taskId !== taskIdRef.current) return;
    void loadSavepoints();
    onChanged();
  };

  // Step 2 自动验证链：running 时每 2s 读一次（readAutoRun 内部有 per-loop 序号/unknown 处理）；
  // 不是 running 不轮询（autoRunPollKey 为空即停）；后台标签页暂停，可见后下一拍继续。
  useEffect(() => {
    if (!autoRunPollKey) return;
    const ids = autoRunPollKey.split(",").map(Number);
    const timer = window.setInterval(() => {
      if (document.hidden) return; // 后台标签页不轮询，回到前台下一拍继续
      for (const id of ids) void readAutoRun(id);
    }, 2000);
    return () => window.clearInterval(timer);
    // readAutoRun 是稳定引用；按轮询 key 重启定时器即可
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id, autoRunPollKey]);

  const refreshEvidence = async (loop: Loop) => {
    setBusyLoop(loop.id);
    setErr(null);
    // 「刷新证据」顺带重拉该轮次的尝试历史（独立 best-effort）
    refreshAttempts(loop.id);
    try {
      const ev = await api.optimizeEvidence(loop.id);
      setEvidenceByLoop((prev) => ({ ...prev, [loop.id]: ev }));
      setNotice(
        ev.status === "none"
          ? `轮次 #${loop.id} 还没有自动优化记录`
          : `已刷新轮次 #${loop.id} 的自动优化证据（${
              ev.guard_reverted ? "已自动回退" : ev.status === "failed" ? "失败" : ev.changed ? "有改动" : "无改动"
            }）`,
      );
      onChanged();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  const revertOptimize = async (loop: Loop) => {
    setBusyLoop(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const ev = await api.revertOptimize(loop.id);
      setEvidenceByLoop((prev) => ({ ...prev, [loop.id]: ev }));
      setNotice(`已把工作区回退到轮次 #${loop.id} 本轮改前（${ev.reverted_clean ? "与改前一致" : "有残留，请检查"}）。`);
      onChanged();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  /**
   * Step 2：一键自动验证链（build → smoke → test → benchmark → 报告）。
   * 后端异步驱动（201 立即返回），这里写入首个快照，进度交给 2s 轮询；409 直接展示 detail（E_AUTO_*）。
   */
  const startAutoRun = async (loop: Loop) => {
    if (verifyingRef.current) {
      setErr("手动验证进行中：等它结束再跑自动验证链（避免两边抢同一张卡）。");
      return;
    }
    const taskId = task.id;
    setBusyLoop(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const res = await api.startAutoRun(loop.id);
      if (taskId !== taskIdRef.current) return;
      applyAutoRun(loop.id, res.auto_run);
      setNotice(
        `已启动轮次 #${loop.id} 的自动验证链（build → smoke → test → benchmark → 报告），进度在下方实时更新；可随时取消。`,
      );
      // 起链成功：刷新证据/尝试/预算并重拉任务，让 phase 与按钮立即同步（后续进度由轮询接管）
      void refreshAutoRunContext(loop.id);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setErr(errorText(e));
      // 409 常见于「已有自动链在跑」：立刻回拉一次最新状态，让进度块接管轮询（不覆盖原始错误文案）
      void readAutoRun(loop.id);
    } finally {
      setBusyLoop(null);
    }
  };

  /** 取消运行中的自动链（后端杀动作作业 + 远端清理；工作区保留现状，不回退）。 */
  const cancelAutoRun = async (loop: Loop) => {
    if (
      !window.confirm(
        `取消轮次 #${loop.id} 的自动验证链？\n\n` +
          "会终止当前动作并清理远端容器；工作区保留现状（不回退）。\n" +
          "之后需要回退，请用「否决」时的回退选项。",
      )
    ) {
      return;
    }
    const taskId = task.id;
    setAutoRunCancelling(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const res = await api.cancelAutoRun(loop.id);
      if (taskId !== taskIdRef.current) return;
      // 取消是同步终态：先记 settled，避免 in-flight 轮询把 running 写回后多触发一次终态刷新
      autoRunSettledRef.current.add(res.auto_run.id);
      applyAutoRun(loop.id, res.auto_run);
      const c = res.cancelled;
      const parts: string[] = [];
      if (c?.action_job_id != null) parts.push(`已终止动作作业 #${c.action_job_id}`);
      if (c?.remote?.ok === true) parts.push("远端容器已清理");
      else if (c?.remote?.ok === false) parts.push(`远端清理未确认${c.remote.why ? `：${c.remote.why}` : ""}`);
      else if (typeof c?.local === "string" && c.local) parts.push(`本地：${c.local}`);
      setNotice(`已取消轮次 #${loop.id} 的自动验证链${parts.length ? `（${parts.join("；")}）` : ""}。工作区保留现状。`);
      void refreshAutoRunContext(loop.id);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setErr(errorText(e));
      // 409「本轮没有运行中的自动链」等：立刻回拉一次，别等下一拍轮询
      void readAutoRun(loop.id);
    } finally {
      setAutoRunCancelling(null);
    }
  };

  const openForm = async () => {
    if (agentRunning || verifyingRef.current) return;
    setFormOpen(true);
    setErr(null);
    setTargets(null);
    setTargetsErr(null);
    try {
      setTargets(await api.optimizationTargets(task.id));
    } catch (e) {
      setTargetsErr(errorText(e));
    }
  };

  const handleCreated = async (res: OptimizationRoundCreated) => {
    setFormOpen(false);
    const attached = res.attached_runs.length
      ? `；复用 ${res.attached_runs.map((r) => `#${r.id} ${r.action}`).join(" ")}`
      : "";
    setNotice(
      `已建轮次 #${res.loop.id}（基线存档 #${res.baseline_savepoint.id}${attached}）` +
        "。先跑基线，再让 agent 改代码，回这里跑候选验证。",
    );
    // 先让父级重拉任务（轮次卡片立刻出现），再去补附属数据：
    // 附属请求慢或失败都不能让「轮次」停在旧值（实测修复前要手动点「刷新」才出现）。
    onChanged();
    try {
      await loadSavepoints();
    } catch {
      /* 非致命：保存点列表下一拍会自己拉回来 */
    }
  };

  const refreshAll = async () => {
    setNotice(null);
    setErr(null);
    try {
      await loadSavepoints();
      await Promise.all([loadIdeas(), loadBudgetAudits()]);
      // 顺手刷新各轮次的 agent 改代码证据（只对有 agent job 的轮次请求）
      const ids = new Set<number>();
      for (const j of task.agent_jobs) {
        if (j.kind === "优化" && j.loop_id != null) ids.add(j.loop_id);
      }
      await Promise.all(
        [...ids].map(async (id) => {
          try {
            const ev = await api.optimizeEvidence(id);
            setEvidenceByLoop((prev) => ({ ...prev, [id]: ev }));
          } catch {
            /* best-effort：刷新证据失败不影响主流程 */
          }
        }),
      );
      // 顺带重拉各轮次的尝试历史（best-effort，失败只写在对应轮次的小块里）
      for (const r of rounds) refreshAttempts(r.id);
      // 顺带重拉各轮次最近一次自动验证链（读失败会标 unknown；running 时由 2s 轮询接管）
      for (const r of rounds) void readAutoRun(r.id);
      // 顺带重拉可归档轮次的报告列表（append-only；失败只写到各自小块里）
      for (const id of archiveLoopIds) void loadReports(id);
    } catch (e) {
      // 预算审计等失败不再静默吞：面板顶部显式报错
      setErr(`刷新失败：${errorText(e)}`);
    } finally {
      // 父级刷新放 finally：任何附属请求失败都不能让「轮次列表」停在旧值
      onChanged();
    }
  };

  async function pollJob(jobId: number, onLog: (log: string) => void): Promise<ActionJob> {
    for (let i = 0; i < 600; i += 1) {
      const job = await api.actionJob(jobId);
      onLog(job.log ?? "");
      if (!job.running) return job;
      await sleep(1500);
    }
    throw new Error("等待执行作业超时（15 分钟）");
  }

  /**
   * 一键验证（P1 前端顺序调度）：plan → test → benchmark，同一张卡（device pin）。
   * 缺哪个动作就跑哪个；失败即停并把错误 + 日志尾部留在轮次卡片上。
   * `resume`：页面刷新/重进后接手一个仍在 running 的动作作业（不重复起作业，先轮询它再补剩余动作）。
   */
  const runVerification = async (loop: Loop, target: VerifyTarget, resume?: ActionJob, force = false) => {
    if (verifyingRef.current) return;
    // 自动链运行中：它会自己串跑动作作业，手动验证再起作业会抢同一份工作区/同一张卡
    if (anyAutoRunRunning) {
      setErr(autoRunBusyHint);
      return;
    }
    const resumeAction =
      resume && (resume.action === "test" || resume.action === "benchmark") ? resume.action : null;
    const sides = sidesOf(runsByLoop.get(loop.id) ?? []);
    let missing = target === "baseline" ? missingActions(sides.before) : missingActions(sides.after);
    if (target === "candidate" && missingActions(sides.before).length > 0) {
      setErr("候选验证前需要基线 test + benchmark 都就绪（先跑基线）。");
      return;
    }
    // 强制重跑（agent 又改了代码 / 恢复新快照的作业）：即使旧候选侧「已就绪」也要重跑两侧动作。
    if ((force || resumeAction) && missing.length === 0) missing = [...ACTIONS];
    // 恢复时：在跑的 action 还没落成 ok run，可能仍在 missing 里；先收尾它，再补剩下的。
    const remaining = resumeAction ? missing.filter((a) => a !== resumeAction) : missing;
    if (!resumeAction && missing.length === 0) {
      setNotice(`轮次 #${loop.id} 的${target === "baseline" ? "基线" : "候选"} run 已就绪，无需重跑。`);
      return;
    }

    verifyingRef.current = true;
    setErr(null);
    // 恢复进度时保留“已恢复…”提示；正常点击开始时清掉旧提示
    if (!resume) setNotice(null);
    setVerifyResult(null);
    const pinStart = resume?.device_id ?? initialDevicePin(task);
    const resumeStarted = resume?.started_at != null ? Date.parse(resume.started_at) : Number.NaN;
    setVerify({
      loopId: loop.id,
      target,
      stage: resumeAction ?? "plan",
      action: resumeAction,
      jobId: resume?.id ?? null,
      deviceId: pinStart,
      log: resume?.log ?? "",
      startedAt: Number.isFinite(resumeStarted) ? resumeStarted : Date.now(),
    });

    let lastLog = resume?.log ?? "";
    const pinned = { v: pinStart };
    try {
      if (resumeAction && resume) {
        // 接手已有作业：只轮询、不再起；它的日志继续进进度区
        const resumed = await pollJob(resume.id, (log) => {
          lastLog = log;
          setVerify((v) => (v ? { ...v, stage: resumeAction, action: resumeAction, log } : v));
        });
        if (resumed.status !== "ok") {
          const exit = resumed.exit_code != null ? `，exit ${resumed.exit_code}` : "";
          const why = resumed.error ? `：${resumed.error}` : "";
          throw new Error(`${resumeAction} 未通过（${resumed.status}${exit}）${why}`);
        }
        // 退出码不是结论：门禁 fail（版本不对齐/产物缺失/用例 NOT_RUN 等）即使 exit 0 也不算通过。
        if (resumed.gate_verdict && resumed.gate_verdict !== "pass") {
          const why = resumed.error ? `：${resumed.error}` : "";
          throw new Error(`${resumeAction} 门禁未通过（${resumed.gate_verdict}）${why}`);
        }
      }
      for (const action of remaining) {
        setVerify((v) => (v ? { ...v, stage: "plan", action: null, jobId: null } : v));
        const plan = await api.planAction(task.id, action, pinned.v);
        if (!plan.ready) {
          const why = [...plan.missing, ...plan.warnings].filter(Boolean).join("；") || "未知原因";
          throw new Error(`执行计划未就绪（${action}）：${why}`);
        }
        pinned.v = pinned.v ?? plan.device_id ?? null;
        setVerify((v) => (v ? { ...v, stage: action, action, deviceId: pinned.v } : v));
        const started = await api.runAction(task.id, action, pinned.v, loop.id);
        setVerify((v) => (v ? { ...v, stage: action, action, jobId: started.action_job_id } : v));
        const job = await pollJob(started.action_job_id, (log) => {
          lastLog = log;
          setVerify((v) => (v ? { ...v, log } : v));
        });
        if (job.status !== "ok") {
          const exit = job.exit_code != null ? `，exit ${job.exit_code}` : "";
          const why = job.error ? `：${job.error}` : "";
          throw new Error(`${action} 未通过（${job.status}${exit}）${why}`);
        }
        // 退出码不是结论：以 gate verdict 为准（见 docs/gate-report-v1.md）。
        if (job.gate_verdict && job.gate_verdict !== "pass") {
          const why = job.error ? `：${job.error}` : "";
          throw new Error(`${action} 门禁未通过（${job.gate_verdict}）${why}`);
        }
      }
      setVerify(null);
      setVerifyResult({
        loopId: loop.id,
        target,
        ok: true,
        text: `${target === "baseline" ? "基线" : "候选"}验证完成：test / benchmark 均通过（门禁 pass）。${
          target === "candidate" ? "候选存档已自动生成，可采纳或否决。" : ""
        }`,
        log: lastLog,
      });
      await loadSavepoints();
      try {
        await loadBudgetAudits();
      } catch (e2) {
        // 验证本身已成功：审计刷新失败只提示，不覆盖验证结论
        setErr(`预算审计刷新失败：${errorText(e2)}`);
      }
      onChanged();
    } catch (e) {
      setVerify(null);
      setVerifyResult({ loopId: loop.id, target, ok: false, text: errorText(e), log: lastLog });
      try {
        await loadBudgetAudits();
      } catch (e2) {
        setErr(`${errorText(e)}；预算审计刷新失败：${errorText(e2)}`);
      }
    } finally {
      verifyingRef.current = false;
    }
  };

  // 刷新/重进页面：若本任务有 running 且挂在某轮次的动作作业，恢复进度并继续轮询（不重复起作业）。
  useEffect(() => {
    if (recoveredRef.current) return;
    recoveredRef.current = true;
    let cancelled = false;
    void (async () => {
      try {
        const jobs = await api.actionJobs(task.id);
        if (cancelled) return;
        const running = jobs.find(
          (j) =>
            j.running &&
            j.loop_id != null &&
            (j.action === "test" || j.action === "benchmark") &&
            rounds.some((l) => l.id === j.loop_id),
        );
        if (!running || running.loop_id == null || verifyingRef.current) return;
        const loop = rounds.find((l) => l.id === running.loop_id);
        if (!loop) return;
        // 该动作作业可能属于自动验证链：先确认 auto_run。
        // 读失败不能当「没有」（否则会接手自动链的 test/benchmark 作业并补跑后续动作，抢设备）：
        // 只提示、不自动恢复；只有明确 auto_run === null（旧任务无记录）才走手动恢复。
        let autoRunSnapshot: LoopAutoRun | null = null;
        try {
          const ar = await api.loopAutoRun(loop.id);
          if (cancelled) return;
          autoRunSnapshot = ar.auto_run;
        } catch {
          if (cancelled) return;
          setAutoRunUnknown((prev) => ({ ...prev, [loop.id]: true }));
          setErr("无法确认该轮次是否有自动链在跑（状态读取失败）：为避免抢设备，本次不自动恢复验证进度；请刷新后重试。");
          return;
        }
        if (cancelled) return;
        setAutoRunUnknown((prev) => (prev[loop.id] ? { ...prev, [loop.id]: false } : prev));
        applyAutoRun(loop.id, autoRunSnapshot);
        if (autoRunSnapshot !== null) return; // 有自动链记录（running 或终态）：交给进度块，不按手动验证恢复
        const sides = sidesOf(runsByLoop.get(loop.id) ?? []);
        const target: VerifyTarget = missingActions(sides.before).length > 0 ? "baseline" : "candidate";
        setNotice(
          `已恢复轮次 #${loop.id} 的运行中 ${running.action} 作业（job #${running.id}，设备 ${
            running.device_id ?? "—"
          }），继续轮询进度（不重复起作业）。`,
        );
        void runVerification(loop, target, running);
      } catch {
        /* 恢复失败不打扰用户：按钮仍可手动重跑 */
      }
    })();
    return () => {
      cancelled = true;
    };
    // runVerification 每次渲染重建，这里只在任务/数据首次就绪时尝试一次（recoveredRef 兜底）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id, rounds, runsByLoop]);


  const acceptRound = async (loop: Loop) => {
    const label = window.prompt("版本名（可选，留空按 cN 自动命名）", "");
    if (label === null) return;
    setBusyLoop(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const res = await api.acceptLoop(loop.id, { label: label.trim() || null });
      setNotice(
        `已采纳轮次 #${loop.id} 为版本 c${res.version_number}（存点 #${res.version.id}）` +
          (res.workspace_matches
            ? "；当前工作区就是候选代码。"
            : "；注意：当前工作区与该候选代码不一致（工作区已被改过），如需保留当前代码请先在「版本」分区存档。"),
      );
      await loadSavepoints();
      onChanged();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  /**
   * 否决本轮（P2 二次修订）：所有未关闭轮次都可否决（含基线未跑/agent 失败/预算耗尽）。
   * 先收原因，再问是否同时回退工作区（默认回退）；回退失败不影响否决，只把原因展示给人。
   */
  const rejectRound = async (loop: Loop) => {
    const reason = window.prompt("否决原因（会写进轮次结论）", "性能回退，不采纳");
    if (reason === null) return;
    const text = reason.trim() || "不采纳";
    const revert = window.confirm(
      "是否同时把工作区回退到本轮改前？\n\n" +
        "确定 = 回退（优先用本轮自动优化的改前快照，其次建轮次时的基线存档）\n" +
        "取消 = 只否决，工作区保持现状",
    );
    const taskId = task.id;
    setBusyLoop(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const res = await api.rejectLoop(loop.id, { reason: text, revert });
      // 切任务后旧响应不能再写新任务状态
      if (taskId !== taskIdRef.current) return;
      const reverted = res.reverted;
      let msg = `已否决轮次 #${loop.id}（${text}）。`;
      if (!revert) {
        msg += "工作区保持现状（未回退）；基线/救援存档仍在，「版本」分区可手动回滚。";
      } else if (reverted?.ok) {
        const mode = reverted.mode ? REVERT_MODE_LABEL[reverted.mode] ?? reverted.mode : "本轮改前";
        msg += `工作区已回退到本轮改前（${mode}）。`;
      } else {
        msg += `但工作区未回退：${reverted?.reason || "后端未返回可用的改前快照"}；基线存档仍在，「版本」分区可手动回滚。`;
      }
      setNotice(msg);
      await loadSavepoints();
      try {
        await loadIdeas();
      } catch (e2) {
        setErr(`否决已生效，但思路池刷新失败：${errorText(e2)}`);
      }
      onChanged();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  const exportReport = async (loop: Loop) => {
    setBusyLoop(loop.id);
    setErr(null);
    setNotice(null);
    try {
      const report = await api.optimizationReport(loop.id);
      downloadJson(`optimization-report-loop${loop.id}.json`, report);
      setNotice(
        `已导出轮次 #${loop.id} 可复现报告` +
          (report.objective_summary?.headline ? `：${report.objective_summary.headline}` : ""),
      );
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusyLoop(null);
    }
  };

  /**
   * 生成并归档报告（Step 3，append-only）：每次写独立目录（report.json + manifest.json + raw/ 副本）。
   * 轮次未关闭 / 缺 after benchmark / 落盘失败时后端 409/400：detail 原样展示在该轮次小块里，不吞。
   */
  const archiveReport = async (loop: Loop) => {
    const taskId = task.id;
    setArchiveBusyLoop(loop.id);
    setArchiveMsg((prev) => ({ ...prev, [loop.id]: null }));
    try {
      const res = await api.archiveOptimizationReport(loop.id);
      if (taskId !== taskIdRef.current) return;
      // manifest.raw 可能含「超限未复制」条目：raw_copied_count/raw_total 由后端给，缺失时按数组兜底
      const rawList = Array.isArray(res.manifest?.raw) ? res.manifest.raw : [];
      const rawCopied = res.manifest?.raw_copied_count ?? rawList.filter((r) => r.copied).length;
      const rawTotal = res.manifest?.raw_total ?? rawList.length;
      const skipped = Math.max(0, rawTotal - rawCopied);
      setArchiveMsg((prev) => ({
        ...prev,
        [loop.id]: {
          ok: true,
          text:
            `${res.stored ? "已归档" : "同内容已归档（复用现有归档）"}轮次 #${loop.id} 报告 #${res.seq}` +
            `（sha ${shortSha(res.report_sha256)}…，raw 副本 ${rawCopied} 个` +
            (skipped > 0 ? `，另有 ${skipped} 个超限未复制` : "") +
            `，manifest ${res.report_valid ? "自校验有效" : "自校验未通过"}）；写入 ${res.written_to}`,
        },
      }));
      // 新归档只追加：重拉列表即可（不影响已有条目）
      void loadReports(loop.id);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      // 409/400 的 detail 是后端人话（本轮未关闭/缺 after benchmark/无法落盘）：原样展示
      setArchiveMsg((prev) => ({ ...prev, [loop.id]: { ok: false, text: archiveErrorText(e) } }));
      // 失败也可能是「已有归档但列表没读到」：顺手重拉一次列表，错误文案保持不变
      void loadReports(loop.id);
    } finally {
      if (taskId === taskIdRef.current) setArchiveBusyLoop((prev) => (prev === loop.id ? null : prev));
    }
  };

  /** 读某归档的 manifest/report 详情（manifest 展开用；按 key 缓存，读取中由 manifestBusyKey 防重复）。 */
  const loadReportDetail = async (loop: Loop, reportSha256: string) => {
    const key = reportKey(loop.id, reportSha256);
    const taskId = task.id;
    setManifestBusyKey(key);
    setReportDetailErrByKey((prev) => ({ ...prev, [key]: null }));
    try {
      const res = await api.getOptimizationReport(loop.id, reportSha256);
      if (taskId !== taskIdRef.current) return;
      setReportDetailByKey((prev) => ({ ...prev, [key]: res }));
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setReportDetailErrByKey((prev) => ({ ...prev, [key]: errorText(e) }));
    } finally {
      if (taskId === taskIdRef.current) setManifestBusyKey((prev) => (prev === key ? null : prev));
    }
  };

  /**
   * 深度校验某个归档：逐文件 sha256 + raw 副本；结果就地展示（valid/errors/首个失败文件）。
   * 行 chip 以深度结果为准（`verifiedByKey`）；成功后让 manifest 详情缓存失效（已展开则重拉），
   * 避免「校验失效但 manifest 面板还是旧的有效」互相冲突。
   */
  const verifyReport = async (loop: Loop, reportSha256: string) => {
    const key = reportKey(loop.id, reportSha256);
    const taskId = task.id;
    setVerifyBusyKey(key);
    setVerifyErrByKey((prev) => ({ ...prev, [key]: null }));
    try {
      const res = await api.verifyOptimizationReport(loop.id, reportSha256);
      if (taskId !== taskIdRef.current) return;
      setVerifiedByKey((prev) => ({ ...prev, [key]: res }));
      setReportDetailByKey((prev) => {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
      if (manifestOpen[key]) void loadReportDetail(loop, reportSha256);
      // 列表里的 valid 是快速校验结果，后端会在深度校验后刷新：重拉同步（行 chip 仍以深度结果优先）
      void loadReports(loop.id);
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      // 请求本身失败（不是校验结论）：保留上一次深度结果，只展示请求错误
      setVerifyErrByKey((prev) => ({ ...prev, [key]: errorText(e) }));
    } finally {
      if (taskId === taskIdRef.current) {
        setVerifyBusyKey((prev) => (prev === key ? null : prev));
      }
    }
  };

  /**
   * 下载归档原始字节：直接走 `.../download`（返回归档里的 report.json 原字节，sha256 == report_sha256）。
   * 绝不能前端 JSON 重新序列化后再下载——那样字节变了，sha 必然对不上。
   */
  const downloadArchivedReport = async (loop: Loop, reportSha256: string) => {
    const key = reportKey(loop.id, reportSha256);
    const taskId = task.id;
    setDownloadBusyKey(key);
    setDownloadMsgByKey((prev) => ({ ...prev, [key]: null }));
    try {
      const res = await fetch(api.archivedReportDownloadUrl(loop.id, reportSha256));
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${text.slice(0, 4000)}`);
      }
      const blob = await res.blob();
      if (taskId !== taskIdRef.current) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = reportFileName(loop.id, reportSha256);
      a.click();
      // 延迟回收：部分浏览器在 click 后异步才开始读 blob，立即 revoke 会中断下载
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setDownloadMsgByKey((prev) => ({
        ...prev,
        [key]: {
          ok: true,
          text: `已下载归档原始字节 ${shortSha(reportSha256, 8)}…（report.json；可直接用 sha256 校验）`,
        },
      }));
    } catch (e) {
      if (taskId !== taskIdRef.current) return;
      setDownloadMsgByKey((prev) => ({ ...prev, [key]: { ok: false, text: `下载失败：${errorText(e)}` } }));
    } finally {
      if (taskId === taskIdRef.current) setDownloadBusyKey((prev) => (prev === key ? null : prev));
    }
  };

  /** 展开/收起某归档的 manifest（首次展开才读详情；读取中重复点击被 manifestBusyKey 挡住）。 */
  const toggleManifest = async (loop: Loop, reportSha256: string) => {
    const key = reportKey(loop.id, reportSha256);
    const open = manifestOpen[key] ?? false;
    setManifestOpen((prev) => ({ ...prev, [key]: !open }));
    if (open) return; // 收起不请求
    if (manifestBusyKey === key) return; // 读取中：防双击重复 GET
    if (reportDetailByKey[key]) return; // 已有缓存
    await loadReportDetail(loop, reportSha256);
  };

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">优化轮次</h2>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          目标/约束 → 采基线 → agent 按目标改代码 → 一键候选验证 → 前后对比 → 采纳成版本 / 否决
        </span>
        <div className="ml-auto flex items-center gap-2">
          <button
            className="btn btn-primary btn-sm"
            onClick={onNewOptimizationChat}
            disabled={verifying || agentRunning || !onNewOptimizationChat}
            title="先到任务对话说优化方向；系统再识别、记录 idea/Loop，并在基线齐全时启动 Agent"
          >
            去对话提出优化方向
          </button>
          <button
            className="btn btn-sm"
            onClick={() => void openForm()}
            disabled={formOpen || verifying || agentRunning || anyAutoRunRunning}
            title={
              anyAutoRunRunning
                ? autoRunBusyHint
                : agentRunning
                  ? "自动优化正在「对话」窗口进行：等它结束再建新轮次（避免把半成品当基线）"
                  : undefined
            }
          >
            高级建轮次
          </button>
          <button className="btn btn-sm" onClick={() => void refreshAll()} disabled={verifying || busyLoop !== null}>
            刷新
          </button>
        </div>
      </div>

      {notice && (
        <div className="mx-4 mt-2 rounded-[8px] px-3 py-2 text-[11px]" style={{ border: "1px solid var(--ok)", color: "var(--ok)" }}>
          {notice}
        </div>
      )}
      {err && (
        <div className="px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>
          {err}
          {budgetErrorHint(err) && (
            <div className="mt-1" style={{ color: "var(--warn)" }}>
              {budgetErrorHint(err)}
            </div>
          )}
        </div>
      )}

      {/* P2：只读诊断得到候选思路；用户思路也进入同一池。选点后才创建一思路一 Loop。 */}
      <div className="mx-4 mt-3 rounded-[10px] px-3 py-3" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <b className="text-[12px]">优化点搜索</b>
          <span className="chip !py-0 !text-[10px]">只读诊断</span>
          <span style={{ color: "var(--fg-mute)" }}>已有思路、已验证/已否决结论与 run 证据会回灌去重；不会改代码或跑真机。</span>
          {activeIdeaSearch && (
            <span className="chip chip-run ml-auto">
              搜索 #{activeIdeaSearch.id} 运行中
              {activeIdeaSearch.agent_job_id ? ` · job #${activeIdeaSearch.agent_job_id}` : ""}
            </span>
          )}
          {activeIdeaSearch && activeIdeaSearch.agent_job_id != null && (
            <button
              className="btn btn-sm !py-0 !text-[10px]"
              onClick={() => void cancelIdeaSearch()}
              disabled={ideaCancelBusy}
              title="终止诊断 Agent；已产生的思路会保留，取消后继续等后端收尾"
            >
              {ideaCancelBusy ? "取消中…" : "取消诊断"}
            </button>
          )}
        </div>
        <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
          <input
            className="input"
            value={ideaExtra}
            onChange={(e) => setIdeaExtra(e.target.value)}
            placeholder="可选：本次重点，例如只找 64K 小 shape 的访存/并行度问题"
            disabled={ideaBusy || ideaRunning}
            maxLength={4000}
          />
          <button className="btn btn-primary btn-sm" onClick={() => void startIdeaSearch()} disabled={ideaBusy || ideaRunning}>
            {ideaRunning ? "诊断中…" : "搜索优化点"}
          </button>
        </div>
        <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
          <input
            className="input"
            value={newIdea}
            onChange={(e) => setNewIdea(e.target.value)}
            placeholder="用户补充思路：输入一句话加入候选池"
            disabled={ideaBusy}
            onKeyDown={(e) => {
              if (e.key === "Enter") void addIdea();
            }}
          />
          <button className="btn btn-sm" onClick={() => void addIdea()} disabled={ideaBusy || !newIdea.trim()}>
            加入思路池
          </button>
        </div>

        {lastFinishedSearch?.summary && (
          <div className="mt-2 text-[10px]" style={{ color: lastFinishedSearch.status === "ok" ? "var(--fg-mute)" : "var(--bad)" }}>
            最近搜索 #{lastFinishedSearch.id}：{lastFinishedSearch.summary}
            {lastFinishedSearch.status === "ok" &&
              `（新增 ${lastFinishedSearch.result.new ?? 0} / 重复 ${lastFinishedSearch.result.duplicates ?? 0} / 更新 ${lastFinishedSearch.result.updated ?? 0}）`}
          </div>
        )}

        <div className="mt-2 grid gap-2">
          {ideas === null && <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>读取思路池…</div>}
          {ideas?.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>还没有候选思路，可搜索或手工补充。</div>}
          {ideas?.map((idea) => (
            <div key={idea.id} className="rounded-[8px] px-3 py-2 text-[11px]" style={{ background: "var(--panel-3)", border: "1px solid var(--border)" }}>
              <div className="flex flex-wrap items-center gap-2">
                <b>#{idea.id} {idea.title}</b>
                <span className="chip !py-0 !text-[10px]">
                  {idea.source === "conversation" ? "对话识别" : idea.source === "user" ? "用户" : "Agent"}
                </span>
                {conversationIdForIdea(idea) != null && (
                  <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    会话 #{conversationIdForIdea(idea)}
                  </span>
                )}
                <span className={`chip !py-0 !text-[10px] ${idea.status === "rejected" ? "chip-bad" : idea.loop_id ? "chip-run" : ""}`}>
                  {idea.status}
                </span>
                {idea.confidence && <span style={{ color: "var(--fg-mute)" }}>置信 {idea.confidence}</span>}
                {idea.effort && <span style={{ color: "var(--fg-mute)" }}>工作量 {idea.effort}</span>}
                {idea.loop_id != null ? (
                  <span className="mono ml-auto" style={{ color: "var(--ok)" }}>已建 Loop #{idea.loop_id}</span>
                ) : idea.status !== "rejected" ? (
                  <div className="ml-auto flex gap-2">
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => setIdeaDialog(idea)}
                      disabled={ideaBusy || verifying || agentRunning || anyAutoRunRunning}
                      title={
                        anyAutoRunRunning
                          ? autoRunBusyHint
                          : "先确认/编辑本轮「需要优化的点」与约束，再建独立 Loop（一思路一 Loop）"
                      }
                    >
                      按此思路建轮次
                    </button>
                    <button className="btn btn-sm" onClick={() => void rejectIdea(idea)} disabled={ideaBusy}>不采用</button>
                  </div>
                ) : null}
              </div>
              {(idea.hypothesis || idea.expected || idea.risk) && (
                <div className="mt-1 leading-relaxed" style={{ color: "var(--fg-dim)" }}>
                  {idea.hypothesis && <div><b>假设</b>：{idea.hypothesis}</div>}
                  {idea.expected && <div><b>预期</b>：{idea.expected}</div>}
                  {idea.risk && <div><b>风险</b>：{idea.risk}</div>}
                  {idea.change_scope && <div className="mono"><b>范围</b>：{idea.change_scope}</div>}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {rounds.length === 0 && (
        <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-dim)" }}>
          还没有优化轮次。建一轮：选目标类型/用例与约束 → 先采基线（可复用刚跑过的 run）→ 让 agent 改代码 →
          跑候选验证 → 采纳成版本或否决。
        </div>
      )}

      {rounds.map((loop) => {
        const obj = loop.objective ?? {};
        const typeKey = obj.type ?? "perf";
        const typeMeta = TYPE_META[typeKey] ?? { label: typeKey, color: "var(--fg-dim)" };
        const focus = (obj.focus ?? "").trim();
        // text 维度（思路轮次）：目标由「优化点」描述，不指定目标用例
        const isTextDimension = obj.target_dimension === "text" || focus.length > 0;
        const dimension = obj.target_dimension === "all" ? "all" : "cases";
        const dimensionLabel = isTextDimension ? "文字优化点" : dimension === "all" ? "全部用例" : "指定用例";
        const targetCases = obj.target_cases ?? [];
        const sides = sidesOf(runsByLoop.get(loop.id) ?? []);
        const beforeMissing = missingActions(sides.before);
        const afterMissing = missingActions(sides.after);
        const beforeReady = beforeMissing.length === 0;
        const afterReady = afterMissing.length === 0;
        const closed = loop.status === "closed";
        const conclusion = loop.conclusion ?? "";
        const stateKey = !closed ? "open" : conclusion.startsWith("accepted") ? "accepted" : conclusion.startsWith("rejected") ? "rejected" : "closed";
        const state = LOOP_STATE[stateKey];
        // accepted/rejected 已由上面的 LOOP_STATE chip 表达，phase chip 不再重复渲染
        const phaseMeta =
          loop.phase && loop.phase !== "accepted" && loop.phase !== "rejected" ? PHASE_META[loop.phase] : null;
        const baselineSp = latestOf(loop.id, "baseline");
        const candidateSp = latestOf(loop.id, "candidate");
        const versionSp = latestOf(loop.id, "accept");
        const rejectedSp = latestOf(loop.id, "rejected");
        const roundVerify = verify && verify.loopId === loop.id ? verify : null;
        const roundResult = verifyResult && verifyResult.loopId === loop.id ? verifyResult : null;
        const busy = busyLoop === loop.id;
        const ev = evidenceByLoop[loop.id] ?? null;
        // Step 2 自动验证链：最近一次 auto_run（有 key = 已读到，null = 后端明确没有；
        // 没 key 且 unknown = 读取失败，不能当「没有」；没 key 且未知 = 还在读（loading））。
        const roundAutoRunKnown = Object.prototype.hasOwnProperty.call(autoRunByLoop, loop.id);
        const roundAutoRunUnknown = autoRunUnknown[loop.id] ?? false;
        const roundAutoRun = autoRunByLoop[loop.id] ?? null;
        const roundAutoRunLoading = !roundAutoRunKnown && !roundAutoRunUnknown;
        const roundAutoRunRunning = roundAutoRun?.status === "running";
        // report 步结论/门禁（regressed 时 status=ok 但 error 有提示、gate 可能是 fail/warn）
        const autoReportStep = roundAutoRun?.steps.find((s) => s.step === "report") ?? null;
        const autoReportConclusion = autoReportStep?.conclusion ?? roundAutoRun?.conclusion ?? null;
        const autoReportGate = autoReportStep?.gate ?? null;
        const autoReportAbnormal =
          autoReportGate === "fail" ||
          autoReportGate === "warn" ||
          autoReportConclusion === "regressed";
        const autoReportColor =
          autoReportConclusion === "regressed" || autoReportGate === "fail"
            ? "var(--bad)"
            : autoReportAbnormal
              ? "var(--warn)"
              : conclusionChip(autoReportConclusion).color;
        // blocked + 远端清理未确认：重试前要先人工确认容器已清理（后端会再次 409 E_AUTO_CLEANUP）
        const autoRunCleanupUnconfirmed = Boolean(
          roundAutoRun?.status === "blocked" &&
            /E_AUTO_CLEANUP|清理未确认/.test(
              `${roundAutoRun.error ?? ""} ${roundAutoRun.steps.map((s) => s.error ?? "").join(" ")}`,
            ),
        );
        // 可（重）试自动链：待验证 = 首次跑；blocked = 取消/清理未确认后直接重试；
        // failed = 失败后重试（工作区已回退时后端 409 E_AUTO_NO_CANDIDATE，detail 原样展示）。
        const canStartAutoRun =
          !closed && loop.phase != null && AUTO_RUN_RETRY_PHASES.has(loop.phase);
        // 动作收尾可能把 phase 恢复为待验证；取消/失败历史仍应显示「重试」。
        const retryingAutoRun =
          loop.phase !== "awaiting_verification" ||
          ["cancelled", "failed", "blocked"].includes(roundAutoRun?.status ?? "");
        // 读不到状态 / 还没读到时不显示主按钮（未知时进度块给「重试读取」）
        const showAutoRunStart =
          canStartAutoRun && !roundAutoRunRunning && !roundAutoRunUnknown && !roundAutoRunLoading;
        // 进度块：有 auto_run 就展示（含 done/failed/blocked/cancelled 历史）；读失败 → 展示未知+重试读取；
        // 已读到没有且 phase 可重试 → 展示引导；还在读 → 先不渲染（不闪「还没跑过」）。按钮与块不重复铺引导。
        const showAutoRunBlock =
          Boolean(roundAutoRun) || roundAutoRunUnknown || (!roundAutoRunLoading && canStartAutoRun);
        // 尝试历史：最近 5 次（按 seq 倒序，最新在上）；仅有记录/错误时才渲染整块，
        // 加载中不单独撑起块（避免每张卡先闪一下「加载中」，已有的旧记录仍在时仍会显示加载态）。
        const attempts = attemptsByLoop[loop.id] ?? [];
        const attemptsBusy = attemptsLoading[loop.id] ?? false;
        const attemptsError = attemptsErr[loop.id] ?? null;
        const recentAttempts = [...attempts].sort((a, b) => b.seq - a.seq).slice(0, 5);
        const attemptsVisible = Boolean(attemptsError) || recentAttempts.length > 0;
        const audits = budgetAudits[loop.id] ?? [];
        // 报告归档（append-only）：最新在前；轮次卡只展示最近 5 条，完整列表由后端接口给。
        // 报告归档（append-only）：只对已关闭 / 有 candidate 存点的轮次渲染（新轮次必然 409，不显示空态）。
        // 默认只显示最近 5 条，可展开全部（后端最多 50）。
        const showArchiveBlock = closed || Boolean(candidateSp);
        const reportEntries = reportsByLoop[loop.id] ?? [];
        const reportsExpanded = expandReports[loop.id] ?? false;
        const shownReports = reportsExpanded ? reportEntries : reportEntries.slice(0, 5);
        const reportsBusy = reportsLoading[loop.id] ?? false;
        const reportsError = reportsErr[loop.id] ?? null;
        const roundArchiveMsg = archiveMsg[loop.id] ?? null;
        // closed 轮次不显示验证/改代码按钮；open 轮次都显示改代码区（基线未就绪时按钮禁用并提示）
        const showAgentControls = !closed;
        // 本轮有过自动优化（job 或证据）→ 提供「在对话中查看」入口（过程/摘要已写进会话）
        const hasOptimizeHistory = Boolean(
          (ev && ev.status !== "none") ||
            task.agent_jobs.some((j) => j.kind === "优化" && j.loop_id === loop.id),
        );
        const showCompare = beforeReady && afterReady;
        const compareKey = `${sides.before?.sha ?? ""}:${sides.after?.sha ?? ""}:${
          sides.after?.byAction.benchmark?.id ?? ""
        }`;
        // agent 在上一轮候选验证之后又改了代码：旧候选侧不再是「当前工作区」，提示重跑候选验证
        const agentChangedAfterCandidate = Boolean(
          !closed &&
            ev?.changed &&
            !ev?.reverted &&
            ev.after_sha256 &&
            sides.after &&
            ev.after_sha256 !== sides.after.sha,
        );

        return (
          <div key={loop.id} className="px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
            {/* 轮次标题 */}
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <b className="text-[12px]">{obj.label || loop.scope || `轮次 #${loop.id}`}</b>
              <span className="chip !py-0 !text-[10px]" style={{ color: typeMeta.color }}>
                {typeMeta.label}
              </span>
              <span className={state.cls}>{state.label}</span>
              {phaseMeta && (
                <span className={phaseMeta.cls} title={loop.phase_error || `phase=${loop.phase}（v${loop.state_version ?? 0}）`}>
                  {phaseMeta.label}
                </span>
              )}
              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                轮次 #{loop.id} · {when(loop.created_at)}
              </span>
            </div>

            {/* 目标：text 维度显示「优化点」，其余维度保持目标用例 */}
            <div className="mt-1 flex flex-wrap items-start gap-1.5 text-[11px]">
              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                目标
              </span>
              {isTextDimension ? (
                <>
                  <span className="chip !py-0 !text-[10px]">优化点</span>
                  {focus ? (
                    <FocusText text={focus} />
                  ) : (
                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      未指定
                    </span>
                  )}
                </>
              ) : dimension === "all" ? (
                <span className="chip !py-0 !text-[10px]">全部用例</span>
              ) : targetCases.length ? (
                <>
                  {targetCases.slice(0, 6).map((c) => (
                    <span key={c} className="chip !py-0 !text-[10px] mono" title={c}>
                      {c}
                    </span>
                  ))}
                  {targetCases.length > 6 && (
                    <span className="chip !py-0 !text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      +{targetCases.length - 6}
                    </span>
                  )}
                </>
              ) : (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  未指定
                </span>
              )}
              {obj.constraints?.allowed_paths?.length ? (
                <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  允许改动 {obj.constraints.allowed_paths.join("，")}
                </span>
              ) : null}
            </div>

            {/* 存点 + 两侧 run 就绪 */}
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px]">
              {baselineSp && (
                <span className="chip !py-0 !text-[10px]" title={`基线存档 #${baselineSp.id} · ${baselineSp.label}`}>
                  基线存点 #{baselineSp.id}
                </span>
              )}
              {candidateSp && (
                <span className="chip chip-run !py-0 !text-[10px]" title={`候选存档 #${candidateSp.id} · ${candidateSp.label}`}>
                  候选存点 #{candidateSp.id}
                </span>
              )}
              {versionSp && (
                <span className="chip chip-ok !py-0 !text-[10px]" title={`版本存档 #${versionSp.id} · ${versionSp.label}`}>
                  版本存点 #{versionSp.id}
                </span>
              )}
              {rejectedSp && (
                <span className="chip chip-bad !py-0 !text-[10px]" title={rejectedSp.note ?? undefined}>
                  否决存点 #{rejectedSp.id}
                </span>
              )}
              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                两侧 run
              </span>
              {(["test", "benchmark"] as const).map((action) => {
                const r = sides.before?.byAction[action] ?? null;
                return (
                  <span
                    key={`b-${action}`}
                    className={`chip !py-0 !text-[10px] ${r ? "chip-ok" : ""}`}
                    style={r ? undefined : { color: "var(--fg-mute)" }}
                    title={r ? `基线 ${action} run #${r.id}` : "基线还没有成功的 " + action}
                  >
                    基线 {action} {r ? `#${r.id}` : "未跑"}
                  </span>
                );
              })}
              {(["test", "benchmark"] as const).map((action) => {
                const r = sides.after?.byAction[action] ?? null;
                return (
                  <span
                    key={`a-${action}`}
                    className={`chip !py-0 !text-[10px] ${r ? "chip-ok" : ""}`}
                    style={r ? undefined : { color: "var(--fg-mute)" }}
                    title={r ? `候选 ${action} run #${r.id}` : "候选还没有成功的 " + action}
                  >
                    候选 {action} {r ? `#${r.id}` : "未跑"}
                  </span>
                );
              })}
              {audits.length > 0 && (
                <span
                  className={`chip !py-0 !text-[10px] ${audits[0].decision === "rejected" ? "chip-bad" : ""}`}
                  title={audits.slice(0, 8).map((a) => `${a.action}: ${a.note ?? a.decision}`).join("\n")}
                >
                  预算审计 {audits.filter((a) => a.decision === "allowed").length} 次允许
                  {audits.some((a) => a.decision === "rejected") ? " · 有拒绝" : ""}
                </span>
              )}
            </div>

            {/* 一键验证进度 / 结果 */}
            {roundVerify && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{ background: "var(--panel-2)", border: "1px solid var(--accent)" }}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="chip chip-run">{roundVerify.target === "baseline" ? "基线验证" : "候选验证"}</span>
                  {(["plan", "test", "benchmark"] as const).map((stage, i) => (
                    <span
                      key={stage}
                      className={`chip !py-0 !text-[10px] ${roundVerify.stage === stage ? "chip-run" : ""}`}
                      style={roundVerify.stage === stage ? undefined : { color: "var(--fg-mute)" }}
                    >
                      {i + 1}. {stage === "plan" ? "计划" : stage}
                    </span>
                  ))}
                  {roundVerify.jobId != null && (
                    <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      job #{roundVerify.jobId}
                    </span>
                  )}
                  <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    设备 {roundVerify.deviceId ?? "—"}（同一轮固定一张卡）
                  </span>
                  <span
                    className="mono text-[10px]"
                    style={{ color: "var(--fg-dim)" }}
                    title="本次验证已运行时长（包含远端同步/起容器时间）"
                  >
                    已运行 {fmtElapsed(nowTick - roundVerify.startedAt)}
                  </span>
                  <span className="chip chip-run !py-0 !text-[10px]">进行中</span>
                </div>
                <div className="mt-1 text-[10px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
                  {roundVerify.stage === "plan"
                    ? "正在计算执行计划（plan，不占设备）…"
                    : roundVerify.stage === "benchmark"
                      ? "benchmark 阶段独占设备（同一时刻只跑一个动作）；耗时期间日志可能长时间静止，用例输出通常批量返回。"
                      : "远端 test 容器内用例输出可能延迟/批量返回；同步代码 + 起容器可能要 1–2 分钟，日志静止不代表没在跑（看上面的已运行时长）。"}
                </div>
                <pre
                  className="mt-1 mb-0 max-h-[160px] overflow-auto rounded-[8px] p-2 text-[10px] leading-[1.6]"
                  style={{
                    background: "var(--panel-3)",
                    border: "1px solid var(--border)",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {logTail(roundVerify.log).trim()
                    ? logTail(roundVerify.log)
                    : "已起作业，等待远端日志…\n同步代码 / 起容器可能 1–2 分钟；远端输出延迟是常态，可看「已运行」确认作业还在跑。"}
                </pre>
              </div>
            )}

            {roundResult && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{
                  background: "var(--panel-2)",
                  border: `1px solid ${roundResult.ok ? "var(--ok)" : "var(--bad)"}`,
                }}
              >
                <div className="text-[11px]" style={{ color: roundResult.ok ? "var(--ok)" : "var(--bad)" }}>
                  {roundResult.ok ? "✓ " : "✘ "}
                  {roundResult.text}
                </div>
                {!roundResult.ok && budgetErrorHint(roundResult.text) && (
                  <div className="mt-1 text-[11px]" style={{ color: "var(--warn)" }}>
                    {budgetErrorHint(roundResult.text)}
                  </div>
                )}
                {!roundResult.ok && roundResult.log && (
                  <pre
                    className="mt-1 mb-0 max-h-[160px] overflow-auto rounded-[8px] p-2 text-[10px] leading-[1.6]"
                    style={{
                      background: "var(--panel-3)",
                      border: "1px solid var(--border)",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {logTail(roundResult.log)}
                  </pre>
                )}
              </div>
            )}

            {/* 操作 */}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {/* Step 2 主入口：phase 可（重）试（待验证 / 阻塞 / 失败）且本 loop 没有 running 自动链时，一键串跑全流程 */}
              {showAutoRunStart && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => void startAutoRun(loop)}
                  disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                  title={
                    anyAutoRunRunning
                      ? autoRunBusyHint
                      : loop.phase === "blocked" || roundAutoRun?.status === "cancelled"
                        ? "上次自动链被取消/远端清理未确认：工作区仍是候选，可直接重试；前置不满足时原样显示后端原因（E_AUTO_*）"
                        : loop.phase === "failed"
                          ? "上次自动链失败：工作区可能已回退到本轮改前，重试若提示 E_AUTO_NO_CANDIDATE 就需先重新让 agent 改代码"
                          : "后端自动串跑 build → smoke → test → benchmark → 报告（会占用工作区与设备，可取消）；前置不满足时直接显示后端原因（如预算不足/基线缺失）"
                  }
                >
                  {busy
                    ? "处理中…"
                    : retryingAutoRun
                      ? "重试自动验证链（build → smoke → test → benchmark → 报告）"
                      : "自动跑完验证（build → smoke → test → benchmark → 报告）"}
                </button>
              )}
              {!closed && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => openOptimizeChat(loop)}
                  disabled={verifying || agentRunning || busy || !beforeReady || anyAutoRunRunning}
                  title={
                    anyAutoRunRunning
                      ? autoRunBusyHint
                      : agentRunning
                        ? "自动优化正在「对话」分区进行：到对话窗口查看过程或停止"
                        : beforeReady
                          ? "跳到「对话」分区：输入参考指令后回车，让 Agent 按本轮目标改代码；过程与续聊都在对话里"
                          : "请先跑完基线 test + benchmark，再启动自动优化"
                  }
                >
                  {agentRunning && agentRunningLoopId === loop.id
                    ? "自动优化进行中（对话）"
                    : "自动优化（对话中改代码）"}
                </button>
              )}
              {!closed && !beforeReady && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => void runVerification(loop, "baseline")}
                  disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                  title={
                    anyAutoRunRunning
                      ? autoRunBusyHint
                      : "依次跑 test → benchmark，带本轮 loop_id 与设备 pin；复用已有 run"
                  }
                >
                  {verifying ? "运行中…" : `跑基线（${beforeMissing.join(" + ")}）`}
                </button>
              )}
              {!closed && beforeReady && !afterReady && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => void runVerification(loop, "candidate")}
                  disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                  title={
                    anyAutoRunRunning
                      ? autoRunBusyHint
                      : "agent 改完代码后跑：依次 test → benchmark，带本轮 loop_id 与设备 pin"
                  }
                >
                  {verifying ? "运行中…" : `跑候选验证（${afterMissing.join(" + ")}）`}
                </button>
              )}
              {!closed && afterReady && (
                <>
                  {agentChangedAfterCandidate && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => void runVerification(loop, "candidate", undefined, true)}
                      disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                      title={
                        anyAutoRunRunning
                          ? autoRunBusyHint
                          : "agent 已再次改代码（工作区 sha 与候选不同）：重跑 test + benchmark，生成新的候选"
                      }
                    >
                      再跑候选验证（重跑 test + benchmark）
                    </button>
                  )}
                  <button
                    className="btn btn-sm"
                    onClick={() => void acceptRound(loop)}
                    disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                    title={
                      anyAutoRunRunning
                        ? autoRunBusyHint
                        : "把最新候选存点升级为版本（cN），并关闭本轮"
                    }
                  >
                    {busy ? "处理中…" : "采纳为版本"}
                  </button>
                </>
              )}
              {/* 否决对所有未关闭轮次可见：基线未跑 / agent 失败 / 候选验证失败 / 预算耗尽都应能标记不采纳 */}
              {!closed && (
                <button
                  className="btn btn-sm"
                  onClick={() => void rejectRound(loop)}
                  disabled={verifying || busy || agentRunning || anyAutoRunRunning}
                  title={
                    anyAutoRunRunning
                      ? autoRunBusyHint
                      : "关闭本轮并标记否决（不要求已有候选）；确认时可选择同时把工作区回退到本轮改前"
                  }
                >
                  {busy ? "处理中…" : "否决"}
                </button>
              )}
              {hasOptimizeHistory && (
                <button
                  className="btn btn-sm"
                  onClick={() => void openLoopConversation(loop)}
                  disabled={busy}
                  title="跳到「对话」分区，查看本轮自动优化的会话（没有会话时直接打开对话入口）"
                >
                  在对话中查看
                </button>
              )}
              <button
                className="btn btn-sm"
                onClick={() => void exportReport(loop)}
                disabled={verifying || busy || agentRunning}
                title="即时计算并下载报告 JSON（不写归档目录；需要 manifest/sha256/raw 副本与可校验归档，用下方「报告归档」）"
              >
                {busy ? "处理中…" : "导出即时报告（不归档）"}
              </button>
              <button className="btn btn-sm" onClick={() => void refreshAll()} disabled={verifying || busy || agentRunning}>
                刷新
              </button>
              {roundAutoRunRunning && (
                <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                  自动验证链运行中（占工作区/设备）：跑基线/候选验证、采纳/否决、回退与自动优化已暂时禁用，可先取消或等它结束。
                </span>
              )}
              {agentRunning && (
                <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                  自动优化在「对话」分区进行：过程与续聊都在对话窗口，需要停止请用该处的「停止」。
                </span>
              )}
              {!closed && beforeReady && !afterReady && !agentRunning && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  自动优化只改算子代码（在对话窗口进行）；完成后请跑候选验证，再由你人工采纳或否决
                </span>
              )}
              {closed && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  {loop.conclusion?.startsWith("accepted")
                    ? "本轮已采纳（版本见「版本」分区）"
                    : loop.conclusion?.startsWith("rejected")
                      ? loop.conclusion
                      : "本轮已关闭"}
                </span>
              )}
            </div>

            {/* 优化前后对比（两侧 run 就绪后展示；原 PerfDiffPanel 的对比部分并入） */}
            {showCompare && <PerfCompare loopId={loop.id} refreshKey={compareKey} />}

            {/* 自动优化证据、diff 与回退 */}
            {(showAgentControls || Boolean(ev && ev.status !== "none")) && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="font-semibold">自动优化证据与回退</span>
                  {ev && ev.status !== "none" && (
                    <span
                      className={`chip ${ev.guard_reverted ? "chip-bad" : ""}`}
                      style={
                        ev.guard_reverted ? undefined : { color: ev.changed ? "var(--ok)" : "var(--fg-dim)" }
                      }
                    >
                      {ev.guard_reverted
                        ? `已自动回退${ev.reverted_clean ? "（= 改前）" : "（有残留）"}`
                        : ev.reverted
                          ? `已回退${ev.reverted_clean ? "（= 改前）" : "（有残留）"}`
                          : ev.status === "failed"
                            ? "失败"
                            : ev.changed
                              ? `改了 ${ev.changed_files.length} 个文件`
                              : "未产生改动"}
                    </span>
                  )}
                  {ev?.verification && (
                    <span className={`chip ${ev.verification === "pass" ? "chip-ok" : "chip-bad"}`}>
                      verifier {ev.verification}
                    </span>
                  )}
                  {ev?.job_id != null && (
                    <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      改代码 job #{ev.job_id}
                    </span>
                  )}
                  <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    本轮目标/约束/用户思路会注入 agent 提示；agent 只改代码，过程在「对话」分区
                  </span>
                </div>

                {ev?.guard_reverted && (
                  <div
                    className="mt-1.5 rounded-[8px] px-3 py-2 text-[11px] leading-relaxed"
                    style={{ border: "1px solid var(--bad)", color: "var(--bad)" }}
                  >
                    本轮改动未保留：因 <b>verifier 未明确 PASS</b> 或 <b>改动越过 allowed_paths</b>，后端已把工作区
                    <b>自动回退</b>到改前（不是「失败但改动还在」）。如需继续，请按新思路重新运行自动优化。
                    {ev.attempted_after_sha256 && (
                      <span className="mono block text-[10px]" style={{ color: "var(--fg-mute)" }}>
                        回退前尝试快照 {ev.attempted_after_sha256.slice(0, 12)}…
                      </span>
                    )}
                  </div>
                )}

                {showAgentControls && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-2">
                    <button
                      className="btn btn-sm"
                      onClick={() => openOptimizeChat(loop)}
                      disabled={verifying || agentRunning || busy || !beforeReady || anyAutoRunRunning}
                      title={
                        anyAutoRunRunning
                          ? autoRunBusyHint
                          : agentRunning
                            ? "自动优化正在「对话」分区进行：到对话窗口查看过程或停止"
                            : "跳到「对话」分区，按本轮 objective（类型/优化点或目标用例/约束/用户思路）让 agent 改代码"
                      }
                    >
                      {agentRunning && agentRunningLoopId === loop.id
                        ? "自动优化进行中（对话）"
                        : !ev || ev.status === "none"
                          ? "运行自动优化"
                          : "再次运行自动优化"}
                    </button>
                    <button
                      className="btn btn-sm"
                      onClick={() => void refreshEvidence(loop)}
                      disabled={verifying || agentRunning || busy}
                    >
                      刷新证据
                    </button>
                    {ev?.changed && !ev.reverted && (
                      <button
                        className="btn btn-sm"
                        onClick={() => void revertOptimize(loop)}
                        disabled={verifying || agentRunning || busy || anyAutoRunRunning}
                        title={anyAutoRunRunning ? autoRunBusyHint : "把工作区还原到这一轮改前"}
                      >
                        回退改动
                      </button>
                    )}
                    {ev?.diff && (
                      <button
                        className="btn btn-sm"
                        onClick={() => setShowAgentDiff((prev) => ({ ...prev, [loop.id]: !prev[loop.id] }))}
                      >
                        {showAgentDiff[loop.id] ? "隐藏 diff" : "看 diff"}
                      </button>
                    )}
                    {!beforeReady && (
                      <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                        先跑基线，再让 agent 改代码（否则改动会被算进基线）
                      </span>
                    )}
                  </div>
                )}

                {ev && ev.status !== "none" && (
                  <div className="mt-1 text-[11px] leading-relaxed" style={{ color: "var(--fg-dim)" }}>
                    <div className="mono" style={{ color: "var(--fg-mute)" }}>
                      快照 {String(ev.before_sha256 ?? "").slice(0, 12)}… → {String(ev.after_sha256 ?? "").slice(0, 12)}…
                      {ev.reference_loop_id != null && <> · 参考 Loop #{ev.reference_loop_id}</>}
                    </div>
                    {ev.agent_summary && (
                      <div className="mt-1">
                        <b>摘要</b>：{ev.agent_summary}
                      </div>
                    )}
                    {(ev.verification ||
                      agentText(ev.pipeline?.architect) ||
                      agentText(ev.pipeline?.developer) ||
                      agentText(ev.pipeline?.verifier)) && (
                      <div className="mt-1 rounded-[7px] px-2 py-1.5" style={{ background: "var(--panel-3)" }}>
                        {ev.verification && (
                          <div>
                            <b>verifier 结论</b>：
                            <span style={{ color: ev.verification === "pass" ? "var(--ok)" : "var(--bad)" }}>
                              {ev.verification}
                            </span>
                          </div>
                        )}
                        {agentText(ev.pipeline?.architect) && (
                          <div>
                            <b>architect</b>：{agentText(ev.pipeline?.architect)}
                          </div>
                        )}
                        {agentText(ev.pipeline?.developer) && (
                          <div>
                            <b>developer</b>：{agentText(ev.pipeline?.developer)}
                          </div>
                        )}
                        {agentText(ev.pipeline?.verifier) && (
                          <div>
                            <b>verifier 摘要</b>：{agentText(ev.pipeline?.verifier)}
                          </div>
                        )}
                      </div>
                    )}
                    {ev.error && (
                      <div className="mt-1" style={{ color: "var(--bad)" }}>
                        错误：{ev.error}
                      </div>
                    )}
                    {ev.changed_files.length > 0 && (
                      <div className="mono mt-1" style={{ color: "var(--fg-mute)" }}>
                        {ev.guard_reverted ? "回退前尝试改动：" : "改动："}
                        {ev.changed_files
                          .map(
                            (f) =>
                              `${f.path}(${f.status}${f.additions != null ? ` +${f.additions}/-${f.deletions}` : ""})`,
                          )
                          .join("；")}
                      </div>
                    )}
                  </div>
                )}

                {showAgentDiff[loop.id] && ev?.diff && (
                  <pre
                    className="mt-1 max-h-[360px] overflow-auto rounded-[8px] p-2 text-[11px]"
                    style={{
                      background: "var(--panel-3)",
                      border: "1px solid var(--border)",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {ev.diff}
                  </pre>
                )}
              </div>
            )}

            {/* 自动优化尝试历史（confirm→start 可审计）：紧凑一行一条，无记录不渲染 */}
            {attemptsVisible && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="font-semibold">尝试历史</span>
                  {attemptsBusy ? (
                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      加载中…
                    </span>
                  ) : attemptsError ? (
                    <span className="text-[10px]" style={{ color: "var(--bad)" }} title={attemptsError}>
                      {attemptsError}
                    </span>
                  ) : (
                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      最近 {recentAttempts.length} 条（接口最多 20 条）
                    </span>
                  )}
                </div>
                {recentAttempts.length > 0 && (
                  <div className="mt-1 grid gap-0.5">
                    {recentAttempts.map((a) => {
                      const meta = ATTEMPT_STATUS[a.status] ?? { cls: "chip", label: a.status };
                      return (
                        <div
                          key={a.id}
                          className="mono flex flex-wrap items-center gap-1.5 text-[10px]"
                          title={a.error ?? `attempt #${a.id} · agent job ${a.agent_job_id ?? "—"}`}
                        >
                          <span style={{ color: "var(--fg-mute)" }}>#{a.seq}</span>
                          <span className={`chip !py-0 !text-[10px] ${meta.cls}`}>{meta.label}</span>
                          <span style={{ color: "var(--fg-dim)" }}>
                            {a.candidate_sha256 ? a.candidate_sha256.slice(0, 8) : "—"}
                          </span>
                          <span style={{ color: "var(--fg-mute)" }}>
                            {a.status === "running"
                              ? a.started_at
                                ? `开始于 ${when(a.started_at)}`
                                : "进行中"
                              : when(a.ended_at ?? a.started_at)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Step 2 自动验证链（build → smoke → test → benchmark → 报告）：运行中展示进度并允许取消 */}
            {showAutoRunBlock && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{
                  background: "var(--panel-2)",
                  border: `1px solid ${
                    roundAutoRun?.status === "failed" || roundAutoRun?.status === "blocked"
                      ? "var(--bad)"
                      : roundAutoRunUnknown
                        ? "var(--warn)"
                        : roundAutoRun?.status === "running"
                          ? "var(--accent)"
                          : "var(--border)"
                  }`,
                }}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="font-semibold">自动验证链</span>
                  {roundAutoRunUnknown && (
                    <span className="chip chip-warn" title={AUTO_RUN_UNKNOWN_HINT}>
                      状态未知
                    </span>
                  )}
                  {roundAutoRun ? (
                    <>
                      <span className={AUTO_RUN_STATUS[roundAutoRun.status]?.cls ?? "chip"}>
                        {AUTO_RUN_STATUS[roundAutoRun.status]?.label ?? roundAutoRun.status}
                      </span>
                      {roundAutoRun.current_step && roundAutoRun.status === "running" && (
                        <span className="text-[10px]" style={{ color: "var(--fg-dim)" }}>
                          当前{" "}
                          {AUTO_STEP_LABEL[roundAutoRun.current_step as LoopAutoRunStep["step"]] ??
                            roundAutoRun.current_step}
                        </span>
                      )}
                      <span
                        className="mono text-[10px]"
                        style={{ color: "var(--fg-mute)" }}
                        title={[
                          `auto_run #${roundAutoRun.id}`,
                          roundAutoRun.current_action_job_id != null
                            ? `当前动作作业 #${roundAutoRun.current_action_job_id}`
                            : "",
                          roundAutoRun.attempt_id != null ? `attempt #${roundAutoRun.attempt_id}` : "",
                          roundAutoRun.baseline_sha256 ? `基线快照 ${roundAutoRun.baseline_sha256.slice(0, 12)}…` : "",
                          roundAutoRun.candidate_sha256 ? `候选快照 ${roundAutoRun.candidate_sha256.slice(0, 12)}…` : "",
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      >
                        #{roundAutoRun.id}
                        {roundAutoRun.attempt_id != null ? ` · attempt #${roundAutoRun.attempt_id}` : ""}
                        {roundAutoRun.started_at ? ` · ${when(roundAutoRun.started_at)}` : ""}
                      </span>
                      {roundAutoRun.current_action_job_id != null && roundAutoRun.status === "running" && (
                        <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          job #{roundAutoRun.current_action_job_id}
                        </span>
                      )}
                      {roundAutoRunUnknown && (
                        <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                          上次读到的状态，可能已变化
                        </span>
                      )}
                    </>
                  ) : roundAutoRunUnknown ? (
                    <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                      读取失败，无法确认是否有自动链在跑（不能当成「没有」）
                    </span>
                  ) : (
                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      {loop.phase === "awaiting_verification"
                        ? "agent 已改完代码（待验证）：点上方主按钮串跑全流程"
                        : "后端没有这条轮次的自动链记录：可直接点上方「重试自动验证链」"}
                    </span>
                  )}
                  {roundAutoRunUnknown && (
                    <button
                      className="btn btn-sm ml-auto !py-0 !text-[10px]"
                      onClick={() => void readAutoRun(loop.id)}
                      title="重新读取该轮次的自动链状态（读取失败不能当成「没有」）"
                    >
                      重试读取
                    </button>
                  )}
                  {roundAutoRun?.status === "running" && (
                    <button
                      className={`btn btn-sm !py-0 !text-[10px] ${roundAutoRunUnknown ? "" : "ml-auto"}`}
                      onClick={() => void cancelAutoRun(loop)}
                      disabled={autoRunCancelling === loop.id}
                      title="终止当前动作并清理远端容器；工作区保留现状（不回退）"
                    >
                      {autoRunCancelling === loop.id ? "取消中…" : "取消自动链"}
                    </button>
                  )}
                </div>

                {(roundAutoRun || !roundAutoRunUnknown) && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    {AUTO_STEP_ORDER.map((stepKey, i) => {
                      const st = roundAutoRun?.steps.find((s) => s.step === stepKey) ?? null;
                      const stMeta = AUTO_STEP_STATUS[st?.status ?? "pending"] ?? AUTO_STEP_STATUS.pending;
                      const title = [
                        `第 ${i + 1} 步 ${AUTO_STEP_LABEL[stepKey]} · ${stMeta.label}`,
                        st?.job_id != null ? `作业 #${st.job_id}` : "",
                        st?.run_id != null ? `run #${st.run_id}` : "",
                        st?.gate ? `门禁 ${st.gate}` : "",
                        st?.conclusion ? `结论 ${st.conclusion}` : "",
                        st?.cleanup
                          ? `远端清理 ${
                              st.cleanup.ok === true ? "ok" : st.cleanup.ok === false ? "未确认" : "—"
                            }${st.cleanup.why ? `：${st.cleanup.why}` : ""}`
                          : "",
                        st?.started_at ? `开始 ${when(st.started_at)}` : "",
                        st?.ended_at ? `结束 ${when(st.ended_at)}` : "",
                        st?.error ? `错误：${st.error}` : "",
                      ]
                        .filter(Boolean)
                        .join(" · ");
                      return (
                        <span key={stepKey} className={stMeta.cls} title={title}>
                          {i + 1}. {AUTO_STEP_LABEL[stepKey]}
                          {roundAutoRun ? ` · ${stMeta.label}` : ""}
                          {st?.gate === "fail" ? " · 门禁 fail" : ""}
                        </span>
                      );
                    })}
                    {roundAutoRun && Object.keys(roundAutoRun.budget ?? {}).length > 0 && (
                      <span
                        className="mono text-[10px]"
                        style={{ color: "var(--fg-mute)" }}
                        title="启动本次自动链时的预算快照（当时已用/上限）；本次执行消耗见预算审计，不足时后端拒绝启动"
                      >
                        起链时预算{" "}
                        {(["build", "smoke", "test", "benchmark"] as const)
                          .map((a) => {
                            const b = roundAutoRun.budget?.[a];
                            return b ? `${a} ${b.used}/${b.limit}` : null;
                          })
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    )}
                  </div>
                )}

                {/* report 结论（regressed/非 pass 单独一行，避免 done 文案误导成「可采纳」） */}
                {roundAutoRun && (autoReportConclusion || autoReportGate || autoReportStep?.error) && (
                  <div
                    className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px]"
                    style={{ color: autoReportColor }}
                    title={autoReportStep?.error ?? undefined}
                  >
                    <b>报告结论</b>
                    <span>{autoReportConclusion ? conclusionChip(autoReportConclusion).label : "未给出"}</span>
                    {autoReportGate && <span>· 门禁 {autoReportGate}</span>}
                    {autoReportStep?.error && <span>· {autoReportStep.error}</span>}
                  </div>
                )}

                {roundAutoRun &&
                  roundAutoRun.steps
                    .filter((s) => s.status === "failed" && (s.error || s.gate === "fail"))
                    .map((s) => (
                      <div key={`auto-run-err-${s.step}`} className="mt-1 text-[10px]" style={{ color: "var(--bad)" }}>
                        {AUTO_STEP_LABEL[s.step]} 失败{s.gate === "fail" ? "（门禁 fail）" : ""}
                        {s.error ? `：${s.error}` : ""}
                        {s.cleanup && s.cleanup.ok === false
                          ? `（远端清理未确认${s.cleanup.why ? `：${s.cleanup.why}` : ""}）`
                          : ""}
                      </div>
                    ))}
                {roundAutoRun?.error && (
                  <div
                    className="mt-1 text-[10px]"
                    style={{ color: roundAutoRun.status === "cancelled" ? "var(--fg-mute)" : "var(--bad)" }}
                  >
                    {roundAutoRun.status === "cancelled" ? roundAutoRun.error : `错误：${roundAutoRun.error}`}
                  </div>
                )}

                {roundAutoRun?.status === "done" && !closed && (
                  <div
                    className="mt-1 text-[10px]"
                    style={{ color: autoReportAbnormal ? "var(--warn)" : "var(--ok)" }}
                  >
                    动作门禁已 pass、报告已落盘
                    {autoReportConclusion ? `（结论：${conclusionChip(autoReportConclusion).label}）` : ""}；
                    采纳仍由 G7 判定（性能回退/精度退化会被拦）。
                  </div>
                )}
                {roundAutoRun?.status === "failed" && !closed && (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-dim)" }}>
                    {canStartAutoRun ? (
                      <>
                        自动链失败：失败步骤与原因见上。按后端失败矩阵，动作失败会回退工作区到本轮改前（报告落盘失败不回退）；
                        可直接点「重试自动验证链」——若提示 E_AUTO_NO_CANDIDATE（候选已被回退），需先重新让 agent 改代码。
                      </>
                    ) : (
                      <>报告落盘失败（轮次已在「待采纳」，动作证据仍有效）：自动链不提供重试；可人工处理报告后再采纳/否决。</>
                    )}
                  </div>
                )}
                {(roundAutoRun?.status === "blocked" || roundAutoRun?.status === "cancelled") && !closed && (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-dim)" }}>
                    {roundAutoRun.status === "cancelled"
                      ? "自动链已取消"
                      : autoRunCleanupUnconfirmed
                        ? "自动链被阻塞：远端容器清理未确认"
                        : "自动链被阻塞（如进程重启）"}
                    ：工作区保留现状（不回退，需要回退请用「否决」时的回退选项）；
                    {autoRunCleanupUnconfirmed
                      ? "请先在执行机上确认容器已清理，再点「重试自动验证链」（否则会再次 409 E_AUTO_CLEANUP）。"
                      : "可直接点「重试自动验证链」。"}
                  </div>
                )}
                {roundAutoRun?.status === "running" && (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    后端正在串跑动作作业（进度每 2s 刷新）：动作门禁不是 pass 就停链并给出原因；可随时取消。
                  </div>
                )}
                {!roundAutoRun && !roundAutoRunUnknown && (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    {loop.phase === "awaiting_verification"
                      ? "还没跑过自动链：一键串跑 build → smoke → test → benchmark → 报告；失败会显示原因，运行中可取消。"
                      : "后端没有这条轮次的自动链记录：可直接点「重试自动验证链」；若提示 E_AUTO_NO_CANDIDATE，说明候选已被回退，需先重新让 agent 改代码。"}
                  </div>
                )}
              </div>
            )}

            {/* 报告归档（Step 3，append-only）：只对已关闭 / 有 candidate 存点的轮次渲染。
                manifest + sha256 + raw 副本，可独立目录校验（不宣称复现已跑通）。 */}
            {showArchiveBlock && (
              <div
                className="mt-2 rounded-[9px] px-3 py-2"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="font-semibold">报告归档</span>
                  <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    manifest + sha256 + raw 副本，可独立目录校验
                  </span>
                  <button
                    className="btn btn-sm ml-auto"
                    onClick={() => void archiveReport(loop)}
                    disabled={archiveBusyLoop === loop.id || verifying || busy || agentRunning || anyAutoRunRunning}
                    title={
                      anyAutoRunRunning
                        ? autoRunBusyHint
                        : "生成并归档报告：独立目录写 report.json + manifest.json（文件 sha256）+ raw/ 原始产物副本；缺 after benchmark（先跑候选 benchmark）或轮次未关闭时后端 409，原因显示在下方"
                    }
                  >
                    {archiveBusyLoop === loop.id ? "归档中…" : "生成并归档报告"}
                  </button>
                  <button
                    className="btn btn-sm"
                    onClick={() => void loadReports(loop.id)}
                    disabled={reportsBusy}
                    title="重拉该轮次归档列表（append-only，只会追加）"
                  >
                    {reportsBusy ? "刷新中…" : "刷新列表"}
                  </button>
                </div>

                {roundArchiveMsg && (
                  <div
                    className="mt-1 rounded-[7px] px-2 py-1.5 text-[10px] leading-relaxed"
                    style={{
                      border: `1px solid ${roundArchiveMsg.ok ? "var(--ok)" : "var(--bad)"}`,
                      color: roundArchiveMsg.ok ? "var(--ok)" : "var(--bad)",
                    }}
                  >
                    {roundArchiveMsg.text}
                  </div>
                )}
                {reportsError && (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--bad)" }} title={reportsError}>
                    归档列表读取失败：{reportsError}
                  </div>
                )}

                {reportsBusy && reportEntries.length === 0 ? (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    加载中…
                  </div>
                ) : reportEntries.length === 0 && !reportsError ? (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    还没有归档报告（点上方生成；归档为 append-only，不会被重跑覆盖）
                  </div>
                ) : reportsError && reportEntries.length === 0 ? (
                  <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    归档列表读取失败，暂时不知道有没有归档（可点「刷新列表」重试）
                  </div>
                ) : (
                  <>
                    <div className="mt-1 grid gap-1">
                      {shownReports.map((rep) => {
                        // 防御：后端字段缺失时也不崩（sha 只用于展示/文件名/缓存 key）
                        const sha = String(rep.report_sha256 ?? "");
                        const key = reportKey(loop.id, sha);
                        const ver = verifiedByKey[key] ?? null;
                        const verErr = verifyErrByKey[key] ?? null;
                        const detail = reportDetailByKey[key] ?? null;
                        const detailErr = reportDetailErrByKey[key] ?? null;
                        const dlBusy = downloadBusyKey === key;
                        const dlMsg = downloadMsgByKey[key] ?? null;
                        const vChip = verdictChip(rep.verdict);
                        const cChip = conclusionChip(rep.conclusion);
                        // 行 chip：深度校验结果优先，没有则用列表返回的快速校验标记
                        const deepValid = ver ? Boolean(ver.valid) : null;
                        const isValid = deepValid ?? Boolean(rep.valid);
                        const verFiles = ver && Array.isArray(ver.files) ? ver.files : [];
                        const verRaw = ver && Array.isArray(ver.raw) ? ver.raw : [];
                        const verErrors = ver && Array.isArray(ver.errors) ? ver.errors : [];
                        const firstBadFile =
                          ver && !ver.valid
                            ? (verFiles.find((f) => !f.ok) ?? verRaw.find((r) => !r.ok) ?? null)
                            : null;
                        const mFiles =
                          detail && Array.isArray(detail.manifest?.files) ? detail.manifest.files : [];
                        const mRaw = detail && Array.isArray(detail.manifest?.raw) ? detail.manifest.raw : [];
                        return (
                          <div
                            key={sha || `seq-${rep.seq}`}
                            className="rounded-[7px] px-2 py-1.5"
                            style={{ background: "var(--panel-3)", border: "1px solid var(--border)" }}
                          >
                            {/* 元信息一行 */}
                            <div className="mono flex flex-wrap items-center gap-1.5 text-[10px]">
                              <span style={{ color: "var(--fg-mute)" }}>#{rep.seq}</span>
                              <span title={sha} style={{ color: "var(--fg-dim)" }}>
                                {shortSha(sha)}
                              </span>
                              <span style={{ color: "var(--fg-mute)" }}>{when(rep.generated_at)}</span>
                              {vChip && (
                                <span className={`chip !py-0 !text-[10px] ${vChip.cls}`}>{vChip.label}</span>
                              )}
                              <span className="chip !py-0 !text-[10px]" style={{ color: cChip.color }}>
                                {cChip.label}
                              </span>
                              <span
                                className={`chip !py-0 !text-[10px] ${isValid ? "chip-ok" : "chip-bad"}`}
                                title={
                                  ver
                                    ? `深度校验${ver.valid ? "通过" : "未通过"}（逐文件 sha256 + raw 副本）；校验于 ${when(ver.checked_at)}`
                                    : "快速校验（列表返回）：report.json sha256 与 manifest 一致 + 文件存在；逐文件 sha256 请点「校验」"
                                }
                              >
                                {isValid ? "✓ 有效" : "✗ 失效"}
                              </span>
                              <span style={{ color: "var(--fg-mute)" }} title={String(rep.dir ?? "")}>
                                {rep.raw_count} raw · {size(rep.size_bytes)}
                              </span>
                            </div>

                            {/* 操作按钮一行 */}
                            <div className="mt-1 flex flex-wrap items-center gap-1.5">
                              <button
                                className="btn btn-sm !py-0 !text-[10px]"
                                onClick={() => void verifyReport(loop, sha)}
                                disabled={verifyBusyKey === key}
                                title="深度校验：逐文件 sha256 + raw 副本是否齐全/一致（结果就地展示，行 chip 以它为准）"
                              >
                                {verifyBusyKey === key ? "校验中…" : "校验"}
                              </button>
                              <button
                                className="btn btn-sm !py-0 !text-[10px]"
                                onClick={() => void downloadArchivedReport(loop, sha)}
                                disabled={dlBusy}
                                title="下载归档里的原始 report.json 字节（sha256 与 report_sha256 一致；不重新序列化）"
                              >
                                {dlBusy ? "下载中…" : "下载"}
                              </button>
                              <button
                                className="btn btn-sm !py-0 !text-[10px]"
                                onClick={() => void toggleManifest(loop, sha)}
                                disabled={manifestBusyKey === key}
                                title="展开/收起 manifest（files/raw 数量与前几个 path）"
                              >
                                {manifestBusyKey === key
                                  ? "读取中…"
                                  : manifestOpen[key]
                                    ? "收起 manifest"
                                    : "查看 manifest"}
                              </button>
                              {dlMsg && (
                                <span
                                  className="text-[10px]"
                                  style={{ color: dlMsg.ok ? "var(--ok)" : "var(--bad)" }}
                                  title={dlMsg.text}
                                >
                                  {dlMsg.text}
                                </span>
                              )}
                            </div>

                            {verErr && (
                              <div className="mt-1 text-[10px]" style={{ color: "var(--bad)" }} title={verErr}>
                                校验请求失败：{verErr}
                              </div>
                            )}
                            {ver && (
                              <div
                                className="mt-1 rounded-[6px] px-2 py-1 text-[10px] leading-relaxed"
                                style={{ border: `1px solid ${ver.valid ? "var(--ok)" : "var(--bad)"}` }}
                              >
                                {ver.valid ? (
                                  <span style={{ color: "var(--ok)" }}>
                                    校验通过（{verFiles.length} 个文件 + {verRaw.length} 个 raw 副本）
                                  </span>
                                ) : (
                                  <div style={{ color: "var(--bad)" }}>
                                    完整性校验未通过：
                                    {verErrors.length > 0 ? (
                                      <ul className="m-0 mt-0.5 list-disc pl-4">
                                        {verErrors.slice(0, 3).map((msg, i) => (
                                          <li key={i}>{msg}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <span> 后端未给出具体错误</span>
                                    )}
                                    {firstBadFile && (
                                      <div className="mono mt-0.5">首个失败文件：{firstBadFile.path}</div>
                                    )}
                                  </div>
                                )}
                                <span style={{ color: "var(--fg-mute)" }}>· 校验于 {when(ver.checked_at)}</span>
                              </div>
                            )}
                            {manifestOpen[key] && (
                              <div
                                className="mt-1 rounded-[6px] px-2 py-1 text-[10px] leading-relaxed"
                                style={{ border: "1px solid var(--border)", background: "var(--panel-2)" }}
                              >
                                {detailErr ? (
                                  <span style={{ color: "var(--bad)" }}>读取 manifest 失败：{detailErr}</span>
                                ) : !detail ? (
                                  <span style={{ color: "var(--fg-mute)" }}>读取中…</span>
                                ) : (
                                  <>
                                    <div className="mono" style={{ color: "var(--fg-dim)" }}>
                                      files {mFiles.length} 个 · raw {mRaw.length} 个 ·{" "}
                                      {detail.valid ? "sha256 一致" : "sha256 不一致"}
                                    </div>
                                    <div className="mono mt-0.5" style={{ color: "var(--fg-mute)" }}>
                                      files：
                                      {mFiles
                                        .slice(0, 3)
                                        .map((f) => f.path)
                                        .join("，") || "—"}
                                      {mFiles.length > 3 ? ` …+${mFiles.length - 3}` : ""}
                                    </div>
                                    <div className="mono mt-0.5" style={{ color: "var(--fg-mute)" }}>
                                      raw：
                                      {mRaw
                                        .slice(0, 3)
                                        .map((raw) => {
                                          if (raw.copied) {
                                            const where = raw.copy || raw.path || "";
                                            const src = raw.path && raw.copy ? `（源 ${raw.path}）` : "";
                                            return `${raw.kind}：归档内 ${where}${src}`;
                                          }
                                          return `${raw.kind}：未复制 ${raw.path || "—"}${
                                            raw.why ? `（${raw.why}）` : ""
                                          }`;
                                        })
                                        .join("，") || "—"}
                                      {mRaw.length > 3 ? ` …+${mRaw.length - 3}` : ""}
                                    </div>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    <div
                      className="mt-1 flex flex-wrap items-center gap-2 text-[10px]"
                      style={{ color: "var(--fg-mute)" }}
                    >
                      <span>
                        共 {reportEntries.length} 条
                        {reportEntries.length > 5 && !reportsExpanded ? "（仅显示最近 5）" : ""}
                        {reportEntries.length >= 50 ? "（接口最多 50 条）" : ""}
                      </span>
                      {reportEntries.length > 5 && (
                        <button
                          className="btn btn-sm !py-0 !text-[10px]"
                          onClick={() =>
                            setExpandReports((prev) => ({ ...prev, [loop.id]: !reportsExpanded }))
                          }
                        >
                          {reportsExpanded ? "收起（只看最近 5）" : `展开全部 ${reportEntries.length} 条`}
                        </button>
                      )}
                    </div>
                  </>
                )}

                <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  复现脚本 <span className="mono">scripts/repro_report.py &lt;报告目录&gt;</span>
                  （只校验/打印命令，真机重跑需人工）
                </div>
              </div>
            )}

            {/* 目标与约束详情 */}
            <details className="mt-1.5">
              <summary className="cursor-pointer text-[10px]" style={{ color: "var(--fg-mute)" }}>
                目标与约束详情
              </summary>
              <div className="mt-1 grid gap-1 text-[11px] leading-relaxed md:grid-cols-2" style={{ color: "var(--fg-dim)" }}>
                <div>
                  <b>类型</b>：{typeMeta.label} · <b>维度</b>：{dimensionLabel}
                </div>
                <div>
                  <b>约束</b>：{obj.constraints?.precision_non_regression ? "精度不退化" : "精度不保证"} ·{" "}
                  {obj.constraints?.perf_non_regression ? "性能不退化" : "性能不保证"}
                </div>
                <div>
                  <b>预算</b>：判断 {obj.budget?.diagnose_rounds ?? "—"} 轮 · 验证 {obj.budget?.verify_rounds ?? "—"}{" "}
                  轮（提示值，非硬限制）· 每动作 {obj.budget?.runs_per_action ?? "—"} 次（后端硬限制并留审计）
                </div>
                {isTextDimension && (
                  <div className="md:col-span-2" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    <b>优化点</b>：{focus || "未指定"}
                  </div>
                )}
                <div className="mono break-all">
                  <b>目标快照</b>：benchmark #{obj.targets_snapshot?.benchmark_run_id ?? "—"} · test #
                  {obj.targets_snapshot?.test_run_id ?? "—"} · 主用例 {obj.targets_snapshot?.primary_case ?? "—"} ·
                  用例集 {obj.targets_snapshot?.suite_ref ?? "—"}
                </div>
                {loop.hypothesis && (
                  <div className="md:col-span-2">
                    <b>假设</b>：{loop.hypothesis}
                  </div>
                )}
                {loop.judge_condition && (
                  <div className="md:col-span-2">
                    <b>判定条件</b>：{loop.judge_condition}
                  </div>
                )}
                {obj.user_text && (
                  <div className="md:col-span-2" style={{ color: "var(--warn)" }}>
                    <b>用户补充思路/约束</b>：{obj.user_text}
                  </div>
                )}
              </div>
            </details>
          </div>
        );
      })}

      <div className="px-4 py-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
        每轮流程：跑基线 → 让 agent 按本轮目标改代码 → 跑候选验证 → 看前后对比 → 采纳/否决。自动验证依次执行
        编译、冒烟、正确性、基准与对比报告，使用同一张卡，失败即停；验证通过后保留候选存点。
        {oldLoops.length > 0 && ` 另有 ${oldLoops.length} 个旧 Loop，已在下方提供只读入口。`}
      </div>

      {oldLoops.length > 0 && (
        <details className="mx-4 mb-3 rounded-[8px] px-3 py-2" style={{ border: "1px solid var(--border)" }}>
          <summary className="cursor-pointer text-[11px]">
            旧版非 objective Loop（只读，{oldLoops.length} 个；已不能自动优化，请新建优化轮次）
          </summary>
          <div className="mt-2 grid gap-2">
            {oldLoops.map((loop) => {
              const runs = runsByLoop.get(loop.id) ?? [];
              return (
                <div key={loop.id} className="rounded-[7px] px-3 py-2 text-[11px]" style={{ background: "var(--panel-2)" }}>
                  <div className="flex flex-wrap gap-2"><b>Loop #{loop.id}</b><span className="chip !py-0 !text-[10px]">{loop.kind}</span><span>{loop.status}</span></div>
                  <div style={{ color: "var(--fg-dim)" }}>{loop.scope || "无范围说明"}{loop.hypothesis ? ` · ${loop.hypothesis}` : ""}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {runs.length ? runs.map((r) => <span key={r.id} className="chip !py-0 !text-[10px]">{r.action} #{r.id} · {r.status}</span>) : <span style={{ color: "var(--fg-mute)" }}>无 run</span>}
                  </div>
                  {loop.conclusion && <div className="mt-1" style={{ color: "var(--fg-mute)" }}>结论：{loop.conclusion}</div>}
                </div>
              );
            })}
          </div>
        </details>
      )}

      {formOpen &&
        createPortal(
          <RoundForm
            task={task}
            targets={targets}
            targetsErr={targetsErr}
            onRetryTargets={() => void openForm()}
            onClose={() => setFormOpen(false)}
            onCreated={(res) => void handleCreated(res)}
          />,
          document.body,
        )}
      {ideaDialog &&
        createPortal(
          <IdeaRoundDialog
            idea={ideaDialog}
            busy={ideaBusy}
            onClose={() => setIdeaDialog(null)}
            onConfirm={(objective, attachBaseline) => createIdeaRound(ideaDialog, objective, attachBaseline)}
          />,
          document.body,
        )}
    </section>
  );
}

/**
 * 文字优化点展示（轮次卡「目标」区）：保留换行；过长默认折叠，可展开/收起。
 */
function FocusText({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > 180 || text.split("\n").length > 3;
  const shown = long && !open ? `${text.slice(0, 180)}…` : text;
  return (
    <span className="flex min-w-[220px] flex-1 flex-col gap-0.5">
      <span style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", color: "var(--fg-dim)" }}>{shown}</span>
      {long && (
        <button
          type="button"
          className="self-start border-0 bg-transparent p-0 text-[10px] underline"
          style={{ color: "var(--accent)" }}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "收起" : `展开全部（${text.length} 字）`}
        </button>
      )}
    </span>
  );
}

/**
 * 「按此思路建轮次」确认/编辑浮层（P2 二次修订）：
 * 思路摘要只读；「需要优化的点」可编辑并作为 text objective 的 focus 注入 Agent 提示；
 * 约束 / 每动作执行次数 / 复用基线 run 在这里一次定好（不再单击直接建轮次）。
 */
function IdeaRoundDialog({
  idea,
  busy,
  onClose,
  onConfirm,
}: {
  idea: OptimizationIdea;
  busy: boolean;
  onClose: () => void;
  onConfirm: (objective: OptimizationObjective, attachBaseline: boolean) => Promise<void>;
}) {
  const [focus, setFocus] = useState(() => defaultFocusForIdea(idea));
  const [perfNonRegression, setPerfNonRegression] = useState(false);
  const [runsPerAction, setRunsPerAction] = useState("3");
  const [attachBaseline, setAttachBaseline] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const focusRef = useRef<HTMLTextAreaElement | null>(null);
  // 同步防双提交：disabled 依赖 state，极快双击时第二次点击可能还在同一帧
  const submittingRef = useRef(false);
  const closed = submitting || busy;

  // 焦点给「需要优化的点」（可直接改预填内容）
  useEffect(() => {
    focusRef.current?.focus();
  }, []);

  // Escape 关闭（提交/父级 busy 时不响应；遮罩点击同样受 closed 保护）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !closed) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closed, onClose]);

  const focusText = focus.trim();
  const canSubmit = !closed && focusText.length > 0;

  const submit = async () => {
    if (!canSubmit || submittingRef.current) return;
    submittingRef.current = true;
    setSubmitting(true);
    setErr(null);
    const runs = Math.min(10, Math.max(1, Number.parseInt(runsPerAction, 10) || 3));
    try {
      await onConfirm(
        {
          type: "perf",
          target_dimension: "text",
          focus: focusText,
          constraints: { precision_non_regression: true, perf_non_regression: perfNonRegression },
          budget: { runs_per_action: runs },
        },
        attachBaseline,
      );
    } catch (e) {
      setErr(errorText(e));
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  };

  const fieldLabel = "text-[10px]";
  const fieldStyle = { color: "var(--fg-mute)" } as const;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "rgba(10, 11, 20, 0.48)" }}
      onClick={() => !closed && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label="按思路建优化轮次"
    >
      <div className="card w-full max-w-[720px] overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="m-0 text-sm font-semibold">按思路建优化轮次</h2>
          <span className="chip chip-run">思路 #{idea.id}</span>
          <span className="ml-auto text-[10px]" style={{ color: "var(--fg-mute)" }}>
            一思路一 Loop；建完先跑基线，再让 Agent 按优化点改代码
          </span>
        </div>

        <div className="grid max-h-[72vh] gap-3 overflow-auto px-4 py-3 text-[11px]">
          {/* 思路摘要（只读） */}
          <div
            className="rounded-[8px] px-3 py-2 leading-relaxed"
            style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
          >
            <div className="flex flex-wrap items-center gap-2">
              <b>#{idea.id} {idea.title}</b>
              <span className="chip !py-0 !text-[10px]">
                {idea.source === "conversation" ? "对话识别" : idea.source === "user" ? "用户" : "Agent"}
              </span>
              {idea.confidence && <span style={{ color: "var(--fg-mute)" }}>置信 {idea.confidence}</span>}
              {idea.effort && <span style={{ color: "var(--fg-mute)" }}>工作量 {idea.effort}</span>}
              <span className="ml-auto text-[10px]" style={{ color: "var(--fg-mute)" }}>
                思路摘要（只读）
              </span>
            </div>
            {idea.hypothesis && <div className="mt-1"><b>假设</b>：{idea.hypothesis}</div>}
            {idea.expected && <div><b>预期</b>：{idea.expected}</div>}
            {idea.risk && <div><b>风险</b>：{idea.risk}</div>}
            {idea.change_scope && <div className="mono"><b>范围</b>：{idea.change_scope}</div>}
          </div>

          {/* 需要优化的点（必填，作为 text objective 的 focus） */}
          <label className="grid gap-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className={fieldLabel} style={fieldStyle}>
                需要优化的点（会作为本轮目标注入 Agent 提示，必填）
              </span>
              <span className="mono ml-auto text-[10px]" style={{ color: "var(--fg-mute)" }}>
                {focusText.length}/2000
              </span>
            </span>
            <textarea
              ref={focusRef}
              className="input min-h-[120px] text-[12px]"
              value={focus}
              onChange={(e) => setFocus(e.target.value)}
              maxLength={2000}
              placeholder="例如：1M 带宽优化：优先分析 Global Memory 访问合并与 tile 大小，保持 64K 路径不退化"
              disabled={closed}
            />
            {!focusText && (
              <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                必填：后端要求 text 维度必须有非空 focus（≤2000 字）。
              </span>
            )}
          </label>

          {/* 约束 / 预算 / 复用基线 */}
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className={fieldLabel} style={fieldStyle}>
                约束
              </div>
              <div className="mt-1 flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-dim)" }}>
                <label className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={perfNonRegression}
                    onChange={(e) => setPerfNonRegression(e.target.checked)}
                    disabled={closed}
                  />
                  性能不退化（同条件 A/B 不得变慢）
                </label>
                <label
                  className="flex items-center gap-1.5"
                  style={{ color: "var(--fg-mute)" }}
                  title="精度不退化是默认约束，后端强制开启"
                >
                  <input type="checkbox" defaultChecked disabled />
                  精度不退化（默认约束，固定开启）
                </label>
              </div>
            </div>
            <div className="grid gap-2">
              <label className="grid gap-0.5">
                <span className={fieldLabel} style={fieldStyle}>
                  每动作执行次数（后端硬限制，1–10）
                </span>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={10}
                  value={runsPerAction}
                  onChange={(e) => setRunsPerAction(e.target.value)}
                  disabled={closed}
                />
              </label>
              <label className="flex items-start gap-2" style={{ color: "var(--fg-dim)" }}>
                <input
                  className="mt-0.5"
                  type="checkbox"
                  checked={attachBaseline}
                  onChange={(e) => setAttachBaseline(e.target.checked)}
                  disabled={closed}
                />
                <span>
                  复用同 sha 已有基线 run（推荐，省一次真机）
                  <span className="block text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    勾选后：该任务下未挂轮次、代码 sha 与基线存档一致的最近 test / benchmark run 会挂到本轮。
                  </span>
                </span>
              </label>
            </div>
          </div>

          {err && (
            <div
              className="rounded-[8px] px-3 py-2 text-[11px]"
              style={{ border: "1px solid var(--bad)", color: "var(--bad)" }}
            >
              {err}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
            本轮目标 = 上面的「优化点」（text 维度，不指定目标用例；perf-diff 按 benchmark 主用例兜底）。
          </span>
          <button className="btn ml-auto" onClick={onClose} disabled={closed}>
            取消
          </button>
          <button className="btn btn-primary" onClick={() => void submit()} disabled={!canSubmit}>
            {submitting || busy ? "创建中…" : "确认建轮次"}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * 优化前后对比（原 PerfDiffPanel 的对比部分并入轮次卡片）：两侧 run 都就绪后拉
 * `GET /api/loops/{id}/perf-diff`，展示前后快照、逐用例 A/B、精度不退化、findings 与门禁结论。
 * 「开优化 Loop / 关联 run / 按结论关闭 / 落盘」不再保留 —— 由轮次流程（基线/候选/采纳/报告）替代。
 */
function PerfCompare({ loopId, refreshKey }: { loopId: number; refreshKey: string }) {
  const [diff, setDiff] = useState<PerfDiff | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setDiff(await api.perfDiff(loopId));
      setErr(null);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setLoading(false);
    }
  }, [loopId]);

  useEffect(() => {
    setDiff(null);
    void load();
  }, [load, refreshKey]);

  const cc = conclusionChip(diff?.conclusion ?? null);

  return (
    <div className="mt-2 rounded-[9px] px-3 py-2" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="font-semibold">优化前后对比（同条件 A/B）</span>
        {diff && (
          <span className="chip" style={{ color: cc.color }}>
            {cc.label}
          </span>
        )}
        {diff && (
          <span className="chip" style={{ color: diff.verdict === "pass" ? "var(--ok)" : "var(--bad)" }}>
            门禁 {diff.verdict}
          </span>
        )}
        <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
          不设加速比硬线；提升不显著如实报告
        </span>
        <button className="btn btn-sm ml-auto" onClick={() => void load()} disabled={loading}>
          {loading ? "…" : "刷新对比"}
        </button>
      </div>

      {err && (
        <div className="mt-1 text-[11px]" style={{ color: "var(--bad)" }}>
          {err}
        </div>
      )}
      {loading && !diff && (
        <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          读取对比…
        </div>
      )}

      {diff && !diff.sides && (
        <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          还未凑齐两个代码快照（优化前/后各需 test + benchmark）。
        </div>
      )}

      {diff && diff.sides && (
        <>
          {/* 前后快照卡 */}
          <div className="mt-1.5 grid gap-3 text-[11px]" style={{ gridTemplateColumns: "1fr 1fr" }}>
            {(["before", "after"] as const).map((side) => (
              <div key={side} className="rounded-md px-3 py-2" style={{ background: "var(--panel-3)" }}>
                <div style={{ color: "var(--fg-mute)" }}>{side === "before" ? "优化前（基线）" : "优化后（候选）"}</div>
                <div className="mono">{diff.sides![side].code_snapshot_sha256.slice(0, 12)}…</div>
                <div style={{ color: "var(--fg-dim)" }}>
                  test #{diff.sides![side].test_run_id ?? "—"} · benchmark #{diff.sides![side].benchmark_run_id ?? "—"}
                </div>
              </div>
            ))}
          </div>

          {/* 逐用例 A/B */}
          {diff.cases.length > 0 && (
            <div className="mt-1 max-h-[280px] overflow-auto">
              <table>
                <thead>
                  <tr>
                    <th>用例</th>
                    <th>N</th>
                    <th>优化前(us)</th>
                    <th>优化后(us)</th>
                    <th>变化</th>
                    <th>不确定性</th>
                    <th>结论</th>
                    <th>基线(前→后)</th>
                  </tr>
                </thead>
                <tbody>
                  {diff.cases.map((c) => {
                    const rowCc = conclusionChip(c.verdict);
                    return (
                      <tr key={c.case_id}>
                        <td className="mono">{c.case_id}</td>
                        <td>{c.n ?? "—"}</td>
                        <td className="mono">{c.before_us.toFixed(3)}</td>
                        <td className="mono">{c.after_us.toFixed(3)}</td>
                        <td
                          className="mono"
                          style={{
                            color:
                              c.delta_pct == null
                                ? "var(--fg-mute)"
                                : c.delta_pct < 0
                                  ? "var(--ok)"
                                  : c.delta_pct > 0
                                    ? "var(--bad)"
                                    : "var(--fg-dim)",
                          }}
                          title={c.delta_pct == null ? "缺有效前后测量（missing_data），无法计算变化" : undefined}
                        >
                          {c.delta_pct == null
                            ? "缺数据"
                            : `${c.delta_pct > 0 ? "+" : ""}${c.delta_pct.toFixed(2)}%`}
                        </td>
                        <td className="mono" style={{ color: "var(--fg-mute)" }}>
                          ±{c.uncertainty_pct.toFixed(2)}%
                        </td>
                        <td>
                          <span className="chip" style={{ color: rowCc.color }}>
                            {rowCc.label}
                          </span>
                        </td>
                        <td className="mono" style={{ color: "var(--fg-dim)" }}>
                          {c.before_baseline_us.toFixed(2)} → {c.after_baseline_us.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* 精度不退化 */}
          {diff.precision && (
            <div className="mt-1 text-[11px]">
              <b>精度不退化</b>：
              {diff.precision.non_regression ? (
                <span className="chip chip-ok ml-2">通过（{diff.precision.cases.length} 例未退化）</span>
              ) : (
                <span className="chip chip-bad ml-2">退化：{diff.precision.regressed_cases.join(" ")}</span>
              )}
            </div>
          )}

          {/* findings（门禁结论在标题行） */}
          <div className="mt-1.5 flex flex-wrap gap-2">
            {diff.findings.map((f, i) => (
              <span key={i} className={`chip ${f.level === "fail" ? "chip-bad" : "chip-run"}`} title={f.message}>
                {f.code}
              </span>
            ))}
            {diff.findings.length === 0 && <span className="chip chip-ok">无 finding</span>}
          </div>

          {/* 建议只读 */}
          {diff.recommendations && diff.recommendations.length > 0 && (
            <details className="mt-1">
              <summary className="cursor-pointer text-[10px]" style={{ color: "var(--fg-mute)" }}>
                优化建议（只读，{diff.recommendations.length} 条）
              </summary>
              {diff.recommendations.map((r) => (
                <div
                  key={r.id}
                  className="mt-1 rounded-md px-3 py-2 text-[11px]"
                  style={{ background: "var(--panel-3)", border: "1px solid var(--border)" }}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="mono">{r.id}</span>
                    {r.title && <b>{r.title}</b>}
                    <span
                      className="chip"
                      style={{
                        color:
                          r.status === "validated"
                            ? "var(--ok)"
                            : r.status === "rejected"
                              ? "var(--bad)"
                              : "var(--fg-dim)",
                      }}
                    >
                      {r.status}
                    </span>
                    <span style={{ color: "var(--fg-mute)" }}>置信 {r.confidence}</span>
                    {r.affected_cases.length > 0 && (
                      <span className="mono" style={{ color: "var(--fg-mute)" }}>
                        用例 {r.affected_cases.join(",")}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 leading-relaxed" style={{ color: "var(--fg-dim)" }}>
                    <div>
                      <b>证据</b>：{r.observed_evidence.join("；")}
                    </div>
                    <div>
                      <b>瓶颈假设</b>：{r.bottleneck_hypothesis}
                    </div>
                    <div>
                      <b>改动</b>：{r.proposed_change}
                    </div>
                    <div>
                      <b>预期观测</b>：{r.expected_observation}
                    </div>
                    <div>
                      <b>约束/风险</b>：{r.constraints_and_risks}
                    </div>
                    <div>
                      <b>验证方法</b>：{r.validation_method}
                      {r.skill ? `（skill: ${r.skill}）` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </details>
          )}
        </>
      )}
    </div>
  );
}

/** 新建轮次浮层（样式参照 TaskDeleteDialog）：目标类型/维度/用例 + 约束 + 预算 + 用户思路/假设。 */
function RoundForm({
  task,
  targets,
  targetsErr,
  onRetryTargets,
  onClose,
  onCreated,
}: {
  task: TaskDetail;
  targets: OptimizationTargets | null;
  targetsErr: string | null;
  onRetryTargets: () => void;
  onClose: () => void;
  onCreated: (res: OptimizationRoundCreated) => void;
}) {
  const [label, setLabel] = useState("");
  const [type, setType] = useState<RoundType>("perf");
  const [dimension, setDimension] = useState<Dimension>("cases");
  const [selected, setSelected] = useState<string[]>([]);
  const [precisionNonRegression, setPrecisionNonRegression] = useState(true);
  const [perfNonRegression, setPerfNonRegression] = useState(true);
  const [allowedPaths, setAllowedPaths] = useState(task.op_name ? `ops/${task.op_name}/` : "ops/");
  const [diagnoseRounds, setDiagnoseRounds] = useState("3");
  const [verifyRounds, setVerifyRounds] = useState("3");
  const [runsPerAction, setRunsPerAction] = useState("3");
  const [userText, setUserText] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [attachBaseline, setAttachBaseline] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const appliedRef = useRef(false);

  const casesForType = (t: RoundType) => {
    if (t === "precision") {
      return (targets?.test_cases ?? []).map((c) => ({
        case_id: c.case_id,
        status: c.status,
        n: null as number | null,
        dtype: null as string | null,
        note: null as string | null,
        primary: false,
        required: false,
      }));
    }
    return (targets?.cases ?? []).map((c) => ({
      case_id: c.case_id,
      status: "",
      n: c.n ?? null,
      dtype: c.dtype ?? null,
      note: c.note ?? null,
      primary: Boolean(c.primary),
      required: Boolean(c.required),
    }));
  };

  /** 默认目标用例：精度 = failing_cases；性能/硬件 = 主用例。 */
  const defaultSelection = (t: RoundType): string[] => {
    if (!targets) return [];
    if (t === "precision") return (targets.failing_cases ?? []).map((c) => c.case_id);
    return targets.primary_case ? [targets.primary_case] : [];
  };

  useEffect(() => {
    if (!targets || appliedRef.current) return;
    appliedRef.current = true;
    setSelected(defaultSelection(type));
    setPerfNonRegression(type !== "precision");
    setPrecisionNonRegression(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targets]);

  // Esc 关闭浮层（与 TaskDeleteDialog 一致；提交中不响应）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !submitting) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [submitting, onClose]);

  const changeType = (next: RoundType) => {
    setType(next);
    setSelected(defaultSelection(next));
    setPerfNonRegression(next !== "precision");
    setPrecisionNonRegression(true);
  };

  const caseOptions = casesForType(type);
  const failing = useMemo(
    () => new Set((targets?.failing_cases ?? []).map((c) => c.case_id)),
    [targets],
  );
  const toggleCase = (caseId: string) => {
    setSelected((prev) => (prev.includes(caseId) ? prev.filter((x) => x !== caseId) : [...prev, caseId]));
  };

  const intOr = (raw: string, fallback: number) => {
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
  };

  const needCases = dimension === "cases";
  const canSubmit = !submitting && Boolean(targets) && (!needCases || selected.length > 0);

  const submit = async () => {
    if (!canSubmit || !targets) return;
    setSubmitting(true);
    setErr(null);
    const snapshot: OptimizationTargetsSnapshot = {
      benchmark_run_id: targets.benchmark_run_id,
      test_run_id: targets.test_run_id,
      primary_case: targets.primary_case,
      suite_ref: targets.suite_ref,
      cases: targets.cases,
      failing_cases: targets.failing_cases,
    };
    const objective: OptimizationObjective = {
      schema: "opx-optimization-objective/1",
      label: label.trim() || null,
      type,
      target_dimension: dimension,
      target_cases: dimension === "cases" ? selected : [],
      constraints: {
        precision_non_regression: precisionNonRegression,
        perf_non_regression: perfNonRegression,
        allowed_paths: allowedPaths
          .split(/[,，\s]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      },
      budget: {
        diagnose_rounds: intOr(diagnoseRounds, 3),
        verify_rounds: intOr(verifyRounds, 3),
        runs_per_action: intOr(runsPerAction, 3),
      },
      user_text: userText.trim() || null,
      targets_snapshot: snapshot,
    };
    const payload: OptimizationRoundCreate = {
      objective,
      label: label.trim() || null,
      hypothesis: hypothesis.trim() || null,
      attach_baseline_runs: attachBaseline,
    };
    try {
      onCreated(await api.createOptimizationRound(task.id, payload));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setSubmitting(false);
    }
  };

  const fieldLabel = "text-[10px]";
  const fieldStyle = { color: "var(--fg-mute)" } as const;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "rgba(10, 11, 20, 0.42)" }}
      onClick={() => !submitting && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label="新建优化轮次"
    >
      <div className="card w-full max-w-[720px] overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="m-0 text-sm font-semibold">新建优化轮次</h2>
          <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
            目标 + 约束 + 用户思路；建完先采基线
          </span>
        </div>

        <div className="max-h-[70vh] overflow-auto px-4 py-3">
          <div className="grid gap-3">
            {/* 标题 */}
            <label className="grid gap-1">
              <span className={fieldLabel} style={fieldStyle}>
                标题（可选）
              </span>
              <input
                className="input"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="例如：1M 带宽优化"
                disabled={submitting}
              />
            </label>

            {/* 目标类型 / 维度 */}
            <div className="grid gap-2 md:grid-cols-2">
              <div>
                <div className={fieldLabel} style={fieldStyle}>
                  目标类型
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {TYPE_OPTIONS.map((t) => (
                    <button
                      key={t.key}
                      className="btn btn-sm"
                      style={type === t.key ? { background: "var(--accent-soft)", borderColor: "var(--accent)" } : undefined}
                      onClick={() => changeType(t.key)}
                      disabled={submitting}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className={fieldLabel} style={fieldStyle}>
                  目标维度
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {(
                    [
                      { key: "cases", label: "指定用例" },
                      { key: "all", label: "全部用例" },
                    ] as { key: Dimension; label: string }[]
                  ).map((d) => (
                    <button
                      key={d.key}
                      className="btn btn-sm"
                      style={dimension === d.key ? { background: "var(--accent-soft)", borderColor: "var(--accent)" } : undefined}
                      onClick={() => setDimension(d.key)}
                      disabled={submitting}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* 目标用例 */}
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={fieldLabel} style={fieldStyle}>
                  目标用例
                </span>
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  {type === "precision" ? "来自最近一次测试报告（test_cases）" : "来自最近一次基准报告（cases）"}
                </span>
                {targets && (
                  <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    基准 #{targets.benchmark_run_id ?? "—"} · 测试 #{targets.test_run_id ?? "—"}
                  </span>
                )}
              </div>

              {targetsErr && (
                <div className="mt-1 flex flex-wrap items-center gap-2 rounded-[8px] px-3 py-2 text-[11px]" style={{ border: "1px solid var(--bad)", color: "var(--bad)" }}>
                  读取目标用例失败：{targetsErr}
                  <button className="btn btn-sm" onClick={onRetryTargets} disabled={submitting}>
                    重试
                  </button>
                </div>
              )}
              {!targets && !targetsErr && (
                <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                  读取目标用例…
                </div>
              )}

              {targets && (
                <>
                  {targets.hint && (
                    <div className="mt-1 text-[11px]" style={{ color: "var(--warn)" }}>
                      {targets.hint}
                    </div>
                  )}
                  {dimension === "all" && (
                    <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      选了「全部用例」：不勾选用例，验收覆盖报告里的全部用例。
                    </div>
                  )}
                  <div
                    className="mt-1 max-h-[180px] overflow-auto rounded-[8px] px-2 py-1"
                    style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
                  >
                    {caseOptions.length === 0 ? (
                      <div className="py-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                        {type === "precision"
                          ? "没有测试用例报告：先跑一次「正确性测试」"
                          : "没有基准用例报告：先跑一次「基准测试」"}
                      </div>
                    ) : (
                      caseOptions.map((c) => (
                        <label
                          key={c.case_id}
                          className="flex items-center gap-2 py-0.5 text-[11px]"
                          style={{ color: dimension === "all" ? "var(--fg-mute)" : "var(--fg-dim)" }}
                        >
                          <input
                            type="checkbox"
                            checked={selected.includes(c.case_id)}
                            onChange={() => toggleCase(c.case_id)}
                            disabled={submitting || dimension === "all"}
                          />
                          <span className="mono">{c.case_id}</span>
                          {c.primary && <span className="chip !py-0 !text-[10px]">主用例</span>}
                          {c.required && <span className="chip !py-0 !text-[10px]">必需</span>}
                          {c.status && (
                            <span
                              className="chip !py-0 !text-[10px]"
                              style={{ color: c.status === "PASS" ? "var(--ok)" : "var(--bad)" }}
                            >
                              {c.status}
                            </span>
                          )}
                          {c.n != null && (
                            <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              n={c.n}
                            </span>
                          )}
                          {c.dtype && (
                            <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              {c.dtype}
                            </span>
                          )}
                          {failing.has(c.case_id) && (
                            <span className="chip chip-bad !py-0 !text-[10px]">失败用例</span>
                          )}
                          {c.note && (
                            <span className="truncate text-[10px]" style={{ color: "var(--fg-mute)" }} title={c.note}>
                              {c.note}
                            </span>
                          )}
                        </label>
                      ))
                    )}
                  </div>
                  {needCases && selected.length === 0 && (
                    <div className="mt-1 text-[10px]" style={{ color: "var(--warn)" }}>
                      「指定用例」维度至少要勾一个目标用例（或切到「全部用例」）。
                    </div>
                  )}
                </>
              )}
            </div>

            {/* 约束 */}
            <div className="grid gap-2 md:grid-cols-2">
              <div>
                <div className={fieldLabel} style={fieldStyle}>
                  约束
                </div>
                <div className="mt-1 flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-dim)" }}>
                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={precisionNonRegression}
                      onChange={(e) => setPrecisionNonRegression(e.target.checked)}
                      disabled={submitting}
                    />
                    精度不退化（正确性用例不得变差）
                  </label>
                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={perfNonRegression}
                      onChange={(e) => setPerfNonRegression(e.target.checked)}
                      disabled={submitting}
                    />
                    性能不退化（同条件 A/B 不得变慢）
                  </label>
                </div>
              </div>
              <label className="grid gap-1">
                <span className={fieldLabel} style={fieldStyle}>
                  允许改动范围（默认 ops/&lt;op&gt;/）
                </span>
                <input
                  className="input mono"
                  value={allowedPaths}
                  onChange={(e) => setAllowedPaths(e.target.value)}
                  placeholder="ops/<op>/ ，多个用逗号分隔"
                  disabled={submitting}
                />
              </label>
            </div>

            {/* 预算 */}
            <div>
              <div className={fieldLabel} style={fieldStyle}>
                预算（只有「每动作执行次数」是后端硬限制并审计；判断/验证轮数为提示值，不参与硬门禁）
              </div>
              <div className="mt-1 grid grid-cols-3 gap-2">
                {[
                  { label: "判断轮数", value: diagnoseRounds, set: setDiagnoseRounds },
                  { label: "验证轮数", value: verifyRounds, set: setVerifyRounds },
                  { label: "每动作执行次数", value: runsPerAction, set: setRunsPerAction },
                ].map((f) => (
                  <label key={f.label} className="grid gap-0.5">
                    <span className="text-[10px]" style={fieldStyle}>
                      {f.label}
                    </span>
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={f.value}
                      onChange={(e) => f.set(e.target.value)}
                      disabled={submitting}
                    />
                  </label>
                ))}
              </div>
            </div>

            {/* 用户思路 / 假设 */}
            <label className="grid gap-1">
              <span className={fieldLabel} style={fieldStyle}>
                用户补充思路 / 约束（原样注入 agent 提示）
              </span>
              <textarea
                className="input min-h-[64px] text-[12px]"
                value={userText}
                onChange={(e) => setUserText(e.target.value)}
                placeholder="例如：只允许改 ops/ 下的候选实现；优先试 128/256 分块；不要动基线实现"
                disabled={submitting}
              />
            </label>
            <label className="grid gap-1">
              <span className={fieldLabel} style={fieldStyle}>
                假设（可选）
              </span>
              <textarea
                className="input min-h-[52px] text-[12px]"
                value={hypothesis}
                onChange={(e) => setHypothesis(e.target.value)}
                placeholder="缺省由目标自动生成：基于目标硬件画像与现有证据，寻找可回退的最小改动"
                disabled={submitting}
              />
            </label>

            <label className="flex items-start gap-2 text-[11px]" style={{ color: "var(--fg-dim)" }}>
              <input
                type="checkbox"
                className="mt-0.5"
                checked={attachBaseline}
                onChange={(e) => setAttachBaseline(e.target.checked)}
                disabled={submitting}
              />
              <span>
                复用同代码快照的基线 run（推荐，省一次真机）
                <span className="block text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  勾选后：该任务下未挂轮次、代码 sha 与基线存档一致的最近 test / benchmark run 会挂到本轮。
                </span>
              </span>
            </label>

            {err && (
              <div className="rounded-[8px] px-3 py-2 text-[11px]" style={{ border: "1px solid var(--bad)", color: "var(--bad)" }}>
                {err}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
            同一轮两次动作会用同一张卡（设备 pin）；有运行中作业时后端会拒绝。
          </span>
          <button className="btn ml-auto" onClick={onClose} disabled={submitting}>
            取消
          </button>
          <button className="btn btn-primary" onClick={() => void submit()} disabled={!canSubmit}>
            {submitting ? "创建中…" : "建轮次"}
          </button>
        </div>
      </div>
    </div>
  );
}
