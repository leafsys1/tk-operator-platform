import { useEffect, useRef, useState } from "react";
import { api, STAGES, when } from "../lib/api";
import type { ChatSeed, Loop, StageTaskView, TaskDetail, TaskGates } from "../types";
import ActionPlanPanel from "./ActionPlanPanel";
import ChatPage from "./ChatPage";
import ContextPanel from "./ContextPanel";
import DeliveryPanel from "./DeliveryPanel";
import GatePanel from "./GatePanel";
import ImplementationPanel from "./ImplementationPanel";
import OptimizationRoundPanel from "./OptimizationRoundPanel";
import ResultsPanel from "./ResultsPanel";
import SavepointPanel from "./SavepointPanel";
import SpecPanel from "./SpecPanel";
import TaskCasePanel from "./TaskCasePanel";
import TaskConfigPanel from "./TaskConfigPanel";
import TaskRecordsPanel from "./TaskRecordsPanel";
import TaskVersionsPanel from "./TaskVersionsPanel";

type TabKey = "overview" | "chat" | "spec" | "impl" | "results" | "perf" | "versions" | "records";

const TABS: { key: TabKey; label: string; hint: string }[] = [
  { key: "overview", label: "概览", hint: "执行七步：开发（→ 对话）→ 编译 → 冒烟 → 正确性 → 基准 → 采集 → 交付；每步结果可 AI 审查" },
  { key: "chat", label: "对话", hint: "本任务的 agent 会话：多轮确认 / 追问 / 附件 / 回退（挂 skills，可改代码）" },
  { key: "spec", label: "需求与规格", hint: "算子需求文档、版本历史、上下文空间与执行配置" },
  { key: "impl", label: "实现与调用", hint: "工作区文件 / diff / 构建日志（发起 agent run 在「概览」）" },
  { key: "results", label: "测试与结果", hint: "用例覆盖、逐例精度，以及 test / benchmark / profile 的机读报告" },
  { key: "perf", label: "性能与优化", hint: "优化前后的同条件对比、结论与结构化建议" },
  { key: "versions", label: "版本", hint: "任务版本（v大.小 / 来源 / 交付状态）+ 源码存档与回滚" },
  { key: "records", label: "记录与交付", hint: "轮次归属、调用/执行台账、交付门禁与交付包下载" },
];

const GATE_LABEL: Record<string, string> = { build: "编译", smoke: "冒烟", test: "正确性", benchmark: "基准", profile: "采集" };
const GATE_ACTIONS = ["build", "smoke", "test", "benchmark", "profile"] as const;

/** seed nonce：同一毫秒连点可能生成相同 timestamp，加自增后缀保证每次带入都触发。 */
let seedNonceSeq = 0;
const nextSeedNonce = () => Date.now() * 1000 + (seedNonceSeq++ % 1000);

/**
 * 任务详情（作业台）的壳：把原先十几个平铺卡片收进六个分区。
 *
 * 设计约束：
 * - 顶栏只保留「一眼判断卡在哪」的信息：阶段、门禁、执行/调用/费用；
 * - 分区只改导航，不改各面板的数据口径；面板保持挂载，切 tab 不会掐断 SSE；
 * - 隐藏分区用 display:none，切回来时补一次 resize（ECharts 在隐藏容器里量不到宽度）。
 */
export default function TaskWorkbench({
  task,
  stageView,
  onChanged,
  onOpenTests,
}: {
  task: TaskDetail;
  stageView: StageTaskView | null;
  onChanged: () => void;
  /** 打开测试管理：带上当前任务 id，由 App 决定是否定位到该任务。 */
  onOpenTests?: (taskId: number) => void;
}) {
  const [tab, setTab] = useState<TabKey>("overview");
  const [gates, setGates] = useState<TaskGates | null>(null);
  const [gatesErr, setGatesErr] = useState<string | null>(null);
  const [gateNonce, setGateNonce] = useState(0);
  // 概览 →「对话」的带入文本（开发 kickoff / AI 审查上下文）；nonce 变化触发对话分区预填。
  // 额外带 taskId：切任务时 ChatPage 会按 key 重挂载，旧 seed 不能带到新任务。
  const [chatSeed, setChatSeed] = useState<(ChatSeed & { taskId: number }) | null>(null);
  /** 对话分区放大：全屏浮层（Esc 关闭）；同一个 ChatPage 实例，不重挂载、不打断 SSE。 */
  const [chatMax, setChatMax] = useState(false);
  // 切任务同一帧丢弃上个任务的 seed（渲染期调整：子组件按新 key 重挂载前先生效，
  // 也避免切回旧任务时把上次的 seed 重新带出来）。
  const seedTaskRef = useRef(task.id);
  if (seedTaskRef.current !== task.id) {
    seedTaskRef.current = task.id;
    setChatSeed(null);
    setChatMax(false);
  }
  const openChat = (text: string) => {
    setChatSeed({ taskId: task.id, text, nonce: nextSeedNonce() });
    setTab("chat");
  };
  /** 新口径：先进任务对话说优化方向，方向确认后再落自动优化台账。 */
  const openNewOptimizationChat = () => {
    setChatSeed({ taskId: task.id, text: "", nonce: nextSeedNonce(), newOptimization: true });
    setTab("chat");
  };
  /**
   * 性能与优化 →「对话」：带着本轮目标上下文跳转，回车即在该会话里启动自动优化。
   * 传 conversationId 时优先打开它（「在对话中查看」/续聊）。
   */
  const openOptimizeChat = (loop: Loop, conversationId?: number | null, startedJobId?: number | null) => {
    const obj = loop.objective ?? {};
    const focus = (obj.focus ?? "").trim();
    // 兜底：任务详情里该轮次是否有正在跑的优化 job（后端会话 running_job 现已覆盖优化 job；
    // 这里带上 job id，ChatPage 可直接接回 SSE，不依赖会话列表刷新的时序）。
    const runningJob = task.agent_jobs.find(
      (j) => j.kind === "优化" && j.status === "running" && j.loop_id === loop.id,
    );
    setChatSeed({
      taskId: task.id,
      text: "",
      nonce: nextSeedNonce(),
      conversationId: conversationId ?? undefined,
      optimize: {
        loopId: loop.id,
        label: obj.label || loop.scope || `轮次 #${loop.id}`,
        focus: focus || null,
        allowedPaths: obj.constraints?.allowed_paths ?? [],
        budget: obj.budget?.runs_per_action ?? 3,
        jobId: startedJobId ?? runningJob?.id ?? null,
      },
    });
    setTab("chat");
  };
  // 只把属于当前任务的 seed 交给 ChatPage；任务刚切换的同一帧也不会带旧 seed 挂载。
  const seedForChat = chatSeed && chatSeed.taskId === task.id ? chatSeed : null;

  useEffect(() => {
    setTab("overview");
  }, [task.id]);

  // 门禁依赖真实执行 + 轮次归属：新建 perf Loop、把 run 关联进 Loop、run 状态变化都要重拉；
  // 否则顶部 perf 门禁与 GatePanel 的验收证据会一直停在旧值。
  const gateSig = [
    task.runs.length,
    task.agent_jobs.length,
    task.agent_jobs.map((j) => `${j.id}:${j.status}`).join(","),
    task.current_spec_version_id, task.version_minor,
    task.cases_confirmed_at, task.cases_confirmed_sha256,
    task.runs.map((r) => `${r.id}:${r.loop_id ?? "-"}:${r.status}`).join(","),
    task.loops.map((l) => `${l.id}:${l.status}`).join(","),
  ].join("|");

  useEffect(() => {
    let alive = true;
    setGatesErr(null);
    api.taskGates(task.id)
      .then((g) => {
        if (!alive) return;
        setGates(g);
        setGatesErr(null);
      })
      .catch((e) => alive && setGatesErr(String(e)));
    return () => {
      alive = false;
    };
  }, [task.id, gateSig, gateNonce]);

  /** 子面板改动后：显式重拉门禁，并让父级刷新任务详情（阶段 / 规格 / 统计）。 */
  const refresh = () => {
    setGateNonce((n) => n + 1);
    onChanged();
  };

  // 切换分区：回到工作台顶部，并让隐藏时初始化的图表重新量宽。
  useEffect(() => {
    window.scrollTo({ top: 0 });
    const raf = window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => window.cancelAnimationFrame(raf);
  }, [tab]);

  // 放大模式：Esc 关闭；锁住 body 滚动，收起后恢复。
  useEffect(() => {
    if (!chatMax) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setChatMax(false);
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [chatMax]);

  const stage = stageView?.derived_stage ?? task.stage;
  const inSync = stageView ? stageView.in_sync : true;
  // 任务流程写入的 next_action 是当前真实进展，优先于阶段模型的通用门禁提示。
  const nextStep = task.next_action ?? stageView?.next_step ?? null;
  const nextStepLabel = task.next_action ? "当前进展" : "下一步";

  // 费用按端点价目实时算（/api/tasks/{id} 已回填）；P0 示例行（is_demo）不计入，与消耗页同口径。
  const billable = task.agent_jobs.filter((j) => !j.is_demo);
  const priced = billable.filter((j) => j.cost != null);
  const unpriced = billable.filter(
    (j) => j.cost == null && (j.tokens_in ?? 0) + (j.tokens_out ?? 0) + (j.cache_read ?? 0) + (j.cache_write ?? 0) > 0,
  ).length;
  const estimated = priced.some((j) => j.cost_estimated);
  const cost = priced.length
    ? `¥${priced.reduce((a, b) => a + (b.cost ?? 0), 0).toFixed(2)}${unpriced ? " +" : ""}${estimated ? "（估）" : ""}`
    : "未定价";
  const currentSpec = task.spec_versions.find((s) => s.id === task.current_spec_version_id);
  const latestBenchmark = [...task.runs].filter((r) => r.action === "benchmark" && r.status === "ok").sort((a, b) => b.id - a.id)[0] ?? null;
  const latestProfile = [...task.runs].filter((r) => r.action === "profile" && r.status === "ok").sort((a, b) => b.id - a.id)[0] ?? null;
  const runningAgent = task.agent_jobs.some((j) => j.status === "running");
  // 「性能与优化」徽标：诊断 job（P2 优化点搜索）也会拦任务删除，运行中也算，保持前端可见性一致。
  const runningOptimize = task.agent_jobs.some(
    (j) => (j.kind === "优化" || j.kind === "诊断") && j.status === "running",
  );

  const badgeFor = (key: TabKey) => {
    if (key === "spec" && task.spec_versions.length > 0) return String(task.spec_versions.length);
    if (key === "impl" && runningAgent) return "运行中";
    // 自动优化已对话化：对话与性能分区都在跑优化 job 时给「运行中」提示
    if ((key === "chat" || key === "perf") && runningOptimize) return "运行中";
    if (key === "records" && task.runs.length > 0) return String(task.runs.length);
    return null;
  };

  const active = TABS.find((t) => t.key === tab) ?? TABS[0];
  const panel = (key: TabKey) => ({ style: { display: tab === key ? undefined : "none" } });

  return (
    <>
      {/* ── 任务摘要：身份 + 阶段 + 门禁 + 规模 ───────────────────── */}
      <section className="card mb-4 overflow-hidden">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>算子</span>
          <span className="mono text-[13px] font-semibold">{task.op_name ?? "—"}</span>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            芯片 <span className="mono" style={{ color: "var(--fg-dim)" }}>{task.target_chip ?? "—"}</span>
          </span>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            判据 <span className="mono" style={{ color: "var(--fg-dim)" }}>{task.criteria_ref ?? "项目默认"}</span>
          </span>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            规格{" "}
            <span className="mono" style={{ color: "var(--fg-dim)" }}>
              {currentSpec
                ? `v${currentSpec.version}.${task.version_minor ?? 0}`
                : task.current_spec_version_id != null
                  ? `#${task.current_spec_version_id}`
                  : "—"}
            </span>
          </span>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            用例{" "}
            <span
              className={task.cases_confirmed_at ? "chip chip-ok !py-0 !text-[10px]" : "chip chip-warn !py-0 !text-[10px]"}
              title={task.cases_confirmed_at ? `用例组确认于 ${when(task.cases_confirmed_at)}` : "精度/性能用例组还没整组确认（在任务对话里确认）"}
            >
              {task.cases_confirmed_at ? "已确认" : "未确认"}
            </span>
          </span>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>创建 {when(task.created_at)}</span>
          <button className="btn btn-sm ml-auto" onClick={() => setTab("spec")}>
            查看规格
          </button>
        </div>

        <div className="grid gap-0 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)_auto]">
          {/* 阶段 */}
          <div className="px-4 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>当前阶段</span>
              <b className="text-xs">{stage}</b>
              {!inSync && <span className="chip chip-warn">存值 {stageView?.stored_stage ?? task.stage}（未同步）</span>}
            </div>
            <div className="mt-2 grid gap-1.5" style={{ gridTemplateColumns: `repeat(${STAGES.length}, minmax(0,1fr))` }}>
              {STAGES.map((s) => {
                const g = stageView?.stages?.find((x) => x.stage === s);
                const cur = s === stage;
                const done = Boolean(g?.ok);
                return (
                  <div key={s}>
                    <div
                      className="h-[5px] rounded-[3px]"
                      style={{ background: done ? "var(--ok)" : cur ? "var(--run)" : "var(--border-strong)" }}
                    />
                    <div
                      className="mt-1 truncate text-[10px]"
                      style={{ color: cur ? "var(--accent)" : done ? "var(--ok)" : "var(--fg-mute)" }}
                      title={g ? `${g.name}${g.ok ? " ✓" : ""}${g.evidence.length ? ` · ${g.evidence.join("；")}` : ""}` : s}
                    >
                      {s}
                    </div>
                  </div>
                );
              })}
            </div>
            {nextStep && (
              <div className="mt-2 text-[11px] leading-relaxed" style={{ color: "var(--fg-dim)" }}>
                {nextStepLabel}：{nextStep}
              </div>
            )}
          </div>

          {/* 门禁 */}
          <div className="px-4 py-3" style={{ borderLeft: "1px solid var(--border)" }}>
            <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>动作证据与门禁</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {gates ? (
                GATE_ACTIONS.map((a) => {
                  const g = gates.gates.find((x) => x.action === a);
                  const verdict = g?.verdict ?? null;
                  const cls = !g || g.run_id === null ? "chip" : verdict === "pass" ? "chip chip-ok" : "chip chip-bad";
                  return (
                    <span
                      key={a}
                      className={cls}
                      title={g?.fail_codes?.length ? g.fail_codes.join(", ") : undefined}
                    >
                      {GATE_LABEL[a]} · {!g || g.run_id === null ? "未跑" : verdict ?? "未定"}
                      {g?.run_id != null && <span style={{ opacity: 0.7 }}> #{g.run_id}</span>}
                    </span>
                  );
                })
              ) : gatesErr ? (
                <span className="text-[11px]" style={{ color: "var(--bad)" }}>读取门禁失败</span>
              ) : (
                <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>读取门禁…</span>
              )}
            </div>
            {gates?.perf && (
              <div className="mt-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                perf Loop #{gates.perf.loop_id}：
                <span style={{ color: gates.perf.verdict === "pass" ? "var(--ok)" : "var(--bad)" }}>
                  {gates.perf.verdict ?? "未定"}
                </span>
              </div>
            )}
            {gatesErr && (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px]" style={{ color: "var(--bad)" }}>
                门禁读取失败：{gatesErr}
                <button className="btn btn-sm !py-0 !text-[10px]" onClick={() => setGateNonce((n) => n + 1)}>
                  重试
                </button>
              </div>
            )}
          </div>

          {/* 规模 */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3" style={{ borderLeft: "1px solid var(--border)" }}>
            {[
              { k: "执行", v: String(task.runs.length) },
              { k: "调用", v: String(task.agent_jobs.length) },
              { k: "轮次", v: String(task.loops.length) },
              { k: "费用", v: cost },
            ].map((s) => (
              <div key={s.k}>
                <div className="text-[10px]" style={{ color: "var(--fg-mute)" }}>{s.k}</div>
                <div className="mt-0.5 text-[13px] font-semibold">{s.v}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 分区导航 ─────────────────────────────────────────────── */}
      <nav
        className="sticky top-0 z-20 -mx-6 mb-3 flex flex-wrap items-center gap-1 px-6 pt-2"
        style={{ background: "var(--bg)", borderBottom: "1px solid var(--border)" }}
      >
        {TABS.map((t) => {
          const badge = badgeFor(t.key);
          return (
            <button
              key={t.key}
              className="wtab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
            >
              {t.label}
              {badge && (
                <span className={`ml-1.5 rounded-full px-1.5 py-0 text-[10px] ${badge === "运行中" ? "chip chip-run !py-0" : ""}`}
                  style={badge === "运行中" ? undefined : { background: "var(--panel-3)", color: "var(--fg-mute)" }}>
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>
      <div className="mb-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>{active.hint}</div>

      {/* ── 分区内容（保持挂载，切走不掐断 SSE） ─────────────────── */}
      <div {...panel("overview")}>
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
          <GatePanel key={`gate-${task.id}-${gateSig}`} taskId={task.id} />
          <ActionPlanPanel
            taskId={task.id}
            opName={task.op_name ?? task.title}
            runs={task.runs}
            gates={gates}
            evidenceRevision={gateSig}
            onDone={refresh}
            onOpenChat={openChat}
          />
        </div>
        <div className="mt-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          七步：① 开发（跳「对话」）→ ② 编译 → ③ 冒烟 → ④ 正确性 → ⑤ 基准 → ⑥ 采集 → ⑦ 交付；执行前先算 plan 再真执行，
          交付由后端本地打包。每步跑完可点「AI 审查」把 run / 门禁 / 报告入口带进对话；结果见「测试与结果」，台账与交付见「记录与交付」。
        </div>
      </div>

      {/* 对话：正常在分区内；「放大」用全屏浮层 —— 同一个 ChatPage 实例只换外层容器，不重挂载、不打断 SSE。 */}
      <div style={chatMax ? undefined : panel("chat").style}>
        <div
          className={chatMax ? "fixed inset-0 z-50 flex flex-col" : ""}
          style={chatMax ? { background: "var(--bg)" } : undefined}
        >
          <div
            className={chatMax ? "flex flex-wrap items-center gap-2 px-6 py-2.5" : "mb-2 flex flex-wrap items-center gap-2"}
            style={chatMax ? { background: "var(--panel)", borderBottom: "1px solid var(--border)" } : undefined}
          >
            <h2 className="m-0 text-xs font-semibold">对话{chatMax ? ` · ${task.title}` : ""}</h2>
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              {chatMax ? "放大模式：Esc 或「还原」回到分区内" : "agent 会话：多轮确认 / 附件 / 回退（挂 skills，可改代码）"}
            </span>
            <button
              className="btn btn-sm ml-auto"
              onClick={() => setChatMax((v) => !v)}
              title={chatMax ? "还原到分区内（Esc）" : "全屏放大（Esc 关闭）"}
            >
              {chatMax ? "还原" : "⤢ 放大"}
            </button>
          </div>
          <div className={chatMax ? "min-h-0 flex-1 overflow-auto p-4" : ""}>
            <ChatPage
              key={`chat-${task.id}`}
              taskId={task.id}
              taskTitle={task.title}
              seed={seedForChat}
              onOptimized={refresh}
            />
          </div>
        </div>
      </div>

      <div {...panel("spec")}>
        <SpecPanel
          key={`spec-${task.current_spec_version_id ?? 0}`}
          taskId={task.id}
          currentVersion={currentSpec?.version ?? null}
          onOpenChat={openChat}
        />
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,360px)]">
          <TaskConfigPanel task={task} onChanged={refresh} />
          <ContextPanel key={`ctx-${task.current_spec_version_id ?? 0}`} taskId={task.id} />
        </div>
      </div>

      <div {...panel("impl")}>
        <ImplementationPanel key={`impl-${gateSig}-${tab === "impl"}`} task={task} />
      </div>

      <div {...panel("results")}>
        <TaskCasePanel taskId={task.id} active={tab === "results"}
          revision={`${task.runs.length}-${task.cases_confirmed_at}-${task.cases_confirmed_sha256}`}
          onOpenTests={() => onOpenTests?.(task.id)} />
        <ResultsPanel task={task} />
      </div>

      <div {...panel("perf")}>
        <section className="card mb-4 flex flex-wrap items-center gap-3 px-4 py-3">
          <div>
            <h2 className="m-0 text-xs font-semibold">性能报告入口</h2>
            <div className="mt-0.5 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              基准 / 采集的机读报告与 Roofline 判定在「测试与结果」里
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <span className={`chip ${latestBenchmark ? "chip-ok" : ""}`}>
              benchmark {latestBenchmark ? `#${latestBenchmark.id}` : "未跑"}
            </span>
            <span className={`chip ${latestProfile ? "chip-ok" : ""}`}>
              profile {latestProfile ? `#${latestProfile.id}` : "未跑"}
            </span>
            <button className="btn btn-sm" onClick={() => setTab("results")}>
              打开结果视图
            </button>
          </div>
        </section>
        <OptimizationRoundPanel
          task={task}
          onChanged={refresh}
          onOptimizeChat={openOptimizeChat}
          onNewOptimizationChat={openNewOptimizationChat}
        />
      </div>

      <div {...panel("versions")}>
        <TaskVersionsPanel task={task} onChanged={refresh} />
        <SavepointPanel task={task} onChanged={refresh} />
      </div>

      <div {...panel("records")}>
        <TaskRecordsPanel task={task} />
        <DeliveryPanel
          key={`delivery-${task.runs.length}`}
          task={task}
          onOpenOverview={() => setTab("overview")}
          onChanged={refresh}
        />
      </div>
    </>
  );
}
