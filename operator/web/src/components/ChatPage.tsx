import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import Markdown from "./Markdown";
import type { AgentEvent, ChatAttachment, ChatSeed, Conversation, ConversationMessage } from "../types";

/** `GET /api/agent-jobs/{id}/live` 的返回（只取本页要用的字段）。 */
type LiveInfo = {
  running: boolean;
  exit_code?: number | null;
  final?: string;
  prose?: string;
  error?: string | null;
};

/** 对话页里的「自动优化」上下文（性能与优化 → 对话带入）：横幅状态机 ready → running → done/failed。 */
type OptimizeCtx = {
  loopId: number;
  label: string;
  focus: string | null;
  allowedPaths: string[];
  budget: number | null;
  /** 横幅所属会话（切到别的会话时清掉横幅，避免错位） */
  conversationId: number | null;
  status: "ready" | "running" | "done" | "failed";
  error: string | null;
  jobId: number | null;
};

/** SSE 事件：`error` 事件后端会带 `error` 字段，`AgentEvent` 类型里还没声明。 */
type ChatEvent = AgentEvent & { error?: string };

/** 断流后最多自动重连几次（1s/2s/4s/8s/15s 退避）。 */
const MAX_RECONNECT = 5;

/** `api.req` 抛的是 `${status} ${body}`，body 多半是 `{"detail": "…"}` —— 还原成人话。 */
function apiError(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  const sp = raw.indexOf(" ");
  if (sp < 0) return raw;
  const body = raw.slice(sp + 1);
  try {
    const data = JSON.parse(body) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
    if (data.detail) return JSON.stringify(data.detail);
  } catch {
    /* body 可能被截断 / 前后有杂字符：下面用正则抠 detail，别把整段 JSON 摊给用户 */
  }
  // 容错：从 `{"detail":"…"}` 里抠出 detail 并反转义（confirm 的多 issue 文案可能很长）
  const m = /"detail"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(body);
  if (m) {
    try {
      return JSON.parse(`"${m[1]}"`) as string;
    } catch {
      return m[1];
    }
  }
  return raw;
}

/** 从 `api.req` 的错误里取出 HTTP 状态码（没有就是 null）。 */
function statusOf(e: unknown): number | null {
  const m = /^(\d{3})\b/.exec(e instanceof Error ? e.message : String(e));
  return m ? Number(m[1]) : null;
}

/** 超时/取消/中断：这类「结束」不是失败，用黄色提示而非红色错误。 */
function isCancelReason(msg: string | null | undefined): boolean {
  return Boolean(msg && /取消|cancel|interrupt|终止|中断/i.test(msg));
}

/** 收尾兜底用的短等待（只在 finish 里用一次，不轮询）。 */
const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** 对话页（P0-2）：会话列表 + 对话 + 逐 token 流式。
 *
 * 引擎是后端 `services/agent_runner`（dsh headless + JSONL → SSE），本页只负责：
 * 建/选会话、发消息、订阅 `GET /api/agent-jobs/{job_id}/events`、落库后回显。
 * 模式：`task` = 任务工作区（挂 40 个 skills，可改代码）；`light` = 空目录（不挂 skills）。
 *
 * SSE 状态机与 agent_runner 事件流约定对齐：`done` 才收尾；断流先查 `/live`，
 * 仍在跑就退避重连（不把断流当完成）；刷新/切页后按会话的 `running_job` 接回。
 */
export default function ChatPage({
  taskId,
  taskTitle,
  seed,
  onOptimized,
}: {
  taskId: number;
  taskTitle?: string;
  /** 外部带入：概览/AI 审查文本，或「性能与优化」的自动优化上下文（见 ChatSeed）。 */
  seed?: ChatSeed | null;
  /** 自动优化 job 结束（兜底重拉会话后）通知父级刷新任务详情/轮次卡证据。 */
  onOptimized?: () => void;
}) {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [active, setActive] = useState<Conversation | null>(null);
  const [opening, setOpening] = useState(false);
  const [mode, setMode] = useState<"task" | "light">("task");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const [stream, setStream] = useState("");
  const [phase, setPhase] = useState("");
  const [tools, setTools] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  // 接回 / 取消 / 回退等中性提示（不是错误）
  const [note, setNote] = useState<string | null>(null);
  // 断流重连提示（断流但 job 仍在跑时显示，不当成完成）
  const [connNote, setConnNote] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deletingAtt, setDeletingAtt] = useState<number | null>(null);

  const esRef = useRef<EventSource | null>(null);
  const retryTimerRef = useRef<number | null>(null);
  const sessionRef = useRef(0);
  const activeIdRef = useRef<number | null>(null);
  const openSeqRef = useRef(0);
  const syncSeqRef = useRef(0);
  const mountedRef = useRef(true);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const seedRef = useRef(0);

  // 「自动优化」对话模式（性能与优化 → 对话带入）：banner 状态机 + 启动防重。
  // state 驱动渲染；ref 给 SSE/async 回调同步读最新值（闭包里的 state 会过期）。
  const [optimizeCtx, setOptimizeCtx] = useState<OptimizeCtx | null>(null);
  const optimizeRef = useRef<OptimizeCtx | null>(null);
  // 「对话在前、台账在后」：有值时，该会话的下一条消息作为新优化方向。
  const [newOptimizationConversationId, setNewOptimizationConversationId] = useState<number | null>(null);
  const optimizeErrorRef = useRef<string | null>(null);
  const optimizeStartingRef = useRef(false);
  const onOptimizedRef = useRef<(() => void) | undefined>(onOptimized);
  useEffect(() => {
    onOptimizedRef.current = onOptimized;
  });

  const loadConvs = useCallback(async () => {
    try {
      const rows = (await api.conversations()).filter((c) => c.task_id === taskId);
      setConvs(rows);
      return rows;
    } catch (e) {
      setErr(apiError(e));
      return null;
    }
  }, [taskId]);

  /** 关掉当前事件流 / 取消待重连定时器，并作废在途的 `/live` 探测回调。 */
  const closeStream = useCallback(() => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    esRef.current?.close();
    esRef.current = null;
    sessionRef.current += 1;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void loadConvs().then((rows) => {
      if (!mountedRef.current || !rows || activeIdRef.current !== null) return;
      // 刷新页面 / 从别页切回来：若还有会话在生成，自动打开它并接回 SSE（不用干等/重发被 409）
      const running = rows.find((c) => c.running_job != null);
      if (running) void open(running.id);
    });
    // 卸载必须关 EventSource + 清 timer，否则切页后回调还在改已卸载组件的 state
    return () => {
      mountedRef.current = false;
      closeStream();
    };
    // open 只用 ref/setState，不放进依赖（否则每次渲染都会重跑挂载逻辑）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadConvs, closeStream]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages?.length, stream, sending]);

  /** 把 `/live` 的结束信息落到页面：失败原因要显示出来，不能静默当成功。 */
  const applyLive = (info: LiveInfo) => {
    if (info.error) {
      if (isCancelReason(info.error)) setNote(`本轮已中断：${info.error}`);
      else setErr(`本轮失败：${info.error}`);
    } else if (info.exit_code != null && info.exit_code !== 0) {
      setErr(`本轮失败（dsh 退出码 ${info.exit_code}），回答可能不完整`);
    }
  };

  /** 同步 banner 状态：ref 给 SSE/async 回调读最新值，state 驱动渲染。 */
  const applyOptimize = useCallback((next: OptimizeCtx | null) => {
    optimizeRef.current = next;
    setOptimizeCtx(next);
  }, []);

  /**
   * 优化 job 终态：banner running → done/failed（只在 running 时生效，重复调用幂等）。
   * 返回 true = 本次确实完成了自动优化（调用方据此让 finish 做兜底重拉 + onOptimized）。
   */
  const markOptimizeFinished = useCallback(
    (ok: boolean, errorMessage?: string | null): boolean => {
      const cur = optimizeRef.current;
      if (!cur || cur.status !== "running") return false;
      applyOptimize({
        ...cur,
        status: ok ? "done" : "failed",
        error: ok ? null : errorMessage || "自动优化失败（未拿到具体原因）",
      });
      return true;
    },
    [applyOptimize],
  );

  /** `/live` 的结束信息 → 优化失败原因（取消/中断不算失败）。 */
  const liveFailure = (info: LiveInfo): string | null => {
    if (info.error && !isCancelReason(info.error)) return info.error;
    if (info.exit_code != null && info.exit_code !== 0) return `dsh 退出码 ${info.exit_code}`;
    return null;
  };

  /**
   * 收尾：拉回会话详情（消息以服务端落库为准）；晚到的旧响应不覆盖当前会话。
   *
   * 竞态兜底：agent_runner 的顺序是「SSE 发 done/关流 → 持久化 job → on_done 回填」，
   * 自动优化 job 的 on_done 还要做工作区快照/落证据，可能比普通对话更慢 —— 刚收到 done
   * 时拉回来的会话可能最后一条仍是 user。若本次是自动优化、或最后一条仍是 user/拉取失败，
   * 延迟 ~1s 再拉一次（只重试一次，不轮询）；重拉仍用 convId + seq + session + mounted 守卫，
   * 避免晚到响应覆盖用户刚发的新消息。`wasOptimize` 由调用方在收尾时传入（done 事件本身
   * 不带 info，失败/取消判定可能晚于 finish 的第一次拉取）。
   */
  const finish = useCallback(
    async (convId: number, wasOptimize = false) => {
      const seq = ++syncSeqRef.current;
      const session = sessionRef.current;
      setSending(false);
      setJobId(null);
      setPhase("");
      setTools(0);
      setConnNote(null);
      const stillCurrent = () =>
        mountedRef.current && seq === syncSeqRef.current && session === sessionRef.current;
      let conv: Conversation | null = null;
      try {
        conv = await api.conversation(convId);
      } catch {
        conv = null;
      }
      if (!stillCurrent()) return;
      const lastMsg = conv?.messages?.[conv.messages.length - 1];
      if (wasOptimize || !conv || lastMsg?.role === "user") {
        await sleep(1000);
        if (!stillCurrent()) return;
        try {
          const again = await api.conversation(convId);
          if (stillCurrent()) conv = again;
        } catch {
          /* 重拉失败：保留第一次结果 */
        }
      }
      if (!stillCurrent()) return;
      if (activeIdRef.current === convId) {
        if (conv) setActive(conv);
        setStream("");
      }
      // task 对话可能在任意轮次里落规格、确认用例或跑动作；不能只在
      // 自动优化结束时刷新父级，否则门禁/版本/用例 chip 会停在旧值（审查 H7）。
      onOptimizedRef.current?.();
      void loadConvs();
    },
    [loadConvs],
  );

  /** 订阅一个 job 的事件流：新发送、断流重连、刷新接回都走同一条路径。
   *
   * `attempt > 0` / `resume`：后端会把 `state.events` 从头重放，此时**不能继续 append**
   * （否则重复上屏）。用重放累加器收敛：显示先保持已上屏文本，重放内容比它长时才替换，
   * 最终等于后端 `live.prose`。
   */
  function attach(job: number, convId: number, attempt = 0, resume = false) {
    closeStream();
    const session = sessionRef.current;
    const replay = resume || attempt > 0;
    let replayText = "";
    let finished = false;
    let probing = false;
    const es = new EventSource(api.eventsUrl(job));
    esRef.current = es;
    if (attempt > 0 && !resume) setPhase("已重新连接，正在重放事件…");

    /** 真正的结束：收到 `done`，或 `/live` 明确不再运行、job 已不存在。 */
    const stop = (info?: LiveInfo) => {
      if (finished) return;
      finished = true;
      closeStream();
      setSending(false);
      setJobId(null);
      setPhase("");
      setConnNote(null);
      setNote(null);
      // 这次收尾是否属于自动优化（done 事件不带 info，失败/取消判定可能晚于下面的 /live 探测；
      // 用收尾瞬间的 banner 状态作为提示传给 finish，保证兜底重拉 + onOptimized 不被漏掉）
      const wasOptimize = optimizeRef.current?.status === "running";
      if (info) {
        applyLive(info);
        const fail = liveFailure(info) ?? optimizeErrorRef.current;
        markOptimizeFinished(!fail, fail);
      } else {
        // `done` 只说明流结束、不带退出码：补一次 `/live` 区分失败/中断（与任务页一致）
        const sess = sessionRef.current;
        void api
          .agentLive(job)
          .then((live) => {
            if (sess !== sessionRef.current) return;
            applyLive(live);
            const fail = liveFailure(live) ?? optimizeErrorRef.current;
            markOptimizeFinished(!fail, fail);
          })
          .catch(() => {
            const fail = optimizeErrorRef.current;
            markOptimizeFinished(!fail, fail);
          });
      }
      void finish(convId, wasOptimize);
    };

    /** 断流但 job 可能仍在跑：退避重连，绝不把断流当完成。 */
    const reconnect = (reason: string) => {
      if (finished) return;
      if (attempt >= MAX_RECONNECT) {
        finished = true;
        closeStream();
        setPhase("");
        // 保持 sending=true：已上屏正文不能清、也不能假装完成；用户可点「停止」或刷新接回
        setConnNote(
          `连接中断，已停止自动重连${reason ? `（${reason}）` : ""}；job #${job} 可能仍在运行 —— 可点「停止」，或刷新页面接回`,
        );
        return;
      }
      const delay = Math.min(1000 * 2 ** attempt, 15000);
      setConnNote(`连接中断，仍在运行，${Math.round(delay / 1000)}s 后自动重连…`);
      setPhase("");
      retryTimerRef.current = window.setTimeout(() => {
        retryTimerRef.current = null;
        if (finished || sessionRef.current !== session) return;
        attach(job, convId, attempt + 1);
      }, delay);
    };

    es.onmessage = (m) => {
      if (finished) return;
      setConnNote(null);
      let ev: ChatEvent;
      try {
        ev = JSON.parse(m.data) as ChatEvent;
      } catch {
        return;
      }

      // 逐 token 正文（打字机）
      if (ev.type === "delta" && ev.kind === "text" && ev.text) {
        setPhase("");
        if (replay) {
          replayText += ev.text;
          setStream((prev) => (replayText.length >= prev.length ? replayText : prev));
        } else {
          setStream((prev) => prev + ev.text);
        }
        return;
      }
      // 没挂上逐 token 插件时（stream=false）只有整段 `final`；有 delta 时以后端落库为准
      if (ev.type === "final") {
        if (ev.text) setStream((prev) => (prev ? prev : ev.text!));
        return;
      }
      if (ev.type === "status") {
        if (ev.phase === "turn_start") setPhase("思考中…");
        else if (ev.phase === "step_end") setPhase("");
        return;
      }
      if (ev.type === "tool_call") {
        setTools((n) => n + 1);
        setPhase("工具调用中…");
        return;
      }
      if (ev.type === "turn_end" && ev.reason?.kind === "error") {
        const msg = ev.reason.error?.message ?? "本轮出错";
        if (!isCancelReason(msg) && optimizeRef.current?.status === "running") optimizeErrorRef.current = msg;
        if (isCancelReason(msg)) setNote(`本轮已中断：${msg}`);
        else setErr(`本轮出错：${msg}`);
        return;
      }
      // 后端在收尾时 publish 的 error 事件（超时/取消/进程错误）—— 之前这里被忽略了
      if (ev.type === "error") {
        const msg = ev.error ?? "未知错误";
        if (!isCancelReason(msg) && optimizeRef.current?.status === "running") optimizeErrorRef.current = msg;
        if (isCancelReason(msg)) setNote(`本轮已中断：${msg}`);
        else setErr(`本轮出错：${msg}`);
      }
    };

    es.addEventListener("done", () => stop());

    es.onerror = () => {
      if (finished || probing) return;
      // EventSource 自己会重连但可能丢/重放事件：主动关掉，用 `/live` 判断运行状态再决定接回
      probing = true;
      es.close();
      if (esRef.current === es) esRef.current = null;
      setPhase("连接中断，正在确认运行状态…");
      void api
        .agentLive(job)
        .then((info) => {
          if (finished || sessionRef.current !== session) return;
          probing = false;
          if (info.running) {
            reconnect("");
            return;
          }
          if (!("exit_code" in info)) {
            // job 不在本进程运行表里（控制台重启 / 已淘汰）：不假装完成
            stop();
            setNote(`控制台已重启或 job #${job} 不存在，进度可能不完整；已刷新历史，请刷新页面确认`);
            return;
          }
          stop(info);
        })
        .catch((e) => {
          if (finished || sessionRef.current !== session) return;
          probing = false;
          if (statusOf(e) === 404) {
            stop();
            setNote(`控制台已重启或 job #${job} 不存在，进度可能不完整；已刷新历史，请刷新页面确认`);
            return;
          }
          reconnect("状态查询失败");
        });
    };
  }

  /** 接回一个仍在跑的 job（刷新/切页/409 后）：先用 `/live` 的 prose 当已上屏文本，再订阅。 */
  const attachRunning = async (convId: number, job: number) => {
    setSending(true);
    setJobId(job);
    setTools(0);
    setStream("");
    setConnNote(null);
    setPhase("已接回运行中的回答，正在同步…");
    try {
      const live = await api.agentLive(job);
      if (!mountedRef.current || activeIdRef.current !== convId) return;
      if (live.running) {
        if (live.prose) setStream(live.prose);
        setPhase("");
        setNote(`已接回运行中的回答（job #${job}）`);
        attach(job, convId, 1, true);
        return;
      }
      applyLive(live);
      // 接回时 job 其实已结束（seed 判定 running 后瞬间跑完）：banner 置终态，别卡在「运行中」
      const fail = liveFailure(live) ?? optimizeErrorRef.current;
      const wasOptimize = markOptimizeFinished(!fail, fail);
      setNote((prev) => prev ?? `job #${job} 已结束，已刷新历史`);
      void finish(convId, wasOptimize);
    } catch (e) {
      if (!mountedRef.current || activeIdRef.current !== convId) return;
      setNote(`无法确认 job #${job} 的运行状态（${apiError(e)}），仍尝试订阅事件…`);
      attach(job, convId, 1, true);
    }
  };

  const open = async (id: number) => {
    if (id === activeIdRef.current) return;
    const seq = ++openSeqRef.current;
    // 自动优化横幅只属于它自己的会话：切到别的会话时清掉，避免上下文错位
    const opt = optimizeRef.current;
    if (opt && opt.conversationId !== id) {
      applyOptimize(null);
      optimizeErrorRef.current = null;
    }
    if (newOptimizationConversationId != null && newOptimizationConversationId !== id) {
      setNewOptimizationConversationId(null);
    }
    // 切会话前先关旧流、清定时器，别让旧 job 的回调改新会话的界面
    closeStream();
    setJobId(null);
    setSending(false);
    setPhase("");
    setStream("");
    setTools(0);
    setConnNote(null);
    setNote(null);
    setErr(null);
    setOpening(true);
    setActiveId(id);
    activeIdRef.current = id;
    setActive(null);
    try {
      const conv = await api.conversation(id);
      // 晚到的旧响应不能覆盖当前会话
      if (!mountedRef.current || seq !== openSeqRef.current || activeIdRef.current !== id) return;
      setActive(conv);
      // 刷新/切页回来：会话还有 running_job 就自动接回 SSE
      if (conv.running_job != null) void attachRunning(id, conv.running_job);
    } catch (e) {
      if (seq === openSeqRef.current) setErr(apiError(e));
    } finally {
      if (seq === openSeqRef.current) setOpening(false);
    }
  };

  const create = async () => {
    setErr(null);
    setNote(null);
    try {
      const conv = await api.createConversation({ task_id: taskId, mode });
      await loadConvs();
      await open(conv.id);
    } catch (e) {
      setErr(apiError(e));
    }
  };

  const armNewOptimization = () => {
    if (!active || active.mode !== "task" || sending) return;
    applyOptimize(null);
    optimizeErrorRef.current = null;
    setNewOptimizationConversationId(active.id);
    setInput("");
    setErr(null);
    setNote("请说明优化对象、期望方向和不能退化的约束；发送后会自动记录到「性能与优化」。");
    window.setTimeout(() => inputRef.current?.focus(), 50);
  };

  // 外部带入：概览/AI 审查（预填文本，不自动发送）或「性能与优化」的自动优化上下文（回车即启动）。
  useEffect(() => {
    if (!seed) return;
    if (seedRef.current === seed.nonce) return;
    seedRef.current = seed.nonce;
    const myNonce = seed.nonce;
    const opt = seed.optimize ?? null;
    const newOptimization = Boolean(seed.newOptimization);
    void (async () => {
      try {
        // 优先 seed 指定的会话（必须属于本任务），否则沿用当前会话 / 最近会话 / 新建
        // 列表读取失败不阻断带入：有活动会话就沿用，否则按新建处理（loadConvs 会另报错误）
        let rows: Conversation[] = [];
        try {
          rows = (await api.conversations()).filter((c) => c.task_id === taskId);
        } catch {
          rows = [];
        }
        // 期间又来了新 seed：旧的那次带入直接放弃，别抢新 seed 的会话/输入框
        if (!mountedRef.current || seedRef.current !== myNonce) return;
        const pick = (id: number | null): number | null =>
          id != null && rows.some((c) => c.id === id) ? id : null;
        let convId = pick(seed.conversationId ?? null) ?? pick(activeIdRef.current);
        if (convId == null && rows.length > 0) convId = rows[0].id;
        // 自动优化只能挂在任务工作区会话（mode=task）：选中的是「轻对话」就另建一个 task 会话
        if ((opt || newOptimization) && rows.find((c) => c.id === convId)?.mode !== "task") {
          const conv = await api.createConversation({ task_id: taskId, mode: "task" });
          if (!mountedRef.current || seedRef.current !== myNonce) return;
          convId = conv.id;
        }
        if (convId == null) {
          const conv = await api.createConversation({ task_id: taskId, mode: "task" });
          if (!mountedRef.current || seedRef.current !== myNonce) return;
          convId = conv.id;
        }
        // 先刷新左侧会话列表：从概览「开发 / AI 审查」跳进来时新会话可能刚建出来，
        // 只 open 不 load 会导致左侧一开始是空的，重进页面才显示（用户反馈的 bug）。
        void loadConvs();
        await open(convId);
        if (!mountedRef.current || seedRef.current !== myNonce) return;
        if (opt) {
          setNewOptimizationConversationId(null);
          // 该会话若已有 running job（刷新/重新进入），open 已接回 SSE；banner 直接显示运行中。
          // seed 里还带着任务详情中该轮次的优化 job id，作为列表 running_job 的时序兜底。
          const rowRunning = rows.find((c) => c.id === convId)?.running_job ?? null;
          const runningJobId = rowRunning ?? opt.jobId ?? null;
          optimizeErrorRef.current = null;
          applyOptimize({
            loopId: opt.loopId,
            label: opt.label,
            focus: opt.focus ?? null,
            allowedPaths: opt.allowedPaths ?? [],
            budget: opt.budget ?? null,
            conversationId: convId,
            // 「在对话中查看」（带 conversationId）默认只看历史，banner 记 done，回车走普通对话；
            // 从「自动优化」按钮进来（无 conversationId）才是 ready（回车启动）。
            status: runningJobId != null
              ? "running"
              : seed.conversationId != null
                ? "done"
                : "ready",
            jobId: runningJobId,
            error: null,
          });
          setInput("");
          setNote(null);
          if (rowRunning == null && opt.jobId != null) {
            // 列表 running_job 的时序兜底：用 seed 的优化 job id 接入 SSE
            void attachRunning(convId, opt.jobId);
          }
        } else if (newOptimization) {
          applyOptimize(null);
          optimizeErrorRef.current = null;
          setNewOptimizationConversationId(convId);
          setInput("");
          setNote("请直接说优化方向；发送后才会记录到「性能与优化」页并安全启动 Agent。");
        } else {
          applyOptimize(null);
          setNewOptimizationConversationId(null);
          setInput(seed.text);
          setNote("已把执行结果带进输入框（可编辑后发送）");
        }
        window.setTimeout(() => inputRef.current?.focus(), 50);
      } catch (e) {
        if (mountedRef.current) setErr(apiError(e));
      }
    })();
    // seed.nonce 变化 = 新的一次带入；open / loadConvs 都是稳定使用 ref 的逻辑
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed?.nonce, taskId]);

  /** 普通对话发送（banner ready 时不会走这里，见下方 send 调度）。 */
  const sendChat = async () => {
    if (!active || !input.trim() || sending) return;
    const draft = input;
    const text = draft.trim();
    const convId = active.id;
    setInput("");
    setErr(null);
    setNote(null);
    setConnNote(null);
    setSending(true);
    setStream("");
    setPhase("已提交，等待第一段…");
    setTools(0);
    // 乐观回显用户消息（服务端也会落库，完成后以服务端为准）
    const optimistic: ConversationMessage = {
      id: -Date.now(),
      role: "user",
      text,
      job_id: null,
      created_at: null,
    };
    setActive((prev) =>
      prev && prev.id === convId ? { ...prev, messages: [...(prev.messages ?? []), optimistic] } : prev,
    );
    try {
      const res = await api.sendMessage(convId, text);
      if (!mountedRef.current || activeIdRef.current !== convId) return;
      setJobId(res.job_id);
      attach(res.job_id, convId);
    } catch (e) {
      const msg = apiError(e);
      // 失败回滚：乐观消息不能留着「已发送」，输入框把草稿还回去
      setActive((prev) =>
        prev && prev.id === convId
          ? { ...prev, messages: (prev.messages ?? []).filter((m) => m.id !== optimistic.id) }
          : prev,
      );
      setInput(draft);
      setSending(false);
      setPhase("");
      setStream("");
      setErr(msg);
      // 409 常见原因：上一轮还在跑 —— 后端给了 job # 就直接接回它，不用干等/反复重发
      if (statusOf(e) === 409 && !isCancelReason(msg) && activeIdRef.current === convId) {
        const m = /job\s*#(\d+)/i.exec(msg);
        if (m) void attachRunning(convId, Number(m[1]));
      }
    }
  };

  /**
   * 新优化的主入口：用户先在任务对话发一条方向，后端再落为
   * conversation message + optimization idea + Loop。基线齐全时直接接回
   * kind=优化 job 的 SSE；不齐时展示后端的明确停点，不会先改代码。
   */
  const startNewOptimization = async () => {
    if (!active || active.id !== newOptimizationConversationId || active.mode !== "task" || sending) return;
    const direction = input.trim();
    if (!direction) return;
    const convId = active.id;
    const draft = input;
    setInput("");
    setErr(null);
    setNote(null);
    setConnNote(null);
    setSending(true);
    setStream("");
    setPhase("正在识别优化方向并创建安全验证链…");
    setTools(0);
    const optimistic: ConversationMessage = {
      id: -Date.now(),
      role: "user",
      text: direction,
      job_id: null,
      created_at: null,
    };
    setActive((prev) =>
      prev && prev.id === convId ? { ...prev, messages: [...(prev.messages ?? []), optimistic] } : prev,
    );
    try {
      const result = await api.startOptimizationConversation(taskId, direction, convId);
      if (!mountedRef.current || activeIdRef.current !== convId) return;
      setNewOptimizationConversationId(null);
      onOptimizedRef.current?.();
      void loadConvs();
      if (result.status === "started" && result.loop && result.job_id != null) {
        const obj = result.loop.objective ?? {};
        optimizeErrorRef.current = null;
        applyOptimize({
          loopId: result.loop.id,
          label: obj.label || result.loop.scope || `轮次 #${result.loop.id}`,
          focus: obj.focus ?? direction,
          allowedPaths: obj.constraints?.allowed_paths ?? [],
          budget: obj.budget?.runs_per_action ?? null,
          conversationId: convId,
          status: "running",
          error: null,
          jobId: result.job_id,
        });
        setJobId(result.job_id);
        setPhase("已识别方向，Agent 正在按新 Loop 优化…");
        attach(result.job_id, convId);
        return;
      }

      setSending(false);
      setPhase("");
      setStream("");
      setNote(result.reason);
      // waiting_baseline / waiting_previous / waiting_agent 都已在后端落消息，以服务端历史为准。
      const conv = await api.conversation(convId);
      if (mountedRef.current && activeIdRef.current === convId) setActive(conv);
    } catch (e) {
      if (!mountedRef.current || activeIdRef.current !== convId) return;
      setActive((prev) =>
        prev && prev.id === convId
          ? { ...prev, messages: (prev.messages ?? []).filter((m) => m.id !== optimistic.id) }
          : prev,
      );
      setInput(draft);
      setSending(false);
      setPhase("");
      setStream("");
      setErr(apiError(e));
    }
  };

  /**

  /**
   * 自动优化对话化（两步、可审计）：banner ready 时回车/点「开始自动优化」
   * → `POST /loops/{id}/confirm`（冻结草案 + 一次性 token）→ `POST /loops/{id}/start`（消费 token 起 kind=优化 job），
   * 复用现有 attach() 的 SSE 展示过程；允许空指令（= 纯按本轮目标）。
   * confirm 的 409 多为前置条件：E_BASELINE_*（基线未就绪）/ E_WORKSPACE_BUSY / E_CONFIRM_BUSY（工作区有作业在跑）；
   * E_CONFIRM_TOKEN / OBJECTIVE_CHANGED / BASELINE_CHANGED 主要出在 start（token 过期/已用/目标或基线变了）。
   * 任一步失败：回滚乐观消息、恢复输入、banner 回 ready 并显示 detail；只有 start 的 409 带 job # 时才接回运行中的 job。
   */
  const startOptimize = async () => {
    const ctx = optimizeRef.current;
    if (!ctx || ctx.status !== "ready" || sending || optimizeStartingRef.current) return;
    const convId = activeIdRef.current ?? ctx.conversationId;
    if (convId == null) {
      setErr("请先新建或选择一个会话，再开始自动优化。");
      return;
    }
    optimizeStartingRef.current = true;
    const instruction = input.trim();
    const draft = input;
    setInput("");
    setErr(null);
    setNote(null);
    setConnNote(null);
    setSending(true);
    setStream("");
    setPhase("已提交自动优化，等待 Agent 开始…");
    setTools(0);
    optimizeErrorRef.current = null;
    applyOptimize({ ...ctx, status: "running", conversationId: convId, jobId: null, error: null });
    // 乐观回显 user 消息（空指令时给占位文本；服务端也会落一条 user 消息，完成后以服务端为准）
    const optimistic: ConversationMessage = {
      id: -Date.now(),
      role: "user",
      text: instruction || "（按本轮目标开始自动优化）",
      job_id: null,
      created_at: null,
    };
    setActive((prev) =>
      prev && prev.id === convId ? { ...prev, messages: [...(prev.messages ?? []), optimistic] } : prev,
    );
    let failedAt: "confirm" | "start" = "confirm";
    try {
      // 两步可审计启动：先 confirm（服务端冻结本轮「确认的优化草案」并发放一次性 token，10 分钟），
      // 再 start（消费 token 起 agent job）。任一步失败都走下面同一套回滚（乐观消息/草稿/banner）。
      // confirm 失败常见 409：E_BASELINE_*（基线未就绪）/ E_WORKSPACE_BUSY / E_CONFIRM_BUSY（工作区有作业在跑）。
      const confirmed = await api.confirmLoop(ctx.loopId, { instruction: instruction || null });
      failedAt = "start";
      setPhase("已确认优化草案，正在启动 Agent…");
      const res = await api.startLoop(ctx.loopId, {
        token: confirmed.token,
        conversation_id: convId,
      });
      if (!mountedRef.current) return;
      if (activeIdRef.current !== convId) {
        // 等待期间用户切走了会话：job 已在后台会话里起来。这里绝不能动页级共享状态
        // （sending/jobId/phase/stream 属于用户刚切到的会话，可能正挂着它自己的 running job）。
        void loadConvs();
        setNote(`已在会话 #${res.conversation_id} 启动自动优化，切回该会话可查看。`);
        return;
      }
      // start 期间 banner 可能已被另一个轮次/会话接管：只认本轮次（+本会话）的 banner，
      // 否则宁可只刷新会话列表，也不要把 B 的 banner 标成 A 的 job。
      const running = optimizeRef.current;
      const bannerMismatch =
        running != null &&
        (running.loopId !== ctx.loopId ||
          (running.conversationId != null && running.conversationId !== res.conversation_id));
      if (bannerMismatch) {
        void loadConvs();
        setNote(`已在会话 #${res.conversation_id} 启动自动优化；横幅已切换，可切回该会话查看。`);
        return;
      }
      if (running) {
        applyOptimize({
          ...running,
          status: "running",
          conversationId: res.conversation_id,
          jobId: res.job_id,
        });
      }
      setJobId(res.job_id);
      attach(res.job_id, res.conversation_id);
      void loadConvs();
    } catch (e) {
      if (!mountedRef.current) return;
      // confirm / start 哪一步失败分开提示（detail 原样保留，如 E_CONFIRM_* / E_BASELINE_* / token 失效）
      const detail = apiError(e);
      // 工作区忙（confirm 的 409 E_CONFIRM_BUSY）：给可行动提示。detail 里的 agent#N 是 AgentJob id，
      // 与 ActionJob/会话 job 不同表，不能当成「job #」去 attach。
      const confirmBusy = failedAt === "confirm" && detail.includes("E_CONFIRM_BUSY");
      const agentRef = /agent#(\d+)/i.exec(detail);
      const msg = confirmBusy
        ? `任务还有作业在跑（${agentRef ? `agent#${agentRef[1]}` : "工作区作业"}），等它结束或去「任务」页取消后再试。`
        : failedAt === "confirm"
          ? `确认优化草案失败：${detail}`
          : `启动自动优化失败：${detail}`;
      // 等待期间用户切走了会话：只报错，不动新会话的输入/乐观消息/页级共享状态
      // （用户刚切到的会话可能正挂着自己的 job，不能把它的「停止」按钮/流式状态抹掉）。
      if (activeIdRef.current !== convId) {
        setErr(msg);
        return;
      }
      // 失败回滚：乐观消息不能留着「已发送」；草稿与 banner（ready）都还回去
      setActive((prev) =>
        prev && prev.id === convId
          ? { ...prev, messages: (prev.messages ?? []).filter((m) => m.id !== optimistic.id) }
          : prev,
      );
      setInput(draft);
      setSending(false);
      setPhase("");
      setStream("");
      // 只有 banner 仍是本轮次/本会话的才回 ready，别把别的轮次的 banner 改成 ready
      const back = optimizeRef.current;
      if (back && back.loopId === ctx.loopId && (back.conversationId == null || back.conversationId === convId)) {
        applyOptimize({ ...back, status: "ready", jobId: null, error: null });
      }
      setErr(msg);
      // 只有 start 的 409 可能指向可接回的「job #」；confirm 的 409（E_BASELINE_*/E_WORKSPACE_BUSY/
      // E_CONFIRM_BUSY）是前置条件，没有可接回的 job —— 尤其 E_CONFIRM_BUSY 的 agent#N 不是会话 job。
      if (failedAt === "start" && statusOf(e) === 409 && !isCancelReason(msg)) {
        const m = /job\s*#(\d+)/i.exec(msg);
        if (m) {
          const jobNum = Number(m[1]);
          try {
            const rows = await api.conversations();
            if (!mountedRef.current) return;
            const owner = rows.find((c) => c.task_id === taskId && c.running_job === jobNum);
            if (owner && owner.id === activeIdRef.current) {
              setErr(null);
              const cur = optimizeRef.current;
              if (cur && cur.loopId === ctx.loopId) applyOptimize({ ...cur, status: "running", jobId: jobNum });
              void attachRunning(owner.id, jobNum);
            } else if (owner) {
              setErr(`${msg}（job #${jobNum} 在会话「${owner.title ?? `#${owner.id}`}」里）`);
              void loadConvs();
            }
          } catch {
            /* 查不到所属会话就保留原错误提示 */
          }
        }
      }
    } finally {
      optimizeStartingRef.current = false;
    }
  };

  /** 查看历史优化时默认不 armed：点一下才切到 ready（回车才会再次启动，避免误触重跑）。 */
  const armOptimize = () => {
    const cur = optimizeRef.current;
    if (!cur || sending) return;
    applyOptimize({ ...cur, status: "ready", jobId: null, error: null });
    setNote("已切换为「重新自动优化」：输入参考指令后回车开始。");
    window.setTimeout(() => inputRef.current?.focus(), 50);
  };

  /** 输入框发送入口：banner ready 时启动自动优化（允许空输入），其它情况走普通对话。 */
  const send = async () => {
    if (sending) return;
    if (active?.id === newOptimizationConversationId) {
      await startNewOptimization();
      return;
    }
    if (optimizeCtx?.status === "ready") {
      await startOptimize();
      return;
    }
    await sendChat();
  };

  /** 停止本轮（后端 kill dsh 进程组并落 interrupted；SSE 随后会收到 done）。 */
  const cancel = async () => {
    const convId = activeIdRef.current;
    if (jobId == null || cancelling || convId == null) return;
    setCancelling(true);
    setErr(null);
    setNote("已发送停止请求，等待进程退出…");
    try {
      const r = await api.cancelAgentJob(jobId);
      setNote(r.result || "已发送停止，等待进程退出…");
    } catch (e) {
      const msg = apiError(e);
      if (statusOf(e) === 409) {
        // 作业其实已经结束（可能断流丢了 done）：直接刷新收尾，别一直卡在「生成中」
        setNote(msg);
        closeStream();
        const wasOptimize = markOptimizeFinished(true, null);
        await finish(convId, wasOptimize);
      } else {
        setErr(`取消失败：${msg}`);
      }
    } finally {
      setCancelling(false);
    }
  };

  const upload = async (f: File) => {
    if (!active || sending) return;
    const convId = active.id;
    setUploading(true);
    setErr(null);
    try {
      await api.uploadAttachment(convId, f);
      const conv = await api.conversation(convId);
      if (mountedRef.current && activeIdRef.current === convId) setActive(conv);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const removeAttachment = async (a: ChatAttachment) => {
    if (!active || sending || deletingAtt !== null) return;
    const convId = active.id;
    setDeletingAtt(a.id);
    setErr(null);
    try {
      await api.deleteAttachment(convId, a.id);
      // 附件会拼进之后每一轮 prompt，删除后必须重载会话（以服务端为准）
      const conv = await api.conversation(convId);
      if (mountedRef.current && activeIdRef.current === convId) setActive(conv);
      setNote(`已删除附件 ${a.filename}；之后的轮次不会再带进 prompt`);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setDeletingAtt(null);
    }
  };

  const revert = async (messageId: number) => {
    if (!active || sending) return;
    const convId = active.id;
    setErr(null);
    setNote(null);
    try {
      const r = await api.revertMessage(convId, messageId);
      const failed = Object.entries(r.files).filter(
        ([, v]) => !v.startsWith("restored") && !v.startsWith("removed"),
      );
      if (failed.length) {
        setErr(`部分文件未能回退：${failed.map(([k, v]) => `${k}（${v}）`).join("、")}`);
      } else {
        setNote("已还原本轮文件改动（只还原文件，对话记录与 agent 会话上下文未回滚）");
      }
      const conv = await api.conversation(convId);
      if (mountedRef.current && activeIdRef.current === convId) setActive(conv);
    } catch (e) {
      // 409 时后端 detail 会说明冲突（例如当前文件已被后续轮改过）
      setErr(apiError(e));
    }
  };

  const remove = async (id: number) => {
    const row = convs.find((c) => c.id === id);
    if (row?.running_job != null) {
      if (!window.confirm(`会话 #${id} 还有作业在跑（job #${row.running_job}），删除会丢掉本轮结果。仍要删除？`)) return;
    } else if (!window.confirm(`删除会话 #${id}？对话记录会一并删除。`)) {
      return;
    }
    setErr(null);
    try {
      await api.deleteConversation(id);
      if (activeIdRef.current === id) {
        closeStream();
        activeIdRef.current = null;
        setActiveId(null);
        setActive(null);
        applyOptimize(null);
        setNewOptimizationConversationId(null);
        optimizeErrorRef.current = null;
        setSending(false);
        setJobId(null);
        setStream("");
        setPhase("");
        setTools(0);
        setConnNote(null);
        setNote(null);
      }
      await loadConvs();
    } catch (e) {
      setErr(apiError(e));
    }
  };

  const messages = active?.messages ?? [];
  const newOptimizationReady = active?.id === newOptimizationConversationId;
  /** 自动优化待启动：回车/发送键改为「开始自动优化」（允许空指令）。 */
  const optimizeReady = optimizeCtx?.status === "ready";

  return (
    <div className="grid gap-4" style={{ gridTemplateColumns: "280px minmax(0, 1fr)" }}>
      <aside className="card p-3" style={{ maxHeight: "calc(100vh - 190px)", overflow: "auto" }}>
        <div className="text-xs font-semibold">会话</div>
        <div className="mt-2 flex flex-col gap-2">
          <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            本任务会话{taskTitle ? ` · ${taskTitle}` : ""}
          </div>
          <div className="flex items-center gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === "task"} onChange={() => setMode("task")} />
              挂 skills
            </label>
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === "light"} onChange={() => setMode("light")} />
              轻对话
            </label>
          </div>
          <button className="btn btn-primary" onClick={() => void create()}>
            + 新建对话
          </button>
        </div>
        <div className="mt-3" style={{ borderTop: "1px solid var(--border)", paddingTop: "8px" }}>
          {convs.length === 0 && (
            <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              还没有会话
            </div>
          )}
          {convs.map((c) => (
            <div
              key={c.id}
              className="mb-1 flex items-center gap-2 rounded-[8px] px-2 py-1 text-[11px]"
              style={{
                background: activeId === c.id ? "var(--panel-2)" : "transparent",
                border: "1px solid " + (activeId === c.id ? "var(--border)" : "transparent"),
              }}
            >
              <button className="min-w-0 flex-1 text-left" onClick={() => void open(c.id)}>
                <div className="flex items-center gap-1">
                  <span className="truncate">{c.title ?? `会话 #${c.id}`}</span>
                  {c.running_job != null && (
                    <span className="chip chip-run !py-0 !text-[9px]" title={`job #${c.running_job} 在跑`}>
                      运行中
                    </span>
                  )}
                </div>
                <div className="truncate text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  {c.mode === "light" ? "轻对话" : "挂 skills"} · {c.message_count} 条
                </div>
              </button>
              <button className="text-[10px]" style={{ color: "var(--fg-mute)" }} onClick={() => void remove(c.id)}>
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="card flex min-h-[520px] flex-col">
        <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="text-xs font-semibold">{active?.title ?? "选择一个会话，或新建"}</div>
          {active && (
            <>
              <span className={`chip ${active.mode === "light" ? "chip-warn" : "chip-ok"}`}>
                {active.mode === "light" ? "轻对话（空目录，提示词约束不碰工作区）" : "任务工作区（挂 skills）"}
              </span>
              <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                {active.task_title} · {active.message_count} 条 · {active.tokens} tokens
              </span>
              {active.dsh_session_id && (
                <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  session {active.dsh_session_id.slice(0, 18)}…
                </span>
              )}
              <button
                className="btn btn-primary btn-sm whitespace-nowrap"
                disabled={sending || active.mode !== "task"}
                title={active.mode === "task" ? "下一条消息按优化方向识别，并记录到性能与优化页" : "轻对话不能改任务工作区"}
                onClick={armNewOptimization}
              >
                提出优化方向
              </button>
              {sending && (
                <span className={`chip ${connNote ? "chip-warn" : "chip-run"}`}>
                  {connNote ? "连接中断" : "生成中"}
                </span>
              )}
              {jobId != null && (
                <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  job #{jobId}
                </span>
              )}
              {(active.attachments ?? []).length > 0 && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  附件 {active.attachments!.length} 个
                </span>
              )}
            </>
          )}
        </div>

        {newOptimizationReady && (
          <div
            className="mx-4 mt-2 rounded-[9px] px-3 py-2 text-[11px]"
            style={{ background: "var(--panel-2)", border: "1px solid var(--accent)" }}
          >
            <div className="flex flex-wrap items-center gap-2">
              <b>新优化·先对话</b>
              <span className="chip !py-0 !text-[10px]">等待你说方向</span>
              <button
                className="btn btn-sm ml-auto whitespace-nowrap"
                disabled={sending}
                onClick={() => setNewOptimizationConversationId(null)}
              >
                退出优化模式
              </button>
            </div>
            <div className="mt-1 leading-relaxed" style={{ color: "var(--fg-dim)" }}>
              直接说你想优化什么。系统会先把这条消息识别为优化点，再记录 idea/Loop 到
              「性能与优化」；只有同快照 test + benchmark 基线齐全时才启动 Agent。
              只产出候选代码，不会自动采纳。
            </div>
          </div>
        )}

        {/* 自动优化上下文横幅：性能与优化 → 对话带入；ready 时回车即启动（允许空指令） */}
        {optimizeCtx && (
          <div
            className="mx-4 mt-2 rounded-[9px] px-3 py-2 text-[11px]"
            style={{
              background: "var(--panel-2)",
              border:
                "1px solid " +
                (optimizeCtx.status === "failed"
                  ? "var(--bad)"
                  : optimizeCtx.status === "done"
                    ? "var(--ok)"
                    : "var(--accent)"),
            }}
          >
            <div className="flex flex-wrap items-center gap-2">
              <b>自动优化</b>
              <span className="chip !py-0 !text-[10px]">Loop #{optimizeCtx.loopId}</span>
              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                {optimizeCtx.label}
              </span>
              <span
                className={`chip ${
                  optimizeCtx.status === "running"
                    ? "chip-run"
                    : optimizeCtx.status === "done"
                      ? "chip-ok"
                      : optimizeCtx.status === "failed"
                        ? "chip-bad"
                        : ""
                }`}
                style={optimizeCtx.status === "ready" ? { color: "var(--fg-mute)" } : undefined}
              >
                {optimizeCtx.status === "running"
                  ? "运行中"
                  : optimizeCtx.status === "done"
                    ? "已完成"
                    : optimizeCtx.status === "failed"
                      ? "失败"
                      : "等待开始"}
              </span>
              {optimizeCtx.jobId != null && (
                <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  job #{optimizeCtx.jobId}
                </span>
              )}
              {optimizeCtx.status === "running" && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  {optimizeCtx.jobId == null
                    ? "正在确认/启动 Agent，暂不可停止"
                    : "需要中止可用输入框旁的「停止」"}
                </span>
              )}
              {optimizeCtx.status === "ready" && (
                <button
                  className="btn btn-primary btn-sm ml-auto"
                  disabled={sending}
                  onClick={() => void startOptimize()}
                >
                  开始自动优化
                </button>
              )}
              {(optimizeCtx.status === "done" || optimizeCtx.status === "failed") && (
                <button className="btn btn-sm ml-auto" disabled={sending} onClick={armOptimize}>
                  重新运行自动优化
                </button>
              )}
            </div>
            <div className="mt-1 leading-relaxed" style={{ color: "var(--fg-dim)" }}>
              {optimizeCtx.focus ? (
                <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  <b>优化点</b>：{optimizeCtx.focus}
                </div>
              ) : (
                <div>
                  <b>目标</b>：{optimizeCtx.label}
                </div>
              )}
              <div>
                <b>允许改动</b>：
                <span className="mono">
                  {optimizeCtx.allowedPaths.length
                    ? optimizeCtx.allowedPaths.join("，")
                    : "后端默认（ops/<op>/）"}
                </span>
                {" · "}
                <b>每动作预算</b>：{optimizeCtx.budget ?? "—"} 次（<b>后端硬限制</b>）
              </div>
              <div className="mt-0.5" style={{ color: "var(--fg-mute)" }}>
                {optimizeCtx.status === "ready"
                  ? "输入参考指令（可选），回车开始自动优化；只产出候选代码，完成后可继续对话/追问。"
                  : optimizeCtx.status === "running"
                    ? "Agent 正在按本轮目标改代码，过程与最终摘要会写进本会话；完成后可继续追问。"
                    : optimizeCtx.status === "done"
                      ? "自动优化已完成：摘要/改动/回退/下一步见会话消息。现在输入是普通对话（可追问/改代码，改完需回「性能与优化」重跑候选验证）；如需再次自动优化，点右上「重新运行自动优化」后再回车。"
                      : `自动优化失败${optimizeCtx.error ? `：${optimizeCtx.error}` : ""}。也可点右上「重新运行自动优化」重试，或回「性能与优化」重新发起。`}
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-auto p-4">
          {opening && (
            <div className="text-[12px]" style={{ color: "var(--fg-mute)" }}>
              加载会话…
            </div>
          )}
          {!active && !opening && (
            <div className="text-[12px]" style={{ color: "var(--fg-mute)" }}>
              这里的会话都绑定当前任务。默认「挂 skills」会把 40 个 CANNBot skill 暴露给 agent，并能读写任务工作区；
              「轻对话」在工作区内的空目录里跑，不挂 skills，靠提示词约束不碰代码（不是硬隔离）。
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`mb-3 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className="max-w-[78%] rounded-[10px] px-3 py-2 text-[12px] leading-[1.7]"
                style={{
                  background: m.role === "user" ? "var(--accent)" : "var(--panel-2)",
                  color: m.role === "user" ? "#fff" : "var(--fg)",
                  border: "1px solid " + (m.role === "user" ? "transparent" : "var(--border)"),
                  whiteSpace: m.role === "user" ? "pre-wrap" : undefined,
                }}
              >
                {m.role === "assistant" ? <Markdown text={m.text} /> : m.text}
                {m.changes && m.changes.changed_files.length > 0 && (
                  <details
                    className="mt-2 rounded-[8px] p-2"
                    style={{ background: "rgba(0,0,0,.15)", border: "1px solid var(--border)" }}
                  >
                    <summary className="cursor-pointer text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      本轮改动 {m.changes.changed_files.length} 个文件（审阅 / 回退）
                    </summary>
                    <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      {m.changes.changed_files.map((f) => f.path).join("、")}
                    </div>
                    <pre className="mt-1 max-h-[220px] overflow-auto text-[10px]" style={{ whiteSpace: "pre-wrap" }}>
                      {m.changes.diff}
                    </pre>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <button className="btn btn-sm" disabled={sending} onClick={() => void revert(m.id)}>
                        回退本轮（仅还原文件）
                      </button>
                      <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                        只还原本轮文件改动，不删除对话记录、不回滚 agent 会话上下文；若文件已被后续轮改过，后端会拒绝并说明冲突
                      </span>
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}
          {sending && (
            <div className="mb-3 flex justify-start">
              <div
                className="max-w-[78%] rounded-[10px] px-3 py-2 text-[12px] leading-[1.7]"
                style={{
                  background: "var(--panel-2)",
                  border: "1px solid " + (connNote ? "var(--warn)" : "var(--border)"),
                  whiteSpace: "pre-wrap",
                }}
              >
                {stream || (phase || "…")}
                <span className="ml-[2px] animate-pulse">▋</span>
              </div>
            </div>
          )}
          {sending && (
            <div className="mb-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
              {phase && <span>{phase} </span>}
              {tools > 0 && <span>· 工具调用 {tools} 次</span>}
              <span> · 实时输出；本轮结束后显示最终答复</span>
            </div>
          )}
          {connNote && (
            <div
              className="mb-2 rounded-[8px] px-2 py-1 text-[10px]"
              style={{ color: "var(--warn)", border: "1px solid var(--warn)", background: "var(--panel-2)" }}
            >
              {connNote}
            </div>
          )}
          {note && (
            <div className="mb-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
              {note}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {err && (
          <div className="px-4 pb-2 text-[11px]" style={{ color: "var(--bad)" }}>
            {err}
          </div>
        )}

        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {(active?.attachments ?? []).map((a) => (
            <span key={a.id} className="chip flex items-center gap-1 text-[10px]" title={a.path}>
              {a.filename}
              <button
                type="button"
                className="text-[10px] leading-none"
                style={{ color: "var(--fg-mute)", opacity: sending ? 0.4 : 1 }}
                title="删除附件（之后每轮 prompt 不再带上它）"
                disabled={sending || deletingAtt !== null}
                onClick={() => void removeAttachment(a)}
              >
                {deletingAtt === a.id ? "…" : "✕"}
              </button>
            </span>
          ))}
        </div>
        <div className="flex items-end gap-2 p-3" style={{ borderTop: "1px solid var(--border)" }}>
          <input
            ref={fileRef}
            type="file"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void upload(f);
            }}
          />
          <button
            className="btn btn-sm"
            disabled={!active || uploading || sending}
            onClick={() => fileRef.current?.click()}
          >
            {uploading ? "上传中…" : "附件"}
          </button>
          <textarea
            ref={inputRef}
            className="input min-h-[42px] flex-1 resize-y"
            rows={2}
            placeholder={
              newOptimizationReady
                ? "例如：重点优化小 shape 并行度，保持大 shape 路径不变，正确性不能退化"
                : optimizeReady
                ? "可选：输入参考指令（如「不要重复上一轮 tile 方案」），Enter 开始自动优化"
                : active
                  ? "直接说；Enter 发送，Shift+Enter 换行"
                  : "先新建/选择一个会话"
            }
            value={input}
            maxLength={newOptimizationReady ? 2000 : undefined}
            disabled={!active || sending}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                void send();
              }
            }}
          />
          {sending ? (
            <button className="btn btn-danger" disabled={jobId == null || cancelling} onClick={() => void cancel()}>
              {cancelling ? "取消中…" : jobId == null ? "发送中…" : "停止"}
            </button>
          ) : newOptimizationReady ? (
            <button className="btn btn-primary whitespace-nowrap" disabled={!active || !input.trim()} onClick={() => void send()}>
              识别方向并自动优化
            </button>
          ) : optimizeReady ? (
            <button className="btn btn-primary" disabled={!active} onClick={() => void send()}>
              开始自动优化
            </button>
          ) : (
            <button className="btn btn-primary" disabled={!active || !input.trim()} onClick={() => void send()}>
              发送
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
