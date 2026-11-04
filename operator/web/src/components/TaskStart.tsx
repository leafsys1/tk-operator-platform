import { useEffect, useRef, useState } from "react";
import { api, size, type TaskWriteBody } from "../lib/api";
import Markdown from "./Markdown";
import type { AgentEvent, ChatMessage, ChatQuestion, Host, HostImage, Project, TaskDocument } from "../types";

/**
 * 「任务开始」页 —— 布局变体 B（对话优先 + 抽屉）。
 * 功能规划：docs/UI-task-start-v1.md ｜ 布局样机：docs/mockups/task-start-v1.html
 *
 * 本步 = **对话式需求澄清**（§3）：和模型真对话，逐步问清算子需求；
 * 业务产出（规格草稿 + 待确认项）落 `spec_draft`，过程（对话）落 `chat_log`。
 * 对话式引擎走 opencode-go 直连（services/llm.py）；`/context` 上下文空间仍是后续。
 */

const TORCH_PRESET =
  "精度要求：与 torch CPU 实现一一对应（oracle = torch CPU；输出 shape/dtype 完全一致；NaN/Inf 逐元素比对）";

/** 规格字段 → 中文标签（未知键就原样显示） */
const SPEC_LABELS: Record<string, string> = {
  op_name: "算子名",
  math_def: "数学定义",
  params: "参数 / 属性",
  inputs: "输入",
  outputs: "输出",
  layout: "布局 / stride",
  support: "支持范围",
  unsupported: "不支持",
  special_behavior: "特殊行为",
  target_chip: "目标芯片",
  target_env: "目标环境",
  oracle: "参考实现（oracle）",
  tolerance: "容差",
  consistency: "一致性口径",
  performance_baseline: "性能基线",
  performance_target: "性能目标",
  delivery: "交付",
  assumptions: "假设",
  unknowns: "未确定项",
};

type Form = {
  title: string;
  opName: string;
  targetChip: string;
  inputForm: string;
  criteriaRef: string;
  deviceId: string;
  timeoutS: string;
};

const EMPTY: Form = {
  title: "",
  opName: "",
  targetChip: "910B1",
  inputForm: "自然语言",
  // 空 = 用项目默认判据（后端优先级：任务 > 项目 > 官方默认）。不要预填官方档，
  // 否则从本页建出的任务永远覆盖项目级判据（审查 #7）。
  criteriaRef: "",
  deviceId: "",
  timeoutS: "",
};

/** 超时是正整数秒；空 = 用默认（任务配置/环境变量）。 */
const intOrNull = (s: string): number | null => {
  const text = s.trim();
  if (!text) return null;
  const n = Number(text);
  return Number.isInteger(n) && n > 0 ? n : null;
};

const show = (v: unknown): string => (typeof v === "string" ? v : JSON.stringify(v));
const clip = (s: string, n = 160): string => (s.length > n ? `${s.slice(0, n)}…` : s);

/** API 错误 → 后端 detail 人话（`<status> {"detail": "..."}`；不是 JSON 就原文）。 */
function errorBrief(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
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

/** dsh 过程事件 → 一行可读文本；`text`/`final` 不展示（那是原始 JSON，答复由 chat_log 呈现）。 */
function evToLine(ev: AgentEvent): { tone: string; text: string } | null {
  switch (ev.type) {
    case "thinking":
      return { tone: "dim", text: `🧠 ${clip(ev.text ?? "")}` };
    case "tool_call":
      return { tone: "accent", text: `🔧 ${ev.tool ?? ""}` };
    case "status":
      return ev.phase === "step_end" ? { tone: "dim", text: `· step ${ev.step} 完成` } : null;
    case "turn_end":
      return ev.reason?.kind === "error"
        ? { tone: "bad", text: `✘ ${ev.reason.error?.message ?? "error"}` }
        : null;
    case "_stderr":
      return { tone: "bad", text: `! ${clip(ev.text ?? "", 120)}` };
    default:
      return null;
  }
}

export default function TaskStart({
  projectId,
  draftId,
  project,
  host,
  image,
  imageNote,
  flushRef,
  onBack,
  onDraftId,
  onCreated,
}: {
  projectId: number | null;
  draftId: number | null;
  project?: Project | null;
  host?: Host | null;
  image?: HostImage | null;
  /** 项目默认镜像不在当前执行机上时的提示（后端 plan 阶段才会报缺项，这里提前说清）。 */
  imageNote?: string | null;
  /** 供父级（侧栏导航）在切走前 flush 防抖中的编辑；返回 false = 有改动没保存成功。 */
  flushRef?: { current: (() => Promise<boolean>) | null };
  onBack: () => void;
  onDraftId: (id: number) => void;
  onCreated: (id: number) => void;
}) {
  const [form, setForm] = useState<Form>(EMPTY);
  const formRef = useRef(form);
  formRef.current = form;

  const idRef = useRef<number | null>(draftId);
  const loadedIdRef = useRef<number | null | undefined>(undefined);
  const timer = useRef<number | undefined>(undefined);
  // 保存/创建 single-flight：同一时刻只允许一个请求在途，避免连点建出多个草稿、PATCH 乱序覆盖。
  const persistInFlightRef = useRef<Promise<boolean> | null>(null);
  const creatingRef = useRef(false);
  const savingRef = useRef(false);
  const dirtyRef = useRef(false);
  /** 哪些字段是用户改过、还没被保存确认的：对话结束回填时不要覆盖它们（审查 #6）。 */
  const dirtyFieldsRef = useRef<Set<keyof Form>>(new Set());
  const loadedOkRef = useRef(draftId == null);
  const retryTimerRef = useRef<number | undefined>(undefined);
  const mountedRef = useRef(true);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [spec, setSpec] = useState<Record<string, unknown>>({});
  const [questions, setQuestions] = useState<ChatQuestion[]>([]);
  const [ready, setReady] = useState(false);

  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(draftId != null);
  const [sending, setSending] = useState(false);
  const [msg, setMsg] = useState("");
  const [drawer, setDrawer] = useState(false);
  const [live, setLive] = useState<{ tone: string; text: string }[]>([]);
  const [procCount, setProcCount] = useState(0);
  const [showProc, setShowProc] = useState(false);
  const [liveText, setLiveText] = useState(""); // 逐 token 上屏的回复（打字机）
  /** 草稿载入是否成功：失败时禁用自动保存与表单，避免用 EMPTY 默认值覆盖服务端字段。 */
  const [loadedOk, setLoadedOk] = useState(draftId == null);
  const [reloadKey, setReloadKey] = useState(0);
  const [persisting, setPersisting] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const [reconnectNote, setReconnectNote] = useState<string | null>(null);
  /** 当前这一轮的 dsh job：用于「取消本轮」（后端 POST /api/agent-jobs/{id}/cancel）。 */
  const [chatJobId, setChatJobId] = useState<number | null>(null);
  const [cancellingChat, setCancellingChat] = useState(false);

  // 设计文档（任务生命周期 v2）：模板下载 + 上传/替换/删除。
  // 草稿未创建时先暂存文件，首次保存拿到 draftId 后再上传（创建期允许上传，is_draft=1）。
  const [docs, setDocs] = useState<TaskDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsErr, setDocsErr] = useState<string | null>(null);
  const [docBusy, setDocBusy] = useState(false);
  const [docNote, setDocNote] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const pendingFileRef = useRef<File | null>(null);
  const docInputRef = useRef<HTMLInputElement | null>(null);
  // 上传在途标记：草稿创建后 effect 会拉一次文档列表，别让「上传前的旧列表」覆盖上传结果。
  const docBusyRef = useRef(false);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (draftId === loadedIdRef.current) return;
    loadedIdRef.current = draftId;
    idRef.current = draftId;
    setErr(null);
    setReconnectNote(null);
    setChatJobId(null);
    setCancellingChat(false);
    if (draftId == null) {
      setForm(EMPTY);
      setMessages([]);
      setSpec({});
      setQuestions([]);
      setReady(false);
      setStatus("idle");
      setLoading(false);
      setLoadedOk(true);
      loadedOkRef.current = true;
      return;
    }
    // 载入草稿：失败时进入「已暂停保存」状态（表单禁用 + 重试），不用 EMPTY 覆盖服务端字段。
    setLoading(true);
    setLoadedOk(false);
    loadedOkRef.current = false;
    let alive = true;
    api
      .task(draftId)
      .then((t) => {
        if (!alive) return;
        const f: Form = {
          title: t.title ?? "",
          opName: t.op_name ?? "",
          targetChip: t.target_chip ?? "",
          inputForm: t.input_form || "自然语言",
          criteriaRef: t.criteria_ref ?? "",
          deviceId: t.device_id ?? "",
          timeoutS: t.timeout_s != null ? String(t.timeout_s) : "",
        };
        setForm(f);
        formRef.current = f;
        setMessages(t.chat_log ?? []);
        setSpec(t.spec_draft?.fields ?? {});
        setQuestions(t.spec_draft?.questions ?? []);
        setReady(t.spec_draft?.ready ?? false);
        setStatus("saved");
        setLoadedOk(true);
        loadedOkRef.current = true;
        dirtyRef.current = false;
        dirtyFieldsRef.current.clear();
      })
      .catch((e) => {
        if (alive) setErr(String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
      // StrictMode 会「挂载 → 清理 → 再挂载」：这次加载被作废时清掉已载入标记，
      // 让重跑真正重新请求，而不是被 loadedIdRef 短路成永远「载入中」。
      if (loadedIdRef.current === draftId) loadedIdRef.current = undefined;
    };
  }, [draftId, reloadKey]);

  // 已有草稿：载入已上传的设计文档（草稿任务也允许上传）。
  useEffect(() => {
    if (draftId == null) {
      setDocs([]);
      setDocsErr(null);
      return;
    }
    if (docBusyRef.current) return;
    let alive = true;
    setDocsLoading(true);
    api
      .documents(draftId)
      .then((list) => {
        if (!alive) return;
        setDocs(list);
        setDocsErr(null);
      })
      .catch((e) => {
        if (alive) setDocsErr(`文档列表读取失败：${errorBrief(e)}`);
      })
      .finally(() => {
        if (alive) setDocsLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [draftId]);

  useEffect(() => () => window.clearTimeout(timer.current), []);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      window.clearTimeout(retryTimerRef.current);
      esRef.current?.close();
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, sending, liveText]);

  /** 上传设计文档（创建期只保留 1 份：上传成功后删掉旧行；同名文件后端已覆盖）。 */
  async function uploadDoc(tid: number, file: File, existing: TaskDocument[]) {
    docBusyRef.current = true;
    setDocBusy(true);
    setDocsErr(null);
    setDocNote(null);
    try {
      const doc = await api.uploadDocument(tid, file);
      for (const old of existing) {
        if (old.id === doc.id) continue;
        try {
          await api.deleteDocument(tid, old.id);
        } catch {
          // 旧文档删除失败不阻塞本次上传（列表里仍会显示，可手动删）
        }
      }
      setDocs(await api.documents(tid));
      pendingFileRef.current = null;
      setPendingFile(null);
      setDocNote(
        doc.extracted_path
          ? `已上传并解析：${doc.filename}`
          : `已上传：${doc.filename}（解析稿未生成，agent 将直接读原文）`,
      );
    } catch (e) {
      setDocsErr(`上传失败：${errorBrief(e)}`);
      // 暂存文件保留：修好后可再点「替换」重试上传
    } finally {
      docBusyRef.current = false;
      setDocBusy(false);
    }
  }

  function pickDoc(file: File | null | undefined) {
    if (!file) return;
    const name = file.name.toLowerCase();
    if (!name.endsWith(".docx") && !name.endsWith(".md")) {
      setDocsErr("仅支持 .docx / .md 设计文档");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setDocsErr(`文件 ${(file.size / 1024 / 1024).toFixed(1)}MB，超过 20MB 上限`);
      return;
    }
    setDocsErr(null);
    setDocNote(null);
    const tid = idRef.current;
    if (tid == null) {
      pendingFileRef.current = file;
      setPendingFile(file);
      setDocNote("草稿还没创建：首次保存后自动上传");
      return;
    }
    void uploadDoc(tid, file, docs);
  }

  function removeDoc(doc: TaskDocument) {
    const tid = idRef.current ?? draftId;
    if (tid == null) return;
    void (async () => {
      setDocBusy(true);
      setDocsErr(null);
      setDocNote(null);
      try {
        await api.deleteDocument(tid, doc.id);
        setDocs(await api.documents(tid));
      } catch (e) {
        setDocsErr(`删除失败：${errorBrief(e)}`);
      } finally {
        setDocBusy(false);
      }
    })();
  }

  async function persist(): Promise<boolean> {
    if (!loadedOkRef.current) {
      setErr("草稿载入失败，已暂停自动保存 —— 请先点「重试」重新载入");
      setStatus("error");
      return false;
    }
    if (projectId == null) {
      setErr("没有可用项目 —— 先在「服务器与镜像」登记项目/执行机");
      setStatus("error");
      return false;
    }
    // single-flight：并发调用复用同一个在途请求（首次创建在途期间不允许再 POST，避免建出多个草稿）。
    const inFlight = persistInFlightRef.current;
    if (inFlight) return inFlight;
    const f = formRef.current;
    const existingId = idRef.current;
    const creating = existingId == null;
    // 超时：空 = 默认；非空必须是正整数（后端 resolve_timeout 会忽略非法值，但这里先给出提示）。
    const timeoutValue = intOrNull(f.timeoutS);
    if (f.timeoutS.trim() && timeoutValue == null) {
      setErr("超时(s) 必须是正整数（留空 = 用默认超时）");
      setStatus("error");
      return false;
    }
    const body: TaskWriteBody = {
      title: f.title.trim() || f.opName.trim() || "未命名任务",
      op_name: f.opName.trim() || null,
      target_chip: f.targetChip.trim() || null,
      input_form: f.inputForm || null,
      criteria_ref: f.criteriaRef.trim() || null,
      device_id: f.deviceId.trim() || null,
      timeout_s: timeoutValue,
    };
    // 只有「首次创建草稿」的 POST 才带 is_draft；之后的 PATCH 不再携带，避免把已定版任务打回草稿。
    if (creating) body.is_draft = true;
    setStatus("saving");
    setPersisting(true);
    setErr(null);
    const run = (async (): Promise<boolean> => {
      try {
        if (creating) {
          creatingRef.current = true;
          const t = await api.createTask(projectId, body);
          idRef.current = t.id;
          loadedIdRef.current = t.id;
          onDraftId(t.id);
          // 草稿创建后把暂存的设计文档传上去（创建期允许上传；失败只影响文档区，不阻断保存）。
          const pending = pendingFileRef.current;
          if (pending) void uploadDoc(t.id, pending, []);
        } else {
          savingRef.current = true;
          await api.patchTask(existingId, body);
        }
        setStatus("saved");
        // 保存期间表单又变了 → 保持 dirty，在 finally 里补存一次（那一轮改了但被 single-flight 合并掉的编辑）。
        if (formRef.current === f) {
          dirtyRef.current = false;
          dirtyFieldsRef.current.clear();
        }
        return true;
      } catch (e) {
        setErr(String(e));
        setStatus("error");
        return false;
      } finally {
        creatingRef.current = false;
        savingRef.current = false;
      }
    })();
    persistInFlightRef.current = run;
    try {
      return await run;
    } finally {
      if (persistInFlightRef.current === run) persistInFlightRef.current = null;
      setPersisting(false);
      if (formRef.current !== f && mountedRef.current) queueSave();
    }
  }

  function queueSave() {
    if (projectId == null || !loadedOkRef.current) return;
    setStatus("idle");
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void persist(), 700);
  }

  function update<K extends keyof Form>(k: K, v: Form[K]) {
    dirtyRef.current = true;
    dirtyFieldsRef.current.add(k);
    setForm((prev) => {
      const next = { ...prev, [k]: v };
      formRef.current = next;
      return next;
    });
    queueSave();
  }

  async function sendText(text: string) {
    const content = text.trim();
    if (!content || sending || finalizing) return;
    if (projectId == null) {
      setErr("没有可用项目 —— 先在「服务器与镜像」登记项目/执行机");
      return;
    }
    if (!loadedOkRef.current) {
      setErr("草稿载入失败，已暂停发送 —— 请先点「重试」重新载入");
      return;
    }
    // 先进入发送态再 await persist：首次输入要建草稿，创建往返期间按钮/输入框必须禁用，
    // 否则连点「发送 / 仅存草稿」会重复 POST 建出多个草稿。
    setSending(true);
    setErr(null);
    setReconnectNote(null);
    // 首次输入即建草稿（对话挂在任务 id 上）
    if (idRef.current == null) {
      const ok = await persist();
      if (!ok || idRef.current == null) {
        setSending(false);
        return;
      }
    }
    const tid = idRef.current;
    setMsg("");
    setLive([]);
    setProcCount(0);
    setShowProc(false);
    setLiveText("");
    try {
      const res = await api.startChatStream(tid, content); // 起 dsh 后台任务，立刻拿 job_id
      setChatJobId(res.job_id);
      setCancellingChat(false);
      setMessages(res.messages ?? []); // 用户这条已落库，先上屏
      let finished = false;
      let attempts = 0;
      const MAX_RECONNECT = 5;
      const clearRetry = () => {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = undefined;
      };
      const finish = async () => {
        if (finished) return;
        finished = true;
        clearRetry();
        esRef.current?.close();
        esRef.current = null;
        setReconnectNote(null);
        setChatJobId(null);
        setCancellingChat(false);
        try {
          const t = await api.task(tid);
          setMessages(t.chat_log ?? []);
          setSpec(t.spec_draft?.fields ?? {});
          setQuestions(t.spec_draft?.questions ?? []);
          setReady(t.spec_draft?.ready ?? false);
          setForm((prev) => {
            // 只回填用户**没改过**的字段：对话期间正在编辑（尚未保存）的字段不能被服务端旧值覆盖（审查 #6）。
            const dirty = dirtyFieldsRef.current;
            const next: Form = {
              ...prev,
              opName: dirty.has("opName") ? prev.opName : (t.op_name ?? prev.opName),
              targetChip: dirty.has("targetChip") ? prev.targetChip : (t.target_chip ?? prev.targetChip),
              title: dirty.has("title")
                ? prev.title
                : prev.title && prev.title !== "未命名任务" ? prev.title : (t.op_name ?? prev.title),
            };
            formRef.current = next;
            return next;
          });
          if ((t.chat_log ?? []).at(-1)?.role !== "assistant") {
            setErr("本轮没有得到回答（dsh 会话可能失败）—— 详见任务详情的「智能体调用」");
          }
        } catch (e) {
          setErr(String(e));
        }
        setSending(false);
        setLive([]);
        setLiveText("");
      };
      const attach = (jobId: number) => {
        // 断流重连会从头回放已缓存事件：先清空逐 token/过程，避免重放内容重复上屏。
        if (attempts > 0) {
          setLiveText("");
          setLive([]);
          setProcCount(0);
        }
        const es = new EventSource(api.eventsUrl(jobId));
        esRef.current = es;
        es.onmessage = (m) => {
          let ev: AgentEvent;
          try {
            ev = JSON.parse(m.data) as AgentEvent;
          } catch {
            return;
          }
          // 逐 token 的回复（插件回流 → 后端切掉 JSON 尾）：直接打进气泡，就是打字机
          if (ev.type === "delta" && typeof ev.text === "string") {
            setLiveText((s) => s + ev.text);
            return;
          }
          // 过程（思考 / 步骤 / 工具）：折叠起来，不抢正文的地方
          const line = evToLine(ev);
          if (line) {
            setLive((prev) => [...prev.slice(-6), line]);
            setProcCount((n) => n + 1);
          }
        };
        es.addEventListener("done", () => void finish());
        // 连接层错误（代理抖动/后端重启导致 404）≠ 本轮结束：先查 /live，
        // 仍在运行就退避重连（有上限）；确认已结束或 /live 也失败时才收尾。
        es.onerror = () => {
          if (finished) return;
          es.close();
          if (esRef.current === es) esRef.current = null;
          void api
            .agentLive(jobId)
            .then((info) => {
              if (finished || !mountedRef.current) return;
              if (!info.running) {
                void finish();
                return;
              }
              if (attempts >= MAX_RECONNECT) {
                setReconnectNote(null);
                setErr("与后台的连接已中断，任务仍在运行 —— 请稍后刷新页面查看结果（也可到任务详情的「智能体调用」查看）");
                setSending(false);
                setLive([]);
                setLiveText("");
                return;
              }
              attempts += 1;
              const delayS = Math.min(2 ** (attempts - 1), 15);
              setReconnectNote(`连接中断，仍在运行 —— ${delayS}s 后第 ${attempts}/${MAX_RECONNECT} 次重连…`);
              clearRetry();
              retryTimerRef.current = window.setTimeout(() => {
                if (!finished && mountedRef.current) attach(jobId);
              }, delayS * 1000);
            })
            // /live 也不可用（job 不存在 / 后端重启）：无法确认仍在运行，按本轮结束收尾，让用户刷新查看
            .catch(() => {
              if (!finished && mountedRef.current) void finish();
            });
        };
      };
      attach(res.job_id);
    } catch (e) {
      setErr(String(e));
      setSending(false);
      setLive([]);
      setLiveText("");
      setChatJobId(null);
      setCancellingChat(false);
    }
  }

  /** 取消本轮 dsh 对话（后端杀进程组并落 interrupted；SSE 会随后收到 done）。 */
  async function cancelChat() {
    const jobId = chatJobId;
    if (jobId == null || cancellingChat) return;
    setCancellingChat(true);
    try {
      const r = await api.cancelAgentJob(jobId);
      setReconnectNote(r.result || "已发送取消，等待进程退出…");
    } catch (e) {
      setErr(`取消失败：${String(e)}`);
    } finally {
      setCancellingChat(false);
    }
  }

  function saveDraft() {
    window.clearTimeout(timer.current);
    void persist();
  }

  async function confirmCreate() {
    if (finalizing || sending) return;
    window.clearTimeout(timer.current);
    // 定版前先置 in-flight：await persist 期间按钮必须禁用，否则双击会重复 finalize（后端每次都落一版 spec）。
    setFinalizing(true);
    setErr(null);
    setStatus("saving");
    try {
      const ok = await persist();
      const tid = idRef.current;
      if (!ok || tid == null) return;
      setSending(true); // 定版期间禁用对话区
      await api.finalize(tid); // 定版：落 spec v1 + 生成上下文空间 + 草稿转正
      onCreated(tid);
    } catch (e) {
      setErr(String(e));
      setStatus("error");
    } finally {
      setFinalizing(false);
      setSending(false);
    }
  }

  /** flush 防抖中的编辑：返回 false = 有改动没保存成功。 */
  async function flushSave(): Promise<boolean> {
    window.clearTimeout(timer.current);
    timer.current = undefined;
    if (!dirtyRef.current) return true;
    return persist();
  }

  /** 返回任务列表前先 flush 防抖中的编辑；保存失败时明确提示并允许强制离开。 */
  async function handleBack() {
    if (leaving) return;
    setLeaving(true);
    setErr(null);
    const ok = await flushSave();
    setLeaving(false);
    if (ok) {
      onBack();
      return;
    }
    if (window.confirm("有改动未能保存（见页面错误提示）。仍要离开吗？离开后这些改动会丢失。")) onBack();
  }

  // 把 flush 交给父级：侧栏切走（不走「返回任务列表」）也要先保存，避免 700ms 内编辑被丢弃。
  useEffect(() => {
    if (!flushRef) return;
    flushRef.current = flushSave;
    return () => {
      if (flushRef.current === flushSave) flushRef.current = null;
    };
  });

  // 关标签页/刷新前提示未保存编辑（防抖窗口内直接关页面时浏览器无法同步保存）。
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!dirtyRef.current) return;
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  /** 载入失败后重试：清掉已载入标记并重跑加载 effect。 */
  function retryLoad() {
    if (draftId == null || loading) return;
    loadedIdRef.current = undefined;
    setErr(null);
    setLoading(true);
    setReloadKey((k) => k + 1);
  }

  const statusLabel = finalizing
    ? "定版中…"
    : sending
      ? "对话中…"
      : status === "saving"
        ? "保存中…"
        : status === "saved"
          ? "已保存"
          : status === "error"
            ? "出错"
            : idRef.current == null
              ? "未保存"
              : "有改动未保存";

  const specKeys = Object.keys(spec).filter((k) => spec[k] !== "" && spec[k] != null);
  // 文档接口的 task id：新建草稿后 idRef 先于 prop 更新，两者取先有的。
  const docTaskId = idRef.current ?? draftId;
  // 载入失败/载入中/定版中：字段与保存入口都锁住，避免默认值覆盖或定版期间并发 PATCH。
  const formLocked = !loadedOk || loading || finalizing;
  // 对话侧同样锁住：草稿未载入完成时发送没有意义（消息会缺上下文）。
  const chatLocked = sending || formLocked;

  return (
    <div className="grid items-start gap-4" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(224px,264px)" }}>
      {/* ── 主区：对话优先 ───────────────────────────────── */}
      <section className="card flex flex-col overflow-hidden">
        <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
          <h2 className="m-0 text-xs font-semibold">需求澄清对话</h2>
          <span className="ml-auto text-[11px]" style={{ color: loading ? "var(--run)" : "var(--fg-mute)" }}>
            {loading ? "载入草稿…" : idRef.current ? `草稿 #${idRef.current}` : "新任务"}
          </span>
        </div>

        <div className="flex items-center gap-2 border-b px-4 py-2" style={{ borderColor: "var(--border)" }}>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            直接说需求即可；数学公式或参考代码也可以直接粘进对话。
          </span>
          <button className="btn btn-sm ml-auto" onClick={() => void handleBack()} disabled={leaving}>
            {leaving ? "保存中…" : "返回任务列表"}
          </button>
        </div>

        {/* 草稿载入失败：表单/自动保存已暂停，给出重试入口（避免用默认值覆盖服务端字段） */}
        {draftId != null && !loadedOk && !loading && (
          <div
            className="flex flex-wrap items-center gap-2 border-b px-4 py-2 text-[11px]"
            style={{ borderColor: "var(--border)", background: "var(--panel-3)", color: "var(--bad)" }}
          >
            <span>草稿 #{draftId} 载入失败：{err ?? "未知错误"} —— 已暂停自动保存，请重试后再编辑。</span>
            <button className="btn btn-sm ml-auto" onClick={retryLoad}>
              重试
            </button>
          </div>
        )}

        {/* 对话流 */}
        <div ref={scrollRef} className="flex min-h-[280px] flex-col gap-3 overflow-auto p-4" style={{ maxHeight: 460 }}>
          {messages.length === 0 && (
            <div className="max-w-[92%]">
              <div className="mb-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                agent · 需求澄清
              </div>
              <div className="px-3 py-3 leading-relaxed" style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 9 }}>
                说说你要开发哪个算子、想达到什么效果吧（例：「我要开发 rms_norm，做到跟 torch CPU 的实现一一对应」）。
                我会逐步把语义、输入输出、精度口径、支持范围问清楚，右侧实时长出规格草稿。
                <br />
                <span style={{ color: "var(--fg-dim)" }}>首次输入会先建一个草稿任务，之后每步自动保存。</span>
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "ml-auto max-w-[92%]" : "max-w-[92%]"}>
              <div
                className="mb-1 text-[11px]"
                style={{ color: "var(--fg-mute)", textAlign: m.role === "user" ? "right" : "left" }}
              >
                {m.role === "user" ? "我" : "agent · 需求澄清"}
              </div>
              {m.role === "user" ? (
                <div className="px-3 py-3 leading-relaxed" style={{ background: "var(--accent-soft)", border: "1px solid var(--accent)", borderRadius: 9 }}>
                  {m.text}
                </div>
              ) : (
                <div className="px-3 py-3 leading-relaxed" style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 9 }}>
                  <Markdown text={m.text} />
                </div>
              )}
            </div>
          ))}

          {sending && (
            <div className="max-w-[92%]">
              <div className="mb-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                agent · 需求澄清（dsh 会话进行中）
              </div>
              {/* 逐 token 上屏的正文（打字机）：和已落库的助手气泡同一套样式 */}
              <div
                className="px-3 py-3 leading-relaxed"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 9, whiteSpace: "pre-wrap", wordBreak: "break-word" }}
              >
                {liveText ? (
                  <>
                    {liveText}
                    <span className="animate-pulse">▋</span>
                  </>
                ) : (
                  <span style={{ color: "var(--run)" }}>正在梳理…</span>
                )}
              </div>
              {/* 过程（思考 / 步骤 / 工具）：默认收起，不抢正文的地方 */}
              {reconnectNote && (
                <div className="mt-1 text-[11px]" style={{ color: "var(--run)" }}>
                  ⟳ {reconnectNote}
                </div>
              )}
              {chatJobId != null && (
                <div className="mt-1 flex items-center gap-2">
                  <button className="btn btn-sm btn-danger" disabled={cancellingChat} onClick={() => void cancelChat()}>
                    {cancellingChat ? "取消中…" : "取消本轮"}
                  </button>
                  <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                    卡住/不想等时可用（后端会杀 dsh 进程并标记中断）
                  </span>
                </div>
              )}
              {(procCount > 0 || live.length > 0) && (
                <div className="mt-1">
                  <button className="btn btn-sm" onClick={() => setShowProc((v) => !v)}>
                    {showProc ? "收起过程" : `过程（${procCount}）`}
                  </button>
                  {showProc && (
                    <div
                      className="mt-1 rounded-[9px] px-3 py-2 text-[11px] leading-[1.9]"
                      style={{ background: "var(--panel-3)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace" }}
                    >
                      {live.length === 0 ? (
                        <span style={{ color: "var(--fg-mute)" }}>…</span>
                      ) : (
                        live.map((l, i) => (
                          <div key={i} style={{ color: l.tone === "bad" ? "var(--bad)" : l.tone === "accent" ? "var(--accent)" : "var(--fg-mute)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                            {l.text}
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 待确认项 */}
        {questions.length > 0 && (
          <div className="border-t px-4 py-3" style={{ borderColor: "var(--border)", background: "var(--panel-3)" }}>
            <div className="flex items-center gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              <b>待确认 {questions.length} 项</b> · 直接在下面回答，或点一项填入
            </div>
            <div className="mt-2 flex flex-col gap-1">
              {questions.map((q, i) => (
                <button
                  key={q.id ?? i}
                  className="text-left text-[12px] leading-relaxed"
                  style={{ color: "var(--fg)", background: "transparent", border: "none", padding: 0, cursor: "pointer" }}
                  onClick={() => setMsg((prev) => (prev ? prev : `关于「${q.text}」：`))}
                >
                  <span className="mono" style={{ color: "var(--accent)" }}>
                    {i + 1}.
                  </span>{" "}
                  {q.text}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 输入 */}
        <div className="border-t p-4" style={{ borderColor: "var(--border)" }}>
          <div className="flex gap-2">
            <textarea
              className="field"
              rows={2}
              style={{ resize: "vertical", minHeight: 44 }}
              placeholder={questions.length ? "回答上面的问题，或补充新信息…（Shift+Enter 换行）" : "继续说…（Shift+Enter 换行）"}
              value={msg}
              disabled={chatLocked}
              onChange={(e) => setMsg(e.target.value)}
              onKeyDown={(e) => {
                // IME 组字期间的回车是「上屏候选」，不是「发送」（中文输入法下 add→回车 会误发）
                const composing = e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229;
                if (e.key === "Enter" && !e.shiftKey && !composing) {
                  e.preventDefault();
                  void sendText(msg);
                }
              }}
            />
            <button className="btn btn-primary btn-sm" disabled={chatLocked || !msg.trim()} onClick={() => void sendText(msg)}>
              {sending ? "…" : "发送"}
            </button>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <button className="tab-x" disabled={chatLocked} onClick={() => void sendText(TORCH_PRESET)}>
              跟 torch CPU 实现一一对应
            </button>
            <button className="tab-x" disabled={chatLocked} onClick={() => void sendText("先只支持 fp32 与 fp16，bf16 暂不考虑")}>
              只先做 fp32/fp16
            </button>
          </div>
          {err && loadedOk && (
            <div className="mt-2 text-[11px]" style={{ color: "var(--bad)" }}>
              错误：{err}
            </div>
          )}
        </div>
      </section>

      {/* ── 右栏：规格草稿 / 配置 / 保存与创建 ─────────────── */}
      <div className="flex flex-col gap-4">
        <section className="card">
          <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">规格草稿</h2>
            <span className={`chip ml-auto ${ready ? "chip-ok" : "chip-run"}`}>{ready ? "可定版" : "澄清中"}</span>
          </div>
          <div className="px-4 py-3">
            {specKeys.length === 0 ? (
              <div className="text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
                还没有确定的信息 —— 先跟左边聊几句。
              </div>
            ) : (
              specKeys.slice(0, 8).map((k) => (
                <div key={k} className="row-line">
                  <span className="k">{SPEC_LABELS[k] ?? k}</span>
                  <span className="v" style={{ wordBreak: "break-word" }}>
                    {show(spec[k])}
                  </span>
                </div>
              ))
            )}
            <button className="btn btn-sm mt-2 w-full" onClick={() => setDrawer(true)}>
              查看完整需求文档
            </button>
          </div>
        </section>

        <section className="card">
          <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">任务字段</h2>
            <span className={`chip ml-auto ${status === "error" ? "chip-bad" : status === "saved" && !sending ? "chip-ok" : "chip-run"}`}>
              {statusLabel}
            </span>
          </div>
          <div className="flex flex-col gap-2 px-4 py-3">
            <label className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              任务名称
              <input className="field mt-1" value={form.title} disabled={formLocked} onChange={(e) => update("title", e.target.value)} placeholder="未命名任务" />
            </label>
            <label className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              算子名
              <input className="field mt-1" value={form.opName} disabled={formLocked} onChange={(e) => update("opName", e.target.value)} placeholder="rms_norm" />
            </label>
            <label className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              目标芯片
              <input className="field mt-1" value={form.targetChip} disabled={formLocked} onChange={(e) => update("targetChip", e.target.value)} placeholder="910B1" />
            </label>
            <label className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              判据档（档名@档版本）
              <input className="field mt-1 mono" value={form.criteriaRef} disabled={formLocked} onChange={(e) => update("criteriaRef", e.target.value)} placeholder="ops-precision-standard@1" />
            </label>
          </div>
        </section>

        {/* 设计文档（任务生命周期 v2）：模板下载 + 上传/替换/删除（docx / md，≤20MB） */}
        <section className="card">
          <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">设计文档</h2>
            <span className="chip ml-auto !py-0 !text-[10px]">可选</span>
          </div>
          <div className="px-4 py-3">
            <div className="text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
              先下载模板填写，再上传（docx / md，≤20MB）。agent 用解析稿生成精度/性能用例；原文会进交付包。
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <a className="btn btn-sm" href={api.templateUrl("docx")} download>
                下载模板 .docx
              </a>
              <a className="btn btn-sm" href={api.templateUrl("md")} download>
                下载模板 .md
              </a>
            </div>

            <div className="mt-3">
              {docsLoading && (
                <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>读取已上传文档…</div>
              )}
              {!docsLoading && docs.length === 0 && !pendingFile && (
                <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                  还没有上传设计文档（可跳过，直接在对话里说需求）。
                </div>
              )}
              {docs.map((doc) => (
                <div
                  key={doc.id}
                  className="mt-1 flex flex-wrap items-center gap-2 rounded-[8px] px-3 py-2 text-[11px]"
                  style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
                >
                  <span className="mono max-w-[150px] truncate" title={doc.filename} style={{ color: "var(--fg)" }}>
                    {doc.filename}
                  </span>
                  <span style={{ color: "var(--fg-mute)" }}>{size(doc.size_bytes)}</span>
                  <span
                    className={`chip !py-0 !text-[10px] ${doc.extracted_path ? "chip-ok" : "chip-warn"}`}
                    title={doc.extracted_path ? `解析稿：${doc.extracted_path}` : "解析稿未生成"}
                  >
                    {doc.extracted_path ? "已解析" : "未解析"}
                  </span>
                  <div className="ml-auto flex items-center gap-1.5">
                    {docTaskId != null && (
                      <a
                        className="btn btn-sm"
                        href={api.downloadDocumentUrl(docTaskId, doc.id)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        下载
                      </a>
                    )}
                    <button className="btn btn-sm btn-danger" disabled={docBusy} onClick={() => removeDoc(doc)}>
                      删除
                    </button>
                  </div>
                </div>
              ))}
              {pendingFile && (
                <div
                  className="mt-1 flex flex-wrap items-center gap-2 rounded-[8px] px-3 py-2 text-[11px]"
                  style={{ background: "var(--panel-3)", border: "1px dashed var(--border-strong)" }}
                >
                  <span className="mono max-w-[150px] truncate" title={pendingFile.name}>{pendingFile.name}</span>
                  <span style={{ color: "var(--fg-mute)" }}>{size(pendingFile.size)}</span>
                  <span className="chip chip-run !py-0 !text-[10px]">待上传（草稿创建后自动上传）</span>
                  {docTaskId != null && (
                    <button
                      className="btn btn-sm ml-auto"
                      disabled={docBusy}
                      onClick={() => void uploadDoc(docTaskId, pendingFile, docs)}
                    >
                      {docBusy ? "上传中…" : "重试上传"}
                    </button>
                  )}
                  <button
                    className={`btn btn-sm ${docTaskId != null ? "" : "ml-auto"}`}
                    onClick={() => {
                      pendingFileRef.current = null;
                      setPendingFile(null);
                      setDocNote(null);
                    }}
                  >
                    取消
                  </button>
                </div>
              )}
            </div>

            <input
              ref={docInputRef}
              type="file"
              accept=".docx,.md,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/markdown"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                e.target.value = ""; // 允许再次选择同一文件（替换）
                pickDoc(f);
              }}
            />
            <button
              className="btn btn-sm mt-3 w-full"
              disabled={docBusy || loading || !loadedOk}
              onClick={() => docInputRef.current?.click()}
            >
              {docBusy ? "上传中…" : docs.length > 0 || pendingFile ? "替换设计文档" : "上传设计文档"}
            </button>
            {docNote && <div className="mt-2 text-[10px] leading-relaxed" style={{ color: "var(--ok)" }}>{docNote}</div>}
            {docsErr && <div className="mt-2 text-[10px] leading-relaxed" style={{ color: "var(--bad)" }}>{docsErr}</div>}
          </div>
        </section>

        <section className="card">
          <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">配置</h2>
            <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
              空 = 用项目默认
            </span>
          </div>
          <div className="px-4 py-3">
            <div className="row-line">
              <span className="k">执行机</span>
              <span className="v mono" style={{ wordBreak: "break-all" }}>
                {host ? `${host.name} · ${host.cards ?? "—"}` : "项目默认未绑定"}
              </span>
            </div>
            <div className="row-line">
              <span className="k">镜像</span>
              <span className="v mono" style={{ wordBreak: "break-all" }}>
                {image
                  ? `${image.image_ref}${image.tag ? `:${image.tag}` : ""}`
                  : project?.image_id != null
                    ? `镜像 #${project.image_id}（清单里未解析到，见「服务器与镜像」）`
                    : "项目默认未绑定"}
              </span>
            </div>
            {imageNote && (
              <div className="mt-1 text-[10px] leading-relaxed" style={{ color: "var(--bad)" }}>
                ⚠ {imageNote}
              </div>
            )}
            <label className="row-line" style={{ alignItems: "center" }}>
              <span className="k">设备</span>
              <input className="field !w-24 mono" value={form.deviceId} disabled={formLocked} onChange={(e) => update("deviceId", e.target.value)} placeholder="自动选空卡" />
            </label>
            <label className="row-line" style={{ alignItems: "center" }}>
              <span className="k">超时(s)</span>
              <input className="field !w-24 mono" value={form.timeoutS} disabled={formLocked} onChange={(e) => update("timeoutS", e.target.value)} placeholder="1800" />
            </label>
            <div className="mt-2 text-[10px] leading-relaxed" style={{ color: "var(--fg-dim)" }}>
              项目工作区：<span className="mono">{project?.workspace_path ?? "—"}</span>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">保存与创建</h2>
          </div>
          <div className="flex flex-col gap-2 px-4 py-3">
            <button
              className="btn btn-primary"
              disabled={projectId == null || loading || !ready || sending || finalizing || persisting || !loadedOk}
              title={ready ? "" : "待澄清项未清零（模型判 ready=false）"}
              onClick={confirmCreate}
            >
              {finalizing ? "定版中…" : ready ? "确认并定版（生成上下文空间）" : `确认创建（待澄清 ${questions.length} 项）`}
            </button>
            <button className="btn" disabled={projectId == null || loading || persisting || finalizing || !loadedOk} onClick={saveDraft}>
              {persisting ? "保存中…" : "仅存草稿"}
            </button>
            <div className="text-[10px] leading-relaxed" style={{ color: "var(--fg-dim)" }}>
              「确认并定版」= 定版门禁（待澄清清零）→ 落 <span className="mono">spec v1</span> →
              生成上下文空间 <span className="mono">tasks/&lt;slug&gt;/</span> → 草稿转正 → 跳任务详情。
            </div>
          </div>
        </section>
      </div>

      {/* ── 抽屉：算子需求文档（规格草稿） ─────────────── */}
      {drawer && <div className="scrim on" onClick={() => setDrawer(false)} />}
      <div className={`drawer ${drawer ? "on" : ""}`} aria-hidden={!drawer}>
        <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)", background: "var(--panel)" }}>
          <h2 className="m-0 text-xs font-semibold">规格草稿 · {form.opName || "未命名"}</h2>
          <span className="ml-auto flex items-center gap-2">
            <span className={`chip mono ${ready ? "chip-ok" : "chip-run"}`}>{ready ? "可定版" : "澄清中"}</span>
            <button className="btn btn-sm" onClick={() => setDrawer(false)}>
              关闭 ✕
            </button>
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-4">
          {specKeys.length === 0 ? (
            <div className="text-[12px]" style={{ color: "var(--fg-mute)" }}>
              规格草稿还是空的 —— 到左边对话里聊几句。
            </div>
          ) : (
            <div className="spec-sec">
              <table>
                <tbody>
                  {specKeys.map((k) => (
                    <tr key={k}>
                      <td style={{ width: 110, color: "var(--fg-mute)", fontSize: 11 }}>{SPEC_LABELS[k] ?? k}</td>
                      <td className="mono" style={{ fontSize: 11, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                        {show(spec[k])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {questions.length > 0 && (
            <div className="mt-4">
              <div className="text-[11px]" style={{ color: "var(--fg-mute)", marginBottom: 6 }}>
                待确认项（{questions.length}）
              </div>
              <ol className="m-0 pl-5 text-[12px] leading-relaxed">
                {questions.map((q, i) => (
                  <li key={q.id ?? i}>{q.text}</li>
                ))}
              </ol>
            </div>
          )}

          <div className="mt-4 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
            这是模型逐轮问出来的「规格草稿」（业务证据）。定版时会落成 <span className="mono">spec_versions v1</span>
            （人读 md 由机读 json 渲染，禁手改）；对话记录（过程证据）另存 <span className="mono">chat_log</span>。
          </div>
        </div>
      </div>
    </div>
  );
}
