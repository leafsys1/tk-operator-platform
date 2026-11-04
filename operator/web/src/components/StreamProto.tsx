import { useRef, useState } from "react";
import { api } from "../lib/api";

/**
 * 流式原型（旁路验证，不进正式链路）。两个数据源可切换：
 * - **dsh 插件**：`opx-typewriter` 插件把 `agent/assistant-stream` 的 token 增量 POST 回后端（`ops/dsh/opx-typewriter/`）。
 * - **opencode SSE**：本机 opencode 服务端 `/api/event` 的 `session.text.delta`。
 */
type Source = "dsh" | "opencode";

export default function StreamProto() {
  const [source, setSource] = useState<Source>("dsh");
  const [prompt, setPrompt] = useState("请用大约 150 字介绍 RMSNorm，不要调用任何工具。");
  const [answer, setAnswer] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [showReasoning, setShowReasoning] = useState(false);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [meta, setMeta] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [deltas, setDeltas] = useState(0);
  const [gaps, setGaps] = useState<number[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const t0 = useRef(0);
  const last = useRef(0);

  const parse = (e: Event): Record<string, unknown> => {
    try {
      return JSON.parse((e as MessageEvent).data);
    } catch {
      return {};
    }
  };

  function run() {
    if (!prompt.trim()) return;
    esRef.current?.close();
    setAnswer("");
    setReasoning("");
    setErr(null);
    setMeta(null);
    setDeltas(0);
    setGaps([]);
    setStatus("running");
    t0.current = performance.now();
    last.current = t0.current;

    const url = source === "dsh" ? api.protoDshStreamUrl(prompt.trim()) : api.protoOpencodeStreamUrl(prompt.trim());
    const es = new EventSource(url);
    esRef.current = es;
    es.addEventListener("meta", (e) => {
      const d = parse(e);
      setMeta(String(d.run ?? d.sessionID ?? ""));
    });
    es.addEventListener("reasoning", (e) => {
      setReasoning((r) => r + String(parse(e).delta ?? ""));
      tick();
    });
    es.addEventListener("text", (e) => {
      setAnswer((a) => a + String(parse(e).delta ?? ""));
      tick();
    });
    es.addEventListener("done", () => {
      setStatus("done");
      es.close();
    });
    es.addEventListener("stream_error", (e) => {
      setErr(String(parse(e).message ?? "错误"));
      setStatus("error");
      es.close();
    });
    es.onerror = () => {
      es.close();
      setStatus((s) => (s === "running" ? "error" : s));
    };
  }

  function tick() {
    const now = performance.now();
    setDeltas((n) => n + 1);
    setGaps((g) => [...g.slice(-120), now - last.current]);
    last.current = now;
  }

  const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
  const elapsed = status === "running" ? ((performance.now() - t0.current) / 1000).toFixed(1) : null;

  return (
    <section className="card">
      <div className="flex items-center gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">流式原型 · 真 token 级打字机</h2>
        <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
          旁路验证，不影响正式对话
        </span>
      </div>

      <div className="p-4">
        <div className="mb-2 flex items-center gap-2 text-[11px]">
          <span style={{ color: "var(--fg-mute)" }}>数据源</span>
          {(["dsh", "opencode"] as Source[]).map((s) => (
            <button
              key={s}
              className="tab-x"
              aria-selected={source === s}
              disabled={status === "running"}
              onClick={() => setSource(s)}
            >
              {s === "dsh" ? "dsh 插件" : "opencode SSE"}
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <textarea
            className="field"
            rows={2}
            style={{ resize: "vertical" }}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={status === "running"}
          />
          <button className="btn btn-primary btn-sm" disabled={status === "running" || !prompt.trim()} onClick={run}>
            {status === "running" ? "流式中…" : "运行"}
          </button>
        </div>

        <div className="mt-2 flex flex-wrap gap-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          <span>状态 {status}</span>
          {meta && <span className="mono">{source === "dsh" ? "run" : "会话"} {meta.slice(0, 16)}…</span>}
          <span>增量 {deltas} 段</span>
          <span>平均间隔 {avg.toFixed(0)} ms</span>
          {elapsed && <span>已 {elapsed}s</span>}
        </div>

        <div
          className="mt-3 min-h-[140px] whitespace-pre-wrap rounded-[9px] p-3 text-[13px] leading-[1.9]"
          style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
        >
          {answer ? (
            <>
              {answer}
              {status === "running" && <span className="animate-pulse">▋</span>}
            </>
          ) : (
            <span style={{ color: "var(--fg-mute)" }}>
              {status === "running" ? "等待第一个 token…" : "点「运行」看打字机效果。"}
            </span>
          )}
        </div>

        {reasoning && (
          <div className="mt-2">
            <button className="btn btn-sm" onClick={() => setShowReasoning((v) => !v)}>
              {showReasoning ? "收起" : "展开"}思考（{reasoning.length} 字）
            </button>
            {showReasoning && (
              <div
                className="mt-2 whitespace-pre-wrap rounded-[9px] p-3 text-[11px] leading-relaxed"
                style={{ background: "var(--panel-3)", border: "1px solid var(--border)", color: "var(--fg-mute)" }}
              >
                {reasoning}
              </div>
            )}
          </div>
        )}

        {err && (
          <div className="mt-2 text-[11px]" style={{ color: "var(--bad)" }}>
            错误：{err}
          </div>
        )}
      </div>
    </section>
  );
}
