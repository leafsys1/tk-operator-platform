import { useEffect, useMemo, useRef, useState } from "react";
import { api, size } from "../lib/api";
import type { TaskDetail, WorkspaceDiff, WorkspaceInfo, WorkspaceFileText } from "../types";

type Tab = "files" | "diff" | "build" | "terminal";

/** 任务详情「实现与接口」：工作区文件浏览 + git diff + 最新构建日志。
 *
 * 只读：文件内容与 diff 都有上限；运行时大件（.dsh/.cannbot/runs/build）不列出。
 */
export default function ImplementationPanel({ task }: { task: TaskDetail }) {
  const [tab, setTab] = useState<Tab>("files");
  const [info, setInfo] = useState<WorkspaceInfo | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<WorkspaceFileText | null>(null);
  const [diff, setDiff] = useState<WorkspaceDiff | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cmd, setCmd] = useState("");
  const [shellRows, setShellRows] = useState<{ command: string; exit_code: number; stdout: string; stderr: string; duration_ms: number }[]>([]);
  const [shellBusy, setShellBusy] = useState(false);
  // 文件预览请求序号：连续点文件时只接受最后一次响应，避免旧文件内容覆盖新选择。
  const fileSeq = useRef(0);

  const buildArtifact = useMemo(() => {
    const runs = task.runs.filter((r) => r.action === "build").sort((a, b) => b.id - a.id);
    for (const r of runs) {
      const a = r.artifacts.find((x) => x.type === "log");
      if (a) return a;
    }
    return null;
  }, [task.runs]);

  useEffect(() => {
    let alive = true;
    api.workspace(task.id)
      .then((d) => {
        if (!alive) return;
        setInfo(d);
        const first = d.files.find((f) => f.path.startsWith("ops/")) ?? d.files[0];
        if (first) void openFile(first.path);
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
      fileSeq.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.id]);

  const openFile = async (path: string) => {
    const seq = ++fileSeq.current;
    setSelected(path);
    setFile(null);
    setErr(null);
    try {
      const data = await api.workspaceFile(task.id, path);
      if (fileSeq.current !== seq) return;
      setFile(data);
    } catch (e) {
      if (fileSeq.current !== seq) return;
      setErr(String(e));
    }
  };

  useEffect(() => {
    if (tab !== "diff" || diff) return;
    setBusy(true);
    api.workspaceDiff(task.id)
      .then(setDiff)
      .catch((e) => setErr(String(e)))
      .finally(() => setBusy(false));
  }, [tab, diff, task.id]);

  useEffect(() => {
    if (tab !== "build" || log !== null || !buildArtifact) return;
    setBusy(true);
    api.artifactText(buildArtifact.id)
      .then((d) => setLog(d.text ?? `（${d.binary ? "二进制产物" : "空"}）`))
      .catch((e) => setErr(String(e)))
      .finally(() => setBusy(false));
  }, [tab, log, buildArtifact]);

  const runShell = async () => {
    const c = cmd.trim();
    if (!c) return;
    setShellBusy(true);
    setErr(null);
    try {
      const r = await api.shell(task.id, c);
      setShellRows((rows) => [{ command: c, exit_code: r.exit_code, stdout: r.stdout, stderr: r.stderr, duration_ms: r.duration_ms }, ...rows]);
      setCmd("");
    } catch (e) {
      setErr(String(e));
    } finally {
      setShellBusy(false);
    }
  };

  const files = (info?.files ?? []).filter((f) => !query || f.path.toLowerCase().includes(query.toLowerCase()));

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">实现与接口</h2>
        <div className="ml-2 flex gap-1">
          {([["files", "工作区文件"], ["diff", "Git diff"], ["build", "构建日志"], ["terminal", "命令"]] as const).map(([k, label]) => (
            <button
              key={k}
              className="btn btn-sm"
              onClick={() => setTab(k)}
              style={tab === k ? { background: "var(--accent-soft)", borderColor: "var(--accent)" } : undefined}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="ml-auto mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
          {info?.root ?? ""}
        </span>
      </div>

      {err && <div className="px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}

      {tab === "files" && (
        <div className="grid" style={{ gridTemplateColumns: "280px minmax(0,1fr)" }}>
          <div className="border-r p-2" style={{ borderColor: "var(--border)" }}>
            <input
              className="input mb-2"
              placeholder="过滤文件…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="max-h-[380px] overflow-auto">
              {files.map((f) => (
                <button
                  key={f.path}
                  className="block w-full truncate rounded px-2 py-1 text-left text-[11px]"
                  style={{ background: selected === f.path ? "var(--panel-2)" : "transparent" }}
                  onClick={() => void openFile(f.path)}
                  title={f.path}
                >
                  {f.path}
                  <span className="ml-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>{size(f.size_bytes)}</span>
                </button>
              ))}
              {info?.truncated && (
                <div className="mt-2 text-[10px]" style={{ color: "var(--warn)" }}>文件过多，已截断到 {info.count} 个</div>
              )}
            </div>
          </div>
          <div className="p-3">
            <div className="mb-2 flex items-center gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              <span className="mono">{selected ?? "（选一个文件）"}</span>
              {file?.truncated && <span className="chip chip-warn">已截断</span>}
              {file?.binary && <span className="chip">二进制</span>}
            </div>
            <pre
              className="mt-0 mb-0 max-h-[380px] overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
              style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace", whiteSpace: "pre-wrap" }}
            >
              {file?.text ?? (file?.binary ? "（二进制文件，不预览）" : "读取中…")}
            </pre>
          </div>
        </div>
      )}

      {tab === "diff" && (
        <div className="p-3">
          {busy && !diff && <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>读取 git 状态…</div>}
          {diff && !diff.is_repo && (
            <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              工作区不在 git 仓库里{diff.error ? `（${diff.error}）` : ""}。
            </div>
          )}
          {diff?.is_repo && (
            <>
              <div className="mb-2 flex flex-wrap gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                <span>分支 <span className="mono">{diff.branch}</span></span>
                <span>{diff.dirty ? "有未提交改动" : "工作区干净"}</span>
                {diff.truncated && <span className="chip chip-warn">diff 已截断</span>}
              </div>
              {diff.status && diff.status.length > 0 && (
                <pre className="mb-2 overflow-auto rounded-[9px] p-2 text-[10px]" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
                  {diff.status.join("\n")}
                </pre>
              )}
              <pre
                className="mt-0 mb-0 max-h-[420px] overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace", whiteSpace: "pre-wrap" }}
              >
                {diff.diff || "（没有已跟踪文件的未提交 diff）"}
              </pre>
            </>
          )}
        </div>
      )}

      {tab === "terminal" && (
        <div className="p-3">
          <div className="flex items-center gap-2">
            <input
              className="input flex-1 font-mono text-[11px]"
              placeholder="在工作区执行一条命令（非交互；30s 默认 / 60s 上限）"
              value={cmd}
              onChange={(e) => setCmd(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void runShell();
                }
              }}
            />
            <button className="btn btn-primary btn-sm" disabled={shellBusy || !cmd.trim()} onClick={() => void runShell()}>
              {shellBusy ? "执行中…" : "执行"}
            </button>
          </div>
          <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
            环境变量里的 KEY/TOKEN/SECRET/PASSWORD 会被剥掉；不提供交互式 PTY。
          </div>
          <div className="mt-2 max-h-[420px] overflow-auto">
            {shellRows.map((r, i) => (
              <div key={i} className="mb-2">
                <div className="mono text-[11px]" style={{ color: "var(--accent)" }}>$ {r.command}</div>
                <div className="text-[10px]" style={{ color: r.exit_code === 0 ? "var(--ok)" : "var(--bad)" }}>
                  exit {r.exit_code} · {r.duration_ms}ms
                </div>
                <pre
                  className="mt-1 mb-0 overflow-auto rounded-[8px] p-2 text-[11px]"
                  style={{ background: "var(--panel-2)", border: "1px solid var(--border)", whiteSpace: "pre-wrap", maxHeight: 240 }}
                >
                  {r.stdout || r.stderr || "（无输出）"}
                </pre>
              </div>
            ))}
            {shellRows.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>还没有执行记录。</div>}
          </div>
        </div>
      )}

      {tab === "build" && (
        <div className="p-3">
          <div className="mb-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            最新构建日志：{buildArtifact ? `artifact #${buildArtifact.id} · ${buildArtifact.path}` : "还没有构建产物"}
          </div>
          <pre
            className="mt-0 mb-0 max-h-[420px] overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
            style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace", whiteSpace: "pre-wrap" }}
          >
            {buildArtifact ? (log ?? (busy ? "读取中…" : "")) : "跑一次 build 后这里能看到容器里的编译日志。"}
          </pre>
        </div>
      )}
    </section>
  );
}
