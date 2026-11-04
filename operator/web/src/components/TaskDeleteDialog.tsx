import { useCallback, useEffect, useState } from "react";
import { api, size } from "../lib/api";
import type { TaskDeletePreview, TaskDeleteResult } from "../types";

const COUNT_LABELS: [keyof TaskDeletePreview["counts"], string][] = [
  ["spec_versions", "规格版本"],
  ["loops", "轮次"],
  ["optimization_ideas", "优化思路"],
  ["idea_searches", "思路搜索"],
  ["loop_budget_audits", "预算审计"],
  ["agent_jobs", "智能体调用"],
  ["runs", "真实执行"],
  ["savepoints", "版本存档"],
  ["artifacts", "产物"],
  ["action_jobs", "执行作业"],
  ["stage_events", "阶段记录"],
  ["conversations", "对话"],
  ["messages", "消息"],
  ["attachments", "附件"],
];

const FILE_LABEL: Record<string, string> = {
  context: "任务上下文空间",
  chat: "需求澄清目录",
  conversation: "对话附件目录",
  run: "本机 run 目录",
};

function errorText(e: unknown): string {
  const raw = String(e);
  const i = raw.indexOf("{");
  if (i >= 0) {
    try {
      const body = JSON.parse(raw.slice(i));
      if (typeof body.detail === "string") return body.detail;
      if (body.detail) return JSON.stringify(body.detail);
    } catch {
      // 保持原文
    }
  }
  return raw;
}

/**
 * 删除任务确认框：先看影响面，再决定要不要清本机文件；
 * 有真实执行（runs）的任务必须输入算子名，避免在列表里误点。
 */
export default function TaskDeleteDialog({
  taskId,
  onClose,
  onDeleted,
}: {
  taskId: number;
  onClose: () => void;
  onDeleted: (taskId: number) => void;
}) {
  const [preview, setPreview] = useState<TaskDeletePreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [typed, setTyped] = useState("");
  // 删除已执行、但本机文件有跳过/失败时先留在弹窗里展示清理结果，关闭时再通知父级。
  const [result, setResult] = useState<TaskDeleteResult | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    api.deletePreview(taskId)
      .then((p) => alive && setPreview(p))
      .catch((e) => alive && setErr(errorText(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [taskId]);

  // 已删成功且带清理结果时，关闭动作也要把「已删除」告诉父级，避免列表残留。
  const closeDialog = useCallback(() => {
    if (busy) return;
    if (result && preview) onDeleted(preview.task_id);
    else onClose();
  }, [busy, result, preview, onDeleted, onClose]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeDialog();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeDialog]);

  const runningCount = preview ? preview.running.action_jobs.length + preview.running.agent_jobs.length : 0;
  const requiredText = preview && preview.counts.runs > 0 ? (preview.op_name || preview.title) : "";
  const typedOk = !requiredText || typed.trim() === requiredText;
  const canDelete = Boolean(preview) && !loading && !busy && runningCount === 0 && typedOk;
  const fileItems = preview?.files.items.filter((f) => f.exists) ?? [];
  const removableBytes = preview?.files.total_bytes ?? 0;
  const shownCounts = preview ? COUNT_LABELS.filter(([k]) => preview.counts[k] > 0) : [];

  const submit = async () => {
    if (!preview || !canDelete) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.deleteTask(preview.task_id, deleteFiles);
      // 本机文件有跳过/失败：先展示清理结果，用户点「关闭」后再通知父级（任务记录此时已删）。
      if (res.files.skipped.length > 0 || res.files.errors.length > 0) {
        setResult(res);
        return;
      }
      onDeleted(preview.task_id);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "rgba(10, 11, 20, 0.42)" }}
      onClick={closeDialog}
      role="dialog"
      aria-modal="true"
      aria-label="删除任务"
    >
      <div
        className="card w-full max-w-[600px] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="m-0 text-sm font-semibold" style={{ color: result ? "var(--ok)" : "var(--bad)" }}>
            {result ? "删除完成" : "删除任务"}
          </h2>
          <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>硬删除，不可恢复</span>
        </div>

        <div className="max-h-[68vh] overflow-auto px-4 py-3">
          {result && preview && (
            <div className="py-1">
              <div className="text-[12px]">
                任务 #{preview.task_id}「{preview.title}」已删除，控制台记录已清理。
              </div>
              {result.files.requested ? (
                <>
                  <div className="mt-3 text-[11px]">
                    本机文件：删除 <b>{result.files.deleted.length}</b> 项 · 跳过{" "}
                    <b style={{ color: result.files.skipped.length ? "var(--warn)" : undefined }}>{result.files.skipped.length}</b> 项 · 失败{" "}
                    <b style={{ color: result.files.errors.length ? "var(--bad)" : undefined }}>{result.files.errors.length}</b> 项
                  </div>
                  {result.files.skipped.length > 0 && (
                    <div className="mt-2 space-y-1 text-[10px]" style={{ color: "var(--warn)" }}>
                      {result.files.skipped.map((s) => (
                        <div key={`skip-${s.path}`}>
                          跳过 <span className="mono break-all">{s.path}</span>：{s.reason}
                        </div>
                      ))}
                    </div>
                  )}
                  {result.files.errors.length > 0 && (
                    <div className="mt-2 space-y-1 text-[10px]" style={{ color: "var(--bad)" }}>
                      {result.files.errors.map((s) => (
                        <div key={`err-${s.path}`}>
                          失败 <span className="mono break-all">{s.path}</span>：{s.error}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="mt-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                  未勾选删除本机文件，任务目录仍留在磁盘上，可之后手工清理。
                </div>
              )}
            </div>
          )}

          {!result && loading && <div className="py-8 text-center text-[12px]" style={{ color: "var(--fg-mute)" }}>正在统计影响面…</div>}

          {!result && preview && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-[12px]">
                <b>{preview.title}</b>
                {preview.op_name && preview.op_name !== preview.title && (
                  <span className="mono" style={{ color: "var(--fg-dim)" }}>{preview.op_name}</span>
                )}
                {preview.is_draft && <span className="chip">草稿</span>}
              </div>
              <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                任务 #{preview.task_id} · 当前阶段 {preview.stage}
              </div>

              {runningCount > 0 && (
                <div className="mt-3 rounded-[9px] px-3 py-2 text-[11px]" style={{ border: "1px solid var(--warn)", color: "var(--warn)" }}>
                  还有 {runningCount} 个运行中的作业（动作 {preview.running.action_jobs.length} / 调用 {preview.running.agent_jobs.length}）。
                  先取消或等结束，再删除任务。
                </div>
              )}

              <div className="mt-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>将删除的记录</div>
              <div className="mt-1 grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                <div className="rounded-[7px] px-2.5 py-2 text-[11px]" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
                  <div style={{ color: "var(--fg-mute)" }}>任务记录</div>
                  <div className="mt-0.5 text-[13px] font-semibold">1</div>
                </div>
                {shownCounts.map(([key, label]) => (
                  <div key={key} className="rounded-[7px] px-2.5 py-2 text-[11px]" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
                    <div style={{ color: "var(--fg-mute)" }}>{label}</div>
                    <div className="mt-0.5 text-[13px] font-semibold">{preview.counts[key]}</div>
                  </div>
                ))}
              </div>
              {shownCounts.length === 0 && (
                <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>该任务还没有规格、调用或执行记录。</div>
              )}

              <div className="mt-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>本机文件</div>
              {fileItems.length === 0 ? (
                <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>没有找到任务专属目录或本机产物。</div>
              ) : (
                <>
                  <label className="mt-1 flex items-start gap-2 rounded-[9px] px-3 py-2 text-[11px]" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={deleteFiles}
                      onChange={(e) => setDeleteFiles(e.target.checked)}
                      disabled={busy}
                    />
                    <span>
                      同时删除本机任务目录与产物
                      <span className="ml-1" style={{ color: "var(--fg-mute)" }}>（可清理 {size(removableBytes)}）</span>
                      <div className="mt-0.5 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                        不勾选则只删控制台记录，文件留在磁盘上，之后可手工清理。
                      </div>
                    </span>
                  </label>
                  <div className="mt-1.5 space-y-1">
                    {fileItems.map((f) => (
                      <div key={`${f.kind}-${f.path}`} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                        <span className="chip !py-0">{FILE_LABEL[f.kind] ?? f.kind}</span>
                        <span className="mono flex-1 truncate" style={{ minWidth: 180 }} title={f.path}>{f.path}</span>
                        <span className="mono">{size(f.bytes)}</span>
                        {!f.removable && <span style={{ color: "var(--warn)" }}>{f.reason ?? "跳过"}</span>}
                      </div>
                    ))}
                  </div>
                </>
              )}

              {requiredText && (
                <label className="mt-3 flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                  该任务已有真实执行记录。请输入算子名 <b className="mono" style={{ color: "var(--fg)" }}>{requiredText}</b> 以确认：
                  <input
                    className="input mono"
                    value={typed}
                    onChange={(e) => setTyped(e.target.value)}
                    placeholder={requiredText}
                    disabled={busy}
                    autoFocus
                  />
                </label>
              )}

              {err && <div className="mt-3 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}
            </>
          )}

          {!result && !loading && !preview && (
            <div className="py-4 text-[11px]" style={{ color: "var(--bad)" }}>{err ?? "读取删除预览失败"}</div>
          )}
        </div>

        <div className="flex items-center gap-2 px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          {result ? (
            <>
              <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                本机文件未全部清理，请按上面的原因手工处理残留。
              </span>
              <button className="btn btn-primary ml-auto" onClick={closeDialog}>关闭</button>
            </>
          ) : (
            <>
              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                {preview?.is_draft ? "这是草稿任务" : "正式任务删除后，阶段看板与消耗统计也会移除"}
              </span>
              <button className="btn ml-auto" onClick={onClose} disabled={busy}>取消</button>
              <button
                className="btn btn-danger"
                onClick={() => void submit()}
                disabled={!canDelete}
                title={runningCount > 0 ? "先取消运行中的作业" : requiredText && !typedOk ? `请输入 ${requiredText}` : undefined}
              >
                {busy ? "删除中…" : "确认删除"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
