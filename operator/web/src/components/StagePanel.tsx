import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { StageBoard, StageTaskView } from "../types";

const TASKS_PER_COLUMN_PAGE = 4;

/**
 * 阶段看板（P3 读侧 / 方案 §5）：任务按**推导阶段**分组。
 *
 * 阶段由真实证据推导（各动作最近一次 run 的门禁 verdict + 是否有规格），不信手填的 `stage` 字段；
 * 存值与推导不一致时，卡片上可「按证据同步」（写 `stage_events` + 更新 `tasks.stage`）。
 */

function GateDots({ view }: { view: StageTaskView }) {
  return (
    <div className="flex items-center gap-1.5">
      {view.stages.map((s) => (
        <span
          key={s.stage}
          title={`${s.name}：${s.ok ? "已达成" : "未达成"}${s.evidence.length ? "（" + s.evidence.join("；") + "）" : ""}`}
          className="inline-block h-[9px] w-[9px] rounded-full"
          style={{
            background: s.ok ? "var(--ok)" : "var(--panel-2)",
            border: `1.5px solid ${s.ok ? "var(--ok)" : "var(--border-strong)"}`,
          }}
        />
      ))}
    </div>
  );
}

export default function StagePanel({
  onOpenTask,
  onOpenDraft,
}: {
  onOpenTask: (id: number) => void;
  /** 草稿卡片走「任务开始」继续编辑/定版（任务详情作业台没有该入口）。 */
  onOpenDraft?: (id: number) => void;
}) {
  const [board, setBoard] = useState<StageBoard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [includeDrafts, setIncludeDrafts] = useState(false);
  /** 每个阶段列各自翻页，避免任务多的一列撑高整块看板。 */
  const [columnPages, setColumnPages] = useState<Record<string, number>>({});

  const load = useCallback(async () => {
    setBusy(true);
    try {
      setBoard(await api.stages(includeDrafts));
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, [includeDrafts]);

  useEffect(() => {
    void load();
  }, [load]);

  const sync = async (id: number) => {
    setBusy(true);
    try {
      await api.syncStage(id);
      await load();
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  };

  const changeColumnPage = (stage: string, nextPage: number, totalPages: number) => {
    const page = Math.max(1, Math.min(nextPage, totalPages));
    setColumnPages((current) => ({ ...current, [stage]: page }));
  };

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px]">
        <span style={{ color: "var(--fg-dim)" }}>
          共 {board?.total ?? 0} 个正式任务
          {board && board.out_of_sync.length > 0 && ` · 存值与证据不一致 ${board.out_of_sync.length} 个`}
        </span>
        <button className="btn btn-sm" onClick={() => void load()} disabled={busy}>{busy ? "…" : "刷新"}</button>
        <label className="ml-auto flex items-center gap-1" style={{ color: "var(--fg-mute)" }}>
          <input type="checkbox" checked={includeDrafts} onChange={(e) => setIncludeDrafts(e.target.checked)} />
          含草稿
        </label>
      </div>

      {err && <div className="card mb-4 px-4 py-3 text-xs" style={{ color: "var(--bad)" }}>{err}</div>}

      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(4, minmax(0,1fr))" }}>
        {(board?.columns ?? []).map((col) => {
          const totalPages = Math.max(1, Math.ceil(col.tasks.length / TASKS_PER_COLUMN_PAGE));
          // 任务数减少后即时夹到有效页，不让用户看到空列。
          const page = Math.min(columnPages[col.stage] ?? 1, totalPages);
          const visibleTasks = col.tasks.slice(
            (page - 1) * TASKS_PER_COLUMN_PAGE,
            page * TASKS_PER_COLUMN_PAGE,
          );

          return (
            <section key={col.stage} className="card flex flex-col overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-2" style={{ borderBottom: "1px solid var(--border)" }}>
                <h2 className="m-0 text-xs font-semibold">{col.stage}</h2>
                <span className="chip ml-auto">{col.tasks.length}</span>
              </div>
              <div className="flex flex-1 flex-col gap-2 p-2">
                {col.tasks.length === 0 && (
                  <div className="px-1 py-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>—</div>
                )}
                {visibleTasks.map((v) => (
                <div key={v.task_id} className="rounded-md p-3" style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}>
                  <div className="flex items-center gap-2">
                    <button
                      className="min-w-0 flex-1 truncate text-left text-xs font-semibold"
                      onClick={() => (v.is_draft && onOpenDraft ? onOpenDraft(v.task_id) : onOpenTask(v.task_id))}
                      title={`数据库任务 ID #${v.task_id} · ${v.title}${v.op_name ? ` · op_name=${v.op_name}` : ""}`}
                    >
                      任务 #{v.task_id} · {v.title}
                    </button>
                    <GateDots view={v} />
                  </div>
                  <div className="mt-1 text-[11px]" style={{ color: "var(--fg-dim)" }}>
                    {v.stages.map((s) => (
                      <span key={s.stage} title={s.evidence.join("；")}>
                        {s.ok ? "✓" : "·"}{s.stage.slice(0, 2)}
                      </span>
                    ))}
                  </div>
                  <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    {v.next_action ? `当前进展：${v.next_action}` : `下一步：${v.next_step}`}
                  </div>
                  {!v.in_sync && (
                    <div className="mt-2 flex items-center gap-2">
                      <span className="chip" style={{ color: "var(--fg-mute)" }} title="手填的 stage 字段">
                        存：{v.stored_stage}
                      </span>
                      <button className="btn btn-sm ml-auto" onClick={() => void sync(v.task_id)} disabled={busy}>
                        按证据同步
                      </button>
                    </div>
                  )}
                  {v.last_event && (
                    <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      末次推进：{v.last_event.from ?? "—"} → {v.last_event.to}
                    </div>
                  )}
                </div>
                ))}
                {totalPages > 1 && (
                  <div
                    className="mt-auto flex items-center justify-between gap-2 px-1 pt-1 text-[10px]"
                    style={{ color: "var(--fg-mute)" }}
                  >
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={page <= 1}
                      aria-label={`${col.stage}上一页`}
                      onClick={() => changeColumnPage(col.stage, page - 1, totalPages)}
                    >
                      上一页
                    </button>
                    <span>第 {page} / {totalPages} 页</span>
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={page >= totalPages}
                      aria-label={`${col.stage}下一页`}
                      onClick={() => changeColumnPage(col.stage, page + 1, totalPages)}
                    >
                      下一页
                    </button>
                  </div>
                )}
              </div>
            </section>
          );
        })}
      </div>

      {board && (
        <div className="mt-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
          {board.notes.map((n, i) => <div key={i}>· {n}</div>)}
        </div>
      )}
    </>
  );
}
