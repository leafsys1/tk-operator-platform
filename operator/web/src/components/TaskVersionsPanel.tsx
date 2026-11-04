import { useCallback, useEffect, useState } from "react";
import { api, when } from "../lib/api";
import type { TaskDetail, TaskVersion, TaskVersions } from "../types";

const SOURCE_LABEL: Record<string, string> = {
  spec: "规格落版",
  optimization: "优化采纳",
  manual: "手工变更",
};
const SOURCE_COLOR: Record<string, string> = {
  spec: "var(--accent)",
  optimization: "var(--ok)",
  manual: "var(--warn)",
};

/** API 错误 → 后端 detail 文案（`<status> {"detail": "..."}`）。 */
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

const versionLabel = (v: TaskVersion) => v.label?.trim() || `v${v.major}.${v.minor}`;

/**
 * 任务版本（任务生命周期 v2，契约 §2.2 / §3.4）：大版本 = 规格版本，小版本 = 优化采纳 / 手工变更。
 *
 * 版本与证据不按「run 打版本标签」，而是靠 `(spec_version_id, code_snapshot_sha256, suite_ref)`
 * 三键匹配 —— 这里「关联 run」按代码快照相等找，交付状态看 `delivered_at`。
 * 放在「版本」分区顶部；下方保留源码存档（savepoints）。
 */
export default function TaskVersionsPanel({
  task,
  onChanged,
}: {
  task: TaskDetail;
  /** 刷新任务详情后由父级触发；交付/新建版本后父级会重拉 runs 以刷新关联 run。 */
  onChanged?: () => void;
}) {
  const [data, setData] = useState<TaskVersions | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const next = await api.versions(task.id);
      setData(next);
      setErr(null);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setLoading(false);
    }
  }, [task.id]);

  useEffect(() => {
    let alive = true;
    setData(null);
    setErr(null);
    api
      .versions(task.id)
      .then((next) => {
        if (alive) setData(next);
      })
      .catch((e) => {
        if (alive) setErr(errorText(e));
      });
    return () => {
      alive = false;
    };
  }, [task.id]);

  // current 排最前，history 去重（后端 history 一般已排除 current，这里防一手）。
  const rows: TaskVersion[] = data
    ? [...(data.current ? [data.current] : []), ...data.history.filter((h) => h.id !== data.current?.id)]
    : [];

  const evidence = data?.evidence ?? null;

  /**
   * 关联 run：代码快照（+ 规格版本，若有）相等的执行记录；当前版本再并入后端 evidence.runs
   * （证据匹配含豁免/用例集对齐，比前端快照匹配更权威）。新的在前。
   */
  const runsFor = (v: TaskVersion): { id: number; action: string; status: string }[] => {
    const out = new Map<number, { id: number; action: string; status: string }>();
    for (const r of task.runs) {
      if (v.code_snapshot_sha256 == null || r.code_snapshot_sha256 !== v.code_snapshot_sha256) continue;
      if (v.spec_version_id != null && r.spec_version_id !== v.spec_version_id) continue;
      out.set(r.id, { id: r.id, action: r.action, status: r.status });
    }
    if (data?.current?.id === v.id && evidence) {
      for (const [action, run] of Object.entries(evidence.runs)) {
        if (run && !out.has(run.run_id)) out.set(run.run_id, { id: run.run_id, action, status: run.status });
      }
    }
    return [...out.values()].sort((a, b) => b.id - a.id);
  };

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">任务版本</h2>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          大版本 = 规格版本；小版本 = 采纳优化 / 交付前变更
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {evidence && (
            <span className={`chip ${evidence.deliverable ? "chip-ok" : "chip-warn"}`}>
              {evidence.deliverable ? "交付门禁齐备" : `交付门禁缺 ${evidence.missing.length} 项`}
            </span>
          )}
          <button className="btn btn-sm" onClick={() => void load()} disabled={loading}>
            {loading ? "…" : "刷新"}
          </button>
        </div>
      </div>

      {err && (
        <div className="flex flex-wrap items-center gap-2 px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>
          版本接口读取失败：{err}
          <button className="btn btn-sm" onClick={() => void load()} disabled={loading}>
            重试
          </button>
        </div>
      )}

      {!data && !err && (
        <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-mute)" }}>
          读取任务版本…
        </div>
      )}

      {data && rows.length === 0 && !err && (
        <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-dim)" }}>
          还没有任务版本记录 —— 定版后会落 v1.0；采纳优化 / 手工变更会递增小版本。
        </div>
      )}

      {rows.length > 0 && (
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>版本</th>
                <th>来源</th>
                <th>交付</th>
                <th>关联 run</th>
                <th>代码快照</th>
                <th>用例集</th>
                <th>时间</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((v) => {
                const isCurrent = data?.current?.id === v.id;
                const runs = runsFor(v);
                return (
                  <tr key={v.id} style={isCurrent ? { background: "var(--panel-3)" } : undefined}>
                    <td>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <b className="mono text-[12px] font-semibold">{versionLabel(v)}</b>
                        {isCurrent && <span className="chip chip-run !py-0 !text-[10px]">当前</span>}
                        {v.label_text && v.label_text !== versionLabel(v) && (
                          <span className="text-[10px]" style={{ color: "var(--fg-mute)" }} title="版本名">
                            {v.label_text}
                          </span>
                        )}
                        {v.note && (
                          <span className="text-[10px]" style={{ color: "var(--fg-mute)" }} title={v.note}>
                            {v.note.length > 40 ? `${v.note.slice(0, 40)}…` : v.note}
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span
                          className="chip !py-0 !text-[10px]"
                          style={{ color: SOURCE_COLOR[v.source] ?? "var(--fg-dim)" }}
                          title={`来源：${v.source}`}
                        >
                          {SOURCE_LABEL[v.source] ?? v.source}
                        </span>
                        {v.spec_version_id != null && (
                          <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                            spec #{v.spec_version_id}
                          </span>
                        )}
                        {v.loop_id != null && (
                          <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                            轮次 #{v.loop_id}
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      {v.delivered_at ? (
                        <span className="chip chip-ok !py-0 !text-[10px]" title={`交付冻结：${when(v.delivered_at)}`}>
                          已交付 · {when(v.delivered_at)}
                        </span>
                      ) : (
                        <span className="chip !py-0 !text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          未交付
                        </span>
                      )}
                    </td>
                    <td>
                      {runs.length === 0 ? (
                        <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          —
                        </span>
                      ) : (
                        <div className="flex flex-wrap gap-1">
                          {runs.slice(0, 4).map((r) => (
                            <span key={r.id} className="chip !py-0 !text-[10px]" title={`${r.action} · ${r.status}`}>
                              #{r.id} {r.action}
                            </span>
                          ))}
                          {runs.length > 4 && (
                            <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              +{runs.length - 4}
                            </span>
                          )}
                        </div>
                      )}
                    </td>
                    <td className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      {(v.code_snapshot_sha256 ?? "—").slice(0, 12)}
                    </td>
                    <td className="mono text-[10px]" style={{ color: "var(--fg-mute)", maxWidth: 180, wordBreak: "break-all" }}>
                      {v.suite_ref ?? "—"}
                    </td>
                    <td className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      {when(v.created_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {evidence && (
        <div className="flex flex-wrap items-center gap-2 px-4 py-2.5 text-[10px]" style={{ borderTop: "1px solid var(--border)", color: "var(--fg-mute)" }}>
          <span>当前版本证据：</span>
          {evidence.required_actions.map((action) => {
            const r = evidence.runs[action] ?? null;
            const ok = r?.status === "ok" && r.gate_verdict === "pass";
            return (
              <span key={action} className={`chip !py-0 !text-[10px] ${ok ? "chip-ok" : r ? "chip-bad" : ""}`}>
                {action} {r ? `#${r.run_id}` : "未跑"}
              </span>
            );
          })}
          {onChanged && (
            <button className="btn btn-sm ml-auto" onClick={onChanged} title="重拉任务详情（关联 run / 交付状态）">
              刷新任务
            </button>
          )}
        </div>
      )}
    </section>
  );
}
