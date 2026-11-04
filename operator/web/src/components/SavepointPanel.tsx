import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { api, size, when } from "../lib/api";
import type { Savepoint, SavepointDiff, SavepointRestoreResult, TaskDetail } from "../types";

const KIND_LABEL: Record<string, string> = {
  manual: "手动",
  auto: "自动",
  rescue: "回滚前",
  // 优化轮次（P1）：基线 / 候选 / 已采纳版本 / 否决
  baseline: "基线",
  candidate: "候选",
  accept: "版本",
  rejected: "否决",
};
const KIND_COLOR: Record<string, string> = {
  manual: "var(--accent)",
  auto: "var(--fg-mute)",
  rescue: "var(--warn)",
  baseline: "var(--fg-dim)",
  candidate: "var(--warn)",
  accept: "var(--ok)",
  rejected: "var(--bad)",
};

const FILE_STATUS_LABEL: Record<string, string> = {
  added: "新增",
  deleted: "删除",
  modified: "修改",
  renamed: "重命名",
};
const FILE_STATUS_COLOR: Record<string, string> = {
  added: "var(--ok)",
  deleted: "var(--bad)",
  modified: "var(--run)",
  renamed: "var(--warn)",
};

/** 证据 run chips 最多显示几个（其余折成 +N）。 */
const MAX_EVIDENCE_RUNS = 6;

/** API 错误 → 后端 detail 文案（与 TaskDeleteDialog 同口径：`<status> {"detail": "..."}`）。 */
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

/** 默认排序：新 → 旧（后端已排，前端再兜一层，防御老数据/时区偏移）。 */
function sortNewestFirst(list: Savepoint[]): Savepoint[] {
  const t = (iso: string) => {
    const n = Date.parse(iso);
    return Number.isFinite(n) ? n : 0;
  };
  return [...list].sort((a, b) => t(b.created_at) - t(a.created_at) || b.id - a.id);
}

function scopeChip(scope: string) {
  return (
    <span
      className="chip !py-0 !text-[10px]"
      style={{ color: scope === "task" ? "var(--accent)" : "var(--warn)" }}
    >
      {scope === "task" ? "任务" : "共享"}
    </span>
  );
}

function changedCounts(changed: SavepointDiff["changed"]) {
  return {
    task: changed.filter((f) => f.scope === "task").length,
    shared: changed.filter((f) => f.scope === "shared").length,
  };
}

/** 对比当前：in_sync + 变更文件表 + 统一 diff（<details> 收起）。 */
function DiffBody({ diff }: { diff: SavepointDiff }) {
  const counts = changedCounts(diff.changed);
  return (
    <div className="py-1">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className={diff.in_sync ? "chip chip-ok" : "chip chip-warn"}>
          {diff.in_sync ? "与工作区一致" : "有差异"}
        </span>
        <span className="mono" style={{ color: "var(--fg-mute)" }}>
          存档 {diff.saved_sha256.slice(0, 12)}…
        </span>
        <span className="mono" style={{ color: "var(--fg-mute)" }}>
          当前 {diff.current_sha256 ? `${diff.current_sha256.slice(0, 12)}…` : "工作区不存在"}
        </span>
        <span style={{ color: "var(--fg-dim)" }}>
          变更 {diff.changed.length} 个文件（任务 {counts.task} / 共享 {counts.shared}）
        </span>
        <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
          in_sync 为全量口径：别的任务改共享文件也会让它不一致
        </span>
      </div>

      {diff.changed.length > 0 ? (
        <div className="mt-1 max-h-[220px] overflow-auto">
          <table>
            <thead>
              <tr>
                <th>路径</th>
                <th>状态</th>
                <th>+ / -</th>
                <th>范围</th>
              </tr>
            </thead>
            <tbody>
              {diff.changed.map((f) => (
                <tr key={`${f.path}:${f.status}`}>
                  <td className="mono break-all">{f.path}</td>
                  <td>
                    <span
                      className="chip !py-0 !text-[10px]"
                      style={{ color: FILE_STATUS_COLOR[f.status] ?? "var(--fg-dim)" }}
                    >
                      {FILE_STATUS_LABEL[f.status] ?? f.status}
                    </span>
                  </td>
                  <td className="mono">
                    <span style={{ color: "var(--ok)" }}>+{f.additions}</span>
                    {" / "}
                    <span style={{ color: "var(--bad)" }}>-{f.deletions}</span>
                  </td>
                  <td>{scopeChip(f.scope)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          工作区与该存档没有文件差异。
        </div>
      )}

      {diff.outside_scope.length > 0 && (
        <div className="mt-1 break-all text-[10px]" style={{ color: "var(--warn)" }}>
          任务范围外（共享）差异：{diff.outside_scope.join("；")}
        </div>
      )}

      {diff.diff ? (
        <details className="mt-1">
          <summary className="cursor-pointer text-[10px]" style={{ color: "var(--fg-mute)" }}>
            统一 diff{diff.diff_truncated ? "（已截断，仅前 200000 字符）" : ""}
          </summary>
          <pre
            className="mt-1 mb-0 max-h-[360px] overflow-auto rounded-[8px] p-2 text-[10px] leading-[1.6]"
            style={{
              background: "var(--panel-2)",
              border: "1px solid var(--border)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {diff.diff}
          </pre>
        </details>
      ) : (
        <div className="mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
          没有文本差异（可能只剩二进制/空目录变化）。
        </div>
      )}
    </div>
  );
}

/** 回滚结果：rescue id / skipped / 范围口径 in_sync。 */
function RestoreResultBody({ result }: { result: SavepointRestoreResult }) {
  return (
    <div className="py-1 text-[11px]">
      <div className="flex flex-wrap items-center gap-2">
        <span className={result.restored ? "chip chip-ok" : "chip"}>
          {result.restored ? "已回滚" : "无需回滚"}
        </span>
        {result.rescue && <span className="chip chip-warn">急救存档 #{result.rescue.id}</span>}
        <span className={result.in_sync ? "chip chip-ok" : "chip chip-bad"}>
          {result.in_sync ? "范围口径一致" : "范围口径仍有差异"}
        </span>
        <span className="mono" style={{ color: "var(--fg-mute)" }}>
          当前 {result.current_sha256 ? `${result.current_sha256.slice(0, 12)}…` : "—"}
        </span>
        <span style={{ color: "var(--fg-dim)" }}>应用 {result.changed.length} 个文件变更</span>
        {result.skipped_outside_scope.length > 0 && (
          <span className="chip" style={{ color: "var(--warn)" }}>
            跳过 {result.skipped_outside_scope.length} 个共享文件
          </span>
        )}
      </div>
      {!result.restored && (
        <div className="mt-1" style={{ color: "var(--fg-dim)" }}>
          当前工作区已与该存档一致，未改动文件（也未生成急救存档）。
        </div>
      )}
      {result.skipped_outside_scope.length > 0 && (
        <div className="mt-1 break-all mono text-[10px]" style={{ color: "var(--warn)" }}>
          跳过（任务范围外）：{result.skipped_outside_scope.join("；")}
        </div>
      )}
      {result.note && (
        <div className="mt-1" style={{ color: "var(--fg-dim)" }}>
          备注：{result.note}
        </div>
      )}
    </div>
  );
}

/**
 * 版本存档（savepoints v1）：源码归档 + 存盘点。
 *
 * 「跑过哪版」的证据是 run 的 code_snapshot_sha256；这里把当时**源码内容**存成 tar.gz，
 * 于是可以对比当前工作区、把任务范围回滚回去（回滚前自动做 rescue 存档，可再回滚回来）。
 * 契约见 `docs/plan-savepoints-v1.md` §4 / §7。
 */
export default function SavepointPanel({ task, onChanged }: { task: TaskDetail; onChanged: () => void }) {
  const [rows, setRows] = useState<Savepoint[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // 存档当前版本（label/note 可选）
  const [formOpen, setFormOpen] = useState(false);
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [creating, setCreating] = useState(false);

  // 对比当前（单行展开）
  const [diffFor, setDiffFor] = useState<number | null>(null);
  const [diff, setDiff] = useState<SavepointDiff | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const diffSeq = useRef(0);

  // 回滚确认（先取 diff，再内联确认）
  const [plan, setPlan] = useState<{ sp: Savepoint; diff: SavepointDiff } | null>(null);
  const [planLoadingId, setPlanLoadingId] = useState<number | null>(null);
  const [scope, setScope] = useState<"task" | "full">("task");
  const [restoreNote, setRestoreNote] = useState("");
  const [restoring, setRestoring] = useState(false);
  const [result, setResult] = useState<SavepointRestoreResult | null>(null);

  const [deletingId, setDeletingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      setRows(sortNewestFirst(await api.listSavepoints(task.id)));
      setErr(null);
    } catch (e) {
      setErr(errorText(e));
      setRows((prev) => prev ?? []);
    }
  }, [task.id]);

  useEffect(() => {
    let alive = true;
    setRows(null);
    setErr(null);
    setFormOpen(false);
    setDiffFor(null);
    setDiff(null);
    setPlan(null);
    setResult(null);
    setDeletingId(null);
    api
      .listSavepoints(task.id)
      .then((list) => {
        if (alive) setRows(sortNewestFirst(list));
      })
      .catch((e) => {
        if (!alive) return;
        setErr(errorText(e));
        setRows([]);
      });
    return () => {
      alive = false;
    };
  }, [task.id]);

  const create = async () => {
    setCreating(true);
    setErr(null);
    try {
      const body: { label?: string; note?: string } = {};
      if (label.trim()) body.label = label.trim();
      if (note.trim()) body.note = note.trim();
      await api.createSavepoint(task.id, body);
      setLabel("");
      setNote("");
      setFormOpen(false);
      await load();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setCreating(false);
    }
  };

  const toggleDiff = async (sp: Savepoint) => {
    setErr(null);
    if (diffFor === sp.id) {
      setDiffFor(null);
      setDiff(null);
      return;
    }
    const seq = ++diffSeq.current;
    setDiffFor(sp.id);
    setDiff(null);
    setDiffLoading(true);
    try {
      const data = await api.savepointDiff(sp.id);
      if (seq === diffSeq.current) setDiff(data);
    } catch (e) {
      if (seq === diffSeq.current) {
        setErr(errorText(e));
        setDiffFor(null);
      }
    } finally {
      if (seq === diffSeq.current) setDiffLoading(false);
    }
  };

  /** 回滚第一步：先看 diff（展示变更 + 共享文件警告），再让用户确认 scope/note。 */
  const startRestore = async (sp: Savepoint) => {
    setErr(null);
    setResult(null);
    setDiffFor(null);
    setDiff(null);
    setPlan(null);
    setPlanLoadingId(sp.id);
    try {
      const d = await api.savepointDiff(sp.id);
      setPlan({ sp, diff: d });
      setScope("task");
      setRestoreNote("");
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setPlanLoadingId(null);
    }
  };

  const confirmRestore = async () => {
    if (!plan) return;
    const body: { scope: "task" | "full"; note?: string } = { scope };
    if (restoreNote.trim()) body.note = restoreNote.trim();
    setRestoring(true);
    setErr(null);
    try {
      const res = await api.restoreSavepoint(plan.sp.id, body);
      setPlan(null);
      setResult(res);
      await load();
      // 回滚改了工作区：通知父级重拉任务 / 门禁 / 阶段对齐。
      onChanged();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setRestoring(false);
    }
  };

  const remove = async (sp: Savepoint) => {
    if (!window.confirm(`删除存档「${sp.label}」？\n只删除存点记录，源码归档文件保留（可能被其它存点引用）。`)) return;
    setDeletingId(sp.id);
    setErr(null);
    try {
      await api.deleteSavepoint(sp.id);
      if (diffFor === sp.id) {
        setDiffFor(null);
        setDiff(null);
      }
      if (plan?.sp.id === sp.id) setPlan(null);
      if (result?.savepoint.id === sp.id) setResult(null);
      await load();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setDeletingId(null);
    }
  };

  /** 列表证据 run：任务里 code_snapshot_sha256 与存档 sha 相等的 run（最多 6 个，新的在前）。 */
  const evidenceRuns = (sp: Savepoint) =>
    task.runs.filter((r) => r.code_snapshot_sha256 === sp.code_snapshot_sha256).sort((a, b) => b.id - a.id);

  const planCounts = plan ? changedCounts(plan.diff.changed) : null;

  return (
    <section className="card mb-4 overflow-hidden">
      {/* 头部：标题 + 说明 + 存档 / 刷新 */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">版本存档</h2>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          源码按内容哈希归档；回滚前自动做急救存档，可再滚回来
        </span>
        <div className="ml-auto flex items-center gap-2">
          <button className="btn btn-primary btn-sm" onClick={() => setFormOpen((v) => !v)} disabled={creating}>
            {formOpen ? "收起" : "存档当前版本"}
          </button>
          <button
            className="btn btn-sm"
            onClick={() => void load()}
            disabled={creating || diffLoading || planLoadingId !== null || restoring || deletingId !== null}
          >
            刷新
          </button>
        </div>
      </div>

      {formOpen && (
        <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="grid gap-2 md:grid-cols-2">
            <input
              className="input"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="存档名（可选，缺省「存档 YYYY-MM-DD HH:mm」）"
              disabled={creating}
            />
            <input
              className="input"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="备注（可选）"
              disabled={creating}
            />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button className="btn btn-primary btn-sm" onClick={() => void create()} disabled={creating}>
              {creating ? "存档中…" : "确认存档"}
            </button>
            <button className="btn btn-sm" onClick={() => setFormOpen(false)} disabled={creating}>
              取消
            </button>
            <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
              对当前工作区打包 tar.gz；同一内容只存一份归档
            </span>
          </div>
        </div>
      )}

      {err && (
        <div className="px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>
          {err}
        </div>
      )}

      {rows === null && (
        <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-mute)" }}>
          读取存档…
        </div>
      )}

      {rows !== null && rows.length === 0 && !err && (
        <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-dim)" }}>
          还没有版本存档。每次真实执行 / 改代码前会自动归档；点「存档当前版本」可手动留一个可回滚的基线。
        </div>
      )}

      {rows !== null && rows.length > 0 && (
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>存档</th>
                <th>时间</th>
                <th>代码快照</th>
                <th>规格</th>
                <th>证据 run</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((sp) => {
                const runs = evidenceRuns(sp);
                return (
                  <Fragment key={sp.id}>
                    <tr>
                      <td>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <b className="text-[12px] font-semibold">{sp.label}</b>
                          <span
                            className="chip !py-0 !text-[10px]"
                            style={{ color: KIND_COLOR[sp.kind] ?? "var(--fg-dim)" }}
                            title={`类型：${KIND_LABEL[sp.kind] ?? sp.kind}`}
                          >
                            {KIND_LABEL[sp.kind] ?? sp.kind}
                          </span>
                          {sp.parent_id != null && (
                            <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              为回滚 #{sp.parent_id} 而做
                            </span>
                          )}
                          {sp.in_sync ? (
                            <span
                              className="chip chip-ok !py-0 !text-[10px]"
                              title="全量工作区快照与该存档一致"
                            >
                              工作区一致
                            </span>
                          ) : (
                            <span
                              className="chip !py-0 !text-[10px]"
                              style={{ color: "var(--fg-mute)" }}
                              title="全量口径不一致：工作区是项目共享的，别的任务改动也会让它 false；按任务范围的差异看「对比当前」"
                            >
                              有差异
                            </span>
                          )}
                        </div>
                        {sp.note && (
                          <div className="mt-0.5 max-w-[320px] truncate text-[10px]" style={{ color: "var(--fg-mute)" }} title={sp.note}>
                            {sp.note}
                          </div>
                        )}
                        <div className="mono mt-0.5 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          {size(sp.archive_bytes)}
                        </div>
                      </td>
                      <td className="text-[11px]" style={{ color: "var(--fg-dim)" }}>
                        {when(sp.created_at)}
                      </td>
                      <td>
                        <span className="mono" title={sp.code_snapshot_sha256}>
                          {sp.code_snapshot_sha256.slice(0, 12)}…
                        </span>
                      </td>
                      <td className="text-[11px]">
                        {sp.spec_version != null ? (
                          <span className="mono">v{sp.spec_version}</span>
                        ) : (
                          <span style={{ color: "var(--fg-mute)" }}>—</span>
                        )}
                        {sp.criteria_ref && (
                          <div className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                            {sp.criteria_ref}
                          </div>
                        )}
                      </td>
                      <td>
                        <div className="flex max-w-[280px] flex-wrap gap-1">
                          {runs.slice(0, MAX_EVIDENCE_RUNS).map((r) => (
                            <span
                              key={r.id}
                              className={`chip !py-0 !text-[10px] ${
                                r.status === "ok" ? "chip-ok" : r.status === "failed" ? "chip-bad" : ""
                              }`}
                              title={`${r.action} ${r.status}${r.device_id ? ` · device ${r.device_id}` : ""}${
                                r.loop_id != null ? ` · loop #${r.loop_id}` : ""
                              }`}
                            >
                              #{r.id} {r.action} {r.status}
                            </span>
                          ))}
                          {runs.length > MAX_EVIDENCE_RUNS && (
                            <span
                              className="chip !py-0 !text-[10px]"
                              style={{ color: "var(--fg-mute)" }}
                              title={runs
                                .slice(MAX_EVIDENCE_RUNS)
                                .map((r) => `#${r.id} ${r.action} ${r.status}`)
                                .join("\n")}
                            >
                              +{runs.length - MAX_EVIDENCE_RUNS}
                            </span>
                          )}
                          {runs.length === 0 && (
                            <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              无匹配 run
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          <button
                            className="btn btn-sm"
                            onClick={() => void toggleDiff(sp)}
                            disabled={diffLoading && diffFor === sp.id}
                          >
                            {diffFor === sp.id ? "收起对比" : "对比当前"}
                          </button>
                          <button
                            className="btn btn-sm"
                            onClick={() => void startRestore(sp)}
                            disabled={planLoadingId !== null || restoring}
                          >
                            {planLoadingId === sp.id ? "读取差异…" : "回滚"}
                          </button>
                          <a
                            className="btn btn-sm"
                            href={api.savepointArchiveUrl(sp.id)}
                            target="_blank"
                            rel="noreferrer"
                            title="下载源码归档（tar.gz）"
                          >
                            下载
                          </a>
                          <button
                            className="btn btn-sm btn-danger"
                            onClick={() => void remove(sp)}
                            disabled={deletingId !== null}
                          >
                            {deletingId === sp.id ? "删除中…" : "删除"}
                          </button>
                        </div>
                      </td>
                    </tr>

                    {/* 对比当前 */}
                    {diffFor === sp.id && (
                      <tr>
                        <td colSpan={6} style={{ background: "var(--panel-2)" }}>
                          {diffLoading && !diff ? (
                            <div className="py-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                              读取差异…
                            </div>
                          ) : diff ? (
                            <DiffBody diff={diff} />
                          ) : null}
                        </td>
                      </tr>
                    )}

                    {/* 回滚确认 */}
                    {plan?.sp.id === sp.id && planCounts && (
                      <tr>
                        <td colSpan={6} style={{ background: "var(--panel-3)" }}>
                          <div className="py-2">
                            <div className="flex flex-wrap items-center gap-2 text-[11px]">
                              <b>回滚到「{plan.sp.label}」</b>
                              <span className="chip !py-0 !text-[10px]">#{plan.sp.id}</span>
                              <span style={{ color: "var(--fg-dim)" }}>
                                {scope === "full"
                                  ? `将应用 ${plan.diff.changed.length} 个文件变更（任务 ${planCounts.task} / 共享 ${planCounts.shared}）`
                                  : `将应用任务范围 ${planCounts.task} 个文件变更${
                                      planCounts.shared ? `；共享 ${planCounts.shared} 个将跳过` : ""
                                    }`}
                              </span>
                            </div>
                            <div className="mono mt-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                              存档 {plan.diff.saved_sha256.slice(0, 12)}… → 当前{" "}
                              {plan.diff.current_sha256 ? `${plan.diff.current_sha256.slice(0, 12)}…` : "工作区不存在"}
                            </div>

                            {plan.diff.changed.length > 0 && (
                              <div
                                className="mt-1 max-h-[160px] overflow-auto rounded-[8px] px-2 py-1"
                                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
                              >
                                {plan.diff.changed.map((f) => (
                                  <div
                                    key={`${f.path}:${f.status}`}
                                    className="mono break-all text-[10px]"
                                    style={{ color: f.scope === "shared" ? "var(--warn)" : "var(--fg-dim)" }}
                                  >
                                    [{f.scope === "task" ? "任务" : "共享"}] {f.path}（
                                    {FILE_STATUS_LABEL[f.status] ?? f.status} +{f.additions}/-{f.deletions}）
                                  </div>
                                ))}
                              </div>
                            )}

                            <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px]">
                              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                                回滚范围
                              </span>
                              <label className="flex items-center gap-1.5" style={{ color: "var(--fg-dim)" }}>
                                <input
                                  type="radio"
                                  name={`sp-scope-${plan.sp.id}`}
                                  checked={scope === "task"}
                                  onChange={() => setScope("task")}
                                  disabled={restoring}
                                />
                                任务范围（推荐）
                              </label>
                              <label className="flex items-center gap-1.5" style={{ color: "var(--fg-dim)" }}>
                                <input
                                  type="radio"
                                  name={`sp-scope-${plan.sp.id}`}
                                  checked={scope === "full"}
                                  onChange={() => setScope("full")}
                                  disabled={restoring}
                                />
                                全工作区
                              </label>
                              {!task.op_name && (
                                <span className="text-[10px]" style={{ color: "var(--warn)" }}>
                                  该任务没有算子名（所有路径都算共享）→ 任务范围会被后端拒绝，请选「全工作区」
                                </span>
                              )}
                            </div>

                            {scope === "full" ? (
                              <div
                                className="mt-2 rounded-[8px] px-3 py-2 text-[11px]"
                                style={{ border: "1px solid var(--bad)", color: "var(--bad)" }}
                              >
                                全工作区回滚会把整个工作区还原成该存档（含其它任务 / 共享文件），可能覆盖别人的改动。
                                {plan.diff.outside_scope.length > 0
                                  ? `下列共享文件会被一并还原/删除：${plan.diff.outside_scope.join("；")}`
                                  : "共享范围也在回滚范围内。"}
                              </div>
                            ) : plan.diff.outside_scope.length > 0 ? (
                              <div
                                className="mt-2 rounded-[8px] px-3 py-2 text-[11px]"
                                style={{ border: "1px solid var(--warn)", color: "var(--warn)" }}
                              >
                                任务范围回滚会跳过 {plan.diff.outside_scope.length} 个任务范围外的共享文件：
                                {plan.diff.outside_scope.join("；")}
                              </div>
                            ) : null}

                            <div className="mt-2 flex flex-wrap items-center gap-2">
                              <input
                                className="input max-w-[320px]"
                                value={restoreNote}
                                onChange={(e) => setRestoreNote(e.target.value)}
                                placeholder="备注（可选，记到回滚结果里）"
                                disabled={restoring}
                              />
                              <button className="btn btn-primary btn-sm" onClick={() => void confirmRestore()} disabled={restoring}>
                                {restoring ? "回滚中…" : "确认回滚"}
                              </button>
                              <button className="btn btn-sm" onClick={() => setPlan(null)} disabled={restoring}>
                                取消
                              </button>
                              <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                                回滚前会先自动做一份急救存档；有运行中作业时会被拒绝
                              </span>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}

                    {/* 回滚结果 */}
                    {result !== null && result.savepoint.id === sp.id && (
                      <tr>
                        <td colSpan={6} style={{ background: "var(--panel-2)" }}>
                          <RestoreResultBody result={result} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="px-4 py-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
        存点只记「标签 + 环境引用 + 时间 + 类型」，源码按 sha 去重存一份；旧 run 仍是旧快照的证据（A/B 配对不受回滚影响）。
      </div>
    </section>
  );
}
