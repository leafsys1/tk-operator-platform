import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CaseReport, Task } from "../types";

/**
 * 测试管理（P3 读侧 / 方案 §7 + 任务生命周期 v2）：两个 tab ——
 * 「精度用例」（矩阵 + 逐例历史 + 失败点 + test 执行记录）与
 * 「性能用例」（benchmark.json 用例 + 最近候选/基线/加速比/门禁 + benchmark 执行记录）。
 *
 * 数据都来自 `GET /api/tasks/{id}/cases`（精度段 + 性能段，契约 §3.6）。只读；判据/门禁口径归 `gate.py`。
 */

function statusChip(s: string | null | undefined) {
  if (!s) return <span className="chip" style={{ color: "var(--fg-mute)" }}>—</span>;
  const cls = s === "PASS" ? "chip-ok" : s === "FAIL" || s === "ERROR" ? "chip-bad" : "chip-run";
  return <span className={`chip ${cls}`}>{s}</span>;
}

const KIND_LABEL: Record<string, string> = { match: "须匹配", reject: "须拒绝" };

/** 默认选中的任务：优先来路任务（从任务详情「打开测试管理」带过来），否则列表里最新的一个（tasks 按 id 倒序）。 */
function pickTaskId(real: Task[], preferred?: number | null): number | null {
  if (preferred != null && real.some((t) => t.id === preferred)) return preferred;
  return real.length ? real[0].id : null;
}

export default function TestPanel({ tasks, initialTaskId = null }: { tasks: Task[]; initialTaskId?: number | null }) {
  const real = tasks.filter((t) => !t.is_draft);
  const [taskId, setTaskId] = useState<number | null>(() => pickTaskId(real, initialTaskId));
  const [report, setReport] = useState<CaseReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // 两类用例：精度（矩阵/逐例结果/执行记录）+ 性能（benchmark.json 用例，只读）。
  const [tab, setTab] = useState<"precision" | "perf">("precision");

  const load = useCallback(async () => {
    if (taskId === null) {
      setReport(null);
      return;
    }
    setBusy(true);
    try {
      setReport(await api.cases(taskId));
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, [taskId]);

  useEffect(() => {
    void load();
  }, [load]);

  // 任务列表刷新后，选中的任务没了（或初始列表还没到）→ 回退到最新任务（同「从任务详情来」的来路任务优先）
  useEffect(() => {
    if (taskId !== null && real.some((t) => t.id === taskId)) return;
    setTaskId(pickTaskId(real, initialTaskId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks, initialTaskId]);

  const perf = report?.performance ?? null;
  // 最近一次 benchmark：契约没约定顺序，按 run_id 倒序取最新。
  const latestPerfRun = perf ? [...perf.runs].sort((a, b) => b.run_id - a.run_id)[0] ?? null : null;
  const fmtUs = (v: number | null | undefined) =>
    v == null || !Number.isFinite(v) ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${v.toFixed(2)} µs`;
  const fmtSpeedup = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(2)}×`);

  if (real.length === 0) {
    return <div className="card px-4 py-3 text-xs" style={{ color: "var(--fg-mute)" }}>还没有正式任务。</div>;
  }

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="text-xs" style={{ color: "var(--fg-dim)" }}>选择任务：</span>
        <select
          className="rounded-[9px] px-3 py-1.5 text-xs"
          style={{ background: "var(--panel-2)", border: "1px solid var(--border-strong)", color: "var(--fg)" }}
          value={taskId ?? ""}
          onChange={(e) => setTaskId(Number(e.target.value))}
        >
          {real.map((t) => (
            <option key={t.id} value={t.id}>#{t.id} {t.title}</option>
          ))}
        </select>
        <button className="btn btn-sm" onClick={() => void load()} disabled={busy}>{busy ? "…" : "刷新"}</button>
      </div>

      {err && <div className="card mb-4 px-4 py-3 text-xs" style={{ color: "var(--bad)" }}>{err}</div>}

      {/* 两类用例（契约 §6）：精度 = 现有矩阵/逐例结果/执行记录；性能 = benchmark.json 只读。 */}
      <nav className="mb-3 flex flex-wrap items-center gap-1" style={{ borderBottom: "1px solid var(--border)" }}>
        <button className="wtab" aria-selected={tab === "precision"} onClick={() => setTab("precision")}>
          精度用例{report ? ` (${report.total_cases})` : ""}
        </button>
        <button className="wtab" aria-selected={tab === "perf"} onClick={() => setTab("perf")}>
          性能用例{perf ? ` (${perf.cases.length})` : ""}
        </button>
      </nav>

      {report && tab === "precision" && (
        <>
          <div className="mb-4 grid grid-cols-4 gap-3">
            {[
              { k: "算子", v: report.op_name || "—", s: report.case_groups.join(" / ") },
              { k: "用例数", v: `${report.total_cases}`, s: `必需 ${report.required_cases}` },
              { k: "判据档", v: report.criteria_ref ?? "—", s: "" },
              { k: "执行记录", v: `${report.runs.length}`, s: `失败点 ${report.failures.length}` },
            ].map((c) => (
              <div key={c.k} className="card px-4 py-3">
                <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{c.k}</div>
                <div className="mt-1 text-[15px] font-semibold">{c.v}</div>
                {c.s && <div className="text-[11px]" style={{ color: "var(--fg-dim)" }}>{c.s}</div>}
              </div>
            ))}
          </div>

          {/* 执行记录 */}
          <section className="card mb-4 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <h2 className="m-0 text-xs font-semibold">执行记录（test runs）</h2>
              <span className="ml-auto mono text-[11px]" style={{ color: "var(--fg-mute)" }}>
                {report.case_source}
                {report.case_source_sha256 ? ` · ${report.case_source_sha256.slice(0, 12)}…` : ""}
              </span>
            </div>
            {report.runs.length === 0 ? (
              <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>这个任务还没有 test run。</div>
            ) : (
              <div className="overflow-auto">
                <table>
                  <thead>
                    <tr>
                      <th>run</th><th>状态</th><th>退出码</th><th>用例集 ref</th><th>代码快照</th>
                      <th>通过 / 失败</th><th>报告</th><th>下载</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.runs.map((r) => {
                      const s = r.summary;
                      return (
                        <tr key={r.run_id}>
                          <td className="mono">#{r.run_id}</td>
                          <td>{statusChip(r.status === "ok" ? "PASS" : r.status === "failed" ? "FAIL" : r.status)}</td>
                          <td className="mono">{r.exit_code ?? "—"}</td>
                          <td className="mono text-[11px]">{r.suite_ref ?? "—"}</td>
                          <td className="mono text-[11px]">{(r.code_snapshot_sha256 ?? "").slice(0, 10) || "—"}</td>
                          <td className="mono">
                            {s ? `${s.PASS ?? 0} / ${s.FAIL ?? 0}` : r.report_available ? "—" : <span style={{ color: "var(--fg-mute)" }}>无报告</span>}
                          </td>
                          <td>{r.report_available ? <span className="chip chip-ok">有</span> : <span className="chip" style={{ color: "var(--fg-mute)" }}>缺</span>}</td>
                          <td>
                            <div className="flex gap-1">
                              {r.downloads.map((d) => (
                                <a key={d.id} className="chip" href={api.artifactUrl(d.id)} target="_blank" rel="noreferrer" title={d.name}>
                                  {d.name.endsWith(".json") ? "json" : "md"}
                                </a>
                              ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* 失败点 */}
          {report.failures.length > 0 && (
            <section className="card mb-4 overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                <h2 className="m-0 text-xs font-semibold">失败点</h2>
                <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>按 run 记录，便于定位</span>
              </div>
              <div className="overflow-auto">
                <table>
                  <thead>
                    <tr><th>用例</th><th>run</th><th>状态</th><th>期望</th><th>匹配率</th><th>最大误差</th><th>说明</th></tr>
                  </thead>
                  <tbody>
                    {report.failures.map((f, i) => (
                      <tr key={`${f.run_id}-${f.case_id}-${i}`}>
                        <td className="mono">{f.case_id}</td>
                        <td className="mono">#{f.run_id}</td>
                        <td>{statusChip(f.status)}</td>
                        <td>{f.expected ? (KIND_LABEL[f.expected] ?? f.expected) : "—"}</td>
                        <td className="mono">{f.matched_ratio != null ? f.matched_ratio.toFixed(4) : "—"}</td>
                        <td className="mono">{f.max_abs_err != null ? f.max_abs_err.toExponential(2) : "—"}</td>
                        <td className="text-[11px]" style={{ color: "var(--fg-dim)", maxWidth: 360 }}>{f.detail ?? f.note ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* 用例矩阵 */}
          <section className="card mb-4 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <h2 className="m-0 text-xs font-semibold">用例矩阵（{report.total_cases}）</h2>
              <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
                {report.runs.length > 1 && "「历史」按 run 从旧到新"}
              </span>
            </div>
            <div className="overflow-auto">
              <table>
                <thead>
                  <tr>
                    <th>用例</th><th>组</th><th>必需</th><th>shape</th><th>dtype</th><th>期望</th>
                    <th>最近</th><th>匹配率</th><th>最大误差</th><th>历史</th><th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {report.cases.map((c) => (
                    <tr key={c.case_id}>
                      <td className="mono">{c.case_id}</td>
                      <td>{c.group ?? "—"}</td>
                      <td>{c.required ? "是" : "否"}</td>
                      <td className="mono text-[11px]">
                        {c.shape ? c.shape.join("×") : "—"}
                        {c.shape_b && c.shape && String(c.shape) !== String(c.shape_b) ? ` / ${c.shape_b.join("×")}` : ""}
                      </td>
                      <td className="mono">{c.dtype ?? "—"}</td>
                      <td>{c.expected ? (KIND_LABEL[c.expected] ?? c.expected) : "—"}</td>
                      <td>
                        {statusChip(c.latest_status)}
                        {c.latest_run_id != null && <span className="mono ml-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>#{c.latest_run_id}</span>}
                      </td>
                      <td className="mono">{c.matched_ratio != null ? c.matched_ratio.toFixed(4) : "—"}</td>
                      <td className="mono">{c.max_abs_err != null ? c.max_abs_err.toExponential(2) : "—"}</td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(c.results).map(([rid, st]) => (
                            <span key={rid} className="chip" title={`run #${rid}`} style={{
                              color: st === "PASS" ? "var(--ok)" : st === "FAIL" ? "var(--bad)" : "var(--fg-mute)",
                            }}>
                              #{rid} {st}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="text-[11px]" style={{ color: "var(--fg-dim)", maxWidth: 300 }}>{c.note ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="px-4 py-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
              {report.notes.map((n, i) => <div key={i}>· {n}</div>)}
            </div>
          </section>
        </>
      )}

      {report && tab === "perf" && (
        <>
          <div className="mb-4 grid grid-cols-4 gap-3">
            {[
              { k: "性能用例", v: `${perf?.cases.length ?? 0}`, s: perf?.primary_case ? `主用例 ${perf.primary_case}` : "无主用例" },
              { k: "benchmark run", v: `${perf?.runs.length ?? 0}`, s: perf?.source ?? "benchmark.json 未生成" },
              {
                k: "最近门禁",
                v: latestPerfRun?.gate_verdict ?? "—",
                s: latestPerfRun ? `run #${latestPerfRun.run_id} · ${latestPerfRun.status}` : "还没有 benchmark run",
              },
              {
                k: "用例来源",
                v: perf?.source ? "benchmark.json" : "—",
                s: perf?.source ?? "先在对话里确认性能用例组",
              },
            ].map((c) => (
              <div key={c.k} className="card px-4 py-3">
                <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{c.k}</div>
                <div className="mt-1 text-[15px] font-semibold">{c.v}</div>
                {c.s && <div className="text-[11px]" style={{ color: "var(--fg-dim)" }}>{c.s}</div>}
              </div>
            ))}
          </div>

          <section className="card mb-4 overflow-hidden">
            <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <h2 className="m-0 text-xs font-semibold">性能用例（{perf?.cases.length ?? 0}）</h2>
              <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
                只读；用例编辑走任务对话（整组确认后落 benchmark.json）
              </span>
            </div>
            {!perf || perf.cases.length === 0 ? (
              <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                还没有性能用例（工作区 cases/&lt;op&gt;/benchmark.json 未生成）。
              </div>
            ) : (
              <div className="overflow-auto">
                <table>
                  <thead>
                    <tr>
                      <th>用例</th><th>n</th><th>dtype</th><th>主用例</th><th>必需</th>
                      <th>最新 run</th><th>候选</th><th>基线</th><th>加速比</th><th>结论</th><th>门禁</th><th>说明</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perf.cases.map((c) => (
                      <tr key={c.case_id}>
                        <td className="mono">{c.case_id}</td>
                        <td className="mono">{c.n ?? "—"}</td>
                        <td className="mono">{c.dtype ?? "—"}</td>
                        <td>{c.primary ? <span className="chip chip-run !py-0 !text-[10px]">主</span> : "—"}</td>
                        <td>{c.required ? "是" : "否"}</td>
                        <td className="mono">{c.latest ? `#${c.latest.run_id}` : "—"}</td>
                        <td className="mono">{fmtUs(c.latest?.candidate_median_us)}</td>
                        <td className="mono">{fmtUs(c.latest?.baseline_median_us)}</td>
                        <td className="mono" style={{ color: c.latest?.speedup != null && c.latest.speedup >= 1 ? "var(--ok)" : undefined }}>
                          {fmtSpeedup(c.latest?.speedup)}
                        </td>
                        <td>
                          {c.latest?.conclusion ? (
                            <span className={`chip !py-0 !text-[10px] ${c.latest.conclusion === "improved" ? "chip-ok" : c.latest.conclusion === "regressed" ? "chip-bad" : ""}`}>
                              {c.latest.conclusion}
                            </span>
                          ) : "—"}
                        </td>
                        <td>
                          {c.latest?.gate_verdict ? (
                            <span className={`chip !py-0 !text-[10px] ${c.latest.gate_verdict === "pass" ? "chip-ok" : "chip-bad"}`}>
                              {c.latest.gate_verdict}
                            </span>
                          ) : "—"}
                        </td>
                        <td className="text-[11px]" style={{ color: "var(--fg-dim)", maxWidth: 280 }}>{c.note ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="card mb-4 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <h2 className="m-0 text-xs font-semibold">benchmark 执行记录</h2>
              <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>性能组历史（只读）</span>
            </div>
            {!perf || perf.runs.length === 0 ? (
              <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>还没有 benchmark run。</div>
            ) : (
              <div className="overflow-auto">
                <table>
                  <thead>
                    <tr><th>run</th><th>状态</th><th>门禁</th><th>用例集 ref</th><th>开始</th></tr>
                  </thead>
                  <tbody>
                    {[...perf.runs].sort((a, b) => b.run_id - a.run_id).map((r) => (
                      <tr key={r.run_id}>
                        <td className="mono">#{r.run_id}</td>
                        <td>{statusChip(r.status)}</td>
                        <td>
                          {r.gate_verdict ? (
                            <span className={`chip ${r.gate_verdict === "pass" ? "chip-ok" : "chip-bad"}`}>{r.gate_verdict}</span>
                          ) : "—"}
                        </td>
                        <td className="mono text-[11px]">{r.suite_ref ?? "—"}</td>
                        <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{r.started_at ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {!report && !err && <div className="card px-4 py-3 text-xs" style={{ color: "var(--fg-mute)" }}>加载中…</div>}
    </>
  );
}
