import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { CaseReport } from "../types";

/** 任务详情「测试与结果」：精度用例（矩阵 + 逐例最新状态 + 失败点 + 执行记录）
 * 与性能用例（benchmark.json 用例 + 最近实测/基线/加速比/门禁）的只读摘要。
 *
 * 数据与「测试管理」页同源（`GET /api/tasks/{id}/cases`，性能段见契约 §3.6）。
 */
export default function TaskCasePanel({ taskId, active = true, revision, onOpenTests }: {
  taskId: number; active?: boolean; revision?: string; onOpenTests?: () => void;
}) {
  const [doc, setDoc] = useState<CaseReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!active) return;
    let alive = true;
    setErr(null);
    api.cases(taskId)
      .then((d) => alive && setDoc(d))
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [taskId, active, revision]);

  const latestRun = useMemo(() => {
    const runs = [...(doc?.runs ?? [])].filter((r) => r.status === "ok").sort((a, b) => b.run_id - a.run_id);
    return runs[0] ?? null;
  }, [doc?.runs]);

  const failing = (doc?.cases ?? []).filter((c) => c.latest_status && c.latest_status !== "PASS");
  // 性能组（契约 §3.6）：与精度组同源 /cases，只读摘要。
  const perf = doc?.performance ?? null;

  const fmtUs = (v: number | null | undefined) =>
    v == null || !Number.isFinite(v) ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${v.toFixed(2)} µs`;
  const fmtSpeedup = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(2)}×`);

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">测试与结果（精度 / 性能）</h2>
        {doc && (
          <>
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              {doc.total_cases} 用例 · 必需 {doc.required_cases} · {doc.case_groups.join(" / ")}
            </span>
            {latestRun?.summary && (
              <span className="chip chip-ok">
                #{latestRun.run_id} PASS {latestRun.summary.PASS ?? 0}/{doc.total_cases}
              </span>
            )}
            {failing.length > 0 && <span className="chip chip-bad">{failing.length} 个未通过</span>}
          </>
        )}
        {onOpenTests && (
          <button className="btn btn-sm ml-auto" onClick={onOpenTests}>打开测试管理</button>
        )}
      </div>

      {err && <div className="px-4 py-3 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}
      {doc && doc.cases.length === 0 && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>还没有用例矩阵（工作区 cases/ 为空）。</div>
      )}

      {doc && doc.cases.length > 0 && (
        <div className="grid gap-0 md:grid-cols-2">
          <div className="max-h-[320px] overflow-auto border-r" style={{ borderColor: "var(--border)" }}>
            <table>
              <thead>
                <tr><th>用例</th><th>组</th><th>最新</th><th>误差</th></tr>
              </thead>
              <tbody>
                {doc.cases.map((c) => (
                  <tr key={c.case_id}>
                    <td className="mono">{c.case_id}</td>
                    <td>{c.group ?? "—"}</td>
                    <td>
                      <span className={`chip ${c.latest_status === "PASS" ? "chip-ok" : c.latest_status ? "chip-bad" : ""}`}>
                        {c.latest_status ?? "—"}
                      </span>
                    </td>
                    <td className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      {c.max_abs_err != null ? `max ${c.max_abs_err.toExponential(2)}` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-3">
            <div className="text-[11px] font-semibold">失败点</div>
            {doc.failures.length === 0 ? (
              <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>没有记录到失败</div>
            ) : (
              <div className="mt-1 max-h-[120px] overflow-auto">
                <table>
                  <thead><tr><th>用例</th><th>run</th><th>状态</th><th>细节</th></tr></thead>
                  <tbody>
                    {doc.failures.map((f) => (
                      <tr key={`${f.case_id}-${f.run_id}`}>
                        <td className="mono">{f.case_id}</td>
                        <td>#{f.run_id}</td>
                        <td><span className="chip chip-bad">{f.status}</span></td>
                        <td className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          {f.detail ?? (f.max_abs_err != null ? `max_abs_err ${f.max_abs_err}` : "—")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="mt-3 text-[11px] font-semibold">最近执行</div>
            <div className="mt-1 max-h-[120px] overflow-auto">
              <table>
                <thead><tr><th>run</th><th>状态</th><th>用例集</th><th>报告</th></tr></thead>
                <tbody>
                  {[...doc.runs].sort((a, b) => b.run_id - a.run_id).slice(0, 5).map((r) => (
                    <tr key={r.run_id}>
                      <td>#{r.run_id}</td>
                      <td><span className={`chip ${r.status === "ok" ? "chip-ok" : "chip-bad"}`}>{r.status}</span></td>
                      <td className="mono text-[10px]">{(r.suite_ref ?? "—").slice(0, 22)}</td>
                      <td>
                        {r.downloads.length > 0 ? (
                          <a className="text-[11px]" href={api.artifactUrl(r.downloads[0].id)}>下载</a>
                        ) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {doc.case_source && (
              <div className="mt-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                用例集来源 {doc.case_source} · {(doc.case_source_sha256 ?? "").slice(0, 12)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 性能用例（契约 §3.6：GET /cases 的 performance 段；只读，编辑走对话） */}
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="flex flex-wrap items-center gap-2 px-4 py-2.5">
          <span className="text-[11px] font-semibold">性能用例</span>
          {perf ? (
            <>
              <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                {perf.cases.length} 条 · {perf.runs.length} 次 benchmark
              </span>
              {perf.primary_case && <span className="chip !py-0 !text-[10px]">主用例 {perf.primary_case}</span>}
              {perf.source && <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>{perf.source}</span>}
            </>
          ) : (
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>无性能用例段</span>
          )}
        </div>
        {!perf || perf.cases.length === 0 ? (
          <div className="px-4 pb-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            还没有性能用例（工作区 cases/&lt;op&gt;/benchmark.json 未生成）。
          </div>
        ) : (
          <div className="max-h-[280px] overflow-auto">
            <table>
              <thead>
                <tr>
                  <th>用例</th><th>n</th><th>dtype</th><th>主用例</th><th>必需</th>
                  <th>最新 run</th><th>候选</th><th>基线</th><th>加速比</th><th>结论 / 门禁</th>
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
                      <div className="flex flex-wrap items-center gap-1">
                        {c.latest?.conclusion ? (
                          <span
                            className={`chip !py-0 !text-[10px] ${c.latest.conclusion === "improved" ? "chip-ok" : c.latest.conclusion === "regressed" ? "chip-bad" : ""}`}
                          >
                            {c.latest.conclusion}
                          </span>
                        ) : (
                          <span style={{ color: "var(--fg-mute)" }}>—</span>
                        )}
                        {c.latest?.gate_verdict && (
                          <span className={`chip !py-0 !text-[10px] ${c.latest.gate_verdict === "pass" ? "chip-ok" : "chip-bad"}`}>
                            {c.latest.gate_verdict}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
