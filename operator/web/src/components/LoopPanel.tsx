import type { TaskDetail } from "../types";
import { num } from "../lib/api";

const KIND: Record<string, string> = { design: "设计", fix: "修复", perf: "优化" };

/**
 * 轮次与关联（方案 §7「执行记录」分区的一部分）：Loop / AgentJob / Run 的归属 + 每轮消耗。
 *
 * 之前只在「自动优化对比」里看得到 perf Loop；这里把**所有**轮次（design/fix/perf）列出来，
 * 让「这些 run / 调用算在哪一轮」一目了然。
 */
export default function LoopPanel({ task }: { task: TaskDetail }) {
  const loops = task.loops;
  const jobsFor = (lid: number) => task.agent_jobs.filter((j) => j.loop_id === lid);
  const runsFor = (lid: number) => task.runs.filter((r) => r.loop_id === lid);
  const orphanRuns = task.runs.filter((r) => r.loop_id === null);
  const orphanJobs = task.agent_jobs.filter((j) => j.loop_id === null);

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">轮次与关联（Loop / AgentJob / Run）</h2>
        <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
          这些执行与调用分别算在哪一轮；未挂轮次的单独列出
        </span>
      </div>

      {loops.length === 0 ? (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          还没有轮次（Loop）—— 优化 Loop 会在「自动优化」区开。
        </div>
      ) : (
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>轮次</th>
                <th>类型</th>
                <th>状态</th>
                <th>假设 / 范围</th>
                <th>结论</th>
                <th>关联 run</th>
                <th>关联调用</th>
                <th>token in</th>
              </tr>
            </thead>
            <tbody>
              {loops.map((l) => {
                const runs = runsFor(l.id);
                const jobs = jobsFor(l.id);
                const tokens = jobs.reduce((s, j) => s + (j.tokens_in ?? 0), 0);
                return (
                  <tr key={l.id}>
                    <td className="mono">#{l.id}</td>
                    <td>{KIND[l.kind] ?? l.kind}</td>
                    <td>
                      <span className={`chip ${l.status === "closed" ? "chip-ok" : "chip-run"}`}>{l.status}</span>
                    </td>
                    <td className="text-[11px]" style={{ color: "var(--fg-dim)", maxWidth: 320 }}>
                      {l.hypothesis || l.scope || "—"}
                    </td>
                    <td className="text-[11px]" style={{ color: "var(--fg-dim)", maxWidth: 220 }}>
                      {l.conclusion ?? "—"}
                    </td>
                    <td className="mono">
                      {runs.length ? runs.map((r) => `#${r.id}(${r.action})`).join(" ") : "—"}
                    </td>
                    <td className="mono">
                      {jobs.length ? jobs.map((j) => `#${j.id}`).join(" ") : "—"}
                    </td>
                    <td className="mono">{num(tokens)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {(orphanRuns.length > 0 || orphanJobs.length > 0) && (
        <div className="px-4 py-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          未挂轮次：
          {orphanRuns.length > 0 && (
            <span className="mono ml-1">run {orphanRuns.map((r) => `#${r.id}(${r.action})`).join(" ")}</span>
          )}
          {orphanJobs.length > 0 && (
            <span className="mono ml-2">调用 {orphanJobs.map((j) => `#${j.id}`).join(" ")}</span>
          )}
        </div>
      )}
    </section>
  );
}
