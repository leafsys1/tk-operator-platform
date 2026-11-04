import { num, when } from "../lib/api";
import type { TaskDetail } from "../types";
import LoopPanel from "./LoopPanel";

/** 同一 run 的产物按类型归并（一批 report 不再刷 12 个同色胶囊）。 */
function artifactCounts(items: { type: string }[]): { type: string; count: number }[] {
  const m = new Map<string, number>();
  for (const a of items) m.set(a.type, (m.get(a.type) ?? 0) + 1);
  return [...m.entries()].map(([type, count]) => ({ type, count }));
}

/**
 * 任务详情「记录」：轮次归属 + 智能体调用台账 + 真实执行台账。
 *
 * 只做只读清点：这些调用 / 执行分别算在哪一轮，产物指针落在哪。
 * 结果内容（精度、性能、diff）不放这里，避免和「测试与结果」「性能与优化」重复。
 */
export default function TaskRecordsPanel({ task }: { task: TaskDetail }) {
  return (
    <>
      <LoopPanel task={task} />

      <section className="card mb-4 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="m-0 text-xs font-semibold">智能体调用（agent_jobs）</h2>
          <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
            跑了什么、花了多少 —— 调用成功 ≠ 算子做对了
          </span>
        </div>
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>job</th>
                <th>类型</th>
                <th>状态</th>
                <th>dsh 会话</th>
                <th>起止</th>
                <th>token in / cache</th>
                <th>费用</th>
              </tr>
            </thead>
            <tbody>
              {task.agent_jobs.map((j) => (
                <tr key={j.id}>
                  <td>#{j.id}</td>
                  <td>{j.kind}</td>
                  <td>
                    <span className={`chip ${j.status === "ok" ? "chip-ok" : j.status === "failed" ? "chip-bad" : "chip-run"}`}>
                      {j.status}
                    </span>
                  </td>
                  <td className="mono">{j.dsh_session_id ?? "—"}</td>
                  <td className="text-[11px]" style={{ color: "var(--fg-dim)" }}>
                    {when(j.started_at)}
                    <br />
                    {when(j.ended_at)}
                  </td>
                  <td className="mono">
                    {num(j.tokens_in)} / {num(j.cache_read)}
                  </td>
                  <td className="mono">
                    {j.cost != null
                      ? `¥${j.cost.toFixed(2)}${j.cost_estimated ? "（估）" : ""}${j.is_demo ? "（示例）" : ""}`
                      : j.is_demo ? "示例" : "—"}
                  </td>
                </tr>
              ))}
              {task.agent_jobs.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    还没有智能体调用。去「实现与调用」发起一次 agent run。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="m-0 text-xs font-semibold">真实执行（runs）</h2>
          <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
            门禁依据：build / smoke / test / benchmark / profile
          </span>
        </div>
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>动作</th>
                <th>状态</th>
                <th>退出码</th>
                <th>代码快照</th>
                <th>判据</th>
                <th>产物</th>
              </tr>
            </thead>
            <tbody>
              {task.runs.map((r) => (
                <tr key={r.id}>
                  <td>#{r.id}</td>
                  <td className="mono">{r.action}</td>
                  <td>
                    <span className={`chip ${r.status === "ok" ? "chip-ok" : r.status === "failed" ? "chip-bad" : "chip-run"}`}>
                      {r.status}
                    </span>
                  </td>
                  <td className="mono">{r.exit_code ?? "—"}</td>
                  <td className="mono">{(r.code_snapshot_sha256 ?? "").slice(0, 10) || "—"}</td>
                  <td className="text-[11px]" style={{ color: "var(--fg-dim)" }}>
                    {r.criteria_ref ?? "—"}
                  </td>
                  <td>
                    <div className="flex flex-wrap gap-1">
                      {artifactCounts(r.artifacts).map((a) => (
                        <span key={a.type} className="chip">
                          {a.type}
                          {a.count > 1 ? ` ×${a.count}` : ""}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
              {task.runs.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    还没有真实执行 —— 在「概览」点击对应动作生成计划，再点「真执行」。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          {task.runs.length === 0
            ? "门禁不会因为「会话跑通了」而变绿：结论只来自真实执行 + 机读产物。"
            : "产物明细（diff / 用例 / 精度图 / 性能图）在「测试与结果」「性能与优化」里渲染；这里只列清单。"}
        </div>
      </section>
    </>
  );
}
