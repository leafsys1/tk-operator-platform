import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { StageTaskView, TaskGates } from "../types";

const ACTIONS = ["build", "smoke", "test", "benchmark", "profile"] as const;
const LABEL: Record<string, string> = { build: "编译", smoke: "冒烟", test: "正确性", benchmark: "基准", profile: "采集" };

/** 阶段与门禁：**以推导阶段 + 门禁 verdict 为准**（不再看产物齐不齐）。
 *
 * 数据来自 `GET /api/tasks/{id}/stage`（推导阶段）与 `/gates`（与阶段看板同口径的当前执行证据
 * `gate.evaluate` 结论）。与「阶段看板」同口径；存值不一致时明确标出。
 */
export default function GatePanel({ taskId }: { taskId: number }) {
  const [stage, setStage] = useState<StageTaskView | null>(null);
  const [gates, setGates] = useState<TaskGates | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api.taskStage(taskId), api.taskGates(taskId)])
      .then(([s, g]) => {
        if (!alive) return;
        setStage(s);
        setGates(g);
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [taskId]);

  const stages = stage?.stages ?? [];
  const perf = gates?.perf ?? stage?.perf ?? null;

  return (
    <section className="card p-4">
      <div className="flex items-center gap-2">
        <h2 className="m-0 text-xs font-semibold">阶段与门禁</h2>
        {stage && (
          <span className="chip" style={{ background: "var(--panel-2)" }}>
            推导：{stage.derived_stage}
          </span>
        )}
        {stage && !stage.in_sync && (
          <span className="chip chip-warn">存值 {stage.stored_stage}（未同步）</span>
        )}
      </div>

      {err && <div className="mt-2 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}

      <div className="mt-3">
        {stages.map((g, i) => (
          <div key={g.stage} className="mb-2 flex gap-3">
            <span
              className="mt-1 inline-block h-[11px] w-[11px] shrink-0 rounded-full"
              style={{
                background: g.ok ? "var(--ok)" : g.stage === stage?.derived_stage ? "var(--run)" : "var(--panel-2)",
                border: `2px solid ${g.ok ? "var(--ok)" : g.stage === stage?.derived_stage ? "var(--run)" : "var(--border-strong)"}`,
              }}
            />
            <div className="min-w-0">
              <div className="text-xs font-semibold">{i + 1} · {g.stage}</div>
              <div className="mt-0.5 text-[11px]" style={{ color: g.ok ? "var(--ok)" : "var(--fg-mute)" }}>
                {g.name} {g.ok ? "✓" : "—"}
                {g.evidence.length > 0 && <span> · {g.evidence.join("；")}</span>}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3" style={{ borderTop: "1px solid var(--border)", paddingTop: "8px" }}>
        <div className="text-[11px] font-semibold">动作证据与门禁</div>
        <div className="mt-1 flex flex-wrap gap-2">
          {ACTIONS.map((a) => {
            const g = gates?.gates.find((x) => x.action === a);
            const ok = g?.verdict === "pass";
            return (
              <div
                key={a}
                className="rounded-md px-2 py-1 text-[11px]"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
              >
                <span className="font-semibold">{LABEL[a]}</span>{" "}
                {!g || g.run_id === null ? (
                  <span style={{ color: "var(--fg-mute)" }}>未跑</span>
                ) : (
                  <>
                    <span className="mono">#{g.run_id}</span>{" "}
                    <span className={`chip ${ok ? "chip-ok" : "chip-bad"}`}>{g.verdict ?? "—"}</span>
                    {(g.fail_codes?.length ?? 0) > 0 && (
                      <div className="mt-0.5 text-[10px]" style={{ color: "var(--bad)" }}>
                        {g.fail_codes.join(", ")}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
        {perf && (
          <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            最近优化轮次 #{perf.loop_id}：<span className={`chip ${perf.verdict === "pass" ? "chip-ok" : "chip-bad"}`}>{perf.verdict}</span>{" "}
            （{perf.conclusion}）
          </div>
        )}
        {stage?.next_step && (
          <div className="mt-2 text-[11px]" style={{ color: "var(--fg-dim)" }}>
            下一步：{stage.next_step}
          </div>
        )}
        <div className="mt-2 text-[10px]" style={{ color: "var(--fg-mute)" }}>
          门禁为硬拦（方案 v1.2 §5）：结果有效 + 业务达标 + 版本一致 + 产物完整；退出码不是结论。
        </div>
      </div>
    </section>
  );
}
