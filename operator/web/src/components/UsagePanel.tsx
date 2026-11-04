import { useEffect, useState } from "react";
import { api, money, num } from "../lib/api";
import type { UsageBucket, UsageReport } from "../types";

/**
 * 消耗情况（P4）：`GET /api/usage` 的渲染。
 *
 * 口径：只统计真实调用（`agent_jobs`），**排除** P0 示例行（`is_demo`，条数见 `excluded_demo_rows`）。费用按调用所属端点
 * 的价目实时算（`endpoints.price_*`，人民币元 / 每百万 token）；端点没配 in/out 单价 → 如实标「未定价」。
 * 早期调用没有端点快照 → 回退当前端点估算（`cost_estimated`），页面用「估」标出来。
 *
 * 金额一律走 `money()`（小额不显示成 ¥0.00）；`cost` 在后端修复前的异常/遗留数据里可能为 null，
 * 这里所有金额渲染都做了空值兜底（不能用 `.toFixed` 直接调，否则整页白屏）。
 */
function Cost({ b }: { b: UsageBucket }) {
  const cost = b.cost ?? null;
  const unpriced = !b.cost_known;
  if (unpriced && cost == null) return <span style={{ color: "var(--fg-mute)" }}>未定价</span>;
  const title = [
    b.cost_estimated ? `${b.cost_estimated} 次早期调用按当前端点价目估算` : "",
    unpriced ? `另有 ${b.unpriced_jobs ?? 0} 次调用没有可用价目（金额未含）` : "",
  ].filter(Boolean).join("；");
  return (
    <span className="mono" title={title || undefined}>
      {money(cost)}
      {unpriced && <span style={{ color: "var(--fg-mute)" }}> +</span>}
      {!!b.cost_estimated && (
        <span className="ml-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>估</span>
      )}
    </span>
  );
}

function tokenCell(b: UsageBucket) {
  return (
    <span className="mono">
      {num(b.tokens_in)} / {num(b.tokens_out)}
    </span>
  );
}

/** 一天的总 token：in + out + cache 读/写（cache 读往往远大于 in+out，不能漏）。 */
const dayTotal = (r: UsageBucket) => r.tokens_in + r.tokens_out + r.cache_read + r.cache_write;

function Bars({ rows }: { rows: UsageBucket[] }) {
  const max = Math.max(1, ...rows.map(dayTotal));
  if (rows.length === 0) return <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>暂无数据</div>;
  return (
    <div className="flex flex-col gap-1 px-4 py-3">
      {rows.map((r) => {
        const total = dayTotal(r);
        return (
          <div key={r.day} className="flex items-center gap-2 text-[11px]">
            <span className="mono w-[86px] shrink-0" style={{ color: "var(--fg-dim)" }}>{r.day}</span>
            <div className="relative h-[10px] min-w-0 flex-1 rounded" style={{ background: "var(--panel-2)" }}>
              <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(total / max) * 100}%`, background: "var(--accent)" }} />
            </div>
            <span className="mono w-[210px] shrink-0 text-right" style={{ color: "var(--fg-mute)" }}>
              in {num(r.tokens_in)} · out {num(r.tokens_out)} · cache {num(r.cache_read + r.cache_write)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function UsagePanel() {
  const [report, setReport] = useState<UsageReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.usage().then(setReport).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="card px-4 py-3 text-xs" style={{ color: "var(--bad)" }}>{err}</div>;
  if (!report) return <div className="card px-4 py-3 text-xs" style={{ color: "var(--fg-mute)" }}>加载中…</div>;

  const t = report.totals;
  const totalCost = t.cost ?? null;
  const priced = report.pricing?.priced ?? false;
  // cost 可能为 null（历史/异常数据）：money() 会显示「未定价」，绝不直接 .toFixed。
  const costValue = t.cost_known
    ? money(totalCost)
    : totalCost != null && totalCost > 0
      ? `${money(totalCost)}+`
      : "未定价";
  const costSub = !priced
    ? "到「API 管理」给端点配输入/输出单价"
    : t.cost_estimated
      ? `含 ${t.cost_estimated} 次按当前价目估算`
      : report.pricing?.endpoint ? `按「${report.pricing.endpoint}」价目` : "";
  const cards = [
    { k: "智能体调用", v: String(t.jobs), s: `成功 ${t.ok} · 失败 ${t.failed}` },
    { k: "输入 token", v: num(t.tokens_in), s: `输出 ${num(t.tokens_out)}` },
    { k: "缓存读取 token", v: num(t.cache_read), s: `写入 ${num(t.cache_write)}` },
    { k: "费用合计", v: costValue, s: costSub },
  ];

  return (
    <>
      <div className="mb-4 grid grid-cols-4 gap-3">
        {cards.map((c) => (
          <div key={c.k} className="card px-4 py-3">
            <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{c.k}</div>
            <div className="mt-1 text-[15px] font-semibold">{c.v}</div>
            {c.s && <div className="text-[11px]" style={{ color: "var(--fg-dim)" }}>{c.s}</div>}
          </div>
        ))}
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
        <section className="card overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">按任务</h2>
          </div>
          <div className="overflow-auto">
            <table>
              <thead>
                <tr><th>任务</th><th>调用</th><th>token in / out</th><th>cache 读</th><th>cache 写</th><th>费用</th></tr>
              </thead>
              <tbody>
                {report.by_task.map((r) => (
                  <tr key={r.task_id}>
                    <td>#{r.task_id} {r.title}</td>
                    <td>{r.jobs}</td>
                    <td>{tokenCell(r)}</td>
                    <td className="mono">{num(r.cache_read)}</td>
                    <td className="mono">{num(r.cache_write)}</td>
                    <td><Cost b={r} /></td>
                  </tr>
                ))}
                {report.by_task.length === 0 && (
                  <tr><td colSpan={6} style={{ color: "var(--fg-mute)" }}>暂无数据</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">按调用类型</h2>
          </div>
          <div className="overflow-auto">
            <table>
              <thead>
                <tr><th>类型</th><th>调用</th><th>token in / out</th><th>cache 读</th><th>cache 写</th><th>费用</th></tr>
              </thead>
              <tbody>
                {report.by_kind.map((r) => (
                  <tr key={r.kind}>
                    <td>{r.kind}</td>
                    <td>{r.jobs}</td>
                    <td>{tokenCell(r)}</td>
                    <td className="mono">{num(r.cache_read)}</td>
                    <td className="mono">{num(r.cache_write)}</td>
                    <td><Cost b={r} /></td>
                  </tr>
                ))}
                {report.by_kind.length === 0 && (
                  <tr><td colSpan={6} style={{ color: "var(--fg-mute)" }}>暂无数据</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card overflow-hidden" style={{ gridColumn: "1 / -1" }}>
          <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <h2 className="m-0 text-xs font-semibold">按天（UTC）</h2>
            <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
              条形 = 当天 token 总量（含 cache 读/写）
            </span>
          </div>
          <Bars rows={report.by_day} />
        </section>
      </div>

      <div className="mt-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
        {report.notes.map((n, i) => <div key={i}>· {n}</div>)}
        {report.excluded_demo_rows > 0 && <div>· 已排除 {report.excluded_demo_rows} 条示例行（P0 假 usage）。</div>}
      </div>
    </>
  );
}
