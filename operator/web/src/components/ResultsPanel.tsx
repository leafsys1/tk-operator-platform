import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { BarChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import { api } from "../lib/api";
import { arithmeticIntensity, baselineLabel } from "../lib/opProfile";
import type {
  BenchmarkCase,
  ChipProfile,
  CorrectnessCase,
  MetricRow,
  TaskDetail,
} from "../types";

echarts.use([BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

type ECharts = ReturnType<typeof echarts.init>;

/** 把对象当 JSON 下载（P2 导出）。 */
export function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name.endsWith(".json") ? name : `${name}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** ECharts 薄封装：option 变化即重绘（notMerge）。 */
function Chart({ option, height = 240 }: { option: EChartsOption; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<ECharts | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    inst.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const onResize = () => inst.current?.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      inst.current?.dispose();
      inst.current = null;
    };
  }, []);
  useEffect(() => {
    inst.current?.setOption(option, true);
  }, [option]);
  return <div ref={ref} style={{ height, width: "100%" }} />;
}

const STATUS_COLOR: Record<string, string> = {
  PASS: "#22c55e", FAIL: "#ef4444", ERROR: "#f59e0b", INVALID: "#a855f7", NOT_RUN: "#94a3b8",
};

const ACTIONS: { key: "test" | "benchmark" | "profile"; report: "correctness" | "performance" | "metrics"; label: string }[] = [
  { key: "test", report: "correctness", label: "正确性（test）" },
  { key: "benchmark", report: "performance", label: "性能基准（benchmark）" },
  { key: "profile", report: "metrics", label: "性能采集（profile）" },
];

type SpecInfo = { version: number; sha256: string | null };

function Alignment({
  report,
  specById,
  currentSpecId,
  runSpecId,
}: {
  report: Record<string, unknown>;
  specById: Map<number, SpecInfo>;
  currentSpecId: number | null;
  runSpecId: number | null;
}) {
  const reportSpecId =
    typeof report.spec_version_id === "number"
      ? report.spec_version_id
      : typeof report.spec_version_id === "string"
        ? Number(report.spec_version_id)
        : null;
  const specId = runSpecId ?? (Number.isFinite(reportSpecId) ? reportSpecId : null);
  const spec = specId != null ? specById.get(specId) : undefined;
  const cur = currentSpecId != null ? specById.get(currentSpecId) : undefined;
  const mismatch = Boolean(spec && cur && specId !== currentSpecId);
  const sameContent = Boolean(mismatch && spec?.sha256 && spec.sha256 === cur?.sha256);
  const items: [string, unknown][] = [
    ["规格", spec ? `v${spec.version}` : specId != null ? `#${specId}` : "—"],
    ["判据", report.criteria_ref ?? "—"],
    ["用例集", report.suite_ref ?? "—"],
    ["代码快照", typeof report.code_snapshot_sha256 === "string" ? `${(report.code_snapshot_sha256 as string).slice(0, 12)}…` : "—"],
  ];
  return (
    <>
      <div className="flex flex-wrap gap-3 px-4 pb-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
        {items.map(([k, v]) => (
          <span key={k}>
            {k} <span className="mono" style={{ color: "var(--fg-dim)" }}>{String(v)}</span>
          </span>
        ))}
      </div>
      {mismatch && spec && cur && (
        <div className="px-4 pb-2 text-[11px]" style={{ color: sameContent ? "var(--warn)" : "var(--bad)" }}>
          本结果基于规格 v{spec.version}；当前规格是 v{cur.version}
          {sameContent ? "（哈希相同，内容未变）" : "（内容已变，旧结果不能直接当作当前规格的验收证据）"}
        </div>
      )}
    </>
  );
}

/**
 * P2 结果视图：把 test/benchmark/profile 的**机读报告**读出来渲染（表 + ECharts 图）。
 *
 * 回答三件事：正确性逐用例过没过、性能与基线·极限怎么比、指标四类的状态与来源。
 * 数据来自 `GET /api/runs/{id}/report?type=...`（内容 = 容器里产出的那些 .json，原样）。
 */
export default function ResultsPanel({ task }: { task: TaskDetail }) {
  // 每个动作取「最近一次成功」的 run
  const latest = useMemo(() => {
    const pick: Record<string, number> = {};
    for (const r of task.runs) {
      if (r.status === "ok" && ["test", "benchmark", "profile"].includes(r.action)) pick[r.action] = r.id;
    }
    return pick;
  }, [task.runs]);

  const [action, setAction] = useState<"test" | "benchmark" | "profile">(() =>
    latest.benchmark ? "benchmark" : latest.test ? "test" : "profile");
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [chip, setChip] = useState<ChipProfile | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .chips()
      .then(async (list) => {
        if (!list.length) return;
        const t = (task.target_chip ?? "").toLowerCase();
        const pick =
          list.find((c) => c.models.some((m) => t && (m.toLowerCase().includes(t) || t.includes(m.toLowerCase())))) ??
          list[0];
        const doc = await api.chip(pick.name);
        if (alive) setChip(doc);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [task.target_chip]);

  const meta = ACTIONS.find((a) => a.key === action)!;
  const candidates = useMemo(
    () => task.runs.filter((r) => r.action === action && r.status === "ok").sort((a, b) => b.id - a.id),
    [task.runs, action],
  );
  const specById = useMemo(
    () => new Map(task.spec_versions.map((s) => [s.id, { version: s.version, sha256: s.sha256 }])),
    [task.spec_versions],
  );
  const selectedRun = useMemo(() => task.runs.find((r) => r.id === runId) ?? null, [task.runs, runId]);

  // 切动作时默认选最近一次成功；用户可再切历史 run
  useEffect(() => {
    setRunId(latest[action] ?? null);
  }, [action, latest]);

  useEffect(() => {
    setReport(null);
    setErr(null);
    if (!runId) return;
    let alive = true;
    setBusy(true);
    api
      .runReport(runId, meta.report)
      .then((r) => alive && setReport(r.report))
      .catch((e) => alive && setErr(String(e)))
      .finally(() => alive && setBusy(false));
    return () => {
      alive = false;
    };
  }, [runId, meta.report]);

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">结果视图 · 正确性 / 性能 / 指标</h2>
        <div className="ml-2 flex gap-1">
          {ACTIONS.map((a) => (
            <button
              key={a.key}
              className="btn btn-sm"
              onClick={() => setAction(a.key)}
              disabled={!latest[a.key]}
              style={a.key === action ? { background: "var(--accent-soft)", borderColor: "var(--accent)" } : undefined}
            >
              {a.label}
              {latest[a.key] ? ` #${latest[a.key]}` : " —"}
            </button>
          ))}
        </div>
        {candidates.length > 1 && (
          <select
            className="input max-w-[320px]"
            value={runId ?? ""}
            onChange={(e) => setRunId(Number(e.target.value))}
            title="切换历史 run（默认最近一次成功）"
          >
            {candidates.map((r) => (
              <option key={r.id} value={r.id}>
                #{r.id} · {(r.code_snapshot_sha256 ?? "—").slice(0, 8)} · loop {r.loop_id ?? "—"}
              </option>
            ))}
          </select>
        )}
        <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
          数据取自该 run 的机读报告（原样）；默认最近一次成功
        </span>
        {report && runId && (
          <button className="btn btn-sm" onClick={() => downloadJson(`run${runId}-${meta.report}.json`, report)}>
            导出 JSON
          </button>
        )}
      </div>

      {err && <div className="px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}
      {!runId && <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-mute)" }}>该动作还没有成功执行。</div>}
      {runId && busy && !report && <div className="px-4 py-4 text-xs" style={{ color: "var(--fg-mute)" }}>读取报告…</div>}
      {report && (
        <Alignment
          report={report}
          specById={specById}
          currentSpecId={task.current_spec_version_id}
          runSpecId={selectedRun?.spec_version_id ?? null}
        />
      )}

      {report && meta.report === "correctness" && <CorrectnessView report={report} />}
      {report && meta.report === "performance" && (
        <PerformanceView report={report} chip={chip} opName={String(report.op ?? task.op_name ?? "")} />
      )}
      {report && meta.report === "metrics" && <MetricsView report={report} />}
    </section>
  );
}

// ---------------------------------------------------------------- 正确性

function CorrectnessView({ report }: { report: Record<string, unknown> }) {
  const cases = (report.cases as CorrectnessCase[]) ?? [];
  const summary = (report.summary as Record<string, number>) ?? {};
  const option: EChartsOption = {
    grid: { left: 60, right: 20, top: 20, bottom: 70 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: cases.map((c) => c.case_id), axisLabel: { rotate: 45, fontSize: 9 } },
    yAxis: { type: "value", name: "max_abs_err" },
    series: [{
      type: "bar",
      data: cases.map((c) => ({
        value: c.max_abs_err ?? 0,
        itemStyle: { color: STATUS_COLOR[c.status] ?? "#94a3b8" },
      })),
    }],
  };
  const first = cases[0];
  return (
    <div className="px-4 pb-4">
      <div className="flex flex-wrap gap-2 py-2 text-[11px]">
        {["PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN"].map((s) => (
          <span key={s} className="chip" style={{ color: STATUS_COLOR[s] }}>{s} {summary[s] ?? 0}</span>
        ))}
        <span className="chip">共 {summary.TOTAL ?? cases.length}</span>
        {first?.tol?.profile && <span className="chip" style={{ color: "var(--fg-mute)" }}>容差档 {first.tol.profile}</span>}
      </div>
      {cases.some((c) => (c.max_abs_err ?? 0) > 0) && <Chart option={option} height={220} />}
      <div className="max-h-[320px] overflow-auto">
        <table>
          <thead>
            <tr><th>用例</th><th>组</th><th>期望</th><th>状态</th><th>matched</th><th>max_abs_err</th><th>说明</th></tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.case_id}>
                <td className="mono">{c.case_id}</td>
                <td>{c.group}</td>
                <td>{c.expected}</td>
                <td><span className="chip" style={{ color: STATUS_COLOR[c.status] }}>{c.status}</span></td>
                <td className="mono">{c.matched_ratio != null ? c.matched_ratio.toFixed(3) : "—"}</td>
                <td className="mono">{c.max_abs_err != null ? c.max_abs_err.toExponential(2) : "—"}</td>
                <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{(c.detail || c.note || "").slice(0, 80)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 性能基准

function PerformanceView({
  report,
  chip,
  opName,
}: {
  report: Record<string, unknown>;
  chip: ChipProfile | null;
  opName: string;
}) {
  const cases = (report.cases as BenchmarkCase[]) ?? [];
  const cond = (report.conditions as Record<string, unknown>) ?? {};
  const primary = cases.find((c) => c.primary) ?? cases[0];
  const hasMissingBaseline = cases.some((c) => c.baseline?.device_us?.median_us == null);
  const fixed = (value: number | null | undefined, digits: number) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

  // 基线名取自报告（不同算子的基线不一定是 torch.add；老报告没有就退成「基线」）
  const baseName = baselineLabel(report, primary);
  // 图例/表头用短名（去掉 "torch.add (torch_npu, device 0)" 里的括注），完整名留在上方元信息
  const baseShort = (baseName.replace(/\s*\(.*$/, "").trim() || baseName).slice(0, 28);
  const option: EChartsOption = {
    legend: { data: ["候选", baseShort], top: 0, textStyle: { fontSize: 10 } },
    grid: { left: 60, right: 20, top: 30, bottom: 50 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: cases.map((c) => c.case_id) },
    yAxis: { type: "value", name: "设备侧耗时 (us)" },
    series: [
      { name: "候选", type: "bar", data: cases.map((c) => c.candidate?.device_us?.median_us ?? "-"), itemStyle: { color: "#38bdf8" } },
      { name: baseShort, type: "bar", data: cases.map((c) => c.baseline?.device_us?.median_us ?? "-"), itemStyle: { color: "#94a3b8" } },
    ],
  };

  // 极限 / Roofline：AI 从报告显式值或算子画像推导，**不再写死 elementwise_add**
  const peakBw = chip?.model_variable?.hbm_bandwidth_gbps?.value;
  const peakFlopsT = chip?.compute?.vector_fp32_peak_tflops?.value;
  const aiInfo = arithmeticIntensity(report, opName, primary);
  const ai = aiInfo?.value ?? null;
  const achievedBw = primary?.derived?.candidate_achieved_bw_gbps;
  const baseBw = primary?.derived?.baseline_achieved_bw_gbps;
  const pct = peakBw && achievedBw ? (achievedBw / peakBw) * 100 : undefined;
  const ridge = peakFlopsT && peakBw ? (peakFlopsT * 1000) / peakBw : undefined; // flop/byte
  const bwBoundFlopsG = peakBw && ai != null ? ai * peakBw : undefined; // GFLOP/s（带宽上限对应的算力）
  const rooflineText =
    ai != null && ridge != null
      ? ai < ridge
        ? "带宽受限"
        : "算力受限"
      : null;
  const limitOption: EChartsOption | null =
    peakBw != null && achievedBw != null
      ? {
          grid: { left: 90, right: 30, top: 10, bottom: 24 },
          tooltip: { trigger: "axis" },
          xAxis: { type: "value", name: "GB/s" },
          yAxis: { type: "category", data: ["HBM 峰值", "基线达成", "候选达成"] },
          series: [{
            type: "bar",
            data: [
              { value: peakBw, itemStyle: { color: "#64748b" } },
              { value: baseBw ?? 0, itemStyle: { color: "#94a3b8" } },
              { value: achievedBw, itemStyle: { color: "#38bdf8" } },
            ],
          }],
        }
      : null;

  return (
    <div className="px-4 pb-4">
      <div className="flex flex-wrap gap-3 py-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
        <span>算子 <span className="mono">{opName || "—"}</span></span>
        <span>基线 <span className="mono">{baseName}</span></span>
        <span>测量范围 <span className="mono">{String(cond.measure_scope ?? "—")}</span></span>
        <span>采集次数 <span className="mono">{String(cond.invocations ?? "—")}</span></span>
        <span>warm-up <span className="mono">{String(cond.warm_up ?? "—")}</span></span>
        <span>设备 <span className="mono">{String(cond.device ?? "—")}</span> / 锁 <span className="mono">{String(cond.lock_mode ?? "—")}</span></span>
      </div>
      {hasMissingBaseline && (
        <div className="pb-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          本报告未定义同设备可比基线；候选耗时照常展示，基线与加速比记为“—”，不据此下性能提升结论。
        </div>
      )}
      <Chart option={option} height={220} />

      <div className="mt-3 mb-1 text-xs font-semibold">极限对比（Roofline 判定）</div>
      {chip && peakBw != null ? (
        <>
          <div className="flex flex-wrap gap-3 py-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            <span>画像 <span className="mono">{chip.name}</span>（{chip.arch}）</span>
            <span>HBM 峰值带宽 <span className="mono">{peakBw} GB/s</span></span>
            {peakFlopsT != null && <span>向量峰值 <span className="mono">{peakFlopsT} TFLOP/s</span></span>}
            {ridge != null && <span>Ridge 点 <span className="mono">{ridge.toFixed(2)} flop/byte</span></span>}
          </div>
          {ai != null ? (
            <div className="py-1 text-[11px]" style={{ color: "var(--fg-dim)" }}>
              算术强度 AI = <span className="mono">{ai.toFixed(3)}</span> flop/byte
              <span className="ml-1 chip !py-0 !text-[10px]">
                {aiInfo?.source === "report" ? "报告值" : "算子画像估算"}
              </span>
              {ridge != null && ` ${ai < ridge ? "≪" : "≫"} ridge ${ridge.toFixed(2)}`}
              {rooflineText && ` → ${rooflineText}`}；
              {bwBoundFlopsG != null && `带宽上限对应算力 ${bwBoundFlopsG.toFixed(0)} GFLOP/s`}；
              {pct != null && `候选达成 HBM 峰值的 ${pct.toFixed(1)}%`}
              {aiInfo?.note && (
                <span className="ml-1" style={{ color: "var(--fg-mute)" }}>（{aiInfo.note}）</span>
              )}
            </div>
          ) : (
            <div className="py-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              该算子没有可靠的算术强度画像（矩阵/注意力类 AI 随形状变化）→ 不做 Roofline 判定；
              达成带宽仍可对照 HBM 峰值看搬运效率。要给结论，请在 `benchmark.json` 里显式给出
              <span className="mono"> memory.reads_per_elem / writes_per_elem / compute.flops_per_elem</span>
              （或在报告里直接写 <span className="mono">arithmetic_intensity</span>）。
            </div>
          )}
          {limitOption && <Chart option={limitOption} height={130} />}
        </>
      ) : (
        <div className="py-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          没有硬件画像（`ops/chips/*.yaml`）→ 极限对比跳过。加一份画像再刷新。
        </div>
      )}

      <div className="mt-2 overflow-auto">
        <table>
          <thead>
            <tr><th>用例</th><th>N</th><th>候选(us)</th><th>{baseShort}(us)</th><th>加速比</th><th>候选带宽(GB/s)</th><th>基线带宽(GB/s)</th></tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.case_id}>
                <td className="mono">{c.case_id}{c.primary ? " ★" : ""}</td>
                <td>{c.n}</td>
                <td className="mono">{fixed(c.candidate?.device_us?.median_us, 3)}</td>
                <td className="mono">{fixed(c.baseline?.device_us?.median_us, 3)}</td>
                <td
                  className="mono"
                  style={{
                    color: c.derived?.speedup_baseline_over_candidate == null
                      ? "var(--fg-mute)"
                      : c.derived.speedup_baseline_over_candidate >= 1 ? "var(--ok)" : "var(--bad)",
                  }}
                >
                  {fixed(c.derived?.speedup_baseline_over_candidate, 3)}
                </td>
                <td className="mono">{fixed(c.derived?.candidate_achieved_bw_gbps, 0)}</td>
                <td className="mono">{fixed(c.derived?.baseline_achieved_bw_gbps, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 指标采集

const METRIC_COLOR: Record<string, string> = {
  measured: "var(--ok)", estimated: "var(--run)", unsupported: "var(--fg-mute)", failed: "var(--bad)",
};

function MetricsView({ report }: { report: Record<string, unknown> }) {
  const metrics = (report.metrics as MetricRow[]) ?? [];
  const info = (report.op_info as Record<string, unknown>) ?? {};
  const ratios = metrics.filter((m) => m.status === "measured" && /_ratio$/.test(m.id));
  const bws = metrics.filter((m) => m.status === "measured" && m.unit === "GB/s");
  const bar = (rows: MetricRow[], name: string): EChartsOption => ({
    grid: { left: 150, right: 30, top: 10, bottom: 30 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "value", name },
    yAxis: { type: "category", data: rows.map((m) => m.id), axisLabel: { fontSize: 10 } },
    series: [{ type: "bar", data: rows.map((m) => ({ value: m.value ?? 0, itemStyle: { color: "#38bdf8" } })) }],
  });
  return (
    <div className="px-4 pb-4">
      <div className="flex flex-wrap gap-3 py-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
        <span>算子 <span className="mono">{String(info.op_name ?? "—")}</span>（{String(info.op_type ?? "—")}）</span>
        <span>Task Duration <span className="mono">{String(info.task_duration_us ?? "—")}</span> us</span>
        <span>Block Dim <span className="mono">{String(info.block_dim ?? "—")}</span></span>
      </div>
      {ratios.length > 0 && <Chart option={bar(ratios, "占比")} height={120} />}
      {bws.length > 0 && <Chart option={bar(bws, "GB/s")} height={140} />}
      <div className="max-h-[340px] overflow-auto">
        <table>
          <thead>
            <tr><th>指标</th><th>类别</th><th>状态</th><th>值</th><th>单位</th><th>来源</th></tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m.id}>
                <td className="mono">{m.id}{m.required ? " *" : ""}</td>
                <td>{m.kind}</td>
                <td><span className="chip" style={{ color: METRIC_COLOR[m.status] }}>{m.status}</span></td>
                <td className="mono">{m.value != null ? (Math.abs(m.value) < 1 ? m.value.toFixed(4) : m.value.toFixed(2)) : "—"}</td>
                <td>{m.unit ?? ""}</td>
                <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{(m.source ?? "").slice(0, 60)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
