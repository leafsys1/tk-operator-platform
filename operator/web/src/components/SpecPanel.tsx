import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { SpecVersion } from "../types";

type GroupKey = "identity" | "semantics" | "validation" | "performance" | "delivery" | "notes" | "other";

const GROUPS: { key: GroupKey; title: string }[] = [
  { key: "identity", title: "标识与环境" },
  { key: "semantics", title: "语义与接口" },
  { key: "validation", title: "正确性口径" },
  { key: "performance", title: "性能口径" },
  { key: "delivery", title: "交付" },
  { key: "notes", title: "假设与未确定项" },
  { key: "other", title: "其他字段" },
];

/** 字段名 → 人话标签 + 分组；没登记过的字段自动落到「其他字段」，不吞数据。 */
const FIELD_META: Record<string, { label: string; group: GroupKey }> = {
  op_name: { label: "算子名", group: "identity" },
  target_chip: { label: "目标芯片", group: "identity" },
  target_env: { label: "目标环境", group: "identity" },
  target_hardware_profile: { label: "目标硬件画像", group: "identity" },
  target: { label: "目标环境", group: "identity" },
  source: { label: "来源", group: "identity" },
  math_def: { label: "数学语义", group: "semantics" },
  semantics: { label: "数学语义", group: "semantics" },
  params: { label: "属性参数", group: "semantics" },
  inputs: { label: "输入", group: "semantics" },
  outputs: { label: "输出", group: "semantics" },
  layout: { label: "布局", group: "semantics" },
  support: { label: "支持范围", group: "semantics" },
  unsupported: { label: "不支持 / 报错", group: "semantics" },
  special_behavior: { label: "特殊行为", group: "semantics" },
  constraints: { label: "约束", group: "semantics" },
  oracle: { label: "参考实现 / 基线", group: "validation" },
  reference: { label: "参考实现 / 基线", group: "validation" },
  tolerance: { label: "精度容差", group: "validation" },
  validation: { label: "正确性口径", group: "validation" },
  consistency: { label: "一致性要求", group: "validation" },
  performance: { label: "性能口径", group: "performance" },
  performance_baseline: { label: "性能基线", group: "performance" },
  performance_target: { label: "性能目标", group: "performance" },
  delivery: { label: "交付内容", group: "delivery" },
  assumptions: { label: "假设", group: "notes" },
  unknowns: { label: "未确定项", group: "notes" },
};

const VALUE_LABEL: Record<string, string> = {
  count: "数量",
  shape: "形状",
  dtype: "数据类型",
  layout: "布局",
  min: "最小",
  max: "最大",
  default: "默认值",
  range: "取值范围",
  required: "必需",
  optional: "可选",
};

function scalar(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "是" : "否";
  return String(v);
}

function isEmpty(v: unknown): boolean {
  return v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);
}

/** 值渲染：短值直接读，数组成列表，对象成 key-value（不把 JSON 原样糊在页面上）。 */
function Value({ value, tone }: { value: unknown; tone?: string }) {
  if (isEmpty(value)) {
    return <span style={{ color: "var(--fg-mute)" }}>—</span>;
  }
  if (typeof value !== "object") {
    return (
      <span className="whitespace-pre-wrap text-[12px] leading-[1.75]" style={{ color: tone }}>
        {scalar(value)}
      </span>
    );
  }
  if (Array.isArray(value)) {
    const primitive = value.every((v) => v === null || typeof v !== "object");
    if (primitive) {
      return (
        <ul className="m-0 list-disc pl-4 text-[12px] leading-[1.75]" style={{ color: tone }}>
          {value.map((v, i) => (
            <li key={i}>{scalar(v)}</li>
          ))}
        </ul>
      );
    }
  }
  if (typeof value === "object" && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return <span style={{ color: "var(--fg-mute)" }}>—</span>;
    return (
      <div className="grid gap-x-5 gap-y-1 sm:grid-cols-2">
        {entries.map(([k, v]) => (
          <div key={k} className="flex min-w-0 gap-2 text-[12px] leading-[1.7]">
            <span className="shrink-0" style={{ color: "var(--fg-mute)" }}>
              {VALUE_LABEL[k] ?? k}
            </span>
            <span className="min-w-0 break-words" style={{ color: tone }}>
              {typeof v === "object" && v !== null ? JSON.stringify(v) : scalar(v)}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return (
    <pre
      className="m-0 max-h-[220px] overflow-auto whitespace-pre-wrap rounded-[8px] p-2 text-[11px] leading-[1.6]"
      style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "inherit" }}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

/** 「修改需求/规格（走对话）」预填文案：把协议和当前版本一起交给对话框，用户补一句要改什么即可。 */
function modifyPrompt(version: number | null): string {
  const lines = [
    `我想修改当前任务的需求/规格${version != null ? `（当前是 v${version}）` : ""}。`,
    "",
    "请按「规格修改走对话」的协议处理：先问清我要改什么，判断是实质变更还是文档性修订。",
    "· 实质变更：起草新的完整 spec_json（缺失项标 UNKNOWN），在对话里给我摘要与差异；我确认后你调 POST /api/tasks/{id}/specs 落新版（大版本 +1、小版本归 0），并提醒我用例需要重新确认、默认需要重跑 test + benchmark。",
    "· 文档性修订：可以不落规格版本，只在对话里留结论。",
    "",
    "我要改的内容：",
  ];
  return lines.join("\n");
}

/** 任务详情「需求与规格」：当前规格 + 版本历史（只读）。
 *
 * 规格是代码生成/测试生成/验收的共同依据。规格修改**走对话**（agent 起草 → 用户确认 → 落新版），
 * 页面不再提供 JSON 编辑器/新建版本表单，避免绕过确认与版本协议。
 * 布局按「标识 / 语义 / 验证 / 性能 / 交付 / 假设」分组，未知字段兜底到「其他字段」。
 */
export default function SpecPanel({
  taskId,
  currentVersion = null,
  onOpenChat,
}: {
  taskId: number;
  /** 任务当前规格版本号（提示文案里的「当前是 vN」用它，与下拉里正在看的版本无关）。 */
  currentVersion?: number | null;
  /** 跳到任务对话并预填「修改需求/规格」文案（由 TaskWorkbench 提供 openChat）。 */
  onOpenChat?: (text: string) => void;
}) {
  const [rows, setRows] = useState<SpecVersion[]>([]);
  const [pick, setPick] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .specs(taskId)
      .then((list) => {
        if (!alive) return;
        setRows(list);
        setPick(list.length ? list[list.length - 1].id : null);
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [taskId]);

  const current = useMemo(() => rows.find((r) => r.id === pick) ?? null, [rows, pick]);

  const fields = useMemo(() => {
    if (!current?.spec_json) return null;
    try {
      const data = JSON.parse(current.spec_json) as Record<string, unknown>;
      return data && typeof data === "object" && !Array.isArray(data) ? data : null;
    } catch {
      return null;
    }
  }, [current]);

  // 分组只保留实际存在的字段；`_meta` 是内部元数据，不占版面。
  const grouped = useMemo(() => {
    if (!fields) return [];
    const buckets = new Map<GroupKey, { key: string; value: unknown; label: string }[]>();
    for (const [key, value] of Object.entries(fields)) {
      if (key.startsWith("_")) continue;
      const meta = FIELD_META[key];
      const group = meta?.group ?? "other";
      if (!buckets.has(group)) buckets.set(group, []);
      buckets.get(group)!.push({ key, value, label: meta?.label ?? key });
    }
    return GROUPS.filter((g) => buckets.has(g.key)).map((g) => ({ ...g, items: buckets.get(g.key)! }));
  }, [fields]);

  const showAssumptions = Boolean(current?.assumptions?.trim()) && current?.assumptions?.trim() !== "[]" && !(fields && "assumptions" in fields);
  const showUnknowns = Boolean(current?.unknowns?.trim()) && current?.unknowns?.trim() !== "[]" && !(fields && "unknowns" in fields);

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">需求与规格</h2>
        {rows.length > 0 && (
          <>
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              {rows.length} 个版本
            </span>
            <select className="input ml-auto max-w-[280px]" value={pick ?? ""} onChange={(e) => setPick(Number(e.target.value))}>
              {rows.map((r) => (
                <option key={r.id} value={r.id}>
                  v{r.version} · {r.op_name ?? "—"} · {r.change_reason ?? "—"}
                </option>
              ))}
            </select>
          </>
        )}
        <button
          className={`btn btn-sm ${rows.length === 0 ? "ml-auto" : ""}`}
          onClick={() => onOpenChat?.(modifyPrompt(currentVersion ?? current?.version ?? null))}
          disabled={!onOpenChat}
          title="走对话：agent 起草新规格 → 你确认 → 落新版（大版本 +1，小版本归 0，用例需重新确认）"
        >
          修改需求/规格（走对话）
        </button>
      </div>

      <div className="px-4 py-2 text-[11px]" style={{ color: "var(--fg-mute)", background: "var(--panel-3)", borderBottom: "1px solid var(--border)" }}>
        规格修改走对话：先说明要改什么 —— agent 判断实质变更/文档性修订，起草新规格并等你确认后落版；页面只读。
      </div>

      {err && <div className="px-4 py-3 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}
      {rows.length === 0 && !err && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          还没有规格版本 —— 先在「任务开始」页完成对话与定版。
        </div>
      )}

      {current && (
        <>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2.5 text-[11px]" style={{ color: "var(--fg-mute)", borderBottom: "1px solid var(--border)" }}>
            <span>版本 <b className="mono" style={{ color: "var(--fg)" }}>v{current.version}</b></span>
            <span>哈希 <span className="mono">{(current.sha256 ?? "—").slice(0, 12)}…</span></span>
            <span>变更原因 {current.change_reason ?? "—"}</span>
            {current.op_name && <span>算子 <span className="mono">{current.op_name}</span></span>}
          </div>

          {fields ? (
            <div className="px-4 py-1">
              {grouped.map((g) => (
                <div key={g.key} className="py-2">
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-mute)" }}>
                    {g.title}
                  </div>
                  {g.items.map(({ key, value, label }) => (
                    <div
                      key={key}
                      className="grid gap-1 py-1.5 md:grid-cols-[104px_minmax(0,1fr)] md:gap-3"
                      style={{ borderTop: "1px dashed var(--border)" }}
                    >
                      <div className="text-[10px] leading-[1.9]" style={{ color: "var(--fg-mute)" }}>{label}</div>
                      <div className="min-w-0">
                        <Value value={value} tone={key === "unknowns" ? "var(--warn)" : undefined} />
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              该版本没有可解析的 spec_json。
            </div>
          )}

          {(showAssumptions || showUnknowns) && (
            <div className="grid gap-2 px-4 pb-3 md:grid-cols-2">
              {showAssumptions && (
                <div>
                  <div className="text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-mute)" }}>版本假设</div>
                  <div className="mt-1 text-[11px] leading-[1.7]" style={{ color: "var(--fg-dim)" }}>{current.assumptions}</div>
                </div>
              )}
              {showUnknowns && (
                <div>
                  <div className="text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-mute)" }}>版本未确定项</div>
                  <div className="mt-1 text-[11px] leading-[1.7]" style={{ color: "var(--warn)" }}>{current.unknowns}</div>
                </div>
              )}
            </div>
          )}

          <details className="px-4 pb-3">
            <summary className="cursor-pointer text-[10px]" style={{ color: "var(--fg-mute)" }}>
              原始 JSON（排查用）
            </summary>
            <pre
              className="mt-2 mb-0 max-h-[260px] overflow-auto rounded-[8px] p-3 text-[10px] leading-[1.6]"
              style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace" }}
            >
              {current.spec_json ?? "{}"}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}
