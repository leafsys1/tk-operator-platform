import { useEffect, useState } from "react";
import { api, size, when, type DeliveryCreateError } from "../lib/api";
import type { DeliveryRecord, TaskDetail, TaskVersions } from "../types";

const ACTION_LABEL: Record<string, string> = {
  build: "编译 build",
  smoke: "冒烟 smoke",
  test: "正确性 test",
  benchmark: "基准 benchmark",
  profile: "采集 profile",
  cases: "用例组确认",
  delivery: "交付 delivery",
};

/** 交付门禁缺项 → 页面提示该去跑哪一步（或需要重跑什么）。 */
const MISSING_HINT: Record<string, string> = {
  build: "去「概览」执行 ② 编译；或确认该动作最新 run 与当前版本（规格 / 代码快照 / 用例集）一致",
  smoke: "去「概览」执行 3 冒烟（编译+最小用例）",
  test: "去「概览」执行 4 正确性测试；或确认该动作最新 run 与当前版本一致且门禁 pass",
  benchmark: "去「概览」执行 5 基准测试；或确认该动作最新 run 与当前版本一致且门禁 pass",
  profile: "去「概览」执行 6 性能采集（可选动作，不影响交付门禁）",
  cases: "在任务对话里整组确认精度/性能用例组（确认后用例集变化要重跑 test + benchmark）",
  delivery: "重新生成交付包",
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

/**
 * 交付面板（任务生命周期 v2，契约 §3.4 / §3.5）：
 * - 交付门禁预览：`GET /versions` 的 evidence（用例已确认 + build/test/benchmark 证据有效 + 产物齐备）；
 * - 生成交付包：`POST /deliveries`（409 → 展示 detail + codes，并刷新缺项）；
 * - 交付历史：`GET /deliveries` + 下载（`download_url` / `downloadDeliveryUrl`）。
 *
 * 后端是门禁权威：预览只是把缺项提前说清楚，「生成交付包」仍以 POST 的 409 为准。
 */
export default function DeliveryPanel({
  task,
  onOpenOverview,
  onChanged,
}: {
  task: TaskDetail;
  /** 缺项时跳「概览」执行对应动作（该分区的执行面板在那边）。 */
  onOpenOverview?: () => void;
  /** 交付成功后通知父级重拉任务详情（版本交付状态 / runs / 阶段）。 */
  onChanged?: () => void;
}) {
  const [versions, setVersions] = useState<TaskVersions | null>(null);
  const [versionsErr, setVersionsErr] = useState<string | null>(null);
  const [records, setRecords] = useState<DeliveryRecord[] | null>(null);
  const [recordsErr, setRecordsErr] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<DeliveryRecord | null>(null);
  const [createErr, setCreateErr] = useState<string | null>(null);
  const [createCodes, setCreateCodes] = useState<string[]>([]);
  const [nonce, setNonce] = useState(0);

  // 门禁 / 历史随 run 变化刷新（这里也兜住「交付后 runs 多一条」的情况）。
  useEffect(() => {
    let alive = true;
    setReloading(true);
    Promise.allSettled([api.versions(task.id), api.deliveries(task.id)]).then(([v, d]) => {
      if (!alive) return;
      if (v.status === "fulfilled") {
        setVersions(v.value);
        setVersionsErr(null);
      } else {
        setVersions(null);
        setVersionsErr(errorText(v.reason));
      }
      if (d.status === "fulfilled") {
        setRecords(d.value.deliveries);
        setRecordsErr(null);
      } else {
        setRecords(null);
        setRecordsErr(errorText(d.reason));
      }
      setReloading(false);
    });
    return () => {
      alive = false;
    };
  }, [task.id, task.runs.length, task.current_spec_version_id, task.version_minor,
      task.cases_confirmed_at, task.cases_confirmed_sha256,
      task.agent_jobs.map((j) => `${j.id}:${j.status}`).join(","), nonce]);

  const refresh = () => setNonce((n) => n + 1);

  const createDelivery = async () => {
    setCreating(true);
    setCreateErr(null);
    setCreateCodes([]);
    setCreated(null);
    try {
      const rec = await api.createDelivery(task.id);
      setCreated(rec);
      refresh();
      onChanged?.();
    } catch (e) {
      const de = e as DeliveryCreateError;
      setCreateErr(de.detail ?? String(e));
      setCreateCodes(de.codes ?? []);
      // 409 后把最新缺项拉出来（用例未确认 / 证据失效 / 产物缺失）。
      try {
        setVersions(await api.versions(task.id));
        setVersionsErr(null);
      } catch {
        /* 保留原来的 versions 状态 */
      }
    } finally {
      setCreating(false);
    }
  };

  const evidence = versions?.evidence ?? null;
  const current = versions?.current ?? null;
  const missing = evidence?.missing ?? [];

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">交付：门禁预览 · 生成交付包 · 交付历史</h2>
        <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
          门禁由真实证据判定；「生成交付包」失败会展示后端缺项
        </span>
        <button className="btn btn-sm" onClick={refresh} disabled={reloading}>
          {reloading ? "…" : "刷新"}
        </button>
      </div>

      {/* ── 门禁预览 ─────────────────────────────────────────── */}
      <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-semibold">交付门禁</span>
          {evidence ? (
            <span className={`chip ${evidence.deliverable ? "chip-ok" : "chip-bad"}`}>
              {evidence.deliverable ? "齐备，可以交付" : `缺 ${missing.length} 项`}
            </span>
          ) : versionsErr ? (
            <span className="chip chip-warn">版本接口不可用</span>
          ) : (
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>读取版本/证据…</span>
          )}
          {current ? (
            <span className="mono text-[11px]" style={{ color: "var(--fg-dim)" }}>
              当前 v{current.major}.{current.minor}
              {current.delivered_at ? ` · 交付于 ${when(current.delivered_at)}` : " · 未交付"}
            </span>
          ) : evidence?.version ? (
            <span className="mono text-[11px]" style={{ color: "var(--fg-dim)" }}>
              当前 {evidence.version}（版本行未落库，按当前规格推导）
            </span>
          ) : null}
          <span
            className={task.cases_confirmed_at ? "chip chip-ok !py-0 !text-[10px]" : "chip chip-bad !py-0 !text-[10px]"}
            title={task.cases_confirmed_at ? `用例组确认于 ${when(task.cases_confirmed_at)}` : "用例组未整组确认（在任务对话里确认）"}
          >
            用例{task.cases_confirmed_at ? "已确认" : "未确认"}
          </span>
          {onOpenOverview && (
            <button className="btn btn-sm ml-auto" onClick={onOpenOverview}>
              去「概览」执行动作
            </button>
          )}
        </div>

        {versionsErr && (
          <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            门禁预览读取失败：{versionsErr}（仍可点「生成交付包」，以接口 409 的缺项为准）
          </div>
        )}

        {evidence && (
          <div className="mt-2 grid gap-1.5 md:grid-cols-2">
            {evidence.required_actions.map((action) => {
              const r = evidence.runs[action] ?? null;
              const miss = missing.find((m) => m.action === action);
              const check = evidence.checks?.find((c) => c.action === action) ?? null;
              // checks 是后端统一判定（含 waived）；缺 checks 时按 runs + missing 兜底。
              const waived = Boolean(check?.waived);
              const ok = check ? check.ok : !miss && r?.status === "ok" && r.gate_verdict === "pass";
              return (
                <div
                  key={action}
                  className="flex flex-wrap items-center gap-2 rounded-[8px] px-3 py-1.5 text-[11px]"
                  style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
                >
                  <span className="mono">{ACTION_LABEL[action] ?? action}</span>
                  {r ? (
                    <>
                      <span className="mono" style={{ color: "var(--fg-mute)" }}>
                        run #{r.run_id} · {r.status}
                      </span>
                      <span className={`chip !py-0 !text-[10px] ${r.gate_verdict === "pass" ? "chip-ok" : "chip-bad"}`}>
                        门禁 {r.gate_verdict ?? "—"}
                      </span>
                    </>
                  ) : (
                    <span className="chip !py-0 !text-[10px]">未跑</span>
                  )}
                  <span className="ml-auto text-[10px]" style={{ color: waived ? "var(--warn)" : ok ? "var(--ok)" : "var(--bad)" }}>
                    {waived ? "对话确认豁免（无需重跑）" : ok ? "证据有效" : miss?.reason ?? "证据失效"}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {missing.length > 0 && (
          <div className="mt-2">
            <div className="text-[11px] font-semibold" style={{ color: "var(--bad)" }}>
              交付缺项（{missing.length}）
            </div>
            <ul className="mt-1 mb-0 pl-4 text-[11px]" style={{ color: "var(--fg-dim)" }}>
              {missing.map((m, i) => (
                <li key={`${m.action ?? m.code ?? "missing"}-${i}`}>
                  <span className="mono" style={{ color: "var(--bad)" }}>{m.label ?? m.action ?? m.code ?? m.item ?? "交付"}</span> · {m.reason}
                  <span style={{ color: "var(--fg-mute)" }}>
                    {" "}—— {(m.action ? MISSING_HINT[m.action] : undefined) ?? "去「概览」重跑对应动作后重新交付"}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {evidence && (evidence.current_snapshot_sha256 || evidence.current_suite_ref) && (
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
            <span>
              当前代码快照 <span className="mono">{(evidence.current_snapshot_sha256 ?? "—").slice(0, 12)}</span>
            </span>
            <span>
              当前用例集 <span className="mono">{evidence.current_suite_ref ?? "—"}</span>
            </span>
          </div>
        )}
      </div>

      {/* ── 生成交付包 ───────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <button
          className="btn btn-primary"
          onClick={() => void createDelivery()}
          disabled={creating}
          title={evidence && !evidence.deliverable ? "门禁当前有缺项：仍会请求后端，409 时展示缺项" : "生成 zip 交付包并登记 artifact"}
        >
          {creating ? "打包中…" : "生成交付包"}
        </button>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          校验：用例已确认且未变 · build/smoke/test/benchmark 证据与当前版本一致 · 五件产物齐备
        </span>

        {created && (
          <div className="flex w-full flex-wrap items-center gap-2 text-[11px]">
            <span className="chip chip-ok">已生成交付包 {created.version}</span>
            <a
              className="btn btn-sm btn-primary"
              href={created.download_url ?? api.downloadDeliveryUrl(created.run_id)}
              target="_blank"
              rel="noreferrer"
            >
              下载交付包
            </a>
            <span className="mono" style={{ color: "var(--fg-mute)" }}>
              run #{created.run_id} · {(created.sha256 ?? "").slice(0, 12) || "—"}… · {size(created.size_bytes)} · {when(created.created_at)}
            </span>
          </div>
        )}

        {createCodes.length > 0 && (
          <div className="flex w-full flex-wrap gap-1.5">
            {createCodes.map((c) => (
              <span key={c} className="chip chip-bad !py-0 !text-[10px] mono">{c}</span>
            ))}
          </div>
        )}

        {createErr && (
          <div className="w-full text-[11px]" style={{ color: "var(--bad)" }}>
            交付失败：{createErr}
            {missing.length > 0 && <span style={{ color: "var(--fg-mute)" }}>（缺项见上方「交付门禁」）</span>}
          </div>
        )}
      </div>

      {/* ── 交付历史 ─────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-4 py-2" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="text-[11px] font-semibold">交付历史</span>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          {records ? `${records.length} 次` : "读取中…"}
        </span>
      </div>
      {recordsErr && (
        <div className="px-4 py-2 text-[11px]" style={{ color: "var(--bad)" }}>
          交付历史读取失败：{recordsErr}
        </div>
      )}
      {records && records.length === 0 && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          还没有交付记录 —— 门禁齐备后点「生成交付包」，交付包可在这里下载。
        </div>
      )}
      {records && records.length > 0 && (
        <div className="overflow-auto">
          <table>
            <thead>
              <tr>
                <th>版本</th>
                <th>run</th>
                <th>生成时间</th>
                <th>大小</th>
                <th>sha256</th>
                <th>下载</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.run_id}>
                  <td className="mono text-[11px]" title={r.version} style={{ maxWidth: 220, wordBreak: "break-all" }}>{r.version || "—"}</td>
                  <td className="mono">
                    #{r.run_id}
                    {r.status && r.status !== "ok" && <span className="chip chip-bad !py-0 !ml-1 !text-[10px]">{r.status}</span>}
                  </td>
                  <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>{when(r.created_at)}</td>
                  <td className="mono">{size(r.size_bytes)}</td>
                  <td className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>{(r.sha256 ?? "—").slice(0, 12)}</td>
                  <td>
                    <a className="btn btn-sm" href={r.download_url ?? api.downloadDeliveryUrl(r.run_id)} target="_blank" rel="noreferrer">
                      下载
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
