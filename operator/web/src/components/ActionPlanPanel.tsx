import { useEffect, useRef, useState } from "react";
import { api, size, when, type DeliveryCreateError } from "../lib/api";
import type { ActionJob, DeliveryRecord, ExecutionPlan, Run, TaskGates, TaskVersions } from "../types";

const ACTIONS: { key: string; label: string }[] = [
  { key: "build", label: "编译" },
  { key: "smoke", label: "冒烟（编译+最小用例）" },
  { key: "test", label: "正确性测试" },
  { key: "benchmark", label: "基准测试" },
  { key: "profile", label: "性能采集" },
];

/** 交付门禁缺项 → 页面上告诉用户该去跑哪一步。 */
const MISSING_HINT: Record<string, string> = {
  build: "去「概览」执行 ② 编译（build）",
  smoke: "去「概览」执行 ③ 冒烟（smoke，编译+最小用例跑通）",
  test: "去「概览」执行 ④ 正确性测试（test）",
  benchmark: "去「概览」执行 ⑤ 基准测试（benchmark）",
  profile: "去「概览」执行 ⑥ 性能采集（profile，可选）",
  cases: "先在「对话」里确认精度/性能用例组",
  delivery: "重跑交付（生成交付包）",
};

const STATUS_CHIP: Record<string, { cls: string; text: string }> = {
  running: { cls: "chip", text: "执行中" },
  ok: { cls: "chip chip-ok", text: "已完成" },
  failed: { cls: "chip chip-bad", text: "失败" },
  interrupted: { cls: "chip chip-warn", text: "已中断" },
};

/** 每个动作最近一次成功执行：用来区分「这次作业失败了」和「这个动作没有可用产物」。 */
function latestOkByAction(runs: Run[]): Record<string, Run> {
  const m: Record<string, Run> = {};
  for (const r of runs) {
    if (r.status !== "ok") continue;
    if (!m[r.action] || r.id > m[r.action].id) m[r.action] = r;
  }
  return m;
}

/** 真实执行的启动面板：算 plan（dry_run）→ 点「真执行」起异步作业 → 轮询进度与门禁结论。
 *
 * 后端 `services/action_runner.py` 照 plan 调 `scripts/nm_test.sh`；脚本自己登记 run，
 * 作业行只记过程（日志/退出码），跑完把 `run_id` 与门禁 verdict 回填。
 * 退出码不是结论 —— 业务达标看 `gate_verdict`。
 *
 * 最近一次作业失败时，若该动作已有成功的 run，明确并列展示两者：
 * 作业失败只说明「这次没跑成」，不推翻已有 run 的门禁结论（常见于脚本/环境问题重跑后）。
 */
export default function ActionPlanPanel({
  taskId,
  opName,
  runs = [],
  gates = null,
  evidenceRevision,
  onDone,
  onOpenChat,
}: {
  taskId: number;
  opName?: string | null;
  runs?: Run[];
  gates?: TaskGates | null;
  evidenceRevision?: string;
  onDone?: () => void;
  /** 跳到任务「对话」分区并预填一段信息（开发 / AI 审查用）。 */
  onOpenChat?: (text: string) => void;
}) {
  const [plan, setPlan] = useState<ExecutionPlan | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<ActionJob | null>(null);
  const [history, setHistory] = useState<ActionJob[]>([]);
  const logRef = useRef<HTMLPreElement | null>(null);
  // 交付（第七步）：本地动作（后端打包）；门禁缺项与交付状态来自 GET /versions。
  const [versions, setVersions] = useState<TaskVersions | null>(null);
  const [versionsErr, setVersionsErr] = useState<string | null>(null);
  const [delivering, setDelivering] = useState(false);
  const [delivery, setDelivery] = useState<DeliveryRecord | null>(null);
  const [deliveryErr, setDeliveryErr] = useState<string | null>(null);
  const [deliveryCodes, setDeliveryCodes] = useState<string[]>([]);

  // 进页面先接上最近一次作业（刷新页面不丢进度）
  useEffect(() => {
    let alive = true;
    api.actionJobs(taskId)
      .then((list) => {
        if (!alive) return;
        setHistory(list);
        if (list.length > 0) setJob(list[0]);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [taskId]);

  // 交付门禁预览：版本/证据随 run 变化刷新（老后端还没有 /versions 时只影响 ⑦ 的预览，不致命）。
  useEffect(() => {
    let alive = true;
    api.versions(taskId)
      .then((v) => {
        if (!alive) return;
        setVersions(v);
        setVersionsErr(null);
      })
      .catch((e) => {
        if (alive) setVersionsErr(String(e));
      });
    return () => {
      alive = false;
    };
  }, [taskId, runs.length, evidenceRevision]);

  const refreshVersions = () => {
    void api.versions(taskId)
      .then((v) => {
        setVersions(v);
        setVersionsErr(null);
      })
      .catch((e) => setVersionsErr(String(e)));
  };

  /** ⑦ 交付：POST /deliveries；409 时展示后端 detail + codes（缺项），并刷新门禁缺项。 */
  const startDelivery = async () => {
    setDelivering(true);
    setDeliveryErr(null);
    setDeliveryCodes([]);
    setDelivery(null);
    try {
      const rec = await api.createDelivery(taskId);
      setDelivery(rec);
      onDone?.();
      refreshVersions();
      void api.actionJobs(taskId).then(setHistory).catch(() => {});
    } catch (e) {
      const de = e as DeliveryCreateError;
      setDeliveryErr(de.detail ?? String(e));
      setDeliveryCodes(de.codes ?? []);
      refreshVersions();
    } finally {
      setDelivering(false);
    }
  };

  // 运行中每 1.5s 拉一次状态（日志在作业行里，跑完自动停）
  useEffect(() => {
    if (!job?.running) return;
    const id = window.setInterval(() => {
      api.actionJob(job.id)
        .then((next) => {
          setJob(next);
          if (!next.running) {
            onDone?.();
            void api.actionJobs(taskId).then(setHistory).catch(() => {});
          }
        })
        .catch(() => {});
    }, 1500);
    return () => window.clearInterval(id);
  }, [job?.id, job?.running, onDone]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log]);

  const compute = async (action: string) => {
    setBusy(action);
    setPlan(null);
    setErr(null);
    try {
      setPlan(await api.planAction(taskId, action));
    } catch (e) {
      setPlan(null);
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  };

  const start = async () => {
    if (!plan || !plan.ready || busy || starting || job?.running) return;
    setStarting(true);
    setErr(null);
    try {
      const res = await api.runAction(taskId, plan.action, plan.device_id);
      setJob(await api.actionJob(res.action_job_id));
      void api.actionJobs(taskId).then(setHistory).catch(() => {});
    } catch (e) {
      setErr(String(e));
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      await api.cancelActionJob(job.id);
    } catch (e) {
      setErr(String(e));
    }
  };

  const chip = job ? STATUS_CHIP[job.status] ?? { cls: "chip", text: job.status } : null;

  /** 每个动作最近一次 run（含失败/中断）—— 步骤条显示进度、AI 审查取它做上下文。 */
  const latest: Record<string, Run> = {};
  for (const r of runs) {
    if (!latest[r.action] || r.id > latest[r.action].id) latest[r.action] = r;
  }

  const devPrompt = () => {
    const op = opName || "本任务算子";
    return `继续开发 ${op}。需求规格见 tasks/${op}/TASK.md 与 tasks/${op}/spec/spec.json；先规划再实现。\n` +
      `交付 5 件：① ops/${op}/ 算子代码 ② ops/${op}/python/${op}.py Python 调用接口（ctypes + host C ABI）` +
      ` ③ tests/test_${op}.py pytest 测试脚本 ④ ops/${op}/README.md 算子说明文档 ⑤ 规划文档补全稿` +
      `（docs/规划文档-补全.md）。\n` +
      `用例见 cases/${op}/{cases.json,benchmark.json}（若用例组还没在对话里确认，先和我确认整组用例）。\n` +
      `开发完成后**先编译验证，再用一条最小用例确认能跑通**：` +
      `scripts/nm.sh --op ${op} --action build，再 scripts/nm.sh --op ${op} --action smoke；` +
      `smoke 返回 SMOKE_VERDICT PASS 后才运行正式 test/benchmark；失败先修，不许跳过。有不确定的先问我。`;
  };

  const REVIEW_REPORT: Record<string, string> = { smoke: "smoke", test: "correctness", benchmark: "performance", profile: "metrics" };

  /** 「AI 审查」上下文：把 run / 门禁 / 报告入口 / 日志尾部带进对话，agent 不用猜。 */
  const reviewPrompt = (
    action: string,
    run: Run,
    gateVerdict?: string | null,
    failCodes?: string[],
    logTail?: string,
  ) => {
    const label = ACTIONS.find((a) => a.key === action)?.label ?? action;
    const gate = gates?.gates.find((g) => g.action === action) ?? null;
    const verdict = gateVerdict ?? gate?.verdict ?? "—";
    const fails = failCodes ?? gate?.fail_codes ?? [];
    const warns = gate?.warn_codes ?? [];
    const report = REVIEW_REPORT[action]
      ? `http://127.0.0.1:8077/api/runs/${run.id}/report?type=${REVIEW_REPORT[action]}`
      : `（该动作没有机读报告，看下面日志尾部与 run 产物）`;
    const lines = [
      `请审查 ${opName || "本任务"} 的「${label}」执行结果：`,
      `- run #${run.id}：status=${run.status}` +
        (run.exit_code != null ? `，exit_code=${run.exit_code}` : "") +
        (run.failure_kind ? `，failure_kind=${run.failure_kind}` : ""),
      `- 门禁：verdict=${verdict}，fail_codes=${fails.length ? fails.join(",") : "无"}，warn_codes=${warns.length ? warns.join(",") : "无"}`,
      `- 门禁接口：http://127.0.0.1:8077/api/runs/${run.id}/gate`,
      `- 报告：${report}`,
      `- 产物/日志：执行机 run 目录 ${run.remote_workdir ?? "—"}（远端 /data/lfx/op-agent/runs/；本机 runs/ 未必有这次 run）`,
    ];
    if (logTail) {
      lines.push("", "作业日志尾部：", "```", logTail.slice(-3000), "```");
    }
    lines.push(
      "",
      "请先读结果与失败点，判断是代码问题还是环境/契约问题；需要改代码就直接在工作区改并说明改了哪些文件，",
      "改完告诉我重跑哪一步；不需要改就给出结论与依据。",
    );
    return lines.join("\n");
  };
  const latestOk = job ? latestOkByAction(runs)[job.action] ?? null : null;
  const actionGate = job ? gates?.gates.find((g) => g.action === job.action) ?? null : null;
  // 交付预览已经按「当前规格 + operator-v1 代码快照 + 当前用例集 + 门禁 PASS」挑出
  // 每个动作的有效 run。执行历史里的红色失败必须保留审计，但不能让用户误以为
  // 已交付任务当前仍失败，所以单独标出“当前有效 / 历史失败”。
  const validEvidenceRunIds = new Set(
    Object.values(versions?.evidence.runs ?? {})
      .map((r) => r?.run_id)
      .filter((id): id is number => typeof id === "number"),
  );
  const historicalFailureCount = history.filter((h) => h.status === "failed").length;
  // 最新作业失败、但同动作已有成功 run：两个事实都摆出来，避免把「重跑失败」读成「动作坏了」。
  const showRunFallback = Boolean(
    job && !job.running && job.status !== "ok" && latestOk !== null,
  );

  return (
    <section className="card mb-4">
      <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">执行七步（开发、验证与交付）</h2>
        <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
          开发去「对话」；编译、冒烟、正确性、基准与采集先算 plan，再点「真执行」；开发完成前必须 build + smoke 通过
        </span>
      </div>
      <div className="p-4">
        <div className="flex flex-wrap gap-2">
          <button
            className="btn"
            onClick={() => onOpenChat?.(devPrompt())}
            title="去「对话」分区规划/开发（多轮确认也在那里）"
          >
            ① 开发
            <span className="ml-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>→ 对话</span>
          </button>
          {ACTIONS.map((a, i) => (
            <button key={a.key} className="btn" onClick={() => void compute(a.key)} disabled={busy !== null || job?.running}>
              {busy === a.key ? "计算中…" : `${i + 2} ${a.label}`}
              <span className="ml-1 mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                {a.key}
                {latest[a.key] ? ` · 最近执行 #${latest[a.key].id}` : ""}
              </span>
            </button>
          ))}
          <button
            className="btn"
            onClick={() => void startDelivery()}
            disabled={delivering || busy !== null || job?.running}
            title="生成交付包：校验用例已确认 + 当前版本证据有效 + 五件产物齐备，打包成可下载的 zip"
          >
            {delivering ? "打包中…" : "⑦ 交付"}
            <span className="ml-1 mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
              delivery
              {delivery ? ` · ${delivery.version}` : versions ? (versions.evidence.deliverable ? " · 可交付" : ` · 缺 ${versions.evidence.missing.length} 项`) : ""}
            </span>
          </button>
        </div>
        {Object.keys(latest).length > 0 && onOpenChat && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            <span>AI 审查（带上该步结果去「对话」）：</span>
            {ACTIONS.filter((a) => latest[a.key]).map((a) => (
              <button
                key={a.key}
                className="btn btn-sm"
                onClick={() =>
                  onOpenChat(
                    reviewPrompt(
                      a.key,
                      latest[a.key],
                      undefined,
                      undefined,
                      job && job.run_id === latest[a.key].id ? job.log : undefined,
                    ),
                  )
                }
                title="把 run / 门禁 / 报告入口带进输入框，确认后再发"
              >
                审查 {a.label}
              </button>
            ))}
          </div>
        )}

        {err && (
          <div className="mt-3 text-[11px]" style={{ color: "var(--bad)" }}>
            {err}
          </div>
        )}

        {/* ⑦ 交付：门禁缺项 / 生成结果 / 下载（本地动作，不占设备） */}
        {(versions || versionsErr || delivery || deliveryErr) && (
          <div className="mt-3" style={{ borderTop: "1px solid var(--border)", paddingTop: "12px" }}>
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="font-semibold">⑦ 交付</span>
              {versions && (
                <span className={`chip ${versions.evidence.deliverable ? "chip-ok" : "chip-warn"}`}>
                  {versions.evidence.deliverable ? "交付门禁齐备" : `交付门禁缺 ${versions.evidence.missing.length} 项`}
                </span>
              )}
              {versions?.current && (
                <span className="mono" style={{ color: "var(--fg-mute)" }}>
                  当前 v{versions.current.major}.{versions.current.minor}
                  {versions.current.delivered_at ? " · 已交付" : " · 未交付"}
                </span>
              )}
              {versionsErr && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  版本接口暂不可用：{versionsErr}
                </span>
              )}
            </div>

            {versions && versions.evidence.missing.length > 0 && (
              <ul className="mt-2 mb-0 pl-4 text-[11px]" style={{ color: "var(--bad)" }}>
                {versions.evidence.missing.map((m, i) => (
                  <li key={`${m.action ?? m.code ?? "missing"}-${i}`}>
                    <span className="mono">{m.label ?? m.action ?? m.code ?? "交付"}</span> · {m.reason}
                    <span style={{ color: "var(--fg-mute)" }}>
                      {" "}—— {(m.action ? MISSING_HINT[m.action] : undefined) ?? "去「概览」重跑对应动作 / 重新交付"}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {deliveryCodes.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {deliveryCodes.map((c) => (
                  <span key={c} className="chip chip-bad !py-0 !text-[10px] mono">{c}</span>
                ))}
              </div>
            )}

            {deliveryErr && (
              <div className="mt-2 text-[11px]" style={{ color: "var(--bad)" }}>
                交付失败：{deliveryErr}
              </div>
            )}

            {delivery && (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                <span className="chip chip-ok">已生成交付包 {delivery.version}</span>
                <a
                  className="btn btn-primary btn-sm"
                  href={api.downloadDeliveryUrl(delivery.run_id)}
                  target="_blank"
                  rel="noreferrer"
                >
                  下载交付包
                </a>
                <span className="mono" style={{ color: "var(--fg-mute)" }}>
                  {(delivery.sha256 ?? "").slice(0, 12) || "—"}… · {size(delivery.size_bytes)} · run #{delivery.run_id}
                </span>
              </div>
            )}
          </div>
        )}

        {plan && (
          <div className="mt-3" style={{ borderTop: "1px solid var(--border)", paddingTop: "12px" }}>
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className={`chip ${plan.ready ? "chip-ok" : "chip-bad"}`}>{plan.ready ? "前置齐备" : "前置不齐"}</span>
              <span style={{ color: "var(--fg-mute)" }}>动作 {plan.action}</span>
              <span style={{ color: "var(--fg-mute)" }}>超时 {plan.timeout_s}s</span>
              {plan.spec_version_id && (
                <span style={{ color: "var(--fg-mute)" }}>
                  规格 {plan.spec_version != null ? `v${plan.spec_version}` : `#${plan.spec_version_id}`}
                </span>
              )}
              {plan.host_ref && <span style={{ color: "var(--fg-mute)" }}>执行机 {plan.host_ref}</span>}
              <button
                className="btn btn-primary ml-auto"
                disabled={!plan.ready || !!busy || starting || job?.running}
                onClick={() => void start()}
              >
                {starting ? "启动中…" : "真执行"}
              </button>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              <span>判据 {plan.criteria_ref ?? "—"}{plan.criteria_source ? `（${plan.criteria_source}）` : ""}</span>
              <span>·</span>
              <span>
                快照 {plan.code_snapshot_sha256.slice(0, 12)}…（{plan.snapshot_files} 文件
                {plan.snapshot_truncated ? "，已截断" : ""}）
              </span>
              {plan.image_digest && (
                <>
                  <span>·</span>
                  <span>镜像 {plan.image_digest.slice(0, 24)}…</span>
                </>
              )}
            </div>
            {(plan.device_id || plan.lock_mode !== "none") && (
              <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                设备 {plan.device_id ?? "—"}（锁：{plan.lock_mode}
                {plan.lock_mode === "exclusive" ? "，基准/采集要独占" : ""}）
              </div>
            )}

            {plan.missing.length > 0 && (
              <ul className="mt-2 mb-0 pl-4 text-[11px]" style={{ color: "var(--bad)" }}>
                {plan.missing.map((m) => (
                  <li key={m}>缺：{m}</li>
                ))}
              </ul>
            )}
            {plan.warnings.map((w) => (
              <div key={w} className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                · {w}
              </div>
            ))}

            <ol className="mt-3 mb-0 pl-4 text-[11px] leading-[1.9]" style={{ color: "var(--fg-dim)" }}>
              {plan.steps.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ol>

            <details className="mt-3">
              <summary className="cursor-pointer text-[11px]" style={{ color: "var(--fg-mute)" }}>
                等价的手工命令（照抄也能跑）
              </summary>
              <div
                className="mt-2 overflow-auto rounded-[9px] p-3 text-[11px]"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace", whiteSpace: "pre-wrap" }}
              >
                {plan.command}
              </div>
            </details>
          </div>
        )}

        {job && (
          <div className="mt-4" style={{ borderTop: "1px solid var(--border)", paddingTop: "12px" }}>
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              {chip && <span className={chip.cls}>{chip.text}</span>}
              <span className="mono text-[10px]" style={{ color: "var(--fg-mute)" }}>
                job #{job.id} · {job.action} · {job.run_ref}
              </span>
              {job.device_id && <span style={{ color: "var(--fg-mute)" }}>设备 {job.device_id}</span>}
              {job.exit_code !== null && <span style={{ color: "var(--fg-mute)" }}>exit {job.exit_code}</span>}
              {job.run_id !== null && (
                <span style={{ color: "var(--fg-mute)" }}>run #{job.run_id}</span>
              )}
              {job.gate_verdict && (
                <span className={`chip ${job.gate_verdict === "pass" ? "chip-ok" : "chip-bad"}`}>
                  门禁 {job.gate_verdict}
                </span>
              )}
              {job.running && (
                <button className="btn ml-auto" onClick={() => void cancel()}>
                  取消
                </button>
              )}
              {!job.running && job.run_id !== null && onOpenChat && (
                <button
                  className="btn btn-sm ml-auto"
                  onClick={() => {
                    const run = runs.find((r) => r.id === job.run_id) ?? null;
                    onOpenChat(
                      run
                        ? reviewPrompt(job.action, run, job.gate_verdict)
                        : `请审查 ${opName || "本任务"} 的「${job.action}」执行结果（run #${job.run_id}，job #${job.id}，门禁 ${job.gate_verdict ?? "—"}）：先读 run 产物与下面的日志，判断是代码问题还是环境/契约问题；需要改代码就直接改并说明，改完告诉我重跑哪一步。\n\n作业日志尾部：\n\`\`\`\n${job.log.slice(-3000)}\n\`\`\``,
                    );
                  }}
                  title="把本次执行结果带进「对话」，让 agent 审查"
                >
                  AI 审查本次
                </button>
              )}
            </div>
            {showRunFallback && latestOk && (
              <div
                className="mt-2 flex flex-wrap items-center gap-2 rounded-[9px] px-3 py-2 text-[11px]"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)" }}
              >
                <span className="chip">历史 run #{latestOk.id} 成功</span>
                <span style={{ color: "var(--fg-dim)" }}>
                  本次作业未成功；该动作历史成功记录为 run #{latestOk.id}
                  {latestOk.ended_at ? `（${when(latestOk.ended_at)}）` : ""}
                  {actionGate?.run_id === latestOk.id && actionGate.verdict
                    ? `，门禁 ${actionGate.verdict}`
                    : ""}
                  。失败原因看下方日志；历史成功不代表当前代码已验证，请核对当前证据与门禁。
                </span>
              </div>
            )}
            {job.error && (
              <div className="mt-2 text-[11px]" style={{ color: "var(--bad)" }}>
                {job.error}
              </div>
            )}
            <pre
              ref={logRef}
              className="mt-2 mb-0 max-h-[320px] overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
              style={{
                background: "var(--panel-2)",
                border: "1px solid var(--border)",
                fontFamily: "ui-monospace, Menlo, monospace",
                whiteSpace: "pre-wrap",
              }}
            >
              {job.log || "（还没有输出）"}
            </pre>
            {!job.running && job.run_id !== null && (
              <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                结果已落进 runs：门禁结论见「阶段与门禁」或{" "}
                <span className="mono">GET /api/runs/{job.run_id}/gate</span>。
              </div>
            )}
          </div>
        )}

        {history.length > 1 && (
          <details className="mt-3">
            <summary className="cursor-pointer text-[11px]" style={{ color: "var(--fg-mute)" }}>
              执行历史（{history.length} 次
              {historicalFailureCount > 0 ? `，含 ${historicalFailureCount} 次历史失败` : ""}）
              · 历史失败保留审计，不代表当前版本失败
            </summary>
            {historicalFailureCount > 0 && versions?.evidence.deliverable && (
              <div
                className="mt-2 rounded-[8px] px-3 py-2 text-[11px]"
                style={{ background: "var(--panel-2)", border: "1px solid var(--border)", color: "var(--fg-dim)" }}
              >
                当前版本四项交付证据均有效。下表红色项是修复前或重试中的历史失败，保留用于追溯；
                以“当前有效”标记的 run 和上方交付门禁为当前结论。
              </div>
            )}
            <div className="mt-2 max-h-[220px] overflow-auto">
              <table>
                <thead>
                  <tr><th>job</th><th>动作</th><th>状态</th><th>run</th><th>证据角色</th><th>门禁</th><th>开始</th></tr>
                </thead>
                <tbody>
                  {history.map((h) => {
                    const isValidEvidence = h.run_id !== null && validEvidenceRunIds.has(h.run_id);
                    return (
                      <tr
                        key={h.id}
                        className={job?.running ? "" : "cursor-pointer"}
                        title={job?.running ? "当前作业运行中，结束后可回看历史" : "点击查看这次作业的日志"}
                        style={h.id === job?.id ? { background: "var(--panel-3)" } : undefined}
                        onClick={() => {
                          if (!job?.running) setJob(h);
                        }}
                      >
                        <td>#{h.id}</td>
                        <td className="mono">{h.action}</td>
                        <td>
                          <span className={`chip ${h.status === "ok" ? "chip-ok" : h.status === "failed" ? "chip-bad" : h.status === "interrupted" ? "chip-warn" : "chip"}`}>
                            {h.status}
                          </span>
                        </td>
                        <td>{h.run_id !== null ? `#${h.run_id}` : "—"}</td>
                        <td>
                          <span className={`chip !py-0 !text-[10px] ${isValidEvidence ? "chip-ok" : h.status === "failed" ? "chip-bad" : ""}`}>
                            {isValidEvidence ? "当前有效" : h.status === "failed" ? "历史失败" : h.running ? "当前执行" : "历史记录"}
                          </span>
                        </td>
                        <td>{h.gate_verdict ?? "—"}</td>
                        <td className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                          {h.started_at ? when(h.started_at) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </div>
    </section>
  );
}
