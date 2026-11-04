import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CaseLibrary, Project } from "../types";

/** `api.req` 抛的是 `${status} ${body}`，body 多半是 `{"detail": "…"}` —— 还原成人话。 */
function apiError(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  const sp = raw.indexOf(" ");
  if (sp < 0) return raw;
  try {
    const data = JSON.parse(raw.slice(sp + 1)) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
  } catch {
    /* 不是 JSON 就用原文 */
  }
  return raw;
}

const IMPORT_CMD = "python3 scripts/import_cases.py --zip ~/cases.zip";
const LIST_CMD = "python3 scripts/import_cases.py --list";

/** 案例库（M2）：读取外置 bundle 索引；一键建任务（复制 cases.json / 可选 workspace、ops）。
 *
 * 案例**不预置进仓库**：由用户用 `scripts/import_cases.py` 从 zip 导入到本机库目录。
 * 索引可能「损坏」（`index.json` 解析失败）或为空 —— 两种都要有明确空态，别留白页。
 */
export default function CasePanel({ onOpenTask }: { onOpenTask?: (id: number) => void }) {
  const [lib, setLib] = useState<CaseLibrary | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      // 项目列表失败不该让整个案例库变空白（一键发起仍可用默认项目）
      const [l, ps] = await Promise.all([
        api.caseLibrary(),
        api.projects().catch(() => [] as Project[]),
      ]);
      setLib(l);
      // 多项目是已知遗留（固定第一个项目）；这里保持既有口径
      setProject(ps[0] ?? null);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const start = async (bundle: string, op: string) => {
    setBusy(`${bundle}/${op}`);
    setErr(null);
    setMsg(null);
    try {
      const r = await api.startCase(bundle, op, project?.id ?? null);
      setMsg(`已创建任务 #${r.task_id}（${r.title}）· 规格/定版需到任务页完成`);
      onOpenTask?.(r.task_id);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setBusy(null);
    }
  };

  const refreshBtn = (
    <button className="btn btn-sm" disabled={loading || busy !== null} onClick={() => void load()}>
      {loading ? "刷新中…" : "刷新"}
    </button>
  );

  return (
    <div className="grid gap-4" style={{ maxWidth: 980 }}>
      <section className="card p-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="m-0 text-sm font-semibold">案例库</h2>
          <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
            外置 bundle · 仓库不预置案例 · 导入后一键建任务
          </span>
          <span className="mono ml-auto text-[10px]" style={{ color: "var(--fg-mute)" }}>
            {lib?.dest ?? ""}
          </span>
          {refreshBtn}
        </div>
        <p className="mt-2 mb-0 text-[11px] leading-[1.7]" style={{ color: "var(--fg-dim)" }}>
          导入：<span className="mono">{IMPORT_CMD}</span>
          ｜查看：<span className="mono">{LIST_CMD}</span>
        </p>
      </section>

      {loading && !lib && (
        <section className="card p-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          正在读取案例库索引…
        </section>
      )}

      {err && (
        <section className="card p-3 text-[11px]" style={{ color: "var(--bad)" }}>
          {err}
        </section>
      )}

      {/* 索引读不了（index.json 损坏 / 权限）：明确报错，别让页面看起来像空库 */}
      {lib?.error && (
        <section className="card p-4" style={{ borderColor: "var(--bad)" }}>
          <div className="text-xs font-semibold" style={{ color: "var(--bad)" }}>
            案例索引读取失败
          </div>
          <div className="mt-1 text-[11px] leading-[1.7]" style={{ color: "var(--fg-dim)" }}>
            <span className="mono">{lib.dest}/index.json</span> 损坏或无法解析：
            <span className="mono"> {lib.error}</span>
          </div>
          <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            重新导入可修复（会覆盖同名 bundle）：<span className="mono">{IMPORT_CMD} --force</span>
          </div>
          <div className="mt-2">{refreshBtn}</div>
        </section>
      )}

      {msg && (
        <section className="card p-3 text-[11px]" style={{ color: "var(--ok)" }}>
          {msg}
        </section>
      )}

      {/* 没导入过：给出导入命令 + 刷新入口 */}
      {lib && !lib.imported && (
        <section className="card p-4">
          <div className="text-xs font-semibold">还没有导入任何案例包</div>
          <div className="mt-1 text-[11px] leading-[1.7]" style={{ color: "var(--fg-dim)" }}>
            本机库目录 <span className="mono">{lib.dest}</span> 里没有 <span className="mono">index.json</span>。
            案例不预置进仓库，需要先从 zip 导入。
          </div>
          <pre
            className="mt-2 overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
            style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace" }}
          >
            {`${IMPORT_CMD}\n${LIST_CMD}`}
          </pre>
          <div className="mt-2">{refreshBtn}</div>
        </section>
      )}

      {/* 索引在、但一个 bundle 都没有：可能是导入中断/手工清空 */}
      {lib && lib.imported && !lib.error && lib.bundles.length === 0 && (
        <section className="card p-4">
          <div className="text-xs font-semibold">索引里没有案例</div>
          <div className="mt-1 text-[11px] leading-[1.7]" style={{ color: "var(--fg-dim)" }}>
            <span className="mono">{lib.dest}/index.json</span> 是空的（或导入中断没写全）。重新导入后点「刷新」。
          </div>
          <pre
            className="mt-2 overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
            style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace" }}
          >
            {`${IMPORT_CMD} --force\n${LIST_CMD}`}
          </pre>
          <div className="mt-2">{refreshBtn}</div>
        </section>
      )}

      {lib?.bundles.map((b) => (
        <section key={b.name} className="card overflow-hidden">
          <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="text-xs font-semibold">{b.name}</div>
            {b.version && <span className="chip">v{b.version}</span>}
            <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
              {b.cases.length} 个案例
            </span>
          </div>
          {b.cases.length === 0 ? (
            <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              这个 bundle 里没有案例。
            </div>
          ) : (
            <div className="overflow-auto">
              <table>
                <thead>
                  <tr>
                    <th>算子</th>
                    <th>场景</th>
                    <th>目标芯片</th>
                    <th>内容</th>
                    <th>状态</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {b.cases.map((c) => (
                    <tr key={`${b.name}/${c.op}`}>
                      <td className="mono">{c.op}</td>
                      <td>{c.scenario ?? "—"}</td>
                      <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                        {c.target_chip ? (
                          <span className="chip" title="一键发起时写入任务的目标芯片">
                            {c.target_chip}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                        cases.json
                        {c.has_workspace ? " · workspace/" : ""}
                        {c.has_ops ? " · ops/" : ""}
                      </td>
                      <td>
                        {c.exists ? (
                          <span className="chip chip-ok">已导入</span>
                        ) : (
                          <span className="chip chip-bad">目录缺失</span>
                        )}
                      </td>
                      <td>
                        <button
                          className="btn btn-primary btn-sm"
                          disabled={!c.exists || busy !== null}
                          onClick={() => void start(b.name, c.op)}
                        >
                          {busy === `${b.name}/${c.op}` ? "创建中…" : "一键发起"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ))}

      <details className="card p-4">
        <summary className="cursor-pointer text-xs font-semibold">Bundle 结构（点开）</summary>
        <pre
          className="mt-2 overflow-auto rounded-[9px] p-3 text-[11px] leading-[1.6]"
          style={{ background: "var(--panel-2)", border: "1px solid var(--border)", fontFamily: "ui-monospace, Menlo, monospace" }}
        >{`my-cases.zip
├── manifest.json            # schema/name/version/cases[]
└── cases/
    └── relu/
        ├── cases.json       # 必需：用例矩阵
        ├── workspace/       # 可选：复制到工作区根（实现/脚本）
        └── ops/             # 可选：复制到 ops/<op>/`}</pre>
        <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          一键发起会拒绝覆盖已存在的目标（cases/&lt;op&gt;、workspace 文件、ops/&lt;op&gt;）；只建任务与工作区文件，
          规格/定版仍需到任务页完成。
        </div>
      </details>
    </div>
  );
}
