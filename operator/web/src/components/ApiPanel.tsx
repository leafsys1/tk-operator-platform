import { useCallback, useEffect, useState } from "react";
import { api, when } from "../lib/api";
import type { EndpointConf, EndpointProbeResult } from "../types";

/**
 * API 管理（P4 / 方案 §7 边界②）：模型端点的**配置态**（用哪个模型 / 哪个 key / 什么价）。
 *
 * - key 可以在页面直接配置，存**本机密钥库**（macOS Keychain / Linux Secret Service / 0600 文件回退），
 *   DB 只存引用；接口只回 `has_key`，**不回显明文**。
 * - 没有配置 key 时，仍回退 `key_ref` 指向的环境变量（兼容旧部署）。
 * - 当前端点（`is_active`）的 key 会同步到进程环境，供 dsh 的 `apiKeyEnv` 读取。
 * - 探针 = `GET {base_url}/models`（OpenAI 兼容，带 Bearer）→ 通不通 + 有哪些模型；不产生生成费用。
 * - **单价（D2）**：人民币元 / 每百万 token；`price_in` + `price_out` 都配了才算「已定价」，
 *   消耗情况页才按 token 算金额。cache 两项留空 = 不单独计价（按 0 计）。
 */

const PROVIDERS = ["dsh-llm-pi-ai", "dsh-llm-deepseek"];

/** key 存储后端的中文名（来自 GET /api/endpoints 的 key_backend） */
const KEY_BACKEND_LABEL: Record<string, string> = {
  keychain: "Keychain",
  "secret-service": "Secret Service",
  file: "本机文件",
  legacy: "旧库明文（待迁移）",
};

const backendLabel = (b: string | null | undefined): string =>
  (b && KEY_BACKEND_LABEL[b]) || b || "本机密钥库";

/** 单价输入框：中文标签（顺序 = 输入 / 输出 / 缓存读 / 缓存写）；后端上限 1e6 ¥/M。 */
const PRICE_FIELDS = [
  { key: "price_in", label: "输入 ¥/M", hint: "必填 · ≤1e6" },
  { key: "price_out", label: "输出 ¥/M", hint: "必填 · ≤1e6" },
  { key: "price_cache_read", label: "缓存读 ¥/M", hint: "留空按 0" },
  { key: "price_cache_write", label: "缓存写 ¥/M", hint: "留空按 0" },
] as const;

type Form = {
  name: string;
  provider: string;
  base_url: string;
  model: string;
  key_ref: string;
  api_key: string;
  max_model_len: string;
  note: string;
  price_in: string;
  price_out: string;
  price_cache_read: string;
  price_cache_write: string;
};

const EMPTY: Form = {
  name: "", provider: PROVIDERS[0], base_url: "", model: "", key_ref: "OPX_LLM_KEY", api_key: "",
  max_model_len: "", note: "", price_in: "", price_out: "", price_cache_read: "", price_cache_write: "",
};

/** 单价格式：空 = null（后端按「未定价 / 不单独计价」处理），否则转 number。 */
const priceOf = (s: string): number | null => (s.trim() === "" ? null : Number(s));

const priceText = (r: EndpointConf): string => {
  const fmt = (v: number | null) => (v == null ? "未填" : String(v));
  const parts = [`in ${fmt(r.price_in)}`, `out ${fmt(r.price_out)}`];
  if (r.price_cache_read != null) parts.push(`缓存读 ${r.price_cache_read}`);
  if (r.price_cache_write != null) parts.push(`缓存写 ${r.price_cache_write}`);
  return parts.join(" · ");
};

function probeTone(s: string) {
  return s === "ok" ? "var(--ok)" : s === "fail" ? "var(--bad)" : "var(--fg-mute)";
}

export default function ApiPanel() {
  const [rows, setRows] = useState<EndpointConf[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<Form>(EMPTY);
  const [probes, setProbes] = useState<Record<number, EndpointProbeResult["probe"]>>({});
  const [editId, setEditId] = useState<number | null>(null);
  const [edit, setEdit] = useState<Form>(EMPTY);
  const [keyEditId, setKeyEditId] = useState<number | null>(null);
  const [keyValue, setKeyValue] = useState("");

  const load = useCallback(async () => {
    try {
      setRows(await api.endpoints());
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const create = async () => {
    setBusy(true);
    try {
      await api.createEndpoint({
        name: form.name,
        provider: form.provider,
        base_url: form.base_url || null,
        model: form.model || null,
        key_ref: form.key_ref || null,
        api_key: form.api_key.trim() || undefined,
        max_model_len: form.max_model_len ? Number(form.max_model_len) : null,
        note: form.note || null,
        price_in: priceOf(form.price_in),
        price_out: priceOf(form.price_out),
        price_cache_read: priceOf(form.price_cache_read),
        price_cache_write: priceOf(form.price_cache_write),
      });
      setForm(EMPTY);
      setShowForm(false);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (r: EndpointConf) => {
    setEditId(r.id);
    setEdit({
      name: r.name, provider: r.provider, base_url: r.base_url ?? "", model: r.model ?? "",
      key_ref: r.key_ref ?? "", api_key: "",
      max_model_len: r.max_model_len != null ? String(r.max_model_len) : "", note: r.note ?? "",
      price_in: r.price_in != null ? String(r.price_in) : "",
      price_out: r.price_out != null ? String(r.price_out) : "",
      price_cache_read: r.price_cache_read != null ? String(r.price_cache_read) : "",
      price_cache_write: r.price_cache_write != null ? String(r.price_cache_write) : "",
    });
  };

  const saveEdit = async () => {
    if (editId === null) return;
    setBusy(true);
    try {
      await api.patchEndpoint(editId, {
        name: edit.name,
        provider: edit.provider,
        base_url: edit.base_url || null,
        model: edit.model || null,
        key_ref: edit.key_ref || null,
        max_model_len: edit.max_model_len ? Number(edit.max_model_len) : null,
        note: edit.note || null,
        price_in: priceOf(edit.price_in),
        price_out: priceOf(edit.price_out),
        price_cache_read: priceOf(edit.price_cache_read),
        price_cache_write: priceOf(edit.price_cache_write),
      });
      setEditId(null);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const activate = async (id: number) => {
    setBusy(true);
    try {
      await api.activateEndpoint(id);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const saveKey = async (id: number, value: string) => {
    const next = value.trim();
    // 留空保存 = 清空该端点的 key（之后回退 key_ref 环境变量）：属于不可逆操作，二次确认。
    if (next === "" && !window.confirm("留空保存 = 清空该端点的 key（之后回退 key_ref 环境变量）。\n\n清空后若运行态依赖这个 key，新的 agent 调用会失败。确定清空？")) {
      return;
    }
    setBusy(true);
    try {
      await api.patchEndpoint(id, { api_key: next });
      setKeyEditId(null);
      setKeyValue("");
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: number) => {
    const row = rows.find((r) => r.id === id);
    const active = row?.is_active
      ? "· 这是当前 API：删除后会一并移除它在控制台进程里的运行态 key，之后新的 agent 调用可能失败，直到切换端点或重新配置 key。\n"
      : "";
    const ok = window.confirm(
      `删除端点 #${id}${row ? `（${row.name}）` : ""}？\n\n` +
        active +
        "· 历史调用若没有端点快照，金额会改按「当前端点」的价目估算 —— 历史费用口径会变。\n" +
        "· 此操作不可恢复。",
    );
    if (!ok) return;
    setBusy(true);
    try {
      await api.deleteEndpoint(id);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const probe = async (id: number) => {
    setBusy(true);
    try {
      const res = await api.probeEndpoint(id);
      setProbes((p) => ({ ...p, [id]: res.probe }));
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const probeAll = async () => {
    setBusy(true);
    try {
      const res = await api.probeAllEndpoints();
      setProbes(Object.fromEntries(res.results.map((r) => [r.endpoint_id, r.probe])));
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const inputCls = "rounded-[8px] px-2 py-1 text-[11px]";
  const inputStyle = { background: "var(--panel-2)", border: "1px solid var(--border-strong)", color: "var(--fg)" };

  const field = (label: string, key: keyof Form, placeholder = "", type = "text") => (
    <label className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
      {label}
      <input
        className={inputCls}
        style={inputStyle}
        type={type}
        placeholder={placeholder}
        value={form[key]}
        onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
      />
    </label>
  );

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px]">
        <span style={{ color: "var(--fg-dim)" }}>共 {rows.length} 个端点</span>
        <button className="btn btn-sm ml-auto" onClick={() => void probeAll()} disabled={busy || rows.length === 0}>全部探针</button>
        <button className="btn btn-primary btn-sm" onClick={() => setShowForm((v) => !v)}>{showForm ? "收起" : "新增端点"}</button>
      </div>

      {err && <div className="card mb-4 px-4 py-3 text-xs" style={{ color: "var(--bad)" }}>{err}</div>}

      {showForm && (
        <section className="card mb-4 p-4">
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(0,1fr))" }}>
            {field("名称 *", "name", "例如 opencode-go / dsh headless")}
            <label className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
              provider
              <select className={inputCls} style={inputStyle} value={form.provider} onChange={(e) => setForm((f) => ({ ...f, provider: e.target.value }))}>
                {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            {field("base_url", "base_url", "https://…/v1")}
            {field("model", "model", "deepseek-v4.1-flash")}
            {field("key_ref（没配 key 时回退它）", "key_ref", "OPX_LLM_KEY")}
            {field("API key（可留空，存本机库）", "api_key", "sk-…", "password")}
            {field("max_model_len（正整数）", "max_model_len", "262144", "number")}
            {PRICE_FIELDS.map((f) => field(f.label, f.key, f.hint, "number"))}
          </div>
          <div className="mt-3 flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <label className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                备注
                <input className={inputCls} style={inputStyle} value={form.note} onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))} />
              </label>
            </div>
            <button className="btn btn-primary btn-sm self-end" onClick={() => void create()} disabled={busy || !form.name.trim()}>登记</button>
          </div>
          <div className="mt-2 text-[11px]" style={{ color: "var(--fg-mute)" }}>
            key 存**本机密钥库**（macOS Keychain / Linux Secret Service / 0600 文件回退），DB 只存引用；
            接口不回显；留空则回退 `key_ref` 环境变量。
            当前 API 的 key 会同步到控制台进程环境，供 dsh 调用。
            <br />
            单价 = <b>人民币元 / 每百万 token</b>，范围 0–1e6（超过会被后端拒绝）：填「输入 + 输出」才算已定价
            （消耗页才按 token 算金额）；缓存读/写留空 = 不单独计价（按 0 计）。改单价后历史消耗实时重算。
            <br />
            max_model_len 填正整数（如 262144）；不确定可留空。
          </div>
        </section>
      )}

      <section className="card overflow-hidden">
        <div className="overflow-auto">
          <table>
            <thead>
              <tr><th>名称</th><th>provider</th><th>base_url</th><th>model</th><th>key_ref</th><th>key</th><th>单价（¥/M）</th><th>状态</th><th>探针结果</th><th>操作</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const editing = editId === r.id;
                const p = probes[r.id];
                return (
                  <tr key={r.id}>
                    <td>
                      {editing ? (
                        <input className={inputCls} style={inputStyle} value={edit.name} onChange={(e) => setEdit((s) => ({ ...s, name: e.target.value }))} />
                      ) : (
                        <span className="font-semibold">{r.name}</span>
                      )}
                      {r.is_active && <span className="chip chip-ok ml-1">当前 API</span>}
                      {r.note && !editing && <div className="text-[10px]" style={{ color: "var(--fg-mute)", maxWidth: 240 }}>{r.note}</div>}
                    </td>
                    <td className="mono text-[11px]">
                      {editing ? (
                        <select className={inputCls} style={inputStyle} value={edit.provider} onChange={(e) => setEdit((s) => ({ ...s, provider: e.target.value }))}>
                          {PROVIDERS.map((x) => <option key={x} value={x}>{x}</option>)}
                        </select>
                      ) : r.provider}
                    </td>
                    <td className="mono text-[11px]">
                      {editing ? (
                        <input className={inputCls} style={inputStyle} value={edit.base_url} onChange={(e) => setEdit((s) => ({ ...s, base_url: e.target.value }))} />
                      ) : r.base_url}
                    </td>
                    <td className="mono text-[11px]">
                      {editing ? (
                        <input className={inputCls} style={inputStyle} value={edit.model} onChange={(e) => setEdit((s) => ({ ...s, model: e.target.value }))} />
                      ) : r.model}
                    </td>
                    <td className="mono text-[11px]">
                      {editing ? (
                        <input className={inputCls} style={inputStyle} value={edit.key_ref} onChange={(e) => setEdit((s) => ({ ...s, key_ref: e.target.value }))} />
                      ) : r.key_ref}
                    </td>
                    <td className="text-[11px]" style={{ minWidth: 150 }}>
                      {keyEditId === r.id ? (
                        <div className="flex flex-wrap items-center gap-1">
                          <input
                            className={inputCls}
                            style={inputStyle}
                            type="password"
                            autoFocus
                            placeholder="sk-…（留空保存=清空）"
                            value={keyValue}
                            onChange={(e) => setKeyValue(e.target.value)}
                          />
                          <button className="btn btn-sm btn-primary" onClick={() => void saveKey(r.id, keyValue)} disabled={busy}>保存</button>
                          <button className="btn btn-sm" onClick={() => { setKeyEditId(null); setKeyValue(""); }} disabled={busy}>取消</button>
                        </div>
                      ) : (
                        <span
                          className={`chip ${r.has_key ? "chip-ok" : "chip-warn"}`}
                          title={
                            r.has_key
                              ? `key 存在「${backendLabel(r.key_backend)}」，DB 只存引用、接口不回显`
                              : "尚未配置 key（会回退 key_ref 环境变量 / 启动时的 OPX_LLM_KEY）"
                          }
                        >
                          {r.has_key ? `已配置 · ${backendLabel(r.key_backend)}` : "未配置"}
                        </span>
                      )}
                    </td>
                    <td className="text-[11px]" style={{ minWidth: 210 }}>
                      {editing ? (
                        <div className="grid grid-cols-2 gap-1">
                          {PRICE_FIELDS.map((f) => (
                            <input
                              key={f.key}
                              className={inputCls}
                              style={inputStyle}
                              type="number"
                              step="any"
                              min="0"
                              placeholder={f.label}
                              value={edit[f.key]}
                              onChange={(e) => setEdit((s) => ({ ...s, [f.key]: e.target.value }))}
                            />
                          ))}
                        </div>
                      ) : (
                        <div className="flex flex-wrap items-center gap-1">
                          <span className="mono" style={{ color: "var(--fg-dim)" }}>{priceText(r)}</span>
                          {!r.priced && (
                            <span className="chip chip-warn" title="输入 + 输出单价都填才算已定价；未定价时「消耗情况」显示「未定价」">
                              未定价
                            </span>
                          )}
                        </div>
                      )}
                    </td>
                    <td>
                      <span className="chip" style={{ color: probeTone(r.probe_state) }}>{r.probe_state}</span>
                      {r.last_probe_at && <div className="text-[10px]" style={{ color: "var(--fg-mute)" }}>{when(r.last_probe_at)}</div>}
                    </td>
                    <td className="text-[11px]" style={{ maxWidth: 300 }}>
                      {p ? (
                        p.ok ? (
                          <span style={{ color: "var(--ok)" }}>
                            {p.model_count ?? 0} 个模型 · {p.latency_ms}ms
                            <div className="text-[10px]" style={{ color: "var(--fg-mute)" }}>{(p.models ?? []).slice(0, 6).join(", ")}{(p.models?.length ?? 0) > 6 ? " …" : ""}</div>
                          </span>
                        ) : (
                          <span style={{ color: "var(--bad)" }}>{p.error}</span>
                        )
                      ) : (
                        <span style={{ color: "var(--fg-mute)" }}>—</span>
                      )}
                    </td>
                    <td>
                      <div className="flex flex-wrap gap-1">
                        <button className="btn btn-sm" onClick={() => void probe(r.id)} disabled={busy}>探针</button>
                        {editing ? (
                          <>
                            <button className="btn btn-sm btn-primary" onClick={() => void saveEdit()} disabled={busy}>保存</button>
                            <button className="btn btn-sm" onClick={() => setEditId(null)} disabled={busy}>取消</button>
                          </>
                        ) : (
                          <>
                            <button className="btn btn-sm" onClick={() => { setKeyEditId(r.id); setKeyValue(""); }} disabled={busy}>配置 key</button>
                            {!r.is_active && (
                              <button className="btn btn-sm" onClick={() => void activate(r.id)} disabled={busy}>设为当前</button>
                            )}
                            <button className="btn btn-sm" onClick={() => startEdit(r)} disabled={busy}>编辑</button>
                            <button className="btn btn-sm" onClick={() => void remove(r.id)} disabled={busy}>删除</button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
              {rows.length === 0 && (
                <tr><td colSpan={10} style={{ color: "var(--fg-mute)" }}>还没有端点 —— 点「新增端点」登记一个。</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="mt-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
        · API 管理是配置态（用哪个模型/key/价），「消耗情况」是运行态 —— 两个入口、一套数据。
        · key 存**本机密钥库**（macOS Keychain / Linux Secret Service / 0600 文件回退），DB 只存引用、接口不回显；
        「当前 API」的 key 会同步到进程环境供 dsh 使用。
        · 探针只验连通性，不产生生成费用；没有配置 key 时回退 `key_ref` 环境变量。
        · **模型路由跟着「当前端点」走**：dsh 的 `--patch` 按当前端点的 provider/base_url/model 动态生成
        （`backend/.runtime/dsh-llm.patch.yml`），切换「当前」即切换 agent 实际使用的模型/网关；改 key/端点无需重启。
        没有端点时回退静态 `ops/dsh/llm-opencode-go.patch.yml`。
        · 单价人民币元 / 每百万 token，范围 0–1e6：输入 + 输出都填才算已定价；缓存两项留空 = 不单独计价（按 0 计）。
        · max_model_len 为正整数（如 262144）。
        费用不落库，改单价后「消耗情况」与任务详情的金额实时重算（早期调用按当前价目估算并标注）。
      </div>
    </>
  );
}
