import { useEffect, useState } from "react";
import { api, size } from "../lib/api";
import type { ContextSpace } from "../types";

/** 任务上下文空间（P0-3 定版时生成）：只指向既定真相源，不复制 cases/criteria/ops。 */
export default function ContextPanel({ taskId }: { taskId: number }) {
  const [doc, setDoc] = useState<ContextSpace | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.context(taskId).then((d) => alive && setDoc(d)).catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [taskId]);

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <h2 className="m-0 text-xs font-semibold">任务上下文空间</h2>
        {doc?.exists && (
          <span className="ml-auto text-[10px]" style={{ color: "var(--fg-mute)" }}>{doc.files.length} 个文件</span>
        )}
      </div>
      {err && <div className="px-4 py-3 text-[11px]" style={{ color: "var(--bad)" }}>{err}</div>}
      {!doc && !err && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>读取上下文空间…</div>
      )}
      {doc && !doc.exists && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          还没有上下文空间 —— 在「任务开始」页完成定版后生成。
        </div>
      )}
      {doc?.exists && doc.files.length === 0 && (
        <div className="px-4 py-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          上下文空间存在，但当前没有登记文件。
        </div>
      )}
      {doc?.exists && doc.files.length > 0 && (
        <>
          <div className="max-h-[300px] overflow-auto">
            {doc.files.map((f) => (
              <div
                key={f.path}
                className="px-4 py-2"
                style={{ borderTop: "1px solid var(--border)" }}
              >
                <div className="mono truncate text-[11px]" title={f.path}>{f.path}</div>
                <div className="mt-0.5 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  {size(f.size_bytes)} · sha {(f.sha256 ?? "").slice(0, 10) || "—"}
                </div>
              </div>
            ))}
          </div>
          <div className="mono truncate px-4 py-2 text-[10px]" style={{ color: "var(--fg-mute)", borderTop: "1px solid var(--border)" }} title={doc.space_dir}>
            {doc.space_dir}
          </div>
        </>
      )}
    </section>
  );
}
