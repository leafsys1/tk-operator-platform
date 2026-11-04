import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Host, HostImage, Project, TaskDetail } from "../types";

/** 任务级执行配置覆盖（空 = 用项目默认）：执行机 / 镜像 / 设备 / 超时 / D8 预算。 */
export default function TaskConfigPanel({ task, onChanged }: { task: TaskDetail; onChanged?: () => void }) {
  const [title, setTitle] = useState(task.title);
  const [opName, setOpName] = useState(task.op_name ?? "");
  const [chip, setChip] = useState(task.target_chip ?? "");
  const [criteria, setCriteria] = useState(task.criteria_ref ?? "");
  const [hosts, setHosts] = useState<Host[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<HostImage[]>([]);
  const [imagesLoaded, setImagesLoaded] = useState(false);
  const [hostId, setHostId] = useState<number | "">(task.host_id ?? "");
  const [imageId, setImageId] = useState<number | "">(task.image_id ?? "");
  const [device, setDevice] = useState(task.device_id ?? "");
  const [timeout, setTimeoutS] = useState(task.timeout_s ? String(task.timeout_s) : "");
  const [fixRounds, setFixRounds] = useState(task.fix_rounds_max ? String(task.fix_rounds_max) : "");
  const [budget, setBudget] = useState(task.budget_per_action ? String(task.budget_per_action) : "");
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([api.hosts(), api.projects()]).then(([hs, ps]) => {
      setHosts(hs);
      setProject(ps.find((p) => p.id === task.project_id) ?? null);
    }).catch(() => {});
  }, [task.project_id]);

  const testHost = hosts.find((h) => h.is_test_server) ?? null;
  // 与后端 build_plan 同口径：任务显式 > 当前测试服务器 > 项目默认
  const effectiveHost = hostId === "" ? testHost?.id ?? project?.host_id ?? null : Number(hostId);
  useEffect(() => {
    let alive = true;
    if (effectiveHost === null) {
      setImages([]);
      setImagesLoaded(true);
      return () => {
        alive = false;
      };
    }
    setImagesLoaded(false);
    void api
      .images(effectiveHost)
      .then((list) => {
        if (!alive) return;
        setImages(list);
        setImagesLoaded(true);
      })
      .catch(() => {
        if (!alive) return;
        setImages([]);
        setImagesLoaded(true);
      });
    return () => {
      alive = false;
    };
  }, [effectiveHost]);

  // 任务选镜像只暴露「测试镜像」；已选中的镜像即使未标记也保留，避免下拉变空
  const selectableImages = images.filter((i) => i.is_test || i.id === Number(imageId));
  const hiddenImageCount = images.filter((i) => !i.is_test && i.id !== Number(imageId)).length;

  // 覆盖执行机但镜像留空 = 回退项目默认镜像；项目默认镜像未必在所选机器上（plan 阶段才会报缺项）。
  const overrideWarn = (() => {
    if (hostId === "" || imageId !== "") return null;
    const hostName = hosts.find((h) => h.id === Number(hostId))?.name ?? `#${hostId}`;
    if (project === null) {
      return `已覆盖执行机「${hostName}」但镜像留空：将回退项目默认镜像（尚未读到项目配置），请确认它有默认镜像且在该执行机上，否则执行会报缺项。`;
    }
    if (project.image_id == null) {
      return `已覆盖执行机「${hostName}」但镜像留空：项目没有默认镜像，执行时会因缺少镜像而失败。请显式选择该机器上的镜像。`;
    }
    if (imagesLoaded && !images.some((i) => i.id === project.image_id)) {
      return `已覆盖执行机「${hostName}」但镜像留空：将回退项目默认镜像 #${project.image_id}，而它不在该执行机上，执行时会报缺项。请显式选择该机器上的镜像。`;
    }
    return `已覆盖执行机「${hostName}」但镜像留空：将回退项目默认镜像 #${project.image_id}，请确认它在该执行机上（不在的话执行会报缺项）。`;
  })();

  const numOk = (v: string) => v.trim() === "" || (Number.isFinite(Number(v)) && Number(v) >= 0);

  const save = async () => {
    const bad = ([
      ["单次超时", timeout],
      ["受控修复轮次上限", fixRounds],
      ["每动作预算", budget],
    ] as const).find(([, v]) => !numOk(v));
    if (bad) {
      setMsg(`${bad[0]}必须是不小于 0 的数字`);
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      await api.patchTask(task.id, {
        title: title.trim() || task.title,
        op_name: opName.trim() || null,
        target_chip: chip.trim() || null,
        criteria_ref: criteria.trim() || null,
        host_id: hostId === "" ? null : Number(hostId),
        image_id: imageId === "" ? null : Number(imageId),
        device_id: device.trim() || null,
        timeout_s: timeout ? Number(timeout) : null,
        fix_rounds_max: fixRounds ? Number(fixRounds) : null,
        budget_per_action: budget ? Number(budget) : null,
      });
      setMsg("已保存（空 = 用项目默认）");
      onChanged?.();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  const num = (v: string) => v.trim();
  const hostLabel =
    hostId === ""
      ? testHost
        ? `${testHost.name}（当前测试）`
        : "项目默认"
      : hosts.find((h) => h.id === Number(hostId))?.name ?? `#${hostId}`;
  const selectedImage = images.find((i) => i.id === Number(imageId));
  const imageLabel =
    imageId === ""
      ? project?.image_id
        ? `项目默认 #${project.image_id}`
        : "项目默认"
      : selectedImage
        ? `${selectedImage.image_ref}${selectedImage.tag ? `:${selectedImage.tag}` : ""}`
        : `#${imageId}`;

  return (
    <section className="card mb-4 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        <h2 className="m-0 text-xs font-semibold">任务与执行配置</h2>
        <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
          空 = 当前测试服务器 / 项目默认（{project ? `host #${project.host_id ?? "—"} / image #${project.image_id ?? "—"}` : "—"}）
        </span>
        <button className="btn btn-sm ml-auto" onClick={() => setOpen((v) => !v)}>
          {open ? "收起" : "编辑"}
        </button>
        {open && (
          <button className="btn btn-primary btn-sm" onClick={() => void save()} disabled={busy}>
            {busy ? "保存中…" : "保存"}
          </button>
        )}
      </div>

      {!open && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 px-4 pb-3 text-[11px]" style={{ color: "var(--fg-mute)" }}>
          {(
            [
              ["标题", title || "—"],
              ["算子", opName || "—"],
              ["芯片", chip || "—"],
              ["判据", criteria || "项目默认"],
              ["执行机", hostLabel],
              ["镜像", imageLabel],
              ["设备", device || "自动选空卡"],
              ["单次超时", timeout ? `${timeout}s` : "默认"],
            ] as const
          ).map(([k, v]) => (
            <span key={k}>
              {k} <span style={{ color: "var(--fg)" }}>{v}</span>
            </span>
          ))}
        </div>
      )}

      {open && (
        <div className="px-4 pb-4 pt-3" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="grid gap-2 text-[11px] md:grid-cols-4">
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>任务标题</span>
              <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>算子名</span>
              <input className="input" value={opName} onChange={(e) => setOpName(e.target.value)} placeholder="如 elementwise_add" />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>目标芯片</span>
              <input className="input" value={chip} onChange={(e) => setChip(e.target.value)} placeholder="如 910B1" />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>判据档（空 = 项目默认）</span>
              <input className="input" value={criteria} onChange={(e) => setCriteria(e.target.value)} placeholder="ops-precision-standard@1" />
            </label>
          </div>
          <div className="mt-3 grid gap-2 text-[11px] md:grid-cols-3">
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>执行机</span>
              <select className="input" value={hostId} onChange={(e) => { setHostId(e.target.value === "" ? "" : Number(e.target.value)); setImageId(""); }}>
                <option value="">{testHost ? `当前测试服务器：${testHost.name}` : "项目默认"}</option>
                {hosts.map((h) => <option key={h.id} value={h.id}>{h.name}{h.ssh_alias ? `（${h.ssh_alias}）` : ""}{h.is_test_server ? " · 当前测试" : ""}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>镜像（只显示测试镜像）</span>
              <select className="input" value={imageId} onChange={(e) => setImageId(e.target.value === "" ? "" : Number(e.target.value))}>
                <option value="">{project?.image_id ? `项目默认镜像 #${project.image_id}` : "项目默认"}</option>
                {imageId !== "" && !images.some((i) => i.id === Number(imageId)) && (
                  <option value={imageId}>#{imageId}（当前任务镜像，不在该执行机）</option>
                )}
                {selectableImages.map((img) => (
                  <option key={img.id} value={img.id}>
                    #{img.id} {img.image_ref}{img.tag ? `:${img.tag}` : ""}{img.is_op_dev ? "（算子开发）" : ""}{img.is_test ? "" : "（未标记测试）"}
                  </option>
                ))}
              </select>
              {hiddenImageCount > 0 && (
                <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                  另有 {hiddenImageCount} 个未标记「测试镜像」的镜像已隐藏；去「服务器与镜像」标记后可选。
                </span>
              )}
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>设备（空 = 自动选空卡）</span>
              <input className="input" value={device} onChange={(e) => setDevice(e.target.value)} placeholder="如 3" />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>单次超时（秒）</span>
              <input className="input" type="number" value={timeout} onChange={(e) => setTimeoutS(num(e.target.value))} placeholder="项目/默认 1800" />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>受控修复轮次上限（D8 未实装）</span>
              <input className="input" type="number" min={0} value={fixRounds} onChange={(e) => setFixRounds(num(e.target.value))} placeholder="默认 3" />
            </label>
            <label className="flex flex-col gap-1">
              <span style={{ color: "var(--fg-mute)" }}>每动作预算（次，D8 未实装）</span>
              <input className="input" type="number" min={0} value={budget} onChange={(e) => setBudget(num(e.target.value))} placeholder="默认 2" />
            </label>
          </div>
          <div className="mt-2 text-[10px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
            修复轮次上限 / 每动作预算目前只落库，执行链路尚未消费（D8 未实装）；需要硬限制请先在执行脚本里控制。
          </div>
        </div>
      )}
      {overrideWarn && (
        <div className="mx-4 mb-3 rounded-[9px] px-3 py-2 text-[11px] leading-relaxed" style={{ border: "1px solid var(--warn)", color: "var(--warn)" }}>
          {overrideWarn}
        </div>
      )}
      {msg && (
        <div className="px-4 pb-3 text-[11px]" style={{ color: msg.startsWith("已保存") ? "var(--ok)" : "var(--bad)" }}>
          {msg}
        </div>
      )}
    </section>
  );
}
