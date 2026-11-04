import { Fragment, useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import CasePanel from "./components/CasePanel";
import ApiPanel from "./components/ApiPanel";
import StagePanel from "./components/StagePanel";
import TaskStart from "./components/TaskStart";
import TaskWorkbench from "./components/TaskWorkbench";
import TaskDeleteDialog from "./components/TaskDeleteDialog";
import TestPanel from "./components/TestPanel";
import UsagePanel from "./components/UsagePanel";
import { api, gb, when, STAGES } from "./lib/api";
import type { Host, HostImage, Project, StageTaskView, Task, TaskDetail } from "./types";

type View = "tasks" | "task" | "task-new" | "hosts" | "usage" | "tests" | "stages" | "api" | "library";

const NAV: { key: View; label: string; icon: string; phase: string }[] = [
  { key: "tasks", label: "任务", icon: "▤", phase: "P0" },
  { key: "stages", label: "阶段看板", icon: "◷", phase: "P3" },
  { key: "tests", label: "测试管理", icon: "✓", phase: "P3" },
  { key: "library", label: "案例库", icon: "📦", phase: "M2" },
  { key: "hosts", label: "服务器与镜像", icon: "🖧", phase: "P0" },
  { key: "api", label: "API 管理", icon: "🔑", phase: "P4" },
  { key: "usage", label: "消耗情况", icon: "¥", phase: "P4" },
  // 「流式原型」页已隐藏（2026-10-21）：流式能力已并入「任务开始 / 对话」真实链路，原型页只作旁路对照。
  // 恢复时：加回 NAV 一项 + View 类型 + 标题/副标题分支 + 渲染块；组件 web/src/components/StreamProto.tsx
  // 与后端 /api/proto/* 均保留未删。
];

const EMPTY_HOST_FORM = { name: "", ssh_alias: "", cred_ref: "", note: "" };

/** 算子开发镜像的参考模板：只在「开启标记且当前没有 recipe」时插入，不覆盖已有配方。 */
const OP_DEV_RECIPE =
  "docker run --rm -v <工作区>:/ws -w /ws --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \\\n  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\\n  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver <本镜像> bash -lc \"export PYTHONPATH=$PYTHONPATH:<pylibs>\"";
const OP_DEV_PREFLIGHT =
  'npu-smi info -t board -i 0 >/dev/null && python3 -c "import torch, torch_npu; print(torch_npu.npu.device_count())"';

/** 镜像元数据表单（空串 = 清空该字段，提交时转 null）。 */
const EMPTY_META_FORM = { digest: "", torch_version: "", python_version: "", cann_version: "" };
type MetaForm = typeof EMPTY_META_FORM;

/** 在镜像清单里按 id 取镜像（清单可能还没拉到，或镜像不在这台机器上）。 */
const findImage = (list: HostImage[], id: number | null | undefined): HostImage | null =>
  id == null ? null : (list.find((i) => i.id === id) ?? null);

export default function App() {
  const [view, setView] = useState<View>("tasks");
  const [taskId, setTaskId] = useState<number | null>(null);
  // 深浅色切换已移除：平台固定浅色，不再保留 theme 状态。
  const [err, setErr] = useState<string | null>(null);

  const [tasks, setTasks] = useState<Task[]>([]);
  const [task, setTask] = useState<TaskDetail | null>(null);
  const [stageMap, setStageMap] = useState<Record<number, StageTaskView>>({});
  const [listFilter, setListFilter] = useState<"all" | "active" | "draft">("all");
  const [hosts, setHosts] = useState<Host[]>([]);
  const [images, setImages] = useState<HostImage[]>([]);
  /** 当前查看服务器的镜像清单错误（与空清单分开：失败不能伪装成「没有镜像」）。 */
  const [imagesErr, setImagesErr] = useState<string | null>(null);
  /** 镜像清单请求的存活守卫：记录最近一次请求目标 host id，晚到的旧响应直接丢弃。 */
  const imagesHostRef = useRef<number | null>(null);
  /** 元数据行内编辑：正在编辑的镜像 id + 四个字段。 */
  const [metaEditId, setMetaEditId] = useState<number | null>(null);
  const [metaForm, setMetaForm] = useState<MetaForm>({ ...EMPTY_META_FORM });
  const [projectImages, setProjectImages] = useState<HostImage[]>([]);
  /** 项目自身执行机（project.host_id）的镜像：用于解析 project.image_id（与执行机解析不是同一条链路）。 */
  const [projectHostImages, setProjectHostImages] = useState<HostImage[]>([]);
  const [selectedHostId, setSelectedHostId] = useState<number | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [draftId, setDraftId] = useState<number | null>(null);
  /** 从任务详情「打开测试管理」带过去的当前任务（TestPanel 初值）。 */
  const [testsTaskId, setTestsTaskId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [showHostForm, setShowHostForm] = useState(false);
  const [editingHostId, setEditingHostId] = useState<number | null>(null);
  const [hostForm, setHostForm] = useState({ ...EMPTY_HOST_FORM });
  const [keyPickedName, setKeyPickedName] = useState<string | null>(null);
  const keyFileRef = useRef<HTMLInputElement | null>(null);
  /** 任务详情请求序号：切换任务/卸载时旧响应作废，避免慢响应覆盖新详情。 */
  const taskSeqRef = useRef(0);
  /** 「任务开始」页的 flush 钩子：侧栏导航切走前先把防抖中的编辑落库。 */
  const taskStartFlushRef = useRef<(() => Promise<boolean>) | null>(null);
  const [onlyTestImages, setOnlyTestImages] = useState(false);
  const [deleteTaskTarget, setDeleteTaskTarget] = useState<number | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      setTasks(await api.tasks());
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      setProjects(await api.projects());
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const loadHosts = useCallback(async () => {
    try {
      const hs = await api.hosts();
      setHosts(hs);
      setSelectedHostId((prev) => {
        if (prev != null && hs.some((h) => h.id === prev)) return prev;
        return hs.find((h) => h.is_test_server)?.id ?? hs[0]?.id ?? null;
      });
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  // 当前查看的服务器镜像（带存活守卫：切服务器/后台刷新时晚到的旧响应不能覆盖当前清单）
  useEffect(() => {
    const targetId = selectedHostId;
    imagesHostRef.current = targetId;
    setImagesErr(null);
    if (targetId == null) {
      setImages([]);
      return;
    }
    setImages([]);
    void api
      .images(targetId)
      .then((list) => {
        if (imagesHostRef.current !== targetId) return;
        setImages(list);
      })
      .catch((e) => {
        if (imagesHostRef.current !== targetId) return;
        setImagesErr(String(e));
      });
  }, [selectedHostId]);

  // 任务默认执行机的镜像（当前测试服务器优先；用于「任务开始」页展示）
  const testHostId = hosts.find((h) => h.is_test_server)?.id ?? null;
  const projectHostId = projects[0]?.host_id ?? null;
  const defaultHostId = testHostId ?? projectHostId;
  useEffect(() => {
    if (defaultHostId == null) {
      setProjectImages([]);
      return;
    }
    void api.images(defaultHostId).then(setProjectImages).catch(() => setProjectImages([]));
  }, [defaultHostId]);

  // 项目自身执行机的镜像：后端镜像解析是「任务 > 项目默认」，与执行机解析（任务 > 测试服务器 > 项目默认）独立。
  // 测试服务器与项目主机不是同一台时，只看当前执行机清单会把已绑定的项目镜像误显示成「项目默认未绑定」。
  useEffect(() => {
    if (projectHostId == null) {
      setProjectHostImages([]);
      return;
    }
    if (projectHostId === defaultHostId) {
      setProjectHostImages(projectImages);
      return;
    }
    let alive = true;
    void api
      .images(projectHostId)
      .then((list) => {
        if (alive) setProjectHostImages(list);
      })
      .catch(() => {
        if (alive) setProjectHostImages([]);
      });
    return () => {
      alive = false;
    };
  }, [projectHostId, defaultHostId, projectImages]);

  // 探针 / 标记镜像后统一刷新：当前查看的清单 + （若就是默认/项目执行机）任务页清单。
  // 主清单只更新「当前查看的服务器」：后台探针可能在他机完成，晚到的响应不能覆盖新选择的清单。
  const refreshImages = async (hostId: number) => {
    const next = await api.images(hostId);
    if (imagesHostRef.current === hostId) {
      setImages(next);
      setImagesErr(null);
    }
    if (hostId === defaultHostId) setProjectImages(next);
    if (hostId === projectHostId) setProjectHostImages(next);
  };

  // 首屏把服务器与项目拉一次（侧栏页脚要显示执行机）
  useEffect(() => {
    void loadHosts();
    void loadProjects();
  }, [loadHosts, loadProjects]);

  // 每次进入「任务」页都刷新列表：刚建的草稿 / 刚改的标题，返回列表要能立刻看到。
  useEffect(() => {
    if (view === "tasks") void loadTasks();
  }, [view, loadTasks]);

  // 任务详情加载（带请求序号）：快速 A→B 切换任务时，A 的慢响应不能覆盖 B。
  const reloadTask = useCallback(async (id: number) => {
    const seq = ++taskSeqRef.current;
    try {
      const t = await api.task(id);
      if (seq === taskSeqRef.current) setTask(t);
    } catch (e) {
      if (seq === taskSeqRef.current) setErr(String(e));
    }
  }, []);

  useEffect(() => {
    if (view !== "task" || taskId === null) return;
    void reloadTask(taskId);
    // 退出详情 / 切换任务时让在途请求作废，避免旧响应覆盖新详情。
    return () => {
      taskSeqRef.current += 1;
    };
  }, [view, taskId, reloadTask]);

  // 阶段推导（stageMap）：进入任务相关页 + 详情里发生变更后都刷新，否则跑完动作后
  // 顶部阶段/进度条停在旧值，与同页 GatePanel 矛盾。
  const refreshStages = useCallback(async () => {
    try {
      const board = await api.stages();
      const m: Record<number, StageTaskView> = {};
      for (const col of board.columns) for (const t of col.tasks) m[t.task_id] = t;
      setStageMap(m);
    } catch {
      // 阶段推导失败不阻塞页面：保留上一次结果
    }
  }, []);

  useEffect(() => {
    if (view !== "tasks" && view !== "task") return;
    void refreshStages();
  }, [view, taskId, refreshStages]);

  const openTask = (id: number) => {
    setTaskId(id);
    setTask(null);
    setErr(null);
    setView("task");
  };

  const openNewTask = () => {
    setDraftId(null);
    setErr(null);
    setView("task-new");
  };

  const openDraft = (id: number) => {
    setDraftId(id);
    setErr(null);
    setView("task-new");
  };

  /** 切视图：当前在「任务开始」页且还有防抖中的编辑时，先 flush 再走（侧栏导航也覆盖）。 */
  const leaveView = async (next: View): Promise<boolean> => {
    if (view === "task-new" && taskStartFlushRef.current) {
      const ok = await taskStartFlushRef.current();
      if (!ok && !window.confirm("有改动未能保存（见任务开始页错误提示）。仍要离开吗？离开后这些改动会丢失。")) {
        return false;
      }
    }
    setView(next);
    setErr(null);
    return true;
  };

  const doProbe = async () => {
    const target = hosts.find((h) => h.id === selectedHostId) ?? hosts[0];
    if (!target) return;
    setBusy(true);
    setErr(null);
    try {
      await api.probe(target.id);
      await loadHosts();
      await refreshImages(target.id);
    } catch (e) {
      // 探针失败也会落 probe_state=fail：重拉一次让页面如实显示
      await loadHosts();
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const resetHostForm = () => {
    setHostForm({ ...EMPTY_HOST_FORM });
    setEditingHostId(null);
    setKeyPickedName(null);
    setShowHostForm(false);
  };

  const startAddHost = () => {
    setEditingHostId(null);
    setHostForm({ ...EMPTY_HOST_FORM });
    setKeyPickedName(null);
    setShowHostForm((v) => !v);
  };

  const startEditHost = (h: Host) => {
    setShowHostForm(false);
    setEditingHostId(h.id);
    setKeyPickedName(null);
    // 历史/API 建的服务器可能只有 address（没有 ssh_alias）：回填地址让页面能改名称/备注/私钥。
    setHostForm({
      name: h.name,
      ssh_alias: h.ssh_alias ?? h.address ?? "",
      cred_ref: h.cred_ref ?? "",
      note: h.note ?? "",
    });
  };

  // 私钥文件选择：浏览器拿不到原文件绝对路径，所以上传给后端复制到本机 key 目录，
  // 回填落点路径给 cred_ref（私钥内容不进库、不回显）。
  const onSshKeyPicked = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // 允许再次选择同一个文件
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.uploadSshKey(file);
      setHostForm((f) => ({ ...f, cred_ref: res.path }));
      setKeyPickedName(res.name);
    } catch (err) {
      setErr(String(err));
    } finally {
      setBusy(false);
    }
  };

  // 保存后后台探一次：卡数 / CANN / 镜像属于「连接目标」的派生信息，目标或私钥变了必须重拉。
  // 探针失败不回滚登记（登记本身已生效），只在页面顶部明确提示。
  const probeInBackground = (hostId: number) => {
    void api.probe(hostId)
      .then(async () => {
        await loadHosts();
        await refreshImages(hostId);
      })
      .catch(async (e) => {
        await loadHosts();
        setErr(`已保存，但自动探针失败：${String(e)}`);
      });
  };

  const saveHost = async () => {
    const sshAlias = hostForm.ssh_alias.trim();
    if (!sshAlias) {
      setErr("请填 SSH 连接目标（~/.ssh/config 里的 alias 或 user@host）");
      return;
    }
    const body = {
      name: hostForm.name.trim() || sshAlias,
      ssh_alias: sshAlias,
      cred_ref: hostForm.cred_ref.trim() || null,
      note: hostForm.note.trim() || null,
    };
    const editingId = editingHostId;
    const before = editingId != null ? hosts.find((h) => h.id === editingId) : null;
    setBusy(true);
    setErr(null);
    try {
      const h = editingId != null ? await api.patchHost(editingId, body) : await api.createHost(body);
      resetHostForm();
      await loadHosts();
      setSelectedHostId(h.id);
      await refreshImages(h.id);
      // 只有连接目标/私钥变了才需要重探；只改名称或备注不打扰远端。
      const connectionChanged = editingId == null || before == null
        || before.ssh_alias !== body.ssh_alias
        || (before.cred_ref ?? null) !== body.cred_ref;
      if (connectionChanged) probeInBackground(h.id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const deleteHost = async (h: Host) => {
    const ok = window.confirm(
      `删除服务器「${h.name}」？\n\n它登记的镜像清单会一并删除，不可恢复。\n仍被项目 / 任务 / 执行引用时会拒绝删除。`,
    );
    if (!ok) return;
    setBusy(true);
    setErr(null);
    try {
      await api.deleteHost(h.id);
      if (editingHostId === h.id) resetHostForm();
      await loadHosts();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const setTestServer = async (id: number) => {
    setBusy(true);
    try {
      await api.setTestHost(id);
      await loadHosts();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleOpDev = async (img: HostImage) => {
    const turningOn = !img.is_op_dev;
    const body: Partial<HostImage> = { is_op_dev: turningOn };
    // 只在「开启且当前没有配方」时插入参考模板；不覆盖已有 recipe，也不在关闭时传 null 清空。
    if (turningOn && !img.env_recipe) {
      body.env_recipe = OP_DEV_RECIPE;
      body.preflight_cmd = OP_DEV_PREFLIGHT;
    }
    setBusy(true);
    try {
      await api.patchImage(img.id, body);
      await loadHosts();
      await refreshImages(img.host_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const startMetaEdit = (img: HostImage) => {
    setMetaEditId(img.id);
    setMetaForm({
      digest: img.digest ?? "",
      torch_version: img.torch_version ?? "",
      python_version: img.python_version ?? "",
      cann_version: img.cann_version ?? "",
    });
  };

  const cancelMetaEdit = () => {
    setMetaEditId(null);
    setMetaForm({ ...EMPTY_META_FORM });
  };

  const saveMeta = async (img: HostImage) => {
    setBusy(true);
    setErr(null);
    try {
      await api.patchImage(img.id, {
        digest: metaForm.digest.trim() || null,
        torch_version: metaForm.torch_version.trim() || null,
        python_version: metaForm.python_version.trim() || null,
        cann_version: metaForm.cann_version.trim() || null,
      });
      cancelMetaEdit();
      await refreshImages(img.host_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleTestImage = async (img: HostImage) => {
    setBusy(true);
    try {
      await api.patchImage(img.id, { is_test: !img.is_test });
      await refreshImages(img.host_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const host = hosts.find((h) => h.id === selectedHostId) ?? hosts[0];
  // 悬空统计排除「远端已无」的陈旧行：它们已被标记为 missing，不应再计入可回收/悬空。
  const dangling = images.filter((i) => i.dangling && i.scan_state !== "missing");
  const reclaim = dangling.reduce((s, i) => s + (i.size_bytes ?? 0), 0);

  const project = projects[0] ?? null;
  const testHost = hosts.find((h) => h.is_test_server) ?? null;
  // 执行机口径与后端一致：当前测试服务器 > 项目默认执行机（不再回退到「当前查看的服务器」，避免展示与实际执行不一致）。
  const projectHost = testHost ?? hosts.find((h) => h.id === projectHostId) ?? null;
  // 项目默认镜像：优先按「项目自身执行机」的清单解析，再兜底当前默认执行机清单。
  const projectImage = findImage(projectHostImages, project?.image_id) ?? findImage(projectImages, project?.image_id);
  const projectImageHost = projectImage ? hosts.find((h) => h.id === projectImage.host_id) ?? null : null;
  const projectImageNote =
    projectImage && projectHost && projectImage.host_id !== projectHost.id
      ? `镜像在另一台执行机「${projectImageHost?.name ?? `#${projectImage.host_id}`}」上（当前执行机「${projectHost.name}」），真执行算 plan 时会报「镜像不在执行机上」`
      : null;
  const visibleImages = onlyTestImages ? images.filter((i) => i.is_test) : images;
  const testImageCount = images.filter((i) => i.is_test).length;
  const draftCount = tasks.filter((t) => t.is_draft).length;
  // 任务详情标题带版本：大版本 = 当前规格版本，小版本 = task.version_minor（契约 §6）。
  const currentSpecRow = task ? task.spec_versions.find((s) => s.id === task.current_spec_version_id) : null;
  const taskMajor = currentSpecRow?.version ?? (task?.spec_versions.length ? task.spec_versions[task.spec_versions.length - 1].version : null);
  const taskVersionLabel = task && taskMajor != null ? `v${taskMajor}.${task.version_minor ?? 0}` : null;

  return (
    <div className="grid min-h-screen" style={{ gridTemplateColumns: "214px 1fr" }}>
      {/* 侧栏 */}
      <aside
        className="px-3 py-4"
        style={{ background: "var(--panel)", borderRight: "1px solid var(--border)", backdropFilter: "blur(16px)" }}
      >
        <div className="flex items-center gap-2 px-2 pb-3">
          <span
            className="inline-block h-[22px] w-[22px] rounded-[7px]"
            style={{ background: "linear-gradient(135deg, var(--accent), #38bdf8)" }}
          />
          <span className="text-[15px] font-semibold">op-agent</span>
        </div>
        <div className="card mb-4 flex items-center gap-2 px-3 py-2 text-xs">
          📁 CANN 算子开发 <span className="ml-auto" style={{ color: "var(--fg-mute)" }}>▾</span>
        </div>
        <nav className="flex flex-col gap-[2px]">
          {NAV.map((n) => {
            const active = view === n.key;
            return (
              <button
                key={n.label}
                onClick={() => {
                  // 侧栏直接进「测试管理」= 无来路任务，TestPanel 按默认（最新任务）选中
                  if (n.key === "tests") setTestsTaskId(null);
                  void leaveView(n.key);
                }}
                className="flex items-center gap-3 rounded-md px-3 py-[7px] text-left text-xs"
                style={{ background: active ? "var(--accent-soft)" : "transparent", color: active ? "var(--fg)" : "var(--fg-dim)", fontWeight: active ? 650 : 400 }}
              >
                <span className="w-4 text-center">{n.icon}</span>
                {n.label}
                <span className="chip ml-auto !px-2 !py-0 !text-[10px]">{n.phase}</span>
              </button>
            );
          })}
        </nav>
        <div className="mt-6 px-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
          {projectHost ? (
            <>
              执行机 {projectHost.name} · {projectHost.ssh_alias || projectHost.address || "未配置"}
              <br />
              {projectHost.cards ?? "—"} · {projectHost.cann_version ?? "—"}
            </>
          ) : hosts.length > 0 ? (
            "执行机未设置：未指定测试服务器 / 项目默认执行机"
          ) : (
            "尚未登记服务器"
          )}
        </div>
      </aside>

      <main className="min-w-0 p-6">
        <header className="mb-4 flex flex-wrap items-end gap-4">
          <div>
            <h1 className="m-0 text-[19px] font-semibold">
              {view === "tasks"
                ? "任务"
                : view === "hosts"
                  ? "服务器与镜像"
                  : view === "task-new"
                    ? "任务开始"
                    : view === "usage"
                      ? "消耗情况"
                      : view === "tests"
                        ? "测试管理"
                        : view === "stages"
                          ? "阶段看板"
                          : view === "api"
                            ? "API 管理"
                            : view === "library"
                                ? "案例库"
                                : view === "task"
                                  ? (
                                    <>
                                      {task?.title || task?.op_name || "任务详情"}
                                      {taskVersionLabel && (
                                        <span
                                          className="mono ml-2 align-middle text-[13px] font-normal"
                                          style={{ color: "var(--fg-mute)" }}
                                          title="大版本 = 当前规格版本；小版本 = 采纳优化 / 交付前变更"
                                        >
                                          {taskVersionLabel}
                                        </span>
                                      )}
                                    </>
                                  )
                                  : "待开发"}
            </h1>
            <div className="mt-1 text-xs" style={{ color: "var(--fg-dim)" }}>
              {view === "tasks" && "这一页回答：现在有哪些算子任务、卡在哪一步"}
              {view === "task-new" && "把「要什么算子、跑在哪、精度怎么判」定清楚 —— 产出算子需求文档与任务上下文空间"}
              {view === "hosts" && "这一页回答：任务能丢到哪台机器、用哪个镜像跑"}
              {view === "usage" && "这一页回答：用了多少 token、花在哪 —— 只算真实调用，P0 示例行已排除"}
              {view === "tests" && "这一页回答：用例集覆盖了什么、每次 test 逐例过没过、失败点在哪"}
              {view === "stages" && "这一页回答：每个任务走到哪一阶段、卡在哪道门禁 —— 阶段由真实证据推导"}
              {view === "api" && "这一页回答：用哪个模型/哪个 key 引用、通不通 —— 凭据只存引用名，探针只验连通性"}
              {view === "task" && "这一页回答：这次 run 做了什么、产物在哪、门禁过没过"}
              {view === "library" && "案例库是外置补充包：独立 zip 由你保管，需要时用导入脚本装进本机库目录"}
            </div>
          </div>
          <div className="ml-auto flex gap-2">
            {view === "task" && (
              <button className="btn" onClick={() => setView("tasks")}>
                ← 任务列表
              </button>
            )}
            {view === "task" && task && (
              <button className="btn btn-danger" onClick={() => setDeleteTaskTarget(task.id)}>
                删除任务
              </button>
            )}
            {view === "hosts" && (
              <button className="btn btn-primary" onClick={doProbe} disabled={busy}>
                {busy ? "探针中…" : "重新探针"}
              </button>
            )}
          </div>
        </header>

        {err && (
          <div className="card mb-4 px-4 py-3 text-xs" style={{ borderColor: "var(--bad)", color: "var(--bad)" }}>
            接口错误：{err}
          </div>
        )}

        {/* ========== 任务列表 ========== */}
        {view === "tasks" && (
          <section className="card overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3">
              <span className="text-xs" style={{ color: "var(--fg-dim)" }}>
                共 {tasks.length} 个任务{draftCount > 0 ? ` · 草稿 ${draftCount}` : ""}
              </span>
              <div className="ml-3 flex gap-1">
                {([["all", "全部"], ["active", "正式"], ["draft", "草稿"]] as const).map(([k, label]) => (
                  <button
                    key={k}
                    className="btn btn-sm"
                    onClick={() => setListFilter(k)}
                    style={listFilter === k ? { background: "var(--accent-soft)", borderColor: "var(--accent)" } : undefined}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button className="btn btn-primary btn-sm ml-auto" onClick={openNewTask}>
                + 新建任务
              </button>
            </div>
            <div className="tbl-wrap overflow-auto">
              <table>
                <thead>
                  <tr>
                    <th>算子</th>
                    <th>阶段</th>
                    <th>芯片</th>
                    <th>输入形态</th>
                    <th>创建</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {tasks.filter((t) => listFilter === "all" || (listFilter === "draft" ? t.is_draft : !t.is_draft)).map((t) => (
                    <tr key={t.id}>
                      <td>
                        <b>{t.title}</b>
                        {t.is_draft && <span className="chip ml-2">草稿</span>}
                        <div className="mono" style={{ color: "var(--fg-mute)" }}>
                          {t.op_name}
                        </div>
                      </td>
                      <td>
                        {t.is_draft ? (
                          "—"
                        ) : (
                          <div>
                            <div className="flex items-center gap-1">
                              {stageMap[t.id]?.derived_stage ?? t.stage}
                              {stageMap[t.id] && !stageMap[t.id].in_sync && <span className="chip">未同步</span>}
                            </div>
                            <div className="mt-1 flex items-center gap-1">
                              {STAGES.map((s, i) => {
                                const cur = stageMap[t.id]?.derived_stage ?? t.stage;
                                return (
                                  <span
                                    key={s}
                                    className="inline-block h-[5px] w-[26px] rounded-[3px]"
                                    style={{
                                      background:
                                        s === cur ? "var(--run)" : STAGES.indexOf(cur) > i ? "var(--ok)" : "var(--border-strong)",
                                    }}
                                  />
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </td>
                      <td>{t.target_chip}</td>
                      <td className="text-xs" style={{ color: "var(--fg-dim)" }}>
                        {t.input_form}
                      </td>
                      <td className="text-xs" style={{ color: "var(--fg-mute)" }}>
                        {when(t.created_at)}
                      </td>
                      <td className="whitespace-nowrap">
                        <div className="flex flex-nowrap gap-1">
                          <button
                            className="btn btn-sm whitespace-nowrap"
                            onClick={() => (t.is_draft ? openDraft(t.id) : openTask(t.id))}
                          >
                            {t.is_draft ? "继续编辑" : "打开"}
                          </button>
                          <button
                            className="btn btn-sm btn-danger whitespace-nowrap"
                            onClick={() => setDeleteTaskTarget(t.id)}
                          >
                            删除
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {tasks.filter((t) => listFilter === "all" || (listFilter === "draft" ? t.is_draft : !t.is_draft)).length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-xs" style={{ color: "var(--fg-mute)" }}>
                        还没有任务。点「+ 新建任务」开始定义第一个算子。
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ========== 消耗情况（P4） ========== */}
        {view === "usage" && <UsagePanel />}

        {/* ========== 测试管理（P3 读侧） ========== */}
        {view === "tests" && <TestPanel tasks={tasks} initialTaskId={testsTaskId} />}

        {/* ========== 阶段看板（P3 读侧） ========== */}
        {view === "stages" && <StagePanel onOpenTask={openTask} onOpenDraft={openDraft} />}

        {/* ========== API 管理（P4 配置态） ========== */}
        {view === "api" && <ApiPanel />}

        {/* ========== 对话（P0-2） ========== */}

        {/* ========== 案例库（占位：外置 bundle + 导入脚本） ========== */}
        {view === "library" && <CasePanel onOpenTask={openTask} />}

        {/* ========== 任务开始（新建任务 · 对话式定义） ========== */}
        {view === "task-new" && (
          <TaskStart
            projectId={project?.id ?? null}
            draftId={draftId}
            project={project}
            host={projectHost}
            image={projectImage}
            imageNote={projectImageNote}
            flushRef={taskStartFlushRef}
            onBack={() => setView("tasks")}
            onDraftId={setDraftId}
            onCreated={(id) => {
              setDraftId(null);
              void loadTasks();
              openTask(id);
            }}
          />
        )}

        {/* ========== 任务详情（作业台：分区见 TaskWorkbench） ========== */}
        {view === "task" && task && (
          <TaskWorkbench
            task={task}
            stageView={stageMap[task.id] ?? null}
            onChanged={() => {
              void loadTasks();
              if (taskId !== null) void reloadTask(taskId);
              void refreshStages();
            }}
            onOpenTests={(id?: number) => {
              // 详情面板会带 task.id 调用；兼容旧的「无参 + 鼠标事件」调用（event 不是 number）
              setTestsTaskId(typeof id === "number" ? id : null);
              setView("tests");
            }}
          />
        )}

        {/* ========== 服务器与镜像 ========== */}
        {view === "hosts" && (
          <>
            <div className="mb-4 grid grid-cols-4 gap-3">
              {[
                { k: "在线机器", v: `${hosts.filter((h) => h.probe_state === "ok").length} / ${hosts.length}` },
                { k: "NPU 卡数", v: host?.cards ?? "—" },
                { k: "镜像总数", v: String(images.length), s: `悬空 ${dangling.length}` },
                { k: "测试镜像", v: String(testImageCount), s: `当前测试服务器 ${testHost?.name ?? "未设置"}` },
              ].map((c) => (
                <div key={c.k} className="card px-4 py-3">
                  <div className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    {c.k}
                  </div>
                  <div className="mt-1 text-[15px] font-semibold">
                    {c.v} {c.s && <span className="text-[11px] font-normal" style={{ color: "var(--fg-dim)" }}>{c.s}</span>}
                  </div>
                </div>
              ))}
            </div>

            <section className="card mb-4 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px]" style={{ color: "var(--fg-mute)" }}>当前查看</span>
                <select
                  className="input max-w-[320px]"
                  value={selectedHostId ?? ""}
                  onChange={(e) => setSelectedHostId(e.target.value === "" ? null : Number(e.target.value))}
                >
                  {hosts.map((h) => (
                    <option key={h.id} value={h.id}>
                      {h.name}{h.ssh_alias ? `（${h.ssh_alias}）` : ""}{h.is_test_server ? " · 当前测试服务器" : ""}
                    </option>
                  ))}
                </select>
                <button className="btn btn-sm" onClick={startAddHost}>
                  {showHostForm ? "收起新增" : "+ 新增服务器"}
                </button>
                {host && (
                  <button
                    className="btn btn-sm"
                    onClick={() => void setTestServer(host.id)}
                    disabled={busy || host.is_test_server}
                    title="任务未显式指定执行机时，优先用这台"
                  >
                    {host.is_test_server ? "已是当前测试服务器" : "设为当前测试服务器"}
                  </button>
                )}
                {testHost && (
                  <span className="chip chip-ok ml-auto">当前测试服务器：{testHost.name}</span>
                )}
              </div>

              {(showHostForm || editingHostId != null) && (
                <div className="mt-3 grid gap-2 md:grid-cols-3" style={{ borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                  <div className="md:col-span-3 text-[11px] font-semibold">
                    {editingHostId != null ? "编辑服务器连接信息" : "新增服务器"}
                  </div>
                  <label className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    SSH 连接目标 *
                    <input className="input mono" value={hostForm.ssh_alias} onChange={(e) => setHostForm((f) => ({ ...f, ssh_alias: e.target.value }))} placeholder="alias、user@host 或 root@1.2.3.4:50023" />
                    <span className="text-[10px]">支持 `user@host:端口` / `ssh://user@host:端口`；端口留空走 ssh 默认</span>
                    <span className="text-[10px]">
                      实际连接目标 = 这里填的 ssh_alias（没有 alias 时回退服务器登记的 address）；保存后以 ssh_alias 为主。
                    </span>
                  </label>
                  <label className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    显示名（可选）
                    <input className="input" value={hostForm.name} onChange={(e) => setHostForm((f) => ({ ...f, name: e.target.value }))} placeholder="留空 = 用 SSH 目标" />
                  </label>
                  <div className="flex flex-col gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    SSH 私钥（可选）
                    <div className="flex gap-1">
                      <input className="input mono min-w-0 flex-1" value={hostForm.cred_ref} onChange={(e) => setHostForm((f) => ({ ...f, cred_ref: e.target.value }))} placeholder="选择私钥文件，或直接填已有路径" />
                      <button type="button" className="btn btn-sm shrink-0" onClick={() => keyFileRef.current?.click()} disabled={busy}>
                        选择文件…
                      </button>
                    </div>
                    <span className="text-[10px]">
                      {keyPickedName
                        ? `已选择 ${keyPickedName}，保存后生效（复制到本机 key 目录，0600）`
                        : "选择后复制到本机 key 目录；留空 = 全局 OPX_SSH_KEY"}
                    </span>
                  </div>
                  <label className="flex flex-col gap-1 text-[11px] md:col-span-3" style={{ color: "var(--fg-mute)" }}>
                    备注
                    <input className="input" value={hostForm.note} onChange={(e) => setHostForm((f) => ({ ...f, note: e.target.value }))} />
                  </label>
                  <div className="flex flex-wrap items-center gap-2 md:col-span-3">
                    <button className="btn btn-primary btn-sm" onClick={() => void saveHost()} disabled={busy || !hostForm.ssh_alias.trim()}>
                      {editingHostId != null ? "保存修改" : "登记服务器"}
                    </button>
                    <button className="btn btn-sm" onClick={resetHostForm} disabled={busy}>取消</button>
                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                      保存后自动探针拉卡数/CANN/镜像；失败可手动重试。
                    </span>
                    <input ref={keyFileRef} type="file" style={{ display: "none" }} onChange={(e) => void onSshKeyPicked(e)} />
                  </div>
                </div>
              )}
            </section>

            {host && (
              <section className="card mb-4 p-4">
                <div className="flex items-start gap-3">
                  <div className="flex-1">
                    <div className="text-xs font-semibold">
                      {host.name} · SSH {host.ssh_alias || host.address || "未配置"}
                      <span
                        className="chip ml-2"
                        style={{
                          background:
                            host.probe_state === "ok"
                              ? "var(--ok)"
                              : host.probe_state === "fail"
                                ? "var(--bad)"
                                : "var(--fg-mute)",
                          color: "#fff",
                          borderColor: "transparent",
                        }}
                        title={host.probe_error || undefined}
                      >
                        {host.probe_state === "ok" ? "已连接" : host.probe_state === "fail" ? "探针失败" : "未探针"}
                      </span>
                      {host.is_test_server && <span className="chip chip-ok ml-2">当前测试服务器</span>}
                    </div>
                    <div className="mt-1 text-[11px]" style={{ color: "var(--fg-dim)" }}>
                      {host.cards ?? "—"} · CANN {host.cann_version ?? "—"} · docker{" "}
                      {host.docker_ok == null ? "—" : host.docker_ok ? "可用" : "不可用"} · 最近探针 {when(host.last_probe_at)}
                    </div>
                    {host.probe_error && (
                      <div
                        className="mt-1 text-[11px]"
                        style={{ color: host.probe_state === "fail" ? "var(--bad)" : "var(--warn, #b45309)" }}
                        title={host.probe_error}
                      >
                        {host.probe_state === "fail" ? "探针失败原因：" : "探针告警（部分能力不可用）："}
                        {host.probe_error.length > 200 ? `${host.probe_error.slice(0, 200)}…` : host.probe_error}
                      </div>
                    )}
                    <div className="mt-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                      {host.ssh_alias && host.address && host.address !== host.ssh_alias ? `地址 ${host.address} · ` : ""}
                      私钥 {host.cred_ref || "全局 OPX_SSH_KEY"}
                      {host.note ? ` · ${host.note}` : ""}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button className="btn btn-sm" onClick={() => startEditHost(host)} disabled={busy}>
                      编辑
                    </button>
                    <button className="btn btn-sm btn-danger" onClick={() => void deleteHost(host)} disabled={busy}>
                      删除
                    </button>
                  </div>
                </div>
              </section>
            )}

            {host && (
              <section className="card overflow-hidden">
                <div className="flex flex-wrap items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                  <h2 className="m-0 text-xs font-semibold">镜像清单 · {host.name}</h2>
                  <label className="ml-2 flex items-center gap-1 text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    <input type="checkbox" checked={onlyTestImages} onChange={(e) => setOnlyTestImages(e.target.checked)} />
                    只看测试镜像
                  </label>
                  <span className="ml-auto text-[11px]" style={{ color: "var(--fg-mute)" }}>
                    悬空镜像 {dangling.length} 个 · 可回收 {gb(reclaim)}
                  </span>
                </div>
                {host.probe_state !== "ok" && (
                  <div className="px-4 py-2 text-[11px]" style={{ background: "var(--panel-3)", color: "var(--fg-dim)", borderBottom: "1px solid var(--border)" }}>
                    以下为上次成功扫描结果，可能属于旧目标 —— 目标/私钥改动后新探针失败时，清单不会被自动清空。
                  </div>
                )}
                <div className="max-h-[560px] overflow-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>镜像</th>
                        <th>torch · digest</th>
                        <th>大小</th>
                        <th>创建</th>
                        <th>状态</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {imagesErr && (
                        <tr>
                          <td colSpan={6} className="px-4 py-3 text-[11px]" style={{ color: "var(--bad)" }}>
                            镜像清单加载失败：{imagesErr}
                            <button
                              className="btn btn-sm ml-2"
                              onClick={() => {
                                const targetId = selectedHostId;
                                if (targetId == null) return;
                                setImagesErr(null);
                                void refreshImages(targetId).catch((e) => setImagesErr(String(e)));
                              }}
                              disabled={busy}
                            >
                              重试
                            </button>
                          </td>
                        </tr>
                      )}
                      {visibleImages.map((img) => {
                        const locked = img.dangling || img.scan_state === "missing";
                        const lockTitle =
                          img.scan_state === "missing"
                            ? "远端已无此镜像（上次扫描未发现），不能标记为测试/算子开发镜像"
                            : img.dangling
                              ? "悬空镜像（<none>）不能标记为测试/算子开发镜像"
                              : undefined;
                        return (
                          <Fragment key={img.id}>
                            <tr style={img.is_test ? { background: "var(--panel-3)" } : undefined}>
                              <td>
                                <div className="mono">
                                  {img.image_ref}
                                  {img.tag ? `:${img.tag}` : ""}
                                </div>
                                {img.env_recipe && (
                                  <div className="mt-1 whitespace-pre-wrap text-[10px]" style={{ color: "var(--fg-mute)" }}>
                                    {img.env_recipe.slice(0, 160)}…
                                  </div>
                                )}
                              </td>
                              <td className="text-[11px]">
                                <div className="flex items-center gap-1">
                                  <span className="mono" title={img.torch_version ? `torch ${img.torch_version}` : "未记录 torch 版本"}>
                                    {img.torch_version ?? "—"}
                                  </span>
                                  {img.is_test && !img.torch_version && (
                                    <span className="chip chip-warn" title="test / benchmark / profile 的 plan 需要 torch/torch_npu 版本；点「元数据」补齐">
                                      缺 torch
                                    </span>
                                  )}
                                </div>
                                <div className="mono" style={{ color: "var(--fg-mute)" }} title={img.digest ?? "未记录 digest"}>
                                  {(img.digest ?? "—").slice(0, 40)}
                                  {img.digest && img.digest.length > 40 ? "…" : ""}
                                </div>
                              </td>
                              <td>{gb(img.size_bytes)}</td>
                              <td className="text-[11px]" style={{ color: "var(--fg-mute)" }}>
                                {img.created_raw ?? "—"}
                              </td>
                              <td>
                                {img.is_test && <span className="chip chip-ok">测试镜像</span>}{" "}
                                {img.is_op_dev && <span className="chip chip-ok">算子开发镜像</span>}{" "}
                                {img.dangling && <span className="chip chip-bad">悬空</span>}{" "}
                                {img.scan_state === "missing" && (
                                  <span className="chip chip-bad" title="远端镜像已不存在（上次扫描未发现）">
                                    远端已无
                                  </span>
                                )}{" "}
                                {img.scan_state === "stale" && (
                                  <span className="chip chip-warn" title={img.last_scanned_at ? `上次成功扫描：${when(img.last_scanned_at)}` : "上次扫描结果"}>
                                    上次扫描
                                  </span>
                                )}
                              </td>
                              <td>
                                <div className="flex flex-wrap gap-1">
                                  <button
                                    className="btn btn-sm"
                                    onClick={() => toggleTestImage(img)}
                                    disabled={busy || locked}
                                    title={lockTitle}
                                  >
                                    {img.is_test ? "取消测试镜像" : "设为测试镜像"}
                                  </button>
                                  <button
                                    className="btn btn-sm"
                                    onClick={() => toggleOpDev(img)}
                                    disabled={busy || locked}
                                    title={lockTitle}
                                  >
                                    {img.is_op_dev ? "取消算子开发" : "设为算子开发"}
                                  </button>
                                  <button
                                    className="btn btn-sm"
                                    onClick={() => (metaEditId === img.id ? cancelMetaEdit() : startMetaEdit(img))}
                                    disabled={busy}
                                    title="编辑 digest / torch / python / CANN 元数据（plan 的前置信息）"
                                  >
                                    元数据
                                  </button>
                                </div>
                              </td>
                            </tr>
                            {metaEditId === img.id && (
                              <tr>
                                <td colSpan={6} style={{ background: "var(--panel-2)" }}>
                                  <div className="flex flex-wrap items-end gap-2">
                                    {([
                                      ["digest", "digest", "sha256:…"],
                                      ["torch_version", "torch", "2.10.0"],
                                      ["python_version", "python", "3.11"],
                                      ["cann_version", "CANN", "cann-9.2.0"],
                                    ] as const).map(([key, label, placeholder]) => (
                                      <label key={key} className="flex flex-col gap-1 text-[10px]" style={{ color: "var(--fg-mute)" }}>
                                        {label}
                                        <input
                                          className="input mono"
                                          style={{ width: 150 }}
                                          value={metaForm[key]}
                                          placeholder={placeholder}
                                          onChange={(e) => setMetaForm((f) => ({ ...f, [key]: e.target.value }))}
                                        />
                                      </label>
                                    ))}
                                    <button className="btn btn-sm btn-primary" onClick={() => void saveMeta(img)} disabled={busy}>
                                      保存元数据
                                    </button>
                                    <button className="btn btn-sm" onClick={cancelMetaEdit} disabled={busy}>
                                      取消
                                    </button>
                                    <span className="text-[10px]" style={{ color: "var(--fg-mute)" }}>
                                      留空 = 清空该字段；digest 用于判定「同一个镜像」。
                                    </span>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        );
                      })}
                      {visibleImages.length === 0 && !imagesErr && (
                        <tr>
                          <td colSpan={6} className="text-center text-[11px]" style={{ color: "var(--fg-mute)" }}>
                            {images.length > 0
                              ? "当前筛选下没有镜像 —— 取消「只看测试镜像」可查看全部。"
                              : "没有镜像。先点「重新探针」从服务器拉取镜像清单。"}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="px-4 py-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-mute)" }}>
                  「测试镜像」= 任务选镜像时可见的镜像；未标记的默认隐藏（本页仍显示以便配置）。
                  「算子开发镜像」= 还能跑算子的镜像，必须带 env_recipe；两者可以同时标记。
                </div>
              </section>
            )}
          </>
        )}
      </main>

      {deleteTaskTarget !== null && (
        <TaskDeleteDialog
          taskId={deleteTaskTarget}
          onClose={() => setDeleteTaskTarget(null)}
          onDeleted={(id) => {
            setDeleteTaskTarget(null);
            void loadTasks();
            if (taskId === id) {
              setTask(null);
              setTaskId(null);
              setView("tasks");
            }
          }}
        />
      )}
    </div>
  );
}
