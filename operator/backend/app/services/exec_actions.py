"""统一执行入口的四动作契约（方案 v1.1 §6）。

**契约 + 前置检查 + 版本固定 + 真执行**：`build_plan()` 算「该跑什么」，`services/action_runner.py`
照 plan 起子进程跑 `scripts/nm_test.sh`（异步作业），脚本自己登记 run，跑完对 run 跑门禁。
（2026-10-20 接通；此前 `execute()` 故意 501。）

契约（每个动作的六步都一样）：
  1. 检查任务配置与本动作的必要环境条件（含**设备占用**与**镜像能不能跑这个动作**）
  2. 固定源码 / 规格 / 测试集 / 环境配方版本
  3. 同步源码快照到远端 Run 目录
  4. 执行编译、测试或性能采集
  5. 回读日志、原始数据与结构化结果
  6. 生成 manifest、登记 Run、调报告校验与门禁判定

两个**实测换来的**硬约束（写在这里，免得下次又绕回去）：
- `env_recipe` 是「怎么把镜像变成可跑算子环境」的真相源（P1.5-a 实测：缺 `--privileged`、缺
  `ASCEND_RT_VISIBLE_DEVICES`、缺 davinci0..7 挂载都会让 torch_npu 看不到卡）。所以命令**不在这里
  手拼 docker run**，而是走 `scripts/nm_test.sh`（配方、超时与兜底清理都在那边固化）；
- 只有 CANN 工具链、不带 torch 的镜像跑不了 test/benchmark/profile（P1.5-a 实测 VERDICT torch_npu: FAIL），
  前置检查必须按**动作**判断镜像能不能用，而不是「有 env_recipe 就算齐」。
"""

from __future__ import annotations

import hashlib
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import ACTIONS, DEFAULT_CRITERIA_REF, Host, HostImage, Project, Run, SpecVersion, Task
from . import devices as dev
from .ssh import ssh_key, ssh_port, ssh_target

#: 每个动作的契约：需要什么前置、产出哪些产物类型、工作区里该有哪个入口脚本
ACTION_SPECS: dict[str, dict] = {
    "build": {
        "title": "编译",
        "entry": "build.sh",
        "requires": ("project_workspace", "host", "image", "env_recipe", "cases_confirmed"),
        "artifact_types": ("log",),
        "needs_device": False,  # 编译要挂设备节点，但不吃算力
        "remote_note": "容器内编译（宿主机 CANN 是编译装，不能直接跑）",
    },
    "smoke": {
        "title": "冒烟（编译+最小用例）",
        "entry": "smoke_check.sh",  # 工作区根的 smoke.sh 是环境自检脚本，动作入口单独一个名字
        "requires": ("project_workspace", "host", "image", "env_recipe", "spec_version", "case_suite",
                     "cases_confirmed", "free_device"),
        "artifact_types": ("log", "smoke"),
        "needs_device": True,  # 最小用例要真跑算子 → 需要一张空闲卡
        "remote_note": "编译 + 挑一条最小 required match 用例跑通（开发内自检；正式测试仍由 test 动作负责）",
    },
    "test": {
        "title": "正确性测试",
        "entry": "test.sh",
        "requires": ("project_workspace", "host", "image", "env_recipe", "spec_version", "case_suite",
                     "cases_confirmed", "smoke_pass"),
        "artifact_types": ("log", "correctness", "cases", "report"),
        "needs_device": True,
        "remote_note": "容器内跑用例集并与参考实现对比",
    },
    "benchmark": {
        "title": "基准测试",
        "entry": "benchmark.sh",
        "requires": ("project_workspace", "host", "image", "env_recipe", "spec_version", "case_suite",
                     "cases_confirmed", "smoke_pass", "free_device"),
        "artifact_types": ("log", "performance"),
        "needs_device": True,
        "remote_note": "与 profiling 分开执行（不混用不同测量条件的耗时）；要独占设备",
    },
    "profile": {
        "title": "性能采集",
        "entry": "profile.sh",
        "requires": ("project_workspace", "host", "image", "env_recipe", "spec_version", "case_suite",
                     "cases_confirmed", "benchmark_pass", "free_device"),
        "artifact_types": ("log", "metrics"),
        "needs_device": True,
        "remote_note": "msopprof 采单算子；四类指标已实测可采（D9 已定），产物 metrics.json",
    },
}

#: 需要「镜像能跑算子」的动作。普通 Ascend C 任务用 torch_npu
#: 作为运行时门禁；`jax_*` 任务改由任务专用 runner 校验 JAX NPU，
#: 不能因为镜像没有 torch_npu 而把已验收的 JAX 镜像误判为不可用。
ACTIONS_NEEDING_TORCH = ("smoke", "test", "benchmark", "profile")


def _is_jax_task(task: Task) -> bool:
    """返回任务是否使用 JAX 专用执行契约。"""
    return (task.op_name or "").startswith("jax_")


@dataclass
class Snapshot:
    """工作区源码快照：**按内容**哈希（不是 mtime），并显式报告有没有被截断。"""

    sha256: str = ""
    files: int = 0
    truncated: bool = False

    def as_dict(self) -> dict:
        return {"sha256": self.sha256, "files": self.files, "truncated": self.truncated}


def cases_gate_reason(task: Task, ws: Path | None) -> str | None:
    """任务生命周期 v2：四动作的「用例组确认」前置检查。

    用例组要在**任务对话**里整组确认（精度 + 性能），未确认或确认后用例集被改动都不允许跑动作
    —— 否则「开发 → 上机」会跑在一份没被用户确认过的用例集上。返回缺项文案；None = 通过。
    """
    if task.cases_confirmed_at is None:
        return "用例组未确认（先在任务对话里确认精度/性能用例组，再跑动作）"
    if not task.cases_confirmed_sha256:
        return "用例组确认记录缺少哈希（需在任务对话里重新确认用例组）"
    if ws is None:
        return "工作区不可用，无法校验已确认的用例组"
    try:
        from . import case_confirm  # 延迟导入：services 之间避免循环
        current = case_confirm.cases_suite_sha256(
            ws, task.op_name if task.cases_confirmed_scope == "operator-v1" else None
        )
    except Exception:  # noqa: BLE001 —— 校验不了就不能证明正在跑用户确认过的用例集
        return "无法计算当前用例集哈希（需修复工作区后重新确认）"
    if current != task.cases_confirmed_sha256:
        return "用例集已变更（需在任务对话里重新确认用例组，再跑动作）"
    return None


def image_digests_match(a: str | None, b: str | None) -> bool:
    """两个镜像 digest 是否指向同一个镜像（容忍「元数据里只记了短 digest」）。

    背景（2026-10-28 实测）：控制台的镜像元数据可能只登记了短 digest
    （如 `local-id:sha256:313b20aa34b4`），而 `nm_test.sh` 在 manifest 里写的是
    `docker image inspect .Id` 的完整值（`local-id:sha256:<64 hex>`）——
    严格字符串比较会把「同一镜像」判成不一致，导致 smoke 永远不满足 test/benchmark 的前置。

    规则：完全相等；或 `sha256:` 之后的十六进制部分一方是另一方的前缀且长度 ≥ 12
    （12 hex = 48 bit，碰撞概率可忽略；更短的短 ID 不认，避免误配）。
    """
    if not a or not b:
        return False
    if a == b:
        return True

    def _hex_part(value: str) -> str:
        s = value.strip().lower()
        idx = s.rfind("sha256:")
        return s[idx + len("sha256:"):] if idx >= 0 else s

    ha, hb = _hex_part(a), _hex_part(b)
    if len(ha) < 12 or len(hb) < 12:
        return False
    return ha.startswith(hb) or hb.startswith(ha)


def smoke_gate_reason(db: Session, task: Task, *, spec_version_id: int | None,
                      code_snapshot_sha256: str, suite_ref: str | None,
                      criteria_ref: str, image_digest: str | None) -> str | None:
    """test/benchmark 只接受当前条件下已通过门禁的 smoke 证据。

    smoke 是正式验证的硬前置，不只看“曾经跑过”：规格、代码快照、用例集、判据与镜像
    任一变化都必须重跑，避免拿旧冒烟结果给新候选放行。
    """
    from . import gate  # 延迟导入，保持服务模块依赖单向

    rows = db.scalars(
        select(Run).where(Run.task_id == task.id, Run.action == "smoke").order_by(Run.id.desc())
    )
    for run in rows:
        if run.status != "ok" or run.exit_code not in (None, 0):
            continue
        if spec_version_id is not None and run.spec_version_id != spec_version_id:
            continue
        effective_snapshot = run.scoped_code_snapshot_sha256 or run.code_snapshot_sha256
        effective_suite = run.scoped_suite_ref or run.suite_ref
        if code_snapshot_sha256 and effective_snapshot != code_snapshot_sha256:
            continue
        if suite_ref and effective_suite != suite_ref:
            continue
        if criteria_ref and run.criteria_ref != criteria_ref:
            continue
        if image_digest and not image_digests_match(run.image_digest, image_digest):
            continue
        try:
            if gate.evaluate(run).get("verdict") == "pass":
                return None
        except Exception:  # noqa: BLE001 —— 证据无法判定就是不能放行
            continue
    return "当前规格/代码/用例集/镜像没有门禁通过的 smoke；先跑冒烟再跑正式 test/benchmark"


def benchmark_gate_reason(db: Session, task: Task, *, spec_version_id: int | None,
                          code_snapshot_sha256: str, suite_ref: str | None,
                          criteria_ref: str, image_digest: str | None) -> str | None:
    """profile 只接受当前条件下已通过门禁的 benchmark 证据。

    文档 TC-009 的前置是 TC-008（基准测试）完成；与 smoke 一样，规格、代码快照、
    用例集、判据或镜像任一变化都必须重新跑基准，不能用旧报告给新候选放行。
    """
    from . import gate  # 延迟导入，保持服务模块依赖单向

    rows = db.scalars(
        select(Run).where(Run.task_id == task.id, Run.action == "benchmark").order_by(Run.id.desc())
    )
    for run in rows:
        if run.status != "ok" or run.exit_code not in (None, 0):
            continue
        if spec_version_id is not None and run.spec_version_id != spec_version_id:
            continue
        effective_snapshot = run.scoped_code_snapshot_sha256 or run.code_snapshot_sha256
        effective_suite = run.scoped_suite_ref or run.suite_ref
        if code_snapshot_sha256 and effective_snapshot != code_snapshot_sha256:
            continue
        if suite_ref and effective_suite != suite_ref:
            continue
        if criteria_ref and run.criteria_ref != criteria_ref:
            continue
        if image_digest and not image_digests_match(run.image_digest, image_digest):
            continue
        try:
            if gate.evaluate(run).get("verdict") == "pass":
                return None
        except Exception:  # noqa: BLE001 —— 证据无法判定就是不能放行
            continue
    return "当前规格/代码/用例集/镜像没有门禁通过的 benchmark；先跑基准测试再采集"


@dataclass
class ExecutionPlan:
    """一次真实执行的完整「该跑什么」——可直接落 runs 表。"""

    task_id: int
    action: str
    workspace: str
    command: str
    cwd: str
    remote_dir: str
    code_snapshot_sha256: str
    spec_version_id: int | None = None
    spec_version: int | None = None  # 人看的 vN（不是数据库 id）
    criteria_ref: str | None = None
    criteria_source: str | None = None
    suite_ref: str | None = None
    snapshot_files: int = 0
    snapshot_truncated: bool = False
    host_id: int | None = None
    host_ref: str | None = None  # 实际 SSH 目标（alias 或 user@host），执行侧据此注入 OPX_SSH
    host_ssh_port: int | None = None  # 从 `user@host:port` 拆出的端口；None = ssh 默认
    host_ssh_key: str | None = None  # host.cred_ref 解析后的私钥路径；空 = 全局 OPX_SSH_KEY
    image_id: int | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    env_recipe: str | None = None
    device_id: str | None = None
    device_state: dict | None = None
    lock_mode: str = "exclusive"
    timeout_s: int = 1800
    artifact_types: tuple[str, ...] = ()
    steps: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id, "action": self.action, "workspace": self.workspace,
            "command": self.command, "cwd": self.cwd, "remote_dir": self.remote_dir,
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "snapshot_files": self.snapshot_files, "snapshot_truncated": self.snapshot_truncated,
            "spec_version_id": self.spec_version_id, "spec_version": self.spec_version,
            "criteria_ref": self.criteria_ref,
            "criteria_source": self.criteria_source, "suite_ref": self.suite_ref,
            "host_id": self.host_id, "host_ref": self.host_ref,
            "host_ssh_port": self.host_ssh_port, "host_ssh_key": self.host_ssh_key,
            "image_id": self.image_id, "image_ref": self.image_ref, "image_digest": self.image_digest,
            "env_recipe": self.env_recipe,
            "device_id": self.device_id, "device_state": self.device_state, "lock_mode": self.lock_mode,
            "timeout_s": self.timeout_s,
            "artifact_types": list(self.artifact_types), "steps": self.steps,
            "missing": self.missing, "warnings": self.warnings, "ready": self.ready,
        }


def snapshot(workspace: Path, limit: int = 20000) -> Snapshot:
    """工作区源码快照指纹：按相对路径 + **文件内容** 排序后哈希。

    为什么是内容而不是 mtime：三硬规则要的是「能证明这次跑的就是这版代码」。用 size+mtime 时
    一次 rsync（mtime 保留策略不同）或容器内 touch 就会被判成「版本不一致」，反过来也可能漏改动。
    超过 limit 个文件时**显式**报 truncated（静默截断等于假装哈希了全部内容）。
    """
    h = hashlib.sha256()
    n = 0
    truncated = False
    skip = {"runs", ".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
            ".skills-src", ".opx-chat"}
    entries: list[tuple[str, Path]] = []
    for path in workspace.rglob("*"):
        rel = path.relative_to(workspace)
        if any(part in skip for part in rel.parts):
            continue
        if path.is_file():
            entries.append((rel.as_posix(), path))
    # 按**相对路径字符串**排序：必须与 scripts/nm_test.sh 里那份实现逐字节一致
    # （两边的一致性由 backend/tests/test_snapshot_parity.py 守着）
    for rel, path in sorted(entries, key=lambda e: e[0]):
        if n >= limit:
            truncated = True
            break
        h.update(rel.encode())
        h.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
        n += 1
    return Snapshot(sha256=h.hexdigest(), files=n, truncated=truncated)


def operator_snapshot(workspace: Path, op: str | None, limit: int = 20000) -> Snapshot:
    """任务/算子级代码指纹：只哈希 ``ops/<op>/`` 中会影响执行的文件。

    全工作区 ``snapshot()`` 仍供存点归档使用；run/版本/交付用本函数。不纳入
    ``build/``、缓存和 Markdown 说明文档：它们不是可执行算子输入，更改交付文档不应
    伪造“代码已变”。域分隔符和 op 名纳入哈希，避免两个算子目录内容恰好相同时
    共用一个证据身份。
    """
    name = str(op or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        return Snapshot()
    root = Path(workspace) / "ops" / name
    if not root.is_dir():
        # 兼容旧/最小工作区：尚未生成 ops/<op>/ 时仍给出可诊断的全局快照。
        # 正式任务在交付代码目录存在后一律进入 operator-v1。
        return snapshot(workspace, limit=limit)
    h = hashlib.sha256()
    h.update(b"opx-operator-snapshot/1\0")
    h.update(name.encode())
    h.update(b"\0")
    entries: list[tuple[str, Path]] = []
    skip = {"build", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        if path.is_file() and path.suffix.lower() != ".md":
            entries.append((rel.as_posix(), path))
    n = 0
    truncated = False
    for rel, path in sorted(entries, key=lambda item: item[0]):
        if n >= limit:
            truncated = True
            break
        h.update(rel.encode())
        h.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
        n += 1
    return Snapshot(sha256=h.hexdigest(), files=n, truncated=truncated)


def resolve_criteria(task: Task, project: Project | None) -> tuple[str, str]:
    """判据档：任务 > 项目 > 默认（方案 §11.1「默认官方档，支持项目自定义档」）。"""
    for ref, src in ((task.criteria_ref, "任务"), (getattr(project, "criteria_ref", None), "项目"),
                     (DEFAULT_CRITERIA_REF, "默认")):
        if ref:
            return ref, src
    return DEFAULT_CRITERIA_REF, "默认"


def resolve_host(db: Session, task: Task, project: Project | None) -> tuple[Host | None, str | None]:
    """执行机优先级：任务显式 > 当前测试服务器 > 项目默认。"""
    if task.host_id:
        return db.get(Host, task.host_id), "任务"
    current = db.scalar(
        select(Host).where(Host.is_test_server.is_(True)).order_by(Host.id).limit(1)
    )
    if current is not None:
        return current, "当前测试服务器"
    if project and project.host_id:
        return db.get(Host, project.host_id), "项目默认"
    return None, None


def resolve_image(db: Session, task: Task, project: Project | None) -> tuple[HostImage | None, str | None]:
    """镜像优先级：任务显式 > 项目默认。"""
    if task.image_id:
        return db.get(HostImage, task.image_id), "任务"
    if project and project.image_id:
        return db.get(HostImage, project.image_id), "项目默认"
    return None, None


def resolve_timeout(task: Task, timeout_s: int | None = None) -> int:
    """单次执行超时（秒）：请求体 > 任务配置 > env `OPX_ACTION_TIMEOUT` > 1800。"""
    for candidate in (timeout_s, task.timeout_s):
        try:
            value = int(candidate) if candidate else 0
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    try:
        env_value = int(os.environ.get("OPX_ACTION_TIMEOUT", "1800"))
    except ValueError:
        return 1800
    return env_value if env_value > 0 else 1800


def build_plan(db: Session, task: Task, action: str, *, remote_dir: str | None = None,
               device_id: str | None = None, timeout_s: int | None = None,
               check_device: bool | None = None,
               defer_smoke: bool = False) -> ExecutionPlan:
    """把「这个动作该跑什么」算清楚，并列出缺什么前置（含设备与镜像可用性）。

    **任务配置在这里接线**（否则页面「任务与执行配置」里保存的 device/timeout 是假的）：
    - 设备优先级：`device_id`（请求体/plan） > `task.device_id`（任务配置） > 自动选空卡；
      显式设备（无论来自请求体还是任务配置）同样走占用检查 —— 指定的卡被占就报缺项，不硬挤。
    - 超时优先级：`timeout_s`（请求体/plan） > `task.timeout_s` > env `OPX_ACTION_TIMEOUT` > 1800。
    """
    if action not in ACTIONS:
        raise ValueError(f"未知动作 {action}；合法动作：{', '.join(ACTIONS)}")

    spec = ACTION_SPECS[action]
    project = db.get(Project, task.project_id)
    host, _host_source = resolve_host(db, task, project)
    image, _image_source = resolve_image(db, task, project)

    ws = Path(project.workspace_path).expanduser() if project and project.workspace_path else None
    missing: list[str] = []
    warnings: list[str] = []

    if ws is None or not ws.is_dir():
        missing.append(f"项目工作区不存在：{ws}")
    if host is None:
        missing.append("项目未绑定执行机（hosts）")
    if image is None:
        missing.append("项目/任务未绑定镜像（host_images）")
    elif host is not None and image.host_id != host.id:
        missing.append(f"镜像 #{image.id} 不在执行机「{host.name}」上 —— 换一张该机器上的镜像")
    elif not image.is_op_dev:
        missing.append(f"镜像 #{image.id} 不是「算子开发镜像」（is_op_dev=0）—— 起得来 ≠ 能跑算子")
    if image is not None and not image.is_test:
        warnings.append(f"镜像 #{image.id} 未标记为「测试镜像」—— 任务选择列表默认隐藏未知镜像")
    if image is not None and not (image.env_recipe or "").strip():
        missing.append(f"镜像 #{image.id} 没有 env_recipe（能起 ≠ 能跑算子）")
    if (image is not None and action in ACTIONS_NEEDING_TORCH and not _is_jax_task(task)
            and not (image.torch_version or "").strip()):
        missing.append(f"镜像 #{image.id} 没记录 torch/torch_npu 版本 —— 这个动作需要能跑算子，"
                       f"只有工具链的镜像会直接失败（P1.5-a 实测 VERDICT torch_npu: FAIL）")
    if (image is not None and action in ACTIONS_NEEDING_TORCH and _is_jax_task(task)
            and "jax" not in (image.note or "").lower()):
        missing.append(f"镜像 #{image.id} 没有 JAX/JAXLIB 验收记录 —— jax_* 动作不使用 "
                       "torch_npu 元数据代替 JAX NPU 运行时验收")

    entry = spec["entry"]
    if ws is not None and ws.is_dir() and not (ws / entry).exists():
        missing.append(f"工作区缺入口脚本 {entry}（四动作按约定调用 <动作>.sh）")

    spec_version_id = task.current_spec_version_id
    spec_version = None
    if spec_version_id is not None:
        spec_row = db.get(SpecVersion, spec_version_id)
        spec_version = spec_row.version if spec_row is not None else None
    if "spec_version" in spec["requires"] and spec_version_id is None:
        missing.append("任务还没有规格版本（spec_versions），先落一版再跑这个动作")
    if "case_suite" in spec["requires"] and not (ws and (ws / "cases").is_dir()):
        missing.append("工作区缺用例集目录 cases/（用例矩阵的载体，§4.5 要求逐用例可追溯）")
    if "cases_confirmed" in spec["requires"]:
        cases_reason = cases_gate_reason(task, ws)
        if cases_reason:
            missing.append(cases_reason)

    criteria_ref, criteria_source = resolve_criteria(task, project)
    if "@" not in criteria_ref:
        warnings.append(f"判据档 {criteria_ref!r} 没有版本号（建议写成 档名@档版本，如 {DEFAULT_CRITERIA_REF}）")

    snap = operator_snapshot(ws, task.op_name) if ws and ws.is_dir() else Snapshot()
    if snap.truncated:
        warnings.append("源码快照被截断（文件数超上限）→ manifest 必须带 truncated 标志，别当全量哈希用")
    current_suite_ref = None
    if ws and ws.is_dir():
        try:
            from . import case_confirm
            current_suite_ref = case_confirm.cases_ref(ws, task.op_name)
        except Exception:  # noqa: BLE001 —— case_suite / cases_confirmed 已给更具体缺项
            pass
    # 自动验证链（build→smoke→test→benchmark）在「启动前一次性预检」时还没有 smoke run，
    # 此时必须把 smoke 硬前置延后到链内真实执行 test/benchmark 的那一步再判；
    # 其它调用方（页面手动四动作）保持默认 fail-closed。
    if "smoke_pass" in spec["requires"] and not defer_smoke:
        smoke_reason = smoke_gate_reason(
            db, task, spec_version_id=spec_version_id, code_snapshot_sha256=snap.sha256,
            suite_ref=current_suite_ref, criteria_ref=criteria_ref,
            image_digest=image.digest if image else None,
        )
        if smoke_reason:
            missing.append(smoke_reason)

    if "benchmark_pass" in spec["requires"]:
        bench_reason = benchmark_gate_reason(
            db, task, spec_version_id=spec_version_id, code_snapshot_sha256=snap.sha256,
            suite_ref=current_suite_ref, criteria_ref=criteria_ref,
            image_digest=image.digest if image else None,
        )
        if bench_reason:
            missing.append(bench_reason)

    # 设备占用：共享机上必须做（不检查要么起不来、要么采到被邻居污染的数据）
    # 显式指定（请求体 > 任务配置）的设备也必须过占用检查；自动选空卡只在没指定时生效。
    explicit_device = device_id or task.device_id or None
    device_state = None
    chosen_device = None
    do_check = check_device if check_device is not None else os.environ.get("OPX_DEVICE_CHECK", "1") != "0"
    if host is not None and do_check:
        count = dev.parse_devices_from_cards(host.cards) or 8
        report = dev.check_devices(ssh_target(host), ssh_key(host), settings.ssh_opts, count,
                                   timeout=settings.ssh_timeout, port=ssh_port(host))
        device_state = report.as_dict()
        if not report.ok:
            warnings.append(f"设备占用检查没做成：{report.error}（不算缺项，但没法保证设备空闲）")
        else:
            if explicit_device:
                want = str(explicit_device).strip().lstrip("dev")
                if want.isdigit() and int(want) in report.busy_devices:
                    missing.append(f"设备 {want} 被占（npu-smi 里有进程）—— 换一张空卡再跑")
                else:
                    chosen_device = want
            elif report.free_devices:
                chosen_device = str(report.free_devices[0])
            if spec["needs_device"] and chosen_device is None:
                missing.append(f"没有空闲设备（{count} 张卡上都有别人的进程）—— 等一会儿或换机器")
            elif not spec["needs_device"] and not report.free_devices:
                warnings.append(f"{count} 张卡都被占：编译能跑，但跑完别接着跑 test/benchmark")
            if report.busy_devices and report.free_devices:
                warnings.append(f"共享机现状：空闲 {report.free_devices}、被占 {report.busy_devices}"
                                f"（基准/采集要独占，否则耗时不可比）")
    elif explicit_device:
        # 设备检查关闭（测试/应急）：仍尊重显式指定，只是没法证明这张卡空闲
        chosen_device = str(explicit_device).strip().lstrip("dev")

    rid = remote_dir or f"{settings.remote_run_root.rstrip('/')}/<run_id>"
    # 不在这里手拼 docker run：配方（privileged / ASCEND_RT_VISIBLE_DEVICES / 设备挂载 / set_env 链 /
    # 超时与兜底清理）在 scripts/nm_test.sh 里固化，plan 只注入任务侧参数（含版本与设备）。
    # 四动作脚本按 OPX_OP 找 ops/<op> 与 cases/<op>；不传就会退回 elementwise_add
    env_prefix = [f"OPX_OP={task.op_name}" if task.op_name else ""]
    env_prefix.append(f"OPX_REMOTE_ROOT={shlex.quote(str(Path(rid).parent))}")
    env_prefix += [f"OPX_IMAGE={image.image_ref}:{image.tag or 'latest'}" if image else ""]
    if image is not None:
        env_prefix += [f"OPX_IMAGE_ID={image.id}", f"OPX_HOST_ID={host.id if host else ''}"]
    if chosen_device:
        env_prefix.append(f"OPX_DEVICES={chosen_device}")
    env_prefix += [f"OPX_TASK_ID={task.id}", f"OPX_SPEC_VERSION_ID={spec_version_id or ''}",
                   f"OPX_CRITERIA_REF={criteria_ref}"]
    if action in ("benchmark", "profile"):
        env_prefix.append("OPX_LOCK_MODE=exclusive")
    command = " ".join(x for x in env_prefix if x and not x.endswith("=")) \
        + f" bash scripts/nm_test.sh --run-id <run_id> --action {action}"

    return ExecutionPlan(
        task_id=task.id,
        action=action,
        workspace=str(ws) if ws else "",
        command=command,
        cwd=str(ws) if ws else "",
        remote_dir=rid,
        code_snapshot_sha256=snap.sha256,
        snapshot_files=snap.files,
        snapshot_truncated=snap.truncated,
        spec_version_id=spec_version_id,
        spec_version=spec_version,
        criteria_ref=criteria_ref,
        criteria_source=criteria_source,
        suite_ref=current_suite_ref,
        host_id=host.id if host else None,
        host_ref=ssh_target(host) if host else None,
        host_ssh_port=ssh_port(host) if host else None,
        host_ssh_key=ssh_key(host) if host else None,
        image_id=image.id if image else None,
        image_ref=(f"{image.image_ref}:{image.tag}" if image else None),
        image_digest=image.digest if image else None,
        env_recipe=image.env_recipe if image else None,
        device_id=chosen_device,
        device_state=device_state,
        timeout_s=resolve_timeout(task, timeout_s),
        artifact_types=tuple(spec["artifact_types"]),
        steps=[
            f"① 检查前置：{', '.join(spec['requires'])}"
            + (f"（设备 {chosen_device} 空闲）" if chosen_device else ""),
            f"② 固定版本：源码快照 {snap.sha256[:12]}…（{snap.files} 个文件"
            + ("，已截断" if snap.truncated else "")
            + f"）+ 规格 v{spec_version if spec_version is not None else '?'} + 判据 {criteria_ref}（{criteria_source}）",
            f"③ 同步快照到远端 {rid}/ws",
            f"④ 容器内执行：bash <工作区>/{entry}（配方走 scripts/nm_test.sh，含设备互斥与超时清理）",
            f"⑤ 回读 {rid}/out → {ws}/runs/<run_id>/out" if ws else "⑤ 回读产物",
            "⑥ 写 manifest（含镜像 digest / 快照 / 判据 / 设备锁）+ 登记 runs + 门禁校验（gate.evaluate）",
        ],
        missing=missing,
        warnings=warnings + [spec["remote_note"]],
    )


class RemoteExecutionNotReady(RuntimeError):
    """远端四动作适配器入口（保留给同步调用方；异步路径见 services/action_runner.py）。"""


def execute(plan: ExecutionPlan, *, confirm: bool = False) -> None:
    """同步执行入口（已由 `services/action_runner.py` 的异步作业取代）。

    真执行必须从**事件循环**里起子进程并立刻返回 job_id（前端要轮询进度），所以路由直接调
    `action_runner.start()`。这里保留一个明确报错的同步入口，避免旧调用方静默走到「什么都不做」。
    """
    if not plan.ready:
        raise RemoteExecutionNotReady("前置不齐，先补齐：" + "；".join(plan.missing))
    raise RemoteExecutionNotReady(
        "四动作已改为异步作业：POST /api/tasks/{id}/actions 时带 dry_run=false，"
        "然后用 GET /api/action-jobs/{job_id} 看进度与门禁结论。"
    )


def cancel(run_id: int) -> str:
    """取消：必须传到远端容器/进程，不能只杀本地会话（方案 v1.1 §6）。

    脚本侧已具备（`nm_test.sh` 用 `timeout` 包住容器执行 + 显式 `docker rm -f` 兜底清理，P1.5-a 实测），
    缺的是**后端这条路径**：记录容器名/远端 run 目录 → ssh kill + 清理结果回写 `runs.note`。
    """
    return f"run#{run_id} 取消：后端 cancel 未实装（远端侧已有超时 + 兜底清理；随四动作适配器一起接通）"
