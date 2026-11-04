"""设备占用检查与「按设备互斥」（方案 v1.1 §1.2 / §4.6 / §6 的最小实现）。

背景（都是实测代价换来的）：
- 远端裸金属是**共享机**：别人的容器在跑，卡会占着。不检查就直接起容器 → 实测
  `dcmi ... device is used ret=-8020`（自己堵自己的路）。
- 就算起得来，邻居在算 → 采到的耗时不干净；而方案 §4.6 要求「同条件对比 + 超过测量不确定性」，
  所以「这张卡是不是独占」必须当成测量条件记下来。
- **HBM 占用不能当判据**：实测 8 张卡都有 ~3.4 GB 常驻（无进程）→ 有进程才算被占。
  判据用 `npu-smi info -t proc-mem -i <n>` 的输出里有没有进程（实测空闲时是 "No process in device."）。

分工：
- 本模块 = **只读检查**（给 plan 的前置检查用）+ 选卡；
- 真正加锁在执行侧：`scripts/nm_test.sh` 在远端用 `flock` 拿 `$OPX_REMOTE_ROOT/.locks/dev<N>.lock`，
  拿不到就报「设备被占」而不是硬跑；锁模式（独占 / 共享）写进 manifest 与 `runs.device_id`。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

from .ssh import validate_ssh_target

#: 空闲时的输出特征（实测）
NO_PROCESS_MARK = "No process in device"

_CHIP_ROW = re.compile(r"^\|\s*(\d+)\s+([0-9A-Za-z_]+)\s*\|", re.MULTILINE)


def parse_chip_count(text: str) -> int:
    """从 `npu-smi info` 里数芯片行（`| 0     910B1 | OK | ...`）。"""
    return len(_CHIP_ROW.findall(text))


def parse_devices_from_cards(cards: str | None) -> int:
    """从 hosts.cards（如 `8×910B1`）里取设备数；解析不出返回 0。"""
    m = re.match(r"\s*(\d+)\s*[×x]\s*", cards or "")
    return int(m.group(1)) if m else 0


def parse_proc_mem(text: str) -> dict[int, tuple[bool, str]]:
    """解析 `npu-smi info -t proc-mem -i <n>` 的输出 → {device: (是否空闲, 原始片段)}。

    实测空闲输出：`NPU ID : 0 / Chip Count : 1 / No process in device.`
    有进程时会列出进程行（含 PID/进程名）→ 视为被占。
    """
    out: dict[int, tuple[bool, str]] = {}
    blocks = re.split(r"(?=NPU ID\s*:)", text)
    for b in blocks:
        m = re.search(r"NPU ID\s*:\s*(\d+)", b)
        if not m:
            continue
        idx = int(m.group(1))
        snippet = " ".join(b.split())[:400]
        out[idx] = (NO_PROCESS_MARK in b, snippet)
    return out


PROC_MEM_LOOP = (
    'for i in $(seq 0 {last}); do echo "NPU ID : $i"; '
    "npu-smi info -t proc-mem -i $i 2>&1 | head -30; done"
)


def ssh_check_devices(alias: str, key: str, opts: str, count: int, timeout: int = 20,
                      port: int | None = None) -> tuple[int, str]:
    """一条 ssh 把 0..count-1 的进程占用全部拉回来（只读，不动服务器）。"""
    # 与 probe.ssh_exec 同口径：`-` 开头的目标会被 ssh 当选项 → 拒绝（审查 M1）
    try:
        safe_alias = validate_ssh_target(alias)
    except ValueError:
        return 2, "invalid ssh target"
    if not safe_alias:
        return 2, "invalid ssh target"
    cmd = PROC_MEM_LOOP.replace("{last}", str(max(count - 1, 0)))
    args = ["ssh", *opts.split()]
    if port:
        args += ["-p", str(port)]
    if key:
        args += ["-i", key]
    args += [safe_alias, cmd]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"
    except OSError as e:  # ssh 不存在等，别让 plan 500
        return 127, f"[ssh 执行失败：{e}]"
    return p.returncode, (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")


@dataclass
class DeviceState:
    index: int
    free: bool
    detail: str = ""


@dataclass
class DeviceReport:
    """一次设备占用检查的结果。`ok=False` 表示「没查成」（网络/命令问题），不是「都被占」。"""

    ok: bool = False
    devices: list[DeviceState] = field(default_factory=list)
    error: str | None = None

    @property
    def free_devices(self) -> list[int]:
        return [d.index for d in self.devices if d.free]

    @property
    def busy_devices(self) -> list[int]:
        return [d.index for d in self.devices if not d.free]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "free": self.free_devices,
            "busy": self.busy_devices,
            "devices": [{"index": d.index, "free": d.free, "detail": d.detail} for d in self.devices],
        }


def check_devices(alias: str, key: str, opts: str, count: int, timeout: int = 20,
                  port: int | None = None) -> DeviceReport:
    # 防御：目标非法时直接判「没查成」，别把 `-oProxyCommand=...` 送进 ssh（审查 M1）
    try:
        validate_ssh_target(alias)
    except ValueError:
        return DeviceReport(ok=False, error="invalid ssh target")
    if count <= 0:
        return DeviceReport(ok=False, error="设备数未知（hosts.cards 解析不出，先跑一次探针）")
    rc, text = ssh_check_devices(alias, key, opts, count, timeout, port=port)
    if rc != 0:
        return DeviceReport(ok=False, error=f"设备占用检查失败（ssh rc={rc}）：{text[:200]}")
    parsed = parse_proc_mem(text)
    if not parsed:
        return DeviceReport(ok=False, error=f"没能解析 npu-smi proc-mem 输出：{text[:200]}")
    return DeviceReport(
        ok=True,
        devices=[DeviceState(i, *parsed.get(i, (False, "（该设备无输出）"))) for i in range(count)],
    )
