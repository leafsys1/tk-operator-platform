"""裸金属探针：SSH 过去拿真实信息（卡型/数量/健康、HBM、CANN 版本、docker 与镜像清单）。

只读：不装任何东西、不动任何现有镜像。所有解析都对不上时保留原始输出（截断）便于排错。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

from .ssh import validate_ssh_target

_SIZE_UNITS = {"B": 1, "kB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15}

NPU_SMI_CMD = "npu-smi info 2>&1 | head -80"
CANN_VER_CMD = (
    "for p in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/latest; do "
    "v=$(basename \"$(readlink -f $p 2>/dev/null)\" 2>/dev/null); "
    "[ -n \"$v\" ] && [ \"$v\" != \".\" ] && { echo \"$v\"; break; }; done"
)
DOCKER_CMD = "docker images --format '{{.Repository}}\\t{{.Tag}}\\t{{.ID}}\\t{{.CreatedAt}}\\t{{.Size}}' 2>&1 | head -200"
HOSTNAME_CMD = "hostname"


def ssh_exec(alias: str, command: str, key: str, opts: str, timeout: int = 25,
             port: int | None = None) -> tuple[int, str]:
    # 防御：以 `-` 开头的目标会被 ssh 当选项（`-oProxyCommand=…` → 本机命令执行，审查 M1）。
    # 正常入口（hosts create/update）已经 400 拦掉，这里兜住历史数据/脚本/其他调用方。
    try:
        safe_alias = validate_ssh_target(alias)
    except ValueError:
        return 2, "invalid ssh target"
    if not safe_alias:
        return 2, "invalid ssh target"
    args = ["ssh", *opts.split()]
    if port:
        args += ["-p", str(port)]
    if key:
        args += ["-i", key]
    args += [safe_alias, command]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"
    except OSError as e:  # ssh 不存在 / 无执行权限等：返回错误码而不是冒泡 500（审查 hosts #10）
        return 127, f"[ssh 执行失败：{e}]"
    out = p.stdout or ""
    if p.stderr:
        out += ("\n" if out else "") + p.stderr
    return p.returncode, out.strip()


@dataclass
class NpuInfo:
    chips: list[dict] = field(default_factory=list)
    hbm_total_mb: int | None = None
    hbm_used_mb: int | None = None
    smi_version: str | None = None
    raw: str = ""

    @property
    def cards(self) -> str | None:
        if not self.chips:
            return None
        names: dict[str, int] = {}
        for c in self.chips:
            names[c["name"]] = names.get(c["name"], 0) + 1
        # 名称数量最多的那一款
        name, n = max(names.items(), key=lambda kv: kv[1])
        return f"{n}×{name}"


def parse_npu_smi(text: str) -> NpuInfo:
    info = NpuInfo(raw=text)
    m = re.search(r"npu-smi\s+([\w.\-]+)", text)
    if m:
        info.smi_version = m.group(1)
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        # 新版 npu-smi 会在设备概览后继续输出进程表。进程行同样以
        # ``| <NPU> <Chip> | <pid> ...`` 开头，会被下面的旧格式正则误认成
        # 名为 ``0`` 的芯片（进程数多时最终显示成 ``10×0``）。进程表开始后
        # 已没有设备概览/HBM 信息，直接停止解析。
        if "Process id" in line:
            break
        chip = re.match(r"^\|\s*(\d+)\s+([A-Za-z0-9_]+)\s*\|\s*(\w+)", line)
        if chip:
            info.chips.append({"index": int(chip.group(1)), "name": chip.group(2), "health": chip.group(3)})
            continue
        # 设备行（含 bus-id）末尾是 "<used> / <total>" 的 HBM
        if re.search(r"\d{4}:[0-9A-Fa-f]{2}:\d{2}\.\d", line):
            hbm = re.findall(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
            if hbm:
                used, total = (int(hbm[0][0]), int(hbm[0][1]))
                info.hbm_used_mb = used
                info.hbm_total_mb = total if info.hbm_total_mb is None else max(info.hbm_total_mb, total)
    return info


def parse_docker_images(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[0].startswith("REPOSITORY") or "command not found" in line:
            continue
        repo, tag, image_id, created, size = (p.strip() for p in parts[:5])
        if not repo:
            continue
        dangling = repo == "<none>" or tag == "<none>"
        out.append(
            {
                "image_ref": repo,
                "tag": None if tag == "<none>" else tag,
                "image_id": image_id[:24] if image_id else None,
                "created_raw": created,
                "size_bytes": parse_size(size),
                "dangling": dangling,
            }
        )
    return out


def parse_size(s: str) -> int | None:
    m = re.match(r"^([\d.]+)\s*([A-Za-z]+)$", s.strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    factor = _SIZE_UNITS.get(unit)
    return int(value * factor) if factor else None


@dataclass
class ProbeResult:
    ok: bool
    hostname: str | None = None
    cards: str | None = None
    cann_version: str | None = None
    docker_ok: bool = False
    npu_raw: str = ""
    images: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def probe_host(alias: str, key: str, opts: str, timeout: int = 25,
               port: int | None = None) -> ProbeResult:
    res = ProbeResult(ok=False)
    rc, out = ssh_exec(alias, HOSTNAME_CMD, key, opts, timeout, port=port)
    if rc != 0:
        res.errors.append(f"ssh 失败（rc={rc}）：{out[:300]}")
        return res
    res.hostname = out.splitlines()[0].strip() if out else None

    rc, npu_out = ssh_exec(alias, NPU_SMI_CMD, key, opts, timeout, port=port)
    if rc == 0 and "npu-smi" in npu_out:
        info = parse_npu_smi(npu_out)
        res.cards = info.cards
        res.npu_raw = npu_out[:4000]
    else:
        res.errors.append(f"npu-smi 不可用（rc={rc}）：{npu_out[:200]}")

    rc, cann_out = ssh_exec(alias, CANN_VER_CMD, key, opts, timeout, port=port)
    if rc == 0 and cann_out and cann_out != ".":
        res.cann_version = cann_out.strip().splitlines()[0][:120]

    rc, dock_out = ssh_exec(alias, DOCKER_CMD, key, opts, timeout, port=port)
    # docker 命令成功即「docker 可用」：0 个镜像也是可用（新机器/清空镜像后必现，审查 M8）。
    # 注意远端命令带了 `| head`，管道 rc 是 head 的（docker 失败也可能 rc=0）→
    # 输出非空却解析不出镜像行时按失败处理，避免把「docker 不可用」误判成「0 个镜像」而清空清单。
    if rc == 0:
        if "\t" in dock_out:
            res.images = parse_docker_images(dock_out)
            res.docker_ok = True
        elif not dock_out.strip():
            res.docker_ok = True
        else:
            res.errors.append(f"docker images 输出无法解析（rc=0）：{dock_out[:200]}")
    else:
        res.errors.append(f"docker images 失败（rc={rc}）：{dock_out[:200]}")

    # 只要 ssh 通、且至少拿到卡或镜像之一，就算探针成功（部分成功也保留）
    res.ok = bool(res.cards or res.images)
    return res
