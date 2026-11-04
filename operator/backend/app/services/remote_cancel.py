"""取消真实执行：把 `docker rm -f opx-<run_ref>` 送到远端（方案 §6「取消要传到远端」）。

本地只杀进程组不够：容器可能继续跑并占卡。脚本自身有 timeout + 兜底清理，
这里补一条**立即清容器**的路径；拿不到 host/run_ref 时明确返回原因。
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Host, Project, Task
from . import exec_actions
from .ssh import ssh_key, ssh_port, ssh_target, validate_ssh_target


def _target(host: Host) -> str:
    return ssh_target(host)


def cancel_container(db: Session, task: Task, run_ref: str | None) -> dict:
    if not run_ref:
        return {"ok": False, "why": "没有 run_ref（这次执行没有远端目录）"}
    project = db.get(Project, task.project_id) if task.project_id else None
    # 与 build_plan 同口径：任务显式 > 当前测试服务器 > 项目默认
    host, _source = exec_actions.resolve_host(db, task, project)
    if host is None:
        return {"ok": False, "why": "任务/项目没有绑定执行机"}
    target = _target(host)
    # 历史脏数据兜底：写入口已校验，但老库里可能仍有以 `-` 开头的目标（会被 ssh 当选项）
    try:
        validate_ssh_target(target)
    except ValueError as e:
        return {"ok": False, "why": f"执行机连接目标非法：{e}"}
    name = f"opx-{run_ref}"
    remote = f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true"
    cmd = ["ssh", *shlex.split(settings.ssh_opts)]
    port = ssh_port(host)
    if port:
        cmd += ["-p", str(port)]
    key = Path(ssh_key(host))
    if key.is_file():
        cmd += ["-i", str(key)]
    cmd += [target, remote]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "why": f"ssh 执行失败：{e}", "target": target, "container": name}
    return {
        "ok": proc.returncode == 0,
        "target": target,
        "container": name,
        "exit_code": proc.returncode,
        "output": ((proc.stdout or "") + (proc.stderr or "")).strip()[:500],
    }
