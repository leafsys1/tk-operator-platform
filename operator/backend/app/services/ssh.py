"""SSH 连接目标与私钥解析 —— 探针 / 设备检查 / 四动作真执行 / 取消的**唯一真相源**。

页面登记的 host 必须走同一套解析，否则会出现「页面选 A、实际跑 B」：
`OPX_SSH` / `OPX_SSH_PORT` / `OPX_SSH_KEY` 由 `action_runner.build_env` 从这里取，
脚本里的 `ops/nm.conf` 只作为没有登记 host 时的回退。

支持的连接目标写法（云主机控制台常直接给 `root@1.2.3.4:50023`）：
- `alias` / `user@host`（走 `~/.ssh/config`，端口由 config 决定）
- `user@host:port` / `host:port`
- `ssh://user@host:port`（可带末尾 `/`）
- `[::1]:port`（IPv6）
"""

from pathlib import Path

from ..config import settings
from ..models import Host

#: SSH 目标长度上限（hosts.ssh_alias=120 / address=200，留余量挡住异常超长串）
_MAX_TARGET_LEN = 400


def validate_ssh_target(value: str | None) -> str | None:
    """校验页面/API 填的 SSH 连接目标；合法则返回 strip 后的值，空值返回 None。

    为什么必须有（任务外页面审查 2026-10-21 M1）：`ssh` 会把以 `-` 开头的目标当选项解析，
    `-oProxyCommand=<cmd>` 等于在控制台主机上执行任意命令。含空白/控制字符同理（argv 拼接、
    日志/脚本注入面），一律拒绝。端口不在这里校验 —— 由 `split_ssh_target` 从目标里拆。

    非法时抛 `ValueError`（路由层转 400），合法时绝不抛。
    """
    target = (value or "").strip()
    if not target:
        return None
    if target.startswith("-"):
        raise ValueError("SSH 连接目标不能以 - 开头（会被 ssh 当成选项解析）")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in target):
        raise ValueError("SSH 连接目标不能含空白或控制字符")
    if len(target) > _MAX_TARGET_LEN:
        raise ValueError(f"SSH 连接目标过长（>{_MAX_TARGET_LEN}）")
    return target


def split_ssh_target(value: str | None) -> tuple[str, int | None]:
    """把页面填的连接目标拆成 `(ssh 目标, 端口)`；端口缺省返回 None。

    裸 IPv6（如 `::1`）含多个冒号且没有方括号，**不当端口**解析。
    """
    target = (value or "").strip()
    if target.lower().startswith("ssh://"):
        target = target[6:].rstrip("/")
    if not target:
        return "", None

    # 方括号 IPv6：[::1]:50023 → ssh 目标用裸 ::1
    if target.startswith("["):
        close = target.find("]")
        if close != -1:
            host = target[1:close]
            rest = target[close + 1:]
            if rest.startswith(":") and rest[1:].isdigit():
                return host, int(rest[1:])
            return host, None

    # user@host:port / host:port —— 只处理「恰好一个冒号」的情况，避免误切裸 IPv6
    if target.count(":") == 1:
        head, port = target.rsplit(":", 1)
        if port.isdigit() and 0 < int(port) < 65536:
            return head, int(port)
    return target, None


def _host_endpoint(host: Host) -> str:
    return host.ssh_alias or host.address or host.name


def ssh_target(host: Host) -> str:
    """连接目标（不含端口）：ssh_alias 优先，其次 address，最后退回 name（兼容老数据）。"""
    return split_ssh_target(_host_endpoint(host))[0]


def ssh_port(host: Host) -> int | None:
    """连接端口：从 `user@host:port` / `ssh://…` 里解析；没有就返回 None（走 ssh 默认/配置）。"""
    return split_ssh_target(_host_endpoint(host))[1]


def ssh_key(host: Host | None) -> str:
    """私钥路径：host.cred_ref 优先，未填回退全局 `OPX_SSH_KEY`；统一展开 `~`。

    注意：展开必须在这里做 —— 这些值最终会以**带引号**的形式进 ssh 命令行，
    bash 不会替我们展开 `~`。
    """
    ref = (host.cred_ref if host is not None else None) or settings.ssh_key
    return str(Path(ref).expanduser())
