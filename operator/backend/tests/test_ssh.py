"""连接目标解析与校验：alias / user@host / user@host:port / ssh:// / IPv6。

2026-10-21 实测踩到：云主机控制台给的 `root@1.2.3.4:50023` 直接当 ssh 目标会
DNS 解析挂住（25s → rc=124），必须拆成 `root@1.2.3.4` + `-p 50023`。

校验（任务外页面审查 M1）：`-` 开头的目标会被 ssh 当选项（`-oProxyCommand=…` → 本机命令执行），
空白/控制字符同理，写入口一律 400、ssh_exec 侧再兜一层。
"""

import pytest

from app.services.ssh import split_ssh_target, validate_ssh_target


def test_split_ssh_target_forms():
    assert split_ssh_target("tk36") == ("tk36", None)
    assert split_ssh_target("root@1.2.3.4") == ("root@1.2.3.4", None)
    assert split_ssh_target("root@1.2.3.4:50023") == ("root@1.2.3.4", 50023)
    assert split_ssh_target("ssh://root@1.2.3.4:50023/") == ("root@1.2.3.4", 50023)
    assert split_ssh_target("[::1]:2222") == ("::1", 2222)
    assert split_ssh_target("::1") == ("::1", None)  # 裸 IPv6 不误切
    assert split_ssh_target("host:notaport") == ("host:notaport", None)
    assert split_ssh_target("") == ("", None)


def test_validate_ssh_target_accepts_supported_forms():
    assert validate_ssh_target(None) is None
    assert validate_ssh_target("   ") is None
    assert validate_ssh_target(" tk36 ") == "tk36"
    assert validate_ssh_target("root@1.2.3.4:50023") == "root@1.2.3.4:50023"
    assert validate_ssh_target("ssh://root@1.2.3.4:50023/") == "ssh://root@1.2.3.4:50023/"
    assert validate_ssh_target("[::1]:2222") == "[::1]:2222"


@pytest.mark.parametrize(
    "bad",
    [
        "-oProxyCommand=touch /tmp/pwn",  # 会被 ssh 当选项 → 本机命令执行
        "-V",
        "root@host extra",  # 空白（argv/脚本注入面）
        "root@host\n-oProxyCommand=x",  # 换行
        "root@ho\tst",
        "bad\x00target",  # 控制字符
        "x" * 401,  # 超长
    ],
)
def test_validate_ssh_target_rejects_injection(bad):
    with pytest.raises(ValueError):
        validate_ssh_target(bad)
