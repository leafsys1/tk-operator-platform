"""测试共用的路径解析 —— 让 tests/ 目录里也能找到 shell/ 下的源文件。

原始开发时 gateway.py / portal.html 与测试同目录；入库后按 shell/ + shell/tests/
分层。这里两个位置都找，避免因为挪目录而改测试逻辑。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def asset(name: str) -> Path:
    """返回 shell/ 下某个文件的路径（优先 tests/ 同级，再退回上一级）。"""
    for base in (HERE, ROOT):
        p = base / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"{name} 既不在 {HERE} 也不在 {ROOT} —— 仓库结构被改动过？")


def text(name: str) -> str:
    return asset(name).read_text(encoding="utf-8")
