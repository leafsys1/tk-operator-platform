"""任务工作区只读浏览（P1「实现与接口」分区用）。

边界：**只读**、只允许工作区内路径、跳过运行时大件（`.dsh` / `.cannbot` / `runs` / `build` / `.git`）；
文件内容与 git diff 都有大小上限 —— 这是给人看的，不是文件管理器。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SKIP_DIRS = {"runs", ".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
             ".skills-src", ".opx-chat", "build"}
SKIP_FILES = {".DS_Store"}
MAX_FILES = 2000
MAX_TEXT_BYTES = 512 * 1024
MAX_DIFF_BYTES = 200 * 1024


class WorkspaceError(ValueError):
    """路径越界 / 不是文件 / 不能读。"""


def _root(workspace: Path) -> Path:
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise WorkspaceError(f"工作区不存在：{root}")
    return root


def safe_path(workspace: Path, rel: str) -> Path:
    """把相对路径解析到工作区内；拒绝绝对路径与 `..`。"""
    root = _root(workspace)
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise WorkspaceError(f"路径越界：{rel}")
    return candidate


def _skipped(rel: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel.parts)


def list_files(workspace: Path, limit: int = MAX_FILES) -> dict:
    """文件清单：遍历时**直接剪掉** SKIP_DIRS（不进入 .git 等大目录），到上限立即停。

    旧实现先 `sorted(root.rglob("*"))` 把整棵树走完再过滤，`limit` 只限制返回条数、
    不限制遍历成本（审查 #11：56 个文件的工作区实测 ~1s）。这里用 os.walk + `dirnames[:]`
    剪枝，`truncated` 恢复「到上限就停」的语义。
    """
    root = _root(workspace)
    rows: list[dict] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name in SKIP_FILES:
                continue
            if len(rows) >= limit:
                truncated = True
                break
            p = Path(dirpath) / name
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            rows.append({"path": p.relative_to(root).as_posix(), "size_bytes": size})
        if truncated:
            break
    rows.sort(key=lambda r: r["path"])  # 输出顺序稳定（与旧的 sorted 遍历一致好对比）
    return {"root": str(root), "files": rows, "truncated": truncated, "count": len(rows)}


def read_text(workspace: Path, rel: str) -> dict:
    p = safe_path(workspace, rel)
    if not p.is_file():
        raise WorkspaceError(f"不是文件：{rel}")
    size = p.stat().st_size
    if size > MAX_TEXT_BYTES:
        return {"path": rel, "size_bytes": size, "truncated": True, "text": p.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", "replace")}
    data = p.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"path": rel, "size_bytes": size, "binary": True, "text": None}
    return {"path": rel, "size_bytes": size, "binary": False, "text": text}


def git_view(workspace: Path) -> dict:
    """工作区的 git 状态与已跟踪文件的 diff（只读；工作区可能不是独立仓库）。"""
    root = _root(workspace)
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"is_repo": False, "error": str(e)}
    if top.returncode != 0:
        return {"is_repo": False, "error": (top.stderr or "").strip()[:300]}
    repo = Path(top.stdout.strip())
    rel = root.relative_to(repo).as_posix() if root != repo else "."

    def run(args: list[str]) -> str:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=15)
        return (r.stdout or "") + (("\n" + r.stderr) if r.returncode != 0 and r.stderr else "")

    try:
        branch = run(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        status = [line for line in run(["status", "--porcelain", "--", rel]).splitlines() if line.strip()]
        diff = run(["diff", "--", rel])[:MAX_DIFF_BYTES]
    except (OSError, subprocess.TimeoutExpired) as e:
        # 慢盘/超大仓库上 git 可能超时：返回可读错误，别让「实现与调用」分区变成 500
        return {
            "is_repo": True,
            "repo_root": str(repo),
            "path": rel,
            "branch": "",
            "dirty": False,
            "status": [],
            "diff": "",
            "truncated": False,
            "error": f"git 读取失败：{type(e).__name__}: {e}",
        }
    return {
        "is_repo": True,
        "repo_root": str(repo),
        "path": rel,
        "branch": branch,
        "dirty": bool(status),
        "status": status[:200],
        "diff": diff,
        "truncated": len(diff) >= MAX_DIFF_BYTES,
    }
