"""源码快照：内容哈希 + 截断标志，并且**必须**与 scripts/nm_test.sh 里那份实现逐字节一致。

为什么要这个测试：两处实现（后端算 plan 用、脚本写 manifest 用）一旦漂移，
「run 行的快照 ≠ manifest 的快照」会被门禁判成版本不一致，而且很难查。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.services import exec_actions

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nm_test.sh"
MARK = "SNAPSHOT_JSON=$(/usr/bin/python3 - \"$OPX_WORKSPACE\" \"$OPX_OP\" <<'PY'"


def _shell_snippet() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index(MARK)
    body = text[start:].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert "hashlib" in body
    return body


def test_content_hash_ignores_mtime(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    first = exec_actions.snapshot(tmp_path)
    import os
    import time

    time.sleep(0.01)
    os.utime(tmp_path / "a.txt", None)  # 只动 mtime
    second = exec_actions.snapshot(tmp_path)
    assert first.sha256 == second.sha256, "只改 mtime 不该改变快照（旧实现用 size+mtime，会误判版本不一致）"

    (tmp_path / "a.txt").write_text("hello!", encoding="utf-8")
    assert exec_actions.snapshot(tmp_path).sha256 != first.sha256


def test_truncation_is_explicit(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(str(i), encoding="utf-8")
    snap = exec_actions.snapshot(tmp_path, limit=3)
    assert snap.truncated is True and snap.files == 3
    assert exec_actions.snapshot(tmp_path, limit=50).truncated is False


def test_skips_runtime_dirs(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.cpp").write_text("x", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "artifact.log").write_text("noise", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("noise", encoding="utf-8")
    # 对话附件/scratch 也是运行时数据：两边快照实现必须都跳过（2026-10-21 实测漂移：
    # nm_test.sh 少跳过 .opx-chat，导致同一次执行「基线存档 sha ≠ run sha」）。
    (tmp_path / ".opx-chat" / "conv-1" / "attachments").mkdir(parents=True)
    (tmp_path / ".opx-chat" / "conv-1" / "attachments" / "a.bin").write_text("noise", encoding="utf-8")
    snap = exec_actions.snapshot(tmp_path)
    assert snap.files == 1


def test_parity_with_nm_test_script(tmp_path: Path, workspace: Path):
    """两边实现必须给出同一个哈希（同一工作区、同一排序、同一 skip 规则）。"""
    # 放进对话附件目录：两处都必须跳过；少跳过一处就会哈希不一致（历史回归点）。
    (workspace / ".opx-chat" / "conv-1" / "attachments").mkdir(parents=True)
    (workspace / ".opx-chat" / "conv-1" / "attachments" / "a.txt").write_text("noise", encoding="utf-8")
    code = _shell_snippet()
    op = "elementwise_add"
    op_root = workspace / "ops" / op
    op_root.mkdir(parents=True)
    (op_root / "kernel.asc").write_text("// kernel\n", encoding="utf-8")
    (op_root / "README.md").write_text("ignored docs\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-c", code, str(workspace), op],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    shell = json.loads(proc.stdout)
    py = exec_actions.operator_snapshot(workspace, op)
    assert shell["sha256"] == py.sha256, f"两处快照实现不一致：{shell} vs {py.as_dict()}"
    assert shell["files"] == py.files
    assert shell["truncated"] == py.truncated


@pytest.mark.parametrize("limit", [1, 2])
def test_parity_with_truncation(tmp_path: Path, limit: int):
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("b.txt", "a.txt", "c/d.txt"):
        p = ws / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding="utf-8")
    code = _shell_snippet().replace("LIMIT = 20000", f"LIMIT = {limit}")
    proc = subprocess.run([sys.executable, "-c", code, str(ws), ""], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    shell = json.loads(proc.stdout)
    py = exec_actions.snapshot(ws, limit=limit)
    assert (shell["sha256"], shell["files"], shell["truncated"]) == (py.sha256, py.files, py.truncated)


def test_operator_snapshot_isolated_from_other_operator_and_docs(tmp_path: Path):
    current = tmp_path / "ops" / "op_a"
    other = tmp_path / "ops" / "op_b"
    current.mkdir(parents=True)
    other.mkdir(parents=True)
    (current / "kernel.asc").write_text("v1", encoding="utf-8")
    (current / "README.md").write_text("docs v1", encoding="utf-8")
    (other / "kernel.asc").write_text("other v1", encoding="utf-8")
    first = exec_actions.operator_snapshot(tmp_path, "op_a")

    (other / "kernel.asc").write_text("other v2", encoding="utf-8")
    (current / "README.md").write_text("docs v2", encoding="utf-8")
    assert exec_actions.operator_snapshot(tmp_path, "op_a").sha256 == first.sha256

    (current / "kernel.asc").write_text("v2", encoding="utf-8")
    assert exec_actions.operator_snapshot(tmp_path, "op_a").sha256 != first.sha256
