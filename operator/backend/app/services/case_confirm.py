"""用例组确认（任务生命周期 v2 §3.3 / §5.1）。

**唯一的用例集哈希实现**：`cases_suite_sha256()` 必须与 `scripts/nm_test.sh` 里那段
「用例集 ref」的 Python 逐字节一致 —— 任务证据只遍历 `<workspace>/cases/<op>/`，
每个目录先 `dirnames.sort()`，每个目录内 `filenames` 排序（`os.walk` 深度优先），
对每个文件依次 `update(相对路径 bytes)`、`update(b"\\0")`、`update(内容按 1MB 分块)`、`update(b"\\0")`。

`suite_ref = "cases@<sha256 hex 前 16 位>"`；`tasks.cases_confirmed_sha256` 存**完整 64 hex**
（列宽 80，够；`confirmation()` 用它判断 in_sync）。

供 versions / delivery / exec_actions plan / 对话 prompt 共用，别在其它文件里复算一遍。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import Project, StageEvent, Task, utcnow

#: 与 nm_test.sh 一致的分块大小
_CHUNK = 1 << 20
#: suite_ref 前缀（契约 §5.1）
_SUITE_REF_PREFIX = "cases@"
#: 用例文件名（契约 §3.3）
CASES_FILENAME = "cases.json"
BENCHMARK_FILENAME = "benchmark.json"


class CaseConfirmError(ValueError):
    """用例组确认失败（文件缺失 / 哈希不匹配 / 工作区不可用）→ 路由映射 409。"""


# ---------------------------------------------------------------------------
# 路径 / 哈希
# ---------------------------------------------------------------------------

def workspace_of(db: Session, task: Task) -> Path | None:
    """任务所属项目的工作区；未配置返回 None（不校验目录是否存在）。"""
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        return None
    return Path(project.workspace_path).expanduser()


def case_dir(workspace: Path, op: str) -> Path:
    return workspace / "cases" / op


def case_files(workspace: Path, op: str) -> tuple[Path, Path]:
    """`(cases/<op>/cases.json, cases/<op>/benchmark.json)`。"""
    base = case_dir(workspace, op)
    return base / CASES_FILENAME, base / BENCHMARK_FILENAME


def sha256_file(path: Path) -> str:
    """单个文件的 sha256 hex（1MB 分块，性能口径与 nm_test.sh 一致）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases_suite_sha256(workspace: Path, op: str | None = None) -> str | None:
    """`cases/<op>/` 的内容哈希（完整 64 hex）；目录不存在返回 None。

    ``op`` 为 None 时保留旧的整树算法，只给存点/历史迁移工具使用；
    任务确认、执行计划与交付门禁必须传 ``task.op_name``。这样新增 #11 的
    用例不会使 #12 的确认与历史 run 失效。遍历顺序与 nm_test 一致：
    `os.walk`（深度优先）+ 每层 `dirnames.sort()` + 每层 `filenames` 排序。
    """
    root = Path(workspace) / "cases"
    if op:
        root = root / op
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            digest.update(rel.encode())
            digest.update(b"\0")
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(_CHUNK), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def cases_ref(workspace: Path, op: str | None = None) -> str | None:
    """用例集 ref：`cases@<hex 前 16>`；目录不存在返回 None。"""
    sha = cases_suite_sha256(workspace, op)
    return f"{_SUITE_REF_PREFIX}{sha[:16]}" if sha else None


def suite_ref_from_sha256(sha256: str | None) -> str | None:
    """完整 64 hex → `cases@<前16>`（给 versions / delivery 拼 suite_ref 用）。"""
    return f"{_SUITE_REF_PREFIX}{sha256[:16]}" if sha256 else None


# ---------------------------------------------------------------------------
# 查询 / 确认
# ---------------------------------------------------------------------------

def confirmation(db: Session, task: Task) -> dict:
    """当前确认状态（契约 §3.3 的 GET 响应结构）。

    - `confirmed_sha256`：确认时记录的 `tasks.cases_confirmed_sha256`（完整 64 hex）或 None
    - `current_suite_sha256`：现在现算的整目录哈希（cases/ 不存在 → None）
    - `in_sync`：已确认且当前哈希 == 确认哈希
    """
    workspace = workspace_of(db, task)
    # 未迁移的历史行仍按当时的整 cases/ 口径判定；一旦确认/迁移为
    # operator-v1，就只看该任务的 cases/<op>/。这避免在未审计时静默改写历史语义。
    current = None
    if workspace is not None:
        current = cases_suite_sha256(
            workspace, task.op_name if task.cases_confirmed_scope == "operator-v1" else None
        )
    confirmed_sha = task.cases_confirmed_sha256
    confirmed_at = task.cases_confirmed_at
    return {
        "confirmed_at": confirmed_at,
        "confirmed_sha256": confirmed_sha,
        "current_suite_sha256": current,
        "in_sync": bool(confirmed_at is not None and confirmed_sha is not None
                        and current is not None and confirmed_sha == current),
    }


def confirm(db: Session, task: Task, *, cases_sha256: str, benchmark_sha256: str,
            by: str | None = None) -> dict:
    """整组确认用例：校验两个文件的 sha256 → 写确认记录 + StageEvent → 返回 confirmation。

    校验失败（工作区/ op_name / 文件缺失 / 哈希不匹配）→ `CaseConfirmError`（路由 409）。
    成功时内部 commit（调用方不必再 commit）。
    """
    workspace = workspace_of(db, task)
    if workspace is None:
        raise CaseConfirmError("项目的 workspace_path 未配置")
    op = str(task.op_name or "").strip()
    if not op:
        raise CaseConfirmError("任务缺少算子名（op_name），无法定位用例文件")
    if "/" in op or "\\" in op or op in (".", "..") or ".." in op:
        raise CaseConfirmError(f"非法算子名：{op!r}")

    cases_json, benchmark_json = case_files(workspace, op)
    for label, path, expected in (
        (f"cases/{op}/cases.json", cases_json, cases_sha256),
        (f"cases/{op}/benchmark.json", benchmark_json, benchmark_sha256),
    ):
        if not path.is_file():
            raise CaseConfirmError(f"用例文件不存在：{label}（先落文件再确认）")
        actual = sha256_file(path)
        if actual != expected:
            raise CaseConfirmError(
                f"{label} 哈希不一致（文件={actual[:16]}…，入参={str(expected)[:16]}…）；"
                "文件可能被改过，请重新读取后再确认"
            )

    suite_sha = cases_suite_sha256(workspace, op)
    if not suite_sha:
        raise CaseConfirmError("cases/ 目录不存在，无法计算用例集哈希")

    task.cases_confirmed_at = utcnow()
    task.cases_confirmed_sha256 = suite_sha
    task.cases_confirmed_scope = "operator-v1"
    task.cases_confirmed_by = (by or "api")[:120]
    # 阶段推进：用例组确认后，按证据重新推导（阶段一达成 → 通常进入「开发」）。
    # 这样页面不会停在「需求与设计 · 未同步」，也把「确认 → 可开发」写成 stage_event 留痕。
    from sqlalchemy import select as _select

    from ..models import Run, SpecVersion
    from . import stageboard

    spec_version = None
    if task.current_spec_version_id is not None:
        spec_row = db.get(SpecVersion, task.current_spec_version_id)
        spec_version = spec_row.version if spec_row is not None else None
    runs = list(db.scalars(_select(Run).where(Run.task_id == task.id).order_by(Run.id)))
    derived = stageboard.build_task_view(task, runs, spec_version=spec_version)["derived_stage"]
    db.add(StageEvent(task_id=task.id, from_stage=task.stage, to_stage=derived,
                      actor=(by or "api")[:80], reason="用例组确认：按证据推进阶段"))
    task.stage = derived
    db.commit()
    db.refresh(task)
    return confirmation(db, task)
