"""版本存档（savepoints v1）：内容寻址源码归档 + 存盘点 + 回滚。

模型（`docs/plan-savepoints-v1.md`）：

- **按内容哈希寻址的源码归档（tar.gz）+ 指向它的存盘点行**。存盘点只存「标签 + 环境引用 +
  时间 + 类型」，不重复存代码；同一 `code_snapshot_sha256` 只归档一份。
- 归档规则必须与 `exec_actions.snapshot()` **完全相同**（同一组 skip 目录、同一文件集合、同一
  排序与哈希算法），解包后重算 `snapshot(解包目录) == 存档 sha` 才算有效。
- 归档先写临时文件再 `os.replace`（原子落盘），单进程内加锁避免并发重复打包。
- 解包逐成员安全校验（拒绝绝对路径 / `..` / 符号链接 / 硬链接 / 非普通文件）；回滚写文件前
  校验目标必须落在 workspace 内。

`restore()` 会懒加载 `action_runner` 查「运行中作业」—— 所以 `action_runner` 顶层**不能** import
本模块（否则循环依赖），挂点处用函数内 import。
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import settings
from ..models import AgentJob, Project, Run, Savepoint, SpecVersion, Task
from . import exec_actions, optimize

logger = logging.getLogger(__name__)

#: 与 `exec_actions.snapshot()` 的 skip 集合逐字一致（两处一旦漂移，归档与 run 指纹就对不上）
SNAPSHOT_SKIP = {"runs", ".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
                 ".skills-src", ".opx-chat"}

#: 单个归档上限（源码总量与压缩后归档都查）
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024

#: 每任务自动存档保留条数（超出按最旧删行，归档文件不删 —— 内容寻址、可能被别的行引用）
MAX_AUTO_KEEP = 50

#: `SavepointOut.runs` 最多列多少条
MAX_RUNS = 20

#: 快照文件数上限（与 exec_actions.snapshot 的默认 limit 一致）；超过即截断 → 拒绝存档
SNAPSHOT_LIMIT = 20000

#: 只要写任务工作区的智能体作业类型（与 conversations._WORKSPACE_JOB_KINDS 同口径）
_RUNNING_AGENT_KINDS = ("对话", "开发", "优化")

#: 同一 sha 只打包一次（单进程部署；跨进程的重复打包由内容寻址 + 原子 replace 兜住）
_ARCHIVE_LOCK = threading.Lock()
#: 回滚互斥：同进程内禁止并发回滚（两个回滚同时写一个工作区会互相踩）
_RESTORE_LOCK = threading.Lock()

_UNSET = object()


class SavepointError(ValueError):
    """存档 / 回滚失败（路由层按语义转 4xx）。"""


class SavepointArchiveError(SavepointError):
    """归档文件缺失 / 损坏 / 路径不合法（路由层转 409）。"""


class SavepointBusyError(SavepointError):
    """有运行中作业或已有回滚在进行（路由层转 409）。"""


# ---------------------------------------------------------------- 归档根 / 路径


def archive_root() -> Path:
    """归档根：`settings.savepoint_root`（env `OPX_SAVEPOINT_ROOT`），空 = `<repo_root>/archives`。"""
    configured = (settings.savepoint_root or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(settings.repo_root).expanduser() / "archives"


def _sources_dir() -> Path:
    d = archive_root() / "sources"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_archive_path(sp: Savepoint) -> Path:
    """存档归档的绝对路径；**必须**位于归档根下（防 DB 被篡改后任意读文件）。"""
    raw = (sp.archive_path or "").strip()
    if not raw:
        raise SavepointArchiveError(f"存档 #{sp.id} 没有归档路径")
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    root = archive_root().resolve()
    if resolved != root and root not in resolved.parents:
        raise SavepointArchiveError(f"存档 #{sp.id} 的归档路径不在允许的目录下（拒绝读文件）")
    return resolved


# ---------------------------------------------------------------- 快照与打包


def task_scope_of(task: Task, path: str) -> str:
    """路径属于任务范围（`ops/<op>/` 或 `cases/<op>/`）还是共享范围（其余）。

    `op_name` 为空时所有路径都算 `shared`（契约 §4.4）。
    """
    op = (getattr(task, "op_name", None) or "").strip()
    if not op:
        return "shared"
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if len(parts) >= 2 and parts[0] in ("ops", "cases") and parts[1] == op:
        return "task"
    return "shared"


def _scan_files(workspace: Path, limit: int = SNAPSHOT_LIMIT) -> list[tuple[str, Path]]:
    """工作区里参与快照的文件（与 `exec_actions.snapshot` 同一过滤 + 同一排序）。"""
    entries: list[tuple[str, Path]] = []
    for path in workspace.rglob("*"):
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in SNAPSHOT_SKIP for part in rel.parts):
            continue
        if path.is_file():
            entries.append((rel.as_posix(), path))
    # 按相对路径字符串排序（与 snapshot 的哈希顺序逐字节一致）
    entries.sort(key=lambda e: e[0])
    if len(entries) > limit:
        raise SavepointError(f"源码快照被截断（文件数超过 {limit}），不能作为完整存档")
    return entries


def _hash_entries(entries: list[tuple[str, Path]]) -> str:
    """按 `exec_actions.snapshot()` 的同一顺序与写法算内容哈希。"""
    h = hashlib.sha256()
    for rel, path in entries:
        h.update(rel.encode())
        h.update(b"\0")
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError as e:
            raise SavepointError(f"读取工作区文件失败：{rel}（{e}）") from e
        h.update(b"\0")
    return h.hexdigest()


def current_sha(workspace: Path, *, op_name: str | None = None) -> str:
    """工作区当前全量源码快照 sha（与 `runs.code_snapshot_sha256` 同算法）；截断则抛错。"""
    snap = (exec_actions.operator_snapshot(Path(workspace), op_name) if op_name
            else exec_actions.snapshot(Path(workspace)))
    if snap.truncated:
        raise SavepointError("源码快照被截断（文件数超上限），不能作为完整存档")
    return snap.sha256


def evidence_op(sp: Savepoint | None) -> str | None:
    return sp.op_name_snapshot if sp is not None and sp.evidence_scope == "operator-v1" else None


def current_evidence_sha(workspace: Path, sp: Savepoint) -> str:
    return current_sha(workspace, op_name=evidence_op(sp))


def _empty_sha() -> str:
    return hashlib.sha256().hexdigest()


class _HashingReader:
    """边写 tar 边算内容哈希（读出多少更新多少）。"""

    def __init__(self, fh, digest):
        self._fh = fh
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        data = self._fh.read(size)
        self._digest.update(data)
        return data


def _write_archive(entries: list[tuple[str, Path]], target: Path) -> str:
    """把 entries 写成 tar.gz；返回**写入内容**的哈希（用于校验打包期间工作区没变）。"""
    h = hashlib.sha256()
    try:
        with tarfile.open(target, "w:gz") as tf:
            for rel, path in entries:
                try:
                    st = path.stat()
                except OSError as e:
                    raise SavepointError(f"读取工作区文件失败：{rel}（{e}）") from e
                info = tarfile.TarInfo(rel)
                info.size = st.st_size
                info.mode = 0o644
                info.mtime = int(st.st_mtime)
                h.update(rel.encode())
                h.update(b"\0")
                try:
                    with path.open("rb") as fh:
                        tf.addfile(info, _HashingReader(fh, h))
                except OSError as e:
                    raise SavepointError(f"读取工作区文件失败：{rel}（{e}）") from e
                h.update(b"\0")
    except tarfile.TarError as e:  # 写 tar 失败（磁盘满等）
        raise SavepointError(f"写归档失败：{e}") from e
    return h.hexdigest()


def ensure_archive(workspace: Path) -> tuple[Path, int, str]:
    """幂等打包当前工作区，返回 `(归档路径, 字节数, sha256)`。

    - 已存在同 sha 的归档 → 直接复用；
    - 先写 `.tmp-*` 再 `os.replace` 原子落盘；
    - 单进程内加锁，避免并发重复打包；
    - 打包期间工作区若被改动（写入内容哈希 ≠ 快照哈希）→ 抛错让调用方重试。
    """
    ws = Path(workspace).expanduser()
    if not ws.is_dir():
        raise SavepointError(f"工作区不存在：{ws}")
    entries = _scan_files(ws)
    sha = _hash_entries(entries)
    sources = _sources_dir()
    target = sources / f"{sha}.tar.gz"
    with _ARCHIVE_LOCK:
        if target.is_file():
            return target, target.stat().st_size, sha
        total_src = 0
        for _rel, path in entries:
            try:
                total_src += path.stat().st_size
            except OSError as e:
                raise SavepointError(f"读取工作区文件失败：{_rel}（{e}）") from e
        if total_src > MAX_ARCHIVE_BYTES:
            raise SavepointError(
                f"工作区源码过大（{total_src} bytes，超过 {MAX_ARCHIVE_BYTES} bytes 上限），拒绝归档")
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".tar.gz", dir=str(sources))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            written_sha = _write_archive(entries, tmp)
            if written_sha != sha:
                raise SavepointError("工作区在打包过程中发生变化，请重试")
            size = tmp.stat().st_size
            if size > MAX_ARCHIVE_BYTES:
                raise SavepointError(
                    f"归档过大（{size} bytes，超过 {MAX_ARCHIVE_BYTES} bytes 上限）")
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return target, size, sha


# ---------------------------------------------------------------- 解包（安全）


def _safe_extract(archive: Path, dest: Path) -> None:
    """把归档解到 `dest`：逐成员拒绝绝对路径 / `..` / 符号链接 / 硬链接 / 非普通文件。"""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    total = 0
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                name = member.name or ""
                if not name or "\x00" in name or name.startswith(("/", "\\")) or "\\" in name:
                    raise SavepointError(f"拒绝解包：非法成员路径 {name!r}")
                parts = PurePosixPath(name).parts
                if PurePosixPath(name).is_absolute() or any(p in ("..", "") for p in parts):
                    raise SavepointError(f"拒绝解包：非法成员路径 {name!r}")
                if not member.isfile():
                    raise SavepointError(f"拒绝解包：非普通文件成员 {name!r}（符号链接/硬链接/目录一律拒绝）")
                total += max(0, int(member.size or 0))
                if total > MAX_ARCHIVE_BYTES:
                    raise SavepointError("拒绝解包：解压后大小超过上限")
                target = root / name
                try:
                    target.resolve().relative_to(root)
                except (ValueError, OSError) as e:
                    raise SavepointError(f"拒绝解包：成员路径越界 {name!r}") from e
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    raise SavepointError(f"拒绝解包：成员没有内容 {name!r}")
                with src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
    except tarfile.TarError as e:
        raise SavepointError(f"归档文件不是合法的 tar.gz：{e}") from e
    except OSError as e:
        raise SavepointError(f"解包归档失败：{e}") from e


@contextlib.contextmanager
def _unpack_verified(sp: Savepoint) -> Iterator[Path]:
    """解包到临时目录并校验「解包后重算的快照 == 存档 sha」；损坏/缺文件 → 409 语义错误。"""
    archive = get_archive_path(sp)
    if not archive.is_file():
        raise SavepointArchiveError(f"存档 #{sp.id} 的归档文件缺失：{archive}")
    with tempfile.TemporaryDirectory(prefix="opx-savepoint-") as td:
        dest = Path(td)
        try:
            _safe_extract(archive, dest)
            snap = exec_actions.snapshot(dest)
        except SavepointArchiveError:
            raise
        except SavepointError as e:
            raise SavepointArchiveError(f"存档 #{sp.id} 的归档已损坏：{e}") from e
        if snap.truncated or snap.sha256 != (sp.archive_snapshot_sha256 or sp.code_snapshot_sha256):
            raise SavepointArchiveError(f"存档 #{sp.id} 的归档已损坏（解包后内容哈希不一致）")
        if evidence_op(sp) and current_sha(dest, op_name=evidence_op(sp)) != sp.code_snapshot_sha256:
            raise SavepointArchiveError(f"存档 #{sp.id} 的算子代码身份与归档不一致")
        yield dest


# ---------------------------------------------------------------- 建行 / 输出


def _workspace_of(db: Session, task: Task) -> Path | None:
    project = db.get(Project, task.project_id) if task is not None else None
    if project is None or not project.workspace_path:
        return None
    return Path(project.workspace_path).expanduser()


def _default_label() -> str:
    return f"存档 {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def _env_refs(db: Session, task: Task, sha: str) -> dict:
    """环境引用：取「同 sha 的最新 ok run」；没有 run 的部分回退任务当前值（可空）。"""
    run = db.scalar(
        select(Run)
        .where(Run.task_id == task.id, Run.code_snapshot_sha256 == sha, Run.status == "ok")
        .order_by(Run.id.desc())
        .limit(1)
    )
    project = db.get(Project, task.project_id)
    criteria_ref = run.criteria_ref if run is not None and run.criteria_ref else None
    if not criteria_ref:
        criteria_ref = exec_actions.resolve_criteria(task, project)[0]
    return {
        "spec_version_id": (run.spec_version_id if run is not None and run.spec_version_id
                            else task.current_spec_version_id),
        "criteria_ref": criteria_ref,
        "suite_ref": run.suite_ref if run is not None else None,
        "image_digest": run.image_digest if run is not None else None,
        "env_recipe_snapshot": run.env_recipe_snapshot if run is not None else None,
    }


def _create_row(db: Session, task: Task, *, sha: str, archive_path: Path, archive_bytes: int,
                label: str, note: str | None, kind: str,
                parent_id: int | None, loop_id: int | None = None,
                evidence_scope: str | None = None, op_name_snapshot: str | None = None,
                archive_snapshot_sha256: str | None = None) -> Savepoint:
    refs = _env_refs(db, task, sha)
    sp = Savepoint(
        project_id=task.project_id,
        task_id=task.id,
        label=(label or _default_label())[:200],
        note=note,
        kind=kind,
        code_snapshot_sha256=sha,
        evidence_scope=evidence_scope,
        op_name_snapshot=op_name_snapshot,
        archive_snapshot_sha256=archive_snapshot_sha256,
        archive_path=str(archive_path),
        archive_bytes=archive_bytes,
        parent_id=parent_id,
        loop_id=loop_id,
        created_by="page",
        created_at=datetime.now(timezone.utc),
        **refs,
    )
    db.add(sp)
    db.flush()
    return sp


def create(db: Session, task: Task, *, label: str | None = None, note: str | None = None,
           kind: str = "manual", parent_id: int | None = None,
           loop_id: int | None = None, evidence_op_name: str | None = None) -> Savepoint:
    """对当前工作区打包（必需成功）并建存盘点行；只 flush，不 commit（调用方提交）。"""
    workspace = _workspace_of(db, task)
    if workspace is None:
        raise SavepointError("项目工作区未配置")
    if not workspace.is_dir():
        raise SavepointError(f"工作区不存在：{workspace}")
    archive_path, archive_bytes, sha = ensure_archive(workspace)
    identity = {}
    if evidence_op_name is not None:
        if evidence_op_name != task.op_name or not (workspace / "ops" / evidence_op_name).is_dir():
            raise SavepointError("算子证据作用域与任务或工作区不一致")
        archive_sha = sha
        sha = current_sha(workspace, op_name=evidence_op_name)
        if current_sha(workspace) != archive_sha:
            raise SavepointError("工作区在生成优化存档时发生变化，请重试")
        identity = dict(evidence_scope="operator-v1", op_name_snapshot=evidence_op_name,
                        archive_snapshot_sha256=archive_sha)
    text = (label or "").strip() or _default_label()
    return _create_row(db, task, sha=sha, archive_path=archive_path, archive_bytes=archive_bytes,
                       label=text, note=note, kind=kind, parent_id=parent_id, loop_id=loop_id, **identity)


def create_for_sha(db: Session, task: Task, sha: str, *, label: str, kind: str,
                   loop_id: int | None = None, note: str | None = None) -> Savepoint:
    """按**已知代码快照 sha** 建存盘点行（P1 自动候选用），只 flush、不 commit。

    优先复用内容寻址的现成归档（`<archive_root>/sources/<sha>.tar.gz`，每次动作前已自动归档）；
    归档不存在时，只有「当前工作区 sha == 该 sha」才现打包，否则说明这一版代码已经不在磁盘上，
    明确报错（不能拿当前代码冒充历史快照）。
    """
    sha = (sha or "").strip()
    if not sha:
        raise SavepointError("缺少代码快照 sha，无法建存档")
    # 已归档的算子证据复用其完整归档；不把算子哈希误当全工作区归档文件名。
    existing = db.scalar(select(Savepoint).where(
        Savepoint.task_id == task.id, Savepoint.code_snapshot_sha256 == sha,
        Savepoint.evidence_scope == "operator-v1").order_by(Savepoint.id.desc()).limit(1))
    if existing is not None:
        with _unpack_verified(existing):
            pass
        archive = get_archive_path(existing)
        return _create_row(db, task, sha=sha, archive_path=archive, archive_bytes=archive.stat().st_size,
                           label=(label or "").strip() or _default_label(), note=note, kind=kind,
                           parent_id=None, loop_id=loop_id, evidence_scope=existing.evidence_scope,
                           op_name_snapshot=existing.op_name_snapshot,
                           archive_snapshot_sha256=existing.archive_snapshot_sha256)
    workspace = _workspace_of(db, task)
    if (loop_id is not None and workspace is not None and task.op_name and (workspace / "ops" / task.op_name).is_dir()
            and current_sha(workspace, op_name=task.op_name) == sha):
        row = create(db, task, label=label, note=note, kind=kind, loop_id=loop_id,
                     evidence_op_name=task.op_name)
        if row.code_snapshot_sha256 != sha:
            raise SavepointError("候选代码在归档期间发生变化")
        return row
    sources = _sources_dir()
    target = sources / f"{sha}.tar.gz"
    if target.is_file():
        archive_path, archive_bytes = target, target.stat().st_size
    else:
        workspace = _workspace_of(db, task)
        if workspace is None:
            raise SavepointError(f"归档不存在，且项目工作区未配置：{sha[:12]}…")
        if not workspace.is_dir():
            raise SavepointError(f"归档不存在，且工作区不可用：{workspace}")
        archive_path, archive_bytes, current = ensure_archive(workspace)
        if current != sha:
            raise SavepointError(
                f"归档不存在，且当前工作区不是该快照（当前 {current[:12]}… ≠ {sha[:12]}…）")
    text = (label or "").strip() or _default_label()
    return _create_row(db, task, sha=sha, archive_path=archive_path, archive_bytes=archive_bytes,
                       label=text, note=note, kind=kind, parent_id=None, loop_id=loop_id)


def _trim_auto(db: Session, task_id: int) -> None:
    """每任务自动存档只保留最新 `MAX_AUTO_KEEP` 条：删更旧的行，**不删归档文件**。"""
    stale = list(db.scalars(
        select(Savepoint.id)
        .where(Savepoint.task_id == task_id, Savepoint.kind == "auto")
        .order_by(Savepoint.created_at.desc(), Savepoint.id.desc())
        .offset(MAX_AUTO_KEEP)
    ))
    if stale:
        db.execute(sql_delete(Savepoint).where(Savepoint.id.in_(stale)))


def auto_capture(db: Session, task: Task, workspace, *, label: str,
                 kind: str = "auto") -> Savepoint | None:
    """自动归档挂点用的 best-effort 版本：**任何异常只 log，不抛**。

    - 最新一条 auto 存档同 sha → 跳过（归档仍幂等补齐，保证文件在）；
    - 建行后 `db.flush()` 并按 `MAX_AUTO_KEEP` 裁剪（只删行）；**不 commit**，由调用方提交。
    """
    try:
        ws = Path(workspace).expanduser()
        if not ws.is_dir():
            return None
        archive_path, archive_bytes, sha = ensure_archive(ws)
        latest = db.scalar(
            select(Savepoint)
            .where(Savepoint.task_id == task.id, Savepoint.kind == "auto")
            .order_by(Savepoint.created_at.desc(), Savepoint.id.desc())
            .limit(1)
        )
        if latest is not None and latest.code_snapshot_sha256 == sha:
            return None
        sp = _create_row(db, task, sha=sha, archive_path=archive_path, archive_bytes=archive_bytes,
                         label=label, note=None, kind=kind, parent_id=None)
        _trim_auto(db, task.id)
        return sp
    except Exception as e:  # noqa: BLE001 —— 自动归档失败不能阻断主流程
        logger.exception("自动归档失败（task=%s, label=%s）", getattr(task, "id", None), label)
        if isinstance(e, SQLAlchemyError):
            try:
                db.rollback()
            except Exception:  # noqa: BLE001
                pass
        return None


def _iso(dt: datetime | None) -> str | None:
    """UTC ISO（SQLite 读回来的 naive datetime 按 UTC 处理）。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _safe_sha(workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    try:
        if not Path(workspace).is_dir():
            return None
        return current_sha(Path(workspace))
    except SavepointError:
        return None


def out(db: Session, sp: Savepoint, *, current_sha=_UNSET, workspace=_UNSET) -> dict:
    """拼 `SavepointOut`（契约 §4.1）。

    `in_sync` 是**全量**口径：工作区当前 sha 是否等于存档 sha（一次请求只算一次，由调用方
    通过 `current_sha` 传入；工作区不存在 / 快照截断 → false）。
    """
    if workspace is _UNSET:
        task = db.get(Task, sp.task_id)
        workspace = _workspace_of(db, task) if task is not None else None
    if current_sha is _UNSET:
        current_sha = _safe_sha(workspace)
    if evidence_op(sp) and workspace is not None:
        try:
            current_sha = current_evidence_sha(Path(workspace), sp)
        except (OSError, SavepointError):
            current_sha = None
    runs = list(db.scalars(
        select(Run)
        .where(Run.task_id == sp.task_id, Run.code_snapshot_sha256 == sp.code_snapshot_sha256)
        .order_by(Run.id)
        .limit(MAX_RUNS)
    ))
    spec_version = None
    if sp.spec_version_id is not None:
        spec_row = db.get(SpecVersion, sp.spec_version_id)
        spec_version = spec_row.version if spec_row is not None else None
    return {
        "id": sp.id,
        "task_id": sp.task_id,
        "label": sp.label,
        "note": sp.note,
        "kind": sp.kind,
        "code_snapshot_sha256": sp.code_snapshot_sha256,
        "archive_bytes": sp.archive_bytes,
        "spec_version_id": sp.spec_version_id,
        "spec_version": spec_version,
        "criteria_ref": sp.criteria_ref,
        "suite_ref": sp.suite_ref,
        "image_digest": sp.image_digest,
        "runs": [
            {"id": r.id, "action": r.action, "status": r.status, "device_id": r.device_id,
             "loop_id": r.loop_id, "ended_at": _iso(r.ended_at)}
            for r in runs
        ],
        "in_sync": bool(current_sha is not None and current_sha == sp.code_snapshot_sha256),
        "parent_id": sp.parent_id,
        "loop_id": sp.loop_id,
        "created_at": _iso(sp.created_at),
    }


def list_for_task(db: Session, task: Task, *, limit: int = 50,
                  loop_id: int | None = None) -> list[dict]:
    """任务的存档列表（`created_at desc, id desc`）；可按优化轮次过滤；全量 sha 一次请求只算一次。"""
    query = select(Savepoint).where(Savepoint.task_id == task.id)
    if loop_id is not None:
        query = query.where(Savepoint.loop_id == loop_id)
    rows = list(db.scalars(
        query.order_by(Savepoint.created_at.desc(), Savepoint.id.desc()).limit(limit)
    ))
    workspace = _workspace_of(db, task)
    sha = _safe_sha(workspace)
    return [out(db, sp, current_sha=sha, workspace=workspace) for sp in rows]


# ---------------------------------------------------------------- 对比 / 回滚


def diff(db: Session, sp: Savepoint, task: Task) -> dict:
    """对比「存档代码」与「当前工作区」（契约 §4.4）。"""
    with _unpack_verified(sp) as saved_dir:
        saved = optimize.capture(saved_dir)
        workspace = _workspace_of(db, task)
        current = optimize.capture(workspace) if workspace is not None else optimize.Capture(
            sha256=_empty_sha())
        raw_changed, diff_text = optimize.diff_captures(saved, current)
        changed = [{**c, "scope": task_scope_of(task, c["path"])} for c in raw_changed]
        outside = [c["path"] for c in changed if c["scope"] == "shared"]
        truncated = len(diff_text) > optimize.MAX_DIFF_CHARS
        code_sha = (current_evidence_sha(workspace, sp)
                    if workspace is not None and evidence_op(sp) else current.sha256)
        return {
            "savepoint": out(db, sp, current_sha=current.sha256, workspace=workspace),
            "saved_sha256": sp.code_snapshot_sha256,
            "current_sha256": code_sha,
            "in_sync": code_sha == sp.code_snapshot_sha256,
            "changed": changed,
            "outside_scope": outside,
            "diff": diff_text,
            "diff_truncated": truncated,
        }


def _ensure_running_guard(db: Session, task: Task, *, scope: str = "task") -> None:
    """运行中作业守卫：有动作作业 / 会写工作区的智能体作业在跑时禁止回滚（409）。

    `scope=full` 会改写**整个共享工作区**（其它任务的代码/共享文件也在里面），所以按
    **项目内全部任务**检查；`scope=task` 只动本任务路径，只看本任务即可。
    """
    # 懒加载：action_runner 顶层不 import 本模块，这里反向引用避免循环依赖
    from . import action_runner

    if scope == "full":
        task_ids = list(db.scalars(select(Task.id).where(Task.project_id == task.project_id)))
    else:
        task_ids = [task.id]
    for tid in task_ids:
        action_ids = action_runner.running_for_task(tid)
        if action_ids:
            raise SavepointBusyError(
                f"任务 #{tid} 有正在执行的动作作业（作业 #{action_ids[0]}），等它结束再回滚")
    job_id = db.scalar(
        select(AgentJob.id)
        .where(AgentJob.task_id.in_(task_ids), AgentJob.status == "running",
               AgentJob.kind.in_(_RUNNING_AGENT_KINDS))
        .limit(1)
    )
    if job_id is not None:
        raise SavepointBusyError(
            f"有运行中的智能体作业（job #{job_id}），等它结束再回滚")


def _scope_in_sync(saved: optimize.Capture, current: optimize.Capture, task: Task, scope: str) -> bool:
    """回滚后的「范围口径」一致性：只看 scope 内路径是否还有差异。"""
    for rel in sorted(set(saved.files) | set(current.files)):
        if scope == "task" and task_scope_of(task, rel) == "shared":
            continue
        before = saved.files.get(rel)
        after = current.files.get(rel)
        if before is None or after is None or before["sha256"] != after["sha256"]:
            return False
    return True


def _apply(workspace: Path, saved_dir: Path, saved: optimize.Capture, current: optimize.Capture,
           applied: list[dict]) -> None:
    """按 diff 结果把「存档 vs 当前」的差异应用到工作区（写/删），路径全部校验在 workspace 内。"""
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)

    writes: list[str] = []
    deletes: list[str] = []
    for item in applied:
        rel = item["path"]
        if rel in saved.files:
            writes.append(rel)
        elif rel in current.files:
            deletes.append(rel)

    # 先全量校验目标路径，再动手：任一越界就整体中止（工作区一个文件都不动）
    for rel in writes:
        target = root / rel
        try:
            resolved = target.resolve()
        except OSError:
            resolved = target
        if resolved != root and root not in resolved.parents:
            raise SavepointError(f"拒绝写入工作区外路径：{rel}")
    for rel in deletes:
        parent = (root / rel).parent.resolve()
        if parent != root and root not in parent.parents:
            raise SavepointError(f"拒绝删除工作区外路径：{rel}")

    for rel in writes:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(saved_dir / rel, target)
    for rel in deletes:
        target = root / rel
        if target.is_symlink() or target.is_file():
            target.unlink()


def restore(db: Session, sp: Savepoint, task: Task, *, scope: str = "task",
            note: str | None = None) -> dict:
    """把工作区回滚到该存档（契约 §4.5）。

    流程：守卫 → **DB 工作区租约**（Step 1b）→ 解包校验 → 急救存档（失败中止）→
    按 scope 应用差异 → 提交并返回。回滚**不动** runs / 规格指针 / loops。
    """
    if scope not in ("task", "full"):
        raise SavepointError("scope 只能是 task / full")
    if scope == "task" and not (task.op_name or "").strip():
        raise SavepointError("任务没有 op_name，无法按任务范围回滚（请改用 scope=full）")
    _ensure_running_guard(db, task, scope=scope)
    # Step 2：自动验证链运行期间工作区必须稳定（链在 build/test/benchmark 之间切换），
    # 回滚会改代码 → 直接 409（项目级租约只覆盖「动作作业存活」的窗口，步间空档靠这个）。
    try:
        from . import loop_orchestrator  # 局部 import：orchestrator 依赖本模块
        auto = loop_orchestrator.running_for_task(db, task.id)
    except Exception:  # noqa: BLE001 —— 判定失败不能误放行
        auto = None
    if auto is not None:
        raise SavepointBusyError(
            f"自动验证链 #{auto.id} 正在跑（当前步骤 {auto.current_step or '未知'}），等它结束再回滚")
    # Step 1b：回滚也持项目级租约。进程内 `_RESTORE_LOCK` 只挡同进程的两个回滚；租约还挡
    # 「另一个 worker/进程里刚起的 agent/action 作业」，避免回滚与写代码同时进行。
    from datetime import timedelta

    from . import workspace_lease  # 局部 import：与本模块「反向引用 action_runner」同一策略
    try:
        # restore 租约没有 job_id（进程崩溃时无从判断存活），用短 TTL：即使走到 SIGKILL
        # 且控制台没重启，也不会把工作区锁 12h，最多锁 15 分钟。
        workspace_lease.acquire(db, task, kind="restore", note=f"回滚到存点 #{sp.id}",
                                ttl=timedelta(minutes=15))
    except workspace_lease.LeaseBusy as e:
        raise SavepointBusyError(f"工作区正被占用，等它结束再回滚：{e}") from e
    try:
        if not _RESTORE_LOCK.acquire(blocking=False):
            raise SavepointBusyError("已有回滚正在进行，等它结束再试")
        try:
            return _restore_locked(db, sp, task, scope=scope, note=note)
        finally:
            _RESTORE_LOCK.release()
    finally:
        try:
            workspace_lease.release(db, project_id=task.project_id, kind="restore")
        except Exception:  # noqa: BLE001 —— 释放失败不能盖掉回滚本身的异常/结果
            logger.exception("回滚后释放工作区租约失败（task=%s）", task.id)


def _restore_locked(db: Session, sp: Savepoint, task: Task, *, scope: str,
                    note: str | None) -> dict:
    workspace = _workspace_of(db, task)
    if workspace is None:
        raise SavepointError("项目工作区未配置")
    with _unpack_verified(sp) as saved_dir:
        saved = optimize.capture(saved_dir)
        current = optimize.capture(workspace)
        if current.sha256 == (sp.archive_snapshot_sha256 or sp.code_snapshot_sha256):
            # 已经一致：不产生急救存档，直接返回
            return {
                "savepoint": out(db, sp, current_sha=current.sha256, workspace=workspace),
                "rescue": None,
                "restored": False,
                "in_sync": True,
                "current_sha256": current_evidence_sha(workspace, sp),
                "changed": [],
                "skipped_outside_scope": [],
                "note": note,
            }

        # 先做急救存档：失败 → 中止回滚，不动工作区（409 语义）
        try:
            rescue = create(db, task, label=f"回滚前自动存档（目标 #{sp.id}）",
                            kind="rescue", parent_id=sp.id)
        except SavepointError as e:
            raise SavepointArchiveError(f"急救存档失败，已中止回滚：{e}") from e

        raw_changed, _diff_text = optimize.diff_captures(saved, current)
        changed = [{**c, "scope": task_scope_of(task, c["path"])} for c in raw_changed]
        applied = [c for c in changed if scope == "full" or c["scope"] == "task"]
        skipped = [c["path"] for c in changed if scope == "task" and c["scope"] == "shared"]

        _apply(workspace, saved_dir, saved, current, applied)
        after = optimize.capture(workspace)
        in_sync = _scope_in_sync(saved, after, task, scope)
        db.commit()
        db.refresh(sp)
        db.refresh(rescue)
        return {
            "savepoint": out(db, sp, current_sha=after.sha256, workspace=workspace),
            "rescue": out(db, rescue, current_sha=after.sha256, workspace=workspace),
            "restored": True,
            "in_sync": in_sync,
            "current_sha256": current_evidence_sha(workspace, sp),
            "changed": applied,
            "skipped_outside_scope": skipped,
            "note": note,
        }
