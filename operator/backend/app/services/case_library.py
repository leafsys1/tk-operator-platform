"""案例库（M2）：读取外置 bundle 索引 + 从案例一键建任务。

案例库**不在仓库里**：由用户用 `scripts/import_cases.py` 导入到本机库目录
（默认 `~/.local/share/op-agent/case-library/`）。本模块只读索引与案例目录。

一键发起做的事（刻意最小）：
1. 把案例的 `cases.json` 复制到工作区 `cases/<op>/cases.json`；
2. 若案例带 `workspace/` 子树 → 复制到工作区根；若带 `ops/` → 复制到 `ops/<op>/`；
3. 建一个任务（`op_name`、标题、目标芯片）。
冲突（目标已存在）→ 明确报错，不覆盖。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Project, Task


class CaseError(ValueError):
    """案例库 / 案例不合法或目标冲突。"""


MAX_OP_NAME_LEN = 120


def root() -> Path:
    return Path(settings.case_library).expanduser()


def _index_path() -> Path:
    return root() / "index.json"


def index() -> dict:
    p = _index_path()
    if not p.is_file():
        return {"dest": str(root()), "bundles": [], "imported": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"dest": str(root()), "bundles": [], "imported": True, "error": str(e)}
    if not isinstance(data, dict):
        return {"dest": str(root()), "bundles": [], "imported": True,
                "error": "index.json 顶层不是对象"}
    bundles = []
    for b in data.get("bundles") or []:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name") or "")
        cases = []
        for c in b.get("cases") or []:
            if not isinstance(c, dict):
                continue
            case_dir = root() / name / str(c.get("path") or f"cases/{c.get('op')}")
            cases.append({
                "op": c.get("op"),
                "scenario": c.get("scenario"),
                "path": c.get("path"),
                # 案例级目标芯片（import 时从 manifest/cases.json 保留下来）；缺省 None
                "target_chip": c.get("target_chip"),
                "exists": case_dir.is_dir(),
                "has_workspace": (case_dir / "workspace").is_dir(),
                "has_ops": (case_dir / "ops").is_dir(),
            })
        bundles.append({"name": name, "version": b.get("version"), "cases": cases})
    return {"dest": str(root()), "bundles": bundles, "imported": True}


def _safe_child(base: Path, *parts: str) -> Path:
    p = base.joinpath(*parts).resolve()
    if p != base.resolve() and base.resolve() not in p.parents:
        raise CaseError("案例路径越界")
    return p


def _single_name(name: str, *, what: str = "算子名") -> str:
    """单段安全名：拒绝空 / `.` / `..` / 含 `/`、`\\`、`..` / 控制字符 / 过长。"""
    text = str(name or "").strip()
    if not text:
        raise CaseError(f"非法{what}：不能为空")
    if len(text) > MAX_OP_NAME_LEN:
        raise CaseError(f"非法{what}：长度超过 {MAX_OP_NAME_LEN}")
    if text in (".", "..") or "/" in text or "\\" in text or ".." in text:
        raise CaseError(f"非法{what}：{text!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise CaseError(f"非法{what}：{text!r}")
    return text


def _find_case(bundle: str, op: str) -> tuple[Path, dict]:
    data = index()
    for b in data.get("bundles", []):
        if b["name"] != bundle:
            continue
        for c in b["cases"]:
            if c["op"] == op:
                path = _safe_child(root(), bundle, str(c.get("path") or f"cases/{op}"))
                if not path.is_dir():
                    raise CaseError(f"案例目录不存在：{path}")
                return path, c
        raise CaseError(f"bundle {bundle} 里没有案例 {op}")
    raise CaseError(f"案例库没有 bundle {bundle}")


def _list_files(src: Path) -> list[tuple[Path, Path]]:
    """目录下的文件清单 [(相对路径, 绝对源路径)]；按路径排序保证可复现。"""
    if not src.is_dir():
        return []
    return [(p.relative_to(src), p) for p in sorted(src.rglob("*")) if p.is_file()]


def _mkdirs(path: Path, created: list[Path]) -> None:
    """逐级创建目录，并记录本次新建的目录（供失败回滚）。"""
    missing: list[Path] = []
    p = path
    while not p.exists():
        missing.append(p)
        if p.parent == p:  # pragma: no cover - 到根了还创建不了，交给 mkdir 抛错
            break
        p = p.parent
    for d in reversed(missing):
        d.mkdir()
        created.append(d)


def _rollback_files(copied: list[Path], created_dirs: list[Path]) -> None:
    for p in reversed(copied):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
    for d in reversed(created_dirs):
        try:
            d.rmdir()
        except OSError:
            shutil.rmtree(d, ignore_errors=True)


def start(db: Session, bundle: str, op: str, *, project_id: int | None = None) -> Task:
    """从案例建任务：先做完整冲突预检，全部通过后再写文件；中途失败回滚本次写入。

    冲突：`cases/<op>/cases.json`、`workspace/` 子树逐文件、`ops/<op>` 任一已存在 → `CaseError`（409）。
    `op` 必须是单段安全名；所有目标路径都经 `_safe_child` 校验在工作区内。
    """
    case_dir, meta = _find_case(bundle, op)
    op = _single_name(op)
    project = db.get(Project, project_id) if project_id else db.scalar(select(Project).order_by(Project.id))
    if not project or not project.workspace_path:
        raise CaseError("项目工作区未配置")
    ws = Path(project.workspace_path).expanduser().resolve()
    if not ws.is_dir():
        raise CaseError(f"工作区不存在：{ws}")

    cases_json = case_dir / "cases.json"
    if not cases_json.is_file():
        raise CaseError(f"案例缺 cases.json：{cases_json}")
    src_workspace = case_dir / "workspace"
    src_ops = case_dir / "ops"

    # ---- 完整冲突预检（任何写入之前）----
    target_cases = _safe_child(ws, "cases", op, "cases.json")
    target_op = _safe_child(ws, "ops", op)
    conflicts: list[str] = []
    if target_cases.exists():
        conflicts.append(str(target_cases))
    for rel, _src in _list_files(src_workspace):
        target = _safe_child(ws, rel.as_posix())
        if target.exists():
            conflicts.append(str(target))
    if src_ops.is_dir() and target_op.exists():
        conflicts.append(str(target_op))
    if conflicts:
        raise CaseError("目标已存在，拒绝覆盖：" + "；".join(conflicts[:5]))

    # ---- 执行：记录本次写入，失败按相反顺序回滚 ----
    copied: list[Path] = []
    created_dirs: list[Path] = []
    try:
        _mkdirs(target_cases.parent, created_dirs)
        shutil.copy2(cases_json, target_cases)
        copied.append(target_cases)
        for rel, src in _list_files(src_workspace):
            target = _safe_child(ws, rel.as_posix())
            _mkdirs(target.parent, created_dirs)
            shutil.copy2(src, target)
            copied.append(target)
        for rel, src in _list_files(src_ops):
            target = _safe_child(target_op, rel.as_posix())
            _mkdirs(target.parent, created_dirs)
            shutil.copy2(src, target)
            copied.append(target)

        task = Task(project_id=project.id, title=f"{op}（案例 {bundle}）", op_name=op,
                    target_chip=meta.get("target_chip"), input_form="案例库")
        db.add(task)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        db.refresh(task)
        return task
    except Exception:
        db.rollback()
        _rollback_files(copied, created_dirs)
        raise
