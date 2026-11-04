#!/usr/bin/env python3
"""任务生命周期 v2 迁移：备份 → 删除全部存量任务 → 清空旧算子产物（2026-10-23 用户定）。

用法（在仓库根执行）：

    backend/.venv/bin/python scripts/reset_tasks_v2.py --dry-run   # 只打印将删除什么
    backend/.venv/bin/python scripts/reset_tasks_v2.py --yes       # 真执行（先备份）

口径（见 docs/plan-task-lifecycle-v2.md §11）：
1. 备份 `backend/opx.db` → `backend/opx.db.bak-<ts>-pre-v2`；
2. 备份 `workspaces/cann-ops/`（排除运行时大件 `.dsh` `.cannbot` `runs` `node_modules`）→
   `archives/workspaces-cann-ops-<ts>-pre-v2.tar.gz`；
3. 逐任务走 `services.task_deletion.delete(delete_files=True)`（DB 行 + `tasks/<slug>` +
   `.opx-chat/*` + 本地 `runs/<run_ref>`；远端 run 目录保留）；
4. 清空旧任务产物（保留共享资产：scripts/ criteria/ tools/run_*.py 入口脚本、.dsh skills 等）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 工作区里属于「旧任务产物」的路径（相对 workspaces/cann-ops）；共享资产不动。
WS = ROOT / "workspaces" / "cann-ops"
CLEAR_GLOBS = [
    "cases/elementwise_*",
    "ops/elementwise_*",
    "tools/torch_*_baseline.py",
    "tests/test_elementwise_*.py",
    "docs/elementwise_*",
    "task-docs/*",
    "tasks/*",
    ".opx-chat/*",
    "runs/*",
]
#: 备份排除的运行时大件（技能/dependencies/远端产物缓存）
TAR_EXCLUDES = (".dsh", ".cannbot", "runs", "node_modules", "build", "__pycache__", ".git")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def backup() -> tuple[Path, Path | None]:
    sqlite = ROOT / "backend" / "opx.db"
    db_bak = sqlite.with_name(f"opx.db.bak-{_ts()}-pre-v2")
    shutil.copy2(sqlite, db_bak)
    tar_path = ROOT / "archives" / f"workspaces-cann-ops-{_ts()}-pre-v2.tar.gz"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(WS.rglob("*")):
            rel = path.relative_to(WS)
            if any(part in TAR_EXCLUDES for part in rel.parts):
                continue
            tar.add(path, arcname=str(Path("cann-ops") / rel))
    return db_bak, tar_path


def clear_items() -> list[Path]:
    out: list[Path] = []
    for pattern in CLEAR_GLOBS:
        out.extend(sorted(WS.glob(pattern)))
    # 去重（嵌套命中时只留最外层）
    unique: list[Path] = []
    for p in out:
        if not any(p != q and q in p.parents for q in out):
            unique.append(p)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="真执行（会先备份）")
    args = parser.parse_args()
    if not args.dry_run and not args.yes:
        print("拒绝执行：加 --dry-run 看影响面，或 --yes 真执行（会先备份）", file=sys.stderr)
        return 2

    # 延迟导入：--dry-run 不需要数据库也能跑。
    # DB URL 默认是 sqlite:///./opx.db（相对进程 CWD）—— 脚本通常在仓库根跑，显式指到 backend/opx.db。
    import os
    os.environ.setdefault("OPX_DB_URL", f"sqlite:///{ROOT / 'backend' / 'opx.db'}")
    sys.path.insert(0, str(ROOT / "backend"))
    from app.db import SessionLocal
    from app.models import Task
    from app.services import task_deletion

    from sqlalchemy import select
    with SessionLocal() as db:
        tasks = list(db.scalars(select(Task).order_by(Task.id)))
        print(f"存量任务 {len(tasks)} 个：")
        for t in tasks:
            counts = task_deletion.counts(db, t)
            print(f"  #{t.id} {t.title!r} op={t.op_name} draft={t.is_draft} "
                  f"spec={t.current_spec_version_id} counts={counts}")
        print("\n将清空的工作区路径：")
        for p in clear_items():
            print(f"  {'[dir] ' if p.is_dir() else '[file]'} {p.relative_to(ROOT)}")
        if args.dry_run:
            print("\n--dry-run：不执行任何删除/备份。")
            return 0

        db_bak, tar_path = backup()
        print(f"\n已备份 DB：{db_bak.relative_to(ROOT)}")
        print(f"已备份工作区：{tar_path.relative_to(ROOT)}")

        for t in tasks:
            result = task_deletion.delete(db, t, delete_files=True)
            print(f"已删除任务 #{result['task_id']}（runs={result['counts'].get('runs')}，"
                  f"files={len(result['files'].get('deleted', []))}）")

    # DB 里删完后，再清工作区旧产物（共享资产保留）
    for p in clear_items():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        print(f"已清理 {p.relative_to(ROOT)}")
    print("\n完成。共享资产（scripts/criteria/tools/入口脚本/.dsh）保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
