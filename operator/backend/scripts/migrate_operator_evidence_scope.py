"""将已核验的历史 run 挂到 operator-v1 证据作用域。

这是元数据迁移：原 run 的 code_snapshot/suite_ref/manifest/报告都不改写，只增加
scoped 别名。调用前必须先核验目标 run 的远端 ws/ops/<op> 与当前算子源码一致。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.models import Project, Run, Task, TaskVersion
from app.services import case_confirm, exec_actions, gate


def _backup(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path("/tmp") / f"opx-before-operator-scope-{stamp}.db"
    with sqlite3.connect(db_path) as src, sqlite3.connect(dest) as out:
        src.backup(out)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--run-ids", type=int, nargs="*", default=[])
    parser.add_argument("--verified-source-sha256", required=True,
                        help="人工/远端核验时 ops/<op> 源码树的哈希，写入审计留痕")
    args = parser.parse_args()
    backup = _backup(args.db.resolve())

    with SessionLocal() as db:
        task = db.get(Task, args.task_id)
        if task is None or not task.op_name:
            raise SystemExit(f"task #{args.task_id} 不存在或没有 op_name")
        project = db.get(Project, task.project_id)
        if project is None or not project.workspace_path:
            raise SystemExit("任务工作区不存在")
        workspace = Path(project.workspace_path).expanduser()
        scoped_snapshot = exec_actions.operator_snapshot(workspace, task.op_name)
        scoped_suite_sha = case_confirm.cases_suite_sha256(workspace, task.op_name)
        scoped_suite_ref = case_confirm.cases_ref(workspace, task.op_name)
        if not scoped_snapshot.sha256 or not scoped_suite_sha or not scoped_suite_ref:
            raise SystemExit("当前算子代码/用例不齐，拒绝迁移")

        rows = [db.get(Run, run_id) for run_id in args.run_ids]
        if any(row is None or row.task_id != task.id for row in rows):
            raise SystemExit("存在不属于目标任务的 run")
        for row in rows:
            assert row is not None
            verdict = gate.evaluate(row).get("verdict")
            if row.status != "ok" or verdict != "pass":
                raise SystemExit(f"run #{row.id} 不是有效 PASS，拒绝迁移")
            original_snapshot = row.code_snapshot_sha256
            original_suite = row.suite_ref
            row.evidence_scope = "workspace-v1+operator-alias"
            row.scoped_code_snapshot_sha256 = scoped_snapshot.sha256
            row.scoped_suite_ref = scoped_suite_ref
            audit = (
                f"evidence_scope_migration=operator-v1 at={datetime.now(timezone.utc).isoformat()} "
                f"verified_source_sha256={args.verified_source_sha256} "
                f"original_snapshot={original_snapshot} original_suite={original_suite}"
            )
            row.note = f"{row.note or ''}\n{audit}".strip()

        task.cases_confirmed_sha256 = scoped_suite_sha
        task.cases_confirmed_scope = "operator-v1"
        task.cases_confirmed_by = f"{task.cases_confirmed_by or 'api'}; scope-migration"
        cur = db.query(TaskVersion).filter(
            TaskVersion.task_id == task.id, TaskVersion.is_current.is_(True)
        ).order_by(TaskVersion.id.desc()).first()
        if cur is not None:
            cur.code_snapshot_sha256 = scoped_snapshot.sha256
            cur.suite_ref = scoped_suite_ref
            cur.evidence_scope = "operator-v1"
            cur.note = f"{cur.note or ''}\n证据作用域迁移：operator-v1（原 run/manifest 保留）".strip()
        db.commit()

    print(f"backup={backup}")
    print(f"task={args.task_id} op={task.op_name}")
    print(f"operator_snapshot={scoped_snapshot.sha256} files={scoped_snapshot.files}")
    print(f"suite_sha256={scoped_suite_sha} suite_ref={scoped_suite_ref}")
    print(f"aliased_runs={','.join(map(str, args.run_ids))}")


if __name__ == "__main__":
    main()
