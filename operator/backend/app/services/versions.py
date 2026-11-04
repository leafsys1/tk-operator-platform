"""任务版本（任务生命周期 v2，2026-10-23）。

模型（见 docs/plan-task-lifecycle-v2.md §4）：

- **大版本** = 当前规格版本号（`spec_versions.version`）：任务定版 = v1，落新规格 = v2…
- **小版本** = 采纳优化 / 交付前人工代码变更时 +1；大版本变更时归 0。
- **证据不按 run 打标**：一次 run 对当前版本有效，当且仅当
  `spec_version_id` / `code_snapshot_sha256` / `suite_ref` 三键与当前版本一致，且门禁 pass。
  因此「采纳优化」时候选 run 直接成为新小版本的证据 —— 不需要重跑。

对外接口（routers/versions.py 与 delivery 共用）：

- `current(db, task)` / `version_number(db, task)` / `history(db, task)`
- `create_spec_version(db, task, spec_row)` —— 定版 / 落新规格时调用（大版本 +1、小版本归 0、用例需重确认）
- `bump_minor(db, task, source=..., loop_id=..., label=..., note=...)` —— 采纳优化 / 手工小版本
- `evidence(db, task)` —— 交付门禁预览：必需动作缺什么、证据是否对得上当前版本
- `add_waiver(db, task, actions, reason, job_id)` —— 对话确认的证据豁免（写当前版本 `waivers_json`）
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Project, Run, SpecVersion, Task, TaskVersion
from . import case_confirm, gate

#: 交付必需动作（smoke 是正式 test/benchmark 的硬前置；profile 可选）
REQUIRED_ACTIONS = ("build", "smoke", "test", "benchmark")
OPTIONAL_ACTIONS = ("profile",)
ACTION_LABELS = {"build": "编译", "smoke": "冒烟", "test": "正确性", "benchmark": "基准", "profile": "采集"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def workspace_of(db: Session, task: Task) -> Path | None:
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        return None
    ws = Path(project.workspace_path).expanduser()
    return ws if ws.is_dir() else None


def current(db: Session, task: Task) -> TaskVersion | None:
    return db.scalar(
        select(TaskVersion).where(TaskVersion.task_id == task.id, TaskVersion.is_current.is_(True))
        .order_by(TaskVersion.id.desc()).limit(1)
    )


def history(db: Session, task_id: int) -> list[TaskVersion]:
    return list(db.scalars(
        select(TaskVersion).where(TaskVersion.task_id == task_id)
        .order_by(TaskVersion.major, TaskVersion.minor, TaskVersion.id)
    ))


def spec_version_number(db: Session, task: Task) -> int | None:
    if task.current_spec_version_id is None:
        return None
    row = db.get(SpecVersion, task.current_spec_version_id)
    return row.version if row is not None else None


def version_number(db: Session, task: Task) -> tuple[int | None, int]:
    """当前 `(大版本, 小版本)`；大版本取当前规格版本，小版本取 task.version_minor（无版本行时的兜底）。"""
    cur = current(db, task)
    if cur is not None:
        return cur.major, cur.minor
    return spec_version_number(db, task), int(task.version_minor or 0)


EVIDENCE_SCOPE = "operator-v1"


def _snapshot(ws: Path | None, task: Task) -> str | None:
    if ws is None or not task.op_name:
        return None
    try:
        from . import exec_actions  # 延迟导入：避免 services 层互相循环
        return exec_actions.operator_snapshot(ws, task.op_name).sha256 or None
    except Exception:  # noqa: BLE001 —— 快照失败只影响证据判定，不该让页面 500
        return None


def _suite(ws: Path | None, task: Task) -> tuple[str | None, str | None]:
    """(完整 ``cases/<op>`` 哈希, `cases@<16>` ref)。"""
    if ws is None or not task.op_name:
        return None, None
    try:
        return (case_confirm.cases_suite_sha256(ws, task.op_name),
                case_confirm.cases_ref(ws, task.op_name))
    except Exception:  # noqa: BLE001
        return None, None


def _clear_confirmation(task: Task) -> None:
    task.cases_confirmed_at = None
    task.cases_confirmed_sha256 = None
    task.cases_confirmed_scope = None
    task.cases_confirmed_by = None


def create_spec_version(db: Session, task: Task, spec_row: SpecVersion) -> TaskVersion:
    """定版 / 落新规格：大版本 = spec.version、小版本归 0、用例需重新确认。"""
    ws = workspace_of(db, task)
    _suite_sha, suite_ref = _suite(ws, task)
    for old in db.scalars(select(TaskVersion).where(TaskVersion.task_id == task.id,
                                                  TaskVersion.is_current.is_(True))):
        old.is_current = False
    row = TaskVersion(
        task_id=task.id, major=int(spec_row.version), minor=0, source="spec",
        spec_version_id=spec_row.id, code_snapshot_sha256=None, suite_ref=suite_ref,
        evidence_scope=EVIDENCE_SCOPE,
        label=f"规格 v{spec_row.version}", is_current=True, created_at=utcnow(),
    )
    db.add(row)
    task.version_minor = 0
    _clear_confirmation(task)
    db.flush()
    return row


def ensure_current_for_delivery(db: Session, task: Task) -> TaskVersion:
    """为早于 v2 的历史任务补当前版本行，不升版本、不清用例确认。

    老任务可能已有完整且与当前工作区匹配的 build/smoke/test/benchmark 证据，但没有
    ``task_versions`` 行。交付时必须补录这一行并冻结 ``delivered_at``，不能只生成 zip 后
    继续显示“版本行未落库”。这里固定的快照/用例集就是已通过交付预检的当前值。
    """
    cur = current(db, task)
    if cur is not None:
        return cur
    major = spec_version_number(db, task) or 0
    minor = int(task.version_minor or 0)
    ws = workspace_of(db, task)
    _suite_sha, suite_ref = _suite(ws, task)
    row = TaskVersion(
        task_id=task.id,
        major=major,
        minor=minor,
        source="spec",
        spec_version_id=task.current_spec_version_id,
        code_snapshot_sha256=_snapshot(ws, task),
        suite_ref=suite_ref,
        evidence_scope=EVIDENCE_SCOPE,
        label=f"v{major}.{minor}（历史任务交付补录）",
        note="首次交付时按已通过门禁的当前版本自动补录",
        is_current=True,
        created_at=utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def bump_minor(db: Session, task: Task, *, source: str, loop_id: int | None = None,
               label: str | None = None, note: str | None = None) -> TaskVersion:
    """小版本 +1。快照取**当前工作区**；采纳优化时工作区就是候选代码，所以候选 run 天然匹配。"""
    cur = current(db, task)
    major = cur.major if cur is not None else (spec_version_number(db, task) or 0)
    minor = (cur.minor if cur is not None else int(task.version_minor or 0)) + 1
    ws = workspace_of(db, task)
    _suite_sha, suite_ref = _suite(ws, task)
    if cur is not None:
        cur.is_current = False
    row = TaskVersion(
        task_id=task.id, major=major, minor=minor, source=source,
        spec_version_id=task.current_spec_version_id,
        code_snapshot_sha256=_snapshot(ws, task), suite_ref=suite_ref,
        evidence_scope=EVIDENCE_SCOPE,
        loop_id=loop_id, label=(label or f"v{major}.{minor}")[:200], note=note,
        is_current=True, created_at=utcnow(),
    )
    db.add(row)
    task.version_minor = minor
    db.flush()
    return row


def accept_optimization(db: Session, task: Task, loop, *, label: str | None = None,
                        note: str | None = None) -> TaskVersion | None:
    """采纳优化 → 小版本 +1（幂等：同一 loop 已有版本行则复用）。"""
    existing = db.scalar(select(TaskVersion).where(TaskVersion.task_id == task.id,
                                                   TaskVersion.loop_id == loop.id).limit(1))
    if existing is not None:
        return existing
    text = (label or "").strip() or f"采纳 Loop #{loop.id}" + (f" · {loop.scope[:60]}" if loop.scope else "")
    return bump_minor(db, task, source="optimization", loop_id=loop.id, label=text, note=note)


def add_waiver(db: Session, task: Task, *, actions: list[str], reason: str,
               job_id: int | None = None) -> dict:
    """对话确认的证据豁免：写当前版本 `waivers_json`（交付 manifest 会如实带出）。"""
    cur = current(db, task)
    if cur is None:
        cur = bump_minor(db, task, source="manual", label="版本基线（自动创建）",
                         note="首次写豁免时自动建版本行")
    try:
        rows = json.loads(cur.waivers_json or "[]")
    except ValueError:
        rows = []
    actions = [a for a in actions if a in (*REQUIRED_ACTIONS, *OPTIONAL_ACTIONS)]
    if not actions:
        raise ValueError("actions 至少包含一个有效动作（build/test/benchmark/profile）")
    rows.append({"actions": actions, "reason": reason, "job_id": job_id,
                 "at": utcnow().isoformat()})
    cur.waivers_json = json.dumps(rows, ensure_ascii=False)
    db.flush()
    return {"waivers": rows}


def _waived_actions(cur: TaskVersion | None) -> set[str]:
    if cur is None or not cur.waivers_json:
        return set()
    try:
        rows = json.loads(cur.waivers_json)
    except ValueError:
        return set()
    out: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            out.update(str(a) for a in (row.get("actions") or []))
    return out


def _effective_snapshot(run: Run) -> str | None:
    """返回交付判定使用的算子级快照。

    旧 run 的全工作区哈希和 manifest 保持不变；经核验后只补一个 scoped 别名。
    新 run 本身就是 operator-v1，直接使用原字段。
    """
    return run.scoped_code_snapshot_sha256 or run.code_snapshot_sha256


def _effective_suite(run: Run) -> str | None:
    return run.scoped_suite_ref or run.suite_ref


def _run_evidence(run: Run | None) -> dict | None:
    if run is None:
        return None
    try:
        verdict = gate.evaluate(run).get("verdict")
    except Exception:  # noqa: BLE001 —— 门禁判不了本身也是结论，不让页面炸
        verdict = None
    return {
        "run_id": run.id,
        "action": run.action,
        "status": run.status,
        "exit_code": run.exit_code,
        "gate_verdict": verdict,
        "spec_version_id": run.spec_version_id,
        "code_snapshot_sha256": run.code_snapshot_sha256,
        "suite_ref": run.suite_ref,
        "evidence_scope": run.evidence_scope,
        "effective_code_snapshot_sha256": _effective_snapshot(run),
        "effective_suite_ref": _effective_suite(run),
        "image_digest": run.image_digest,
        "device_id": run.device_id,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
    }


def evidence(db: Session, task: Task) -> dict:
    """交付门禁预览：逐动作证据 + 缺失原因 + 用例确认状态 + 豁免。"""
    ws = workspace_of(db, task)
    cur = current(db, task)
    major, minor = version_number(db, task)
    expected_spec_id = task.current_spec_version_id
    current_snapshot = _snapshot(ws, task)
    # 旧版本行没有 evidence_scope，其 code_snapshot 是全工作区哈希，不能拿来
    # 与新的算子级哈希比较。迁移期以当前算子指纹为基准，原行保留。
    expected_snapshot = (
        cur.code_snapshot_sha256
        if cur is not None and cur.evidence_scope == EVIDENCE_SCOPE and cur.code_snapshot_sha256
        else current_snapshot
    )
    suite_sha_now, suite_ref_now = _suite(ws, task)
    waived = _waived_actions(cur)

    runs = list(db.scalars(select(Run).where(Run.task_id == task.id).order_by(Run.id)))
    required_checks: list[dict] = []
    missing: list[dict] = []
    # 小版本会固化工作区快照。版本建立后若代码又被直接修改，旧 run 虽仍能匹配版本行，
    # 却不能证明即将打进交付包的当前代码；必须先建新小版本再重跑，不能 fail-open。
    if (cur is not None and cur.evidence_scope == EVIDENCE_SCOPE and cur.code_snapshot_sha256
            and current_snapshot != cur.code_snapshot_sha256):
        missing.append({"action": "version", "label": "当前版本",
                        "reason": "工作区代码与版本快照不一致（先创建新小版本并重跑）"})
    run_views: dict[str, dict | None] = {}
    for action in (*REQUIRED_ACTIONS, *OPTIONAL_ACTIONS):
        action_runs = [row for row in reversed(runs) if row.action == action]
        run = None
        # 不用“该动作最新一条”覆盖旧的有效证据；选最新一条与当前
        # (规格, 算子快照, 算子用例集) 一致且门禁通过的 run。失败历史仍保留在 run 列表。
        for candidate in action_runs:
            if candidate.status != "ok":
                continue
            if expected_spec_id is not None and candidate.spec_version_id != expected_spec_id:
                continue
            if expected_snapshot and _effective_snapshot(candidate) != expected_snapshot:
                continue
            if suite_ref_now and _effective_suite(candidate) != suite_ref_now:
                continue
            try:
                if gate.evaluate(candidate).get("verdict") != "pass":
                    continue
            except Exception:  # noqa: BLE001
                continue
            run = candidate
            break
        display_run = run or (action_runs[0] if action_runs else None)
        run_views[action] = _run_evidence(display_run)
        if action not in REQUIRED_ACTIONS:
            continue
        if action in waived:
            required_checks.append({"action": action, "ok": True, "waived": True})
            continue
        reason = None
        if run is None and not action_runs:
            reason = "未跑"
        elif run is None:
            reason = "没有与当前规格/算子代码/用例集一致且门禁通过的证据"
        elif run.status != "ok":
            reason = f"状态是 {run.status}（需重跑）"
        elif (run_views[action] or {}).get("gate_verdict") != "pass":
            reason = "门禁未过"
        elif expected_spec_id is not None and run.spec_version_id != expected_spec_id:
            reason = "规格版本不一致（大版本变更后需重跑）"
        elif expected_snapshot and _effective_snapshot(run) != expected_snapshot:
            reason = "代码快照不一致（版本变更后需重跑）"
        elif suite_ref_now and _effective_suite(run) != suite_ref_now:
            reason = ("run 缺少用例集标识（需重跑）" if not _effective_suite(run)
                      else "用例集不一致（用例变更后需重跑）")
        if reason:
            missing.append({"action": action, "label": ACTION_LABELS.get(action, action), "reason": reason})
        required_checks.append({"action": action, "ok": reason is None, "waived": False,
                                "run_id": run.id if run else None, "reason": reason})

    confirmation = case_confirm.confirmation(db, task)
    cases_ok = bool(confirmation.get("confirmed_at")) and bool(confirmation.get("in_sync"))
    if not cases_ok:
        missing.append({"action": "cases", "label": "用例组确认",
                        "reason": "未确认" if not confirmation.get("confirmed_at") else "用例集已变更，需重新确认"})

    # 交付包齐备性不在这里判（需要读工作区文件，由 delivery.py 做）；这里只回答「证据是否有效」。
    return {
        "major": major,
        "minor": minor,
        "version": f"v{major}.{minor}" if major is not None else "未定版",
        "required_actions": list(REQUIRED_ACTIONS),
        "optional_actions": list(OPTIONAL_ACTIONS),
        "runs": run_views,
        "checks": required_checks,
        "missing": missing,
        "waived_actions": sorted(waived),
        "deliverable": not missing,
        "expected_spec_version_id": expected_spec_id,
        "current_snapshot_sha256": current_snapshot,
        "expected_snapshot_sha256": expected_snapshot,
        "current_suite_sha256": suite_sha_now,
        "current_suite_ref": suite_ref_now,
        "cases": confirmation,
        "history_count": len(history(db, task.id)),
        "current_version_id": cur.id if cur is not None else None,
        "delivered_at": cur.delivered_at.isoformat() if cur is not None and cur.delivered_at else None,
    }
