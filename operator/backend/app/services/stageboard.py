"""阶段看板（P3 读侧 / 方案 §5）：从**真实证据**推导任务走到哪一阶段。

四阶段是线性的（需求与设计 → 开发 → 验收 → 交付），每段的门禁由**门禁校验器**（`gate.evaluate`）
判定，而不是信 `tasks.stage` 那个手填字段（它会长期停在种子值上）。本模块：

- 对 operator-v1 任务，取当前源码、规格、用例和判据下各动作最近一次 run；
- 按阶段门禁汇总成「哪一段过了」，推出**当前阶段**（第一段没过的地方）与 `next_hint`；
- 与 `tasks.stage`（存值）对比，标出不一致 —— 页面上可用「按证据同步」把 stage 与 `stage_events` 落正。

阶段门禁口径（§5，落到四动作）：
  一 · 需求与设计 = 有规格版本（`current_spec_version_id`）
  二 · 开发       = `build` 与 `smoke` 门禁 pass
  三 · 验收       = 最近一次 `test` 与 `benchmark` 门禁都 pass（精度 + 性能报告达标）
  四 · 交付       = 当前版本交付包门禁 pass
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, object_session

from ..models import Loop, LoopRunLink, Project, Run, SpecVersion, StageEvent, Task, TaskVersion
from . import gate, perf_diff

#: 四阶段（顺序即推进方向）
STAGES = ["需求与设计", "开发", "验收", "交付"]

#: 各阶段门禁：`actions` 表示需要哪几个动作最近一次 run 的门禁 pass
STAGE_GATES: list[dict] = [
    {"stage": "需求与设计", "name": "需求规格齐备 + 用例组已确认", "actions": [],
     "hint": "先在「任务开始」页定版，再在任务对话里确认精度/性能用例组"},
    {"stage": "开发", "name": "编译 + 最小用例冒烟通过", "actions": ["build", "smoke"],
     "hint": "先 build（容器内真编），再跑 smoke（编译+一条最小用例跑通）；都 pass 才算开发完成"},
    {"stage": "验收", "name": "精度 + 性能报告达标", "actions": ["test", "benchmark"],
     "hint": "跑 test 与 benchmark 且门禁都 pass"},
    {"stage": "交付", "name": "交付包齐备", "actions": ["delivery"],
     "hint": "在「交付」分区生成交付包（版本证据有效 + 5 件产物齐备）"},
]

ALL_ACTIONS = ("build", "smoke", "test", "benchmark", "profile")
DELIVERY_ACTION = "delivery"


def _latest_by_action(runs: list[Run]) -> dict[str, Run]:
    out: dict[str, Run] = {}
    for r in sorted(runs, key=lambda x: x.id):
        out[r.action] = r  # 同动作取最新
    return out


def spec_version_of(db: Session, task: Task) -> int | None:
    """当前规格的人看的版本号（`vN`），不是 `spec_versions.id`。"""
    if task.current_spec_version_id is None:
        return None
    row = db.get(SpecVersion, task.current_spec_version_id)
    return row.version if row is not None else None


def _spec_version_from_task(task: Task) -> int | None:
    """给没有 db 的调用方兜底：关系已加载时直接取 version。"""
    if task.current_spec_version_id is None:
        return None
    try:
        for row in task.spec_versions:
            if row.id == task.current_spec_version_id:
                return row.version
    except Exception:  # detached / 关系不可用：宁可显示未知，也不把 id 当版本号
        return None
    return None


def _gate_view(run: Run) -> dict:
    rep = gate.evaluate(run)
    return {
        "run_id": run.id,
        "status": run.status,
        "verdict": rep.get("verdict"),
        "fail_codes": [f.get("code") for f in rep.get("findings", []) if f.get("level") == "fail"],
        "warn_codes": [f.get("code") for f in rep.get("findings", []) if f.get("level") == "warn"],
        "code_snapshot_sha256": run.scoped_code_snapshot_sha256 or run.code_snapshot_sha256,
        "device_id": run.device_id,
        "loop_id": run.loop_id,
    }


def _perf_payload(loop: Loop, rep: dict) -> dict:
    """perf-diff 结论 → 看板/门禁共用的精简视图。"""
    return {
        "loop_id": loop.id,
        "verdict": rep.get("verdict"),
        "conclusion": rep.get("conclusion"),
        "loop_conclusion": loop.conclusion,
        "accepted": (loop.conclusion or "").strip().lower() == "accepted",
        "fail_codes": [f.get("code") for f in rep.get("findings", []) if f.get("level") == "fail"],
    }


def perf_evidence(db: Session, task: Task, *, cache: dict[int, dict] | None = None) -> dict | None:
    """本任务的性能对比证据（A3 口径 B）：**有 perf Loop 才要求**，没有返回 None。

    阶段三只认「最近一轮 perf Loop 的 perf-diff verdict=pass」——即数据有效（配对完整、
    同条件、精度不退化）。结论是 improved / no_significant_change / regressed 都算有结论，
    **不因「没提速」判失败**（不设加速比硬线）。

    ⚠️ 只取 `loop_id == 该 Loop` 的 run：任务里可能还有冒烟轮 / 其它 Loop 的 run，
    混进来会让配对与可比性判断失真（2026-10-20 实测：混入冒烟轮后 Loop #3 被误判 inconclusive）。

    `cache`：同一次看板构建内按 loop_id 复用 perf-diff 结果（perf-diff 要读盘，别重复算）。
    """
    loop = db.scalar(
        select(Loop).where(Loop.task_id == task.id, Loop.kind == "perf")
        .order_by(Loop.id.desc()).limit(1)
    )
    if loop is None:
        return None
    # 复用来的基线 run 只挂 loop_run_links（Step 1b），不走改 runs.loop_id；读取走并集，
    # 与建轮次前的行为一致（否则新轮次的 before 侧会从看板里消失）。
    from . import optimization  # 局部 import：避免与 optimization→perf_diff 的依赖顺序打架
    loop_runs = optimization.loop_runs(db, loop.id)
    return _perf_evidence_from_loop(loop, loop_runs, cache)


def _perf_evidence_from_loop(loop: Loop, loop_runs: list[Run],
                             cache: dict[int, dict] | None = None) -> dict:
    rep = cache.get(loop.id) if cache is not None else None
    if rep is None:
        rep = perf_diff.evaluate_loop(loop, loop_runs)
        if cache is not None:
            cache[loop.id] = rep
    return _perf_payload(loop, rep)


def build_task_view(task: Task, runs: list[Run], last_event: StageEvent | None = None,
                    perf: dict | None = None, spec_version: int | None = None,
                    current_versions: dict[int, TaskVersion] | None = None,
                    projects: dict[int, Project] | None = None) -> dict:
    """一个任务的阶段视图（`runs` 是该任务的全部真实执行，顺序不限）。

    `perf` 由 `perf_evidence()` 算出（A3）：有 perf Loop 时，阶段三多一道「对比报告有效」的门槛。
    `spec_version` 是当前规格的**版本号**（v1/v2），不是数据库 id；不传时从 task 关系兜底。
    """
    if spec_version is None:
        spec_version = _spec_version_from_task(task)
    db = object_session(task)
    if current_versions is not None:
        current_version = current_versions.get(task.id)
    elif db is not None:
        current_version = db.scalar(select(TaskVersion).where(
            TaskVersion.task_id == task.id, TaskVersion.is_current.is_(True)
        ).order_by(TaskVersion.id.desc()).limit(1))
    else:
        current_version = None

    spec_ok = task.current_spec_version_id is not None
    cases_ok = task.cases_confirmed_at is not None
    # operator-v1 任务只展示当前工作区的有效身份。回退后较新的候选 run
    # 仍属于历史，不能覆盖已恢复源码的证据；历史记录本身保持不变。
    scoped = (task.cases_confirmed_scope == "operator-v1" or
              (current_version is not None and current_version.evidence_scope == "operator-v1"))
    current_runs = runs
    identity = None
    if scoped:
        from . import versions
        project = projects.get(task.project_id) if projects is not None else (
            db.get(Project, task.project_id) if db is not None else None)
        ws = Path(project.workspace_path).expanduser() if project and project.workspace_path else None
        snapshot = versions._snapshot(ws, task)
        suite_sha, suite_ref = versions._suite(ws, task)
        criteria = task.criteria_ref or (project.criteria_ref if project else None)
        cases_ok = bool(cases_ok and suite_sha and task.cases_confirmed_sha256 == suite_sha)
        identity = {"code_snapshot_sha256": snapshot, "suite_ref": suite_ref,
                    "spec_version_id": task.current_spec_version_id}
        current_runs = [r for r in runs if snapshot and suite_ref
                        and r.spec_version_id == task.current_spec_version_id
                        and versions._effective_snapshot(r) == snapshot
                        and versions._effective_suite(r) == suite_ref
                        and (not criteria or r.criteria_ref == criteria)]
    latest = _latest_by_action(current_runs)
    actions: dict[str, dict | None] = {a: (_gate_view(latest[a]) if a in latest else None) for a in ALL_ACTIONS}
    latest_delivery = latest.get(DELIVERY_ACTION)
    delivery_view = _gate_view(latest_delivery) if latest_delivery is not None else None

    def stage_ok(stage: dict) -> bool:
        if stage["stage"] == "需求与设计":
            # 任务生命周期 v2：规格齐备 **且** 精度/性能用例组已在对话里确认
            return spec_ok and cases_ok
        if stage["stage"] == "交付":
            # 同一规格下采纳优化会升小版本；旧版本的包不能证明当前版本已交付。
            if latest_delivery is None or delivery_view is None:
                return False
            if delivery_view.get("verdict") != "pass" or latest_delivery.status != "ok":
                return False
            if latest_delivery.spec_version_id != task.current_spec_version_id:
                return False
            if current_version is not None:
                if current_version.delivered_at is None:
                    return False
                if (current_version.code_snapshot_sha256 and
                        current_version.code_snapshot_sha256 !=
                        (latest_delivery.scoped_code_snapshot_sha256 or latest_delivery.code_snapshot_sha256)):
                    return False
                if (current_version.suite_ref and current_version.suite_ref !=
                        (latest_delivery.scoped_suite_ref or latest_delivery.suite_ref)):
                    return False
            return True
        ok = all(actions[a] is not None and actions[a]["verdict"] == "pass" for a in stage["actions"])
        if stage["stage"] == "验收" and perf is not None:
            # A3 口径 B：有 perf Loop 时，最近一轮对比报告必须可判（verdict=pass）
            ok = ok and perf.get("verdict") == "pass"
        return ok

    stages = []
    for g in STAGE_GATES:
        ok = stage_ok(g)
        ev = []
        if g["stage"] == "需求与设计":
            if spec_ok:
                ev.append(f"spec v{spec_version}" if spec_version is not None else "spec（版本号缺失）")
            ev.append("用例组已确认" if cases_ok else "用例组未确认或已变更")
        for a in g["actions"]:
            if a == DELIVERY_ACTION:
                if latest_delivery is not None:
                    ev.append(f"delivery #{latest_delivery.id} 门禁 {delivery_view.get('verdict') if delivery_view else '—'}")
                else:
                    ev.append("delivery 未跑")
            elif actions[a]:
                ev.append(f"{a} #{actions[a]['run_id']} 门禁 {actions[a]['verdict']}")
            else:
                ev.append(f"{a} 未跑")
        if g["stage"] == "验收" and perf is not None:
            ev.append(f"perf-diff loop #{perf['loop_id']} verdict {perf.get('verdict')}"
                      f"（{perf.get('conclusion')}）")
        stages.append({**g, "ok": ok, "evidence": ev})

    # 当前阶段 = 第一段没过的地方（全过则停在最后一段）。
    # 历史优化 Demo 上线早于 v2 的「用例确认 + build/smoke」门禁；若已有同条件
    # test/benchmark PASS、perf-diff PASS 且 Loop 明确 accepted，则展示到「交付」待办列。
    # 早期缺失的圆点仍保持未达成，不伪造新证据。
    legacy_accepted = bool(not scoped and perf and perf.get("accepted") and next(
        (s["ok"] for s in stages if s["stage"] == "验收"), False
    ))
    if legacy_accepted:
        derived = "交付"
    else:
        derived = STAGES[-1]
        for s in stages:
            if not s["ok"]:
                derived = s["stage"]
                break

    cur_idx = STAGES.index(derived)
    next_stage = STAGES[cur_idx]
    next_hint = next((s["hint"] for s in stages if s["stage"] == next_stage), "")
    if next_stage == "需求与设计" and spec_ok and not cases_ok:
        next_hint = "规格已定版；请在任务对话里整组确认精度/性能用例"
    if legacy_accepted:
        delivery_ok = next((s["ok"] for s in stages if s["stage"] == "交付"), False)
        if delivery_ok:
            next_hint = "优化版本已人工采纳，交付包已生成；当前版本已完成交付并冻结"
        elif not all(s["ok"] for s in stages[:2]):
            next_hint = "优化版本已人工采纳；历史任务的 v2 前置证据缺失保持原样，待生成交付包"
        else:
            next_hint = "优化版本已人工采纳；当前版本尚未交付，待生成交付包"
    elif all(s["ok"] for s in stages):
        next_hint = "当前版本已完成交付；交付包与验证证据可在「记录与交付」查看"

    return {
        "task_id": task.id,
        "title": task.title,
        "op_name": task.op_name,
        "is_draft": bool(task.is_draft),
        "stored_stage": task.stage,
        "derived_stage": derived,
        "in_sync": task.stage == derived,
        "spec_version": spec_version,
        "version_minor": int(task.version_minor or 0),
        "cases_confirmed_at": task.cases_confirmed_at.isoformat() if task.cases_confirmed_at else None,
        "next_step": next_hint,
        "next_action": task.next_action,
        "stages": stages,
        "actions": actions,
        "evidence_identity": identity,
        "delivery": delivery_view,
        "run_count": len(runs),
        "perf": perf,
        "last_event": None if last_event is None else {
            "from": last_event.from_stage, "to": last_event.to_stage,
            "reason": last_event.reason, "actor": last_event.actor,
            "created_at": last_event.created_at.isoformat() if last_event.created_at else None,
        },
    }


def _last_events(db: Session, task_ids: list[int]) -> dict[int, StageEvent]:
    """一次查询取每个任务的**最近一条** stage_event（按 task 分组的 max(id) 子查询）。"""
    if not task_ids:
        return {}
    latest_ids = (
        select(func.max(StageEvent.id).label("event_id"))
        .where(StageEvent.task_id.in_(task_ids))
        .group_by(StageEvent.task_id)
        .subquery()
    )
    rows = db.scalars(select(StageEvent).where(StageEvent.id.in_(select(latest_ids.c.event_id))))
    return {row.task_id: row for row in rows}


def _spec_versions(db: Session, tasks: list[Task]) -> dict[int, int | None]:
    """一次查询取各任务的当前 spec 版本号（vN，不是 spec_versions.id）。"""
    wanted = {t.current_spec_version_id for t in tasks if t.current_spec_version_id is not None}
    version_by_id: dict[int, int] = {}
    if wanted:
        rows = db.execute(select(SpecVersion.id, SpecVersion.version)
                          .where(SpecVersion.id.in_(wanted))).all()
        version_by_id = {row.id: row.version for row in rows}
    return {t.id: version_by_id.get(t.current_spec_version_id) for t in tasks}


def _perf_by_task(db: Session, task_ids: list[int],
                  cache: dict[int, dict]) -> dict[int, dict]:
    """一次查询取各任务最近一轮 perf Loop + 该 Loop 的 run，再逐个算 perf-diff。

    `cache` 由调用方（`build_board`）持有：同一次构建内同一个 loop 只算一次。
    """
    if not task_ids:
        return {}
    loops = list(db.scalars(
        select(Loop).where(Loop.task_id.in_(task_ids), Loop.kind == "perf")
        .order_by(Loop.id.desc())
    ))
    loop_of_task: dict[int, Loop] = {}
    for loop in loops:
        loop_of_task.setdefault(loop.task_id, loop)  # 倒序 → 先到的是最新
    if not loop_of_task:
        return {}
    loop_ids = [loop.id for loop in loop_of_task.values()]
    runs_by_loop: dict[int, list[Run]] = {loop_id: [] for loop_id in loop_ids}
    seen: set[tuple[int, int]] = set()
    for run in db.scalars(select(Run).where(Run.loop_id.in_(loop_ids)).order_by(Run.id)):
        runs_by_loop[run.loop_id].append(run)
        seen.add((run.loop_id, run.id))
    # Step 1b：复用基线 run 走 link，不再改写 runs.loop_id；这里补上（批量子查询，避免 N+1）。
    for link in db.scalars(select(LoopRunLink).where(LoopRunLink.loop_id.in_(loop_ids))):
        if (link.loop_id, link.run_id) in seen:
            continue
        run = db.get(Run, link.run_id)
        if run is not None:
            runs_by_loop[link.loop_id].append(run)
    for rows in runs_by_loop.values():
        rows.sort(key=lambda r: r.id)
    return {task_id: _perf_evidence_from_loop(loop, runs_by_loop.get(loop.id, []), cache)
            for task_id, loop in loop_of_task.items()}


def build_board(db: Session, *, include_drafts: bool = False) -> dict:
    """整块看板：所有任务按推导阶段分组。

    批量取数（runs / 最近 stage_event / 当前 spec 版本 / perf loop 与 loop runs 各一次 `in_`
    查询），在 Python 分组后再逐任务算视图 —— 避免「任务数 N → 5N+ 次查询」的 N+1（审查 #12）。
    返回结构保持不变。
    """
    tasks = list(db.scalars(select(Task).order_by(Task.id)))
    visible = [t for t in tasks if not t.is_archived and (include_drafts or not t.is_draft)]
    views: list[dict] = []
    if visible:
        task_ids = [t.id for t in visible]
        runs_by_task: dict[int, list[Run]] = {task_id: [] for task_id in task_ids}
        for run in db.scalars(select(Run).where(Run.task_id.in_(task_ids)).order_by(Run.id)):
            runs_by_task[run.task_id].append(run)
        last_events = _last_events(db, task_ids)
        spec_versions = _spec_versions(db, visible)
        current_versions = {v.task_id: v for v in db.scalars(select(TaskVersion).where(
            TaskVersion.task_id.in_(task_ids), TaskVersion.is_current.is_(True)
        ).order_by(TaskVersion.id))}
        projects = {p.id: p for p in db.scalars(select(Project).where(
            Project.id.in_({t.project_id for t in visible})
        ))}
        perf_cache: dict[int, dict] = {}
        perf_by_task = _perf_by_task(db, task_ids, perf_cache)
        for t in visible:
            views.append(build_task_view(
                t, runs_by_task[t.id], last_events.get(t.id),
                perf=perf_by_task.get(t.id), spec_version=spec_versions.get(t.id),
                current_versions=current_versions,
                projects=projects,
            ))

    columns = {s: [] for s in STAGES}
    for v in views:
        columns[v["derived_stage"]].append(v)
    return {
        "stages": STAGES,
        "columns": [{"stage": s, "tasks": columns[s]} for s in STAGES],
        "stored_columns": {s: [v["task_id"] for v in views if v["stored_stage"] == s] for s in STAGES},
        "total": len(views),
        "out_of_sync": [v["task_id"] for v in views if not v["in_sync"]],
        "notes": [
            "阶段由真实证据推导；operator-v1 任务只计当前源码、规格、用例和判据对应的最近一次动作，历史执行仍保留。",
            "归档任务默认不进入主看板；任务、run、Loop 与日志证据仍完整保留。",
            "历史任务若已有 test/benchmark、有效 A/B 且优化 Loop 明确人工采纳，则展示到「交付」；"
            "旧流程缺失的用例确认、build/smoke 圆点仍保持未完成，不补造证据。",
        ],
    }
