"""阶段看板（P3 读侧）：从真实证据推导阶段 + 按证据同步（写 stage_events）。"""

from __future__ import annotations

from datetime import datetime, timezone

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.models import Project, Run, SpecVersion, StageEvent, Task, TaskVersion
from app.services import stageboard

SNAP = "a" * 64
DIGEST = "sha256:deadbeef"
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@0123456789abcdef"
VER = {"spec_version_id": 1, "code_snapshot_sha256": SNAP, "criteria_ref": CRITERIA, "suite_ref": SUITE}


def _mk(tmp_path: Path, name: str, *, correctness: dict | None = None,
        performance: dict | None = None, smoke: dict | None = None) -> Path:
    rd = tmp_path / name
    (rd / "out").mkdir(parents=True)
    arts: list[dict] = []

    def add(rel: str, body: bytes, atype: str) -> None:
        p = rd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        arts.append({"type": atype, "path": rel, "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)})

    add("out/run.log", b"ok\n", "log")
    if correctness is not None:
        add("out/correctness.json", json.dumps(correctness).encode(), "correctness")
    if performance is not None:
        add("out/performance.json", json.dumps(performance).encode(), "performance")
    if smoke is not None:
        add("out/smoke.json", json.dumps(smoke).encode(), "smoke")
    man = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0,
           "code_snapshot_sha256": SNAP, "spec_version_id": 1, "criteria_ref": CRITERIA, "suite_ref": SUITE,
           "image_digest": DIGEST, "image": "x", "device": "0", "lock_mode": "exclusive",
           "snapshot": {"files": 1, "truncated": False}, "artifacts": arts}
    (rd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return rd / "manifest.json"


def _run(db, task: Task, action: str, man: Path) -> Run:
    r = Run(task_id=task.id, action=action, status="ok", manifest_path=str(man), spec_version_id=1,
            code_snapshot_sha256=SNAP, suite_ref=SUITE, criteria_ref=CRITERIA, image_digest=DIGEST,
            device_id="0", exit_code=0)
    db.add(r)
    db.commit()
    return r


def test_spec_only_lands_in_development(seeded, db_session):
    view = stageboard.build_task_view(seeded["task"], [])
    assert view["derived_stage"] == "开发"
    assert view["stages"][0]["ok"] is True      # 需求与设计（有 spec）
    assert view["spec_version"] == 1
    assert "spec v1" in view["stages"][0]["evidence"]
    assert view["stages"][1]["ok"] is False     # 开发（还没 build）
    assert view["stages"][3]["ok"] is False     # 交付永不为真


def test_custom_command_does_not_replace_correctness(seeded, db_session, tmp_path):
    task = seeded["task"]
    formal = _run(db_session, task, "test", _mk(
        tmp_path, "formal", correctness={**VER, "cases": [
            {"case_id": "required", "status": "PASS", "required": True}
        ]}))
    custom = _run(db_session, task, "custom", _mk(tmp_path, "custom"))
    view = stageboard.build_task_view(task, [formal, custom])
    assert view["actions"]["test"]["verdict"] == "pass"
    assert view["actions"]["test"]["run_id"] == formal.id
    assert stageboard.build_task_view(task, [custom])["actions"]["test"] is None


def test_advances_with_build_then_test_benchmark(seeded, db_session, tmp_path: Path):
    task = seeded["task"]
    build = _run(db_session, task, "build", _mk(tmp_path, "b1"))
    view = stageboard.build_task_view(task, [build])
    assert view["actions"]["build"]["verdict"] == "pass"
    assert view["derived_stage"] == "开发"  # v2：开发段还要求 smoke（编译+最小用例）通过
    assert view["stages"][1]["ok"] is False

    smoke = _run(db_session, task, "smoke",
                 _mk(tmp_path, "s1", smoke={**VER, "verdict": "PASS",
                                            "case": {"case_id": "c-min", "status": "PASS"}}))
    view = stageboard.build_task_view(task, [build, smoke])
    assert view["actions"]["smoke"]["verdict"] == "pass"
    assert view["derived_stage"] == "验收"

    test = _run(db_session, task, "test",
                _mk(tmp_path, "t1", correctness={**VER, "cases": [{"case_id": "c1", "status": "PASS", "required": True}]}))
    bench = _run(db_session, task, "benchmark", _mk(tmp_path, "k1", performance={**VER}))
    view = stageboard.build_task_view(task, [build, smoke, test, bench])
    assert view["stages"][2]["ok"] is True
    assert view["derived_stage"] == "交付"  # 验收过了 → 当前在交付段（交付动作未实装）


def test_perf_loop_blocks_acceptance_until_diff_is_valid(seeded, db_session, tmp_path: Path):
    """A3 口径 B：有 perf Loop 时，阶段三要求最近一轮 perf-diff verdict=pass。"""
    from app.models import Loop

    task = seeded["task"]
    build = _run(db_session, task, "build", _mk(tmp_path, "b2"))
    smoke = _run(db_session, task, "smoke", _mk(tmp_path, "s2", smoke={**VER, "verdict": "PASS",
                                                                     "case": {"case_id": "c-min", "status": "PASS"}}))
    test = _run(db_session, task, "test",
                _mk(tmp_path, "t2", correctness={**VER, "cases": [{"case_id": "c1", "status": "PASS", "required": True}]}))
    bench = _run(db_session, task, "benchmark", _mk(tmp_path, "k2", performance={**VER}))
    runs = [build, test, bench]

    # 没有 perf Loop：不强制（普通任务按测量 + 分析验收）
    plain = stageboard.build_task_view(task, runs)
    assert plain["stages"][2]["ok"] is True

    # 有 perf Loop 但对比不完整 → 阶段三卡住
    loop = Loop(task_id=task.id, kind="perf", status="open")
    db_session.add(loop)
    db_session.flush()
    for r in runs:  # 只把本 Loop 的 run 归进去（不能混入冒烟轮/别的 Loop）
        r.loop_id = loop.id
    db_session.commit()
    perf = stageboard.perf_evidence(db_session, task)
    assert perf is not None and perf["verdict"] == "fail"
    blocked = stageboard.build_task_view(task, runs, perf=perf)
    assert blocked["stages"][2]["ok"] is False
    assert any("perf-diff" in e for e in blocked["stages"][2]["evidence"])


def test_stage_sync_endpoint_writes_event(client, workspace: Path):
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.flush()
        sv = SpecVersion(task_id=t.id, version=1, op_name="elementwise_add", spec_json="{}")
        db.add(sv)
        db.flush()
        t.current_spec_version_id = sv.id
        t.cases_confirmed_at = datetime.now(timezone.utc)  # v2：阶段一要求用例组已确认
        db.commit()
        tid = t.id

    board = c.get("/api/stages").json()
    assert board["total"] == 1
    col = next(x for x in board["columns"] if x["stage"] == "开发")
    assert [v["task_id"] for v in col["tasks"]] == [tid]
    assert tid in board["out_of_sync"]  # 存值还是「需求与设计」

    r = c.post(f"/api/tasks/{tid}/stage/sync")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True and body["from"] == "需求与设计" and body["to"] == "开发"

    with Session() as db:
        assert db.get(Task, tid).stage == "开发"
        ev = db.scalar(select(StageEvent).where(StageEvent.task_id == tid))
        assert ev is not None and ev.from_stage == "需求与设计" and ev.to_stage == "开发"

    # 再同步一次：已一致 → 不写事件
    assert c.post(f"/api/tasks/{tid}/stage/sync").json()["changed"] is False

def test_build_board_batched_queries_same_result(seeded, db_session, tmp_path: Path):
    """看板批量取数：查询数不随任务数线性增长，推导结果与 per-task 视图一致（审查 #12）。"""
    from sqlalchemy import event

    project = seeded["project"]
    task = seeded["task"]
    others: list[Task] = []
    for i in range(3):
        t = Task(project_id=project.id, title=f"t{i}", op_name="add")
        db_session.add(t)
        db_session.flush()
        sv = SpecVersion(task_id=t.id, version=1, op_name="add", spec_json="{}")
        db_session.add(sv)
        db_session.flush()
        t.current_spec_version_id = sv.id
        t.cases_confirmed_at = datetime.now(timezone.utc)  # v2：阶段一要求用例组已确认
        others.append(t)
    draft = Task(project_id=project.id, title="draft", is_draft=True)
    db_session.add(draft)
    db_session.flush()

    for t in [task, *others]:
        _run(db_session, t, "build", _mk(tmp_path, f"b-{t.id}"))
        _run(db_session, t, "smoke",
             _mk(tmp_path, f"s-{t.id}", smoke={**VER, "verdict": "PASS",
                                               "case": {"case_id": "c-min", "status": "PASS"}}))
        db_session.add(StageEvent(task_id=t.id, from_stage="需求与设计", to_stage="开发", actor="test"))
    db_session.commit()

    selects: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(db_session.bind, "before_cursor_execute", _record)
    try:
        board = stageboard.build_board(db_session)
    finally:
        event.remove(db_session.bind, "before_cursor_execute", _record)

    assert board["total"] == 4  # 草稿不进统计
    assert len(selects) <= 8, selects  # 5 次左右；不能是 1+5N
    expected = [task.id, *[t.id for t in others]]
    assert [v["task_id"] for v in board["columns"][2]["tasks"]] == expected  # build 过了 → 验收
    assert board["stored_columns"]["需求与设计"] == expected
    assert set(board["out_of_sync"]) == set(expected)

    # 结果与逐任务算的一致（批量不能改口径）
    for t in [task, *others]:
        runs = list(db_session.scalars(select(Run).where(Run.task_id == t.id).order_by(Run.id)))
        single = stageboard.build_task_view(t, runs, perf=stageboard.perf_evidence(db_session, t),
                                            spec_version=stageboard.spec_version_of(db_session, t))
        view = next(v for v in board["columns"][2]["tasks"] if v["task_id"] == t.id)
        assert view["derived_stage"] == single["derived_stage"]
        assert [s["ok"] for s in view["stages"]] == [s["ok"] for s in single["stages"]]


def test_task_stage_and_gates_endpoints(client, workspace: Path):
    c, Session = client
    with Session() as db:
        p = Project(name="p", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        t = Task(project_id=p.id, title="elementwise_add", op_name="elementwise_add")
        db.add(t)
        db.flush()
        sv = SpecVersion(task_id=t.id, version=1, op_name="elementwise_add", spec_json="{}")
        db.add(sv)
        db.flush()
        t.current_spec_version_id = sv.id
        t.cases_confirmed_at = datetime.now(timezone.utc)  # v2：阶段一要求用例组已确认
        db.commit()
        tid = t.id

    stage = c.get(f"/api/tasks/{tid}/stage").json()
    assert stage["derived_stage"] == "开发" and stage["in_sync"] is False

    gates = c.get(f"/api/tasks/{tid}/gates").json()
    assert [g["action"] for g in gates["gates"]] == ["build", "smoke", "test", "benchmark", "profile"]
    assert all(g["run_id"] is None for g in gates["gates"])
    # 未跑过的动作也必须给完整形状：前端按 fail_codes.length 渲染（缺字段会白屏）
    assert all(g["verdict"] is None and g["fail_codes"] == [] and g["warn_codes"] == []
               for g in gates["gates"])
    assert gates["perf"] is None


def test_accepted_legacy_optimization_is_shown_in_delivery(seeded, db_session, tmp_path: Path,
                                                            monkeypatch):
    """旧 Demo 已有人审采纳 + 同条件 A/B 通过时，不应仍卡在需求列。

    早期流程缺少 v2 的用例确认/build/smoke 证据；这里仅调整展示位置，门禁圆点仍如实为 false。
    """
    task = seeded["task"]
    task.cases_confirmed_at = None
    db_session.commit()
    test = _run(
        db_session,
        task,
        "test",
        _mk(tmp_path, "legacy-test", correctness={
            **VER,
            "cases": [{"case_id": "c1", "status": "PASS", "required": True}],
        }),
    )
    bench = _run(db_session, task, "benchmark", _mk(tmp_path, "legacy-bench", performance={**VER}))

    view = stageboard.build_task_view(task, [test, bench], perf={
        "loop_id": 5,
        "verdict": "pass",
        "conclusion": "improved",
        "loop_conclusion": "accepted",
        "accepted": True,
        "fail_codes": [],
    })

    assert view["derived_stage"] == "交付"
    assert view["stages"][0]["ok"] is False
    assert view["stages"][1]["ok"] is False
    assert view["stages"][2]["ok"] is True
    assert "已人工采纳" in view["next_step"]

    # 真正生成 delivery 后，不能仍显示“待生成交付包”。
    delivery = Run(id=99, task_id=task.id, action="delivery", status="ok",
                   spec_version_id=task.current_spec_version_id)
    original_gate_view = stageboard._gate_view
    monkeypatch.setattr(
        stageboard,
        "_gate_view",
        lambda run: {
            "run_id": 99,
            "status": "ok",
            "verdict": "pass",
            "fail_codes": [],
            "warn_codes": [],
        } if run.action == "delivery" else original_gate_view(run),
    )
    delivered = stageboard.build_task_view(task, [test, bench, delivery], perf=view["perf"])
    assert delivered["stages"][3]["ok"] is True
    assert "交付包已生成" in delivered["next_step"]
    assert "待生成" not in delivered["next_step"]


def test_archived_tasks_are_excluded_from_stage_board(seeded, db_session):
    task = seeded["task"]
    task.is_archived = True
    db_session.commit()

    board = stageboard.build_board(db_session)

    assert board["total"] == 0
    assert all(not column["tasks"] for column in board["columns"])


def test_task_list_hides_archived_by_default_but_supports_history_query(client, workspace: Path):
    c, Session = client
    with Session() as db:
        p = Project(name="archive-list", workspace_path=str(workspace))
        db.add(p)
        db.flush()
        active = Task(project_id=p.id, title="active")
        archived = Task(project_id=p.id, title="archived", is_archived=True)
        db.add_all([active, archived])
        db.commit()
        active_id, archived_id = active.id, archived.id

    default_ids = {row["id"] for row in c.get("/api/tasks").json()}
    history = c.get("/api/tasks?include_archived=true").json()
    history_by_id = {row["id"]: row for row in history}

    assert active_id in default_ids
    assert archived_id not in default_ids
    assert history_by_id[archived_id]["is_archived"] is True


def test_old_delivery_does_not_freeze_new_minor_version(seeded, db_session, monkeypatch):
    task = seeded['task']
    task.version_minor = 1
    cur = TaskVersion(task_id=task.id, major=1, minor=1, source='optimization',
                      spec_version_id=task.current_spec_version_id,
                      code_snapshot_sha256='b' * 64, suite_ref=SUITE,
                      is_current=True)
    db_session.add(cur)
    db_session.flush()
    runs = [Run(id=i, task_id=task.id, action=a, status='ok',
                spec_version_id=task.current_spec_version_id,
                code_snapshot_sha256='b' * 64, suite_ref=SUITE)
            for i, a in enumerate(['build', 'smoke', 'test', 'benchmark'], 1)]
    old = Run(id=5, task_id=task.id, action='delivery', status='ok',
              spec_version_id=task.current_spec_version_id,
              code_snapshot_sha256=SNAP, suite_ref=SUITE)
    monkeypatch.setattr(stageboard, '_gate_view', lambda r: {
        'run_id': r.id, 'status': 'ok', 'verdict': 'pass', 'fail_codes': [], 'warn_codes': []})
    perf = {'loop_id': 7, 'verdict': 'pass', 'accepted': True,
            'conclusion': 'no_significant_change'}
    view = stageboard.build_task_view(task, runs + [old], perf=perf)
    assert view['stages'][3]['ok'] is False
    assert '当前版本尚未交付' in view['next_step']
    assert '历史任务' not in view['next_step']
    cur.delivered_at = datetime.now(timezone.utc)
    # A timestamp alone cannot bless an old code snapshot or a different suite.
    assert not stageboard.build_task_view(task, runs + [old], perf=perf)['stages'][3]['ok']
    new = Run(id=6, task_id=task.id, action='delivery', status='ok',
              spec_version_id=task.current_spec_version_id,
              code_snapshot_sha256=cur.code_snapshot_sha256, suite_ref='cases@wrong')
    assert not stageboard.build_task_view(task, runs + [old, new], perf=perf)['stages'][3]['ok']
    new.suite_ref = SUITE
    view = stageboard.build_task_view(task, runs + [old, new], perf=perf)
    assert view['stages'][3]['ok'] is True
    assert '交付包已生成' in view['next_step']


def test_scoped_stage_uses_restored_code_not_newer_rejected_candidate(seeded, db_session, workspace, monkeypatch):
    from app.services import case_confirm, versions
    task = seeded["task"]
    op = task.op_name
    (workspace / "ops" / op).mkdir(parents=True)
    (workspace / "ops" / op / "kernel.cpp").write_text("restored baseline")
    (workspace / "cases" / op).mkdir(parents=True)
    (workspace / "cases" / op / "cases.json").write_text('{"cases": []}')
    task.cases_confirmed_scope = "operator-v1"
    task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace, op)
    snapshot = versions._snapshot(workspace, task)
    suite = case_confirm.cases_ref(workspace, op)
    task.criteria_ref = CRITERIA
    db_session.flush()
    monkeypatch.setattr(stageboard, "_gate_view", lambda r: {
        "run_id": r.id, "status": r.status, "verdict": "pass", "fail_codes": [], "warn_codes": []})
    actions = ["build", "smoke", "test", "benchmark", "profile"]
    restored = [Run(id=i+1, task_id=task.id, action=a, status="ok",
                    spec_version_id=task.current_spec_version_id,
                    code_snapshot_sha256=snapshot, suite_ref=suite, criteria_ref=CRITERIA)
                for i, a in enumerate(actions)]
    rejected = [Run(id=i+20, task_id=task.id, action=a, status="ok",
                    spec_version_id=task.current_spec_version_id,
                    code_snapshot_sha256="f" * 64, suite_ref=suite, criteria_ref=CRITERIA)
                for i, a in enumerate(actions)]
    view = stageboard.build_task_view(task, restored + rejected)
    assert [view["actions"][a]["run_id"] for a in actions] == [r.id for r in restored]
    assert view["stages"][1]["ok"] and view["stages"][2]["ok"]
    assert view["run_count"] == 10  # Historical rows are preserved.
    only_stale = stageboard.build_task_view(task, rejected)
    assert all(v is None for v in only_stale["actions"].values())
    assert not only_stale["stages"][1]["ok"]
    # The task header must use the same identity filter as the stage board.
    from app.routers.tasks import task_gates
    db_session.add_all(restored + rejected)
    db_session.flush()
    gates = task_gates(task.id, db_session)
    assert [g["run_id"] for g in gates["gates"]] == [r.id for r in restored]
    # A later failure for the current code must not be hidden by an older pass.
    failed = Run(id=30, task_id=task.id, action="test", status="error",
                 spec_version_id=task.current_spec_version_id,
                 code_snapshot_sha256=snapshot, suite_ref=suite, criteria_ref=CRITERIA)
    db_session.add(failed)
    db_session.flush()
    assert next(g for g in task_gates(task.id, db_session)["gates"]
                if g["action"] == "test")["run_id"] == failed.id
    # Actual edits invalidate previously passing evidence and case confirmation.
    (workspace / "ops" / op / "kernel.cpp").write_text("new untested code")
    assert not stageboard.build_task_view(task, restored)["stages"][1]["ok"]
    (workspace / "cases" / op / "cases.json").write_text('{"cases": ["changed"]}')
    drift = stageboard.build_task_view(task, restored)
    assert not drift["stages"][0]["ok"]
    assert drift["derived_stage"] == "需求与设计"
