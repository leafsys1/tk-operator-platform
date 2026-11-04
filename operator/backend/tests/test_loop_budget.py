"""预算硬门禁：并发不超发 / 一次性预留 / 启动失败回收 / 运行中不重复计数 / 旧 Loop 只读。

覆盖审查 `runs/review/2026-10-23-pull-1cf4408.md` 的 M1/M3/M4/F2/M5：

- M3：单进程内 check-then-insert 由模块锁串行化，两线程 barrier 下 limit=1 不得双放行；
- M4：服务端签发的一次性预留（reservation_id）只能消费一次；伪造 id / action、loop 不匹配 /
  重复使用都 fail-closed（不新建审计）；旧的无 objective Loop 不启用预算（enforced=False）；
- M1：`plan_action` 先审计后启动，启动失败（ValueError）要把 allowed 审计改成 rejected 回收额度；
- F2：脚本已登记 run、`ActionJob.run_id` 尚未回填的窗口里，同一次尝试不能被 jobs+run 数两遍；
- M5：无 objective 旧 Loop 的自动优化入口直接 409。

不碰真机：`action_runner.start` 在路由集成测试里被 monkeypatch，只验证「预留 → 透传 → 回收」这条胶水。
"""

from __future__ import annotations

from datetime import datetime, timezone

import json
import threading

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.models import (ActionJob, Host, HostImage, Loop, LoopBudgetAudit, Project, Run,
                        SpecVersion, Task)
from app.services import action_runner, case_confirm, loop_budget


def _seed_loop(db, task: Task, *, limit: int = 2) -> Loop:
    loop = Loop(task_id=task.id, kind="perf", status="open",
                objective_json=json.dumps({"type": "perf", "budget": {"runs_per_action": limit}}))
    db.add(loop)
    db.commit()
    db.refresh(loop)
    return loop


def _seed_runnable_api(Session, workspace, *, budget: int = 2) -> tuple[int, int]:
    """项目/执行机/镜像/任务/规格 + 带 objective 的 Loop：让 build 动作的 plan 是 ready。"""
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, env_recipe="docker run --privileged ...",
                          torch_version="2.10.0")
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id,
                          image_id=image.id)
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name="elementwise_add")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, op_name="elementwise_add",
                           spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        task.cases_confirmed_at = datetime.now(timezone.utc)  # v2：四动作 plan 要求用例组已确认
        task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace)
        loop = Loop(task_id=task.id, kind="perf", status="open",
                    objective_json=json.dumps({"type": "perf", "budget": {"runs_per_action": budget}}))
        db.add(loop)
        db.commit()
        return task.id, loop.id


# ---------------------------------------------------------------- 并发（M3）


def test_concurrent_check_and_audit_cannot_overbook(db_session, seeded):
    """两线程同时过门禁、limit=1：必须一放行一拒绝，allowed 审计只能有 1 条。"""
    task = seeded["task"]
    loop = _seed_loop(db_session, task, limit=1)
    task_id, loop_id = task.id, loop.id
    # 每线程用自己的 Session/Engine（SQLite 连接不能跨线程复用），指向同一个临时库文件。
    db_path = db_session.get_bind().url.database
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    barrier = threading.Barrier(2)
    allowed: list[dict] = []
    rejected: list[str] = []

    def attempt() -> None:
        with Session() as db:
            task_row = db.get(Task, task_id)
            loop_row = db.get(Loop, loop_id)
            barrier.wait(timeout=10)
            try:
                allowed.append(loop_budget.check_and_audit(db, task_row, loop_row, "benchmark"))
            except loop_budget.BudgetExceeded as e:
                rejected.append(str(e))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    engine.dispose()

    assert len(allowed) == 1, f"并发双放行：{allowed}"
    assert len(rejected) == 1 and "预算已用完" in rejected[0]
    db_session.expire_all()
    audits = list(db_session.scalars(select(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == loop_id).order_by(LoopBudgetAudit.id)))
    assert [a.decision for a in audits] == ["allowed", "rejected"]
    assert audits[0].limit_count == 1


# ---------------------------------------------------------------- 一次性预留（M4）


def test_reservation_can_only_be_claimed_once(db_session, seeded):
    task = seeded["task"]
    loop = _seed_loop(db_session, task)
    first = loop_budget.check_and_audit(db_session, task, loop, "benchmark")
    audit_id = first["audit_id"]

    claimed = loop_budget.check_and_audit(db_session, task, loop, "benchmark",
                                          reservation_id=audit_id)
    assert claimed == {"enforced": True, "reused": True, "audit_id": audit_id,
                       "limit": 2, "used": 0}
    row = db_session.get(LoopBudgetAudit, audit_id)
    assert row.decision == "allowed" and row.claimed_at is not None

    # 同一个预留 id 再用一次 → fail-closed（这正是旧 OPX_BUDGET_RESERVED 绕过的口子）
    with pytest.raises(loop_budget.BudgetExceeded, match="无效或已被使用"):
        loop_budget.check_and_audit(db_session, task, loop, "benchmark", reservation_id=audit_id)
    # 复用/被拒都不新建审计
    assert db_session.scalar(select(func.count()).select_from(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == loop.id)) == 1


def test_invalid_reservation_is_rejected_without_creating_audit(db_session, seeded):
    task = seeded["task"]
    loop = _seed_loop(db_session, task)
    audit_id = loop_budget.check_and_audit(db_session, task, loop, "benchmark")["audit_id"]

    # 伪造 id
    with pytest.raises(loop_budget.BudgetExceeded, match="无效或已被使用"):
        loop_budget.check_and_audit(db_session, task, loop, "benchmark", reservation_id=987654)
    # action 不匹配
    with pytest.raises(loop_budget.BudgetExceeded, match="无效或已被使用"):
        loop_budget.check_and_audit(db_session, task, loop, "test", reservation_id=audit_id)
    # loop 不匹配
    other = _seed_loop(db_session, task)
    with pytest.raises(loop_budget.BudgetExceeded, match="无效或已被使用"):
        loop_budget.check_and_audit(db_session, task, other, "benchmark", reservation_id=audit_id)

    # 非法尝试不新建审计、也不消耗合法预留
    assert db_session.scalar(select(func.count()).select_from(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == loop.id)) == 1
    assert db_session.get(LoopBudgetAudit, audit_id).claimed_at is None
    ok = loop_budget.check_and_audit(db_session, task, loop, "benchmark", reservation_id=audit_id)
    assert ok["reused"] is True


# ---------------------------------------------------------------- 启动失败回收（M1）


def test_void_audit_recycles_budget(db_session, seeded):
    task = seeded["task"]
    loop = _seed_loop(db_session, task, limit=1)
    first = loop_budget.check_and_audit(db_session, task, loop, "benchmark")
    assert loop_budget.used_for(db_session, loop.id, "benchmark") == 1

    assert loop_budget.void_audit(db_session, first["audit_id"]) is True
    assert loop_budget.void_audit(db_session, first["audit_id"]) is False  # 幂等
    assert loop_budget.void_audit(db_session, None) is False
    assert loop_budget.used_for(db_session, loop.id, "benchmark") == 0

    row = db_session.get(LoopBudgetAudit, first["audit_id"])
    assert row.decision == "rejected"
    assert "预留已回收" in (row.note or "")
    # 回收后同预算还能重新发起（不会被一次失败点击烧光）
    again = loop_budget.check_and_audit(db_session, task, loop, "benchmark")
    assert again["enforced"] is True


def test_plan_action_passes_reservation_id_to_start(client, workspace, monkeypatch):
    c, Session = client
    task_id, loop_id = _seed_runnable_api(Session, workspace)
    captured: dict = {}

    class FakeJob:
        id = 88
        run_ref = "rFAKE-88"

    def fake_start(db, task, plan, **kwargs):
        captured.update(kwargs)
        return FakeJob()

    monkeypatch.setattr(action_runner, "start", fake_start)
    action_runner.REGISTRY.clear()
    response = c.post(f"/api/tasks/{task_id}/actions",
                      json={"action": "build", "dry_run": False, "loop_id": loop_id})
    assert response.status_code == 200, response.text
    body = response.json()
    audit_id = body["budget"]["audit_id"]
    assert body["action_job_id"] == 88
    assert captured["loop_id"] == loop_id
    assert captured["budget_audit_id"] == audit_id
    with Session() as db:
        row = db.get(LoopBudgetAudit, audit_id)
        assert row.decision == "allowed" and row.claimed_at is None
        assert loop_budget.used_for(db, loop_id, "build") == 1


def test_plan_action_recycles_reservation_when_start_fails(client, workspace, monkeypatch):
    """启动失败（ValueError）→ 审计改 rejected，额度还回去，可以立刻重试（M1）。"""
    c, Session = client
    task_id, loop_id = _seed_runnable_api(Session, workspace)

    def boom(*args, **kwargs):
        raise ValueError("起个进程试试：测试注入失败")

    monkeypatch.setattr(action_runner, "start", boom)
    response = c.post(f"/api/tasks/{task_id}/actions",
                      json={"action": "build", "dry_run": False, "loop_id": loop_id})
    assert response.status_code == 409, response.text
    assert "测试注入失败" in response.text
    with Session() as db:
        rows = list(db.scalars(select(LoopBudgetAudit).where(
            LoopBudgetAudit.loop_id == loop_id)))
        assert len(rows) == 1
        assert rows[0].decision == "rejected"
        assert "预留已回收" in (rows[0].note or "")
        assert loop_budget.used_for(db, loop_id, "build") == 0

    class FakeJob:
        id = 99
        run_ref = "rFAKE-99"

    monkeypatch.setattr(action_runner, "start", lambda *a, **kw: FakeJob())
    retry = c.post(f"/api/tasks/{task_id}/actions",
                   json={"action": "build", "dry_run": False, "loop_id": loop_id})
    assert retry.status_code == 200, retry.text
    assert retry.json()["budget"]["used"] == 0  # 回收后重试看到的用量没有涨


def test_panel_reservation_is_claimed_via_budget_check_endpoint(client, workspace, monkeypatch):
    """脚本侧闭环：页面预留的 audit_id 交给 nm_test.sh，脚本带 OPX_BUDGET_AUDIT_ID 回来 claim。

    这是 `POST /api/tasks/{id}/actions` → `scripts/nm_test.sh` → `POST /api/loops/{id}/budget-check`
    的真实握手；同一个 id 只能 claim 一次，第二次必须 409（fail-closed）。
    """
    c, Session = client
    task_id, loop_id = _seed_runnable_api(Session, workspace)

    class FakeJob:
        id = 88
        run_ref = "rFAKE-88"

    monkeypatch.setattr(action_runner, "start", lambda *a, **kw: FakeJob())
    body = c.post(f"/api/tasks/{task_id}/actions",
                  json={"action": "build", "dry_run": False, "loop_id": loop_id}).json()
    audit_id = body["budget"]["audit_id"]

    claimed = c.post(f"/api/loops/{loop_id}/budget-check",
                     json={"action": "build", "reservation_id": audit_id})
    assert claimed.status_code == 200, claimed.text
    data = claimed.json()
    assert data["reused"] is True and data["audit_id"] == audit_id
    assert data["enforced"] is True

    again = c.post(f"/api/loops/{loop_id}/budget-check",
                   json={"action": "build", "reservation_id": audit_id})
    assert again.status_code == 409, again.text
    assert "无效或已被使用" in again.text


def test_budget_check_accepts_smoke_action(client, workspace):
    """二期自动链要跑 `--action smoke --loop`：budget-check 不能把 smoke 当非法 action 400。

    smoke 是独立动作、单独按 `loop×action` 计预算；非法 action 仍要 400。
    """
    c, Session = client
    task_id, loop_id = _seed_runnable_api(Session, workspace)
    r = c.post(f"/api/loops/{loop_id}/budget-check",
               json={"action": "smoke", "reservation_id": None})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enforced"] is True and body["audit_id"] is not None
    # 预算审计落在 action=smoke 上（与 test/benchmark 分账）
    with Session() as db:
        row = db.get(LoopBudgetAudit, body["audit_id"])
        assert row.action == "smoke" and row.decision == "allowed"

    bad = c.post(f"/api/loops/{loop_id}/budget-check",
                 json={"action": "profile-x", "reservation_id": None})
    assert bad.status_code == 400


def test_plan_action_busy_precheck_does_not_consume_budget(client, workspace):
    """已有作业在跑时先拒绝，不能先写 allowed 审计再失败（否则白扣一格额度）。"""
    c, Session = client
    task_id, loop_id = _seed_runnable_api(Session, workspace)
    action_runner.REGISTRY[777] = action_runner.JobState(job_id=777, task_id=task_id)
    try:
        response = c.post(f"/api/tasks/{task_id}/actions",
                          json={"action": "build", "dry_run": False, "loop_id": loop_id})
    finally:
        action_runner.REGISTRY.clear()
    assert response.status_code == 409 and "已有正在执行" in response.text
    with Session() as db:
        assert db.scalar(select(func.count()).select_from(LoopBudgetAudit).where(
            LoopBudgetAudit.loop_id == loop_id)) == 0


# ---------------------------------------------------------------- 运行中不重复计数（F2）


def test_inflight_run_is_not_double_counted(db_session, seeded):
    task = seeded["task"]
    loop = _seed_loop(db_session, task, limit=2)
    # 脚本已登记 run、ActionJob.run_id 还没回填的窗口：jobs 和 run 各算一次会虚增到 2。
    db_session.add(ActionJob(task_id=task.id, loop_id=loop.id, action="benchmark",
                             status="running", run_ref="rINFLIGHT-1"))
    db_session.add(Run(task_id=task.id, loop_id=loop.id, action="benchmark",
                       status="running", remote_workdir="rINFLIGHT-1"))
    db_session.commit()
    assert loop_budget.used_for(db_session, loop.id, "benchmark") == 1

    # remote_workdir 为 NULL 的历史 run 不能被 NOT IN 静默漏掉
    db_session.add(Run(task_id=task.id, loop_id=loop.id, action="benchmark",
                       status="ok", remote_workdir=None))
    db_session.commit()
    assert loop_budget.used_for(db_session, loop.id, "benchmark") == 2

    # 收尾回填后仍是「一次尝试 = 一格」
    job = db_session.scalar(select(ActionJob).where(ActionJob.run_ref == "rINFLIGHT-1"))
    run = db_session.scalar(select(Run).where(Run.remote_workdir == "rINFLIGHT-1"))
    job.run_id = run.id
    db_session.commit()
    assert loop_budget.used_for(db_session, loop.id, "benchmark") == 2


# ---------------------------------------------------------------- 旧 Loop 只读（M5）


def test_legacy_loop_without_objective_is_not_enforced(db_session, seeded):
    task = seeded["task"]
    legacy = Loop(task_id=task.id, kind="perf", status="open")
    db_session.add(legacy)
    db_session.commit()
    assert loop_budget.check_and_audit(db_session, task, legacy, "benchmark") == {
        "enforced": False}
    assert db_session.scalar(select(func.count()).select_from(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == legacy.id)) == 0


def test_malformed_budget_falls_back_to_default(db_session, seeded):
    """脏数据（budget 不是 dict）不能把门禁打成 500（审查 LOW 9）。"""
    task = seeded["task"]
    loop = Loop(task_id=task.id, kind="perf", status="open",
                objective_json=json.dumps({"type": "perf", "budget": "oops"}))
    db_session.add(loop)
    db_session.commit()
    assert loop_budget.limit_for(loop) == 2


def test_optimize_rejects_legacy_loop_without_objective(client, workspace):
    c, Session = client
    with Session() as db:
        project = Project(name="legacy", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="legacy", op_name="elementwise_add")
        db.add(task)
        db.flush()
        loop = Loop(task_id=task.id, kind="perf", status="open")
        db.add(loop)
        db.commit()
        loop_id = loop.id
    response = c.post(f"/api/loops/{loop_id}/optimize", json={})
    assert response.status_code == 409, response.text
    assert "旧 Loop（无 objective）已改为只读" in response.text
    assert "请新建优化轮次后再自动优化" in response.text
