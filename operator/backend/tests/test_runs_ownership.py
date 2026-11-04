"""runs 归属校验 + agent job 取消路由（审查 #13 / #1）。

覆盖：
- `POST /api/tasks/{id}/runs` 只接受属于该任务的 loop / spec_version；
- `manifest_path` / 带路径的 `remote_workdir` 必须落在允许的 runs/ 根目录下；
- `PATCH /api/runs/{id}` 不能把 run 挂到别的任务的 loop；
- `POST /api/agent-jobs/{id}/cancel` 的 404 / 409 / 正常调用契约。
"""

from __future__ import annotations

from pathlib import Path

from app.models import AgentJob, Loop, Project, Run, SpecVersion, Task
from app.services import agent_runner


def _seed(Session, workspace: Path) -> dict:
    with Session() as db:
        project = Project(name="p", workspace_path=str(workspace))
        db.add(project)
        db.flush()

        def mk_task(title: str) -> Task:
            task = Task(project_id=project.id, title=title, op_name=title)
            db.add(task)
            db.flush()
            spec = SpecVersion(task_id=task.id, version=1, op_name=title, spec_json="{}")
            db.add(spec)
            db.flush()
            task.current_spec_version_id = spec.id
            loop = Loop(task_id=task.id, kind="perf", status="open")
            db.add(loop)
            db.flush()
            return task, spec, loop

        a = mk_task("add")
        b = mk_task("relu")
        ids = {
            "workspace": workspace,
            "task_a": a[0].id, "spec_a": a[1].id, "loop_a": a[2].id,
            "task_b": b[0].id, "spec_b": b[1].id, "loop_b": b[2].id,
        }
        db.commit()
        return ids


def _run_dir(base: Path, name: str = "r1-20261021-120000") -> Path:
    rd = base / "runs" / name
    (rd / "out").mkdir(parents=True)
    (rd / "manifest.json").write_text("{}", encoding="utf-8")
    return rd


def test_register_run_rejects_cross_task_loop_and_spec(client, workspace: Path):
    c, Session = client
    ids = _seed(Session, workspace)

    body = {"action": "build", "status": "ok", "exit_code": 0,
            "loop_id": ids["loop_b"], "spec_version_id": ids["spec_a"]}
    r = c.post(f"/api/tasks/{ids['task_a']}/runs", json=body)
    assert r.status_code == 400 and "loop" in r.json()["detail"]

    body["loop_id"] = ids["loop_a"]
    body["spec_version_id"] = ids["spec_b"]
    r = c.post(f"/api/tasks/{ids['task_a']}/runs", json=body)
    assert r.status_code == 400 and "spec_version" in r.json()["detail"]

    # 都来自本任务 → 正常登记
    body["spec_version_id"] = ids["spec_a"]
    r = c.post(f"/api/tasks/{ids['task_a']}/runs", json=body)
    assert r.status_code == 201, r.text
    assert r.json()["loop_id"] == ids["loop_a"] and r.json()["spec_version_id"] == ids["spec_a"]


def test_register_run_checks_local_run_roots(client, workspace: Path):
    c, Session = client
    ids = _seed(Session, workspace)
    allowed = _run_dir(workspace)

    ok = c.post(f"/api/tasks/{ids['task_a']}/runs", json={
        "action": "build", "status": "ok", "manifest_path": str(allowed / "manifest.json"),
        "remote_workdir": allowed.name,
    })
    assert ok.status_code == 201, ok.text

    # 裸 remote_workdir（nm_test.sh 的写法）不当作路径校验
    bare = c.post(f"/api/tasks/{ids['task_a']}/runs", json={
        "action": "build", "status": "ok", "remote_workdir": "r-bare-name",
    })
    assert bare.status_code == 201, bare.text

    outside = workspace.parent / "outside-runs" / "manifest.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}", encoding="utf-8")
    bad_manifest = c.post(f"/api/tasks/{ids['task_a']}/runs", json={
        "action": "build", "status": "ok", "manifest_path": str(outside),
    })
    assert bad_manifest.status_code == 400 and "manifest_path" in bad_manifest.json()["detail"]

    bad_workdir = c.post(f"/api/tasks/{ids['task_a']}/runs", json={
        "action": "build", "status": "ok", "remote_workdir": str(outside.parent),
    })
    assert bad_workdir.status_code == 400 and "remote_workdir" in bad_workdir.json()["detail"]


def test_patch_run_loop_must_belong_to_same_task(client, workspace: Path):
    c, Session = client
    ids = _seed(Session, workspace)
    with Session() as db:
        run = Run(task_id=ids["task_a"], action="build", status="ok")
        db.add(run)
        db.commit()
        run_id = run.id

    bad = c.patch(f"/api/runs/{run_id}", json={"loop_id": ids["loop_b"]})
    assert bad.status_code == 400 and "不属于该 run 的任务" in bad.json()["detail"]

    ok = c.patch(f"/api/runs/{run_id}", json={"loop_id": ids["loop_a"]})
    assert ok.status_code == 200, ok.text
    assert ok.json()["loop_id"] == ids["loop_a"]


def test_agent_job_cancel_route_contract(client, workspace: Path, monkeypatch):
    """取消路由：不存在 404、已结束 409、运行中调 agent_runner.cancel 并回 {result}。"""
    c, Session = client
    ids = _seed(Session, workspace)

    calls: list[int] = []

    def fake_cancel(job_id: int) -> str:
        calls.append(job_id)
        return f"作业 #{job_id} 已发送终止信号"

    monkeypatch.setattr(agent_runner, "cancel", fake_cancel, raising=False)

    with Session() as db:
        running = AgentJob(task_id=ids["task_a"], kind="需求澄清", status="running")
        done = AgentJob(task_id=ids["task_a"], kind="需求澄清", status="ok")
        db.add_all([running, done])
        db.commit()
        running_id, done_id = running.id, done.id

    body = c.post(f"/api/agent-jobs/{running_id}/cancel")
    assert body.status_code == 200, body.text
    assert body.json() == {"result": f"作业 #{running_id} 已发送终止信号"}
    assert calls == [running_id]

    assert c.post(f"/api/agent-jobs/{done_id}/cancel").status_code == 409
    assert c.post("/api/agent-jobs/999999/cancel").status_code == 404
