"""四动作「点一下就真执行」：作业跑通 / 日志落库 / 找回 run + 门禁 / 并发拒绝 / 无台账判定。

不碰真机：作业命令由测试注入（`cmd=`），只验证后端这段胶水 ——
子进程、日志、退出码、`run_ref → runs.remote_workdir` 的找回、门禁 verdict 回填，
以及「脚本 exit 0 但没登记 run」不能显示成功（退出码不是结论）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import ActionJob, Run
from app.services import action_runner, case_confirm, exec_actions

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def runner_db(db_session, seeded, monkeypatch):
    """让作业收尾用**测试库**（生产里是 app.db.SessionLocal 指向 opx.db）。"""
    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(action_runner, "SessionLocal", Session)
    action_runner.REGISTRY.clear()
    yield db_session
    action_runner.REGISTRY.clear()


def _run_job(db, task, plan, **kw) -> int:
    """在事件循环里起作业并等它收尾（收尾时会把 REGISTRY 里的条目弹掉）。"""

    async def go() -> int:
        job = action_runner.start(db, task, plan, **kw)
        for _ in range(200):
            if job.id not in action_runner.REGISTRY:
                return job.id
            await asyncio.sleep(0.02)
        raise AssertionError("作业没有在 4s 内收尾")

    return asyncio.run(go())


def test_action_job_runs_and_captures_log(runner_db, seeded):
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")
    assert plan.ready, plan.missing

    job_id = _run_job(runner_db, task, plan, cmd=["bash", "-c", "echo BUILD_OK; echo tail"],
                      run_ref="rTEST-1")

    row = runner_db.get(ActionJob, job_id)
    assert row.status == "failed" and row.exit_code == 0  # 退出码 0 但没台账 → 不是成功
    assert "BUILD_OK" in (row.log or "") and "tail" in (row.log or "")
    assert row.run_ref == "rTEST-1"
    # 脚本没登记 run → 明确记一条原因，而不是假装成功
    assert row.run_id is None and "exit 0 但未登记 run" in (row.error or "")


def test_action_job_resolves_run_and_gate(runner_db, seeded, tmp_path):
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")
    # 预置脚本「会登记」的那条 run：run_ref 就是 remote_workdir
    run = Run(task_id=task.id, action="build", status="ok", exit_code=0,
              remote_workdir="rTEST-2", manifest_path=str(tmp_path / "missing" / "manifest.json"))
    runner_db.add(run)
    runner_db.commit()

    job_id = _run_job(runner_db, task, plan, cmd=["bash", "-c", "echo ok"], run_ref="rTEST-2")

    runner_db.expire_all()
    row = runner_db.get(ActionJob, job_id)
    assert row.run_id == run.id
    assert row.gate_verdict == "fail"  # 没有 manifest → 门禁判 fail，但作业本身仍然 ok
    assert row.status == "ok"


def test_action_job_nonzero_exit_is_failed(runner_db, seeded):
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")
    job_id = _run_job(runner_db, task, plan, cmd=["bash", "-c", "echo boom; exit 7"],
                      run_ref="rTEST-3")
    row = runner_db.get(ActionJob, job_id)
    assert row.status == "failed" and row.exit_code == 7
    assert "boom" in (row.log or "")


def test_same_task_concurrent_job_rejected(runner_db, seeded):
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")

    async def go():
        first = action_runner.start(runner_db, task, plan, cmd=["bash", "-c", "sleep 1"],
                                    run_ref="rTEST-4")
        with pytest.raises(ValueError, match="已有正在执行"):
            action_runner.start(runner_db, task, plan, cmd=["bash", "-c", "true"], run_ref="rTEST-5")
        for _ in range(200):
            if first.id not in action_runner.REGISTRY:
                break
            await asyncio.sleep(0.02)

    asyncio.run(go())


def test_run_ref_contains_task_id(runner_db, seeded):
    """run_ref 必须带 task id：否则两个任务同一秒会落到同一个 runs/<ref>（审查 #7）。"""
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")
    job_id = _run_job(runner_db, task, plan, cmd=["bash", "-c", "true"])
    runner_db.expire_all()
    row = runner_db.get(ActionJob, job_id)
    assert re.fullmatch(rf"r{task.id}-\d{{8}}-\d{{6}}", row.run_ref or ""), row.run_ref
    # 仍然秒级可读
    assert row.run_ref.split("-", 1)[1].replace("-", "").isdigit()


def _manifest_script() -> str:
    """取出 nm_test.sh 里写 manifest 的那段内嵌 Python（不跑真机也能验证字段）。"""
    text = (REPO_ROOT / "scripts" / "nm_test.sh").read_text(encoding="utf-8")
    start = text.index("import hashlib, json, os, sys, datetime")
    end = text.index("\nPY\n", start)
    return text[start:end]


def _write_manifest(base: Path, task_arg: str, *, device: str = "0",
                    requested: str = "0", switch_flag: str = "1") -> dict:
    out = base / "runs" / "r1-20261021-120000"
    (out / "out").mkdir(parents=True)
    (out / "out" / "run.log").write_text("ok\n", encoding="utf-8")
    snap = json.dumps({"sha256": "a" * 64, "files": 1, "truncated": False})
    argv = [str(out), out.name, "0", "tk36", "img:tag", "sha256:beef", snap, "build",
            "1", "crit@1", "cases@x", device, "exclusive", "yes", "1", "1",
            str(base), "off", "", task_arg, requested, switch_flag]
    proc = subprocess.run([sys.executable, "-c", _manifest_script(), *argv],
                          cwd=str(base), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_nm_test_manifest_carries_task_id(tmp_path: Path):
    """manifest 必须带 task_id（删除归属校验读它）；原有字段一个不少，空 task 为 None。"""
    man = _write_manifest(tmp_path / "with", "7")
    assert man["task_id"] == 7
    assert man["schema"] == "opx-manifest/1" and man["action"] == "build"
    assert man["code_snapshot_sha256"] == "a" * 64
    assert [a["type"] for a in man["artifacts"]] == ["log"]
    # 2026-10-28 自动切卡：实际设备与请求设备都记进 manifest，换过卡要能看出来
    assert man["device"] == "0" and man["device_requested"] == "0"
    assert man["device_auto_switched"] is False
    switched = _write_manifest(tmp_path / "switched", "7", device="3", requested="0")
    assert switched["device"] == "3" and switched["device_requested"] == "0"
    assert switched["device_auto_switched"] is True

    legacy = _write_manifest(tmp_path / "without", "")
    assert legacy["task_id"] is None  # 旧 manifest 无该字段时的兼容语义


def test_plan_and_env_bind_registered_host(runner_db, seeded):
    """执行目标必须跟页面登记的 host 走：否则「页面选 A、实际跑 B」。"""
    task = seeded["task"]
    host = seeded["host"]

    host.cred_ref = "~/.ssh/id_tk36"
    host.ssh_alias = "root@10.0.0.9:50023"  # 云主机控制台常见写法
    runner_db.commit()
    plan = exec_actions.build_plan(runner_db, task, "build")
    assert plan.host_id == host.id
    assert plan.host_ref == "root@10.0.0.9"  # 端口不能留在目标里
    assert plan.host_ssh_port == 50023
    assert plan.host_ssh_key == str(Path("~/.ssh/id_tk36").expanduser())
    env = action_runner.build_env(task, plan, None)
    assert env["OPX_SSH"] == "root@10.0.0.9"
    assert env["OPX_SSH_PORT"] == "50023"
    assert env["OPX_SSH_KEY"] == plan.host_ssh_key

    # 未登记私钥 → 回退全局 OPX_SSH_KEY（环境变量在 nm_test.sh 里优先于 ops/nm.conf）
    host.cred_ref = None
    host.ssh_alias = "tk36"
    runner_db.commit()
    plan2 = exec_actions.build_plan(runner_db, task, "build")
    assert plan2.host_ssh_port is None
    env2 = action_runner.build_env(task, plan2, None)
    assert env2["OPX_SSH_KEY"] == settings.ssh_key


def test_nm_test_scp_uses_registered_ssh_port():
    """runner/locker 上传也必须带页面登记的非默认 SSH 端口。"""
    script = (REPO_ROOT / "scripts" / "nm_test.sh").read_text(encoding="utf-8")
    assert 'SCP+=(-P "$OPX_SSH_PORT")' in script
    assert script.count('"${SCP[@]}"') == 2


def test_build_env_budget_reservation_env(db_session, seeded, monkeypatch):
    """预算预留只认服务端一次性 id（OPX_BUDGET_AUDIT_ID），旧 OPX_BUDGET_RESERVED 不再存在。"""
    task = seeded["task"]
    plan = exec_actions.build_plan(db_session, task, "build")
    monkeypatch.delenv("OPX_BUDGET_RESERVED", raising=False)
    monkeypatch.delenv("OPX_BUDGET_AUDIT_ID", raising=False)
    monkeypatch.delenv("OPX_LOOP_ID", raising=False)

    # 无预留：只透传 loop id，脚本会走「新建审计」的直跑路径
    env = action_runner.build_env(task, plan, 7)
    assert env["OPX_LOOP_ID"] == "7"
    assert "OPX_BUDGET_AUDIT_ID" not in env
    assert "OPX_BUDGET_RESERVED" not in env

    # 有服务端预留：注入一次性审计 id；旧的绕过开关不再出现
    env = action_runner.build_env(task, plan, 7, budget_audit_id=42)
    assert env["OPX_BUDGET_AUDIT_ID"] == "42"
    assert "OPX_BUDGET_RESERVED" not in env

    # 父进程 shell 里残留的旧变量/旧 id 不能泄漏给子进程
    monkeypatch.setenv("OPX_BUDGET_RESERVED", "1")
    monkeypatch.setenv("OPX_BUDGET_AUDIT_ID", "999")
    monkeypatch.setenv("OPX_LOOP_ID", "5")
    env = action_runner.build_env(task, plan, 7, budget_audit_id=42)
    assert env["OPX_LOOP_ID"] == "7"
    assert env["OPX_BUDGET_AUDIT_ID"] == "42"
    assert "OPX_BUDGET_RESERVED" not in env
    # 不是 Loop 的动作：连 loop id 都不传（避免父进程环境误挂到某轮）
    env = action_runner.build_env(task, plan, None)
    assert "OPX_LOOP_ID" not in env and "OPX_BUDGET_AUDIT_ID" not in env


def test_start_forwards_budget_audit_id_to_env(runner_db, seeded, monkeypatch):
    task = seeded["task"]
    plan = exec_actions.build_plan(runner_db, task, "build")
    seen: dict = {}
    real_build_env = action_runner.build_env

    def spy(task_arg, plan_arg, loop_id, *, budget_audit_id=None):
        seen["args"] = (loop_id, budget_audit_id)
        return real_build_env(task_arg, plan_arg, loop_id, budget_audit_id=budget_audit_id)

    monkeypatch.setattr(action_runner, "build_env", spy)

    async def go() -> None:
        job = action_runner.start(runner_db, task, plan, loop_id=3, budget_audit_id=42,
                                  cmd=["bash", "-c", "true"])
        for _ in range(200):
            if job.id not in action_runner.REGISTRY:
                return
            await asyncio.sleep(0.02)

    asyncio.run(go())
    assert seen["args"] == (3, 42)


def test_nm_test_budget_gate_uses_server_reservation():
    """脚本不再认 OPX_BUDGET_RESERVED；带 --loop 必须带 reservation_id 走服务端校验。"""
    script = (REPO_ROOT / "scripts" / "nm_test.sh").read_text(encoding="utf-8")
    # 旧的「调用方自己设开关跳过门禁」已删干净（注释里说明它被删掉不算）：
    assert "OPX_BUDGET_RESERVED=${OPX_BUDGET_RESERVED" not in script
    assert 'OPX_BUDGET_RESERVED" ]' not in script and "export OPX_BUDGET_RESERVED" not in script
    assert "OPX_BUDGET_AUDIT_ID=${OPX_BUDGET_AUDIT_ID:-}" in script
    assert '"reservation_id": reservation_id' in script
    # fail-closed：门禁失败仍然 exit 4，--loop 缺 --action 仍然 exit 2
    assert "预算门禁拒绝" in script and "exit 4" in script
    assert '--loop 需要同时指定 --action' in script


def test_endpoint_starts_job(client, workspace, monkeypatch):
    """`dry_run=false` 走异步作业，响应里带 action_job_id（真执行由上面几个测试覆盖）。"""
    from app.models import Host, HostImage, Project, SpecVersion, Task

    c, Session = client
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, env_recipe="docker run ...", torch_version="2.10")
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id, image_id=image.id)
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name="elementwise_add")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, op_name="elementwise_add", spec_json="{}",
                           sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        task.cases_confirmed_at = datetime.now(timezone.utc)  # v2：四动作 plan 要求用例组已确认
        task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace)
        db.commit()
        task_id = task.id

    class FakeJob:
        id = 77
        run_ref = "rFAKE-77"

    monkeypatch.setattr(action_runner, "start", lambda *a, **kw: FakeJob())
    body = c.post(f"/api/tasks/{task_id}/actions",
                  json={"action": "build", "dry_run": False}).json()
    assert body["action_job_id"] == 77 and body["run_ref"] == "rFAKE-77"
    assert body["ready"] is True
