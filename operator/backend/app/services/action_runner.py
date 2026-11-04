"""四动作「点一下就真执行」：把 plan 变成一次异步作业（2026-10-20）。

**为什么不让后端手拼 docker run**：配方（privileged / 设备映射 / set_env 链 / 超时与兜底清理 /
设备互斥）全部固化在 `scripts/nm_test.sh`（P1.5-a 踩出来的）。这里只做四件事：

1. 照 `ExecutionPlan` 组装环境变量（镜像 / 执行机 / 设备 / 规格 / 判据 / 任务）；
2. 起一个子进程跑 `nm_test.sh --run-id <ref> --action <动作>`，边跑边收日志；
3. 脚本跑完**自己登记 run**（它 POST `/api/tasks/{id}/runs`）—— 这里按 `run_ref`（= `runs.remote_workdir`）
   把那次 run 找回来；
4. 对找回来的 run 跑一次 `gate.evaluate`，把 verdict 记在作业行上（「退出码不是结论」）。

**边界**：只杀本地进程组时，远端容器仍由脚本自己的 `timeout` + `docker rm -f` 兜底清理
（`scripts/nm_test.sh` 已固化）；作业行只留日志尾部，完整日志在 `runs/<run_ref>/`。
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import SessionLocal
from ..models import ActionJob, Loop, LoopRunLink, Run, Savepoint, Task
from . import gate, loop_state, workspace_lease
from .exec_actions import ExecutionPlan

logger = logging.getLogger(__name__)

#: 作业日志只留尾部（完整日志在 runs/<run_ref>/，库里没必要存全量）
LOG_LIMIT = 200_000

#: 运行中的作业：action_jobs.id → 状态（同进程内可读实时日志；重启后从库里读尾部）
REGISTRY: dict[int, "JobState"] = {}


@dataclass
class JobState:
    job_id: int
    task_id: int
    proc: asyncio.subprocess.Process | None = None
    chunks: list[str] = field(default_factory=list)
    done: bool = False
    cancelled: bool = False

    def text(self) -> str:
        return "".join(self.chunks)[-LOG_LIMIT:]

    def add(self, line: str) -> None:
        self.chunks.append(line)
        total = sum(len(c) for c in self.chunks)
        if total > LOG_LIMIT * 2:  # 定期把前缀折叠掉，避免长跑把内存吃满
            self.chunks = [self.text()]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def script_path() -> Path:
    return Path(settings.repo_root) / "scripts" / "nm_test.sh"


def build_command(plan: ExecutionPlan, run_ref: str) -> list[str]:
    return ["bash", str(script_path()), "--run-id", run_ref, "--action", plan.action]


def build_env(task: Task, plan: ExecutionPlan, loop_id: int | None, *,
              budget_audit_id: int | None = None) -> dict[str, str]:
    """把 plan 里的版本/设备/执行机注入 nm_test.sh（环境变量优先于 ops/nm.conf）。"""
    env = os.environ.copy()
    # 预算相关的环境变量一律由本函数按本次动作重新决定：父进程 shell 里残留的同名变量
    # 不能泄漏给子进程（旧 `OPX_BUDGET_RESERVED` 是调用方自己就能设的绕过开关，已删除）。
    env.pop("OPX_BUDGET_RESERVED", None)
    env.pop("OPX_BUDGET_AUDIT_ID", None)
    env.pop("OPX_LOOP_ID", None)
    env.update({
        "OPX_WORKSPACE": plan.workspace,
        "OPX_REMOTE_ROOT": str(Path(plan.remote_dir).parent),
        "OPX_OP": task.op_name or "",
        "OPX_TASK_ID": str(task.id),
        "OPX_SPEC_VERSION_ID": str(plan.spec_version_id or ""),
        "OPX_CRITERIA_REF": plan.criteria_ref or "",
        "OPX_API": settings.console_api,
        "OPX_LOCK_MODE": "exclusive",
        "OPX_TIMEOUT": str(plan.timeout_s),
    })
    if plan.image_ref:
        env["OPX_IMAGE"] = plan.image_ref
    if plan.image_id:
        env["OPX_IMAGE_ID"] = str(plan.image_id)
    if plan.host_id:
        env["OPX_HOST_ID"] = str(plan.host_id)
    # 页面登记的 host 是执行目标的唯一真相源：不注入这两项，nm_test.sh 会退回
    # ops/nm.conf 里写死的 OPX_SSH（默认 tk36），出现「页面选 A、实际跑 B」。
    if plan.host_ref:
        env["OPX_SSH"] = plan.host_ref
    if plan.host_ssh_port:
        env["OPX_SSH_PORT"] = str(plan.host_ssh_port)
    if plan.host_ssh_key:
        env["OPX_SSH_KEY"] = plan.host_ssh_key
    if plan.device_id:
        env["OPX_DEVICES"] = str(plan.device_id)
    if loop_id:
        env["OPX_LOOP_ID"] = str(loop_id)
        # P3 预算预留：/api/tasks/{id}/actions 已在起进程前写过 allowed 审计并把 id 传下来。
        # 脚本执行前带这个 id 回 /budget-check claim（服务端一次性校验，防伪造/复用）；
        # 没有预留（如旧无 objective Loop）就不传，脚本走「新建审计」的直跑路径。
        if budget_audit_id:
            env["OPX_BUDGET_AUDIT_ID"] = str(budget_audit_id)
    return env


def running_for_task(task_id: int) -> list[int]:
    return [jid for jid, st in REGISTRY.items() if not st.done and st.task_id == task_id]


def start(db: Session, task: Task, plan: ExecutionPlan, *, loop_id: int | None = None,
          budget_audit_id: int | None = None, cmd: list[str] | None = None,
          run_ref: str | None = None) -> ActionJob:
    """起一次真实执行。返回作业行（立刻返回，不阻塞等容器）。

    `budget_audit_id`：调用方（plan_action）已写好的 P3 预算预留审计 id，透传成脚本的
    `OPX_BUDGET_AUDIT_ID`（脚本执行前 claim，只能消费一次）。
    """
    if not plan.ready:
        raise ValueError("前置不齐，先补齐：" + "；".join(plan.missing))
    busy = running_for_task(task.id)
    if busy:
        raise ValueError(f"任务 #{task.id} 已有正在执行的动作（作业 #{busy[0]}），等它结束再发")
    # Step 1：DB 级工作区租约（agent/action 共用），挡住“benchmark 跑着时 agent 改码”。
    # 失败抛 ValueError，由 plan_action 捕获并回收预算预留。
    try:
        lease = workspace_lease.acquire(db, task, kind="action")
    except workspace_lease.LeaseBusy as e:
        raise ValueError(str(e)) from e

    # 自动归档：每次真实执行前留一份可回滚的源码（best-effort，失败只 log 不影响执行）。
    # 懒加载 savepoints：它的 restore() 会反向懒加载本模块查运行中作业，顶层 import 会成环。
    if plan.workspace:
        try:
            from . import savepoints

            savepoints.auto_capture(db, task, Path(plan.workspace),
                                    label=f"{plan.action} 执行前自动存档")
        except Exception:  # noqa: BLE001 —— 自动归档失败不能阻断真执行
            logger.exception("动作执行前自动归档失败（task=%s, action=%s）", task.id, plan.action)

    # run_ref 必须能唯一定位到本任务的产物目录：带 task id 才能避免「两个任务同一秒」
    # 撞成同一个 runs/<ref>（进而被 delete_files 误删，审查 #7）；时间仍保持可读。
    ref = run_ref or f"r{task.id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    argv = cmd or build_command(plan, ref)
    env = build_env(task, plan, loop_id, budget_audit_id=budget_audit_id)
    job = ActionJob(
        task_id=task.id, loop_id=loop_id, action=plan.action, status="running",
        run_ref=ref, device_id=plan.device_id, command=" ".join(argv), started_at=_now(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    lease.job_id = job.id
    db.commit()

    state = JobState(job_id=job.id, task_id=task.id)
    REGISTRY[job.id] = state
    asyncio.get_running_loop().create_task(
        _run(job.id, argv, env, str(Path(settings.repo_root)), plan.timeout_s + 120, state)
    )
    return job


async def _run(job_id: int, argv: list[str], env: dict[str, str], cwd: str,
               timeout: int, state: JobState) -> None:
    rc: int | None = None
    error: str | None = None
    try:
        state.proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert state.proc.stdout is not None
        try:
            async with asyncio.timeout(timeout):
                async for raw in state.proc.stdout:
                    state.add(raw.decode("utf-8", "replace"))
                rc = await state.proc.wait()
        except TimeoutError:
            error = f"超时（{timeout}s）—— 已终止本地进程；远端容器由脚本自身的 timeout 兜底清理"
            _kill(state)
            rc = -1
    except Exception as e:  # 起不来（脚本不存在 / 权限）也要落一条可读的失败
        error = f"{type(e).__name__}: {e}"
    finally:
        state.done = True
        _finalize(job_id, rc if rc is not None else -1, error, state)


def _kill(state: JobState) -> None:
    proc = state.proc
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def cancel(job_id: int) -> str:
    """取消一次动作作业（本地进程组；远端由脚本 timeout 兜底）。"""
    state = REGISTRY.get(job_id)
    if state is None or state.done:
        return f"作业 #{job_id} 不在运行（无法取消）"
    state.cancelled = True
    _kill(state)
    return f"作业 #{job_id} 已发送终止信号（远端容器随脚本 timeout 清理）"


def _maybe_candidate_savepoint(db: Session, job: ActionJob, run: Run) -> None:
    """benchmark 成功 + 该作业挂在有 objective 的优化轮次 → 自动落 `candidate` 存点。

    - 基线那一轮跳过（以**本轮最早 ok run 的快照**为准；历史数据可能因快照口径漂移
      导致基线存档 sha 与 run sha 不相等，不能只比存档）；
    - 候选侧必须有同 sha 的 test 成功 run —— 只跑 benchmark 不能生成候选（没有精度证据）；
    - 同 Loop 最新 candidate 同 sha → 去重；
    - 任何异常只 log + rollback 候选事务，不影响动作作业（best-effort）。
    """
    if job.action != "benchmark" or job.status != "ok" or job.loop_id is None:
        return
    sha = run.code_snapshot_sha256
    if not sha:
        return
    try:
        loop = db.get(Loop, job.loop_id)
        if loop is None or loop.objective is None:
            return
        task = db.get(Task, job.task_id)
        if task is None:
            return
        # 懒加载：savepoints 的 restore() 会反向 import 本模块查运行中作业，顶层 import 会成环
        from . import savepoints

        # 本轮可见的 run = 直接归属 ∪ 复用 link（Step 1b：before run 不再改写 runs.loop_id）。
        # 只按 loop_id 取会把 link-only 的基线 run 漏掉，导致把候选 run 误判成「基线那一轮」，
        # 候选存点永远不落 → 采纳门禁 G7 永久 409 E_ACCEPT_NO_CANDIDATE。
        loop_runs_cond = or_(
            Run.loop_id == loop.id,
            Run.id.in_(select(LoopRunLink.run_id).where(LoopRunLink.loop_id == loop.id)),
        )
        first_sha = db.scalar(
            select(Run.code_snapshot_sha256)
            .where(loop_runs_cond, Run.status == "ok",
                   Run.code_snapshot_sha256.is_not(None))
            .order_by(Run.id)
            .limit(1)
        )
        if first_sha == sha:
            return  # 这是基线那一轮（本轮最早的代码快照），不是候选
        baseline = db.scalar(
            select(Savepoint)
            .where(Savepoint.loop_id == loop.id, Savepoint.kind == "baseline")
            .order_by(Savepoint.id.desc())
            .limit(1)
        )
        if baseline is not None and baseline.code_snapshot_sha256 == sha:
            return  # 代码与基线存档一致（无实际改动）
        # 候选 = test + benchmark 成对证据；benchmark 单独成功不生成候选。
        # （run 的 gate verdict 由采纳门禁 G7 再强校验。）
        candidate_test = db.scalar(
            select(Run.id)
            .where(Run.loop_id == loop.id, Run.action == "test", Run.status == "ok",
                   Run.code_snapshot_sha256 == sha)
            .order_by(Run.id.desc())
            .limit(1)
        )
        if candidate_test is None:
            logger.info("候选 benchmark 已成功但缺同 sha 的 test run，暂不生成 candidate"
                        "（task=%s, loop=%s, sha=%s）", job.task_id, loop.id, sha[:12])
            return
        latest = db.scalar(
            select(Savepoint)
            .where(Savepoint.loop_id == loop.id, Savepoint.kind == "candidate")
            .order_by(Savepoint.created_at.desc(), Savepoint.id.desc())
            .limit(1)
        )
        if latest is not None and latest.code_snapshot_sha256 == sha:
            return  # 同 sha 去重
        savepoints.create_for_sha(
            db, task, sha, label=f"候选 · {loop.scope or f'Loop #{loop.id}'}",
            kind="candidate", loop_id=loop.id,
        )
        db.commit()
    except Exception:  # noqa: BLE001 —— 自动候选失败不能影响动作作业
        logger.exception("自动候选存档失败（task=%s, job=%s）", job.task_id, job.id)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


def _finalize(job_id: int, rc: int, error: str | None, state: JobState) -> None:
    with SessionLocal() as db:
        job = db.get(ActionJob, job_id)
        if job is None:
            return
        job.exit_code = rc
        job.log = state.text()
        job.ended_at = _now()
        if state.cancelled:
            job.status = "interrupted"
        elif error or rc != 0:
            job.status = "failed"
        else:
            job.status = "ok"
        if error:
            job.error = error

        # 脚本自己登记了 run：按 run_ref（= runs.remote_workdir）找回来
        run = db.scalar(
            select(Run).where(Run.task_id == job.task_id, Run.remote_workdir == job.run_ref)
            .order_by(Run.id.desc())
        )
        if run is not None:
            job.run_id = run.id
            try:
                job.gate_verdict = gate.evaluate(run).get("verdict")
            except Exception as e:  # 门禁算不出来不能把作业判死
                job.error = (job.error + "；" if job.error else "") + f"门禁判定失败：{type(e).__name__}: {e}"
        elif error is None and not state.cancelled:
            tail = [ln.strip() for ln in state.text().splitlines() if ln.strip()][-3:]
            tail_text = " | ".join(tail)[:600] if tail else "（空）"
            if rc == 0:
                # 退出码不是结论：脚本 exit 0 只是「容器跑完了」，没登记 run = 没有证据，
                # 不能显示绿色通过（审查 #8）。任务页据此提示「无台账」。
                job.status = "failed"
                job.error = f"脚本 exit 0 但未登记 run（台账回填失败）；日志尾部：{tail_text}"
            else:
                job.error = f"脚本没有登记 run（exit {rc}）；日志尾部：{tail_text}"
        db.commit()
        # P1 优化轮次：benchmark 成功后自动落候选存点。作业已先落库，候选另起事务 ——
        # 失败只 log + rollback 候选，不影响动作作业本身的结果。
        if run is not None:
            _maybe_candidate_savepoint(db, job, run)
        # Step 1：动作结束释放工作区租约，并把本轮 phase 从证据同步一次
        # （基线动作 → baseline_ready；候选动作 → awaiting_adoption/verification）。
        workspace_lease.release(db, task_id=job.task_id, kind="action", job_id=job.id,
                                commit=False)
        if job.loop_id is not None:
            loop = db.get(Loop, job.loop_id)
            if loop is not None and isinstance(loop.objective, dict):
                try:
                    loop_state.sync_phase(db, loop, event="action_done", commit=False)
                except Exception:  # noqa: BLE001 —— 同步失败不能把动作作业判死
                    logger.exception("动作结束后同步 phase 失败（loop=%s）", loop.id)
        db.commit()
    REGISTRY.pop(job_id, None)


def view(job: ActionJob) -> dict:
    """对外的作业视图：运行中优先给内存里的实时日志。"""
    state = REGISTRY.get(job.id)
    running = bool(state and not state.done)
    return {
        "id": job.id,
        "task_id": job.task_id,
        "loop_id": job.loop_id,
        "action": job.action,
        "status": "running" if running else job.status,
        "running": running,
        "run_ref": job.run_ref,
        "run_id": job.run_id,
        "device_id": job.device_id,
        "command": job.command,
        "log": state.text() if running else (job.log or ""),
        "exit_code": job.exit_code,
        "error": job.error,
        "gate_verdict": job.gate_verdict,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
    }
