"""P3 优化 Loop 的真实执行预算硬兜底与审计。

**并发前提**：当前部署是「单 uvicorn worker、单进程」（见 ops 启动方式），所以
`check-then-insert` 用模块级 `threading.Lock` 串行化就足够（FastAPI 的同步端点跑在线程池里，
两个请求可以真的并行；只有进程内互斥没有跨进程互斥）。**多 worker / 多实例部署时这里不够**，
必须换成数据库层原子写（`INSERT … SELECT WHERE count < limit`、或唯一约束 + 重试），
否则仍可能超发（审查 M3）。

预算的两条证据链（同一格预算不能被算两次）：
- 页面动作：allowed 审计（= 预留）→ ActionJob → Run；
- `scripts/nm_test.sh` 直跑：budget-check 新建 allowed 审计 → 脚本登记 Run。
`used_for` 取 `max(动作证据, allowed 审计数)`：同一尝试在两个口径上都只算一格，而还没落 Run
的预留先占住格子（避免「先点出去、脚本还没跑」的窗口超发）。
"""

from __future__ import annotations

import threading

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import ActionJob, Loop, LoopBudgetAudit, Run, Task, utcnow

#: check-then-insert 的进程内串行化。前提见模块 docstring：单 uvicorn worker。
#: 预算预留的「claim 一次」也必须走同一把锁，否则两个请求能同时把 claimed_at 从空改掉。
_LOCK = threading.Lock()


class BudgetExceeded(ValueError):
    pass


def limit_for(loop: Loop) -> int:
    objective = loop.objective or {}
    budget = objective.get("budget") if isinstance(objective, dict) else {}
    if not isinstance(budget, dict):  # 脏数据（budget 写成了字符串/列表）不能把门禁打成 500
        budget = {}
    try:
        value = int(budget.get("runs_per_action", 2))
    except (TypeError, ValueError):
        value = 2
    return max(1, value)


def used_for(db: Session, loop_id: int, action: str) -> int:
    jobs = int(db.scalar(select(func.count()).select_from(ActionJob).where(
        ActionJob.loop_id == loop_id, ActionJob.action == action
    )) or 0)
    linked = select(ActionJob.run_id).where(
        ActionJob.loop_id == loop_id, ActionJob.action == action,
        ActionJob.run_id.is_not(None),
    )
    # 运行中瞬时重复计数（审查 F2）：脚本跑完会**自己**先登记 run，而 ActionJob.run_id 要等
    # 本进程收尾（_finalize）才回填。这段窗口里同一个 run 同时出现在「ActionJob」与「Run」
    # 两条证据上 → jobs 和 direct_runs 各算一次。排除「同 loop+action 的正在执行作业所对应
    # 的 run_ref」即可，收尾回填后它自然只被 jobs 计数。
    inflight_refs = select(ActionJob.run_ref).where(
        ActionJob.loop_id == loop_id, ActionJob.action == action,
        ActionJob.run_id.is_(None), ActionJob.status == "running",
    )
    direct_runs = int(db.scalar(select(func.count()).select_from(Run).where(
        Run.loop_id == loop_id, Run.action == action, Run.id.not_in(linked),
        # Run.remote_workdir 可为 NULL：NULL NOT IN (...) 的结果是 NULL，会把这些历史 run
        # 静默漏掉，所以显式保留 NULL 行。
        or_(Run.remote_workdir.is_(None), Run.remote_workdir.not_in(inflight_refs)),
    )) or 0)
    measured_attempts = jobs + direct_runs
    reservations = int(db.scalar(select(func.count()).select_from(LoopBudgetAudit).where(
        LoopBudgetAudit.loop_id == loop_id, LoopBudgetAudit.action == action,
        LoopBudgetAudit.decision == "allowed",
    )) or 0)
    # 页面动作 = allowed 审计 + ActionJob；nm.sh 直跑 = allowed 审计 + Run。
    # 两条证据描述的是同一次尝试，取 max 可避免重复计数，同时预留尚未落 Run 的直跑名额。
    return max(measured_attempts, reservations)


def _claim_reservation(db: Session, task: Task, loop: Loop, action: str,
                       reservation_id: int) -> dict:
    """校验并消费一次服务端预留（调用方必须已持有 `_LOCK`）。

    预留由 `routers/runs.py::plan_action` 在起进程前写下（allowed 审计）。脚本执行前带着
    `OPX_BUDGET_AUDIT_ID` 回来 claim；**只能消费一次**（claimed_at），且必须与本次
    task/loop/action 对得上 —— 调用方无法用环境变量/伪造 id 跳过或重复使用门禁（审查 M4）。
    任何不满足都 fail-closed：不新建审计、不放行。
    """
    row = db.get(LoopBudgetAudit, reservation_id)
    valid = (
        row is not None
        and row.task_id == task.id
        and row.loop_id == loop.id
        and row.action == action
        and row.decision == "allowed"
        and row.claimed_at is None
    )
    if not valid:
        raise BudgetExceeded(
            f"预算预留 #{reservation_id} 无效或已被使用（loop/action 不匹配、已被消费或不是 allowed 审计）；"
            "拒绝执行，请从控制台页面重新发起动作"
        )
    row.claimed_at = utcnow()
    db.commit()
    return {"enforced": True, "reused": True, "audit_id": row.id,
            "limit": row.limit_count, "used": row.used_count}


def check_and_audit(db: Session, task: Task, loop: Loop, action: str, *,
                    reservation_id: int | None = None) -> dict:
    """允许/拒绝都写审计；只对带 objective 的新优化轮次启用。

    - `reservation_id` 不为空：消费服务端一次性预留（见 `_claim_reservation`），不再按用量判一次；
    - `reservation_id` 为空：脚本直跑 `budget-check` 的老路径，按当前用量新建一条审计。
    """
    if not loop.objective:
        return {"enforced": False}
    # 整段 check-then-insert（以及 claim）都在进程锁内：否则两个并发请求会同时读到 used=0
    # 各写一条 allowed（审查 M3 的复现）。见模块 docstring 的部署前提。
    with _LOCK:
        if reservation_id is not None:
            return _claim_reservation(db, task, loop, action, reservation_id)
        limit = limit_for(loop)
        used = used_for(db, loop.id, action)
        decision = "allowed" if used < limit else "rejected"
        note = (f"允许第 {used + 1}/{limit} 次 {action}"
                if decision == "allowed" else f"已用 {used}/{limit} 次，拒绝继续执行 {action}")
        row = LoopBudgetAudit(
            task_id=task.id, loop_id=loop.id, action=action, decision=decision,
            limit_count=limit, used_count=used, note=note,
        )
        db.add(row)
        db.commit()
        if decision == "rejected":
            raise BudgetExceeded(
                f"Loop #{loop.id} 的 {action} 预算已用完（{used}/{limit}）；"
                "请新建下一轮（失败/环境类尝试也会占用预算），拒绝不会产生 run"
            )
        return {"enforced": True, "limit": limit, "used": used,
                "remaining_after_start": limit - used - 1, "audit_id": row.id}


def void_audit(db: Session, audit_id: int | None) -> bool:
    """回收一次「已预留但没能真正启动」的额度（审查 M1）。

    `plan_action` 的次序是「先写 allowed 审计（预留）→ 再起进程」；启动前的检查
    （plan 不齐 / 并发 busy）失败时并没有产生 ActionJob/Run，如果不回收这格额度，
    用户就会因为一次无效点击被扣掉预算、无从重试。把该审计改成 rejected 后，
    `used_for` 的 reservations 口径就不再计它。返回是否真的回收（幂等：非 allowed 不改）。
    """
    if audit_id is None:
        return False
    row = db.get(LoopBudgetAudit, audit_id)
    if row is None or row.decision != "allowed":
        return False
    row.decision = "rejected"
    note = row.note or ""
    row.note = (note + "；" if note else "") + "动作启动失败，预留已回收"
    db.commit()
    return True


def list_audits(db: Session, loop_id: int) -> list[dict]:
    rows = db.scalars(select(LoopBudgetAudit).where(LoopBudgetAudit.loop_id == loop_id)
                      .order_by(LoopBudgetAudit.id.desc()).limit(100))
    return [{
        "id": row.id, "task_id": row.task_id, "loop_id": row.loop_id,
        "action": row.action, "decision": row.decision,
        "limit_count": row.limit_count, "used_count": row.used_count,
        "claimed_at": row.claimed_at,
        "note": row.note, "created_at": row.created_at,
    } for row in rows]
