"""P2 优化点搜索：只读诊断（工程强制）、服务端去重、用户思路入池、按点建独立 Loop。

审查 1cf4408 的 P2 项在这里落地：
- M6：``start`` 前抓工作区快照，``_finish`` 后比对 + 自动回滚 —— 只读不再只靠 prompt；
- M14：``_finish`` 先纯函数解析、再入库；入库/``_ingest`` 任何异常 rollback 后重标 failed，
  不会出现「搜索 failed 但新思路已落库」；
- M15：去重查重 + 唯一索引 ``IntegrityError`` 兜底，``_norm`` 归一到空时回退原文；
- M11：以 Loop 状态判定「一思路一 Loop」串行，模块级锁包住「检查 + 建轮 + 写 loop_id」；
- M10：证据包/提示词分项限长 + 总长硬上限，避免证据爆炸拖垮 build_prompt；
- M16：``SPEC_SEP`` 与兜底解析统一严格校验 ``ideas`` 必须是数组，prose 统一清洗。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..models import IdeaSearch, Loop, OptimizationIdea, Run, Task
from . import agent_runner, chat, gate, optimization, optimize
from .prose_relay import SPEC_SEP

#: 运行中的诊断：agent_jobs.id → {search_id, workspace, before(改前快照)}
#: 跑完回调取用；类似 ``optimize.PENDING``，只读强制的 before 快照就存在这里。
PENDING: dict[int, dict] = {}

#: 证据包/提示词体积上限（审查 M10）：单 Loop 证据可达 0.9MB，多轮后 prompt 会到数 MB。
MAX_EVIDENCE_CHARS = 120_000  # 证据包 JSON 总长硬上限（build_prompt 不会因此失败）
MAX_RUN_REPORT_CHARS = 8_000  # 单个 run 的报告 JSON 上限
MAX_LOOP_OPTIMIZE_CHARS = 2_000  # 每个 Loop 的 optimize 证据投影上限
MAX_IDEA_FIELD_CHARS = 2_000  # 每条思路 evidence/outcome 投影上限
MAX_IDEAS = 60  # 证据包里最多带这么多条思路
MAX_EXTRA_CHARS = 4_000  # 用户本次补充说明上限（schemas.IdeaSearchRequest 同步）

#: 思路状态：用户侧只能改这三个；selected/validating/verified 由系统（建轮/采纳/否决）写入。
USER_IDEA_STATUSES = frozenset({"proposed", "rejected", "archived"})
SYSTEM_IDEA_STATUSES = frozenset({"selected", "validating", "verified"})

#: 建轮「检查 → 建 Loop → 写 idea.loop_id」必须原子：FastAPI 同步端点在线程池里并发执行。
_ROUND_LOCK = threading.Lock()


class IdeaSearchError(ValueError):
    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _dumps(value) -> str:
    """证据包里的 JSON 序列化：坏对象也不能让 build_prompt 直接炸（default=str 兜底）。"""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "{}"


def _load(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _norm(value: str | None) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (value or "").strip().lower())


def _norm_or_raw(value: str | None) -> str:
    """归一化；纯符号标题归一后为空时回退原文（审查 M15：否则所有符号标题撞成一个 key）。"""
    text = (value or "").strip().lower()
    return _norm(text) or text


def dedupe_key(title: str, change_scope: str | None = None) -> str:
    raw = f"{_norm_or_raw(title)}|{_norm_or_raw(change_scope)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _find_duplicate(db: Session, task_id: int, key: str) -> OptimizationIdea | None:
    return db.scalar(select(OptimizationIdea).where(
        OptimizationIdea.task_id == task_id, OptimizationIdea.dedupe_key == key
    ).limit(1))


def idea_view(row: OptimizationIdea) -> dict:
    return {
        "id": row.id,
        "task_id": row.task_id,
        "loop_id": row.loop_id,
        "search_id": row.search_id,
        "source": row.source,
        "title": row.title,
        "evidence": _load(row.evidence_json),
        "hypothesis": row.hypothesis,
        "change_scope": row.change_scope,
        "expected": row.expected,
        "risk": row.risk,
        "confidence": row.confidence,
        "effort": row.effort,
        "skill": row.skill,
        "status": row.status,
        "dedupe_key": row.dedupe_key,
        "outcome": _load(row.outcome_json),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def search_view(row: IdeaSearch) -> dict:
    result = _load(row.result_json) or {}
    return {
        "id": row.id,
        "task_id": row.task_id,
        "agent_job_id": row.agent_job_id,
        "status": row.status,
        "summary": row.summary,
        "result": result,
        "created_at": row.created_at,
        "ended_at": row.ended_at,
    }


def list_ideas(db: Session, task_id: int) -> list[dict]:
    rows = db.scalars(
        select(OptimizationIdea)
        .where(OptimizationIdea.task_id == task_id)
        .order_by(OptimizationIdea.updated_at.desc(), OptimizationIdea.id.desc())
    )
    return [idea_view(row) for row in rows]


def add_user(db: Session, task: Task, payload: dict, *, source: str = "user") -> tuple[OptimizationIdea, bool]:
    title = str(payload.get("title") or "").strip()
    if not title:
        raise IdeaSearchError("思路标题不能为空")
    if len(title) > 300:
        raise IdeaSearchError("思路标题过长（最多 300 字）")
    scope = str(payload.get("change_scope") or "").strip() or None
    key = dedupe_key(title, scope)
    existing = _find_duplicate(db, task.id, key)
    if existing is not None:
        return existing, True
    row = OptimizationIdea(
        task_id=task.id,
        source=source,
        title=title,
        evidence_json=_json(payload.get("evidence")),
        hypothesis=str(payload.get("hypothesis") or "").strip() or None,
        change_scope=scope,
        expected=str(payload.get("expected") or "").strip() or None,
        risk=str(payload.get("risk") or "").strip() or None,
        confidence=str(payload.get("confidence") or "").strip() or None,
        effort=str(payload.get("effort") or "").strip() or None,
        skill=str(payload.get("skill") or "").strip() or None,
        status="proposed",
        dedupe_key=key,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # 唯一索引 uq_optimization_ideas_task_dedupe 兜底：并发插入撞车 → 重查按重复处理。
        db.rollback()
        existing = _find_duplicate(db, task.id, key)
        if existing is not None:
            return existing, True
        raise
    db.refresh(row)
    return row, False


# ---------------------------------------------------------------- 证据包（有界投影）


def _shrink_json(value, *, depth: int = 0):
    """把超大 JSON 递归瘦身（截字符串 / 截列表 / 限制深度），尽量保留有用字段。"""
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:300] + "…（截断）"
    if isinstance(value, dict):
        if depth >= 3:
            return {"truncated": True}
        return {str(k)[:80]: _shrink_json(v, depth=depth + 1)
                for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        if depth >= 3:
            return ["…（截断）"]
        items = [_shrink_json(v, depth=depth + 1) for v in value[:8]]
        if len(value) > 8:
            items.append({"truncated": True, "omitted": len(value) - 8})
        return items
    return value


def _bounded_json(value, limit: int, *, hint: str = "证据"):
    """把任意 JSON 证据压进 limit 字符：能放下原样返回，放不下先瘦身，再不行用截断投影。"""
    if value is None:
        return None
    text = _dumps(value)
    if len(text) <= limit:
        return value
    shrunk = _shrink_json(value)
    if len(_dumps(shrunk)) <= limit:
        return shrunk
    summary = ""
    if isinstance(value, dict):
        for key in ("summary", "conclusion", "verdict", "case_id", "source"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                summary = candidate.strip()[:300]
                break
    projected = {
        "truncated": True,
        "summary": f"{hint}过大（{len(text)} 字符），已省略明细",
        "keys": [str(k)[:80] for k in list(value)][:12] if isinstance(value, dict) else [],
    }
    if summary:
        projected["summary"] = summary
    return projected


def _bound_run_reports(reports: dict) -> dict:
    """每个 run 的 reports 聚合 JSON ≤ MAX_RUN_REPORT_CHARS（逐报告限长后再兜底降级）。"""
    if len(_dumps(reports)) <= MAX_RUN_REPORT_CHARS:
        return reports
    per_kind = {kind: _bounded_json(data, MAX_RUN_REPORT_CHARS, hint=f"{kind} 报告")
                for kind, data in reports.items()}
    if len(_dumps(per_kind)) <= MAX_RUN_REPORT_CHARS:
        return per_kind
    markers = {kind: {"truncated": True, "summary": f"{kind} 报告过大，已省略明细"}
               for kind in reports}
    if len(_dumps(markers)) <= MAX_RUN_REPORT_CHARS:
        return markers
    return {"truncated": True, "summary": "run 报告过多/过大，已全部省略明细"}


def _project_loop_optimize(value) -> dict | None:
    """Loop 的 optimize 证据只投影摘要字段，禁止把 diff/revert/agent_output 大字段塞进 prompt。"""
    if not isinstance(value, dict):
        return None
    projected = {
        key: value.get(key)
        for key in ("status", "changed", "agent_summary", "verification",
                    "guard_reverted", "after_sha256")
    }
    return _bounded_json(projected, MAX_LOOP_OPTIMIZE_CHARS, hint="Loop optimize 证据")


def _project_idea(idea: dict) -> dict:
    out = dict(idea)
    out["evidence"] = _bounded_json(idea.get("evidence"), MAX_IDEA_FIELD_CHARS, hint="思路证据")
    out["outcome"] = _bounded_json(idea.get("outcome"), MAX_IDEA_FIELD_CHARS, hint="思路结果")
    for key in ("hypothesis", "change_scope", "expected", "risk"):
        text = out.get(key)
        if isinstance(text, str) and len(text) > 1000:
            out[key] = text[:1000] + "…（截断）"
    return out


def _bounded_evidence(evidence: dict) -> dict:
    """证据包总长硬上限：超出逐步降级并打 ``truncated`` 标记，保证 json.dumps 可用。"""
    if len(_dumps(evidence)) <= MAX_EVIDENCE_CHARS:
        return evidence
    counts = {
        "runs": len(evidence.get("runs") or []),
        "loops": len(evidence.get("loops") or []),
        "ideas": len(evidence.get("ideas") or []),
        "skipped_run_reports": True,
    }
    trimmed = dict(evidence)
    trimmed["truncated"] = True
    trimmed["truncated_note"] = (
        f"证据包超过 {MAX_EVIDENCE_CHARS} 字符上限，已丢弃 run 报告明细"
    )
    trimmed["counts"] = counts
    trimmed["runs"] = [{"id": r.get("id"), "truncated": True} for r in evidence.get("runs") or []]
    if len(_dumps(trimmed)) <= MAX_EVIDENCE_CHARS:
        return trimmed
    # 还不够小：loops/ideas 也只留定位字段。
    trimmed["loops"] = [
        {"id": loop.get("id"), "status": loop.get("status"),
         "conclusion": loop.get("conclusion"), "truncated": True}
        for loop in (evidence.get("loops") or [])[:8]
    ]
    trimmed["ideas"] = [
        {"id": idea.get("id"), "title": str(idea.get("title") or "")[:120],
         "status": idea.get("status"), "source": idea.get("source"), "truncated": True}
        for idea in (evidence.get("ideas") or [])[:30]
    ]
    if len(_dumps(trimmed)) <= MAX_EVIDENCE_CHARS:
        return trimmed
    # 极端兜底：只留任务定位 + 计数，保证 build_prompt 永远能拼出可解析的 JSON。
    return {
        "truncated": True,
        "summary": "证据包远超上限，已整体省略；请仅依据任务描述与工作区只读证据诊断。",
        "task": evidence.get("task"),
        "counts": counts,
    }


def evidence_pack(db: Session, task: Task) -> dict:
    runs = list(db.scalars(
        select(Run).where(Run.task_id == task.id).order_by(Run.id.desc()).limit(16)
    ))
    loops = list(db.scalars(
        select(Loop).where(Loop.task_id == task.id).order_by(Loop.id.desc()).limit(12)
    ))
    ideas = list_ideas(db, task.id)
    run_rows = []
    for r in runs:
        reports: dict[str, object] = {}
        for kind, path in gate.artifact_paths(r).items():
            if kind not in {"performance", "correctness", "metrics", "profile"} or path.suffix != ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            reports[kind] = _bounded_json(data, MAX_RUN_REPORT_CHARS, hint=f"{kind} 报告")
        reports = _bound_run_reports(reports)
        run_rows.append({
            "id": r.id, "loop_id": r.loop_id, "action": r.action, "status": r.status,
            "snapshot": r.code_snapshot_sha256, "suite_ref": r.suite_ref,
            "criteria_ref": r.criteria_ref, "device_id": r.device_id,
            "image_digest": r.image_digest, "manifest_path": r.manifest_path,
            "reports": reports,
        })
    evidence = {
        "task": {"id": task.id, "op_name": task.op_name, "target_chip": task.target_chip},
        "runs": run_rows,
        "loops": [{
            "id": loop.id, "scope": loop.scope, "hypothesis": loop.hypothesis,
            "status": loop.status, "conclusion": loop.conclusion,
            "objective": _bounded_json(loop.objective, MAX_LOOP_OPTIMIZE_CHARS,
                                       hint="Loop objective"),
            "optimize": _project_loop_optimize(loop.optimize),
        } for loop in loops],
        "ideas": [_project_idea(idea) for idea in ideas[:MAX_IDEAS]],
    }
    if len(ideas) > MAX_IDEAS:
        evidence["ideas_truncated"] = len(ideas) - MAX_IDEAS
    return _bounded_evidence(evidence)


def build_prompt(db: Session, task: Task, workspace: Path, extra: str | None) -> str:
    evidence = evidence_pack(db, task)
    hardware = optimize.hardware_brief(workspace, task.target_chip) or "无可用硬件画像"
    extra_text = (extra or "").strip()
    if len(extra_text) > MAX_EXTRA_CHARS:
        extra_text = extra_text[:MAX_EXTRA_CHARS] + "…（已截断）"
    return f"""你是 op-agent 的优化点诊断 Agent。请针对算子 {task.op_name or task.title} 做一次有界、只读的优化点搜索。

硬约束：
- 这是诊断 job：只读代码、报告和已有结论；禁止修改任何文件，禁止调用 scripts/nm.sh，禁止发起 build/test/benchmark/profile。
- 控制台会在诊断前后比对工作区快照：**只要改动了工作区（含新增/删除文件），本次诊断一律判失败并自动回滚**。
- 只能引用下面证据包或工作区内能实际读到的证据；没有证据的猜想要降低 confidence，不能伪装成结论。
- 已采纳、已验证、已否决和已有 proposed 思路都在证据包里；不要换句话重复。已否决项是负样本。
- 最多返回 5 个高价值新思路。每个思路必须能作为“一思路一 Loop”串行验证，change_scope 只能在 ops/{task.op_name or ''}/ 下。
- 不写死 shape；目标用例来自报告。自动采纳永不允许。
- 证据包可能被截断：标注 truncated/omitted 的部分表示明细已省略，请以可见证据为准，不要臆造缺失数字。

工作区：{workspace}
硬件证据：{hardware}
用户本次补充：{extra_text or '无'}
证据包 JSON：
{_dumps(evidence)}

输出两段：先用中文简述证据与搜索结论；另起一行输出 {SPEC_SEP}，再输出严格 JSON：
{{"summary":"...","ideas":[{{"title":"...","evidence":{{}},"hypothesis":"...","change_scope":"ops/.../","expected":"...","risk":"...","confidence":"high|medium|low","effort":"small|medium|large","skill":"可选"}}],"updates":[{{"idea_id":1,"status":"proposed|verified|rejected|archived","outcome":{{}}}}]}}
JSON 后不要再输出任何文字。"""


# ---------------------------------------------------------------- 解析（纯函数，不碰 DB）


def _clean_prose(text: str | None) -> str:
    """prose 清洗：去掉 markdown fence 行 / 残留分隔符（审查 M16：兜底路径会残留 ```）。"""
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = cleaned.replace(SPEC_SEP, "\n")
    return cleaned.strip()


def _require_ideas(parsed) -> dict:
    if not isinstance(parsed, dict):
        raise IdeaSearchError("诊断 Agent 返回的 JSON 不是对象")
    ideas = parsed.get("ideas")
    if not isinstance(ideas, list):
        raise IdeaSearchError("诊断 Agent 返回的 ideas 缺失或不是数组")
    return parsed


def _parse_output(text: str) -> tuple[str, dict]:
    """解析 dsh 两段式输出；SPEC_SEP 分支与 fence/裸 JSON 兜底分支用同一套严格校验。"""
    raw = text or ""
    if SPEC_SEP in raw:
        prose, tail = raw.split(SPEC_SEP, 1)
        try:
            parsed = json.loads(tail.strip())
        except json.JSONDecodeError:
            parsed = None  # 分隔符后可能还包了 fence：继续走兜底扫描
        if isinstance(parsed, dict) and "ideas" in parsed:
            return _clean_prose(prose), _require_ideas(parsed)
    # 兼容模型包了 markdown fence / 只回裸 JSON：取最后一个含 ideas 的 JSON 对象。
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidates += list(chat._iter_json_objects(raw))  # 同一服务内复用成熟的平衡括号扫描
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or "ideas" not in parsed:
            continue
        return _clean_prose(raw[: raw.rfind(candidate)]), _require_ideas(parsed)
    raise IdeaSearchError("诊断 Agent 未返回可解析的 ideas JSON")


# ---------------------------------------------------------------- 入库（事务内，失败由 _finish 回滚）


def _ingest(db: Session, search: IdeaSearch, parsed: dict) -> dict:
    """入库；并发撞唯一索引时回滚并**重跑一次**（重跑时预查能看到已提交的重复 → duplicate）。

    不用 ``begin_nested``：SQLite + pysqlite 的延迟 BEGIN 下，SAVEPOINT 若成为最外层事务，
    RELEASE 会直接提交（等于提前 commit 半批数据），所以这里用「整事务回滚 + 幂等重放」。
    """
    search_id = search.id
    try:
        return _ingest_once(db, search, parsed)
    except IntegrityError:
        db.rollback()
        fresh = db.get(IdeaSearch, search_id)
        if fresh is None:
            raise
        return _ingest_once(db, fresh, parsed)


def _ingest_once(db: Session, search: IdeaSearch, parsed: dict) -> dict:
    new_ids: list[int] = []
    duplicate_ids: list[int] = []
    updated_ids: list[int] = []
    for item in parsed.get("ideas") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        scope = str(item.get("change_scope") or "").strip() or None
        key = dedupe_key(title, scope)
        existing = _find_duplicate(db, search.task_id, key)
        if existing is not None:
            duplicate_ids.append(existing.id)
            continue
        row = OptimizationIdea(
            task_id=search.task_id, search_id=search.id, source="agent", title=title[:300],
            evidence_json=_json(item.get("evidence")),
            hypothesis=str(item.get("hypothesis") or "").strip() or None,
            change_scope=scope,
            expected=str(item.get("expected") or "").strip() or None,
            risk=str(item.get("risk") or "").strip() or None,
            confidence=str(item.get("confidence") or "").strip() or None,
            effort=str(item.get("effort") or "").strip() or None,
            skill=str(item.get("skill") or "").strip() or None,
            status="proposed", dedupe_key=key,
        )
        db.add(row)
        db.flush()
        new_ids.append(row.id)
    for update in parsed.get("updates") or []:
        if not isinstance(update, dict) or not update.get("idea_id"):
            continue
        row = db.get(OptimizationIdea, int(update["idea_id"]))
        if row is None or row.task_id != search.task_id:
            continue
        status = str(update.get("status") or "").strip()
        if status in {"proposed", "selected", "validating", "verified", "rejected", "archived"}:
            row.status = status
        if update.get("outcome") is not None:
            row.outcome_json = _json(update["outcome"])
        row.updated_at = datetime.now(timezone.utc)
        updated_ids.append(row.id)
    return {
        "new": len(new_ids), "duplicates": len(duplicate_ids), "updated": len(updated_ids),
        "new_ids": new_ids, "duplicate_of": duplicate_ids, "updated_ids": updated_ids,
    }


# ---------------------------------------------------------------- 只读强制 + 事务收尾


def _finish(job_id: int, state) -> None:
    """dsh 跑完：先做只读强制（快照比对 + 自动回滚），再解析/入库；异常路径保证不留半截数据。"""
    ctx = PENDING.pop(job_id, None)
    if ctx is None:
        return
    search_id: int = ctx["search_id"]
    workspace = Path(ctx["workspace"])
    before: optimize.Capture = ctx["before"]
    ended_at = datetime.now(timezone.utc)

    capture_error: str | None = None
    changed: list[dict] = []
    try:
        after = optimize.capture(workspace)
        changed, _diff = optimize.diff_captures(before, after)
    except Exception as exc:  # noqa: BLE001 —— 快照失败也必须结束搜索，不能留 running
        capture_error = f"诊断结束快照失败：{type(exc).__name__}: {exc}"

    rollback_failed: list[str] = []
    if changed:
        try:
            rollback_failed = optimize.apply_revert(
                workspace, optimize._revert_payload(before, changed))
        except Exception as exc:  # noqa: BLE001 —— 回滚失败要写进结果，不能吞
            rollback_failed = [f"{type(exc).__name__}: {exc}"]

    readonly_result = {
        "new": 0, "duplicates": 0, "updated": 0,
        "readonly_violation": True,
        "changed_files": changed,
        "rollback_failed": rollback_failed,
    }

    with SessionLocal() as db:
        search = db.get(IdeaSearch, search_id)
        if search is None:  # 搜索行已被删除（任务删除等）：工作区改动仍已在上面回滚
            return
        try:
            if capture_error:
                search.status = "failed"
                search.summary = capture_error
                search.result_json = _json({"new": 0, "duplicates": 0, "updated": 0,
                                            "snapshot_error": True,
                                            "error": capture_error[:500]})
            elif changed:
                # 只读强制：诊断期间改动工作区 → 一律失败 + 自动回滚，不解析、不入库。
                search.status = "failed"
                summary = f"诊断 Agent 改动了工作区（{len(changed)} 个文件），已自动回滚"
                if rollback_failed:
                    summary += "；以下文件无法自动恢复：" + "、".join(rollback_failed)
                search.summary = summary
                search.result_json = _json(readonly_result)
            elif state.error or state.exit_code != 0:
                search.status = "failed"
                search.summary = state.error or f"诊断 Agent 退出码 {state.exit_code}"
                search.result_json = _json({"new": 0, "duplicates": 0, "updated": 0})
            else:
                prose, parsed = _parse_output(state.final)  # 纯函数：先解析（不碰 DB）
                stats = _ingest(db, search, parsed)  # 再同一事务入库
                search.status = "ok"
                search.summary = str(parsed.get("summary") or prose).strip() or "搜索完成"
                search.result_json = _json(stats)
            search.ended_at = ended_at
            db.commit()
        except Exception as exc:  # noqa: BLE001 —— 解析/入库失败：rollback 后重标 failed
            db.rollback()
            row = db.get(IdeaSearch, search_id)
            if row is None:
                return
            row.ended_at = ended_at
            row.status = "failed"
            row.summary = (str(exc) if isinstance(exc, IdeaSearchError)
                           else f"诊断结果处理失败：{type(exc).__name__}: {exc}")
            row.result_json = _json({
                "new": 0, "duplicates": 0, "updated": 0,
                "readonly_violation": False,
                "changed_files": changed,
                "error": str(exc)[:500],
            })
            db.commit()


def start(search: IdeaSearch, job: AgentJob, prompt: str, workspace: Path, dsh_home: Path):
    """抓改前快照 → 起诊断 agent job；跑完回调做只读比对/回滚 + 解析入库。"""
    before = optimize.capture(Path(workspace))
    PENDING[job.id] = {"search_id": search.id, "workspace": str(workspace), "before": before}
    try:
        return agent_runner.start(
            job.id, prompt, cwd=workspace, dsh_home=dsh_home,
            on_done=lambda state: _finish(job.id, state), stream=True,
            prose_of=lambda text: text.split(SPEC_SEP, 1)[0].strip(),
        )
    except Exception:
        PENDING.pop(job.id, None)  # 起进程就失败：别留一个永远不会被回调的 PENDING
        raise


# ---------------------------------------------------------------- 一思路一 Loop（串行）


def _active_idea_loop(db: Session, task_id: int) -> tuple[OptimizationIdea, Loop] | None:
    """本任务里挂了「未关闭 Loop」的思路（以 Loop 状态为准，PATCH idea.status 绕不过去）。"""
    row = db.execute(
        select(OptimizationIdea, Loop)
        .join(Loop, OptimizationIdea.loop_id == Loop.id)
        .where(OptimizationIdea.task_id == task_id, Loop.status != "closed")
        .order_by(OptimizationIdea.id)
        .limit(1)
    ).first()
    if row is None:
        return None
    return row[0], row[1]


def create_round(db: Session, task: Task, idea: OptimizationIdea, *, objective: dict | None,
                 attach_baseline_runs: bool) -> dict:
    # 全局锁：FastAPI 同步端点并发时，「检查 + 建 Loop + 写 idea.loop_id」必须原子。
    with _ROUND_LOCK:
        return _create_round_locked(
            db, task, idea, objective=objective, attach_baseline_runs=attach_baseline_runs)


def default_focus_for_idea(idea: OptimizationIdea) -> str:
    """把一条思路转成「需要优化的点」文字描述（文字目标轮的 focus 缺省值）。

    文字目标（target_dimension=text）不指定 benchmark 用例，Agent 按这段文字自己选要优化的
    shape/路径；把假设/范围/预期/风险一起带上，避免只给一句标题导致候选范围漂移。
    """
    lines = [f"优化点：{idea.title}"]
    if idea.hypothesis:
        lines.append(f"假设：{idea.hypothesis}")
    if idea.change_scope:
        lines.append(f"改动范围：{idea.change_scope}")
    if idea.expected:
        lines.append(f"预期：{idea.expected}")
    if idea.risk:
        lines.append(f"风险：{idea.risk}")
    return "\n".join(lines)[:2000]


def _create_round_locked(db: Session, task: Task, idea: OptimizationIdea, *,
                         objective: dict | None, attach_baseline_runs: bool) -> dict:
    if idea.loop_id is not None:
        raise IdeaSearchError(f"思路 #{idea.id} 已关联 Loop #{idea.loop_id}", status=409)
    if idea.status != "proposed":
        raise IdeaSearchError(
            f"思路 #{idea.id} 当前状态为 {idea.status}，只有 proposed 的思路才能新建轮次"
            "（已否决/归档/已验证的思路不能复用）", status=409)
    busy = _active_idea_loop(db, task.id)
    if busy is not None:
        other, open_loop = busy
        raise IdeaSearchError(
            f"思路 #{other.id} 的 Loop #{open_loop.id} 尚未采纳或否决；优化点必须串行验证",
            status=409,
        )
    focus = default_focus_for_idea(idea)
    if objective is not None:
        raw = dict(objective)
        if (str(raw.get("target_dimension") or "") == "text"
                and not str(raw.get("focus") or "").strip()):
            raw["focus"] = focus
    else:
        raw = {
            # 思路池建轮：目标用「文字描述的优化点」，不指定目标用例（perf-diff 按主用例兜底）。
            "type": "perf", "target_dimension": "text", "focus": focus, "target_cases": [],
            "constraints": {"precision_non_regression": True, "perf_non_regression": False,
                            "allowed_paths": [f"ops/{task.op_name}/"] if task.op_name else []},
            "budget": dict(optimization.DEFAULT_BUDGET),
            "user_text": f"来自优化点 #{idea.id}：{idea.title}",
        }
    try:
        result = optimization.create_round(
            db, task, objective=raw, label=idea.title, hypothesis=idea.hypothesis,
            attach_baseline_runs=attach_baseline_runs,
        )
        idea.loop_id = int(result["loop"]["id"])
        idea.status = "validating"
        idea.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        # 锁内先回滚：异常路径别把未提交的写事务带出锁，否则并发建轮会撞 database is locked。
        db.rollback()
        raise
    result["idea"] = idea_view(idea)
    return result
