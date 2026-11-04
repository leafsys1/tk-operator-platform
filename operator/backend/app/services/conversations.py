"""P0-2 对话页：会话索引 + 消息 + 复用 agent_runner（dsh headless）。

**不新写引擎**：一次发送 = 起一个 `agent_jobs` 作业（`kind=对话`），走
`agent_runner.start()`（子进程 `dsh --profile headless --json` + SSE + 逐 token 插件）。

两种模式（由会话的 `mode` 决定）：

- `task`：cwd = 任务工作区 → **挂 skills**（40 个 CANNBot skill）、能读写代码；默认。
- `light`：cwd = 空目录 `.opx-chat/conv-<id>` → 不挂 skills、不碰工作区；用于轻问答。

边界：同一任务同时只允许一个对话作业（两个 agent 同时改一个工作区会互相踩）；
`messages` 只存用户可见正文，dsh 的完整事件仍在它自己的 JSONL（`agent_jobs.dsh_jsonl_path`）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..db import SessionLocal
from ..models import AgentJob, Attachment, Message, Project, Task
from . import agent_runner, optimize, pricing, savepoints, workspace_lease

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 2000

#: 单进程内「检查 running + 落用户消息 + 插入 job」的互斥锁。
#: 数据库没有唯一约束（SQLite 加部分唯一索引代价大），单进程部署下用模块锁兜住 check-then-act 竞态。
_SEND_LOCK = threading.Lock()

#: 会写任务工作区的作业类型：同一任务同时只允许一个（两个 agent 同时改一个工作区会互相踩）。
_WORKSPACE_JOB_KINDS = ("对话", "开发", "优化")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec_fields(db: Session, task: Task) -> dict:
    """当前规格版本的字段（给每轮 prompt 的固定层用；缺失就空）。"""
    if not task.current_spec_version_id:
        return {}
    from ..models import SpecVersion

    row = db.get(SpecVersion, task.current_spec_version_id)
    if not row or not row.spec_json:
        return {}
    try:
        data = json.loads(row.spec_json)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _clip(value, limit: int = MAX_CONTEXT_CHARS) -> str:
    text = str(value) if value not in (None, "", [], {}) else "UNKNOWN"
    return text[:limit]


ATTACH_MAX_BYTES = 20 * 1024 * 1024
#: 每个会话附件数上限：`build_prompt` 每轮都拼全部附件，无限累积会拖垮 prompt。
ATTACH_MAX_COUNT = 20


def attachment_dir(workspace: Path, conv) -> Path:
    d = workspace / ".opx-chat" / f"conv-{conv.id}" / "attachments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_attachment(db: Session, conv, workspace: Path, filename: str, data: bytes) -> Attachment:
    """保存附件到工作区（文件名去路径 + 去危险字符；重名加后缀）。"""
    if len(data) > ATTACH_MAX_BYTES:
        raise ValueError("附件超过 20MB 上限")
    count = int(db.scalar(
        select(func.count(Attachment.id)).where(Attachment.conversation_id == conv.id)
    ) or 0)
    if count >= ATTACH_MAX_COUNT:
        raise ValueError(f"每个会话最多 {ATTACH_MAX_COUNT} 个附件，请先删除不用的附件")
    name = Path(filename or "upload.bin").name
    name = "".join(c if (c.isalnum() or c in "._- ") else "_" for c in name).strip() or "upload.bin"
    d = attachment_dir(workspace, conv)
    target = d / name
    n = 2
    while target.exists():
        target = d / f"{Path(name).stem}-{n}{Path(name).suffix}"
        n += 1
    target.write_bytes(data)
    row = Attachment(conversation_id=conv.id, filename=target.name,
                     path=target.relative_to(workspace).as_posix(),
                     size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), created_at=_now())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_attachment(db: Session, conv, workspace: Path | None, attachment_id: int) -> bool:
    """删除会话的一个附件（DB 行 + 磁盘文件）。文件已不存在也算成功。

    路径经 `workspace.resolve()` 校验，避免恶意/损坏的 `path` 删库外文件。
    返回 False 表示附件不存在或不属于该会话。
    """
    row = db.get(Attachment, attachment_id)
    if row is None or row.conversation_id != conv.id:
        return False
    if workspace is not None:
        root = workspace.resolve()
        p = (workspace / row.path).resolve()
        if p != root and root in p.parents and p.is_file():
            try:
                p.unlink()
            except OSError:  # 文件删不掉不阻断 DB 行清理（DB 是真相源）
                pass
    db.execute(sql_delete(Attachment).where(Attachment.id == row.id))
    db.commit()
    return True


def _job_still_alive(job: AgentJob) -> bool:
    """DB 标 running 的作业是否真的还在跑。

    对齐 task_deletion 的僵尸兜底：本进程 REGISTRY 里没有它、且 started_at 已超过
    `agent_timeout`，说明后台任务随进程崩溃/被杀，不能再永久挡住会话删除。
    """
    if agent_runner.running(job.id) is not None:
        return True
    started = job.started_at
    if started is None:
        return True  # 无法判断就保守拒绝
    if started.tzinfo is None:  # SQLite 读回来是 naive；写库一律 UTC
        started = started.replace(tzinfo=timezone.utc)
    timeout = max(1, int(settings.agent_timeout or 0))
    return (datetime.now(timezone.utc) - started).total_seconds() <= timeout


def delete(db: Session, conv) -> None:
    """删除一个会话：running 作业存在则拒绝（不产生幽灵 agent）；显式删 attachments/messages/会话行。

    不依赖 SQLite 外键级联（`app/db.py` 没有 `PRAGMA foreign_keys=ON`），否则会留下孤儿行，
    且会话 id 复用时新会话会继承旧附件。磁盘目录 `<workspace>/.opx-chat/conv-<id>` 一并删除；
    工作区不存在时只删库、跳过文件。僵尸 running 行（进程已崩 + 超时）视为可删并标 interrupted。
    """
    from ..models import Conversation

    conv_id = conv.id
    running_jobs = list(db.scalars(
        select(AgentJob).where(
            AgentJob.conversation_id == conv_id, AgentJob.status == "running"
        ).order_by(AgentJob.id)
    ))
    live = [job for job in running_jobs if _job_still_alive(job)]
    if live:
        raise ValueError(f"会话还有运行中的作业（job #{live[0].id}），先取消再删")
    for job in running_jobs:  # 僵尸：落 interrupted，别让台账永远 running
        job.status = "interrupted"
        job.ended_at = job.ended_at or datetime.now(timezone.utc)
    if running_jobs:
        db.commit()

    project = db.get(Project, conv.project_id) if conv.project_id else None
    workspace: Path | None = None
    if project and project.workspace_path:
        ws = Path(project.workspace_path).expanduser()
        if ws.is_dir():
            workspace = ws

    atts = list(db.scalars(select(Attachment).where(Attachment.conversation_id == conv_id)))
    att_paths = [a.path for a in atts]
    db.execute(sql_delete(Attachment).where(Attachment.conversation_id == conv_id))
    db.execute(sql_delete(Message).where(Message.conversation_id == conv_id))
    db.execute(sql_delete(Conversation).where(Conversation.id == conv_id))
    db.commit()

    if workspace is None:
        return
    root = workspace.resolve()
    # 附件文件（按 DB 路径删，带 workspace 越界校验）
    for rel in att_paths:
        p = (workspace / rel).resolve()
        if p != root and root in p.parents and p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    # 会话目录（含轻对话 scratch 与 attachments 残留）
    d = (workspace / ".opx-chat" / f"conv-{conv_id}").resolve()
    if d != root and root in d.parents:
        shutil.rmtree(d, ignore_errors=True)


def _changes_payload(before: optimize.Capture, after: optimize.Capture) -> dict | None:
    """本轮改动证据：逐文件 diff + 回退数据（改前内容，≤512KB）。"""
    changed, diff = optimize.diff_captures(before, after)
    if not changed:
        return None
    revert: dict[str, dict] = {}
    for item in changed:
        rel = item["path"]
        b = before.files.get(rel)
        a = after.files.get(rel)
        existed = b is not None
        content = b.get("content") if b else None
        revert[rel] = {
            "existed": existed,
            "restorable": (not existed) or content is not None,
            "content_b64": base64.b64encode(content).decode("ascii") if content is not None else None,
            # 改后哈希：回退前校验「当前文件仍是这一轮结束时的样子」，防止覆盖后续轮改动。
            # added 文件有值；removed 文件为 None（回退时要求文件仍不存在）。
            "after_sha256": (a.get("sha256") if a else None),
        }
    return {
        "before_sha256": before.sha256,
        "after_sha256": after.sha256,
        "changed_files": changed,
        "diff": diff,
        "revert": revert,
    }


def _file_sha256(path: Path) -> str | None:
    """文件不存在/读不了 → None（用于「改后哈希」比对）。"""
    if not path.is_file():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def revert_message(db: Session, conv, msg, workspace: Path) -> dict:
    """把某一轮对话的改动还原到改前（按 `changes_json` 里的回退数据）。

    **先全量校验再动手**：每个文件当前 sha256 必须等于该轮「改后哈希」`after_sha256`
    （旧记录没有该字段则跳过校验）。任一不一致 → `ValueError`（router 409），且一个文件都不改，
    避免回退早期轮次时静默覆盖后续轮的改动。
    """
    if not msg.changes_json:
        raise ValueError("这条消息没有可回退的改动证据")
    try:
        data = json.loads(msg.changes_json)
    except (ValueError, TypeError) as e:
        raise ValueError("改动证据不是合法 JSON") from e
    if not isinstance(data, dict):
        raise ValueError("改动证据格式不正确")
    revert = data.get("revert") or {}
    if not isinstance(revert, dict):
        raise ValueError("改动证据格式不正确")

    root = workspace.resolve()
    results: dict[str, str] = {}
    plans: dict[str, tuple[Path, dict]] = {}
    conflicts: list[str] = []
    for rel, info in revert.items():
        if not isinstance(info, dict):
            results[rel] = "skipped: 回退数据格式不正确"
            continue
        p = (root / rel).resolve()
        if root not in p.parents and p != root:
            results[rel] = "skipped: 路径越界"
            continue
        plans[rel] = (p, info)
        # 旧记录没有 after_sha256 → 跳过冲突校验（兼容历史数据）
        if "after_sha256" not in info:
            continue
        if info.get("existed") and (not info.get("restorable") or info.get("content_b64") is None):
            continue  # 原文件过大/未存内容，本来就不会被还原，不参与冲突判定
        if _file_sha256(p) != info.get("after_sha256"):
            conflicts.append(rel)
    if conflicts:
        raise ValueError("文件已被后续改动，拒绝回退：" + "、".join(conflicts))

    # 全部校验通过后才开始写
    for rel, (p, info) in plans.items():
        if info.get("existed"):
            if not info.get("restorable") or info.get("content_b64") is None:
                results[rel] = "skipped: 原文件过大，未存回退内容"
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(info["content_b64"]))
            results[rel] = "restored"
        else:
            if p.exists():
                p.unlink()
            results[rel] = "removed"
    return {"reverted": True, "files": results}


# ---------------------------------------------------------------- 任务生命周期 v2 固定层
# 对话 prompt 每轮注入的「当前版本 / 用例确认状态 / 交付物 / 规格修改协议」（契约 §7）。
# 这一切只读；任何判定异常都降级成「按未确认处理」，不允许因为 T1 的 case_confirm
# 尚未落地、工作区缺目录而让对话本身发不出去。


def _case_confirm_api():
    """延迟 import T1 的 `services.case_confirm`（并行开发中）；不可用/半成品返回 None。"""
    try:
        from . import case_confirm  # noqa: PLC0415 —— 刻意延迟：该模块由 T1 并行落地
        return case_confirm
    except Exception:  # noqa: BLE001 —— ImportError 及模块内半成品异常都不能拖垮对话
        return None


def _hash_cases_tree(root: Path) -> str | None:
    """契约 §5.1 / `scripts/nm_test.sh` 的 **整个 cases/ 目录树**哈希（仅 case_confirm 不可用时的降级）。

    算法：os.walk（每层 dirnames.sort()、文件名 sorted()）→ 每文件
    `update(相对路径 bytes)` + `b"\\0"` + 内容（1MB 分块）+ `b"\\0"`。
    """
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                p = Path(dirpath) / name
                h.update(p.relative_to(root).as_posix().encode())
                h.update(b"\0")
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                h.update(b"\0")
    except OSError:
        return None
    return h.hexdigest()


def _current_suite_sha256(workspace: Path, op: str | None = None) -> str | None:
    """按任务作用域计算 full SHA；未传 op 时兼容历史整树确认。"""
    mod = _case_confirm_api()
    fn = getattr(mod, "cases_suite_sha256", None) if mod is not None else None
    if callable(fn):
        try:
            value = fn(workspace, op) if op else fn(workspace)
            if value:
                return str(value).strip().lower()
        except Exception:  # noqa: BLE001 —— 降级到本地实现
            pass
    return _hash_cases_tree(workspace / "cases" / op if op else workspace / "cases")


def _current_suite_ref(workspace: Path) -> str | None:
    """人读的用例集 ref：`cases@<hash16>`（契约 §5.1）。"""
    mod = _case_confirm_api()
    fn = getattr(mod, "cases_ref", None) if mod is not None else None
    if callable(fn):
        try:
            value = fn(workspace)
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            pass
    sha = _current_suite_sha256(workspace)
    return f"cases@{sha[:16]}" if sha else None


def _sha_same(confirmed: str | None, current: str | None) -> bool:
    """确认哈希与当前哈希是否一致：容错 16 位前缀口径（一方是另一方前缀即视为一致）。"""
    x = (confirmed or "").strip().lower()
    y = (current or "").strip().lower()
    if x.startswith("cases@"):
        x = x[len("cases@"):]
    if y.startswith("cases@"):
        y = y[len("cases@"):]
    if not x or not y:
        return False
    return x == y or (len(x) >= 8 and y.startswith(x)) or (len(y) >= 8 and x.startswith(y))


def _fmt_time(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        stamp = value.strftime("%Y-%m-%d %H:%M:%S")
        return f"{stamp} UTC" if value.tzinfo is not None else stamp
    return str(value)


def _cases_confirmation(db: Session, task: Task, workspace: Path, op: str) -> dict:
    """用例组确认状态（契约 §3.3 / §7）。

    `state`：
    - `unconfirmed`：`cases_confirmed_at` 为空（**硬门禁：第一优先级先确认用例组**）；
    - `confirmed`：已确认且 `cases_confirmed_sha256` 与当前 `cases/` 目录树哈希一致；
    - `drift`：有确认记录但哈希已不一致（改过用例/规格落版后清空前的过渡），需重新确认；
    - `unknown`：确认记录在、但当前哈希算不出来（cases/ 缺失/不可读），保守按未确认对待。
    """
    confirmed_at = getattr(task, "cases_confirmed_at", None)
    confirmed_by = getattr(task, "cases_confirmed_by", None)
    confirmed_sha = (getattr(task, "cases_confirmed_sha256", None) or "").strip().lower() or None
    current_sha: str | None = None
    in_sync: bool | None = None

    mod = _case_confirm_api()
    fn = getattr(mod, "confirmation", None) if mod is not None else None
    if callable(fn):
        try:
            info = fn(db, task) or {}
            confirmed_at = info.get("confirmed_at", confirmed_at)
            raw_confirmed = info.get("confirmed_sha256")
            if raw_confirmed:
                confirmed_sha = str(raw_confirmed).strip().lower()
            raw_current = info.get("current_suite_sha256")
            if raw_current:
                current_sha = str(raw_current).strip().lower()
            in_sync = bool(info.get("in_sync"))
        except Exception:  # noqa: BLE001 —— T1 模块异常/半成品：降级自己算
            in_sync = None

    if current_sha is None:
        scope_op = op if getattr(task, "cases_confirmed_scope", None) == "operator-v1" else None
        current_sha = _current_suite_sha256(workspace, scope_op)
    if in_sync is None:
        in_sync = bool(confirmed_at and _sha_same(confirmed_sha, current_sha))

    if not confirmed_at:
        state = "unconfirmed"
    elif in_sync:
        state = "confirmed"
    elif current_sha is None:
        state = "unknown"
    else:
        state = "drift"
    return {
        "state": state,
        "op": op,
        "confirmed_at": confirmed_at,
        "confirmed_by": confirmed_by,
        "confirmed_sha": confirmed_sha,
        "current_sha": current_sha,
        "current_ref": f"cases@{current_sha[:16]}" if current_sha else None,
        "in_sync": bool(in_sync),
    }


def _spec_version_number(db: Session, task: Task) -> int | None:
    """当前规格的大版本号（= `spec_versions.version`）；未定版返回 None。"""
    if not task.current_spec_version_id:
        return None
    from ..models import SpecVersion

    row = db.get(SpecVersion, task.current_spec_version_id)
    if not row or row.version is None:
        return None
    try:
        return int(row.version)
    except (TypeError, ValueError):
        return None


def _version_label(db: Session, task: Task) -> tuple[str, int | None]:
    """`(展示字符串, 大版本号)`：有规格 → `v{major}.{minor}`；没有 → 「未定版」。"""
    major = _spec_version_number(db, task)
    minor = int(task.version_minor or 0)
    if major is None:
        return "未定版", None
    return f"v{major}.{minor}", major


def _task_doc_excerpts(workspace: Path, task: Task) -> list[str]:
    """任务上传的设计文档解析稿（相对工作区路径）：`task-docs/task-<id>/*.extracted.md`。"""
    root = workspace / "task-docs" / f"task-{task.id}"
    if not root.is_dir():
        return []
    try:
        return sorted(
            p.relative_to(workspace).as_posix()
            for p in root.iterdir()
            if p.is_file() and p.name.endswith(".extracted.md")
        )
    except OSError:
        return []


def _confirm_command(api: str, task_id: int, op: str, by: str) -> str:
    """用例组确认的 curl 命令（契约 §3.3：两个文件各自 sha256 + by）。"""
    body = (
        '{"cases_sha256":"<cases/<op>/cases.json 的 64 位 sha256>",'
        '"benchmark_sha256":"<cases/<op>/benchmark.json 的 64 位 sha256>",'
        f'"by":"{by}"' + "}"
    ).replace("<op>", op)
    return (
        f"curl -sS -X POST {api}/api/tasks/{task_id}/cases/confirm "
        "-H 'Content-Type: application/json' "
        f"-d '{body}'"
    )


def _cases_prompt(db: Session, task: Task, workspace: Path, api: str, by: str) -> str:
    """用例确认状态段（未确认 = 第一优先级；已确认 = 切到开发交付物清单）。"""
    op = task.op_name or task.title
    state = _cases_confirmation(db, task, workspace, op)
    cases_ok = (workspace / "cases" / op / "cases.json").is_file()
    bench_ok = (workspace / "cases" / op / "benchmark.json").is_file()
    confirm_cmd = _confirm_command(api, task.id, op, by)

    if state["state"] == "unconfirmed":
        docs = _task_doc_excerpts(workspace, task)
        docs_line = "、".join(f"`{p}`" for p in docs) if docs else "未发现（用户没上传设计文档）"
        return "\n".join([
            "【第一优先级：用例组确认（硬门禁；未确认前不要改代码、不要跑四动作）】",
            f"- 状态：用例组**未确认**（`tasks.cases_confirmed_at` 为空）。未确认时控制台动作 plan 会以"
            "「用例组未确认」拦下 build/smoke/test/benchmark/profile。",
            "- 用例来源（优先级）：**用户在对话里的最新要求 > 上传的设计文档附录 > 你按规格自动生成**。"
            f"设计文档解析稿：{docs_line}。",
            "- 若用户上传过文档：先读附录 A（测试用例要点）。文档没有用例 → 先在对话里问用户"
            "「你是否有精度/性能测试用例」；用户没有/不提供 → 按规格 + `criteria/ops-precision-standard@1` 自动生成。",
            "- 生成后**以整组清单**在对话里呈现，等用户整组确认（可多轮修改；不要逐条问、不要直接落文件）：",
            "  · 精度组：case_id / 分组（常规·边界·数值边界·异常·回归）/ shape（或 n_a、n_b）/ dtype /"
            " 输入生成方式与 seed / 期望（match，或 reject + 拒绝码）/ 说明；",
            "  · 性能组：case_id / n / dtype / 主用例（primary）/ 必需 / 说明。",
            "- 用户明确回复「确认」后按顺序执行：",
            f"  1. 写 `cases/{op}/cases.json`（精度组）与 `cases/{op}/benchmark.json`（性能组）。字段沿用工作区 "
            "`tools/run_cases.py` / `tools/run_benchmark.py` 的读取口径（参考同工作区已有 `cases/<其他算子>/`"
            " 样例；无样例时用完整最小字段集：精度 schema `opx-cases/1`：schema/op/spec_version/dtype/criteria_ref/"
            "constraints/case_groups/cases[{case_id,group,required,n_a,n_b,dtype,gen_a,gen_b,seed,expected,"
            "expect_reject_code,note}]；性能 schema `opx-benchmark/1`：schema/op/dtype/baseline/memory/compute/"
            "measure/primary_case/cases[{case_id,required,n,dtype,gen_a,gen_b,seed,note}]）。",
            "  2. 用 python3 算两个文件的 sha256："
            f"`python3 -c \"import hashlib;print(hashlib.sha256(open('cases/{op}/cases.json','rb').read()).hexdigest())\"`"
            f"（`cases/{op}/benchmark.json` 同理）。",
            "  3. 调控制台接口确认（HTTP 409 = 文件不存在或哈希与入参不一致，修正后重试）：",
            f"     `{confirm_cmd}`",
            "  4. 把返回的 `confirmed_at` / `cases_suite_sha256` 简要告诉用户，然后才开始开发（交付物清单见下段）。",
        ])

    confirm_cmd_line = f"     `{confirm_cmd}`"
    if state["state"] == "drift":
        return "\n".join([
            "【用例组需要重新确认（已确认记录与当前 cases/ 目录哈希不一致）】",
            f"- 确认记录：{_fmt_time(state['confirmed_at'])}（by={state['confirmed_by'] or '—'}），"
            f"确认哈希={ (state['confirmed_sha'] or '—')[:16] }；"
            f"当前哈希={ (state['current_sha'] or '—')[:16] }（ref={state['current_ref'] or '—'}）。",
            "- 原因通常是 cases.json / benchmark.json 被改过，或规格变更后确认状态被清空。"
            "旧 test/benchmark 证据因 suite_ref 变化已失效。",
            "- 处理：重新把当前精度/性能用例组以整组清单呈现给用户 → 用户确认 → 更新文件 → **重新确认**：",
            confirm_cmd_line,
            "- **重新确认前不得改代码、不得触发四动作**；确认成功后再进入开发/重跑。",
        ])
    if state["state"] == "unknown":
        return "\n".join([
            f"【用例组状态无法核验（case_confirm 不可用或 `cases/` 读取失败）】候选文件："
            f"cases/{op}/cases.json 存在={cases_ok}，benchmark.json 存在={bench_ok}。",
            "- 保守处理：**按未确认对待** —— 不要改代码、不要跑四动作；先让用户确认用例组，或排查"
            " `cases/` 目录后重新确认：",
            confirm_cmd_line,
        ])
    # confirmed
    docs = _task_doc_excerpts(workspace, task)
    docs_line = ("规划文档补全依据：" + "、".join(f"`{p}`" for p in docs) + "（原文在 task-docs/ 同目录）"
                 if docs else "无用户上传文档 → 规划文档按模板五段从头生成")
    standards = f"{str(settings.repo_root).rstrip('/')}/docs/reference/deliverable-standards.md"
    template = f"{str(settings.repo_root).rstrip('/')}/docs/templates/算子设计文档模板.md"
    return "\n".join([
        f"【开发阶段：用例组已确认（{_fmt_time(state['confirmed_at'])}，by={state['confirmed_by'] or 'api'}），"
        f"当前 cases/ 哈希与确认记录一致（ref={state['current_ref'] or '—'}）】",
        f"交付物清单（缺一不可，执行细则见控制台仓库 `{standards}`）：",
        f"1. 算子代码：`ops/{op}/`（kernel / tiling / host / CMakeLists，沿用现有 AscendC 直调工程）；",
        f"2. Python 调用接口：`ops/{op}/python/{op}.py` + `lib{op}_host.so` —— ctypes 加载共享库，"
        f"接受 torch_npu tensor、校验 dtype/shape/连续性，返回新 tensor；host 侧在 `ops/{op}/op_host/*.asc`"
        f" 增加 `extern \"C\"` 导出 `int {op}_run(...)`，CMake 在保留可执行文件的同时产出 `lib{op}_host.so`"
        "（**不改变四动作调用可执行文件的路径**）；",
        f"3. pytest 测试脚本：`tests/test_{op}.py`（读 `cases/{op}/cases.json`，用 Python 接口跑候选、"
        f"`ops/{op}/scripts/golden.py` 跑参考，按 `criteria/check.py` 同口径判 PASS/FAIL/拒绝码；共享 runner 仍是门禁权威）；",
        f"4. 算子说明文档：`ops/{op}/README.md`（接口/调用示例/约束/精度与性能口径）；",
        f"5. 规划文档：`docs/{op}/design.md`（按控制台模板 `{template}` 五段；{docs_line}）。",
        "- **开发顺序（硬要求，2026-10-23 用户定）**：代码写完后先 `scripts/nm.sh --op <op> --action build`，"
        "再 `scripts/nm.sh --op <op> --action smoke`（编译 + 一条最小 required match 用例跑通）。"
        "**只有 smoke 返回 `SMOKE_VERDICT PASS` 才允许进入正式 test/benchmark，也才允许说「开发完成」**；"
        "smoke FAIL → 先修代码再重跑，不许跳过；设备被占/跑不了 → 如实说阻塞，不许声称完成。"
        "smoke 会登记一条 action=smoke 的 run，阶段「开发」要求 build + smoke 门禁都 pass。",
        "- **四动作仍走可执行文件**：`scripts/nm.sh --op <op> --action build|smoke|test|benchmark|profile`，"
        "benchmark 测量口径不变；pytest / README 是交付件，不替代四动作门禁。",
        "- 开发前先说明计划；每个节点开始/完成/失败在对话里简短汇报（控制台用户偏好，不要静默）。",
        "- 交付前若改了算子代码（非优化采纳）→ 按规格修改/版本协议由用户确认小版本 +1 并重跑 build+test+benchmark"
        "（profile 可选，由用户对话确认）。",
    ])


def _spec_change_prompt(api: str, task_id: int, by: str) -> str:
    """规格修改协议（契约 §5 / §7：对话判断实质变更 → 起草完整 spec_json → 用户确认 → 落版）。"""
    payload_example = (
        '{"spec_json":"<完整 spec 的 json.dumps 字符串>",'
        f'"change_reason":"对话确认（{by}）：<改了什么、为什么>",'
        '"set_current":true,"assumptions":"...","unknowns":"..."}'
    )
    return "\n".join([
        "【规格修改协议（用户要求改需求/规格时）】",
        "1. 先问清要改什么，判断是**实质变更**（影响语义 / 输入输出 / 支持范围 / 判据 / 交付）还是"
        "**文档性修订**（措辞、说明、错别字）。",
        "2. 实质变更：起草**完整** spec_json（所有字段重写；缺失的标 \"UNKNOWN\"，不要只给 diff）→ "
        "在对话里给摘要与差异 → 等用户明确确认（未确认不要落版）。",
        "3. 用户确认后用 python3 写 payload 文件（注意 `spec_json` 必须是**字符串**，不是对象），再落版：",
        f"   payload 形如：`{payload_example}`",
        f"   `curl -sS -X POST {api}/api/tasks/{task_id}/specs -H 'Content-Type: application/json' --data-binary @<payload 文件>`",
        "4. 落版成功后提醒用户：**大版本 +1、小版本归 0、用例组需重新确认**（确认状态已清空，按上文"
        "用例确认流程重新整组确认）、交付前需重跑 build/test/benchmark（profile 可选，由用户对话确认）。",
        "5. 文档性修订：默认**不落版本**（避免版本膨胀），只在对话里留结论；落不落版由用户确认。",
    ])


def build_prompt(db: Session, task: Task, conv, workspace: Path, text: str,
                 *, job_id: int | None = None) -> str:
    """每轮重拼固定层（不依赖「模型记得」）。

    v2 增补（契约 §7）：当前版本 `v{major}.{minor}`、用例确认状态（未确认 → 第一优先级整组确认
    用例；已确认 → 开发交付物清单）、规格修改协议、控制台 API base（`settings.console_api`，
    不写死地址）。`job_id` 用于提示词里的确认留痕 `by=chat:job#<id>`（可缺省，缺省用 `chat`）。
    """
    fields = _spec_fields(db, task)
    atts = list(db.scalars(select(Attachment).where(Attachment.conversation_id == conv.id).order_by(Attachment.id)))
    if conv.mode == "light":
        head = ("你是「op-agent 算子开发控制台」的助手。**轻对话模式**：不要读写工作区、"
                "不要调用工具；直接、简洁地回答用户。\n")
        if atts:
            head += "用户上传的附件（相对当前目录）：\n" + "\n".join(
                f"- attachments/{a.filename}（{a.size_bytes} bytes）" for a in atts) + "\n"
        return head + f"\n用户：{text}"

    op = task.op_name or task.title
    api = str(settings.console_api).rstrip("/")  # 只从配置读，prompt 里不写死第二个地址
    by = f"chat:job#{job_id}" if job_id else "chat"
    version_label, major = _version_label(db, task)
    parts = [
        f"你是「op-agent 算子开发控制台」的助手，正在和用户讨论任务「{task.title}」"
        f"（算子 `{op}`，目标芯片 {task.target_chip or '910B1'}）。",
        f"当前工作区（cwd）：{workspace}。领域能力在 `.dsh/skills/`（需要时先用 skill 工具查阅）；"
        f"改代码前先说明意图，改完告诉用户改了哪些文件。",
        # 控制台默认策略（2026-10-21）：用户不要「静默开发」，要能看到节点级进度。
        "控制台用户偏好：**不要静默** —— workflow 启动、每个节点开始/完成/失败、等待决策时都在对话里"
        "简短汇报（一两句，不刷屏）；长节点轮询期间也报一行状态；不要只说「静默开发」。",
        "真机验证入口：工作区 `scripts/nm.sh`（`--op <op> --action build|test|benchmark|profile`，先 `--dry-run`）；"
        "它按**当前任务**配置的执行机/镜像执行并回填 run 台账，不要与控制台「概览 → 四动作」同时触发同一动作。",
        f"规格要点：数学定义={_clip(fields.get('math_def'))}；"
        f"输入={_clip(fields.get('inputs'))}；输出={_clip(fields.get('outputs'))}；"
        f"oracle={_clip(fields.get('oracle'))}；容差={_clip(fields.get('tolerance'))}。",
        # ---- v2 固定层：版本 + API base ----
        f"任务当前版本：{version_label}（大版本 = 当前规格版本号；小版本 = 任务内变更计数）"
        + ("" if major is not None else "。当前任务**还没有定版规格**（未定版）：先澄清需求并对用例组完成确认，"
                                        "再由用户在控制台定版；不要自己调 finalize。"),
        f"控制台 API base：{api}。本轮对话作业：`{by}`。对话里需要调后端接口（用例确认 / 规格落版）时"
        "就用上面这个 base 和这个 by，**不要写死其它地址**。",
        # ---- v2 固定层：用例确认 / 开发交付物 ----
        _cases_prompt(db, task, workspace, api, by),
        # ---- v2 固定层：规格修改协议 ----
        _spec_change_prompt(api, task.id, by),
    ]
    if atts:
        parts.append("用户上传的附件（相对工作区）：\n" + "\n".join(
            f"- {a.path}（{a.size_bytes} bytes）" for a in atts))
    parts += ["", f"用户：{text}"]
    return "\n".join(parts)


def _dsh_home(project: Project, workspace: Path) -> Path:
    home = Path(project.dsh_home or (workspace.parent / ".dsh")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def workspace_of(db: Session, conv) -> Path:
    project = db.get(Project, conv.project_id) if conv.project_id else None
    if not project or not project.workspace_path:
        raise ValueError("项目工作区未配置")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise ValueError(f"工作区不存在：{workspace}")
    return workspace


def conversation_cwd(db: Session, conv) -> tuple[Path, Path]:
    """会话的 (cwd, dsh_home)。task 模式 = 工作区（挂 skills）；light = 空目录。"""
    project = db.get(Project, conv.project_id) if conv.project_id else None
    if not project or not project.workspace_path:
        raise ValueError("项目工作区未配置")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise ValueError(f"工作区不存在：{workspace}")
    if conv.mode == "light":
        scratch = workspace / ".opx-chat" / f"conv-{conv.id}"
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch, _dsh_home(project, workspace)
    return workspace, _dsh_home(project, workspace)


def create(db: Session, task_id: int, *, title: str | None = None, mode: str = "task"):
    from ..models import Conversation

    task = db.get(Task, task_id)
    if not task:
        raise ValueError("任务不存在")
    if mode not in ("task", "light"):
        raise ValueError("mode 只能是 task / light")
    conv = Conversation(
        project_id=task.project_id, task_id=task.id,
        title=(title or f"{task.title} · 对话")[:300], mode=mode,
        message_count=0, created_at=_now(), last_active_at=_now(),
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def load(db: Session, conv_id: int, *, with_messages: bool = False):
    from ..models import Conversation

    q = select(Conversation).where(Conversation.id == conv_id)
    if with_messages:
        q = q.options(selectinload(Conversation.messages))
    return db.scalar(q)


def list_for(db: Session, *, task_id: int | None = None, project_id: int | None = None, limit: int = 50):
    from ..models import Conversation

    q = select(Conversation).order_by(Conversation.last_active_at.desc(), Conversation.id.desc()).limit(limit)
    if task_id is not None:
        q = q.where(Conversation.task_id == task_id)
    if project_id is not None:
        q = q.where(Conversation.project_id == project_id)
    return list(db.scalars(q))


def send(db: Session, conv, text: str) -> AgentJob:
    """发一条消息：落用户消息 → 起 dsh 作业（立刻返回，过程走 SSE）。"""
    if not text.strip():
        raise ValueError("消息不能为空")

    # 检查 running + 落用户消息 + 插入 job 在同一把进程内锁里完成：单进程下不会两个请求同时通过。
    # 任务行也在锁内读：避免两个线程各自先开读事务，导致后进者看不到先提交的 running 行。
    with _SEND_LOCK:
        task = db.get(Task, conv.task_id) if conv.task_id else None
        if not task:
            raise ValueError("会话没有绑定任务（P0-2 首版要求绑定任务）")
        # Step 2：自动验证链运行期间工作区必须稳定（对话会改代码），等链结束再发。
        try:
            from . import loop_orchestrator
            auto = loop_orchestrator.running_for_task(db, task.id)
        except Exception:  # noqa: BLE001
            auto = None
        if auto is not None:
            raise ValueError(
                f"自动验证链 #{auto.id} 正在跑（当前步骤 {auto.current_step or '未知'}），"
                "等它结束再继续对话（否则会改到正在验证的代码）")
        running = db.scalar(
            select(AgentJob).where(AgentJob.task_id == task.id,
                                   AgentJob.status == "running",
                                   AgentJob.kind.in_(_WORKSPACE_JOB_KINDS)).limit(1)
        )
        if running is not None:
            if running.kind == "对话":
                raise ValueError(f"任务 #{task.id} 已有一个对话作业在跑（job #{running.id}），等它结束再发")
            raise ValueError(
                f"任务 #{task.id} 已有 {running.kind} 作业在跑（job #{running.id}），等它结束再发"
                "（可在任务页取消）")

        # Step 1：任务工作区租约（与动作作业共用）；light 会话不写工作区，不占租约。
        lease = None
        if conv.mode == "task":
            try:
                lease = workspace_lease.acquire(db, task, kind="agent")
            except workspace_lease.LeaseBusy as e:
                raise ValueError(str(e)) from e

        cwd, dsh_home = conversation_cwd(db, conv)
        before = optimize.capture(cwd) if conv.mode == "task" else None
        # 自动归档：对话可能改代码，改前留一份可回滚的源码（best-effort，失败只 log）。
        if conv.mode == "task":
            try:
                savepoints.auto_capture(db, task, cwd, label="对话改前自动存档")
            except Exception:  # noqa: BLE001 —— 自动归档失败不能阻断对话
                logger.exception("对话改前自动归档失败（task=%s, conv=%s）", task.id, conv.id)
        # 先插入 job 拿 id，再拼 prompt：prompt 里要告诉 agent 用例确认留痕用 `by=chat:job#<id>`
        # （契约 §7 / §3.3）。job 行仍在同一事务里，拼 prompt 失败时随会话回滚。
        job = AgentJob(task_id=task.id, conversation_id=conv.id, kind="对话", status="running",
                       prompt="", started_at=_now())
        pricing.attach_endpoint(db, job)
        db.add(job)
        db.flush()  # 只 flush 不 commit：拿 id 用
        if lease is not None:
            lease.job_id = job.id
        prompt = build_prompt(db, task, conv, cwd, text, job_id=job.id)
        job.prompt = prompt[:6000]
        db.add(Message(conversation_id=conv.id, role="user", text=text, created_at=_now()))
        conv.last_active_at = _now()
        db.commit()
        db.refresh(job)

    def on_done(state) -> None:
        _finish(job.id, conv.id, state, before=before, cwd=cwd)

    # 自由聊天没有「prose + JSON 尾」两段式契约 → plain_stream 原样放行，
    # 否则以代码围栏/反引号/{/[ 开头的回答会被 ProseRelay 整段 suppress。
    agent_runner.start(job.id, prompt, cwd=cwd, dsh_home=dsh_home, session_id=conv.dsh_session_id,
                       on_done=on_done, stream=True, plain_stream=True,
                       prose_of=lambda s: s.final)
    return job


def _finish(job_id: int, conv_id: int, state, *, before=None, cwd: Path | None = None) -> None:
    """作业跑完：落助手消息 + 回填会话（session id / tokens / 计数）。

    失败/取消也可见：有正文 → 正文 + `（本轮中断：…）`；无正文 → 只落中断说明。
    """
    with SessionLocal() as db:
        from ..models import Conversation

        conv = db.get(Conversation, conv_id)
        if conv is None:
            return
        # 优先落**最终答复**：`prose` 是逐 token 流拼接的「中间叙述」（agent 多步执行时会把
        # 所有 assistant 消息串在一起，看着像「回复全拼在一起」），只有拿不到 final 时才兜底。
        text = (state.final or state.prose or "").strip()
        error = getattr(state, "error", None)
        cancelled = bool(getattr(state, "cancelled", False))
        if error or cancelled:
            note = "（本轮中断：" + (error or "已取消") + "）"
            text = (text + "\n\n" + note) if text else note
        elif not text:
            text = "（本轮没有产生可显示的回答；错误：未知）"
        usage = state.usage or {}
        conv.dsh_session_id = state.session_id or conv.dsh_session_id
        conv.last_active_at = _now()
        conv.message_count = (conv.message_count or 0) + 2
        conv.tokens = (conv.tokens or 0) + sum(
            int(usage.get(key) or 0)
            for key in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")
        )
        changes = None
        if before is not None and cwd is not None:
            try:
                changes = _changes_payload(before, optimize.capture(cwd))
            except Exception:  # noqa: BLE001 —— 审阅证据失败不影响消息落库
                changes = None
        db.add(Message(conversation_id=conv.id, role="assistant", text=text, job_id=job_id,
                       changes_json=(json.dumps(changes, ensure_ascii=False) if changes else None),
                       created_at=_now()))
        # Step 1：对话作业结束释放工作区租约（只删本 job 的租约，别误删别人的）。
        if conv.task_id:
            workspace_lease.release(db, task_id=conv.task_id, kind="agent", job_id=job_id,
                                    commit=False)
        db.commit()
