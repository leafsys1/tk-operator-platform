"""把 dsh headless 跑成一次**智能体调用**（agent_jobs）：子进程 + JSONL 事件流 + SSE 订阅。

契约面只有一条（方案 v1.1 §6）：`dsh --profile headless --json "<任务>"` 的 **JSONL 事件流**。
事件明细不落库（真相源是 dsh 的 sessions JSONL），这里只存摘要：session id、usage、退出码。
**注意**：agent job 跑成功 ≠ 算子做对了 —— 业务通过状态由真实执行（runs）+ 报告校验决定（v1.1 §3.1）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from . import llm_config
from .prose_relay import ProseRelay, split_prose


class JobState:
    """一次正在跑的智能体调用：事件缓冲 + 订阅者 + 结果摘要。"""

    def __init__(self, job_id: int, prompt: str, *, plain_stream: bool = False) -> None:
        self.job_id = job_id  # = agent_jobs.id
        self.prompt = prompt
        # 正式 agent run 的答复没有「prose + JSON 尾」两段式契约 → 原样放行；
        # 任务开始/对话的澄清轮才走 ProseRelay（吞掉 JSON 尾）。
        self.plain_stream = plain_stream
        self.events: list[dict] = []
        self.subscribers: list[asyncio.Queue] = []
        self.done = False
        self.cancelled = False  # cancel() 置位：落库写 interrupted，而不是 failed
        self.proc: asyncio.subprocess.Process | None = None  # 供 cancel()/超时杀进程组
        self.exit_code: int | None = None
        self.session_id: str | None = None
        self.final: str = ""
        self.error: str | None = None
        self.usage = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0, "cacheWriteTokens": 0}
        self.stderr_tail: list[str] = []
        self.tool_calls = 0
        self.turns = 0
        self.started_at = datetime.now(timezone.utc)
        self.ended_at: datetime | None = None
        # 逐 token 流式（opx-typewriter 插件把增量 POST 回来；见 push_delta）
        self.relay = ProseRelay()
        self.prose = ""  # 已经上屏给用户的自然语言回复
        self.raw_text = ""  # 模型原始增量（含被吞掉的 JSON 尾，仅供排查）
        self.text_deltas = 0  # 收到的 text 增量段数（0 = 插件没生效 → 走兜底）
        self.reason_deltas = 0

    def publish(self, ev: dict) -> None:
        self.events.append(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:  # pragma: no cover - 订阅者太慢就丢给它自己
                pass

    def close(self) -> None:
        self.done = True
        self.ended_at = datetime.now(timezone.utc)
        for q in list(self.subscribers):
            try:
                q.put_nowait(None)
            except Exception:
                pass

    def push_delta(self, kind: str, text: str) -> None:
        """吃一条**逐 token 增量**（来自 dsh 插件），把可上屏的 prose 立刻广播出去。

        - `kind=text`：正常路径。经 `ProseRelay` 切掉 JSON 尾，只发 prose 增量
          （事件 `{"type": "delta", "text": ...}`；前端累加即成打字机）。
        - `kind=reason`：推理增量只记数、不上屏 —— dsh 的 reasoning 已经由 stderr 事件带出来了，
          再上屏会重复。（留数是为了排查「插件到底有没有收到东西」。）
        """
        if kind == "reason":
            self.reason_deltas += 1
            return
        if kind != "text" or not text:
            return
        self.text_deltas += 1
        self.raw_text += text
        if self.plain_stream:
            # 普通 agent 答复：不切 prose、不吞 JSON 尾（它本来就不是两段式契约）
            self.prose += text
            self.publish({"type": "delta", "kind": "text", "text": text})
            return
        chunk = self.relay.feed(text)
        if chunk:
            self.prose += chunk
            self.publish({"type": "delta", "kind": "text", "text": chunk})


#: 运行中的智能体调用：agent_jobs.id → 状态
REGISTRY: dict[int, JobState] = {}


def dsh_binary() -> str:
    return shutil.which("dsh") or "dsh"


def build_command(prompt: str, session_id: str | None = None, stream: bool = False) -> list[str]:
    # 模型路由跟着「API 管理」页的当前端点走（llm_config 生成的 patch）；没有再回退静态 patch。
    patch = llm_config.dsh_patch_for_run()
    cmd = [dsh_binary(), "--profile", settings.dsh_profile, "--patch", str(patch)]
    if stream:
        # 逐 token 流式：挂 opx-typewriter 插件（相对 name 锚定在 patch 文件旁，不需要 pnpm）
        cmd += ["--patch", str(Path(settings.repo_root) / settings.dsh_stream_patch)]
    if session_id:
        cmd += ["--session-id", session_id]  # 续会话（P0-3 对话式澄清）
    cmd += ["--json", prompt]
    return cmd


def delta_url(job_id: int) -> str:
    """插件把增量 POST 回的地址（插件只认 `OPX_DELTA_URL`，body 里带 `run`）。"""
    return f"{settings.delta_base_url.rstrip('/')}/api/agent-jobs/{job_id}/delta"


def delta_env(job_id: int) -> dict:
    return {"OPX_DELTA_URL": delta_url(job_id), "OPX_RUN_ID": str(job_id)}


def available() -> bool:
    return shutil.which("dsh") is not None


def emit_fallback_prose(state: JobState, prose_of=None) -> None:
    """插件没生效时的兜底：把**解析出来的 prose** 一次性补发上屏（`fallback=True`）。

    这样即便 dsh 改了扩展点、插件静默，用户仍然看得到答复 —— 只是没有逐字效果。
    """
    if state.text_deltas:
        return
    text = ""
    try:
        text = (prose_of or split_prose)(state.final or "")
    except Exception:  # pragma: no cover - 兜底不能自己炸
        text = ""
    if not text:
        return
    state.prose = text
    state.publish({"type": "delta", "kind": "text", "text": text, "fallback": True})


def _apply_event(state: JobState, ev: dict) -> None:
    t = ev.get("type")
    if t == "session":
        state.session_id = ev.get("sessionId")
    elif t == "status":
        if ev.get("phase") == "turn_start":
            state.turns += 1
        if ev.get("phase") == "step_end" and isinstance(ev.get("usage"), dict):
            for k in state.usage:
                state.usage[k] += int(ev["usage"].get(k) or 0)
    elif t == "tool_call":
        state.tool_calls += 1
    elif t == "final":
        state.final = ev.get("text") or ""
    elif t == "turn_end":
        reason = ev.get("reason") or {}
        if reason.get("kind") == "error":
            state.error = (reason.get("error") or {}).get("message") or "unknown error"


#: 跑完的调用（成功/失败/取消）保留最近这么多份状态：给晚订阅的前端重放事件 / 查 `/live`
#: （失败原因在 state.error），超出即淘汰 —— 有界保留而不是永久泄漏（审查 #10）。
RECENT_LIMIT = 32


def _kill(state: JobState) -> None:
    """杀掉 dsh 进程组（`start_new_session=True`；只杀 dsh 本身会留下它起的子进程）。"""
    proc = state.proc
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def _retire_state(state: JobState) -> None:
    """收尾清理 REGISTRY：结束的调用（成功/失败/取消）有界保留最近 RECENT_LIMIT 份。

    保留是为了 `/live`、`/events` 在跑完之后仍可读（失败原因在 `state.error` 里，前端晚订阅也看得到）；
    超出上限就淘汰最旧的，避免每次对话的完整 events/prose 常驻内存（审查 #10）。
    """
    finished = [jid for jid, st in REGISTRY.items() if st.done]
    for jid in finished[:-RECENT_LIMIT]:
        REGISTRY.pop(jid, None)


def _persist_outcome(state: JobState, cwd: Path, dsh_home: Path) -> None:
    """把本次调用摘要落库（独立 session：请求早已返回，且失败路径也要落）。"""
    from ..db import SessionLocal  # 延迟导入，避免循环
    from ..models import AgentJob

    with SessionLocal() as db:
        row = db.get(AgentJob, state.job_id)
        if row is None:
            return
        if state.cancelled:
            row.status = "interrupted"
        elif state.error or state.exit_code != 0:
            row.status = "failed"
        else:
            row.status = "ok"
        row.exit_code = state.exit_code
        row.ended_at = state.ended_at
        row.dsh_session_id = state.session_id or row.dsh_session_id
        row.tokens_in = state.usage["inputTokens"] or None
        row.tokens_out = state.usage["outputTokens"] or None
        row.cache_read = state.usage["cacheReadTokens"] or None
        row.cache_write = state.usage["cacheWriteTokens"] or None
        if state.session_id:
            # dsh 的会话目录名 = "-" + 工作区路径（把 / 换成 -，且带尾部斜杠）+ "-"
            escaped = (str(cwd).rstrip("/") + "/").replace("/", "-")
            row.dsh_jsonl_path = str(dsh_home / "sessions" / f"-{escaped}-" / state.session_id)
        db.commit()


def provider_runtime_env(extra_env: dict) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env)
    # DSH removes variables named *SECRET* from shell tools. These aliases carry
    # only store configuration, never a key; workflow children still read the
    # same endpoint through the existing local secret store.
    env["OPX_PROVIDER_STORE_DIR"] = str(settings.secret_dir)
    env["OPX_PROVIDER_STORE_BACKEND"] = settings.secret_backend
    return env


async def _consume(state: JobState, cwd: Path, dsh_home: Path, extra_env: dict,
                   session_id: str | None = None, on_done=None,
                   stream: bool = False, prose_of=None) -> None:
    cmd = build_command(state.prompt, session_id=session_id, stream=stream)
    env = provider_runtime_env(extra_env)
    env["DSH_HOME"] = str(dsh_home)
    state.publish({"type": "_meta", "command": " ".join(cmd), "cwd": str(cwd),
                   "dsh_home": str(dsh_home), "stream": stream})
    timeout = max(1, int(settings.agent_timeout or 0))
    stderr_task: asyncio.Task | None = None
    crashed = False

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        state.proc = proc
        if state.cancelled:  # 进程起来之前就被 cancel() 了：立刻收掉
            _kill(state)

        async def read_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                state.stderr_tail.append(text)
                state.publish({"type": "_stderr", "text": text})

        stderr_task = asyncio.create_task(read_stderr())

        assert proc.stdout is not None
        # 总超时：卡住的 dsh 到点就杀进程组并落 failed（否则任务页永久「对话中」）
        async with asyncio.timeout(timeout):
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    ev = json.loads(text)
                except json.JSONDecodeError:
                    state.publish({"type": "_raw", "text": text})
                    continue
                _apply_event(state, ev)
                state.publish(ev)
            state.exit_code = await proc.wait()
        await stderr_task
    except TimeoutError:
        crashed = True
        state.error = state.error or f"智能体调用超时（{timeout}s），已终止 dsh 进程"
        _kill(state)
        if state.proc is not None:
            with contextlib.suppress(Exception):
                state.exit_code = await asyncio.wait_for(state.proc.wait(), 5)
    except asyncio.CancelledError:
        # 后台 task 被取消（进程退出/测试收尾）：也要保证落库 + 清理 REGISTRY，然后继续抛
        crashed = True
        state.cancelled = True
        _kill(state)
        raise
    except Exception as e:  # 起进程失败（dsh 不在 PATH）/ 读流异常：同样落 failed，不留 running 僵尸
        crashed = True
        state.error = state.error or f"{type(e).__name__}: {e}"
        _kill(state)
        if state.proc is not None:
            with contextlib.suppress(Exception):
                state.exit_code = await asyncio.wait_for(state.proc.wait(), 5)
    finally:
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
        if state.exit_code is None:
            rc = state.proc.returncode if state.proc is not None else None
            state.exit_code = rc if rc is not None else -1
        # 逐 token 没生效（插件静默 / dsh 改扩展点）时兜底：把 prose 一次性补上屏
        # plain_stream 的答复没有 JSON 尾，整段就是正文，不能再按两段式切。
        if not crashed and stream:
            emit_fallback_prose(state, (lambda s: s) if state.plain_stream else prose_of)
        if state.error:
            state.publish({"type": "error", "error": state.error})
        state.close()
        try:
            # 落库摘要（新开 session，别用请求期的）—— v1.1：写 agent_jobs，不写 runs
            _persist_outcome(state, cwd, dsh_home)
        finally:
            _retire_state(state)
        # 完成回调（流式对话/对话页/自动优化：把这一轮的结果落进任务/会话/Loop）；
        # 失败路径也调用，调用方按 state.error / exit_code 自己判断，避免消息半途而废。
        if on_done is not None:
            try:
                on_done(state)
            except Exception:  # pragma: no cover - 回调失败不能拖垮消费者
                pass


def start(job_id: int, prompt: str, cwd: Path, dsh_home: Path,
          session_id: str | None = None, on_done=None,
          stream: bool = False, prose_of=None, plain_stream: bool = False) -> JobState:
    """登记状态并起后台消费者（不阻塞请求）。

    `stream=True`：挂 opx-typewriter 插件逐 token 回流（前端打字机）；跑完若一个增量都没收到，
    用 `prose_of(state.final)` 兜底补发 prose（见 `emit_fallback_prose`）。

    `plain_stream=True`：答复原样放行（正式 agent run 用）；否则按「prose + JSON 尾」两段式切分
    （任务开始/对话澄清用）。
    """
    state = JobState(job_id, prompt, plain_stream=plain_stream)
    REGISTRY[job_id] = state
    extra_env = {settings.llm_key_env: os.environ.get(settings.llm_key_env, "")} if settings.llm_key_env else {}
    if stream:
        extra_env.update(delta_env(job_id))
    asyncio.create_task(_consume(state, cwd, dsh_home, extra_env, session_id, on_done, stream, prose_of))
    return state


def running(job_id: int) -> JobState | None:
    st = REGISTRY.get(job_id)
    return st if st and not st.done else None


def _mark_interrupted(job_id: int) -> None:
    """cancel() 立即把 DB 里的 running 行改成 interrupted（不等 `_consume` 收尾，杀进程慢也不卡删除）。"""
    from ..db import SessionLocal  # 延迟导入，避免循环
    from ..models import AgentJob

    with SessionLocal() as db:
        row = db.get(AgentJob, job_id)
        if row is not None and row.status == "running":
            row.status = "interrupted"
            row.ended_at = row.ended_at or datetime.now(timezone.utc)
            db.commit()


def cancel(job_id: int) -> str:
    """取消一次智能体调用：置 cancelled → 杀 dsh 进程组 → 落库写 interrupted。

    返回一句人读结果（B 线 `POST /api/agent-jobs/{job_id}/cancel` 原样回给前端）。
    """
    state = REGISTRY.get(job_id)
    if state is None or state.done:
        return f"智能体调用 #{job_id} 不在运行（无法取消）"
    state.cancelled = True
    state.error = state.error or "用户取消"
    _kill(state)
    _mark_interrupted(job_id)
    return f"智能体调用 #{job_id} 已发送终止信号"


async def run_once(prompt: str, cwd: Path, dsh_home: Path, *,
                   session_id: str | None = None, timeout: float = 180.0) -> JobState:
    """跑一轮 dsh 并**等它结束**（P0-3 对话式澄清：请求/响应式，不做 SSE）。

    与 `start()`（后台 + SSE）共用 `_apply_event` 与事件契约，差别只在「阻塞到出最终回答」。
    带 `session_id` 时续会话（dsh 记忆在它自己的 sessions 目录里，所以 cwd / DSH_HOME 必须稳定）。
    """
    state = JobState(0, prompt)
    cmd = build_command(prompt, session_id=session_id)
    env = os.environ.copy()
    if settings.llm_key_env:
        env[settings.llm_key_env] = os.environ.get(settings.llm_key_env, "")
    env["DSH_HOME"] = str(dsh_home)

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async def pump_stdout() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                continue
            _apply_event(state, ev)

    async def pump_stderr() -> None:
        assert proc.stderr is not None
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            state.stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    try:
        await asyncio.wait_for(asyncio.gather(pump_stdout(), pump_stderr(), proc.wait()), timeout)
    except asyncio.TimeoutError:
        state.error = f"dsh 超时（{timeout:.0f}s）"
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), 5)

    state.exit_code = proc.returncode
    state.close()
    return state
