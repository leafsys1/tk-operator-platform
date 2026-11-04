"""最小原型（旁路验证，不进正式链路）：

1. **dsh 插件版**（`/dsh/stream`）：起一轮 `dsh headless`，挂 `opx-typewriter` 插件；
   插件把 `agent/assistant-stream` 的 token 级增量 POST 回 `/dsh/delta`，这里用 SSE 转发给前端。
2. **opencode SSE 版**（`/opencode/stream`）：读本机 opencode 服务的 `/api/event`，
   转发 `session.text.delta`。

两者的差别正是这轮调研的结论：dsh 程序化面（headless/ACP）只给提交态，
要真 token 得**自己挂插件**；opencode 服务端则原生就有增量 SSE。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import settings
from ..services import llm_config

router = APIRouter(prefix="/api/proto", tags=["prototype"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================ dsh 插件版

#: run_id → 该轮增量的队列（同进程内；单人本机够用）
DASH_RUNS: dict[str, asyncio.Queue] = {}


@router.post("/dsh/delta")
async def dsh_delta(payload: dict) -> dict:
    """收 dsh 插件 POST 来的增量。"""
    q = DASH_RUNS.get(str(payload.get("run")))
    if q is None:
        return {"ok": False, "why": "unknown run"}
    k = payload.get("k")
    if k == "text":
        q.put_nowait({"k": "text", "t": payload.get("t") or ""})
    elif k == "reason":
        q.put_nowait({"k": "reasoning", "t": payload.get("t") or ""})
    return {"ok": True}


@router.get("/dsh/stream")
async def dsh_stream(prompt: str, request: Request) -> StreamingResponse:
    run = uuid.uuid4().hex
    q: asyncio.Queue = asyncio.Queue()
    DASH_RUNS[run] = q
    delta_url = str(request.base_url).rstrip("/") + "/api/proto/dsh/delta"

    async def gen():
        proc = None
        try:
            yield _sse("meta", {"run": run, "delta_url": delta_url})
            dsh = shutil.which("dsh") or "dsh"
            patch_llm = llm_config.dsh_patch_for_run()
            patch_plugin = Path(settings.repo_root) / "ops/dsh/opx-typewriter.patch.yml"
            cmd = [dsh, "--profile", settings.dsh_profile,
                   "--patch", str(patch_llm), "--patch", str(patch_plugin), prompt]

            cwd = Path(tempfile.gettempdir()) / f"opx-dsh-proto-{run}"
            cwd.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["OPX_DELTA_URL"] = delta_url
            env["OPX_RUN_ID"] = run

            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(cwd), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out_buf: list[str] = []
            err_buf: list[str] = []

            async def drain(stream, sink: list[str]) -> None:
                while True:
                    b = await stream.readline()
                    if not b:
                        break
                    sink.append(b.decode("utf-8", "replace"))

            t_out = asyncio.create_task(drain(proc.stdout, out_buf))
            t_err = asyncio.create_task(drain(proc.stderr, err_buf))

            async def wait_proc() -> None:
                await proc.wait()
                q.put_nowait(None)

            waiter = asyncio.create_task(wait_proc())
            got_text = False
            try:
                async with asyncio.timeout(300):
                    while True:
                        item = await q.get()
                        if item is None:
                            break
                        if item["k"] == "text":
                            got_text = True
                        yield _sse(item["k"], {"delta": item["t"]})
            except TimeoutError:
                yield _sse("stream_error", {"message": "dsh 超时（300s）"})
                proc.kill()
            finally:
                await asyncio.gather(t_out, t_err, return_exceptions=True)
                await waiter

            rc = proc.returncode
            if not got_text:  # 插件没生效时的兜底：把最终答案一次性发出来
                answer = "".join(out_buf).strip()
                if answer:
                    yield _sse("text", {"delta": answer})
            if rc not in (0, None) and err_buf:
                yield _sse("stream_error", {"message": err_buf[-1][:200]})
            yield _sse("done", {"ok": rc == 0, "exit": rc, "plugin": got_text})
        except Exception as e:  # pragma: no cover - 原型
            yield _sse("stream_error", {"message": str(e)})
            if proc is not None and proc.returncode is None:
                proc.kill()
        finally:
            DASH_RUNS.pop(run, None)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# ============================================================ opencode SSE 版

SERVICE_JSON = Path.home() / ".local/state/opencode/service.json"


def _opencode_service() -> tuple[str, dict]:
    if not SERVICE_JSON.is_file():
        raise HTTPException(400, f"找不到 {SERVICE_JSON}（本机 opencode 服务没在跑？）")
    cfg = json.loads(SERVICE_JSON.read_text(encoding="utf-8"))
    url, pw = cfg.get("url"), cfg.get("password")
    if not url or not pw:
        raise HTTPException(400, "service.json 缺 url/password")
    basic = base64.b64encode(f"opencode:{pw}".encode()).decode()
    return url, {"Authorization": f"Basic {basic}"}


@router.get("/opencode/stream")
async def opencode_stream(prompt: str) -> StreamingResponse:
    url, headers = _opencode_service()

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                r = await client.post(f"{url}/api/session", json={}, headers=headers)
                if r.status_code >= 400:
                    yield _sse("stream_error", {"message": f"建会话失败 {r.status_code}: {r.text[:200]}"})
                    return
                sid = (r.json().get("data") or {}).get("id")
                yield _sse("meta", {"sessionID": sid, "server": url})

                async with client.stream("GET", f"{url}/api/event", headers=headers) as resp:
                    if resp.status_code >= 400:
                        yield _sse("stream_error", {"message": f"订阅事件失败 {resp.status_code}"})
                        return

                    async def send_prompt() -> None:
                        await client.post(f"{url}/api/session/{sid}/prompt",
                                          json={"text": prompt}, headers=headers)

                    task = asyncio.create_task(send_prompt())
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            m = json.loads(payload)
                        except ValueError:
                            continue
                        t = m.get("type") or ""
                        d = m.get("data") or {}
                        if d.get("sessionID") != sid:
                            continue
                        if t == "session.reasoning.delta":
                            yield _sse("reasoning", {"delta": d.get("delta") or ""})
                        elif t == "session.text.delta":
                            yield _sse("text", {"delta": d.get("delta") or ""})
                        elif t in ("session.execution.succeeded", "session.execution.failed"):
                            yield _sse("done", {"ok": t.endswith("succeeded")})
                            break
                    await task
        except httpx.HTTPError as e:
            yield _sse("stream_error", {"message": f"连不上 opencode 服务：{e}"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
