# opx-typewriter（逐 token 流式插件）

把 dsh 的 **agent assistant 增量**转发给 op-agent 控制台，让「任务开始」页的对话变成真·打字机。

**状态（2026-10-15）**：已从旁路原型**并入正式链路** —— 澄清对话（`POST /api/tasks/{id}/chat/stream`）
每轮都挂它。原型页（控制台的「流式原型」）保留作对照实验，两条路互不影响（收增量的端点不同）；
**2026-10-21 起该页在侧栏隐藏**（`StreamProto.tsx` 与 `/api/proto/*` 保留，恢复方式见 `web/src/App.tsx` 的 NAV 注释）。

## 它 hook 什么

```js
ctx.on("agent/assistant-stream", ({ agent, frame }) => { … })
```

- 事件是 **agent 作用域**的（`dsh-headless` 自己也用它把 reasoning 写 stderr）；
- `frame.type ∈ {start, chunk, end}`；chunk 帧里 `frame.chunk.type ∈ {text-delta, reasoning-delta, tool-call-delta, block-start/end, usage, finish}`，文本在 `chunk.text`；
- 会话标识：`agent.id`（= `agent.session.id`）。
- **只读观察者**：不干扰 agent loop、不干扰工具调用（实测带工具的一轮里工具照常执行）。

## 怎么挂

```bash
dsh --profile headless \
    --patch ops/dsh/llm-opencode-go.patch.yml \
    --patch ops/dsh/opx-typewriter.patch.yml \
    --json "你的任务"
```

`opx-typewriter.patch.yml` 用 `insert:` + 相对 `name: ./opx-typewriter/index.mjs`
（相对路径**锚定在 patch 文件旁**）→ **不需要 pnpm / node_modules**。

控制台侧由 `agent_runner.start(..., stream=True)` 自动加这条 `--patch`（配置项
`settings.dsh_stream_patch`，总开关 `OPX_CHAT_STREAM=0` 可关掉）。

## 环境变量

| 变量 | 作用 |
|---|---|
| `OPX_DELTA_URL` | 后端收增量的端点。正式链路 = `{OPX_DELTA_BASE_URL}/api/agent-jobs/<job_id>/delta`；原型 = `/api/proto/dsh/delta`。**不设则插件静默** |
| `OPX_RUN_ID` | 本轮标识（正式链路 = `agent_jobs.id`），用于把增量归属到某次调用 |

## 与后端的契约

插件按 `{run, k, t}` POST（`k ∈ {start, text, reason, end}`），串行发送保序：

- 后端 `POST /api/agent-jobs/{job_id}/delta` → `JobState.push_delta()`；
- 经 `backend/app/services/prose_relay.py` 切分：**prose 上屏**（`{"type":"delta"}` SSE 事件）、
  **JSON 尾吞掉**（模型按「先流式说人话 + `<<<SPEC>>>` + JSON」的两段式契约输出，见
  `docs/UI-task-start-v1.md` §3.1）；
- `reason` 增量**只记数不上屏**（dsh 的 reasoning 已由 stderr 事件带出来，再上屏会重复）。

## 注意

- **不写 stdout/stderr**：`--json` 模式的 stdout 被事件流占用；插件只走独立 HTTP 通道。
- **真 token 级**：实测一轮约 47~78 段 `text-delta`（+ 同量级 `reasoning-delta`），
  带工具的一轮还有 78 段 `tool-call-delta`。
- **兜底**：插件静默或 dsh 改了扩展点时，后端在 job 结束时把解析出的 prose 一次性补发
  （`fallback: true`）—— 退化成「不逐字但看得到答案」，不会白屏。
- dsh 处开发者预览期，`agent/assistant-stream` 是公开 JSDoc 事件，但版本仍可能变 —— 升级 dsh 后
  用日志里 `text_deltas` 判断插件是否还有效。
