// op-agent 流式原型插件：把 dsh 的 agent assistant 增量 POST 到控制台后端，
// 让「任务开始」类对话能做真·打字机。
//
// 挂载：`--patch ops/dsh/opx-typewriter.patch.yml`（相对 name 锚定在 patch 文件旁）
// 环境：OPX_DELTA_URL（后端收增量端点）、OPX_RUN_ID（本轮标识，用于归属）
//
// 规矩：**不写 stdout/stderr**（`--json` 模式的 stdout 被事件流占用），只走独立 HTTP 通道。
// 事件：`agent/assistant-stream`（agent 作用域）→ 只含真正的 agent 生成，标题/压缩等旁路不会进来。
export const name = "opx-typewriter";

export function apply(ctx) {
  const url = process.env.OPX_DELTA_URL;
  const run = process.env.OPX_RUN_ID || "unknown";
  if (!url) return; // 没配就静默不干活

  // 串行发送，保序 + 限并发（本机回环，开销可忽略）
  let chain = Promise.resolve();
  const send = (body) => {
    chain = chain
      .then(() =>
        fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ run, ...body }),
        }),
      )
      .catch(() => {});
  };

  ctx.effect(
    () =>
      ctx.on("agent/assistant-stream", ({ agent, frame }) => {
        if (frame.type === "start") return void send({ k: "start", session: agent.id });
        if (frame.type === "end") return void send({ k: "end" });
        if (frame.type !== "chunk") return;
        const c = frame.chunk;
        if (c.type === "text-delta") send({ k: "text", t: c.text });
        else if (c.type === "reasoning-delta") send({ k: "reason", t: c.text });
      }),
    "opx-typewriter",
  );
}
