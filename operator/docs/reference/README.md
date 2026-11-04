# 调研证据快照

抓取时间：2026-10-14 16:12 CST ｜ 方式：`curl -L` 经 Clash 代理（`http://127.0.0.1:7897`）

| 文件 | 来源 URL | HTTP | 大小 |
|---|---|---|---|
| `cannbot-README.snapshot.md` | `https://raw.gitcode.com/cann/cannbot/raw/master/README.md` | 200 | 2639 B |
| `cannbot-skills-feature-list.snapshot.md` | `https://raw.gitcode.com/cann/cannbot-skills/raw/master/docs/feature-list.md` | 200 | 19035 B |

为什么留快照：GitCode 仓库页是前端渲染 + 仓群演进快（同一仓库的 skill 计数在不同入口已不一致），
正文快照是 `docs/prior-art.md` 里引用的唯一可回溯依据。

已知未取到：

- `https://raw.gitcode.com/Ascend/agent-skills/raw/master/README.md` → **429**（限流），未重试成功；
  该仓内容仅以仓库页与 GitHub 镜像 README 的搜索摘要为据（`prior-art.md` 中标 `[转述]`）。
- GitCode 站内搜索接口需要 `private-token`（`/api/v5/...` 返回 400 required），游客网页搜索返回
  "当前游客访问过多，请稍后重试或登录后访问"，故站内检索改走 org 页枚举 + 搜索引擎 `site:gitcode.com`。
