# dsh 运行时配置（op-agent）

## 接 opencode go 网关（OpenAI 兼容）—— 已实测跑通

```bash
export OPX_LLM_KEY=<你的 opencode go key>      # 只放环境变量，别写进仓库
dsh --profile headless \
    --patch ops/dsh/llm-opencode-go.patch.yml \
    --json "<任务>"
```

### 要点（都是实测换来的）

| # | 事实 |
|---|---|
| 1 | opencode go 的 `/v1/chat/completions` **要求 `x-opencode-session` 请求头**，缺了返回 400 `MissingSessionID`；**任何稳定 id 都能用**（patch 里已塞 `dsh-op-agent-0001`） |
| 2 | 它**不能用**官方的 `dsh-llm-deepseek` 适配器（那条走 DeepSeek Messages 风格，会 404）；要用 **`pi-ai` + `api: openai-completions`** |
| 3 | patch 语法：**同 id 的裸条目 = 覆写，且替换整行 `config`（不是深合并）**；新增条目才用 `insert:`。`llm-pi-ai` 已在 profile 里，重复 `insert` 会因"路由重复声明"而**静默不激活**（只在 stderr 留 warning） |
| 4 | 默认模型锚点是一条独立条目 **`agent-default-model`**（`provider` + `model`），改路由时要一起覆写它，否则模型名会被路由回官方 route |
| 5 | 官方文档要求客户端**带自己的 User-Agent**（不要通用 SDK 名）并保持稳定 session id，否则可能被按滥用限流（patch 里设了 `op-agent-console/0.1 (dsh)`） |
| 6 | `GET /v1/models` 可直接列可用模型（走 bearer 鉴权）：当前含 `deepseek-v4.1-flash`、`deepseek-v4-flash`、`deepseek-v4-pro`、`glm-5.1` 等 |

## CANNBot skills / agents（按阶段加载的领域能力）

安装（**装进任务工作区**，不是缓存目录）：

```bash
scripts/install_skills.sh              # 默认目标 workspaces/cann-ops
scripts/install_skills.sh --verify     # 按 ops/dsh/skills.lock 复核（文件级 sha256）
```

要点（2026-10-15 复查时踩出来的）：

| # | 事实 |
|---|---|
| 1 | 落点必须是**任务工作区** `<workspace>/.dsh/{skills,agents}`（dsh 按 projectRoot 找能力）。之前装到 `~/.hermes/cache/scratch/...` —— 那是 scratch 目录，**空闲 24h 会被清**。 |
| 2 | 来源换成仓内 `.skills-src/cannbot`（git 忽略、不随缓存清理）；来自 `https://gitcode.com/cann/cannbot.git`。安装器的 source 模式是 **symlink**，脚本装完会**物化成真文件**，让工作区自包含。 |
| 3 | 共享 skills（`vendor/cannbot-skills`）那份是 **dirty** 的 → 安装器默认拒绝，须加 `--skills <path>`（本地路径覆盖保留未提交内容）。 |
| 4 | macOS 上必须 `CANNBOT_SKIP_INIT=1`：CLI 默认跑插件 `init.sh`，而 `source-plugin-init.sh` 在 `set -u` 下展开空数组 `"${SKILLS_ARGS[@]}"`，**系统 bash 3.2** 会报 `unbound variable` 直接失败。 |
| 5 | 装了 skills 之后工作区会多出 `.dsh`(14 MB) 与 `.cannbot/dependencies`(~450 MB) —— `nm_test.sh` 同步到远端时**必须排除**（容器只要源码与脚本），否则一次 sync 就拖住整个 run（实测卡在 rsync，容器根本没起来）。 |
| 6 | `ops/dsh/skills.lock` 记 cannbot commit + skills commit + dirty 标志 + **每个文件的 sha256**：光写 pin commit 不够（dirty 内容复现不出来）。 |
