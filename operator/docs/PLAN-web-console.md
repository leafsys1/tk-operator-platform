# 方案：算子开发智能体 Web 控制台（op-agent console）

> ⚠️ **本文件（v1.0）已被 [`PLAN-web-console-v1.1.md`](./PLAN-web-console-v1.1.md) 取代**
> （2026-10-14 合并外部建议稿《算子开发智能体优化方案 v2.0》）。本文件保留为 **v1.0 决策溯源记录**，不再更新；
> 后续一切以 v1.1 为准。v1.0 与 v1.1 的差异处置见 v1.1 的 **附录 A**。

状态：**v1.0 · 定稿（可开工）** ｜ 日期：2026-10-14 ｜ 前置文档：`REQUIREMENT.md`、`prior-art.md`
v0.3→v1.0 变更：① **部署形态定为「(ii) 控制台 + dsh 在独立机器，执行委托到裸金属」**，新增 §3.2 执行委托设计；
② A/B/C/N 组决策全部落定（§8.1），§8.2 只剩 D 组；③ 门禁、判据口径、性能采集层写入正文。

> 证据级别沿用本仓约定：`[已核验]` = 我实际打开过源码或官方文档并引用原文；`[转述]` = 二手；
> `[推断]` = 我的判断；`[待实测]` = 必须真跑一次才知道。

---

## 0. 一句话定位

**给自己用的算子开发控制台**：管裸金属服务器与算子开发镜像 → 发起与编排算子开发任务 → 和 agent 对话看过程 → 看结果（精度 / 性能）→ 管消耗。
**agent 能力不外采、不自研内核**：harness 用 DSH（含它的对话 UI），领域能力用 CANNBot 的 skills。

## 1. 边界

| 项 | 处置 | 为什么 |
|---|---|---|
| agent 内核 / LLM loop | **不做** | DSH 是 MIT、web-first、插件化的成熟 harness；自研等于把成熟件换成自写版 `[已核验]` |
| CANNBot skills 内容 | **不重写** | 与 CANN 团队同步的领域知识资产（手册/基准集/判据标准/模板），上游月级迭代 `[已核验]` |
| **裸金属服务器 + 镜像管理** | **做**（原 v0.1 误判为"不做资源调度器"，已更正） | 目标是 NPU 裸金属，与 PAI-EAS 无关。范围：登记服务器 → SSH 连通 → 拉取镜像清单 → 标记若干为「**算子开发镜像**」。**不做跨机调度/排队编排**，也不接管任何既有平台 |
| **控制台与 dsh 的部署位置** | **独立机器**（(ii) 分支，本轮定） | 控制台与 dsh 跑在普通机器（本机或公司服务器），**裸金属只负责编译/运行算子**。代价是要做"执行委托"（§3.2），换来的是：不必在共享的裸金属上放模型 key，agent 工作区与执行环境解耦 |
| **对话页** | **复用 DSH，不自研** | DSH 自带 Web UI（工作区 + 会话列表 + 对话 + 权限审批），资源路径全是相对路径、重定向 `./` → **可反代挂到子路径** `[已核验·实测]`。本控制台导航里放一个入口即可 |
| 通用 LLM 观测（token / 费用 / 上下文 / 轨迹） | **不自研** | `headless --json` 的 `step_end` 事件**直接带 usage**（input/output/cacheRead/cacheWrite/total）`[已核验·实测]`，不用另做埋点；文本明细留在 sessions JSONL 里 |
| 算子结果视图（精度 / 性能） | **做**（无人可替） | 现有工具都没有算子维度的精度/性能数据模型；`cannbot-insight` 只有一个"手工填报 CANN-Bench 总分"的口 `[已核验]` |

## 2. 技术栈

| 层 | 选型 | 为什么是它 |
|---|---|---|
| 后端 | **Python 3.12（uv）+ FastAPI + Pydantic v2 + Uvicorn** | 领域资产全在 Python：SSH 探针、容器/镜像操作、读 `sessions/*.jsonl`、子进程驱动 dsh |
| ORM / 迁移 | **SQLAlchemy 2.0 + Alembic** | 实体有真关系（project→task→run→artifact），手写 SQL 到 P3 会崩 |
| 存储 | **SQLite（v1 单人）→ PostgreSQL（要多人/登录时同层切换）**（D4 待定） | 换库不动业务代码 |
| 实时 | **SSE（Server-Sent Events）** | run 事件单向推送，比 WebSocket 简单够用 |
| 本机 agent 执行 | **子进程 `dsh headless --json`**，每 run 一进程，默认串行（可配上限） | 与 `evalscope-perf` 同构；并发跑会互相抢资源污染指标 |
| **远端执行委托** | **本机 wrapper 脚本 + `rsync` + `ssh docker run`**（§3.2） | 把 ssh/docker 细节封在脚本里，**不交给模型自己拼参数** |
| 远端访问 | SSH（本机已有 tk36 免密 PEM） | 裸金属上没有别的通道 |
| 前端 | **React 19 + TypeScript + Vite + Tailwind CSS v4 + shadcn/ui** | 现代控制台最成熟的一套；`cannbot-insight` 同代栈 `[已核验]`，IA 可参考（**代码不可抄，CANN OSL**） |
| 表格 / 数据 | **TanStack Table + TanStack Query** | 任务/用例/镜像/消耗都是重表格 + 服务端查询 |
| 图表 | **ECharts** | 精度误差分布、性能对比（含基线/极限）、Roofline、时间线一库全包 |
| 其他前端件 | `lucide-react`、`sonner`、`framer-motion`（少量） | 与 shadcn 同生态 |
| 代码组织 | monorepo：`backend/`（FastAPI）+ `web/`（Vite SPA）+ `ops/`（wrapper 脚本） | 开发期 Vite 反代 FastAPI；生产 FastAPI 托管 build 产物，**部署仍是单进程** |
| agent 运行时 | **DSH**（MIT），profile + YAML patch；对话 UI 用 `dsh --profile web`；**v1 统一走 CLI `0.2.0-rc.2`** | SDK 自带 `0.1.5-rc.1`，两套版本不同，别混（B5） |
| 领域能力 | CANNBot skills，统一安装器直调装到 `.dsh/skills|agents`，**pin commit**（主仓 `1f9b84f` / skills `81d2bdb`） | 安装器 `update` 保留用户改动 `[已核验·源码]` |
| 密钥 | 只存引用名 + 环境注入 / 600 文件，**页面不明文回显** | 模型 key 只落在"控制台+dsh"那台机器；裸金属上不放模型 key |
| 时区 | **库里统一存 UTC**，界面按本地时区显示 | 避免 UTC/CST 混用（压测那边已踩过） |

**不选的两条**：① **全 Node（Next.js + Prisma）全栈**——探针/执行/真机脚本都在 Python，会被迫在 Node 里 spawn python；② **Jinja2 服务端渲染**——做不出行内筛选、拖拽推进、实时事件流这类交互。

## 3. 数据模型（**唯一不可逆的部分**）

```
Project（一个算子开发项目 = 一个工作区 + 一台执行机 + 一个算子开发镜像）
 └─ Task（一个算子开发需求）
     └─ Run（一次执行：设计/开发/验收/交付/测试/性能）
         └─ Artifact（规格 / 代码 diff / 用例 / 精度报告 / 性能数据 / 日志 / 交付件）
Conversation（对话，对应一个 DSH 会话）
```

| 表 | 关键字段 |
|---|---|
| `projects` | id、name、workspace_path（本机工作区，= dsh 的 cwd / projectRoot）、host_id、image_id、dsh_profile、dsh_home、created_at |
| `tasks` | id、**project_id**、title、op_name、target_chip、input_form（自然语言/数学式/参考代码）、spec_json、stage、created_at |
| `runs` | id、task_id、kind、**host_id / image_id（本次实际用的）**、`dsh_session_id`、`dsh_jsonl_path`、status、started_at、ended_at（UTC）、exit_code、**usage 摘要**（tokens_in/out/cache_read/cache_write）、cost、**remote_workdir**（裸金属上的临时目录）、sync_state |
| `run_commands` | id、run_id、seq、argv、cwd、exit_code、stdout_path、stderr_path、started_at、ended_at（委托执行的每条命令，用于审计与复现） |
| `artifacts` | id、run_id、type、path、sha256、size、created_at |
| `stage_events` | id、task_id、from_stage、to_stage、actor、reason、created_at |
| `hosts` | id、name、**kind=裸金属**、address、ssh_alias/凭据引用、cards（型号×数量）、cann_version、docker_ok、npu_state、probe_state、last_probe_at、note |
| `host_images` | id、host_id、image_ref、tag、digest、size、`is_op_dev`、scan_state、cann_version、python/torch/torch_npu 版本、**env_recipe**（把镜像配成可跑环境的命令）、**preflight_cmd**（可选自检命令）、last_scanned_at、note |
| `conversations` | id、project_id、`dsh_session_id`、title、created_at、last_active_at、message_count、tokens、cost |
| `endpoints` | id、name、provider（`dsh-llm-deepseek` / `dsh-llm-pi-ai`）、base_url、model、key_ref、probe_state、max_model_len、last_probe_at |

**三个设计要点**

1. `runs.dsh_session_id` + `dsh_jsonl_path` 指向 DSH 会话（`<dsh_home>/sessions/<工作区转义名>/<session-id>/`）——"过程视图"与"结果视图"天然对齐。
2. **事件不做全量落库**：`sessions/*.jsonl` 已是全量真相源；库里只存**索引与摘要**（run↔session 映射、usage、首末时间、错误），直播走内存 → SSE。避免两份口径打架，也省存储。
3. **`host_images.env_recipe` 不是可选装饰**：能起的镜像 ≠ 跑得动算子的环境。宿主机 CANN 是编译/开发装（缺 GE 插件与 `libhccl`），能跑的只有容器里那套（hccl shim + pylibs + uv cpython3.12 + PYTHONPATH 追加）`[既有记录·待复核]`。

### 3.1 阶段与门禁

四阶段：**需求与设计 → 开发 → 验收 → 交付**（原「上库」改名）

| 阶段 | 进入条件 | 门禁（产物齐备即通过） | 主要产物 |
|---|---|---|---|
| 一 · 需求与设计 | — | **需求规格齐备**：输入形态 + 输入/输出形状 + dtype + 目标硬件 | `spec.json`、方案设计.md |
| 二 · 开发 | 需求规格齐备 ✓ | **代码编译通过**：容器内编译无错 + 同步配对检查（`ascendc-sync-audit`） | kernel 源码、`run.sh`、代码 diff |
| 三 · 验收 | 编译通过 ✓ | **精度报告达标**：常规/边界/异常用例跑完且判据通过（口径见 §3.3） | 用例集、精度报告、性能数据 |
| 四 · 交付 | 精度达标 ✓ | **交付包齐备**：补丁 + run 包 + 交付文档 | `delivery/{*_DELIVERY.md, patches/, packages/}` |

- **门禁硬拦**（已定）：未齐不给推进，并列出缺哪一项。
- 每次推进/打回写 `stage_events` → 阶段推进不靠人工记忆。
- 「验收」≠「测试管理」：前者看**门禁是否达标**，后者看**用例集与执行记录**。

### 3.2 上机测试（(ii) 部署形态的执行模型）

**工作流：本地 agent 改代码 → 改到满意 → 上机测一次。** 因此不需要"命令级代理 / wrapper 框架"，
只需要一个**显式动作**（已实现：`scripts/nm_test.sh`，配置 `ops/nm.conf`）：

```
本机：工作区（agent 随便改，平时不下发任何东西）
        │  scripts/nm_test.sh '<测试命令>'
        ▼
① rsync 工作区 → tk36:/data/lfx/op-agent/runs/<run_id>/ws
② 测试命令写进 run.sh 并挂进容器执行：
   docker run --rm -v …/ws:/ws -v …/run.sh:/run.sh -v …/out:/out -w /ws <算子开发镜像> \
     bash -lc 'set -o pipefail; bash /run.sh 2>&1 | tee /out/run.log'
③ rsync 回读 …/out → 本机 runs/<run_id>/out
④ 写 manifest.json（每个产物 sha256）+ docker_cmd.txt（原样命令，用于复现）
⑤ 可选：POST /api/tasks/<id>/runs 回填台账（配 OPX_API + OPX_TASK_ID）
```

要点：

- **这不是封装框架**：agent 平时自由改代码（本机），只有"要真机验证"时才调这一条命令。命令原文落 `run.sh`／`docker_cmd.txt` 随 run 留档，`tool_call` 事件另有一份 → 审计够用。
- **不做命令级白名单**：个人专用机上先靠约定（项目 `AGENTS.md`），不做硬拦；真出现误操作再补。
- **不留远端垃圾**：默认保留远端目录便于复核，`--clean` 才删；每 run 一个 namespace 隔离。
- **镜像与配方来自控制台**：`OPX_IMAGE` 从「服务器与镜像」页复制；`OPX_ENV_RECIPE` 处理"镜像能起 ≠ 能跑算子"。
- **已验证**：`--dry-run` 与 `--sync-only` 真跑通（43 文件 / 464K 往返，自测目录已清理）；**容器那一步未跑**（等你点头）。
- 备选（P6 再评估）：DSH 的 `packages/ssh` 提供方（"共享连接 + 远端文件系统 + 托管子进程"`[已核验·官方文档]`），把远端执行做成原生能力。

### 3.3 判据与性能采集口径（本轮定）

- **精度判据**：默认走官方 `ops-precision-standard`（按 dtype 分类的 atol/rtol），**同时支持自定义档**（你自己的 gamma_k / ULP 口径作为可选档，按项目切换）。
- **异常输入判定**：默认**必须明确报错且错误码/信息与参考实现一致**才算 PASS；可自定义降级为"不崩即算过"（官方无对应错误码时用）。
- **性能采集**：**只到算子层**（`msopprof` / `msprof` 采单算子），**不采模型层**；结果视图要做**基线/极限对比**（对 `aclnnMatmul` 基线与上一版，并在可能时给出理论上限）。

## 4. 页面（IA）

| 页面 | 回答什么决策 | 优先级 |
|---|---|---|
| **任务详情（作业台）** | 这次 run 做了什么、产物在哪、结果怎样 | **MVP 核心** |
| 任务列表 / 项目 | 现在有哪些项目与任务、卡在哪 | MVP |
| **对话** | 和 agent 说话、看历史 | **P0-2（挂上就有）** |
| **服务器与镜像** | 任务在哪台机器、用哪个镜像跑 | **P0**（没有它跑不了） |
| 阶段看板 | 某个算子卡在哪一阶段、门禁过没过 | P3 |
| 测试管理 | 用例集与执行记录如何、精度/性能是否达标 | P3 |
| API 管理 | 用哪个模型/key、通不通、窗口多大（配置态） | P4 |
| 消耗情况 | 这批任务花了多少、花在哪（运行态） | P4 |

**边界**：① 阶段（流程状态机）≠ 测试（用例与执行记录）；② API 管理（配置态）≠ 消耗（运行态）；③ 对话（与 agent 交互）≠ 任务台账，**两个入口、一套数据**。

### 4.1 服务器与镜像

- 登记：名称、地址、SSH 凭据引用；探针：可达性、`npu-smi`（卡型号/数量/健康）、CANN 版本、docker 可用性。
- **镜像清单**：SSH 列 `docker images`（ref/tag/digest/size/创建时间）落 `host_images`；**悬空镜像单独标记**（实测 tk36 有 4 个 `<none>`、合计 ≈81.8 GB 可回收）。
- **标记「算子开发镜像」**：勾选 + 记 `env_recipe`（已定：**只勾选 + 记配方，不做预检门禁**；`preflight_cmd` 保留为可选手动动作）。
- **每个 run 绑定 (host, image, env_recipe)** —— 结果可复现的最低要求。

### 4.2 对话页（已定：反代挂载）

- `dsh --profile web` 起在某个端口；反代**剥离挂载前缀、保留浏览器可见 Host、转发 WebSocket upgrade、改写 cookie Path**；`/?token=…` 换签名 cookie（实测 `dsh-auth-<实例哈希>`、`Path=/`、`HttpOnly`、`SameSite=Strict`、30 天）`[已核验·实测]`。
- 本机实测 **`--public-url` 在 `0.2.0-rc.2` 不存在**（文档超前于发布版），但**前端资源全用相对路径**，所以挂子路径这条路成立；`--trusted-host` 可用。
- **仍未验**：WebSocket 升级经反代（部署时用 caddy/nginx 顺手验）。
- 多用户：官方模型是单实例 + 代理 + token/cookie；要隔离就每人一个实例（独立 `DSH_HOME` + 端口）`[推断]`。

## 4.5 界面与设计（已定：皮肤 B 深色玻璃 + 亮）

**设计目标**：现代控制台——信息密度高但层级清楚，状态色语义化；统一间距与圆角节奏、卡片分层、图表配色成套、空态/加载/错误态都认真做；不做花哨动效。
**样机已出**：`docs/mockups/console-ia-v1.html`（三皮肤对比，已选 B+亮）、`docs/mockups/console-pages-v1.html`（七页）。**内容先冻结，落码后按实际使用再调。**

| 屏幕 | 视觉要点 |
|---|---|
| 任务列表 / 项目 | 重表格（排序筛选）+ 状态胶囊 + 四格门禁条 + 行内动作 |
| **任务详情** | 三段式：左＝阶段时间线／中＝产物与报告（diff、用例表、精度图、性能图，含基线/极限）／右＝实时事件流（SSE，可暂停）；顶部 KPI |
| 服务器与镜像 | 卡片 + 探针状态灯 + 镜像表（算子开发镜像标记、悬空镜像、可回收空间） |
| 对话 | 导航入口 + 会话台账（session ↔ task/run） |
| 阶段看板 | 四列看板 + 拖拽推进 + 门禁未满原因 |
| 测试管理 | 用例表 + 误差分布图 + 性能对比图（判据可切"官方档/自定义档"） |
| 消耗情况 | 按项目/阶段/模型堆叠柱 + 趋势 + 明细（金额与 token 分开计价） |

**设计 token 化**：颜色/间距/字号/圆角/阴影一套 CSS 变量，亮暗两套主题。

## 5. 路线（纵切 MVP）

| 阶段 | 交付物 | 验收 |
|---|---|---|
| **P0-1 设计** ✅ | 2–3 套 HTML 样机（暗亮双主题，同页切换），定下设计 token | 已选定 **B 深色玻璃 + 亮** |
| **P0 骨架 + 资源** | FastAPI + SQLAlchemy/Alembic + SPA 壳；`projects/hosts/host_images` 表 + 服务器登记与镜像清单页；任务详情页渲染**手工登记**的一次 run | 能登记 tk36、列出镜像、标出一个算子开发镜像 |
| **P0-2 对话与运行时** | 装 dsh（`--allow-scripts` 放开原生包）、反代挂载 `dsh web` 当对话页、每项目一个 `dsh_home`、`session-log-deepseek: enabled:false` | 控制台里能进对话页并跑通一个真实小任务 |
| **P1 执行链** ✅ | 上机测试脚本（`scripts/nm_test.sh`）+ `dsh headless --json` 子进程（本机改代码）+ 事件经 SSE 上屏、摘要落库 + 台账回填 | ✅ **已实测**：控制台点「发起」→ dsh 跑通 → 事件流实时上屏 → 台账回填 session/usage/exit |
| **P2 结果视图** | 精度报告 + 性能数据渲染（ECharts），含**基线/极限对比** | C3/C4 两类结果可读、可导出 |
| **P3 阶段 + 测试** | 状态机（门禁硬拦 = 产物齐备）+ 推进/打回；用例集与执行记录；判据官方档/自定义档 | 阶段推进不靠人工记忆 |
| **P4 API + 消耗** | API 登记与探针（掩码）、消耗聚合 | 两个薄页面 |
| **P5 能力接入** | CANNBot skills 装进 `.dsh/skills`（pin commit）+ 必要的自写 skill | 用官方 skill 跑通一条真实算子任务 |
| **P6（可选）** | SDK 常驻替换子进程；`packages/ssh` 原生远端执行；SQLite → PostgreSQL | 多人/长任务时再考虑 |

**MVP 纵切候选**：202608 那批交付里的 complex64 mm（复数 matmul，直调/注册两种接入都跑过）。

## 6. 前置验证清单与结果

| # | 验什么 | 结论 |
|---|---|---|
| **V1** | `dsh headless --json` 能否跑通并吐机器可读事件 | ✅ **通过**（§6.1） |
| **V2** | Python SDK 能否免 Node 起 | ✅ **通过**（§6.1） |
| **V3** | CANNBot skills 装到 `.dsh/skills/` 后 dsh 认不认 | ✅ **通过**（§6.1） |
| **V4** | `dsh web` 反代挂载 | 🟡 部分通过（差 WebSocket 过反代，部署时验） |
| **V5** | 裸金属"镜像 → 可跑环境"链路 | 🟡 只读部分完成（凭据/镜像盘点完成；容器预检按你意见跳过） |
| **V6** | 多用户隔离 | ⏸ 未跑（仅在确定要多人时才需要；见 D4） |

### 6.1 实测结果（2026-10-14，本机 macOS，dsh `0.2.0-rc.2`）

**环境**：`npm i -g @deepseek-ai/dsh` → `/Users/lfx/.local/bin/dsh`。
npm 11.19 新增的 `allow-scripts` 门禁会**默认跳过 dsh 的原生辅助包**（`dsh-subprocess-local`、`koffi`、`node-pty`），需显式放开：
`npm i -g @deepseek-ai/dsh --allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,@google/genai,protobufjs`。

**V1 ✅ 通过 —— P1 的集成地基**

`dsh --profile headless --json "<任务>"` 输出**真 JSONL 事件流**：

| 事件 | 字段 | 用途 |
|---|---|---|
| `session` | `sessionId`、`cwd` | run 锚点（写进 `runs.dsh_session_id`） |
| `status` | `phase`: `turn_start`/`step_start`/`step_end`/`turn_end`；`turn`、`step` | 进度；`step_end` 带 **`usage`**（input/output/cacheRead/cacheWrite/total） |
| `thinking` | `text` | 思考（可折叠） |
| `tool_call` | `callId`、`tool`、`input` | 工具调用（含 `bash` 命令原文） |
| `tool_result` | `callId`、`status`、`result` | 工具结果 |
| `text` / `final` | `text` | 正文 / 最终回答 |
| `turn_end.reason` | `{kind:"completed"}` 或 `{kind:"error", error:{message,code,status}}` | **结构化成功失败判定** |

- 实测任务：数 `.dsh/skills` 目录数 → 退出码 0、答出 **40**（并指出是指向 skills 仓的符号链接）。两轮共 **13,734 in / 240 out**，第二轮缓存命中 11,008 → 成本可忽略。
- **续会话可用**：`--session-id <id> "刚才那个数字是多少"` → 答 `40`，`cacheReadTokens:13,696` → 历史被恢复。
- **会话落盘**：`$DSH_HOME/sessions/<工作区路径转义名>/` → **按工作区分桶**。
- **端点坑**：`dsh-llm-deepseek` 走 **Messages 风格**，默认端点 `https://api.deepseek.com/anthropic`；手设 `DEEPSEEK_BASE_URL=https://api.deepseek.com` 会 `HTTP_404`。**只注入 `DEEPSEEK_API_KEY` 即可**；接 OpenAI 兼容网关才用 `dsh-llm-pi-ai`。

**V2 ✅ 通过 —— SDK 路线可用且免系统 Node**

- `deepseek-harness-sdk 0.1.5rc1` + `deepseek-harness-runtime-bin 0.1.5rc1`（72 MiB 原生运行时）；venv 自带 `dsh`（`0.1.5-rc.1`）。
- 用 `env -i PATH=/usr/bin:/bin`（`which('node')` = `None`）跑通 `DeepSeekHarness(...).run()`。
- profile 是**生成在 home 里的 pnpm workspace**（`cordis.yml`/`cordis.patch.yml`/`package.json`/`pnpm-workspace.yaml`/`node_modules`）→ 插件安装走 pnpm。
- **版本差异**：SDK `0.1.5-rc.1` vs CLI `0.2.0-rc.2` → v1 统一走 CLI。

**V3 ✅ 通过（关键结论）**

- cannbot 装到 `.dsh/skills/`（+ `.dsh/agents/`）：**40 个 skill 目录 + 3 个 agent 定义** + 依赖仓（`cann-samples`/`ops-tensor`/`cann-bench` → `.cannbot/dependencies/`）+ 工作区根 `AGENTS.md`。
- dsh 原生 skill 体系：`dsh-skill` + **`dsh-skill-filesystem`** + `dsh-tool-skill` + `dsh-client-ui-skill`。
- **扫描根 rank 100 = `<projectRoot>/.dsh/skills`** —— 与 cannbot 落点**正好重合**；格式 `<name>/SKILL.md` + frontmatter `name`/`description`，**兼容**。
- `projectRoot` = 含 `.git` 的最近祖先，无则取 cwd。

**踩到的两个坑**

1. **macOS 上 cannbot 源码安装路径直接崩**：`source-plugin-init.sh:71: SKILLS_ARGS[@]: unbound variable`（系统 bash 3.2.57 + `set -u` + 空数组）。绕过：`CANNBOT_SKIP_INIT=1 node <repo>/script/bin/cannbot.js install <plugin> --tool dsh --target <ws> --source <repo> --plugin-dir <repo>/plugins-official/<plugin> --skills <repo>/vendor/cannbot-skills`（根治：用 bash 5）。
2. **安装器会改自己缓存里的文件**（`~/.cache/cannbot/sources/checkout-*/`），二次安装报脏子模块 → 用 `--skills <路径>` 绕。**安装要留痕**（pin commit：主仓 `1f9b84f` / skills `81d2bdb`）。

**V4 🟡 部分通过**

- 无 token 访问 `/` 与 `/api/` 均 **401**；`/?token=<43 字符>` → **303 `location: ./`** + `set-cookie: dsh-auth-<实例哈希>=…; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000` → 跟随重定向 **200**（34.8 KB SPA）。
- **资源全相对路径** → 子路径挂载可行。
- `--public-url` **在 0.2.0-rc.2 不存在**（文档超前）；`--trusted-host` 可用。
- 未验：WebSocket 过反代、iframe。

**V5 🟡 只读部分完成**

- tk36 = `bms-fae`，**8×910B1**，`npu-smi 26.0.rc1`，HBM 65536 MB/卡，近空载。
- 相关镜像：`claude-ascend-agents:20260916-aarch64`（11.9GB，待弄清是什么）、**`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12`（12.2GB，最自然的算子开发镜像底座）**、`cann-9.2.0-py311-openruler-a3:v1`（a3 机型，与 910B1 不匹配）、`asi_bench_evalscope:v1`、以及 **4 个 `<none>` 悬空镜像 ≈81.8 GB 可回收**。

## 7. 已知风险与对策

| 风险 | 对策 |
|---|---|
| DSH 处于**开发者预览期**，承诺破坏性变更 `[已核验]` | 集成层薄：只依赖契约面（`headless --json` 事件、sessions JSONL、web 反代约定） |
| DSH 默认 `danger-full-access` + 持久 shell，会直接改工作区 `[已核验]` | **每 run 一个一次性工作区**（git worktree/复制）；MVP 串行 1；远端跑在容器里 |
| 代码在"控制台机 ↔ 裸金属"之间同步 | 每 run 独立远端目录 + `rsync` 增量 + 跑完清理（保留 N 天）；产物回读后算 sha256 |
| 模型 key 落点 | **只落在控制台+dsh 那台机**（600 文件/环境注入），裸金属不放模型 key；页面只掩码 |
| `session-log-deepseek` 默认把未接受的日志事件上传 DeepSeek（≤8 MiB/请求）`[已核验]` | 每个 `dsh_home` 的 profile 都 `enabled: false` |
| 许可叠加：DSH 是 MIT，CANNBot/skills 是 **CANN OSL 2.0**（仅限面向华为 AI 处理器的用途、可撤销、须保留声明）`[已核验]` | 内部工具无碍；对外开源时这条限制会跟着发布走 |
| skills 上游迭代快（cannbot npm 月级 20 个版本） | pin commit + **手动 bump**：记录"旧 commit → 新 commit + 变更摘要"，升一次跑一次冒烟，不自动跟 |
| 镜像能起 ≠ 环境能跑算子 | `env_recipe` 是必填；`preflight_cmd` 作为可选手动验证 |
| 时间口径 | 库里**统一 UTC**，界面本地化显示 |
| 远端目录/容器堆积 | 每 run 一个临时目录 + 跑完清理；容器默认 `--rm` |
| 悬空镜像吃盘（实测 81.8 GB） | 镜像页标悬空 + 给清理建议（不自动删） |

## 8. 决策状态

### 8.1 已定

| 项 | 结论 |
|---|---|
| 技术栈 | Python 3.12 + FastAPI + SQLAlchemy/Alembic；React 19 + Vite + TS + Tailwind v4 + shadcn/ui + ECharts（§2） |
| **部署形态** | **(ii) 控制台 + dsh 在独立机器；裸金属只做编译/运行** → 执行委托层（§3.2） |
| **工作区** | 在控制台机（本机）；每 run 一个**一次性工作区** |
| **镜像职责** | 纯"编译/运行环境"（CANN + torch_npu），**dsh 不进镜像** |
| **对话页** | 反代挂载 `dsh web`（资源相对路径已验证可行） |
| 阶段与门禁 | 四阶段（需求与设计/开发/验收/**交付**）；门禁四项；**硬拦** |
| 判据口径 | 默认官方 `ops-precision-standard`，**支持自定义档** |
| 异常输入口径 | 默认**错误码/信息与参考实现一致**才算过，可降级为"不崩即算过" |
| 性能采集 | **只到算子层**（不采模型层），**要基线/极限对比** |
| 运行时接法 | 子进程 `dsh headless --json`；**v1 统一 CLI `0.2.0-rc.2`**；SDK 留 P6 |
| **执行模型** | **本地 agent 改代码 → 显式「上机测试」**（`scripts/nm_test.sh`：rsync + 容器 + 回读 + 台账；配置 `ops/nm.conf`）。**不做命令级封装/白名单** —— 留痕靠 `dsh` 的 `tool_call` 事件与 run 目录存档 |
| skills 安装 | 统一安装器直调 + **pin commit**，手动 bump |
| 每项目独立 `dsh_home` | 是 |
| 密钥 | 引用名 + 环境注入/600 文件，页面掩码；裸金属不放模型 key |
| 事件存储 | **不全量落库**，只落索引与摘要；明细留 sessions JSONL |
| 时区 | 库内 UTC |
| 设计方向 | 皮肤 **B 深色玻璃 + 亮**，可切暗 |
| 页面 IA | 七页；样机已出，内容先冻结 |

### 8.2 待定（均为不阻塞项）

| # | 待定 | 备注 |
|---|---|---|
| D2 | 费用单价口径：DeepSeek 价目 + `cacheRead` 是否单独计价；按 run 还是按 task 汇总 | P4 前定 |
| D3 | 保留策略：sessions JSONL / 产物 / 日志各留多久；磁盘治理 | P3 前定 |
| D4 | **控制台+dsh 具体部署在哪台机**；v1 是否要登录/多人（决定 SQLite→PostgreSQL 时机与 V6 验不验） | **唯一有技术后果的一项**，P1 前定 |
| D5 | 与既有资产的关系：把 `mm-operator` / 202608 交付件导入做基线，还是一切从零跑 | P2 前定 |
| D6 | MVP 纵切算子确认（默认 complex64 mm） | P1 前定 |
| D7 | 队列与重试：默认串行 1、超时可配、失败不自动重试 | 已按默认实现，要改再说 |
| D1 | 「算子开发镜像」是否恢复"预检通过才算可用"的门禁 | 当前＝只勾选 + 记配方 |

## 附：与既有材料的对应关系

- 需求拆解 C1–C4 → `REQUIREMENT.md`；C3/C4 的落点 = P2 结果视图 + 任务详情页。
- 现成能力盘点 → `prior-art.md`；本方案 §1 的处置逐条对应该文档的覆盖度结论。
- CANNBot / DSH / CANN OSL 的事实与取证方式 → 技能库 `ascend-operator-research` 的 `references/cann-agent-ecosystem.md`；
  build/fork/reimplement 判据 → 技能库 `build-vs-adopt-review`。
- 设计样机 → `docs/mockups/console-ia-v1.html`（三皮肤对比）、`docs/mockups/console-pages-v1.html`（七页，B+亮）。
