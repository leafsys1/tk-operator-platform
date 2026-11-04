# 方案：算子开发智能体 Web 控制台（op-agent console）

状态：**v1.2 · 落版（2026-10-20）** ｜ 初版 2026-10-14（v1.1 合并稿） ｜ 2026-10-15 复查修正见 §9.3 ｜ **2026-10-15 需求调整已并入正文（见 §13 变更记录）** ｜ 前置文档：`REQUIREMENT.md`、`prior-art.md`
> 文件名保留 `PLAN-web-console-v1.1.md` 只为**路径稳定**（多处引用）；版本以本行为准。
> **2026-10-23 任务生命周期 v2 设计（文档输入 / 用例确认 / 大·小版本 / 交付）见 `docs/plan-task-lifecycle-v2.md`，与本文冲突处以该文为准。**
本文件 = 原 **v1.0**（`PLAN-web-console.md`，保留为决策溯源记录）+ 外部建议稿 **《算子开发智能体优化方案 v2.0》**（逐字归档于 `docs/proposals/`，sha256 `b6c44aa0…`）的合并，并已并入 2026-10-15 需求调整（A1–A13，2026-10-20 逐条确认）。
**v1.1 → v1.2 一句话**：自动优化从「可选扩展」升为**必需能力**（不设加速比硬线），新增**案例库**与**验收硬指标**（10 个案例全部真机全链路），并把对比报告、预算口径、硬件画像等差额逐条写进正文。

**已定（2026-10-14）· D4**：控制台 + dsh 跑**本机（这台 Mac）**、**单人使用** ⇒ 存储保持 **SQLite**、不做登录/鉴权、**V6（多用户隔离）关闭**、P6 的 PostgreSQL 切换只在将来真要多人时才考虑。

> 证据级别沿用本仓约定：`[已核验]` = 我实际打开过源码或官方文档并引用原文；`[转述]` = 二手；
> `[推断]` = 我的判断；`[待实测]` = 必须真跑一次才知道。

---

## 0. 一句话定位

**给自己用的算子开发控制台**：管裸金属服务器与算子开发镜像 → 发起与编排算子开发任务 → 和 agent 对话看过程 → 看结果（精度 / 性能）→ 管消耗。
**agent 能力不外采、不自研内核**：harness 用 DSH（含它的对话 UI），领域能力用 CANNBot 的 skills。
**v1.1 补充**：控制台是**入口**，业务契约是**骨架**——每个阶段的输入/输出/通过条件写死，会话结果必须转成可验证的算子开发结果才算数。

## 1. 边界与范围

### 1.1 边界

| 项 | 处置 | 为什么 |
|---|---|---|
| agent 内核 / LLM loop | **不做** | DSH 是 MIT、web-first、插件化的成熟 harness；自研等于把成熟件换成自写版 `[已核验]` |
| CANNBot skills 内容 | **不重写** | 与 CANN 团队同步的领域知识资产（手册/基准集/判据标准/模板），上游月级迭代 `[已核验]` |
| **裸金属服务器 + 镜像管理** | **做** | 目标是 NPU 裸金属，与 PAI-EAS 无关。范围：登记服务器 → SSH 连通 → 拉镜像清单 → 标记若干为「**算子开发镜像**」。**不做跨机调度/排队编排** |
| **业务编排层（v1.1 新增）** | **做，但做薄** | 只做：规格版本、阶段推进、能力路由、结果校验、受控修复、任务恢复。**是一层状态机 + 校验器，不碰 LLM loop、不重写 skills** —— 所以不违反上一行 |
| **控制台与 dsh 的部署位置** | **本机**（D4 已定：这台 Mac；(ii) 分支） | 控制台与 dsh 跑**本机**、**裸金属只负责编译/运行算子**；代价是要做执行委托（§6），换来「共享裸金属上不放模型 key」 |
| **对话页** | **复用 DSH，不自研** | DSH 自带 Web UI（工作区 + 会话列表 + 对话 + 权限审批），资源全相对路径 → 可反代挂子路径 `[已核验·实测]` |
| 通用 LLM 观测（token/费用/上下文/轨迹） | **不自研** | `headless --json` 的 `step_end` **直接带 usage**（input/output/cacheRead/cacheWrite/total）`[已核验·实测]` |
| **算子结果视图（精度/性能）** | **做**（无人可替） | 现有工具都没有算子维度的精度/性能数据模型；`cannbot-insight` 只有一个「手工填报 CANN-Bench 总分」的口 `[已核验]` |
| **对话过程 vs 业务证据（v1.1 改）** | **分开存、建关联** | 原 v1.0 让 sessions JSONL 当全量真相源 —— 但「agent 说的话」与「容器里跑出来的数」不是一类证据，混在一起无法机判 |

### 1.2 首版范围（采建议稿 §1.2）

| 项 | 首版 |
|---|---|
| 硬件 | 先面向昇腾 NPU；具体芯片与软件环境在任务中声明 |
| 需求输入 | 自然语言 / 数学表达式 / 参考代码；统一转成结构化规格 |
| 实现路径 | 从已接入、已验证的 CANNBot 能力里选；**首版先跑通一条路径** |
| 代码产物 | 算子实现、调用接口、构建/运行脚本、使用示例 |
| 正确性 | 常规 / 边界 / 异常输入，与**独立**参考实现对比 |
| 性能 | 算子层基准、采集、瓶颈分析与建议 |
| 执行 | 控制台与 DSH 跑**本机（D4）**；裸金属容器只负责编译/测试/采集 |
| 使用方式 | **首版单人，任务并发默认 1；性能执行按设备互斥**（互斥有载体：远端 `flock`，见 §6） |
| **自动优化 = 必需能力** | 必须完成「优化前后同条件 A/B + 对比报告」；**验收不设加速比硬线**，提升不显著时如实报告（结论：`improved` / `no_significant_change` / `regressed` / `inconclusive`）。**自动 = agent 执行 + 工具判定；人保留触发、审批（定版/采纳）与回退权。** |

### 1.3 分层（采建议稿 §2，五层）

| 层 | 职责 | 实现 |
|---|---|---|
| 交互层 | 需求提交、规格查看、对话、进度、报告与产物下载 | Web 控制台 + DSH 对话入口 |
| **业务编排层** | 规格版本、阶段推进、能力路由、结果校验、受控修复、任务恢复 | 后端状态机 + 结构化契约（**薄**） |
| 智能体能力层 | 需求理解、方案设计、代码生成、错误分析、优化建议 | DSH + **按阶段加载**的 CANNBot skills |
| 执行层 | 工作区快照、远端同步、容器执行、产物回读、超时取消 | 执行适配器 + SSH/rsync + 容器 |
| 证据与存储层 | 规格、用例、代码版本、执行记录、原始数据、结构化报告 | DB 索引 + 文件产物 + 会话日志 |

**不要求按阶段拆多个智能体**：首版一个主控执行流程；要并行开发或独立审查时再拆。

## 2. 技术栈

| 层 | 选型 | 为什么是它 / v1.1 新增要求 |
|---|---|---|
| 后端 | **Python 3.12（uv）+ FastAPI + Pydantic v2 + Uvicorn** | 领域资产全在 Python（SSH 探针、容器/镜像操作、读 JSONL、子进程驱动 dsh）；**新增**：阶段编排、报告校验、执行适配器、恢复与失败分类 |
| ORM / 迁移 | **SQLAlchemy 2.0 + Alembic** | 实体有真关系（project→task→loop→run→artifact）；**新增**：版本关系、逐用例结果、指标与建议证据 |
| 存储 | **SQLite**（D4 已定：本机单人）→ PostgreSQL 只在将来真要多人时同层切换 | 换库不动业务代码 |
| 实时 | **SSE** | **新增**：事件带序号、可补读、断线重连、**最终状态以持久化记录核对**（不能只信内存事件） |
| 本机 agent 执行 | **子进程 `dsh headless --json`**，默认串行 | 与 `evalscope-perf` 同构；并发会互相抢资源污染指标 |
| 远端执行 | **wrapper 脚本 + `rsync` + `ssh docker run`** | ssh/docker 细节封在脚本里，**不交给模型自己拼参数** |
| 前端 | **React 19 + TS + Vite + Tailwind v4**；**手写小组件，不引 shadcn/ui** | `cannbot-insight` 同代栈 `[已核验]`（代码不可抄，CANN OSL）。2026-10-15 复查：实际只有 react/react-dom + tailwind、无 `components.json` → 按「能静态不上复杂架构」保持手写 |
| 表格 / 数据 | **原生 fetch + 手写表格**（重表格真出现时再上 TanStack Table/Query） | 2026-10-15 复查：现有页面与样机都是手写小组件；引库的收益要等用例表（P3）落地才体现 |
| 图表 | **ECharts（P2 引入）** | 精度误差分布、性能对比（含基线/极限）、Roofline、时间线；**新增**：指标来源与缺失状态要能显示 |
| agent 运行时 | **DSH（MIT）**，profile + YAML patch；对话 UI 用 `dsh --profile web`；**v1 统一走 CLI `0.2.0-rc.2`** | SDK 自带 `0.1.5-rc.1`，两套版本别混（B5）；**新增**：独立记录调用与业务结果 |
| 领域能力 | CANNBot skills/agents 装到**任务工作区** `.dsh/{skills,agents}`：`scripts/install_skills.sh` + `ops/dsh/skills.lock`（commit + **逐文件 sha256**） | **v1.1 提前**：从「P5 才接」提到**首条流程就用**。2026-10-15 复查：光 pin commit 不够（vendored skills 是 dirty 的），且**不能**装在会被清理的缓存/scratch 目录（原装法就在 `~/.hermes/cache/scratch`，空闲 24h 即被清） |
| 密钥 | 只存引用名 + 环境注入 / 600 文件，页面不明文回显 | 模型 key 只落在「控制台+dsh」那台机 |
| 时区 | 库里统一 **UTC**，界面本地化 | 避免 UTC/CST 混用（压测那边踩过） |

**不选的两条**：① 全 Node（Next.js + Prisma）全栈——探针/执行/真机脚本都在 Python；② Jinja2 服务端渲染——做不出行内筛选、拖拽推进、实时事件流。

## 3. 数据模型（唯一不可逆的部分）

### 3.1 目标形态（v1.1：Project → Task → **Loop** → Run → Artifact）

```
Project（工作区 + 默认执行机/镜像 + 运行时配置）
 └─ Task（一个算子开发需求：目标、当前规格版本、阶段、当前认知、下一步）
     ├─ SpecVersion（原始输入、结构化规格、假设、变更原因、哈希）
     ├─ Loop（一轮设计/修复/诊断/优化方向：范围、假设、判定条件、结论）
     │   ├─ AgentJob（一次智能体调用：会话、输入摘要、产物、状态、usage）
     │   └─ Run（一次真实构建/测试/基准/采集）
     └─ Artifact / Recommendation
```

| 对象 | 关键字段 |
|---|---|
| `projects` | name、workspace_path（本机持续开发工作区，= dsh 的 cwd/projectRoot）、**default_host_id / default_image_id**（默认配置，不构成项目身份）、dsh_profile、dsh_home、**criteria_ref**（判据档默认值）、created_at |
| `tasks` | project_id、title、op_name、target_chip、input_form、**current_spec_version_id**、stage、**criteria_ref**（空 = 用项目默认）、**next_action**、created_at；创建期字段：**is_draft**、**raw_input**、host_id/image_id/device_id/timeout_s（空 = 用项目默认）、**fix_rounds_max** / **budget_per_action**（D8）、**spec_draft** / **chat_log** / **dsh_session_id**（P0-3 对话）。v1.1 起**不再有** `spec_json`：规格唯一真相源是 `spec_versions` |
| `spec_versions` | task_id、raw_input、**spec_json**（含 `target.hardware_profile` 引用硬件画像）、assumptions、unknowns、change_reason、sha256、created_at |
| `loops` | task_id、kind（design/fix/perf）、scope、hypothesis、judge_condition、conclusion、status、**optimize_json**（agent 自动改代码的证据：前后快照/逐文件 diff/回退数据）、created_at |
| **`agent_jobs`**（v1.1 新增，原 v1.0「runs」的一半） | task_id、loop_id、prompt、`dsh_session_id`、`dsh_jsonl_path`、status、**is_demo**（P0 示例行：usage/cost 是假的，**聚合必须排除**）、exit_code、**usage**（tokens_in/out/cache_read/cache_write）、started_at、ended_at |
| **`runs`**（v1.1 改义：真实执行） | task_id、loop_id、`agent_job_id`、**action**（build/test/benchmark/profile）、spec_version_id、code_snapshot_sha256、**suite_ref**（用例集内容哈希 ref；原 `suite_version_id` 是指向**不存在的表**的半截外键，2026-10-15 复查改掉，P3 真上用例集表时再迁）、criteria_ref、host_id、device_id、image_digest、env_recipe_snapshot、command、cwd、**remote_workdir**、started_at、ended_at、exit_code、manifest_path、sync_state、**failure_kind**、**note** |
| **`action_jobs`**（2026-10-20 新增） | task_id、loop_id、action、status、**run_ref**（= `runs.remote_workdir`）、run_id、device_id、command、log（尾部）、exit_code、error、**gate_verdict**、started_at、ended_at —— 「点一下就真执行」的作业行，不承载业务结果 |
| `artifacts` | run_id、type、path、sha256、size、created_at |
| `case_results` / `metrics` | **首版用 JSON 文件承载**（`reports/*.json`），库里只存索引 + 状态枚举 + 值摘要（最小集） |
| `recommendations` | **首版 JSON 字段**：evidence→hypothesis→confidence→change→validation→status |
| `stage_events` | task_id、from_stage、to_stage、actor、reason、created_at |
| `conversations` | project_id、dsh_session_id、title、last_active_at、message_count、tokens、cost |
| `hosts` / `host_images` | cards、cann_version、docker_ok、npu_state；镜像的 `is_op_dev`、`env_recipe`、`preflight_cmd`、digest |
| `endpoints` | name、provider（`dsh-llm-deepseek` / `dsh-llm-pi-ai`）、base_url、model、key_ref、probe_state、max_model_len |

**载体（2026-10-20 更新）**：`projects`·`tasks`·`spec_versions`·`loops`·`agent_jobs`·`runs`·`artifacts`·`stage_events`·`hosts`·`host_images`·`endpoints`·`action_jobs` 已是**真表** ✅；`case_results`·`metrics`·`recommendations` 首版用 **JSON 文件**承载；`conversations`（P0-2）**已建表**（迁移 `d4e5f6a7b8c9`，2026-10-20）。

**案例库不建表（A10）**：案例库 = 工作区 `cases/<op>/`（规格/用例/基准/参考实现）+ **`cases/index.json` 索引清单**；页面读索引 + 目录。对比结果 = `performance-diff.json` + `runs` 成对关联。若将来需要跨项目检索/权限，再评估建表（当前不做）。

**三条硬规则（v1.1）**

1. **业务通过状态不由会话退出码决定**：`agent_jobs.exit_code == 0` 只说明 agent 跑完了；验收看 `runs` + 报告校验结果。
2. **一轮 agent 调用可以产生多个真实执行 Run**；批量测试可归一个 Run，但逐用例结果要有稳定 Case ID。
3. **版本关联必须留**：代码快照哈希、规格版本、测试集版本、判据版本、镜像 digest、环境配方快照 —— 这是「可复现」的最低要求。

### 3.2 最小集与迁移（落地口径）

- **最小集**：`agent_jobs` / `loops` / `spec_versions` 建真表；`case_results` / `metrics` / `recommendations` **先用 JSON 字段 + 报告文件**（建议稿 §8.1 明确允许），跑顺了再拆表。
- **迁移**：现库里 `runs`（P1 期存的其实是 agent 调用）→ 改名 `agent_jobs`；新建 `runs` 供真实执行。单人自用，测试数据可直接重建，不做数据搬运。
- **现有代码影响面**：`agent_runner.py` 的落库目标从 `runs` 改到 `agent_jobs`；`/api/tasks/{id}/agent-runs` 语义不变（它本来就是发起 agent 调用）。

### 3.3 实时与恢复

- SSE 事件**带序号**；断线可从我方落盘的 `reports/events/*.jsonl` 补读。
- **服务重启后核对执行状态**：现状是启动钩子把 `running` 一律标 `interrupted`（安全但粗）→ 目标是查远端容器/run 目录状态再判定（P1.5 起逐步）。
- 任务恢复优先读**紧凑摘要**：目标、当前规格、活动 Loop、保护用例、按用例基线、已接受结论、最近失败原因、下一步动作；完整历史按需走证据索引。

## 4. 业务流程契约（建议稿 §3–§7 的最小集）

### 4.1 主流程

```
输入（自然语言/数学式/参考代码）
 → ① 需求解析 →（关键语义不明确则向用户澄清）→ ② 用例规划 + ③ 方案设计
 → ④ 代码生成（实现/接口/构建脚本） + 参考实现与测试脚本
 → ⑤ 目标环境编译运行 → ⑥ 正确性验证 →（不过则⑦受控修复回④）
 → ⑧ 建立基线 + 性能采集 → ⑨ 分析与建议 → 交付
```

### 4.2 各步的输入 / 必需输出 / 通过条件

| 步骤 | 必需输出 | 通过条件 |
|---|---|---|
| 需求解析 | 结构化规格、待澄清项、显式假设 | 影响语义与实现的关键问题已解决 |
| 用例规划 | 常规/边界/异常用例矩阵 | 覆盖要求有对应用例或明确范围说明 |
| 方案设计 | 实现路径、算法、分块与存储设计 | 符合规格，环境支持所选路径 |
| 代码生成 | kernel、调用接口、构建/运行脚本 | 产物可被**统一执行入口**调用 |
| 构建验证 | 构建日志、构建结果、实现专项检查 | 构建通过且专项检查通过 |
| 正确性验证 | 逐用例结果 + 正确性报告 | 所有必需用例符合任务判据 |
| 性能测量 | 基准样本、原始数据、指标 | 测量范围明确、数据有效、环境可追溯 |
| 分析与建议 | 瓶颈假设、建议、适用范围与验证方式 | 建议引用证据，区分观测/推断/待验证 |
| 交付 | 交付包、接口说明、复现步骤 | 交付版本与验收版本一致，产物齐备 |

### 4.3 规格 schema（最小字段）

来源（原始描述/输入方式/参考代码）｜语义（算子名、数学定义、参数、属性、广播与归约规则）｜输入输出（数量、名称、shape、dtype、layout、stride/连续性、输出推导）｜支持范围（动态维度、合法参数集、不支持的组合）｜特殊行为（空输入、NaN/Inf、非法 shape/dtype 行为）｜环境（目标芯片、框架、编译运行要求）｜正确性（参考实现、容差策略、特殊值比较、异常预期）｜性能（测量范围、代表用例、基线、可选目标）｜交付（调用形式、接口约束、产物、复现要求）｜版本（版本号、假设、未确定项、变更记录）。

- 缺失事实标 `UNKNOWN`；**关键语义/支持范围/硬件约束未定时，不进入依赖它的代码生成**。
- 可用合理默认值的事项**明确记录**即可，不要求用户逐项确认所有工程参数。

### 4.4 失败分类与受控修复

- **失败分类**（至少）：规格不明确、生成产物不完整、构建失败、正确性失败、测量无效、环境失败、超时。
- 构建/正确性失败 → 主控提取**错误摘要 + 失败用例 + 最小证据** → 要求**有边界的修复** → 生成新代码版本与新执行记录；**历史失败记录保留**。
- **修复轮次与总预算可配**（D8）；达上限输出：失败原因 + 已尝试方向 + 下一步建议。
- **基础设施失败不默认自动重试**，不把环境问题当算子实现错误。
- 需求实质变化 → 生成**新规格版本**，并重新判断既有测试证据是否仍有效。

### 4.5 用例矩阵与报告

| 用例组 | 覆盖 |
|---|---|
| 常规 | 典型 shape、支持的 dtype/属性组合、代表性数据分布 |
| 边界 | 最小/最大合法尺寸、非对齐尺寸、尾块、特殊布局；空输入按规格处理 |
| 数值边界 | 零值、极值、正负混合、适用的 NaN/Inf |
| 异常 | 非法 shape、类型不支持、参数越界、约束不满足 |
| 回归 | 已发现问题的最小复现用例、交付必须保护的用例 |

- 每用例记 shape/dtype/layout/参数/输入生成方式/**随机种子**/分组/预期行为；不同 shape/dtype/布局**不能只按元素总数合并覆盖**。
- 报告状态枚举：`PASS` / `FAIL` / `ERROR` / `INVALID` / `NOT_RUN`；合法跳过要记理由，**必需用例未执行不计通过**。
- 通过条件：所有必需用例**实际执行且达标**，且报告来源与**当前候选版本一致**；报告缺失、解析失败、证据版本不匹配 → 不满足门禁。
- 验证项含「**实际执行的是否为候选算子路径，有无未声明的回退**」。

### 4.6 性能采集与分析

- 只采**算子层**；沿用 `msopprof` / `msprof`（**命令/格式/指标支持已实测固化**：`msopprof --application=<exe|python app> --aic-metrics=Default ...`，读 `OpBasicInfo.csv` 的 `Task Duration(us)` 与各能力组 CSV；见 `workspaces/cann-ops/tools/run_profile.py` 与 `docs/PROGRESS.md`「M1-b 实测记录」）。
- 指标四类：执行耗时 / 数据搬运 / 存储使用 / 计算单元负载。**每项标记 `measured` / `estimated` / `unsupported` / `failed` 并记来源**；估算值附假设，**不得当实测值展示**；必需指标失败 → 任务标「性能分析未完成」。
- **基准测试与 Profiling 分开执行**，不混用不同测量条件的耗时。
- **优化前后对比**：`loops.kind=perf` 的一轮里，优化前/后各跑一次 `test` + `benchmark`（同机、同卡、同镜像 digest、同用例集与测量条件）。由 `GET /api/loops/{id}/perf-diff` 产出 **`performance-diff.json`**：逐用例前后耗时、精度不退化、配对与可比性检查、样本数与不确定性、结论枚举（`improved` / `no_significant_change` / `regressed` / `inconclusive`）。**不设加速比硬线**；提升不显著或回退都如实记录。精度退化是**硬失败**（`verdict=fail`），但报告仍落盘、结论仍记录。四类指标的**成对对比**属后续增强（当前对比以 benchmark 耗时 + 正确性为准）。
- 基线按算子与用例选（官方对应实现 / 既有可信实现 / 上一版已验证实现）；**v1.0 的 `aclnnMatmul` 基线只适用于语义、dtype、支持范围匹配的任务**。
- 性能记录固定条件：用例与输入、代码版本、硬件/设备、软件栈、**镜像 digest**、预热次数、重复次数、同步方式、测量范围、聚合方法；保存原始样本与波动。**提升需同条件对比且超过测量不确定性**，否则输出「无法确认提升」；理论上下界要说明模型、假设与适用范围。
- 分析按**观测 → 竞争假设 → 建议 → 验证**组织（低计算负载 ≠ 确定瓶颈成因）。建议结构化：

```yaml
recommendation:
  id: R1
  affected_cases: [case_ids]
  observed_evidence: [metric_ids, artifact_paths]
  bottleneck_hypothesis: …      # 可能的机制
  confidence: low|medium|high
  proposed_change: …            # 具体修改方向
  expected_observation: …       # 改完应出现的变化
  constraints_and_risks: …
  validation_method: …          # 同条件基准 + 必要时再采集
  status: proposed|validated|rejected|inconclusive
```

## 5. 阶段与门禁

外部仍是**四阶段**（需求与设计 → 开发 → 验收 → 交付）；内部用 Loop / Run 管细分动作。

| 阶段 | 门禁（v1.1：**结果有效 + 业务达标 + 版本一致 + 产物完整**） | 主要产物 |
|---|---|---|
| 一 · 需求与设计 | 关键语义明确；规格、用例规划、环境要求、实现路径完整 | `spec.json`、用例矩阵、设计 |
| 二 · 开发 | **当前版本真实编译通过**；代码/接口/脚本完整；专项检查通过 | 源码、接口、构建日志、检查结果 |
| 三 · 验收 | 必需正确性用例通过；有效性能报告与证据引用齐备；瓶颈分析与建议完成；**若本任务有 `perf` Loop，须有一份可判的 `performance-diff.json`**（同条件配对 + 精度不退化 + 样本数与不确定性有记录 + 来源齐；结论可为 `improved` / `no_significant_change` / `regressed`） | 正确性报告、性能数据、分析与建议、`performance-diff.json` |
| 四 · 交付 | 交付代码与验收版本一致；文档、脚本、复现材料齐备 | 交付包、接口文档、复现说明 |

- **硬拦保留**，但**不再只看"产物齐备"**：缺项 / 失败 / 证据失效都要展示具体原因（v1.0→v1.1 的关键升级）。
- 判定由**门禁校验器 v1** 落地：`GET /api/runs/{id}/gate`，输出 `gate_report`（fail code 清单与三个契约文件见 **`docs/gate-report-v1.md`**）。口径 = 产物完整 + 结果有效 + 业务达标 + 版本一致；**退出码不是结论**。
- 每次推进/打回写 `stage_events`；「验收」≠「测试管理」（前者看门禁，后者看用例集与执行记录）。
- **任务另有明确性能目标时**验收才检查目标；没有目标时以「测量与分析完成」为准，**不擅自加"必须更快"**。

## 6. 执行模型（上机测试 → 统一执行入口）

**工作流不变**：本地 agent 改代码 → 改到满意 → 上机测一次。

- **持续开发工作区**（v1.1 改，原为「每 run 一次性」）：任务维护一个长期工作区；**每次真实执行生成确定的源码快照**，同步到远端独立目录执行并回读。
- **统一执行入口四动作**：`build` / `test` / `benchmark` / `profile`。每个动作：① 检查任务配置与本动作的必要环境条件 → ② 固定源码/规格/测试集/环境配方版本 → ③ 同步快照到远端 Run 目录 → ④ 执行 → ⑤ 回读日志/原始数据/结构化结果 → ⑥ 生成 manifest、登记 Run、调报告校验与门禁判定。**`dry_run=false` 已接通（2026-10-20）**：后端起异步 `action_jobs` 作业（`services/action_runner.py`），脚本登记 run 后按 `run_ref` 找回并回填门禁 verdict。
- **对比报告不新增第 5 动作**：优化前/后各跑一次 `test` + `benchmark`，由 `loops.kind=perf` 成对关联（按 `code_snapshot_sha256` 配对），`GET /api/loops/{id}/perf-diff` 产出 `performance-diff.json`。**不引入 `benchmark --baseline-run`**：配对依据是代码快照与轮次，不是命令行自报。
- 实现上**沿用 `scripts/nm_test.sh` 路线**作为适配器后端，但**参数与返回结果要形成明确契约**；主机/镜像/设备/配方**从任务配置注入**（不靠手工复制，否则必漏登记）。
- **取消/超时必须传到远端容器/进程**，并记录清理结果 —— 不能只杀本地会话而远端还在跑。**脚本侧已具备**（`nm_test.sh`：`timeout --kill-after` 包住容器执行 + 显式 `docker rm -f` 兜底，P1.5-a 实测泄漏过两次才加的）；**后端侧：本地终止已实装**（2026-10-20，`services/action_runner.cancel()` 杀本地进程组并标 `interrupted`），**远端清理也已补**（`services/remote_cancel.py` → `docker rm -f opx-<run_ref>`，run / action job 取消都走它）。远端目录按保留策略清理，**产物回读校验成功后才删证据**。
- **设备互斥（共享机硬要求，2026-10-15 补落地）**：tk36 是共享机（别人的容器在跑）。① 前置检查读 `npu-smi info -t proc-mem` 判每张卡有没有进程（**HBM 占用不能当判据**：8 张卡都有 ~3.4 GB 常驻但无进程），选一张空卡写进 plan；② 执行侧在远端用 `flock -n $OPX_REMOTE_ROOT/.locks/dev<N>.lock` 拿独占锁，拿不到就 **exit 3（env_failed）不跑**，而不是硬挤（实测卡被占时容器起不来 `dcmi -8020`；挤上去采到的耗时带邻居，“同条件对比”失效）；③ 设备与锁模式写进 `manifest`（`device`/`lock_mode`/`lock_acquired`）与 `runs.device_id`，门禁对非独占的基准/采集给 `W_DEVICE_SHARED`。
- **命令不由控制台手拼 docker run**：配方（privileged / `ASCEND_RT_VISIBLE_DEVICES` / 设备挂载 / set_env 链 / 超时清理）在 `scripts/nm_test.sh` 里固化，plan 只注入任务侧参数（镜像、设备、任务、规格版本、判据档），`plan.command` 就是一次可复制的 `nm_test.sh` 调用。
- 不做命令级白名单/封装框架：个人专用机上先靠约定（项目 `AGENTS.md`）；留痕靠 `tool_call` 事件 + run 目录存档 + `docker_cmd.txt`。
- 镜像登记继续「标记 + 配方」，并**按动作**判可用性：`torch_version` 为空的镜像跑 test/benchmark/profile 直接拦（P1.5-a 实测：只有 CANN 工具链的镜像 `VERDICT torch_npu: FAIL`）；`is_op_dev` 只表示「可以拿来做算子开发」，不表示「能跑算子」。**镜像 digest 必须回填**（`backend/scripts/sync_image_meta.py`：`docker images --digests` + 本地构建镜像用 image ID 兜底），否则「两次跑的是不是同一镜像」判不了。

## 7. 页面（IA）

| 页面 | 回答什么 | 优先级 |
|---|---|---|
| **任务详情（作业台）** | 这次 run 做了什么、结果是否可信、下一步做什么 | **MVP 核心** |
| **任务开始**（新建任务 · 对话式定义算子） | 这个算子到底要什么、跑在哪、精度怎么判 —— 并产出**任务上下文空间** | **P0-3**（跟随 P0-2；功能规划见 `docs/UI-task-start-v1.md`） |
| 任务列表 / 项目 | 有哪些任务、卡在哪 | MVP |
| **对话** | 和 agent 说话、看历史（任务会话默认挂 skills；轻对话不挂） | ✅ P0-2（自研 B，2026-10-20） |
| **服务器与镜像** | 在哪台机器、用哪个镜像跑 | P0（没有它跑不了） |
| 阶段看板 | 某算子卡在哪一阶段、门禁过没过 | P3 |
| 测试管理 | 用例集与执行记录、精度/性能是否达标 | P3 |
| **案例库** | 有哪些典型算子案例、覆盖哪些场景；按案例一键发起任务。**案例不预置进仓库**：独立 zip（外部 bundle）+ 导入脚本；页面先做说明与导入指引 | **M2（最后做；已做导入列表 + 一键发起，案例内容待做）** |
| API 管理 / 消耗情况 | 用哪个模型/key、通不通；花了多少、花在哪 | P4 |

**任务详情分区（采建议稿 §11）**：需求与规格（含假设/未确定项/版本变更）｜实现与接口（设计、diff、构建结果）｜测试与正确性（覆盖、逐例结果、误差、失败输入、复现命令）｜性能与建议（测量范围、基线、指标、证据、假设、建议）｜**优化前后对比**（假设/判定、前后快照与 run、逐用例耗时、精度不退化、结论）｜执行记录（Loop / AgentJob / Run 关联 + 命令 + 环境 + 消耗）｜交付（验收状态、产物清单、下载、复现说明）。

**边界**：① 阶段 ≠ 测试；② API 管理（配置态）≠ 消耗（运行态）；③ 对话 ≠ 任务台账（两个入口、一套数据）。

**设计**：皮肤 **B 深色玻璃 + 亮**（保留亮暗与设计变量体系）；样机 `docs/mockups/console-ia-v1.html`（三皮肤）、`console-pages-v1.html`（**八页**；案例库页在 M2 时补样机）。内容先冻结，落码后按实际使用再调。

## 8. 路线（P 命名；与建议稿 M 的映射见附录 A.5）

| 阶段 | 交付物 | 验收 |
|---|---|---|
| **P0-1 设计** ✅ | HTML 样机 + 设计 token | 已选 **B 深色玻璃 + 亮** |
| **P0 骨架 + 资源** ✅ | FastAPI + SQLAlchemy/Alembic + SPA 壳；资源页（服务器登记/探针/镜像清单/标记算子开发镜像）；任务详情渲染手工登记的 run | 已登记 tk36、列出镜像、标出算子开发镜像 |
| **P1 执行链** ✅ | `dsh headless --json` 子进程 + SSE 上屏 + 摘要落库 | ✅ 已实测（§9.2） |
| **P1.5-a 容器闭环** ✅ | 上机测试的"起容器"那步真跑（= 建议稿 M0 验收）：容器里看到 8 张卡、torch_npu 可用 | ✅ **已实测**（§9.2）：`npu devices 8` / `VERDICT npu: OK` / `VERDICT torch_npu: OK`，产物回读 + manifest |
| **P1.5-b 契约最小集** ✅ | 三张表迁移（`agent_jobs`/`loops`/`spec_versions`，现 `runs` 改名 `agent_jobs`，新建真实执行 `runs`）+ spec schema（elementwise_add 首版规格）+ 统一执行入口四动作契约（`build`/`test`/`benchmark`/`profile`，含取消/超时预留）+ 任务详情按"过程/结果"分列 | ✅ **已实测**（§9.2）：迁移后旧数据落 agent_jobs、plan 算出前置齐备、`dry_run=false` 当时明确拒绝（**2026-10-20 已接通为异步作业**） |
| **P1.5-c 门禁与环境** ✅ | 门禁校验器 v1（`docs/gate-report-v1.md` + `GET /api/runs/{id}/gate`）＋设备互斥（远端 flock + 前置检查）＋版本对齐（内容哈希快照 / 判据档 / 用例集 ref / 镜像 digest）＋控制台自测与一键起服务＋**CANNBot skills 落地进任务工作区（含 lock 指纹）** | ✅ **已落地**（§9.3）：迁移 `c9a4e2f7b6d1`、`pytest` 38 项通过、真实 run 跑通「取锁 → 容器 → manifest → 登记 → 门禁」 |
| **P2 结果视图** | 精度报告 + 性能数据渲染（ECharts），基线/极限对比；指标状态与来源可见；建议按结构化 schema 展示 | C3/C4 结果可读、可导出；**未支持指标明确标记，不按已采集验收**。✅ **完成（2026-10-16）**：正确性/性能/指标三类报告渲染 + **极限对比（Roofline 判定，达成带宽 vs HBM 峰值）** + 指标状态与来源 + **结构化建议（§4.6 R1）** + 导出 JSON |
| **P3 阶段 + 测试** | 状态机（门禁硬拦 = 校验器）+ 推进/打回；用例矩阵与执行记录；判据官方档/自定义档；报告状态枚举 | 阶段推进不靠人工记忆；**必需用例未跑通不算过**。🟡 **读侧完成（2026-10-19）**：阶段看板（按证据推导 + `stage/sync`）+ 测试管理（用例矩阵 / 逐例历史）；**写侧未做**（手动推进/打回、判据档管理） |
| **P4 API + 消耗** | API 登记与探针（掩码）、消耗聚合 | 两个薄页面。✅ **完成（2026-10-19；2026-10-20 支持页面配 key / 当前端点 / 掩码）** |
| **P5 能力路由** | **按阶段选择能力**（避免全量 skills 入上下文）+ 必要的自写 skill（安装与 pin 已随 P1.5-c 落地） | 用官方 skill 跑通一条真实算子任务。⬜ **未开工**（当前任务会话固定挂全量 40 skills，尚未按阶段选子集） |
| **M3 自动优化闭环** ✅ | 硬件画像 + `perf` Loop + 同条件 A/B + `performance-diff.json` + 门禁规则 + agent 自动改代码/回退 | **已完成**（2026-10-16 / 10-19 / 10-20 两轮真机 A/B） |
| **M2 案例库** 🟡 | **10 个**典型案例（矩阵 2 / 归一化 3 / 激活 3 / 注意力 2）+ 案例库页 + 一键发起 | **10 个全部真机全链路（硬性，2026-10-20 定）**；**交付形态 = 独立 zip bundle + 导入脚本**（仓库不预置案例，2026-10-20 定）；**算子清单可替换**（同场景内换算子，但保持 10 个 + 四场景覆盖）；已延后到最后做（2026-10-19）。**框架已完成（2026-10-20）**：导入脚本 + 列表/一键发起 + `CasePanel`；**案例内容 0/10**（剩余工作量在此） |
| **P6**（可选） | SDK 常驻替换子进程；`packages/ssh` 原生远端执行；SQLite → PostgreSQL | 多人/长任务时再考虑（不含自动优化 —— 已在 M3） |

**首条流程算子（D6）✅ 已定（2026-10-14）**：用**语义明确的简单算子** `elementwise_add`（常规/边界/异常全覆盖、参考实现无争议）；**`complex64 mm` 留作后续代表性验证任务**（它的接口与基线支持范围要先确认）。

## 9. 验证清单与结果

### 9.1 前置验证（v1.0 保留）

| # | 验什么 | 结论 |
|---|---|---|
| V1 | `dsh headless --json` 跑通并吐机器可读事件 | ✅ 通过 |
| V2 | Python SDK 能否免 Node 起 | ✅ 通过 |
| V3 | CANNBot skills 装到 `.dsh/skills/` 后 dsh 认不认 | ✅ 通过（40 skill + 3 agent）。**2026-10-15 复查补**：原装法落在 Hermes scratch 目录（空闲 24h 会被清）+ symlink 指向 `~/.cache/cannbot`（同样会被清）→ 已改成 `scripts/install_skills.sh` 装进**任务工作区**并物化成真文件，逐文件指纹写进 `ops/dsh/skills.lock` |
| V4 | `dsh web` 反代挂载 | ⬜ **已被 P0-2 取代（2026-10-20）**：对话页采用**自研（B）**，复用 `agent_runner`；不反代 `dsh web`，WebSocket 过反代不再是前置 |
| V5 | 裸金属「镜像 → 可跑环境」链路 | ✅ **完成**（只读探针 + 容器那步都跑了）：见 §9.2 的 P1.5-a；仍未验的是多机 / 多设备并发 |
| V6 | 多用户隔离 | ⏸ **关闭**（D4 已定：本机单人使用，不需要） |

### 9.2 已实测进展

- **P1.5-a（2026-10-14）容器闭环**：走正式管线 `scripts/nm_test.sh`，`VERDICT npu: OK` + `VERDICT torch_npu: OK`（`torch 2.10.0+cpu / torch_npu 2.10.0.post4 / npu devices 8`、`Ascend910B1`），exit 0、产物回读、manifest 带 sha256（run `r20261014-205455`）。
  - **配方（逐项实测出来的，已写进 `ops/nm.conf` 与镜像 #52 的 env_recipe）**：`--privileged` + `-e ASCEND_RT_VISIBLE_DEVICES=0..7` + 11 个设备节点 + `driver/dcmi/npu-smi/ascend_install.info` 挂载 + `--entrypoint bash` 绕开自带 ENTRYPOINT + 容器内 source 完整 CANN set_env 链。
  - **踩到的 6 条**：① 只挂 `davinci_manager/devmm_svm/hisi_hdc` → `dcmi -8005`；② 缺 `davinci0..7` 与 `/etc/ascend_install.info` 就看不到卡；③ 自带 CANN 的镜像靠挂载不够、要 `--privileged`；④ 不给 `ASCEND_RT_VISIBLE_DEVICES` 时 torch_npu 只认 1 张卡（而容器里 npu-smi 能看到 8 张）；⑤ 镜像自带 ENTRYPOINT 会 source set_env 卡死，且 `--entrypoint bash` 之后不能再传 `bash`（`bash bash -lc` → exit 126）；⑥ **`timeout` 只杀 `docker run` 客户端、容器会继续跑并占住卡** —— 必须显式 `docker rm -f` 兜底（我泄漏过两次，其中一次反过来导致后续 `dcmi -8020 device is used`）。
  - **镜像结论**：`lfx-op-dev-910b:torch210-cann910` 是能用那个（已标 is_op_dev + 配方）；`cann:9.2.0-beta.2-910b` 只适合做编译/工具链底座（npu-smi 正常但没有 torch/torch_npu）。
  - **自检脚本**：`smoke.sh` 的判定改成**按输出判定**（`npu-smi` 会「打印报错却返回 0」，只看退出码会误判成 OK）。

- **P1.5-b（2026-10-14）契约最小集**：
  - 迁移 `b7f2c9a4e1d3` 真跑：老 `runs` 5 行按语义搬进 `agent_jobs`（还在跑的东西确实是智能体调用）；新 `runs`（真实执行）建空；`artifacts` 按既定口径清掉（demo 数据可重建）；`tasks` 加 `current_spec_version_id`/`next_action`；`host_images` 加 `digest`。
  - seed 落 `elementwise_add` 首版规格（v1，sha256 `b29b1852…`）+ 一轮 design Loop + `next_action`，并**不造假 run**。
  - 四动作契约：`POST /api/tasks/{id}/actions` 算出 plan（前置检查 / 版本固定 / 同步 / 执行 / 回读 / manifest 六步），实测 `ready=true`（执行机 tk36、镜像 cann 9.2.0-beta.2、规格 v1、判据官方档、超时 1800s）；`dry_run=false` 返回 **501** 并说清卡在哪 —— 远端执行故意不接通。**（2026-10-20 已接通：起异步 `action_jobs` 作业，脚本登记 run 后回填门禁 verdict。）**
  - 前端按"过程（agent_jobs）/ 结果（runs）"分列，并加了四动作 plan 面板（浏览器实测点通）。
  - 回归：拆分后 `POST /api/tasks/2/agent-runs` → job#6 `ok`、session、usage in 1351 / out 195 / cache 4096；`runs` 保持 0（语义正确）。

- **P1（2026-10-14）**：控制台发起 agent run → dsh 子进程 → SSE 事件流实时上屏（thinking / tool_call / step 级 usage / final）→ 台账回填（`status=ok`、`exit=0`、session id、tokens in 5615 / out 146 / cache 5376）；失败态由启动钩子标 `interrupted`。
- **执行脚本**：`nm_test.sh` 的 `--dry-run` 与 `--sync-only` 真跑通（43 文件 / 464K 往返）；**容器那一步已完成**（见上条 P1.5-a 实测）。
- **模型接入**：dsh 经 `pi-ai` 适配器接 opencode go 网关已跑通并固化（`ops/dsh/`）。

### 9.3 复查修正（2026-10-15，P1.5-c）

一次对 v1.1 的复查（事实修正 + 缺失补齐 + 前后矛盾）落地，逐条证据：

- **F5 默认镜像与实测结论相反**：项目默认绑的是 `ascendhub/cann:9.2.0-beta.2-910b`（**不带 torch**，跑 test/benchmark 必失败），而脚本默认镜像早已切到能跑的那个。迁移 `c9a4e2f7b6d1` 把项目默认镜像切到 `lfx-op-dev-910b:torch210-cann910`、摘掉工具链镜像的 `is_op_dev`、回填 `torch_version`；前置检查新增「本动作需要 torch 而镜像没记版本 → 拦」。
- **F6 镜像 digest 回填**：`backend/scripts/sync_image_meta.py` 真跑：`docker images --digests` 命中 30 条 + 本地构建镜像用 image ID 兜底 64 条 → 库内 **94/98** 张镜像有 digest；`manifest.json` 从此必带 `image_digest`（门禁缺它就判 fail）。
- **F7/F8/F9 数据模型**：`runs.suite_version_id`（指向不存在的表）→ `suite_ref`；删 `tasks.spec_json`（规格唯一真相源 = `spec_versions`，迁移时把旧值搬过去）；`agent_jobs.is_demo` 标记 P0 假 usage 行（消耗聚合必须排除）。
- **F1/F2/F3/F4 文档与代码里的过期话**：脚本已加固（取消/超时）、501 的理由改成「四动作适配器未实装」（不再说容器没验过）、V5 改 ✅、"独立机器"措辞对齐 D4。
- **快照算法换成内容哈希**（原来 size+mtime：rsync/touch 就会被误判「版本不一致」，且有静默截断）→ `snapshot()` 返回 `{sha256, files, truncated}`，并与 `nm_test.sh` 里的实现**逐字节一致**（`backend/tests/test_snapshot.py` 直接抽脚本里的那段代码跑对比）。
- **判据档有载体了**：`projects.criteria_ref` / `tasks.criteria_ref`（`档名@档版本`），plan 按 任务 > 项目 > 默认 解析并标注来源；没写版本号会被门禁判 fail。
- **门禁校验器 v1 + 控制台自测**：`backend/app/services/gate.py`（fail code 见 `docs/gate-report-v1.md`）、`backend/tests/`（38 项：迁移往返、plan 缺项分支、fail code、快照一致性、501）、`scripts/dev.sh`（一键起前后端）。
- **CANNBot skills 落地**：`scripts/install_skills.sh`（装进任务工作区 + 物化 symlink + lock 指纹）真跑：**40 skills / 3 agents / 1274 个文件**，零残留 symlink。
- **实测一条完整链路**（run `r20261015-1040-test`）：`nm_test.sh --action test` → 远端 `OPX_LOCK_OK devices=0 mode=exclusive` → 容器里跑 `test.sh`（当时的 NOT IMPLEMENTED 存根，exit 2）→ 回读产物 → manifest（digest/快照/判据/设备锁齐）→ 自动登记 run #1 → `GET /api/runs/1/gate` 判 **fail**，理由是 `E_EXIT_NONZERO`、`E_REQUIRED_ARTIFACT_MISSING(correctness)`、`E_SUITE_REF_MISSING`、`E_CORRECTNESS_REPORT_MISSING` —— 即「存根跑过一次」不会被判过，**这正是 M1 要填的洞**。

### 9.4 M1-a 实测（2026-10-15）

首个算子 `elementwise_add` 的 **Ascend C 直调**实现真机跑通：19 条用例全过、第一份 `correctness.json` 落盘、run #5 门禁 **pass**；同时抓出三个真问题（CANNBot 模板分块不回推 GM 偏移 / 容器每次干净导致 `test` 依赖 build 产物 / 门禁同类产物撞车造成假阴性），都已修 + 加回归。
路线定档与容器事实见 `docs/M1-spike.md`；逐条证据、run 台账、D8 预算用量见 `docs/PROGRESS.md`「M1-a 实测记录」。
**M1 已补齐**：`benchmark`/`profile` 实装 + **D9 已定**（2026-10-16 实测四类指标均可采，见 §9.5）。

### 9.5 M1-b 实测（2026-10-16）

`benchmark` / `profile` 两个动作实装并真机跑通（run #6 / #7 门禁 **pass**）：

- **D9 定档**：`msopprof`（= `msprof op`，CANN 9.1 自带）在 `lfx-op-dev-910b:torch210-cann910` 上**四类指标都能采到** —— 耗时 `OpBasicInfo.Task Duration`、数据搬运 `Memory/PipeUtilization`、存储使用 `MemoryUB/Memory`、计算单元负载 `ArithmeticUtilization/PipeUtilization`；门禁据此把 `profile` 的必需产物从 `log` 收紧为 `log + metrics`。
- **基准的一处纠偏（重要）**：最初用 host 墙钟（启动+同步）对比，得到「候选比 `torch.add` 快 5–6×」—— 那是把 **PyTorch 的 host 派发开销**算成了算子差异（实测 torch.add host 端到端约 80us，设备侧只有约 8us）。改成**两侧都走 msopprof 量设备侧 `Task Duration(us)`**（候选跑可执行文件那一次 kernel、基线跑 `tools/torch_add_baseline.py` 的 `torch.add`），同卡同条件后真实结果是 **8.46us（torch.add） vs 7.98us（候选）≈ 1.06×**，并在小尺寸（64k）如实报出候选反而更慢（固定开销占比高）。
- 候选 exe sha256 `c4039046…` 与 `test` 验证过的**逐字节相同**（基准/采集没有改动算子源码）。
- 细节与逐条证据见 `docs/PROGRESS.md`「M1-b 实测记录」。

### 9.6 M3-a 实测（2026-10-16）

**自动优化闭环**跑通首个真机 A/B（Loop #2，run #8/#9 优化前 vs #10/#11 优化后）：

- **硬件画像** `ops/chips/dav-2201.yaml`（§13-A11）：UB/L1/L0/核数/频率/对齐/Vector 宽度，源自 `npu-arch` skill，带 source/confidence，**不硬编码进算子**。
- **`perf` Loop + 成对 run**：两侧按 `code_snapshot_sha256` 配对，各需 test + benchmark。
- **`performance-diff.json`**（契约见 `docs/perf-diff-v1.md`）：前/后耗时（设备侧中位数）+ 精度不退化 + 不确定性 + 结论枚举；**可比性硬条件**= 规格/判据/镜像/设备 + **基准用例集指纹**（`suite_ref` 只记录不硬判）。
- **门禁规则**：**精度退化硬拦**（`E_PRECISION_REGRESSION`）；**性能回退只 warn**（不设加速比硬线）。
- **P2 对比视图**：任务详情「自动优化 · 优化前后对比」分区。

**首个 A/B 结论（如实）**：`TILE_LENGTH` 4096→8192 后，主用例 1M **-1.2%（±2.99%，落在不确定性内）= `no_significant_change`**，64k 反而 **+19.6% 回退**；精度 19/19 不退化；门禁 pass。**该优化不予采纳** —— 并留下「小 shape 应走 core-shrink」的实测线索。细账见 `docs/PROGRESS.md`「M3-a 实测记录」。

### 9.7 M3-b 实测（2026-10-19，agent 自动改代码）

把 M3 余项「agent 自动改代码」接进 `perf` Loop（§13-A5「agent 提议、工具判定」）：

- **入口**：`POST /api/loops/{id}/optimize` 起一次 dsh job（工作区 cwd、挂 skills；prompt 固定层 = 规格 + 本轮 Loop 假设/判定 + **参考轮次的结构化建议** + 硬件画像）；`GET …/optimize` 读证据；`POST …/optimize/revert` 一键回退。证据落 `loops.optimize_json`（迁移 `f1a2b3c4d5e6`）。
- **范围**：agent **只改代码**，**不跑** build/test/benchmark —— 真机执行仍照 `plan.command` 单独触发（**2026-10-20 起四动作已可异步真执行**，但 optimize 仍不自动串起它们；两者是独立动作）。
- **真机冒烟**：只读勘察一轮（不改代码）在真 dsh 上通过（15 次工具调用、590 段逐 token、证据 `changed=false`）。
- **顺带证伪**：M3-a 的 core-shrink 线索**前提不成立** —— `op_host/elementwise_add.asc` 已用 `blockNum = ceil(totalTiles / tilesPerCore)` **隐式裁核**（64k 只起 16 核）；下一轮换线索（详见 `docs/PROGRESS.md`「M3-b 实测记录」）。

### 9.8 阶段看板（2026-10-19，P3 读侧）

§5 的四阶段看板落地（读侧）：`GET /api/stages` 按**真实证据推导**阶段（各动作最近一次 run 的 `gate.evaluate` verdict + 是否有规格），不信手填的 `tasks.stage`；`POST /api/tasks/{id}/stage/sync` 按证据把 stage 落正并写 `stage_events`。**「交付」段当前不会达成**（四动作不含 delivery）。详见 `docs/PROGRESS.md`「阶段看板实测记录」。


## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| DSH 处**开发者预览期**，承诺破坏性变更 `[已核验]` | 集成层薄：只依赖契约面（`headless --json` 事件、sessions JSONL、web 反代约定） |
| DSH 默认 `danger-full-access` + 持久 shell `[已核验]` | 任务级持续工作区 + **每次真实执行打源码快照**；首版串行 1；远端跑在容器里 |
| 代码「控制台机 ↔ 裸金属」同步 | 每 run 独立远端目录 + rsync 增量 + 保留策略；回读后算 sha256 |
| 模型 key 落点 | 只落控制台+dsh 那台机（600 文件/环境注入），裸金属不放；页面只掩码 |
| `session-log-deepseek` 默认上传日志事件到 DeepSeek `[已核验]` | 每个 `dsh_home` 的 profile `enabled: false` |
| **业务层被"看起来成功了"骗过** | 门禁以**报告校验 + 版本一致性**判定；命令退出码 ≠ 业务通过 |
| **环境问题被当成算子 bug** | 失败分类 + 基础设施失败不自动重试；执行前检查环境条件 |
| **指标采不到却按已采集展示** | 指标状态枚举 `measured/estimated/unsupported/failed` + 来源；估算值标假设 |
| **性能提升无统计学意义** | 同条件对比 + 测量不确定性阈值；否则输出「无法确认提升」 |
| 许可叠加：DSH MIT，CANNBot/skills **CANN OSL 2.0**（限华为 AI 处理器用途、可撤销、须保留声明）`[已核验]` | 内部工具无碍；对外开源时限制跟着发布走 |
| skills 上游迭代快（月级 20 版） | pin commit + 手动 bump（记录旧→新 commit + 变更摘要，升一次跑一次冒烟） |
| 镜像能起 ≠ 环境能跑算子 | `env_recipe` 必填；执行前按动作检查环境条件（`torch_version` 为空的镜像直接拦 test/benchmark/profile） |
| **共享机上设备被占 / 邻居污染耗时** | 前置检查选空卡（读 `npu-smi proc-mem`；HBM 占用不是判据）+ 远端 `flock` 独占锁，拿不到就 exit 3 不跑；`lock_mode` 记进 manifest，门禁对非独占的采集给 warn |
| **agent 能力（skills）装在会被清理的目录** | 只装进任务工作区（`scripts/install_skills.sh`），来源换成仓内 `.skills-src/cannbot`，`ops/dsh/skills.lock` 记 commit + 逐文件 sha256；`--verify` 可随时复核 |
| 时间口径 / 远端堆积 / 悬空镜像吃盘（实测 81.8 GB） | 库内 UTC；每 run 临时目录 + 清理策略；镜像页标悬空、给清理建议（不自动删） |
| **编排层膨胀成"自研 orchestrator"** | 明确最小集（§3.2）：3 张真表 + 校验器；其余用 JSON 扛；**不碰 LLM loop** |

## 11. 决策状态

### 11.1 已定

| 项 | 结论 |
|---|---|
| 技术栈 | Python 3.12 + FastAPI + SQLAlchemy/Alembic；React 19 + Vite + TS + Tailwind v4 + shadcn/ui + ECharts |
| **部署形态** | (ii) 控制台 + dsh 在独立机器；裸金属只做编译/运行 |
| **工作区** | **任务级持续开发工作区 + 每次真实执行打源码快照**（v1.1 改） |
| **对话过程 vs 业务证据** | **分开存、建关联**（v1.1 改） |
| **数据模型** | Project → Task → **Loop** → Run → Artifact；**AgentJob 与 Run 分离**；最小集见 §3.2 |
| **门禁** | 四阶段；**结果有效 + 业务达标 + 版本一致 + 产物完整**；硬拦（v1.1 升级） |
| **异常输入口径** | 按规格定义的预期行为验证（**"不崩即过"降级档取消**，v1.1 改） |
| 判据口径 | 默认官方 `ops-precision-standard`，支持项目自定义档 |
| 性能采集 | 只到算子层；指标带状态与来源；基准与 profiling 分开；**基线需语义匹配**；要基线与极限对比 |
| **费用口径（D2，2026-10-21 定）** | **端点单价可配**（人民币元 / 每百万 token：`price_in` / `price_out` / `price_cache_read` / `price_cache_write`）；**in+out 都配才算已定价**；先按调用（job）实时算、再按任务 / 类型 / 天聚合；cache 两项留空 = 不单独计价（按 0）；费用不落库，改单价历史实时重算（早期无端点快照的调用按当前价目估算并标注） |
| 运行时接法 | 子进程 `dsh headless --json`；v1 统一 CLI `0.2.0-rc.2`；SDK 留 P6 |
| 执行模型 | 本地改代码 → **统一执行入口四动作**（`nm_test.sh` 作适配器后端）；不做命令级封装/白名单 |
| 受控修复 | 失败分类 + 有边界修复 + 轮次/预算可配 + 基础设施失败不自动重试 |
| 能力接入 | CANNBot skills pin commit，**首条流程即接入**并按阶段选能力（v1.1 提前） |
| 每项目独立 `dsh_home` / 密钥 / 事件存储 / 时区 | 密钥只引用名；明细留 JSONL，库只索引与摘要；库内 UTC |
| 设计方向 / 页面 IA | 皮肤 B 深色玻璃 + 亮；**八页**（含案例库）；任务详情按 §7 分区 |
| **自动优化 = 必需能力**（不设提升幅度硬线） | 必须完成「同条件 A/B + 对比报告」；验收以**至少一份有效报告**为准（不要求每个案例都做优化）；提升不显著时如实报告。**自动 = agent 执行 + 工具判定；人保留触发、审批与回退权** |
| **部署位置（D4）** | **本机（这台 Mac）+ 单人使用**；存储保持 SQLite；不做登录/鉴权；V6 关闭 |

### 11.2 待定

| # | 待定 | 卡到哪一步 |
|---|---|---|
| ~~D4~~ ✅ | ~~控制台+dsh 部署在哪台机 / 是否要登录多人~~ → **已定：本机 + 单人**（见 §11.1） | 2026-10-14 定 |
| **D10**（新） | 本机常驻方式：launchd 自启（含 600 环境文件注入 `OPX_LLM_KEY`）还是手动起服务 | **仍未定**（当前手动起服务，launchd 未做） |
| ~~D6~~ ✅ | ~~首条流程算子~~ → **已定：`elementwise_add` 类简单算子**；`complex64 mm` 作后续代表验证（见 §8） | 2026-10-14 定 |
| D5 | 与既有资产的关系：把 `mm-operator` / 202608 交付件导入做基线还是一切从零 | **仍未定**（P2 已完成且未使用 D5；后续基线对比前需定） |
| D1 | 「算子开发镜像」是否恢复「预检通过才算可用」的门禁（当前＝勾选 + 记配方） | 现在＝不恢复（仍未改口径） |
| ~~D2~~ ✅ | ~~费用单价口径（`cacheRead` 是否单独计价；按 run 还是按 task 汇总）~~ | **已定（2026-10-21）**：端点可配单价（¥/百万 token，cache 可选）+ 按调用实时算再聚合；已实现并实测，见 PROGRESS「端点定价实测记录」 |
| D3 | 保留策略：sessions JSONL / 产物 / 日志 / 远端目录各留多久 | **仍未定**（P3 读侧已完成，保留策略未落地） |
| D7 | 队列与重试：默认串行 1、超时可配、失败不自动重试 | 已按默认实现 |
| **D8** | 执行预算 → **按每案例 / 每 Loop 计**：每个案例 `build`/`test`/`benchmark`/`profile` 各 **1–2 次**；同一 Loop 最多 **3 轮**受控修复；**环境/契约类失败不计入**（设备被占 `exit 3`、容器干净导致的契约缺口、存根式跑法），但仍要如实登记 run 行。跨 Loop 各自独立计。**算例**：首次实现 = `build` 1 + `test` 1 + `benchmark` 1（可选 `profile` 1）；一轮 A/B = 优化前 `test`+`benchmark`、优化后 `test`+`benchmark` = **4 条**。超限输出「失败原因 + 已尝试方向 + 下一步建议」。**当前为人工核对的预算口径，超限靠评审；自动计数属后续** | 要改就改这里 |
| **D9** ✅ | 指标体系支持范围 | ✅ **已定（2026-10-16 实测）**：`msopprof` 四类指标（执行耗时 / 数据搬运 / 存储使用 / 计算单元负载）均可采，命令与字段固化在 `workspaces/cann-ops/tools/run_profile.py`；门禁对 `profile` 收紧为**必需 `log`+`metrics`**（缺 → `E_METRICS_REPORT_MISSING`，fail）。**曾是 M3 的硬前置，现已解除**（M3 已在其上完成）。不适用当前算子的指标（如 vector 算子的 cube 列）如实标 `unsupported` | 已定 |

## 12. 验收清单（MVP，采建议稿 §12.2）

首个 MVP 必须提供以下**真实证据**：

1. 三种需求输入方式均能形成结构化规格；至少一种完成全流程演示。
2. 关键语义缺失时能提出具体澄清问题，假设有记录。
3. 生成算子实现、调用接口、构建脚本、测试脚本和使用示例。
4. 在声明的目标硬件与环境中**真实编译、运行**候选实现。
5. 常规/边界/异常用例有覆盖；报告含形状、dtype、数值误差与异常行为判定。
6. 能保留并展示**失败用例、错误原因**与修复后的新执行记录。
7. 采集耗时/搬运/存储/计算单元负载四类目标指标；**未支持项明确标记**，不得按已采集验收。
8. 至少一条优化建议引用性能证据并说明适用范围与验证方法；并完成**优化前后同条件 A/B 对比报告**（**不设加速比硬线**，提升不显著时如实报告「无法确认提升」）。
9. 报告与交付代码可通过版本、环境、命令与产物哈希相互对应。
10. 从独立工作目录按复现说明运行，能重现正确性结论并再次采集性能。
11. 提供**不少于 10 个典型算子开发案例的可导入 bundle**（独立 zip；仓库不预置案例，用 `scripts/import_cases.py` 导入），导入后清单可查、可一键发起任务，覆盖矩阵计算 / 归一化 / 激活函数 / 注意力机制四个场景；**10 个案例全部完成真机全链路**（规格 → 实现 → `build` → `test` → `benchmark` → 门禁 pass；`profile` 可选）。**算子清单不写死**：某个算子开发受阻时可在**同场景内替换**，但必须保持 10 个案例、四场景覆盖与全部真机全链路。
12. **优化前后对比报告可复现**：报告带版本、环境、命令与产物哈希；从**独立工作目录**按复现说明重跑，能得到同一组前后快照与可判的结论（结论可以不同，但配对与可比性不能丢）。

> 自然语言解释、会话退出成功、页面展示正常、报告文件存在 —— **都须与上述执行证据结合**才算验收。

**落库硬约束（2026-10-15 复查补）**：验收那一刻，「契约算 plan → 真实执行 → 报告 → 门禁判定」四步必须**全部落库**（`runs` 行 + `manifest.json` + 报告文件 + `gate_report`），
只落一半（例如有 run 行但没门禁判定、或报告文件在但没登记）不算通过。当前状态（**2026-10-15 M1-a 起**）：`runs`/`artifacts` 已有真实数据（task #2 共 5 条 run，含 1 条 `verdict=pass` 的 test）；`stage_events` 仍空 —— 阶段推进在 P3。

**交付目录（采建议稿附录）**

```text
delivery/
├── README.md            # 调用方式、范围与复现步骤
├── spec.json            # 验收规格
├── design.md            # 实现设计
├── src/ examples/ scripts/ tests/
├── reports/             # correctness.{json,md} performance.{json,md} recommendations.md
├── evidence/            # 原始样本、日志与采集数据
└── manifest.json        # 版本、环境、来源与产物哈希
```

---

## 13. 需求调整（2026-10-15）：差额修订清单

> **本节已并入正文（v1.2，2026-10-20 用户逐条确认）**：下表保留为**变更记录**（原始建议与理由），以正文为准。原「本节优先于正文」的交叉引用已全部落到正文，`⚠️ §13-A#` 标记不再承担优先级作用。

> **2026-10-20 补充（M2 交付形态）**：案例库改为**外置 bundle（独立 zip）+ 导入脚本**，仓库不预置案例；A9/A10 里「工作区 `cases/<op>/` + `cases/index.json`」仍然成立，但它是**导入后的落点**，不是仓库内容。
> 新需求原文逐字归档于 **`REQUIREMENT.md` §四**（与（一）的关系待确认，旧原文保留）。

**差异总览**（逐条对照见 `REQUIREMENT.md` §4.1）：D1「辅助生成」→「**自动**生成」（措辞升格，主流程本已自动）；**D2 新增「内置不少于 10 个典型算子开发案例，覆盖矩阵计算 / 归一化 / 激活函数 / 注意力机制」**；**D3「自动优化」从可选扩展升为必需能力**（基于目标硬件特性生成优化代码 + 优化前后精度与性能对比 + 可复现测试报告）；D4 四类指标与"结合形状/dtype/硬件信息"未逐字要求（我方保留，只多不少）。

**2026-10-15 已定口径（用户已答）**：① 案例库深度 = **10 个案例全部齐备**（规格 / 用例 / 参考实现 / 实现代码 / 基线），覆盖四个场景的 **≥3 个真机跑通全链路**，其余给可复现的离线或模拟器验证记录（**2026-10-20 已改：10/10 全部真机，见 §8 / §12 #11；本条仅作历史记录**）；② 自动优化验收 = **完成闭环 + 对比报告，不设加速比硬线**（提升不显著时如实报告）；③ **D8 改按案例 / 按 Loop 计**（每案例四动作各 1–2 次，受控修复仍 3 轮）。

### 13.1 要改的条款（逐条）

| # | 位置 | 现状 | 建议修订 |
|---|---|---|---|
| A1 | §1.2 / §11.1 | 「不以获得性能提升为必需验收；自动改代码反复调优 = 可选扩展（P6）」 | **自动优化 = 必需能力**（优化前后同条件 A/B + 对比报告）；**验收仍不设提升幅度硬线**——「不必提速」与「必须做优化」是两件事，原文把两者混在一句里 |
| A2 | §8 路线 P6 | 自动调优列在 P6（可选） | 移出 P6，拆成 **M2 案例库** / **M3 自动优化闭环** 两个里程碑（排期见 §13.3） |
| A3 | §5 阶段三·验收门禁 | 必需正确性用例通过 + 有效性能报告与证据引用齐备 | 增加**性能对比报告有效性**校验：成对版本**同条件**（同卡 / 同镜像 digest / 同预热与重复次数 / 同测量范围）+ **精度不退化** + 样本数与不确定性有记录 + 来源齐 |
| A4 | §12 验收 #8 | 「…**不要求已获得加速**」 | 改为「必须完成**优化前后同条件 A/B 对比报告**；不设加速比硬线，提升不显著时如实报告」；并**新增** #11「内置 ≥10 个典型案例（清单可查、可一键发起任务，覆盖四场景）」、#12「对比报告可复现（版本/环境/命令/产物哈希对齐，从独立工作目录可重跑）」 |
| A5 | §4.6 性能采集与分析 | 基准与 profiling 分开；指标带状态与来源 | 增加**优化前后对比**：`loops.kind=perf` 的成对 Run 关联 + 新报告 **`performance-diff.json`**（优化前/后各自四类指标 + 精度不退化 + 不确定性 + 结论枚举 `improved` / `no_significant_change` / `inconclusive`） |
| A6 | §11.2 D8 | 每任务 ≤6 次真实执行 | 按案例 / 按 Loop 计：每案例 `build`/`test`/`benchmark`/`profile` 各 1–2 次；受控修复 3 轮；环境/契约类失败仍不计入 |
| A7 | §11.2 D9 | 「与 `profile` 动作一起验」 | **升级为自动优化（M3）的硬前置**：采不到四类指标就没有"性能瓶颈分析"的数据基础 |
| A8 | §6 执行模型四动作 | `build`/`test`/`benchmark`/`profile` | **不新增第 5 动作**：对比报告由 `benchmark` 产物承载。**最终未采用 `--baseline-run` 方案**（与 §6「不引入」一致）：改由 `loops.kind=perf` 成对关联 + `performance-diff.json`（见 A5 / §4.6） |
| A9 | §7 页面 IA | 七页，无案例库 | 新增「**案例库**」页（浏览/筛选 → 一键按案例发起任务）；任务详情增加「**优化前后对比**」分区 |
| A10 | §3.1 数据模型 | `loops.kind` 已含 `perf`；`case_results`/`metrics`/`recommendations` 用 JSON | **不建新表**（守住 §3.2 最小集）：案例库 = 工作区 `cases/<op>/` + JSON 索引清单；优化对比 = `performance-diff.json` + `runs` 成对关联（复用 `code_snapshot_sha256` / `loop_id`） |
| A11 | 新增资产 | — | **`ops/chips/dav-2201.yaml`**：目标硬件特性画像（UB / L1 / L0 容量、核数、cache line/burst 对齐、Vector 指令宽度、HBM 带宽与算力），内容源自 `npu-arch` skill；规格的环境段**引用**它 —— 这是「基于目标硬件特性生成优化代码」的输入，不能让模型凭记忆猜 |
| A12 | 措辞 | 全文「辅助生成 / 辅助识别」 | 统一改为「**自动生成**」「自动分析与优化」（**除** `REQUIREMENT.md` §一 原文照抄不动） |
| A13 | 落版 v1.2 时同步 | — | `docs/gate-report-v1.md`（对比报告校验 + 新 fail code）、后端 `gate.py` 与动作契约、`cases/` 索引与工作区、`ops/chips/` |

### 13.2 建议的案例清单（≥10，覆盖四个场景，待定稿）

| # | 场景 | 算子 | 备注 |
|---|---|---|---|
| 1 | 矩阵计算 | matmul (GEMM) | 基线 `aclnnMatmul`，**基线语义/dtype 支持范围须先确认**（§4.6 的既有约束） |
| 2 | 矩阵计算 | batched matmul | 批维与 broadcast |
| 3 | 归一化 | softmax | 数值稳定性（减最大值） |
| 4 | 归一化 | layer_norm | 双 pass / Welford |
| 5 | 归一化 | rms_norm | 仅平方均值 |
| 6 | 激活 | relu | 最快，作最小样张 |
| 7 | 激活 | gelu（tanh / erf 两档） | 超越函数精度口径 |
| 8 | 激活 | silu / swish | 融合乘 |
| 9 | 注意力 | attention 核心（QKᵀ → softmax → PV） | 融合 + 中间精度口径，**风险最高** |
| 10 | 注意力 | flash-attention（分块 + 在线 softmax + 重计算） | 视 #9 进展，可降级 |
| — | 回归载体 | `elementwise_add`（M1-a 已完成） | 留作回归载体与最小闭环样张，**不计入 10 个**（合计 11 个可交付案例） |

### 13.3 排期建议（先闭环、后批量）

| 序 | 块 | 估时 | 前置 |
|---|---|---|---|
| 1 | **M1-b**：`benchmark` / `profile` 实装 + **D9 摸底**（固化进镜像配方与门禁） | ✅ **已完成（2026-10-16）** | run #6 / #7 门禁 pass；D9 已定（§9.5） |
| 2 | **M3-a 自动优化闭环**：拿**已完成的 `elementwise_add`** 当首个 A/B 样张 —— 硬件画像 + perf Loop + `performance-diff.json` + 门禁 + P2 对比视图 | ✅ **已完成（2026-10-16，§9.6）** | run #8/#9 vs #10/#11；首个 A/B 结论如实 |
| 3 | **M2 案例库** ⚠️ **已延后到最后做（2026-10-19 用户定）**：框架（索引 + 案例库页 + 一键发起）1–2 天；10 个案例内容 6–9 天 | 7–11 天（≥3 真机）→ **改 15–25 天（10/10 真机，2026-10-20 定）** | 各案例基线支持范围确认 |
| 4 | **v1.2 落版**（本文档 + `gate-report-v1` 交叉引用） | ✅ **已完成（2026-10-20）** | A1–A13 已逐条确认 |

**为什么是这个顺序**：闭环（M3-a）风险最高且卡在 D9，先用已有算子打通最省成本；案例库是工作量最大、可分批的一项，且它的"真机全链路"名额需要先有好用的 benchmark/对比能力。反过来先堆 10 个案例，会先撞上 D8 预算与 D9 未定两个坑。

> **2026-10-19 更新（用户定）**：**M2 案例库延后，放最后做**（本表原排第 3 位）。内容口径不变，只改先后；最新待办顺序以 `docs/PROGRESS.md` §待办 为准（自动优化增强 / P0-2 / P3·P4 页面占位均排在 M2 之前）。**M3 余项「agent 自动改代码」已于 2026-10-19 完成（§9.7）**。

**一处口径已确认（2026-10-20）**：原「≥3 个真机全链路」按 **10 个全部真机全链路（硬性）** 执行；案例算子清单不写死，同场景内可替换，但必须保持 10 个 + 四场景覆盖 + 全部真机全链路。

---

## 附录 A：与建议稿的差异处置（v1.0 → v1.1 逐条）

### A.1 冲突点（必须选边，已定）

| # | 项 | v1.0 | 建议稿 v2 | v1.1 采用 | 理由 |
|---|---|---|---|---|---|
| 1 | 主线顺序 | 控制台优先 | 编排优先，控制台排 M3 | **各取一半**：保留 P0/P1 成果（控制台已在跑），**P1.5 插进契约最小集 + 容器闭环** | 建议稿是照 v1.0 文本写的，不知道 P0/P1 已完成；按它的 M3 重排等于推翻已做的东西 |
| 2 | 工作区 | 每 run 一次性 | 持续工作区 + 执行快照 | **建议稿** | 算子开发是"改一版→修一版"，一次性工作区是反的；快照哈希照样能隔离与复现 |
| 3 | 门禁语义 | 产物齐备 | 结果有效 + 版本一致 + 业务达标 | **建议稿** | 「齐备」会被"报告文件存在但内容废"骗过 |
| 4 | 异常输入 | 可降级「不崩即过」 | 按规格预期行为验证 | **建议稿** | 降级档掩盖真实缺陷 |
| 5 | 真相源 | sessions JSONL 全量 | 对话与业务证据分开 | **建议稿** | 两类证据性质不同，混一起无法机判 |
| 6 | Run 语义 | 一表两义 | 拆 AgentJob / Run + Loop | **建议稿** | 一轮 agent 调用确实产出多个真实执行 |

### A.2 已吸收的增量

Loop 概念｜指标状态枚举与来源｜建议的结构化 schema｜受控修复与失败分类｜统一执行入口四动作 + 取消/超时传远端｜任务详情六分区｜交付目录与 MVP 验收 10 条｜规格字段表｜报告状态枚举｜基线需语义匹配｜按阶段加载 skills（不让全部 skills 进上下文）｜SSE 事件序号与补读、重启后按持久化状态核对。

### A.3 降级/分期实施（不照搬的部分）

| 建议稿要求 | v1.1 处置 | 理由 |
|---|---|---|
| 13 个对象全部建表 | **最小集**：`agent_jobs`/`loops`/`spec_versions` 建真表；`case_results`/`metrics`/`recommendations` 先用 JSON + 报告文件 | 建议稿自己也允许；单人自用不需要一次到位 |
| 完整业务编排层（状态机 + 契约校验 + 能力路由 + 恢复） | **P1.5 只做**：规格 schema、四动作执行契约、门禁校验器 v1（有效+版本一致）；能力路由放 P5 | 一次全做会把首个可用时间拉长数倍 |
| 每阶段独立审查/多智能体 | 不做（首版单主控） | 建议稿同样说不要求 |
| 指标体系与命令细节 | 标 `[待实测]`（D9），真机跑完再固化 | 建议稿自述"具体可用命令与指标支持待目标环境验证" |

### A.4 建议稿的盲点（落地时需注意）

1. **没看到代码进展**：它依据的是 v1.0 文本，把控制台排到 M3，而控制台 P0/P1 已跑通。
2. **没接 v1.0 的待定项**：D2/D3/D4 它都没提，只留了"实施前需明确"7 项（与 D1/D5/D6 部分重叠）。
3. **体量对单人自用偏重**：见 A.3。

### A.5 路线映射（我方 P ↔ 建议稿 M）

| 我方 | 建议稿 | 说明 |
|---|---|---|
| P0 ✅ / P1 ✅ | （M3 的一部分提前完成） | 控制台骨架、资源页、发起 agent run、事件流 |
| **P1.5** | **M0 契约与环境** + M1 的一部分 | 容器闭环 = M0 验收；规格 schema、AgentJob/Run/Loop = M1 的地基 |
| P2 | M2 性能与分析 | 结果视图与指标状态 |
| P3 | M1/M3/M4 的门禁与测试管理部分 | 校验器接进门禁 |
| P4 | M4 | API/消耗 |
| P5 | M1（能力接入） | 建议稿要求首条流程即接入，已提前到 P1.5 |
| P6 | M5 | 可选扩展（**不含自动调优** —— 已并入 M3） |

> 注：我方自 2026-10-15 起另用 **M2 = 案例库、M3 = 自动优化闭环**（见 §8）；与建议稿的 M 编号不是同一套，引用时以 M 前缀的上下文为准。

## 附：与既有材料的对应关系

- 需求拆解 C1–C4 → `REQUIREMENT.md`；C3/C4 的落点 = P2 结果视图 + 任务详情页。
- 现成能力盘点 → `prior-art.md`；§1 的处置逐条对应该文档的覆盖度结论。
- **外部建议稿原文** → `docs/proposals/算子开发智能体优化方案-v2.md`（逐字归档，sha256 `b6c44aa0…`）。
- 进展与实测记录 → `docs/PROGRESS.md`。
- CANNBot / DSH / CANN OSL 事实与取证 → 技能库 `ascend-operator-research` 的 `references/cann-agent-ecosystem.md`；build/fork/reimplement 判据 → 技能库 `build-vs-adopt-review`。
- 设计样机 → `docs/mockups/console-ia-v1.html`、`console-pages-v1.html`。
- v1.0 原文 → `PLAN-web-console.md`（保留为决策溯源记录，顶部已加取代说明）。
