# 同类项目调研（首轮，2026-10-14）

调研范围：GitCode（昇腾/CANN 官方主力托管）、GitHub。目标：回答「这个需求有没有现成实现、能不能直接用」。

证据级别：`[已核验]` = 我实际打开过该页并引用原文；`[转述]` = 只来自搜索摘要/二手页面；`[推断]` = 我的判断。
所有 URL 均为 2026-10-14 实际访问（HTTP 200，除注明）。原始快照见 `reference/`。

---

## 0. 结论先说

1. **需求里的四件事，昇腾官方 CANNBot 仓群已经基本做完了**：需求解析 / 代码生成 / 用例+精度报告 / 性能采集+优化建议，都有对应的 Agent 和 Skill，且是 Apache-2.0、2026-10 仍在活跃更新。`[已核验]`
2. **连"评测"这一层官方也有**：`cann/cann-bench`（CANN 领域 AI 代码任务评测，三层框架、四级复杂度）和 `cann/cannbot-sentry`（观测/审计/AB 门禁）。`[已核验]`
3. GitHub 上昇腾算子 Agent 的个人项目不少但都很小（0–39★），没有能对标 CANNBot 的。`[已核验]`
4. 因此**不建议从零造**。选题要落在官方没覆盖的缝隙上：具体业务算子族的垂直固化、私有数据集/判据口径、与现有交付流程（补丁+run 包+交付文档）的对接。`[推断]`

---

## 1. GitCode：昇腾官方 CANNBot 仓群（同类度最高）

来源：`cann/cannbot` README（快照 `reference/cannbot-README.snapshot.md`）原文：「CANNBot 是 CANN 社区的 Infra 智能体层，用 Agent 完成 AscendC/PyPTO/TileLang/Triton 等各类语言的算子开发、模型迁移与推理优化，并延伸至图模式、Runtime 等更多 CANN 开发场景。」`[已核验]`

| 仓库 | 定位（引用官方简介/README） | 证据 |
|---|---|---|
| [`cann/cannbot`](https://gitcode.com/cann/cannbot) | 应用仓与用户入口：工作流编排 + 场景插件；npm 安装 `npx @cannbot-plugin/cannbot@latest install <plugin> --tool opencode`；Apache-2.0；支持 OpenCode / Codex / Claude Code | `[已核验]` README 原文 + 仓库页 |
| [`cann/cannbot-skills`](https://gitcode.com/cann/cannbot-skills) | Skills + Agents 全量清单，覆盖 Ascend C / TileLang / Triton / Catlass / PyPTO / Runtime / 图模式 / 模型推理优化 / 训练精度诊断 / CANN 工具链 | `[已核验]` 功能清单快照 |
| [`cann/cannbot-knowledge`](https://gitcode.com/cann/cannbot-knowledge) | 可追溯工程知识库：文档+源码+实验知识整理成 Markdown 知识卡，提供生产/检索/校验/审计 | `[已核验]` 仓库页简介 |
| [`cann/cannbot-dsl`](https://gitcode.com/cann/cannbot-dsl) | 基于 CANNBot-DSL 生成的复杂算子示例（VoxelConv、FlashAttention 等），面向 NPU ARCH 3510 | `[已核验]` 仓库页简介 |
| [`cann/cann-bench`](https://gitcode.com/cann/cann-bench) | CANN 领域 AI 代码任务评测：算子生成、优化等领域，三层评测框架、四级复杂度、统一量化标准、"识别 Agent 短板" | `[已核验]` 仓库页简介 |
| [`cann/cannbot-sentry`](https://gitcode.com/cann/cannbot-sentry) | LLM 编码 Agent 观测/审计/评测：时间线回放、网关捕获、合规对账、安全审查、A/B 门禁，面向 opencode/Claude Code | `[已核验]` 仓库页简介 |
| [`Ascend/agent-skills`](https://gitcode.com/Ascend/agent-skills) | 昇腾 AI 辅助研发的核心技能管理仓（GitHub 有同名镜像）；A 类 AscendC 算子开发、C 类 Triton、E 类 PyPTO、H 类 ATB 迁移、G 类自动化测试 | `[已核验]` 仓库页 + GitHub README |

> 计数口径提醒：`Ascend/agent-skills` 的 AscendC 算子开发 Skill 数量，GitCode 侧搜索摘要写 52 个、GitHub 镜像 README 写 9 个、本次抓到的 `cannbot-skills` 功能清单 A 类表格列了 29 个 skill + 10 个 agent。**三处数字不一致**，说明仓库在快速演进且各入口未同步 —— 要引用具体 skill 名单时，应以当时 `cannbot-skills/docs/feature-list.md` 为准。`[已核验]`

### 官方插件（对应需求 C2 代码生成）

| 插件 | 能力 | 官方原文 |
|---|---|---|
| `ops-direct-invoke` | Ascend C 直调算子开发 | 以 `<<<>>>` 直接调用核函数的方式接入 |
| `ops-registry-invoke` | Ascend C 注册算子开发 | 经 ACLNN / GEIR 接入框架，以标准算子形式调用 |
| `triton-op-generator` | Triton-Ascend 算子代码生成与优化 | Phase 0-7 直接执行模式，知识卡片由 cannbot-knowledge 供给 |
| `hccl-op-generator` | HCCL AICPU 标准通信算子开发 | 四 Agent 状态机工作流 |
| `model-infer-optimize` | NPU 模型迁移、精度对齐与推理性能优化 | — |

以上均为 `cannbot` README 原文表格 `[已核验]`。

---

## 2. 覆盖度对照：需求 C1–C4 ↔ CANNBot 现有能力

（Skill/Agent 名称取自 `cannbot-skills/docs/feature-list.md` 快照，为官方原文名称）

| 需求能力 | CANNBot 对应 | 覆盖判断 |
|---|---|---|
| **C1 需求解析**（自然语言/数学表达式/参考代码 + 形状/dtype/硬件） | Agent `ascendc-ops-architect`（需求分析+方案设计）、`tilelang-op-analyst`、`model-infer-analyzer`；Skill `pypto-intent-understand`（需求意图理解与规格生成）、`triton-task-extractor`（从用户输入提取算子构建任务文件）、`npu-arch`（芯片型号映射）| 覆盖。**"数学表达式"作为输入形态**没有独立的 skill 明确对应，`[推断]` 由 `pypto-intent-understand` / `triton-task-extractor` 部分承担 —— 未核实 |
| **C2 代码生成**（实现 + 调用接口 + 测试脚本） | Agent `ascendc-ops-developer`、`developer-code`、`catlass-op-generator`；Skill `ascendc-registry-invoke-template`（完整自定义算子工程模板）、`ascendc-direct-invoke-template`、`ascendc-tiling-design`、`ascendc-api-best-practices`、`aiss-tiling-solver`（自动求解最优 Tiling） | 覆盖，且比我原以为的细（有工程模板、tiling 求解器） |
| **C3 正确性与精度报告** | Agent `ascendc-ops-tester`（ST/UT 用例生成与执行）、`pypto-op-verifier`（对抗测试+精度/layout 校验）；Skill `ascendc-ut-develop`、`ascendc-st-design`（L0/L1 用例生成）、`ascendc-whitebox-design`、`ops-precision-standard`（**按 dtype 分类的 atol/rtol 标准**）、`ascendc-precision-debug`、`pypto-golden-generate`（golden 参考实现生成） | 覆盖，连"精度标准"和"边界/白盒用例"都有 |
| **C4 性能采集与优化建议** | Skill `ops-profiling`（**CSV 指标解读、瓶颈定位、优化建议**）、`ops-theoretical-performance-estimation`（访存量/指令吞吐/流水重叠/目标时延）、`ascendc-performance-best-practices`（按算子族组织）、`tool-npu-compute`（带宽/流水线/存储访问/缓存/资源冲突）、`aiss-tiling-solver`；Agent `pypto-op-optimizer`、`tilelang-op-perf-tuner`、`triton-latency-optimizer` | 覆盖，四类指标（耗时/搬运/存储/计算单元负载）与工具采集项对得上 |
| （需求未提，官方也有）编译调试 | `ascendc-runtime-debug`（错误码 161xxx/361xxx/561xxx）、`ascendc-crash-debug`、`ascendc-sync-audit`（SetFlag/WaitFlag 配对）、`ops-simulator`（精度/性能仿真、流水线分析） | 超出需求 |
| （需求未提）交付与检视 | `ascendc-code-review`、`ascendc-docs-gen`（需求分析/详细设计模板）、`cannbot-skill-reviewer`、`gitcode-pr-handler`、`gitcode-issue-handler` | 超出需求，且贴近你现有的交付流程 |

---

## 3. GitHub：同类项目（个人/小体量）

Stars 为 2026-10-14 实测；均 `[已核验]`（GitHub Search API 返回）。

| 项目 | ★ | 说明 |
|---|---|---|
| [`Just-it/AscendOpGenAgent`](https://github.com/Just-it/AscendOpGenAgent) | 39 | Ascend operator generation，同类里最高星 |
| [`MagicBowen/optest`](https://github.com/MagicBowen/optest) | 1 | An Operator Test Framework for Ascend C Developing Agent（测的部分） |
| [`Sunye1213106/ascendc-codemap-mcp`](https://github.com/Sunye1213106/ascendc-codemap-mcp) | 6 | AscendC 算子 CodeMap MCP（给 coding agent 用） |
| [`Sunye1213106/testcase-generate-agent`](https://github.com/Sunye1213106/testcase-generate-agent) | 0 | 用例生成 agent |
| [`duoduoyi6/ascendC_debug_agent`](https://github.com/duoduoyi6/ascendC_debug_agent) | 2 | 调试 agent |
| [`ChaosWang666/AscendC-Kernel-Agent`](https://github.com/ChaosWang666/AscendC-Kernel-Agent) | 1 | Kernel agent |
| [`Bo1222/CannAgent`](https://github.com/Bo1222/CannAgent) | 0 | Ascend operator generation |
| [`KuaaMU/KuaaMU`](https://github.com/KuaaMU/KuaaMU) | 4 | Ascend NPU 算子优化 + MCP 工具 + agent skills 合集 |
| [`aloschilov/agentic-ascend-operator-views`](https://github.com/aloschilov/agentic-ascend-operator-views) | 1 | 编译器派生的算子知识视图（给 agentic AscendC 优化用） |
| [`s0ngY0ung/cann-op-agent-mla-ascendc`](https://github.com/s0ngY0ung/cann-op-agent-mla-ascendc) | 0 | MLA 算子 + KernelEvo 优化循环 |
| [`baicai-1145/ascend-dev-skill`](https://github.com/baicai-1145/ascend-dev-skill) | 0 | pi agent 用的 Ascend NPU 算子开发 skill（AscendC/CCE/Runtime/PTO-ISA） |
| [`jxliu1216/ascendc-rl-datasets`](https://github.com/jxliu1216/ascendc-rl-datasets) | 1 | AscendC RL 数据集（做训练数据的方向） |
| [`Westreety/Code-Review-Agent-for-Ascend-Operators`](https://github.com/Westreety/Code-Review-Agent-for-Ascend-Operators) | 0 | 昇腾算子代码检视 agent |
| [`KernelHive/AscendOptimizer`](https://github.com/KernelHive/AscendOptimizer) | 1 | 优化方向 |

判断：**都不构成"现成可用"**，多数是个人 demo/课设级别的仓库，无硬件闭环与精度体系。`[推断]`

## 4. GPU 侧可借鉴的方法论（不是同类，但路线可抄）

| 项目 | ★ | 可借鉴点 |
|---|---|---|
| [`ScalingIntelligence/KernelBench`](https://github.com/ScalingIntelligence/KernelBench) | 1290 | "LLM 能不能写 kernel"的评测范式：Torch→CUDA 任务集 + 正确性/加速比双指标，已被广泛引用 |
| [`alibaba/atrex-kernel-agent`](https://github.com/alibaba/atrex-kernel-agent) | 177 | 端到端 GPU kernel agent：实现→分析→profiling→迭代优化的闭环工程化 |
| [`AMD-AGI/Apex`](https://github.com/AMD-AGI/Apex) | 78 | Agent + RL 环境做 kernel 优化，含基准与奖励设计 |
| [`kcxain/Awesome-LLM4Kernel`](https://github.com/kcxain/Awesome-LLM4Kernel) | 82 | LLM for GPU kernel 开发综述（找选题/找 gap 的入口） |
| [`RLsys-Foundation/TritonForge`](https://github.com/RLsys-Foundation/TritonForge) | 152 | 训模型把 PyTorch op 转成 Triton kernel（数据合成方向） |
| [`AMDResearch/NPUEval`](https://github.com/AMDResearch/NPUEval) | 34 | 非 NVIDIA 硬件上的 kernel 生成评测集（AMD AIE），可类比造昇腾版 |

---

## 5. 增量空间在哪（`[推断]`，待你判断）

1. **垂直固化**：官方是通用编排器，未必覆盖你的具体算子族（复数、sparse、通信算子、特定融合算子）与私有判据口径 —— 你 202608 那批交付（complex64/SpMM/SpGEMM）就是现成的垂直样本。
2. **判据/口径自有化**：官方 `ops-precision-standard` 给标准，但你实际交付用的 gamma_k 类判据（见 `~/Projects/personal/mm-operator`）与 ULP 细粒度口径是自己的资产。
3. **对接现有交付链**：补丁 + run 包 + 交付文档（`delivery/{*_DELIVERY.md, patches/, packages/}`）这一套只有你在跑，官方 skill 到 PR 就止步了。
4. **评测闭环**：官方已做 `cann-bench`，但**场景/数据集是通用的**；用自己压测/交付过的算子做回归集（可机判 PASS/FAIL + 性能劣化门禁）是能自证价值的部分。
5. 反过来说：**"从零实现一个算子开发智能体"这件事本身已经不值得做**，做完只能证明官方做过。`[推断]`

---

## 6. 未核验清单（别当成事实）

- 没有真机跑过 CANNBot：安装、插件工作流、skill 实际生成质量**全部未验证**。目前只核到"仓库存在 + 官方自述能力"。`[未核验]`
- `cannbot-skills` README 里的三层架构图（PLUGINS/AGENTS/SKILLS）来自仓库页面转述，未逐文件核对目录。`[未核验]`
- `cann-bench` / `cannbot-sentry` 的"三层评测框架""四级复杂度""AB 门禁"只有仓库页简介，未读其文档与评测集构成。`[未核验]`
- GitHub 各小项目的 star 数与描述来自 Search API 返回，未逐个打开仓库确认能力真伪。`[未核验]`
- `Ascend/agent-skills` 的 skill 数量三处不一致（见 §1 提醒），未追到权威计数。`[未核验]`
- GitCode 站内搜索需要登录（游客返回"当前游客访问过多"），本次是靠 org 页枚举 + 搜索引擎 `site:gitcode.com` 交叉得到，**可能漏掉站内未被索引的同名项目**。`[已核验]`（受此局限）
