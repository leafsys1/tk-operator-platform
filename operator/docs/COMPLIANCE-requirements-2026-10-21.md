# 需求符合性分析（算子开发 + 自动优化）

- **分析时点**：2026-10-21 16:30–16:55（Asia/Shanghai）
- **分析对象**：`/Users/lfx/Projects/personal/op-agent`，HEAD `32b48d4` + 未提交工作树
- **数据快照**：`backend/opx.db` 只读（30 runs / 5 tasks / 5 loops / 7 spec_versions / 7 action_jobs；分析期间仍有真机作业在跑）
- **方法**：3 路独立只读审计（代码 + SQLite + 磁盘报告 + 少量只读单测/复算）+ Lead 复核；**未修改任何代码、数据库或工作区**
- **证据级别**：`[已核验]` = 本次直接读到代码/产物/DB 行/命令输出；`[文档转述]` = 只来自项目文档自述；`[推断]` = 由证据外推
- **子报告**（只读审计，未入 git）：
  - `runs/review/audit-a-generation.md`（三种输入与三件产物）
  - `runs/review/audit-a-cases.md`（10 案例与四场景）
  - `runs/review/audit-b-optimization.md`（自动优化六项）

---

## 0. 一句话结论

**按需求字面，当前项目尚未满足完整开发要求，不能整体验收通过。**

- **需求 A（算子开发能力）＝ 部分满足（偏不满足）**：自然语言→实现/接口/测试脚本的真实端到端**已跑通**（elementwise_sub 13/13、elementwise_mul 19/19），但**数学公式、参考代码两种输入只有名义通道**；**「内置 ≥10 个典型案例、覆盖矩阵/归一化/激活/注意力」为 0/10、四场景 0/4**，这是硬缺口。
- **需求 B（自动优化能力）＝ 部分满足（机制层较扎实，核心两项未达字面）**：正确性验证、优化前后精度/性能对比有真机证据；但**「自动优化」实际是 agent 只改代码、build/test/benchmark 与采纳/回退全靠人逐步触发**；**「可复现的测试报告」只有哈希/版本链，独立工作目录复现（项目验收 #10/#12）自认「未验」**；硬件画像只进 prompt，**没有「画像→优化参数」的推导路径**。

项目自己也承认主要缺口：`docs/PROGRESS.md:13` 写明「M2 案例内容 **0/10**、交付/独立目录复现（验收 #10/#12）**未验**」；`README.md:36` 把 M2 标为「🟡 框架完成，案例内容 0/10」[已核验]。

**判定汇总（按可判条目）**：13 个细项中 **5 项满足 / 7 项部分满足 / 1 项不满足**；但未满足的是需求里最硬的两块——10 个案例内容与「自动/可复现」的闭环强度。

---

## 1. 判定总表

| # | 需求条目 | 判定 | 一句话依据 |
|---|---|---|---|
| A1 | 自然语言输入 | ✅ **满足** | 草稿→对话澄清→结构化 spec→上下文空间→agent 写码→真机验证全链路；task#10 有 13/13 证据、task#11 有 19/19 证据 |
| A2 | 数学公式输入 | 🟡 **部分满足（仅名义）** | 无字段/解析/校验/LaTeX 渲染/UI tabs；公式只能当自由文本粘进对话，spec 里落成 `math_def` 字符串；无任何真实任务用公式输入 |
| A3 | 参考代码输入 | 🟡 **部分满足** | 任务对话有附件通道（≤20MB，落 `.opx-chat/`），但**不进需求澄清/spec/上下文空间/`references/`**；真实附件只有一条 17B 冒烟文本 |
| A4 | 自动生成算子实现代码 | ✅ **满足（能力）** | sub/mul 由 agent 真实写盘（逐字节比对通过）并真机 PASS；add 是人工按 CANNBot 模板改造；**只有 elementwise 一类** |
| A5 | 自动生成调用接口 | ✅ **满足** | 每算子均有参数化 CLI + README 接口契约 + `EXEC` 标记 + 拒绝码，被 `run_cases.py` 真实调用 |
| A6 | 自动生成测试脚本 | 🟡 **部分满足** | 按算子生成 `cases.json`/`golden.py`/`benchmark.json`/`torch_*_baseline.py`；`test.sh`、`run_cases.py`、`criteria/` 是 M1-a 起固定的共享资产；sub 的 13 条用例由人给出 |
| A7 | **内置 ≥10 个典型案例，覆盖四场景** | ❌ **不满足** | 内容 **0/10**、四场景 **0/4**；案例库目录/index/zip bundle **全不存在**；现有 3 个算子全是 elementwise，不属四场景 |
| B1 | 正确性验证 | ✅ **满足** | 门禁 + 用例执行器 + 判据档三层；run#4 失败（16/18）→ 修 → run#5 19/19 PASS；异常输入是「非 0 退出 + 指定拒绝码」的真拒绝；报告含 shape/dtype/误差/NaN-Inf |
| B2 | 性能瓶颈分析 | 🟡 **部分满足** | profile 真采四类指标（耗时/搬运/存储/计算负载，CSV 复算与报告一致）；结构化建议引用真实 run/指标；但**瓶颈诊断由人/agent 写进假设**，后端无自动分类；Roofline 只在前端 |
| B3 | 自动优化 | 🟡 **部分满足** | agent 真自动改代码（Loop#3/#4），一键回退可用；但 build/test/benchmark、对比落盘、采纳/回退**全部人工逐步触发**（一次 A/B ≥7–8 步）；Loop#2 还是手工改 |
| B4 | 基于目标硬件特性生成优化代码 | 🟡 **部分满足** | 硬件画像质量高、被内联进优化 prompt；代码用 `aclrtGetDeviceInfo` 取核数不硬编码；但**画像只到 prompt 文本**，tile/阈值来自实测扫描 + 人在 prompt 指定，无「画像→参数」推导 |
| B5 | 优化前后精度与性能对比 | ✅ **满足（有保留）** | 3 份真机 A/B JSON；硬校验规格/判据/镜像 digest/设备/基准用例指纹，精度退化硬拦；保留：物理卡只比台账字符串、预热/样本数只记不校验 |
| B6 | 可复现的测试报告 | 🟡 **部分满足** | manifest 有镜像 digest/源码快照 sha/判据/用例集/设备锁/产物 sha，可下载、gate/perf-diff 可复算；但 **#10/#12 未验**、30/30 run 同一工作目录、env recipe 与时间戳全 NULL、gate_report 仅 3/30 落盘 |

> 需求 A 的「三种输入 → 三件产物」按字面整体是 **部分满足（偏不满足）**；需求 B 六项是 **2 满足 / 4 部分满足**，但缺失项恰好是需求标题里的「自动」与「可复现」。

---

## 2. 需求 A：算子开发能力

### 2.1 三种输入方式：只有自然语言是真通道

**A1 自然语言 — 满足。**
- 代码链：`backend/app/routers/tasks.py:292/349`（澄清对话）、`:409`（定版）、`:439`（`input_form` 随 spec 落库）；`backend/app/services/chat.py:30-74`（澄清 prompt + 结构化输出）；`context_space.generate()` 落 `tasks/<op>/`。
- 真实证据：task#10 `elementwise_sub` 的 `raw_input` 946B、`spec_draft` 1330B，spec v1 的 `change_reason=由「任务开始」页对话定版`；agent job#25 的 dsh 会话有 9 次 `write`，其中 8 个文件与磁盘逐字节相同；run#25 `correctness.json` = **13/13 PASS**，action_job#3 门禁 **pass** [已核验]。
- task#11 `elementwise_mul` 同链：spec v1 → 对话开发 → build#28 → test#29（18/19，设备类 ERROR）→ test#30 **19/19 PASS**（16:31）[已核验]。
- 反例：task#9 `addrelu` 已定版 spec 但 `ops/addrelu` 不存在、0 run，说明「NL→spec」通了，「spec→code」仍靠人再去发起 [已核验]。

**A2 数学公式 — 部分满足（仅名义）。**
- `tasks.input_form` 只是自由字符串（`backend/app/models.py:127` 的注释；schemas 无枚举）；后端无任何行为分支。
- 前端 `TaskStart.tsx` 没有「输入方式 tabs」控件（`input_form` 只在载入/保存时读写，默认 `"自然语言"`）；实际 UI 文案是「数学公式或参考代码也可以直接粘进对话」（`TaskStart.tsx:622`）；设计稿里的 tabs 见 `docs/UI-task-start-v1.md:122`，**未落地**。
- 全仓无公式解析/校验/LaTeX 渲染（`web/package.json` 无 katex/remark-math）；DB 5 个任务的 `input_form` 无「数学」。公式只在 task#9 的自然语言正文里出现过一句，被抽成 `math_def` 字符串 [已核验]。
- 结论：能「收下公式文本」，但系统没有把它当成独立输入形态来支持。

**A3 参考代码 — 部分满足。**
- 任务对话附件可用：`conversations.py:74-105`（落 `.opx-chat/conv-N/attachments/`，≤20MB/≤20 个）、`conversations.py:301-327`（下一轮 prompt 列出附件路径）、`routers/conversations.py:96`（上传接口）；有测试覆盖 [已核验]。
- 但：需求澄清接口 `ChatRequest` 只有 `message`，不收附件；`finalize` 不会把附件复制进 `tasks/<op>/references/`；`context_space.py:167/185` 只创建占位 README；全仓没有代码读取 `references/` 参与规格或生成。
- 真实使用：`attachments` 表只有 1 行 17B 的 `note.txt`（内容 `smoke attachment`）；三个 `tasks/*/references/` 都只有 80B README；task#1 的 `input_form=参考代码 + 形状/dtype` 但 `raw_input=NULL`、spec 只有 96B 的 shape/dtype/chip 登记桩，无实现/用例/run [已核验]。
- 结论：有「传文件给 agent 看」的通道，**没有「参考代码→规格→代码」的端到端证据**。

### 2.2 三件产物

| 产物 | 判定 | 事实 |
|---|---|---|
| 实现代码 | ✅ 满足（能力） | `elementwise_sub`/`elementwise_mul` 由 agent 生成并真机 PASS；`elementwise_add` 是人工按 CANNBot 模板改造（commit `fee3900` 早于该任务所有 agent job）；三者均 7 文件（kernel/tiling/host/CMake/golden/README），模板痕迹明示（`/* Generated By CANNBot */`）[已核验] |
| 调用接口 | ✅ 满足 | 形态为 Ascend C 直调 host CLI：`elementwise_<op> --n-a N --n-b M --dtype float32 --in-a ... --out ...`；README 有「调用接口」段 + 退出码/拒绝码表；成功 stdout 打 `EXEC op=… arch=dav-2201 impl=ascendc-direct-invoke`，执行器缺标记判 `INVALID` [已核验] |
| 测试脚本 | 🟡 部分满足 | 按算子生成：`cases/<op>/cases.json`、`ops/<op>/scripts/golden.py`、`benchmark.json`、`tools/torch_<op>_baseline.py`；固定共享：`test.sh`、`tools/run_cases.py`、`tools/gen_inputs.py`、`criteria/`；sub 的 13 条用例是人在开发 prompt 里列好的，mul 才是 agent 自主设计 19 条 [已核验] |

### 2.3 「自动生成」的真实档位

- 需求字面「自动生成」= 输入后尽量少人工产出三件产物；项目实际：**agent 自动但人触发**——定版需人点、开发 kickoff 需人发、四动作需人触发或 agent 自行决定、采纳/回退权在人 [已核验]。
- 项目文档把「自动」定义为「agent 执行 + 工具判定；人保留触发、审批、回退权」（`PLAN-web-console-v1.1.md:416`、`REQUIREMENT.md:35`）——这是一个合理的范围收缩，但**与需求字面并不等价**，验收前必须对齐口径。
- 证据显示生成能力是真实的（sub/mul 会话 `write` 与磁盘逐字节相同），不是「页面假装生成」；但生成多样性只被 elementwise 一类验证，且 add 是人工模板改造。

### 2.4 10 个典型案例与四场景覆盖 —— 硬缺口

| 场景 | 计划（PLAN §13.2） | 实际内容 | 完成度 |
|---|---|---|---|
| 矩阵计算 | matmul / batched matmul | 仅 DB 里 `complex64_mm` 迁移草稿（task#1），无 ops/cases/run | **0/2** |
| 归一化 | softmax / layer_norm / rms_norm | 无任何项目内文件 | **0/3** |
| 激活函数 | relu / gelu / silu | 仅 `addrelu` 规格草稿（task#9），无实现/用例/run | **0/3** |
| 注意力机制 | attention / flash-attention | 无任何项目内文件 | **0/2** |
| **合计** | **10** | **0 个四场景案例** | **0/10（四场景 0/4）** |

- 仓库内只有 3 个 elementwise 算子（add 19 用例 / sub 13 / mul 19，共 51 条）；`PLAN:514` 明确 elementwise_add 是「回归载体与最小闭环样张，不计入 10 个」[已核验/文档]。
- 案例库目录 `~/.local/share/op-agent/case-library/` **不存在**；`index.json` 不存在；zip bundle 全盘不存在（`/Users/lfx` 下仅 pytest 临时夹具）；运行中后端 `GET /api/case-library` 返回 `{"bundles":[],"imported":false}` [已核验]。
- 框架本身可用：`scripts/import_cases.py`（校验/解包/索引）、`GET /api/case-library` + `POST /api/case-library/start`、`CasePanel`、7 个测试实测通过；但「框架完成 ≠ 内容完成」，且一键发起只建任务+拷文件，**不落 spec/上下文空间** [已核验]。
- **口径变更**：需求原文是「**内置**不少于 10 个」；项目 2026-10-20 把验收 #11 改写为「提供 ≥10 案例的**可导入 bundle**」（`PROGRESS.md:728` / `PLAN:448`）。即便按改后口径也**不满足**（bundle 根本不存在、真机全链路 0/10）。项目文档内部还有冲突：`PROGRESS.md:1194` 仍写「#11『内置 ≥10 案例』口径不变，只改先后」[已核验/文档]。
- 真实工作量：项目自评 10 个案例内容 15–25 人天，attention/flash-attention 风险最高（`PROGRESS.md:244/1188`、`PLAN:522/529`）[文档转述]。

---

## 3. 需求 B：算子自动优化能力

### 3.1 正确性验证 — 满足

- **规则层**：`backend/app/services/gate.py` —— 产物类型/哈希、退出码、版本一致（源码快照/规格/判据/镜像）、用例逐条状态（`NOT_RUN` 不算过）、指标状态、设备独占；形状/dtype/数值误差在 `tools/run_cases.py` + `criteria/check.py`；异常输入要求「非 0 退出 + 命中指定拒绝码」，不是「不崩即过」[已核验]。
- **真机证据**：run#4 `correctness.json` = 16/18（`add_norm_003`/`add_edge_005` FAIL，`max_abs_err 7.17`、`matched_ratio 0.168`）→ 修复 → run#5 **19/19 PASS**；只读复算门禁：run#4 fail（`E_EXIT_NONZERO`,`E_CASE_FAILED`）、run#5 pass；4 条异常用例分别以 `rc=4 REJECT_N_OUT_OF_RANGE` / `rc=3 REJECT_SHAPE_MISMATCH` / `rc=5 REJECT_UNSUPPORTED_DTYPE` 被按预期拒绝 [已核验]。
- 报告覆盖需求：#5「含形状、dtype、数值误差与异常行为判定」成立；失败用例、错误原因、修复后新执行记录均保留。
- 保留项：门禁不校验「用例集应覆盖多少条」，只对报告里出现的用例判状态；数值阈值唯一口径是判据档（设计选择）[已核验]。

### 3.2 性能瓶颈分析 — 部分满足

- **采集**：benchmark 用 msopprof 量设备侧 `Task Duration`（5 次取中位，含预热），派生达成带宽/算术强度；profile run#7 真采四类指标 —— 耗时 `op_time_us=7.78`、搬运 `mte2_ratio=0.7486`/`mte3_ratio=0.2845`、存储 UB/L2、计算单元 `vec_ratio=0.0921`（cube 标 `unsupported`，如实）[已核验]。
- **可信度**：审计员按 msopprof 原始 CSV 逐列复算均值，与 `metrics.json` **完全相等**，说明报告由原始数据派生、非手写 [已核验]。
- **结构化建议**：`perf_diff.py:211-257` 产出含 `affected_cases`/`observed_evidence`（run id + 具体指标值）/`bottleneck_hypothesis`/`confidence`/`proposed_change`/`validation_method`/`status` 的建议，证据可追溯 [已核验]。
- **短板**：瓶颈「诊断」不是系统自动产出——`bottleneck_hypothesis` 第 1 条直接取人写的 `loop.hypothesis`，第 2 条是写死的小 shape 启发式；后端没有任何代码读 `mte2_ratio/vec_ratio` 做分类；Roofline/极限对比只在**前端**（`ResultsPanel.tsx`），不进报告也无测试；全库 profile **只有 1 条**（run#7）[已核验]。

### 3.3 自动优化 — 部分满足（需求字面差距最大）

- **已实现**：`POST /api/loops/{id}/optimize`（`runs.py:424-460`）→ dsh agent job（`kind=优化`）**只改代码**；跑完落改前/后快照 + 逐文件 diff + 回退数据（`optimize.py:342-390`）；`POST …/optimize/revert` 一键回退；`GET /api/loops/{id}/perf-diff` 计算对比；结论由人通过 `POST /loops/{id}/close?conclusion=…` 传入 [已核验]。
- **边界（代码注释明写）**：`optimize.py:12-14` 与 `runs.py:430-431` 均写明「**不跑 build/test/benchmark**，真机执行仍由控制台单独触发」。
- **三轮真实形态**：

| Loop | 谁改代码 | 谁跑 build/test/benchmark | 谁判定/采纳 | 结论 |
|---|---|---|---|---|
| #2 | **人**（prog 自认「扮演 agent 工程动作」，`optimize_json=NULL`） | 人逐次跑 | 人 | 主用例 no_significant_change，64k 回退 +19.6%，不采纳并回退 |
| #3 | **agent**（job#16，1 文件 +13/−6） | **人逐步触发** run#15–#18 | 工具算 `regressed`，人点落盘/关闭/回退 | 不采纳，已回退 |
| #4 | **agent**（job#17，3 文件，tile 运行时化） | **人逐步触发** run#20–#23 | 工具算主用例 no_sig、64k improved −30.4%，人关闭=accepted | **首次采纳** |

- **人工步骤**：建 Loop → 写假设 → 点「让 agent 自动优化」→ 等 job → 分别点 build/test/benchmark → 点「落盘」→ 点「关闭」→（不采纳时）点「回退」；**一次完整 A/B 至少 7–8 次人工触发** [已核验]。
- 结论：这是「agent 提议/改码 + 工具判定 + 人保留触发/审批/回退」，**不是端到端自动优化**；按需求字面「自动优化」只能判部分满足。项目自定的验收口径（至少一份有效 A/B 报告、不设加速比硬线）**本身与需求字面存在差距**，验收前需书面确认。

### 3.4 基于目标硬件特性生成优化代码 — 部分满足

- **画像质量高**：`workspaces/cann-ops/ops/chips/dav-2201.yaml` 每个 fact 带 `source`/`confidence`，含 `runtime_authoritative`（要求以 `PlatformAscendC`/`aclrtGetDeviceInfo` 为准）与 `uncertain` 段；UB=196608B、向量核 48、频率 1850MHz、对齐 32B 等 [已核验]。
- **三条消费通道**：
  1. 规格引用 `target_hardware_profile`（只出现在 task#10/#11；**唯一有完整优化证据的 task#2 规格 v1 是旧 schema，未引用画像**）；
  2. `optimize.hardware_brief`（`optimize.py:245-280`）把关键事实压成一行注入优化 prompt，并约束「不得硬编码、low/medium 不当定论」；
  3. 前端 Roofline 读 `/api/chips` 做极限对比。
- **代码层面**：核数用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 取运行时真值（非硬编码 48）；但 deviceId 仍写死 0（靠远端单卡可见兜底）、UB 预算是静态分配；**没有任何代码把画像事实推导成优化参数**——Loop#4 的 tile=512/阈值 65536 来自实测扫描，且方案由人在 prompt 里直接给出，agent 照实现 [已核验]。
- 结论：机制存在（画像 + prompt + 运行时真值），但「基于硬件特性**生成**优化代码」目前依赖 agent 的通用推理与人工给方案，**没有可复现的硬件→参数推导链路**；证据也只有 1 个算子、1 轮生效优化。

### 3.5 优化前后精度与性能对比 — 满足（有保留）

- **硬校验**：配对完整（两侧各 test+benchmark）、同规格/同判据/同镜像 digest/同设备、同基准用例指纹；**精度退化硬拦 `E_PRECISION_REGRESSION`**（优化后不得有 FAIL/ERROR、逐用例不得比 before 差）；性能回退只 warn（项目已定不设加速比硬线）；不确定性参与显著性判定（差异需 > max(std/median 之和, 2%)）[已核验]。
- **3 份真机 A/B**：Loop#2 `no_significant_change`（1M 8.28→8.18us；64k +19.6% regressed）；Loop#3 `regressed`（1M 8.00→8.18us，+2.25%）；Loop#4 主用例 `no_significant_change`（8.10→8.10us）、64k **7.44→5.18us（−30.38%，improved）**，精度 19/19 不退化；三份均可只读复算出与落盘一致的结论 [已核验]。
- **保留**：
  - 「同卡」只比 `runs.device_id` **台账字符串**，未实证物理卡（历史上 run#12/#13 台账 dev1、实际 dev0；项目自己也把升级列为待办）；
  - 镜像 digest 是 `local-id:sha256:…`，同 host 可比、跨机不可移植；
  - 预热/重复次数/样本数**只记录不比对**；
  - 精度退化硬拦**只有单测**，3 轮真机均未触发过该分支 [已核验]。

### 3.6 可复现的测试报告 — 部分满足（项目自认未验）

- **已有的**：每次 run 的 manifest 带 `image_digest`、`code_snapshot_sha256`、`spec_version_id`、`criteria_ref`、`suite_ref`、设备/锁模式、逐产物 sha256；run 目录有 `docker_cmd.txt`/`run.sh`/`lock.log`；报告可经 `GET /api/artifacts/{id}/download` 下载，gate/perf-diff 可用当前代码只读复算 [已核验]。
- **缺的**：
  - **验收 #10/#12「从独立工作目录按复现说明重跑」自认「未验」**（`PROGRESS.md:13`），DB 30/30 条 run 的 `cwd` 都是同一个 `workspaces/cann-ops`，从未从独立目录重跑；
  - 无 `delivery/` 交付目录/REPRO 文档；`runs.env_recipe_snapshot` **30/30 为 NULL**；`runs.started_at/ended_at` **30/30 为 NULL**；
  - `gate_report` 落盘只有 3/30（run#1/#6/#7），无 `gate_reports` 表；异步作业只在 `action_jobs.gate_verdict` 存了 5 条 → 项目自己定的「四件全落库（runs 行 + manifest + 报告 + gate_report）」对 Loop#2–#4 的 15 条 run **不成立**；
  - 源码只有 sha256、无可运行归档；报告内容会随代码版本漂移（Loop#2 的 JSON 是旧 schema，缺 `recommendations`）[已核验]。
- 结论：**「版本可追溯」成立，「可复现」尚未被演示**；严格按需求「形成可复现的测试报告」只能判部分满足。

---

## 4. 已经扎实的部分（不应被上面的缺口淹没）

1. **执行闭环是真的**：契约算 plan → 真机执行 → 机读报告 → 门禁判定，run/manifest/artifact/门禁贯穿；真实跑了 30 条 run、466 个 artifact；失败 run 如实保留并分类（正确性失败、设备类失败）[已核验]。
2. **证据纪律好**：报告带源码快照哈希、判据档、用例集 ref、镜像 digest、设备锁；门禁明确「退出码 0 ≠ 业务通过」；异常输入真拒绝；精度退化硬拦（代码层）。
3. **自动优化的「工具判定」是机判**：`performance-diff.json` 有同条件校验、显著性、精度不退化、结论枚举；三轮 A/B 结果**如实**（包括不采纳/回退），没有为了好看伪造加速比。
4. **A/B 报告可复算**：审计用当前代码只读复算 3 份 JSON，结论/配对/精度一致；msopprof 四类指标来自原始 CSV 逐列复算 [已核验]。
5. **工程质量**：本次全量 `pytest` **277 passed**；前端 `tsc --noEmit && vite build` 通过（2026-10-21 复核）[已核验]。

---

## 5. 关键口径与风险（验收前必须定）

| # | 问题 | 影响 |
|---|---|---|
| R1 | **能力归属**：项目定位是控制台，agent/领域能力外采 DSH + CANNBot skills（40 skills + 3 agents，pin 版本）。 | 若验收要求「交付物自包含/离线可复现该能力」，需要额外确认许可、版本 pin 与依赖交付；若按「系统级能力」验收，则当前集成方式成立。 |
| R2 | **「自动」口径被收缩**：需求字面「自动生成/自动优化」 vs 项目「agent 执行 + 工具判定 + 人保留触发/审批/回退」。 | 这直接决定 A「自动生成」、B「自动优化」是判满足还是部分满足。**应在验收前书面确认**，否则按字面判会失分。 |
| R3 | **「内置」被改为「外置可导入 bundle」**：`PROGRESS.md:728` vs `:1194` 自相矛盾。 | 即便按改后口径也未达成（bundle 不存在）；按原文更不满足。对外汇报时不可引用规划清单冒充已内置。 |
| R4 | **只有 1 个算子（elementwise_add）有完整优化证据**：Loop#2–#4；sub 无 perf Loop，mul 的 Loop#5 仍 open、run 未挂 Loop。 | 「自动优化能力」的覆盖度不足，不能证明对矩阵/归一化/激活/注意力类算子同样有效。 |
| R5 | **性能证据的物理卡未实证、预热/样本数未比对、精度硬拦无真机触发**。 | 对比报告的「同条件」在严格审计下是可质疑的；建议补物理卡 ID 校验与真实精度退化演练。 |
| R6 | **案例内容工作量 15–25 人天**，attention/flash-attention 风险最高；当前工作树仍在开发 elementwise_mul。 | 10/10 是最大验收缺口，不能靠文档或页面框架补齐；需按项目建议「风险前置」排期（relu → attention → matmul …）。 |
| R7 | **时点敏感性**：本报告快照 16:30–16:55，task#11 在窗口内从 test FAIL 变为 19/19 PASS，benchmark/profile 尚未完成。 | 最终状态以复核时点为准；不要把「test 刚过」当「全链路完成」。 |

---

## 6. 补齐建议（按验收影响排序）

**P0（不做则无法整体验收）**
1. **10 个案例内容 + 10/10 真机全链路 + 四场景覆盖**：按 PLAN §13.2 清单产出 zip bundle，导入后逐案例走「规格→实现→build→test→benchmark→门禁 pass」，修掉「一键发起不落规格」；alignment：需要先和验收方确认「内置」是否接受「可导入 bundle」。
2. **数学公式 / 参考代码两条输入通道真正接入**：至少补「输入方式选择 + 参考代码附件进澄清/spec/references + 公式→math_def 的结构化映射 + 一个真实案例」，否则 A 的「三种输入」只有 1/3。
3. **独立工作目录复现演示**：按验收 #10/#12 从干净目录重跑一次（登记为 run），补齐 `delivery/` + REPRO 说明 + env recipe + 时间戳 + gate_report 全落库。

**P1（把「部分满足」变「满足」）**
4. **自动优化端到端编排**：agent 改完代码后自动串联 build/test/benchmark → 自动算 perf-diff → 自动回填 Loop 结论，人只保留审批/回退（预计可把一次 A/B 从 7–8 次人工降到 1 次审批）。
5. **硬件画像→参数推导**：把 UB 预算/核数/对齐从 prompt 文本变成 tiling 推导输入（如按 UB 预算自动选 tile 上限、按实际核数自动选 blockNum），并强制规格引用 `target_hardware_profile`。
6. **报告与门禁补全**：gate_report 全量落盘、补 `gate_reports` 表；perf-diff 增加物理卡校验与预热/样本数比对；Roofline 进报告并加测试。

**P2（提升可信度/覆盖度）**
7. 给至少 2–3 个四场景算子补 perf Loop 与 A/B 证据（不只 elementwise_add）。
8. 做一次真实的「精度退化 → 门禁硬拦」演练（当前只有单测）。
9. 同步修正文档口径：`PROGRESS.md:728` 与 `:1194` 冲突、README 的「下一步」与验收状态、`REQUIREMENT.md` 的三种输入支持现状。

---

## 7. 最终判定

| 需求块 | 判定 | 说明 |
|---|---|---|
| **算子开发能力（整句）** | **部分满足（偏不满足）** | 自然语言→三件产物有真实闭环（sub/mul 真机 PASS）；但三种输入只 1 种真通，10 个案例 0/10、四场景 0/4 —— 硬指标未达成。 |
| **算子自动优化能力（整句）** | **部分满足** | 正确性验证、A/B 对比、报告版本链扎实且有真机证据；但「自动优化」非端到端、「可复现报告」未演示、硬件驱动仅静态画像，均与需求字面有实质差距。 |
| **整体** | **未满足完整验收** | 若按项目自定 v1.2 验收清单，仍有 #11（10 案例）、#10/#12（独立目录复现）明确未达成；若按需求原文逐字，缺口更大。可考虑按「阶段性交付」验收执行闭环与 B 的机制，同时把上述 P0/P1 列为整改项。 |

> **建议的对外表述**：项目已具备「单算子从自然语言需求到真机正确性/性能证据」的完整骨架，自动优化闭环也已用 3 轮真机 A/B 证明机制可用；但需求要求的 **10 个四场景案例、三种输入全通道、端到端自动优化、独立目录可复现报告** 尚未交付，当前不能按开发要求整体验收通过。

---

## 附录 A：关键证据索引

| 主题 | 证据 |
|---|---|
| 需求原文/拆解 | `docs/REQUIREMENT.md:12`（原文）、`:34-35`（10/10 与自动口径）、`:21`（D2 案例硬指标） |
| 验收清单 | `docs/PLAN-web-console-v1.1.md:438-449`（#1–#12）、`:453`（四件全落库）、`:416`（自动口径） |
| 项目自评缺口 | `docs/PROGRESS.md:13`（M2 0/10、#10/#12 未验）、`:728`（外置 bundle）、`:1194`（口径冲突）、`README.md:36/109` |
| 三种输入 | `backend/app/models.py:127`；`backend/app/schemas.py`；`web/src/components/TaskStart.tsx:622/840-900`；`backend/app/services/chat.py:30-74`；`conversations.py:74-105/301-327`；`context_space.py:167/185` |
| 代码生成证据 | agent_jobs #25/#41 的 dsh 会话（`write` 调用 ×9，与磁盘逐字节比对）；`runs/r20261020-165548/out/correctness.json`（13/13）；`workspaces/cann-ops/runs/r20261021-163103/out/correctness.json`（19/19） |
| 调用接口/测试脚本 | `ops/*/op_host/*.asc` main + README；`tools/run_cases.py:169-297`；`criteria/check.py`；`cases/*/cases.json` |
| 10 案例/案例库 | `workspaces/cann-ops/{cases,ops}`（3 个 elementwise）；`~/.local/share/op-agent/`（无 case-library）；`GET /api/case-library`（空）；`scripts/import_cases.py`；`backend/app/services/case_library.py`；`docs/PLAN-web-console-v1.1.md:500-514` |
| 门禁/正确性 | `backend/app/services/gate.py:24-306/312-362`；`runs/r20261015-1134-test/out/correctness.json`（16/18）；`runs/r20261015-1250-test/out/correctness.json`（19/19）；`action_jobs#3 gate_verdict=pass` |
| 指标采集 | `runs/r20261016-103425/out/metrics.json` + `out/evidence/profile/*.csv`；`workspaces/cann-ops/tools/run_benchmark.py`、`run_profile.py` |
| 自动优化 | `backend/app/services/optimize.py:12-14/245-316/342-408`；`routers/runs.py:424-482`；`loops #3/#4.optimize_json`；`agent_jobs #16/#17` |
| A/B 对比 | `runs/r20261016-112226/performance-diff.json`（Loop#2）、`runs/r20261020-101447/performance-diff.json`（Loop#3）、`runs/r20261020-123736/performance-diff.json`（Loop#4）；`backend/app/services/perf_diff.py:279-380` |
| 可复现 | `runs/*/manifest.json`、`docker_cmd.txt`；`runs.env_recipe_snapshot`/`started_at`/`ended_at` 全 NULL；`find runs -name gate_report.json` 仅 3 条 |
| 硬件画像 | `workspaces/cann-ops/ops/chips/dav-2201.yaml`；`optimize.py:245-316`；`routers/chips.py`；`ResultsPanel.tsx:333-405` |

## 附录 B：本次复核命令（可重复）

```bash
cd /Users/lfx/Projects/personal/op-agent

# 数据库快照（真实库是 backend/opx.db；根目录 opx.db 是 0 字节空文件）
cd backend && python3 - <<'PY'
import sqlite3
c = sqlite3.connect('file:opx.db?mode=ro', uri=True)
print(c.execute("select count(*) from runs").fetchone())
print(list(c.execute("select id,task_id,action,status from runs order by id desc limit 10")))
print(list(c.execute("select task_id,action,status,count(*) from runs group by task_id,action,status")))
PY

# 案例库/案例内容
ls -la ~/.local/share/op-agent/
find workspaces/cann-ops/cases -maxdepth 2 -type d
find workspaces/cann-ops/ops -maxdepth 1 -type d
curl -s http://127.0.0.1:8077/api/case-library

# 项目自身回归（本次已跑：277 passed；前端 build 已通过）
cd backend && .venv/bin/python -m pytest -q
cd ../web && npm run build
```

---

**分析结论**：当前项目是一个**执行闭环与证据链做得较扎实、但内容与自动化程度尚未达到需求**的算子开发控制台原型。最大的验收阻断项是 **10 个四场景案例 0/10**；其次是 **三种输入只通了自然语言**、**自动优化仍靠人工触发**、**可复现报告未做独立目录复现**。建议先与需求方对齐 R2/R3 两项口径，再按 P0/P1 补齐；在此之前不宜对外宣称「已满足开发要求」。
