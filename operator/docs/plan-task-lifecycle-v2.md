# 任务生命周期 v2 · 设计（文档输入 / 用例确认 / 版本 / 交付）

状态：**设计定稿（2026-10-23），P0–P6 已实施并通过真机端到端验收（2026-10-23）**；记录见 PROGRESS「2026-10-23 真机端到端验收」 ｜ 依据：2026-10-23 用户口径（见 §0）
补充/替代：`PLAN-web-console-v1.1.md`（v1.2）中「任务开始 → 开发 → 验收 → 交付」相关章节；其余章节不变。
实施前需确认项：见 §13。

---

## 0. 本轮口径（2026-10-23 用户确认，原文语义）

| # | 问题 | 定稿口径 |
|---|---|---|
| 1 | 用例在哪确认 | **任务对话过程中确认**；改用例也在对话里做（方便 agent 诊断）；未确认不允许开始开发（硬门禁） |
| 2 | 确认粒度 | **整组确认，只确认 case**（精度组 / 性能组各确认一次；不逐项确认测量条件） |
| 3 | 来源冲突 | 默认：用户对话最新要求 > 上传文档 > agent 自动生成（待追认，见 §13） |
| 4 | 用例集变更 | 默认：需重新确认 + 旧证据失效；交付前用新用例集重跑 test/benchmark；不单独升版本号（待追认） |
| 5 | 需求/规格改动 | **走对话判断**：是否实质变更、变更后是否重跑，都由对话确认并留痕 |
| 6 | 小版本触发 | 采纳的优化轮次 / 交付前非优化代码变更；否决、回滚、单次 build/test 不升；大版本内递增 |
| 7 | 优化采纳后证据 | **采纳 == 候选已经跑过 test+benchmark**，候选 run 直接作为新小版本证据，**不需要重跑** |
| 8 | 版本变更后重跑范围 | **精度 test + 性能 benchmark 必须**；profile 可选（可提供）；build 为 test/benchmark 的前置 |
| 9 | 设计文档模板格式 | **docx + md 两种都提供**（下载/回传均可） |
| 10 | 上传文档格式 | 与模板一致：**docx + md** |
| 11 | 规划文档 | **agent 补全**；交付包含**用户原始文档 + agent 生成/补全文档** |
| 12 | 对话放大 | **任务详情里的对话**（独立对话页已移除） |
| 13 | 规格修改入口 | 走对话；移除页面 JSON 编辑器；agent 起草 → 用户确认 → 落版（按建议） |
| 14 | 测试管理 | 显示**精度 + 性能两类用例**，只读（编辑走对话） |
| 15 | Python 调用接口 | **封装 host API 的 Python 接口**（采用 ctypes 方案，见 §6.2）；**四动作链路不变** |
| 16 | 测试脚本 | 每算子 pytest 脚本作为交付件；共享 runner 仍是门禁权威（按建议） |
| 17 | 交付 | 新增交付动作 + 交付包（5 件产物 + manifest），可下载 |
| 18 | 存量任务 | **已有的任务全部删除**（执行前备份，见 §11） |

---

## 1. 现状差距（为什么要改）

| 现状 | 目标 |
|---|---|
| 用例矩阵是**开发阶段的交付物**（写代码时顺手写 cases.json） | 用例是**开发前的输入**：对话确认后才允许开发 |
| 任务开始只能对话/粘贴文本，无文档输入 | 可选上传《算子设计文档》（docx/md），有模板下载 |
| 测试管理页只有精度用例，性能用例只在结果视图 | 测试管理页并列显示精度 / 性能两类用例 |
| 只有「规格版本」这一种版本概念 | 大版本（规格）+ 小版本（优化/交付前变更），详情页标题显示 `v大.小` |
| 规格修改有 JSON 编辑器，改完直接落版 | 规格修改走对话，agent 起草 + 用户确认 + 落版 |
| 交付只有展示页（DeliveryPanel），四动作没有 delivery，阶段四永远不达成 | 新增交付动作 + 交付包 + 阶段四门禁 |
| 调用接口 = C++ host CLI；测试 = 共享 `test.sh` + `run_cases.py` | 交付物新增 **Python 调用接口 + pytest 测试脚本**（四动作不变） |
| agent 交付只有代码 | 交付 5 件：算子代码 + 调用接口 + 测试脚本 + 算子说明文档 + 规划文档 |

---

## 2. 目标流程

```
① 任务开始（草稿任务）
   ├─ [可选] 下载模板 → 填《算子设计文档》→ 上传（docx/md，≤20MB）
   ├─ 对话澄清（窗口可放大）
   ├─ 用例规划
   │    ├─ 有文档：agent 按文档生成精度组 + 性能组
   │    ├─ 没文档：agent 先问「你是否有测试用例」
   │    ├─ 用户没有 → agent 自动生成 → 用户整组确认（可要求修改，循环直到确认）
   │    └─ 确认后 agent 落 cases/<op>/cases.json + benchmark.json，写确认记录
   └─ 定版（finalize）
        ├─ 校验：规格 ready（用例确认不在定版时拦，见 §3.3 实现口径）
        ├─ 落 spec v1 → 大版本 v1.0
        ├─ 生成 tasks/<slug>/ 上下文空间
        └─ 草稿转正 → 进入任务详情，第一件事：任务对话里确认用例组
              ↓
② 开发（对话）
   ├─ 算子代码（AscendC）
   ├─ Python 调用接口
   ├─ pytest 测试脚本（读 cases.json）
   ├─ 算子说明文档
   └─ 规划文档（补全用户文档 / 无文档时按模板生成）
              ↓
③ 动作链（v2 追加 smoke）：build → **smoke（编译 + 一条最小用例跑通）** → test（精度全部用例）
   → benchmark（性能）→ profile（可选）；smoke 未 PASS 不允许进入 test/benchmark
              ↓
④ 验收 / 优化
   ├─ 优化轮次：候选验证跑 test+benchmark → 采纳 → 小版本 +1（证据直接用候选 run）
   └─ 交付前代码变更（非优化）→ 对话确认 → 小版本 +1 → 重跑 build+test+benchmark
              ↓
⑤ 交付（第五个动作 delivery）
   ├─ 校验：当前版本证据有效（test 全过 + benchmark 门禁 pass）+ 5 件产物齐备
   ├─ 打包 zip + manifest → 登记 artifact → 下载
   └─ 版本冻结：再改代码必须新小版本 + 重跑 + 重新交付
```

---

## 3. 任务创建：文档 + 对话 + 用例确认

### 3.1 设计文档上传（可选）

- 位置：任务开始页（草稿任务，`tasks.is_draft=1` 期间）。
- 格式：`.docx` / `.md`；单文件 ≤20MB；创建期允许 1 份设计文档（可删除/替换，定版后走对话附件补充）。
- 存储（工作区）：
  ```
  <workspace>/task-docs/task-<task_id>/
      <原名>.docx|.md          # 原文，交付时原样带上
      <stem>.extracted.md      # 解析稿：docx 的段落 + 表格转 Markdown；md 原样复制
  ```
- 数据库：新表 `task_documents`（见 §9）。
- 解析：后端用 `python-docx` 读段落与表格，输出 `extracted.md`；解析失败 → 上传报错且不落库（原文可保留待手工处理）。
- 上下文：`context.yaml` 增 `pointers.docs: ../../task-docs/task-<id>/`；agent 的创建期 prompt 注入文档路径；agent 用文件读取工具读解析稿（不把全文塞进每轮 prompt）。
- 交付：原始文件 + agent 补全稿都进交付包（§7.2）。

### 3.2 设计文档模板

- 存放：`docs/templates/算子设计文档模板.docx`、`docs/templates/算子设计文档模板.md`（本仓已生成）。
- 下载：任务开始页「下载模板」按钮（docx / md 两个），后端 `GET /api/templates/operator-design-doc?format=docx|md` 直出文件。
- 章节结构（按用户给的「背景—分析—设计—验证」主线）：
  1. 需求背景（需求来源、业务背景、数学语义、历史 TBE 实现与现状分析：功能范围 / 输入输出 dtype、format、shape / 计算流程）
  2. 需求分析（外部组件依赖、内部适配模块、需求模块设计、算子原型：输入/输出/属性/推导规则）
  3. 需求详细设计（使能方式、总体设计、支持硬件、约束限制；Host 侧：tiling 与分核、shape/dtype 校验、workspace；Kernel 侧：数据搬运、向量计算流水、精度处理、结果写回）
  4. 特性交叉分析（混合精度、图模式、动态 shape、其他）
  5. 可维可测分析（精度标准、性能标准、兼容性：接口 / 数据类型 / 硬件形态）
  6. 附录 A（可选）：测试用例要点（精度组表格 + 性能组表格，供 agent 生成用例）
  7. 附录 B（可选）：参考资料
- 模板只给结构与填写说明，**不预填业务内容**；用户不填的章节由 agent 在补全稿里补。

### 3.3 用例规划与确认（对话内，硬门禁）

流程（创建期对话）：
1. agent 读完规格草稿 +（可选）文档解析稿后，判断来源：
   - 文档里有测试用例（附录 A）→ 按文档生成；
   - 文档没用例 → **明确问用户「你是否有精度/性能测试用例」**；
   - 用户没有/不提供 → agent 按规格 + 判据档自动生成（维度：常规 / 边界 / 数值边界 / 异常 / 回归，性能用例含 n/dtype/主用例/必需）。
2. agent 在对话里**以整组清单形式**呈现：
   - 精度组：case_id、分组、shape、dtype、输入生成与种子、期望（match/reject）、说明；
   - 性能组：case_id、n、dtype、primary、required、说明。
3. 用户可要求增删改 → agent 调整后再次呈现；用户回复「确认」→ agent：
   - 写 `cases/<op>/cases.json` + `cases/<op>/benchmark.json`；
   - 调用 `POST /api/tasks/{id}/cases/confirm`（带两文件内容哈希）；
   - 后端写 `tasks.cases_confirmed_at / cases_confirmed_sha256`，并在 `stage_events` 留一条记录。
4. **硬门禁位置（2026-10-23 实施口径）**：创建期对话是无工具的结构化澄清（只产出 spec 草稿），
   写 `cases.json`/`benchmark.json` 需要工具 → 所以用例确认放在**建任务后的任务对话**里：
   - `finalize` **不拦**用例确认（只拦 spec 未 ready / 重复定版）；
   - 四动作（build/test/benchmark/profile）的 plan 在 `cases_confirmed_at` 为空时直接报缺项
     「用例组未确认」——没确认就跑不了任何真机动作（等价于「未确认不允许开始开发/验证」）；
   - 任务详情对话的 prompt 会注入「未确认 → 第一优先级先确认用例组」，这是软引导，硬约束在 plan；
   - 阶段一「需求与设计」的推导门禁也要求 `cases_confirmed_at` 非空。
5. 建任务后改用例：同样走对话 → agent 改动并重新调 confirm；用例集哈希（suite_ref）变化 → 旧 test/benchmark 证据失效，页面提示需重跑；不单独升版本号（默认口径，§13 待追认）。

确认的口径：**整组确认，只确认 case**。性能用例的测量条件（baseline、warm-up、重复次数、聚合、设备）用 `benchmark.json` 默认模板，不要求用户逐项确认；用户明确要求改时走对话。

---

## 4. 版本模型

### 4.1 定义

- **大版本 M** = 当前规格版本号（`spec_versions.version`）。任务创建时定版 = v1；每次落新规格 = M+1。
- **小版本 m** = 任务内的小版本计数（`tasks.version_minor`），初始 0；大版本变更时归 0。
- 显示：任务详情页左上标题 = `{任务标题（或算子名）} v{M}.{m}`；任务列表、报告、交付包同名带上版本。

### 4.2 触发矩阵

| 事件 | 大版本 | 小版本 | 证据处理 |
|---|---|---|---|
| 规格落版（对话确认实质变更） | +1 | 归 0 | 旧 run 因 spec_version_id 不一致失效；是否重跑由对话确认（默认重跑 test+benchmark+build，可豁免并留痕） |
| 用例集修改（确认后） | 不变 | 不变 | `suite_ref` 变化 → 旧 test/benchmark 证据失效；交付前必须用新用例集重跑 |
| 优化轮次采纳 | 不变 | +1 | **候选 run（同一 code_snapshot + suite_ref）直接作为新版本证据，不重跑** |
| 优化轮次否决 / 回滚 | 不变 | 不变 | 无 |
| 交付前非优化代码变更（经对话确认） | 不变 | +1 | 必须重跑 build + test + benchmark（profile 可选） |
| 纯文档 / 说明更新 | 不变 | 不变 | 不影响证据 |

### 4.3 证据有效性判定（交付门禁的核心）

一次 run 对「当前版本」有效，当且仅当：

```
run.action ∈ {build, test, benchmark}         # profile 可选
run.status == "ok" 且 gate.verdict == "pass"
run.spec_version_id == 当前规格版本 id
run.code_snapshot_sha256 == 当前版本的 code_snapshot_sha256
run.suite_ref == 当前用例集 suite_ref
```

- 小版本记录（`task_versions`）绑定 `code_snapshot_sha256`；采纳优化时直接取候选 run 的 code snapshot，所以跑过即有效。
- 不按 run tag 打版本号，避免「先跑后升版」顺序问题；版本与证据靠快照/规格/用例集三键匹配。
- 规格变更后如对话确认「不需重跑」（例如只改文档性描述），写一条豁免记录：
  `{actions: [...], reason, message_id, at}` 记到 `task_versions.waivers_json`；门禁据此放行并在交付 manifest 里如实记录。

### 4.4 重跑与门禁

- 交付前若证据失效：交付面板列出缺哪个动作、哪个版本维度不一致，提供「按当前版本重跑」按钮（依次 build → test → benchmark；profile 可选勾选）。
- 默认**硬门禁**：不满足不允许交付；对话确认的豁免是唯一例外，且必须留痕。
- 重跑范围：精度 test + 性能 benchmark 必须；profile 可选；build 作为 test/benchmark 的前置（干净容器会自动补编，但仍建议重跑并登记）。

### 4.5 版本记录与展示

- 新表 `task_versions`（§9）保存历史：`major.minor`、来源（`spec|optimization|manual`）、`spec_version_id`、`code_snapshot_sha256`、`suite_ref`、关联 `loop_id`、备注、豁免记录、是否当前。
- 任务详情「版本」分区：上半部分任务版本列表（当前版本高亮，显示来源/关联 run/交付状态），下半部分保留现有 savepoints（源码存档）。
- 每个小版本建议自动关联一个 savepoint（采纳优化/手工变更时 `auto_capture`），便于回滚。

---

## 5. 规格修改走对话

- `SpecPanel` 移除 JSON 编辑器和「新建版本」表单，改为只读展示 + 「修改需求/规格（走对话）」按钮：跳转任务对话并预填上下文（当前规格摘要 + 让用户说清要改什么）。
- 对话中的协议（写进 agent 系统提示词）：
  1. 先澄清要改什么；判断是**实质变更**还是文档性修订；
  2. 实质变更 → agent 起草新规格（完整 spec_json，缺失标 UNKNOWN）→ 在对话里给摘要与差异 → 用户确认 → agent 调 `POST /api/tasks/{id}/specs` 落新版（`change_reason` 写明对话确认）→ 大版本 +1、小版本归 0；
  3. 落版后 agent 提醒：用例需要重新确认（大版本变更 → 用例确认状态置为未确认，force 重新确认），以及是否需要重跑（默认需要；用户可在对话里确认豁免）；
  4. 文档性修订（不影响语义/接口/判据）→ 可不落规格版本，仅在对话留结论（是否落版由用户确认，避免版本膨胀）。
- 高级入口保留：`POST /api/tasks/{id}/specs` 仍可被脚本直接调用（不诱导页面使用）。

---

## 6. 开发交付物（对话内产出）

### 6.1 交付清单

| 产物 | 形态 | 说明 |
|---|---|---|
| 算子代码 | `ops/<op>/`（kernel / tiling / host / CMakeLists） | 沿用现有 AscendC 直调工程 |
| Python 调用接口 | `ops/<op>/python/<op>.py` + `lib<op>_host.so` | ctypes 封装 host API（§6.2） |
| 测试脚本 | `tests/test_<op>.py`（pytest）+ `tests/cases.json` + `tests/benchmark.json` + `tests/golden.py` | 交付件；门禁仍走共享 runner |
| 算子说明文档 | `docs/算子说明文档.md`（接口/调用示例/约束/精度与性能口径） | 由 `ops/<op>/README.md` 升级/派生 |
| 规划文档 | 用户原始文档（若有）+ agent 补全稿 `docs/规划文档-补全.md` | 无上传时 agent 按模板生成完整规划文档 |

### 6.1.1 开发内冒烟 smoke（2026-10-23 追加）

用户要求：算子开发完先编译 + 一条最小用例确认能跑通，避免后续 test/benchmark 才发现构建失败/跑不通。

- 新增动作 **`smoke`**（第 5 个受控动作，入口脚本 `smoke_check.sh`；根目录 `smoke.sh` 是环境自检，别混）：
  - 内容：容器内编译（缺产物自动补编）+ 从 `cases/<op>/cases.json` 挑**一条最小 required match 用例**
    （n 最小、n_a==n_b），用与正式 test 相同的输入生成/候选调用/参考/判据跑一次；
  - 产物：`smoke.json`（机读，含版本对齐字段、case、verdict）+ `smoke.md`；门禁 `verdict=PASS` 才算过；
  - 与正式测试的关系：smoke 只 1 条用例，**不替代 test**；test 仍跑全部用例 + 逐例结果。
- 阶段门禁：**开发 = build 门禁 pass + smoke 门禁 pass**（此前只有 build）。
- Agent 协议：开发对话里代码写完 → `scripts/nm.sh --op <op> --action build` → `--action smoke`；
  smoke PASS 后才允许进入 test/benchmark，也才允许说「开发完成」；失败先修，设备被占如实报阻塞。
- 前端执行步骤：开发 / 编译 / **冒烟** / 正确性 / 基准 / 采集 / 交付（页面可手动触发，也支持 agent 自动跑）。

### 6.2 Python 调用接口（ctypes 方案）

- Host 侧新增导出 C ABI（同一次编译产出两个目标）：
  ```
  int <op>_run(const void* a, const void* b, void* out, int64_t n, int dtype_code, int device_id);
  int <op>_last_error(char* buf, int buflen);
  ```
  编出 `lib<op>_host.so`；现有可执行文件保留，四动作仍调可执行文件（**四动作链路不变**）。
- Python 侧 `ops/<op>/python/<op>.py`：
  ```python
  import torch, torch_npu
  from . import _lib  # ctypes.CDLL

  def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      # 校验 dtype/shape/连续性 → torch_npu.empty_like → _lib.<op>_run(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, dtype_code, dev)
      # 返回 out；失败按拒绝码抛 ValueError/RuntimeError
  ```
- 选择 ctypes 的原因：镜像里没有 pybind11 依赖，C ABI 零额外依赖；接口语义对齐 torch（同 shape/dtype、不广播、拒绝码）。
- **风险 / 前置**：Python 进程里 torch_npu 已初始化 ACL 上下文，ctypes 直调 host API 必须在同一 device 上下文（先做一次真机 spike 验证，再批量要求 agent 生成；spike 不通过则退回「Python 子进程 wrapper」备选）。
- 四动作不受影响：`test.sh` / `benchmark.sh` 仍走可执行文件 + `run_cases.py`，避免改测量口径。

### 6.3 测试脚本（pytest 交付件）

- `tests/test_<op>.py`：
  - 读 `cases.json`，逐用例走 **Python 调用接口** 跑候选，`golden.py`（torch CPU）跑参考，按 `criteria/check.py` 同口径判 PASS/FAIL/拒绝码；
  - `pytest -q tests/` 可直接在算子开发镜像里跑（不依赖控制台）；
  - 与共享 runner 读同一份用例文件，口径一致；**门禁权威仍是 `test.sh` + `run_cases.py`**（四动作不变）。
- 交付包必须包含 pytest 脚本与用例文件；agent 开发完成后应自行跑一次 pytest 并报告结果（作为对话证据，不替代四动作）。

---

## 7. 交付（第五个动作 delivery）

### 7.1 动作定义

- 在 `ACTIONS` 中新增 `delivery`，但它是**本地动作**（后端打包，不连远端容器、不需要设备）：
  - 前置：规格版本存在；用例已确认；当前版本证据有效（§4.3，或存在对话确认的豁免）；5 件产物齐备。
  - 产物：`delivery`（zip）、`manifest`（JSON）。
  - 落库：`runs`（action=delivery）+ artifacts；版本标记 `delivered_at`。
- 阶段看板「四 · 交付」门禁改为：最近一次 delivery run 门禁 pass 且版本与当前一致（不再永远不达成）。
- 交付门禁 fail code（草案）：`E_CASES_UNCONFIRMED`、`E_VERSION_EVIDENCE_STALE`、`E_DELIVERY_DOCS_MISSING`、`E_DELIVERY_PYTHON_IFACE_MISSING`、`E_DELIVERY_TESTS_MISSING`、`E_DELIVERY_PACKAGE_FAILED`。

### 7.2 交付包结构

```
<op>_v<M>.<m>_<yyyymmdd>/
  ops/<op>/                     # 算子代码（不含 build 产物）
  ops/<op>/python/              # Python 调用接口 + lib<op>_host.so（可选带 .so）
  tests/                        # cases.json / benchmark.json / golden.py / test_<op>.py
  docs/算子说明文档.md
  docs/规划文档-原始.<docx|md>   # 用户上传原文（若有）
  docs/规划文档-补全.md          # agent 补全/生成
  evidence/                     # 可选：最近一次 test/benchmark 机读报告（脱敏版）
  manifest.json                 # 版本/规格/快照/用例集/镜像/门禁结论/文件哈希
```

- 打包位置：`archives/deliveries/task-<id>/v<M>.<m>/<zip>`；提供下载 URL。
- 交付后版本冻结：任务新增代码变更 → 小版本 +1 → 必须重跑 → 重新交付（新包覆盖/并列历史）。
- manifest 必须记录：`major.minor`、`spec_version_id + sha256`、`code_snapshot_sha256`、`suite_ref`、`image_digest`、门禁 verdict、豁免记录、文件清单 sha256。

---

## 8. UI 调整

| 页面/组件 | 改动 |
|---|---|
| 任务开始页 | 上传设计文档（docx/md、进度/解析状态/删除）；下载模板（docx/md）；用例确认在对话里完成（整组清单 + 确认）；对话窗口可放大 |
| 任务详情 · 头部 | 左上标题显示 `{任务名称} v{M}.{m}`（大版本=规格、小版本=优化/变更）；tooltip 显示版本来源与时间 |
| 任务详情 · 对话 | 对话窗口放大：全屏浮层（Esc 关闭）+ 可拖拽高度；保留 SSE/断线重连；「开发」kickoff 提示词更新为 5 件交付物 |
| 任务详情 · 需求与规格 | 只读展示 + 「修改需求/规格（走对话）」按钮；移除 JSON 编辑器 |
| 任务详情 · 测试与结果 | 精度用例 + 性能用例摘要（两类） |
| 测试管理页 | 两个 tab：「精度用例」（现有矩阵 + 逐例结果）与「性能用例」（benchmark.json 用例 + 最近实测/基线/加速比/门禁，只读） |
| 任务详情 · 版本 | 版本列表（v大.小、来源、关联 run、交付状态）+ 现有 savepoints |
| 任务详情 · 交付 | 交付门禁预览（缺什么/证据是否有效）、「按当前版本重跑」、「生成交付包」、交付历史下载 |
| 执行五步 | 变为「开发 + 四动作 + 交付」（交付为本地动作，不占设备） |

---

## 9. 数据模型 / API 变更（草案）

### 9.1 新表

```
task_versions
  id, task_id FK, major INTEGER, minor INTEGER, source VARCHAR(32)  -- spec|optimization|manual
  spec_version_id FK NULL, code_snapshot_sha256 VARCHAR(80) NULL, suite_ref VARCHAR(200) NULL
  loop_id FK NULL, label VARCHAR(200) NULL, note TEXT NULL
  waivers_json TEXT NULL            -- 对话确认的证据豁免记录
  delivered_at DATETIME NULL
  created_at DATETIME, is_current BOOLEAN
  UNIQUE (task_id, major, minor)

task_documents
  id, task_id FK, filename VARCHAR(300), kind VARCHAR(32) DEFAULT 'design'
  path VARCHAR(600), extracted_path VARCHAR(600) NULL
  sha256 VARCHAR(80), size_bytes BIGINT, created_at DATETIME

deliveries（若 delivery 走 runs 表则只存扩展元数据；二选一，实施时定）
  id, task_id, version_id FK task_versions, run_id FK runs NULL
  zip_path VARCHAR(600), sha256 VARCHAR(80), size_bytes BIGINT
  manifest_json TEXT, created_at DATETIME
```

### 9.2 表字段

- `tasks`：`version_minor INTEGER DEFAULT 0`、`cases_confirmed_at DATETIME NULL`、`cases_confirmed_sha256 VARCHAR(80) NULL`、`cases_confirmed_by VARCHAR(80) NULL`（对话 message/job 引用）。
- `spec_versions`：沿用；大版本 = `version`；`change_reason` 记录「对话确认」。
- `runs`：沿用 `spec_version_id / code_snapshot_sha256 / suite_ref` 做证据匹配；action 枚举增加 `delivery`。

### 9.3 API 变更

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/templates/operator-design-doc?format=docx|md` | 下载模板 |
| POST | `/api/tasks/{id}/documents` | 上传设计文档（multipart） |
| GET | `/api/tasks/{id}/documents` | 列表（含解析状态） |
| GET | `/api/tasks/{id}/documents/{doc_id}/download` | 下载原文 |
| DELETE | `/api/tasks/{id}/documents/{doc_id}` | 删除（定版前可） |
| POST | `/api/tasks/{id}/cases/confirm` | 用例组确认（写确认记录；校验哈希） |
| POST | `/api/tasks/{id}/finalize` | 增加用例确认硬门禁 |
| GET | `/api/tasks/{id}/versions` | 版本列表（含当前版本、证据状态、交付状态） |
| POST | `/api/tasks/{id}/versions` | 手工小版本（交付前非优化代码变更，对话确认后调用） |
| GET | `/api/tasks/{id}/test-cases` | 精度 + 性能两类用例（新页面数据源；也可扩展现有 `/cases`） |
| POST | `/api/tasks/{id}/deliveries` | 创建交付动作（异步） |
| GET | `/api/tasks/{id}/deliveries` | 交付历史 |
| GET | `/api/deliveries/{id}/download` | 下载交付包 |

### 9.4 Agent 协议（写进创建/开发/变更 prompt）

- 创建期：文档解析稿路径 + 「生成精度/性能用例 → 整组呈现 → 用户确认 → 落文件 → 调 confirm」+ 「无文档时先问用户是否有用例」。
- 开发期：5 件交付物清单 + Python 接口规范（ctypes + C ABI）+ pytest 规范。
- 变更期：规格修改协议（§5）+ 用例修改协议（§3.3）+ 版本/重跑建议（由用户确认）。

---

## 10. 门禁规则汇总

| 门禁 | 新增/修改 |
|---|---|
| 定版（finalize） | **新增**：用例已确认（否则 409） |
| build | 不变 |
| test | 不变（新增：交付门禁检查 test 证据版本有效） |
| benchmark | 不变 |
| delivery | **新增**：版本证据有效（§4.3）+ 5 件产物齐备 + 用例已确认 + 生成包成功 |
| 阶段四 · 交付 | 由「永远不达成」改为「最近一次 delivery 门禁 pass 且版本一致」 |

---

## 11. 迁移与清理（存量任务全部删除）

> **破坏性操作，执行前先备份，且单独向用户确认一次执行清单。**

1. 备份：
   - `backend/opx.db` → `backend/opx.db.bak-20261023-pre-v2`；
   - `workspaces/cann-ops/` 整目录 tar.gz → `archives/workspaces-cann-ops-20261023-pre-v2.tar.gz`。
2. 删除任务（5 个：complex64_mm / elementwise_add / addrelu / elementwise_sub / elementwise_mul）：
   - 走 `services/task_deletion` 的删除流程（DB 行 + 上下文空间 `tasks/<slug>` + `.opx-chat/conv-*` + 本地 `runs/<run_ref>`）；远端 run 目录与 dsh JSONL 保留（v1 口径）；
   - 清理表：tasks / loops / agent_jobs / runs / artifacts / stage_events / messages / conversations / attachments / spec_versions / savepoints / optimization_ideas / idea_searches / loop_budget_audits。
3. 工作区清理（待确认）：
   - 建议同时清掉旧任务产物，保证新流程从零验证：`cases/elementwise_*`、`ops/elementwise_*`、`tools/torch_*_baseline.py`、`tasks/*`、`.opx-chat/*`、`runs/*`；
   - 保留共享资产：`scripts/`、`tools/run_cases.py|run_benchmark.py|run_profile.py|gen_inputs.py`、`criteria/`、根入口 `build.sh/test.sh/benchmark.sh/profile.sh`、`.dsh/skills`。
   - 若想保留 3 个 elementwise 示例作参考，则只清 `tasks/`、`.opx-chat/`，并在新任务创建时避免同 op 冲突（新任务复用前先确认覆盖）。
4. 迁移脚本：Alembic 新 revision（新表 + tasks 字段 + action 枚举无 DB 约束不用改）；清理脚本 `scripts/reset_tasks.py`（带 `--dry-run` 与备份校验）。

---

## 12. 实施阶段（建议顺序）

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 模板文件 + 模板下载 API + 文档上传/解析/存储 + 任务开始页 UI | 上传 docx/md 能解析成 md，原文可下载；无文档流程不受影响 |
| P1 | 用例确认协议（对话 prompt）+ confirm API + 四动作 plan 硬门禁 | 不确认用例无法跑四动作/阶段一不达成；确认后可进入开发；改用例需重新确认 |
| P2 | 版本模型：task_versions + tasks.version_minor + 标题/版本列表 + 证据有效性判定 + 豁免记录 | 采纳优化→小版本+1 且不重跑；手工变更→+1 且要求重跑；规格→大版本+1、小版本归零 |
| P3 | 开发交付物扩展：Python ctypes 接口 + pytest 脚本 + 说明/规划文档 prompt（先做一次真机 spike） | 新任务由 agent 生成 5 件产物；pytest 可跑；四动作不受影响 |
| P4 | delivery 动作 + 打包 + 交付门禁 + 阶段四 + 交付历史下载 | 门禁拦住证据失效/缺产物；交付包可下载且 manifest 自洽 |
| P5 | UI 收口：对话放大、规格改对话、测试管理双类、交付面板、执行步骤 | `tsc` + `pytest` 全过、页面手工验收 |
| P6 | 迁移清理（§11）+ 端到端验收 | 新任务全流程：文档→用例确认→开发→四动作→优化采纳→交付；旧任务已清空 |

---

## 13. 默认口径与待追认项

以下为用户未明确回答、本设计先按默认处理，实施前请确认（不确认则按默认）：

1. **来源冲突优先级**（用户对话最新要求 > 上传文档 > agent 自动生成）。
2. **用例集变更**：重新确认 + 旧证据失效 + 交付前重跑 test/benchmark；不单独升小版本。
3. **硬门禁 + 对话豁免**：版本证据失效默认阻止交付；对话确认的豁免写入 `waivers_json` 并在 manifest 留痕。
4. **docx 解析深度**：段落 + 表格转 Markdown 文本；不做强结构化（用例仍由 agent 生成后整组确认）。
5. **规格落版**：agent 起草 → 用户对话确认 → agent 调 `POST /specs` 落版；文档性修订可不落版本（对话判断）。
6. **交付包内容**：5 件产物 + manifest（可选附证据 JSON）；交付后版本冻结。
7. **工作区清理范围**（§11.3）：默认清掉旧算子产物，只保留共享资产。
8. **Python 接口**：ctypes + C ABI；若真机 spike 发现 ACL 上下文不兼容，退回子进程 wrapper（届时再确认）。
