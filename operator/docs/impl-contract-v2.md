# 任务生命周期 v2 · 实现契约（2026-10-23）

> 设计依据：`docs/plan-task-lifecycle-v2.md`。本文件是**并行实现的接口契约**；实现时以本文件为准，
> 冲突处以「先落库的代码 + 本契约」协调，任何人改契约先在这里加一行记录。

## 1. 写作用域（谁改哪些文件，别越界）

| 工作流 | 负责人 | 写作用域 |
|---|---|---|
| 数据模型 / 迁移 / 版本 / 交付 / 门禁 / 集成 | Lead | `backend/app/models.py`、`backend/alembic/versions/*`（新迁移）、`backend/app/services/versions.py`、`backend/app/services/delivery.py`、`backend/app/services/gate.py`、`backend/app/services/stageboard.py`、`backend/app/services/exec_actions.py`、`backend/app/routers/tasks.py`、`backend/app/routers/templates.py`（若需要）、`backend/app/routers/versions.py`、`backend/app/routers/deliveries.py`、`backend/app/services/case_report.py`、`backend/app/main.py`、`backend/app/schemas.py`、`backend/tests/test_versions*.py`、`backend/tests/test_delivery*.py` |
| 文档上传 / 解析 / 用例确认 | T1 | **新建文件**：`backend/app/services/documents.py`、`backend/app/routers/documents.py`、`backend/app/services/case_confirm.py`、`backend/tests/test_documents.py`、`backend/tests/test_case_confirm.py`；允许在 `backend/app/main.py` 只加 include 行（与 Lead 协调） |
| 前端 UI | T2 | `web/src/**`（全部） |
| Agent 提示词 / 交付物规范 | T3 | `backend/app/services/chat.py`、`backend/app/services/conversations.py`、`backend/app/services/optimize.py`、`backend/app/services/agent_runner.py`（如需）、`workspaces/cann-ops/AGENTS.md`、`docs/reference/deliverable-standards.md`（新建） |

## 2. 数据模型（Lead 落）

### 2.1 `tasks` 新列

| 列 | 类型 | 语义 |
|---|---|---|
| `version_minor` | INTEGER NOT NULL DEFAULT 0 | 小版本号（大版本 = 当前 spec_versions.version） |
| `cases_confirmed_at` | DATETIME NULL | 用例组确认时间（UTC） |
| `cases_confirmed_sha256` | VARCHAR(80) NULL | 确认时用例集哈希（= cases/<op> 目录哈希前 80 位中的 sha256 hex；见 §5.1） |
| `cases_confirmed_by` | VARCHAR(120) NULL | 确认来源（如 `chat:job#123` / `api`） |

### 2.2 新表 `task_versions`

| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| task_id | FK tasks.id CASCADE，index | |
| major | INTEGER | = spec 版本号 |
| minor | INTEGER | |
| source | VARCHAR(32) | `spec` / `optimization` / `manual` |
| spec_version_id | FK spec_versions.id NULL | |
| code_snapshot_sha256 | VARCHAR(80) NULL | 该版本代码快照 |
| suite_ref | VARCHAR(200) NULL | 用例集 ref（`cases@<hash16>`） |
| loop_id | FK loops.id NULL | 采纳优化时关联 |
| label | VARCHAR(200) NULL | 人看的名字 |
| note | TEXT NULL | |
| waivers_json | TEXT NULL | 对话确认的证据豁免 `[{actions, reason, job_id, at}]` |
| delivered_at | DATETIME NULL | 交付冻结时间 |
| is_current | BOOLEAN NOT NULL DEFAULT 0 | 当前版本（每 task 唯一） |
| created_at | DATETIME | |

- **当前版本** = `is_current=1` 的那行；任务定版时创建 `(major=spec.version, minor=0, source=spec)`。
- 落新规格（`POST /tasks/{id}/specs`，set_current=true）→ 旧 current 置 0；新建 `(major=新spec.version, minor=0, source=spec)`；`tasks.version_minor=0`；`cases_confirmed_*` 清空（要求重新确认）。
- `POST /loops/{id}/close?conclusion=accepted*`（kind=perf）→ 小版本 +1，新建 `(major=当前spec, minor=旧+1, source=optimization, loop_id, code_snapshot=当前工作区快照)`；若该 loop 已有对应版本则不重复（幂等按 loop_id 查）。
- `POST /tasks/{id}/versions`（手工小版本）→ minor+1，source=manual，快照=当前工作区。
- 版本号展示：`v{major}.{minor}`。

### 2.3 新表 `task_documents`

| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | |
| task_id | FK tasks.id CASCADE，index | |
| filename | VARCHAR(300) | 原名 |
| kind | VARCHAR(32) DEFAULT 'design' | design / other |
| path | VARCHAR(600) | 相对工作区：`task-docs/task-<id>/<filename>` |
| extracted_path | VARCHAR(600) NULL | 相对工作区：`task-docs/task-<id>/<stem>.extracted.md` |
| sha256 | VARCHAR(80) | 原文 sha256 |
| size_bytes | BIGINT | |
| created_at | DATETIME | |

### 2.4 `runs.action` 增加 `delivery` 与 `smoke`（均为字符串枚举，无 DB 约束，无需迁移）

- `smoke`（2026-10-23 追加）：开发内冒烟动作，入口 `smoke_check.sh`；产物 `smoke`（smoke.json）+ `log`；
  阶段「开发」门禁 = build + smoke 都 pass。与 `delivery` 不同，smoke 走远端容器（需要空闲设备）。

## 3. 后端 API 契约

### 3.1 模板（T1）

```
GET /api/templates/operator-design-doc?format=docx|md
→ FileResponse（文件名「算子设计文档模板.docx/md」，application/vnd.openxmlformats-officedocument.wordprocessingml.document / text/markdown）
文件源：<repo_root>/docs/templates/算子设计文档模板.docx|.md
```

### 3.2 设计文档（T1）

```
POST /api/tasks/{task_id}/documents        multipart: file
→ 201 {id, task_id, filename, kind, path, extracted_path, sha256, size_bytes, created_at}
   仅 .docx/.md，≤20MB；docx 用纯 stdlib（zipfile + ElementTree）抽段落+表格转 markdown；
   解析失败 → 400 且不落库；同名文件覆盖（先删旧行+文件再写）。
   草稿任务（is_draft=1）允许上传。
GET  /api/tasks/{task_id}/documents        → [同结构]
GET  /api/tasks/{task_id}/documents/{doc_id}/download → FileResponse（原文）
DELETE /api/tasks/{task_id}/documents/{doc_id} → 204（定版后也允许，交付前自己保证）
```

### 3.3 用例确认（T1 + Lead 集成）

```
POST /api/tasks/{task_id}/cases/confirm
body {"cases_sha256": "<64hex>", "benchmark_sha256": "<64hex>", "by": "chat:job#123"|null}
→ 200 {"confirmed_at","cases_sha256","cases_suite_sha256","suite_ref"}
校验：cases/<op>/cases.json 与 benchmark.json 存在且各自 sha256 等于入参 → 否则 409；
      cases_suite_sha256 = §5.1 的整目录哈希；
      写 tasks.cases_confirmed_at/sha256/by + StageEvent（reason="用例组确认"）。
GET /api/tasks/{task_id}/cases/confirmation → {confirmed_at, confirmed_sha256, current_suite_sha256, in_sync}
```

- 未确认时：`POST /tasks/{id}/finalize` **不拦**（定版先发生）；但**四动作 build/test/benchmark/profile 的 plan** 增加缺项 `"用例组未确认"`（Lead 改 exec_actions）。

### 3.4 版本（Lead）

```
GET  /api/tasks/{task_id}/versions
→ {"current": {id, major, minor, label:"v1.2", source, spec_version_id, code_snapshot_sha256,
               suite_ref, delivered_at, created_at}, "history": [...], 
   "evidence": {"required_actions":["build","test","benchmark"], 
                "runs": {action: {run_id, status, gate_verdict, spec_version_id, code_snapshot_sha256, suite_ref} | null},
                "missing": [{"action":"benchmark","reason":"版本快照不一致"|"未跑"|"门禁未过"}...],
                "deliverable": true|false,
                "current_snapshot_sha256": "<workspace snapshot or null>",
                "current_suite_ref": "cases@..."}}
POST /api/tasks/{task_id}/versions   body {"note": "...", "label": "..."} → 手工小版本（minor+1）
```

### 3.5 交付（Lead）

```
POST /api/tasks/{task_id}/deliveries
→ 202/201 {run_id, download_url, sha256, size_bytes, version:"v1.2", created_at}
  校验（任一失败 → 409 {detail, codes:[...]}）：
   - 用例已确认且未变（cases_confirmed_at 非空、confirmation.in_sync）
   - 版本证据齐全：build/test/benchmark 最新 run 均 ok+gate pass 且 spec_version_id/code_snapshot_sha256/suite_ref
     与当前版本一致；profile 可选
   - 产物齐备：ops/<op>、ops/<op>/python、tests/test_<op>.py、cases/<op>/{cases,benchmark}.json、
     ops/<op>/README.md、规划文档（task_documents 有原文或 docs/<op>/design.md 存在）
  打包到 <repo_root>/archives/deliveries/task-<id>/v<major>.<minor>/<op>_v<major>.<minor>_<yyyymmdd>.zip
  登记 Run(action="delivery", status="ok", spec/snapshot/suite 同上, manifest_path=zip 路径? 用 note)
  + Artifact(type="delivery", path=zip 相对/绝对, sha256, size_bytes) + Artifact(type="manifest", ...)
  + task_versions.delivered_at = now
GET /api/tasks/{task_id}/deliveries → [{run_id, version, sha256, size_bytes, created_at, download_url}]
GET /api/deliveries/{run_id}/download → FileResponse
```

- 交付包结构见设计 §7.2；zip 内路径固定：`ops/<op>/`、`tests/`、`docs/算子说明文档.md`、`docs/规划文档-补全.md`（如有）、`docs/规划文档-原始.<ext>`（如有）、`manifest.json`、`evidence/`（可选：最新 correctness.json/performance.json）。

### 3.6 测试管理：两类用例（Lead）

```
GET /api/tasks/{task_id}/cases   （既有接口，追加 "performance" 段，不破坏旧字段）
→ { ...precision 现有字段...,
    "performance": {"source": "cases/<op>/benchmark.json"|null,
                    "primary_case": "bench_1m"|null,
                    "cases": [{case_id, n, dtype, required, primary, note,
                               "latest": {run_id, candidate_median_us, baseline_median_us,
                                          speedup, conclusion, gate_verdict} | null}],
                    "runs": [{"run_id","status","gate_verdict","suite_ref","started_at"}] } }
```

## 4. 门禁 / 阶段（Lead）

- `gate.evaluate`：action=delivery 时校验 `delivery` + `manifest` 两类 artifact、run.status=ok；版本对齐沿用。
  action=smoke 时必需 `log` + `smoke` artifact，`smoke.json.verdict` 必须为 PASS（fail code `E_SMOKE_FAILED` /
  `E_SMOKE_REPORT_MISSING`），且报告版本字段与 run 行一致。
- `exec_actions.ACTION_SPECS`：build/smoke/test/benchmark/profile 的 `requires` 增加 `cases_confirmed`；plan 缺项文案 `"用例组未确认（先在任务对话里确认精度/性能用例组）"`。
- `stageboard`：
  - 阶段一「需求与设计」= 有 spec **且** `cases_confirmed_at` 非空；
  - 阶段二「开发」= build **且** smoke 门禁 pass（2026-10-23 追加）；
  - 阶段四「交付」= 最近一次 delivery run 门禁 pass 且其 spec/snapshot 与当前版本一致。
- 阶段推导/`/api/tasks/{id}/gates` 的四动作列表保持 build/test/benchmark/profile；交付单独显示。

## 5. 工作区路径约定（T3 / 交付打包共同遵守）

| 产物 | 路径 |
|---|---|
| 设计文档原文/解析稿 | `task-docs/task-<id>/...`（DB rows 为准） |
| 精度/性能用例 | `cases/<op>/cases.json`、`cases/<op>/benchmark.json` |
| 算子代码 | `ops/<op>/` |
| Python 调用接口 | `ops/<op>/python/<op>.py`（+ `lib<op>_host.so`，可缺省） |
| pytest 测试脚本 | `tests/test_<op>.py`（读 `cases/<op>/cases.json`） |
| 参考实现 | `ops/<op>/scripts/golden.py`（沿用） |
| 算子说明文档 | `ops/<op>/README.md` |
| 规划文档（agent 补全） | `docs/<op>/design.md` |

### 5.1 用例集哈希（suite_ref，必须与 `scripts/nm_test.sh` 一致）

`NM_TEST` 的 `suite_ref = cases@<sha256 of cases/ dir tree, first16>`：遍历 `cases/`（排序），
对每个文件依次 `update(相对路径 bytes); update(b"\0"); update(文件内容按 1MB 分块); update(b"\0")`。
Python 实现放 `backend/app/services/case_confirm.py`，供 confirm / 版本证据 / 交付共用。

## 6. 前端契约（T2）

- 类型（web/src/types.ts）：`TaskDetail` 增 `version_minor/cases_confirmed_at/cases_confirmed_sha256`；新 `TaskVersion`、`TaskDocument`、`CaseReport.performance`、`DeliveryRecord`。
- `lib/api.ts` 增加：`documents(taskId)`、`uploadDocument(taskId, file)`、`deleteDocument`、`downloadDocumentUrl`、`templateUrl(format)`、`confirmCases(taskId, body)`、`casesConfirmation(taskId)`、`versions(taskId)`、`createVersion(taskId, body)`、`deliveries(taskId)`、`createDelivery(taskId)`、`downloadDeliveryUrl(runId)`。
- 任务开始页：模板下载按钮（docx/md）+ 上传区（选择/拖拽 docx/md，≤20MB；提示解析状态；可删除/替换）。
- 任务详情：
  - 左上标题 `{title/op_name} v{major}.{minor}`（major 取当前 spec 版本，minor 取 task.version_minor）。
  - 对话分区：右上「放大」按钮 → 全屏浮层呈现 ChatPage（Esc 关闭）；其余布局不变。
  - 概览执行步骤：开发 / 编译 / 正确性 / 基准 / 采集 / 交付；「交付」按钮走 `createDelivery`，展示门禁缺项（调 `versions`）。
  - 「需求与规格」分区：移除 JSON 编辑器/新建版本表单，改为只读 + 「修改需求/规格（走对话）」按钮（跳 chat 并 seed 文案）。
  - 「测试与结果」：精度/性能两类用例摘要（性能来自 `/cases` 的 performance 段）。
  - 「版本」分区：顶部任务版本列表（v大.小、来源、当前高亮、交付状态、关联 run），下方保留 savepoints。
  - 「交付」分区（原 DeliveryPanel）：交付门禁预览（缺项 + 一键去跑对应动作或显示需重跑）、生成交付包、交付历史下载。
- 测试管理页 `TestPanel`：两个 tab「精度用例」「性能用例」，只读；性能列 n/dtype/主用例/最近候选/基线/加速比/门禁。
- 任务列表/阶段页可继续用 title，不强制版本号。

## 7. Agent 提示词契约（T3）

1. **创建期对话**（`chat.py`）：`build_prompt(..., doc_text: str | None = None)`；`doc_text` = 上传文档解析稿前 ~12k 字符，注入 system/user prompt（"用户上传的算子设计文档摘录"）。SYSTEM_PROMPT 增加：若文档含测试用例附录，提示后续由任务对话确认。
2. **任务对话**（`conversations.build_prompt`）：
   - 注入任务当前版本 `v{major}.{minor}`、用例确认状态；
   - 若 `cases_confirmed_at` 为空：**第一优先级**是生成精度/性能用例组并在对话里整组呈现，等待用户「确认」；确认后用 `python3`/`curl` 写文件并调 `POST /api/tasks/{id}/cases/confirm`（API 地址 `http://127.0.0.1:8077`），在确认前**不改代码、不调四动作**；
   - 已确认后：开发交付物清单（算子代码 / Python ctypes 调用接口 / pytest / README / 规划文档），指向 `docs/reference/deliverable-standards.md`；
   - 规格修改协议：语义变更 → 起草新 spec_json → 用户确认 → `POST /api/tasks/{id}/specs`（大版本 +1、小版本归 0、用例需重新确认）；文档性修订可不落版本。
3. **交付物规范文件**（T3 新建 `docs/reference/deliverable-standards.md` + `workspaces/cann-ops/AGENTS.md` 引用）：
   - Python 接口：ctypes 加载 `lib<op>_host.so`，C ABI `int <op>_run(const void*, const void*, void*, int64_t n, int dtype_code, int device_id)`；host `.asc` 增加 `extern "C"` 导出与共享库构建目标；Python 包装接受 torch_npu tensor、校验 dtype/shape、返回新 tensor；
   - pytest：`tests/test_<op>.py`，读 `cases/<op>/cases.json`，比对 `ops/<op>/scripts/golden.py`；拒绝用例校验拒绝码；
   - 说明文档 README 结构；规划文档 `docs/<op>/design.md` 结构（按模板五段）。

## 8. 验证命令

```bash
# 后端
backend/.venv/bin/python -m pytest backend/tests -q
backend/.venv/bin/python -m alembic -c backend/alembic.ini upgrade head   # 在 backend/ 下执行注意 db_url
# 前端（用 DSH 自带 node/pnpm）
"$NODE" "$PNPM" --dir web run build       # 或 cd web && pnpm run build
cd web && npx tsc --noEmit
```

## 9. 变更记录 / 实施口径登记

- 2026-10-23 初版（Lead）：建立并行契约。实现偏差必须在此登记。
- 2026-10-23 实施口径（backend-docs / Lead）：
  1. `tasks.cases_confirmed_sha256` 存**整个 `cases/` 目录树哈希的 full 64hex**（不是单文件哈希）；
     `suite_ref = "cases@" + hex[:16]`，与 `scripts/nm_test.sh` 逐字节一致。
  2. 用例确认发生在**建任务后的任务对话**（创建期澄清对话无工具、只产出 spec 草稿）；
     `finalize` 不拦用例确认；四动作 plan 增加缺项「用例组未确认」作为硬门禁。
  3. 模板下载与用例确认接口都落在 `routers/documents.py`（路径与 §3.1/§3.3 一致）。
  4. `GET /api/tasks/{id}/deliveries` 返回 `{task_id, deliveries: [...]}`（不是裸数组）；列表项
     `version` 是人读备注（如「交付包 xxx.zip」），结构化版本看 `GET /versions` 的 `current.label`。
  5. 另提供 `GET /api/tasks/{id}/deliveries/preview`（契约 §3.5 未列）：交付门禁预检，返回
     `{ok, missing[], evidence}`，页面可直接用来展示「5 件产物/证据缺什么」。
  6. 任务版本 `label` = 自动格式 `vM.m`，人写名字在 `label_text`。
