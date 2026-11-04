# 自动优化门禁 v1（对话确认 → 基线 → 改码 → smoke → 全量 → 回写）

> 状态：2026-10-26 起实现。上游设计：`docs/plan-optimize-rounds-v1.md`（轮次/存点/报告）、
> `docs/perf-diff-v1.md`（同条件 A/B）、`docs/plan-task-lifecycle-v2.md`（build/smoke/test/benchmark 动作链）、
> `docs/plan-savepoints-v1.md`（源码存档/回滚）。
>
> 本文回答一个具体问题：**“对话里确认优化思路 → 页面记录 → 基线门禁 → agent 改码 → smoke →
> 全量精度/性能验证 → 结果回写 → 人工采纳”这条链，靠什么保证结论可信？**
> 结论先行：**靠显式证据绑定 + 硬门禁，不靠 run/job 拓扑反推，也不靠页面按钮或 agent 自觉。**
>
> **实现状态（2026-10-27）**：S0.1–S0.4、Step 1a（phase 状态机/事件/幂等/工作区租约/重启恢复）、
> Step 1b（confirm 一次性 token / `loop_run_links` / `loop_attempts` / restore 租约）、
> Step 2（自动验证链 orchestrator）与 Step 3（append-only 报告归档 + manifest/sha + raw 副本 +
> 独立目录校验脚本）已落地，`pytest` **485 passed**，`tsc --noEmit` / `npm run build` 通过；
> 迁移 `c9d0e1f2a3b4`、`e9f0a1b2c3d4`、`a3b4c5d6e7f8`；报告仓库 `reports/loops/<id>/…`。

## 1. 目标 / 非目标

### 1.1 目标

1. 用户在对话里审查并确认优化思路；确认落成**结构化轮次（Loop）**，页面是唯一真相源。
2. 改码前必须证明“基线就绪”：同一代码快照、同一环境、同一用例集上的 before test + benchmark
   都通过，且当前工作区仍是基线代码。
3. 改码后先跑单例 smoke（编译 + 一条最小 required-match 用例）作为**快速失败门禁**，失败立即回滚；
   smoke 不是精度/质量结论。
4. 全量 test（精度硬拦）+ benchmark（同条件 A/B）出 `perf-diff` 与报告；**采纳必须过证据门禁**。
5. 自动判定不等于自动采纳：人保留采纳/否决/回退权。
6. 所有“是否就绪/是否通过/哪一版”都由服务端显式绑定与 gate verdict 判定，可审计、可恢复。

### 1.2 非目标（v1）

- 不做“确认即全自动跑完”：一期 smoke 之后的全量动作仍可人工一键触发；二期才串自动链。
- 不做多用户权限体系；不做跨任务共享工作区；不做自动采纳。
- 不在 v1 解决“独立目录复现报告”（只保证报告可复算，并如实标注未验）。
- 不把 allowed_paths 当安全沙箱：v1 仍是“事后快照比对 + 回滚”，只把证据不得改动的路径单独硬拦。

## 2. 术语与核心模型

| 术语 | 含义 |
|---|---|
| `Loop` | 优化轮次，优化单位；`objective_json` 冻结目标/约束/预算/目标快照 |
| `attempt` | 一次“agent 改码 + 后续验证”的尝试；一个 Loop 可多次 attempt（重试/失败后重来） |
| `baseline_sha` | 本轮基线代码快照（= 建轮次时 baseline 存点的 sha） |
| `candidate_sha` | 某个 attempt 改码后的代码快照（= 改后快照 sha） |
| `attempt 证据` | 该 attempt 的 `smoke/test/benchmark` run，必须全部挂同一 `candidate_sha` |
| `gate verdict` | `gate.evaluate(run).verdict`，`pass/fail`；**不是**进程 exit code |
| `phase` | 轮次状态机（见 §5），持久化，不用 runs/jobs 反推 |

**核心不变量**：

1. before = `baseline_sha` 上的 test+benchmark；after = `candidate_sha` 上的 test+benchmark。
   两侧不得由“run 出现顺序”推断。
2. 同一 attempt 的 smoke/test/benchmark 必须同 `candidate_sha`；跨 sha 的候选证据无效。
3. 任何“通过/就绪”都必须有 gate verdict=pass，`Run.status=ok` 只是“脚本跑完”。
4. 参考实现（golden）、判据（criteria）、用例（cases）是**证据输入**，不是可优化代码；
   agent 不得修改，改了就判证据失效。

## 3. 门禁总表

| 门禁 | 什么时候查 | 判据 | 失败码（消息前缀） |
|---|---|---|---|
| G0 只读讨论 | 对话“审查思路”阶段 | job 级 read-only；快照比对无改动 | `E_READONLY_VIOLATION` |
| G1 基线就绪 | 启动改码前（后端 409） | 见 §3.1 | `E_BASELINE_*` |
| G2 改码证据 | agent job 结束后 | `status=done`、`changed=true`、`guard_reverted=false`、`after_sha=attempt.candidate_sha` | `E_NO_CANDIDATE` |
| G3 smoke | 改码后、全量前 | `smoke.json.verdict=PASS` 且 run.sha=candidate_sha；失败回滚 | `E_SMOKE_FAILED` |
| G4 验证 | test/benchmark 各动作结束后 | action job ok **且** gate verdict=pass；两侧同 sha | `E_RUN_GATE_FAIL` |
| G5 精度 | 全量 test 后 | before/after 用例集不得缩水；PASS→FAIL 硬拦 | `E_PRECISION_*` |
| G6 性能 | 全量 benchmark 后 | 同条件键一致 + 有有效测量 + 结论枚举；无测量不得 pass | `E_PERF_*` |
| G7 采纳 | accept 前（后端 409） | 见 §3.2 | `E_ACCEPT_*` |

### 3.1 G1 基线就绪（`baseline_evidence`）

必须同时满足：

1. `Loop.objective` 存在，`status=open`，无 running 工作区作业；
2. baseline 存点存在，且 `baseline_sha == current_sha(workspace)`（工作区不得在建轮次后被改过）；
3. 存在 before `test` 与 before `benchmark` 两条 run：
   - `run.code_snapshot_sha256 == baseline_sha`；
   - `run.status == "ok"`；
   - `gate.evaluate(run).verdict == "pass"`（产物齐、版本对齐、用例非 NOT_RUN）；
4. 两条 run 的环境指纹一致：
   `spec_version_id / criteria_ref / image_digest` 必须非空且相等；`host_id` 有值时
   必须一致（历史数据两边都为空按“未记录”处理，Step 1 起强制非空）。
   （2026-10-28 修订：**去掉「必须同卡」的要求**——执行侧指定卡被占会自动切空闲卡，
   `device_id` 只记录进 `env.device_id / env.device_consistent`，不再作为 G1 硬条件；）
5. 两条 run 的 benchmark case 指纹一致（`perf_diff._case_fingerprint`，before 侧自身校验即可；
   与 after 的一致性在 G6 再做）；
6. 复用条件：同 sha、同指纹、未被其它 Loop 占用；否则必须重跑基线。

失败返回结构化原因（`issues[]`），后端 409；不得静默降级成“仅有 run 就算就绪”。

### 3.2 G7 采纳就绪（`accept_evidence`）

必须同时满足：

1. Loop `status=open`，无 running 工作区/动作作业；
2. 最新 `candidate` 存点存在，`candidate_sha` 明确；
3. before/after 各有 `test` + `benchmark`，且 after 两侧 sha = `candidate_sha`；
4. 四条 run 的 gate verdict 全为 pass；
5. `perf_diff.evaluate_loop(...)`：`verdict=pass`、`precision.regressions` 为空、
   `conclusion != regressed`（精度退化/性能回退一律不得采纳）；
6. `optimization.report(write=True)` 成功落盘（失败 409，不允许“报告稍后补”）；
7. 当前工作区 sha == `candidate_sha`；不一致 409（可先存档/回滚，或重跑候选验证）。

> 允许显式 override 是 v2 的事；v1 一律 409。

## 4. 基线证据契约（后端派生，不新增表）

`GET /api/loops/{id}/baseline-evidence`（新增，Step 0 内部先用 service；对外接口 Step 1 补）：

```json
{
  "schema": "opx-baseline-evidence/1",
  "loop_id": 12,
  "baseline_savepoint_id": 34,
  "baseline_sha256": "…",
  "workspace_sha256": "…",
  "ready": false,
  "issues": [
    {"code": "E_BASELINE_GATE_FAIL", "message": "benchmark run #40 门禁未通过：E_REPORT_VERSION_MISMATCH"}
  ],
  "before": {
    "test": {"run_id": 39, "status": "ok", "gate": "pass", "sha256": "…"},
    "benchmark": {"run_id": 40, "status": "ok", "gate": "fail", "sha256": "…"}
  },
  "env": {
    "spec_version_id": 3, "criteria_ref": "ops-precision-standard@1",
    "image_digest": "sha256:…", "device_id": "1", "host_id": 2,
    "bench_cases_fingerprint": [["bench_1m", 1048576, "float32", true]]
  }
}
```

- 一切字段来自 DB + 产物 + `gate.evaluate`，不信任前端缓存。
- `ready=true` 才能放行改码；`issues` 用于页面提示与回归测试断言。

## 5. 状态机（Step 1a 已落库）

`loops.phase`（实现见 `services/loop_state.py`）：

```
idle / baseline_pending → baseline_ready → agent_running
   → awaiting_verification → smoke_running → full_running → awaiting_adoption
   → accepted / rejected / failed / blocked / cancelled
```

- 所有转移带 `state_version` CAS + `active_job_id` 单占用；事件追加到 `loop_events`
  （created/start/agent_done/action_done/accept/reject/cancel/recover）。
- 建轮次写 `baseline_pending` 并尝试派生 `baseline_ready`；启动改码用 `start_agent` CAS
  （重复/并发启动返回 409）；agent 结束 `finish_agent`：有候选 → `awaiting_verification`，
  无改动/guard 回退 → `failed`。
- **no-op 终止**：agent 结束且 `changed=false` / `guard_reverted=true` → `failed`。
- **取消**：`POST /api/loops/{id}/cancel` 杀 agent 进程 + `phase=cancelled` + 释放租约；
  不静默回滚工作区（回退走「否决并回退」）；agent 收尾不会把 cancelled 改回 failed。
- **重启恢复**：启动时把 `agent_running/smoke_running/full_running` 标 `blocked`、清
  `active_job_id`、清空租约；远端容器清理仍属 Step 2。
- **前端**：轮次卡显示 phase chip（`PHASE_META`），phase_error 作为 tooltip。

Step 1b 余项：draft + 一次性 confirm token、`loop_run_links`（不再用 `runs.loop_id` 表达复用）、
savepoint restore 也持租约、`attempts` 历史与后台 orchestrator。

## 6. API 契约

### Step 0（本次实现，尽量不新增端点）

| 变更 | 契约 |
|---|---|
| `POST /api/loops/{id}/optimize`、`/optimize/chat` | 启动前查 G1；不满足 409，返回 `{detail, issues?}` 文案含 fail code |
| `POST /api/loops/{id}/accept` | 查 G7；不满足 409；成功前 `report(write=True)` |
| `POST /api/loops/{id}/budget-check` | action 白名单加 `smoke`；smoke 单独按 `loop×action` 计预算 |
| `POST /api/tasks/{id}/optimization-rounds` | 默认 `allowed_paths` 收窄为 `ops/<op>/op_host/`、`ops/<op>/op_kernel/` |
| `GET /api/loops/{id}/optimization-report?write=true` | 无法落盘时报错，不再静默跳过 |
| 候选存点 | 仅当同 sha 的 test 成功 run 存在时才自动落；四条 run 的 gate pass 在采纳门禁 G7 强校验 |

### Step 1（已实现部分）

| 变更 | 契约 |
|---|---|
| `GET /api/loops/{id}/phase` | phase / state_version / active_job_id / error / 最近事件 |
| `GET /api/loops/{id}/baseline-evidence` | G1 证据（ready + issues + before run + env） |
| `POST /api/tasks/{id}/optimization-rounds` | 新增可选 `idempotency_key`；同 key 返回已有轮次 |
| `POST /api/loops/{id}/optimize`、`/optimize/chat` | `start_agent` CAS + DB 级工作区租约 |
| `POST /api/loops/{id}/cancel` | 取消：杀 agent、phase=cancelled、释放租约 |
| `POST /api/loops/{id}/confirm` | 冻结 draft + 一次性 token（10 分钟，库里只存 sha256），见 §15.1 |
| `POST /api/loops/{id}/start` | 校验并 CAS 消费 token 后启动；目标/基线/工作区变化 → 409 |
| `GET /api/loops/{id}/attempts` | 尝试历史（seq/状态/基线+候选 sha/时间），见 §15.3 |
| `loop_run_links` | 复用 run 只写 link，不改 `runs.loop_id`；证据读取走并集，见 §15.2 |
| `loops.phase/...` + `loop_events` + `workspace_leases` | 迁移 `c9d0e1f2a3b4` |
| `loop_confirmations` + `loop_attempts` + `loop_run_links` | 迁移 `e9f0a1b2c3d4` |

### Step 1b / Step 2+（后续）

- `POST /api/loops/{id}/confirm`（draft + 一次性 token）与独立 `POST /api/loops/{id}/start`
- `POST /api/loops/{id}/attempts`（history）；savepoint restore 租约；`loop_run_links`
- 二期：后端 orchestrator（`optimization_runs`），smoke/test/benchmark 自动串行 + 远端清理确认。

## 7. 失败 / 回滚矩阵

| 失败点 | 工作区 | 轮次状态 | 产物 |
|---|---|---|---|
| agent verifier/allowed_paths fail | 自动回滚到改前 | `failed` | 保留 `optimize_json`（attempted_after、guard_reason） |
| agent 无改动 / guard 回退 | 保持原样 | `failed`（no-op） | 记录原因，不进 smoke/全量 |
| build/smoke fail | 自动回滚到 attempt 前 | `failed` | 落 `failed_candidate` 存点（Step 1） |
| 精度退化 | 默认回滚（可配置保留现场） | `failed` | 报告 + 候选侧证据 |
| 性能回退 | **保留现场**，不自动回滚 | `awaiting_adoption` / `rejected` 由人定 | 报告如实写 regressed |
| 报告落盘失败 | 不动 | 停在 `awaiting_adoption`，禁止 accept | 409 |
| 取消 | 按 keep/revert 语义 | `aborted` / `blocked` | 记录取消点与远端清理状态 |

## 8. 预算与复用

- `runs_per_action` 按 `(loop_id, action)` 计；`smoke` 是独立 action，与 test/benchmark 分账。
- 复用基线 run 不计入“本轮执行次数”，但必须出现在 `baseline_evidence` 里；
  Step 1 用 `loop_run_links` 表达复用，不再靠改 `runs.loop_id`。
- 预留（allowed 审计）必须可回收：启动失败 `void`、取消/超时/重启恢复显式回收；Step 1 加 TTL。
- 预算不足时在**确认阶段** 409，不允许跑到一半才失败（二期编排）。

## 9. Oracle 完整性（防“改参考实现”）

1. **路径硬拦**：`optimize._finish` 对以下改动的处理一律自动回滚并判 fail（`E_PROTECTED_PATH`）：
   - `ops/<op>/scripts/`（golden/参考实现）
   - `cases/`、`criteria/`、`tools/`、`scripts/`（用例/判据/执行器）
2. **默认允许路径收窄**：`allowed_paths` 默认 `ops/<op>/op_host/`、`ops/<op>/op_kernel/`；
   用户显式追加也只能落在 `ops/<op>/` 内，且仍受第 1 条保护。
3. **指纹留痕**：`correctness.json` 增加：
   ```json
   "oracle": {
     "golden_path": "ops/<op>/scripts/golden.py",
     "golden_sha256": "…",
     "criteria_path": "criteria/check.py",
     "criteria_sha256": "…"
   }
   ```
   `perf_diff` 在两侧都带该字段时比对，不一致 → `E_ORACLE_CHANGED`（fail）。
4. 报告与证据里保留 oracle 指纹，事后可复算。

## 10. 报告与可复现

- `write=true` 必须成功落盘；路径写到 after benchmark 的 run 目录（v1），并在返回中带 `written_to`。
- 报告必须包含：两侧 run 与 sha、环境指纹、case 集指纹、oracle 指纹、perf-diff、逐例精度、
  复现命令、生成时间；**不宣称**“独立目录复现已验”。
- Step 1 起改为 append-only 报告目录 + manifest + sha，避免被重跑覆盖。

## 11. 分期

### Step 0（本次实现，2026-10-26 已完成）

- S0.1 oracle 隔离与指纹：
  - `optimize.PROTECTED_PATHS` + `protected_violations()`：改 `ops/<op>/scripts/`、`cases/`、
    `criteria/`、`tools/`、`scripts/` → `E_PROTECTED_PATH` + 自动回滚；证据落 `protected_paths`。
  - 默认 `allowed_paths` 收窄为 `ops/<op>/op_host/`、`ops/<op>/op_kernel/`（显式放宽仍受保护路径硬拦）。
  - `tools/run_cases.py` 的 `correctness.json` 增 `oracle.{golden,criteria}_sha256`；
    `perf_diff` 两侧都有该字段且不同 → `E_ORACLE_CHANGED`（fail）。
- S0.2 baseline / accept 硬门禁：
  - `optimization.baseline_evidence / require_baseline`：同 sha、gate pass、环境一致、工作区仍为基线；
    启动 `/optimize`、`/optimize/chat` 前调用，未就绪 409。
  - `optimization.accept_evidence / require_accept`：before/after 成对 test+benchmark、四条 gate pass、
    perf-diff 绑定 baseline/candidate、精度不退化、报告 write 成功、工作区仍为候选；否则 409。
  - `action_runner._maybe_candidate_savepoint`：缺同 sha test run 不生成 candidate。
- S0.3 gate verdict 替代 exit code：
  - 前端 `OptimizationRoundPanel` 在动作 `status=ok` 后继续检查 `gate_verdict=pass`；
  - 后端基线/候选/采纳统一用 `gate.evaluate`。
- S0.4 smoke 纳入 loop 预算/动作枚举：`POST /loops/{id}/budget-check` 白名单加 `smoke`，
  独立按 `loop×action` 分账；非法 action 仍 400。

**验收**：`pytest` **436 passed**；`tsc --noEmit` / `npm run build` 通过；无迁移。
新增回归：`test_optimization_rounds.py`（baseline/accept/report/candidate）、
`test_optimize.py`（protected golden、cases）、`test_perf_diff.py`（E_ORACLE_CHANGED）、
`test_loop_budget.py`（smoke budget-check）。

### Step 1a（2026-10-26 已完成）

- 迁移 `c9d0e1f2a3b4`：`loops.phase/state_version/active_job_id/idempotency_key/phase_error/phase_updated_at`
  + `(task_id, idempotency_key)` 唯一索引 + `loop_events` + `workspace_leases`（project 唯一）。
  存量回填：非优化 Loop=idle/closed；优化轮次按关闭结论或 baseline_pending。
- `services/loop_state.py`：PHASES、CAS 转移、start_agent/finish_agent、derive/sync_phase、
  reconcile_startup（running → blocked）、事件 append-only。
- `services/workspace_lease.py`：DB 级单飞（agent/action 共用；过期或 job 不在 running 可抢占；
  重启清空）；已接入 optimize 启动、action_runner.start/_finalize、conversations.send/_finish。
- 幂等：`OptimizationRoundCreate.idempotency_key`（唯一索引 + IntegrityError 重查）；
  启动用 state_version + active_job_id CAS，并发/重复 → 409。
- 取消：`POST /loops/{id}/cancel`；`optimize._finish` 不覆盖 cancelled。
- 前端：`Loop.phase` 类型 + 轮次卡 phase chip。
- 验收：`pytest` **443 passed**；`tsc --noEmit` / `npm run build` 通过；迁移往返测试通过。

### Step 1b（2026-10-27 已完成）

- 迁移 `e9f0a1b2c3d4`：`loop_confirmations`（token_hash 唯一 / draft / 冻结目标与基线 / used_at）、
  `loop_attempts`（轮内 seq 唯一 / 状态 / 基线+候选 sha）、`loop_run_links`（(loop_id, run_id) 唯一）。
- 确认与启动（§15.1）：`POST /loops/{id}/confirm` 冻结 draft 并返回 10 分钟一次性 token（库里只存
  sha256）；`POST /loops/{id}/start` 在同事务 CAS 消费 token 后走与对话化启动同一条链；确认后
  目标/基线/工作区变化 → 409（`E_CONFIRM_OBJECTIVE_CHANGED` / `E_CONFIRM_BASELINE_CHANGED`）；
  重放/过期/跨轮次 token → 409（`E_CONFIRM_TOKEN`）。`/optimize/chat` 保留（可选 `confirm_token`，
  不传写 `confirm_auto` 事件），页面改走 confirm → start。
- run 复用（§15.2）：建轮次复用基线 run 只写 `loop_run_links`（不再改 `runs.loop_id`）；证据读取
  （baseline/accept/report/perf-diff/stageboard/参考轮次）按 loop_id ∪ links 并集；预算仍按
  `runs.loop_id` 计（复用 run 不计本轮执行次数）。
- attempt 历史（§15.3）：`start_agent` 开 attempt，退出 running phase 时由 `transition` 统一收尾
  （succeeded/failed/blocked/cancelled，带候选 sha），取消/重启对账收尾；`GET /loops/{id}/attempts`。
- 回滚租约（§15.4）：`savepoints.restore` 持 `kind=restore` 项目级租约，结束/失败都释放。
- 前端：对话页回车改 confirm → start（失败回滚乐观消息/输入/banner）；轮次卡新增「尝试历史」
  （最近 5 次、状态 chip、候选 sha、时间）。
- 验收：`pytest` **443 → 459 passed**（新增 `test_step1b.py` 16 项）；`tsc --noEmit` / `npm run build`
  通过；迁移往返测试通过。两路只读审查（后端/前端）已完成，修复见 §15.5。

### Step 2（自动验证链 orchestrator，2026-10-27 已完成 v2.1）

- 迁移 `a3b4c5d6e7f8`：`loop_auto_runs`（status/current_step/steps_json/budget_json/错误/时间）。
- `services/loop_orchestrator.py`：持久化步骤机 + 进程内 asyncio 驱动；每步
  「预算 allowed 审计 → `action_runner.start` 挂 loop → 轮询终态 → `status=ok` 且
  `gate_verdict=pass` → 远端清理确认」；失败落 `failed_candidate` 存点 + 持租约回退 +
  phase failed；benchmark regressed 保留现场 awaiting_adoption；report 落盘失败
  awaiting_adoption + failed；取消/重启 running → blocked。
- API：`POST/GET /api/loops/{id}/auto-run`、`POST /api/loops/{id}/auto-run/cancel`；
  `POST /loops/{id}/cancel` 会一并停自动链；启动只允许 `awaiting_verification`（首次）或
  `blocked/failed`（重试；候选/基线/预算重新校验）。
- 前置一次性预检：G1（忽略「工作区=基线」这一条，agent 改码后天然不成立）、候选 sha =
  工作区 sha、四动作预算、四动作 plan；不满足 409 `E_AUTO_*`。
- 前端：轮次卡「自动跑完验证」按钮 + 五步进度块（状态/门禁/失败原因/清理结果/预算）+
  取消；running 时 2s 轮询并禁用抢工作区按钮。
- 验收：`pytest` **459 → 472 passed**（新增 `test_auto_run.py` 13 项）；`tsc --noEmit` /
  `npm run build` 通过；迁移往返测试通过。

### Step 2 审查修复（2026-10-27）

两路独立只读审查（后端并发/失败路径、前端轮询/互斥）后修复：

- **取消竞态**：`_start_job` 起完动作后复查 `run.status`，已取消就补杀 + 回收预算 + 远端清理；
  `_fail` 在回退前/写 phase 前复查 `run.status`，被取消只清步骤状态、不静默回退、不覆盖 blocked；
  取消/重启对账把 running 步标 skipped 并清 `current_action_job_id`；`sync_phase` 对 blocked 粘性
  （被取消作业的 `_finalize` 不会再把它改回 awaiting_verification）。
- **失败分类**：plan/预算/租约占用等资源类失败 → `blocked`（不回退，可原地重试）；只有动作真实
  失败或门禁 fail → `failed` + 回退（回退失败/清理未确认也归 blocked）。
- **超时**：orchestrator 的等待比子进程自身 timeout 多 60s；超时后先等终态、确认远端清理，再回退；
  确认不了 → blocked。
- **报告前校验**：benchmark 后必须存在同 sha 的 candidate 存点才进入 report/done，否则
  `E_AUTO_CANDIDATE` + failed（避免 G7 永久 409 的假成功）；regressed 结论落 `steps[report].conclusion`
  与 view `conclusion`。
- **事件循环**：preflight/建行/phase CAS 丢线程执行；report 落盘也丢线程；驱动任务保存强引用。
- **跨入口互斥**：自动链 running 时手动动作 / 一键回退 / savepoint restore / 对话改码 / 采纳 / 否决
  一律 409（见 §16.3 末段）。
- **前端**：GET 状态失败不当“没有”（unknown 保守禁用 + 重试）、done 文案与 regressed/conclusion 渲染、
  刷新恢复不再误接手自动链作业、取消/失败回读状态与清理提示。

### Step 2（自动链）

后端 orchestrator：改码 → build → smoke → test → benchmark → report；预算一次性预检；
失败候选存点；远端清理确认后才进下一步。

### Step 3（报告与复现，2026-10-27 已完成 v3）

- `services/optimization_reports.py`：append-only 报告仓库（见 §17）；`optimization.report(write=True)`
  委托归档；`accept_evidence` 的 G7 报告门禁改为「最新归档存在且深度校验通过」。
- API：`GET/POST /loops/{id}/optimization-reports`、`GET .../{sha}`、`POST .../{sha}/verify`；
  `GET /optimization-report?write=true` 兼容返回归档字段。
- `scripts/repro_report.py`：独立目录只读校验（manifest/files/raw；`--json/--show-commands`）。
- 报告 payload 增 `evidence` 块：两侧 run 的 manifest/产物路径与环境/可比性指纹/oracle 指纹/
  raw 清单/复现命令；对外只表述「可独立目录校验/复算」，不宣称「独立目录复现已跑通」。
- 前端：轮次卡「报告归档」块（生成/刷新/最近 5 条/校验/下载/查看 manifest/就地错误），
  「导出报告」改「导出即时报告（不归档）」。
- 验收：`pytest` **472 → 485 passed**（新增 `test_report_store.py` 13 项）；`tsc --noEmit` /
  `npm run build` 通过。

## 12. 验收测试清单

- G1：无 baseline 存点 / 无 before run / before gate fail / spec 不一致 /
  工作区 sha 漂移 → 启动 409；全通过 → 放行。（设备不一致**不再**是拦项，允许自动切卡。）
- G2/G4：agent `changed=false` / `guard_reverted` → 不进 smoke；test/benchmark gate fail
  但 exit 0 → 页面与后端均判失败；候选存点不生成。
- G7：无候选 / 候选缺 test / 候选 test gate fail / 精度退化 / 性能回退 / 报告无法落盘 /
  工作区 sha 不匹配 → accept 409；全通过 → 采纳并写报告。
- Oracle：改 `ops/<op>/scripts/golden.py` → `E_PROTECTED_PATH` + 回滚；两侧 golden sha 不同 →
  perf-diff `E_ORACLE_CHANGED`。
- Smoke：`--action smoke --loop` 预算门禁通过；超预算 409；smoke 失败回滚（Step 1 自动，Step 0 手动拒绝）。
- 前端：gate fail 不再显示“均通过”；错误里带 fail code。
- Step 1：幂等建轮（同 key 仅一个 Loop）；start CAS 重复启动 409；phase/events 正确；
  agent 无候选 → failed；取消 → cancelled + 租约释放；重启 → blocked + 租约清空；
  同一 project 的 agent/action 互斥；迁移含新列/新表。

## 13. 文件所有权 / 影响面

- 后端：`services/optimization.py`（门禁函数就地实现，避免新模块与路由错误映射的循环）、
  `services/loop_state.py`（新）、`services/workspace_lease.py`（新）、
  `services/optimize.py`、`services/action_runner.py`、`services/conversations.py`、
  `services/perf_diff.py`、`main.py`、`routers/runs.py`、`routers/optimization.py`、
  `models.py`、`schemas.py`、`alembic/versions/c9d0e1f2a3b4_optimization_state.py`
- 工作区工具：`workspaces/cann-ops/tools/run_cases.py`（oracle 指纹）
- 前端：`web/src/components/OptimizationRoundPanel.tsx`、`web/src/types.ts`
- 测试：`backend/tests/test_optimization_rounds.py`、`test_optimize.py`、`test_loop_budget.py`、
  `test_perf_diff.py`、`test_step1_state.py`、`test_migration.py`

## 14. 已定默认（有异议再改）

1. 一期不做全自动链；smoke 之后仍人工点“跑候选验证”。
2. accept 一律硬门禁，不提供 override。
3. smoke 计入 `loop×action` 预算，独立分账。
4. oracle/golden 路径硬拦 + 指纹；不把 allowed_paths 当安全沙箱。
5. 报告 v1 写 run 目录，Step 1 再改 append-only。

## 15. Step 1b 契约（2026-10-26 实施，迁移 `e9f0a1b2c3d4`）

### 15.1 draft / confirm / start（一次性 token）

目的：让「对话里确认的方案」成为服务端显式记录；启动只能消费一次，防页面重放、并发双击、
以及「确认后目标被改却照旧启动」。

- `POST /api/loops/{id}/confirm`
  - body：`{instruction?: str, reference_loop_id?: int, note?: str}`
  - 前置：loop 存在、`objective` 存在（旧 Loop 409）、`status=open`、G1 `require_baseline` 通过。
  - 行为：冻结 `draft = {loop_id, objective_sha256, baseline_sha256, workspace_sha256,
    instruction, reference_loop_id, state_version, created_at}` 写入 `loop_confirmations`；
    返回明文 token（DB 只存 `sha256(token)`）、`expires_at`（默认 10 分钟）与 `draft`；
    追加 `confirm` 事件。
  - 失败：409 + `E_CONFIRM_*` 前缀（含 G1 的 `E_BASELINE_*` issues 文案）。
- `POST /api/loops/{id}/start`
  - body：`{token: str, conversation_id?: int}`
  - 前置：token 存在 / 未过期 / 未使用 / 属于该 loop；`objective_sha256`、`baseline_sha256`
    与当前一致；重新跑 G1（工作区 sha 必须仍等于 baseline）。instruction 与参考轮次**只取
    draft**，请求体不能覆盖。
  - 行为：同一事务里 CAS 消费 token（`used_at` 只能写一次）+ 建会话/用户消息 + 起 agent job；
    任一步失败整体回滚，token 不被消耗。
  - 返回同 `/optimize/chat`：`{job_id, loop_id, conversation_id, message_id, prompt}`。
  - 失败：409 + `E_CONFIRM_TOKEN` / `E_CONFIRM_OBJECTIVE_CHANGED` / `E_CONFIRM_BASELINE_CHANGED`。
- 兼容：`/optimize`、`/optimize/chat` 保留；`/optimize/chat` 可选 `confirm_token`（传了就校验
  并消费），不传按 legacy 处理并写 `confirm_auto` 事件。页面改走 confirm → start。

### 15.2 `loop_run_links`（复用 run 不再改 `runs.loop_id`）

- 新表 `loop_run_links(loop_id, run_id, role, created_at, created_by)`，唯一 `(loop_id, run_id)`。
- 建轮次复用基线 run：写 link（`role=baseline`），**不再** `run.loop_id = loop.id`；
  证据读取（baseline / accept / report / perf-diff）按
  `Run.loop_id == loop.id OR run_id IN (links)` 的并集取，历史数据（loop_id 已被改写）继续可读。
- 预算/执行次数仍按 `runs.loop_id` 算：复用来的基线 run 不计入本轮执行次数（§8）。

### 15.3 attempts 历史

- 新表 `loop_attempts`：一次 agent 改码尝试一行（`seq` 轮内自增）。
- `loop_state.start_agent` 开 attempt（`status=running`）；`finish_agent` 收尾
  （`succeeded` / `failed`，带 `candidate_sha256` 与原因）；取消/重启对账把 running → `blocked`。
- `GET /api/loops/{id}/attempts` → `{loop_id, attempts: [...]}`。

### 15.4 savepoint restore 持租约

- `savepoints.restore` 进入前 acquire `kind=restore` 的工作区租约（项目唯一），结束/失败释放；
  租约忙 → 409 `SavepointBusyError`。与既有 running 作业守卫形成两层门禁（DB 级 + 进程级）。

### 15.5 审查修复（2026-10-27）

两路独立只读审查（后端并发/事务、前端契约/交互）发现的问题与修法：

- **`/start` 必须 async**：同步端点会被 FastAPI 丢进线程池，而 `agent_runner.start` 内部
  `asyncio.create_task` 需要 running loop → 新主链必 500 且白烧 token。已改 `async def` 并加
  `inspect.iscoroutinefunction` 回归。
- **link-only 基线也生成候选存点**：`action_runner._maybe_candidate_savepoint` 的「本轮最早快照」
  改为 loop_id ∪ links 并集；否则 before run 只有 link 时会把候选误判成基线，candidate 永不落、
  accept 永久 409 `E_ACCEPT_NO_CANDIDATE`。
- **启动失败清理内存运行态**：`_fail_agent_start` + `optimize.start` 兜底 pop
  `optimize.PENDING`/`agent_runner.REGISTRY`，避免该 Loop 永远 409「已有正在跑」。
- **`transition` 不再替调用方 rollback**：CAS 失败只 `expire(loop)` 返回 False；accept/reject/cancel
  检查返回值并 409，避免并发下「200 但没落库 / job 永挂 / 证据丢失」。终态
  （accepted/rejected/cancelled）不可再迁移（同态除外），reject 在有运行中作业时 409 `E_REJECT_BUSY`。
- **租约作用域**：取消只按 `kind=agent + job_id` 释放本轮自己的租约（不再按 task 一刀切）；
  `reject(revert=True)` 的 optimize 改前回退与 `POST /optimize/revert` 都持 `kind=restore` 租约；
  restore 租约用 15 分钟短 TTL（无 job_id，崩溃时不等 12h）。
- **复用排他**：已挂 `loop_run_links` 的 run 不再被后续轮次复用（契约 §3.1(6)）。
- **契约细节**：`/start` 缺 token 返回 409 `E_CONFIRM_TOKEN`（不是 422）；`/optimize/chat` 带
  `confirm_token` 时指令/参考轮次只取冻结 draft；confirm/start 对工作区忙单独报 `E_CONFIRM_BUSY`
  （不再埋在「基线未就绪」里）。
- **降级可逆**：迁移 `e9f0a1b2c3d4` 的 downgrade 先把 link 回填成 `runs.loop_id`（最小 loop_id），
  1b→1a 后旧页面仍能看到复用来的 before 证据。
- **`draft.state_version`** 只作审计快照，不参与启动校验（目标用 objective sha、基线用
  baseline/workspace sha 强校验）。

## 16. Step 2 契约（自动链 orchestrator，2026-10-27 实施，迁移 `a3b4c5d6e7f8`）

目标：把「agent 改码完成 → build → smoke → test → benchmark → report」串成一条**服务端可恢复、
可取消、证据绑定**的链，替代页面上手动逐个点动作。v2.1 从 `awaiting_verification` 开始；
「confirm/start 即自动串起」（`auto_run` flag）与多 attempt 编排属 Step 2b。

### 16.1 触发与状态

- `POST /api/loops/{id}/auto-run`（async）：前置
  1. `phase ∈ {awaiting_verification, blocked, failed}`（首次 / 重试）、`loop.status=open`、
     无 running agent/action 作业；
  2. 候选明确：`loops.optimize.after_sha256`（或最新 succeeded `loop_attempts.candidate_sha256`）
     存在，且当前工作区 sha == 候选 sha；
  3. G1 仍就绪（before test+benchmark gate pass、环境一致）；
  4. 预算一次性预检：build/smoke/test/benchmark 四个动作 `used_for < limit_for`，不足 409
     `E_AUTO_BUDGET`（一次列全，不跑到一半才失败）；
  5. 四动作 plan 全部 `ready`，缺项 409 `E_AUTO_PLAN`（列 `missing`）。
  - 并发：phase CAS（上列 phase → `smoke_running`）+ `active_job_id IS NULL`，失败或已有
    running auto_run → 409 `E_AUTO_BUSY`；建 `loop_auto_runs` 行 commit 后 `asyncio.create_task`
    驱动，端点立即返回（preflight 的重活在线程里跑，不堵事件循环）。
- `GET /api/loops/{id}/auto-run`：最近一次 auto_run（无 → `{auto_run: null}`）。
- `POST /api/loops/{id}/auto-run/cancel`：杀当前 action job + 远端清理；`status=cancelled`、
  `phase=blocked`、**不静默回滚工作区**。

步骤：`build → smoke`（phase=`smoke_running`）→ `test → benchmark`（phase=`full_running`）
→ `report`（phase=`awaiting_adoption`）。每步：

1. 预算预留（`loop_budget.check_and_audit`，allowed 审计）→
2. `action_runner.start` 挂 loop → 3. 轮询作业终态 → 4. 要求 `status=ok` **且**
   `gate_verdict=pass` → 5. 远端清理确认（`remote_cancel.cancel_container` 返回 ok）→ 下一步。
   benchmark 成功后由既有 `_maybe_candidate_savepoint` 落 candidate；report 用
   `optimization.report(write=True)` 落盘。

失败矩阵（与 §7 对齐）：

| 失败点 | 工作区 | 轮次 phase | auto_run 状态 |
|---|---|---|---|
| build / smoke / test 失败（含门禁 fail） | 先落 `failed_candidate` 存点，再回退到 agent 改前 | `failed` | `failed` |
| benchmark 作业失败 / 门禁 fail / 无有效测量 | 同上回退 | `failed` | `failed` |
| benchmark 成功但 perf-diff `conclusion=regressed` | **保留现场**，不自动回退 | `awaiting_adoption` | `done`（记结论） |
| report 落盘失败 | 不动 | `awaiting_adoption`（禁止 accept，G7 已拦） | `failed`（`E_AUTO_REPORT`） |
| 用户取消 | 保留现场（回退走「否决并回退」） | `blocked` | `cancelled` |
| 进程重启 | 保持 | `blocked`（既有 reconcile） | `blocked` |

### 16.2 数据与视图

新表 `loop_auto_runs`：`id / loop_id / status(running|done|failed|blocked|cancelled) /
current_step / current_action_job_id / attempt_id / baseline_sha256 / candidate_sha256 /
steps_json / budget_json / error / started_at / ended_at / updated_at`。

```json
{
  "auto_run": {
    "id": 1, "loop_id": 12, "status": "running", "current_step": "smoke",
    "current_action_job_id": 55, "attempt_id": 3,
    "baseline_sha256": "…", "candidate_sha256": "…", "error": null,
    "steps": [
      {"step": "build", "status": "ok", "job_id": 54, "run_id": 41, "gate": "pass",
       "error": null, "started_at": "…", "ended_at": "…",
       "cleanup": {"ok": true, "container": "opx-r2-…"}},
      {"step": "smoke", "status": "running", "job_id": 55, "run_id": null, "gate": null,
       "error": null, "started_at": "…", "ended_at": null, "cleanup": null},
      {"step": "test", "status": "pending", "job_id": null, "run_id": null, "gate": null,
       "error": null, "started_at": null, "ended_at": null, "cleanup": null},
      {"step": "benchmark", "status": "pending", "job_id": null, "run_id": null, "gate": null,
       "error": null, "started_at": null, "ended_at": null, "cleanup": null},
      {"step": "report", "status": "pending", "job_id": null, "run_id": null, "gate": null,
       "error": null, "started_at": null, "ended_at": null, "cleanup": null}
    ],
    "budget": {"build": {"limit": 3, "used": 1}, "smoke": {"limit": 3, "used": 0},
               "test": {"limit": 3, "used": 1}, "benchmark": {"limit": 3, "used": 1}},
    "conclusion": null,
    "started_at": "…", "ended_at": null, "updated_at": "…"
  }
}
```

步骤状态：`pending | running | ok | failed | skipped`（`build/smoke/test/benchmark` 是动作，
`report` 落盘结果记 `gate`/`error`，`job_id` 为空）。

### 16.3 失败码

`E_AUTO_PHASE`（phase 不对）/ `E_AUTO_BUSY`（已有自动链/作业在跑）/ `E_AUTO_NO_CANDIDATE` /
`E_AUTO_BASELINE` / `E_AUTO_BUDGET` / `E_AUTO_PLAN` / `E_AUTO_CLEANUP`（远端清理未确认）/
`E_AUTO_TIMEOUT`（动作超时且未收尾）/ `E_AUTO_CANDIDATE`（benchmark 成功但候选存点没落）/
`E_AUTO_REPORT`（报告落盘失败）。全部 409（内部异常 500 由路由兜底）。

**步骤状态语义**：`pending | running | ok | failed | skipped`；取消/重启对账时把仍在 running
的步骤标 `skipped` 并清空 `current_action_job_id`。`steps[report].conclusion` 记录
`improved / no_significant_change / inconclusive / regressed`，view 顶层 `conclusion` 同步它。

**跨入口互斥**：自动链运行期间（`GET /auto-run` 为 running），手动动作（`POST /tasks/{id}/actions`
带 loop）、`POST /loops/{id}/optimize/revert`、`savepoints.restore`、对话改码（`conversations.send`）、
采纳/否决都 409 —— 工作区在 build/test/benchmark 之间必须稳定。

### 16.4 边界

- 不自动采纳；性能回退/精度退化仍由 G7 人工采纳门禁判定。
- 链内动作参数固定用任务/项目默认（device/timeout/criteria），不支持链内改参。
- 自动链只处理「当前 attempt」；`failed_candidate` 存点用于人工排查，不自动重试。

## 17. Step 3 契约（报告与复现，2026-10-27 实施，无迁移）

目标：报告不再写进会被重跑覆盖的 run 目录，而是 **append-only 报告仓库**；每次生成带
manifest + sha256 + 原始数据路径 + 复现命令，可脱离控制台/DB 在独立目录校验与复算。

### 17.1 报告仓库（append-only，文件系统真相源）

- 根目录 `settings.report_root`（空 = `<repo_root>/reports`）；布局：
  `<root>/loops/<loop_id>/<seq>-<report_sha12>/`：
  - `report.json`：**规范化** JSON（`sort_keys=True, ensure_ascii=False, indent=2` + 末尾换行），
    其 sha256 = `report_sha256`；
  - `manifest.json`：`{schema:"opx-report-manifest/1", loop_id, task_id, seq, report_sha256,
    content_sha256, generated_at, files:[{path,sha256,bytes}],
    raw:[{kind,path,source,sha256,bytes,copied,copy,why?}], report:{schema,verdict,conclusion},
    size_bytes, raw_copied_count, raw_total, source:{app,repo_root}}`；其中 `path` 是源侧相对路径，
    `copy` 是归档目录内相对路径（仅 `copied=true` 有）；
  - `raw/`：复制关键原始产物（两侧 run 的 `manifest.json` + manifest 里列出的 artifacts
    （performance/correctness/log 等）+ `gate_report.json`，归档前会 best-effort 先落 gate 报告）；
    单文件 ≤ 8MB、**每侧 ≤ 32MB（after 优先，避免 before 侧饿死候选证据）**；
    超限/源文件缺失只记路径 + sha + `why`（manifest 里 `copied:false`），不复制；
  - `repro.md`：复现/校验命令（报告 sha、manifest 校验、两侧 nm_test 命令）。
- 追加语义：同 loop 每次生成 `seq = max+1`，**绝不覆盖**已有目录；同内容重复生成复用最新
  （不新增 seq）。内容包含 raw 副本清单（raw 变了必须新归档）；`gate_report.json` 的时间戳
  不参与幂等判定（它是 run 证据的派生物）。写盘用「临时目录 → 原子 rename」+ 进程锁；
  `.tmp-*` 残留不参与列表/采纳，超过 1h 自动清理。
- 安全：run manifest 里的产物路径必须是相对 run 目录、不含 `..`（远端拉回的 manifest 按不可信
  输入处理）；manifest 里的相对路径读取限制在报告目录内。
- `report_sha256` 只由 `report.json` 内容决定。**校验分两档**：
  - 列表里的 `valid` = **快速校验**（report.json 的 sha 一致 + manifest 文件/raw 副本存在）；
  - `POST .../verify` 与采纳门禁 = **深度校验**（manifest.files + raw 副本逐个 sha256 一致，
    且 report.json sha == manifest.report_sha256）。
- 归档原始字节可通过 `GET .../optimization-reports/{sha}/download` 下载（sha 与 manifest 一致；
  不要用「重新序列化」的 JSON 去比 sha）。
- 真实库/run 目录后续清理不影响报告：raw 副本自包含（`copied:true` 的才可独立复算）。

### 17.2 API

| 端点 | 契约 |
|---|---|
| `GET /api/loops/{id}/optimization-report?write=true` | 兼容保留：生成 append-only 报告；响应在原报告字段上追加 `report_sha256 / report_dir / written_to / manifest / report_valid` |
| `GET /api/loops/{id}/optimization-report?write=false` | 兼容保留：即时计算（不落盘，旧行为） |
| `GET /api/loops/{id}/optimization-reports` | `{reports:[{report_sha256, seq, dir, generated_at, verdict, conclusion, valid, size_bytes, raw_count}]}`（最新在前，≤50） |
| `POST /api/loops/{id}/optimization-reports` | 生成/归档一次报告（201）；返回**原 report 字段平铺** + `report_sha256/content_sha256/seq/report_dir/written_to/manifest/report_valid/stored`；前置：必须有优化后的 benchmark run（否则 409） |
| `GET /api/loops/{id}/optimization-reports/{report_sha256}` | `{manifest, report, valid}`（从磁盘读，校验 sha；不存在 404） |
| `POST /api/loops/{id}/optimization-reports/{report_sha256}/verify` | `{report_sha256, valid, checked_at, files:[{path,ok,expected,actual}], raw:[...], errors:[...]}`；**深度校验**，只读，不重跑 |
| `GET /api/loops/{id}/optimization-reports/{report_sha256}/download` | 原始 `report.json` 字节（`FileResponse`；sha 与 manifest 一致） |

- **采纳门禁 G7 升级**：`accept_evidence` 要求最新归档报告存在且 `valid=true`（替代仅
  “write=True 成功”），报告缺失/校验失败 → 409 `E_ACCEPT_REPORT_UNWRITTEN`。
- 报告内新增 `evidence` 块：两侧 run 的 manifest/产物绝对路径、环境指纹、case 集指纹、
  oracle 指纹（来自 correctness.json）、生成时间；对外表述：「报告可复算/可校验」，
  **不宣称**「独立目录复现已跑通」。

### 17.3 独立目录复现演示

- `scripts/repro_report.py <report_dir 或 report.json>`：校验 manifest 内每个文件 sha256、
  report.json 自身 sha、raw 副本 sha；默认只校验、不执行任何真机命令；`--show-commands`
  打印报告里的两侧复现命令；`--json` 输出结构化结果。
- 演示方式（不依赖 DB/控制台）：`cp -r <report_dir> /tmp/repro-demo && python scripts/repro_report.py
  /tmp/repro-demo` → `ok / valid`。真机重跑仍按报告里的 nm_test 命令人工执行（v1 不自动）。

### 17.4 边界

- raw 复制有大小上限；不复制 run 目录以外的文件；不修改 run 目录。
- 报告结论口径不变（perf-diff / objective_summary）；证据不足时不编造结论。
- 报告仓库只追加不删除（GC 留后续步骤）；`reports/` 已进 `.gitignore`。
- 单进程写盘锁（与 loop_budget 同口径）；多实例/脚本共用同一 report_root 需换文件锁。
- 报告内绝对路径（`evidence.runs[].manifest_path/run_dir`、`source.repo_root`）是本机生成时
  的原路径，换机后以 `raw/` 副本为准（repro.md 已注明）。
- `POST /optimization-reports` 幂等复用时仍返回 201（`stored=false`）。
- 归档目录里 `valid`（列表）= 快速校验；逐文件 sha 以 `POST .../verify` 为准；采纳门禁用最新
  归档的**深度**校验，且要求归档与当前候选/两侧 run 绑定（不一致会自动 append 新归档）。
