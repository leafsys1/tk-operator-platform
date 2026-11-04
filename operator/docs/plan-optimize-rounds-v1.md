# 优化轮次 v1（功能阶段一）· 设计与契约

> 注：本文的“阶段一/二/三”只指优化轮次功能的分期，不是 `PROGRESS.md` 里的历史里程碑 P1/P2/P3。
> 2026-10-21 起实现。目标：把「性能与优化」页从「手动 A/B」升级为
> **建轮次（目标/约束/用户思路）→ 自动复用或用真机采基线 → agent 改代码 → 一键候选验证 → 人工采纳成版本 → 可复现报告**。
>
> 阶段一原定不做：思路搜索/思路池、测量预算硬兜底、跨芯片 port、报告 Markdown 渲染、自动采纳。
> **2026-10-22 更新**：思路搜索/思路池与测量预算硬兜底已在阶段二/三落地；跨芯片 port、报告 Markdown 仍未做；自动采纳继续永久关闭。
> **2026-10-23 更新（`1cf4408` 审查修复）**：预算 `runs_per_action` 仍是硬限制，但审计改为「页面预留 → `nm_test.sh` 带 `OPX_BUDGET_AUDIT_ID` claim」的一次性预留（不再认 `OPX_BUDGET_RESERVED`），启动失败会回收额度；自动优化只允许 objective 轮次（旧 Loop 只读 409），且 `pipeline.verifier` 为空会回退；`allowed_paths` 为空时 fail-closed、外指符号链接按越界回滚；perf-diff 的精度轮/`dimension=all`/缺测量口径见 `docs/perf-diff-v1.md`。
> **2026-10-23 二次修订（按用户规划）**：`target_dimension` 新增 `text` 维度——思路池建轮次用「文字描述的优化点」（`objective.focus`）作为目标，不指定目标用例（perf-diff 按 benchmark primary 兜底）；否决对所有未关闭轮次可用且可同时回退工作区（`reject.revert`，优先自动优化改前快照、其次 baseline 存点）；默认预算 `runs_per_action` 2→3（基线+候选后留 1 次失败重试）。
>
> **实现状态（2026-10-21 收工）**：本阶段已落地并本机验证。按用户实测反馈，旧 `PerfDiffPanel` 已删除，
> 「让 agent 改代码 + 优化前后对比」并入 `OptimizationRoundPanel` 的轮次卡片；验证进度支持已运行时长、
> 延迟输出提示与刷新恢复。另修复 `nm_test.sh` 快照 skip 少 `.opx-chat` 的口径漂移（基线存档 sha ≠ run sha）。
> 当前实机有一个待重跑基线的轮次（任务 #11 / Loop #6），交接事项见 `docs/NEXT-2026-10-22.md`。

## 1. 数据模型（迁移 `b1c2d3e4f5a6`，down_revision `a9b8c7d6e5f4`）

- `loops.objective_json`（Text，可空）：本轮优化的目标/约束/预算/用户补充文本/目标快照。
- `savepoints.loop_id`（Integer，FK `loops.id`，可空，索引 `ix_savepoints_loop_id`）。
- `Loop.objective` property（解析 JSON，只读）；`LoopOut.objective: dict | None`。
- `SavepointOut.loop_id: int | None`。
- 结论：轮次（Loop）是优化单位；存盘点通过 `loop_id` 归属轮次；`kind` 扩展
  `baseline | candidate | accept | rejected`（原有 manual/auto/rescue 不变）。

`objective_json` schema：

```json
{
  "schema": "opx-optimization-objective/1",
  "label": "1M 带宽优化",
  "type": "perf | precision | hardware",
  "target_dimension": "cases | all",
  "target_cases": ["bench_1m"],
  "constraints": {
    "precision_non_regression": true,
    "perf_non_regression": false,
    "allowed_paths": ["ops/<op>/"]
  },
  "budget": {"diagnose_rounds": 3, "verify_rounds": 3, "runs_per_action": 2},
  "user_text": "用户补充的优化思路/约束（原样保存）",
  "targets_snapshot": {
    "benchmark_run_id": 32, "test_run_id": 30,
    "primary_case": "bench_1m", "suite_ref": "cases@...",
    "cases": [{"case_id": "bench_1m", "n": 1048576, "dtype": "float32",
               "primary": true, "required": true, "note": "..."}],
    "failing_cases": [{"case_id": "case_001", "status": "FAIL"}]
  }
}
```

## 2. 目标用例：数据驱动，不写死 shape

`GET /api/tasks/{task_id}/optimization-targets`

- 取该任务**最近一次 status=ok 的 benchmark run** 与**最近一次 status=ok 的 test run**；
  用 `gate.artifact_paths(run)` 找 `performance.json` / `correctness.json` 并解析。
- 返回：

```json
{
  "task_id": 11,
  "benchmark_run_id": 32,
  "test_run_id": 30,
  "suite_ref": "cases@...",
  "primary_case": "bench_1m",
  "cases": [{"case_id": "bench_1m", "n": 1048576, "dtype": "float32",
             "primary": true, "required": true, "note": "..."}],
  "failing_cases": [{"case_id": "case_001", "status": "FAIL"}],
  "test_cases": [{"case_id": "case_001", "status": "FAIL"}],
  "hint": "缺少基准报告：先跑一次「基准测试」" 
}
```

- 没有基准 → `cases: []`、`benchmark_run_id: null`、`hint` 提示；没有 test → `failing_cases: []`。
- 字段一律有默认：`n`/`dtype`/`primary`/`required`/`note` 缺失就给 null/false（不同算子的 case 语义不同，UI 不假设 `n`）。
- 精度轮次默认目标用例 = `failing_cases` 的 case_id。

## 3. 建轮次

`POST /api/tasks/{task_id}/optimization-rounds`

```json
{
  "objective": { "...上一节 schema..." },
  "label": "可选标题",
  "hypothesis": "可选假设（缺省由 objective 生成）",
  "attach_baseline_runs": true
}
```

- 校验：`type ∈ perf|precision|hardware`；`target_dimension ∈ cases|all`；
  `cases` 维度必须给非空 `target_cases`，且必须在目标快照里存在（精度类型按 test 用例校验，性能/硬件按 benchmark 用例校验）。
- 建 `Loop`：`kind="perf"`（三类都用 perf-diff 所需成对 run；类型在 objective 里区分）：
  - `scope = label or f"{type} 优化：{target_cases 或 全部}"`
  - `hypothesis = payload.hypothesis or "基于目标硬件画像与现有证据，寻找可回退的最小改动"`
  - `judge_condition = "目标用例 ...；精度不退化；同设备/镜像/规格/用例集同条件 A/B"`
  - `objective_json` 写入（含 `targets_snapshot`）。
- 自动建**基线存档**：`savepoints` 行 `kind="baseline"`、`loop_id=loop.id`、`label="基线 · ..."`。
- `attach_baseline_runs=true` 时，把该任务下**未挂轮次**、`code_snapshot_sha256 == 基线 sha` 的
  最近一次 ok `test` 与 `benchmark` run 挂到本 Loop（`runs.loop_id`）——复用刚跑过的基线，省一次真机。
- 返回：

```json
{
  "loop": {...LoopOut with objective...},
  "baseline_savepoint": {...SavepointOut...},
  "attached_runs": [{"id": 30, "action": "test"}, {"id": 32, "action": "benchmark"}],
  "targets": {...optimization-targets...}
}
```

- 错误：404 任务；400 参数/目标用例不在快照；409 工作区不可用 / 基线存档失败。

## 4. 保存点扩展（`services/savepoints.py`）

- `Savepoint.loop_id`；`create(..., loop_id=None)`；`out()` 增 `loop_id`。
- 新增 `create_for_sha(db, task, sha, *, label, kind, loop_id=None, note=None)`：
  - 优先用 `<archive_root>/sources/<sha>.tar.gz`（v1 每次动作前已自动归档）；
  - 归档不存在时，若当前工作区 sha == `sha` 则现打包，否则抛 `SavepointError`。
  - 返回 `Savepoint`（只 flush，不 commit）。
- `GET /api/tasks/{id}/savepoints` 增加可选 `loop_id` 过滤。
- **自动候选存档**：`action_runner._finalize` 在「benchmark 成功 + job.loop_id 存在 + loop.objective 存在」时：
  - 取 run 的 `code_snapshot_sha256`；若它 == 基线存档 sha（即这是基线那轮）→ 跳过；
  - 否则 `create_for_sha(kind="candidate", loop_id=job.loop_id, label="候选 · ...")`；
  - 同 Loop 最新 candidate 同 sha → 跳过；异常只 log，不影响动作作业。
- 这样即使页面关了，候选验证完也会自动留下候选版本。

## 5. 验证流程（阶段一用前端顺序调度，不新增后端队列）

- 基线：建轮次时若能复用已有 run 就复用；否则页面「跑基线 test+benchmark」：
  依次 `POST /api/tasks/{id}/actions`（`dry_run=false, action=test|benchmark, loop_id=<id>, device_id=<pin>`），
  轮询 `GET /api/action-jobs/{jobId}` 直到结束；两个都 ok 才算基线就绪。
- 候选：agent 改完代码后「跑候选验证」，同上；benchmark 成功时后端自动落 candidate。
- **设备 pin**：`task.device_id` > 最近一次 ok run 的 device > 首个 plan 返回的 device；同一轮两次动作必须用同一张卡。
- 同一任务同一时刻只允许一个动作作业（后端已有守卫）；失败即停，展示错误和日志尾部。
- 前端可复用 `planAction` / `runAction`（`planAction` 需补 `device_id` 参数，后端 `ActionRequest` 已支持）。

## 6. 采纳 / 否决

`POST /api/loops/{id}/accept`

```json
{"label": "可选版本名", "note": "可选"}
```

- 必须存在**该 Loop 最新 candidate**，否则 409「先跑一轮候选验证」。
- 版本号 `n = 该任务 kind=="accept" 的存点数 + 1`；把 candidate 行原地升级：
  `kind="accept"`、`label=payload.label or f"c{n} · {loop.scope}"`、`note`。
- 关闭 Loop：`status="closed"`、`conclusion="accepted"`、`ended_at=now`。
- 返回 `{loop, version, version_number, workspace_matches}`；`workspace_matches` = 当前工作区 sha 是否等于候选 sha（不一致只提示，不改工作区）。
- 已采纳过 → 409。

`POST /api/loops/{id}/reject`

```json
{"reason": "性能回退，不采纳"}
```

- 最新 candidate 行改为 `kind="rejected"`、`note=reason`；关闭 Loop `conclusion="rejected：<reason>"`。
- 不自动回滚工作区（让人在「版本」分区一键回滚到 baseline；rescue/baseline 都在）。

## 7. 可复现报告

`GET /api/loops/{id}/optimization-report?write=false`

```json
{
  "schema": "opx-optimization-report/1",
  "task_id": 11, "loop_id": 6,
  "label": "...", "objective": {...},
  "targets": {"benchmark_run_id": 30, "test_run_id": 32, "primary_case": "bench_1m",
              "target_cases": ["bench_1m"], "suite_ref": "cases@..."},
  "code": {"baseline_savepoint_id": 12, "candidate_savepoint_id": 13,
           "version_savepoint_id": 14, "baseline_sha256": "...", "candidate_sha256": "..."},
  "runs": {"before": {"test": 30, "benchmark": 31}, "after": {"test": 32, "benchmark": 33}},
  "perf_diff": {...GET /api/loops/{id}/perf-diff 的完整结果...},
  "precision": {"before": {"case_id": "PASS"}, "after": {},
                "target_cases": [{"case_id": "case_001", "before": "FAIL", "after": "PASS",
                                  "newly_passed": true, "regressed": false}],
                "newly_passed": 1, "regressions": [], "non_regression": true},
  "objective_summary": {"type": "perf", "verdict": "pass", "conclusion": "improved",
                        "headline": "主用例 -3.2%，精度不退化"},
  "repro": {"before": {"test_command": "...", "benchmark_command": "..."},
            "after": {"test_command": "...", "benchmark_command": "..."},
            "env": {"spec_version_id": 7, "criteria_ref": "...", "image_digest": "...", "device_id": "1"}},
  "generated_at": "..."
}
```

- `objective_summary`：
  - `perf` / `hardware`：取 `perf_diff.verdict/conclusion`（hardware 额外带 `chip`：`target_chip` + 选中的 `ops/chips/*.yaml` 摘要）；
  - `precision`：按目标用例逐例 before/after 状态；`newly_passed` / `regressions`；
    `verdict = fail` 当且仅当有 regressions（已 PASS 变差）；`conclusion = improved` 当 `newly_passed > 0` 且无回退，
    否则 `no_significant_change`/`inconclusive`。性能对比照常放进 `perf_diff` 作为上下文。
- `write=true` 写到「优化后那轮 benchmark run」目录下的 `optimization-report.json`。

## 8. Agent prompt 注入（`services/optimize.py`）

- `build_prompt` 增加「## 本轮目标与约束」：objective type、目标用例、约束、用户补充文本；
  hardware 类型额外强调芯片画像。
- `hardware_brief(workspace, chip)`：按 `task.target_chip` 在 `ops/chips/*.yaml` 里找
  `models` 包含该型号的文件；找不到退回第一份（本阶段实现，别喂错画像）。

## 9. 前端（阶段一）

- 新组件 `OptimizationRoundPanel.tsx`，作为 perf 分区**唯一主面板**（原 `PerfDiffPanel` 已并入并删除）：
  - 头部「优化轮次」+ 说明 +「新建优化轮次」。
  - 表单（浮层，样式参照 `TaskDeleteDialog`）：
    - 标题、目标类型（性能/精度/硬件适配）、目标维度（用例/全部）；
    - 目标用例多选：性能/硬件取 `targets.cases`，精度取 `targets.test_cases` 且默认勾 `failing_cases`；
    - 约束：精度不退化（性能/硬件默认勾）、性能不退化（精度默认不勾）、允许改动范围（默认 `ops/<op>/`）；
    - 预算：判断轮数/验证轮数/每动作执行次数（阶段一只存；2026-10-22 起每动作执行次数由后端与 `nm.sh --loop` 强制并审计）；
    - **用户补充思路/约束** textarea；
    - 假设 textarea（可选）；
    - 提交 → `POST optimization-rounds` → 刷新。
  - 轮次列表（`loops.filter(l => l.objective)`，新→旧）：label、类型/状态 chip、目标用例 chip、
    基线/候选/版本存点 chip、两侧 run 就绪状态；
    操作：`跑基线`（缺 before 时）/`跑候选验证`（缺 after 时）/`采纳为版本`/`否决`/`导出报告`/`刷新`；
    运行时展示阶段（plan→test→benchmark）+ 日志尾部，禁用按钮。
  - 提示：「让 agent 改代码在下方『自动优化』区；改完回这里跑候选验证」。
- `TaskWorkbench.tsx`：perf 分区只渲染 `<OptimizationRoundPanel task={task} onChanged={refresh} />`。
- `types.ts` / `lib/api.ts`：新增 `OptimizationTargets`、`OptimizationObjective`、
  `OptimizationRoundCreate`、`OptimizationReport`；`Loop.objective`、`Savepoint.loop_id`；
  `savepoints(taskId, loopId?)`、`optimizationTargets`、`createOptimizationRound`、
  `acceptLoop`、`rejectLoop`、`optimizationReport`；`planAction(taskId, action, deviceId?)`。
- 「版本」分区不用改（`loop_id` 只是多一个字段）；SavepointPanel 可忽略。

## 10. 验收

- `pytest` 全量绿（新增 `tests/test_optimization_rounds.py`）：
  targets 解析（有/无产物、缺字段兜底）；建轮次（objective/基线存档/复用 run/校验）；
  `create_for_sha`（归档存在/现打包/找不到）；自动 candidate（基线轮跳过、候选轮生成、去重）；
  accept（编号、无候选 409、重复采纳 409）；reject；报告（perf 与 precision 的 objective_summary、
  write 落盘）；savepoints loop_id 过滤；迁移往返含两列。
- `tsc --noEmit` / `npm run build` 通过。
- 真实库迁移 + 重启；`GET /api/tasks/11/optimization-targets` 200 且能看到 case；
  页面「性能与优化」出现「优化轮次」卡片与表单；建轮次 → 基线存档出现。
- 端到端验收只在临时库/测试里建轮次，**不要**在真实任务上留测试轮次（Loop 目前无删除接口）。

## 11. 文件所有权

- 后端（backend-dev）：`app/models.py`、`schemas.py`、`main.py`、`services/optimization.py`（新）、
  `routers/optimization.py`（新）、`services/savepoints.py`、`services/action_runner.py`、
  `services/optimize.py`、`routers/savepoints.py`、`alembic/versions/b1c2d3e4f5a6_optimization_rounds.py`（新）、
  `tests/test_optimization_rounds.py`（新）、`tests/test_savepoints.py`（如需）。
- 前端（frontend-dev）：`web/src/components/OptimizationRoundPanel.tsx`（新）、`TaskWorkbench.tsx`、
  `lib/api.ts`、`types.ts`。
- Lead：本文档、真实库迁移、全量回归、真实接口冒烟、README/PROGRESS、最终验收。
