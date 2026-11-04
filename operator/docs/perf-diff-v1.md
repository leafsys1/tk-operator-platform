# 自动优化对比报告 · `performance-diff.json` 契约（v1）

> 方案 v1.2 §4.6 / §5（`loops.kind=perf` 的成对 Run 关联 + `performance-diff.json`；阶段三在存在 perf Loop 时要求本报告 `verdict=pass`）与
> 需求 C6（自动优化闭环）的**机器可读落地**。实现：`backend/app/services/perf_diff.py`；
> 入口：`GET /api/loops/{id}/perf-diff`（`?write=true` 把报告落到优化后那次 benchmark 的产物目录）。
> 状态：**M3-a 起生效**（2026-10-16 实测，Loop #2 / run #8–#11；Loop #3 / run #15–#18 第二轮验证）。

## 1. 为什么要有它

需求的自动优化不是「给建议」，而是**闭环 + 对比证据**：必须真的做优化前后的**同条件 A/B**，出可复现报告。
难点是「同条件」——两次执行的**规格版本 / 判据档 / 镜像 / 基准用例集**必须一致，
否则对比的是两组不同条件，结论无效。本报告把这件事做成机判。
（2026-10-28 修订：**设备不再作为硬条件**——执行侧指定卡被占时会自动切空闲卡，
`device_id` 只记录进 `comparability.device_id`，跨卡绝对耗时的可比性交由报告使用方判断。）

## 2. 输入：优化 Loop 的成对 Run

一个 `loops.kind == "perf"` 的优化 Loop（`runs.loop_id` 关联）里：

- 按 `runs.code_snapshot_sha256` 分成**两侧**：出现早的 = **优化前**，晚的 = **优化后**；
- 每侧各需一次 `test`（`correctness.json` → 精度不退化）与一次 `benchmark`（`performance.json` → 耗时）；
  同一快照重跑属「同侧复现」，取最新一条，不改变结论。

关联方式：`POST /api/tasks/{id}/runs` 带 `loop_id`，或 `scripts/nm_test.sh` 设 `OPX_LOOP_ID=<id>`
（登记时自动挂到该 Loop），或 `PATCH /api/runs/{id}`（只许改归属这一个字段）。

## 3. 判定口径

| 段 | 条件 | 处理 |
|---|---|---|
| 配对 | ≥2 个代码快照、两侧各有 test+benchmark | 不足 → `E_PERF_LOOP_INCOMPLETE` |
| 可比性（硬） | `spec_version_id` / `criteria_ref` / `image_digest` 前后一致 | 不一致 → `E_PERF_DIFF_NOT_COMPARABLE` |
| 可比性（软） | `device_id` 只记录不硬判（允许执行侧自动切卡；跨卡绝对耗时的可比性交由使用方判断） | — |
| 可比性（硬） | **基准用例集指纹**（`case_id`/`n`/`dtype`/`primary`）前后一致 | 不一致 → `E_PERF_DIFF_NOT_COMPARABLE` |
| 可比性（软） | `suite_ref` 只记录不硬判（它是 `cases/` 的内容哈希，会随基准配置等非正确性文件一起变） | — |
| 耗时 | 逐用例比较 benchmark 的**候选设备侧中位数**；差异 > max(不确定性, 最小效应 2%) 才算「显著」 | 见下表结论枚举 |
| 精度（硬） | 优化后 `correctness` 不得出现 FAIL/ERROR，且逐用例状态不得比优化前差 | 退化 → `E_PRECISION_REGRESSION` |
| 性能回退 | — | **只 warn（不设加速比硬线，需求已定）** `W_PERF_REGRESSED` |

逐用例结论 `verdict` ∈ `improved | no_significant_change | inconclusive | regressed`，保守聚合为总结论
（`regressed > inconclusive > improved > 全部 improved 才 improved > 其余 no_significant_change`）。
驱动总结论的用例集合按 `loop.objective` 判定（2026-10-23 修复后口径）：

- `type=perf|hardware` + `target_dimension=cases`：只用 `target_cases`（benchmark case_id）；
- `type=perf|hardware` + `target_dimension=all`：用**全部可配对用例**（任一 regressed 即 regressed）；
- `type=perf|hardware` + `target_dimension=text`（文字优化点，思路池建轮）：不指定目标用例，
  按 benchmark primary 兜底（focus 只进 prompt/展示，不参与用例配对）；
- `type=precision`：`target_cases` 是 correctness/test 用例 id，不与 benchmark 配对 → 回到 benchmark primary
  （精度目标由 `optimization.report` 的 precision 段判定）；
- 无 objective 的旧 Loop：benchmark primary（兼容口径）。

缺有效测量（任一侧 `median_us` 缺失或 ≤0）的用例：`delta_pct=null`、`missing_data=true`、
`verdict=inconclusive`，记 `W_PERF_MISSING_DATA`，**不参与** improved/regressed；目标用例 id 在报告里
根本不存在才记 `E_TARGET_CASE_MISSING`（并强制 conclusion=inconclusive，避免 fail + improved 自相矛盾）。
`summary.decision_basis` 取值 `target_cases | all_cases | benchmark_primary`。

## 4. 输出：`performance-diff.json`

```json
{
  "schema": "opx-perf-diff/1",
  "loop_id": 2, "task_id": 2,
  "verdict": "pass | fail",
  "conclusion": "improved | no_significant_change | inconclusive | regressed",
  "hypothesis": "…本轮优化假设…", "judge_condition": "…判定条件…",
  "min_effect_pct": 2.0,
  "sides": {
    "before": {"code_snapshot_sha256": "…", "test_run_id": 8, "benchmark_run_id": 9, "run_ids": [8, 9]},
    "after":  {"code_snapshot_sha256": "…", "test_run_id": 10, "benchmark_run_id": 11, "run_ids": [10, 11]}
  },
  "comparability": {"spec_version_id": {"same": true, "hard": true}, "suite_ref": {"same": false, "hard": false}, "bench_cases": {"same": true, "hard": true}},
  "cases": [
    {"case_id": "bench_1m", "n": 1048576,
     "before_us": 8.28, "after_us": 8.18, "delta_pct": -1.21, "missing_data": false,
     "uncertainty_pct": 2.99, "verdict": "no_significant_change",
     "before_baseline_us": 8.45, "after_baseline_us": 8.50},
    {"case_id": "bench_64k", "n": 65536,
     "before_us": 5.24, "after_us": 0.0, "delta_pct": null, "missing_data": true,
     "uncertainty_pct": 0.0, "verdict": "inconclusive"}
  ],
  "precision": {"cases": [{"case_id": "add_norm_001", "before": "PASS", "after": "PASS", "regressed": false}],
                "regressed_cases": [], "non_regression": true},
  "summary": {"cases": 2, "fail": 0, "warn": 2, "primary_verdict": "inconclusive",
              "decision_basis": "target_cases", "target_case_ids": ["bench_64k"]},
  "recommendations": [
    {"id": "R2-1", "title": "本轮优化尝试", "affected_cases": ["bench_1m"],
     "observed_evidence": ["after runs [10, 11]（benchmark #11）", "op_time_us=8.18"],
     "bottleneck_hypothesis": "…", "confidence": "medium", "proposed_change": "…",
     "expected_observation": "…", "constraints_and_risks": "…",
     "validation_method": "同条件 A/B（本报告 cases）", "status": "inconclusive"},
    {"id": "R2-2", "status": "proposed", "affected_cases": ["bench_64k"], "skill": "common/core_shrink_design.md"}
  ],
  "findings": [{"code": "W_PERF_NO_SIGNIFICANT_CHANGE", "level": "warn", "message": "…"}],
  "notes": ["自动优化验收 = 闭环 + 对比报告；性能提升幅度不设硬线。"]
}
```

`verdict = fail` 当且仅当存在 `level = fail` 的 finding。**性能回退不判 fail**，只如实记 `regressed` + warn；
但 objective 声明 `constraints.perf_non_regression=true` 时，目标/全部对比用例显著回退记
`E_TARGET_PERF_REGRESSION` → fail（默认性能轮不勾该约束，保持「回退只 warn」的历史口径）。

**2026-10-23 增补（同一 schema `opx-perf-diff/1`，字段向后兼容）**：逐用例新增 `missing_data`，
缺有效测量时 `delta_pct` 为 `null`；`summary` 新增 `decision_basis` / `target_case_ids`；
`optimization.report` 的 `objective_summary` 在「软口径回退」时 verdict 记 `warn`（不再误报 `pass`）。

## 5. code 清单（v1）

| code | level | 触发条件 |
|---|---|---|
| `E_NOT_PERF_LOOP` | fail | Loop 不是 `kind=perf` |
| `E_PERF_LOOP_INCOMPLETE` | fail | 不足两个代码快照，或某侧缺 test/benchmark |
| `E_PERF_DIFF_INCOMPLETE` | fail | 优化前/后某一侧没有 benchmark 或 test |
| `E_PERF_DIFF_NOT_COMPARABLE` | fail | 规格/判据/镜像 或 基准用例集指纹 前后不一致（设备只记录、不在此列） |
| `E_REPORT_UNREADABLE` | fail | 读不到某侧的 `performance.json` / `correctness.json` |
| `E_PERF_DIFF_NO_CASES` | fail | 前后基准报告没有可配对的用例（case_id 对不上） |
| `E_TARGET_CASE_MISSING` | fail | objective 指定的 benchmark 目标用例在报告里不存在（无法配对） |
| `E_TARGET_PERF_REGRESSION` | fail | objective 声明 `perf_non_regression=true`，且目标/全部用例显著回退 |
| `E_PRECISION_REGRESSION` | fail | 优化后有用例从 PASS 变差（**硬拦**） |
| `E_PRECISION_UNCHECKED` | fail | 优化后正确性报告没有用例，精度不退化判不了 |
| `W_PERF_MISSING_DATA` | warn | 用例缺 before/after 有效耗时中位数（`delta_pct=null`），不参与判定 |
| `W_PERF_REGRESSED` | warn | 目标/主用例耗时变慢（超过不确定性）——不设加速比硬线，如实记录 |
| `W_PERF_INCONCLUSIVE` | warn | 耗时差异落在测量不确定性内，无法确认提升 |
| `W_PERF_NO_SIGNIFICANT_CHANGE` | warn | 耗时无显著变化（提升不显著如实报告） |

## 6. 边界（明确不做的）

- **不判「提升幅度」是否达标**：需求已定「不设加速比硬线」——本报告只给**可复现的对比证据 + 诚实结论**。
- **不替 agent 改代码**：报告只回答「这轮优化是否有效、精度有没有退化」；改代码是 Loop + agent 的事。
- **不重算候选耗时**：耗时主值直接取 benchmark 的 `Task Duration(us)` 中位数，与 `gate-report-v1.md` 同源。
