# 门禁校验器 v1 · gate_report 契约

> 方案 v1.2 §5（门禁=结果有效 + 业务达标 + 版本一致 + 产物完整）的**机器可读落地**。
> 实现：`backend/app/services/gate.py`；入口：`GET /api/runs/{id}/gate`（`?write=true` 会把
> `gate_report.json` 落到该 run 的产物目录，与 `manifest.json` 在一起）。
> 状态：**P1.5-c 起生效**（退出码不再是结论）。
>
> **对比报告不在本契约里**（A3 口径 B）：优化前后 A/B 跨多个 run，判定见 **`docs/perf-diff-v1.md`**；
> 阶段三在任务**存在 `perf` Loop** 时要求最近一轮 `perf-diff` 的 `verdict=pass`（方案 v1.2 §5）。

## 1. 为什么要有它

v1.0 的门禁只看「产物齐备」，会被两类东西骗过：

1. **报告文件存在但内容是废的**（解析失败、空 cases、上一版代码跑出来的报告）；
2. **退出码 0 但业务没过**（用例 FAIL 被判成 PASS，或必需用例根本没跑）。

所以 v1 的判定拆成四段，每段都可机判：产物完整 → 结果有效 → 业务达标 → 版本一致。

## 2. 输入：三个契约文件

### 2.1 `manifest.json`（每次真实执行都写，由 `scripts/nm_test.sh` 生成）

```json
{
  "schema": "opx-manifest/1",
  "run_id": "r20261015-1040-test",
  "host": "tk36", "host_id": 1,
  "image": "lfx-op-dev-910b:torch210-cann910", "image_id": 52,
  "image_digest": "local-id:sha256:56c1…",     // 必须有：tag 会被重新 push 覆盖，digest 才判得动
  "action": "test",
  "exit_code": 0,
  "code_snapshot_sha256": "44aeffb4…",          // 工作区源码**内容**哈希
  "snapshot": {"files": 7, "truncated": false},
  "spec_version_id": 1,
  "criteria_ref": "ops-precision-standard@1",
  "suite_ref": "cases@0123456789abcdef",
  "device": "0", "lock_mode": "exclusive", "lock_acquired": "yes",
  "finished_at": "2026-10-15T02:45:03+00:00",
  "artifacts": [{"type": "log", "path": "out/run.log", "sha256": "…", "size_bytes": 1655}]
}
```

产物 `type` 由脚本按契约命名（`log` / `correctness` / `performance` / `metrics` / `cases` / `report`），
门禁按类型找文件，不按文件名猜。

### 2.2 `correctness.json`（test 动作，工作区的 `test.sh` 产出）

```json
{
  "suite_ref": "cases@0123456789abcdef",
  "spec_version_id": 1,
  "code_snapshot_sha256": "44aeffb4…",
  "criteria_ref": "ops-precision-standard@1",
  "cases": [
    {"case_id": "add_norm_001", "group": "常规", "status": "PASS", "required": true,
     "shape": [1024], "dtype": "float16", "seed": 20261015, "max_abs_err": 0.0, "tol": 1e-3}
  ],
  "summary": {"PASS": 1, "FAIL": 0, "ERROR": 0, "INVALID": 0, "NOT_RUN": 0}
}
```

`status` ∈ `PASS | FAIL | ERROR | INVALID | NOT_RUN`（枚举外的值直接判无效）；`required`（缺省 true）
决定「没跑」是否算不合格。**报告里带的版本字段必须与这次 run 一致**，否则视为「报告不是这一版跑出来的」。

### 2.3 `metrics.json` / `performance.json`（benchmark / profile 动作）

```json
{"metrics": [
  {"id": "op_time_us", "kind": "duration", "status": "measured", "source": "msopprof OpBasicInfo.csv Task Duration(us)",
   "value": 12.3, "required": true},
  {"id": "cube_ratio", "kind": "compute", "status": "unsupported", "source": "msopprof ArithmeticUtilization.csv aic_cube_ratio",
   "required": false, "assumption": "vector 算子无 cube 单元数据（msopprof 填 NA）"}
]}
```

`status` ∈ `measured | estimated | unsupported | failed`；`estimated` 必须附 `assumption`；
必需指标 `failed` → 判「性能分析未完成」。和 `correctness.json` 一样，报告里带的
`spec_version_id / code_snapshot_sha256 / criteria_ref / suite_ref` 也必须与这次 run 一致，
否则视为「报告不是这一版跑出来的」（`E_REPORT_VERSION_MISMATCH`）。

## 3. 输出：`gate_report`

```json
{
  "run_id": 1, "task_id": 2, "action": "test",
  "generated_at": "2026-10-15T02:46:00+00:00",
  "verdict": "pass | fail",
  "summary": {"checks": 4, "fail": 1, "warn": 2},
  "version_alignment": {"spec_version_id": 1, "code_snapshot_sha256": "…", "suite_ref": "…",
                        "criteria_ref": "…", "image_digest": "…", "device_id": "0"},
  "findings": [{"code": "E_REQUIRED_ARTIFACT_MISSING", "level": "fail", "message": "…",
                "evidence": {"need": "correctness"}}],
  "notes": ["退出码 0 不等于业务通过（§3.1 硬规则 1）", "必需用例未执行（NOT_RUN）不计通过（§4.5）"]
}
```

`verdict = fail` 当且仅当存在 `level = fail` 的 finding（warn 不影响判定，但要在界面上显示出来）。

## 4. fail code 清单（v1）

| code | level | 触发条件 |
|---|---|---|
| `E_NO_MANIFEST` | fail | `runs.manifest_path` 空或文件不存在 |
| `E_MANIFEST_UNREADABLE` | fail | manifest 不是合法 JSON |
| `E_MANIFEST_INCOMPLETE` | fail | 缺 `run_id/exit_code/artifacts/code_snapshot_sha256/spec_version_id/criteria_ref/image_digest` |
| `E_RUN_NOT_FINISHED` | fail | run 状态不是 `ok/failed`（还在跑、或被重启标成 interrupted） |
| `E_EXIT_UNKNOWN` / `E_EXIT_NONZERO` | fail | 退出码缺失 / 非 0 |
| `E_ARTIFACT_EMPTY` | fail | manifest 里没有任何产物 |
| `E_ARTIFACT_MISSING` | fail | 产物文件不存在 |
| `E_ARTIFACT_SHA_MISMATCH` | fail | 产物 sha256 与实际不符 |
| `E_REQUIRED_ARTIFACT_MISSING` | fail | 按动作缺必需产物类型（见 §5） |
| `E_SNAPSHOT_MISSING` / `E_SNAPSHOT_MISMATCH` | fail | 代码快照缺失 / run 行与 manifest 不一致 |
| `E_SPEC_VERSION_MISSING` / `E_SPEC_VERSION_MISMATCH` | fail | 规格版本缺失 / 不一致 |
| `E_CRITERIA_MISSING` / `E_CRITERIA_MISMATCH` | fail | 判据档缺失 / 不一致 |
| `E_CRITERIA_VERSION_MISSING` | fail | 判据档没写 `@版本`（判不了「报告与本判据一致」） |
| `E_IMAGE_DIGEST_MISSING` | fail | 镜像 digest 缺失（三硬规则第 3 条） |
| `E_SUITE_REF_MISSING` | fail（test） / warn（其他） | 没记用例集 ref |
| `E_CORRECTNESS_REPORT_MISSING` | fail | test 动作没有 `correctness` 类型产物 |
| `E_SMOKE_REPORT_MISSING` | fail | smoke 动作没有 `smoke` 类型产物（应产出 `smoke.json`） |
| `E_SMOKE_FAILED` | fail | 冒烟最小用例未通过（`verdict != PASS`；开发完成前必须修） |
| `E_SMOKE_CASE_MISSING` | fail | `smoke.json` 里没有 `case.case_id`（不知道跑的是哪条最小用例） |
| `E_METRICS_REPORT_MISSING` | fail | profile 动作没有 `metrics` 类型产物（D9 已定：四类指标可采） |
| `E_REPORT_UNREADABLE` | fail | correctness/metrics JSON 解析失败 |
| `E_REPORT_VERSION_MISMATCH` | fail | 报告里的版本字段与 run 不一致（报告不是这一版跑的） |
| `E_CASE_LIST_EMPTY` | fail | `correctness.json` 没有 cases |
| `E_CASE_STATUS_INVALID` | fail | 用例状态不在枚举里 |
| `E_CASE_NOT_RUN` | fail | 必需用例 `NOT_RUN` |
| `E_CASE_FAILED` | fail | 有用例 `FAIL` 或 `ERROR` |
| `E_METRIC_STATUS_INVALID` | fail | 指标状态不在枚举里 |
| `E_METRIC_REQUIRED_FAILED` | fail | 必需指标 `failed` → 性能分析未完成 |
| `W_ARTIFACT_NO_SHA` | warn | 产物没记 sha256 |
| `W_METRIC_NO_SOURCE` | warn | 指标没记来源 |
| `W_METRIC_ESTIMATED_NO_ASSUMPTION` | warn | 估算值没写假设 |
| `W_DEVICE_UNRECORDED` | warn | 基准/采集没记设备与锁模式 |
| `W_DEVICE_SHARED` | warn | 基准/采集不是独占 → 耗时可能被邻居污染 |

## 5. 按动作的必需产物（`gate.REQUIRED_ARTIFACTS`）

| 动作 | 必需产物类型 | 说明 |
|---|---|---|
| `build` | `log` | 构建日志 |
| `smoke` | `log`, `smoke` | 开发内冒烟（2026-10-23）：编译 + 一条最小 required match 用例跑通；`smoke.json.verdict=PASS` 才算过 |
| `test` | `log`, `correctness` | 逐用例结果 + 正确性报告 |
| `benchmark` | `log`, `performance` | 与 profiling 分开执行 |
| `profile` | `log`, `metrics` | D9 已定（2026-10-16 真机实测）：msopprof 四类指标均可采 |

## 6. 边界（明确不做的）

- **不重试、不改代码**：门禁只判「过 / 不过 + 为什么」，修复是 Loop（§4.4）的事。
- **不把 warn 当拦**：`W_DEVICE_SHARED` 这类只提示测量条件不干净，判定权在人（方案 §5 不擅自加条件）。
- **不解析人写的报告**：只认上表的机器可读文件；`.md` 只是给人看的伴生品。
