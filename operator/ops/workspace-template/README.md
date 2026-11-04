# 工作区契约模板（四动作入口）

方案 v1.1 §6 的统一执行入口在容器里调用**工作区根的** `<动作>.sh`，产物写 `/out`，日志由入口 tee 到 `/out/run.log`：

| 动作 | 入口 | 该产出什么 |
|---|---|---|
| `build` | `build.sh` | 容器内编译通过；构建日志 |
| `test` | `test.sh` | 用例矩阵（常规/边界/异常）逐例结果 + 正确性报告 `correctness.{json,md}` |
| `benchmark` | `benchmark.sh` | 与 profiling **分开**执行；`performance.json` + 原始样本（/out/evidence/） |
| `profile` | `profile.sh` | 采指标（耗时/搬运/存储/计算单元负载）；每项标 `measured\|estimated\|unsupported\|failed` |

约定：

- 缺哪个入口脚本，`POST /api/tasks/{id}/actions` 算 plan 时就会报「工作区缺入口脚本 X」——**前置检查是真拦的**。
- `test`/`benchmark`/`profile` 还要求工作区有 **`cases/` 用例集目录**（用例矩阵的载体）；缺了同样拦。
- 模板里的脚本**故意以 2 退出**并写明「NOT IMPLEMENTED」：宁可明确失败，也不让门禁被"看起来跑过了"骗过。
- 把模板复制进新任务的工作区，再由交付约定的实现替换掉内容（不要在模板里塞真实现）。
- 产出的报告必须是**机器可读**的（`correctness.json` / `performance.json` / `metrics.json`），
  字段与 fail code 见 `docs/gate-report-v1.md`：门禁只认这些文件，`.md` 只是给人看的伴生品。
- 报告里要带 `spec_version_id` / `code_snapshot_sha256` / `criteria_ref` / `suite_ref`，
  与这次 run 对齐；不一致会被判成「报告不是这一版跑出来的」。
