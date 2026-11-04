# 用例矩阵（cases）

方案 v1.1 §4.5 的载体。门禁按**用例集内容哈希**判版本：`nm_test.sh` 对 `cases/` 整目录算
`suite_ref = cases@<16位哈希>`，报告里的 `suite_ref` 必须与本次 run 一致。

```
cases/
  README.md                    ← 本文件
  elementwise_add/cases.json   ← 首个算子的用例矩阵（规格 v1：float32、同 shape、不广播）
```

## 用例字段

| 字段 | 说明 |
|---|---|
| `case_id` | 稳定 ID（§3.1 硬规则 2：逐用例结果要有稳定 Case ID），一旦产生证据就不再改名 |
| `group` | `常规` / `边界` / `数值边界` / `异常` / `回归`（§4.5 的组） |
| `required` | 缺省 true。**必需用例没跑（NOT_RUN）不算通过**（门禁 `E_CASE_NOT_RUN`） |
| `n_a` / `n_b` | 两个输入的元素数；不等即「形状不一致」 |
| `dtype` | 规格 v1 只支持 `float32`；其他 dtype 的用例是**异常组**（要求实现拒绝） |
| `gen_a` / `gen_b` | 输入生成方式（`uniform`/`normal`/`const`/`zeros`/`special`）；**参数与 seed 一起决定输入**，是可复现的一部分 |
| `seed` | 随机种子（§4.5：每用例要记输入生成方式与种子） |
| `expected` | `match`（与标杆比对通过）或 `reject`（实现必须拒绝） |
| `expect_reject_code` | `reject` 用例期望的拒绝码（实现必须打印同一码，避免「因为别的原因失败」被当成拒绝通过） |
| `note` | 这条用例为什么存在 |

## 分组现状

| 组 | 条数 | 说明 |
|---|---|---|
| 常规 | 4 | 典型尺寸 / 多核切分 / N 上界 / 大值域 |
| 边界 | 6 | 最小 N、TILE±1、非 32B 对齐、N 上界-1、**每核多 TILE**（`add_edge_006`） |
| 数值边界 | 5 | 零值、次正规数、正负相消、NaN/Inf 传播、上溢 |
| 异常 | 4 | N=0、N 超上界、形状不一致、dtype 不支持 |
| 回归 | **0** | 尚无「已发现问题的最小复现用例」——M1 是第一版实现，**不造假用例**；发现缺陷后按 §4.5 补在此组 |

## 与硬约束的关系

- `N ∈ [1, 1048576]`、`same_shape = true`、`broadcasting = false`、`supported_dtypes = [float32]`
  都来自**规格 v1**（`spec_versions` 表里那一版），不是这里现编的。
- 数值边界组的「NaN/Inf 传播」是为了把规格 v1 里 `nan_inf_policy: UNKNOWN` 这一项**用实测结论补上**；
  结论回填规格时需要出新的规格版本（v2），不改这一版的判定口径。
