# ops-precision-standard@1 · 出处与指纹

| 项 | 值 |
|---|---|
| 档名@版本 | `ops-precision-standard@1`（门禁 `criteria_ref` 逐字认这个字符串，含 `@1`） |
| 实现副本 | `mixed_tolerance_check.py` — sha256 `ccc41e7f083c5781e4bee24a25d140100f2b7bc3d91e85940d749aab02c5747a` |
| 判据正文副本 | `float_compute.md` — sha256 `2dd3af46857f86d86fa7379a1923e2ede1d5e1c7c0ae73f71aaf7955b9c01000` |
| 上游 | CANNBot skills → 本工作区 `.dsh/skills/ops-precision-standard/` |
| 上游 commit | cannbot `1f9b84fc490353ddcfeccb2f0a7130a513751d17`；skills `81d2bdbb1e57f9b737a205c3edfda50c7d0b0a31`（`skills_dirty: true`） |
| 指纹台账 | `ops/dsh/skills.lock`（`ops-precision-standard/scripts/mixed_tolerance_check.py` 同一 sha256） |

两份副本都是 **`cp` 出来的逐字节拷贝**，哈希与 `ops/dsh/skills.lock` 里的记录一致
（复核命令：`shasum -a 256 criteria/ops-precision-standard@1/*.py`）。

## 为什么要有副本

`.dsh/` 是 agent 能力目录（含 .cannbot 依赖仓几百 MB），`nm_test.sh` 在同步时**排除**它 ——
容器里只有工作区源码。而 `test.sh` 必须在容器里完成机判（不能把输出搬回 Mac 再判，
那样报告的「实际执行」就变成了本机脚本）。所以判据实现必须是工作区里的一份可见文件。

## float32 档值（本档用于 v1 规格）

| 项 | 值 | 出处 |
|---|---|---|
| rtol | 2⁻¹⁰ ≈ 9.77e-4 | `float_compute.md` §3 混合容差阈值表 |
| atol | 2⁻¹⁶ ≈ 1.53e-5 | 同上 |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | max(1e-2, 32×ULP)，ULP(fp32@1.0) = 2⁻²³ | 同上（固定上限 与 32×ULP 取宽松者） |

通过条件（两条同时满足）：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit`。
