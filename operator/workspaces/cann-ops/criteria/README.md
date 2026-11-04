# 判据档（criteria）

门禁的 `criteria_ref`（档名`@`档版本）在这里有**实体**：一个目录一份可执行实现。

```
criteria/
  README.md                      ← 本文件：契约与版本绑定规则
  ops-precision-standard@1/      ← 档名@版本 = 目录名
    mixed_tolerance_check.py     ← 官方档实现的**逐字节副本**（sha256 见 PROVENANCE.md）
    float_compute.md             ← 官方档正文（判据表出处）
    PROVENANCE.md                ← 出处、指纹、为什么必须有副本
  check.py                       ← 机判入口（官方实现 + 非有限值口径），test.sh 用它
```

## 规则

1. **判据档实现必须能在容器里被调用**。`.dsh/` 不进容器（`nm_test.sh` 的 rsync 排除，几十~几百 MB），
   而 `test.sh` 在容器里做机判 —— 所以实现必须是工作区里可见的文件，不能只存在于 skills 目录。
2. **改判据 = 换版本号**：`ops-precision-standard@1` 的实现一旦落地就冻结；要改口径就出 `@2`
   （新目录），并在 `tasks.criteria_ref` / `projects.criteria_ref` 上换代。
   否则历史 run 的 `criteria_ref` 会指向与当时不同的口径，门禁的「版本一致」就失去意义。
3. **报告必须回写档名@版本**（`correctness.json.criteria_ref`），与本次 run 的 `criteria_ref` 逐字相等；
   门禁的 `E_REPORT_VERSION_MISMATCH` 就是守这条。
4. 官方档只规定**有限浮点指标**（atol/rtol 通过率 + max_abs_error 上限）。
   NaN/Inf 按 IEEE 语义逐元素精确比对的口径写在 `check.py` 顶部，不被当成「容差通过」。
