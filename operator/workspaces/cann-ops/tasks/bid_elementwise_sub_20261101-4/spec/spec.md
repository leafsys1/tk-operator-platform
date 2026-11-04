# 算子需求规格 · bid_elementwise_sub_20261101

> spec v1 · sha256 179c44a208a5 · 2026-11-03 12:26 UTC · 判据档 ops-precision-standard@1
> 本文件由 `spec.json` 渲染，禁手改。

| 字段 | 值 |
|---|---|
| op_name | bid_elementwise_sub_20261101 |
| math_def | 逐元素减法 out = a - b，按完整 numpy 广播规则对齐后逐元素计算，输出与广播结果同 shape |
| params | {} |
| inputs | [{'name': 'a', 'dtype': 'float32', 'layout': '连续内存'}, {'name': 'b', 'dtype': 'float32', 'layout': '连续内存'}] |
| outputs | [{'name': 'out', 'dtype': 'float32', 'shape': 'a、b 按完整 numpy 广播规则得到的 shape（含 0 维标量参与）'}] |
| layout | 仅连续内存输入；不支持非连续 stride、不支持原地写回 |
| support | ['目标芯片 910B1，硬件画像 ops/chips/dav-2201.yaml', '输入输出 dtype float32', '完整 numpy 广播（含 0 维标量）', '最大秩 8 维', '广播后元素总数不设额外上限，按显存自然约束', '空张量（0 元素）返回空结果'] |
| unsupported | ['非连续内存输入（拒绝码 EPARAM_LAYOUT）', '秩超过 8 维（拒绝码 EPARAM_RANK）', '非 float32 dtype（拒绝码 EPARAM_DTYPE）', '原地写回'] |
| special_behavior | 空张量返回空结果且不报错；NaN/Inf 逐元素精确透传；不支持的组合返回上述固定拒绝码，错误信息格式统一为 "ERR <code>: <reason>" |
| oracle | 与 torch CPU 实现一一对应（torch 的 a-b） |
| tolerance | ops-precision-standard@1 |
| consistency | 输出 shape/dtype 完全一致 + NaN/Inf 逐元素精确比对 |
| performance_baseline | 本轮不作要求 |
| performance_target | 本轮不作要求，验收以 build/smoke/正确性通过为准 |
| delivery | 算子源码、测试、文档 |
| target_env | CANN 9.2.0（与执行镜像一致） |
| target_chip | 910B1 |
| target_hardware_profile | ops/chips/dav-2201.yaml |
| assumptions | ['8 维秩上限按输入与广播结果均不超过 8 维执行'] |
| unknowns | [] |
