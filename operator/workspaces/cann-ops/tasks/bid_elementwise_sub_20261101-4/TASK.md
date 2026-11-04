# 任务：统一入口真机流程验证 20261103

> 由 op-agent「任务开始」页生成 · 2026-11-03 12:26 UTC · spec v1（sha256 179c44a208a5）· 判据档 ops-precision-standard@1

## 目标

- 算子：bid_elementwise_sub_20261101
- 目标芯片：910B1

## 需求（用户原话）

我要验证向量减法（elementwise sub）算子在 910B 上的完整流程。输入两个 float32 张量 a、b，输出 a-b，允许形状广播为同 shape；精度判据按 ops-precision-standard@1。本轮只做需求澄清与规格确认，请不要修改任何文件、不要编译、不要执行任何命令或真机动作。
1) op_name 用 bid_elementwise_sub_20261101（单段名，沿用工作区已有算子目录与源码，不加库前缀）。
2) 输出 dtype 恒为 float32；广播按完整 numpy 广播规则（含 0 维标量）；布局只要求连续内存输入，暂不支持非连续/原地写回。
3) 空张量返回空结果不报错；NaN/Inf 逐元素精确透传；芯片是 910B1，用 dav-2201 硬件画像；不支持的组合返回明确拒绝码。
1) 广播最大秩限 8 维；广播后元素总数不设额外上限，按显存/内存自然约束；超限统一返回拒绝码。
2) 拒绝码就用这三类固定值：非法布局 EPARAM_LAYOUT、秩超限 EPARAM_RANK、dtype 不支持 EPARAM_DTYPE；错误信息统一格式 "ERR <code>: <reason>"。正确性口径采用项目默认：参考实现 torch CPU 的 a-b 一一对应，输出 shape/dtype 完全一致、NaN/Inf 逐元素精确比对。
3) 本轮性能基线与性能目标不作要求，先只要求 build/smoke/正确性通过；交付物包含算子源码、测试与文档；目标环境 CANN 9.2.0（与执行镜像一致）。本轮只做规格确认，仍然不要修改文件、不要编译、不要执行任何命令。

## 语义与接口

- 数学定义：逐元素减法 out = a - b，按完整 numpy 广播规则对齐后逐元素计算，输出与广播结果同 shape
- 参数 / 属性：{}
- 输入：[{'name': 'a', 'dtype': 'float32', 'layout': '连续内存'}, {'name': 'b', 'dtype': 'float32', 'layout': '连续内存'}]
- 输出：[{'name': 'out', 'dtype': 'float32', 'shape': 'a、b 按完整 numpy 广播规则得到的 shape（含 0 维标量参与）'}]
- 支持范围：['目标芯片 910B1，硬件画像 ops/chips/dav-2201.yaml', '输入输出 dtype float32', '完整 numpy 广播（含 0 维标量）', '最大秩 8 维', '广播后元素总数不设额外上限，按显存自然约束', '空张量（0 元素）返回空结果']
- 不支持：['非连续内存输入（拒绝码 EPARAM_LAYOUT）', '秩超过 8 维（拒绝码 EPARAM_RANK）', '非 float32 dtype（拒绝码 EPARAM_DTYPE）', '原地写回']
- 特殊行为：空张量返回空结果且不报错；NaN/Inf 逐元素精确透传；不支持的组合返回上述固定拒绝码，错误信息格式统一为 "ERR <code>: <reason>"

## 正确性

- 参考实现：与 torch CPU 实现一一对应（torch 的 a-b）
- 容差：ops-precision-standard@1
- 一致性：输出 shape/dtype 完全一致 + NaN/Inf 逐元素精确比对

## 性能

- 基线：本轮不作要求
- 目标：本轮不作要求，验收以 build/smoke/正确性通过为准

## 下一步

按本规格落实现（build / test / benchmark / profile）。装载清单见 `context.yaml`。
