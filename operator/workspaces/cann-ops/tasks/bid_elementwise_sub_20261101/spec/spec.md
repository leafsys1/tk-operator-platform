# 算子需求规格 · bid_elementwise_sub_20261101

> spec v1 · sha256 1055c7fca56b · 2026-11-01 09:41 UTC · 判据档 ops-precision-standard@1
> 本文件由 `spec.json` 渲染，禁手改。

| 字段 | 值 |
|---|---|
| op_name | bid_elementwise_sub_20261101 |
| math_def | y[i] = x1[i] - x2[i]；参考实现 def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())；不广播。 |
| params | 长度 N 取值范围 1..1048576；无广播参数。 |
| inputs | ['x1: shape=[N], dtype=float32，NPU 标准连续 ND 张量；支持非零 storage_offset 的切片，按 data_ptr 指向的逻辑元素计算', 'x2: shape=[N], dtype=float32，NPU 标准连续 ND 张量；支持非零 storage_offset 的切片，按 data_ptr 指向的逻辑元素计算'] |
| outputs | y: shape=[N], dtype=float32，同设备新连续 ND 张量。 |
| layout | 标准连续 ND；非零 storage_offset 的 NPU 切片按 data_ptr 指向的逻辑元素正确计算；输出为新连续张量。 |
| support | 一维连续 ND、float32、同设备 NPU 张量；N∈[1,1048576]；不广播；非零 storage_offset 连续切片按逻辑元素正确。 |
| unsupported | 非张量/CPU 输入/CPU-NPU 混用 → TypeError+REJECT_BAD_ARGS；跨 NPU/非一维/非连续 → ValueError+REJECT_BAD_ARGS；长度不一致 → ValueError+REJECT_SHAPE_MISMATCH；空(N=0)/超长(N>1048576) → ValueError+REJECT_N_OUT_OF_RANGE；非 float32（float16/int32 等）→ ValueError+REJECT_UNSUPPORTED_DTYPE；底层执行失败 → RuntimeError（不吞错）。 |
| special_behavior | 空输入拒绝；NaN 按位置类别核对、Inf 按符号核对；NaN payload 与零符号位不作逐位要求。 |
| cli_exit_codes | BAD_ARGS=2、SHAPE=3、N=4、DTYPE=5、执行失败=1（CLI 如适用）。 |
| target_chip | Ascend 910B1 |
| target_hardware_profile | ops/chips/dav-2201.yaml |
| target_env | 当前服务器执行机；镜像 lfx-op-dev-910b:torch210-cann910；物理卡 2，独占且不自动换卡，不干扰其他任务。 |
| oracle | torch.sub(x1.cpu(), x2.cpu())（def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())）；无候选 CPU/torch 代算回退。 |
| tolerance | ops-precision-standard@1：rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_err<=0.01（有限值）。 |
| consistency | 输出 shape/dtype/设备/数值与参考一致；NaN 按位置类别、Inf 按符号逐元素核对，NaN payload 与零符号位不作逐位要求。 |
| performance_baseline | torch.sub 同卡同条件对比；N=65536 与 N=1048576；warmup=2、迭代=5、独立重复=3；设备侧 Task Duration 中位数（微秒）。 |
| performance_target | 以实测为准，不预造（不预承诺）加速比；基础实现清晰常规，后续基于实测调优。 |
| delivery | 算子实现(kernel/tiling/host/CMake)；共享库及 Python 接口 subtract(x1,x2)；测试脚本；可独立运行最小示例；README；设计文档；正确性/性能报告。 |
| assumptions | 本阶段仅澄清需求、不执行开发、不占卡（本阶段工作方式，非算子最终功能限制）；跨卡拒绝为接口约束，本次仅用物理卡 2，双卡用例未经额外独占条件不执行并如实标记未验证、不写已测通过。 |
