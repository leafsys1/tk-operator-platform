# 任务：服务器版投标复验 20261101 向量减法

> 由 op-agent「任务开始」页生成 · 2026-11-01 09:41 UTC · spec v1（sha256 1055c7fca56b）· 判据档 ops-precision-standard@1

## 目标

- 算子：bid_elementwise_sub_20261101
- 目标芯片：Ascend 910B1

## 需求（用户原话）

开发算子 bid_elementwise_sub_20261101，任务名“服务器版投标复验 20261101 向量减法”。自然语言：两个同形状一维连续 float32 NPU 张量逐元素相减，返回同形状同类型同设备连续张量。不广播，长度 N=1..1048576。Python 接口 subtract(x1,x2)。非零 storage_offset 的连续视图也按逻辑元素正确计算。CPU 输入、CPU/NPU 混用、跨 NPU、非一维、非连续输入拒绝；空或超长、长度不一致、float16/int32 拒绝，明确异常类型和拒绝码。输出需核对形状、dtype、设备及数值。参考为 torch CPU 的 torch.sub，无候选 CPU/torch 代算回退。目标 Ascend 910B1 / dav-2201，执行机当前服务器，镜像 lfx-op-dev-910b:torch210-cann910，物理卡2，独占且不自动换卡，不干扰其他任务。判据 ops-precision-standard@1：rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_err<=0.01（有限值）；NaN 按位置类别、Inf 按符号，NaN payload 和零符号位不作逐位要求。基准 N=65536 和1048576，torch.sub 同卡同条件对比，warmup2、迭代5、独立重复3，设备侧 Task Duration 中位数微秒。性能优化以实测为准，不预承诺加速比。交付算子实现、共享库及 Python 接口、测试脚本、可独立运行最小示例、README、设计文档及正确性/性能报告。当前阶段只澄清需求，先不生成实现或占卡。
确认：数学公式 y[i]=x1[i]-x2[i]。参考代码：def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())。输入标准连续 ND，非零 storage_offset 的 NPU 切片按 data_ptr 指向的逻辑元素正确计算，输出新连续张量。拒绝约定：非张量、CPU 输入、CPU/NPU 混用：TypeError + REJECT_BAD_ARGS；跨 NPU、非一维、非连续：ValueError + REJECT_BAD_ARGS；长度不一致：ValueError + REJECT_SHAPE_MISMATCH；空/超长：ValueError + REJECT_N_OUT_OF_RANGE；非 float32（含 float16/int32）：ValueError + REJECT_UNSUPPORTED_DTYPE；底层执行失败：RuntimeError，不吞错。CLI 如适用对应退出码 BAD_ARGS=2、SHAPE=3、N=4、DTYPE=5、执行失败1。跨卡拒绝是接口约束，本次只用物理卡2，实际双卡用例未经额外独占条件不执行，并如实标记未验证，不能写已测通过。基础开发选择清晰常规实现，后续允许基于实测调优，不预造加速结果。上述信息完整，可以把需求置为可定版；本阶段不执行开发、不占卡的限制不要误写成最终功能限制。

## 语义与接口

- 数学定义：y[i] = x1[i] - x2[i]；参考实现 def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())；不广播。
- 参数 / 属性：长度 N 取值范围 1..1048576；无广播参数。
- 输入：['x1: shape=[N], dtype=float32，NPU 标准连续 ND 张量；支持非零 storage_offset 的切片，按 data_ptr 指向的逻辑元素计算', 'x2: shape=[N], dtype=float32，NPU 标准连续 ND 张量；支持非零 storage_offset 的切片，按 data_ptr 指向的逻辑元素计算']
- 输出：y: shape=[N], dtype=float32，同设备新连续 ND 张量。
- 支持范围：一维连续 ND、float32、同设备 NPU 张量；N∈[1,1048576]；不广播；非零 storage_offset 连续切片按逻辑元素正确。
- 不支持：非张量/CPU 输入/CPU-NPU 混用 → TypeError+REJECT_BAD_ARGS；跨 NPU/非一维/非连续 → ValueError+REJECT_BAD_ARGS；长度不一致 → ValueError+REJECT_SHAPE_MISMATCH；空(N=0)/超长(N>1048576) → ValueError+REJECT_N_OUT_OF_RANGE；非 float32（float16/int32 等）→ ValueError+REJECT_UNSUPPORTED_DTYPE；底层执行失败 → RuntimeError（不吞错）。
- 特殊行为：空输入拒绝；NaN 按位置类别核对、Inf 按符号核对；NaN payload 与零符号位不作逐位要求。

## 正确性

- 参考实现：torch.sub(x1.cpu(), x2.cpu())（def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())）；无候选 CPU/torch 代算回退。
- 容差：ops-precision-standard@1：rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_err<=0.01（有限值）。
- 一致性：输出 shape/dtype/设备/数值与参考一致；NaN 按位置类别、Inf 按符号逐元素核对，NaN payload 与零符号位不作逐位要求。

## 性能

- 基线：torch.sub 同卡同条件对比；N=65536 与 N=1048576；warmup=2、迭代=5、独立重复=3；设备侧 Task Duration 中位数（微秒）。
- 目标：以实测为准，不预造（不预承诺）加速比；基础实现清晰常规，后续基于实测调优。

## 下一步

按本规格落实现（build / test / benchmark / profile）。装载清单见 `context.yaml`。
