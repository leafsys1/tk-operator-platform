# 需求澄清问答记录

> 导出 2026-11-01 09:41 UTC（真相源：任务 `chat_log`）

**用户**：开发算子 bid_elementwise_sub_20261101，任务名“服务器版投标复验 20261101 向量减法”。自然语言：两个同形状一维连续 float32 NPU 张量逐元素相减，返回同形状同类型同设备连续张量。不广播，长度 N=1..1048576。Python 接口 subtract(x1,x2)。非零 storage_offset 的连续视图也按逻辑元素正确计算。CPU 输入、CPU/NPU 混用、跨 NPU、非一维、非连续输入拒绝；空或超长、长度不一致、float16/int32 拒绝，明确异常类型和拒绝码。输出需核对形状、dtype、设备及数值。参考为 torch CPU 的 torch.sub，无候选 CPU/torch 代算回退。目标 Ascend 910B1 / dav-2201，执行机当前服务器，镜像 lfx-op-dev-910b:torch210-cann910，物理卡2，独占且不自动换卡，不干扰其他任务。判据 ops-precision-standard@1：rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_err<=0.01（有限值）；NaN 按位置类别、Inf 按符号，NaN payload 和零符号位不作逐位要求。基准 N=65536 和1048576，torch.sub 同卡同条件对比，warmup2、迭代5、独立重复3，设备侧 Task Duration 中位数微秒。性能优化以实测为准，不预承诺加速比。交付算子实现、共享库及 Python 接口、测试脚本、可独立运行最小示例、README、设计文档及正确性/性能报告。当前阶段只澄清需求，先不生成实现或占卡。

**助手**：已收到。本轮已记入草稿的内容：算子名 `bid_elementwise_sub_20261101`；语义为两个同形状一维连续 float32 NPU 张量逐元素相减、返回同形状同类型同设备连续张量、不广播；长度 N 范围 1..1048576；支持非零 storage_offset 的连续视图按逻辑元素计算；Python 接口 `subtract(x1,x2)`。目标 Ascend 910B1 / dav-2201，镜像 lfx-op-dev-910b:torch210-cann910，物理卡 2 独占且不自动换卡、不干扰其他任务。参考实现取 torch CPU 的 `torch.sub`、无代算回退；容差采用 ops-precision-standard@1；NaN 按位置类别、Inf 按符号核对，NaN payload 与零符号位不作逐位要求；性能基准 N=65536 与 1048576、同卡同条件对比、以实测为准不预承诺加速比。各类拒绝场景、交付物清单以及“本阶段只澄清需求、不生成实现不占卡”均已记录。

还需要向你确认三点：

1. 减法方向：`subtract(x1,x2)` 的结果是 `x1 - x2`（即 `torch.sub(x1,x2)`，元素级 `x1[i] - x2[i]`），对吗？
2. 异常类型与拒绝码：CPU 输入、CPU/NPU 混用、跨 NPU、非一维、非连续、空输入、超长、长度不一致、float16/int32 这些拒绝场景，各自的异常类型（如 TypeError / ValueError / NotImplementedError / RuntimeError）与拒绝码有没有既定约定？还是由开发侧自定，并在设计文档、测试用例与报告中声明留痕？
3. 存储布局：NPU 侧的 format 是否就采用标准连续 ND？非零 storage_offset 的连续视图按什么方式传入（例如带 offset 的 ND 引用）？如无特殊要求，我按“标准 ND、offset 视图按逻辑元素计算”理解。

**用户**：确认：数学公式 y[i]=x1[i]-x2[i]。参考代码：def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())。输入标准连续 ND，非零 storage_offset 的 NPU 切片按 data_ptr 指向的逻辑元素正确计算，输出新连续张量。拒绝约定：非张量、CPU 输入、CPU/NPU 混用：TypeError + REJECT_BAD_ARGS；跨 NPU、非一维、非连续：ValueError + REJECT_BAD_ARGS；长度不一致：ValueError + REJECT_SHAPE_MISMATCH；空/超长：ValueError + REJECT_N_OUT_OF_RANGE；非 float32（含 float16/int32）：ValueError + REJECT_UNSUPPORTED_DTYPE；底层执行失败：RuntimeError，不吞错。CLI 如适用对应退出码 BAD_ARGS=2、SHAPE=3、N=4、DTYPE=5、执行失败1。跨卡拒绝是接口约束，本次只用物理卡2，实际双卡用例未经额外独占条件不执行，并如实标记未验证，不能写已测通过。基础开发选择清晰常规实现，后续允许基于实测调优，不预造加速结果。上述信息完整，可以把需求置为可定版；本阶段不执行开发、不占卡的限制不要误写成最终功能限制。

**助手**：已确认并记入草稿：减法方向为 `y[i] = x1[i] - x2[i]`，参考实现即 `def ref(x1, x2): return torch.sub(x1.cpu(), x2.cpu())`。输入布局为标准连续 ND，非零 storage_offset 的 NPU 切片按 data_ptr 指向的逻辑元素正确计算，输出为新连续张量。拒绝协议也全部落地：

- 非张量 / CPU 输入 / CPU-NPU 混用 → TypeError + REJECT_BAD_ARGS；
- 跨 NPU / 非一维 / 非连续 → ValueError + REJECT_BAD_ARGS；
- 长度不一致 → ValueError + REJECT_SHAPE_MISMATCH；
- 空 (N=0) / 超长 (N>1048576) → ValueError + REJECT_N_OUT_OF_RANGE；
- 非 float32（含 float16/int32）→ ValueError + REJECT_UNSUPPORTED_DTYPE；
- 底层执行失败 → RuntimeError，不吞错；CLI 如适用：BAD_ARGS=2、SHAPE=3、N=4、DTYPE=5、执行失败=1。

并已记录：跨卡拒绝属接口约束，本次仅用物理卡 2，双卡用例未经额外独占条件不执行、如实标记未验证、不写“已测通过”；基础开发采用清晰常规实现、后续按实测调优、不预造加速结果；“本阶段不开发、不占卡”是本阶段工作方式，不写进算子最终功能限制。

至此关键项（算子名、数学语义、输入输出与布局、正确性口径、支持与拒绝范围、特殊行为）均已明确，需求可定版，本轮无需再提问。
