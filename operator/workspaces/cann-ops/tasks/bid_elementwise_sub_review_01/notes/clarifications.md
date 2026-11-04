# 需求澄清问答记录

> 导出 2026-11-02 03:57 UTC（真相源：任务 `chat_log`）

**用户**：任务名称为向量减法检测，算子名为 bid_elementwise_sub_review_01，请按以下预置条件解析需求，先澄清、不开发、不运行。
输入 x1、x2 为同一 NPU 上的一维连续 ND float32 向量，长度相同，1≤N≤1048576，支持连续且 storage_offset 非零的视图；y[i]=x1[i]-x2[i]，不广播、不带 alpha、不原位修改，输出新分配、同设备同形状 float32。数值参考为 CPU torch.sub。Python 接口 subtract(x1,x2)：非 Tensor、CPU 输入或 CPU/NPU 混用为 TypeError 及 REJECT_BAD_ARGS；非一维、非连续或跨 NPU 为 ValueError 及 REJECT_BAD_ARGS；长度不同为 ValueError 及 REJECT_SHAPE_MISMATCH；空或超长为 ValueError 及 REJECT_N_OUT_OF_RANGE；非 float32 为 ValueError 及 REJECT_UNSUPPORTED_DTYPE；底层执行失败为 RuntimeError。多重错误以实际入口检查顺序核对；跨卡验证另需两张可独占卡。目标芯片 Ascend 910B1，判据 ops-precision-standard@1；要求实现、调用接口、测试脚本、独立示例、说明及测试报告。

**助手**：已收到，本任务为「向量减法检测」，按你的要求先澄清、不开发、不运行。以下已确定信息我直接采入草稿：

- 算子名：bid_elementwise_sub_review_01；
- 语义：同长度一维向量逐元素相减 y[i]=x1[i]-x2[i]，无广播、无 alpha、不原位修改，输出为新分配、同设备同形状 float32；
- 输入：同一 NPU 上的一维连续 float32 向量，长度相同且 1≤N≤1048576，支持连续且 storage_offset 非零的视图；
- 参考实现 CPU torch.sub，容差档 ops-precision-standard@1，目标芯片 Ascend 910B1；
- 拒绝路径已记录：非 Tensor/CPU 输入/CPU 与 NPU 混用 → TypeError+REJECT_BAD_ARGS；非一维/非连续/跨 NPU → ValueError+REJECT_BAD_ARGS；长度不同 → ValueError+REJECT_SHAPE_MISMATCH；空或超长 → ValueError+REJECT_N_OUT_OF_RANGE；非 float32 → ValueError+REJECT_UNSUPPORTED_DTYPE；底层执行失败 → RuntimeError；多重错误按实际入口检查顺序核对；
- 交付物：实现、Python 调用接口、pytest 测试脚本、独立示例、说明文档、测试报告。

还需确认三点：

1）NaN/Inf 的比对口径：是否沿用项目默认——输出与 CPU torch.sub 对 NaN/Inf 逐元素精确比对（NaN 按等价判断、Inf 按 IEEE 754 减法规格）？还是只要求功能正确、不做逐元素精确比对？

2）性能口径：任务名含「检测」，是否有性能基线或目标（如 N=1048576 时设备侧时延/吞吐基准）？若无显式指标，测试报告是否仍须包含性能（benchmark）章节？

3）「跨卡验证另需两张可独占卡」：是否指为验证「跨 NPU 拒绝」路径，需用两张可独占卡构造跨设备输入？另外运行环境（CANN 版本/镜像）是沿用平台默认，还是由你指定？

**用户**：补充确认三点：
1）NaN/Inf 比对口径：沿用项目默认并写入规格——NaN 按位置和类别核对，Inf 按符号核对；NaN payload 与零符号位不作逐位要求；有限值按 ops-precision-standard@1（rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_err≤0.01）。
2）性能口径：不设固定加速比硬线；测试报告必须包含 benchmark 章节，主用例 N=65536、另一用例 N=1048576，基线为同设备 torch.sub，warm_up=2、invocations=5、retries_per_invocation=3，取设备 Task Duration 中位数（μs）；结论按实测如实记录。
3）跨卡验证：跨 NPU 拒绝路径需要两张可独占卡构造输入；本次若只有一张卡，则该项记为未验证，不冒充通过。运行环境沿用平台默认镜像与 CANN。
另补数学式：对 0≤i<N，y[i]=x1[i]-x2[i]。

**助手**：已收到你补充的三点确认与数学式，全部采入草稿：

- 数学式：对 0≤i<N，y[i]=x1[i]-x2[i]；
- 特殊值/一致性：NaN 按位置和类别核对，Inf 按符号核对；NaN payload 与零符号位不作逐位要求；有限值按 ops-precision-standard@1（rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_err≤0.01）；
- 性能：测试报告必须含 benchmark 章节；主用例 N=65536、另一用例 N=1048576，基线为同设备 torch.sub，warm_up=2、invocations=5、retries_per_invocation=3，取设备 Task Duration 中位数（μs）；不设固定加速比硬线，结论按实测如实记录；
- 跨卡验证：跨 NPU 拒绝路径需两张可独占卡构造输入；若本次仅一张卡，该项记为未验证、不冒通过；
- 运行环境：沿用平台默认镜像与 CANN。

至此六个关键项（算子名、语义、输入输出、正确性口径、支持/不支持范围、空输入及 NaN/Inf 特殊行为）均已确认，规格草稿已可执行，本轮无需你再确认其他事项。

**用户**：补充参考代码：
def ref(x1, x2):
    return torch.sub(x1.cpu(), x2.cpu())
说明：参考代码仅作数值对比（CPU oracle），候选实现必须在目标 NPU 上执行；输出设备与输入一致，不允许 CPU/torch 代算回退。

**助手**：已记录：参考实现为 CPU oracle `torch.sub(x1.cpu(), x2.cpu())`，仅作数值对比；候选实现必须在目标 NPU 上执行，输出设备与输入一致，不允许 CPU/torch 代算回退。该点与现有「输出同设备」一致，无新增待确认项。
