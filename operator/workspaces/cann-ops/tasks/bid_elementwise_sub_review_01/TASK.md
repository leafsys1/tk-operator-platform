# 任务：向量减法检测

> 由 op-agent「任务开始」页生成 · 2026-11-02 03:57 UTC · spec v1（sha256 52333ec51f49）· 判据档 ops-precision-standard@1

## 目标

- 算子：bid_elementwise_sub_review_01
- 目标芯片：Ascend 910B1

## 需求（用户原话）

任务名称为向量减法检测，算子名为 bid_elementwise_sub_review_01，请按以下预置条件解析需求，先澄清、不开发、不运行。
输入 x1、x2 为同一 NPU 上的一维连续 ND float32 向量，长度相同，1≤N≤1048576，支持连续且 storage_offset 非零的视图；y[i]=x1[i]-x2[i]，不广播、不带 alpha、不原位修改，输出新分配、同设备同形状 float32。数值参考为 CPU torch.sub。Python 接口 subtract(x1,x2)：非 Tensor、CPU 输入或 CPU/NPU 混用为 TypeError 及 REJECT_BAD_ARGS；非一维、非连续或跨 NPU 为 ValueError 及 REJECT_BAD_ARGS；长度不同为 ValueError 及 REJECT_SHAPE_MISMATCH；空或超长为 ValueError 及 REJECT_N_OUT_OF_RANGE；非 float32 为 ValueError 及 REJECT_UNSUPPORTED_DTYPE；底层执行失败为 RuntimeError。多重错误以实际入口检查顺序核对；跨卡验证另需两张可独占卡。目标芯片 Ascend 910B1，判据 ops-precision-standard@1；要求实现、调用接口、测试脚本、独立示例、说明及测试报告。
补充确认三点：
1）NaN/Inf 比对口径：沿用项目默认并写入规格——NaN 按位置和类别核对，Inf 按符号核对；NaN payload 与零符号位不作逐位要求；有限值按 ops-precision-standard@1（rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_err≤0.01）。
2）性能口径：不设固定加速比硬线；测试报告必须包含 benchmark 章节，主用例 N=65536、另一用例 N=1048576，基线为同设备 torch.sub，warm_up=2、invocations=5、retries_per_invocation=3，取设备 Task Duration 中位数（μs）；结论按实测如实记录。
3）跨卡验证：跨 NPU 拒绝路径需要两张可独占卡构造输入；本次若只有一张卡，则该项记为未验证，不冒充通过。运行环境沿用平台默认镜像与 CANN。
另补数学式：对 0≤i<N，y[i]=x1[i]-x2[i]。
补充参考代码：
def ref(x1, x2):
    return torch.sub(x1.cpu(), x2.cpu())
说明：参考代码仅作数值对比（CPU oracle），候选实现必须在目标 NPU 上执行；输出设备与输入一致，不允许 CPU/torch 代算回退。

## 语义与接口

- 数学定义：对 0≤i<N，y[i]=x1[i]-x2[i]；无广播、无 alpha、不原位修改，输出为新分配张量
- 参数 / 属性：无额外参数（subtract(x1, x2)）
- 输入：[{'name': 'x1', 'dtype': 'float32', 'shape': '一维 [N]', 'layout': '连续(contiguous)', 'device': 'NPU', 'constraints': '1≤N≤1048576；支持连续且 storage_offset 非零的视图'}, {'name': 'x2', 'dtype': 'float32', 'shape': '一维 [N]，与 x1 同长度', 'layout': '连续(contiguous)', 'device': '与 x1 同一 NPU', 'constraints': '同上'}]
- 输出：[{'name': 'y', 'dtype': 'float32', 'shape': '与输入同形状（一维 [N]）', 'layout': '连续(contiguous)', 'device': '同设备（与输入一致）', 'note': '新分配'}]
- 支持范围：同一 NPU 上、同长度一维连续 float32 向量；1≤N≤1048576；支持连续且 storage_offset≠0 的视图
- 不支持：广播、alpha、原位修改；非一维、非连续、跨 NPU、长度不同、空或超长（N 越界）、非 float32
- 特殊行为：{'empty_input': 'ValueError + REJECT_N_OUT_OF_RANGE', 'nan_inf': 'NaN 按位置和类别核对、Inf 按符号核对；NaN payload 与零符号位不作逐位要求；有限值按 ops-precision-standard@1'}

## 正确性

- 参考实现：CPU torch.sub（torch.sub(x1.cpu(), x2.cpu())），仅作数值对比
- 容差：ops-precision-standard@1（有限值：rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_err≤0.01）
- 一致性：输出 shape/dtype 完全一致；NaN 按位置和类别核对、Inf 按符号核对；NaN payload 与零符号位不作逐位要求

## 性能

- 基线：同设备 torch.sub
- 目标：不设固定加速比硬线；结论按实测如实记录

## 下一步

按本规格落实现（build / test / benchmark / profile）。装载清单见 `context.yaml`。
