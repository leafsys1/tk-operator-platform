# elementwise_add 规划文档（任务生命周期 v2 交付件）

## 1. 需求背景
逐元素加法 y=a+b；来源：op-agent 首个贯通样例（task #2，历史已归档）；目标芯片 910B1（dav-2201）。
数学语义：float32、同 shape、不广播，NaN/Inf 按 IEEE 语义逐元素传播。

## 2. 需求分析
依赖 CANN 9.2 / Ascend C 直调工程；输入 a、b 各 N 个 float32（N∈[1,1048576]），输出 y 同 shape。
原型：CLI `elementwise_add --n-a N --n-b M --dtype float32 --in-a a.bin --in-b b.bin --out c.bin`；
拒绝码：N 越界 4、shape 不一致 3、dtype 不支持 5、参数错误 2。

## 3. 需求详细设计
Host：shape/dtype 校验 → 按 N 自适应 tile（≤64k 用 512，否则 4096）→ 按可用向量核数分核；
Kernel：GM→UB CopyIn、Add、UB→GM CopyOut，尾块用 DataCopyPad；精度与 torch CPU 对齐（fp32 一次舍入）。

## 4. 特性交叉分析
混合精度：规格 v1 仅 float32，不涉及；图模式：直调二进制，不涉及；动态 shape：运行时可变的 N 由 host tiling 处理。

## 5. 可维可测分析
精度标准 ops-precision-standard@1（rtol=2^-10、atol=2^-16，非有限位精确比对）；性能口径为设备侧
Task Duration、同卡同条件、median；兼容性：接口/CLI 与既有契约一致，dtype 仅 float32，目标硬件 910B 系列。
