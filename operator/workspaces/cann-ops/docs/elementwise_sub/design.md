# elementwise_sub 规划文档（任务生命周期 v2 交付件）

- 任务：#2 · 规格：`tasks/elementwise_sub/spec/spec.json`（spec_version_id 2，v1，sha256 `d20bca80…`）
- 用例集：`cases/elementwise_sub/{cases.json,benchmark.json}`（`cases@9d2f6e10cf34a828`，2026-10-23 确认）
- 目标芯片：Ascend910B1（`dav-2201`，硬件画像 `ops/chips/dav-2201.yaml`）

## 1. 需求背景

逐元素减法算子 `elementwise_sub`：`y = a - b`；`alpha` 非 1 时按 torch 语义 `y = a - alpha*b`。
承接 `elementwise_add` 的首个贯通样例，把「调用接口参数化 + 拒绝码 + 五件交付物」的工程框架复用到
一个**语义上有真实分支**的算子上（alpha、out 复用、特殊值符号位）。行为与报错对齐 torch CPU `torch.sub`。

## 2. 需求分析

| 项 | 内容 |
|---|---|
| 数学语义 | `y = a - alpha*b`，逐元素、无跨元素依赖；IEEE 754 fp32 减法结果唯一 |
| 输入 | `a`、`b`：float32、连续、形状相同（**不广播**）；`N ∈ [1, 1048576]` |
| 参数 | 对齐 `torch.sub(input, other, alpha=1, out=None)`；alpha 任意 float；`out=` 复用 |
| 输出 | `y`：float32、同 shape、连续；`out=` 时写入调用方缓冲（含原地 `out=a`） |
| 拒绝 | N=0、N=1048577、形状不一致（`REJECT_SHAPE_MISMATCH`）、dtype float16/int32（`REJECT_UNSUPPORTED_DTYPE`）、缺参 / 未知参数（`REJECT_BAD_ARGS`） |
| 标杆 | CPU **fp64** 计算 `a - alpha*b` → 一次舍入到 fp32；禁止用 NPU 实现当参考 |
| 判据 | 规格要求按位精确；门禁按 `criteria/ops-precision-standard@1`（见 §4 风险项） |
| 性能基线 | 同卡 `torch.sub`（`tools/torch_sub_baseline.py`），指标 = msopprof 设备侧 Task Duration |

外部依赖与原型：沿用 Ascend C 直调工程（`ascendc-direct-invoke-template` 的 add_custom 改造），
CLI 契约与 `tools/run_cases.py` 一致（`--n-a/--n-b/--dtype/--in-a/--in-b/--out`，argv 不含 alpha）。

## 3. 需求详细设计

### 3.1 两套调用面（分工明确）

| 调用面 | 参数 | 用途 |
|---|---|---|
| 可执行文件 `ops/elementwise_sub/build/elementwise_sub` | `--n-a --n-b --dtype --in-a --in-b --out`（**alpha 固定 1**） | 四动作门禁（smoke/test/benchmark），与 `run_cases.py` 协议一致 |
| 共享库 `libelementwise_sub_host.so` + Python `sub(a, b, alpha=1, out=None)` | 张量 + alpha + out | pytest 交付件；alpha/out 正向用例只能走这里 |

(argv 为什么只能 alpha=1：`sub_rej_007` 要求「未知参数 → reject」，所以调用协议本身是接口的一部分，
不能随手加 `--alpha`。)

### 3.2 Host 侧

1. 参数解析与校验顺序（先于任何设备操作，拒绝码见上表）：未知参数/缺参 → dtype → N 范围 → 形状一致。
2. Tiling（沿用 add 的实测结论，`ascendc-tiling-design` elewise 路由）：
   - **P0 分摊式 tiling（v2 起）**，实现统一在 `op_kernel/elementwise_sub_tiling.h::BuildTiling`，可执行文件与 `.so` 共用：
     - 大 shape（N > 8192）：`used = clamp(N/512, 1, cores)`，`tile = min(4096, alignUp(ceil(N/used), 8))`，
       `blockNum = ceil(N/tile)`，除尾块外每核正好一个 tile（单核一次搬运，无跨 tile 串行）；
     - 小 shape（N ≤ 8192）：单核，`tileLength = 32`，`tileNum = ceil(N/32)`（小 shape 的定点开销在核启动上，多开核无收益）；
     - 核数：可执行文件用运行时 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)`；共享库用常量
       `DEFAULT_CORE_NUM = 48`（910B1），避免每次 Python 调用走设备查询；
   - 尾块：`DataCopyPad` 按真实字节数读写；写回时向上对齐到 32B，最多多写 7 个元素到填充槽。
     **不经标量回落路径**（非 64 元素对齐的 `DataCopyPad` 写回在真机会回落为逐元素标量格式）。
3. UB 预算：3 个队列 × DOUBLE_BUFFER(2) × MAX_TILE_LENGTH(4096) × 4B = 96KB / 192KB ≈ 50%。
   不扩到 8192 的原因：本算子搬运受限（MTE2 占比高），tile 翻倍只减固定开销，留余量给后续对齐尾块。

### 3.3 Kernel 侧

- 结构：`CopyIn(DataCopyPad) → Compute → CopyOut(DataCopyPad)` 三段式 + double buffer；每块推进 GM 偏移
  （add 的 `add_edge_006` 曾暴露「不推进偏移」缺陷，这里继承修正版）。
- 计算：`AscendC::Sub(dst, a, b, count)`。
- **alpha≠1 走单次候选 kernel（`OP_MODE_FMASUB`）**：kernel 内在 UB 上 `Muls(z, b, alpha)` 再 `Sub(z, a, z)`，
  与 torch CPU 的求值顺序（先舍入 `alpha*b` 再相减）一致；alpha 为 2 的幂时该式无二次舍入损失，与 oracle
  （fp64 + 单次舍入）逐位一致。Python 层只做一次调用，不代算（编码规则 A1）。
  |alpha*b| > FLT_MAX 时先上溢为 ±inf，规格 v2 已把该范围写进支持边界（见 §4 已知限制）。
- 特殊值：Sub 指令遵循 IEEE 754（NaN 传播、±Inf、`x - x = +0`）；±0 结果符号由 `sub_num_007` 锁住。

### 3.4 被否决的方案

| 方案 | 否决理由 |
|---|---|
| 接口层两次调用（先 `_scale` 再 sub） | v1 曾采用；后被 §3.5 的单 kernel `Muls + Sub` 取代 —— 少一次 kernel 启动、少一次中间张量，且 alpha 进 tiling 后语义更集中 |
| 设备侧 fp64 求值 | **硬件不支持**：DaVinci vector 核不允许 double，编译即报 `cast to/from double precision floating variable is not allowed in aicore function`。详见 §4 风险项 |
| 手写补偿算法（拆分高低位做两次舍入补偿） | 先在 numpy 上验证过一代实现，与 fp64 标杆严重不一致（sub_alpha_001 1/4093、sub_alpha_002 0/4096、sub_alpha_003 0/1048576），整体废弃。教训：数值算法必须先离线验证再进 kernel |
| 广播 / 多 shape 支持 | 规格明确不广播，`sub_rej_003` 要求拒绝 |
| 在包装层用 torch 计算 | 违反编码规则 A1（候选真实性） |
| 改判据档为「按位」 | `criteria/` 阈值受守护，改口径要换新档（`@2`），超出本任务范围 |

### 3.5 alpha≠1 的最终实现

kernel 内 `OP_MODE_FMASUB` 分支：`Muls(z, y, alpha)` 再 `Sub(z, x, z)`，即 fp32「先舍入 alpha*b、再相减」，
与 torch CPU 的求值顺序一致。alpha 为 2 的幂时（2.0 / 0.5 / -1.0 都是）该式无二次舍入损失，
与 fp64「一次舍入」标杆逐位等价 —— 真机实测 sub_alpha_002 / sub_alpha_003 100% 逐位。
代价是 `|alpha*b| > FLT_MAX` 时会先上溢为 Inf，故规格 v2 把该前提写入支持范围，而不是放宽判据。

## 4. 特性交叉分析

| 维度 | 结论 |
|---|---|
| 混合精度 / 量化 | 规格仅 fp32；fp16/bf16 属不覆盖分支（拒绝） |
| 动态 shape | N 运行时确定，host tiling 自适应；N=1 与 N=1M 同一 kernel |
| 图模式 | 直调二进制，不涉及 |
| 多流并发 | 每次调用独立 stream，由调用方负责上下文；接口层不隐式同步（`aclrtSynchronizeStream` 只在 kernel 后同步，返回即数据就绪） |
| 原地运算 | `out=a` 支持（`sub_out_002`）：kernel 读 a、写 y 同一 GM，按块顺序读在写前，块间不重叠 → 安全 |
| 风险：判据与规格不一致 | 规格要求按位精确，`ops-precision-standard@1` 是混合容差（rtol 2⁻¹⁰ / atol 2⁻¹⁶ / 通过率 0.99）。真机 29/29 PASS；pytest 另报 `bit_exact` 作为「接近按位」证据。注意 `check.py` 对非有限位是**逐元素精确比对**：inf 与有限值互换会直接判失败，比官方容差更严 |
| 风险：硬件无 fp64（已确认） | DaVinci vector 核禁止 double，device 侧无法复刻标杆的「双精度 + 一次舍入」。**处理方式**：规格 v2 明确 alpha≠1 的口径为 fp32 先乘后减 + 前提 `\|alpha*b\| ≤ FLT_MAX`，并论证 alpha 为 2 的幂时与标杆逐位等价。**残留**：非 2 的幂 alpha 的逐位性未穷举验证（规格 `_meta.unknowns` 登记） |
| 风险：alpha≠1 上溢点 | `a = b = FLT_MAX, α = 2` 时设备侧 `-Inf`、标杆 `-3.4028235e+38`。这是先乘后减与双精度一次舍入的固有差异，由规格的前提覆盖，用例 `sub_alpha_004` 已调整为满足前提 |
| 风险：507057 毒化下次启动 | scale 路径传 `bDev == nullptr` 却仍对 b 侧发 `DataCopyPad`，报错出现在**下一次** kernel 启动（`507057 SUSPECT REMOTE ERROR` / `[Bisheng] Launch kernel failure!`），极难定位。现由 `CopyIn` 的 `opMode != OP_MODE_SCALE` guard 守住；**重构时不可移除** |

## 5. 可维可测分析

- **正确性**：29 条用例（常规 9 / 数值边界 12 / 边界 1 / 异常 7 / 回归 0）；golden 为独立 fp64 实现；拒绝码机判。
- **性能**：7 个 N 点（1 … 1048576），主用例 1M；同卡 torch.sub 同尺度对比，报告由
  `tools/run_benchmark.py` 产出 baseline/candidate 双列与加速比（`performance.json` + `.md`）。
- **可维护性**：新增源码仅 `ops/elementwise_sub/`。对共享工具的改动：新增 `tools/torch_sub_baseline.py`（新增文件），
  以及 `tools/gen_inputs.py`、`tools/run_cases.py` 的两处健壮性修复（拒绝用例 `gen_a/gen_b` 为 `null` 时不再崩、
  支持 `argv` 字段表达参数级拒绝用例）—— 均为向后兼容的增量，其它算子不受影响。
- **已知限制**：`UNKNOWN` —— 设备侧 `Sub` 指令对次正规数的处理是否有 FTZ/DAZ（真机用例 `sub_num_*` 已跑通，
  但与标杆逐位比对仍受判据的非有限位规则约束，未单独隔离验证）。
