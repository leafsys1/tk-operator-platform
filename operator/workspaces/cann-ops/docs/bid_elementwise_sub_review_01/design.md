# 向量减法检测（bid_elementwise_sub_review_01）算子设计文档

| 项目 | 内容 |
|---|---|
| 算子名称 | `bid_elementwise_sub_review_01`（向量减法检测） |
| 目标芯片 / 架构 | Ascend 910B1 / dav-2201（`--npu-arch=dav-2201`） |
| 文档版本 | v1.0 |
| 作者 / 日期 | ops-direct-invoke-architect（节点 0 方案设计）/ 2026-11-02 |
| 规格版本 | v4（current_spec_version_id=4，version_minor=0） |

### 修订记录

| 版本 | 日期 | 修改人 | 修改说明 |
|---|---|---|---|
| v1.0 | 2026-11-02 | ops-direct-invoke-architect | 依规格 v4 首版：五段设计（常规分块单缓冲 / 固定 8192 元素 tile / 无 broadcast / 无 alpha），从已验证参考设计 `docs/bid_elementwise_sub_20261101/design.md`（参考来源）适配重命名 |

---

## 1. 需求背景

### 1.1 需求来源

- 任务：task id=2，project id=1，标的算子 `bid_elementwise_sub_review_01`，任务标题「向量减法检测」。
- 规格：`spec v4`（`current_spec_version_id=4`，`version_minor=0`）。
- 用例集：`suite_ref = cases@6afae1abee1c9e22`；`cases_confirmed_sha256 = 6afae1abee1c9e22d2b67f06de46f063d740da7d6663d35b0d7035b9fd4bc9e7`（2026-11-02T04:03:58.997416 by chat:job#18）。
- 判据档：`criteria_ref = ops-precision-standard@1`。
- 权威输入：`$WORK_DIR/需求分析.md`（v4）、`$WORK_DIR/环境信息.md`（只读副本）、`cases/bid_elementwise_sub_review_01/{cases.json(spec_version=4),benchmark.json(无 spec_version)}`。
- 目标设备：device_id=4，timeout_s=1800。

### 1.2 业务背景与使用场景

向量逐元素减法检测场景：以「复现同类算子语义、但以独立算子身份交付」的方式验证算子开发链路（规格 → 用例 → 代码 → 四动作 → 交付）。算子本体是向量逐元素减法，属 Ascend C 直调（direct-invoke）最简二元算子形态，用于复验从 tiling 到 host/kernel 的完整交付链路，不承担 alpha 缩放、广播、原位写回等扩展语义。

典型调用方式：算子直调（Python 接口 `subtract(x1, x2) -> Tensor`）+ 四动作门禁用的可执行文件 CLI；另有独立示例 `examples/bid_elementwise_sub_review_01_demo.py`（示例 [3,1,2]-[1,1,1]=[2,0,1]）。无图模式接入点。

### 1.3 算子数学语义

$$ y[i] = x1[i] - x2[i], \quad i \in [0, N) $$

- 输入：`x1`、`x2` 为同 shape 的 1-D 连续 float32 NPU 张量，shape=`[N]`，**不广播**、**不混合 CPU/NPU**、**不跨 NPU**。
- 输出：与输入同设备、连续、shape=`[N]`、dtype float32 的**新张量**（每次调用新建，非 in-place、非 out 复用）。
- N 范围：1 ≤ N ≤ 1048576（上界 = 2^20）。
- 权威参考（oracle）：`torch.sub(x1.cpu(), x2.cpu())`（CPU 逐元素减法）。无候选 CPU 回退 / 无 torch 回退，候选实现必须是真实 Ascend C kernel。
- **无 alpha/scale**：纯 `y = x1 - x2`，两参接口。`alpha`、`out=`、`in-place` 属参考算子扩展能力，**不纳入**本算子。
- **特殊值语义**：纯减法无 trap；NaN 保持按位置传播、Inf 按符号相减（Inf-Inf=NaN、Inf-(-Inf)=+Inf 等），与 CPU torch.sub 对齐；NaN 按位置和类别核对、Inf 按符号核对，NaN payload 与零符号位不作逐位要求；有限值按 `ops-precision-standard@1` 容差核对（由判据档容差与 NaN/Inf 分类核对兜底）。

### 1.4 历史实现与现状分析

无首版实现。复用依据（参考来源）：已验证参考工程 `ops/bid_elementwise_sub_20261101/` 及其设计 `docs/bid_elementwise_sub_20261101/design.md`（同算子语义、已验证、v4）。该参考实现的底层 tiling/尾块/ABI 机制继承自直调工程 `ops/elementwise_sub/`（减法 + alpha，P0 分摊式 tiling + 双缓冲 + opMode）与 `ops/elementwise_add/`（常规分块 tiling + `extern "C"` ctypes ABI）。本算子沿用参考实现中「常规分块单缓冲 / 固定 8192 元素 tile / 无 opMode（仅 Sub）」的 tiling 授权，复用其已验证的尾块对齐机制与 host/ABI 结构；目标交付件由参考工程全量重命名 `bid_elementwise_sub_20261101` → `bid_elementwise_sub_review_01` 得到。

---

## 2. 需求分析

### 2.1 外部组件依赖

| 组件 | 用途 | 来源 |
|---|---|---|
| `acl/acl.h` | host 侧设备管理（SetDevice/CreateStream/Malloc/Memcpy/GetDeviceInfo） | CANN 运行时 |
| `kernel_operator.h` | Ascend C kernel 编程（TPipe/GlobalTensor/LocalTensor/DataCopyPad/Sub） | CANN Ascend C 工具链 |
| `tiling_api` / `platform` / `register` | 编译期链接 | CMake 目标库 |
| `libbid_elementwise_sub_review_01_host.so` | ctypes 加载的 host C ABI | 本算子 CMake 产出（本轮不编译，真机 build 阶段产出） |
| 无第三方数学库 | 减法为硬件 `Sub` 原语，无需 libm | — |

### 2.2 内部适配模块（复用点）

直接复用源：已验证参考工程 `ops/bid_elementwise_sub_20261101/`（参考来源，同算子语义 v4）。其内部适配机制沿袭如下直调工程：

| 复用模块 | 来源文件 | 复用内容 |
|---|---|---|
| 常规分块 tiling 公式 + kernel Init 区间 + 每 tile 推进 GM 偏移 | `ops/elementwise_add/op_kernel/elementwise_add_tiling.h`、`op_kernel/elementwise_add_kernel.asc` | `totalTiles/tilesPerCore/blockNum/numPerCore/tailNumLastCore` 分核公式（继承 add 的 per-block GM 偏移修正） |
| DataCopyPad 尾块对齐机制（32B 对齐写回 / 向量段 + `<8` 标量收尾；本算子适配为「只搬运 `count * sizeof(float)` 有效字节 + DataCopyPad 在 UB 侧补齐」，不再显式写 0） | `ops/elementwise_sub/op_kernel/elementwise_sub_kernel.asc`（L80-175） | 非 32B 尾块与 N<8 的搬运对齐 |
| host CLI 退出码映射 / `EXEC` 标记行 / KernelCall（aclrtMalloc+Memcpy H2D + ReadFile） | `ops/elementwise_sub/op_host/elementwise_sub.asc`、`ops/elementwise_add/op_host/elementwise_add.asc` | 四动作可执行文件契约 |
| ctypes `.so` C ABI + 不 aclInit/Finalize 约定 + 运行时取核数 | `ops/elementwise_add/op_host/elementwise_add_pyapi.asc`、`python/elementwise_add.py` | `extern "C"` run ABI + `abi_version` + `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` |
| 数据工具 | `ops/elementwise_sub/op_host/data_utils.h` | `ReadFile`/`WriteFile` 裸二进制读写 |

差异化（必做）：参考 `elementwise_sub` 为「P0 分摊式 + DOUBLE_BUFFER=2 + opMode（SUB/SCALE/FMASUB）」，本算子为「常规分块单缓冲 + 固定 8192 元素 tile + 无 opMode（仅 Sub）」；复用其已验证的尾块对齐与 ABI 结构，不复用其 tiling 与 alpha/scale/fmasub 扩展。

### 2.3 需求模块设计

| 模块 | 目标文件（相对 `ops/bid_elementwise_sub_review_01/`） | 职责 |
|---|---|---|
| 常量/结构体 | `op_kernel/bid_elementwise_sub_tiling.h` | `BidSubTilingData`（纯 POD）+ `BuildTiling`（纯 C/C++，kernel/host 共用；短名不变） |
| kernel | `op_kernel/bid_elementwise_sub_kernel.asc` | `bid_elementwise_sub_kernel` 入口 + Init/Process（CopyIn→Compute→CopyOut 三段式；短名不变） |
| host CLI | `op_host/bid_elementwise_sub_review_01.asc` | `main`（参数解析/拒绝/EXEC 行/KernelCall），产出可执行文件 |
| host ABI | `op_host/bid_elementwise_sub_review_01_pyapi.asc` | `extern "C"` run ABI，产出 `libbid_elementwise_sub_review_01_host.so` |
| 数据工具 | `op_host/data_utils.h` | `ReadFile`/`WriteFile` 裸二进制读写 |
| Python 接口 | `python/bid_elementwise_sub_review_01.py` | ctypes 封装 + 入参校验 + `subtract(x1, x2)` |
| 参考实现 | `scripts/golden.py` | golden（权威 = CPU torch.sub，numpy fp64 仅辅助） |
| 构建 | `CMakeLists.txt` | 双目标（可执行文件 + 共享库，保留原可执行目标） |
| 说明 | `README.md` | 语义/接口/构建/精度性能口径/已知限制 |
| pytest | `tests/test_bid_elementwise_sub_review_01.py` | 读 cases.json，候选走 Python 接口、golden 走 `scripts/golden.py`，按 `criteria/check.py` 同口径 |
| 独立示例 | `examples/bid_elementwise_sub_review_01_demo.py` | 真实调用 `subtract`，示例 [3,1,2]-[1,1,1]=[2,0,1] |

### 2.4 算子原型

#### 2.4.1 输入

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `x1` | float32 | ND 连续（一维连续） | `[N]` | NPU 张量，N∈[1,1048576]，支持非零 `storage_offset` 连续视图 |
| `x2` | float32 | ND 连续（一维连续） | `[N]` | NPU 张量，与 `x1` 同 shape、同 dtype、同设备 |

#### 2.4.2 输出

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `y` | float32 | 连续 | `[N]` | 与输入同设备的新连续张量，`y[i]=x1[i]-x2[i]` |

#### 2.4.3 属性 / 参数

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| —（无算子属性） | — | — | — | 纯两参逐元素减法，无 alpha/scale/广播等属性 |

#### 2.4.4 推导规则

- 输出 shape = `[N]`（与输入一致）；dtype = float32；设备 = 输入所在 NPU。
- 约束校验优先级：host CLI 为参数解析 `REJECT_BAD_ARGS` → `REJECT_UNSUPPORTED_DTYPE`（非 float32） → `REJECT_SHAPE_MISMATCH`（长度不一致，不广播）→ `REJECT_N_OUT_OF_RANGE`（N<1 或 N>1048576）；Python 接口为非 Tensor → `REJECT_UNSUPPORTED_DTYPE` → `REJECT_SHAPE_MISMATCH` → rank/连续性 → NPU 类型/同设备 → `REJECT_N_OUT_OF_RANGE`。
- 拒绝码唯一合法集合：`REJECT_BAD_ARGS` / `REJECT_SHAPE_MISMATCH` / `REJECT_N_OUT_OF_RANGE` / `REJECT_UNSUPPORTED_DTYPE`（执行字符串带 `REJECT_` 前缀，与 cases.json `expect_reject_code` 逐字一致）。Python 层异常类映射见规格/需求分析 §3，多重错误以实际入口检查顺序核对。

---

## 3. 需求详细设计

### 3.1 使能方式

Ascend C 直调（direct-invoke）：host 通过 `aclrtMalloc`/`aclrtMemcpy` 管理裸二进制缓冲，`kernel<<<blockNum,nullptr,stream>>>` 拉起，无需框架算子切分。编译目标架构 `dav-2201`，实现标签 `impl=ascendc-direct-invoke`。Python 接口经 ctypes 加载同源共享库 `libbid_elementwise_sub_review_01_host.so`，不做 aclInit/aclFinalize（torch_npu 可能已持有 ACL context）。

### 3.2 总体设计方案

#### 3.2.1 Host 侧

**tiling 与分核策略（常规分块单缓冲，固定 8192 元素 tile）**

紧凑型 tiling 结构（纯 POD，kernel 从 GM 读）：

```c
constexpr uint32_t TILE_LENGTH      = 8192;     // 固定 tile 长度（元素）；8192 % 8 == 0 → 32B 对齐
constexpr uint32_t VEC_ALIGN_LENGTH = 8;        // 8 元素 = 32B 对齐粒度
constexpr uint32_t DOUBLE_BUFFER    = 1;        // 单缓冲
constexpr uint32_t DEFAULT_CORE_NUM = 48;       // 910B 逻辑核静态参考；运行时以 aclrtGetDeviceInfo 为准
constexpr int64_t  N_MIN = 1, N_MAX = 1048576;  // 2^20

struct BidSubTilingData {
    uint32_t blockNum;          // 实际使用的 block 数（≤ 物理核数）
    uint32_t tileLength;        // 恒 = TILE_LENGTH = 8192
    uint64_t totalLength;       // N
    uint64_t numPerCore;        // 非尾核处理元素数 = tilesPerCore * tileLength
    uint64_t tailNumLastCore;   // 尾核处理元素数 = N - numPerCore*(blockNum-1)
};
```

分核公式（host `BuildTiling` 算一次，kernel 复用同一套区间）：

```text
coreNum     = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM)，无效时取 DEFAULT_CORE_NUM
totalTiles  = ceil(N / TILE_LENGTH)                              // N>0 恒 ≥1
tilesPerCore= ceil(totalTiles / coreNum)                         // 每核摊到多少 tile
blockNum    = ceil(totalTiles / tilesPerCore)                    // 恒 ≤ coreNum（见下证）
numPerCore  = tilesPerCore * TILE_LENGTH
tailNumLastCore = N - numPerCore * (blockNum - 1)                // ∈ [1, numPerCore]
```

**blockNum ≤ coreNum 证明**：`tilesPerCore = ceil(totalTiles / coreNum) ≥ totalTiles / coreNum`，故 `totalTiles / tilesPerCore ≤ coreNum`，因此 `blockNum = ceil(totalTiles / tilesPerCore) ≤ coreNum`。

tiling 推算实算（`coreNum=48` 参考）：

| N | totalTiles | tilesPerCore | blockNum | numPerCore | tailNumLastCore |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 8192 | 1 |
| 17 | 1 | 1 | 1 | 8192 | 17 |
| 1024 | 1 | 1 | 1 | 8192 | 1024 |
| 8192 | 1 | 1 | 1 | 8192 | 8192 |
| 65536 | 8 | 1 | 8 | 8192 | 8192 |
| 1048576 | 128 | 3 | 43 | 24576 | 16384 |

字节对齐实算（`bytes = elements × 4`，float32）：

| N | 元素数 | dtype 字节 | 总字节 | bytes % 32 |
|---|---|---|---|---|
| 1 | 1 | 4 | 4 | 4 ≠ 0 |
| 17 | 17 | 4 | 68 | 4 ≠ 0 |
| 1024 | 1024 | 4 | 4096 | 0 |
| 8192 | 8192 | 4 | 32768 | 0 |
| 65536 | 65536 | 4 | 262144 | 0 |
| 1048576 | 1048576 | 4 | 4194304 | 0 |

- 边界覆盖：N=1（单元素）与 `bytes%32≠0`（N=17：68%32=4）为非 32B 尾块（DataCopyPad 的 pad 补齐支路，rightPad>0）；N=1024/8192/65536/1048576 均 32B 对齐（pad=0）。无论对齐与否，本轮都统一走 `DataCopyPad`（无 `DataCopy` 对齐支路）。
- 8192×4=32768 ≤ 65535（DataCopyPad blockLen 上限 uint16），满足单块长度约束。

**UB 预算（单缓冲）**：三队列（xGm→UB、yGm→UB、UB→zGm）× 1 buffer × 8192 × 4B = 98304B（96KB）≤ 196608B（UB 192KB，`dav-2201.yaml` ub_size_bytes=196608 high）。单缓冲运行态安全；不设 workspace。

**shape / dtype 校验（拒绝码）**：见 §2.4.4；校验在 host 侧 launch 前完成，`REJECT_*` 字符串与退出码一并返回，不进入 kernel。

#### 3.2.2 Kernel 侧

- **数据搬运**：`GlobalTensor<float>` xGm/yGm/zGm；搬运长度恒为有效元素数 `count`，只搬 `count * sizeof(float)` 字节、不对 GM 侧向上取整到 32B（GM 侧不读/写逻辑 in/out 元素之外的任何字节）。整块与非 32B 尾块 / N<8 **统一走 `DataCopyPad`**（`DataCopy` 在非 32B 对齐源上不适用、编码规则黑名单）：非 32B 尾块由 `padParams`（isPad=true + rightPad = (count%8==0 ? 0 : 8-count%8) + paddingValue=0）在 **UB 侧**补齐 32B（dummy=0），GM 侧只移动有效字节；32B 对齐整块 pad=0、同样走 DataCopyPad。
- **向量计算流水**：`TQue<TPosition::VECIN,1>` inQueueX / inQueueY + `TQue<TPosition::VECOUT,1>` outQueueZ，单缓冲（DOUBLE_BUFFER=1），CopyIn→Compute→CopyOut 三段式按 tileNum 循环；每 block 处理区间 `[blockIdx*numPerCore, blockIdx*numPerCore + total)`，`total = (blockIdx<blockNum-1) ? numPerCore : tailNumLastCore`；第 i 块 GM 偏移 = 基址 + `i*tileLength`（继承 add 的 per-block 偏移修正）。
- **计算**：先取 `vecCount = count / 8 * 8`，`Sub(zLocal, xLocal, yLocal, vecCount)` 向量段；`[vecCount, count)` 不足 8 的残段按标量逐元素 `x - y` 收尾（尾核 last 块）。只覆盖 `[0, count)` 有效元素，不对补齐槽显式写 0（补齐由 DataCopyPad 在搬运侧负责，与计算无关）。
- **精度处理**：float32 直算，无精度晋升、无累加；**不断言与 fp64 逐位相等**，精度门禁以 oracle（CPU torch.sub）在 rtol/atol 容差内逐元素核对为准（见 §5.1）。
- **结果写回**：`DataCopyPad`（UB→GM，无 padParams）写回 zGm；UB→GM 由框架自动补齐 UB 读取所需 dummy 并在写回 GM 时丢弃，只写 `count * sizeof(float)` 有效字节，不越界写、不显式写 0。写回尺寸由 `tileNum / tailTileElementNum` 控制，确保新张量全量覆盖。

### 3.3 支持硬件

目标芯片 Ascend 910B1（SoC `dav-2201`），`--npu-arch=dav-2201`。静态画像（`ops/chips/dav-2201.yaml`）：UB 196608B、AIV 逻辑核 48（历史实测参考 M1-b）、fp32 向量宽度 64 元素/repeat、DataCopy 对齐 32B。运行时**仅**核数以 `aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM)` 返回为准（host 在 `aclrtSetDevice` 后查询，无效/≤0 时回落 `DEFAULT_CORE_NUM=48`），代码不硬编码物理核数；UB 不运行时查询（不承诺不存在的 UB/内存查询能力），UB 预算按固定 8192 单缓冲静态上限（96KB ≤ 192KB）设计，见 §3.2.1。多芯片形态本算子不特殊支持（单卡，双卡不执行、记为「未验证」）。

### 3.4 算子约束与限制

- 仅 float32 / 1-D 连续 / 同 shape NPU 输入；不广播、不混合 CPU/NPU、不跨 NPU、无 alpha/scale。
- N 范围 [1, 1048576]；N=0 或 N>1048576 → `REJECT_N_OUT_OF_RANGE`。
- CPU 输入、CPU/NPU 混用、跨 NPU、非一维、非连续 → `REJECT_BAD_ARGS`（唯一合法集合内）；float16/int32 → `REJECT_UNSUPPORTED_DTYPE`。
- 非零 `storage_offset` 的连续视图：候选经 `data_ptr()` 透传偏移后的设备地址，按逻辑元素计算；pytest 构造（本轮必须）：先将整个 storage 放到 NPU 再切片得到 `storage_offset≠0` 视图，覆盖 N=8/17 与 offset=1/4，调用候选前断言实际 storage_offset 与预期一致。
- 性能前提假设：固定 8192 tile 的常规分块，单缓冲；真实耗时、带宽、核利用率待真机 profiling，结论一律「待测」。

---

## 4. 特性交叉分析

### 4.1 混合精度

**不涉及**。本算子仅支持 float32 输入/输出，无混合 dtype、无类型晋升、无精度转换点；非 float32 一律 `REJECT_UNSUPPORTED_DTYPE`。

### 4.2 图模式

**不涉及**。本算子为 Ascend C 直调（direct-invoke），无图引擎接入、无图融合、无动态 shape 图推导；Python 接口直接 ctypes 调用 host C ABI。

### 4.3 动态 shape

**运行时 tiling 自适应 N**。N 在 [1,1048576] 自由取值，tiling 结构体在 launch 前按 N 实时 `BuildTiling`，无 shape 特化、无重编译；kernel 从 GM tiling 读参数，支持任意合法 N（含 N=1 与非 32B 尾块）单次 launch。

### 4.4 其他特性

- **storage_offset**：见 §3.4；实现经 `data_ptr()` 天然携带偏移，零拷贝、无额外 workspace。
- **广播 / 切分 / 量化 / 内存复用**：均不涉及（不广播、单算子、float32 无量化；输出新张量、不复用输入存储）。
- **多流 / 多卡**：不特殊支持；双卡默认不执行、如实记「未验证」，不声称验证。

---

## 5. 可维可测分析

### 5.1 精度标准

- 权威参考（oracle）：`golden = torch.sub(x1.cpu(), x2.cpu())`（CPU 逐元素减法），与规格 oracle 逐字一致；`scripts/golden.py` 的权威判定必须直接使用 CPU torch.sub 对照。
- numpy fp64 单次舍入（`a.astype(np.float64) - b.astype(np.float64)` 后回 float32）**仅作辅助交叉核对，不能替代权威 golden**；若与其一致但 torch.sub 不一致，以 torch.sub 为准。**不得**把 numpy fp64 写成权威 golden，**不得**把 torch 写成「第二参照（非权威）」。
- 判据档 `ops-precision-standard@1`：rtol=2^-10，atol=2^-16，matched_ratio ≥ 0.99，max_abs_err ≤ 0.01。命中即 PASS。
- shape/dtype 对照 CPU 参考（shape 同 `[N]`、dtype float32）；device 对照输入所在 NPU（输出与输入同设备）。
- 特殊值：NaN 按位置类别、Inf 按符号逐元素核对；NaN payload 与零符号位不做逐位要求。
- 拒绝码核对：与 cases.json `expect_reject_code` 逐字一致（`REJECT_*`）。

### 5.2 性能标准

- 测量口径（benchmark.json）：warmup=2，invocations=5，retries_per_invocation=3；device 侧 Task Duration（μs），取中位数；主用例 N=65536、另一用例 N=1048576；基线同设备 torch.sub。
- 用例：`cases/bid_elementwise_sub_review_01/benchmark.json`（2 条，**无 spec_version 字段**，运行期规格由平台注入）。
- 目标：无硬性目标，本轮不采集，**性能结论一律写「待测」**；真机 build/smoke/test/benchmark/profile 延后至网页四动作或后续轮次。

### 5.3 兼容性分析

- **接口兼容性**：Python 接口 `subtract(x1, x2) -> Tensor`（两参，无 alpha/out/in-place）；host C ABI 与参考 `elementwise_sub_pyapi` 同构（`run(aDev,bDev,outDev,n,dtypeCode,deviceId)` + `abi_version`），命名带上 `bid_elementwise_sub_review_01_` 前缀；四动作可执行文件 CLI 沿用参考 `--n-a/--n-b/--dtype/--in-a/--in-b/--out` 契约，退出码沿用 0/2/3/4/5/1 映射（0=成功、1=运行期错误、2=REJECT_BAD_ARGS、3=REJECT_SHAPE_MISMATCH、4=REJECT_N_OUT_OF_RANGE、5=REJECT_UNSUPPORTED_DTYPE）。
- **数据类型兼容性**：仅 float32；减法无精度损失晋升，误差预期在 §5.1 容差内。
- **硬件形态兼容性**：目标 dav-2201（910B1）。核数运行时动态读取（UB 不运行时查询、按固定 8192 单缓冲静态预算），不同核数下 `blockNum ≤ coreNum` 公式不变量成立；非 910B1 子型号行为记 `UNKNOWN`。

---

## 附录 A（选填）：测试用例要点

已确认用例集 `cases/bid_elementwise_sub_review_01/cases.json`（spec_version=4，15 条：10 match + 5 reject）。

### A.1 精度用例

| Case ID | 分组 | shape(N) | dtype | 输入生成方式 / 种子 | 期望（match / reject + 拒绝码） | 说明 |
|---|---|---|---|---|---|---|
| sub_norm_001 | 常规 | 1 | float32 | uniform[-1,1] / 20261101 | match | 最小合法 N=1 |
| sub_norm_002 | 常规 | 17 | float32 | uniform[-1,1] / 20261102 | match | 非 32B 尾块（68B%32=4） |
| sub_norm_003 | 常规 | 1024 | float32 | uniform[-1,1] / 20261103 | match | 32B 对齐 |
| sub_norm_004 | 常规 | 1048576 | float32 | uniform[-1,1] / 20261104 | match | 上界规模 |
| sub_num_001 | 数值边界 | 17 | float32 | 双 leading nan/inf/-inf / 20261105 | match | NaN/Inf 双侧 |
| sub_num_002 | 数值边界 | 17 | float32 | a 特殊,b 均匀 / 20261106 | match | NaN/Inf 仅 a |
| sub_num_003 | 数值边界 | 17 | float32 | a 均匀,b 特殊 / 20261107 | match | NaN/Inf 仅 b |
| sub_num_004 | 数值边界 | 17 | float32 | 双零 / 20261108 | match | 0-0=0 |
| sub_num_005 | 数值边界 | 1024 | float32 | a 均匀,b 零 / 20261109 | match | 右零 |
| sub_num_006 | 数值边界 | 1024 | float32 | a 零,b 均匀 / 20261110 | match | 左零 |

| Case ID | 分组 | shape(N) | dtype | 输入生成方式 | 期望（reject + 拒绝码） | 说明 |
|---|---|---|---|---|---|---|
| sub_rej_001 | 异常 | 0 | float32 | — | reject + REJECT_N_OUT_OF_RANGE | 空输入 N=0 |
| sub_rej_002 | 异常 | 1048577 | float32 | — | reject + REJECT_N_OUT_OF_RANGE | 超上界 |
| sub_rej_003 | 异常 | 1024/1023 | float32 | — | reject + REJECT_SHAPE_MISMATCH | 长度不一致不广播 |
| sub_rej_004 | 异常 | 1024 | float16 | — | reject + REJECT_UNSUPPORTED_DTYPE | 非 float32 |
| sub_rej_005 | 异常 | 1024 | int32 | — | reject + REJECT_UNSUPPORTED_DTYPE | 非 float32 |

### A.2 性能用例

| Case ID | n | dtype | 主用例 | 必需 | 说明 |
|---|---|---|---|---|---|
| （benchmark.json 2 条） | 见 benchmark.json | float32 | 见文件 | 见文件 | 无 spec_version 字段，运行期规格平台注入 |

---

## 附录 B（选填）：参考资料

| 用途 | 路径 |
|---|---|
| 需求分析（权威输入） | `$WORK_DIR/需求分析.md`（v4） |
| 环境信息 | `$WORK_DIR/环境信息.md` |
| 交付物规范 | `app/docs/reference/deliverable-standards.md` |
| 设计模板 | `app/docs/templates/算子设计文档模板.md` |
| 已验证参考工程 / 参考设计（复用源，参考来源） | `ops/bid_elementwise_sub_20261101/`、`docs/bid_elementwise_sub_20261101/design.md` |
| 常规分块 tiling | `ops/elementwise_add/op_kernel/elementwise_add_tiling.h`、`elementwise_add_kernel.asc` |
| 尾块机制 + host CLI/ABI | `ops/elementwise_sub/op_kernel/elementwise_sub_kernel.asc`、`op_host/elementwise_sub.asc`、`op_host/elementwise_sub_pyapi.asc`、`python/elementwise_sub.py` |
| ctypes ABI + golden 参考 | `ops/elementwise_add/op_host/elementwise_add_pyapi.asc`、`python/elementwise_add.py` |
| 芯片画像 | `ops/chips/dav-2201.yaml` |
| 用例 | `cases/bid_elementwise_sub_review_01/cases.json`（spec_version=4）、`benchmark.json`（无 spec_version） |
| 精度判据 | `criteria/ops-precision-standard@1`、`criteria/check.py` |

---

> 未验证声明：`build / smoke / test / benchmark / profile` 五动作本轮均未触发，真机/精度/性能结论一律「待测 / 未验证（待网页执行）」。本节点仅做静态设计与落盘，不编译、不上机、不宣称任何硬件或精度通过。
