# jax_lax_scan 算子设计文档（AscendC 实现）

| 项目 | 内容 |
|---|---|
| 算子名称 | `jax_lax_scan` |
| 目标芯片 / 架构 | Ascend 910B1 / `dav-2201` |
| 目标镜像 | `opx-xla-npu:jax081-cann920-910b`（CANN 9.2.0-beta.2、bisheng、JAX/JAXLIB 0.8.1、`xla-npu/AFIR`） |
| 文档版本 | v1.1 |
| 作者 / 日期 | 节点 5（文档准备）/ 2026-10-27 |

> 路径约定：文中 `0-*` / `2-*` / `3-*` / `4-*` 报告、探针与证据目录均位于本轮工作目录
> `.cannbot/jax_lax_scan/workflow1/`，属过程证据，不随交付包分发。

### 修订记录

| 版本 | 日期 | 修改人 | 修改说明 |
|---|---|---|---|
| v1.0 | 2026-10-26 | 节点 2 | 首版：按已确认 spec v1 / 冻结用例集与 `0-黑盒测试设计.md` §1.3 契约实现 |
| v1.1 | 2026-10-27 | 节点 5 | 按 `docs/templates/算子设计文档模板.md` 补全 §3.1 使能方式、§3.3 支持硬件、§3.4 约束与限制、§4.1–4.4、§5.1–5.6 与附录 A/B；同步节点 4 修复后的验证证据（pytest 19 条 / 33 次、run #73/#74/#76、`body_id` 断言）；未改实现、判据与用例集 |

---

## 1. 需求背景

### 1.1 需求来源

JAX 侧模型/算法在昇腾上落地时，`lax.scan`（静态循环）是 RNN、线性递归、迭代求解等结构的基础控制流原语。
本任务由控制台任务 #12 发起，验收方为控制台「四动作 + 判据档」门禁与用户；真相源为
`tasks/jax_lax_scan/spec/spec.json`（spec v1，`spec_version_id=4`），用例集为
`cases/jax_lax_scan/cases.json`（7 条 / 13 次，`suite_ref=cases@4ff7be4f08af6df4`）与
`benchmark.json`（2 条）。

### 1.2 业务背景与使用场景

`lax.scan` 在 trace 期展开为 `fori_loop` 等价的静态循环。本批在 Ascend 侧交付：

* **直调路径**：Python 入口解析已注册 `body` → C ABI（`.so`）→ Ascend C kernel；
* **兼容路径**：未注册 callable（以及自动微分）由 JAX 自己在 `xla-npu` 后端 trace 执行。

典型调用为算子直调（非图内节点）；本批不涉及图引擎接入或融合子图。

### 1.3 算子数学语义

给定 `init`（carry pytree）、`xs`（pytree 或 `None`）、静态长度 `L`、`reverse`：

```
carry_0 = init
for t in 0..L-1:
    idx  = (L-1-t) if reverse else t
    carry_{t+1}, y_t = f(carry_t, xs[idx])
ys[t] = y_t                       # 始终按原始索引写回
return carry_L, stack(ys)
```

* 与 `jax.lax.scan`（JAX 0.8.1）一致：`reverse=True` 先反转 `xs`、扫描后再把 `ys` 反转回来，
  因此 `ys[t]` 对应原始索引 `t`（§3.2.2 的 `StepOnce` 与 `0-设计复核探针.log` P3/P4 已核实）；
* NaN / ±Inf：逐元素语义一致（`criteria/check.py` 先做非有限位精确比对）；
* `carry` 结构 / shape / dtype 每步不变；本批冻结用例中改变 carry 的 body（4/5/6）在 trace 期
  稳定拒绝为 `carry_shape_mismatch`（未注册 callable 的同一类错误由 JAX 自身报出，见 §3.2.1 注）。

### 1.4 历史 TBE 实现与现状分析

无历史 TBE 实现，首版。参考依据：JAX 0.8.1 `lax.scan` 语义、`0-黑盒测试设计.md` §1.3 的候选接口锁定结论、
以及本仓 `ops/elementwise_sub` 已验证的 Ascend C 直调工程布局（kernel 三段式 / tiling / pyapi / 四动作入口）。

---

## 2. 需求分析

### 2.1 外部组件依赖

| 组件 | 版本 / 位置 | 用途 |
|---|---|---|
| CANN toolkit | 9.2.0-beta.2（`/usr/local/Ascend/cann-9.2.0-beta.2`） | bisheng / ccec / ACL / `find_package(ASC)` |
| 编译平台 | `dav-2201`（`--npu-arch=dav-2201`） | Ascend 910B1 |
| Python | 3.11.17（`/opt/conda/bin/python3`） | host driver / 判据 / golden 子进程 |
| JAX / JAXLIB | 0.8.1（`xla-npu` PJRT 插件，backend `npu`） | trace 路径与 CPU oracle |
| numpy | 有 | `inputs.npz` / `outputs.npz` |
| 设备 | 910B1 ×1，device 0，独占 | 真机执行 |

> **镜像与验证环境（实现期实测，见 `4-测试执行记录.md`）**：生产镜像 `opx-xla-npu:jax081-cann920-910b`
> （#136）只有 JAX/JAXLIB，**没有** `torch`/`torch_npu`/`pytest`；真机 `build`/`smoke`/`test` 使用控制台
> 测试镜像 `opx-xla-npu:jax081-cann920-910b-test`（#145，同一 JAX 0.8.1 + CANN 9.2.0-beta.2，另含
> `pytest`，但不含 `torch`/`torch_npu`）。JAX 主门禁固定使用 #145；torch 公开接口在已验收的
> `lfx-op-dev-910b:torch210-cann910`（#80）上做补充真机探针（5 条冻结数值用例 × eager/jit +
> length=2 回归），证据为
> `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch-family-run.log` 与
> `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch/scan_correctness.json`。

### 2.2 内部适配模块

| 适配点 | 说明 |
|---|---|
| 四动作入口 | `build.sh` 只 `cmake .. && make jax_lax_scan`；可执行文件路径固定 `ops/jax_lax_scan/build/jax_lax_scan` |
| 共享 runner | `tools/run_cases.py` 已按 `op == "jax_lax_scan"` 分派（节点 1 的 A1 改动），本节点不改 `tools/` |
| golden | `ops/jax_lax_scan/scripts/golden.py` + `golden_cli.py`（CPU 子进程），本节点不改其 oracle 语义 |
| 判据 | `criteria/check.py`（`ops-precision-standard@1`），只读复用 |
| 用例 | `cases/jax_lax_scan/cases.json`（7 条 / 13 次）与 `benchmark.json`（2 条），冻结、不改 |
| 图引擎 | 不涉及：本批为算子直调，不注册 GEIR/ACLNN，不做图内 shape/dtype 推导 |
| workspace | 无：kernel 只用 UB 持有 carry；host 不做 workspace 申请，设备缓冲由 Python/CLI 侧管理 |
| 错误码体系 | 统一 `0/1/2/3/4`（见 §3.2.1），CLI 退出码、`.so` 返回值与 Python 异常文本同表 |
| 日志 / 维测 | stdout `EXEC ...` 标记行、`meta.json`、`_last_dispatch`、`.so` 的 `err` 缓冲 |

### 2.3 需求模块设计

| 模块 | 文件 | 职责 |
|---|---|---|
| device kernel | `op_kernel/jax_lax_scan_kernel.asc` | carry 常驻 UB，逐步取 `x_t`、算 `new_carry`/`y`、写回 `ys` |
| tiling | `op_kernel/jax_lax_scan_tiling.h` | body / dtype / 步区间等纯 C 结构，kernel 与 host 共用 |
| 可执行入口 | `op_host/jax_lax_scan.asc` | argv 协议校验 + 转交 host driver；`--self-check` 校验交付布局与 `.so` |
| C ABI | `op_host/jax_lax_scan_pyapi.asc` + `op_host/jax_lax_scan_abi.h` | `extern "C" jax_lax_scan_run`：参数校验 / tiling / kernel launch / 设备缓冲辅助 |
| Python 接口 | `python/jax_lax_scan.py` | `scan` / `scan_compiled` / `_plan_dispatch` / `_last_dispatch`；族分派与校验 |
| host driver | `python/jax_lax_scan_cli.py` | `--case-json/--inputs/--mode/--outdir` 协议、`outputs.npz`/`meta.json`、AD 与拒绝分类 |
| 参考实现 | `scripts/golden.py`（节点 1） | 同版本 JAX CPU oracle，独立子进程 |
| 测试 | `tests/test_jax_lax_scan.py`（节点 1/4） | pytest 交付件（19 条 / 33 次），与门禁同源判据 |

### 2.4 算子原型

#### 2.4.1 输入

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `init` | float32 / int32 | ND | 任意（pytree） | 初始 carry；本批 `f32[W]` 或 `(int32 标量, f32[W])` |
| `xs` | float32 | ND | `(L, …)` | 序列输入；pytree 时各叶子前导轴 = `L`；可为 `None` |
| `f` | callable | — | — | `f(carry, x_t) -> (new_carry, y)`；已注册 body 用 `body_id` 标识 |

#### 2.4.2 输出

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `final_carry` | 同 `init` | ND | 同 `init` | pytree 结构完全一致 |
| `ys` | float32 | ND | `(L,) + y.shape` | 各步 `y` 沿新轴堆叠 |

#### 2.4.3 属性 / 参数

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `length` | int | 由 `xs` 前导轴推导 | ≥ 1 | 静态步数；`xs=None` 时必填 |
| `reverse` | bool | `False` | — | 逆序扫描 |
| `unroll` | int | 1 | 仅 1 | 本批不支持展开 |

#### 2.4.4 推导规则

* `length`：`xs != None` 时取前导轴，显式给出则必须相等；`xs == None` 时必须有显式值；
* `ys.shape[0] == length`；`final_carry` 的结构 / dtype / shape 由 `init` 推导，逐叶子保持；
* `y` 的 shape 取 `carry` 中**首个 float32 叶子**的 shape（本批 5 个可执行 body 均满足）。

```text
length                                   # xs=None 且未给 length → length_mismatch
  → body 路由（4/5/6 先拒绝）            # carry_shape_mismatch（abi_calls=0）
  → 族/设备校验（torch：连续、同 npu 设备、requires_grad）  # invalid_argument / autodiff_unsupported
  → .so 侧 carry 不变式复校验             # carry_shape_mismatch / invalid_argument
  → kernel launch                        # device_error
```

拒绝时机：Python 层（trace 期，任何 device 调用前）与 `.so` 入口（kernel 启动前）各有一道校验；
`.so` 不信任 Python 层，重复校验 carry 叶子数 / dtype / numel / `ys_numel`。

---

## 3. 详细设计

### 3.1 使能方式

采用 **Ascend C 直调**（与 `ops/elementwise_sub` 同一工程形态）：

* 四动作门禁调用可执行文件 `ops/jax_lax_scan/build/jax_lax_scan`；该文件只做 argv 协议校验，
  再 `execvp` 到 `python/jax_lax_scan_cli.py`（见 §3.2.1），数值计算全部在
  `libjax_lax_scan_host.so` → AscendC kernel 内完成；
* Python 交付接口通过 ctypes 直接加载同一份 `.so` 的 `extern "C"` ABI；
* 不注册 ACLNN / GEIR 算子，不做图内 shape 推导或融合子图——本批的交付边界是直调算子与 pytest/CLI 入口。

### 3.2 总体设计方案

#### 3.2.1 Host 侧

**总体数据流**

```
调用方 (Python)
  └─ scan / scan_compiled  [python/jax_lax_scan.py]
       ├─ _plan_dispatch: body_id + 张量族 + 是否 tracer  → so / trace / reject
       ├─ so    : 设备缓冲（torch.data_ptr 或 numpy+ACL）→ jax_lax_scan_run  [.so]
       │            └─ ScanTilingData → jax_lax_scan_kernel<<<1>>>
       │                 └─ UB 持有 carry，逐步 DataCopyPad(x_t) → 向量计算 → DataCopyPad(ys[t])
       ├─ trace : jax.lax.scan / jax.jit(...)（NPU 后端；AD 与未注册 callable）
       └─ reject: trace 期抛 carry_shape_mismatch（abi_calls = 0）

四动作 gate (tools/run_cases.py)
  └─ build/jax_lax_scan [op_host/jax_lax_scan.asc]
       └─ execv python3 python/jax_lax_scan_cli.py
            ├─ eager/jit : _run_so_numpy → .so kernel（一次整循环 或 L 次逐步）
            ├─ jvp/vjp   : JAX NPU trace → jvp_primal/jvp_tangent 或 vjp_primal/vjp_cotangent
            └─ reject    : rc=3 + `EXEC path=reject ...`
```

**可执行入口与四动作协议**

`op_host/jax_lax_scan.asc` 的 `main()` 只做三件事：识别 `--case-json/--inputs/--mode/--outdir`
（未知参数/缺值直接 `rc=2`，不进入算子路径）、定位 `python/jax_lax_scan_cli.py`、转交 argv。
`--self-check` 只 `dlopen` `.so` 并读 ABI 版本，不需要 NPU。CLI 负责写 `outputs.npz` / `meta.json`、
打印 `EXEC ...` 标记行，并按 `body_id` 与模式分派到 `.so`（eager/jit）或 JAX trace（jvp/vjp）。

**C ABI 与 Python 接口**

* ABI 结构体见 `op_host/jax_lax_scan_abi.h`；v1 字段语义不变，末尾追加
  `abi_version / total_length / step_start / step_count`（布局版本 2，ctypes 侧断言 472 字节）；
* 逐叶子 `carry_dtype[8]` / `xs_dtype[8]`：pytree carry 的 `int32 counter` 与 `float32 accumulator`
  不共用全局 dtype 推断；
* `carry_in` / `carry_out` 分离；逐步模式轮换缓冲，输入张量不被原地修改；
* `.so` 在**任何 kernel 启动前**复校验 carry 不变式（`n_carry` / `carry_dtype` / `carry_numel` /
  `ys_numel`），不满足即 `rc=3`（`carry_shape_mismatch`）。

**路由规则（`_plan_dispatch`）**

| 条件 | `path` | `exec_mode` | `abi_calls` |
|---|---|---|---|
| `body_id ∈ {4,5,6}` | `reject` | `reject` | 0 |
| `body_id ∈ {1,2,3,7,8}` 且族为 torch/numpy、无 tracer | `so` | 逐步 / 整循环 | `L` / `1` |
| 其余（JAX tracer、未注册 callable、具体 `jax.Array`） | `trace` | `jax_eager` / `jax_jit` | 0 |

AD（`jvp`/`vjp`）在 `scan` 被 `jax.jvp`/`jax.vjp` 追踪时表现为 tracer → `trace`，
`reason=so_no_autodiff`（`.so` 无自动微分这一事实，不等价于"后端不支持"）。
本版本实测 `xla-npu` 对 `lax.scan` 的 JVP/VJP lowering 报
`UNIMPLEMENTED: Unsupported opcode: tuple (instruction: while.1)`，因此 `scan_jvp_vjp` 稳定分类为
**显式 unsupported**（jvp/vjp 各 3 次同码 + 后端原文 + `path=trace`/`abi_calls=0`/`backend=npu`），
不记 PASS。
**公开 torch 接口的 AD**：`.so` 未注册 torch autograd，JAX 也不能穿过 torch→jax 的 DLPack 边界；
`_run_so_torch` 在 kernel 启动前检查叶子 `requires_grad`，命中即抛
`ValueError("UNSUPPORTED:autodiff_unsupported: ...")`，不返回无 `grad_fn` 的结果。冻结用例
`scan_jvp_vjp` 用测试侧 `jax.Array`，其结论只代表 JAX trace 路径，不冒充 torch 接口 AD。

**异常与拒绝码**

| code | 触发 | 行为 |
|---|---|---|
| `carry_shape_mismatch` | body 4/5/6；`.so` 侧 carry 不变式不满足 | Python 抛 `ValueError` / `.so` 返回 3；`abi_calls=0` |
| `dtype_mismatch` | dtype 不在 `{float32,int32}` | `ValueError` |
| `length_mismatch` | `xs` 前导轴不一致 / `xs=None` 未给 `length` / `length < 1` | `ValueError` |
| `invalid_argument` | 非连续 torch 张量、未知参数、`unroll != 1`、`width ∉ [1,4096]` | `ValueError` / `TypeError`；`.so` 返回 2 时 Python 抛 `RuntimeError`（文本带码） |
| `device_error` | ACL / kernel / 非 CPU 后端要求不满足 | `RuntimeError` / rc=1 |
| `unsupported_body` | 张量族不支持 | `ValueError` |
| `autodiff_unsupported` | 后端明确不支持 AD；torch 输入 `requires_grad=True` | rc=4 + 后端原文，`UNSUPPORTED`，不记 PASS |

> **拒绝码覆盖边界**：稳定映射到 `carry_shape_mismatch` 的是**已注册的拒绝 body（4/5/6）**；它们
> 在 Python trace 期、任何 device 调用前返回（`abi_calls=0`）。未注册 callable 若真的改变 carry
> 结构/shape/dtype，走 JAX trace 时由 JAX 自身报错，当前不统一改写成 `carry_shape_mismatch`
> （本批确认范围之外的通用能力；若后续要求通用校验，需加 trace 期不变式检查并补用例）。

**设备缓冲管理**

* torch 族：调用方 `torch.empty_like` 分配输出，`data_ptr()` 传给 ABI；carry 每叶子两块缓冲轮换，
  输入 `init` 只读；
* host driver（numpy 族）：`.so` 导出的 `jax_lax_scan_malloc/h2d/d2h/free` 管理设备缓冲，
  `aclInit` 由 `jax_lax_scan_acl_init` 幂等执行；进程退出前不 `aclFinalize`（生命周期交给调用方）；
* **workspace**：不使用额外 workspace；设备侧仅 carry / xs / ys 三类数据缓冲。

#### 3.2.2 Kernel 侧

**为什么单核**：`carry` 必须在 L 步之间保持在 UB 内（避免每步 GM 往返与跨核同步），且本批 `W <= 256`
（benchmark 上界 256，接口上界 4096），单核向量吞吐足够；`blockNum` 恒为 1。

**三段式（沿用 elementwise_sub 已验证模式）**：

1. `Init`：按 32B 对齐分配 UB；浮点 `carry` 用 `TBuf<VECCALC>`（`carryBuf`），int32 `counter`
   是 **UB 单元素队列**（`inQI`/`outQI` 各深度 1，读入后以标量 `int32_t` 寄存器持有，写回时
   `Duplicate` 到 UB 再 `DataCopyPad`；kernel 中没有 `counterBuf`）；`x0/x1/y` 用深度 1 的
   `TQue<VECIN>/<VECOUT>`（`EnQue/DeQue` 自带 MTE2→V、V→MTE3 同步）；
2. `Process`：载入初始 carry / counter → 按 `[start, start+count)` 循环 → 写回 carry；
3. `StepOnce`：对逻辑步 `step` 取 `idx = reverse ? length-1-step : step`，`DataCopyPad` 取 `x[idx]`
   → 按 `bodyId` 用 `Add/Adds/Muls/Mul` 算 `new_carry` 与 `y` → `DataCopyPad` 写 `ys[idx]`
   （读与写都用**原始索引** `idx`，因此 reverse 时 `ys` 仍按原始顺序排列）。

**body 分支**（`op_kernel/jax_lax_scan_tiling.h` 的 `SCAN_BODY_*`）：

| body | 计算序列（Ascend C 原语） |
|---|---|
| 1 | `Add(y, carry, x)` → `Adds(carry, y, 0)` |
| 2 | 同 1，且 `counter += 1` |
| 3 | `Muls(y, carry, .5)` → `Add(y, y, x)` → `Adds(carry, y, 0)` → `Mul(y, y, y)` |
| 7 | `Mul(y, add, scale)` → `Add(y, y, carry)` → `Adds(carry, y, 0)` |
| 8 | `Adds(y, carry, 1.0)` → `Adds(carry, y, 0)` |

`y` 与 `new_carry` 在向量 pipe 内按序计算，`carry` 的写回发生在读取旧 `carry` 之后，避免原地覆盖语义错误。
搬运统一用 `DataCopyPad`（支持非 32B 对齐的 `blockLen`），`alignedWidth = ceil(W/8)*8` 只用于 UB 分配；
逐元素算术为 fp32（int32 counter 在标量寄存器上 `+1`，写回前 `Duplicate` 到 UB），
误差按 `ops-precision-standard@1` 判定（run #76 的 28 个非 AD 数值 mode 全部通过）。

**Tiling 设计**

```c
struct ScanTilingData {
    int32_t bodyId, reverse, nCarry, nXs;   // nCarry=1/2, nXs=0/1/2（dtype 槽位由 host 校验，不入 tiling）
    int64_t length;   // 总逻辑步数 L（ys 步轴 / xs 前导轴）
    int64_t width;    // 浮点 carry 叶子元素数 W
    int64_t start;    // 本次 launch 的第一个逻辑步
    int64_t count;    // 本次 launch 步数（整循环 = L，逐步 = 1）
};
```

逐步模式（`scan`）下 Python 层做 L 次 launch，用两份 carry 设备缓冲轮换，`start=t`、`count=1`；
整循环模式（`scan_compiled`）一次 launch，`start=0`、`count=L`。两种模式的输出必须逐叶子一致
（门禁会互比 eager 与 jit）。

### 3.3 支持硬件

* 目标芯片：Ascend 910B1（Atlas A2 系列），架构 `dav-2201`；编译选项 `--npu-arch=dav-2201`；
* 计算单元：kernel 只使用 **AIV 向量核**（`Add/Adds/Muls/Mul` 与 `DataCopyPad`），不使用 AIC；
  `blockNum` 恒为 1，不做多核切分；
* 设备形态：单卡（device 0），单进程；多卡/多设备不涉及；
* 运行环境：CANN 9.2.0-beta.2 + bisheng + ACL；JAX/JAXLIB 0.8.1 的 `xla-npu` 插件用于 trace/AD 路径。

### 3.4 算子约束与限制

| 约束 | 范围 / 行为 |
|---|---|
| dtype | `carry`：仅 float32，pytree 额外允许 1 个 int32 标量 counter（非法 carry dtype → `dtype_mismatch`）；`xs`：float32。`xs_dtype` 未参与校验：`xs=int32` 会通过 Python 层并被 kernel 按 float32 读取（未定义行为，不属支持范围）；其它不支持 dtype 仍由 dtype 码拒绝 |
| carry 浮点宽度 | `1 ≤ W ≤ 4096`（`SCAN_MAX_WIDTH`）；越界 → `.so` rc=2 / Python `invalid_argument` |
| carry 结构 | 1 个 float32 叶子，或 `(int32 标量, float32[W])` pytree；每步结构/shape/dtype 不变 |
| `length` | `≥ 1` 且 trace 期静态；`xs=None` 时必填；`length=0` 不支持 |
| `unroll` | 仅 1 |
| `y` 形状 | 取 carry 中首个 float32 叶子的形状；pytree `y`、异构 `xs` 叶子不支持 |
| 设备 | 单卡 device 0；多设备 / 跨设备不支持 |
| 自动微分 | `.so` 无 AD；JAX trace 路径在当前 `xla-npu` 上显式 unsupported（不记 PASS） |
| 并发 | 每次调用独立 ACL stream；调用方负责输入生产与输出消费的同步（`.so` 返回前同步自己的 stream） |
| 性能前提 | 单核、W ≤ 4096；本批只记录同设备同镜像的实测值，不设加速比门槛 |

---

## 4. 特性交叉分析

### 4.1 混合精度

**部分涉及**：pytree carry 的 `int32` counter 与 `float32` accumulator 逐叶子使用独立 dtype 码
（`carry_dtype[8]`），不做全局 dtype 推断；kernel 对 int32 counter 在标量寄存器上自增、写回时
`Duplicate` 到 UB，float32 叶子走向量浮点运算，两者不混算。
**不涉及** float16 / bfloat16 / float64：本批 dtype 仅 `{float32,int32}`，其它直接拒绝。
不存在隐式类型提升或跨 dtype 转换点。

### 4.2 图模式

**不涉及**：本批是 Ascend C 直调算子，不是 GEIR/ACLNN 注册算子，不进入图引擎，也没有图内 shape/dtype
推导、图融合或属性传递。JAX 侧的 `jit` 是候选内部的整循环入口（`scan_compiled`）或 trace 路径的 `jax.jit`，
与昇腾图引擎不是同一概念。

### 4.3 动态 shape

**不支持**：`length` 在 trace 期静态确定，carry 的 shape/dtype 每步不变；每个新 shape 都需要重新 trace，
tiling 在 host 侧按当次 `length/width` 生成，不做运行期动态维度适配。本批所有用例为静态 shape。

### 4.4 其他特性

| 交叉维度 | 组合 | 结论 / 依据 |
|---|---|---|
| `reverse` × `pytree carry` | body 2 + reverse | 支持；kernel 用 `length-1-t` 取步、`ys[t]` 原序写回（补充用例 `scan_sup_rev_pytree_len6`） |
| `reverse` × `xs=None` | body 8 + reverse | 支持；索引变换对 `xs=None` 无影响（`scan_sup_len1_reverse_f32` 覆盖 length=1 边界） |
| `length=1` × eager/jit | body 1 | 支持；`abi_calls` 分别为 1 / 1（eager L=1 恰好等于整循环） |
| `xs` pytree × carry 单叶子 | body 7 | 支持；两个 `f32[L,W]` 叶子分别走 `xs[0]`/`xs[1]`，契约要求同构 |
| `xs=None` × 显式 `length` | body 8 | 支持；必须显式给 `length`，否则 `length_mismatch` |
| `int32` carry × `float32` carry | body 2 | 支持；逐叶子 dtype 码 + int32 counter 走 UB 单元素队列、以标量持有（无独立 `counterBuf`） |
| NaN / ±Inf × reverse / pytree | 补充用例 `scan_sup_naninf_len8` | 支持；kernel 逐元素算术，NaN/Inf 传播与 CPU 一致，判据逐元素精确比对非有限位 |
| `eager` × `jit` | 所有可执行 body | 必须输出一致；门禁对同用例两模式做互比（`_consistency_problems`） |
| AD × `.so` | body 3 | 不交叉：`.so` 无 AD，AD 一律 trace（`reason=so_no_autodiff`），当前后端显式 unsupported |
| 拒绝 × eager/jit | body 4/5/6 | 都在 device 前拒绝，`abi_calls=0`，不写 `outputs.npz` |
| 多流 × 单核 | torch / numpy 两族 | 支持顺序调用；`.so` 自建 stream 并在返回前同步；并发调用未验证 |
| 多设备 × carry 常驻 UB | — | 不支持（本批单卡 device 0） |

---

## 5. 可维可测分析

### 5.1 精度标准

* **oracle**：同版本 JAX CPU（JAX/JAXLIB 0.8.1），`scripts/golden.py` + `golden_cli.py` 独立子进程，
  设 `JAX_PLATFORMS=cpu`；禁止用候选 NPU 结果当参考；
* **判据档**：`criteria/check.py`（`ops-precision-standard@1`）；
* **容差**：float32 `rtol=2^-10`（≈9.77e-4）、`atol=2^-16`（≈1.53e-5）、`matched_ratio ≥ 0.99`、
  `max_abs_error ≤ 1e-2`；int32 逐元素精确相等；
* **结构与类型**：树结构、叶子数、dtype、shape 完全一致（0 容差）；
* **特殊值**：NaN/±Inf 位置与符号逐元素精确一致，有限子集再走浮点容差；
* **设备守卫**：逐叶子必须位于非 CPU 设备；
* **异常输入**：`scan_bad_carry_shape` 在 trace 期拒绝，要求 `rc=3` + `carry_shape_mismatch` +
  `abi_calls=0` 且不写 `outputs.npz`；
* **AD**：`scan_jvp_vjp` 允许「与 CPU 一致」或「显式 unsupported」；本版本为显式 unsupported
  （jvp/vjp 各 3 次同码，`path=trace`/`abi_calls=0`/`reason=so_no_autodiff`/`backend=npu`，
  附后端原文），不记 PASS。

### 5.2 性能标准

* **基线**：不设 CPU / torch 基线（`baseline: null`、`comparison: same_device_only`）；CPU 延迟不是门槛；
* **测量口径**：`scripts/nm.sh --op jax_lax_scan --action benchmark` → `tools/run_benchmark.py` →
  `scripts/scan_bench.py`；设备侧 `msopprof` 的 `Task Duration(us)`；blocking JIT 稳态，
  `warm_up=3`、`invocations=10`、median、`block_until_ready`（读 `benchmark.json` 的 `measure`）；
* **代表用例**：`scan_len128_width256_f32`、`scan_len1024_width256_f32`（**primary**，L=1024、W=256）；
* **目标**：仅记录，不设硬性加速比；本节点未执行 benchmark，**不宣称性能结论**，
  采集结果由 PM 在整轮退出后落盘。

### 5.3 兼容性分析

* **接口兼容性**：Python 签名与 `jax.lax.scan(f, init, xs, length, reverse)` 对齐（额外 `unroll` 仅接受 1）；
  `scan_compiled` 是测试/门禁的编译态入口，不是第二个公开 API；CLI 协议固定
  `--case-json/--inputs/--mode/--outdir`；拒绝码与 `.so` 返回值、Python 异常文本同表；
* **数据类型兼容性**：仅 `{float32,int32}`；无 f16/bf16/f64，无隐式提升；pytree 叶子逐 dtype 处理；
* **硬件形态兼容性**：仅 910B1（`dav-2201`）单卡；单核 `blockNum=1`，不随 AIV 核数扩展；
  多设备/多卡未实现；不同芯片需重新编译并复核 UB 预算。

### 5.4 可观测性与可测试性

* 可执行文件 stdout 首行固定 `EXEC op=... kernel=... arch=dav-2201 mode=... body=... body_id=...
  family=... path=... exec_mode=... abi_calls=... backend=...`；缺失即判 `INVALID`；
* `meta.json`（`schema=opx-scan-candidate/1`）逐叶子记录 `path/dtype/shape/platforms/device`，
  并带 `inputs_sha256` / `outputs_sha256` / `jax` / `jaxlib` / `error`；
* `_last_dispatch` 暴露路由结果，供 pytest 断言 `path/exec_mode/abi_calls/reason/backend/body_id`；
* `.so` 失败路径统一写 `code: detail` 到调用方提供的 `err` 缓冲；
* 路由函数 `_plan_dispatch` 纯函数、不执行 device，可单测；
* host driver 与四动作 runner 共用同一 `case.json`/`inputs.npz`/判据，避免"两套口径"；
* 拒绝 / 显式 unsupported / 数值 PASS 三态分离，`UNSUPPORTED` 不记 PASS；
* AD 判定要求 3 次同码 + 后端原文 + `path=trace`/`abi_calls=0`/非 CPU backend。

### 5.5 可维护性

* kernel / tiling / ABI / Python 接口分层清晰，body 语义集中在 `StepOnce` 的单一 `if-else` 链与
  `SCAN_BODY_*` 常量，新增 body 只需加常量与一个分支；
* 工程布局与 `ops/elementwise_sub` 一致（CMake 双 target、`op_host/*.asc` 导出 `extern "C"`），
  降低后续算子复用成本；
* 已知限制（README §6）与镜像差异（#136/#145 无 torch_npu）在交付记录中显式列出；
  torch-family 另有 #80 设备 0 补充真机证据（5 条数值用例 × eager/jit + length=2 回归，
  `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch/scan_correctness.json`），不把 JAX 或
  numpy host-driver 结果冒充 torch 接口通过。

### 5.6 验证口径

以 `tools/run_cases.py`（门禁 7 条 / 13 次）为权威，`tests/test_jax_lax_scan.py`（pytest 19 条 / 33 次：
冻结 7 + 基准 2 + 补充 6 + 白盒 4）为交付件入口；判据统一 `criteria/check.py`，oracle 统一 CPU 子进程
`golden_cli.py`。节点 4 修复后的代码/测试版本真机证据（快照 `6d660cd0…`，81 文件；节点 5 仅改
README 与本文档，代码/测试哈希未变）：build #73、smoke #74、test #76 均 `exit=0` 且 gate `pass`；
门禁 7 条 / 13 次 = 6 PASS + 1 显式 UNSUPPORTED；pytest 19 条 / 33 次 = 18 PASS + 1 UNSUPPORTED
（`32 passed, 1 xfailed`，`tests=33 failures=0 errors=0`）；白盒设备 4 条 / 8 mode 全 PASS。
原始日志、逐用例结果与哈希见 `.cannbot/jax_lax_scan/workflow1/4-证据-r1/`、
`4-测试执行记录.md` 与 `4-验收报告.md`。

---

## 附录 A：测试用例要点

### A.1 精度用例（冻结 `cases/jax_lax_scan/cases.json`，7 条 / 13 次）

| Case ID | 分组 | shape / 结构 | dtype | 期望 | 说明 |
|---|---|---|---|---|---|
| `scan_sum_len8_f32` | 常规 | `xs[8,16]` | float32 | match_cpu（eager/jit） | 加法 carry 与 stacked ys |
| `scan_reverse_len7_f32` | 反向扫描 | `xs[7,5]`，`reverse=true` | float32 | match_cpu（eager/jit） | 非 2 次幂长度，确认 reverse 输出顺序 |
| `scan_pytree_carry` | pytree | carry `(int32 counter, f32[4])`，length=6 | float32/int32 | match_cpu（eager/jit） | 结构、叶子 shape/dtype 与数值一致 |
| `scan_pytree_xs` | pytree | xs `dict(add[6,4], scale[6,4])` | float32 | match_cpu（eager/jit） | xs 为 pytree，body 7 |
| `scan_xs_none_length5` | 可选输入 | carry `f32[4]`，`xs=None`，length=5 | float32 | match_cpu（eager/jit） | body 8 增量循环 |
| `scan_jvp_vjp` | 自动微分 | `xs[5,4]`，body 3 | float32 | match_cpu_or_explicit_unsupported | 本版本显式 unsupported，不记 PASS |
| `scan_bad_carry_shape` | 异常 | body 4 改变 carry shape | — | reject + `carry_shape_mismatch` | rc=3，`abi_calls=0` |

> 另有 `tests/jax_lax_scan_supplement_cases.json`（6 条补充）与
> `tests/jax_lax_scan_whitebox_cases.json`（4 条白盒）在同一 pytest 入口执行，**不属冻结
> `cases.json`**，也不改变门禁 7 条 / 13 次的结论。

### A.2 性能用例（`cases/jax_lax_scan/benchmark.json`，2 条）

| Case ID | length | width | dtype | 主用例 | 必需 | 说明 |
|---|---|---|---|---|---|---|
| `scan_len128_width256_f32` | 128 | 256 | float32 | 否 | 是 | 中等长度稳态 |
| `scan_len1024_width256_f32` | 1024 | 256 | float32 | **是** | 是 | primary，长循环稳态 |

---

## 附录 B：参考资料

* JAX 0.8.1 `jax.lax.scan` 官方语义（`reverse` / `length` / `xs=None`）；
* `tasks/jax_lax_scan/spec/spec.json`（spec v1）与 `tasks/jax_lax_scan/notes/clarifications.md`；
* `0-黑盒测试设计.md` §1.3（候选接口锁定）与 `3-白盒测试设计.md`（源码分支覆盖）；
* 本仓 `ops/elementwise_sub` 的 Ascend C 直调工程布局与 `docs/templates/算子设计文档模板.md`。
