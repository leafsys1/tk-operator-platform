# jax_lax_while_loop 算子设计文档（AscendC 实现）

| 项目 | 内容 |
|---|---|
| 算子名称 | `jax_lax_while_loop` |
| 目标芯片 / 架构 | Ascend 910B1 / `dav-2201` |
| 目标镜像 | #145 `opx-xla-npu:jax081-cann920-910b-test`（CANN 9.2.0-beta.2、bisheng、JAX/JAXLIB 0.8.1、`xla-npu/AFIR`） |
| 文档版本 | v1.0 |
| 作者 / 日期 | 节点 1（文档准备）/ 2026-10-29 |

> 路径约定：文中 `.cannbot/jax_lax_while_loop/workflow6/` 下的 `0-*`～`1-*` 报告、探针与证据目录均为
> 本轮工作目录的过程证据，不随交付包分发；交付件为 `ops/jax_lax_while_loop/`（含
> `ops/jax_lax_while_loop/README.md`）、`tests/test_jax_lax_while_loop.py` 与本文件。
> 本文件与 `README.md` 均以修复后最终代码为准，同步 node 0 修复记录与四动作证据。
>
> 真相源：`tasks/jax_lax_while_loop/spec/spec.json`（spec v1，`spec_version_id=7`；
> `_meta.sha256=53be437582588ba1f90beb5baca9bbbece776976a54d659ad7ed8a76115f9b32`）；冻结用例
> `cases/jax_lax_while_loop/{cases,benchmark}.json`（`suite_ref=cases@288876910a5e2843`，
> `cases.json` 文件 sha256 `89e5376f7d31f34e6594e6284fc572b93b67d14b5b604996f0557bcd0b0f81`）；
> 判据 `criteria/check.py`（`ops-precision-standard@1`）。

### 修订记录

| 版本 | 日期 | 修改人 | 修改说明 |
|---|---|---|---|
| v1.0 | 2026-10-29 | 节点 1 | 首版：按 spec v1、冻结用例、当前 operator-v1 `cb954eb1…`（13 文件）与修复后最终代码（`repair_file` sha256 `e0037bc1…`）编写；同步 node 0 修复记录与 build #251 / smoke #252 / test #253 / benchmark #254 四动作证据；未修改代码、测试、用例、判据或阈值 |

---

## 1. 需求背景

### 1.1 需求来源

JAX 模型在昇腾上落地时，`jax.lax.while_loop` 是动态迭代次数的结构化控制流原语：它以
`cond_fun(carry)` 的标量布尔结果决定是否继续，以 `body_fun(carry)` 推进状态，直到条件为假。
它覆盖「while 循环」与「do-while / 迭代算法」两类典型形态，是 RNN 展开、迭代求解、早停训练
循环等场景的基础算子。本任务由控制台任务 #9 发起，验收方为控制台「四动作 + 判据档」门禁、
verifier 与用户；node 1 的职责是以修复后最终代码与证据补全 `README.md` / `design.md`，并如实
记录能力边界与复现入口。

冻结输入与身份：

- 规格：spec v1，`spec_version_id=7`；
- 正确性用例：16 条 required（黑盒 mode 执行 51 次）；性能用例：2 条；
- 判据：`ops-precision-standard@1`；oracle：同版本 JAX 0.8.1 CPU；
- 目标硬件：Ascend 910B1；目标镜像 #145；真机设备 5（容器内 `device_id=0`）、独占锁；
- 代码与用例证据作用域：`operator-v1`（只哈希 `ops/jax_lax_while_loop/` 下非 `.md` 文件，
  本版 13 文件；`cases/jax_lax_while_loop/` 单独哈希）。

本批不修改已确认接口、cases、criteria、golden 与阈值；`README.md`/`design.md` 等文档变化不改变
operator-v1 快照。

### 1.2 业务背景与使用场景

`lax.while_loop` 的典型调用形态是「同一段 `body_fun` 反复执行，直到 `cond_fun` 判定停止」。
在本批的候选（Python → host C ABI → AscendC kernel）实现中：

- **cond**：归一化为返回 **0-d scalar bool** 的谓词；迭代在设备侧逐阶段求值；
- **body**：被 `jax.make_jaxpr` 追踪并线性化，每个 carry 叶子映射为仿射线性表达式
  `out = const + Σ scale_i * (reduce_i ? Σ src_i : src_i)`（最多 2 项，含整叶求和 / `mean`）；
- **host_step 路由**：一次宿主入口推进一个阶段（stage 0 = 只求值确认 cond；stage 1 = 执行一次
  body 再求值 cond），host 在 Python 侧循环直到 cond 为假或安全上限触发；
- **自动微分**：正向 JVP 只对浮点 carry 叶求切向量；reverse-mode VJP / `grad` 不是 v1 强制能力。

本批的交付边界是**算子直调 + pytest/CLI 入口**，不注册 ACLNN/GEIR、不做图内子图融合或
shape 推导；典型调用为算子直调，不是 XLA 图内节点。`jit` 路径（`while_loop_compiled`）复用
「已线性化并缓存的设备计划」，不是把 Python `lax.while_loop` 交给 XLA jit。

### 1.3 算子数学语义

对 callable `cond_fun`、`body_fun` 与 pytree `init_carry`：

```
while_loop(cond_fun, body_fun, init_carry) =
    carry = init_carry
    while cond_fun(carry) is scalar True:
        carry = body_fun(carry)
    return carry
```

语义与 JAX 0.8.1 `jax.lax.while_loop` 对齐：

- `cond_fun` 必须返回 **0-d `bool`**；非标量或非 bool → `INVALID_CONDITION`；
- `body_fun` 必须返回与输入 carry **完全相同**的 pytree 结构，每叶 shape/dtype/`weak_type` 不变：
  结构漂移 → `CARRY_STRUCTURE_MISMATCH`，shape 漂移 → `CARRY_SHAPE_MISMATCH`，
  dtype 漂移 → `CARRY_DTYPE_MISMATCH`；
- 支持运行期动态迭代次数（0 次 / 1 次 / 多次）；0 次循环直接返回 `init_carry`；
- 安全迭代上限（`safety_limit` 参数 / 设备安全阀）触发 → `ITERATION_LIMIT_EXCEEDED`
  （运行期设备安全阀，不是参数校验拒绝）；
- 正常终止时 `body_hits = iterations = n`、`cond_hits = n + 1`；安全阀触发时三者都等于 `limit`；
- 错误归一化保留上游 JAX 异常的类型与单行摘要（`upstream_exception_type` /
  `upstream_message` / `upstream_message_sha256`）；
- 不做广播；不原地修改输入、输出不与输入别名；每次返回新 pytree / 新数组；
- NaN / ±Inf 逐元素传播，与 CPU golden 的非有限位逐元素精确比对；有限子集再走浮点容差；
- 规格将 `layout` 登记为「随输入张量默认布局」；实现按连续 ND / C-contiguous 叶子处理，
  不引入额外的设备地址对齐假设。

### 1.4 历史 TBE 实现与现状分析

无历史 TBE 实现，首版。参考依据：

- JAX 0.8.1 `jax.lax.while_loop` 官方语义（标量 cond、pytree 连续性、weak type、动态迭代）；
- node 0 修复记录 `0-修复记录.md` 与验收报告 `0-验收报告.md`（int32 张量叶截断缺陷的根因与修复）；
- 本仓 `ops/jax_lax_cond` / `ops/jax_lax_scan` 已验证的 Ascend C 直调工程布局（kernel 分段、
  tiling、pyapi、C ABI、四动作入口、CMake 双 target、设备计数器/token 证据）。

workflow1–5 已完成规格对齐、黑盒/白盒测试设计与接入、以及缺陷暴露（workflow5 在 run
`runs/r20261029-161341` 真实暴露 6 项 int32 张量叶白盒失败）；本版为修复后正式实现，
未改动用例/golden/阈值/spec/checker。

---

## 2. 需求分析

### 2.1 外部组件依赖

| 组件 | 版本 / 用途 | 说明 |
|---|---|---|
| JAX / JAXLIB | 0.8.1 | 语义对齐目标；`jax.make_jaxpr` 追踪 `body_fun`/`cond_fun` 生成 jaxpr，作为线性化计划的解释输入 |
| CANN / bisheng | CANN 9.2.0-beta.2 | AscendC kernel 编译与设备运行 |
| ctypes + `libjax_lax_while_loop_host.so` | ABI `opx-while-loop-python-abi/1` | Python 包装通过 C ABI 下发达成 host→device 调用 |
| `criteria/check.py` | `ops-precision-standard@1` | 数值判定（浮点容差 / 整型精确 / 非有限位规则 / 结构一致） |
| JAX 0.8.1 CPU | oracle | 独立子进程产出 golden，与候选在同一份场景上对比 |

主机侧无 PyTorch/CANN 内置算子代算；数值计算全部由 kernel 在 NPU 上完成，包装层不做
transpose/contiguous/cast 等实质性数据变换。

### 2.2 内部适配模块

| 模块 | 路径 | 职责 |
|---|---|---|
| 四动作入口 | `op_host/jax_lax_while_loop.asc` | CLI 协议校验，转交 case.json / inputs.npz 协议 |
| C ABI 头 | `op_host/jax_lax_while_loop_abi.h` | ctypes 对齐的 C 结构：`JaxLaxWhileLoopArgs` |
| pyapi | `op_host/jax_lax_while_loop_pyapi.asc` | `extern "C"` C ABI：tiling 组装、launch、D2H 读回计数器 |
| kernel | `op_kernel/jax_lax_while_loop_kernel.asc` | AscendC kernel：body / cond / JVP / 设备计数器 |
| tiling | `op_kernel/jax_lax_while_loop_tiling.h` | kernel / host / pyapi 共享 `WhileTilingData` |
| Python | `python/jax_lax_while_loop.py` | 公开接口 `while_loop` / `while_loop_compiled`、jaxpr 解释、plan 推导、证据封装 |
| CLI driver | `python/jax_lax_while_loop_cli.py` | 用例协议驱动（默认不直接对外） |
| oracle / runner 适配 | `scripts/` | 独立 CPU golden、输入生成、runner 适配（测试工程节点） |

### 2.3 需求模块设计

按 klib 分层拆分为计划推导（Python）、执行驱动（Python + pyapi）、数值计算（kernel）三层：

1. **计划推导（Python）**：`_interpret_body` / `_interpret_cond` 把 jaxpr 解释成 body 线性表达式
   与 cond 比较形态；`_derive_plan` 生成 Tiling 计划，`_plan_id` 取计划的 sha256 前 16 位
   （只依赖计算计划、不含运行期 limit 值，故同一计划在不同 limit 下 `plan_id` 稳定）。
2. **执行驱动（Python + pyapi）**：`host_step` 逐阶段推进；每次推进 launch 一次 kernel；
   `abi_calls = 1 + iterations`（安全阀触发时为 `1 + limit`）；计数器全部 kernel 写设备内存，
   host 只做 D2H 读回。
3. **数值计算（kernel）**：body 的 `ComputeBody`（浮点叶 `ComputeFloatLeaf` / 整型叶
   `ComputeIntLeaf`）、cond 的 `EvalCond`、JVP 的切线叶，以及证据计数器写入。

### 2.4 算子原型

#### 2.4.1 输入

- `cond_fun`：`carry → scalar bool`，必须返回 0-d `bool`；
- `body_fun`：`carry → carry`，返回与输入同结构 / 同 shape / 同 dtype / 同 `weak_type` 的 pytree；
- `init_carry`：任意 JAX pytree（叶子 dtype ∈ {int32, float32}）；
- `安全迭代上限`：可选 `safety_limit`（设备安全阀，默认值见实现，触发即 `ITERATION_LIMIT_EXCEEDED`）。

#### 2.4.2 输出

- `final carry`：与 `init_carry` 结构一致的 pytree，每叶 shape / dtype / `weak_type` 精确一致。

#### 2.4.3 属性 / 参数

| 参数 | 取值 / 含义 |
|---|---|
| `safety_limit` | 安全迭代上限，触发安全阀 |
| `mode`（编译态入口） | `primal` 返回 carry；`jvp` 返回 `(primal, tangent)` |
| `tangent`（JVP） | 浮点 carry 叶的切向量；整型叶为 `float0` 零尺寸标记 |
| `jit`（编译态入口） | `True` 复用已线性化并缓存的设备计划 |

#### 2.4.4 推导规则

- 一次调用最多 8 个 carry 叶子（`WHILE_MAX_LEAVES=8`，pytree 展开后），超限按 `invalid_argument`
  拒绝；
- body 叶子支持 `add / sub / mul(常量) / div(常量) / reduce_sum / mean` 与广播，逐元素项要求
  源叶子 numel 与输出一致，归约项源叶子 numel ≤ 256（`WHILE_SCAN_MAX`）；不认识的 primitive 显式
  报 `UNSUPPORTED_PLAN`；
- cond 归一化为 `leaf <op> leaf|const`、`all(leaf <op> const)` 或恒真；`all()` 的叶子 numel ≤ 256；
- 每个输出叶 type 由实际 dtype（int32 / float32）决定；逐叶独立 tiling。

---

## 3. 详细设计

### 3.1 使能方式

候选实现为 AscendC 直调算子：Python 通过 ctypes 加载 `libjax_lax_while_loop_host.so`，launch
AscendC kernel 完成 body/cond/JVP 数值与设备计数器；四动作门禁走
`scripts/nm.sh --op jax_lax_while_loop --action <build|smoke|test|benchmark>`（`--npu-arch=dav-2201`）。
不注册 GEIR/ACLNN，不在框架图内接算子。

### 3.2 总体设计方案

#### 3.2.1 Host 侧

- **pyapi**：`extern "C"` C ABI 组装 `WhileTilingData` 并下发，返回证据计数器；一次调用按
  `stage` 只推进一个阶段：
  - `WHILE_STAGE_COND_ONLY`（stage 0）：只求值一次 cond（`cond_hits += 1`）；
  - `WHILE_STAGE_BODY_COND`（stage 1）：执行一次 body，再求值一次 cond；达到安全上限时不再求值
    cond（`body_hits += 1`，必要时 `cond_hits += 1`）。
- **Python**：`_validate` 前置校验（cond 0-d bool、pytree 连续性、dtype/shape 支持表）；
  `_derive_plan` 生成 plan；`_execute` 循环调用 `host_step`，读到 `cond_result = false` 或
  `limit_flag = 1` 即返回；下载最终 carry 并用 `stable_leaf_meta` 还原 `weak_type`。
- **证据封装**：`last_dispatch` 记录 `path/route/abi_calls/iterations/cond_hits/body_hits/
  runtime_token/counter_source/plan_id/plan_cache/compile_count/leaves/tree`。

#### 3.2.2 Kernel 侧

- **UB / buffer**：`WHILE_TILE_ELEMS=8192` float（32KB）的 `in1/in2/out` 各一个 tile、
  1KB 标量区（<192KB）；计数器区 `WHILE_COUNTER_SLOTS=8` 槽位。
- **body（`ComputeBody`）**：按 `WhileTilingData` 的每叶 `outSrc1/outScale1/outRed1` 与
  `outSrc2/outScale2/outRed2/outConst` 计算仿射线性表达式。
  - `ComputeFloatLeaf`（float32 叶）：以 `WHILE_TILE_ELEMS` 分块，`count=min(remain, WHILE_TILE_ELEMS)`，
    逐 tile 搬运源到 UB，逐元素计算后按 `{1, count * sizeof(float), 0, 0}` 写回；归约项（`Red>0`）
    在标量区求和后广播。
  - `ComputeIntLeaf`（int32 叶）：**与 float 路径同一 tile 循环契约**（node 0 修复后一致）。
    element-wise 项按 `count * sizeof(int32_t)` 读入 `x1/x2`，逐元素 `GetValue → RoundToInt →
    SetValue`；纯 reduce / 常量叶整块广播 `cEff`（`RoundToInt(常量 + Σ scale_i * reduce_i)`）；
    每 tile 以 `{1, count * sizeof(int32_t), 0, 0}` 全量写回 `outI_[j][off]`，并以
    `PipeBarrier<PIPE_ALL>` 同步。修复前该路径只用 `ReadScalarInt` 读首元素、只写 4 字节，
    导致 int32 张量叶截断。
- **cond（`EvalCond`）**：标量比较走 `ReadScalarInt`/`ReadScalarFloat`；`all(leaf <op> const)`
  在单核把叶子搬进标量区逐元素归约比较，写成布尔结果。
- **JVP**：`hasTangent=1` 时逐浮点叶计算切线；整型叶切向量不生成实际数值（`float0` 由 Python
  侧以结构标记返回）。
- **计数器 / token**：`WriteCounters` 写 `bodyHits / condHits / limitFlag / runtimeToken /
  condResult / iterations` 到设备内存，host 经 `ReadCounters` D2H 读回；`counter_source=kernel`
  保证设备真实性。

### 3.3 支持硬件

- Ascend 910B1，`dav-2201`；镜像 #145（CANN 9.2.0-beta.2、bisheng、JAX 0.8.1）。
- 单核执行；UB 用量约 97KB（3 × 32KB tile + 1KB 标量区 + 计数器区），小于 192KB，无 workspace
  外扩需求。

### 3.4 算子约束与限制

1. body 仅仿射线性计划；非线性 body（`sin`、`c*c` 等）与 slice/concatenate → `UNSUPPORTED_PLAN`；
2. cond 标量比较（int32/float32）或 `all(tensor > const)`；`all()` 叶子 numel ≤ 256；多叶布尔
   组合、动态索引不在 v1 范围；
3. 归约项源叶子 numel ≤ 256（kernel 标量区）；
4. carry 叶子 dtype ∈ {int32, float32}；JVP 仅浮点叶；
5. `weak_type` 仅 0-d weak 叶精确还原；非 0-d weak 叶 → `UNSUPPORTED_WEAK_TYPE`；
6. reverse-mode VJP / `grad` 不提供（v1 非强制）；
7. 单核执行，不做多核切分 / 流水优化；性能仅基线；
8. pytree 叶子经 `numpy` 上传到设备缓冲区（I/O，非计算），大 carry 的 H2D 计入基线。

---

## 4. 特性交叉分析

### 4.1 混合精度

不涉及（v1 仅 int32 / float32 两档，无 fp16 / bf16 混合）。逐叶 dtype 独立处理，无跨叶转换。

### 4.2 图模式

部分涉及：`jit` 路径（`while_loop_compiled`）复用已线性化并缓存的设备计划（`plan_cache ∈
{miss, hit}`、`compile_count`），但数值与计数仍全部来自设备 kernel；不是 XLA 子图。本算子不注册
框架图内节点。

### 4.3 动态 shape

部分涉及：迭代次数是运行期动态的（0 / 1 / 多次，由 cond 结果决定），但 carry 叶子的 shape /
dtype / pytree 结构在循环内不变（size-stable carry）。运行期 shape 推导不在本批范围。

### 4.4 其他特性

- **自动微分**：JVP（forward）只对浮点 carry 叶；reverse-mode VJP / grad 不涉及（v1 非强制）。
- **多核 / 并行**：不涉及，单核。
- **设备异常**：`device_error`、`invalid_argument` 等运行错误记 `ERROR`，不计 PASS。

---

## 5. 可维可测分析

### 5.1 精度标准

- 浮点：`rtol=2**-10`、`atol=2**-16`、`matched_ratio>=0.99`、`max_abs_error<=1e-2`，逐输出叶子
  独立判定（`criteria/check.py`，`ops-precision-standard@1`）；非有限位逐元素精确；
- 整型：逐元素精确（修复后全量写出，无截断）；
- `float0` 切向量：dtype / shape / nbytes=0 结构精确；
- 结构 / shape / dtype / `weak_type`：精确相等，不进入容差；
- 设备真实性：`path=so`、`backend != cpu`、`platforms` 含 `npu`、原始整数 `abi_calls>=1`、
  `counter_source=kernel`、`route ∈ {device_loop, host_step}`、`device_id=0`、`runtime_token`
  非空；`path=trace`、CPU、模拟、`xfail`、`unsupported`、`NOT_RUN` 一律不计 PASS。

### 5.2 性能标准

只记录基线，不设硬阈值：`blocking` 执行、`warm_up=3`、`invocations=10`、`median`、
`block_until_ready`、同设备 / 镜像 / 规格 / 用例集，设备侧 Task Duration。
`host_step` 每次 body 需要一次宿主入口与一次 D2H 读回，短循环小 carry 主要受 ABI / 同步开销，
长循环大张量主要受 kernel tile 吞吐。单核执行，未见多核优化承诺（不在本轮范围）。

### 5.3 兼容性分析

- 环境：Ascend 910B1 / `dav-2201`、镜像 #145、JAX 0.8.1；
- 接口：`while_loop(cond_fun, body_fun, init_carry, *, safety_limit=None, tangent=None)` 与
  `while_loop_compiled(cond_fun, body_fun, init_carry, *, mode="primal", tangent=None,
  safety_limit=None, jit=True)`；
- 身份：`spec_version_id=7`、`suite_ref=cases@288876910a5e2843`、`criteria_ref=ops-precision-standard@1`、
  镜像 #145、宿主设备 5 / 容器设备 0。

### 5.4 可观测性与可测试性

- 每次调用通过 `last_dispatch` 暴露 `path/route/abi_calls/iterations/cond_hits/body_hits/
  runtime_token/counter_source/plan_id/plan_cache/compile_count/leaves/tree`；
- 设备计数器 / token 全部 kernel 写入设备内存，host 只做 D2H 读回，可被白盒证据断言；
- pytest 交付件 `tests/test_jax_lax_while_loop.py` 读 `cases.json`，用 Python 接口跑候选、
  `ops/jax_lax_while_loop/scripts/golden.py` 跑参考，按 `criteria/check.py` 同口径判定；
  拒绝用例必须匹配 `expect_reject_code`。

### 5.5 可维护性

- 单一 `WhileTilingData` 结构在 tiling.h 定义，kernel / host / pyapi 三方共享；
- `ComputeIntLeaf` 与 `ComputeFloatLeaf` 采用一致的 tile 循环与同步范式，降低后续维护分叉；
- `plan_id` 只依赖计算计划（与运行期 limit 解耦），`plan_cache` / `compile_count` 明确暴露缓存
  行为，便于排查「同计划不同 limit」问题；
- 上层代码只做 jaxpr 解释、pytree 校验、弱类型还原与证据封装，不代算数值，责任边界清晰。

### 5.6 验证口径

见附录 A 与 `README.md` §9：

| 动作 | run / 台账 | 结论 |
|---|---|---|
| build | `runs/r20261029-174551/` / #251 | BUILD_OK，exit=0 |
| smoke | `runs/r20261029-174641/` / #252 | SMOKE_VERDICT PASS（`while_one_iteration` eager） |
| test | `runs/r20261029-174734/` / #253 | TEST_VERDICT PASS（黑盒 51/51；pytest 138 passed；白盒 120/120） |
| benchmark | `runs/r20261029-180418/` / #254 | BENCH_VERDICT done（仅基线） |

---

## 附录 A：测试用例要点

### A.1 正确性用例（冻结 `cases/jax_lax_while_loop/cases.json`，16 条 / 黑盒 51 mode）

| case_group | case_id（要点） |
|---|---|
| 动态迭代 | `while_zero_iterations`、`while_one_iteration`、`while_dynamic_multi_iterations`（limit 2/7/13） |
| carry pytree | `while_scalar_carry`、`while_tuple_carry`、`while_nested_dict_carry` |
| jit | `while_jit_runtime_limit`（limit 0/1/8） |
| 自动微分 | `while_jvp_multi_iteration`、`while_jvp_zero_iteration` |
| 边界与规模 | `while_unaligned_tail_widths`（元素 17/33，bytes 68/132，均 `%32 != 0`）、`while_large_tensor`（f32[1024,256] × 12 次） |
| 执行证据 | `while_iteration_token_and_hits`（limit 0/1/7） |
| 错误分类 | `while_invalid_condition`（bool[1]/bool[2]/int32/float32）、`while_carry_structure_mismatch`、`while_carry_shape_or_dtype_mismatch`（2 变体）、`while_iteration_limit_exceeded`（safety_limit 64） |

黑盒逐 mode：eager 14 / jit 17 / jvp_eager 2 / jvp_jit 2 / reject 16 = 51；白盒
`tests/while_loop_correctness.json`：eager 57 / jit 37 / jvp_eager 4 / jvp_jit 4 / reject 18 = 120，
全部 PASS。node 0 修复翻转的 6 项 int32 白盒：`while_wb_int_sum_reduce` / `while_wb_int_two_reduce` /
`while_wb_int_two_elem_terms` × eager+jit。

### A.2 性能用例（`cases/jax_lax_while_loop/benchmark.json`，2 条）

| case_id | 描述 | 本版本实测（median） |
|---|---|---|
| `while_short_small_carry` | f32[16] × 4 次 | 4.910us |
| `while_long_large_tensor`（主用例） | f32[1024,256] × 64 次 | 4.890us |

口径：设备侧 Task Duration、10 样本 median、同卡同条件；`performance_target=null`、仅基线。

## 附录 B：参考资料

- spec v1：`tasks/jax_lax_while_loop/spec/spec.json`（`spec_version_id=7`）；
- 判据：`criteria/check.py`（`ops-precision-standard@1`）；
- 用例集：`cases/jax_lax_while_loop/{cases,benchmark}.json`（`cases@288876910a5e2843`）；
- node 0 修复记录：`.cannbot/jax_lax_while_loop/workflow6/0-修复记录.md`；
- node 0 测试执行记录：`.cannbot/jax_lax_while_loop/workflow6/0-测试执行记录.md`；
- node 0 验收报告：`.cannbot/jax_lax_while_loop/workflow6/0-验收报告.md`；
- workflow5 缺陷暴露证据：`runs/r20261029-161341/`；
- 同类算子文档：`ops/jax_lax_cond/README.md`、`docs/jax_lax_cond/design.md`、
  `ops/jax_lax_scan/README.md`、`docs/jax_lax_scan/design.md`；
- 交付物规范：控制台仓库 `docs/reference/deliverable-standards.md`（§4 README、§5 design 五段）。