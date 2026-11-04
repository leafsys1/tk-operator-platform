# jax_lax_cond 算子设计文档（AscendC 实现）

| 项目 | 内容 |
|---|---|
| 算子名称 | `jax_lax_cond` |
| 目标芯片 / 架构 | Ascend 910B1 / `dav-2201` |
| 目标镜像 | #145 `opx-xla-npu:jax081-cann920-910b-test`（CANN 9.2.0-beta.2、bisheng、JAX/JAXLIB 0.8.1、`xla-npu/AFIR`） |
| 文档版本 | v1.0 |
| 作者 / 日期 | 节点 5（文档准备）/ 2026-10-28 |

> 路径约定：文中 `.cannbot/jax_lax_cond/workflow3/` 下的 `0-*`～`5-*` 报告、探针与证据目录均为本轮
> 工作目录的过程证据，不随交付包分发；交付件为 `ops/jax_lax_cond/`、`tests/test_jax_lax_cond.py`、
> `ops/jax_lax_cond/README.md` 与本文件。
>
> 真相源：`tasks/jax_lax_cond/spec/spec.json`（spec v1，`spec_version_id=6`；
> `_meta.sha256=79556ac1659564a7f57763d3d4a50650ec5bbc6233caf04547a276de1684396f`，
> 文件 sha256 `3a08e86813f52b98adfa231f5c4158a6822049d0e74c3cf9e50e6574dbf7229b`）；冻结用例
> `cases/jax_lax_cond/{cases,benchmark}.json`（`suite_ref=cases@233778f456b4eb9b`，完整
> `233778f456b4eb9b06114cc6a66a8352caee5162ac33065cb85cf6f26c2b1fe8`）；判据
> `criteria/check.py`（`ops-precision-standard@1`）。

### 修订记录

| 版本 | 日期 | 修改人 | 修改说明 |
|---|---|---|---|
| v1.0 | 2026-10-28 | 节点 5 | 首版：按 spec v1、冻结用例、`0-黑盒测试设计.md` §1.3 契约、当前 operator-v1 `55b69ca6…`（13 文件）与修复后测试入口 `66a3d0a1…` 编写；同步节点 4 run #232 全量证据与 build #226 / smoke #227 / benchmark #229 复用证据；未修改代码、测试、用例、判据或阈值 |

---

## 1. 需求背景

### 1.1 需求来源

JAX 模型在昇腾上落地时，`jax.lax.cond` 是条件分支、分段函数、门控与迭代算法中的基础控制流原语：
它在 trace 阶段访问 true/false 两个分支并校验两侧输出结构，运行期只执行由标量布尔谓词选中的一支。
本任务由控制台任务 #10 发起，验收方为控制台「四动作 + 判据档」门禁、verifier 与用户。

冻结输入与身份：

- 规格：spec v1，`spec_version_id=6`；
- 正确性用例：16 条 required（49 个 mode 执行）；性能用例：2 条（4 个序列）；
- 判据：`ops-precision-standard@1`；oracle：同版本 JAX 0.8.1 CPU；
- 目标硬件：Ascend 910B1；目标镜像 #145；真机设备 5（容器内 `device_id=0`）、独占锁；
- 代码与用例证据作用域：`operator-v1`（只哈希 `ops/jax_lax_cond/**` 非 `.md` 文件，本版 13 文件）。

本批不修改已确认接口、cases、criteria、golden 与阈值；`README.md`/`design.md` 等文档变化不改变
operator-v1 快照。

### 1.2 业务背景与使用场景

`lax.cond` 的典型调用形态是「同一段程序按运行期标量布尔值走两条分支之一」，两条分支通常具有
相同的 pytree 结构，但计算不同。本批在 Ascend 侧交付两种可执行路径：

- **numpy family（门禁数值路径）**：operand 全为 `numpy.ndarray`、分支为本仓已注册原语时，候选把
  选中 `branch_id` 与逐输出叶子的 `(源叶子, 原语, 步长)` 计划交给 `libjax_lax_cond_host.so` →
  AscendC kernel；kernel 写设备侧 token/counters，host 通过 C ABI 的 D2H 读回；
- **jax family（trace / 自动微分路径）**：operand 含 JAX tracer 或全为 `jax.Array` 时，候选走 JAX
  在 `xla-npu` 后端的 trace；两分支都真实 trace，运行期只调用选中分支。`grad` 全部由 JAX 反向模式
  AD 产生。

本批的交付边界是**算子直调 + pytest/CLI 入口**，不注册 ACLNN/GEIR、不做图内 shape/dtype 推导或
融合子图；典型调用为算子直调，不是图内节点。

### 1.3 算子数学语义

对严格标量布尔谓词 `pred`、两个 callable 与 pytree operands：

```
cond(pred, true_fun, false_fun, *operands) =
    true_fun(*operands)    if pred is True
    false_fun(*operands)   otherwise
```

语义与 JAX 0.8.1 `jax.lax.cond` 对齐：

- `pred` 必须是**严格 scalar bool**（Python `bool`、`np.bool_`、0-d bool 数组，或可静态求值的
  0-d `jax.Array(bool)`）；`bool[1]`、`bool[2]`、整型 `0/1` 均不做隐式提升。整型 `0/1` 按已确认
  spec v1 / PM 最终确认归一化为 `INVALID_PREDICATE`；CPU JAX 0.8.1 对它们按真值接受、不抛错，
  该原始行为单独留证，不伪造上游异常；
- **两分支都参与 trace**：每次调用都真实访问 `true_fun` 与 `false_fun`，比较两者输出的 pytree
  结构、逐叶 shape、dtype 与 `weak_type`；JAX 负责抛出上游校验异常；
- **运行期只执行选中分支**：numpy family 只下发一次选中 `branch_id` 的 kernel；jax family 只调用
  选中分支的 callable（grad 路径另见 §3.2.1）；
- 结构/shape/dtype 不一致分别归一化为 `BRANCH_STRUCTURE_MISMATCH` /
  `BRANCH_SHAPE_MISMATCH` / `BRANCH_DTYPE_MISMATCH`，并保留上游异常类型与消息 sha256；
- 不做广播；不原地修改输入、输出不与输入别名；每次返回新 pytree/新数组；
- NaN / ±Inf 逐元素传播，与 CPU golden 的非有限位逐元素精确比对；有限子集再走浮点容差；
- `pred` 不可导；grad 只对浮点 operand 求导；`weak_type` 与 JAX 0.8.1 语义严格对齐；
- 规格将 `layout` 登记为 `UNKNOWN`；实现按连续 ND / C-contiguous 叶子处理，不引入额外的设备
  地址对齐假设。

### 1.4 历史 TBE 实现与现状分析

无历史 TBE 实现，首版。参考依据：

- JAX 0.8.1 `jax.lax.cond` 官方语义（两分支 trace、标量谓词、pytree 一致性、weak type）；
- `0-黑盒测试设计.md` §1.3 的候选接口契约（Python 入口、mode 可执行定义、CLI 协议、host ABI
  基线、设备守卫、分支证据、错误归一化）；
- 本仓 `ops/jax_lax_scan` / `ops/elementwise_sub` 已验证的 Ascend C 直调工程布局（kernel 分段、
  tiling、pyapi、C ABI、四动作入口、CMake 双 target）。

workflow2 的穿刺已验证「扁平布局 VJP、真实 cotangent、0-d `weak_type`、独立进程隔离、设备
token/counters、数据派生 predicate」路线；本版为正式实现，未 import 或直接拷贝穿刺文件，
`ops/` 中不存在对 `.cannbot/jax_lax_cond/workflow2/0-穿刺/` 的依赖。

---

## 2. 需求分析

### 2.1 外部组件依赖

| 组件 | 版本 / 位置 | 用途 |
|---|---|---|
| CANN toolkit | 9.2.0-beta.2（容器内 `/usr/local/Ascend/cann-9.2.0-beta.2`） | bisheng / ccec / ACL / `find_package(ASC)` |
| 编译平台 | `dav-2201`（`--npu-arch=dav-2201`） | Ascend 910B1 |
| Python | 3.11 | host driver、pytest、golden 父进程 |
| JAX / JAXLIB | 0.8.1（`xla-npu` PJRT 插件，backend `npu`） | jax family trace / AD；CPU oracle |
| numpy | 有 | `inputs.npz` / `outputs.npz`、ctypes 缓冲 |
| pytest | 测试镜像 #145 内 | pytest 交付件入口 |
| 设备 | Ascend 910B1 ×1；宿主设备 5，容器内 `device_id=0`，独占 | 真机执行 |

本批**不依赖** `torch` / `torch_npu`：数值路径是 numpy → ctypes → ACL，trace/AD 路径是 JAX；
镜像 #145 也不含 `torch`/`torch_npu`。

### 2.2 内部适配模块

| 适配点 | 说明 |
|---|---|
| 四动作入口 | `build.sh` 只 `make jax_lax_cond`；可执行文件固定 `ops/jax_lax_cond/build/jax_lax_cond`；CMake 通过 `add_dependencies` 顺带产出 `libjax_lax_cond_host.so` |
| 共享 runner | `tools/run_cases.py` 按 `op == "jax_lax_cond"` 分派到 `ops/jax_lax_cond/scripts/cond_engine.py`；`tools/smoke_case.py`、`tools/run_benchmark.py` 同样按 op 分派；本节点不改共享 runner |
| golden | `ops/jax_lax_cond/scripts/golden.py` + `golden_cli.py`（CPU 子进程，`JAX_PLATFORMS=cpu`） |
| 判据 | `criteria/check.py`（`ops-precision-standard@1`），只读复用 |
| 用例 | `cases/jax_lax_cond/{cases,benchmark}.json`，冻结、不改；补充/白盒用例只在 `tests/` |
| 图引擎 | 不涉及：本批为 Ascend C 直调，不注册 GEIR/ACLNN，不做图内 shape/dtype 推导 |
| workspace | kernel 不使用额外 workspace；每次 `.so` 调用临时分配 tiling/token/counters 设备缓冲并在返回前释放；数据缓冲由 Python 侧通过 `.so` 的 `malloc/h2d/d2h/free` 管理 |
| 错误码体系 | C 侧 `0/1/2/3` 与 Python `CondError` 四码同表；CLI 退出码与 `.so` 返回值一致 |
| 日志 / 维测 | stdout `EXEC ...` 标记行、`<outdir>/meta.json`、模块级 `_last_dispatch`、`.so` 的 `err` 缓冲 |

### 2.3 需求模块设计

| 模块 | 文件 | 职责 |
|---|---|---|
| device kernel | `op_kernel/jax_lax_cond_kernel.asc` | 逐输出叶子向量原语；连续叶子按 tile 多核 grid-stride，跨步叶子单核抽取；0 号核写 token/counters |
| tiling | `op_kernel/jax_lax_cond_tiling.h` | `CondTilingData`、`COND_OP_*`、`COND_MAX_LEAVES=8`、`COND_TILE_ELEMS=8192`、`COND_MAX_STRIDED_ELEMS=4095`；kernel/host/pyapi 共用 |
| 可执行入口 | `op_host/jax_lax_cond.asc` | argv 协议校验 + `execvp` 到 host driver；`--self-check` 只 dlopen `.so` 并读 ABI，不含数值计算 |
| C ABI | `op_host/jax_lax_cond_pyapi.asc` + `op_host/jax_lax_cond_abi.h` | `extern "C" jax_lax_cond_run`：参数校验 / tiling / kernel launch / 证据 D2H；辅助导出 ACL 初始化与设备缓冲接口 |
| Python 接口 | `python/jax_lax_cond.py` | `cond` / `cond_compiled` / `CondError` / `_last_dispatch` / `_reset_evidence` / `_plan_dispatch` / `library_fingerprint`；族分派、pred 解析、plan 缓存、错误归一化、grad 路线 |
| host driver | `python/jax_lax_cond_cli.py` | `--case-json/--inputs/--mode/--variant/--outdir` 协议；`outputs.npz`/`meta.json`、`EXEC` 标记、拒绝分类、grad 扁平化与 host shape 恢复 |
| 参考实现 | `scripts/golden.py` + `golden_cli.py` | 同版本 JAX 0.8.1 CPU oracle，独立子进程，不 import 候选 |
| runner 适配 | `scripts/cond_engine.py` | 门禁 cond 分支：用例解析、输入生成、候选调用、golden 子进程、逐叶判定、分支证据与报告组装 |
| 基准 | `scripts/cond_bench.py` | benchmark 分支：设备侧 Task Duration 采集与序列汇总 |
| fixtures | `scripts/fixtures.py` | 分支词表、输入 pytree、用例展开、golden 参数绑定 |
| 测试 | `tests/test_jax_lax_cond.py` + `tests/jax_lax_cond_{supplement,whitebox}_cases.json` | pytest 交付件：29 身份 / 83 mode + 15 自检 + 4 诊断（102 testcase） |
| 构建 | `CMakeLists.txt` | 可执行文件与共享库双 target；POST_BUILD 打印 `.so` 指纹并运行 `--self-check` |

### 2.4 算子原型

#### 2.4.1 输入

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `pred` | bool | — | 0-d | 严格 scalar bool；Python `bool`、`np.bool_`、0-d bool 数组或可静态求值的 0-d `jax.Array(bool)` |
| `true_fun` | callable | — | — | `(*operands) -> pytree`；必须被 trace |
| `false_fun` | callable | — | — | 同上；两分支输出结构/逐叶 shape/dtype/weak_type 必须一致 |
| `*operands` | 数值叶为 float32；numpy family 为 `numpy.ndarray`，jax family 为 `jax.Array` | ND | 任意（pytree） | 每个输入叶子 C-contiguous；pytree 展开后输入叶子数 1…8；v1 不承诺空 pytree |

#### 2.4.2 输出

| 参数名 | dtype | format | shape | 说明 |
|---|---|---|---|---|
| `result` | 与两分支输出一致（数值用例 float32） | ND | 与两分支输出逐叶一致 | pytree 结构 / 叶子数 / shape / dtype / weak_type 精确一致；新数组，不与输入别名 |

#### 2.4.3 属性 / 参数

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `warm_up` | int | 3 | 3 | 性能用例预热次数（benchmark 口径，不是算子入参） |
| `invocations` | int | 10 | 10 | 性能用例重复次数（benchmark 口径） |
| `mode` | str | — | `eager` / `jit` / `grad_eager` / `grad_jit` | 测试/门禁的执行模式；`jit` 在 numpy family 使用 `cond_compiled` |
| `branch_id` | int | — | 1…14 | 内部注册表标识（1–8、11–14 可下发 kernel；9/10 仅 trace/grad） |

本算子没有形状/数值类算子属性（没有 axis、alpha、keepdims 等）；`pred` 是输入而不是属性。

#### 2.4.4 推导规则

- 输出结构/逐叶 shape/dtype/weak_type 由两分支 trace 结果推导，必须完全一致；不做广播、不做
  隐式类型提升；
- 校验顺序固定为 **pred → 结构 → shape → dtype**；多个不一致同时存在时只报最高优先级码；
- 拒绝必须发生在任何 device kernel 启动前：`path=reject`、`abi_calls=0`、不写 `outputs.npz`，
  并保留上游异常类型/消息 sha256；唯一例外是 c13 的 `int32(0/1)`（按已确认规格规则拒绝，
  CPU 不抛错，`cpu_rejects=false` 原样留证）；
- `jit` 的 plan key 只含 `(branch_ids, 输入 shape/dtype, pred_kind, 输出 shape)`，**不含 pred 值**；
  `jit_true` / `jit_false` 共享同一 `plan_id`，每次调用仍真实 trace 两分支；首次建立 plan 时
  `compile_count=1`，后续同 plan 命中缓存 `plan_cache="hit"`；
- 路由规则：

| 条件 | `path` | `exec_mode` | `abi_calls` |
|---|---|---|---|
| numpy family、已注册 branch、`eager` | `so` | `selected_only` | 1 |
| numpy family、已注册 branch、`jit` | `so` | `selected_only` | 1 |
| numpy family、未注册 callable | 拒绝 `invalid_argument` | `reject` | 0 |
| jax family、concrete non-grad | `trace` | `jaxpr_runtime_token` | 0 |
| jax family、grad | `trace` | `trace_autodiff` | 0 |
| tracer pred（outer jit） | `trace` 后 `device_error` | `jaxpr_runtime_token` | 0 |
| 四类错误输入 | `reject` | `reject` | 0 |

```text
pred 解析（scalar bool / 规格规则拒绝）
  → operand 展开与 family 判定（numpy / jax）
  → numpy: jax.eval_shape 两分支 trace + JAX 结构/shape/dtype 校验
         → 已注册 branch → 选中 branch 的逐叶 plan → .so kernel
         → 未注册 branch → invalid_argument
  → jax:  backend 非 CPU → 两分支 eval_shape → 选中分支（direct / flat_vjp / eager_vjp_jit_bwd）
  → 拒绝：CondError(code + upstream type/message/sha256)，abi_calls=0
```

---

## 3. 详细设计

### 3.1 使能方式

采用 **Ascend C 直调**（与 `ops/jax_lax_scan` / `ops/elementwise_sub` 同一工程形态）：

- 四动作门禁调用可执行文件 `ops/jax_lax_cond/build/jax_lax_cond`；该文件只做 argv 协议校验，
  再 `execvp("python3", python/jax_lax_cond_cli.py ...)`；数值计算全部在
  `libjax_lax_cond_host.so` → AscendC kernel（numpy family）或 JAX trace（jax family）内完成；
- Python 交付接口通过 ctypes 直接加载同一份 `.so` 的 `extern "C"` ABI；
- 不注册 ACLNN / GEIR 算子，不做图内 shape 推导或融合子图；`cond_compiled` 是 host driver 的
  编译态入口，不是第二个公开 JAX API。

### 3.2 总体设计方案

#### 3.2.1 Host 侧

**总体数据流**

```text
调用方 (Python)
  └─ cond / cond_compiled  [python/jax_lax_cond.py]
       ├─ _pred_info / _family_of / _operand_leaves
       ├─ numpy family
       │    ├─ _trace_cond_abstract：jax.eval_shape 触发两分支 trace 与 JAX 校验
       │    ├─ _build_output_plan：选中 branch → 逐输出 (src, op, stride)
       │    └─ _run_so：ctypes 设备缓冲 malloc/H2D → jax_lax_cond_run [.so]
       │         └─ Validate → Launch → jax_lax_cond_kernel<<<blockNum>>> →
       │              token/counters 写设备内存 → aclrtMemcpy(D2H) → Python 读回
       └─ jax family
            ├─ backend 非 CPU；两分支 eval_shape；runtime token 仅按分支约定记录
            └─ 选中分支：direct_branch / flat_vjp / eager_vjp_jit_bwd

四动作 gate (tools/run_cases.py → scripts/cond_engine.py)
  └─ build/jax_lax_cond [op_host/jax_lax_cond.asc]
       └─ execv python3 python/jax_lax_cond_cli.py
            ├─ eager/jit : cond / cond_compiled → .so kernel
            ├─ grad_*    : cond → JAX trace / AD（NPU backend）
            └─ reject    : rc=3 + `REJECT code=...` + `EXEC path=reject ...`
```

**可执行入口与四动作协议**

`op_host/jax_lax_cond.asc` 的 `main()` 只做三件事：

1. 校验 argv：只接受 `--case-json/--inputs/--mode/--variant/--outdir`，未知 key 或缺值直接
   `rc=2`（`REJECT_BAD_ARGS ...`），不进入算子路径；
2. 定位 `python/jax_lax_cond_cli.py`，`execvp("python3", ...)`；找不到 driver 时 `rc=1`；
3. `--self-check` 只 dlopen `libjax_lax_cond_host.so`、读 ABI 版本并打印 `SELF_CHECK ...`，
   不跑用例、不需要 NPU。

CLI（host driver）负责：

- 读 `case.json`（含 `_resolved` 的 branch/树/输出 spec）与 `inputs.npz`；
- `eager`/`jit` 走 `cond`/`cond_compiled`；`jit` 的 numpy family 使用 `cond_compiled`，
  jax family 使用零参数/普通 `jax.jit` closure（0-d operand 或 concrete pred 作为闭包常量）；
- `grad_eager`/`grad_jit` 走 `jax.grad`；`grad_jit` 在 host 把 operand 展平成 `[N]` 进入设备编译，
  比较前在 host 恢复原 shape（不改变数值）；
- 成功写 `outputs.npz` + `meta.json`，stdout 首行 `EXEC ...`；拒绝写 `meta.json`、
  `REJECT code=...`/`UPSTREAM ...`、`EXEC path=reject ...`，返回 3 且**不写** `outputs.npz`；
- 其它失败打印 `ERROR ...` 并返回 1；本任务没有合法的 `UNSUPPORTED`/`rc=4`。

**C ABI 与 Python 接口**

`op_host/jax_lax_cond_abi.h` 定义 `jax_lax_cond_args_t`（Python `ctypes.Structure` 逐字段对应，
布局 472 B = 59×8）：

| 字段组 | 字段 | 语义 |
|---|---|---|
| 分支/叶子 | `branch_id`、`n_outputs`、`n_inputs`、`dtype[8]` | `branch_id` 1=true/2=false；叶子数 1…8；`dtype=0` 仅 float32 |
| 输入 | `in_numel[8]`、`in[8]` | 设备地址，按 pytree 稳定顺序 |
| 输出 | `out[8]`、`out_numel[8]`、`out_src[8]`、`out_op[8]`、`out_stride[8]` | 每个输出叶子的源输入、`COND_OP_*` 原语与元素步长 |
| 运行/证据 | `device_id`、`token`、`counters`、`err`、`err_len`、`abi_version` | `token/counters` 是 host 侧 int32 缓冲；kernel 写设备内存后由 `.so` D2H 读回 |

导出（`extern "C"`）：

```c
int         jax_lax_cond_run(const jax_lax_cond_args_t *args);  /* 0/1/2，拒绝在 Python 层 */
const char *jax_lax_cond_abi_version(void);                    /* "opx-cond-python-abi/1" */
int         jax_lax_cond_acl_init(void);                       /* 幂等；绑定 device 0 */
int         jax_lax_cond_malloc(void **dev_ptr, int64_t bytes);
int         jax_lax_cond_free(void *dev_ptr);
int         jax_lax_cond_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
int         jax_lax_cond_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
```

`out_src/out_op/out_stride` 相对 `0-黑盒测试设计.md` §1.3.4 的非绑定基线是追加字段：没有它们就
无法在不做任何 host 侧布局重排的前提下表达 `c06` 的 `aux = x[:,0] ± 1`（跨步叶子）。Python 侧
在加载 `.so` 时用 `ctypes.sizeof == 472` 做布局自检，并校验 ABI tag；不一致直接 `device_error`。

**路由与模式**

| 用例 `mode` | 入口 | `path` / `exec_mode` | 关键硬断言 |
|---|---|---|---|
| `eager` / `eager_true` / `eager_false` | `cond` | numpy：`so`/`selected_only`；jax：`trace`/`jaxpr_runtime_token` | numpy：`abi_calls==1`、`trace_hits.true/false≥1`、`selected` 与 pred 一致、`runtime_token` 1/2、`exec_hits` 只选中支 +1；jax：`abi_calls==0`、backend/设备平台为 npu、`trace_hits` 两支 ≥1 |
| `jit` / `jit_true` / `jit_false` | numpy：`cond_compiled`；jax：`cond`/`jax.jit` | numpy：`so`/`selected_only`；jax：`trace`/`jaxpr_runtime_token` | numpy：首次 `compile_count==1`、同 plan `plan_cache=="hit"`、每次 `abi_calls==1`、动态谓词两次共享 `plan_id`；其余同 eager；jax：`abi_calls==0`、backend 非 CPU、`trace_hits` 两支 ≥1、`weak_type` 精确 |
| `grad_eager` | `jax.grad(lambda x: sum(cond(pred, true, false, x)))` | `trace`/`trace_autodiff` | `abi_calls==0`、backend 非 CPU、输出设备平台含 npu、`trace_hits` 两支 ≥1、梯度与 CPU 一致、`cotangent_route=eager_vjp_jit_bwd` |
| `grad_jit` | `jax.jit(jax.grad(...))` | `trace`/`trace_autodiff` | 同上；`cotangent_route=flat_vjp`；operand 以扁平 `[N]` 进入 |
| 拒绝（c13–c16） | `cond`/`cond_compiled` | `reject`/`reject` | `code` 精确、`abi_calls==0`、不写输出、上游类型/摘要可复算 |

**jit 的「动态谓词」口径**：本批 `c04` 的 `jit_true`/`jit_false` 是**运行期具体标量 bool**
（同一 plan 的两种取值，`plan_id` 相同、`runtime_token` 分别为 1/2），不是 `jax.jit` 图内的 tracer
pred。outer `jax.jit` 的 tracer pred 与 stock `lax.cond` 在 jit 内的 conditional 受 AFIR 后端限制，
D4 记录为 `BACKEND_LIMITATION`，**不计 PASS**（§5.4/§5.6）。

**异常与拒绝码**

| code | 触发 | 行为 |
|---|---|---|
| `INVALID_PREDICATE` | pred 非 scalar 或 dtype 非 bool（含整型 0/1） | `abi_calls=0`；保留上游 TypeError 摘要；int32 0/1 按 `spec_rule=scalar_bool_pred` 留证 |
| `BRANCH_STRUCTURE_MISMATCH` | 两分支 tree structure 不同（容器/键/叶子数） | `abi_calls=0`；上游 TypeError + sha256 |
| `BRANCH_SHAPE_MISMATCH` | 同结构下某叶 shape 不同 | 同上 |
| `BRANCH_DTYPE_MISMATCH` | 同结构/shape 下某叶 dtype 不同 | 同上 |
| `invalid_argument` | 非 callable、空 pytree、>8 叶、未注册 numpy 分支、非连续/非 float32、`cond_compiled` 在 trace 内 | Python `ValueError`/`CondError`；`.so` rc=2 |
| `device_error` | ACL/kernel 失败、backend 是 CPU、tracer pred、2-D jit grad | `CondError`/rc=1；不得记 PASS |
| `internal_error` | 候选内部不变量失败 | `CondError`；不得记拒绝通过 |

C 侧返回码：`0` 成功 / `1` 运行失败（`device_error`）/ `2` 参数错误（`invalid_argument`）/ `3`
拒绝（本层不产生，拒绝在 Python 层设备操作前完成）。CLI 退出码：`0` 成功、`1` ERROR、`2`
argv 错误、`3` 拒绝、`127` `execvp` 失败；`4`/`UNSUPPORTED` 在本任务不是合法状态。

**设备缓冲与 workspace**

- Python 侧：对每个输入叶子 `jax_lax_cond_malloc(nbytes)` + `h2d`；每个输出叶子 `malloc` +
  `d2h`；调用结束统一 `free`；输入/输出缓冲分离，输入不被原地修改；
- `.so` 每次调用分配 `CondTilingData`、1×int32 token、2×int32 counters 三块设备缓冲，
  kernel 后同步并 D2H 读回，随后释放；
- **不使用额外 workspace**；kernel 的 UB 预算为 in 32KB + out 32KB + evidence 32B（远小于 192KB）；
- ACL 生命周期：`aclInit` 由 `.so` 幂等执行（PJRT 已初始化时视为可用），每次 `aclrtSetDevice`；
  **不调用 `aclFinalize`**，避免破坏进程内其它 ACL 使用者（PJRT/xla-npu）的生命周期。

**plan 缓存与证据字段**

- plan key 只含 `(branch_ids, 输入 shape/dtype, pred_kind, 输出 shape)`；pred 值不进入 key；
- `compile_count` 是候选自己的 plan 编译次数（不是 JAX/XLA 编译次数）；首次为 1，后续同 key
  不增；`cond`（eager）不做编译，`compile_count=0`、`plan_cache="eager"`；
- `_last_dispatch` 记录 `path/exec_mode/abi_calls/selected/branch_id/compile_count/plan_id/
  plan_cache/backend/device_id/pred_kind/pred_source/pred_platform/pred_sha256/token_source/
  trace_hits/exec_hits/runtime_token/evidence_level/family/cotangent_route`；
- numpy `.so` 路径的 `token_source=kernel_token_counters`，`exec_hits` 来自 kernel 写入并 D2H
  读回的 counters；jax trace 路径的 `token_source=branch_id_convention_not_raw_kernel_token`，
  不把约定值写成 kernel token（`abi_calls=0`）。

**grad 路线（全部由 JAX 反向模式 AD 产生，无 `branch_id` 导数表）**

| 模式 | 路由 | 实现要点 | 限制 |
|---|---|---|---|
| `grad_jit` | `flat_vjp` | `custom_vjp` 的 backward 用真实 `jax.vjp(branch_fn, value)`；内部扁平 `[N]`；转置图乘输入相关恒等项 `((v+1)-(v-1))/2` | 只承诺扁平 `[N]`；2-D operand 显式 `device_error`；恒等项在 `abs(v) ≥ 2**24` 退化 |
| `grad_eager` | `eager_vjp_jit_bwd` | `custom_vjp` forward 保存真实 `value`；backward 调 `jax.jit(jax.vjp(branch_fn, value))(cotangent)`，消费真实 cotangent | 单叶 grad；多叶 pytree grad 未验证 |
| 非 grad jax trace | `direct_branch` | 只调用选中分支 callable | 无 device 侧 conditional |

被替换的旧路线（`_AD_CONST_DERIV` 按 `branch_id` 硬编码导数、`eager_const_cotangent`）已删除，
当前源码 grep 为零；该路线违反 `0-黑盒测试设计.md` 的“不得按 `branch_id` 硬编码导数”要求。

#### 3.2.2 Kernel 侧

**分核策略**

- 连续输出叶子的 tile 数 `totalTiles = Σ ceil(outNumel[j] / 8192)`；
- `blockNum = min(totalTiles, VECTOR_CORE_NUM)`，且至少为 1；小 shape（如 `[16]`）为 1 核，
  大 shape（如 `[1024,256]`，32 tiles）最多 32 核；
- 每个连续输出叶子按 `t = blockIdx; t < tiles; t += blockNum` 的 grid-stride 循环处理，因此
  `tiles > blockNum` 时单核处理多个 tile；
- 跨步叶子只由 0 号核处理，host 已限制 `out_numel ≤ 4095` 且源输入 `numel ≤ 8192`。

**Init**

- `tokenGm`（1×int32）与 `counterGm`（2×int32）建立 GlobalTensor；
- UB：`inQ` 1×32KB、`outQ` 1×32KB、`eviQ` 1×32B；`COND_MAX_LEAVES=8` 个输入/输出槽；
- `CondTilingData` 从 GM 读取，逐叶子设置 `inGm[i]/outGm[i]` 的边界。

**Process（逐输出叶子）**

1. `stride == 1`：按 `COND_TILE_ELEMS=8192` 分块；`count = min(remain, 8192)`；`CopyInContig`
   → `Compute` → `CopyOut`；
2. `stride > 1`：0 号核把整个源叶子 `CopyInStrided`（`blockCount=1`），用 `LocalTensor::GetValue`
   按 `i*stride` 抽取，`SetValue` 写 UB，`PipeBarrier<PIPE_ALL>` 保序；
3. 所有输出叶子处理完后，0 号核 `WriteEvidence`：
   - `token = branchId`（1/2）；
   - `counters[0] = (branch==1)`、`counters[1] = (branch==2)`；
   - token/counters 由 kernel 写、host D2H 读回，Python 不自报。

**原语表**

| `out_op` | 常量 | 向量计算 | 等价表达式 |
|---|---|---|---|
| 1 | `COND_OP_ADD1` | `Adds(y, x, 1.0f, count)` | `x + 1` |
| 2 | `COND_OP_SUB1` | `Adds(y, x, -1.0f, count)` | `x - 1` |
| 3 | `COND_OP_MUL2` | `Muls(y, x, 2.0f, count)` | `x * 2` |
| 4 | `COND_OP_HALF` | `Muls(y, x, 0.5f, count)` | `x / 2`（二进制浮点下与 `x*0.5` 逐位等价） |
| 5 | `COND_OP_AFFINE_TRUE` | `Muls(1.125f)` → `Adds(0.5f)` | `x*1.125 + 0.5` |
| 6 | `COND_OP_AFFINE_FALSE` | `Muls(0.875f)` → `Adds(-0.5f)` | `x*0.875 - 0.5` |

**分支表（`_BRANCH_NAME_TO_ID` / `fixtures.BRANCH_FUNCS`）**

| `branch_id` | 名称 | 语义 | 可下发 kernel |
|---|---|---|---|
| 1 | `x+1` | `x + 1` | 是（单输入/单输出） |
| 2 | `x-1` | `x - 1` | 是 |
| 3 | `x*2` | `x * 2` | 是 |
| 4 | `x/2` | `x / 2` | 是 |
| 5 | `tree_add` | pytree 逐叶 `+1` | 是（叶数相同） |
| 6 | `tree_sub` | pytree 逐叶 `-1` | 是 |
| 7 | `out_tree_true` | `(x+1, {"aux": x[:,0]+1})` | 是（1 输入/2 输出，aux 步长 `shape[1]`） |
| 8 | `out_tree_false` | `(x-1, {"aux": x[:,0]-1})` | 是 |
| 9 | `grad_true` | `2*x` | 否（仅 JAX trace/grad） |
| 10 | `grad_false` | `3*x` | 否 |
| 11 | `list_inc` | list 逐叶 `+1` | 是 |
| 12 | `list_dec` | list 逐叶 `-1` | 是 |
| 13 | `p02_true` | `x*1.125 + 0.5` | 是 |
| 14 | `p02_false` | `x*0.875 - 0.5` | 是 |

**Tiling 结构**

```c
struct CondTilingData {
    int32_t branchId;         // 1=true, 2=false（只用于证据写入）
    int32_t nInputs;          // 1..8
    int32_t nOutputs;         // 1..8
    int32_t outOp[8];         // COND_OP_*
    int32_t outSrc[8];        // 输入叶子下标
    int64_t outNumel[8];      // 输出叶子元素数
    int64_t outStride[8];     // 源输入叶子内的元素步长（连续=1）
    int64_t inNumel[8];       // 输入叶子元素数
};
```

常量：`COND_MAX_LEAVES=8`、`COND_TILE_ELEMS=8192`（32KB/tile）、`COND_MAX_STRIDED_ELEMS=4095`
（`DataCopy` blockCount 上限）。本算子没有按 tilingkey 分派的模板；分派由 host 侧 `outOp` 与
`stride` 几何决定。

**搬运、对齐与精度**

- 连续叶子统一 `DataCopyPad`，`blockLen = count * sizeof(float)`；尾块允许非 32B 对齐：
  `float32[17]` = 17×4 = 68 B，`68 % 32 = 4`；`float32[33]` = 33×4 = 132 B，`132 % 32 = 4`；
  `float32[8192]` = 32768 B，`32768 % 32 = 0`；`float32[8193]` = 32772 B，`32772 % 32 = 4`；
  `float32[1024,256]` = 262144×4 = 1,048,576 B，`% 32 = 0`；
- 逐元素 fp32 计算（`Adds/Muls`），无跨元素归约、无累加舍入差异；`x/2` 用 `Muls(0.5f)` 与
  IEEE 754 二进制浮点的 `x/2` 逐位等价；
- NaN/±Inf 由向量原语逐元素传播，判据对非有限位做精确比对；
- 跨步叶子用 `LocalTensor` 标量读写，`PipeBarrier<PIPE_ALL>` 保证 MTE2→S 与 S→MTE3 的顺序；
- 0 号核写证据，不需要跨核同步。

### 3.3 支持硬件

- 目标芯片：Ascend 910B1（Atlas A2 系列），架构 `dav-2201`；编译选项 `--npu-arch=dav-2201`；
- 计算单元：kernel 只使用 **AIV 向量核**（`Adds/Muls/DataCopyPad`），不使用 AIC；
- 核数：`blockNum` 由 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 与输出 tile 数取
  `min`，不写死核数；小 shape 1 核、大 shape 最多 32 核（本批实测 `[1024,256]` 为 32）；
- 设备形态：单卡（宿主设备 5 / 容器 `device_id=0`），单进程；多卡/多设备不涉及；
- 运行环境：CANN 9.2.0-beta.2 + bisheng + ACL；JAX/JAXLIB 0.8.1 的 `xla-npu` 插件用于
  trace/AD 路径；
- HBM 带宽、AIC 核数等未在本批采集，标 **UNKNOWN**，不作为交付承诺。

### 3.4 算子约束与限制

| 约束 | 范围 / 行为 |
|---|---|
| `pred` | 严格 scalar bool；`bool[1]`/`bool[2]` 在 numpy family 由 JAX scalar 校验拒绝为 `INVALID_PREDICATE`；整型 `0/1` 按已确认规格规则拒绝（CPU JAX 接受，`cpu_rejects=false` 留证）；jax family 的 pred 必须是可静态求值的 scalar bool，非 scalar jax bool 数组不在本批 required 覆盖内；pred 不可导 |
| operands | pytree；v1 不承诺空 pytree；输入叶子 1…8（`COND_MAX_LEAVES=8`）；数值叶 C-contiguous |
| dtype | 数值叶仅 float32；int32 只出现在“分支 dtype 不一致”的拒绝用例；无 float16/bfloat16/float64；无隐式提升 |
| numpy family 分支 | 必须是本仓已注册 callable（`_BRANCH_NAME_TO_ID` 的 1–8、11–14）；未注册 callable 只能走 jax family trace，numpy family 显式 `invalid_argument` |
| 两分支输出 | tree structure / 逐叶 shape / dtype / weak_type 完全一致；结构/shape/dtype 不一致按四码拒绝；weak_type mismatch 不在本批四码 required 用例内 |
| `jit` plan | pred 值不进入 plan key；`jit_true`/`jit_false` 共享 `plan_id`、`runtime_token` 1/2；plan 缓存不跨 case/进程（fresh worker） |
| outer `jax.jit` tracer pred | 目标 AFIR 无 `stablehlo.conditional`/`select` 实现、`convert` 不可执行、`debug_callback`/`pure_callback` 未注册 npu lowering；显式 `device_error`，D4 `BACKEND_LIMITATION`，不记 PASS |
| stock `lax.cond` 在 jit 内 | `Unsupported instruction opcode: conditional`；D4 `BACKEND_LIMITATION`，不计 PASS |
| grad | 只对浮点 operand、单叶、逐元素分支；`grad_jit` 只承诺扁平 `[N]`，2-D/多叶显式 `device_error` 或未验证；eager 的后向 VJP 用 `jax.jit` 编译 |
| jit 恒等项 | `((v+1)-(v-1))/2` 仅在 `abs(v) < 2**24` 时精确等于 1（fp32）；冻结输入 O(1)/×1000 在范围内 |
| 0-d jit（S1） | 镜像 #145 不能把 0-d operand/动态 pred 当 `jax.jit` 实参；S1 用零参数 closure，HLO 是常量图，分支在 closure 外以 eager 设备执行；数值与 `weak_type` 精确，但**不宣称分支在 jit 图内运行** |
| 跨步输出叶子 | 仅用于小 aux：`out_numel ≤ 4095` 且源输入 `numel ≤ 8192`；本批 c06 为 7 元素 |
| 设备 | 单卡；多设备/SPMD 不支持；每次调用独立 ACL stream，返回前同步 |
| layout | spec 登记 `UNKNOWN`；实现按连续 ND / C-contiguous 处理，不承诺其它物理布局 |
| 性能 | 本轮只记基线（`performance_target=null`），不设阈值，不宣称性能达标 |

---

## 4. 特性交叉分析

### 4.1 混合精度

**不涉及混合精度**：数值叶子只有 float32，没有 float16/bfloat16/float64，也没有隐式类型提升；
int32 只出现在“分支 dtype 不一致”的拒绝用例中，不作为数值输出支持范围。`weak_type` 是 JAX 的
aval 元数据（weak/strong 标量语义），不是 dtype 混合；S1 要求 weak/strong 的 `weak_type` 与 CPU
golden 逐叶精确。kernel 内部全部 fp32 向量运算，不存在跨 dtype 转换点。

### 4.2 图模式

**不涉及图引擎**：本批是 Ascend C 直调算子，不是 GEIR/ACLNN 注册算子，不进入图引擎，也没有图内
shape/dtype 推导、图融合或属性传递。JAX 侧的 `jit` 是候选内部的编译态入口（`cond_compiled`）或
trace 路径的 `jax.jit`，与昇腾图引擎不是同一概念。目标 AFIR 后端对 `stablehlo conditional`
的 lowering 缺失是**后端能力限制**（D4），不是图模式接入问题。

### 4.3 动态 shape

**不支持运行期动态维度**：每个 plan 的输入/输出 shape/dtype 在 trace 期静态确定；tiling 由 host
按当次 `outNumel/outStride/nInputs/nOutputs` 生成，kernel 只按元素数分块，不做运行期 shape 推导。
新 shape 需要重新 trace/建立 plan；pred 是运行期标量值、不进入 plan key，因此同一 shape 下的
`jit_true`/`jit_false` 复用同一 plan。本批所有用例为静态 shape。

### 4.4 其他特性

| 交叉维度 | 组合 | 结论 / 依据 |
|---|---|---|
| 输入 pytree | dict/list 嵌套 + numpy/jax family | 支持；叶子按稳定顺序展开，`tree_add/tree_sub`/`list_inc/list_dec` 逐叶计算（c05/S3/WB-05/WB-06） |
| 输出 pytree | tuple + dict（`out/0`、`out/1/aux`） | 支持；host 按模板 `_rebuild` 重组（c06/WB-04） |
| 跨步输出叶 | `x[:,0] ± 1`，stride=`shape[1]` | 支持小 aux（c06 `[7,5]`→7 元素；WB-04 `[64,128]`→源 8192 上界）；host 限制 4095/8192 |
| 非 32B 尾宽 | 17/33 元素 | 支持；`DataCopyPad` 处理 68 B/132 B（`%32=4`），c11 8 个 mode 全 PASS |
| 大 shape | `[1024,256]`（262144 元素，32 tiles） | 支持；多核 grid-stride，c12 4 个 mode 全 PASS |
| 0-d weak/strong | `float32[]`，eager/jit | 支持；S1 4 个 mode PASS，`weak_type` 与 CPU 精确；jit 为常量图（§3.4） |
| NaN/±Inf | 8 个特殊值 | 支持；S5 4 个 mode PASS，非有限位逐元素精确 |
| 动态谓词 | 同一 plan 的 true/false | 支持；c04/WB 的 jit_true/jit_false 共享 `plan_id`，`runtime_token` 1/2 |
| 两分支 trace | eager/jit + 两分支 | 支持；`trace_hits.true/false ≥1`，c07 PASS |
| 单分支执行 | `.so` 的 4 个 mode | 支持；kernel token/counters 证明只选中支 +1，c08/D1 PASS |
| eager × jit | 同输入 | 必须一致；门禁/ pytest 对同用例两模式做互比，全部 PASS |
| grad × eager/jit | 单叶逐元素分支 | 支持；c09/c10/S4/D2 PASS，真实 cotangent，无 `branch_id` 导数表 |
| reject × eager/jit | c13–c16 | 都在 device 前拒绝，`abi_calls=0`，不写 `outputs.npz` |
| 多流 / 并发 | 每次调用独立 stream | 顺序调用支持；并发调用未验证 |
| 多设备 / SPMD | — | 不支持（单卡） |
| 性能 × 大小 shape | p01 `[16]` / p02 `[1024,256]` | 仅基线；block_dim 1/32，见 §5.2 |

---

## 5. 可维可测分析

### 5.1 精度标准

- **oracle**：同版本 JAX 0.8.1 **CPU 独立子进程**（`ops/jax_lax_cond/scripts/golden.py` +
  `golden_cli.py`，父进程设置 `JAX_PLATFORMS=cpu`），不 import 候选、不依赖 NPU 结果；golden 与
  候选读同一份 `case.json`/`inputs.npz`；
- **判据档**：`criteria/check.py`（`ops-precision-standard@1`）；
- **容差**：float32 `rtol=2^-10`（≈9.77e-4）、`atol=2^-16`（≈1.53e-5）、`matched_ratio ≥ 0.99`、
  `max_abs_error ≤ max(1e-2, 32×ULP@1.0)`；float32 下 32×ULP@1.0≈3.8e-6，实际下限为 1e-2；
  通过条件为两条同时满足；
- **结构与类型**：tree structure、叶子数、dtype、shape 完全一致（0 容差）；
- **weak_type**：jax family 逐叶与 CPU golden 精确；numpy family 记为 `null`，不冒充覆盖；
- **特殊值**：NaN/±Inf 位置与符号逐元素精确，有限子集再走浮点容差；`-0.0` 符号不单独断言；
- **设备守卫**：numpy family 必须 `path=so`、`abi_calls=1`、`backend=npu`、`device_id=0` 且有
  `so_fingerprint`；jax family 必须 `jax.default_backend() != "cpu"` 且输出设备平台含 `npu`；
  任一 CPU 回退判 `INVALID`；
- **异常输入**：四码必须精确、`abi_calls=0`、不写输出；CPU 可确认的拒绝要求候选
  `upstream.type == cpu_exception_type`；c13 的 int32 0/1 是已确认的规格规则拒绝，CPU 不抛错，
  以 `cpu_probe.rejects=false` / `spec_rule=scalar_bool_pred` 留证；
- **per-mode**：eager/jit/grad_eager/grad_jit 分开记录；xfail/unsupported 不计 PASS；
- **本版实测**：gate 16/16（49/49 mode，0 FAIL/ERROR/INVALID/NOT_RUN）；pytest 29 身份 / 83 mode
  全 PASS，`matched_ratio=1.0`；除 `cond_large_tensor_throughput` 的 `jit_true`/`jit_false` 两个
  benchmark 复用 mode 为 `max_abs_error=1.1920928955078125e-07` 外，其余为 `0.0`；两者均远低于
  1e-2 且 `matched_ratio=1.0`，不影响 PASS。

### 5.2 性能标准

- **基线**：不设 CPU / torch 基线（`baseline: null`、`comparison: same_device_only`）；
  `performance_target=null`，本轮只记基线，不设硬阈值；
- **测量口径**：`scripts/nm.sh --op jax_lax_cond --action benchmark` → `tools/run_benchmark.py` →
  `scripts/cond_bench.py`；设备侧 `msopprof` 的 `Task Duration(us)`；blocking JIT 稳态，
  `warm_up=3`、`invocations=10`、median、`block_until_ready`；
- **代表用例**：`cond_small_branch_overhead`（`float32[16]`，小分支开销）、
  `cond_large_tensor_throughput`（`float32[1024,256]`，**primary**，大张量吞吐）；
- **实测**（run `r20261028-164927` / 台账 #229，设备 5，exclusive，镜像 #145）：small true
  `4.660093 us`、small false `4.390088 us`；large true `7.460150 us`、large false
  `7.440149 us`；`block_dim` 1（small）/32（large），每序列 10 样本、无采集失败；
- **目标**：仅记录，不宣称性能达标；性能结论不替代正确性门禁。

### 5.3 兼容性分析

- **接口兼容性**：Python `cond(pred, true_fun, false_fun, *operands)` 与 `jax.lax.cond` 参数顺序
  一致；`cond_compiled` 是测试/门禁的编译态入口，不是第二个公开 JAX API，不得在
  `jax.grad`/`jax.jit` trace 内使用；CLI 协议固定
  `--case-json/--inputs/--mode/--variant/--outdir`；拒绝码、C 返回码、CLI 退出码同表；
- **数据类型兼容性**：仅 float32 数值叶；无 f16/bf16/f64；无隐式提升；weak_type 逐叶精确；
  numpy family 要求 C-contiguous，jax family 由 JAX 管理布局；
- **硬件形态兼容性**：仅 910B1（`dav-2201`）单卡；`blockNum` 随 `VECTOR_CORE_NUM` 与 tile 数
  自适应，但本批只在 910B1 的 `npu:0` 上验证；不同芯片需重新编译并复核 UB 预算/后端能力；
- **共享 runner 兼容性**：`tools/run_cases.py`/`smoke_case.py`/`run_benchmark.py` 的 cond 分支
  按 op 显式分派，elementwise、`jax_lax_scan`、`jax_lax_associative_scan` 旧路径保持原样
  （节点 1 运行级自检：scan 7 passed、assoc 11 passed；elementwise 静态未改）。

### 5.4 可观测性与可测试性

- 可执行文件 stdout 首行固定 `EXEC op=jax_lax_cond kernel=jax_lax_cond_kernel arch=dav-2201
  mode=... variant=... pred_kind=... selected=... path=... exec_mode=... abi_calls=...
  backend=... device_id=... trace_true=... trace_false=... exec_true=... exec_false=...
  runtime_token=... compile_count=... plan_id=... plan_cache=...`；缺任一键判 `INVALID`；
- `meta.json` 记录 `case_id/variant/mode/status`、`dispatch{path,exec_mode,abi_calls,selected,
  branch_id,compile_count,plan_id,plan_cache,backend,device_id}`、`evidence{trace_hits,exec_hits,
  runtime_token,evidence_level,token_source,pred_source,pred_platform,pred_sha256,
  cotangent_route,grad_input_layout}`、`leaves[{path,dtype,shape,weak_type,sha256}]`、
  `inputs_sha256`、`outputs_sha256`、`so_fingerprint`；拒绝时含 `upstream{type,message,sha256,
  summary,keyword_hit}`，无 `leaves`；
- `_last_dispatch` 暴露路由结果，供 pytest 断言 `path/exec_mode/abi_calls/selected/runtime_token/
  plan_id/plan_cache/trace_hits/exec_hits/cotangent_route`；
- `_plan_dispatch` 是纯 CPU 路由/校验函数，不执行 device、不修改 `_last_dispatch`，可单测；
- `.so` 失败路径统一写 `code: detail` 到调用方 `err` 缓冲；ABI 版本/布局自检在 dlopen 时完成；
- 每个会触发 JAX trace/compile 的 `(case_id, variant, mode)` 在 fresh worker 子进程执行，父进程
  只聚合 JSON/日志；记录 PID、spawn_pid、program sha256、spawn_index；D3 验证 83 个 case worker
  全部 `pid_matches_spawn=true`、无跨 case 缓存；
- 拒绝 / 显式 backend limitation / 数值 PASS 三态分离；D4 的 `BACKEND_LIMITATION` 与聚合
  `RECORDED` **不是 PASS**；D1（数据派生 pred/token）、D2（真实 cotangent）、D3（进程隔离）
  单列，不替代 16+2 / 29+83 的计数。

### 5.5 可维护性

- Python 接口 / host CLI / C ABI / kernel / tiling 分层清晰；branch 语义集中在
  `_SINGLE_OUTPUT_OP`、`_build_output_plan` 与 `fixtures.BRANCH_FUNCS`，新增原语只需加一个
  `COND_OP_*` 与 kernel switch 分支；
- `CondTilingData` 由 kernel/host/pyapi 三方共用，避免三份布局漂移；`out_src/out_op/out_stride`
  让“输出叶从哪个输入叶、用什么原语、按什么步长”成为显式计划，便于审计；
- 工程布局与 `ops/jax_lax_scan` / `ops/elementwise_sub` 一致（CMake 双 target、host `.asc`
  导出 `extern "C"`），降低复用成本；
- 证据字段（`token_source`、`cotangent_route`、`grad_input_layout`、`pred_source`）显式区分
  “真实设备证据”与“分支约定值”，避免把 trace 路径写成 kernel token；
- 已知限制（§3.4 / README §6）与后端能力限制（D4）在交付文档中显式列出，不把
  `BACKEND_LIMITATION` 写成支持。

### 5.6 验证口径

当前交付版本（operator-v1 `55b69ca6a7dbc10b22be8976091cdac6d00931f68eee552fd29f9f8ee121c426`，
13 文件；测试入口 `66a3d0a1340c6f256ac9edc5a719ed8199c80fedf0febf9074013ea9e81f8f2a`，
113,823 B）的四动作身份与结果：

| 动作 | run / 台账 | exit | 关键结论 | 本版关系 |
|---|---|---|---|---|
| build | `r20261028-162821` / #226 | 0 | `BUILD_OK`；`.so` 561,952 B / `bdf8530d…`，exe 381,696 B / `25d4eebf…`，ABI `opx-cond-python-abi/1` | 复用节点 2（同 operator-v1） |
| smoke | `r20261028-162906` / #227 | 0 | `SMOKE_VERDICT PASS`，`cond_true_eager/eager` 1/1 | 复用节点 2 |
| test | `r20261028-174512` / #232 | 0 | gate 16/16（49/49 mode）；pytest 29 身份 / 83 mode / 102 testcase，0 fail/error/skip；白盒 12/12；D2 8/8；D1/D3 PASS；D4 2×`BACKEND_LIMITATION` + 1×`RECORDED` | 节点 4 修复后新执行 |
| benchmark | `r20261028-164927` / #229 | 0 | 2 用例 / 4 序列，10 样本/序列；仅基线，`performance_target=null` | 复用节点 2 |

身份一致性：四个 run 的 `spec_version_id=6`、`code_snapshot_sha256=55b69ca6…`、
`suite_ref=cases@233778f456b4eb9b`、镜像 #145（digest `e99a62ad…`）、设备 5、exclusive、
`lock_acquired=yes` 一致；控制台 `GET /api/tasks/10/versions` 的 evidence gate 四项
`ok=true / gate_verdict=pass`、`missing=[]`、`deliverable=true`。

计数口径：gate `PASS=16, FAIL=0, ERROR=0, INVALID=0, NOT_RUN=0, mode_executions=49,
variant_executions=20`；pytest `PASS=83, FAIL=0, ERROR=0, INVALID=0, NOT_RUN=0,
identity_count=29, mode_executions=83, selfcheck_count=15, selfcheck_pass=15`；pytest XML
102 testcase / 0 failure / 0 error / 0 skipped；来源分布 frozen 49 / benchmark 4 / supplement 18 /
whitebox 12；`by_mode`：eager 12、jit 11、jit_true 12、jit_false 12、eager_true 15、
eager_false 15、grad_eager 3、grad_jit 3；dispatch 路径 `.so` 57 / `reject` 14 / `trace` 12。
D4 的 3 条受限记录不计入 PASS。

节点 5 只修改 `ops/jax_lax_cond/README.md` 并新增本文件；`ops/jax_lax_cond/**` 非 `.md` 文件、
`tests/`、`cases/`、`criteria/`、`golden`、阈值均未改（哈希见 `5-清单.json`），因此上述
operator-v1 证据对当前交付仍有效。原始证据索引：`.cannbot/jax_lax_cond/workflow3/4-证据-r1/`、
`4-测试执行记录.md`、`4-验收报告.md` 与 `4-清单.json`。

---

## 附录 A：测试用例要点

### A.1 冻结正确性用例（`cases/jax_lax_cond/cases.json`，16 条 / 49 mode）

| Case ID | 分组 | pred / operand / 输出 | 分支 | mode | 期望 |
|---|---|---|---|---|---|
| `cond_true_eager` | 基本分支 | true；`float32[8,16]` | `x+1` / `x-1` | eager | match_cpu |
| `cond_false_eager` | 基本分支 | false；`float32[8,16]` | `x+1` / `x-1` | eager | match_cpu |
| `cond_jit_static_pred` | jit | 静态 true；`float32[4,17]` | `x*2` / `x/2` | jit | match_cpu |
| `cond_jit_dynamic_pred` | jit | 运行期 true/false；`float32[4,33]` | `x*2` / `x/2` | jit_true / jit_false | match_cpu |
| `cond_input_pytree` | pytree | `dict(a:[6,4], b:[6,2])` | `tree_add` / `tree_sub` | eager/jit × true/false | match_cpu |
| `cond_output_pytree` | pytree | `float32[7,5]` → `(out, dict(aux:[7]))` | `out_tree` | eager/jit × true/false | match_cpu |
| `cond_both_branches_traced` | 分支证据 | true；`float32[8]` | 两支计数 | eager/jit | both_branches_traced |
| `cond_selected_branch_only_executes` | 分支证据 | true/false；`float32[8]` | token/counters | eager/jit × true/false | only_selected_branch_runtime_hit |
| `cond_grad_true` | 自动微分 | true；`float32[5,4]` | `grad_true` / `grad_false` | grad_eager / grad_jit | match_cpu |
| `cond_grad_false` | 自动微分 | false；`float32[5,4]` | `grad_true` / `grad_false` | grad_eager / grad_jit | match_cpu |
| `cond_unaligned_tail_widths` | 边界与规模 | 17 / 33 元素（68 B / 132 B，`%32=4`） | `x+1` / `x-1` | 4 模式 × 2 元素数 | match_cpu |
| `cond_large_tensor` | 边界与规模 | `float32[1024,256]`（262144 元素，1,048,576 B） | `x+1` / `x-1` | eager/jit × true/false | match_cpu |
| `cond_invalid_predicate` | 错误分类 | `bool[1]`、`bool[2]`、`int32(0)`、`int32(1)` | `x+1` / `x-1` | eager/jit × 4 variant | `INVALID_PREDICATE` |
| `cond_branch_structure_mismatch` | 错误分类 | `tuple(f32[4],f32[4])` vs `dict(x,y)` | `struct_true/false` | eager/jit | `BRANCH_STRUCTURE_MISMATCH` |
| `cond_branch_shape_mismatch` | 错误分类 | `f32[4]` vs `f32[5]` | `shape_true/false` | eager/jit | `BRANCH_SHAPE_MISMATCH` |
| `cond_branch_dtype_mismatch` | 错误分类 | `f32[4]` vs `int32[4]` | `dtype_true/false` | eager/jit | `BRANCH_DTYPE_MISMATCH` |

> 另有 `tests/jax_lax_cond_supplement_cases.json`（5 条 / 18 mode：S1 weak_type、S2 0-d pred、
> S3 list pytree、S4 grad 两分支 trace、S5 NaN/Inf）与 `tests/jax_lax_cond_whitebox_cases.json`
> （6 条 / 12 mode：tile 8192、tile 尾 8193、64 tiles grid-stride、跨步源 8192 上界、jax dict
> pytree、8 叶上界）。两者在同一 pytest 入口执行，**不进入冻结 `cases.json`**，不改变
> `suite_ref` 与 16+2 的结论。

### A.2 性能用例（`cases/jax_lax_cond/benchmark.json`，2 条 / 4 序列）

| Case ID | operand | 分支 | 序列 | 实测 median（设备侧 Task Duration） | 备注 |
|---|---|---|---|---|---|
| `cond_small_branch_overhead` | `float32[16]` | `x+1` / `x-1` | true / false | true `4.660093 us`；false `4.390088 us` | 小分支开销；block_dim=1 |
| `cond_large_tensor_throughput` | `float32[1024,256]` | `x*1.125+0.5` / `x*0.875-0.5` | true / false | true `7.460150 us`；false `7.440149 us` | **primary**；block_dim=32 |

> 测量条件：run #229、设备 5、exclusive、镜像 #145、`warm_up=3`、`invocations=10`、median、
> `block_until_ready`；`performance_target=null`，仅基线，不设阈值。

## 附录 B：参考资料

- JAX 0.8.1 `jax.lax.cond` 官方语义（scalar predicate、两分支 trace、pytree 一致性、weak type）；
- `tasks/jax_lax_cond/spec/spec.json`（spec v1，`spec_version_id=6`）与
  `tasks/jax_lax_cond/notes/clarifications.md`（PM 代用户确认的三项：CPU oracle / 四码归一化 /
  性能仅基线；int32 0/1 拒绝；c11 17/33 元素；design.md/test 必须交付）；
- `0-黑盒测试设计.md` §1.3（候选接口契约）、§2（golden/精度口径）、§5（L0/L1/L2 用例）；
- `3-白盒测试设计.md` §2（源码分支清单）与 `4-白盒设计勘误.md`（K-13/PY-03/H-07 行号校正）；
- `4-修复记录.md`、`4-测试执行记录.md`、`4-验收报告.md`、`4-清单.json`（当前交付证据）；
- 本仓 `ops/jax_lax_scan` / `ops/elementwise_sub` 的 Ascend C 直调工程布局与
  `ops/jax_lax_cond/README.md`；
- `docs/reference/deliverable-standards.md`（五件交付物与 README 六项 / design 五段要求）。
