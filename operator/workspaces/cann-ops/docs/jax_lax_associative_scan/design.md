# jax_lax_associative_scan 算子设计文档（AscendC 实现）

> 目标仓：`/Users/huzhengyu/Desktop/工作文件/模型与算子开发智能体/op-agent/workspaces/cann-ops`
> 目标芯片：Ascend 910B1（`dav-2201`）；运行时：JAX/JAXLIB 0.8.1 + xla-npu/AFIR（镜像 #145）
> 规格：`tasks/jax_lax_associative_scan/spec/spec.json`（spec v1，spec_version_id 5）
> 用例：`cases/jax_lax_associative_scan/{cases.json,benchmark.json}`（按 `scripts/nm_test.sh` 同算法重算
> `cases@6d6bc47fa93738dc`（op-scoped 门禁标签；同一冻结内容按整棵 `cases/` 复算为
> `cases@b852239bc7476550`，与 prompt/需求分析引用一致）；控制台 `cases_confirmed_sha256=6d6bc47f…`
> 与全部四动作 run 清单一致）
> 判据：`criteria/check.py`（`ops-precision-standard@1`）

### 修订记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1 | 2026-10-27 | 节点 2 首版：与已确认规格、黑盒方案、测试工程契约对齐 |
| v1.1 | 2026-10-27 | 节点 2 恢复执行：按冻结文件与控制台确认值修正 `suite_ref` 为 `cases@6d6bc47fa93738dc`（该行对 `cases@b852239bc7476550` 的「上游笔误」判断已由 v1.2 更正）；同步四动作重跑结果与错误路径资源释放说明 |
| v1.2 | 2026-10-27 | 文档准备：按仓内同类算子（`jax_lax_scan`）文档规范补全/校正——§5.4 pytest 用例计数改为修复后实际的 27 身份 / 48 参数化项（+11 自检 +12 白盒分支 = 71 项）、§5.6 验证口径补 `#99/#100/#101` 实测结果、§5.2/§3.4 性能证据状态改为当前快照 `UNKNOWN`（历史两次 benchmark 快照不匹配）、§1.4/§2.2/§5.5 去掉节点内措辞、附录 A.3 补白盒用例范围；并更正 v1.1 的判断：`cases@b852239bc7476550` 是**整棵 `cases/`** 复算值（可复现，非上游笔误），op-scoped 门禁标签为 `cases@6d6bc47fa93738dc`，两者对应同一冻结内容。仅改本文件，不动实现 |

---

## 1. 需求背景

### 1.1 需求来源

JAX 生态在 Ascend 上的算子能力补齐：`jax.lax.associative_scan` 是并行前缀扫描原语，被
`jax.lax.cumsum`/`cummax`、linear attention、状态空间模型等大量使用。本任务在
Ascend 910B1 + xla-npu/AFIR 环境按 CANN Bench 直调工程交付该能力，并以同版本 JAX CPU 为 oracle。

### 1.2 业务背景与使用场景

* 前缀/后缀归约：`sum`、`max` 沿静态 axis 的前缀结果；
* pytree 结构：多个叶子共享同一扫描长度、逐叶子独立扫描；
* 非 2 次幂长度、`length=1`、`reverse`、eager/jit 两种调用形态；
* JVP/VJP：自动微分边界必须有可审计证据，后端不支持时显式 `UNSUPPORTED`，不得伪报 PASS。

### 1.3 算子数学语义

对任一叶子，令 `axis` 前维乘积 `outer = prod(shape[:axis])`、`L = shape[axis]`、
`inner = prod(shape[axis+1:])`。输入可视为 `[outer, L, inner]`，扫描是 `outer × inner` 条独立链：

* `reverse=False`：`out[o, i, k] = x[o,0,k] ⊕ … ⊕ x[o,i,k]`；
* `reverse=True`：`out[o, i, k] = x[o,L-1,k] ⊕ … ⊕ x[o,i,k]`（后缀）；
* `⊕` 本批注册 `sum`（`a+b`）、`max`（`maximum`）、`pytree_add`（逐叶子 `+`）；
* pytree 输出树结构与输入完全一致；各叶子 `shape[axis]` 必须相等，其余维可不同。

### 1.4 历史实现与现状

* 目标仓已有 `ops/jax_lax_scan`：JAX `lax.scan` 直调实现（kernel + host C ABI + CLI + pytest），
  本算子复用其工程骨架、四动作入口、`.so` 指纹与 host-driver 协议；
* `jax_lax_scan` 的语义是「逐步 carry 循环」，本算子是「沿 axis 的前缀归约」，几何抽象与
  kernel 数据流不同，不能直接复用其 kernel；
* `cases/`、`criteria/`、`scripts/golden*.py`、`scripts/fixtures.py`、`scripts/assoc_scan_engine.py`
  由测试工程交付并已通过验收；本算子实现不改其语义，只按既有契约接入。

---

## 2. 需求分析

### 2.1 外部组件依赖

| 组件 | 版本/路径 | 用途 |
|---|---|---|
| CANN / AscendC | 镜像 #145（`opx-xla-npu:jax081-cann920-910b-test`） | kernel 编译、ACL runtime |
| Ascend 910B1 | device 0，独占 | kernel 执行 |
| JAX / JAXLIB | 0.8.1（CPU 子进程） | oracle |
| xla-npu/AFIR | 镜像 #145 | JAX trace 路径（AD）的 NPU 后端 |
| numpy | 镜像运行时版本 | host driver I/O、输入生成 |
| pytest | 镜像运行时版本 | 交付件入口（`test.sh` 第二段） |

### 2.2 内部适配模块

| 模块 | 文件 | 职责 |
|---|---|---|
| Python 接口 | `python/jax_lax_associative_scan.py` | 校验、pytree、ctypes 调用、路由与自省 |
| host driver | `python/jax_lax_associative_scan_cli.py` | 读 case/inputs、写 outputs/meta、EXEC/REJECT 协议 |
| C ABI | `op_host/jax_lax_associative_scan_pyapi.asc` | 参数校验 + Tiling + kernel launch、malloc/h2d/d2h |
| kernel | `op_kernel/jax_lax_associative_scan_kernel.asc` | 真正的前缀/后缀扫描计算 |
| 可执行入口 | `op_host/jax_lax_associative_scan.asc` | argv 协议校验并转交 host driver |
| 测试执行器 | `scripts/assoc_scan_engine.py`（测试工程交付） | 门禁/冒烟的候选调用、meta/EXEC 校验、golden 子进程、逐叶子判定 |

### 2.3 需求模块设计

1. **调用面分离**：Python 接口承接 JAX API 形态（pytree、callable、reverse、axis）；门禁 CLI
   承接 `--case-json/--inputs/--mode/--outdir` 固定协议。callable 在 Python 层解析为
   `combine_id`，C ABI 只接收整数 id 与设备地址。
2. **设备真实性**：numpy family 的数值全部由 `.so` → ACL → AscendC kernel 计算；Python 层只做
   参数校验、设备缓冲管理与 I/O，不做扫描计算（编码规则 A1/A2）。
3. **两种执行形态**：
   * `eager`/`associative_scan`：每叶子一次 `.so` 调用（`exec_mode=per_leaf`，
     `abi_calls=叶子数`）；
   * `jit`/`associative_scan_compiled`：整树一次 `.so` 调用（`exec_mode=whole_tree`，
     `abi_calls=1`），一次 kernel launch 覆盖全部叶子。
4. **AD 路径**：host-driver 用 kernel 计算（`path=so`）；JAX 数组/tracer 用 trace 路径
   （`path=trace`，`abi_calls=0`）。两条路径的结论分开记录，不互相冒充。
5. **拒绝前置**：dtype/axis/结构/连续性错误在任何 device 操作前拒绝，`abi_calls=0`。

### 2.4 算子原型

#### 2.4.1 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| `elems` | pytree（numpy / jax.Array） | 叶子 `float32`、C-contiguous、rank>=1；各叶子 `shape[axis]` 相等 |
| `fn` | Python callable | 已注册 combine（`combine_id` ∈ {1,2,3}） |
| `reverse` | bool | 默认 `False` |
| `axis` | int | 默认 `0`；`0 <= axis < rank` |

#### 2.4.2 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| `out` | pytree | 与输入同结构/shape/dtype；新结果，不与输入共享存储 |

#### 2.4.3 属性 / 参数

* `combine_id`：1=sum、2=max、3=pytree_add；
* `reverse`：0/1；
* `dtype`：本批只 0=float32（逐叶子）；
* 几何：`outer/length/inner` 逐叶子；
* `device_id`：0（单卡）。

#### 2.4.4 推导规则

* `outer = prod(shape[:axis])`、`length = shape[axis]`、`inner = prod(shape[axis+1:])`；
* 输入按 `[outer, length, inner]` 连续解释；输出同 shape；
* `reverse` 只改变扫描方向，不改变 shape/dtype/树结构；
* pytree 逐叶子独立推导，只有 `length` 必须一致。

---

## 3. 详细设计

### 3.1 使能方式

* 目标仓根 `build.sh` 按 `$OP` 找到 `ops/$OP`，`cmake ... && make $OP`；
* `CMakeLists.txt` 产出两个 target：
  * `jax_lax_associative_scan`（可执行文件，四动作入口固定路径 `build/jax_lax_associative_scan`）；
  * `jax_lax_associative_scan_host`（共享库 `libjax_lax_associative_scan_host.so`，Python C ABI）；
* `build.sh` 不需要为新算子改动（与算子名无关）。

### 3.2 总体设计方案

#### 3.2.1 Host 侧

1. `op_host/jax_lax_associative_scan.asc`：
   * 校验 `--case-json/--inputs/--mode/--outdir` 与配对值；未知 key 直接 rc=2；
   * `readlink("/proc/self/exe")` 定位交付目录，`execvp python3` 转交
     `python/jax_lax_associative_scan_cli.py`；
   * `--self-check` 只做 `dlopen(.so)` + ABI 版本读取，不跑用例。
2. `python/jax_lax_associative_scan_cli.py`：
   * 读 `case.json`（含 `_resolved` 元数据）与 `inputs.npz`，按 `elems_tree` 还原 pytree；
   * 调用 Python 接口；成功写 `outputs.npz` + `meta.json`，stdout 首行 `EXEC`；
   * 捕获 `_AssocReject` → rc=3，写 `status="reject"` 的 meta，不写 `outputs.npz`；
   * 捕获包含 `UNSUPPORTED:autodiff_unsupported` 的异常 → rc=4，`EXEC path=trace abi_calls=0`；
   * `inputs_sha256` 与 `leaves[].sha256` 用测试工程冻结的同一算法（键序/dtype/shape/字节）；
   * `so_fingerprint` 来自 Python 接口实际 `dlopen` 的库。
3. `python/jax_lax_associative_scan.py`：
   * `_analyze` 在 device 前完成全部校验并生成几何；
   * numpy family：`acl_init → malloc → h2d → jax_lax_associative_scan_run → d2h → free`；
     per_leaf 循环 N 次，whole_tree 一次传 N 个叶子；
   * jax family/tracer：trace 路径（见 3.2.3）；
   * `_last_dispatch` 记录 `path/exec_mode/abi_calls/combine_id/family/backend/device_id/
     plan_id/plan_cache/reason`；`_plan_dispatch` 是纯路由。
4. `op_host/jax_lax_associative_scan_pyapi.asc`：
   * 校验 ABI 版本、combine、reverse、叶子数、dtype、几何、numel、设备地址；
   * 组 `AssocTilingData`，`aclrtMalloc + Memcpy H2D`，单核 launch；
   * `aclrtSynchronizeStream` 后返回；`aclInit` 只做一次，`aclrtSetDevice(device_id)`；
   * 导出 `run/abi_version/acl_init/malloc/free/h2d/d2h`，不 `aclFinalize`（生命周期归调用方）。

#### 3.2.2 Kernel 侧

`op_kernel/jax_lax_associative_scan_kernel.asc`：

```
for leaf in 0..nLeaves-1:                       // whole_tree 一次 launch 覆盖多叶子
  for chunkStart in 0..inner step chunkCap:     // inner 切块；块间是独立链
    for o in 0..outer-1:
      carry = first element (按 reverse 取 L-1 或 0)
      for t in 0..L-1:
        idx = reverse ? L-1-t : t
        x = DataCopyPad(GM row [idx])            // MTE2 → UB
        y = (t==0) ? x : combine(carry, x)       // Add / Max
        carry = y
        DataCopyPad(GM row [idx] ← y)            // UB → MTE3
```

* 单核（`blockNum=1`）：carry 必须跨扫描步保持在 UB，不能跨核切分一条链；
* 三级缓冲：`TQue<VECIN>`（输入行）、`TBuf<VECCALC>`（carry）、`TQue<VECOUT>`（输出行）；
  `EnQue/DeQue` 负责 MTE2→V、V→MTE3 同步；
* `DataCopyPad` 处理非 32B 对齐的行宽（如 `inner=257`、`inner=1`）；UB 起始地址始终是
  buffer 基址（32B 对齐），行内不做非对齐 UB 偏移；
* `reverse` 时按原始索引写回，等价于「先反转、扫描、再反转」；
* `chunkCap = min(maxInner, 2048)`，UB 只按该上限分配（3 × 8KB = 24KB），更大的 `inner`
  由 chunk 循环覆盖；
* `nLeaves=1..8`（`ASSOC_MAX_LEAVES`，只约束**单次 launch**）；`per_leaf` 每次 launch
  只带 1 个叶子，pytree 叶子总数不受 8 限制；`whole_tree` 一次带全部叶子，超过 8 个叶子
  由 `_analyze(whole_tree=True)` 在任何 device 操作之前按 `invalid_argument` 稳定拒绝。

#### 3.2.3 AD 路径

* **host-driver（`--mode jvp|vjp`）**：kernel 路径。AD 定义与 golden 一致（seed 全 1）：
  * JVP：`primal = scan(x)`，`tangent = scan(ones)`（同 reverse）；
  * VJP：`primal = scan(x)`，`cotangent = scan(ones)`（reverse 翻转）；
  * 两次 `.so` 调用（`path=so`、`exec_mode=jvp|vjp`、`abi_calls=2`）。
* **JAX 数组/tracer**：trace 路径。用 `jnp.take` 逐位置取切片、`jnp.stack` 拼回，按
  elementwise combine 展开前缀扫描，由 JAX 自动微分穿过；不依赖
  `lax.associative_scan` 在 xla-npu 后端的 lowering。`path=trace`、`abi_calls=0`。
* 若 trace 路径失败：仅当异常是**非 CPU 后端 runtime 错误**且带 AFIR / lowering /
  未实现签名时，才包装为 `UNSUPPORTED:autodiff_unsupported` + 后端原文，
  并置 `path=trace`/`abi_calls=0`/`backend` 非 CPU；3 次同码才允许 `UNSUPPORTED`，
  且不记 PASS。实现侧 shape/编程异常保留原始类型与文本抛出，不得伪装成 AD 不支持。

### 3.3 支持硬件

* Ascend 910B1（`dav-2201`），单卡 device 0；
* UB 预算：3 × 8KB（chunkCap=2048 float）= 24KB，远小于 910B 的 UB；
* 不依赖 Cube/Matmul，纯 Vector + MTE2/MTE3。

### 3.4 算子约束与限制

| 约束 | 说明 |
|---|---|
| dtype | 仅 float32；其他 dtype 在 device 前拒绝 |
| axis | 静态、非负、`< rank`；负 axis 不支持 |
| rank | `>= 1`；rank-0 拒绝 |
| 结构 | 单叶子 / tuple / list / dict；空树、叶子类型混用拒绝 |
| 长度 | `L >= 1`；非 2 次幂、`L=1` 支持 |
| pytree | 各叶子 `shape[axis]` 必须相等 |
| 设备 | 单卡、单核；多设备/SPMD 不支持 |
| 特殊值 | 不定义 NaN/Inf/±0/次正规数的专门语义 |
| 性能 | 不设性能阈值（`performance_target=UNKNOWN`）；当前快照未采集 benchmark/profile，无当前版本性能结论，见 §5.2 |

---

## 4. 特性交叉分析

| 维度 | 设计结论 |
|---|---|
| dtype | 仅 float32；fp16/bf16/int 走拒绝（拒绝依据是已确认 spec，不是 oracle 报错） |
| 动态 shape | `outer/length/inner` 运行时由 tiling 下发；同一 kernel 覆盖 `L=1`、非 2 次幂、大 `inner`、rank 1–3 |
| 非对齐 | 行宽不是 32B 倍数时用 `DataCopyPad`；UB 基址 32B 对齐 |
| reverse | 只改扫描取步方向与写回索引，不改几何；eager/jit 输出一致 |
| pytree | 逐叶子独立扫描；`whole_tree` 一次 launch 覆盖多叶子；`per_leaf` 每叶子一次 ABI 调用 |
| eager/jit | 由 Python 入口区分；输出必须彼此一致（测试做 eager×jit 互比） |
| AD | host-driver 走 kernel AD；jax.Array/tracer 走 trace；两条路径的证据分开记录 |
| 多流/并发 | 每次调用独立 ACL stream，kernel 后同步；ACL 生命周期交给调用方 |
| 图模式/量化 | 不适用（直调二进制、float32 only） |
| 风险 | 单核串行扫描，`L=1024` 时性能有限；不设性能阈值，当前快照未采集 benchmark（见 §5.2） |
| 风险 | `Max` 对 NaN 的行为未定义专门语义；测试输入不含 NaN/Inf |
| 风险 | xla-npu 对 trace 路径的 take/stack/AD 支持需真机证据；不支持时按 3.2.3 显式 UNSUPPORTED |

---

## 5. 可维可测分析

### 5.1 精度标准

* 结构/叶子路径集合/每叶子 shape/dtype：精确相等，无容差；
* 浮点：`criteria/check.py`（`ops-precision-standard@1`），
  `matched_ratio >= 0.99` 且 `max_abs_error <= 1e-2`，元素阈 `atol + rtol*|golden|`
  （`rtol=2^-10=9.765625e-4`、`atol=2^-16=1.52587890625e-5`）；
* 非有限位逐元素精确；pytree 与 AD 的每个输出叶子独立判定；
* oracle：同版本 JAX CPU 子进程（`JAX_PLATFORMS=cpu`），父进程只跑候选，不互相污染。

### 5.2 性能标准

* `blocking_jit_execution`；`warm_up=3`、`invocations=10`、`aggregate=median`、
  `synchronize=block_until_ready`；
* 用例：`assoc_sum_len256_width256`、`assoc_sum_len1024_width256`；主用例 1024×256；
* `performance_target=UNKNOWN`：只报告同卡同镜像的设备侧 `Task Duration`，不设通过阈值；
* 不设 CPU / torch 基线（CPU 延迟不是门禁）；
* **当前证据状态**：`#99/#100/#101`（快照 `cf080296…`）只覆盖 build/smoke/test，**未执行**
  `--action benchmark`/`profile`，当前版本性能结论 = `UNKNOWN`。历史采集 #81（快照 `e88deade…`，
  87.642 / 307.676 us）与 #85（快照 `605ce734…`，88.082 / 308.766 us）均不对应当前代码，
  仅用于说明口径，不得作为当前版本结论；重跑须同设备、同镜像、同规格、同 `suite_ref`。

### 5.3 兼容性分析

* JAX `lax.associative_scan` 的参数顺序、默认值（`reverse=False, axis=0`）与返回结构对齐；
* `associative_scan_compiled` 是测试/门禁的编译态入口，不是第二个公开 API；
* 不支持的 dtype/axis/结构在 device 前拒绝，拒绝码稳定、可机判；
* 与 `jax_lax_scan` 共用工程骨架与四动作入口，但 kernel/ABI/CLI 独立，互不影响。

### 5.4 可观测性与可测试性

* `_last_dispatch` 暴露路径、执行形态、ABI 调用次数、combine、family、backend、device_id、
  plan_id/plan_cache；`_plan_dispatch` 可在 CPU 上单测路由；
* 门禁 CLI 的 `EXEC`/`REJECT` 行 + `meta.json`（含 `so_fingerprint`）可逐项核对；
* pytest 交付件：`tests/test_jax_lax_associative_scan.py`，用例 = 冻结 7 条 + benchmark 2 条 +
  补充 11 条 + 白盒 7 条（`tests/jax_lax_associative_scan_whitebox_cases.json`），共
  **27 条身份 / 48 个 `(case_id, mode)` 参数化项**；另有 11 项自检与 12 个白盒分支用例
  （CPU 路由/校验 + 设备侧 trace/数值），合计 **71 项**（#101 实测：`tests=71, failures=0,
  errors=0, skipped=2`，`69 passed, 2 xfailed`）；
* 拒绝用例：`assoc_l2_dtype_f16`、`assoc_l2_axis_oob`、`assoc_l2_axis_rank1`，要求 rc=3、
  稳定码、`abi_calls=0`、无 `outputs.npz`；
* AD 用例 `assoc_grad_sum`：PASS 或 3 次同码 `UNSUPPORTED`；`UNSUPPORTED` 不记 PASS；
* 失败证据保留：`runs/<run_id>/out/` 下的 `correctness.json`、`pytest_*.log/xml`、
  `smoke.json`、`evidence/`。

### 5.5 可维护性

* 目录边界清晰：`ops/jax_lax_associative_scan/`（算子）、`cases/`（冻结用例，只读）、
  `tests/`（pytest）、`docs/`（设计）；
* tiling 常量集中在 `op_kernel/*_tiling.h`，kernel/host/pyapi 三方共用；
* ABI 布局版本 1（456 字节），Python 侧 `ctypes.sizeof` 自检；字段只追加、不改语义；
* `build.sh`/`test.sh`/`tools/*.py` 的新算子分派沿用既有向后兼容扩展（已回归旧算子路径）；
  本算子交付只新增 `ops/jax_lax_associative_scan/`、`tests/`、`docs/jax_lax_associative_scan/`；
* license 头统一保留。

### 5.6 验证口径

| 动作 | 入口 | 判据 |
|---|---|---|
| build | `scripts/nm.sh --op jax_lax_associative_scan --action build` | rc=0，`build/jax_lax_associative_scan` 与 `.so` 存在，`BUILD_OK` |
| smoke | `--action smoke` | 最小用例 `assoc_sum_len1`（jit）PASS |
| test | `--action test` | `test.sh` 两段：`run_cases.py` 7 条 / 14 次全 PASS；pytest 71 项（48 参数化 + 11 自检 + 12 白盒分支），AD 允许 3 次同码显式 `UNSUPPORTED` 且不记 PASS |
| benchmark | `--action benchmark` | 只报告设备侧中位数，不设通过阈值（`performance_target=UNKNOWN`） |

**当前版本实测**（快照 `cf080296…`、`cases@6d6bc47fa93738dc`、镜像 #145、device 0 独占）：

| 动作 | run | 结果 |
|---|---|---|
| build | `r20261027-132336` / #99 | `exit=0`，`BUILD_OK`，gate `pass` |
| smoke | `r20261027-132421` / #100 | `SMOKE_VERDICT PASS`（`assoc_sum_len1`/jit），gate `pass` |
| test | `r20261027-132509` / #101 | 门禁 7/7、14/14 PASS（0 FAIL/ERROR/INVALID）；pytest `69 passed, 2 xfailed`（`tests=71`）；白盒 48 次 = 46 PASS + 2 显式 UNSUPPORTED |
| benchmark | 未执行 | 当前快照无性能结论（历史 #81/#85 快照不匹配，见 §5.2） |

> 原始证据：`.cannbot/jax_lax_associative_scan/workflow1/4-证据-r1/runs-99|100|101/`；
> 2 个 `UNSUPPORTED` 不记 PASS；本文件为 `.md`（不在 `ops/` 内），不影响 `operator-v1` 快照
> `cf080296…`，故上述证据仍对应当前算子代码。

---

## 附录 A：测试用例要点

### A.1 冻结正确性用例（`cases/jax_lax_associative_scan/cases.json`，7 条 / 14 次）

| case_id | op | shape | axis | reverse | modes |
|---|---|---|---|---|---|
| `assoc_sum_len8_axis0` | sum | [8,16] | 0 | false | eager/jit |
| `assoc_max_len7_axis0` | max | [7,5] | 0 | false | eager/jit |
| `assoc_reverse_sum_len9` | sum | [9,4] | 0 | true | eager/jit |
| `assoc_pytree_add_len6` | pytree_add | dict a[6,4], b[6,2] | 0 | false | eager/jit |
| `assoc_sum_axis1` | sum | [3,8,4] | 1 | false | eager/jit |
| `assoc_sum_len1` | sum | [1,16] | 0 | false | eager/jit |
| `assoc_grad_sum` | sum | [5,4] | 0 | false | jvp/vjp |

### A.2 补充用例（`tests/jax_lax_associative_scan_supplement_cases.json`，8 L1 + 3 L2）

* L1：max×axis1×reverse（rank3）、sum `1000×257`、pytree reverse、rank1 `L=5`、
  axis1 `L=1`、pytree axis1、axis1 rank2、max axis0 rank3；
* L2：f16 `dtype_mismatch`、axis 越界 `invalid_axis`（rank2/rank1）。

### A.3 白盒用例（`tests/jax_lax_associative_scan_whitebox_cases.json`，6 条数值 + 1 条拒绝）

* 覆盖 `inner` 切块边界（`inner=2048`、`inner=2049` reverse、`inner=2500` max、`outer=2` axis1）、
  混合 `inner` 的 pytree、8 叶子整树 eager/jit、per_leaf 越界 axis 拒绝；
* 另有 12 个白盒分支用例在 `tests/test_jax_lax_associative_scan.py` 内（CPU 路由、ABI 校验、
  argv 协议、trace 异常分类、9 叶子上限、设备侧 trace 数值等），不进入上面的用例参数化计数。

### A.4 基准用例（`cases/jax_lax_associative_scan/benchmark.json`，2 条）

| case_id | length | width | axis | reverse | op |
|---|---|---|---|---|---|
| `assoc_sum_len256_width256` | 256 | 256 | 0 | false | sum |
| `assoc_sum_len1024_width256` | 1024 | 256 | 0 | false | sum |

---

## 附录 B：参考资料

* `tasks/jax_lax_associative_scan/spec/spec.json`（spec v1）
* `cases/jax_lax_associative_scan/{cases.json,benchmark.json}`（`cases@6d6bc47fa93738dc`）
* `$WORK_DIR/0-黑盒测试设计.md`（§1.3 接口契约、§2 判据、§5 用例）
* `$WORK_DIR/1-测试工程说明.md`（§5 执行契约、§6 精度口径）
* `criteria/check.py`、`criteria/ops-precision-standard@1/`
* `docs/jax_lax_scan/design.md`（工程骨架与设计结构参考）、`ops/jax_lax_scan/README.md`（文档格式参考）
* AscendC API best practices：`DataCopyPad`/`DataCopyExtParams`、`TQue/TBuf`、`Add/Max/Adds`
