# jax_lax_cond · JAX `lax.cond` Ascend C 直调实现

JAX `lax.cond` 在 Ascend 910B1（`dav-2201`）上的直调实现：Python 调用接口 → C ABI
（`libjax_lax_cond_host.so`）→ Ascend C kernel（numpy family 数值路径）；jax family 的
trace / 自动微分走 JAX 的 `xla-npu` 后端。参考 oracle 为**同版本 JAX 0.8.1 CPU**
（`ops/jax_lax_cond/scripts/golden.py`，独立子进程）。

- 规格真相源：`tasks/jax_lax_cond/spec/spec.json`（spec v1，`spec_version_id=6`）。
- 冻结用例：`cases/jax_lax_cond/{cases.json,benchmark.json}`（16 条正确性 / 2 条性能，
  `suite_ref=cases@233778f456b4eb9b`）。
- 判据：`criteria/check.py`（`ops-precision-standard@1`）。
- 设计文档：`docs/jax_lax_cond/design.md`。

## 目录

```text
op_kernel/jax_lax_cond_tiling.h       Tiling 常量/结构体（kernel、host、pyapi 三方共用）
op_kernel/jax_lax_cond_kernel.asc     AscendC kernel（逐输出叶向量原语 + token/counters）
op_host/jax_lax_cond.asc              可执行入口（四动作门禁，argv 协议 + 转交 host driver）
op_host/jax_lax_cond_pyapi.asc        Python 接口共享库（extern "C" C ABI）
op_host/jax_lax_cond_abi.h            C ABI 结构体（布局 472 字节）
python/jax_lax_cond.py                Python 调用接口（ctypes → libjax_lax_cond_host.so）
python/jax_lax_cond_cli.py            host driver（--case-json/--inputs/--mode/--variant/--outdir）
scripts/golden.py                     **独立** CPU oracle（同版本 JAX，独立子进程）
scripts/golden_cli.py                 oracle 子进程入口（JAX_PLATFORMS=cpu）
scripts/fixtures.py                   注册分支词表 / 输入 pytree / 用例展开（测试侧共用）
scripts/cond_engine.py                门禁 cond 分支编排（用例加载、候选调用、判据、证据）
scripts/cond_bench.py                 基准分支（被 tools/run_benchmark.py 按 op 分派）
CMakeLists.txt                        两个 target：可执行文件 + 共享库 + 构建指纹
```

---

## 1. 语义与支持范围

```python
result = jax.lax.cond(pred, true_fun, false_fun, *operands)

# 数学语义：
#   result = true_fun(*operands)    if pred is True
#   result = false_fun(*operands)   otherwise
```

语义与 JAX 0.8.1 `jax.lax.cond` 对齐；本批不以 torch 同名算子作为 oracle 或语义依据。

- `pred` 必须是**严格 scalar bool**（Python `bool`、`np.bool_`、0-d bool 数组，或可静态求值的
  0-d `jax.Array(bool)`）；`bool[1]`、`bool[2]`、整型 `0/1` 均不做隐式提升；
- **两个分支都参与 trace**：每次调用都真实访问 `true_fun` 与 `false_fun`，比较两者输出的
  pytree 结构、逐叶 shape、dtype 与 `weak_type`；
- **运行期只执行选中分支**：numpy family 只下发选中 `branch_id` 的 kernel；jax family 只调用
  选中分支的 callable；
- 结构/shape/dtype 不一致分别拒绝为 `BRANCH_STRUCTURE_MISMATCH` /
  `BRANCH_SHAPE_MISMATCH` / `BRANCH_DTYPE_MISMATCH`，并保留上游异常类型与消息 sha256；
- 不做广播；不原地修改输入、输出不与输入别名；NaN/±Inf 逐元素传播；`pred` 不可导。

### 已注册分支（numpy family 的门禁路径）

| `branch_id` | 名称 | 语义 | 可下发 kernel |
|---|---|---|---|
| 1 | `x+1` | `x + 1` | 是 |
| 2 | `x-1` | `x - 1` | 是 |
| 3 | `x*2` | `x * 2` | 是 |
| 4 | `x/2` | `x / 2` | 是 |
| 5 | `tree_add` | pytree 逐叶 `+1` | 是 |
| 6 | `tree_sub` | pytree 逐叶 `-1` | 是 |
| 7 | `out_tree_true` | `(x+1, {"aux": x[:,0]+1})` | 是（1 输入 / 2 输出） |
| 8 | `out_tree_false` | `(x-1, {"aux": x[:,0]-1})` | 是（1 输入 / 2 输出） |
| 9 | `grad_true` | `2*x` | 否（仅 JAX trace/grad） |
| 10 | `grad_false` | `3*x` | 否（仅 JAX trace/grad） |
| 11 | `list_inc` | list 逐叶 `+1` | 是 |
| 12 | `list_dec` | list 逐叶 `-1` | 是 |
| 13 | `p02_true` | `x*1.125 + 0.5` | 是 |
| 14 | `p02_false` | `x*0.875 - 0.5` | 是 |

> numpy family 的 `true_fun`/`false_fun` 必须是上表已注册 callable（按 `__name__` 或显式
> `branch_id` 解析）；未注册 callable 需改用 jax family 的 operands 走 JAX trace，不会冒充
> `.so` 设备执行（numpy family 直接显式 `invalid_argument`）。

### 支持范围（本批）

| 项 | 范围 | 越界行为 |
|---|---|---|
| `pred` | 严格 scalar bool（numpy family 门禁路径） | 整型 `0/1`、`bool[1]`、`bool[2]` → `INVALID_PREDICATE`（整型按已确认规格规则拒绝，CPU JAX 实际接受，原始行为单独留证；jax family 的 pred 契约见 §6） |
| operands | pytree；输入叶子 1…8；numpy family 叶子 C-contiguous | 空 pytree / >8 叶 → `invalid_argument` |
| dtype | 数值叶 float32；int32 只用于“分支 dtype 不一致”的拒绝用例 | 其它 dtype → `invalid_argument`（numpy family） |
| 两分支输出 | tree structure / 逐叶 shape / dtype / weak_type 完全一致 | 四码拒绝；`abi_calls=0` |
| 模式 | `eager` / `jit` / `grad_eager` / `grad_jit` | — |
| 设备 | 910B1 单卡（宿主设备 5 / 容器 `device_id=0`） | 多设备 / SPMD 不支持 |
| grad | 浮点单叶、逐元素分支；`grad_jit` 只承诺扁平 `[N]` | 2-D jit grad → `device_error`；多叶 pytree grad 未验证 |
| 后端限制 | outer `jax.jit` tracer pred、stock `lax.cond` 在 jit 内的 conditional | `BACKEND_LIMITATION`，不计 PASS（见 §6） |

---

## 2. 调用接口

### 2.1 Python 接口

```python
cond(pred, true_fun, false_fun, *operands) -> pytree
cond_compiled(pred, true_fun, false_fun, *operands) -> pytree
```

- `cond`：eager 语义，每次调用重新 trace/校验；
- `cond_compiled`：测试/门禁的**编译态入口**；每次调用都真实 trace 两分支，首次建立 plan，
  后续同 plan 命中缓存（`compile_count` 不增、`plan_cache="hit"`）；不是第二个公开 JAX API，
  **不得**在 `jax.grad` / `jax.jit` trace 内使用；
- 返回新 pytree（numpy family 返回新 `numpy.ndarray`；jax family 返回 `jax.Array`）；
- 失败抛 `CondError`（`code`、`upstream_exception_type`、`upstream_message`、
  `upstream_message_sha256`）；`invalid_argument`/`device_error`/`internal_error` 同样走
  `CondError`，但不会被当成四码拒绝通过。

以下示例在工作区根（`cann-ops/`）执行：

**jax family（通用 callable，走 JAX trace）**：

```python
import sys
sys.path.insert(0, "ops/jax_lax_cond/python")
import jax.numpy as jnp
from jax_lax_cond import cond

x = jnp.ones((8, 16), jnp.float32)
y = cond(True, lambda v: v + 1.0, lambda v: v - 1.0, x)  # 需要 NPU backend
```

**numpy family（门禁用例路径，走 `.so` + AscendC kernel）**：

```python
import sys
import numpy as np
sys.path[:0] = ["ops/jax_lax_cond/python", "ops/jax_lax_cond/scripts"]
import jax_lax_cond as iface
import fixtures

x = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
y = iface.cond(True, fixtures.branch_fn("x+1"), fixtures.branch_fn("x-1"), x)
```

> 以上示例需要先完成 `scripts/nm.sh --op jax_lax_cond --action build` 并具备 NPU 环境；
> 本仓的 jax family 在 CPU 后端会按设备守卫显式报 `device_error`，不会用 CPU 结果冒充 NPU。

**自省/证据接口**：`_last_dispatch`（最近一次路由与证据）、`_reset_evidence()`（清空计数，
不清 plan 缓存）、`_plan_dispatch(...)`（纯 CPU 路由，不执行 device）、`library_fingerprint()`
（运行期实际加载的 `.so` 路径/大小/sha256/ABI）、`abi_version()`。

### 2.2 四动作可执行入口（门禁 CLI）

```bash
ops/jax_lax_cond/build/jax_lax_cond \
  --case-json <case.json> --inputs <inputs.npz> \
  --mode <eager|jit|grad_eager|grad_jit> [--variant <name>] --outdir <dir>
```

- 可执行文件只做 argv 校验并 `execvp` 到 `python/jax_lax_cond_cli.py`；数值在
  `libjax_lax_cond_host.so` → AscendC kernel（或 AD 的 NPU trace）内完成；
- `case.json`/`inputs.npz` 由共享 runner 的 cond 分支（`scripts/cond_engine.py`）生成；
- 成功：`rc=0`，写 `<outdir>/outputs.npz` + `meta.json`，stdout 首行 `EXEC ...`；
- 拒绝：`rc=3`，stdout 含 `REJECT code=<四码之一>` 与 `UPSTREAM type=... sha256=... summary=...`，
  `EXEC ... path=reject exec_mode=reject abi_calls=0`，**不写** `outputs.npz`；
- 其它非 0：`ERROR`，不算拒绝通过；本任务没有合法的 `UNSUPPORTED`/`rc=4`。

| 退出码 | 含义 |
|---|---|
| `0` | 成功（输出与 `meta.json` 已写） |
| `1` | `ERROR`（`device_error`/`invalid_argument`/`internal_error`、case/inputs 缺失、`_resolved` 缺失或指纹读取失败） |
| `2` | argv 协议错误（未知参数/缺值），不进入算子路径 |
| `3` | 按预期拒绝（四码之一） |
| `127` | `execvp("python3")` 失败 |

`EXEC` 必需键：`op`、`kernel`、`arch`、`mode`、`variant`、`pred_kind`、`selected`、`path`、
`exec_mode`、`abi_calls`、`backend`、`device_id`、`trace_true`、`trace_false`、`exec_true`、
`exec_false`、`runtime_token`、`compile_count`、`plan_id`、`plan_cache`；缺任一键判 `INVALID`。
`meta.json` 记录 dispatch/evidence/leaves/输入输出哈希/`so_fingerprint`；拒绝时含 `upstream`。

### 2.3 C ABI（`libjax_lax_cond_host.so`）

`op_host/jax_lax_cond_abi.h` 与 Python `ctypes.Structure` 逐字段对应，布局 472 B（59×8）：

```c
typedef struct {
    int32_t branch_id;            /* 1=true, 2=false */
    int32_t n_outputs, n_inputs;
    int32_t dtype[8];             /* 逐输入叶子：0=float32 */
    int64_t in_numel[8];  void *in[8];
    void *out[8];         int64_t out_numel[8];
    int32_t out_src[8];   int32_t out_op[8];  int64_t out_stride[8];
    int32_t device_id;
    void *token;                  /* host int32*：kernel 写 1/2，本库 D2H 读回 */
    void *counters;               /* host int32[2]：kernel 写 [true,false] 计数 */
    void *err;  int32_t err_len;  int32_t abi_version;  /* = 1 */
} jax_lax_cond_args_t;

extern "C" int jax_lax_cond_run(const jax_lax_cond_args_t *args);
```

另导出 `jax_lax_cond_abi_version()`、`jax_lax_cond_acl_init()`、`jax_lax_cond_malloc()`、
`jax_lax_cond_free()`、`jax_lax_cond_h2d()`、`jax_lax_cond_d2h()`。返回码：`0` 成功 / `1`
运行失败 / `2` 参数错误；拒绝在 Python 层设备操作前完成（`abi_calls=0`）。`out_src/out_op/
out_stride` 是可表达 `c06` 跨步 aux（`x[:,0] ± 1`）的追加字段；`.so` 不做任何数值重排。

### 2.4 交付文件

| 文件 | 作用 |
|---|---|
| `python/jax_lax_cond.py` | Python 入口：`cond` / `cond_compiled`、`CondError`、`_last_dispatch`、`_reset_evidence`、`_plan_dispatch`、`library_fingerprint` |
| `python/jax_lax_cond_cli.py` | 四动作 host driver：读 case/inputs，写 outputs/meta，stdout 打印 `EXEC` |
| `op_host/jax_lax_cond_abi.h` | C ABI 结构体（ctypes 逐字段对应，472 B） |
| `op_host/jax_lax_cond_pyapi.asc` | 参数校验 + Tiling + kernel launch + 证据 D2H |
| `op_host/jax_lax_cond.asc` | `build/jax_lax_cond`：argv 协议校验 + 转交 host driver（无数值计算） |
| `op_kernel/jax_lax_cond_kernel.asc` | AscendC kernel：逐输出叶子向量原语 + token/counters |
| `op_kernel/jax_lax_cond_tiling.h` | kernel/host/pyapi 共用的 tiling 结构与 `COND_OP_*` |
| `tests/test_jax_lax_cond.py`（仓库根 `tests/`） | pytest 交付件：29 身份 / 83 mode + 15 自检 + 4 诊断 |

---

## 3. 构建与运行

### 3.1 四动作（工作区根执行）

```bash
scripts/nm.sh --op jax_lax_cond --action build      # 产物 build/jax_lax_cond + libjax_lax_cond_host.so
scripts/nm.sh --op jax_lax_cond --action smoke      # 最小 required match 用例
scripts/nm.sh --op jax_lax_cond --action test       # gate 16 条 / 49 mode + pytest
scripts/nm.sh --op jax_lax_cond --action benchmark  # 2 用例 / 4 序列（仅基线）
```

`build.sh` 只 `make jax_lax_cond`；CMake 通过 `add_dependencies(jax_lax_cond jax_lax_cond_host)`
顺带产出 `.so`，并在 POST_BUILD 打印 `SO_OK path=... size_bytes=... sha256=...`，再运行
`build/jax_lax_cond --self-check`（dlopen `.so` + ABI，不需要 NPU）。

### 3.2 无 NPU 的本地检查（不替代真机门禁）

```bash
python -m py_compile ops/jax_lax_cond/python/*.py
JAX_PLATFORMS=cpu python -m pytest tests/test_jax_lax_cond.py -q -k selfcheck
```

### 3.3 运行期加载与指纹

Python 侧按 `Path(__file__).resolve().parent.parent / "build" / "libjax_lax_cond_host.so"`
定位并 `ctypes.CDLL`；加载时校验 ABI tag `opx-cond-python-abi/1` 与 `ctypes.sizeof == 472`。
`library_fingerprint()` 返回实际加载的路径/大小/sha256/ABI，可与 CMake POST_BUILD 打印值逐字比对。

### 3.4 最近一次真机验证（当前交付版本）

> 版本：operator-v1 `55b69ca6a7dbc10b22be8976091cdac6d00931f68eee552fd29f9f8ee121c426`
> （13 文件）；测试入口 sha256 `66a3d0a1340c6f256ac9edc5a719ed8199c80fedf0febf9074013ea9e81f8f2a`；
> spec v1（`spec_version_id=6`）、`cases@233778f456b4eb9b`、镜像 #145、设备 5、exclusive。

| 动作 | run / 台账 | exit | 结果 |
|---|---|---|---|
| build | `r20261028-162821` / #226 | 0 | `BUILD_OK`；`.so` 561,952 B / `bdf8530d…`，exe 381,696 B / `25d4eebf…`，ABI `opx-cond-python-abi/1` |
| smoke | `r20261028-162906` / #227 | 0 | `SMOKE_VERDICT PASS`，`cond_true_eager/eager` 1/1 |
| test | `r20261028-174512` / #232 | 0 | gate `PASS=16 / FAIL=0 / ERROR=0 / INVALID=0 / NOT_RUN=0`，49/49 mode；pytest 29 身份 / 83 mode / 102 testcase，0 fail/error/skip；白盒 12/12；D2 8/8；D1/D3 PASS；D4 2 条 `BACKEND_LIMITATION` + 1 条 `RECORDED`（**不计 PASS**） |
| benchmark | `r20261028-164927` / #229 | 0 | 2 用例 / 4 序列，10 样本/序列；`performance_target=null`，仅基线 |

> build/smoke/benchmark 为同一 operator-v1 快照的复用证据（节点 2）；test 为节点 4 修复测试后
> 的全量重跑。四个 run 的 spec/快照/用例集/镜像/设备一致；控制台 evidence gate 四项
> `ok=true / gate_verdict=pass`、`missing=[]`、`deliverable=true`。原始证据见
> `.cannbot/jax_lax_cond/workflow3/4-证据-r1/`、`4-测试执行记录.md`、`4-验收报告.md`。

---

## 4. 精度口径

- **oracle**：同版本 JAX 0.8.1 CPU，`scripts/golden.py` + `golden_cli.py` 独立子进程
  （`JAX_PLATFORMS=cpu`）；不 import 候选，不使用 NPU 结果当参考；
- **判据档**：`criteria/check.py`（`ops-precision-standard@1`）；float32 `rtol=2^-10`
  （≈9.77e-4）、`atol=2^-16`（≈1.53e-5）、`matched_ratio ≥ 0.99`、
  `max_abs_error ≤ max(1e-2, 32×ULP@1.0)`；float32 下 32×ULP@1.0≈3.8e-6，实际下限为 1e-2；
- **结构与类型**：tree structure / 叶子数 / dtype / shape 完全一致（0 容差）；jax family 的
  `weak_type` 逐叶与 CPU 精确；numpy family 的 `weak_type` 记为 `null`（不可观测，不冒充覆盖）；
- **特殊值**：NaN/±Inf 位置与符号逐元素精确，有限子集再走浮点容差；`-0.0` 符号不单独断言；
- **设备守卫**：numpy family 必须 `path=so`、`abi_calls=1`、`backend=npu`、`device_id=0` 且有
  `so_fingerprint`；jax family 必须 `jax.default_backend() != "cpu"` 且输出设备平台含 `npu`；
- **字节与对齐实算**（`bytes = elements × dtype_size`）：`float32[17]` = 17×4 = 68 B，
  `68 % 32 = 4`；`float32[33]` = 33×4 = 132 B，`132 % 32 = 4`；`float32[4,17]` = 68×4 =
  272 B，`272 % 32 = 16`；`float32[8192]` = 32768 B，`%32=0`；`float32[8193]` = 32772 B，
  `%32=4`；`float32[1024,256]` = 262144×4 = 1,048,576 B，`%32=0`；kernel 用 `DataCopyPad`
  处理非 32B 尾块，host 不要求 32B 对齐；
- **实测**：gate 16/16（49/49 mode）、pytest 83/83 mode 全 PASS，`matched_ratio=1.0`；除
  `cond_large_tensor_throughput` 的 `jit_true`/`jit_false` 两个 benchmark 复用 mode 为
  `max_abs_error=1.1920928955078125e-07` 外，其余为 `0.0`；两者均远低于 1e-2，不影响 PASS；
- **拒绝**：四码精确、`abi_calls=0`、不写输出、保留上游类型/消息 sha256；c13 的 int32 `0/1`
  按已确认规格规则拒绝（`cpu_rejects=false`、`spec_rule=scalar_bool_pred` 留证，不伪造异常）。

---

## 5. 性能口径

- **基线**：不设 CPU/torch 基线（`baseline: null`、`comparison: same_device_only`）；
  `performance_target=null`，仅记录，不设硬阈值；
- **测量**：`scripts/nm.sh --op jax_lax_cond --action benchmark` → `tools/run_benchmark.py` →
  `scripts/cond_bench.py`；设备侧 `msopprof` 的 `Task Duration(us)`；blocking JIT 稳态，
  `warm_up=3`、`invocations=10`、median、`block_until_ready`；
- **代表用例**：`cond_small_branch_overhead`（`float32[16]`，小分支开销）、
  `cond_large_tensor_throughput`（`float32[1024,256]`，**primary**，大张量吞吐）；
- **实测**（run #229，设备 5，exclusive，镜像 #145）：

| 用例 | pred | median（设备侧 Task Duration） | block_dim |
|---|---|---|---|
| `cond_small_branch_overhead`（`float32[16]`） | true | 4.660093 us | 1 |
| 同上 | false | 4.390088 us | 1 |
| `cond_large_tensor_throughput`（`float32[1024,256]`） | true | 7.460150 us | 32 |
| 同上 | false | 7.440149 us | 32 |

> 每序列 10 个样本、无采集失败；同设备/镜像/用例集。本轮不宣称性能达标，性能优化与
> profile 不在本节点范围。

---

## 6. 已知限制（UNKNOWN 不当作已支持）

- v1 不承诺空 pytree operand；不承诺对 `pred` 求导；不承诺多设备/SPMD 与并发调用。
- numpy family 的数值分支必须是本仓已注册分支（`_BRANCH_NAME_TO_ID` 的 1–8、11–14）；未注册
  callable 只能走 jax family trace（不冒充 `.so` 设备执行）。
- jax family 的 `pred` 必须是可静态求值的 scalar bool；非 scalar 的 jax bool 数组不在本批
  required 覆盖内（冻结 c13 的四码断言在 numpy family 路径），实现未承诺对其报
  `INVALID_PREDICATE`。
- 跨步输出叶子（`x[:,0]`）只用于小 `aux`：host 限制 `out_numel ≤ 4095` 且源输入
  `numel ≤ 8192`；本批 c06 为 7 元素，WB-04 源恰为 8192 上界。
- `weak_type` 只在 jax family 可观测；numpy family 的 `meta.leaves[].weak_type` 记 `null`。
- **目标镜像后端限制（有原始证据，workflow2 run #202 / workflow3 D4）**：
  xla-npu/AFIR 不支持 `stablehlo conditional` / `select` 的 lowering、`convert`
  (bool→int/float) 不可执行，`debug_callback` / `pure_callback` 未注册 npu lowering。
  因此 jax family 在 jit 内遇到 **tracer 谓词**时显式报 `device_error`，不生成必然
  失败的 conditional 图、不伪造 `selected/runtime_token`；stock `jax.lax.cond` 在
  jit 内的 conditional 同样不受支持（D4 以 `BACKEND_LIMITATION` 如实记录，不计 PASS）。
  本批 `c04` 的“动态谓词”是**运行期具体标量 bool**（同一 plan 的 true/false 两次调用），
  不是 outer `jax.jit` 的 tracer pred。
- **grad 路线（jit / eager 分开，梯度均由 JAX 反向模式 AD 产生，无 `branch_id` 导数表）**：
  - `jax.jit(jax.grad(...))` 内的单叶 grad 用真实 `jax.vjp`（消费真实 cotangent），
    转置图乘输入相关恒等项 `((v+1)-(v-1))/2`，规避 AFIR 对 2-D 常数转置图只写首列
    的缺陷。该镜像的 AFIR **只支持扁平 `[N]` 的 jit grad**：2-D auto-fuse 输出只写
    首列（设备探针 5/20），且 `20→5x4` 的 device reshape 被明确拒绝；因此 2-D
    jit grad 调用会显式报 `device_error`，测试/CLI 以 host 扁平化进入、比较前在
    host 恢复原 shape（不改变数值）。
  - **eager** `jax.grad`：单叶 grad 用 `custom_vjp`，后向的 `jax.vjp(branch, value)`
    由 `jax.jit` 编译成独立程序（真实 residual + 调用方真实 cotangent），绕开 AFIR
    对 eager 嵌套转置的 `Unsupported opcode: tuple` 限制；2-D 布局可直接写出全部
    元素（设备探针 `vjp_jit_res:eager2d` = 20/20）。对比探针：不改走路的 plain eager
    AD 对线性分支同样报 `tuple`（`2-探针-eager-ad-r1/r2*.log`），因此候选统一用
    上述“eager VJP 后向编译”路线。
  - 恒等项仅在 `|v| < 2**24`（fp32）精确等于 1，超出该范围时 jit 路线不承诺梯度
    正确（冻结输入 O(1)/×1000 在范围内）；后续若需更大范围的 jit AD，应改为
    kernel 计算 VJP。
  - 多叶 pytree grad 未验证；`grad_eager` 的 0-d 兼容性未验证。
- **0-d jit（R2）**：镜像 #145 不能编译 0-d `jax.jit` 动态参数，S1 的 weak/strong
  jit 使用零参数 closure（operand/pred 均为闭包常量，pred 由设备标量派生后同步为
  具体值）。其 HLO 是常量图（`stablehlo.constant`），分支在 closure 外以 eager 设备
  执行；数值与 `weak_type` 精确，但不宣称“分支在 jit 图内运行”。
- **2-D jit grad / 跨步源 >8192**：已记录上界/后端限制，本批未作为正常用例；见
  `.cannbot/jax_lax_cond/workflow3/2-实现记录.md` §7.4/§7.5。
- 性能只记基线；profile、性能优化与性能达标验收不在本节点范围。
- 镜像/设备/用例集身份以控制台任务配置与 run manifest 为准；本 README 不写未实测的环境结论。

### 未验证项（如实保留，不写成通过）

1. outer `jax.jit` tracer pred 与 stock `lax.cond` 在 jit 内的 conditional：`BACKEND_LIMITATION`，
   不是 PASS。
2. 多叶 pytree grad、2-D jit grad、eager 0-d grad 的 jit 后向兼容性：未验证。
3. 多设备/SPMD、并发调用、不同芯片形态：未验证。
4. `profile` 动作：未执行；性能仅 build/test/benchmark 范围内的基线。
5. torch/torch_npu family：本批不适用（镜像 #145 无 torch；JAX 语义以 JAX CPU 为 oracle）。

---

## 7. 证据与参考

- 规格：`tasks/jax_lax_cond/spec/spec.json`（spec v1，`spec_version_id=6`）；
- 冻结用例：`cases/jax_lax_cond/{cases,benchmark}.json`（`cases@233778f456b4eb9b`）；
- 判据：`criteria/check.py`（`ops-precision-standard@1`）；
- 设计：`docs/jax_lax_cond/design.md`；
- 当前执行证据：`.cannbot/jax_lax_cond/workflow3/4-测试执行记录.md`、`4-证据-r1/`、
  `4-验收报告.md`、`4-清单.json`；
- 黑盒/白盒设计：`.cannbot/jax_lax_cond/workflow3/0-黑盒测试设计.md`、
  `3-白盒测试设计.md` 与 `4-白盒设计勘误.md`。
