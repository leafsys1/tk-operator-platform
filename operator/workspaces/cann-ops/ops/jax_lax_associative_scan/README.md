# jax_lax_associative_scan · Ascend C 直调算子

JAX `lax.associative_scan` 在 Ascend 910B1（`dav-2201`）上的直调实现：Python 调用接口 → C ABI
（`libjax_lax_associative_scan_host.so`）→ Ascend C kernel
（`op_kernel/jax_lax_associative_scan_kernel.asc`）。
参考 oracle 为**同版本 JAX CPU**（`ops/jax_lax_associative_scan/scripts/golden.py`，独立子进程）。
设计文档见 `docs/jax_lax_associative_scan/design.md`。

## 目录

```
op_kernel/jax_lax_associative_scan_tiling.h   Tiling 常量/结构体（kernel、host、pyapi 三方共用）
op_kernel/jax_lax_associative_scan_kernel.asc AscendC kernel（单核持 carry，按 outer→inner-chunk→步 扫描）
op_host/jax_lax_associative_scan.asc          可执行入口（四动作门禁，argv 协议 + 转交 host driver）
op_host/jax_lax_associative_scan_pyapi.asc    Python 接口共享库（extern "C" C ABI）
op_host/jax_lax_associative_scan_abi.h        C ABI 结构体（布局版本 1，456 字节）
python/jax_lax_associative_scan.py            Python 调用接口（ctypes → lib…_host.so）
python/jax_lax_associative_scan_cli.py        host driver（--case-json/--inputs/--mode/--outdir）
scripts/golden.py                             **独立** CPU oracle（同版本 JAX，独立子进程）
scripts/golden_cli.py                         oracle 子进程入口（JAX_PLATFORMS=cpu）
scripts/fixtures.py                           测试输入生成 / pytree 编解码 / combine 注册表
scripts/assoc_scan_engine.py                  test/smoke 的 assoc-scan 执行器（测试工程交付）
scripts/assoc_scan_bench.py                   benchmark 分支（被 tools/run_benchmark.py 按 op 分派）
CMakeLists.txt                                两个 target：可执行文件 + 共享库
```

---

## 1. 语义

```python
out = jax.lax.associative_scan(fn, elems, reverse=R, axis=A)
```

* `fn` 必须是满足结合律的二元函数；本批注册 `combine_sum`（`a+b`）、`combine_max`
  （`maximum(a,b)`）、`combine_tree_add`（pytree 逐叶子 `+`），按对象身份或 `fn.combine_id` 解析；
* `reverse=False`：`out[..., i, ...] = elems[..., 0, ...] ⊕ … ⊕ elems[..., i, ...]`；
* `reverse=True`：`out[..., i, ...] = elems[..., L-1, ...] ⊕ … ⊕ elems[..., i, ...]`（后缀）；
  结果仍按**原始索引**写回（等价于「反转输入 → 扫描 → 反转输出」）；
* 输出与输入**结构、shape、dtype 完全一致**，是新结果，不原地修改输入；
* pytree 各叶子 `shape[axis]` 必须相等（JAX 的共同迭代长度），其余维可不同。

### 支持范围（本批）

| 项 | 范围 | 越界行为 |
|---|---|---|
| dtype | `float32` | 其他 dtype → `dtype_mismatch`（device 前拒绝） |
| `axis` | 静态整数，`0 <= axis < rank`；本批确认数值用例覆盖 0/1 | 非法 → `invalid_axis`（device 前拒绝） |
| `reverse` | `True/False` | — |
| 结构 | 单叶子 / tuple / list / dict pytree | 空树、rank-0、叶子类型不一致 → `invalid_argument` |
| `n_leaves`（**单次 launch**） | `1..8`（`ASSOC_MAX_LEAVES`）；`whole_tree` 一次带整树叶子 | 超限 → `invalid_argument`（device 前拒绝；`associative_scan` 的 `per_leaf` 每次只带 1 个叶子，**不受**该上限约束） |
| 长度 `L=shape[axis]` | `>= 1`（含 `L=1`、非 2 次幂） | — |
| `inner` | 任意；kernel 按 `ASSOC_MAX_INNER_CHUNK=2048` 切块 | — |
| 设备 | 单卡（device 0），单核 kernel | 多设备/SPMD 不支持 |

与 `jax.lax.associative_scan`（JAX/JAXLIB 0.8.1）逐输出对齐：树结构 / 叶子路径集合 / shape /
dtype 完全一致，数值按 `ops-precision-standard@1`；NaN/±Inf 不定义专门语义（测试输入不使用）。

---

## 2. 调用接口

### 2.1 Python 接口（交付件 #2）

```python
from ops.jax_lax_associative_scan.python.jax_lax_associative_scan import (
    associative_scan, associative_scan_compiled,
)

out = associative_scan(fn, elems, reverse=False, axis=0)
out = associative_scan_compiled(fn, elems, reverse=False, axis=0)
```

在工作区根目录执行（`ops/` 为 namespace package，无 `__init__.py`）。最小示例：

```python
import numpy as np
from ops.jax_lax_associative_scan.python.jax_lax_associative_scan import associative_scan


def combine_sum(a, b):          # 已注册 combine：函数对象带 combine_id 属性
    return a + b


combine_sum.combine_id = 1      # 1=sum；2=max；3=pytree_add

elems = np.random.default_rng(0).uniform(-1, 1, (8, 16)).astype(np.float32)
out = associative_scan(combine_sum, elems, axis=0)   # out.shape == (8, 16)
```

| 入口 | 输入 family | `path` | `exec_mode` | `abi_calls` |
|---|---|---|---|---|
| `associative_scan` | numpy | `so` | `per_leaf` | 叶子数（单叶子 1；pytree 2） |
| `associative_scan_compiled` | numpy | `so` | `whole_tree` | 1（一次 launch 覆盖整棵树） |
| `associative_scan`（在 `jax.jvp`/`jax.vjp` 内或输入为 `jax.Array`） | jax | `trace` | `trace` | 0 |
| 非法输入 | — | `reject` | `reject` | 0 |

* `associative_scan_compiled` 是**测试/门禁的编译态入口**，不是第二个公开 API；
* `jit` 的 `plan_id` = combine id + 各叶子 shape + dtype + axis + reverse 的摘要；同进程第二次
  调用同一计划时 `plan_cache="hit"`，`eager` 为 `None`；
* 自省：`_last_dispatch`（每次调用后更新）与 `_plan_dispatch(...)`（纯路由，不执行 device、不改
  `_last_dispatch`），字段 `op/mode/path/exec_mode/combine_id/abi_calls/family/backend/device_id/
  plan_id/plan_cache/reason`；
* 返回 **新** numpy 数组 / JAX 数组：不原地改输入，输出不与输入共享存储。

Python 侧拒绝映射（异常文本带稳定 code，与 CLI 同一张表）：

| 拒绝码 | 触发条件 |
|---|---|
| `dtype_mismatch` | 叶子 dtype 不是 float32 |
| `invalid_axis` | `axis` 非整数、越界、或对当前 rank 不合法 |
| `invalid_argument` | 空 pytree、rank-0、叶子类型不一致、非 C-contiguous、`shape[axis]` 不一致、`whole_tree` 单次 launch 叶子数 > 8 |
| `unsupported_combine_fn` | 未注册 callable（trace 路径的显式分类，不静默返回错值） |

### 2.2 门禁 CLI（host driver，交付件 #2 的可执行侧）

```bash
ops/jax_lax_associative_scan/build/jax_lax_associative_scan \
  --case-json <cf>/case.json --inputs <cf>/inputs.npz \
  --mode <eager|jit|jvp|vjp> --outdir <cf>/out
```

| 结果 | rc | 产物 |
|---|---|---|
| 成功 | 0 | `outdir/outputs.npz` + `outdir/meta.json`；stdout 首行 `EXEC ...` |
| 拒绝 | 3 | `meta.json`（`status="reject"`、`path="reject"`、`abi_calls=0`、无 `leaves`）；**不写** `outputs.npz` |
| 显式 unsupported | 4 | `UNSUPPORTED:autodiff_unsupported` + 后端原文；`EXEC ... path=trace abi_calls=0 backend=<非cpu>` |
| 参数/IO 错误 | 2 | stderr 说明 |
| 其他失败 | 1 | stderr 说明 |

* `outputs.npz` 键：单叶子 `out`；pytree `out/a`、`out/b`；AD `jvp_primal`+`jvp_tangent`
  或 `vjp_primal`+`vjp_cotangent`；
* `meta.json` 记录 `case_id/mode/status/inputs_sha256/dispatch{...}/leaves[...]`，成功路径另记
  `so_fingerprint{path,size_bytes,sha256,abi}`（运行期实际 `dlopen` 的 `.so`）；
* `EXEC` 键：`op/kernel/arch/mode/combine/combine_id/family/path/exec_mode/abi_calls/backend/
  device_id/plan_id/plan_cache`。

### 2.3 C ABI（`libjax_lax_associative_scan_host.so`）

```c
int         jax_lax_associative_scan_run(const jax_lax_associative_scan_args_t *args); /* 0/1/2 */
const char *jax_lax_associative_scan_abi_version(void);      /* "opx-assoc-scan-host-abi/1" */
int         jax_lax_associative_scan_acl_init(void);
int         jax_lax_associative_scan_malloc(void **dev_ptr, int64_t bytes);
int         jax_lax_associative_scan_free(void *dev_ptr);
int         jax_lax_associative_scan_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
int         jax_lax_associative_scan_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
```

* 结构体 `jax_lax_associative_scan_args_t` 逐字段定义见 `op_host/jax_lax_associative_scan_abi.h`：
  布局版本 1、固定 **456 字节**（Python 侧 `ctypes.sizeof` 断言）；字段只追加，不改已用字段语义；
* 一次 `run` = 一次 kernel launch：`per_leaf` 时 `n_leaves=1`，`whole_tree` 时 `n_leaves=叶子数`
  （`1..8`）；`in`/`out` 均为 device 地址，由 Python 侧持有并负责 `malloc/h2d/d2h/free`；
* 返回码：`0` 成功 / `1` 运行失败（ACL、kernel）/ `2` 参数错误。`3`（拒绝）与 `4`（显式 unsupported）
  只在 Python 预校验 / trace 路径产生，**本层不伪造**；
* `err`（≥256 B）由调用方提供，失败时写 `code: detail`；
* ACL 生命周期：`aclInit` 只做一次（明确失败才报错）+ `aclrtSetDevice(device_id)` + 自建 stream +
  kernel 后 `aclrtSynchronizeStream`；**不** `aclFinalize`，生命周期归调用方（Python 进程内 PJRT 等
  其它 ACL 使用者由调用方管理）。

---

## 3. 构建与四动作入口

```bash
# 在目标仓根目录
scripts/nm.sh --op jax_lax_associative_scan --action build
scripts/nm.sh --op jax_lax_associative_scan --action smoke
scripts/nm.sh --op jax_lax_associative_scan --action test
scripts/nm.sh --op jax_lax_associative_scan --action benchmark
```

* 构建产物：`ops/jax_lax_associative_scan/build/jax_lax_associative_scan`（四动作入口）与
  `build/libjax_lax_associative_scan_host.so`（Python C ABI）；目标名必须为算子名；
* `build.sh` 只按 `$OP` 找 `ops/$OP` 并 `make $OP`，本算子不需要改 `build.sh`；
* `test.sh` / `tools/run_cases.py` / `tools/smoke_case.py` / `tools/run_benchmark.py` 已按 op
  显式分派到 `scripts/assoc_scan_engine.py` / `scripts/assoc_scan_bench.py`；
* 顺序约束：`build → smoke → test → benchmark`；smoke 未 PASS 不进入 test，test 未全 PASS
  不进入 benchmark。设备 0 独占，不与其他动作并发。

Python 接口直接运行（不经过四动作）：

```bash
python3 -c "from ops.jax_lax_associative_scan.python.jax_lax_associative_scan import abi_version; print(abi_version())"
```

pytest 交付件（读冻结 `cases/`，判据与门禁同源；镜像 #145 自带 pytest，`test` 动作第二段会自动跑）：

```bash
pytest -q tests/test_jax_lax_associative_scan.py
```

### 最近一次真机验证（当前交付版本）

| 动作 | run | 结论 | 证据（`$WORK_DIR` = `.cannbot/jax_lax_associative_scan/workflow1`） |
|---|---|---|---|
| build | `r20261027-132336` / #99 | `exit=0`，`BUILD_OK`，gate `pass` | `$WORK_DIR/4-证据-r1/runs-99/out/run.log` |
| smoke | `r20261027-132421` / #100 | `SMOKE_VERDICT PASS`（`assoc_sum_len1`/jit），gate `pass` | `$WORK_DIR/4-证据-r1/runs-100/out/smoke.json` |
| test（共享 runner，权威） | `r20261027-132509` / #101 | `TEST_VERDICT PASS`，门禁 **7 条 / 14 次全 PASS**（0 FAIL/ERROR/INVALID） | `$WORK_DIR/4-证据-r1/runs-101/out/correctness.json` |
| test（pytest 交付件） | 同上 #101 | **27 条身份 / 48 次 = 46 PASS + 2 显式 UNSUPPORTED**；`69 passed, 2 xfailed`（`tests=71, failures=0, errors=0`） | `$WORK_DIR/4-证据-r1/runs-101/out/pytest_jax_lax_associative_scan.log`、`.../assoc_scan_correctness.json` |
| benchmark / profile | — | **当前快照未执行**（本阶段 out-of-scope；不得据历史值推算当前版本性能） | 见 §4.3 历史记录（快照不匹配） |

* 版本对齐：`#99/#100/#101` 的 `code_snapshot_sha256=cf080296b95fd8d9…`（`operator-v1`，13 个非 `.md`
  文件）、`suite_ref=cases@6d6bc47fa93738dc`（门禁 op-scoped 标签；同一冻结内容按整棵 `cases/`
  复算为 `cases@b852239bc7476550`，与 prompt/需求分析引用一致）、`spec_version_id=5`、
  `criteria_ref=ops-precision-standard@1`、镜像 digest `local-id:sha256:e99a62ad…`、device 0、
  `lock_mode=exclusive`；
* 本次文档补全只改 `README.md`（`.md`）与 `docs/jax_lax_associative_scan/design.md`（不在 `ops/` 下），
  `operator-v1` 快照口径跳过 `.md`，按 `scripts/nm_test.sh` 同算法复算仍为 `cf080296…`，
  故上述 build/smoke/test 证据仍对应当前算子代码；
* 2 个 `UNSUPPORTED` 是 `assoc_grad_sum` 的 jvp/vjp 显式分类（`xfail`），**不记 PASS**，见 §4.2。

---

## 4. 精度与性能口径

### 4.1 精度

* 逐输出张量判定，结构与 shape/dtype 精确（无容差）；
* 浮点按 `criteria/check.py`（`ops-precision-standard@1`）：`matched_ratio >= 0.99` 且
  `max_abs_error <= 1e-2`，元素阈 `atol + rtol*|golden|`（`rtol=2^-10`、`atol=2^-16`），
  非有限位逐元素精确；
* pytree 逐叶子独立判定，不合并通过率；AD 的 `primal` 与 `tangent`/`cotangent` 各自判定；
* oracle：同版本 JAX CPU（JAX/JAXLIB 0.8.1）的子进程；候选只跑 NPU，不为 golden 设置父进程
  环境变量；
* 候选设备证据：numpy family 必须 `path=so` + `backend=npu` + `abi_calls>=1` + `device_id=0`；
  任一 CPU 回退判 `INVALID`。

### 4.2 自动微分（JVP/VJP）边界

* **host-driver（CLI `--mode jvp|vjp`）**：`.so` kernel 路径。AD 定义与 golden 一致：
  tangent/cotangent seed 均为全 1；对 sum，`jvp_tangent = scan(ones)`，
  `vjp_cotangent = scan(ones, reverse 翻转)`，`primal = scan(elems)`。两次 kernel 调用，
  `path=so`、`exec_mode=jvp|vjp`；
* **JAX 数组 / tracer（pytest 交付件）**：`path=trace`、`abi_calls=0`、`backend` 非 CPU。
  trace 路径用 take/stack + elementwise combine 展开前缀扫描，由 JAX 自动微分穿过；
  **该路径证明的是 xla-npu 后端在该 AD 组合上的执行，不证明自定义 kernel 有 AD**；
* 若 trace 路径被后端明确拒绝：抛 `UNSUPPORTED:autodiff_unsupported` + 后端原文，
  `path=trace`/`abi_calls=0`/`backend` 非 CPU；连续 3 次同码才允许分类为 `UNSUPPORTED`，
  且 `UNSUPPORTED` **不记 PASS**。

### 4.3 性能

* 口径冻结：`blocking_jit_execution`，`warm_up=3`、`invocations=10`、`aggregate=median`、
  `synchronize=block_until_ready`；
* 基准用例：`assoc_sum_len256_width256`、`assoc_sum_len1024_width256`（axis=0、reverse=false、
  sum、float32）；主用例 `assoc_sum_len1024_width256`；
* `performance_target=UNKNOWN`：本任务只报告设备侧 `Task Duration` 中位数，不设通过阈值；
  不设 CPU/torch 基线，比较仅限同设备、同镜像、同规格、同用例集。

**当前快照的性能状态 = UNKNOWN**：`cf080296…` 尚未执行 `--action benchmark`/`profile`（本阶段
out-of-scope）。下表的历史采集**不对应当前代码**，仅用于说明测量口径，不得据此推算当前版本结论：

| 历史 run | `code_snapshot_sha256` | `assoc_sum_len256_width256` median | `assoc_sum_len1024_width256` median（主用例） |
|---|---|---|---|
| #81 `r20261027-112212` | `e88deade93a3…` | 87.642 us | 307.676 us |
| #85 `r20261027-113654` | `605ce734b7b2…` | 88.082 us | 308.766 us |

> 两次均为「同镜像 #145、device 0 独占、`warm_up=3`、`invocations=10`、median」，原始证据在仓库根
> `runs/<run_id>/out/performance.json`；当前快照需按同一口径重跑后才有性能结论。

---

## 5. 已知限制

1. 仅 float32；fp16/bf16/int 输入按规格拒绝（JAX 本身支持 fp16，拒绝依据是已确认 spec）。
2. `fn` 只注册 sum / max / pytree_add；未注册 callable 在 numpy family 显式报
   `unsupported_combine_fn`，不静默返回错值。
3. `axis` 实现对任意合法 rank 维度通用，但已确认数值用例只覆盖 0/1；负 axis 不支持。
4. 单卡、单核 kernel；不覆盖多设备/SPMD/shmap。
5. 不定义 NaN/Inf/±0/次正规数的专门语义（测试输入不使用）。
6. `associative_scan_compiled` 的 `jit` 是「整树一次 `.so` 调用」的门禁定义，不等价于
   JAX `jit` 的编译缓存语义；其单次 launch 的 `n_leaves <= 8`（`ASSOC_MAX_LEAVES`），
   9 叶子及以上在 device 前按 `invalid_argument` 稳定拒绝。该上限**只约束单次 launch**：
   `associative_scan`（`per_leaf`）每次只带 1 个叶子，pytree 叶子数不受 8 的限制。
7. 性能只报告设备侧 `Task Duration` 中位数（warmup 3、measure 10），`performance_target=UNKNOWN`，
   不设通过阈值。
8. 运行时 `.so` 与构建日志中的 sha256/size 必须一致；若镜像内构建缓存只回填 exe，
   `test.sh`/`smoke_case.py`/`run_benchmark.py` 的 `.so` preflight 会按同一份源码补编一次，
   补编失败即 rc≠0，不降级为跳过。

### 未验证项（如实保留，不写成通过）

* **性能**：当前快照 `cf080296…` 未执行 `benchmark`/`profile`，无当前版本性能结论（见 §4.3）；
* **AD**：`assoc_grad_sum` 的 jvp/vjp 在 #101 为**显式 `UNSUPPORTED`**（2 项 `xfail`），不是数值
  PASS；它证明的是 xla-npu 后端在该组合上的拒绝，不证明自定义 kernel 有 AD；
* **`_is_backend_unsupported` 的 `backend=None/cpu → False` 防御分支**：设备侧固定 `backend=npu`
  不进入该段，仅由本机纯 Python 探针覆盖（非 NPU 证据），统一 pytest 入口未覆盖；
* **上游 L-01**：`tests/test_jax_lax_associative_scan.py` 的 `pytest.importorskip("numpy")` 与未注册
  `opx_case` mark（#101 日志 1 warning）保留未改，属既有低风险项。
