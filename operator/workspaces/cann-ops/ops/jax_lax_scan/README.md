# jax_lax_scan · Ascend C 直调算子

JAX `lax.scan` 静态循环在 Ascend 910B1（`dav-2201`）上的直调实现：Python 调用接口 → C ABI
（`libjax_lax_scan_host.so`）→ Ascend C kernel（`op_kernel/jax_lax_scan_kernel.asc`）。
参考 oracle 为**同版本 JAX CPU**（`ops/jax_lax_scan/scripts/golden.py`，独立子进程）。
设计文档见 `docs/jax_lax_scan/design.md`。

## 目录

```
op_kernel/jax_lax_scan_tiling.h       Tiling 常量/结构体（kernel、host、pyapi 三方共用）
op_kernel/jax_lax_scan_kernel.asc     AscendC kernel（单核持 carry，逐步取 x_t / 写 ys[t]）
op_host/jax_lax_scan.asc              可执行入口（四动作门禁，argv 协议 + 转交 host driver）
op_host/jax_lax_scan_pyapi.asc        Python 接口共享库（extern "C" C ABI）
op_host/jax_lax_scan_abi.h            C ABI 结构体（布局版本 2，472 字节）
python/jax_lax_scan.py                Python 调用接口（ctypes → libjax_lax_scan_host.so）
python/jax_lax_scan_cli.py            host driver（--case-json/--inputs/--mode/--outdir）
scripts/golden.py                     **独立** CPU oracle（同版本 JAX，独立子进程）
scripts/golden_cli.py                 oracle 子进程入口（JAX_PLATFORMS=cpu）
scripts/fixtures.py                   注册 body / 输入生成（测试侧共用；host driver 独立实现同算法）
scripts/scan_engine.py                pytest 侧编排（用例加载、判据、证据）
scripts/scan_bench.py                 基准分支（被 tools/run_benchmark.py 按 op 分派）
CMakeLists.txt                        两个 target：可执行文件 + 共享库
```

---

## 1. 语义

```python
final_carry, ys = jax.lax.scan(f, init, xs, length=L, reverse=R)
```

* 逐步计算 `f(carry, x_t) -> (new_carry, y)`；`y` 沿新轴堆叠成 `ys`（`shape == (L,) + y.shape`）；
* `length` 在 trace 期静态确定；`xs=None` 时必须显式给 `length`；
* `reverse=True`：按 `x[L-1] … x[0]` 顺序消费，`ys[t]` 仍对应**原始**索引 `t`（JAX 语义：先反转 xs，再反转 ys）；
* `carry` 的结构 / shape / dtype 每步必须不变，非法改变在 trace 期拒绝（`carry_shape_mismatch`）。

### 注册 body（本批确认范围）

| `body_id` | 名称 | carry | xs | 每步语义 | `y` |
|---|---|---|---|---|---|
| 1 | `B1_add` | `f32[W]` | `f32[L,W]` | `new = carry + x` | `new` |
| 2 | `B3_pytree` | `(int32 counter, f32[W] acc)` | `f32[L,W]` | `new = (c+1, acc+x)` | `acc+x` |
| 3 | `B4_quad` | `f32[W]` | `f32[L,W]` | `new = 0.5*carry + x` | `new*new` |
| 7 | `pytree_xs_add_scale` | `f32[W]` | `dict(add: f32[L,W], scale: f32[L,W])` | `new = carry + add*scale` | `new` |
| 8 | `none_input_increment` | `f32[W]` | `None`（显式 `length`） | `new = carry + 1.0` | `new` |
| 4 / 5 / 6 | `B5_bad_shape` / `B6_bad_dtype` / `B7_bad_struct` | — | — | 故意破坏 carry 不变式 | 拒绝 |

`4/5/6` 是**拒绝用例的 body**：在 Python trace 期直接抛 `carry_shape_mismatch`，`abi_calls == 0`，
永不进入 device kernel。

### 支持范围（本批）

| 项 | 范围 | 越界行为 |
|---|---|---|
| `carry` / `xs` dtype | `carry`：`float32`（pytree 另允许 1 个 `int32` 标量 counter）；`xs`：`float32` | carry 非法 dtype → `dtype_mismatch`；`xs=int32` 未被拒绝、按 float32 解释（见 §6 第 10 条） |
| carry 浮点叶子宽度 `W` | `1 … 4096`（`SCAN_MAX_WIDTH`） | `.so` 返回 2（`invalid_argument`） |
| `length` | `≥ 1`，trace 期静态；`xs=None` 时必填 | `length_mismatch` |
| `reverse` | 支持 | — |
| `unroll` | 仅 `1` | `invalid_argument` |
| 设备 | 单卡（device 0），单核 kernel | 多设备不支持 |

与 `jax.lax.scan`（JAX/JAXLIB 0.8.1）逐输出对齐：树结构 / 叶子数 / shape / dtype 完全一致，
数值按 `ops-precision-standard@1`，NaN/±Inf 逐元素语义一致。

---

## 2. 调用接口

### 2.1 Python 接口（交付件 #2）

```python
from ops.jax_lax_scan.python.jax_lax_scan import scan, scan_compiled

final_carry, ys = scan(f, init, xs=None, length=None, reverse=False, *, unroll=1)
final_carry, ys = scan_compiled(f, init, xs=None, length=None, reverse=False, *, unroll=1)
```

在工作区根目录执行（`ops/` 为 namespace package，无 `__init__.py`）。最小示例：

```python
import torch
import torch_npu
from ops.jax_lax_scan.python.jax_lax_scan import scan


def add_body(carry, x):          # 已注册 body：函数对象带 body_id 属性
    new = carry + x
    return new, new


add_body.body_id = 1             # 1 = B1_add；未注册 callable 走 JAX trace

init = torch.zeros(16, dtype=torch.float32, device="npu:0")
xs = torch.randn(8, 16, dtype=torch.float32, device="npu:0")
final_carry, ys = scan(add_body, init, xs)   # ys.shape == (8, 16)
```

| 入口 | 路由 | `exec_mode` | `abi_calls` |
|---|---|---|---|
| `scan` + 已注册 body + torch/numpy 族 | `.so`（逐步） | `stepwise` | `length` |
| `scan_compiled` + 已注册 body + torch/numpy 族 | `.so`（整循环一次下发） | `compiled_whole_loop` | `1` |
| 未注册 callable / JAX tracer（含 JVP/VJP）/ **具体 `jax.Array`** | JAX trace | `jax_eager` / `jax_jit` | `0` |
| `body_id ∈ {4,5,6}` | trace 期拒绝 | `reject` | `0` |

* `init` / `xs`：`torch_npu.Tensor` pytree（交付接口目标族，补充镜像 #80 已实测），或 numpy /
  `jax.Array`（#145 主门禁的 host driver / AD 测试用；#145 不含 torch_npu）；
  `carry` dtype ∈ {float32, int32}（`int32` 仅允许作 pytree 的 counter 标量）；`xs` 叶子为 float32；
  **torch 张量必须连续**（非连续直接 `invalid_argument`），
  **numpy 非连续输入会被 `ascontiguousarray` 规范化为连续副本**（只影响输入表示，不做语义变换）；
* 返回**新**张量：`final_carry` 结构同 `init`，`ys.shape == (length,) + y.shape`；不原地改输入；
* 自省：`_last_dispatch`（每次调用后更新）与 `_plan_dispatch(...)`（纯路由，不执行 device），
  字段 `op/mode/path/exec_mode/body_id/abi_calls/family/reason/backend`。

> `scan_compiled` 是测试/门禁的**编译态入口**，不是第二个公开 API。

Python 侧拒绝映射（异常文本带拒绝码，与 CLI/`.so` 同一张表）：

| 拒绝码 | 触发条件 | Python 侧 |
|---|---|---|
| `carry_shape_mismatch` | 拒绝 body（4/5/6）；`.so` 侧 carry 不变式不满足 | `ValueError` |
| `dtype_mismatch` | dtype 不在 `{float32,int32}` | `ValueError` |
| `length_mismatch` | `xs` 前导轴不一致 / `xs=None` 未给 `length` / `length < 1` | `ValueError` |
| `invalid_argument` | 非连续 torch 张量、未知参数、`unroll != 1`、非 npu 张量、`width ∉ [1,4096]` | `ValueError` / `TypeError`；`.so` 返回 2 时 `RuntimeError`（文本带码） |
| `device_error` | ACL / kernel / 设备侧失败 | `RuntimeError` |
| `unsupported_body` | 张量族不支持 | `ValueError` |
| `autodiff_unsupported` | 公开 torch 接口输入 `requires_grad=True`（kernel 启动前拒绝） | `ValueError`（`UNSUPPORTED:` 前缀） |

### 2.2 C ABI（`libjax_lax_scan_host.so`）

```c
int  jax_lax_scan_run(const jax_lax_scan_args_t *args);  /* 0/1/2/3/4 */
const char *jax_lax_scan_abi_version(void);               /* "opx-scan-python-abi/2" */
int  jax_lax_scan_acl_init(void);
int  jax_lax_scan_malloc(void **dev_ptr, int64_t bytes);
int  jax_lax_scan_free(void *dev_ptr);
int  jax_lax_scan_h2d(void *dev_ptr, const void *host_ptr, int64_t bytes);
int  jax_lax_scan_d2h(void *host_ptr, const void *dev_ptr, int64_t bytes);
```

结构体逐字段定义见 `op_host/jax_lax_scan_abi.h`（布局版本 2，ctypes 侧断言语义尺寸 472 字节）。
返回码：`0` 成功 / `1` 运行失败 / `2` 参数错误 / `3` carry 不变式拒绝 / `4` 显式 unsupported。
`err` 由调用方提供（≥256B），失败时写入 `code: detail`。

### 2.3 四动作可执行入口

```bash
ops/jax_lax_scan/build/jax_lax_scan \
  --case-json <case.json> --inputs <inputs.npz> --mode <eager|jit|jvp|vjp> --outdir <dir>
```

* 成功：`rc=0`，写 `outputs.npz` + `meta.json`，stdout 首行
  `EXEC op=jax_lax_scan kernel=jax_lax_scan_kernel arch=dav-2201 mode=… body=… body_id=…
  family=… path=… exec_mode=… abi_calls=… backend=…`（缺失即判 `INVALID`）；
* 拒绝：`rc=3` + stdout 含 `carry_shape_mismatch` + `EXEC path=reject abi_calls=0`，不写 `outputs.npz`；
* 显式 unsupported：`rc=4` + `UNSUPPORTED:autodiff_unsupported` + 后端原文；
* 其它失败：`rc=1`（`ERROR`，不算拒绝通过）；参数协议错误：`rc=2`（`invalid_argument`）。

| 退出码 | 拒绝码 | 条件 |
|---|---|---|
| 0 | — | 成功（写 `outputs.npz` + `meta.json`） |
| 1 | `device_error` | ACL / kernel / 文件读写失败 |
| 2 | `invalid_argument` | 参数缺失/未知、`.so` 侧参数或范围非法 |
| 3 | `carry_shape_mismatch` | body 4/5/6；carry 结构/shape/dtype 不变式不满足 |
| 4 | `autodiff_unsupported` | 后端明确不支持 AD（附后端原始报错，不记 PASS） |

构建后自检：`ops/jax_lax_scan/build/jax_lax_scan --self-check`（只校验交付布局与 `.so` 可加载，不跑用例）。

---

## 3. 构建与运行

```bash
# 容器内（CANN 9.2.0-beta.2 + bisheng），工作区根：
OPX_OP=jax_lax_scan bash build.sh          # → ops/jax_lax_scan/build/jax_lax_scan
                                           #   + ops/jax_lax_scan/build/libjax_lax_scan_host.so

# 本机（工作区根，真机串行）：
scripts/nm.sh --op jax_lax_scan --action build
scripts/nm.sh --op jax_lax_scan --action smoke
scripts/nm.sh --op jax_lax_scan --action test
```

CMake 目标：`jax_lax_scan`（可执行文件，路径固定）与 `jax_lax_scan_host`（共享库）；
`--npu-arch=dav-2201`；`add_dependencies(jax_lax_scan jax_lax_scan_host)` 保证 `make jax_lax_scan`
同时产出 `.so`。

构建日志里 `.so` 编好后会打印一行 `[build] SO_OK path=... size_bytes=... sha256=...`
（`CMakeLists.txt` 的 POST_BUILD），运行期 `python/jax_lax_scan.py` 的 `_load_lib` 也会把实际加载的
同一组指纹打到 stderr（`[jax_lax_scan] SO path=...`）；两者可逐字比对，用于证明“构建产物 = 运行时加载库”。
`test.sh` 对 `jax_lax_scan` 额外检查 `build/libjax_lax_scan_host.so` 是否存在，缺失时按同一份源码补编；
`test` 动作第二段同时跑 `tests/test_jax_lax_scan.py`（pytest 交付件，判据与门禁同源）。

### 最近一次真机验证（当前交付版本）

| 动作 | run | 结论 | 证据 |
|---|---|---|---|
| build | `r20261026-231419` / #73 | `exit=0`，`BUILD_OK` + `SO_OK`，gate `pass` | `.cannbot/jax_lax_scan/workflow1/4-证据-r1/build-r1.log` |
| smoke | `r20261026-231512` / #74 | `SMOKE_VERDICT PASS`（`scan_xs_none_length5`/jit），gate `pass` | `.cannbot/jax_lax_scan/workflow1/4-证据-r1/smoke-r1.log` |
| test（共享 runner，权威） | `r20261026-233618` / #76 | `TEST_VERDICT PASS`，门禁 **7 条 / 13 次 = 6 PASS + 1 显式 UNSUPPORTED**（0 FAIL/ERROR/INVALID），gate `pass` | `.cannbot/jax_lax_scan/workflow1/4-证据-r1/runs-76/out/correctness.json` |
| test（pytest 交付件） | 同上 #76 | **19 条 / 33 次 = 18 PASS + 1 UNSUPPORTED**；`32 passed, 1 xfailed`（`tests=33 failures=0`） | `.cannbot/jax_lax_scan/workflow1/4-证据-r1/runs-76/out/pytest_jax_lax_scan.log` |
| benchmark / profile | — | **本轮未执行**（按 task 由 PM 在整轮退出后发起；本 README 不宣称性能结论） | — |

> 节点 5 只更新本 README 与 `docs/jax_lax_scan/design.md`；8 个算子代码文件、8 个测试资产、冻结输入与
> 统一入口的哈希均与 #73/#74/#76 记录一致，因此上述证据仍对应当前代码。文档改动会使工作区**总快照**
> 变化（文件数仍为 81）；若后续以总快照哈希做门禁，需在整轮退出后按新快照重跑。
>
> 同一版本的 test 第 1 次尝试（#75）在容器内已打印 `TEST_VERDICT PASS`，但收尾阶段远端 SSH 断连、
> 进程 `exit=255`，门禁判 `E_EXIT_NONZERO`，**不计通过**；#76 为唯一有效 test 证据。详见
> `.cannbot/jax_lax_scan/workflow1/4-修复记录.md` 与
> `.cannbot/jax_lax_scan/workflow1/4-测试执行记录.md`。

---

## 4. 精度口径

* 树结构 / 叶子数 / dtype / shape：完全一致（0 容差）；
* float32 叶子：`criteria/check.py`（`ops-precision-standard@1`）——`rtol=2^-10`、`atol=2^-16`、
  `matched_ratio >= 0.99`、`max_abs_error <= 1e-2`；
* int32 叶子（pytree `counter`）：逐元素精确相等（不进浮点判据）；
* NaN/±Inf：位置与符号逐元素精确一致，有限子集再走浮点容差；
* 设备守卫：逐叶子必须位于非 CPU 设备；
* 拒绝用例：`scan_bad_carry_shape` 必须 `rc=3` + `carry_shape_mismatch` + `abi_calls=0`；
* AD 用例：`scan_jvp_vjp` 允许「与 CPU 一致」或「显式 unsupported」；本版本为**显式 unsupported**
  （jvp/vjp 各 3 次同码，`path=trace`/`abi_calls=0`/`reason=so_no_autodiff`/`backend=npu`，附后端原文
  `UNIMPLEMENTED: Unsupported opcode: tuple (instruction: while.1)`），**不记 PASS**。

---

## 5. 性能口径

* 基线：**不设 CPU / torch 基线**（`baseline: null`、`comparison: same_device_only`）；CPU 延迟不作门槛；
* 测量：`scripts/nm.sh --op jax_lax_scan --action benchmark` → `tools/run_benchmark.py` →
  `scripts/scan_bench.py`，设备侧 `msopprof` 的 `Task Duration(us)`；
* 口径：blocking JIT 稳态，`warm_up=3`、`invocations=10`、median，`block_until_ready`（读
  `cases/jax_lax_scan/benchmark.json` 的 `measure`，不用 runner 旧默认值）；
* 用例：`scan_len128_width256_f32`、`scan_len1024_width256_f32`（**primary**）；
* 状态：**本轮未采集**。交付时不得据上文推算性能结论；采集结果由 PM 在整轮退出后以
  `--action benchmark` 落盘。

---

## 6. 已知限制

1. **`.so` 不接收 Python callable**：callable 只能在 Python 层解析成 `body_id`；未注册 callable 走 JAX trace。
2. **`.so` 没有自动微分**：JVP/VJP 一律走 JAX trace（`reason=so_no_autodiff`），不是“后端不支持”的等价说法；
   后端真的不支持时必须以 `UNSUPPORTED:autodiff_unsupported` + 后端原文稳定分类，不得记 PASS。
   **公开 torch 接口的 AD**：`.so` 未注册 torch autograd，JAX 也无法穿过 torch→jax 的 DLPack 边界；
   对 `requires_grad=True` 的 torch 输入会在 kernel 启动前直接抛
   `UNSUPPORTED:autodiff_unsupported`（不返回没有 `grad_fn` 的结果，不用 jax-family 的结论冒充）。
   本批冻结 AD 用例 `scan_jvp_vjp` 使用测试侧 `jax.Array`，其结论只代表 JAX trace 路径。
3. **kernel 单核执行**：carry 必须跨步留在 UB，故 `blockNum == 1`；本批 `W <= 4096`。
4. **范围外**：`unroll != 1`、`length = 0`、仅由 rank-0 叶子组成且没有 float32 累加器的 carry、`pytree y`、
   异构 `xs` 叶子、多设备。
5. **`y` 形状 = `carry` 中首个 float32 叶子的形状**（本批 5 个可执行 body 都满足）。
6. **运行环境依赖**：Python 接口的 torch 路径需要 `torch` + `torch_npu`；host driver（四动作入口）走
   numpy + ACL 设备缓冲路径，不依赖 torch。生产镜像 `opx-xla-npu:jax081-cann920-910b`（#136）只带
   JAX/JAXLIB，不含 `torch`/`pytest`；`test` 动作使用控制台测试镜像
   `opx-xla-npu:jax081-cann920-910b-test`（#145，含 pytest，不含 torch/torch_npu）。
   torch 交付接口使用已验收的 `lfx-op-dev-910b:torch210-cann910`（#80）作补充真机探针
   （5 条冻结数值用例 × eager/jit + length=2 回归），证据见
   `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch-family-run.log`、
   `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch/scan_correctness.json`；
   该探针不改变 JAX 主门禁的镜像和测量口径。
7. `carry` 为 pytree 时，叶子顺序固定为 flatten 顺序（tuple/list 按序、dict 按键序），与 `fixtures` 一致。
8. **稳定拒绝码只覆盖已注册的拒绝 body（`body_id ∈ {4,5,6}`）**：它们在 Python trace 期被映射为
   `carry_shape_mismatch`（`abi_calls == 0`）。未注册 callable 若真的改变 carry 结构/shape/dtype，
   走 JAX trace 时由 JAX 自身的错误报出，当前**不**统一改写成 `carry_shape_mismatch`；这是本批
   确认范围之外的通用能力，若要支持需另加 trace 期不变式校验并补用例。
9. **torch 路径的 stream 依赖**：`.so` 在自建 ACL stream 上启动 kernel 并同步该 stream；调用前会
   `torch.npu.current_stream().synchronize()` 等待输入生产完成。该路径已在补充镜像 #80 上由
   pytest 的 torch-family 用例（`tests/test_jax_lax_scan.py`）在设备 0 验证：5 条冻结数值用例
   eager/jit 全 PASS，并额外覆盖 length=2 偶数长度回归；逐叶与 NumPy 参考按 `criteria/check.py`
   同口径比较，且断言 init/xs 不被改写、输出与输入不别名。证据见
   `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch-family-run.log`、
   `.cannbot/jax_lax_scan/workflow1/2-证据-r6/torch/scan_correctness.json`。

10. **`xs=int32` 未被拒绝**：契约要求 `xs` 叶子为 `float32`；当前 Python/.so 只校验 carry dtype，
    `xs_dtype` 仅随 ABI 传入、未参与校验。`int32` 的 `xs` 会通过 Python 层，并被 kernel 按 float32
    解释（**未定义行为**，不属支持范围）；其它不支持 dtype 在构造 dtype 码时仍按 `dtype_mismatch`
    拒绝。若后续允许 `int32` xs，需在 Python 与 `.so` 两侧补显式转换/校验并加用例。
    （冻结用例 `scan_bad_carry_shape` 只覆盖 carry 不变式拒绝，不覆盖该分支。）

### 未验证项（如实保留，不写成通过）

* **R1**：torch-family 6 个校验错误分支——`test` 主镜像 #145 无 `torch_npu`，未覆盖；
* **R2**：`_run_trace` 的 compiled / `xs=None` 非 AD 路径——属交付接口需求之外，未覆盖；
* **性能**：`benchmark` / `profile` 本轮未执行，无性能结论；
* **AD**：`scan_jvp_vjp` 是显式 unsupported，不是数值 PASS；公开 torch 接口 AD 只有「kernel 前拒绝」的证据。
