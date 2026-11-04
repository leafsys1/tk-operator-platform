# jax_lax_while_loop · Ascend C 直调算子

JAX 0.8.1 `lax.while_loop(cond_fun, body_fun, init_carry)` 在 Ascend 910B1（DAV_2201）
上的设备实现与交付件说明。语义以 `tasks/jax_lax_while_loop/spec/spec.json`（spec v1，
`spec_version_id=7`）为准；精度判据 `ops-precision-standard@1`；oracle 为同版本 JAX 0.8.1 CPU。
设计文档见 `docs/jax_lax_while_loop/design.md`。

> 证据作用域 `operator-v1`：只覆盖 `ops/jax_lax_while_loop/` 与
> `cases/jax_lax_while_loop/`。本文不声称任何未在目标镜像上取证的性能或能力。
> 本文所述实现为**修复后最终代码**（node 0 修复 `ComputeIntLeaf` int32 张量叶「只写 4 字节」
> 截断缺陷），修复内容与 build/smoke/test/benchmark 四动作证据见 §9。

## 1. 语义

`while_loop(cond_fun, body_fun, init_carry)` 反复执行 `carry = body_fun(carry)`，直到
`cond_fun(carry)` 返回**标量 bool** 为假，返回最终 carry。

- `cond_fun` 必须返回 0-d `bool`；非标量 / 非 bool → `INVALID_CONDITION`。
- `body_fun` 必须返回与输入 carry **完全相同**的 pytree 结构，每叶 shape/dtype/`weak_type`
  不变；结构漂移 → `CARRY_STRUCTURE_MISMATCH`，shape 漂移 → `CARRY_SHAPE_MISMATCH`，
  dtype 漂移 → `CARRY_DTYPE_MISMATCH`。
- 支持运行期动态迭代次数（0 次 / 1 次 / 多次）；`eager`、`jit`、浮点 carry 的 JVP。
- 安全迭代上限（本实现的 `safety_limit` 参数）触发 →
  `ITERATION_LIMIT_EXCEEDED`（设备安全阀，不是参数校验拒绝）。
- reverse-mode VJP / `grad` **不是 v1 强制能力**；本接口不提供反向后端，遇到时按未支持处理，
  不计 PASS。

错误归一化保留上游 JAX 异常的类型与单行摘要（`upstream_exception_type` /
`upstream_message` / `upstream_message_sha256`）。

## 2. 调用接口

```python
from ops.jax_lax_while_loop.python.jax_lax_while_loop import while_loop, while_loop_compiled

final = while_loop(cond_fun, body_fun, init_carry, safety_limit=None)

# 测试/门禁的编译态入口（不是第二个公开 JAX API）：
final = while_loop_compiled(cond_fun, body_fun, init_carry,
                            mode="primal", safety_limit=None, jit=True)
primal, tangent = while_loop_compiled(cond_fun, body_fun, init_carry,
                                      mode="jvp", tangent=tangent, jit=True)
```

- 返回值是与 `init_carry` 同结构的 JAX pytree；`weak_type` 逐叶与 CPU golden 对齐。
- 两条入口都经宿主 C ABI `libjax_lax_while_loop_host.so` → AscendC kernel 执行，
  输出证据写入模块级 `last_dispatch`（`path/route/abi_calls/iterations/cond_hits/body_hits/
  runtime_token/counter_source/plan_id/plan_cache/compile_count/leaves/tree`）。
- Python 日志使用 `logging`。

### 设备执行路线（host_step）

一次宿主入口只推进一个阶段：

| stage | 设备动作 | 计数 |
|---|---|---|
| 0 | 只求值一次 `cond` | `cond_hits += 1` |
| 1 | 执行一次 `body`，再求值一次 `cond`；达到安全上限时不再求值 `cond` | `body_hits += 1`，必要时 `cond_hits += 1` |

`body_hits`/`cond_hits`/`limit_flag`/`runtime_token`/`cond_result` 全部由 kernel 写设备内存，
host 只做 D2H 读回（`counter_source=kernel`）。`abi_calls` 是 C ABI 按实际 launch 次数累加的
**原始整数**（`host_step` 路由下 = `1 + iterations`，安全阀触发时为 `1 + limit`）。
正常终止：`body_hits=iterations=n`、`cond_hits=n+1`；安全阀：三者都等于 `limit`。

## 3. 实现结构

```text
ops/jax_lax_while_loop/
├── CMakeLists.txt                       # 两个 target：exe + host .so
├── op_host/
│   ├── jax_lax_while_loop.asc           # 四动作入口（协议校验 + 转交 CLI）
│   ├── jax_lax_while_loop_abi.h         # ctypes 对齐的 C ABI 结构
│   └── jax_lax_while_loop_pyapi.asc     # extern "C" C ABI 实现（Tiling + launch + D2H）
├── op_kernel/
│   ├── jax_lax_while_loop_kernel.asc    # AscendC kernel（body/cond/JVP + 设备计数）
│   └── jax_lax_while_loop_tiling.h      # kernel/host/pyapi 共享 tiling 结构
├── python/
│   ├── jax_lax_while_loop.py            # Python 调用接口（交付件 #2）
│   └── jax_lax_while_loop_cli.py        # host driver（case.json/inputs.npz 协议）
└── scripts/                             # 独立 CPU oracle、输入生成、runner 适配（测试工程节点）
```

计算分工（编码规则 A1/A2）：Python 层只做 pytree 校验、cond/body 的 jaxpr 线性化、
host_step 循环控制、弱类型还原与证据封装；**全部数值**（body、cond、JVP 切向量）
由 kernel 按实际 carry 在 NPU 上计算。包装层不做 Torch/CANN 内置算子代算，不做
transpose/contiguous/cast 等实质性数据变换。

### 计划来源（不是按 case 预置结果）

- `body_fun` 被 `jax.make_jaxpr` 追踪，解释成每个 carry 叶子的线性表达式：
  `out = const + Σ scale_i * (reduce_i ? Σ src_i : src_i)`，最多 2 项，项可为整叶求和
  （支持 `a + mean(b)`）。逐元素项要求源叶子 numel 与输出一致；归约项源叶子
  numel ≤ 256。
- `cond_fun` 归一化为 `leaf <op> leaf|const`、`all(leaf <op> const)` 或恒真。
- 不认识的 primitive（非线性 body、slice/concatenate 等）显式报 `UNSUPPORTED_PLAN`，
  不猜测、不静默降级；本批冻结用例的合法 body/cond 都能线性化。
- `plan_id` 是计划的 sha256 前 16 位，只依赖计算计划（不含运行期 limit 值），因此
  同一计划下不同 runtime limit 的 `plan_id` 稳定；`compile_count` 为进程内该计划
  首次物化次数（fresh worker 首调为 1），`plan_cache ∈ {miss, hit}`。

## 4. 构建与四动作

在目标镜像（镜像 #145 `opx-xla-npu:jax081-cann920-910b-test`，容器设备 0 / 宿主设备 5）
按顺序执行：

```bash
scripts/nm.sh --op jax_lax_while_loop --action build
scripts/nm.sh --op jax_lax_while_loop --action smoke
scripts/nm.sh --op jax_lax_while_loop --action test
scripts/nm.sh --op jax_lax_while_loop --action benchmark
```

`build.sh` 在 `ops/jax_lax_while_loop/build/` 产出：

- `jax_lax_while_loop`（四动作入口，路径固定不变）；
- `libjax_lax_while_loop_host.so`（Python 接口的 C ABI；`--npu-arch=dav-2201`）。

构建日志会打印 `.so` 的路径/大小/sha256 指纹；Python 接口加载时再次打印运行期实际加载的
同一指纹，用于“源码 → 构建产物 → 实际加载库”的关联。

> **复现入口**：`scripts/nm.sh --op jax_lax_while_loop --action <build|smoke|test|benchmark>`
> （镜像 #145 / 宿主设备 5 / 容器设备 0 / 用例集 `cases@288876910a5e2843`，独占卡串行）。
> 修复后四动作的 run 台账与结论见 §9。

## 5. 精度与证据口径

- 浮点：`rtol=2**-10`、`atol=2**-16`、`matched_ratio>=0.99`、`max_abs_error<=1e-2`，
  逐输出叶子独立判定（`criteria/check.py`，`ops-precision-standard@1`）；非有限位逐元素精确。
- 整型：逐元素精确；`float0` 切向量：dtype/shape/nbytes=0 结构精确。
- 结构/shape/dtype/`weak_type`：精确相等，不进入容差。
- 设备真实性：`path=so`、`backend != cpu`、`platforms` 含 `npu`、原始整数 `abi_calls>=1`、
  `counter_source=kernel`、`route ∈ {device_loop, host_step}`、`device_id=0`、
  `runtime_token` 非空，并通过独立后端探测与设备节点探测。
- 身份：`spec_version_id=7`、`suite_ref=cases@288876910a5e2843`、
  `criteria_ref=ops-precision-standard@1`、镜像 #145、宿主设备 5、容器设备 0。

`path=trace`、CPU、模拟、`xfail`、`unsupported`、`NOT_RUN` 一律不计 PASS。

## 6. 性能口径

只记录基线，不设硬阈值：`blocking` 执行、`warm_up=3`、`invocations=10`、`median`、
`block_until_ready`、同设备/镜像/规格/用例集。`host_step` 路由每个 body 需要一次宿主入口
与一次 D2H 读回，**短循环小 carry**（p01）主要受 ABI/同步开销影响；**长循环大张量**
（p02）主要受 kernel tile 吞吐影响。本实现单核执行，未做多核切分与流水优化（不在本轮范围）。

## 7. 已知限制

1. **body 必须是仿射线性计划**：支持 `add/sub/mul(常量)/div(常量)/reduce_sum/mean` 与广播；
   非线性 body（如 `sin`、`c*c`）或 slice/concatenate 会显式报 `UNSUPPORTED_PLAN`，不在
   v1 覆盖范围。
2. **cond 形态**：支持标量比较（int32/float32）与 `all(tensor > const)`；更复杂的谓词
   （多叶布尔组合、动态索引）不在 v1 覆盖范围。`all()` 的叶子 numel ≤ 256（kernel 标量区）。
3. **weak_type**：0-d weak 叶子可精确还原；非 0-d weak 叶子显式报 `UNSUPPORTED_WEAK_TYPE`。
4. **JVP 只对浮点 carry 叶**：整型叶切向量为 `float0`（零尺寸标记），按结构精确断言。
5. **VJP/grad 不提供**：v1 非强制能力；没有反向后端，不以任何形式冒充。
6. **`jit` 的含义**：`while_loop_compiled` 复用“已线性化并缓存的设备计划”，不是把 Python
   `lax.while_loop` 交给 XLA jit；数值与计数仍全部来自设备 kernel。
7. **单核执行**：为数据相关 cond 与读改写计数选择单核；性能只作基线，不做多核优化承诺。
8. **输入上传**：pytree 叶子经 `numpy` 上传到本算子的设备缓冲区（I/O，不是计算）；
   大 carry 的 H2D 开销计入基线。

## 8. 错误码

| code | 触发时机 | 证据 |
|---|---|---|
| `INVALID_CONDITION` | cond 非标量/非 bool（device 之前） | `path=reject`、`abi_calls=0`、上游类型/摘要 |
| `CARRY_STRUCTURE_MISMATCH` | body 返回不同 pytree 结构 | 同上 |
| `CARRY_SHAPE_MISMATCH` | body 返回不同 shape | 同上 |
| `CARRY_DTYPE_MISMATCH` | body 返回不同 dtype | 同上 |
| `ITERATION_LIMIT_EXCEEDED` | 设备安全阀触发 | `path=so`、`iterations=cond_hits=body_hits=limit`、`limit_flag=1`、原始整数 `abi_calls>0` |

`UNSUPPORTED_PLAN` / `device_error` / `invalid_argument` 属运行错误（非拒绝）；runner 记
`ERROR`，不计 PASS。

## 9. 修复记录与证据（node 0）

workflow5 白盒验收在真机（run `runs/r20261029-161341/`）暴露 6 项 int32 张量叶白盒失败
（`while_wb_int_sum_reduce` / `while_wb_int_two_reduce` / `while_wb_int_two_elem_terms`
的 eager+jit）。根因在 `op_kernel/jax_lax_while_loop_kernel.asc` 的 `ComputeIntLeaf`：
element-wise 项只用 `ReadScalarInt` 回读源张量首元素，写出也只 `DataCopyPad` 4 字节，
其余元素保留陈旧初始值。node 0 将其重写为与 `ComputeFloatLeaf` 一致的 tile 循环契约：

- 以 `WHILE_TILE_ELEMS=8192` 为步长地 tile：`count = min(remain, WHILE_TILE_ELEMS)`；
- 纯 reduce/常量叶整块广播 `cEff`（`RoundToInt(常量 + Σ scale_i * reduce_i)`）；
- 逐元素叶按 `count * sizeof(int32_t)` 读入 `x1/x2`，逐元素 `GetValue → RoundToInt →
  SetValue`；每 tile 以 `{1, count * sizeof(int32_t), 0, 0}` 全量写回 `outI_[j][off]`。

修复后 int32 张量叶全量写出、6 项历史失败全部转 PASS，且未改动 spec/用例/golden/阈值。

- 修复文件：`ops/jax_lax_while_loop/op_kernel/jax_lax_while_loop_kernel.asc`
  sha256 `e0037bc15def494a4da9b564614e21cecbd2fc4ca64c87e8cf1603f883270743`；
- 修复后 operator-v1 快照（13 文件）：`cb954eb1d713e80c4070a9d8a45066c85e5162a9581cf5f2d09010e7857dd077`。

四动作门禁（同镜像 #145 / 宿主设备 5 / 容器设备 0 / Ascend 910B1，独占锁）：

| 动作 | run / 台账 | 结论 |
|---|---|---|
| build | `runs/r20261029-174551/` / #251 | BUILD_OK，exit=0 |
| smoke | `runs/r20261029-174641/` / #252 | SMOKE_VERDICT PASS（`while_one_iteration` eager） |
| test | `runs/r20261029-174734/` / #253 | TEST_VERDICT PASS（黑盒 run_cases 51/51；pytest 138 passed；白盒 120/120） |
| benchmark | `runs/r20261029-180418/` / #254 | BENCH_VERDICT done（仅基线，无硬阈值） |

构建产物（真机）：`libjax_lax_while_loop_host.so` sha256 `c494afb6…ac771`（593456 B）、
可执行文件 `jax_lax_while_loop` sha256 `c0fa2081…0a943ce4`（381992 B）、
ABI `opx-while-loop-python-abi/1`。性能基线（设备侧 Task Duration，10 样本 median）：
`while_short_small_carry 4.910us`、`while_long_large_tensor 4.890us`（仅基线，未设达标阈值）。

> 能力边界保持不变：本节修复不扩大语义范围——§1/§7 的 body 仿射线性计划、cond 形态、
> weak_type、JVP（仅浮点叶）、VJP/grad 不提供、`jit` 语义、单核执行等边界均未改变。
