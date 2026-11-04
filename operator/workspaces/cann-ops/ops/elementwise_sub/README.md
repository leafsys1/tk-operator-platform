# elementwise_sub · Ascend C 直调实现

逐元素减法：`y = a - b`；`alpha ≠ 1` 时为 `y = a - alpha*b`（torch 语义）。
规格 v1：**float32、同 shape、不广播**、`N ∈ [1, 1048576]`、非法 shape / 不支持的 dtype 一律 **reject**；
判据档 `ops-precision-standard@1`，设计文档见 `docs/elementwise_sub/design.md`。

## 目录

```
op_kernel/elementwise_sub_tiling.h     Tiling 常量与结构体（kernel 与 host 共用，含 opMode/alpha）
op_kernel/elementwise_sub_kernel.asc   NPU 侧计算（KernelSub 类 + elementwise_sub_kernel 入口）
op_host/elementwise_sub.asc            Host 直调层 + main（四动作走的可执行文件）
op_host/elementwise_sub_pyapi.asc      Python 接口共享库（extern "C" C ABI）
op_host/data_utils.h                   文件读写小工具
python/elementwise_sub.py              Python 调用接口（ctypes → libelementwise_sub_host.so）
scripts/golden.py                      **独立**参考实现（CPU fp64 → 一次舍入到 fp32）
CMakeLists.txt                         两个 target：可执行文件 + 共享库
```

## 调用接口

### 1) 可执行文件（四动作门禁按此调用，协议与 elementwise_add 完全一致）

```bash
elementwise_sub --n-a <N> --n-b <M> --dtype float32 --in-a a.bin --in-b b.bin --out y.bin
```

- 输入/输出都是**裸二进制** float32（无头部），元素数分别是 N / M / N。
- **alpha 不在命令行上**：协议固定 `alpha=1`（规格要求「未知参数必须拒绝」，所以这份 argv 协议本身是接口）。
- 成功：退出码 0，stdout 含
  `EXEC op=elementwise_sub kernel=elementwise_sub_kernel arch=dav-2201 impl=ascendc-direct-invoke`，末行 `OK n=… dtype=float32 blocks=… cores=…`。
- 拒绝：非 0 退出 + 打印拒绝码（机判要认码）。

| 退出码 | 拒绝码 | 条件 |
|---|---|---|
| 3 | `REJECT_SHAPE_MISMATCH` | `N_A ≠ N_B`（v1 不广播） |
| 4 | `REJECT_N_OUT_OF_RANGE` | `N ∉ [1, 1048576]` |
| 5 | `REJECT_UNSUPPORTED_DTYPE` | dtype ≠ float32 |
| 2 | `REJECT_BAD_ARGS` | 参数缺失/未知 |
| 1 | — | ACL/设备/读写文件失败 |

### 2) Python 接口（交付件，签名对齐 `torch.sub`）

```python
from ops.elementwise_sub.python.elementwise_sub import sub

y = sub(a, b)                    # y = a - b
y = sub(a, b, alpha=2.0)         # y = a - 2*b
sub(a, b, out=a)                 # 原地：a -= b（等价 a.sub_(b)）
```

- `a`、`b`：npU 上的 float32 连续张量，同 shape；
- 返回：新张量（`out=` 时返回该缓冲本身）；
- 拒绝：`REJECT_UNSUPPORTED_DTYPE` / `REJECT_SHAPE_MISMATCH` / `REJECT_N_OUT_OF_RANGE` /
  `REJECT_BAD_ARGS` 以 `ValueError` / `TypeError` 抛出，消息里带拒绝码。
- **alpha≠1 的实现**：torch 在 fp32 下是「先算 `alpha*b` 并舍入，再减」；接口做**一次**候选 kernel 调用
  （`elementwise_sub_alpha_run` → `OP_MODE_FMASUB`，UB 内 `Muls` + `Sub`），Python 层不做任何数值计算
  （编码规则 A1）。alpha 为 2 的幂时该口径与 oracle（fp64 + 单次舍入）逐位一致。
- ABI：`int elementwise_sub_run(const void*, const void*, void*, int64_t n, int dtype_code, int device_id)`、
  `int elementwise_sub_scale_run(const void* x, void* out, int64_t n, float alpha, int device_id)`、
  `const char* elementwise_sub_abi_version()` → `"opx-python-abi/1"`。本库不 `aclInit`/`aclFinalize`。

## 构建与运行（只在容器里做）

```bash
scripts/nm.sh --op elementwise_sub --action build    # 产出 build/elementwise_sub + build/libelementwise_sub_host.so
scripts/nm.sh --op elementwise_sub --action smoke    # 开发内自检：编译 + 一条最小用例
scripts/nm.sh --op elementwise_sub --action test     # 正式正确性（共享 runner 为门禁权威）
```

最近一次真机验证（规格 v2 / `cases@8df658c27183dac8`）：

| 动作 | 结论 | 证据 |
|---|---|---|
| build | `BUILD_OK`，exe sha256 `271f121f1a790eb6…`，383480 B | `runs/r20261023-152406` |
| smoke | `SMOKE_VERDICT PASS` | 同 run 的 `/out/smoke.json` |
| test | `TEST_VERDICT PASS`，29/29（PASS 29 / FAIL 0 / ERROR 0） | `/out/correctness.json` |
| pytest | `29 passed` | `tests/test_elementwise_sub.py` |

手动构建等价于：`mkdir -p build && cd build && cmake .. && make -j8 elementwise_sub`（`add_dependencies` 会顺带编出共享库）。

pytest（真机）：`pytest -q tests/test_elementwise_sub.py`，读 `cases/elementwise_sub/cases.json`，
用 Python 接口跑候选、`scripts/golden.py` 跑参考、`criteria/check.py` 判 PASS/FAIL/拒绝码，
并额外打印 `bit_exact`（逐位一致元素数）与 `torch_diff`（与同卡 torch.sub 的差异数）。

## 精度口径

- 标杆：CPU fp64 算 `a - alpha*b`，**一次**舍入到 fp32（`scripts/golden.py`）；禁止用 NPU 实现当参考。
- 判据：`criteria/check.py`（`ops-precision-standard@1`）：fp32 `rtol 2⁻¹⁰`、`atol 2⁻¹⁶`、`matched_ratio ≥ 0.99`、
  非有限位逐元素精确（NaN/±Inf 位置与符号一致）。
- 规格文本要求按位精确，而当前判据档是混合容差（`@1`；按位档 `@2` 不存在）。pytest 会把 `bit_exact` 打印出来，
  作为「接近按位」的证据，不当作门禁放宽。
- **alpha≠1 的求值口径**（规格 v2）：设备侧 `y = a - alpha*b` 按 fp32「先舍入 `alpha*b`、再与 a 相减」实现，
  与 torch CPU 的求值顺序一致。
  - alpha 取 2 的幂（2.0 / 0.5 / -1.0，用例覆盖的三个值）时该式无二次舍入损失，与 fp64 一次舍入的标杆**逐位一致**
    （真机实测：sub_alpha_002 4096/4096、sub_alpha_003 1048576/1048576 逐位；sub_alpha_001 4092/4093、
    sub_alpha_004 1023/1024 的唯一差异是 `a = b = FLT_MAX, α = 2` 的上溢点，见「已知限制」）。
  - 不在 2 的幂上的 alpha：理论上存在二次舍入口径差，**未穷举验证**（规格 `_meta.unknowns` 已登记）。

## 性能口径

- 基线：同卡官方 `torch.sub`（`tools/torch_sub_baseline.py`），与候选**同一把尺子** —— 两侧都走 `msopprof`
  量设备侧 `Task Duration(us)`；不设加速比硬线。
- 用例：`cases/elementwise_sub/benchmark.json`（N = 1 … 1048576，主用例 1M，alpha=1）；
  `scripts/nm.sh --op elementwise_sub --action benchmark` 产出 `performance.json` / `.md`。

## 已知限制

- 仅 float32；不广播；`N ≤ 1048576`（与规格一致，超出走 reject）。
- 命令行协议不含 alpha/out（alpha/out 只能经 Python 接口）。
- 设备侧对次正规数是否 FTZ/DAZ：**UNKNOWN**，由数值边界用例在真机确认。
- 性能报告里 `baseline_op_time_us` 指标文案在共享 runner 中固定写作 "torch.add kernel"（共享脚本未改），
  实际采集的是 `benchmark.json` 指定的 torch.sub 基线（`op_name` 字段为真实值）。

### alpha≠1 的范围前提（规格 v2 明确限定）

- 规格要求 alpha≠1 时 **`|alpha*b| ≤ FLT_MAX`**。超出该前提时 `alpha*b` 在 fp32 内已经上溢为 ±Inf，
  结果不做逐位承诺。实例：`a = b = FLT_MAX, alpha = 2` → 设备侧得 `-Inf`，fp64 标杆得 `-3.4028235e+38`。
  这是「先乘后减」与「双精度一次舍入」的固有差异，**不是实现缺陷**，故以规格限定范围而非放宽判据处理。
- 用例 `sub_alpha_004` 已按该前提调整：a 走 special（NaN/±Inf/FLT_MAX），b 走 normal 分布。

### 为什么 device 侧不用 fp64

DaVinci vector 核**不支持 double**。真机编译实测：

```
elementwise_sub_kernel.asc:104: cast to/from double precision floating variable is not allowed in aicore function
```

因此 device 侧无法复刻标杆的「双精度求值 + 单次舍入」，只能走 fp32 先乘后减（上面已论证 alpha 为 2 的幂时逐位等价）。

### 踩坑记录：507057 / Launch kernel failure

scale 路径曾把 `bDev == nullptr` 传给 kernel，但 `CopyIn` 仍无条件对 b 侧发 `DataCopyPad`。后果不是本次调用报错，
而是**毒化下一次 kernel 启动**（`507057 SUSPECT REMOTE ERROR` + `[Bisheng] Launch kernel failure!`），排查成本很高。
现在 `CopyIn` 用 `if (tiling->opMode != OP_MODE_SCALE)` 守住 b 侧搬运 —— **重构时不要去掉这个 guard**。
