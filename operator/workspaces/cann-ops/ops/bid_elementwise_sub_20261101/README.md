# bid_elementwise_sub_20261101 · Ascend C 直调实现

逐元素减法：`y[i] = x1[i] - x2[i]`（float32 / 同 shape / 不广播）。规格 **v3.0**：输入为两个同
shape 的一维连续 float32 NPU 张量，输出为同 shape、同 dtype、与输入同一 NPU 设备的新连续张量；
`N ∈ [1, 1048576]`；非法 shape / 不支持的 dtype / N 越界一律 **reject**（无 broadcast、无 alpha、无
out=、无候选 CPU 回退 / 无 torch 回退）。判据档 `criteria/ops-precision-standard@1`，权威参考
（oracle）= CPU `torch.sub(x1.cpu(), x2.cpu())`；设计文档见 `docs/bid_elementwise_sub_20261101/design.md`。

> 本版已通过真机验证：build / smoke / test（15/15）/ benchmark 均通过；profile 已在优化前基线代码上采集。
> v3.1 优化后 bench_64k 设备耗时 5.74us（相对优化前 8.88us，-35.36%），bench_1m 8.22us（无显著变化）。
> 双卡/跨 NPU 仍不执行、记为未验证。详见文末「已知限制」。

## 目录

```
op_kernel/bid_elementwise_sub_tiling.h       Tiling 常量与结构体（host / kernel 共用，固定 8192 tile）
op_kernel/bid_elementwise_sub_kernel.asc     NPU 侧计算（KernelBidSub + bid_elementwise_sub_kernel 入口）
op_host/bid_elementwise_sub_20261101.asc     Host 直调层 + main（四动作走的可执行文件）
op_host/bid_elementwise_sub_20261101_pyapi.asc  Python 接口共享库（extern "C" C ABI）
op_host/data_utils.h                         文件读写小工具
python/bid_elementwise_sub_20261101.py       Python 调用接口（ctypes → libbid_elementwise_sub_20261101_host.so）
scripts/golden.py                            **独立**参考实现（权威 = CPU torch.sub；fp64 仅辅助）
CMakeLists.txt                               两个 target：可执行文件 + 共享库
```

## 调用接口

### 1) 可执行文件（四动作门禁按此调用；入口 `ops/<op>/build/<op>` 保持不变）

```bash
bid_elementwise_sub_20261101 --n-a <N> --n-b <M> --dtype float32 \
    --in-a a.bin --in-b b.bin --out y.bin
```

- 输入/输出都是**裸二进制** float32（无头部），元素数分别是 N / M / N。
- 成功：退出码 0，stdout 含
  `EXEC op=bid_elementwise_sub_20261101 kernel=bid_elementwise_sub_kernel arch=dav-2201 impl=ascendc-direct-invoke`，
  末行 `OK n=… dtype=float32 blocks=… cores=…`。
  `EXEC` 标记是**用例执行器的路由标记**（要求存在以确认候选走 Ascend C 直调路径，缺失判 `INVALID`），
  单凭这一行**不能**证明设备 kernel 已真实执行：真实执行证据 = 退出码 0 + 非空输出文件 + `OK …` 行
  （或真机 test/benchmark 的运行结果）。本节仅为静态描述。
- 拒绝：非 0 退出 + 打印拒绝码（机判要认码，不只看退出码）。

| 退出码 | 拒绝码 | 条件 |
|---|---|---|
| 3 | `REJECT_SHAPE_MISMATCH` | `N_A ≠ N_B`（v3.0 不广播，必须同 shape） |
| 4 | `REJECT_N_OUT_OF_RANGE` | `N ∉ [1, 1048576]` |
| 5 | `REJECT_UNSUPPORTED_DTYPE` | dtype ≠ float32 |
| 2 | `REJECT_BAD_ARGS` | 参数缺失/未知参数 |
| 1 | — | ACL/设备/读写文件失败 |

### 2) Python 接口（交付件，ctypes 封装 host C ABI）

```python
from ops.bid_elementwise_sub_20261101.python.bid_elementwise_sub_20261101 import subtract

y = subtract(x1, x2)      # y = x1 - x2，out 为新张量
```

- `x1`、`x2`：**NPU 上的一维连续 float32 张量**，同 shape；返回**同形状、同 dtype、同 NPU 设备的新连续张量**
  （非原地，不与输入共享 storage）。不广播、无 alpha、无 out=；`N = a.numel() ∈ [1, 1048576]`。
- 非零 `storage_offset` 的连续视图：经 `int(a.data_ptr())` 取址（data_ptr 已含 offset），按逻辑元素计算。
- 包装层不做任何数值计算（编码规则 A1），只做 dtype/shape/rank/连续性/设备校验并调用 `run`。
- 拒绝：以 `ValueError` / `TypeError` 抛出，消息里带拒绝码（供 pytest 断言）：

| 条件 | 异常 | 拒绝码 |
|---|---|---|
| 非张量（`None` / `list` 等） | TypeError | `REJECT_BAD_ARGS` |
| CPU 输入 / CPU-NPU 混用 | TypeError | `REJECT_BAD_ARGS` |
| 非 float32 | ValueError | `REJECT_UNSUPPORTED_DTYPE` |
| shape 不一致 | ValueError | `REJECT_SHAPE_MISMATCH` |
| rank≠1 | ValueError | `REJECT_BAD_ARGS` |
| 非连续 | ValueError | `REJECT_BAD_ARGS` |
| 跨 NPU（同类型、不同设备索引） | ValueError | `REJECT_BAD_ARGS` |
| `N ∉ [1, 1048576]` | ValueError | `REJECT_N_OUT_OF_RANGE` |
| run 返回 1（ACL/建流/同步失败） | RuntimeError | `RUN_FAILED` |

- ABI（`ops/bid_elementwise_sub_20261101/op_host/bid_elementwise_sub_20261101_pyapi.asc`，`extern "C"` 导出）：

```c
int  bid_elementwise_sub_20261101_run(const void *a_dev, const void *b_dev, void *out_dev,
                                      int64_t n, int dtype_code, int device_id);  // dtype_code: 0=float32
const char *bid_elementwise_sub_20261101_abi_version(void);  // "opx-python-abi/1"
```

  C ABI 返回码与可执行文件一致：0 成功 / 1 运行失败 / 2 REJECT_BAD_ARGS / 4 REJECT_N_OUT_OF_RANGE /
  5 REJECT_UNSUPPORTED_DTYPE。本库**不 `aclInit` / 不 `aclFinalize`**（逻辑见 pyapi.asc 注释）：
  仅 `aclrtSetDevice` + 查核数 + 建流 + 同步，把 ACL 上下文生命周期交给调用方（torch_npu）。
  shape（码 3）在 Python 层已校验，故 `.so` 侧 `run` 不返回 3——属有意设计，非缺码。
- 同步语义（阻塞）：`subtract` 在把 `data_ptr` 交给候选 ABI（内部新建**独立 ACL 流**）之前，对目标 NPU 设备
  做一次 `torch_npu.npu.synchronize(device)` 保守整体同步，确保 torch_npu 已把输入产出到显存；候选 ABI 侧
  `aclrtSynchronizeStream` 同步并传播返回码。整体为**同步阻塞**语义，不做多卡/跨流异步承诺。

## 构建与运行（真机；经 `scripts/nm.sh` 远程入口）

```bash
# 先算计划（不动容器）
scripts/nm.sh --dry-run --op bid_elementwise_sub_20261101 --action build

# 上机编译（产出 build/bid_elementwise_sub_20261101 与 build/libbid_elementwise_sub_20261101_host.so）
scripts/nm.sh --op bid_elementwise_sub_20261101 --action build

# 冒烟：编译 + 一条最小用例（开发内自检；SMOKE_VERDICT PASS 后才允许 test/benchmark）
scripts/nm.sh --op bid_elementwise_sub_20261101 --action smoke

# 正式正确性测试（全部用例；自动补编）
scripts/nm.sh --op bid_elementwise_sub_20261101 --action test

# 基准（独占卡，慢）
scripts/nm.sh --op bid_elementwise_sub_20261101 --action benchmark

# 采集（独占卡，慢）
scripts/nm.sh --op bid_elementwise_sub_20261101 --action profile
```

- 四动作 build/test/benchmark/profile 入口不变：走工作区根脚本 `scripts/nm.sh` 转发控制台仓库根
  `scripts/nm_test.sh`，最终落到可执行文件 `ops/bid_elementwise_sub_20261101/build/bid_elementwise_sub_20261101`
  （CMake 可执行 target 保持不变）。
- ctypes `.so` 由 **build 动作产出**：CMake 同源 `add_library(bid_elementwise_sub_20261101_host SHARED …)`
  产出 `ops/bid_elementwise_sub_20261101/build/libbid_elementwise_sub_20261101_host.so`，
  `add_dependencies(…exe …host)` 保证只 `make` 可执行文件时也顺带编出共享库；本轮**未实际编译** `.so`
  （真机 build 阶段才产出）。
- **开发顺序硬规则**：代码完成后先 `build` 再 `smoke`，仅 `SMOKE_VERDICT PASS` 才允许 `test` / `benchmark` /
  宣称「开发完成」；本轮不提前执行，仅记录规则。
- 手动构建等价于：`mkdir -p build && cd build && cmake .. && make -j8 bid_elementwise_sub_20261101`
  （`add_dependencies` 顺带编出共享库）。
- pytest（真机）：`pytest -q tests/test_bid_elementwise_sub_20261101.py`，读
  `cases/bid_elementwise_sub_20261101/cases.json`，候选走 Python 接口、`scripts/golden.py` 跑参考、
  `criteria/check.py` 判 PASS/FAIL/拒绝码。

## 精度口径

- 权威标杆（golden）= **CPU `torch.sub(x1.cpu(), x2.cpu())`**，与规格 v3.0 oracle 逐字一致
  （`scripts/golden.py::compute_golden`，numpy float32 入/出，`inf-inf=nan`、`x-nan=nan`、`±0` 按符号均
  遵循 CPU torch 的 IEEE 754 语义）。
- numpy fp64 单次舍入（`compute_golden_fp64`）**仅作辅助交叉核对，非权威、不参与判据**；两者不一致时以
  `torch.sub` 为准。
- 判据：`criteria/ops-precision-standard@1`（`criteria/check.py`）。float32 混合容差：
  `rtol = 2⁻¹⁰`、`atol = 2⁻¹⁶`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 0.01`
  （`max_abs_error_limit = max(0.01, 32×ULP(float32)=2⁻¹⁸) ≈ 0.01`，固定限占优）。
  非有限位（NaN/±Inf）在 `check.py` 中**单独逐元素精确判定**（同号 Inf / 同为 NaN），不参与容差通过率统计，
  也不放宽——非有限位是比容差更严的精确判定。

## 性能口径

- 用例：`cases/bid_elementwise_sub_20261101/benchmark.json`（主 `bench_64k` = N 65536、上界 `bench_1m` = N 1048576）。
- 基线：官方 `torch.sub`（torch_npu, device 0，`tools/torch_sub_baseline.py`），与候选**同一把尺子**——两侧
  都走 `msopprof` 量设备侧任务时长（`device_kernel_task_duration`）。
- 采集口径：`warm_up=2`、`invocations=5`、`retries_per_invocation=3`、`aggregate=median`，主值 =
  设备侧 `Task Duration` 中位数（微秒）。benchmark/profile 需要**独占卡**（远端 flock）。
- **性能结论以实测为准**：bench_64k 5.74us（优化前 8.88us，-35.36%）、bench_1m 8.22us（无显著变化）；不虚构加速比 / HAP / 时延数值。

## 已知限制

- 仅 float32；不广播；`N ≤ 1048576`（超出走 reject）；无 alpha、无 out=、无 scale。候选实现无任何
  CPU 回退 / torch 回退（候选路径为 Ascend C 直调 kernel，`EXEC … impl=ascendc-direct-invoke` 标记行标识
  该路由；真实设备执行以退出码 0 + 输出/运行证据为准，单凭 EXEC 行不构成执行证明）。
- 搬运长度恒为有效元素数 `count`：kernel 只搬 `count * sizeof(float)` 有效字节、不对 GM 侧向上取整到 32B，
  因此 GM 侧不读/写逻辑 in/out 元素之外的任何字节。非 32B 对齐尾块（`N % 8 ≠ 0`，如 N=1/17）由 `DataCopyPad`
  的 `padParams`（isPad + rightPad = `8 - count%8` + paddingValue=0）在 **UB 侧**补齐 32B（dummy=0）；
  UB→GM 时框架自动丢弃 dummy，GM 侧只写回有效字节。不存在「最多越界读写 7 元素、依赖分配器余量」的行为。
- 设备侧对次正规数是否 FTZ/DAZ：**UNKNOWN**，由数值边界用例在真机确认（与本仓参考算子口径一致）。
- **真机 build / smoke / test / benchmark 已在 v3.1 候选上通过**（runs #13–#16；test 15/15 PASS）；
  profile 已在优化前基线代码上采集（run #10）；v3.1 候选未重复 profile（可选动作）。
  本工作区提供 `scripts/nm.sh` 四动作入口和 `scripts/run_pytest.sh` 受控 pytest/示例入口，可复现上机验证。
- **双卡未执行（未验证）**：跨 NPU 拒绝逻辑在 Python 接口层实现但未真机验证，不声称覆盖双卡路由。
- Python 接口需真机 `torch`/`torch_npu` 与已构建的 `.so`；本机后端起 venv 无 torch/torch_npu（仅本机
  静态环境缺 NPU 后端；可经 `scripts/nm.sh` 远程 Docker 入口上真机）。缺依赖时 ImportError 记为失败
  而非跳过（不 `importorskip`，不静默跳过缺依赖）。
- 性能结论：bench_64k 5.74us（相对优化前 -35.36%，相对 torch.sub 0.645×）；bench_1m 8.22us（无显著变化，相对 torch 1.027×）。