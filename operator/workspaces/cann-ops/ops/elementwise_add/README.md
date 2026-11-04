# elementwise_add · Ascend C 直调实现

首个贯通样例算子（D6）。规格来源：`spec_versions` v1 —— **float32、同 shape、不广播**、
`N ∈ [1, 1048576]`、非法 shape / 不支持的 dtype 一律 **reject**。

## 目录

```
op_kernel/elementwise_add_tiling.h     Tiling 常量与结构体（kernel 与 host 共用）
op_kernel/elementwise_add_kernel.asc   NPU 侧计算（KernelAdd 类 + elementwise_add_kernel 入口）
op_host/elementwise_add.asc            Host 直调层 + main（参数化调用接口）
op_host/data_utils.h                   文件读写小工具（样例原样）
scripts/golden.py                      **独立**参考实现（CPU fp64 → 按输出 dtype 舍入）
CMakeLists.txt                         单一 target：直调可执行文件 elementwise_add
```

## 调用接口（交付件之一，用例执行器按此调用）

```bash
elementwise_add --n-a <N> --n-b <M> --dtype float32 --in-a a.bin --in-b b.bin --out c.bin
```

- 输入/输出都是**裸二进制** float32（无头部），元素数分别是 N / M / N。
- 成功：退出码 0，stdout 含 `EXEC op=elementwise_add kernel=elementwise_add_kernel arch=dav-2201 impl=ascendc-direct-invoke`，
  末行 `OK n=… dtype=float32 blocks=… cores=…`。
  这行 `EXEC` 标记不是装饰：用例执行器要求它存在，缺了就判 `INVALID`
  （方案 §4.5 的验证项「实际执行的是否为候选算子路径，有无未声明的回退」）。
- 拒绝：非 0 退出 + 打印拒绝码（机判要认码，不只看退出码）。

| 退出码 | 拒绝码 | 条件 |
|---|---|---|
| 3 | `REJECT_SHAPE_MISMATCH` | `N_A ≠ N_B`（v1 不广播） |
| 4 | `REJECT_N_OUT_OF_RANGE` | `N ∉ [1, 1048576]` |
| 5 | `REJECT_UNSUPPORTED_DTYPE` | dtype ≠ float32 |
| 2 | `REJECT_BAD_ARGS` | 参数缺失/未知 |
| 1 | — | ACL/设备/写文件失败 |

## 构建与运行（只在容器里做）

```bash
# 构建（工作区根的 build.sh 干的就是这件事）
mkdir -p build && cd build && cmake .. && make -j8 elementwise_add

# 手跑一条形状（真实跑用例走 tools/run_cases.py）
./elementwise_add --n-a 4096 --n-b 4096 --dtype float32 \
    --in-a a.bin --in-b b.bin --out c.bin
```

容器事实（M1-0 实测，见 `docs/M1-spike.md`）：CANN `9.1.0-beta.1` 完整装、
`find_package(ASC)` 由 bisheng 编译链解析、`--npu-arch=dav-2201` 即 910B（Atlas A2）。

## 性能（M1-b，`benchmark`/`profile`）

- **基准**（`benchmark.sh` → `tools/run_benchmark.py`）：基线 = 官方 `torch.add`（`tools/torch_add_baseline.py`），
  与候选**同一把尺子** —— 两侧都走 `msopprof` 量**设备侧 `Task Duration(us)`**（不用 host 墙钟：
  那会把 PyTorch 的派发开销当成算子差异）。用例/条件见 `cases/elementwise_add/benchmark.json`。
  实测（1M）：torch.add 8.46us vs 候选 7.98us ≈ 1.06×。
- **采集**（`profile.sh` → `tools/run_profile.py`）：`msopprof --aic-metrics=Default`，四类指标
  （耗时/数据搬运/存储使用/计算单元负载）落 `metrics.json`；vector 算子无 cube 数据列标 `unsupported`。
  实测（1M）：`op_time 8.1us`、`mte2_ratio 0.75`（搬运受限取向）、`vec_ratio 0.09`。
- 两个动作都需要**独占设备**（`nm_test.sh` 远端 flock），否则耗时会被邻居污染。

## 与 CANNBot 模板的关系

改造自 `ascendc-direct-invoke-template/references/add_custom`：
- 改：算子名、调用接口参数化、**只保留直调 target**（去掉 PyTorch 接入层 `lib*_ops.so`）。
- 未改：kernel 计算逻辑（`AscendC::Add` + UB 三段式 + `DataCopyPad` 尾块）、Tiling 计算方式、`data_utils.h`。
- 许可：模板文件带 CANN OSL 2.0 头（该许可限华为 AI 处理器用途，须保留声明）。
