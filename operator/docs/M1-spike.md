# M1-0 穿刺：Ascend C 直调编译链（容器实测）

> 状态：✅ **通过** ｜ 2026-10-15 ｜ 方式：`scripts/nm_test.sh` 走正式管线（镜像 `lfx-op-dev-910b:torch210-cann910`，
> 设备 0 独占锁，run_id `m1-0-spike`，**调试型会话不计入 D8 预算、不进 runs 台账**）
> 脚本：`workspaces/cann-ops/spike/m1_0_probe.sh`（临时，穿刺完删）

## 一、验的是什么

M1 选定的实现路线是 **Ascend C 直调（KernelLaunch）**，用 CANNBot 的
`ascendc-direct-invoke-template` → `references/add_custom/` 样例工程改造。开工前必须先证明
**这个镜像里能把直调工程编出来、能在 NPU 上跑、能过精度**，否则后面全是空中楼阁。

## 二、容器环境事实（实测）

| 项 | 值 |
|---|---|
| CANN | `/usr/local/Ascend/cann-9.1.0-beta.1`（**完整装**，与宿主机那份「开发/编译装」不同） |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.1.0-beta.1`（由 `OPX_ENV_RECIPE` 的 set_env 链给出） |
| ASC 编译链 | `find_package(ASC)` **成功**：`CMAKE_ASC_COMPILER=…/bin/bisheng`、`CMAKE_ASC_LLD_LINKER=…/aarch64-linux/ccec_compiler/bin/ld.lld` |
| CXX | GNU 12.3.1；`cmake`/`make`/`g++` 齐全 |
| 算子编译工具 | `ccec`、`opc`、`bishengir-compile`、`msopgen`、`msopst`(`mindstudio_opst-26.0.0`) 都在；`tikcpp`/`ascendc` 无（不需要） |
| 性能采集 | `msprof` 有 `op` 子命令（= msopprof 入口）、`msopprof` 独立命令也在 → **D9 有戏**；2026-10-16（M1-b）已在容器里定档：四类指标均可采（见 `docs/PROGRESS.md`「M1-b 实测记录」） |
| Python 依赖 | python 3.12.13、`torch 2.10.0+cpu`、`torch_npu 2.10.0.post4`、`numpy 1.26.4` |
| OS | openEuler 24.03 (LTS-SP3) |

## 三、实测结果（原样模板，一行未改）

```
cmake ..                        → rc=0（ASC 语言包解析成功）
make -j8 add_custom             → [ 50%] Building ASC object CMakeFiles/add_custom.dir/op_host/add_custom.asc.o
                                  [100%] Linking ASC executable add_custom      → rc=0
python3 scripts/gen_data.py     → 32768 元素 fp32，golden.bin 写出
./add_custom（device 0）         → rc=0，output/output.bin 写出（131072 B）
python3 scripts/verify_result.py→ Verification PASSED! Shape: (32768,) Max diff: 0.0
```

## 四、结论与对后续的影响

1. **路线定档**：M1 走「Ascend C 直调 + 模板改造」。模板默认 `--npu-arch=dav-2201`＝ Atlas A2 系列
   （910B 对应值），**不需要改架构参数**。
2. **只编可执行目标是可行的**（`make add_custom`）；模板 `CMakeLists.txt` 的第二个 target
   （`libadd_custom_ops.so`，torch 接入层）M1 **不编**（它对 benchmark 基线不是必需 —— 基线直接用
   `torch.add`；它依赖 torch cmake prefix，是额外风险面）。若后续要给 PyTorch 通路再加。
3. **判据/标杆都能在容器里做**：numpy 1.26.4 + torch/torch_npu 在容器内齐备，
   所以「独立参考实现（fp64 标杆）」「torch.add 基线」都放容器里跑，产物直接落 `/out` —— 不把输出搬回本机判。
4. **本机不需要 Ascend 工具链**：编译只在容器里发生，本机负责源码与用例（符合方案 §6「本机改代码 → 上机测一次」）。
5. **D9 前置成立**：`msprof op` / `msopprof` 存在；2026-10-16（M1-b）已定档「四类指标均可采」，命令与字段固化在 `workspaces/cann-ops/tools/run_profile.py`。

## 五、风险更新

| 原风险 | 现状 |
|---|---|
| 镜像里 ASC 编译链不完整 | ✅ 排除（bisheng 编译链完整，`find_package(ASC)` 成功） |
| 直调 exe 在 910B 起不来 | ✅ 排除（device 0 上跑通，逐元素精度 0 误差） |
| msprof/msopprof 缺失 | ✅ 排除（命令都在；支持范围待 M1-8） |
| 共享机设备锁 | 本次 8 张卡当时全空闲，锁一次拿到（`OPX_LOCK_OK devices=0 mode=exclusive`） |
