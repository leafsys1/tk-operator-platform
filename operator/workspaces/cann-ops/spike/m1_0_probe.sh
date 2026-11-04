#!/usr/bin/env bash
# M1-0 穿刺（临时脚本，穿刺完即删）：Ascend C「直调」编译链在这个镜像里到底能不能用。
# 目的：① 有没有 ASC cmake 语言包（模板 CMakeLists 要 find_package(ASC)）
#       ② ccec/编译器在不在、--npu-arch=dav-2201 认不认
#       ③ 原样样例 add_custom 能不能编译 → 运行 → 过精度
#       ④ msprof/msopprof 在不在（D9 摸底前哨）
# 不做任何写入工作区的事（除 build 临时产物）。
set -uo pipefail

echo "### 0 环境"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-（空）}"
echo "user=$(id -un) pwd=$(pwd)"
head -2 /etc/os-release 2>/dev/null

echo "### 1 工具链"
for t in cmake make g++ gcc python3 msprof msopprof ccec opc tikcpp bishengir-compile ascendc; do
  printf '%-20s: ' "$t"; command -v "$t" || echo '(缺)'
done

echo "### 2 CANN 目录与 ASC cmake 包"
ls -d /usr/local/Ascend/ascend-toolkit/latest/*/ 2>/dev/null | head -20
echo "--- ASC cmake"
find /usr/local/Ascend -maxdepth 8 -name 'ASCConfig.cmake' 2>/dev/null | head -3
find /usr/local/Ascend -maxdepth 8 -name '*ASC*onfig*.cmake' 2>/dev/null | head -5
echo "--- tools/"
ls /usr/local/Ascend/ascend-toolkit/latest/tools/ 2>/dev/null | head -20
ls /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin 2>/dev/null | head

echo "### 3 python 依赖"
python3 -c "import torch, torch_npu, numpy; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| numpy', numpy.__version__)" 2>&1 | tail -3

echo "### 4 cmake 配置（模板原样，不改一行）"
cd /ws/spike/add_custom || exit 9
mkdir -p build && cd build || exit 9
cmake .. 2>&1 | tail -30
echo "=== cmake_rc=${PIPESTATUS[0]}"

echo "### 5 make 目标 add_custom（只编可执行文件目标）"
make -j8 add_custom 2>&1 | tail -80
echo "=== make_rc=${PIPESTATUS[0]}"
ls -la /ws/spike/add_custom/build/add_custom 2>/dev/null

echo "### 6 数据生成 + 运行 + 精度校验"
cd /ws/spike/add_custom/build || exit 9
python3 ../scripts/gen_data.py 2>&1 | tail -6
echo "=== gendata_rc=$?"
ls -la input output 2>/dev/null | head -12
rm -f output/output.bin
./add_custom 2>&1 | tail -25
echo "=== kernel_rc=$?"
ls -la output 2>/dev/null | head -8
python3 ../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tail -12
echo "=== verify_rc=$?"

echo "### 7 msprof 前哨"
msprof --help 2>&1 | head -15
echo "=== 穿刺结束"
