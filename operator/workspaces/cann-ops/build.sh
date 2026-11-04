#!/usr/bin/env bash
# 四动作契约之一：build（编译）—— 在容器里编 ops/<op> 的 Ascend C 直调工程。
# 契约（方案 v1.1 §6）：统一执行入口调用工作区根的 <动作>.sh，产物写 /out，日志由入口 tee 到 /out/run.log。
# 本动作的必需产物类型是 log（构建日志）。
set -uo pipefail

OP=${OPX_OP:-elementwise_add}
WS=${OPX_WS:-/ws}
JOBS=${OPX_BUILD_JOBS:-8}
proj="$WS/ops/$OP"

echo "[build] 算子=$OP 工作区=$WS 作业数=$JOBS"
if [ ! -d "$proj" ]; then
  echo "[build] FAIL 找不到算子工程：$proj"
  exit 2
fi

# 镜像自带的 CANN 环境（OPX_ENV_RECIPE 通常已经 source 过；这里只是兜底，失败不致命）
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
  # shellcheck disable=SC1090
  . "$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1 || true
fi

mkdir -p "$proj/build" || exit 1
cd "$proj/build" || exit 1
rm -f CMakeCache.txt

echo "[build] --- cmake 配置 ---"
cmake .. 2>&1
rc_cmake=$?
if [ "$rc_cmake" != "0" ]; then
  echo "[build] FAIL cmake 配置失败 rc=$rc_cmake"
  exit "$rc_cmake"
fi

echo "[build] --- make 目标 $OP ---"
make -j"$JOBS" "$OP" 2>&1
rc_make=$?
exe="$proj/build/$OP"
if [ "$rc_make" != "0" ] || [ ! -x "$exe" ]; then
  echo "[build] FAIL 编译失败 rc=${rc_make}（产物 ${exe} 不存在或不可执行）"
  exit 1
fi

sha=$(sha256sum "$exe" | cut -d' ' -f1)
size=$(stat -c %s "$exe")
echo "[build] BUILD_OK target=$OP exe=ops/$OP/build/$OP sha256=$sha size_bytes=$size"
exit 0
