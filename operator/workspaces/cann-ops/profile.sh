#!/usr/bin/env bash
# 四动作契约之一：profile（性能采集）。
# 契约：用 msopprof（= msprof op）采单算子；四类指标（耗时/数据搬运/存储使用/计算单元负载）
# 每项标 measured|estimated|unsupported|failed 并记来源；采不到就明确标，不按已采集验收。
# D9（2026-10-16 真机）：这四类在本镜像上**都能采到**（详见表头与 tools/run_profile.py）；
# 因此本动作的必需产物类型升级为 log + metrics（门禁曾对 profile 只要求 log）。
#
# 所需产物类型：log + metrics。
# 退出码：0 = 采集完成；非 0 = 跑不起来或解析失败（门禁判 fail）。
set -uo pipefail

OP=${OPX_OP:-elementwise_add}
WS=${OPX_WS:-/ws}
OUT=${OPX_OUT:-/out}
exe="$WS/ops/$OP/build/$OP"

echo "[profile] 算子=$OP 工作区=$WS 产物目录=$OUT"
if [ ! -x "$exe" ]; then
  # 容器每次都是干净的：profile 同样依赖 build 产物，缺了就先补编。
  echo "[profile] 本次容器里没有 build 产物（每次 run 都是干净容器）→ 自动补编一次；这是正常流程，不是缺 build"
  bash "$WS/build.sh" || { echo "[profile] FAIL 补编失败（看上面的编译日志）"; exit 2; }
  [ -x "$exe" ] || { echo "[profile] FAIL 补编后仍没有产物：$exe"; exit 2; }
fi

python3 "$WS/tools/run_profile.py" \
  --op "$OP" \
  --workspace "$WS" \
  --exe "$exe" \
  --bench "$WS/cases/$OP/benchmark.json" \
  --out "$OUT/metrics.json" \
  --md "$OUT/metrics.md" \
  --evidence-dir "$OUT/evidence" \
  --workdir "${OPX_PROFILE_WORKDIR:-/tmp/opx-profile}" \
  --aic-metrics "${OPX_PROFILE_METRICS:-Default}" \
  --warm-up "${OPX_PROFILE_WARMUP:-5}" \
  --launch-count "${OPX_PROFILE_LAUNCH_COUNT:-1}"
rc=$?

if [ "$rc" = "0" ]; then
  echo "[profile] PROFILE_VERDICT done（报告 $OUT/metrics.json）"
else
  echo "[profile] PROFILE_VERDICT failed rc=$rc"
fi
exit "$rc"
