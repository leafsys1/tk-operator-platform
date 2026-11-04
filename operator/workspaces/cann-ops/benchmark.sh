#!/usr/bin/env bash
# 四动作契约之一：benchmark（基准测试）。
# 契约：与 profiling **分开执行**；固定条件（用例/代码快照/镜像 digest/预热次数/重复次数/
# 同步方式/测量范围/聚合方法）写进 /out/performance.json，原始样本进 /out/evidence/。
# 对照口径（方案 §4.6）：基线 = 官方对应实现 torch.add，**同卡同条件**（同用例/dtype/预热/重复/
# 测量范围/聚合）；设备由 nm_test.sh 的远端 flock 独占，否则耗时会被邻居污染。
#
# 所需产物类型：log + performance（门禁按类型找，见 docs/gate-report-v1.md §5）。
# 退出码：0 = 采集完成；非 0 = 跑不起来或基线/候选采样失败（门禁判 fail，不写假数据）。
set -uo pipefail

OP=${OPX_OP:-elementwise_add}
WS=${OPX_WS:-/ws}
OUT=${OPX_OUT:-/out}
exe="$WS/ops/$OP/build/$OP"
so="$WS/ops/$OP/build/lib${OP}_host.so"

echo "[benchmark] 算子=$OP 工作区=$WS 产物目录=$OUT"
if [ ! -x "$exe" ]; then
  # 容器每次都是干净的（--rm + 全新 rsync）：benchmark 与 test 一样依赖 build 产物，缺了就先补编。
  echo "[benchmark] 本次容器里没有 build 产物（每次 run 都是干净容器）→ 自动补编一次；这是正常流程，不是缺 build"
  bash "$WS/build.sh" || { echo "[benchmark] FAIL 补编失败（看上面的编译日志）"; exit 2; }
  [ -x "$exe" ] || { echo "[benchmark] FAIL 补编后仍没有产物：$exe"; exit 2; }
fi

# jax_lax_scan 的 benchmark 与 test 共用 Python host-driver，它在运行时会
# dlopen 同一构建目录里的 libjax_lax_scan_host.so。当前构建缓存只回填
# 可执行文件，所以缓存命中后仅检查 exe 会让 msopprof 里的应用先以 1
# 退出，表面上看起来像“没采到 kernel”。这里与 test.sh 保持同一契约：
# 缺 .so 就用同一份源码快照补编，不改用例或测量口径。
if [ "$OP" = "jax_lax_scan" ] && [ ! -f "$so" ]; then
  echo "[benchmark] 缺少 $so（构建缓存只回填了 exe）→ 按同一份源码补编一次"
  bash "$WS/build.sh" || { echo "[benchmark] FAIL .so 补编失败（看上面的编译日志）"; exit 2; }
  [ -f "$so" ] || { echo "[benchmark] FAIL 补编后仍没有共享库：$so"; exit 2; }
fi

python3 "$WS/tools/run_benchmark.py" \
  --op "$OP" \
  --workspace "$WS" \
  --exe "$exe" \
  --bench "$WS/cases/$OP/benchmark.json" \
  --out "$OUT/performance.json" \
  --md "$OUT/performance.md" \
  --evidence-dir "$OUT/evidence" \
  --workdir "${OPX_BENCH_WORKDIR:-/tmp/opx-bench}"
rc=$?

if [ "$rc" = "0" ]; then
  echo "[benchmark] BENCH_VERDICT done（报告 $OUT/performance.json）"
else
  echo "[benchmark] BENCH_VERDICT failed rc=$rc"
fi
exit "$rc"
