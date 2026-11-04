#!/usr/bin/env bash
# 四动作契约之一：test（正确性测试）—— 容器里跑用例矩阵，逐例与**独立**参考实现比对。
# 契约：产物 /out/correctness.json（机判）+ /out/correctness.md（人看的伴生品）+ /out/evidence/（最小证据）；
# 本动作的必需产物类型是 log + correctness（门禁按类型找，不按文件名猜）。
# 退出码：0 = 全部用例通过（run 才会是 ok）；非 0 = 有用例没过或跑不起来（门禁会判 E_EXIT_NONZERO + E_CASE_FAILED）。
set -uo pipefail

OP=${OPX_OP:-elementwise_add}
WS=${OPX_WS:-/ws}
OUT=${OPX_OUT:-/out}
exe="$WS/ops/$OP/build/$OP"
so="$WS/ops/$OP/build/lib${OP}_host.so"

echo "[test] 算子=$OP 工作区=$WS 产物目录=$OUT"
if [ ! -x "$exe" ]; then
  # 容器每次都是干净的（--rm + 每次全新 rsync），build 产物不会跨 run 存在。
  # 所以这里按同一份源码快照补编一次 —— 不复制编译逻辑，直接调 build.sh；
  # 编译是否可复现由 correctness.json 里记的候选 exe sha256 兜着（两次编译同哈希 = 可复现）。
  echo "[test] 本次容器里没有 build 产物（每次 run 都是干净容器）→ 自动补编一次；这是正常流程，不是缺 build"
  bash "$WS/build.sh" || { echo "[test] FAIL 补编失败（看上面的编译日志）"; exit 2; }
  [ -x "$exe" ] || { echo "[test] FAIL 补编后仍没有产物：$exe"; exit 2; }
fi

# jax_lax_scan / jax_lax_associative_scan / jax_lax_cond 的 Python 接口依赖同一目录下的 lib<op>_host.so。
# 构建缓存目前只回填可执行文件（控制台 scripts/nm_test.sh），缓存命中时 exe 存在但 .so 缺失，
# Python 接口在 _load_lib 处必然失败 —— 这与 kernel 数值无关。这里对这些算子单独补 .so
# 存在性检查（向后兼容扩展；其他算子行为不变），缺失则按同一份源码补编，不伪造产物。
if { [ "$OP" = "jax_lax_scan" ] || [ "$OP" = "jax_lax_associative_scan" ] || [ "$OP" = "jax_lax_cond" ]; } && [ ! -f "$so" ]; then
  echo "[test] 缺少 $so（构建缓存只回填了 exe）→ 按同一份源码补编一次"
  bash "$WS/build.sh" || { echo "[test] FAIL .so 补编失败（看上面的编译日志）"; exit 2; }
  [ -f "$so" ] || { echo "[test] FAIL 补编后仍没有共享库：$so"; exit 2; }
fi

run_shared_runner() {
  python3 "$WS/tools/run_cases.py" \
    --op "$OP" \
    --workspace "$WS" \
    --exe "$exe" \
    --cases "$WS/cases/$OP/cases.json" \
    --golden "$WS/ops/$OP/scripts/golden.py" \
    --out "$OUT/correctness.json" \
    --md "$OUT/correctness.md" \
    --evidence-dir "$OUT/evidence" \
    --workdir "${OPX_CASE_WORKDIR:-/tmp/opx-cases}"
}
run_shared_runner
rc1=$?
# 仅对 NPU 后端初始化的瞬时错误重试一次；保留首次原始结果，不隐藏算子/用例失败。
if [ "$rc1" -ne 0 ] && [ -f "$OUT/correctness.json" ] && \
   grep -qE "Unable to initialize backend 'npu'|no supported devices found for platform npu" \
     "$OUT/correctness.json"; then
  cp "$OUT/correctness.json" "$OUT/correctness-attempt1.json"
  echo "[test] shared runner 命中 NPU 初始化瞬时错误 → 保留 attempt1 后重试一次"
  run_shared_runner
  rc1=$?
  [ "$rc1" -eq 0 ] || cp "$OUT/correctness.json" "$OUT/correctness-attempt2.json"
fi

# ---- 第二段：jax_lax_scan / jax_lax_associative_scan 的 pytest 交付件 ----
# 说明：pytest 是交付件入口，判据与门禁同源（criteria/check.py + CPU 子进程 golden），
# 但不替代上面的共享 runner；两段都跑，退出码取非 0 最大值。
rc2=0
if [ "$OP" = "jax_lax_scan" ]; then
  echo "[test] scan 分支第二段：pytest 交付件（tests/test_jax_lax_scan.py）"
  if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "[test] PYTEST_UNAVAILABLE（镜像里没有 pytest，第二段不能静默跳过）"
    rc2=2
  else
    pushd "$WS" >/dev/null || exit 2
    python3 -m pytest tests/test_jax_lax_scan.py -q -p no:cacheprovider \
      --junitxml="$OUT/pytest_jax_lax_scan.xml" 2>&1 | tee "$OUT/pytest_jax_lax_scan.log"
    rc2=${PIPESTATUS[0]}
    popd >/dev/null || true
    echo "[test] pytest_rc=${rc2}（日志 $OUT/pytest_jax_lax_scan.log）"
  fi
elif [ "$OP" = "jax_lax_associative_scan" ]; then
  echo "[test] assoc-scan 分支第二段：pytest 交付件（tests/test_jax_lax_associative_scan.py）"
  if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "[test] PYTEST_UNAVAILABLE（镜像里没有 pytest，第二段不能静默跳过）"
    rc2=2
  else
    pushd "$WS" >/dev/null || exit 2
    python3 -m pytest tests/test_jax_lax_associative_scan.py -q -p no:cacheprovider \
      --junitxml="$OUT/pytest_jax_lax_associative_scan.xml" 2>&1 | tee "$OUT/pytest_jax_lax_associative_scan.log"
    rc2=${PIPESTATUS[0]}
    popd >/dev/null || true
    echo "[test] pytest_rc=${rc2}（日志 $OUT/pytest_jax_lax_associative_scan.log）"
  fi
elif [ "$OP" = "jax_lax_cond" ]; then
  echo "[test] cond 分支第二段：pytest 交付件（tests/test_jax_lax_cond.py）"
  if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "[test] PYTEST_UNAVAILABLE（镜像里没有 pytest，第二段不能静默跳过）"
    rc2=2
  else
    pushd "$WS" >/dev/null || exit 2
    python3 -m pytest tests/test_jax_lax_cond.py -q -p no:cacheprovider \
      --junitxml="$OUT/pytest_jax_lax_cond.xml" 2>&1 | tee "$OUT/pytest_jax_lax_cond.log"
    rc2=${PIPESTATUS[0]}
    popd >/dev/null || true
    echo "[test] pytest_rc=${rc2}（日志 $OUT/pytest_jax_lax_cond.log）"
  fi
fi

rc=$rc1
if [ "$rc2" -ne 0 ] && { [ "$rc" -eq 0 ] || [ "$rc2" -gt "$rc" ]; }; then
  rc=$rc2
fi

if [ "$rc" = "0" ]; then
  echo "[test] TEST_VERDICT PASS（全部用例通过；报告 $OUT/correctness.json）"
else
  echo "[test] TEST_VERDICT FAIL rc=${rc}（逐例原因见 ${OUT}/correctness.md 与 ${OUT}/evidence/）"
fi
exit "$rc"
