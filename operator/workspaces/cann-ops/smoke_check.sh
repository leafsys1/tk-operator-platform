#!/usr/bin/env bash
# 动作 smoke（开发内冒烟，2026-10-23 新增）：编译 + **一条最小用例**跑通。
#
# 为什么：算子刚写完就等全套 test/benchmark 才发现构建失败/跑不通，返工成本高。
# 开发对话约定：代码写完后必须先 build + smoke；smoke 不 PASS 不许进入正式 test/benchmark，
# 也不许对用户说「开发完成」。
#
# 注意：工作区根已有 `smoke.sh`（那是**环境自检**脚本，不是动作入口）；本动作入口叫
# `smoke_check.sh`，由统一执行器按 ACTION_SPECS.smoke.entry 调用：
#   scripts/nm.sh --op <op> --action smoke
# 产物 /out/smoke.json（机读）+ /out/smoke.md + log；用例/参考/判据与正式 test 同源
# （cases/<op>/cases.json、ops/<op>/scripts/golden.py、criteria/check.py）。
# 退出码：0 = 最小用例 PASS；1 = 未通过/运行失败；2 = 缺 build 产物或前置文件。
set -uo pipefail

OP=${OPX_OP:-elementwise_add}
WS=${OPX_WS:-/ws}
OUT=${OPX_OUT:-/out}
exe="$WS/ops/$OP/build/$OP"

echo "[smoke] 算子=$OP 工作区=$WS 产物目录=$OUT"
if [ ! -x "$exe" ]; then
  # 与 test.sh/benchmark.sh 同一约定：容器每次干净，缺产物就按同一份源码补编一次
  echo "[smoke] 本次容器里没有 build 产物 → 自动补编一次；这是正常流程，不是缺 build"
  bash "$WS/build.sh" || { echo "[smoke] FAIL 补编失败（看上面的编译日志）"; exit 2; }
  [ -x "$exe" ] || { echo "[smoke] FAIL 补编后仍没有产物：$exe"; exit 2; }
fi

python3 "$WS/tools/smoke_case.py" \
  --op "$OP" \
  --workspace "$WS" \
  --exe "$exe" \
  --cases "$WS/cases/$OP/cases.json" \
  --golden "$WS/ops/$OP/scripts/golden.py" \
  --out "$OUT/smoke.json" \
  --md "$OUT/smoke.md" \
  --evidence-dir "$OUT/evidence" \
  --workdir "${OPX_SMOKE_WORKDIR:-/tmp/opx-smoke}"
rc=$?

if [ "$rc" = "0" ]; then
  echo "[smoke] SMOKE_VERDICT PASS（最小用例跑通；报告 $OUT/smoke.json）"
else
  echo "[smoke] SMOKE_VERDICT FAIL rc=${rc}（先修到冒烟通过，再跑正式 test/benchmark）"
fi
exit "$rc"
