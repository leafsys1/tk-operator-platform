#!/usr/bin/env bash
# 按类别并行探针：batch_probe_cat.sh <category> <card>
set -uo pipefail
ROOT=/data/model_corpus
C="$1"; CARD="$2"
DONE="$ROOT/_probe/_done_${C}.txt"; touch "$DONE"
n=0; ok=0; fail=0
[ -d "$ROOT/$C" ] || { echo "no dir $C"; exit 0; }
for m in "$ROOT/$C"/*/; do
  M=$(basename "$m")
  [ -f "$m/.download_ok.json" ] || continue
  grep -qx "$M" "$DONE" && continue
  echo "### $(date '+%T') [$C] $M"
  out=$("$ROOT/_probe/probe.sh" "$C" "$M" "$CARD" 2>&1 | tail -6)
  echo "$out"
  echo "$M" >> "$DONE"
  if echo "$out" | grep -q "VERDICT=LOAD-OK"; then ok=$((ok+1)); else fail=$((fail+1)); fi
  n=$((n+1))
done
echo "===== [$C] 结束: 共 $n, OK $ok, FAIL $fail ====="
