#!/usr/bin/env bash
# 批量探针：对已下载完成的模型逐个跑，结果落 _probe/<model>/probe.json
set -uo pipefail
ROOT=/data/model_corpus
CARD="${1:-2}"
DONE="$ROOT/_probe/_done.txt"; touch "$DONE"
n=0; ok=0; fail=0
for c in nlp multi-modal audio cv scientific-computing; do
  [ -d "$ROOT/$c" ] || continue
  for m in "$ROOT/$c"/*/; do
    M=$(basename "$m")
    [ -f "$m/.download_ok.json" ] || continue
    grep -qx "$M" "$DONE" && continue
    echo "### $(date '+%T') $c/$M"
    out=$("$ROOT/_probe/probe.sh" "$c" "$M" "$CARD" 2>&1 | tail -6)
    echo "$out"
    if echo "$out" | grep -q "VERDICT=LOAD-OK"; then ok=$((ok+1)); else fail=$((fail+1)); fi
    echo "$M" >> "$DONE"
    n=$((n+1))
  done
done
echo "===== 批量探针结束: 共 $n, OK $ok, FAIL $fail ====="
