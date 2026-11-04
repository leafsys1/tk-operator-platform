#!/usr/bin/env bash
# 用法: perf.sh <分类> <模型目录名> <卡号> [warmup] [iters]
# 在指定卡上跑性能测试，产物落到 _probe/<模型>/perf.json + PERF_REPORT.md
set -uo pipefail
C="$1"; M="$2"; CARD="${3:-0}"; WARM="${4:-3}"; ITERS="${5:-20}"
ROOT=/data/model_corpus
MDIR="$ROOT/$C/$M"
OUT="$ROOT/_probe/$M"
mkdir -p "$OUT"
IMG="${PROBE_IMG:-model-corpus-probe:v8}"
echo "=== 性能测试: $C/$M  卡=$CARD  warmup=$WARM iters=$ITERS ==="
docker run --rm --privileged \
  --device "/dev/davinci${CARD}" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e "ASCEND_RT_VISIBLE_DEVICES=${CARD}" \
  --network host --ipc host --shm-size 16g \
  -v "$MDIR:/workspace/model:ro" \
  -v "$OUT:/workspace/out:rw" \
  -v "$ROOT/_probe:/workspace/probe:ro" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMG" \
  bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
            cd /workspace/out && PYTHONPATH=/workspace/probe:/workspace/probe/_stubs \
            python3 -u /workspace/probe/perf_test.py \
              --model /workspace/model --out /workspace/out \
              --warmup $WARM --iters $ITERS" 2>&1
