#!/usr/bin/env bash
# 在探针容器里执行指定 python 脚本（迁移平台 Agent 的 run_py 工具专用）
#
# 用法: agent_exec.sh <分类> <模型目录名> <卡号> <脚本文件名>
#   脚本文件需已放在 $ROOT/_probe/<模型目录名>/_agent/ 下（容器内即 /workspace/out/_agent/）
#
# 为什么单独一个脚本：docker run 的参数必须与 probe.sh 完全一致，否则
# agent 在 run_py 里验证通过、probe 换个环境又失败，等于白验。
set -uo pipefail
C="$1"; M="$2"; CARD="${3:-0}"; SCRIPT="$4"
ROOT=/data/model_corpus
MDIR="$ROOT/$C/$M"
OUT="$ROOT/_probe/$M/_agent"
IMG="${PROBE_IMG:-model-corpus-probe:v8}"

mkdir -p "$OUT"
if [ ! -f "$OUT/$SCRIPT" ]; then
  echo "AGENT_EXEC_ERROR: 脚本不存在 $OUT/$SCRIPT"
  exit 2
fi

docker run --rm --privileged \
  --device "/dev/davinci${CARD}" --device /dev/devmm_svm \
  --device /dev/davinci_manager --device /dev/hisi_hdc \
  -e "ASCEND_RT_VISIBLE_DEVICES=${CARD}" \
  --network host --ipc host --shm-size 16g \
  -v "$MDIR:/workspace/model:ro" \
  -v "$OUT:/workspace/out:rw" \
  -v "$ROOT/_probe:/workspace/probe:ro" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMG" \
  bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
            if [ -f /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh ]; then set +u; source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh; set -u; fi
            cd /workspace/out
            PYTHONPATH=/workspace/model:/workspace/probe python3 -u /workspace/out/$SCRIPT" 2>&1
