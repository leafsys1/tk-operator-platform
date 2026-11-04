#!/usr/bin/env bash
# 用法: probe.sh <分类> <模型目录名> <卡号>
#
# 两条执行路线：
#   ① 模型目录里有 entry.py（迁移时沉淀的专用入口）-> 优先用它。
#      这是"配方真正生效"的前提。早期版本永远跑通用探针，
#      导致配方命中了、但实际加载路线没变（实测 GroundVTS 反复报同一个错）。
#   ② 没有 entry.py -> 用通用探针 probe_model.py（多路线依次试）。
set -uo pipefail
C="$1"; M="$2"; CARD="${3:-2}"
ROOT=/data/model_corpus
MDIR="$ROOT/$C/$M"
OUT="$ROOT/_probe/$M"
mkdir -p "$OUT"
IMG="${PROBE_IMG:-model-corpus-probe:v8}"

if [ -f "$MDIR/entry.py" ]; then
  MODE="专用入口 entry.py"
  PYCMD='cd /workspace/out && PYTHONPATH=/workspace/model:/workspace/probe python3 -u /workspace/model/entry.py --model /workspace/model --out /workspace/out/entry_result.json'
else
  MODE="通用探针"
  PYCMD='cd /workspace/out && PYTHONPATH=/workspace/probe python3 -u /workspace/probe/probe_model.py --model /workspace/model --out /workspace/out/probe.json'
fi
echo "=== 探针: $C/$M  卡=$CARD  镜像=$IMG  模式=$MODE ==="

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
            $PYCMD" 2>&1
RC=$?

# 专用入口写的是 entry_result.json；这里补一份归一化的 probe.json，
# 让平台后续读 probe.json 的逻辑（深挖结论、产物校验）不用改。
if [ -f "$OUT/entry_result.json" ]; then
  python3 - "$OUT" <<'PYEOF' 2>/dev/null || true
import json, os, sys
out = sys.argv[1]
try:
    r = json.load(open(os.path.join(out, "entry_result.json"), encoding="utf-8"))
except Exception:
    raise SystemExit(0)
p = {
    "model": r.get("model"),
    "route": r.get("route") or "custom-entry",
    "loaded": bool(r.get("loaded")),
    "verdict": "LOAD-OK" if r.get("loaded") else "LOAD-FAIL",
    "n_params": r.get("n_params"),
    "elapsed_s": r.get("elapsed_s"),
    "device": r.get("device"),
    "source": "entry.py",
    "metrics": r.get("parts") or {},
    "caveat": r.get("caveat"),
}
json.dump(p, open(os.path.join(out, "probe.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PYEOF
fi

exit $RC
