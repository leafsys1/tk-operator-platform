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
  # 入口对 --out 的约定不统一，实测两种都有：
  #   · 当文件用：open(a.out,"w")                        —— GroundVTS
  #   · 当目录用：makedirs(a.out) + join(a.out,"…json")  —— Florence-2 / MERT / GOT-OCR / BiRefNet / RMBG
  # 传错了就是 FileExistsError / NotADirectoryError（实测两条都踩过：
  # 先是没有拍平导致读不到结论，拍平后目录型入口又撞上同名文件）。
  # 所以按入口源码判定它要哪种，各给各的路径，最后统一收口到 $OUT。
  if grep -qE 'makedirs\([[:space:]]*a\.out|join\([[:space:]]*a\.out' "$MDIR/entry.py"; then
    OUTARG="/workspace/out/_entry_out"
    OUTKIND="dir"
    mkdir -p "$OUT/_entry_out"
  else
    OUTARG="/workspace/out/entry_result.json"
    OUTKIND="file"
  fi
  PYCMD="cd /workspace/out && PYTHONPATH=/workspace/model:/workspace/probe python3 -u /workspace/model/entry.py --model /workspace/model --out $OUTARG"
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

# 目录型入口的产物要搬回 $OUT，否则归一化段落读不到 entry_result.json
# （顺带把 alpha.png / mask.png 这类产物一起搬上来，交付包里要用）。
if [ "${OUTKIND:-}" = "dir" ] && [ -d "$OUT/_entry_out" ]; then
  cp -a "$OUT/_entry_out/." "$OUT/" 2>/dev/null
  rm -rf "$OUT/_entry_out"
  echo "[probe] 已从 _entry_out 回收入口产物"
fi
# 兼容历史遗留：早期跑出来的嵌套目录 entry_result.json/
if [ -d "$OUT/entry_result.json" ]; then
  NEST="$OUT/entry_result.json"
  TMP=$(mktemp -d)
  cp -a "$NEST/." "$TMP/" 2>/dev/null
  rm -rf "$NEST"
  cp -a "$TMP/." "$OUT/" 2>/dev/null
  rm -rf "$TMP"
  echo "[probe] 拍平历史遗留的 entry_result.json 目录"
fi

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
