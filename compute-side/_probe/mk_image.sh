#!/usr/bin/env bash
# 用法: mk_image.sh <分类> <模型目录名> [是否保留本地 0|1]
# 把模型权重打成 docker 镜像 → save → 上传 OBS → 删本地。
# 产物: _probe/<模型>/weights_image.json
#
# 三道闸门：
#   ① 体积 —— 超过 MAX_IMG_GB 直接跳过（默认 20）
#   ② 磁盘 —— 需要 3×权重 + 6GB 可用
#   ③ 串行 —— 由调用方保证一次只跑一个
set -uo pipefail
C="$1"; M="$2"; KEEP="${3:-0}"
ROOT=/data/model_corpus
MDIR="$ROOT/$C/$M"
OUT="$ROOT/_probe/$M"
mkdir -p "$OUT"
OBS="${OBS_IMG_BASE:-obs://model-corpus/images/weights/}"
MAX_GB="${MAX_IMG_GB:-20}"
TMP="/tmp/_imgone"
mkdir -p "$TMP"

SLUG=$(echo "$M" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9._-]/-/g' | cut -c1-60)
TAG="model-corpus-weights:${SLUG}"
TAR="$TMP/${SLUG}.tar.gz"

# ① 体积闸门
WG=$(du -sm "$MDIR" 2>/dev/null | cut -f1)
WG=${WG:-0}
if [ "$WG" -gt $((MAX_GB * 1024)) ]; then
  echo "SKIP_SIZE $WG MB > ${MAX_GB} GB"
  python3 - "$OUT" "$M" "$WG" "$MAX_GB" <<'PY'
import json, sys, time
out, model, wg, mg = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
json.dump({"model": model, "skipped": True,
           "reason": "权重 %d MB 超过镜像上限 %s GB（MAX_IMG_GB）" % (wg, mg),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(out + "/weights_image.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PY
  exit 0
fi

# ② 磁盘闸门（需要 3 份 + 6GB 余量）
NEED=$(( (WG * 3 / 1024) + 6 ))
FREE=$(df -BG /data | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${FREE:-0}" -lt "$NEED" ]; then
  echo "SKIP_DISK 需 ${NEED}G 可用 ${FREE}G"
  python3 - "$OUT" "$M" "$NEED" "$FREE" <<'PY'
import json, sys, time
out, model, need, free = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
json.dump({"model": model, "skipped": True,
           "reason": "磁盘不足：需 %s GB，可用 %s GB" % (need, free),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(out + "/weights_image.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PY
  exit 0
fi

echo "=== 造镜像 $TAG  (权重 ${WG} MB) ==="
tar -C "$MDIR" -cf "$TMP/_layer.tar" --transform='s|^\./|models/|' . 2>/dev/null || {
  echo "FAIL tar"; exit 1; }

docker rmi "$TAG" >/dev/null 2>&1
cat "$TMP/_layer.tar" | docker import \
  -c "ENV MODEL_DIR=/models" \
  -c "LABEL model=$M" \
  -c "LABEL model-corpus=true" \
  -c 'CMD ["/bin/false"]' \
  - "$TAG" >/dev/null || { echo "FAIL import"; rm -f "$TMP/_layer.tar"; exit 1; }
rm -f "$TMP/_layer.tar"

docker save "$TAG" | gzip -1 > "$TAR" || { echo "FAIL save"; exit 1; }
SZ=$(du -m "$TAR" | cut -f1)

echo "=== 上传 OBS ==="
UP=$(obsutil cp "$TAR" "${OBS}${SLUG}.tar.gz" 2>&1 | tail -2)
echo "$UP" | tail -1
if ! echo "$UP" | grep -qiE "successfully|status \[200\]"; then
  echo "FAIL upload"
  python3 - "$OUT" "$M" "$TAG" "$SZ" <<'PY'
import json, sys, time
out, model, tag, sz = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
json.dump({"model": model, "skipped": False, "ok": False,
           "reason": "上传 OBS 失败", "tag": tag, "size_mb": sz,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(out + "/weights_image.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PY
  rm -f "$TAR"; docker rmi "$TAG" >/dev/null 2>&1; exit 1
fi

python3 - "$OUT" "$M" "$TAG" "$SZ" "$OBS" "$SLUG" "$WG" <<'PY'
import json, sys, time
out, model, tag, sz, obs, slug, wg = sys.argv[1:8]
json.dump({"model": model, "skipped": False, "ok": True,
           "tag": tag, "obs": obs + slug + ".tar.gz",
           "size_mb": int(sz), "weights_mb": int(wg),
           "load_cmd": "docker load -i %s.tar.gz" % slug,
           "extract_cmd": "docker create --name w %s && docker cp w:/models . && docker rm w" % tag,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(out + "/weights_image.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PY

echo "OK ${SZ} MB"
rm -f "$TAR"
[ "$KEEP" = "1" ] || docker rmi "$TAG" >/dev/null 2>&1
