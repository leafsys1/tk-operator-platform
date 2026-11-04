#!/usr/bin/env bash
# =============================================================================
# gitcode 镜像取官方代码（内网访问不了 GitHub 时的替代方案）
#
# 背景：算力服务器访问不了 github.com，但 gitcode.com 是它的国内镜像。
#       实测可用：deepseek-ai/DeepSeek-VL、deepseek-ai/Janus、microsoft/TRELLIS、
#       facebookresearch/sapiens、OpenBMB/MiniCPM-V、VAST-AI-Research/TripoSR、
#       KwaiVGI/LivePortrait。
#
# 关键：gitcode 对不同仓库用**不同的镜像路径**（GitHub_Trending/xx、gh_mirrors/xx、
#       或 ai.gitcode.com），所以要**先跟随重定向拿到真实地址**，再 clone。
#
# 用法：./gitcode-clone.sh <org/repo> <目标目录>
# =============================================================================
set -uo pipefail
REPO="${1:?用法: gitcode-clone.sh <org/repo> <目标目录>}"
DEST="${2:?需要目标目录}"
BASE="https://gitcode.com/${REPO}"

# 1) 跟随重定向，拿到真实仓库页地址
REAL=$(curl -sIL -o /dev/null -w "%{url_effective}" "$BASE" 2>/dev/null || echo "")
echo "[gitcode] $BASE -> ${REAL:-（无重定向）}"

CANDIDATES=()
[ -n "$REAL" ] && CANDIDATES+=("${REAL}.git")
CANDIDATES+=("${BASE}.git")

rm -rf "$DEST"
for url in "${CANDIDATES[@]}"; do
  echo "[gitcode] 尝试 $url"
  if timeout 240 git clone --depth 1 "$url" "$DEST" 2>/dev/null; then
    echo "[gitcode] ✅ 成功 -> $DEST"
    exit 0
  fi
done
echo "[gitcode] ❌ 全部失败（该仓库可能没有镜像）"
exit 1
