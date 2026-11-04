#!/usr/bin/env bash
# 一键起开发环境：后端 8077（uvicorn，托管 web/dist）+ 前端 5173（Vite，/api 反代到 8077）。
# Ctrl-C 一起退（后端会跟着停）。
#
# 为什么要这个脚本：之前只有 README 里的几行手敲命令，起服务/停服务靠记忆，
# 「控制台自身怎么跑」没有一键入口（2026-10-15 复查 ②4）。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BACKEND="$ROOT_DIR/backend"
PY="$BACKEND/.venv/bin/python"

PORT_BACKEND=${OPX_PORT:-8077}

# 载入可选的本机环境文件（600 权限）：把 OPX_LLM_KEY 等凭据放在**进程环境**里，
# 不进库、不进仓库。不存在就跳过（页面仍能开，只有「发起 agent run」需要 key）。
# 路径可用 OPX_ENV_FILE 覆盖；轮换 key 只改那个文件。
OPX_ENV_FILE=${OPX_ENV_FILE:-$HOME/.config/op-agent/env}
if [ -f "$OPX_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$OPX_ENV_FILE"
fi

if [ ! -x "$PY" ]; then
  echo "后端虚拟环境不存在：$BACKEND/.venv" >&2
  echo "先建：cd backend && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ." >&2
  exit 2
fi

if [ -z "${OPX_LLM_KEY:-}" ]; then
  echo "提示：未设置 OPX_LLM_KEY —— 页面能开，但「发起 agent run」会报缺凭据（只从环境变量取，不进库）" >&2
fi

cleanup() {
  if [ -n "${BACK_PID:-}" ] && kill -0 "$BACK_PID" 2>/dev/null; then
    echo "-- 停止后端（pid ${BACK_PID}）"
    kill "$BACK_PID" 2>/dev/null || true
    wait "$BACK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "-- 后端：http://127.0.0.1:$PORT_BACKEND （文档 /docs）"
(cd "$BACKEND" && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT_BACKEND" --reload) &
BACK_PID=$!

# 就绪探测（别用 sleep 猜）
for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT_BACKEND/api/health" >/dev/null 2>&1; then
    echo "-- 后端就绪"
    break
  fi
  kill -0 "$BACK_PID" 2>/dev/null || { echo "后端启动失败" >&2; exit 1; }
  sleep 0.5
done

echo "-- 前端：http://127.0.0.1:5173 （Ctrl-C 退出）"
cd "$ROOT_DIR/web"
if [ ! -d node_modules ]; then
  echo "   首次运行：npm install"
  npm install
fi
npm run dev
