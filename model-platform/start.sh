#!/usr/bin/env bash
# =============================================================================
# 一键模型迁移平台 · 启动脚本
#
# 用法（在本目录内执行）：
#     ./start.sh                    前台启动（Ctrl-C 停止）
#     ./start.sh daemon             后台启动，写 server.pid
#     ./start.sh stop               停止后台实例
#     ./start.sh status             查看状态
#     ./start.sh tunnel [本地端口]  起 SSH 隧道把远端 8090 映射到本地（便于本机浏览器访问）
#
# 可覆盖环境变量：
#     CORPUS_ROOT   语料库根目录（缺省 /data/model_corpus）
#     PORT          监听端口（缺省 8090）
#     HOST          监听地址（缺省 0.0.0.0）
#     PY            指定 python 解释器
#
# 依赖：仅 Python 3 标准库（无需 pip install 任何东西）。
# =============================================================================
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"

CORPUS_ROOT="${CORPUS_ROOT:-/data/model_corpus}"
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
PY="${PY:-python3}"

cmd="${1:-foreground}"

case "$cmd" in
  tunnel)
    local_port="${2:-$PORT}"
    echo "[tunnel] 本地 $local_port -> 远端 127.0.0.1:$PORT（Ctrl-C 断开）"
    exec ssh -N -L "$local_port:127.0.0.1:$PORT" "${TUNNEL_HOST:-localhost}"
    ;;

  stop)
    # server.pid 可能是上一代进程留下的（实测：daemon 重启后 pid 文件还指向老进程，
    # stop 打一句"未在运行"就走了，端口其实还被占着，新实例 bind 直接 EADDRINUSE）。
    # 所以 pid 文件不可信时，按端口反查真正的监听者。
    killed=0
    if [ -f server.pid ]; then
      pid="$(cat server.pid)"
      if kill "$pid" 2>/dev/null; then
        sleep 1
        echo "[stop] 已停止 pid=$pid（来自 server.pid）"
        killed=1
      fi
    fi
    portpid="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
    if [ -n "${portpid:-}" ]; then
      if kill "$portpid" 2>/dev/null; then
        sleep 1
        echo "[stop] 已停止 pid=$portpid（按端口 :$PORT 反查）"
        killed=1
      fi
    fi
    rm -f server.pid
    if [ "$killed" = "0" ]; then
      echo "[stop] 端口 :$PORT 上没有在跑的实例"
    fi
    ;;

  status)
    code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || echo 000)"
    echo "[status] http://127.0.0.1:$PORT/ -> HTTP $code"
    if [ -f server.pid ]; then
      pid="$(cat server.pid)"
      ps -p "$pid" >/dev/null 2>&1 && echo "[status] 进程 pid=$pid 存活" || echo "[status] pid=$pid 已不在"
    fi
    ;;

  daemon)
    if [ -f server.pid ] && ps -p "$(cat server.pid)" >/dev/null 2>&1; then
      echo "[daemon] 已在运行 pid=$(cat server.pid)"
      exit 0
    fi
    setsid nohup "$PY" app.py --port "$PORT" --root "$CORPUS_ROOT" \
      > server.log 2>&1 < /dev/null &
    # setsid 会 detach，$! 拿不到真正的 python 进程；按端口反查真实 PID
    sleep 2
    realpid="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
    echo "${realpid:-$!}" > server.pid
    sleep 3
    code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || echo 000)"
    echo "[daemon] pid=$(cat server.pid)  HTTP=$code  日志: $PKG_DIR/server.log"
    ;;

  foreground|*)
    echo "[start] 语料库根目录: $CORPUS_ROOT"
    echo "[start] 监听: http://$HOST:$PORT/"
    exec "$PY" app.py --port "$PORT" --root "$CORPUS_ROOT"
    ;;
esac
