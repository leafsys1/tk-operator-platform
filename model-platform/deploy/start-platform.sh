#!/usr/bin/env bash
# =============================================================================
# 一键模型迁移平台 · 虚拟机侧启动脚本
#
# 架构：代码与数据在虚拟机（本机），模型权重与 NPU 在算力服务器 61.241.77.3。
#       平台在本机跑，迁移任务通过 SSH 派发到算力服务器执行并回收结果。
#
# 用法：
#     ./start-platform.sh            前台启动
#     ./start-platform.sh daemon     后台启动（写 platform.pid）
#     ./start-platform.sh stop       停止
#     ./start-platform.sh status     查看状态
#     ./start-platform.sh restart    重启
#
# 环境变量（一般不用改）：
#     CORPUS_ROOT   本机代码与数据根目录（缺省=脚本所在目录）
#     PORT          监听端口（缺省 8090）
#     COMPUTE_SSH   算力服务器（缺省 root@61.241.77.3）
#     COMPUTE_PORT  算力服务器 SSH 端口（缺省 31236）
# =============================================================================
set -uo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CORPUS_ROOT="${CORPUS_ROOT:-$PKG_DIR}"
PORT="8090" 
PY="${PY:-python3}"
PIDF="$PKG_DIR/platform.pid"
LOG="$PKG_DIR/_platform/server.log"

cmd="${1:-foreground}"

_realpid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

case "$cmd" in
  stop)
    pid="$(_realpid)"
    [ -z "$pid" ] && pid="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null; then
      sleep 2; echo "[stop] 已停止 pid=$pid"
    else
      echo "[stop] 未在运行"
    fi
    rm -f "$PIDF"
    ;;
  status)
    code="$(curl -s -m 5 --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null || echo 000)"
    echo "[status] http://127.0.0.1:$PORT/ -> HTTP $code"
    echo "[status] 本机访问:   http://$(hostname -I | awk '{print $1}'):$PORT/"
    [ -f "$PIDF" ] && echo "[status] pid=$(cat "$PIDF")"
    ;;
  daemon|restart)
    [ "$cmd" = "restart" ] && "$0" stop >/dev/null 2>&1
    if [ -n "$(_realpid)" ]; then echo "[daemon] 已在运行 pid=$(_realpid)"; exit 0; fi
    cd "$PKG_DIR/_platform"
    setsid nohup "$PY" app.py --port "$PORT" --root "$CORPUS_ROOT" \
      >> "$LOG" 2>&1 < /dev/null &
    sleep 4
    p="$(_realpid)"; echo "${p:-$!}" > "$PIDF"
    code="$(curl -s -m 5 --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null || echo 000)"
    echo "[daemon] pid=${p:-?}  HTTP=$code"
    echo "[daemon] 访问: http://$(hostname -I | awk '{print $1}'):$PORT/   日志: $LOG"
    ;;
  *)
    cd "$PKG_DIR/_platform"
    echo "[start] 代码数据: $CORPUS_ROOT"
    echo "[start] 算力服务器: ${COMPUTE_SSH:-root@61.241.77.3}:${COMPUTE_PORT:-31236}"
    echo "[start] 监听: http://0.0.0.0:$PORT/"
    exec "$PY" app.py --port "$PORT" --root "$CORPUS_ROOT"
    ;;
esac
