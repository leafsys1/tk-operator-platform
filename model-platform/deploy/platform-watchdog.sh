#!/usr/bin/env bash
# 平台看门狗：进程挂了自动拉起（systemd 装好后可停掉这个）
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while true; do
  if ! curl -s -m 5 --noproxy '*' -o /dev/null "http://127.0.0.1:8090/" 2>/dev/null; then
    echo "$(date '+%F %T') 平台无响应，重启" >> "$D/_platform/watchdog.log"
    "$D/start-platform.sh" restart >> "$D/_platform/watchdog.log" 2>&1
  fi
  sleep 30
done
