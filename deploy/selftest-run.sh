#!/usr/bin/env bash
# 运行时自测：把端口改写成非默认值后，网关还能不能正确路由到两个上游
# 用法：bash deploy/selftest-run.sh
# 不碰生产：全程用 28078/28090/28080 和 /tmp 下的临时目录
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/.." && pwd)"
T="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; rm -rf "$T"; }
trap cleanup EXIT

# 两个假上游
cat > "$T/up.py" <<'PY'
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
name = sys.argv[2]
class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def do_GET(self):
        body = ('UPSTREAM=%s PATH=%s' % (name, self.path)).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
PY
python3 "$T/up.py" 28090 model &  PIDS+=($!)
python3 "$T/up.py" 28078 operator & PIDS+=($!)
sleep 1

# 改端口后的网关
cp "$R/shell/gateway.py" "$T/gateway.py"
python3 - "$T/gateway.py" 28090 28078 <<'PY'
import re, sys
path, mport, oport = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path, encoding="utf-8").read()
pat = re.compile(r"\{'model':\s*\(\s*'127\.0\.0\.1'\s*,\s*\d+\s*\)\s*,\s*"
                 r"'operator':\s*\(\s*'127\.0\.0\.1'\s*,\s*\d+\s*\)\s*\}")
out, n = pat.subn("{'model': ('127.0.0.1', %s), 'operator': ('127.0.0.1', %s)}" % (mport, oport), src)
assert n == 1, n
open(path, "w", encoding="utf-8").write(out)
PY
cp "$R/shell/portal.html" "$T/portal.html"
umask 077; printf 'u:p\n' > "$T/access.txt"; chmod 600 "$T/access.txt"

WORKBENCH_BIND=127.0.0.1 WORKBENCH_PORT=28080 \
WORKBENCH_AUTH_REQUIRED=0 WORKBENCH_ALLOWED_CIDRS=127.0.0.0/8 \
WORKBENCH_ALLOWED_HOSTS='*' WORKBENCH_CREDENTIAL_FILE="$T/access.txt" \
  python3 "$T/gateway.py" & PIDS+=($!)
sleep 1.5

fail=0
chk() { if [ "$2" = "$3" ]; then echo "  ✓ $1"; else echo "  ✗ $1（期望 $3，实得 $2）"; fail=1; fi; }

chk "/model/api/x 路由到 model 上游" \
  "$(curl -s --noproxy '*' -m 5 http://127.0.0.1:28080/model/api/x)" "UPSTREAM=model PATH=/api/x"
chk "/operator/api/y 路由到 operator 上游" \
  "$(curl -s --noproxy '*' -m 5 http://127.0.0.1:28080/operator/api/y)" "UPSTREAM=operator PATH=/api/y"
chk "/ 返回壳页" \
  "$(curl -s --noproxy '*' -m 5 http://127.0.0.1:28080/ | grep -q 'company-logo' && echo yes)" "yes"
chk "未知路径 404" \
  "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 5 http://127.0.0.1:28080/nope)" "404"
chk "/model/ 剥前缀后到上游 /" \
  "$(curl -s --noproxy '*' -m 5 http://127.0.0.1:28080/model/)" "UPSTREAM=model PATH=/"

echo
[ "$fail" = "0" ] && echo "运行时自测通过（全程用 28078/28090/28080 与 /tmp，未触碰生产）" || echo "有检查项失败"
exit "$fail"
