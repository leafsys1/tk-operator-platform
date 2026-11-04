#!/usr/bin/env bash
# 只验「渲染」不验「安装」：把 site.env 换成一份假值，把单元文件渲染出来检查
# 用法：bash deploy/selftest-render.sh
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

cat > "$T/site.env" <<'ENV'
TARGET_HOST=10.0.0.9
RUN_USER=tester
OPERATOR_ROOT=/opt/tk/op
MODEL_ROOT=/opt/tk/model
SHELL_ROOT=/opt/tk/shell
OPERATOR_PORT=28078
MODEL_PORT=28090
GATEWAY_PORT=28080
COMPUTE_SSH=root@10.0.0.8
COMPUTE_PORT=22222
COMPUTE_KEY=/opt/tk/key.pem
COMPUTE_MODEL_ROOT=/opt/tk/cm
COMPUTE_OPAGENT_ROOT=/opt/tk/co
LLM_BASE_URL=https://example.invalid/v1
LLM_MODEL=test-model
LLM_CONTEXT_WINDOW=1024
ALLOWED_CIDRS=127.0.0.0/8,10.0.0.0/24
ALLOWED_HOSTS=portal.example.invalid
ENV

fail=0
chk() { if eval "$2"; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=1; fi; }

# ---------- 网关单元渲染 ----------
sed -e "s#/data/lfx/op-agent-console-20261101/preview-gateway#/opt/tk/shell#g" \
    -e "s#^User=hzzg#User=tester#" \
    -e "s#^Group=hzzg#Group=tester#" \
    -e "s#WORKBENCH_PORT=18080#WORKBENCH_PORT=28080#" \
    -e "s#WORKBENCH_ALLOWED_CIDRS=.*#WORKBENCH_ALLOWED_CIDRS=127.0.0.0/8,10.0.0.0/24#" \
    -e "s#WORKBENCH_ALLOWED_HOSTS=.*#WORKBENCH_ALLOWED_HOSTS=portal.example.invalid#" \
    "$R/shell/deploy/tk-gateway-preview.service" > "$T/gw.service"
echo "网关单元渲染:"
chk "WorkingDirectory 指向 SHELL_ROOT" "grep -q '^WorkingDirectory=/opt/tk/shell$' '$T/gw.service'"
chk "User 已改成 RUN_USER" "grep -q '^User=tester$' '$T/gw.service'"
chk "端口已改" "grep -q 'WORKBENCH_PORT=28080' '$T/gw.service'"
chk "CIDR 已改" "grep -q 'WORKBENCH_ALLOWED_CIDRS=127.0.0.0/8,10.0.0.0/24' '$T/gw.service'"
chk "Host 已改" "grep -q 'WORKBENCH_ALLOWED_HOSTS=portal.example.invalid' '$T/gw.service'"
chk "ExecStart 指向 gateway-v2.py" "grep -q 'ExecStart=.*gateway-v2.py$' '$T/gw.service'"

# ---------- 算子单元渲染 ----------
sed -e "s#/data/lfx/op-agent-console-20261101#/opt/tk/op#g" \
    -e "s#--port 18078#--port 28078#g" \
    "$R/operator/deploy/tk-operator-preview.service" > "$T/op.service"
echo "算子单元渲染:"
chk "EnvironmentFile 指向 OPERATOR_ROOT" "grep -q 'EnvironmentFile=/opt/tk/op/preview-gateway/operator-preview.env' '$T/op.service'"
chk "uvicorn 端口已改" "grep -q -- '--port 28078' '$T/op.service'"
chk "WorkingDirectory 已改" "grep -q '^WorkingDirectory=/opt/tk/op/app/backend$' '$T/op.service'"

# ---------- 模型平台单元渲染 ----------
sed -e "s#/home/hzzg/workspace/model-corpus#/opt/tk/model#g" \
    -e "s#root@61.241.77.3#root@10.0.0.8#g" \
    -e "s#31236#22222#g" \
    -e "s#/home/hzzg/workspace/.ssh/KeyPair-fae2027.pem#/opt/tk/key.pem#g" \
    -e "s#/data/model_corpus#/opt/tk/cm#g" \
    -e "s#--port 8090#--port 28090#g" \
    "$R/model-platform/deploy/model-migration-platform.service" > "$T/mp.service"
sed -i "s#^User=hzzg#User=tester#" "$T/mp.service"
echo "模型平台单元渲染:"
chk "COMPUTE_SSH 已改" "grep -q 'COMPUTE_SSH=root@10.0.0.8' '$T/mp.service'"
chk "COMPUTE_PORT 已改" "grep -q 'COMPUTE_PORT=22222' '$T/mp.service'"
chk "COMPUTE_KEY 已改" "grep -q 'COMPUTE_KEY=/opt/tk/key.pem' '$T/mp.service'"
chk "COMPUTE_ROOT 已改" "grep -q 'COMPUTE_ROOT=/opt/tk/cm' '$T/mp.service'"
chk "端口已改" "grep -q -- '--port 28090' '$T/mp.service'"
chk "User 已改" "grep -q '^User=tester$' '$T/mp.service'"
chk "WorkingDirectory 已改" "grep -q '^WorkingDirectory=/opt/tk/model/_platform$' '$T/mp.service'"

# ---------- 上游端口改写（与 setup-shell.sh 同一段逻辑）----------
cp "$R/shell/gateway.py" "$T/gateway.py"
python3 - "$T/gateway.py" 28090 28078 <<'PY' || fail=1
import re, sys
path, mport, oport = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path, encoding="utf-8").read()
pat = re.compile(r"\{'model':\s*\(\s*'127\.0\.0\.1'\s*,\s*\d+\s*\)\s*,\s*"
                 r"'operator':\s*\(\s*'127\.0\.0\.1'\s*,\s*\d+\s*\)\s*\}")
out, n = pat.subn("{'model': ('127.0.0.1', %s), 'operator': ('127.0.0.1', %s)}" % (mport, oport), src)
if n != 1:
    sys.exit("找到 %d 处 upstream 声明（期望 1）" % n)
open(path, "w", encoding="utf-8").write(out)
PY
echo "网关上游端口改写:"
chk "model 端口已改" "grep -q \"'model': ('127.0.0.1', 28090)\" '$T/gateway.py'"
chk "operator 端口已改" "grep -q \"'operator': ('127.0.0.1', 28078)\" '$T/gateway.py'"
chk "改写后语法仍可通过" "python3 -c 'import ast,sys; ast.parse(open(\"$T/gateway.py\").read())'"

echo
[ "$fail" = "0" ] && echo "渲染自测全部通过（未安装、未重启任何服务）" || echo "有检查项失败"
exit "$fail"
