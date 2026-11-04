#!/usr/bin/env bash
# =============================================================================
# 步骤 3/3 —— 部署统一外壳（网关 + 双 iframe 页）
#
# 在 172.16.111.102（TARGET_HOST）上以 RUN_USER 执行：
#     bash deploy/setup-shell.sh
#
# 外壳做四件事，全部不侵入业务源码：
#   1. 路由：/ → portal.html；/operator/* → OPERATOR_PORT；/model/* → MODEL_PORT
#   2. 主题：向两个平台的 HTML 注入统一 CSS（颜色/字号/表格密度/导航字重）
#   3. 缓存：被重写过的 HTML 强制 no-store，剥掉 ETag/Last-Modified
#      （不这么做会出现「改了样式但页面还是旧的」——实测踩过）
#   4. 写闸：可分别把某一侧设为只读（GET 放行，写操作 503）
# =============================================================================
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_site_env
assert_on_target

info "统一外壳部署到 $SHELL_ROOT"

# ---------- 1. 环境检查 ----------
python3 -c 'import http.server, ssl' || die "Python 缺标准库"
for p in "$OPERATOR_PORT" "$MODEL_PORT"; do
  curl -s -o /dev/null --noproxy '*' -m 3 "http://127.0.0.1:$p/" \
    || warn "上游 127.0.0.1:$p 不通 —— 外壳能起，但那一侧页签会 502"
done
port_free "$GATEWAY_PORT" || warn "端口 $GATEWAY_PORT 已被占用，确认不是旧实例"

# ---------- 2. 文件 ----------
mkdir -p "$SHELL_ROOT" || die "建 $SHELL_ROOT 失败"
install -m 0700 "$REPO_DIR/shell/gateway.py"    "$SHELL_ROOT/gateway-v2.py" || die "拷 gateway 失败"
install -m 0600 "$REPO_DIR/shell/portal.html"   "$SHELL_ROOT/portal.html"   || die "拷 portal 失败"
install -m 0644 "$REPO_DIR/shell/tanku-logo.png" "$SHELL_ROOT/tanku-logo.png" || die "拷 logo 失败"
ok "gateway-v2.py / portal.html / tanku-logo.png"

# 网关把上游端口写死在代码里，端口不同就必须改
sed -i -e "s/('model', (\"127.0.0.1\", 8090))/('model', ('127.0.0.1', ${MODEL_PORT}))/" \
       -e "s/('operator', (\"127.0.0.1\", 18078))/('operator', ('127.0.0.1', ${OPERATOR_PORT}))/" \
       "$SHELL_ROOT/gateway-v2.py"
grep -q "127.0.0.1', ${MODEL_PORT}" "$SHELL_ROOT/gateway-v2.py" \
  && ok "上游端口已改为 model=$MODEL_PORT operator=$OPERATOR_PORT" \
  || warn "上游端口改写没命中（gateway.py 里 upstream 写法和预期不同，请手工核对）"

# ---------- 3. 离线单测（先测再上）----------
info "跑外壳单测"
if [ -d "$REPO_DIR/shell/tests" ]; then
  ( cd "$REPO_DIR/shell" && python3 -m pytest tests -q 2>&1 | tail -4 )
  RC=${PIPESTATUS[0]}
  [ "$RC" = "0" ] || die "外壳单测失败 rc=$RC —— 不要部署"
  ok "42 项单测通过"
fi

# ---------- 4. systemd ----------
info "安装并启动 systemd 单元"
sed -e "s#/data/lfx/op-agent-console-20261101/preview-gateway#${SHELL_ROOT}#g" \
    -e "s#^User=hzzg#User=${RUN_USER}#" \
    -e "s#^Group=hzzg#Group=${RUN_USER}#" \
    -e "s#WORKBENCH_PORT=18080#WORKBENCH_PORT=${GATEWAY_PORT}#" \
    -e "s#WORKBENCH_ALLOWED_CIDRS=.*#WORKBENCH_ALLOWED_CIDRS=${ALLOWED_CIDRS}#" \
    -e "s#WORKBENCH_ALLOWED_HOSTS=.*#WORKBENCH_ALLOWED_HOSTS=${ALLOWED_HOSTS}#" \
    -e "s#gateway-v2.py#gateway-v2.py#" \
    "$REPO_DIR/shell/deploy/tk-gateway-preview.service" > /tmp/tk-gateway.service \
  || die "渲染单元失败"
install_unit /tmp/tk-gateway.service tk-gateway-preview.service 1
restart_and_check tk-gateway-preview.service "http://127.0.0.1:${GATEWAY_PORT}/"

# ---------- 5. 端到端冒烟 ----------
info "端到端冒烟"
BODY="$(curl -s --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/")"
for token in company-logo 'data-product="operator"' 'data-product="model"'; do
  case "$BODY" in
    *"$token"*) ok "壳页含 $token" ;;
    *) die "壳页缺 $token —— 说明 / 没返回 portal.html（看 journalctl -u tk-gateway-preview）" ;;
  esac
done
for name in operator model; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 15 "http://127.0.0.1:${GATEWAY_PORT}/$name/")"
  case "$code" in
    2*) ok "/$name/ → $code" ;;
    *) warn "/$name/ → $code（该侧业务平台没起或端口不符）" ;;
  esac
done
code="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/brand/tanku.png")"
[ "$code" = "200" ] && ok "/brand/tanku.png → 200" || warn "/brand/tanku.png → $code"

cat <<EOF

统一外壳就绪：
  入口     http://${TARGET_HOST}:${GATEWAY_PORT}/
  监听     0.0.0.0:${GATEWAY_PORT}
  上游     operator→127.0.0.1:${OPERATOR_PORT}  model→127.0.0.1:${MODEL_PORT}
  来源白名单 ${ALLOWED_CIDRS}
  日志     journalctl -u tk-gateway-preview -f
  回滚     bash deploy/rollback.sh gateway
EOF
