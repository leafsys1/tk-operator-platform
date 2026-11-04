#!/usr/bin/env bash
# =============================================================================
# 回滚 —— 把某一个组件退回上一份或指定的备份文件
#
#   bash deploy/rollback.sh gateway          # 外层网关退回上一份
#   bash deploy/rollback.sh gateway <文件>    # 退回到指定备份
#   bash deploy/rollback.sh operator         # 算子后端退回上一次部署（git 上一个 tag）
#   bash deploy/rollback.sh model            # 模型平台退回上一份 app.py
#   bash deploy/rollback.sh list             # 列出可用备份
#
# 备份命名：部署脚本/手工修改前都会留 <文件名>.bak-<用途>-<时间戳>，
#           本脚本按时间倒序取最新一份。
# =============================================================================
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_site_env

GW="$SHELL_ROOT/gateway-v2.py"
MP="$MODEL_ROOT/_platform/app.py"

list_backups() {
  echo "== 网关备份（$SHELL_ROOT）=="
  ls -1t "$SHELL_ROOT"/gateway-v2*.py 2>/dev/null | grep -v 'gateway-v2\.py$' | head -10 || echo "  （无）"
  echo "== 模型平台备份（$MODEL_ROOT/_platform）=="
  ls -1t "$MODEL_ROOT"/_platform/app.py.bak-* 2>/dev/null | head -10 || echo "  （无）"
  echo "== 仓库 tag（算子后端按 tag 回退）=="
  ( cd "$REPO_DIR" && git tag -l 2>/dev/null | tail -10 ) || echo "  （无）"
}

restore_file() {
  local target="$1" src="$2" svc="$3"
  [ -f "$src" ] || die "备份不存在: $src"
  cp -a "$target" "$target.bak-before-rollback-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
  install -m 0700 "$src" "$target" || die "覆盖失败"
  ok "已用 $(basename "$src") 覆盖 $(basename "$target")"
  as_root systemctl restart "$svc" || die "重启 $svc 失败"
  sleep 2
  systemctl is-active --quiet "$svc" && ok "$svc active" || die "$svc 未起来 —— 立即看 journalctl -u $svc -n 80"
}

case "${1:-}" in
  list) list_backups ;;
  gateway)
    if [ -n "${2:-}" ]; then restore_file "$GW" "$2" tk-gateway-preview.service
    else
      SRC="$(ls -1t "$SHELL_ROOT"/gateway-v2*.py 2>/dev/null | grep -v 'gateway-v2\.py$' | head -1)"
      [ -n "$SRC" ] || die "没有网关备份，可用: bash deploy/rollback.sh list"
      restore_file "$GW" "$SRC" tk-gateway-preview.service
    fi
    curl -s --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/" | grep -q company-logo \
      && ok "回滚后壳页正常" || warn "回滚后壳页异常，继续回退更早一份"
    ;;
  model)
    if [ -n "${2:-}" ]; then restore_file "$MP" "$2" model-migration-platform.service
    else
      SRC="$(ls -1t "$MODEL_ROOT"/_platform/app.py.bak-* 2>/dev/null | head -1)"
      [ -n "$SRC" ] || die "没有模型平台备份"
      restore_file "$MP" "$SRC" model-migration-platform.service
    fi
    ;;
  operator)
    TAG="${2:-$(cd "$REPO_DIR" && git describe --tags --abbrev=0 2>/dev/null)}"
    [ -n "$TAG" ] || die "没有可用 tag；用 'bash deploy/rollback.sh list' 看，或给第二个参数指定"
    info "算子后端回退到 tag $TAG"
    ( cd "$REPO_DIR" && git checkout "$TAG" -- operator/backend operator/web operator/ops operator/scripts ) \
      || die "git checkout 失败"
    rsync -a --exclude='__pycache__' "$REPO_DIR/operator/backend/" "$OPERATOR_ROOT/app/backend/" || die "同步失败"
    rsync -a --exclude='node_modules' "$REPO_DIR/operator/web/" "$OPERATOR_ROOT/app/web/" || die "同步失败"
    as_root systemctl restart tk-operator-preview.service || die "重启失败"
    sleep 3
    systemctl is-active --quiet tk-operator-preview.service && ok "算子后端 active（$TAG）" || die "未起来"
    warn "注意：代码回退了，数据库 schema 不回退。若 tag 之间有过 migration，需手工 alembic downgrade"
    ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
