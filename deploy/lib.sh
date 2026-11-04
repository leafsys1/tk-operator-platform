#!/usr/bin/env bash
# =============================================================================
# 公共库 —— 被 setup-*.sh / verify.sh 引用，不单独执行
# 命名：deploy/lib.sh
# =============================================================================
set -uo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LIB_DIR/.." && pwd)"
SITE_ENV="$LIB_DIR/site.env"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ! $*" >&2; }

load_site_env() {
  [ -f "$SITE_ENV" ] || die "缺少 $SITE_ENV —— 先执行: cp deploy/site.env.example deploy/site.env && vi deploy/site.env"
  # shellcheck disable=SC1090
  . "$SITE_ENV"
  : "${TARGET_HOST:?}" "${RUN_USER:?}" "${OPERATOR_ROOT:?}" "${MODEL_ROOT:?}" "${SHELL_ROOT:?}"
  : "${OPERATOR_PORT:?}" "${MODEL_PORT:?}" "${GATEWAY_PORT:?}"
  : "${COMPUTE_SSH:?}" "${COMPUTE_PORT:?}" "${COMPUTE_KEY:?}"
  : "${COMPUTE_MODEL_ROOT:?}" "${COMPUTE_OPAGENT_ROOT:?}"
  : "${ALLOWED_CIDRS:?}" "${ALLOWED_HOSTS:?}"
  # 一致性检查：三个端口不能撞
  if [ "$OPERATOR_PORT" = "$MODEL_PORT" ] || [ "$GATEWAY_PORT" = "$MODEL_PORT" ] || [ "$GATEWAY_PORT" = "$OPERATOR_PORT" ]; then
    die "OPERATOR_PORT/MODEL_PORT/GATEWAY_PORT 有重复"
  fi
}

# 确认当前就在目标机上（脚本设计为「在 TARGET_HOST 上以 RUN_USER 执行」）
assert_on_target() {
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ "$ip" != "$TARGET_HOST" ]; then
    warn "当前机器主 IP = $ip，site.env 的 TARGET_HOST = $TARGET_HOST"
    warn "本脚本必须在那台机器上执行（不要从跳板机直接跑远程安装）"
    [ "${FORCE_ON_HOST:-0}" = "1" ] || die "如在等价测试机上演练，请设 FORCE_ON_HOST=1"
  fi
  if [ "$(id -un)" != "$RUN_USER" ]; then
    die "当前用户 $(id -un) ≠ RUN_USER=$RUN_USER"
  fi
}

# sudo 包装：需要密码时从 SUDO_PASS 读
as_root() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [ -n "${SUDO_PASS:-}" ]; then
    printf '%s\n' "$SUDO_PASS" | sudo -S "$@"
  else
    sudo "$@"   # 交互输入
  fi
}

# 安装/更新 systemd 单元：$1=源文件 $2=单元名 $3=是否 enable
install_unit() {
  local src="$1" name="$2" enable="${3:-1}"
  [ -f "$src" ] || die "找不到单元文件 $src"
  as_root install -m 0644 -o root -g root "$src" "/etc/systemd/system/$name" \
    || die "写 /etc/systemd/system/$name 失败"
  as_root systemctl daemon-reload || die "daemon-reload 失败"
  if [ "$enable" = "1" ]; then
    as_root systemctl enable "$name" >/dev/null 2>&1 || warn "enable $name 失败（不致命）"
  fi
  ok "已安装 $name"
}

restart_and_check() {
  local name="$1" url="${2:-}"
  as_root systemctl restart "$name" || die "重启 $name 失败"
  sleep 2
  systemctl is-active --quiet "$name" || { as_root systemctl status "$name" --no-pager -l | tail -20; die "$name 未 active"; }
  ok "$name active"
  if [ -n "$url" ]; then
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 10 "$url" || echo 000)"
    case "$code" in
      2*|3*) ok "健康检查 $url → $code" ;;
      *) warn "健康检查 $url → $code（看 journalctl -u $name -n 50）" ;;
    esac
  fi
}

port_free() {
  local p="$1"
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$p\$" && return 1 || return 0
}
