#!/usr/bin/env bash
# =============================================================================
# 步骤 1/3 —— 部署算子开发平台后端
#
# 在 172.16.111.102（TARGET_HOST）上以 RUN_USER 执行：
#     bash deploy/setup-operator.sh
#
# 做的事：
#   1. 校验 site.env、端口、Python
#   2. 建目录树（代码/数据/runs/archives/keys/secrets）
#   3. 建 venv 并装依赖（锁到 deploy/requirements.lock.txt 的精确版本）
#   4. 跑后端回归测试（必须 571 passed 才算装对）
#   5. 初始化数据库、写入 LLM 端点与密钥
#   6. 装 systemd 单元并启动，回读健康检查
#
# 幂等：重复执行只更新代码与单元，不动已有数据库/密钥/runs。
# =============================================================================
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_site_env
assert_on_target

VENV="$OPERATOR_ROOT/control-venv312"
PY="$VENV/bin/python"
APP="$OPERATOR_ROOT/app"

info "算子平台部署到 $OPERATOR_ROOT"

# ---------- 1. 环境检查 ----------
command -v python3 >/dev/null || die "缺 python3"
PYV="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYV" = "3.12" ] || warn "Python $PYV（生产实测 3.12；3.10/3.11 依赖版本需同步下调）"
command -v node >/dev/null || warn "缺 node（前端已预构建在 app/web/dist，不装 node 也能跑；要改前端才需要）"
port_free "$OPERATOR_PORT" || warn "端口 $OPERATOR_PORT 已被占用，确认不是旧实例"

# ---------- 2. 目录树 ----------
info "建目录树"
mkdir -p "$OPERATOR_ROOT"/{app,data,runs,archives,reports,case-library,keys,secrets/values,config} \
  || die "建目录失败（检查 $OPERATOR_ROOT 写权限）"
chmod 700 "$OPERATOR_ROOT/keys" "$OPERATOR_ROOT/secrets" "$OPERATOR_ROOT/secrets/values"
ok "$OPERATOR_ROOT/{app,data,runs,archives,reports,case-library,keys,secrets,config}"

# 代码（仓库里的 operator/ 就是 app/ 的内容）
if [ -d "$APP/backend/app" ]; then
  info "已存在 app/，只同步代码（保留 data/runs/密钥）"
  rsync -a --delete \
    --exclude='.runtime' --exclude='__pycache__' --exclude='*.pyc' \
    "$REPO_DIR/operator/backend/" "$APP/backend/"
  rsync -a --delete --exclude='node_modules' "$REPO_DIR/operator/web/" "$APP/web/"
else
  info "首次安装，拷贝代码"
  mkdir -p "$APP"
  rsync -a --exclude='__pycache__' "$REPO_DIR/operator/backend/" "$APP/backend/"
  rsync -a --exclude='node_modules' "$REPO_DIR/operator/web/" "$APP/web/"
fi
rsync -a --exclude='__pycache__' --exclude='*.bak*' "$REPO_DIR/operator/ops/" "$APP/ops/"
rsync -a --exclude='__pycache__' --exclude='*.bak*' "$REPO_DIR/operator/scripts/" "$APP/scripts/"
rsync -a "$REPO_DIR/operator/docs/" "$APP/docs/" 2>/dev/null || warn "docs 同步跳过"
rsync -a "$REPO_DIR/operator/config/" "$OPERATOR_ROOT/config/"
ok "代码就位（backend/web/ops/scripts/docs/config）"

# ---------- 3. venv 与依赖 ----------
if [ ! -x "$PY" ]; then
  info "建 venv（目标机 python3.12 若缺 ensurepip，用 --without-pip + pip --python 兜底）"
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    warn "标准 venv 失败，改用 --without-pip"
    python3 -m venv --without-pip "$VENV" || die "venv 创建失败"
    curl -sS --noproxy '*' -m 60 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
      || die "下载 get-pip.py 失败（离线环境请预置 wheels）"
    python3 /tmp/get-pip.py --python "$PY" --no-warn-script-location >/dev/null || die "pip 引导失败"
  fi
fi
if [ -f "$REPO_DIR/deploy/requirements.lock.txt" ]; then
  "$PY" -m pip install -q --disable-pip-version-check -r "$REPO_DIR/deploy/requirements.lock.txt" \
    || die "依赖安装失败（看完整输出）"
else
  "$PY" -m pip install -q "$APP/backend" || die "依赖安装失败"
fi
"$PY" -m pip check || die "pip check 有冲突，不能继续"
ok "依赖装好，无冲突"

# ---------- 4. 后端回归测试 ----------
info "跑后端回归测试（这一步不过就不许启动）"
if [ -d "$APP/backend/tests" ]; then
  ( cd "$APP/backend" && "$PY" -m pytest -q 2>&1 | tail -5 )
  RC=${PIPESTATUS[0]}
  [ "$RC" = "0" ] || die "后端测试失败 rc=$RC —— 不要启动服务，先看上面失败用例"
  ok "后端测试通过"
else
  warn "没有 tests/，跳过（不可据此认为功能正常）"
fi

# ---------- 5. 数据库 ----------
DB="$OPERATOR_ROOT/data/opx.db"
[ -f "$DB" ] || { info "首次初始化数据库"; ( cd "$APP/backend" && "$PY" -m alembic upgrade head ) || die "alembic 失败"; }
"$PY" - "$DB" <<'PY' || die "数据库自检失败"
import sqlite3, sys
p = sys.argv[1]
c = sqlite3.connect(p)
print("  integrity_check:", c.execute("PRAGMA integrity_check").fetchone()[0])
print("  tables:", len(c.execute("select name from sqlite_master where type='table'").fetchall()))
PY
ok "数据库就绪 $DB"

# ---------- 6. 大模型端点 ----------
KEYFILE="$OPERATOR_ROOT/secrets/values/server-active-endpoint.key"
if [ ! -s "$KEYFILE" ]; then
  info "写入算子平台 agent 用的大模型 API Key"
  echo "  端点: ${LLM_BASE_URL:-<未配置>}   模型: ${LLM_MODEL:-<未配置>}"
  printf '  粘贴 API Key（回车跳过，之后可在网页「API 管理」填）: '
  read -r -s _k || true; echo
  if [ -n "${_k:-}" ]; then
    umask 077; printf '%s' "$_k" > "$KEYFILE"
    ok "Key 已写入 $KEYFILE（0600）"
  else
    warn "跳过写 Key —— agent 调用会 401，直到你在网页配置端点"
  fi
else
  ok "Key 已存在（长度 $(wc -c < "$KEYFILE")），不动"
fi

# ---------- 7. 算力机连通性 ----------
if [ -f "$COMPUTE_KEY" ]; then
  install -m 0600 "$COMPUTE_KEY" "$OPERATOR_ROOT/keys/executor.pem" || die "拷私钥失败"
  ssh -i "$OPERATOR_ROOT/keys/executor.pem" -p "$COMPUTE_PORT" -o IdentitiesOnly=yes \
      -o BatchMode=yes -o ConnectTimeout=10 "$COMPUTE_SSH" 'hostname' >/dev/null 2>&1 \
    && ok "算力机 $COMPUTE_SSH:$COMPUTE_PORT 可达" \
    || warn "算力机不可达 —— 平台能起，但真机 build/test 会失败"
else
  warn "COMPUTE_KEY=$COMPUTE_KEY 不存在，跳过算力机检查"
fi

# ---------- 8. systemd ----------
info "安装并启动 systemd 单元"
sed -e "s#/data/lfx/op-agent-console-20261101#${OPERATOR_ROOT}#g" \
    -e "s#--port 18078#--port ${OPERATOR_PORT}#g" \
    "$REPO_DIR/operator/deploy/tk-operator-preview.service" > /tmp/tk-operator.service \
  || die "渲染单元失败"
install_unit /tmp/tk-operator.service tk-operator-preview.service 1
restart_and_check tk-operator-preview.service "http://127.0.0.1:${OPERATOR_PORT}/api/health"

cat <<EOF

算子平台就绪：
  代码     $APP
  数据库   $DB
  监听     127.0.0.1:${OPERATOR_PORT}（仅回环，对外必须经 shell）
  日志     journalctl -u tk-operator-preview -f
  下一步   部署模型平台 → bash deploy/setup-model-platform.sh
EOF
