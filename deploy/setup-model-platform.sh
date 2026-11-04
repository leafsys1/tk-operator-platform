#!/usr/bin/env bash
# =============================================================================
# 步骤 2/3 —— 部署模型迁移工作台
#
# 在 172.16.111.102（TARGET_HOST）上以 RUN_USER 执行：
#     bash deploy/setup-model-platform.sh
#
# 注意这个平台的特殊之处（别按「agent 平台」的思路排查）：
#   * 它是纯 Python stdlib + http.server，**没有任何 LLM 调用**
#   * "智能"来自三处：配方库 recipes / 坑库 pitfalls 关键词打分 / 算力机真推理
#   * 平台只做派发与回收：SSH 到算力机跑 probe.sh + mk_image.sh，rsync 回结果
#   * 权重不在本机，在算力机 $COMPUTE_MODEL_ROOT
# =============================================================================
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_site_env
assert_on_target

PLAT="$MODEL_ROOT/_platform"
info "模型迁移平台部署到 $MODEL_ROOT"

# ---------- 1. 环境检查 ----------
command -v python3 >/dev/null || die "缺 python3"
python3 -c 'import sqlite3, http.server' || die "Python 缺 sqlite3/http.server"
port_free "$MODEL_PORT" || warn "端口 $MODEL_PORT 已被占用，确认不是旧实例"

# ---------- 2. 目录树 ----------
info "建目录树"
mkdir -p "$MODEL_ROOT"/{_platform,_manifest,_probe,_delivery,_tools} || die "建目录失败"
mkdir -p "$MODEL_ROOT"/{nlp,multi-modal,audio,cv,scientific-computing}
ok "$MODEL_ROOT/{_platform,_manifest,_probe,_delivery,_tools} + 5 个分类目录"

# ---------- 3. 代码 ----------
info "同步平台代码"
rsync -a "$REPO_DIR/model-platform/app.py" "$PLAT/app.py" || die "拷 app.py 失败"
rsync -a --delete "$REPO_DIR/model-platform/static/" "$PLAT/static/" || die "拷 static 失败"
cp -f "$REPO_DIR/model-platform/start.sh" "$MODEL_ROOT/start-platform.sh"
chmod +x "$MODEL_ROOT/start-platform.sh"
ok "app.py（$(wc -l < "$PLAT/app.py") 行）+ static/ + start-platform.sh"

# ---------- 4. 算力机侧探针脚本 ----------
# 探针脚本必须放在算力机的 $COMPUTE_MODEL_ROOT/_probe 下，且**路径与平台里写死的一致**
if [ -d "$REPO_DIR/compute-side/_probe" ] && [ -f "$COMPUTE_KEY" ]; then
  info "把探针脚本推到算力机 $COMPUTE_SSH:$COMPUTE_MODEL_ROOT/_probe"
  ssh -i "$COMPUTE_KEY" -p "$COMPUTE_PORT" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 \
      "$COMPUTE_SSH" "mkdir -p '$COMPUTE_MODEL_ROOT/_probe'" || die "算力机不可达或建目录失败"
  rsync -a -e "ssh -i $COMPUTE_KEY -p $COMPUTE_PORT -o IdentitiesOnly=yes -o BatchMode=yes" \
      "$REPO_DIR/compute-side/_probe/" "$COMPUTE_SSH:$COMPUTE_MODEL_ROOT/_probe/" || die "探针脚本推送失败"
  ok "探针脚本已推送（probe.sh / mk_image.sh / npu_shim.py / perf_test.py）"
else
  warn "跳过探针推送（缺 compute-side/_probe 或 COMPUTE_KEY）—— 迁移任务会在远端失败"
fi

# ---------- 5. 自检 ----------
info "平台自检"
python3 - "$PLAT/app.py" <<'PY' || die "自检失败"
import ast, sys
src = open(sys.argv[1], encoding="utf-8").read()
ast.parse(src)
bad = [m for m in ("import requests", "import httpx", "urllib.request", "openai") if m in src]
print("  语法 OK，行数", src.count("\n"))
print("  出站 HTTP 客户端:", bad or "无（这是设计如此：平台不调用任何 LLM）")
PY
ok "自检通过"

# ---------- 6. systemd ----------
info "安装并启动 systemd 单元"
sed -e "s#/home/hzzg/workspace/model-corpus#${MODEL_ROOT}#g" \
    -e "s#root@61.241.77.3#${COMPUTE_SSH}#g" \
    -e "s#31236#${COMPUTE_PORT}#g" \
    -e "s#/home/hzzg/workspace/.ssh/KeyPair-fae2027.pem#${COMPUTE_KEY}#g" \
    -e "s#/data/model_corpus#${COMPUTE_MODEL_ROOT}#g" \
    -e "s#--port 8090#--port ${MODEL_PORT}#g" \
    "$REPO_DIR/model-platform/deploy/model-migration-platform.service" > /tmp/tk-model-platform.service \
  || die "渲染单元失败"
# 单元里 User= 写死 hzzg，按 RUN_USER 改写
sed -i "s#^User=hzzg#User=${RUN_USER}#" /tmp/tk-model-platform.service
install_unit /tmp/tk-model-platform.service model-migration-platform.service 1
restart_and_check model-migration-platform.service "http://127.0.0.1:${MODEL_PORT}/api/phases"

cat <<EOF

模型迁移平台就绪：
  代码     $PLAT/app.py
  监听     127.0.0.1:${MODEL_PORT}（仅回环，对外必须经 shell）
  日志     journalctl -u model-migration-platform -f（另有一份 $PLAT/server.log）
  语料索引 $MODEL_ROOT/_manifest
  权重     在算力机 $COMPUTE_SSH:$COMPUTE_MODEL_ROOT（本机没有）
  下一步   部署统一外壳 → bash deploy/setup-shell.sh
EOF
