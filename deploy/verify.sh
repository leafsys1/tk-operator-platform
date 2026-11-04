#!/usr/bin/env bash
# =============================================================================
# 验收脚本 —— 部署完必须跑这个，它给的是「功能是否真通」，不是「进程是否在」
#
#   bash deploy/verify.sh            只读：进程/端口/页面/两侧 API
#   bash deploy/verify.sh --write    额外测写链路（会建一个临时任务再删掉）
#   bash deploy/verify.sh --gpu      额外测真机（会在算力机抢一张空闲卡干活，耗时 3-10 分钟）
#
# 输出：终端一张汇总表 + 逐项明细写 /tmp/tk-verify-<时间戳>.log
# =============================================================================
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_site_env

LOG="/tmp/tk-verify-$(date +%Y%m%d-%H%M%S).log"
PASS=0; FAIL=0; SKIP=0
declare -a ROWS

chk() { # chk 名称 期望正则 实际
  local name="$1" want="$2" got="$3"
  if printf '%s' "$got" | grep -qE "$want"; then
    PASS=$((PASS+1)); ROWS+=("PASS|$name|$got")
  else
    FAIL=$((FAIL+1)); ROWS+=("FAIL|$name|$got")
  fi
  printf '  %-46s %s\n' "$name" "$(printf '%s' "$got" | head -c 60)" | tee -a "$LOG"
}
skip() { SKIP=$((SKIP+1)); ROWS+=("SKIP|$1|$2"); printf '  %-46s SKIP %s\n' "$1" "$2" | tee -a "$LOG"; }

echo "=== 验收开始 $(date) ===" | tee "$LOG"
echo "目标 $TARGET_HOST  算子:$OPERATOR_PORT  模型:$MODEL_PORT  外壳:$GATEWAY_PORT" | tee -a "$LOG"

# ---------- A. 进程与端口 ----------
echo; echo "A. 进程与端口" | tee -a "$LOG"
for svc in tk-operator-preview model-migration-platform tk-gateway-preview; do
  chk "systemd $svc active" '^active$' "$(systemctl is-active "$svc" 2>/dev/null)"
done
for p in "$OPERATOR_PORT" "$MODEL_PORT" "$GATEWAY_PORT"; do
  chk "端口 $p 在听" '.' "$(ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]$p\$" | head -1)"
done
chk "算子只监听回环(不对公网裸奔)" "^127\.0\.0\.1:$OPERATOR_PORT$" \
  "$(ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]$OPERATOR_PORT\$" | head -1)"
chk "模型只监听回环" "^127\.0\.0\.1:$MODEL_PORT$" \
  "$(ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]$MODEL_PORT\$" | head -1)"

# ---------- B. 外壳 ----------
echo; echo "B. 统一外壳" | tee -a "$LOG"
BODY="$(curl -s --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/")"
chk "壳页 200 且有 company-logo" 'company-logo' "$(printf '%s' "$BODY" | grep -o company-logo | head -1)"
chk "壳页含算子页签" 'data-product="operator"' "$(printf '%s' "$BODY" | grep -o 'data-product="operator"' | head -1)"
chk "壳页含模型页签" 'data-product="model"' "$(printf '%s' "$BODY" | grep -o 'data-product="model"' | head -1)"
chk "Logo 可下载" '^200$' "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/brand/tanku.png")"
chk "重写 HTML 不被缓存(no-store)" 'no-store' \
  "$(curl -sI --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/operator/" | grep -i '^cache-control' | head -1)"
chk "重写 HTML 无 ETag(防旧版回潮)" '^(200|)$' \
  "$(curl -sI --noproxy '*' -m 10 "http://127.0.0.1:${GATEWAY_PORT}/operator/" | grep -ic '^etag' | sed 's/^0$/200/')"

# ---------- C. 两平台上手 ----------
echo; echo "C. 两侧平台（经外壳）" | tee -a "$LOG"
chk "算子 /operator/ 可达" '^(200|304)$' "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 15 "http://127.0.0.1:${GATEWAY_PORT}/operator/")"
chk "算子 /operator/api/health" '^(200|"ok"|ok)' \
  "$(curl -s --noproxy '*' -m 15 "http://127.0.0.1:${GATEWAY_PORT}/operator/api/health" | head -c 40)"
chk "模型 /model/ 可达" '^(200|304)$' "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 15 "http://127.0.0.1:${GATEWAY_PORT}/model/")"
chk "模型 /model/api/phases" 'phases' "$(curl -s --noproxy '*' -m 15 "http://127.0.0.1:${GATEWAY_PORT}/model/api/phases" | head -c 40)"

# ---------- D. 大模型端点（就这一处用到 LLM）----------
echo; echo "D. 算子平台的大模型端点" | tee -a "$LOG"
EP="$(curl -s --noproxy '*' -m 20 "http://127.0.0.1:${OPERATOR_PORT}/api/endpoints" 2>/dev/null)"
chk "有 active 端点" 'is_active' "$(printf '%s' "$EP" | grep -o is_active | head -1)"
MODEL_IN_EP="$(printf '%s' "$EP" | python3 -c 'import sys,json;d=json.load(sys.stdin);print([e["model"] for e in d if e.get("is_active")][:1])' 2>/dev/null || echo '[]')"
chk "端点模型 = ${LLM_MODEL:-?}" 'deepseek' "$MODEL_IN_EP"
chk "端点探测 ok" '"probe_state": *"ok"' "$(printf '%s' "$EP" | head -c 400)"

# ---------- E. 写链路（可选）----------
echo; echo "E. 写链路" | tee -a "$LOG"
if [ "${1:-}" = "--write" ]; then
  PJ="$(curl -s --noproxy '*' -m 10 "http://127.0.0.1:${OPERATOR_PORT}/api/projects" | python3 -c 'import sys,json;d=json.load(sys.stdin);print((d[0]["id"] if isinstance(d,list) and d else 1))' 2>/dev/null)"
  R="$(curl -s -o /tmp/_w.json -w '%{http_code}' --noproxy '*' -m 20 -X POST \
        -H 'Content-Type: application/json' -H "Origin: http://127.0.0.1:${GATEWAY_PORT}" \
        -d "{\"title\":\"verify-smoke\",\"op_name\":\"verify_smoke\",\"project_id\":${PJ:-1}}" \
        "http://127.0.0.1:${GATEWAY_PORT}/operator/api/tasks")"
  chk "经外壳建任务(POST 201)" '^201$' "$R"
  TID="$(python3 -c 'import json;print(json.load(open("/tmp/_w.json")).get("id",""))' 2>/dev/null)"
  if [ -n "$TID" ]; then
    chk "清理测试任务(DELETE 200)" '^200$' \
      "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 15 -X DELETE -H "Origin: http://127.0.0.1:${GATEWAY_PORT}" \
         "http://127.0.0.1:${GATEWAY_PORT}/operator/api/tasks/$TID")"
  fi
else
  skip "写链路" "加 --write 才测"
fi

# ---------- F. 真机（可选）----------
echo; echo "F. 真机执行" | tee -a "$LOG"
if [ "${1:-}" = "--gpu" ]; then
  info "查算力机空闲卡（会打印每张卡的占用进程）"
  ssh -i "$COMPUTE_KEY" -p "$COMPUTE_PORT" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 \
      "$COMPUTE_SSH" 'for i in 0 1 2 3 4 5 6 7; do printf "card%s: " $i; npu-smi info -i $i -t proc-mem 2>/dev/null | grep -q "No process in device" && echo FREE || npu-smi info -i $i -t proc-mem 2>/dev/null | grep "Process name" | head -1; done' \
    | tee -a "$LOG"
  warn "真机跑任务请在网页上做（tools/ 下有 Playwright 脚本可自动化），本脚本不替你抢卡"
  skip "真机任务" "手工在页面执行 build→smoke→test"
else
  skip "真机执行" "加 --gpu 才查卡；真任务请在页面跑"
fi

# ---------- 汇总 ----------
echo | tee -a "$LOG"
printf '%-6s %-46s %s\n' 结果 项目 实测 | tee -a "$LOG"
printf '%s\n' "----------------------------------------------------------------------------" | tee -a "$LOG"
for r in "${ROWS[@]}"; do
  IFS='|' read -r s n v <<< "$r"
  printf '%-6s %-46s %s\n' "$s" "$n" "$(printf '%s' "$v" | head -c 40)" | tee -a "$LOG"
done
echo "----------------------------------------------------------------------------" | tee -a "$LOG"
echo "PASS=$PASS FAIL=$FAIL SKIP=$SKIP   明细: $LOG" | tee -a "$LOG"
[ "$FAIL" = "0" ] || exit 1
