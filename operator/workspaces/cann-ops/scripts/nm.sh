#!/usr/bin/env bash
# 真机验证入口（工作区内可见）——转调控制台仓库根的 scripts/nm_test.sh。
#
# 为什么要有这个 wrapper：nm_test.sh + 仓库根 ops/nm.conf 在**控制台仓库**里，不在任务工作区；
# 而且 `ops/nm.conf` 只是本机兜底（默认 tk36），**不是任务配置**。这个 wrapper 先向控制台取
# 「当前任务」的 plan（执行机 / 私钥 / 镜像 / 设备 / 规格 / 判据 —— 任务覆盖 > 项目默认），
# 再把同一套值注入 nm_test.sh，保证 agent 在对话里发起的上机验证与控制台「概览 → 四动作」同源，
# 不会出现「页面选 A、实际跑 B」。
#
# 用法（在任务工作区根执行）：
#   scripts/nm.sh --dry-run --op elementwise_mul --action build   # 只算计划 + 同步，不动容器
#   scripts/nm.sh --op elementwise_mul --action build             # 上机编译
#   scripts/nm.sh --op elementwise_mul --action smoke             # 冒烟：编译+最小用例跑通（开发内自检）
#   scripts/nm.sh --op elementwise_mul --action test              # 上机正确性测试
#   scripts/nm.sh --op elementwise_mul --action benchmark         # 基准（独占卡，慢）
#   scripts/nm.sh --op elementwise_mul --action profile           # 性能采集（独占卡，慢）
#   scripts/nm.sh --task 11 --op elementwise_mul --action test    # 同任务有多个时显式指定
#   scripts/nm.sh --task 11 --loop 6 --op elementwise_mul --action test  # 优化 Loop（预算硬限制）
#
# 任务 id 的确定顺序：--task > 环境变量 OPX_TASK_ID > 按 --op 找最近的未定版任务。
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$WS_ROOT/../.." && pwd)"
API="${OPX_API:-http://127.0.0.1:8077}"

# agent 的 bash 工具在 workspace-write 沙箱里：仓库根 runs/ 不可写。本地产物落工作区 runs/。
export OPX_LOCAL_RUNS="${OPX_LOCAL_RUNS:-$WS_ROOT/runs}"

TASK="${OPX_TASK_ID:-}"
OP=""; ACTION=""
ORIG_ARGS=("$@")
i=0
while [ $i -lt ${#ORIG_ARGS[@]} ]; do
  case "${ORIG_ARGS[$i]:-}" in
    --task)   TASK="${ORIG_ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
    --op)     OP="${ORIG_ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
    --action) ACTION="${ORIG_ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
    *) i=$((i+1)) ;;
  esac
done

# 没显式给任务：按算子名找最近的未定版任务（同一 op 多个任务时请显式 --task）
if [ -z "$TASK" ] && [ -n "$OP" ]; then
  TASK=$(/usr/bin/python3 - "$API" "$OP" <<'PY'
import json, sys, urllib.request
api, op = sys.argv[1], sys.argv[2]
rows = json.load(urllib.request.urlopen(api + "/api/tasks", timeout=10))
cand = [t for t in rows if t.get("op_name") == op and not t.get("is_draft")]
cand.sort(key=lambda t: t.get("id") or 0)
print(cand[-1]["id"] if cand else "")
PY
  ) || TASK=""
fi
if [ -z "$TASK" ]; then
  echo "[nm.sh] 无法确定当前任务：请加 --task <id>（或确认该算子已有未定版任务）" >&2
  exit 2
fi

# 取「当前任务」的执行 plan（host/image/device/规格/判据；任务覆盖 > 项目默认）
PLAN_JSON=$(/usr/bin/python3 - "$API" "$TASK" "${ACTION:-build}" <<'PY'
import json, sys, urllib.request
api, task, action = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"action": action, "dry_run": True}).encode()
req = urllib.request.Request(api + f"/api/tasks/{task}/actions", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print(json.dumps(json.load(r), ensure_ascii=False))
except urllib.error.HTTPError as e:
    print("ERR " + e.read().decode("utf-8", "replace")[:300])
    sys.exit(1)
PY
) || { echo "[nm.sh] 取任务 plan 失败：$PLAN_JSON" >&2; exit 2; }
case "$PLAN_JSON" in ERR*) echo "[nm.sh] 取任务 plan 失败：${PLAN_JSON#ERR }" >&2; exit 2 ;; esac

# plan → 与控制台四动作同一套环境变量
eval "$(/usr/bin/python3 - "$PLAN_JSON" <<'PY'
import json, shlex, sys
from pathlib import PurePosixPath
p = json.loads(sys.argv[1])

def e(k, v):
    if v in (None, ""):
        return
    print(f"export {k}={shlex.quote(str(v))}")

e("OPX_TASK_ID", p.get("task_id"))
e("OPX_WORKSPACE", p.get("workspace"))
if p.get("remote_dir"):
    e("OPX_REMOTE_ROOT", str(PurePosixPath(p["remote_dir"]).parent))
e("OPX_SPEC_VERSION_ID", p.get("spec_version_id"))
e("OPX_CRITERIA_REF", p.get("criteria_ref"))
e("OPX_SSH", p.get("host_ref"))
e("OPX_SSH_PORT", p.get("host_ssh_port"))
e("OPX_SSH_KEY", p.get("host_ssh_key"))
e("OPX_IMAGE", p.get("image_ref"))
e("OPX_IMAGE_ID", p.get("image_id"))
e("OPX_HOST_ID", p.get("host_id"))
e("OPX_DEVICES", p.get("device_id"))
e("OPX_LOCK_MODE", p.get("lock_mode"))
e("OPX_TIMEOUT", p.get("timeout_s"))
print("export OPX_PLAN_READY=" + shlex.quote("1" if p.get("ready") else "0"))
print("export OPX_PLAN_MISSING=" + shlex.quote("；".join(p.get("missing") or [])))
PY
)"

if [ "${OPX_PLAN_READY:-0}" != "1" ]; then
  echo "[nm.sh] 任务 #${TASK} 前置不齐，未上机：${OPX_PLAN_MISSING:-未知}" >&2
  exit 3
fi

# 把控制台 plan 的权威身份写进工作区（随 rsync 进容器），供测试/验收节点做身份门禁。
# 测试侧仍优先读环境变量；文件里缺失的字段写 null，由测试记 INVALID，不猜测。
/usr/bin/python3 - "$PLAN_JSON" "$WS_ROOT/.opx-run-identity.json" <<'PY'
import json, sys
plan = json.loads(sys.argv[1])
out = sys.argv[2]
dev = plan.get("device_id")
container = 0 if dev not in (None, "") and "," not in str(dev) else None
payload = {
    "schema": "opx-run-identity/1",
    "source": "console_plan",
    "task_id": plan.get("task_id"),
    "image_id": plan.get("image_id"),
    "host_device_id": dev,
    "container_device_id": container,
    "spec_version_id": plan.get("spec_version_id"),
    "criteria_ref": plan.get("criteria_ref"),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"[nm.sh] run identity -> {out} "
      f"(image={payload['image_id']}, host={payload['host_device_id']}, "
      f"container={payload['container_device_id']})")
PY

echo "[nm.sh] 任务 #${TASK} · 动作 ${ACTION:-?} · 执行机 ${OPX_SSH:-?} · 镜像 ${OPX_IMAGE:-?} · 设备 ${OPX_DEVICES:-自动}"
exec "$ROOT/scripts/nm_test.sh" --task "$TASK" "${ORIG_ARGS[@]}"
