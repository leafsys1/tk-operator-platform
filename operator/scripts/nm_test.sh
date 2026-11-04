#!/usr/bin/env bash
# 上机测试（nm_test）：把工作区同步到裸金属 → 在「算子开发镜像」容器里跑一次动作 → 回读产物 → 回填台账。
#
# 这是「本地改代码 → 改到满意 → 上机测一次」里的那「一次」，不是一个命令级代理：
# agent 平时在本机随便改（不下发任何东西），只有需要真机验证时才调这个脚本。
# 它同时是**四动作契约（方案 v1.1 §6）的执行侧后端**：配方、设备互斥、超时与兜底清理都在这里固化。
#
# 用法：
#   scripts/nm_test.sh --dry-run                      # 只看会同步/跑什么，不动服务器
#   scripts/nm_test.sh --sync-only                    # 只同步到服务器（不跑命令）
#   scripts/nm_test.sh --action test                  # 按契约跑工作区根的 test.sh（四动作之一）
#   scripts/nm_test.sh --action smoke                 # 冒烟：编译 + 最小用例跑通（开发内自检，非正式测试）
#   scripts/nm_test.sh --op relu --action test         # 指定算子（脚本按 ops/<op> / cases/<op> 找）
#   scripts/nm_test.sh 'bash build.sh && ./run_cases.sh'   # 也可以直接给命令
#
# 选项：--run-id <id> | --task <id> | --loop <id> | --op <name> | --action <build|smoke|test|benchmark|profile> |
#       --devices <0,1> | --dry-run | --sync-only | --clean
# 设备：指定单卡被占时默认自动换一张空闲卡再跑（OPX_DEVICE_SWITCH=0 关闭；
#       OPX_DEVICE_CANDIDATES=0,2,3 显式给候选顺序）；实际用卡记进 manifest/台账。
# 构建缓存：默认开（key = op + 源码快照 + 镜像 digest）；OPX_BUILD_CACHE=0 关闭，
#           OPX_FORCE_BUILD=1 强制重编并覆盖缓存。
# 配置：ops/nm.conf（见 nm.conf.example），环境变量优先
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CONF="${OPX_CONF:-$ROOT_DIR/ops/nm.conf}"

# 先读配置文件，再补默认值 —— 只有这个顺序才能让「环境变量 > 配置文件 > 脚本默认」成立。
# （反过来写的话，配置文件里的 ${VAR:-默认} 会因为脚本已经赋过值而静默失效。）
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

OPX_SSH=${OPX_SSH:-tk36}
OPX_SSH_PORT=${OPX_SSH_PORT:-}
OPX_SSH_KEY=${OPX_SSH_KEY:-$HOME/work/pem/KeyPair-fae2027.pem}
OPX_REMOTE_ROOT=${OPX_REMOTE_ROOT:-${OPX_REMOTE_RUN_ROOT:-/data/lfx/op-agent/runs}}
OPX_LOCK_ROOT=${OPX_LOCK_ROOT:-$OPX_REMOTE_ROOT/.locks}
OPX_WORKSPACE=${OPX_WORKSPACE:-$ROOT_DIR}
# 四动作脚本按 OPX_OP 找 ops/<op> 与 cases/<op>；空 = 脚本自己的默认（elementwise_add）
OPX_OP=${OPX_OP:-}
OPX_IMAGE=${OPX_IMAGE:-}
OPX_IMAGE_ID=${OPX_IMAGE_ID:-}
OPX_HOST_ID=${OPX_HOST_ID:-}
OPX_ENV_RECIPE=${OPX_ENV_RECIPE:-}
# docker run 的额外参数（设备映射/驱动挂载等）——「能起 ≠ 能跑算子」那一半在这里
OPX_DOCKER_ARGS=${OPX_DOCKER_ARGS:-}
# 有些算子开发镜像的 ENTRYPOINT 是会 source 一堆 set_env.sh 的包装脚本（实测会卡死），
# 用 OPX_ENTRYPOINT=bash 绕开它，环境准备改由 OPX_ENV_RECIPE 负责。
OPX_ENTRYPOINT=${OPX_ENTRYPOINT:-}
OPX_TIMEOUT=${OPX_TIMEOUT:-900}
OPX_TASK_ID=${OPX_TASK_ID:-}
OPX_LOOP_ID=${OPX_LOOP_ID:-}
# 页面起的动作会把**服务端预算预留审计 id** 注入进来；脚本执行前必须拿它回 /budget-check
# claim（一次性）。命令行直跑没有这个 id → 传 reservation_id=null，由服务端按用量新开审计。
OPX_BUDGET_AUDIT_ID=${OPX_BUDGET_AUDIT_ID:-}
OPX_API=${OPX_API:-http://127.0.0.1:8077}
OPX_SPEC_VERSION_ID=${OPX_SPEC_VERSION_ID:-}
OPX_CRITERIA_REF=${OPX_CRITERIA_REF:-}
OPX_SUITE_REF=${OPX_SUITE_REF:-}
OPX_DEVICES=${OPX_DEVICES:-}
# 构建缓存：同一 op + 源码快照 + 镜像 digest 命中时，复用上次编好的可执行文件。
# 1=开（默认）| 0=关；OPX_FORCE_BUILD=1 强制重编并覆盖缓存。
OPX_BUILD_CACHE=${OPX_BUILD_CACHE:-1}
OPX_FORCE_BUILD=${OPX_FORCE_BUILD:-0}
OPX_BUILD_CACHE_ROOT=${OPX_BUILD_CACHE_ROOT:-$OPX_REMOTE_ROOT/.build-cache}
# 设备互斥模式：exclusive（默认，基准/采集必须）| shared（明确知道有邻居还要跑时）
OPX_LOCK_MODE=${OPX_LOCK_MODE:-exclusive}
# 设备自动切换（2026-10-28 用户口径：去掉「必须同卡」的要求）：指定单卡被占（flock 拿不到）时，
# 按候选顺序换一张空闲卡再跑；实际锁到哪张，容器/manifest/台账就记哪张。
# 1=开（默认）| 0=关；OPX_DEVICE_CANDIDATES=0,2,3 可显式指定候选顺序（逗号分隔）。
OPX_DEVICE_SWITCH=${OPX_DEVICE_SWITCH:-1}
OPX_DEVICE_CANDIDATES=${OPX_DEVICE_CANDIDATES:-}

RUN_ID=""; DRY=""; SYNC_ONLY=""; CLEAN=""; ACTION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --task)   OPX_TASK_ID="$2"; shift 2 ;;
    --loop)   OPX_LOOP_ID="$2"; shift 2 ;;
    --action) ACTION="$2"; shift 2 ;;
    --op)     OPX_OP="$2"; shift 2 ;;
    --devices) OPX_DEVICES="$2"; shift 2 ;;
    --dry-run|-n) DRY=1; shift ;;
    --sync-only) SYNC_ONLY=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
CMD="${*:-}"
if [ -n "$ACTION" ]; then
  case "$ACTION" in
    build|test|benchmark|profile) CMD="bash $ACTION.sh" ;;
    smoke) CMD="bash smoke_check.sh" ;;  # 环境自检 smoke.sh 是另一条脚本，别混
    *) echo "--action 只能是 build|smoke|test|benchmark|profile（收到 ${ACTION}）" >&2; exit 2 ;;
  esac
fi

# P3 预算硬门禁：带 --loop 的真执行必须在碰远端前向控制台校验预算（dry-run/sync-only
# 不动远端，不计费）。门禁是**服务端校验的一次性预留**：页面起动作时后端已写 allowed 审计
# 并把 id 注入成 OPX_BUDGET_AUDIT_ID，这里带着它 claim（同一审计只能消费一次）；没有该 id
# （命令行直跑）则传 reservation_id=null，由服务端按用量新开审计。
# 不再认 OPX_BUDGET_RESERVED —— 旧版里调用方自己设 =1 就能跳过门禁（审查 M4）。
# fail-closed：网络失败 / 非 2xx / 非法响应 / 预留无效 → exit 4，绝不进入远端同步阶段。
if [ -z "$DRY" ] && [ -z "$SYNC_ONLY" ] && [ -n "$OPX_LOOP_ID" ]; then
  [ -n "$ACTION" ] || { echo "--loop 需要同时指定 --action，才能执行预算门禁" >&2; exit 2; }
  BUDGET_JSON=$(/usr/bin/python3 - "$OPX_API" "$OPX_LOOP_ID" "$ACTION" "$OPX_BUDGET_AUDIT_ID" <<'PY'
import json, sys, urllib.error, urllib.request
api, loop_id, action, reservation = sys.argv[1:]
reservation_id = int(reservation) if reservation else None
req = urllib.request.Request(
    api.rstrip("/") + f"/api/loops/{loop_id}/budget-check",
    data=json.dumps({"action": action, "reservation_id": reservation_id}).encode(),
    headers={"Content-Type": "application/json"}, method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=15) as response:
        print(json.dumps(json.load(response), ensure_ascii=False))
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", "replace")
    print(f"ERR {exc.code} {detail}")
    raise SystemExit(1)
PY
  ) || { echo "[nm_test] Loop #${OPX_LOOP_ID} 预算门禁拒绝：${BUDGET_JSON:-请求失败}" >&2; exit 4; }
  echo "-- Loop #${OPX_LOOP_ID} 预算门禁通过：$BUDGET_JSON"
fi

RUN_ID=${RUN_ID:-r$(date +%Y%m%d-%H%M%S)}
REMOTE_DIR="$OPX_REMOTE_ROOT/$RUN_ID"
# 本地产物目录：默认仓库根 runs/；agent 在沙箱里跑时可覆盖到工作区内（OPX_LOCAL_RUNS）
LOCAL_OUT="${OPX_LOCAL_RUNS:-$ROOT_DIR/runs}/$RUN_ID"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$OPX_SSH_KEY")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=10 -i "$OPX_SSH_KEY")
RSSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $OPX_SSH_KEY"
# 页面登记的目标可能是 user@host:port（云主机控制台常见）；端口拆出来交给 ssh -p
if [ -n "$OPX_SSH_PORT" ]; then
  SSH+=(-p "$OPX_SSH_PORT")
  SCP+=(-P "$OPX_SSH_PORT")
  RSSH="$RSSH -p $OPX_SSH_PORT"
fi

echo "== 上机测试 =="
echo "  run_id  : $RUN_ID"
echo "  工作区  : $OPX_WORKSPACE"
echo "  远端    : $OPX_SSH${OPX_SSH_PORT:+:$OPX_SSH_PORT}:$REMOTE_DIR"
echo "  镜像    : ${OPX_IMAGE:-（未配置，见 ops/nm.conf）}"
echo "  动作    : ${ACTION:-（未指定，按命令跑）}"
echo "  设备    : ${OPX_DEVICES:-（未指定：不取锁，锁模式 none）}"
echo "  命令    : ${CMD:-（未给，仅同步）}"

RSYNC_OPTS=(-a --delete --exclude .git --exclude .venv --exclude node_modules
            --exclude runs --exclude dist --exclude __pycache__ --exclude '*.pyc'
            # agent 能力与它们的依赖仓（几十~几百 MB）不进容器：容器只需要源码与脚本。
            # 2026-10-15 实测：装上 skills 后工作区多了 .dsh(14M) + .cannbot(453M)，
            # 不排除的话一次 sync 就会拖住整个 run（rsync 卡在那儿，容器根本没起来）。
            --exclude .dsh --exclude .cannbot --exclude .skills-src
            --exclude .uvcache --exclude .workflow-venv --exclude .opx-chat)
[ -n "$DRY" ] && RSYNC_OPTS+=(-n -v)

if [ -z "$DRY" ]; then
  "${SSH[@]}" "$OPX_SSH" "mkdir -p '$REMOTE_DIR/ws' '$REMOTE_DIR/out' '$OPX_LOCK_ROOT'"
fi
mkdir -p "$LOCAL_OUT"

echo "-- ① 同步工作区 → 远端"
rsync "${RSYNC_OPTS[@]}" -e "$RSSH" "$OPX_WORKSPACE/" "$OPX_SSH:$REMOTE_DIR/ws/"

# 任务源码快照：按 ops/<op>/ 隔离（与 exec_actions.operator_snapshot() 同算法）。
# OPX_OP 为空时仅为命令行旧用法保留全工作区哈希；页面任务永远传 op。
SNAPSHOT_JSON=$(/usr/bin/python3 - "$OPX_WORKSPACE" "$OPX_OP" <<'PY'
import hashlib, json, os, sys
ws, op = sys.argv[1:3]
LIMIT = 20000
h, n, truncated = hashlib.sha256(), 0, False
paths = []
if op and os.path.isdir(os.path.join(ws, "ops", op)):
    root = os.path.join(ws, "ops", op)
    scope = "operator-v1"
    skip = {"build", "__pycache__", ".pytest_cache"}
    h.update(b"opx-operator-snapshot/1\0"); h.update(op.encode()); h.update(b"\0")
else:
    root = ws
    scope = "workspace-v1"
    skip = {"runs", ".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
            ".skills-src", ".opx-chat"}
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for f in filenames:
        if scope == "operator-v1" and f.lower().endswith(".md"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), root)
        if any(part in skip for part in rel.split(os.sep)):
            continue
        paths.append(rel)
for rel in sorted(paths):
    if n >= LIMIT:
        truncated = True
        break
    full = os.path.join(root, rel)
    h.update(rel.encode()); h.update(b"\0")
    with open(full, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(b"\0")
    n += 1
print(json.dumps({"sha256": h.hexdigest(), "files": n, "truncated": truncated, "scope": scope}))
PY
)
# 用例集 ref：没显式给就只按 cases/<op>/ 算，其它算子变更不污染本任务。
if [ -z "$OPX_SUITE_REF" ] && [ -n "$OPX_OP" ] && [ -d "$OPX_WORKSPACE/cases/$OPX_OP" ]; then
  OPX_SUITE_REF="cases@$(/usr/bin/python3 - "$OPX_WORKSPACE/cases/$OPX_OP" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
h = hashlib.sha256()
for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    for f in sorted(filenames):
        p = os.path.join(dirpath, f)
        h.update(os.path.relpath(p, root).encode()); h.update(b"\0")
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
print(h.hexdigest()[:16])
PY
)"
fi

if [ -n "$DRY" ] || [ -n "$SYNC_ONLY" ]; then
  echo "-- 到此为止（$([ -n "$DRY" ] && echo 'dry-run' || echo 'sync-only')：未跑容器）"
  echo "   源码快照：$SNAPSHOT_JSON"
  exit 0
fi
[ -n "$CMD" ] || { echo "缺少要跑的命令：--action test 或 nm_test.sh 'bash build.sh'" >&2; exit 2; }
[ -n "$OPX_IMAGE" ] || { echo "未配置 OPX_IMAGE（算子开发镜像），见 ops/nm.conf" >&2; exit 2; }

SNAP_SHA=$(printf '%s' "$SNAPSHOT_JSON" \
  | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["sha256"])' 2>/dev/null || echo '')

# 镜像 digest：三硬规则要求「镜像 digest 必须留」（比 tag 硬，tag 会被重新 push 覆盖）。
# 放在容器启动前算一次：构建缓存 key 要用它，manifest 也复用它。
IMAGE_DIGEST=""
if [ -n "$OPX_IMAGE" ]; then
  IMAGE_DIGEST=$("${SSH[@]}" "$OPX_SSH" \
    "docker image inspect '$OPX_IMAGE' --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null" || true)
fi
if [ -z "$IMAGE_DIGEST" ]; then
  IMAGE_DIGEST=$("${SSH[@]}" "$OPX_SSH" "docker image inspect '$OPX_IMAGE' --format '{{.Id}}' 2>/dev/null" || true)
  [ -n "$IMAGE_DIGEST" ] && IMAGE_DIGEST="local-id:$IMAGE_DIGEST"
fi
echo "-- 镜像 digest：${IMAGE_DIGEST:-（拿不到）}"

# ③ 构建缓存：smoke/test/benchmark/profile 启动前把「同一代码版本」编好的 exe 放回 /ws/ops/<op>/build/，
#    容器里就不再补编（build 是编译动作本身，不在此列；缓存 key = op + 源码快照 + 镜像 digest）。
BUILD_CACHE_STATUS="off"
BUILD_CACHE_KEY=""
CACHE_DIR=""
CACHE_EXE=""
if [ "$OPX_BUILD_CACHE" = "1" ] && [ -n "$OPX_OP" ] && [ -n "$SNAP_SHA" ]; then
  IMAGE_KEY=$(printf '%s' "${IMAGE_DIGEST:-$OPX_IMAGE}" \
    | /usr/bin/python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:16])')
  CACHE_OP=$(printf '%s' "$OPX_OP" \
    | /usr/bin/python3 -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9_.-]","_",sys.stdin.read().strip()))')
  CACHE_DIR="$OPX_BUILD_CACHE_ROOT/$CACHE_OP/$SNAP_SHA/$IMAGE_KEY"
  CACHE_EXE="$CACHE_DIR/$CACHE_OP"
  CACHE_SO="$CACHE_DIR/lib${CACHE_OP}_host.so"
  BUILD_CACHE_KEY="$OPX_OP/$SNAP_SHA/$IMAGE_KEY"
  # 只要不是 build 本身，都优先复用同代码版本的编译产物（smoke 以前漏了 → 每次重编）
  if [ "$ACTION" = "smoke" ] || [ "$ACTION" = "test" ] || [ "$ACTION" = "benchmark" ] \
     || [ "$ACTION" = "profile" ]; then
    if [ "$OPX_FORCE_BUILD" != "1" ]; then
      if "${SSH[@]}" "$OPX_SSH" "test -x '$CACHE_EXE'" >/dev/null 2>&1; then
        if "${SSH[@]}" "$OPX_SSH" \
          "mkdir -p '$REMOTE_DIR/ws/ops/$OPX_OP/build' && cp '$CACHE_EXE' '$REMOTE_DIR/ws/ops/$OPX_OP/build/$OPX_OP' && chmod +x '$REMOTE_DIR/ws/ops/$OPX_OP/build/$OPX_OP'; if [ -f '$CACHE_SO' ]; then cp '$CACHE_SO' '$REMOTE_DIR/ws/ops/$OPX_OP/build/lib${OPX_OP}_host.so'; fi" >/dev/null 2>&1; then
          BUILD_CACHE_STATUS="hit"
          echo "-- 构建缓存命中：${BUILD_CACHE_KEY}（跳过容器内补编）"
        else
          BUILD_CACHE_STATUS="miss"
          echo "-- 构建缓存命中但复制失败 → 退回容器内补编"
        fi
      else
        BUILD_CACHE_STATUS="miss"
        echo "-- 构建缓存未命中：${BUILD_CACHE_KEY}（容器内自动补编）"
      fi
    else
      BUILD_CACHE_STATUS="forced"
      echo "-- OPX_FORCE_BUILD=1 → 强制重编（完成后覆盖缓存）"
    fi
  fi
fi

# ②-b 设备选择：候选顺序 = 指定卡优先，其后是 npu-smi 报告「无进程」的卡。
#   权威判据是执行侧 flock：npu-smi 看不到「拿了锁但还在编译」的 run（2026-10-28 实测：
#   dev0 被别的 benchmark/test 锁着时 npu-smi 仍显示 No process，plan 会误判空闲）。
#   指定单卡时默认自动换卡；显式多卡（--devices 0,1）仍按整组原子取锁，不换卡。
HELD_DEVICE_TOKEN="__OPX_HELD_DEVICE__"
RUN_DEVICES="$OPX_DEVICES"
SWITCH_MODE=0
if [ "${OPX_DEVICE_SWITCH}" = "1" ] && [ -n "${OPX_DEVICES:-}" ] \
   && [ "${OPX_DEVICES//,/}" = "$OPX_DEVICES" ]; then
  SWITCH_MODE=1
  if [ -n "${OPX_DEVICE_CANDIDATES:-}" ]; then
    DEVICE_CANDIDATES="${OPX_DEVICE_CANDIDATES//,/ }"
  else
    DEVICE_CANDIDATES="$OPX_DEVICES"
    OTHER_FREE=$("${SSH[@]}" "$OPX_SSH" \
      "for i in \$(seq 0 15); do [ -e /dev/davinci\$i ] || break; npu-smi info -t proc-mem -i \$i 2>/dev/null | grep -q 'No process in device' && echo \$i; done" \
      2>/dev/null || true)
    for d in $OTHER_FREE; do
      [ "$d" = "$OPX_DEVICES" ] && continue
      DEVICE_CANDIDATES="$DEVICE_CANDIDATES $d"
    done
  fi
  echo "-- 设备自动切换：指定 dev${OPX_DEVICES}；候选 $(printf '%s' "$DEVICE_CANDIDATES" | tr ' ' ',')"
fi
# 容器/runner 里最终写的设备号：自动切换时先写占位符，锁到哪张就替换成哪张
RENDER_DEVICE="$OPX_DEVICES"
[ "$SWITCH_MODE" = "1" ] && RENDER_DEVICE="$HELD_DEVICE_TOKEN"

# ② 远端 runner：命令写进文件再挂进容器，避免层层引号地狱
#    **版本对齐字段也在这里注入**：test/benchmark/profile 的报告要回写 spec_version_id /
#    criteria_ref / suite_ref / 代码快照哈希，门禁逐字比对它们与 run 行是否一致
#    （不一致 → E_REPORT_VERSION_MISMATCH）。哈希**不在容器里重算** —— 两边各算一次就多一个
#    漂移面，而门禁要的是「真实执行那一版」的哈希。
# 显式模板兼容 BSD/macOS 与 GNU/Linux；GNU 的 -t 不接受没有 X 后缀的前缀。
RUNNER=$(mktemp "${TMPDIR:-/tmp}/nm_test_runner.XXXXXX")
{
  echo 'set -e'
  echo 'cd /ws'
  echo "export OPX_RUN_ID='$RUN_ID'"
  echo "export OPX_ACTION='$ACTION'"
  echo "export OPX_SPEC_VERSION_ID='$OPX_SPEC_VERSION_ID'"
  echo "export OPX_CRITERIA_REF='$OPX_CRITERIA_REF'"
  echo "export OPX_SUITE_REF='$OPX_SUITE_REF'"
  echo "export OPX_CODE_SNAPSHOT_SHA256='$SNAP_SHA'"
  echo "export OPX_OUT='/out'"
  [ -n "$OPX_OP" ] && echo "export OPX_OP='$OPX_OP'"
  echo "export OPX_DEVICES='$RENDER_DEVICE'"
  # 容器内再用 export 覆写一次设备可见性：`-e` 重复时只靠 docker 的参数顺序不可靠
  # （glibc getenv 取第一个、bash 取最后一个），这里显式 export 保证「锁哪张就跑哪张」。
  # 自动切换时这里先写占位符，远端 lockrun 拿到实际锁后会就地替换。
  [ -n "$OPX_DEVICES" ] && echo "export ASCEND_RT_VISIBLE_DEVICES='$RENDER_DEVICE'"
  echo "export OPX_LOCK_MODE='$OPX_LOCK_MODE'"
  [ -n "$OPX_ENV_RECIPE" ] && echo "$OPX_ENV_RECIPE"
  echo "$CMD"
} > "$RUNNER"
"${SCP[@]}" "$RUNNER" "$OPX_SSH:$REMOTE_DIR/run.sh"
cp "$RUNNER" "$LOCAL_OUT/run.sh"
rm -f "$RUNNER"

# 远端容器执行必须带超时：2026-10-14 实测 torch_npu 初始化会挂死（npu-smi 卡住）
# → 没有 timeout 的话本地脚本会无限等，run 也永远不结束。
INNER="set -o pipefail; bash /run.sh 2>&1 | tee /out/run.log"
if [ -n "$OPX_ENTRYPOINT" ]; then
  # 已经用 --entrypoint 指定了解释器 → 这里只能给参数，否则会变成 `bash bash -lc ...`（实测 exit 126）
  ENTRY_ARGS="--entrypoint $OPX_ENTRYPOINT"
  TAIL="-lc '$INNER'"
else
  ENTRY_ARGS=""
  TAIL="bash -lc '$INNER'"
fi
CNAME="opx-${RUN_ID}"
# 设备可见性必须与「锁住的那张卡」一致（2026-10-20 实测踩到）：
# nm.conf 的 OPX_DOCKER_ARGS 固定暴露 0..7，而算子写死 aclrtSetDevice(0) —— 于是
# `--devices 1` 锁的是 dev1、真正跑的却是物理 dev0（台账 device_id 会撒谎，且会和别人的
# 进程抢卡：实测 aclrtSetDevice 失败 ret=507033、单例卡 150s）。这里用**后写的 -e 覆盖**
# 前面的同名变量，让容器内 device 0 == 锁住的第一张卡。
# 自动切换时先用占位符，远端 lockrun 锁到实际卡后再替换（见 ③）。
if [ "$SWITCH_MODE" = "1" ]; then
  DEVICE_ENV="-e ASCEND_RT_VISIBLE_DEVICES=$HELD_DEVICE_TOKEN"
elif [ -n "$OPX_DEVICES" ]; then
  DEVICE_ENV="-e ASCEND_RT_VISIBLE_DEVICES=$OPX_DEVICES"
else
  DEVICE_ENV=""
fi
DOCKER_CMD="timeout --kill-after=15 ${OPX_TIMEOUT:-900}s docker run --rm --name $CNAME $ENTRY_ARGS $OPX_DOCKER_ARGS $DEVICE_ENV -v $REMOTE_DIR/ws:/ws -v $REMOTE_DIR/run.sh:/run.sh -v $REMOTE_DIR/out:/out -w /ws $OPX_IMAGE $TAIL"
printf '%s\n' "$DOCKER_CMD" > "$LOCAL_OUT/docker_cmd.txt"

# ③ 设备互斥（共享裸金属的硬要求）：先锁后跑，绝不硬挤（卡被硬占时容器起不来 dcmi -8020，
#    挤上去耗时会被邻居污染）。但与旧版不同：拿锁失败**默认自动换一张空闲卡重试**，
#    候选全占才判「设备被占」不执行（2026-10-28 用户口径：去掉「必须同卡」的要求）。
LOCK_MODE="none"
LOCK_ACQUIRED="no"
replace_device_token() {
  /usr/bin/python3 - "$1" "$HELD_DEVICE_TOKEN" "$2" <<'PY'
import sys
path, token, value = sys.argv[1:4]
try:
    text = open(path, encoding="utf-8").read()
except OSError:
    raise SystemExit(0)
open(path, "w", encoding="utf-8").write(text.replace(token, str(value)))
PY
}
if [ -n "$OPX_DEVICES" ]; then
  LOCKER=$(mktemp "${TMPDIR:-/tmp}/nm_test_locker.XXXXXX")
  {
    echo '#!/usr/bin/env bash'
    echo 'set -u'
    echo "LOCK_ROOT='$OPX_LOCK_ROOT'"
    echo "RUN_ID='$RUN_ID'"
    echo "MODE='$OPX_LOCK_MODE'"
    if [ "$SWITCH_MODE" = "1" ]; then
      echo "CANDIDATES='$DEVICE_CANDIDATES'"
      echo "RUNNER_PATH='$REMOTE_DIR/run.sh'"
      echo "HELD_TOKEN='$HELD_DEVICE_TOKEN'"
    else
      echo "DEVICES='$OPX_DEVICES'"
    fi
    echo 'mkdir -p "$LOCK_ROOT"'
    # 远端要执行的命令（含 docker run）通过环境变量传进去，避免引号地狱；
    # **必须在锁体之前导出**：自动换卡分支要按实际锁到的设备替换 INNER_CMD 里的占位符。
    printf 'export INNER_CMD=%s\n' "$(printf '%q' "$DOCKER_CMD")"
    if [ "$SWITCH_MODE" = "1" ]; then
      cat <<'LOCKER_SWITCH'
fd=9
LOCKED=""
for d in $CANDIDATES; do
  eval "exec $fd>\"\$LOCK_ROOT/dev\$d.lock\""
  if [ "$MODE" = exclusive ]; then
    if ! flock -n "$fd"; then
      echo "OPX_LOCK_BUSY dev$d 被占；持有者记录：$(cat "$LOCK_ROOT/dev$d.owner" 2>/dev/null || echo '（无 owner 记录，可能是进程崩溃后残留）')" >&2
      eval "exec $fd>&-"
      fd=$((fd+1))
      continue
    fi
  else
    if ! flock -s "$fd"; then
      echo "OPX_LOCK_BUSY dev$d（共享锁也拿不到）" >&2
      eval "exec $fd>&-"
      fd=$((fd+1))
      continue
    fi
  fi
  echo "$RUN_ID $(whoami)@$(hostname) $(date -Is) mode=$MODE" > "$LOCK_ROOT/dev$d.owner"
  LOCKED="$d"
  break
done
if [ -z "$LOCKED" ]; then
  echo "OPX_LOCK_BUSY 候选设备全被占：$CANDIDATES" >&2
  exit 3
fi
# 实际锁到哪张，容器命令与 runner 就写哪张（自动切卡的落点）
if [ -n "${HELD_TOKEN:-}" ]; then
  INNER_CMD=${INNER_CMD//$HELD_TOKEN/$LOCKED}
  if [ -f "$RUNNER_PATH" ]; then
    sed -i "s/$HELD_TOKEN/$LOCKED/g" "$RUNNER_PATH" 2>/dev/null || true
  fi
fi
echo "OPX_LOCK_OK devices=$LOCKED mode=$MODE"
LOCKER_SWITCH
    else
      cat <<'LOCKER_BODY'
fd=9
held=()
for d in ${DEVICES//,/ }; do
  eval "exec $fd>\"\$LOCK_ROOT/dev\$d.lock\""
  if [ "$MODE" = exclusive ]; then
    if ! flock -n "$fd"; then
      echo "OPX_LOCK_BUSY dev$d 被占；持有者记录：$(cat "$LOCK_ROOT/dev$d.owner" 2>/dev/null || echo '（无 owner 记录，可能是进程崩溃后残留）')" >&2
      exit 3
    fi
  else
    flock -s "$fd" || { echo "OPX_LOCK_BUSY dev${d}（共享锁也拿不到）" >&2; exit 3; }
  fi
  echo "$RUN_ID $(whoami)@$(hostname) $(date -Is) mode=$MODE" > "$LOCK_ROOT/dev$d.owner"
  held+=("$d")
  fd=$((fd+1))
done
echo "OPX_LOCK_OK devices=${held[*]} mode=$MODE"
LOCKER_BODY
    fi
    echo 'eval "$INNER_CMD"'
    echo 'rc=$?'
    if [ "$SWITCH_MODE" = "1" ]; then
      echo 'echo "OPX_LOCK_RELEASED devices=$LOCKED rc=$rc"'
    else
      echo 'echo "OPX_LOCK_RELEASED devices=$DEVICES rc=$rc"'
    fi
    echo 'exit $rc'
  } > "$LOCKER"
  "${SCP[@]}" "$LOCKER" "$OPX_SSH:$REMOTE_DIR/lockrun.sh"
  cp "$LOCKER" "$LOCAL_OUT/lockrun.sh"
  rm -f "$LOCKER"

  echo "-- ② 在容器里跑一次（镜像：${OPX_IMAGE##*/}，设备：${OPX_DEVICES}，模式：${OPX_LOCK_MODE}）"
  set +e
  "${SSH[@]}" "$OPX_SSH" "bash $REMOTE_DIR/lockrun.sh; rc=\$?; exit \$rc" 2>&1 | tee "$LOCAL_OUT/lock.log"
  RC=${PIPESTATUS[0]}
  set -e
  if grep -q "OPX_LOCK_OK" "$LOCAL_OUT/lock.log"; then
    LOCK_MODE="$OPX_LOCK_MODE"; LOCK_ACQUIRED="yes"
    if [ "$SWITCH_MODE" = "1" ]; then
      RAN_DEVICE=$(sed -n 's/.*OPX_LOCK_OK devices=\([^ ]*\) mode=.*/\1/p' "$LOCAL_OUT/lock.log" | tail -1)
      if [ -n "$RAN_DEVICE" ]; then
        RUN_DEVICES="$RAN_DEVICE"
        replace_device_token "$LOCAL_OUT/docker_cmd.txt" "$RAN_DEVICE"
        if [ "$RAN_DEVICE" != "$OPX_DEVICES" ]; then
          echo "-- 指定 dev${OPX_DEVICES} 被占 → 自动切到空闲 dev${RAN_DEVICE}"
        fi
      fi
    fi
  elif [ "$RC" = "3" ]; then
    LOCK_MODE="$OPX_LOCK_MODE"; LOCK_ACQUIRED="no"
    echo "-- 设备被占：候选 ${DEVICE_CANDIDATES:-$OPX_DEVICES} 全不可用，本次不执行（稍后重跑；不当作算子 bug）" >&2
  fi
else
  echo "-- ② 在容器里跑一次（镜像：${OPX_IMAGE##*/}，未指定设备：不取锁）"
  set +e
  "${SSH[@]}" "$OPX_SSH" "$DOCKER_CMD; rc=\$?; exit \$rc"
  RC=$?
  set -e
fi
echo "-- 容器退出码：$RC"
# 兜底：timeout 只杀 docker run 客户端，容器可能还在跑并占住卡（2026-10-14 实测泄漏过两次）
"${SSH[@]}" "$OPX_SSH" "docker rm -f $CNAME >/dev/null 2>&1 || true"

# 成功执行后把候选 exe 存进远端构建缓存：build 动作存；test/benchmark/profile 未命中时
# 容器内自动补编也会顺带存，下一次同快照/同镜像就能直接命中。
if [ "$OPX_BUILD_CACHE" = "1" ] && [ -n "$CACHE_DIR" ] && [ "$RC" = "0" ]; then
  case "$ACTION" in
    build|smoke|test|benchmark|profile)
      if "${SSH[@]}" "$OPX_SSH" "test -x '$REMOTE_DIR/ws/ops/$OPX_OP/build/$OPX_OP'" >/dev/null 2>&1; then
        if "${SSH[@]}" "$OPX_SSH" \
          "mkdir -p '$CACHE_DIR' && cp '$REMOTE_DIR/ws/ops/$OPX_OP/build/$OPX_OP' '$CACHE_EXE' && chmod +x '$CACHE_EXE' && sha256sum '$CACHE_EXE' > '$CACHE_EXE.sha256'; if [ -f '$REMOTE_DIR/ws/ops/$OPX_OP/build/lib${OPX_OP}_host.so' ]; then cp '$REMOTE_DIR/ws/ops/$OPX_OP/build/lib${OPX_OP}_host.so' '$CACHE_SO' && sha256sum '$CACHE_SO' > '$CACHE_SO.sha256'; fi" >/dev/null 2>&1; then
          echo "-- 构建缓存已保存：$BUILD_CACHE_KEY"
          case "$BUILD_CACHE_STATUS" in
            off|miss) BUILD_CACHE_STATUS="saved" ;;
          esac
        else
          echo "-- 构建缓存保存失败（不影响本次结果）：$BUILD_CACHE_KEY" >&2
        fi
      fi
      ;;
  esac
fi

echo "-- ④ 回读产物"
rsync -a -e "$RSSH" "$OPX_SSH:$REMOTE_DIR/out/" "$LOCAL_OUT/out/" || true
rsync -a -e "$RSSH" "$OPX_SSH:$REMOTE_DIR/run.sh" "$LOCAL_OUT/run.sh" 2>/dev/null || true

echo "-- ⑤ 写 manifest"
/usr/bin/python3 - "$LOCAL_OUT" "$RUN_ID" "$RC" "$OPX_SSH" "$OPX_IMAGE" "$IMAGE_DIGEST" \
  "$SNAPSHOT_JSON" "$ACTION" "$OPX_SPEC_VERSION_ID" "$OPX_CRITERIA_REF" "$OPX_SUITE_REF" \
  "$RUN_DEVICES" "$LOCK_MODE" "$LOCK_ACQUIRED" "$OPX_IMAGE_ID" "$OPX_HOST_ID" "$OPX_WORKSPACE" \
  "$BUILD_CACHE_STATUS" "$BUILD_CACHE_KEY" "$OPX_TASK_ID" "$OPX_DEVICES" "$SWITCH_MODE" <<'PY'
import hashlib, json, os, sys, datetime
(out, run_id, rc, host, image, digest, snap_json, action, spec_id, criteria, suite,
 devices, lock_mode, lock_acquired, image_id, host_id, workspace,
 build_cache_status, build_cache_key, task_id, requested_devices,
 device_switch) = sys.argv[1:23]

# 产物类型按契约命名（gate 校验要按类型找 correctness/performance/metrics）。
# ⚠️ **类型要看扩展名**：correctness.json 与 correctness.md 的 stem 都是 "correctness"，
#    若不加区分，两者都会被打成 correctness 类型 —— 门禁按类型取值时会取到 .md，
#    于是「报告根本没被解析」却没报错（2026-10-15 实测踩到：run#4 的用例 FAIL 被判漏）。
#    规则：只有 .json 才认那些机读类型；.md/.txt 一类人看的伴生品一律 report。
TYPE_MAP = {"correctness": "correctness", "performance": "performance", "metrics": "metrics",
            "smoke": "smoke"}
arts = []
root = os.path.join(out, "out")
for dirpath, _, files in os.walk(root):
    for f in sorted(files):
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, out)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        stem, _, ext = f.rpartition(".")
        stem = stem or f
        if rel.startswith("out/cases/") or "/cases/" in rel:
            atype = "cases"
        elif stem == "run":                     # 统一执行入口 tee 出来的日志
            atype = "log"
        elif ext.lower() == "json":
            atype = TYPE_MAP.get(stem, "report")
        else:
            atype = "report"
        arts.append({"type": atype, "path": rel, "sha256": h.hexdigest(),
                     "size_bytes": os.path.getsize(p)})

snap = json.loads(snap_json)
man = {
    "schema": "opx-manifest/1",
    "run_id": run_id,
    # 任务归属：删除任务时按 manifest 里的 task_id 校验 run 目录归属，防止误删别的任务产物。
    # 旧 manifest 没有这个字段（None），调用方需兼容（不能因为缺字段就当成「不属于任何任务」）。
    "task_id": int(task_id) if task_id else None,
    "host": host,
    "host_id": int(host_id) if host_id else None,
    "image": image,
    "image_id": int(image_id) if image_id else None,
    "image_digest": digest or None,
    "action": action or "custom",
    "exit_code": int(rc),
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workspace": workspace,
    # 版本对齐（三硬规则第 3 条）
    "code_snapshot_sha256": snap["sha256"],
    "evidence_scope": snap.get("scope"),
    "snapshot": {"files": snap["files"], "truncated": snap["truncated"]},
    "spec_version_id": int(spec_id) if spec_id else None,
    "criteria_ref": criteria or None,
    "suite_ref": suite or None,
    # 测量条件
    "device": devices or None,
    "device_requested": requested_devices or None,
    "device_auto_switched": bool(device_switch == "1" and requested_devices
                                 and devices and str(devices) != str(requested_devices)),
    "lock_mode": lock_mode,
    "lock_acquired": lock_acquired,
    "build_cache": {"status": build_cache_status or "off", "key": build_cache_key or None},
    "artifacts": arts,
}
with open(os.path.join(out, "manifest.json"), "w") as fh:
    json.dump(man, fh, ensure_ascii=False, indent=2)
print(f"   产物 {len(arts)} 个 → {os.path.relpath(out, os.getcwd())}/manifest.json")
PY

if [ -n "$OPX_API" ] && [ -n "$OPX_TASK_ID" ]; then
  echo "-- ⑥ 回填台账（task #${OPX_TASK_ID}）"
  /usr/bin/python3 - "$LOCAL_OUT" "$RC" "$OPX_API" "$OPX_TASK_ID" "$OPX_LOOP_ID" <<'PY'
import json, os, sys, urllib.request
out, rc, api, task, loop = sys.argv[1:6]
man = json.load(open(os.path.join(out, "manifest.json")))
run_action = man.get("action") or "custom"
# 退出码 3 = 设备被占 → 环境问题（方案 §4.4 的失败分类），不当算子 bug
failure_kind = None
if int(rc) == 3:
    failure_kind = "env_failed"
elif int(rc) != 0:
    failure_kind = ("execution_failed" if run_action == "custom" else
                    "build_failed" if run_action == "build" else "correctness_failed")
body = {
    "action": run_action,
    "loop_id": int(loop) if loop else None,
    "status": "ok" if int(rc) == 0 else "failed",
    "exit_code": int(rc),
    "failure_kind": failure_kind,
    "host_id": man.get("host_id"),
    "image_id": man.get("image_id"),
    "image_digest": man.get("image_digest"),
    "spec_version_id": man.get("spec_version_id"),
    "code_snapshot_sha256": man.get("code_snapshot_sha256"),
    "evidence_scope": man.get("evidence_scope"),
    "suite_ref": man.get("suite_ref"),
    "criteria_ref": man.get("criteria_ref"),
    "device_id": man.get("device"),
    "command": (open(os.path.join(out, "docker_cmd.txt")).read().strip()
                if os.path.exists(os.path.join(out, "docker_cmd.txt")) else None),
    "cwd": man.get("workspace"),
    "remote_workdir": man.get("run_id"),
    "manifest_path": os.path.join(out, "manifest.json"),
    "sync_state": "pulled",
    "note": (f"lock_mode={man.get('lock_mode')} lock_acquired={man.get('lock_acquired')}"
             + (f" device={man.get('device')}" if man.get("device") else "")
             + (" device_auto_switched" if man.get("device_auto_switched") else "")),
    "artifacts": [{"type": a["type"], "path": a["path"], "sha256": a["sha256"],
                   "size_bytes": a["size_bytes"]} for a in man.get("artifacts", [])],
}
req = urllib.request.Request(f"{api}/api/tasks/{task}/runs", data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        run_id = json.load(r)["id"]
    print("   台账 run #", run_id, "→ 门禁：", f"{api}/api/runs/{run_id}/gate")
except Exception as e:
    print("   台账回填失败（不影响测试结果）：", e)
PY
else
  echo "-- ⑥ 跳过台账回填（未给 OPX_API / OPX_TASK_ID）"
fi

if [ -n "$CLEAN" ] && [ -z "$DRY" ]; then
  echo "-- ⑦ 清理远端目录"
  "${SSH[@]}" "$OPX_SSH" "rm -rf '$REMOTE_DIR'"
fi

echo "== 完成：exit=${RC}（锁：${LOCK_MODE}/${LOCK_ACQUIRED}，设备：${RUN_DEVICES:-无}），本地产物在 ${LOCAL_OUT} =="
exit "$RC"
