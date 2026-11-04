#!/usr/bin/env bash
# 把 CANNBot 的 skills / agents 装进**任务工作区**，并把「装了什么」写成可复核的 lock 文件。
#
# 为什么要有这个脚本（2026-10-15 复查的落地）：
#   1. 之前的安装落在 `~/.hermes/cache/scratch/dsh-ws/.dsh/` —— Hermes 的 scratch 目录，**空闲 24h 会被清**；
#      而且是 **symlink 到 `~/.cache/cannbot/...` 的 checkout**（同样在会被清的缓存里）。
#      M1 一开工就要用 skills，这样放着等于随时会丢。
#   2. dsh 按「projectRoot/.dsh/skills|agents」找能力 → 正确落点是**任务工作区**。
#   3. 光写「pin commit」不够：那次安装的 cannbot-skills 子模块是 **dirty** 的，commit 复现不出内容。
#      所以 lock 里记每个文件的 sha256，`--verify` 能随时复核。
#
# 安装方式说明：cannbot 的 source 安装是 live symlink（官方文档也写明 dirty 内容不随 commit 复现），
# 本脚本装完**物化成真文件**（cp -RL），让工作区自包含、内容可哈希。
#
# 用法：
#   scripts/install_skills.sh                     # 装到默认工作区 workspaces/cann-ops
#   scripts/install_skills.sh --target <dir>
#   scripts/install_skills.sh --source <cannbot-repo>   # 指定来源仓库（默认 .skills-src/cannbot）
#   scripts/install_skills.sh --verify            # 只按 lock 复核当前安装（V3 可复现性）
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

PLUGIN="ops-direct-invoke"
TOOL="dsh"
TARGET="${OPX_WORKSPACE:-$ROOT_DIR/workspaces/cann-ops}"
SRC="${OPX_CANNBOT_SRC:-$ROOT_DIR/.skills-src/cannbot}"
LOCK="$ROOT_DIR/ops/dsh/skills.lock"
VERIFY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --source) SRC="$2"; shift 2 ;;
    --verify) VERIFY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [ -n "$VERIFY" ]; then
  exec /usr/bin/python3 - "$TARGET" "$LOCK" <<'PY'
import hashlib, json, os, sys
target, lock_path = sys.argv[1], sys.argv[2]
if not os.path.exists(lock_path):
    print(f"没有 lock 文件：{lock_path}（先跑一次 install）"); raise SystemExit(2)
lock = json.load(open(lock_path))
bad, missing, extra = [], [], []
for rel, want in lock["files"].items():
    p = os.path.join(target, rel)
    if not os.path.isfile(p):
        missing.append(rel); continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != want:
        bad.append(rel)
root = os.path.join(target, ".dsh")
for base, _, files in os.walk(root):
    for f in files:
        rel = os.path.relpath(os.path.join(base, f), target)
        if rel not in lock["files"]:
            extra.append(rel)
print(f"lock: {lock['plugin']}@{lock['plugin_version']} 装于 {lock['installed_at']}")
print(f"目标: {target}")
print(f"一致 {len(lock['files']) - len(bad) - len(missing)} / 缺失 {len(missing)} / 内容不符 {len(bad)} / 多出 {len(extra)}")
for label, items in (("缺失", missing), ("内容不符", bad), ("多出", extra)):
    for x in items[:10]:
        print(f"  {label}: {x}")
raise SystemExit(0 if not (bad or missing) else 1)
PY
fi

# ---- ① 来源仓库：必须在**不会被清**的地方（cache/scratch 都不行）-------------
if [ ! -d "$SRC" ]; then
  CACHED=$(ls -d "$HOME"/.cache/cannbot/sources/*/repository 2>/dev/null | head -1 || true)
  if [ -n "$CACHED" ]; then
    echo "-- 来源缺失，从缓存 checkout 拷一份到 ${SRC}（离线、内容与那次安装一致）"
    mkdir -p "$(dirname "$SRC")"
    cp -R "$CACHED" "$SRC"
  else
    echo "-- 来源缺失且无缓存，从 gitcode 克隆（需要网络）"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 https://gitcode.com/cann/cannbot.git "$SRC"
    git -C "$SRC" submodule update --init --depth 1 vendor/cannbot-skills || true
  fi
fi
if [ ! -f "$SRC/script/bin/cannbot.js" ]; then
  echo "来源目录 $SRC 里找不到 script/bin/cannbot.js —— 不是 cannbot 仓库？" >&2
  exit 2
fi
command -v node >/dev/null || { echo "需要 node（>=20）来跑 cannbot 安装器" >&2; exit 2; }
echo "-- node $(node -v)"

# ---- ② 安装（source 模式：cannbot 会把 skills/agents 以 symlink 挂进 <target>/.dsh）----
mkdir -p "$TARGET"
echo "-- 安装 ${PLUGIN} → ${TARGET}/.dsh（tool=${TOOL}）"
# `--skills <path>` 是必须的：共享 skills（vendor/cannbot-skills）那份是 **dirty** 的（有未提交改动），
# 安装器默认拒绝这种状态（"has uncommitted changes"）。官方文档说本地路径覆盖会保留未提交内容、
# 且不会去动 submodule —— 我们要的正是「与那次安装一致的那份内容」，然后把它逐文件指纹写进 lock。
#
# `CANNBOT_SKIP_INIT=1` 也是必须的（macOS 上的坑）：CLI 默认会去跑插件的 `init.sh`，而
# `script/bin/source-plugin-init.sh` 在 `set -u` 下展开**空数组** `"${SKILLS_ARGS[@]}"`，
# macOS 自带 bash 3.2 会把它当未定义变量直接报错（`SKILLS_ARGS[@]: unbound variable`）。
# 置这个标志即让 CLI 跳过 init.sh，直接做安装（内容一致；插件的依赖仓仍会由安装器拉取）。
CANNBOT_SKIP_INIT=1 node "$SRC/script/bin/cannbot.js" install "$PLUGIN" --source "$SRC" --tool "$TOOL" \
  --target "$TARGET" --skills "$SRC/vendor/cannbot-skills"

# ---- ③ 物化：把 symlink 换成真文件（工作区自包含）-----------------------------
/usr/bin/python3 - "$TARGET" <<'PY'
import os, shutil, sys
target = os.path.join(sys.argv[1], ".dsh")
converted = 0
for base in ("skills", "agents"):
    root = os.path.join(target, base)
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.islink(p):
            continue
        real = os.path.realpath(p)
        if not os.path.exists(real):
            print(f"  !! 悬空 symlink（来源已被清）：{p} → {os.readlink(p)}")
            continue
        if os.path.isdir(real):
            os.unlink(p); shutil.copytree(real, p, symlinks=False)
        else:
            os.unlink(p); shutil.copy2(real, p)
        converted += 1
print(f"-- 物化 {converted} 个 symlink → 真文件")
PY

# ---- ③.5 控制台策略覆盖：追加「不要静默 + 真机验证入口」到项目 AGENTS.md（幂等）----
# 为什么放在安装后：AGENTS.md 来自 cannbot 模板（上游要求「静默执行」），控制台用户不要静默；
# 模板更新/重装会覆盖它，所以每次装完都补这段（已存在则跳过）。
/usr/bin/python3 - "$TARGET" <<'PY'
import os, sys

target = sys.argv[1]
path = os.path.join(target, "AGENTS.md")
if not os.path.isfile(path):
    raise SystemExit(0)
text = open(path, encoding="utf-8").read()
marker = "## 用户偏好（控制台追加，优先于上文）"
if marker in text:
    raise SystemExit(0)

block = """

## 真机验证入口（控制台追加）

本工作区新增了 `scripts/nm.sh`（转发控制台仓库根的 `scripts/nm_test.sh`）：dsh agent 可直接在**本工作区**发起上机验证。
任务感知：它按当前任务配置的执行机/镜像/设备执行（不是 `ops/nm.conf` 的写死默认值）。

```bash
scripts/nm.sh --dry-run --op <op> --action build     # 先算计划，不动容器
scripts/nm.sh --op <op> --action build               # 上机编译
scripts/nm.sh --op <op> --action test                # 上机正确性测试（自动补编）
scripts/nm.sh --op <op> --action benchmark           # 基准（独占卡，慢）
scripts/nm.sh --op <op> --action profile             # 采集（独占卡，慢）
```

- SSH/镜像/设备锁/超时兜底/run 台账回填都由原脚本处理；脚本会占用远端卡，**不要与控制台「概览 → 四动作」同时触发同一动作**；
- 本机没有容器/工具链不代表不能验证：verifier/修复节点应优先用 `scripts/nm.sh` 拿真实证据；实在没有执行条件才记录阻塞，不得伪造结果。

## 用户偏好（控制台追加，优先于上文）

本控制台用户**明确不要静默**；本节优先于上文「启动后尽量静默执行」：

- workflow 启动、每个节点开始 / 完成 / 重试 / 失败、需要用户决策时，都在对话里发一条**简短**进度（一两句，不刷屏）；
- 不要再说「你可以离开了，即将开始静默开发」；改为「开发已开始，我会在节点完成时同步进度」；
- 长节点或轮询期间定期播报一行状态，让用户能判断没有卡死；
- 最终交付仍按原要求汇报：交付位置、验收结论、未验证项清单。
"""

open(path, "w", encoding="utf-8").write(text.rstrip() + block)
print("-- 追加控制台策略覆盖（不静默 / 真机验证入口）到 AGENTS.md")
PY

# ---- ③.6 证据质量策略：把真机开发复盘出的防错规则固化进每个任务工作区 --------
# 独立于上面的用户偏好 marker，确保老工作区升级/重装时也能补上新策略。
/usr/bin/python3 - "$TARGET" <<'PY'
import os, sys

target = sys.argv[1]
path = os.path.join(target, "AGENTS.md")
if not os.path.isfile(path):
    raise SystemExit(0)
text = open(path, encoding="utf-8").read()
marker = "## 证据质量门禁（控制台追加）"
if marker in text:
    raise SystemExit(0)

block = r'''

## 证据质量门禁（控制台追加）

以下规则来自真实算子工作流复盘，优先于角色自行推断；验收报告违反任一项时不得 PASS：

- **身份取权威值**：规格、`suite_ref`、代码快照、镜像、设备以控制台 API / 当前任务配置 / 本次 run manifest 为准；prompt、旧报告和人工摘要里的值只作线索。发现冲突必须写明并使用权威值。
- **证据按算子隔离**：新任务统一使用 `operator-v1`，代码只哈希 `ops/<op>/`，用例只哈希 `cases/<op>/`；其它算子变化不得使本任务证据失效。
- **宣称命中必须有原始证据**：源码中“存在某分支”不等于测试“命中该分支”。分支、ABI 次数、NPU 路由、设备号、PASS 数量都必须能回指日志、报告字段或测试断言；无法证明就写 `UNKNOWN` / 未覆盖。
- **执行模式不可混算**：`per_leaf`、`whole_tree/compiled`、eager、jit、JVP/VJP 分开记录。聚合叶子数、拒绝条件和预期异常只适用于对应模式，不能把 whole-tree 的 `n_leaves` 约束套到 per-leaf 单次调用。
- **字节与对齐必须实算**：用 `bytes = elements * dtype_size` 计算，再判断 `bytes % 32`；报告同时写元素数、dtype 字节数和总字节数，禁止凭 shape 名称或印象标“32B 对齐”。
- **结果计数从日志解析**：`passed / failed / xfailed / unsupported` 分开报告；`xfail`、`unsupported` 永远不得写成 PASS。重试后保留旧失败，只把与当前身份四元组一致且门禁通过的 run 标为当前有效证据。
- **交付前再对齐一次**：算子实现或用例变更后，重新计算 operator-v1 快照和用例哈希，只重跑受影响动作；纯文档变化不应使算子快照漂移。生成交付包前必须确认 build → smoke → test → benchmark 均指向当前身份。
'''

open(path, "w", encoding="utf-8").write(text.rstrip() + block)
print("-- 追加证据质量门禁到 AGENTS.md")
PY

# ---- ④ 校验 + 写 lock -------------------------------------------------------
/usr/bin/python3 - "$TARGET" "$SRC" "$LOCK" "$PLUGIN" "$TOOL" <<'PY'
import hashlib, json, os, subprocess, sys, datetime

target, src, lock_path, plugin, tool = sys.argv[1:6]
dsh = os.path.join(target, ".dsh")
skills_dir, agents_dir = os.path.join(dsh, "skills"), os.path.join(dsh, "agents")
skills = sorted(d for d in os.listdir(skills_dir)) if os.path.isdir(skills_dir) else []
agents = sorted(f for f in os.listdir(agents_dir)) if os.path.isdir(agents_dir) else []

problems = []
for s in skills:
    if not os.path.isfile(os.path.join(skills_dir, s, "SKILL.md")):
        problems.append(f"skill 缺 SKILL.md：{s}")
for base, dirs, files in os.walk(dsh):
    for n in dirs + files:
        p = os.path.join(base, n)
        if os.path.islink(p):
            problems.append(f"仍是 symlink：{os.path.relpath(p, target)}")

files = {}
for base, _, fs in os.walk(dsh):
    for f in fs:
        p = os.path.join(base, f)
        rel = os.path.relpath(p, target)
        files[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", src, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def git_in(path, *args):
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


skills_src = os.path.join(src, "vendor", "cannbot-skills")
lock = {
    "schema": "opx-skills-lock/1",
    "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tool": tool,
    "plugin": plugin,
    "plugin_version": json.load(open(os.path.join(src, "script", "package.json")))["version"],
    "install_mode": "source→materialized（装完把 symlink 物化成真文件）",
    "target": os.path.relpath(target, os.path.dirname(lock_path)),
    "source": {
        "path_hint": ".skills-src/cannbot（仓内、git 忽略、不随缓存清理）",
        "cannbot_commit": git("rev-parse", "HEAD"),
        "cannbot_remote": git("remote", "get-url", "origin"),
        "skills_commit": git_in(skills_src, "rev-parse", "HEAD") if os.path.isdir(skills_src) else "",
        "skills_dirty": bool(git_in(skills_src, "status", "--porcelain")) if os.path.isdir(skills_src) else None,
    },
    "counts": {"skills": len(skills), "agents": len(agents)},
    "skills": skills,
    "agents": agents,
    "file_count": len(files),
    "files": files,
}
json.dump(lock, open(lock_path, "w"), ensure_ascii=False, indent=2, sort_keys=False)

print(f"-- skills {len(skills)} / agents {len(agents)} / 文件 {len(files)} → lock: {lock_path}")
print(f"   cannbot commit {lock['source']['cannbot_commit'][:12]}…，"
      f"skills commit {str(lock['source']['skills_commit'])[:12]}…，dirty={lock['source']['skills_dirty']}")
if problems:
    print("   !! 校验问题：")
    for p in problems[:10]:
        print("     -", p)
    raise SystemExit(1)
print("   校验通过：无悬空/残留 symlink，每个 skill 都有 SKILL.md")
PY
