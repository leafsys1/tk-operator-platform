"""「agent 自动改代码」这一步（M3 余项 / 方案 §13-A5、§4.6 R1）。

自动优化闭环里，原本手工做的「按优化假设改算子代码」交给 agent：

1. **起一轮**：后端重拼固定层 prompt（任务规格 + 本轮 Loop 假设/判定 + **参考轮次的结构化建议** +
   硬件画像 + 相关 skill 提示），在**项目工作区**里起一次 dsh agent job（`kind=优化`、`loop_id` 关联）。
   cwd = 工作区，所以 dsh 按 projectRoot 能找到 `.dsh/skills`。
2. **落证据**：跑完回调里算**改后代码快照**、与改前对比、出逐文件 unified diff 与 agent 结构化摘要，
   写进 `loops.optimize_json`。**代码没变**就明确记 `changed=false`。
3. **可回退**：证据里带改动前内容，`revert()` 能把工作区还原到改前（把「不予采纳」变成一键）。

**边界（方案 §13-A5「agent 提议、工具判定」）**：本模块**只改代码**，不跑 build/test/benchmark ——
真机执行仍由控制台按 `plan.command` 单独触发（四动作远端执行器是另一件事）。所以它不产生 `runs`，
产出的是「一个待验证的候选快照」。
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..models import (AgentJob, Conversation, Loop, LoopRunLink, Message, Project, Run, Savepoint,
                      Task)
from . import agent_runner, chat, exec_actions, loop_state, perf_diff, savepoints, workspace_lease
from .prose_relay import SPEC_SEP

logger = logging.getLogger(__name__)

#: 与 `exec_actions.snapshot` 同一套跳过目录（口径一致，别两处漂移）
SNAPSHOT_SKIP = {"runs", ".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
                 ".skills-src", ".opx-chat"}

#: 单个文件超过它就不留在证据里（只记哈希，不做行级 diff / 不提供原样回退）
MAX_DIFF_FILE_BYTES = 512 * 1024

#: 存进 `optimize_json` 的 diff 上限（超出截断，避免把库撑大）
MAX_DIFF_CHARS = 200_000

#: 运行中的优化：agent_jobs.id → 上下文（改前快照等），跑完回调取用
PENDING: dict[int, dict] = {}

#: 证据输入路径：参考实现（golden）/用例/判据/执行器不得被 agent 修改；改了就判证据失效。
#: 这是**证据完整性**边界，不是完整安全沙箱（allowed_paths 本身仍是事后比对 + 回滚）。
PROTECTED_PATHS = ("cases/", "criteria/", "tools/", "scripts/")


def protected_violations(op_name: str | None, changed: list[dict]) -> list[str]:
    """返回本次改动里触碰受保护证据路径的文件（空 = 没碰）。"""
    prefixes = list(PROTECTED_PATHS)
    if op_name:
        prefixes.append(f"ops/{op_name}/scripts/")
    return [
        str(item.get("path"))
        for item in changed
        if any(str(item.get("path", "")).startswith(prefix) for prefix in prefixes)
    ]


SYSTEM_PROMPT = """你是「op-agent 算子开发控制台」的自动优化 PM。任务：针对下面这一轮**优化 Loop**，使用 CANNBot 多角色工作流对算子 `{op}` 形成并实现一处**最小、可回退**的候选改动。

## CANNBot 工作流（务必遵守）
- 用户已经明确授权本轮多角色自动优化，不要再次提问。先完整读取 `AGENTS.md`、`.dsh/skills/ops-direct-invoke/SKILL.md`、`.dsh/skills/workflow-orchestrator/SKILL.md` 和与本算子匹配的性能 Skill。
- **必须使用 workflow 工具**编排以下串行角色；PM 自己只汇总，不直接改算子代码：
  1. `ops-direct-invoke-architect`：只读诊断。读取本提示里的历史 A/B、实际报告、源码、硬件画像和性能 Skill；复算 tiling/核数/尾块，给出一个有证据的候选及可证伪观测。禁止改代码。
  2. `ops-direct-invoke-developer`：只在 architect 方案通过证据门槛后实施一个最小候选；只改允许路径。禁止改测试、判据、执行器或接口。
  3. `ops-direct-invoke-verifier`：只读独立复核 diff、整数数学、尾块、路径边界、注释一致性和可回退性；必须明确输出 PASS 或 FAIL。不能用开发者自述代替检查。
- verifier 为 FAIL 时，把具体意见交回 developer 修复一次并重新验证；仍失败则恢复本轮改前内容并返回 `changed=false`。没有 verifier PASS，不得留下代码改动。
- workflow 中的角色调用必须同步等待（`run_in_background=false`，或显式 await/join 等价机制）；前一个角色没有返回完整产物时不得启动下一角色。禁止留下后台 subagent 后让 PM 提前结束。
- PM 只有在 architect、developer（若有改动）和 verifier 都已结束后才能输出最终分隔符与 JSON；“仍在运行/等待结果/稍后继续”不是完成状态。
- architect 若发现候选只是恢复既有基线、与已失败结论重复、无法保证安全边界，或者没有可证伪的收益机制，直接返回不改代码；不得为了“有结果”而猜一个参数。
- 候选必须遵守**单变量归因**：一次只验证一个性能机制。architect 要列出“保持不变的关键量”和“唯一变化量”；developer 不得顺手改变其他 tile / 核数 / buffer / 搬运变量。若假设是核间负载分配，必须保持单 tile 粒度、搬运 API、buffer 深度不变；若无法隔离变量则 `changed=false`。
- verifier 必须逐项核对单变量约束；发现候选同时改变两个以上可能影响性能的机制时判 FAIL，即使整数数学与路径范围都正确。
- 这个阶段的产物是“待真机 A/B 的候选”，不是已证明更快的结论。**不得要求必须先有该候选本身的 A/B 才允许实施**，否则会形成无法开始实验的死循环。只要现有证据支持收益机制、整数数学/尾块/边界安全可证、单变量可隔离、结果可由后续 test+benchmark 证伪，就应产生一个最小候选，并在 expected/risks 里明确说明收益尚待实测。
- 性能 Skill 的“逐 shape/核数扫描、实测校准”用来形成和验证假设；已经有与唯一变量相关的实测曲线、失败反例或源码定量证据时，不能仅因“精确参数还没有 A/B”就 `changed=false`。

## 代码与执行边界
- 在**当前工作目录**（项目工作区）里读写文件：算子在 `ops/{op}/`，用例矩阵 / 判据 / 执行器在 `cases/` `criteria/` `tools/`。
- **只改算子实现**（kernel / tiling / host 侧的 `.asc` `.h` 等）。**不要**改用例语义、判据档、执行器脚本，
  也**不要**改参考实现 `ops/{op}/scripts/golden.py`（改参考实现会让精度对比失效，改动会被自动回滚）。
- **不要**在本机跑编译/测试/基准，**不要**调用 `scripts/nm_test.sh` 或任何远端命令 —— 真机执行由控制台在改完之后单独触发。
- 改动要小而聚焦：能一句话说清、便于回退。若假设站不住或证据不足，**宁可不动代码**，在回复里说明理由。
- 禁止把泛化经验直接当成本机结论；所有数字必须用当前源码公式和本轮报告复算。

## 本轮 Loop
- 假设：{hypothesis}
- 范围：{scope}
- 判定条件：{judge}
{reference}
## 任务规格（节选）
- 算子：{op}
- 目标芯片：{chip}
- 数学定义：{math_def}
- 输入 / 输出：{inputs} / {outputs}
- 正确性：oracle={oracle}；容差={tolerance}；一致性={consistency}

## 输出格式（两段式，必须严格遵守）
最终回复必须简短：中文说明不超过 300 字，JSON 不超过 1500 字。详细角色过程留在前面的工具结果中，不要把整份角色报告嵌套进最终 JSON。pipeline.architect、pipeline.developer、pipeline.verifier 必须各为不超过 100 字的字符串摘要，verifier 摘要保留独立复核结论和检查范围。优先保证分隔符之后的 JSON 完整闭合，不得在 JSON 后继续输出。
第一段：**给用户看的中文说明**（改了什么、为什么、预期观测）—— 会被逐字实时显示，不要出现 JSON / 代码围栏。
第二段：说明写完后，**另起一行单独输出分隔符** `{sep}`，**紧接着另起一行**输出一个 JSON 对象（不要围栏、是最后内容）：
{{"summary": "一句话改动摘要", "changed_files": ["相对路径", "..."], "hypothesis_check": "改动如何验证假设", "expected": "预期观测", "risks": "风险/约束", "changed": true, "verification": "pass", "pipeline": {{"architect": "诊断结论", "developer": "实施摘要", "verifier": "独立复核摘要"}}}}
若没有改任何文件，则 `"changed": false`、`changed_files` 为空数组、`verification` 填 `"pass"`（确认不改是正确结论），并在 `summary` 里说明为什么不动。若 verifier 未通过则 `verification` 必须为 `"fail"`，且工作区必须恢复到改前。
**verifier 证据是硬要求**：只要有代码改动，`pipeline.verifier` 必须是 verifier 角色给出的非空独立复核摘要（写清核对过 diff/整数数学/尾块/路径边界并给出 PASS）；`verification=pass` 但没有 `pipeline.verifier` 证据的候选会被控制台判无效并自动回退。"""


# ---------------------------------------------------------------- 快照与 diff


@dataclass
class Capture:
    """工作区代码快照：总哈希（与 runs.code_snapshot_sha256 同口径）+ 逐文件内容/哈希。"""

    sha256: str = ""
    files: dict[str, dict] = field(default_factory=dict)  # rel → {sha256, size, content(bytes|None)}


def capture(workspace: Path) -> Capture:
    """抓一次工作区快照（跳过运行时大件；单文件超限只记哈希、不留内容）。

    符号链接不跟随读取目标内容：记录链接目标 + 是否指向工作区外（`external`）。
    跟随读取会让「在允许路径里放一个指向外部的链接再写穿」只表现为允许路径内的文件变化；
    这里显式把链接本身和它的外指属性带进证据，由 `_finish` 的越界门禁拒绝。
    """
    snap = exec_actions.snapshot(workspace)
    files: dict[str, dict] = {}
    root_resolved = workspace.resolve()
    for path in workspace.rglob("*"):
        rel = path.relative_to(workspace)
        if any(part in SNAPSHOT_SKIP for part in rel.parts):
            continue
        if path.is_symlink():
            try:
                link = os.readlink(path)
            except OSError:
                continue
            try:
                external = not path.resolve().is_relative_to(root_resolved)
            except OSError:
                external = True
            files[rel.as_posix()] = {
                "sha256": hashlib.sha256(link.encode("utf-8", "surrogateescape")).hexdigest(),
                "size": 0,
                "content": None,
                "symlink": link,
                "external": external,
            }
            continue
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        files[rel.as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "content": data if len(data) <= MAX_DIFF_FILE_BYTES else None,
        }
    return Capture(sha256=snap.sha256, files=files)


def _text(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def diff_captures(before: Capture, after: Capture) -> tuple[list[dict], str]:
    """前后快照的逐文件变化 + unified diff 文本。返回 (changed_files, diff_text)。"""
    changed: list[dict] = []
    chunks: list[str] = []
    for p in sorted(set(before.files) | set(after.files)):
        b, a = before.files.get(p), after.files.get(p)
        if b is not None and a is not None and b["sha256"] == a["sha256"]:
            continue
        status = "added" if b is None else "removed" if a is None else "modified"
        if (b and b.get("symlink")) or (a and a.get("symlink")):
            # 符号链接单独标注（不跟随读取目标，避免把「写穿外部文件」伪装成普通改动）
            old_link = (b or {}).get("symlink")
            new_link = (a or {}).get("symlink")
            if status == "added":
                chunks.append(f"（新增符号链接：{p} -> {new_link}）\n")
            elif status == "removed":
                chunks.append(f"（删除符号链接：{p} -> {old_link}）\n")
            else:
                chunks.append(f"（符号链接目标改动：{p} {old_link} -> {new_link}）\n")
            changed.append({"path": p, "status": status, "additions": 0, "deletions": 0,
                            "symlink": new_link if a is not None else None,
                            "external": bool((a or b).get("external"))})
            continue
        btext = _text(b["content"]) if b else ""
        atext = _text(a["content"]) if a else ""
        adds = dels = 0
        if btext is not None and atext is not None:
            bl = (btext or "").splitlines(keepends=True)
            al = (atext or "").splitlines(keepends=True)
            for line in difflib.unified_diff(bl, al,
                                             fromfile=f"a/{p}" if b is not None else "/dev/null",
                                             tofile=f"b/{p}" if a is not None else "/dev/null"):
                chunks.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    adds += 1
                elif line.startswith("-") and not line.startswith("---"):
                    dels += 1
        else:
            chunks.append(f"（二进制或超限文件改动：{p}）\n")
        changed.append({"path": p, "status": status, "additions": adds, "deletions": dels})
    diff = "".join(chunks)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n…（diff 超过 {MAX_DIFF_CHARS} 字符，已截断）\n"
    return changed, diff


def _revert_payload(before: Capture, changed: list[dict]) -> dict:
    """回退数据：每个变化文件改前是否存在、以及（能拿到时）改前内容/符号链接目标。"""
    out: dict[str, dict] = {}
    for c in changed:
        b = before.files.get(c["path"])
        if b is None:
            out[c["path"]] = {"existed": False}
        elif b.get("symlink") is not None:
            out[c["path"]] = {"existed": True, "symlink": b["symlink"],
                              "restorable": True}
        elif b["content"] is None:
            out[c["path"]] = {"existed": True, "restorable": False,
                              "note": "文件超过上限，未留内容，无法原样回退"}
        else:
            out[c["path"]] = {"existed": True, "restorable": True,
                              "content_b64": base64.b64encode(b["content"]).decode("ascii")}
    return out


def apply_revert(workspace: Path, revert_data: dict) -> list[str]:
    """按回退数据还原工作区；返回无法回退的路径（空 = 全部还原）。

    符号链接按「链接本身」还原（不写穿目标）；回退新建链接/文件时用 `is_symlink() or exists()`
    判断，避免断链（broken symlink）漏删。
    """
    failed: list[str] = []
    for rel, info in (revert_data or {}).items():
        target = workspace / rel
        if not info.get("existed"):
            try:
                if target.is_symlink() or target.is_file():
                    target.unlink()
            except OSError:
                failed.append(rel)
            continue
        if info.get("symlink") is not None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(str(info["symlink"]), target)
            except OSError:
                failed.append(rel)
            continue
        b64 = info.get("content_b64")
        if not b64:
            failed.append(rel)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            target.write_bytes(base64.b64decode(b64))
        except (OSError, ValueError):
            failed.append(rel)
    return failed


# ---------------------------------------------------------------- prompt 与参考轮次


def _spec_fields(db: Session, task: Task) -> dict:
    sv = None
    if task.current_spec_version_id is not None:
        from ..models import SpecVersion
        sv = db.get(SpecVersion, task.current_spec_version_id)
    if sv is None:
        return {}
    try:
        data = json.loads(sv.spec_json or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _loop_runs(db: Session, loop_id: int) -> list[Run]:
    """轮次可见 run：直接归属 ∪ link 复用（Step 1b；不改写 runs.loop_id）。"""
    return list(db.scalars(select(Run).where(
        or_(Run.loop_id == loop_id,
            Run.id.in_(select(LoopRunLink.run_id).where(LoopRunLink.loop_id == loop_id)))
    ).order_by(Run.id)))


def reference_text(db: Session, loop: Loop, reference_loop_id: int | None = None) -> tuple[str | None, int | None]:
    """取「参考轮次」的对比结论与结构化建议，拼成 prompt 里的一段文本。

    默认取本任务下**最近的、已凑齐两个代码快照**的 perf Loop（排除自己）—— 通常就是上一轮。
    """
    ref = db.get(Loop, reference_loop_id) if reference_loop_id is not None else None
    if ref is None:
        cands = list(db.scalars(select(Loop)
                                .where(Loop.task_id == loop.task_id, Loop.kind == "perf")
                                .order_by(Loop.id.desc())))
        for cand in cands:
            if cand.id == loop.id:
                continue
            rs = _loop_runs(db, cand.id)
            if len({r.code_snapshot_sha256 for r in rs if r.code_snapshot_sha256}) >= 2:
                ref = cand
                break
    if ref is None:
        return None, None
    runs = _loop_runs(db, ref.id)
    report = perf_diff.evaluate_loop(ref, runs)
    lines = [f"[参考：Loop #{ref.id} 的优化前后对比结论]", f"- 结论：{report.get('conclusion')}（门禁 {report.get('verdict')}）"]
    for c in report.get("cases") or []:
        # 缺有效测量时 perf_diff 给 delta_pct=None（不再伪造成 -100%），这里如实写成「缺有效测量」。
        delta = c.get("delta_pct")
        delta_text = f"{delta:+.1f}%" if isinstance(delta, (int, float)) else "缺有效测量"
        before_us = float(c.get("before_us") or 0)
        after_us = float(c.get("after_us") or 0)
        lines.append(f"- 用例 {c['case_id']}：{before_us:.3f}→{after_us:.3f}us（{delta_text}）")
    for r in report.get("recommendations") or []:
        if r.get("status") != "proposed":
            continue
        lines.append(f"- 待验证建议 {r['id']}：{r.get('title', '')}")
        lines.append(f"  瓶颈假设：{r.get('bottleneck_hypothesis')}")
        lines.append(f"  建议改动：{r.get('proposed_change')}")
        lines.append(f"  预期观测：{r.get('expected_observation')}")
        if r.get("skill"):
            lines.append(f"  参考 skill：{r['skill']}")
    return "\n".join(lines) + "\n", ref.id


def current_evidence_text(db: Session, loop: Loop) -> str | None:
    """把当前 Loop 已验证过的候选注入下一次优化，防止重复撞同一失败方向。"""
    runs = _loop_runs(db, loop.id)
    if len({r.code_snapshot_sha256 for r in runs if r.code_snapshot_sha256}) < 2:
        return None
    report = perf_diff.evaluate_loop(loop, runs)
    sides = report.get("sides") if isinstance(report.get("sides"), dict) else {}
    before = sides.get("before") if isinstance(sides.get("before"), dict) else {}
    after = sides.get("after") if isinstance(sides.get("after"), dict) else {}
    lines = [
        f"[当前 Loop #{loop.id} 已验证候选：不得忽略、不得重复]",
        f"- 结论：{report.get('conclusion')}（门禁 {report.get('verdict')}）",
        f"- 基线 runs：{before.get('run_ids') or []}；sha={before.get('code_snapshot_sha256') or 'unknown'}",
        f"- 最近候选 runs：{after.get('run_ids') or []}；sha={after.get('code_snapshot_sha256') or 'unknown'}",
    ]
    for case in report.get("cases") or []:
        delta = case.get("delta_pct")
        delta_text = f"{float(delta):+.1f}%" if isinstance(delta, (int, float)) else "缺有效测量"
        lines.append(
            f"- 用例 {case.get('case_id')}：{float(case.get('before_us') or 0):.3f}→"
            f"{float(case.get('after_us') or 0):.3f}us（{delta_text}；"
            f"{case.get('verdict') or 'unknown'}；不确定性 ±{float(case.get('uncertainty_pct') or 0):.1f}%）"
        )
    previous = loop.optimize
    if isinstance(previous, dict):
        summary = str(previous.get("agent_summary") or "").strip()
        if summary:
            lines.append(f"- 上次 Agent 改动：{summary}")
    lines.append("- 新候选必须与上述已验证候选有实质差异；仅恢复基线不算新的优化。")
    return "\n".join(lines) + "\n"


#: 优化目标类型 → 人看的标签（与 `services/optimization.py` 的枚举同一套）
OBJECTIVE_TYPE_LABELS = {"perf": "性能", "precision": "精度", "hardware": "硬件适配"}


def _load_chip_profile(path: Path) -> dict | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _chip_profile_for(workspace: Path, chip: str | None) -> tuple[Path, dict] | None:
    """按目标芯片型号选 `ops/chips/*.yaml`：`models` 命中优先，找不到退回第一份（P1）。

    「别喂错画像」：`task.target_chip`（如 Ascend910B1）只有出现在画像的 `models` 里才算命中；
    没命中时退回第一份但调用方仍会带上型号说明，避免 agent 把别的型号参数当定论。
    """
    chips = sorted((workspace / "ops" / "chips").glob("*.yaml"))
    if not chips:
        return None
    want = (chip or "").strip().lower()
    first: tuple[Path, dict] | None = None
    for path in chips:
        data = _load_chip_profile(path)
        if data is None:
            continue
        if first is None:
            first = (path, data)
        models = [str(m).strip().lower() for m in (data.get("models") or []) if m]
        if want and want in models:
            return path, data
    return first


def hardware_brief(workspace: Path, chip: str | None = None) -> str:
    """把硬件画像里的**关键事实**压成几行，直接放进优化 prompt（A11）。

    只内联「影响决策的几条」，完整文件仍让 agent 自己读；置信度一起带上，low/medium 不得当定论。
    `chip` = `task.target_chip`：按画像里的 `models` 选文件，找不到退回第一份（别喂错画像）。
    """
    chips = sorted((workspace / "ops" / "chips").glob("*.yaml"))
    if not chips:
        return ""
    selected = _chip_profile_for(workspace, chip)
    if selected is None:
        return f"硬件画像：{chips[0].as_posix()}（解析失败，请自行阅读）"
    path, data = selected
    facts = {f.get("key"): f for f in (data.get("facts") or []) if isinstance(f, dict)}
    parts: list[str] = []
    for key, label in (("ub_size_bytes", "UB"), ("l1_size_bytes", "L1"),
                       ("l0c_size_bytes", "L0C"), ("bt_size_bytes", "BT")):
        f = facts.get(key)
        if f:
            parts.append(f"{label}={f.get('value')}B({f.get('confidence')})")
    mv = data.get("model_variable") or {}
    for key, label in (("vector_core_count", "向量核"), ("frequency_mhz", "频率MHz"),
                       ("hbm_bandwidth_gbps", "HBM GB/s")):
        v = mv.get(key)
        if isinstance(v, dict):
            parts.append(f"{label}={v.get('value')}({v.get('confidence')})")
    align = (data.get("alignment") or {}).get("datacopy_min_align_bytes")
    if align:
        parts.append(f"DataCopy对齐={align}B")
    vw = (data.get("vector_width") or {}).get("fp32_elements_per_repeat")
    if vw:
        parts.append(f"fp32 repeat={vw}")
    rel = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path)
    brief = "；".join(parts)
    return (f"硬件画像：{rel}（{brief}）。**以运行时 PlatformAscendC/aclrtGetDeviceInfo 为准，"
            f"不得把常量硬编码进算子；confidence=low/medium 的不得当定论。**")


def _objective_prompt(task: Task, loop: Loop) -> str | None:
    """把「本轮优化目标与约束」（`loops.objective_json`）拼成 prompt 的一段（P1 §8）。"""
    objective = loop.objective
    if not isinstance(objective, dict):
        return None
    otype = str(objective.get("type") or "").strip()
    dimension = str(objective.get("target_dimension") or "all").strip() or "all"
    cases = [str(c) for c in (objective.get("target_cases") or []) if str(c).strip()]
    constraints = objective.get("constraints") if isinstance(objective.get("constraints"), dict) else {}
    budget = objective.get("budget") if isinstance(objective.get("budget"), dict) else {}
    snapshot = (objective.get("targets_snapshot")
                if isinstance(objective.get("targets_snapshot"), dict) else {})

    lines = ["## 本轮目标与约束",
             f"- 目标类型：{OBJECTIVE_TYPE_LABELS.get(otype, otype or '未指定')}"
             f"（{otype or '未指定'}）"]
    focus = str(objective.get("focus") or "").strip()
    if dimension == "text" and focus:
        # 思路池/文字目标轮：目标是「需要优化的点」的文字描述，不指定目标用例。
        lines.append("- **本轮需要优化的点（文字描述，按它选择要优化的 shape/路径）**：")
        for chunk in focus.splitlines():
            lines.append(f"  {chunk}")
        lines.append("- 目标用例：不限定（由你根据上面的文字和有证据的报告选择；"
                     "在给用户的说明里写清选择依据和预期观测）")
    else:
        lines.append(f"- 目标用例：{'、'.join(cases) if cases else '全部'}"
                     f"（维度：{'指定用例' if dimension == 'cases' else '全部用例'}）")
    constraint_texts: list[str] = []
    if constraints.get("precision_non_regression"):
        constraint_texts.append("精度不退化")
    if constraints.get("perf_non_regression"):
        constraint_texts.append("性能不退化")
    allowed = [str(p) for p in (constraints.get("allowed_paths") or []) if str(p).strip()]
    if allowed:
        constraint_texts.append("允许改动：" + "、".join(allowed))
    if constraint_texts:
        lines.append("- 约束：" + "；".join(constraint_texts))
    if budget:
        lines.append("- 预算（控制台硬限制真实执行次数）：判断轮数 {} / 验证轮数 {} / 每动作执行次数 {}".format(
            budget.get("diagnose_rounds", "?"), budget.get("verify_rounds", "?"),
            budget.get("runs_per_action", "?")))
    snapshot_texts = []
    if snapshot.get("benchmark_run_id") is not None:
        snapshot_texts.append(f"benchmark run #{snapshot['benchmark_run_id']}")
    if snapshot.get("test_run_id") is not None:
        snapshot_texts.append(f"test run #{snapshot['test_run_id']}")
    if snapshot.get("primary_case"):
        snapshot_texts.append(f"主用例 {snapshot['primary_case']}")
    if snapshot.get("suite_ref"):
        snapshot_texts.append(f"用例集 {snapshot['suite_ref']}")
    if snapshot_texts:
        lines.append("- 目标快照：" + "；".join(snapshot_texts))
    user_text = str(objective.get("user_text") or "").strip()
    if user_text:
        lines.append("- 用户补充的思路/约束：" + user_text)
    if otype == "hardware":
        chip = task.target_chip or "未指定"
        lines.append(f"- 本轮是**硬件适配**优化：以目标芯片 {chip} 的硬件画像为准"
                     "（运行时通过 PlatformAscendC / aclrtGetDeviceInfo 确认参数，不得硬编码常量）。")
    return "\n".join(lines)


def build_prompt(*, db: Session, task: Task, loop: Loop, workspace: Path,
                 reference: str | None, extra: str | None = None) -> str:
    op = task.op_name or task.title
    fields = _spec_fields(db, task)

    def g(key: str) -> str:
        v = fields.get(key)
        return str(v) if v not in (None, "", []) else "UNKNOWN"

    chips_dir = workspace / "ops" / "chips"
    impl_files: list[str] = []
    op_dir = workspace / "ops" / op
    if op_dir.is_dir():
        impl_files = sorted(p.relative_to(workspace).as_posix() for p in op_dir.rglob("*") if p.is_file())
    hints = []
    brief = hardware_brief(workspace, task.target_chip)
    if brief:
        hints.append(brief)
    elif chips_dir.is_dir():
        hints.append("硬件画像：" + "、".join(sorted(p.name for p in chips_dir.glob("*.yaml"))))
    if impl_files:
        hints.append("当前实现文件：\n  - " + "\n  - ".join(impl_files[:40]))

    body = SYSTEM_PROMPT.format(
        op=op, chip=task.target_chip or "910B1", sep=SPEC_SEP,
        hypothesis=loop.hypothesis or "（未记假设）",
        scope=loop.scope or "（未记范围）",
        judge=loop.judge_condition or "（未记判定条件）",
        reference=(reference + "\n" if reference else ""),
        math_def=g("math_def"), inputs=g("inputs"), outputs=g("outputs"),
        oracle=g("oracle"), tolerance=g("tolerance"), consistency=g("consistency"),
    )
    objective = _objective_prompt(task, loop)
    if objective:
        # 放在「输出格式」之前：目标/约束属于任务上下文，别被格式说明挤到最后
        marker = "## 输出格式"
        body = (body.replace(marker, objective + "\n\n" + marker, 1)
                if marker in body else body + "\n\n" + objective)
    current_evidence = current_evidence_text(db, loop)
    if current_evidence:
        body += "\n\n## 当前轮次的已验证反馈\n" + current_evidence
    if hints:
        body += "\n\n## 工作区提示\n- " + "\n- ".join(hints)
    if extra:
        body += f"\n\n## 用户补充\n{extra.strip()}"
    return body


# ---------------------------------------------------------------- 起一轮 / 回调


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start(job_id: int, prompt: str, *, loop_id: int, workspace: Path, dsh_home: Path,
          reference_loop_id: int | None = None,
          conversation_id: int | None = None,
          session_id: str | None = None) -> agent_runner.JobState:
    """抓改前快照 → 起 dsh agent job（工作区 cwd，挂 skills）→ 跑完回调落证据。

    `conversation_id`：对话化自动优化时关联的会话；跑完由 `_finish` 往会话里回写助手消息。
    `session_id`：同一会话续聊时传入 conv.dsh_session_id（首次为空 → dsh 新建会话并回填）。
    """
    before = capture(workspace)
    with SessionLocal() as db:
        loop = db.get(Loop, loop_id)
        baseline = db.scalar(select(Savepoint).where(
            Savepoint.loop_id == loop_id, Savepoint.kind == "baseline"
        ).order_by(Savepoint.id.desc()).limit(1)) if loop is not None else None
        op_name = savepoints.evidence_op(baseline)
    scoped = op_name is not None
    before_code = exec_actions.operator_snapshot(workspace, op_name).sha256 if scoped else before.sha256
    PENDING[job_id] = {"loop_id": loop_id, "workspace": str(workspace), "before": before,
                       "before_code_sha256": before_code, "evidence_scope": "operator-v1" if scoped else None,
                       "op_name": op_name,
                       "reference_loop_id": reference_loop_id, "started_at": _now(),
                       "conversation_id": conversation_id}
    try:
        return agent_runner.start(
            job_id, prompt, cwd=workspace, dsh_home=dsh_home, session_id=session_id,
            on_done=lambda state: _finish(job_id, state),
            stream=True, prose_of=chat.prose_only,
        )
    except BaseException:
        # 起不来（没有 running loop / dsh 不在 / 会话目录不可写…）：PENDING 只在 _finish 里 pop，
        # 这里必须兜底清掉，否则该 Loop 之后永远被判成「已有正在跑的自动优化」。
        PENDING.pop(job_id, None)
        raise


def _finish(job_id: int, state) -> None:
    """dsh 跑完：算改后快照 + diff + 摘要，写 `loops.optimize_json`（用独立 session）。"""
    ctx = PENDING.pop(job_id, None)
    if ctx is None:
        return
    workspace = Path(ctx["workspace"])
    before: Capture = ctx["before"]
    ok = state.exit_code == 0 and not state.error

    reply, parsed = "", None
    parse_error = None
    if ok:
        try:
            reply, parsed = chat.parse_output(state.final)
        except chat.ChatError as exc:
            reply = chat.prose_only(state.final or "")
            parse_error = f"优化结果的结构化输出解析失败（可能不完整）：{exc}"

    after = capture(workspace)
    changed, diff = diff_captures(before, after)
    summary = ""
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
    if not summary:
        summary = reply

    verification = str(parsed.get("verification") or "").strip().lower() if isinstance(parsed, dict) else ""
    pipeline = parsed.get("pipeline") if isinstance(parsed, dict) and isinstance(parsed.get("pipeline"), dict) else {}
    verifier_note = str(pipeline.get("verifier") or "").strip()
    allowed: list[str] = []
    op_name: str | None = None
    with SessionLocal() as guard_db:
        guard_loop = guard_db.get(Loop, ctx["loop_id"])
        objective = guard_loop.objective if isinstance(guard_loop.objective, dict) else None
        constraints = objective.get("constraints") if objective is not None else None
        if isinstance(constraints, dict):
            allowed = [
                str(path).strip().rstrip("/") + "/"
                for path in (constraints.get("allowed_paths") or [])
                if str(path).strip()
            ]
        if guard_loop is not None:
            guard_task = guard_db.get(Task, guard_loop.task_id)
            op_name = guard_task.op_name if guard_task is not None else None
    # 符号链接不在允许路径内跟随读取：外指链接等于绕开 allowed_paths 的写穿风险，一律拒绝。
    allowed_symlinks = [
        rel for rel, info in after.files.items()
        if info.get("external") and (not allowed or any(rel.startswith(prefix) for prefix in allowed))
    ]
    root_resolved = workspace.resolve()

    def _outside(rel: str) -> bool:
        if allowed and not any(rel.startswith(prefix) for prefix in allowed):
            return True
        try:
            return not (workspace / rel).resolve().is_relative_to(root_resolved)
        except OSError:
            return True

    outside = [item["path"] for item in changed if _outside(item["path"])]
    protected = protected_violations(op_name, changed)
    guard_error = None
    scoped = ctx.get("evidence_scope") == "operator-v1"
    attempted_after_sha256 = (exec_actions.operator_snapshot(workspace, ctx["op_name"]).sha256
                              if scoped else after.sha256)
    if parse_error:
        guard_error = parse_error + "；已自动回退候选，请重新生成完整结果"
    elif verification != "pass":
        guard_error = "verifier 未明确 PASS，已自动回退候选"
    elif changed and not verifier_note:
        guard_error = "缺少 verifier 独立复核证据（pipeline.verifier 为空），已自动回退候选"
    elif allowed_symlinks:
        guard_error = ("允许改动路径下存在指向工作区外的符号链接（可能是本次新建）："
                       + "、".join(allowed_symlinks) + "；已自动回退")
    elif protected:
        guard_error = ("E_PROTECTED_PATH: 改动触碰受保护证据路径（参考实现/用例/判据/执行器），"
                       "已自动回退：" + "、".join(protected))
    elif changed and not allowed:
        guard_error = "本轮缺少 allowed_paths，无法校验改动边界，已按安全默认回退候选"
    elif outside:
        guard_error = "改动越过允许路径，已自动回退：" + "、".join(outside)
    if guard_error:
        failed = apply_revert(workspace, _revert_payload(before, changed))
        if failed:
            guard_error += "；以下文件无法自动恢复：" + "、".join(failed)
        after = capture(workspace)

    ev = {
        "schema": "opx-optimize/1",
        "loop_id": ctx["loop_id"],
        "job_id": job_id,
        "status": "done" if ok and not guard_error else "failed",
        "error": "；".join(part for part in (state.error, guard_error) if part) or None,
        "started_at": ctx["started_at"],
        "ended_at": _now(),
        "session_id": getattr(state, "session_id", None),
        "evidence_scope": ctx.get("evidence_scope"),
        "op_name": ctx.get("op_name"),
        "before_sha256": ctx.get("before_code_sha256", before.sha256),
        "after_sha256": (exec_actions.operator_snapshot(workspace, ctx["op_name"]).sha256
                         if scoped else after.sha256),
        "before_workspace_sha256": before.sha256,
        "after_workspace_sha256": after.sha256,
        "attempted_after_sha256": attempted_after_sha256,
        "changed": bool(changed) and not bool(guard_error),
        "changed_files": changed,
        "diff": diff or None,
        "agent_summary": summary,
        "agent_output": parsed,
        "raw_output": state.final if parse_error else None,
        "verification": verification or None,
        "pipeline": pipeline,
        "guard_reverted": bool(guard_error),
        "guard_reason": guard_error,
        "external_symlinks": allowed_symlinks,
        "protected_paths": protected,
        "reference_loop_id": ctx.get("reference_loop_id"),
        "revert": _revert_payload(before, changed),
        "reverted": bool(guard_error),
        "reverted_clean": bool(guard_error) and after.sha256 == before.sha256,
    }
    with SessionLocal() as db:
        loop = db.get(Loop, ctx["loop_id"])
        if loop is not None:
            loop.optimize_json = json.dumps(ev, ensure_ascii=False)
            # Step 1：改码结束推进状态机（有候选 → awaiting_verification，否则 failed），
            # 并释放工作区租约。与证据同事务，避免“证据写了但状态还停在 running”。
            # 用户已取消的轮次：保持 cancelled（取消不能让收尾把状态改回 failed/verification）。
            if loop_state.phase_of(loop) == "cancelled":
                loop_state.transition(db, loop, "cancelled", event="agent_cancelled",
                                      force=True, active_job_id=None,
                                      error=loop.phase_error or "用户取消", commit=False)
            else:
                loop_state.finish_agent(db, loop.id, ev, commit=False)
            workspace_lease.release(db, task_id=loop.task_id, kind="agent",
                                    job_id=job_id, commit=False)
            db.commit()
    # 对话化自动优化：把结果回写到关联会话（失败不影响 loop 证据）
    _write_conversation_result(job_id, ctx.get("conversation_id"), ev, state)


def _conversation_result_text(ev: dict) -> str:
    """把 loop 证据压成对话里可读的助手消息（摘要 + 改动 + 回退/verifier + 下一步）。"""
    parts = [str(ev.get("agent_summary") or "").strip() or "（本轮自动优化没有可显示摘要）"]
    changed = ev.get("changed_files") or []
    if ev.get("changed") and changed:
        parts.append("改动文件：" + "、".join(
            f"{c.get('path')}（{c.get('status')}）" for c in changed[:20]))
    if ev.get("guard_reverted"):
        parts.append("候选未保留：" + str(
            ev.get("guard_reason") or ev.get("error") or "verifier 未 PASS / 改动越界，已自动回退"))
    elif ev.get("reverted"):
        parts.append("候选已回退到本轮改前")
    if ev.get("verification"):
        parts.append(f"verifier 结论：{ev['verification']}")
    if ev.get("status") != "done" and ev.get("error") and not ev.get("guard_reverted"):
        parts.append(f"本轮失败：{ev['error']}")
    if ev.get("status") == "done" and ev.get("changed") and not ev.get("reverted"):
        parts.append("下一步：到「性能与优化」轮次卡跑候选验证（test + benchmark），再由人工采纳/否决。")
    else:
        parts.append("下一步：本轮没有可验证的候选。先核对失败或未改码原因，再重新运行自动优化；保留有效候选后才可验证，也可否决本轮。")
    body = "\n".join(f"- {part}" for part in parts[1:])
    return parts[0] + ("\n\n" + body if body else "")


def _write_conversation_result(job_id: int, conversation_id: int | None, ev: dict, state) -> None:
    """自动优化跑完：往关联会话落一条助手消息 + 回填 session / 计数（对话化自动优化收尾）。"""
    if not conversation_id:
        return
    try:
        with SessionLocal() as db:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                return
            usage = getattr(state, "usage", None) or {}
            tokens = sum(int(usage.get(key) or 0) for key in
                         ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"))
            conv.dsh_session_id = getattr(state, "session_id", None) or conv.dsh_session_id
            conv.message_count = (conv.message_count or 0) + 2
            conv.tokens = (conv.tokens or 0) + tokens
            conv.last_active_at = datetime.now(timezone.utc)
            db.add(Message(conversation_id=conv.id, role="assistant",
                           text=_conversation_result_text(ev), job_id=job_id,
                           created_at=datetime.now(timezone.utc)))
            db.commit()
    except Exception:  # noqa: BLE001 —— 会话回写失败不能影响 loop 证据
        logger.exception("自动优化结果回写会话失败（job=%s, conv=%s）", job_id, conversation_id)


def revert(db: Session, loop: Loop, workspace: Path) -> dict:
    """把工作区还原到「这一轮改前」。返回更新后的证据。"""
    ev = loop.optimize
    if not ev:
        raise ValueError("这一轮还没有自动优化记录，无法回退")
    failed = apply_revert(workspace, ev.get("revert") or {})
    ev["reverted"] = True
    ev["revert_failed"] = failed
    ev["reverted_at"] = _now()
    # 回退后再抓一次快照：确认工作区回到了改前哈希（拿不到就是有东西没还原）
    now = capture(workspace)
    ev["current_sha256"] = (exec_actions.operator_snapshot(workspace, ev.get("op_name")).sha256
                            if ev.get("evidence_scope") == "operator-v1" else now.sha256)
    ev["reverted_clean"] = (now.sha256 == ev.get("before_workspace_sha256", ev.get("before_sha256"))) and not failed
    loop.optimize_json = json.dumps(ev, ensure_ascii=False)
    db.commit()
    return ev


def evidence(loop: Loop) -> dict:
    """对外读：没有记录时给一个明确的空态（而不是 404，前端好处理）。"""
    if loop.optimize:
        return loop.optimize
    return {"schema": "opx-optimize/1", "loop_id": loop.id, "status": "none",
            "changed": False, "changed_files": [], "diff": None, "agent_summary": None}
