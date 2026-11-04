"""任务上下文空间（方案 v1.1 §6）：把「对话澄清出的规格」落成开发阶段要吃的文件。

位置：`<project.workspace_path>/tasks/<slug>/`（不复制真相源 —— `cases/` `criteria/` `ops/`
仍在工作区既定位置，这里只**指向**）。

写盘顺序（§6「先库后盘」）：调用方**先**落 `spec_versions`，再把 `spec_version_id` + 规格哈希
回填进文件内容，最后写盘并回填每个文件的 sha256。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models import Task


def slugify(op_name: str | None) -> str | None:
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", (op_name or "").strip()).strip("-").lower()
    return s or None


def _marker_owner(base: Path) -> str | None:
    """读目录里的 `.opx-task` 归属标记（缺失/读不了返回 None，不能因为权限问题 500）。"""
    try:
        return (base / ".opx-task").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def space_dir(workspace: Path, task: Task) -> Path:
    """任务空间目录的**实时推导**；`tasks/<slug>`，冲突（属于别的任务）时退化成 `tasks/<slug>-<id>`。

    注意：目录名依赖可变的 `task.op_name`，只适合「还没写过盘」的推导。已有目录的读取/删除
    请用 `find_space_dir()`（按 `.opx-task` 归属标记兜底，改名/同 slug 任务删除后仍找得到）。
    """
    slug = slugify(task.op_name) or f"task-{task.id}"
    base = workspace / "tasks" / slug
    owner = _marker_owner(base)
    if owner not in (None, "", str(task.id)):
        base = workspace / "tasks" / f"{slug}-{task.id}"
    return base


def _scan_owner(workspace: Path, task_id: int) -> Path | None:
    """扫 `workspace/tasks/*/.opx-task`，找标记等于 `task_id` 的目录（稳定归属的最后兜底）。"""
    tasks_dir = workspace / "tasks"
    try:
        candidates = sorted(p for p in tasks_dir.iterdir() if p.is_dir())
    except OSError:
        return None
    for base in candidates:
        if _marker_owner(base) == str(task_id):
            return base
    return None


def find_space_dir(workspace: Path, task: Task) -> Path:
    """找「确认属于本任务」的上下文空间目录（优先按当前 op_name，找不到再按标记扫描）。

    为什么不能只用 `space_dir()`：目录名按 `op_name` 现算 —— 改名后实时推导指向新目录（不存在），
    同 slug 的另一个任务被删后推导又会从 `tasks/<slug>-<id>` 漂回 `tasks/<slug>`。
    已有目录一律以 `.opx-task` 标记（生成时写入 task.id）为准；扫描不到才回落到实时推导目录
    （兼容手工创建/标记缺失的老目录，读侧仍可展示，删侧由 `task_deletion` 另做标记校验）。
    """
    base = space_dir(workspace, task)
    if _marker_owner(base) == str(task.id):
        return base
    return _scan_owner(workspace, task.id) or base


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _yaml_scalar(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_task_md(task: Task, fields: dict, spec_version: int, spec_sha: str,
                    criteria_ref: str | None, at: str) -> str:
    def g(key: str) -> str:
        v = fields.get(key)
        return str(v) if v not in (None, "", []) else "UNKNOWN"

    return (
        f"# 任务：{task.title}\n\n"
        f"> 由 op-agent「任务开始」页生成 · {at} · spec v{spec_version}"
        f"（sha256 {spec_sha[:12]}）· 判据档 {criteria_ref or '（项目默认）'}\n\n"
        "## 目标\n\n"
        f"- 算子：{g('op_name')}\n"
        f"- 目标芯片：{g('target_chip')}\n\n"
        "## 需求（用户原话）\n\n"
        f"{(task.raw_input or '（空）').strip()}\n\n"
        "## 语义与接口\n\n"
        f"- 数学定义：{g('math_def')}\n"
        f"- 参数 / 属性：{g('params')}\n"
        f"- 输入：{g('inputs')}\n"
        f"- 输出：{g('outputs')}\n"
        f"- 支持范围：{g('support')}\n"
        f"- 不支持：{g('unsupported')}\n"
        f"- 特殊行为：{g('special_behavior')}\n\n"
        "## 正确性\n\n"
        f"- 参考实现：{g('oracle')}\n"
        f"- 容差：{g('tolerance')}\n"
        f"- 一致性：{g('consistency')}\n\n"
        "## 性能\n\n"
        f"- 基线：{g('performance_baseline')}\n"
        f"- 目标：{g('performance_target')}\n\n"
        "## 下一步\n\n"
        "按本规格落实现（build / test / benchmark / profile）。装载清单见 `context.yaml`。\n"
    )


AGENTS_MD = """# 本任务约定（继承项目级 AGENTS.md）

- 真相源：`cases/` `criteria/` `ops/` 在工作区既定位置，**不要**复制进本目录；本目录只指向它们。
- 规格：以库里的 `spec_versions` 为版本真相源；`spec/spec.md` 是人读渲染，**禁手改**（改需求请走新规格版本）。
- 判据：默认官方档；**改判据必须换版本号**（换档版本 = 换口径）。
- 实现路线：首版 Ascend C 直调（见 `ops/`）。
"""


def _render_context_yaml(criteria_ref: str | None) -> str:
    return (
        "# 上下文装载契约（方案 v1.1 §8）。键名固定，供后端每轮拼 prompt 用。\n"
        "always_load:\n"
        "  - TASK.md\n"
        "  - spec/spec.json              # 语义 / 输入输出 / 正确性三段\n"
        "by_stage:\n"
        "  design: [ops-design-paradigms, ascendc-tiling-design]\n"
        "  code: [repo-coding-rules]\n"
        "  verify: [ops-precision-standard]\n"
        "  perf: [ops-profiling, ascendc-performance-best-practices]\n"
        "by_action:\n"
        f"  test: [{_yaml_scalar(criteria_ref or 'ops-precision-standard@1')}, cases/]\n"
        "  benchmark: [baseline, cases/]\n"
        "  profile: [criteria]\n"
        "never_default:\n"
        "  - 全量会话历史\n"
        "  - 全部 40 个 skills\n"
        "  - 原始大产物（日志 / bin / jsonl）\n"
        "pointers:\n"
        "  cases: ../../cases/        # 相对工作区\n"
        "  criteria: ../../criteria/\n"
        "  ops: ../../ops/\n"
    )


def generate(*, workspace: Path, task: Task, fields: dict, questions: list,
             spec_version: int, spec_version_id: int | None, spec_sha: str,
             criteria_ref: str | None, chat_log: list[dict]) -> tuple[Path, list[dict]]:
    """写上下文空间，返回 (目录, 文件清单[{path,sha256,size_bytes}])。"""
    at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    base = space_dir(workspace, task)
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "references").mkdir(exist_ok=True)
    (base / "notes").mkdir(exist_ok=True)
    (base / ".opx-task").write_text(str(task.id), encoding="utf-8")

    unknowns = [q.get("text") for q in questions if isinstance(q, dict)]
    spec_obj = {
        "_meta": {"generated_by": "op-agent", "spec_version_id": spec_version_id,
                  "spec_version": spec_version, "sha256": spec_sha, "created_at": at,
                  "unknowns": unknowns},
        **fields,
    }

    files: dict[str, str] = {
        "TASK.md": _render_task_md(task, fields, spec_version, spec_sha, criteria_ref, at),
        "AGENTS.md": AGENTS_MD,
        "context.yaml": _render_context_yaml(criteria_ref),
        "spec/spec.json": json.dumps(spec_obj, ensure_ascii=False, indent=2) + "\n",
        "spec/spec.md": _render_spec_md(task, fields, spec_version, spec_sha, criteria_ref, at, unknowns),
        "references/README.md": "# 参考实现\n\n`golden.py` 等参考实现放这里（实现阶段生成）。\n",
        "notes/clarifications.md": _render_notes(chat_log, at),
    }

    out: list[dict] = []
    for rel, content in files.items():
        data = content.encode("utf-8")
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        out.append({"path": rel, "sha256": _sha256(data), "size_bytes": len(data)})
    return base, out


def _render_spec_md(task: Task, fields: dict, version: int, spec_sha: str,
                    criteria_ref: str | None, at: str, unknowns: list) -> str:
    lines = [
        f"# 算子需求规格 · {fields.get('op_name') or task.op_name or task.title}",
        "",
        f"> spec v{version} · sha256 {spec_sha[:12]} · {at} · 判据档 {criteria_ref or '（项目默认）'}",
        "> 本文件由 `spec.json` 渲染，禁手改。",
        "",
        "| 字段 | 值 |",
        "|---|---|",
    ]
    for k, v in fields.items():
        lines.append(f"| {k} | {v if v not in (None, '') else 'UNKNOWN'} |")
    if unknowns:
        lines += ["", "## 未确定项", ""]
        lines += [f"- {u}" for u in unknowns]
    lines.append("")
    return "\n".join(lines)


def _render_notes(chat_log: list[dict], at: str) -> str:
    lines = ["# 需求澄清问答记录", "", f"> 导出 {at}（真相源：任务 `chat_log`）", ""]
    for m in chat_log:
        who = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"**{who}**：{m.get('text', '')}")
        lines.append("")
    return "\n".join(lines)
