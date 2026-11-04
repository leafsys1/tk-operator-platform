"""P0-3 第二步：对话式需求澄清 —— **走 dsh 会话**（`--session-id` 续会话）。

与 `agent_runner` 分工：agent_runner 管「起进程 / 读 JSONL / 落摘要」；这里管「拼 prompt +
解析模型结构化输出」。每轮固定层由后端重拼（方案 §8.1）：DSH 每轮是独立进程，"模型记得读过"
不能当依赖 —— 所以当前规格草稿与待确认项每次都重新喂进去；同时用 `--session-id` 让它保有会话。

**输出契约（两段式，支持逐 token 上屏）**：模型先写给人看的自然语言回复（= `reply`，会被
逐 token 流给前端，见 `prose_relay`），另起一行输出分隔符 `<<<SPEC>>>`，其后是承载结构化
产物的 JSON `{spec, questions, ready}`。解析入口 `parse_output()`；**旧格式**（只有一个 JSON、
`reply` 在 JSON 里）继续兼容，兜底见 `extract_json`。

**文档输入（任务生命周期 v2）**：`build_prompt(..., doc_text=...)` 把任务上传的《算子设计文档》
解析稿摘录（截断到 ~12k 字符）注入本轮 prompt；文档只是需求输入，**不改变**上面的两段式输出契约。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from . import agent_runner
from .prose_relay import SPEC_SEP, split_prose


class ChatError(RuntimeError):
    pass


SYSTEM_PROMPT = """你是「op-agent 算子开发控制台」的需求澄清助手。目标：把用户对某个昇腾（Ascend）算子的模糊描述，通过一问一答逐步问清楚，产出一份可执行的「算子需求规格草稿」。

工作规则：
- 每轮先从用户这句话里**提取已确定的信息**更新规格草稿；再就**仍然缺失的关键项**最多问 2~3 个问题。
- 不要臆造。没确认的字段不要写进草稿（或写 "UNKNOWN"）；宁可多问。
- 若本轮下方提供了「用户上传的算子设计文档摘录」：把它当作需求输入来源之一（**低于用户本轮明确要求、高于你自己的推断**）。文档已明确的事实可直接采用并写进草稿；文档没写的仍要提问或标 "UNKNOWN"。
- 若文档含**测试用例附录**（如「附录 A：测试用例要点」里的精度组 / 性能组表格）：在规格草稿里用 `document_test_cases` 字段保留其要点（精度：case_id / 分组 / shape / dtype / 输入生成 / 期望或拒绝码；性能：case_id / n / dtype / 主用例 / 必需），并在第一段自然语言回复里提示用户「用例将在后续任务对话中整组确认」。**不要**在本轮逐条确认用例；文档只是参考输入，**不改变下面的两段式输出契约**。
- 下面这些关键项没清楚之前，不要判 ready=true：① 合法且稳定的单段算子名 `op_name`（不能含路径分隔符）② 算子的数学语义（做什么）③ 输入输出（shape / dtype / 布局）④ 正确性口径（参考实现 oracle、容差、一致性）⑤ 支持范围与不支持的组合 ⑥ 特殊行为（空输入 / NaN / Inf）。
- 回复用简洁、专业的中文，可以带编号问题。
- **不要调用任何工具**，不要读写文件，不要联网。

规格草稿建议字段（有则填，无则省略）：
op_name, math_def, params, inputs, outputs, layout, support, unsupported, special_behavior,
target_chip, target_hardware_profile（硬件画像文件引用，如 ops/chips/dav-2201.yaml）, target_env,
oracle, tolerance, consistency, performance_baseline, performance_target, delivery, assumptions, unknowns,
document_test_cases（仅当上传文档的测试用例附录里有内容时填写：原样保留用例要点，供后续任务对话整组确认）

项目默认（可参考，用户可改）：目标芯片 {chip}；**硬件画像默认 `ops/chips/dav-2201.yaml`**（目标芯片确定后填进 target_hardware_profile）；参考实现 oracle = 「与 torch CPU 实现一一对应」；容差档 = ops-precision-standard@1；一致性 = 输出 shape/dtype 完全一致 + NaN/Inf 逐元素精确比对。

【输出格式（必须严格遵守，两段式）】
第一段：**给用户看的自然语言回复** —— 就是你会直接对用户说的话（确认已定的信息、提出你要问的问题）。
这一段会被**逐字实时显示**给用户，所以：不要出现 JSON、字段名、代码围栏、分隔符本身；不要写"以下是我的 JSON"之类的话；保持简洁专业的中文。
第二段：自然语言回复写完后，**另起一行单独输出分隔符** `{sep}`，**紧接着另起一行**输出一个 JSON 对象（不要 markdown 围栏、不要任何解释、必须是最后一个字符是 `}}`）。结构：
{{"spec": {{...已确定的规格字段...}},
  "questions": [{{"id": "q1", "text": "待确认项"}}],
  "ready": false}}"""



@dataclass
class TurnResult:
    ok: bool
    prompt: str
    parsed: dict | None = None
    reply: str = ""  # 自然语言回复（= prose；旧格式从 JSON 的 reply 字段取）
    session_id: str | None = None
    usage: dict = field(default_factory=dict)
    exit_code: int | None = None
    error: str | None = None


#: 创建期对话注入的文档解析稿上限（字符）。上传文档可能几万字，全塞进每轮 prompt 会拖垮上下文；
#: 超过就截断并在 prompt 里**明说已截断**，让模型不要假装看过没给的部分（完整稿在任务工作区）。
DOC_TEXT_MAX_CHARS = 12000


def _doc_excerpt(doc_text: str | None) -> str | None:
    """规整上传文档解析稿：去首尾空白；空 → None；超过 `DOC_TEXT_MAX_CHARS` 截断并留标记。"""
    text = (doc_text or "").strip()
    if not text:
        return None
    if len(text) <= DOC_TEXT_MAX_CHARS:
        return text
    return (
        text[:DOC_TEXT_MAX_CHARS].rstrip()
        + f"\n\n…（摘录已截断，仅保留前 {DOC_TEXT_MAX_CHARS} 字符；"
          "未覆盖的章节请向用户询问，或让用户在后续任务对话里补充）"
    )


def build_prompt(chip: str, spec: dict, questions: list, message: str,
                 doc_text: str | None = None) -> str:
    """拼创建期对话的完整 prompt。

    `doc_text` 可选（默认 None，保持旧调用方向后兼容）：传任务上传的《算子设计文档》**解析稿**
    （`task-docs/task-<id>/*.extracted.md` 的内容，由 router 读取后传入）。注入位置在
    「当前待确认项」之后、「用户本轮输入」之前，标注为「用户上传的算子设计文档摘录」，
    并截断到 `DOC_TEXT_MAX_CHARS`。文档只作为需求输入，不改变两段式输出契约。
    """
    prompt = (
        SYSTEM_PROMPT.format(chip=chip, sep=SPEC_SEP)
        + "\n\n[当前规格草稿 JSON]\n" + json.dumps(spec, ensure_ascii=False)
        + "\n[当前待确认项 JSON]\n" + json.dumps(questions, ensure_ascii=False)
    )
    doc = _doc_excerpt(doc_text)
    if doc:
        prompt += "\n[用户上传的算子设计文档摘录]\n" + doc
    prompt += "\n[用户本轮输入]\n" + message
    return prompt


def _iter_json_objects(s: str):
    """按平衡花括号切出候选 JSON 片段（容忍回复里夹带别的文字/工具输出）。"""
    depth = 0
    start = -1
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield s[start : i + 1]


def extract_json(text: str) -> dict:
    """从模型输出里抠出我们的 JSON 对象：优先代码围栏，其次平衡括号，并校验含约定键。"""
    if not (text or "").strip():
        raise ChatError("dsh 没有返回最终回答（final 为空）")
    candidates: list[str] = [b.strip() for b in re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)]
    candidates.append(text)
    wanted = {"reply", "spec", "questions", "ready"}
    for cand in candidates:
        for obj in _iter_json_objects(cand):
            try:
                data = json.loads(obj)
            except ValueError:
                continue
            if isinstance(data, dict) and wanted & set(data):
                return data
        try:
            data = json.loads(cand.strip())
        except ValueError:
            continue
        if isinstance(data, dict) and wanted & set(data):
            return data
    raise ChatError(f"模型没有按要求返回 JSON：{text[:200]}")


def parse_output(text: str) -> tuple[str, dict]:
    """解析一轮模型输出 → `(reply, parsed)`（**两段式契约 + 旧格式兼容**）。

    - 新格式：分隔符 `<<<SPEC>>>` 之前是 prose（= reply），之后是 JSON `{spec, questions, ready}`；
    - 旧格式（无分隔符、整段是 JSON）：`reply` 从 JSON 里取 —— 兼容模型不听话/历史记录；
    - 解析不出 JSON 一律抛 `ChatError`（调用方决定是 502 还是留痕跳过）。
    """
    if not (text or "").strip():
        raise ChatError("dsh 没有返回最终回答（final 为空）")
    i = text.find(SPEC_SEP)
    if i >= 0:
        prose = text[:i].rstrip("\n").strip()
        tail = text[i + len(SPEC_SEP):]
        try:
            parsed = extract_json(tail)
        except ChatError:
            parsed = _any_json(tail)
            if parsed is None:
                raise ChatError(f"分隔符之后的 JSON 解析失败：{tail[:200]}") from None
        return (prose or str(parsed.get("reply") or "").strip()), parsed
    parsed = extract_json(text)
    return str(parsed.get("reply") or "").strip(), parsed


def _any_json(text: str) -> dict | None:
    """宽松一档：只要是个 JSON 对象就收（新契约的尾段不一定带 `reply` 键）。"""
    for cand in [b.strip() for b in re.findall(r"```(?:json)?\s*(.*?)```", text or "", re.S)] + [text or ""]:
        for obj in _iter_json_objects(cand):
            try:
                data = json.loads(obj)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def prose_only(text: str) -> str:
    """兜底用：从**完整输出**里取 prose（流式插件没生效时，把答复一次性补上屏）。"""
    try:
        reply, _ = parse_output(text)
        return reply
    except ChatError:
        return split_prose(text)


async def converse(*, prompt: str, cwd: Path, dsh_home: Path,
                   session_id: str | None, timeout: float) -> TurnResult:
    """跑一轮 dsh 会话；失败不抛（返回 ok=False + error），好让调用方留痕。"""
    if not agent_runner.available():
        return TurnResult(ok=False, prompt=prompt, session_id=session_id,
                          error="找不到 dsh 可执行文件（控制台进程 PATH 里要有）")
    if settings.llm_key_env and not os.environ.get(settings.llm_key_env):
        return TurnResult(ok=False, prompt=prompt, session_id=session_id,
                          error="缺少凭据：先在「API 管理」页配置 key（或设置环境变量 "
                                f"{settings.llm_key_env}）")

    state = await agent_runner.run_once(prompt, cwd, dsh_home, session_id=session_id, timeout=timeout)
    usage = dict(state.usage or {})
    sid = state.session_id or session_id
    if state.error or state.exit_code != 0:
        tail = state.stderr_tail[-1] if state.stderr_tail else ""
        detail = state.error or f"dsh 退出码 {state.exit_code}"
        return TurnResult(ok=False, prompt=prompt, session_id=sid, usage=usage,
                          exit_code=state.exit_code, error=f"{detail} {tail}".strip())
    try:
        reply, parsed = parse_output(state.final)
    except ChatError as e:
        return TurnResult(ok=False, prompt=prompt, session_id=sid, usage=usage,
                          exit_code=state.exit_code, error=str(e))
    return TurnResult(ok=True, prompt=prompt, parsed=parsed, reply=reply, session_id=sid,
                      usage=usage, exit_code=state.exit_code)
