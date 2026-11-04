#!/usr/bin/env python3
"""agent_bridge.py — 模型迁移工作台的「Agent 介入层」（纯标准库，无第三方依赖）。

## 它解决什么问题

迁移平台原本是确定性流水线：探针试完几条固定路线，失败就报错结束。
失败的模型需要人来读日志、改入口、再跑一遍 —— 这一步现在由 agent 做。

## 什么情况下才会叫 agent（这是本模块的边界，不是可选项）

平台侧（app.py）先做路线判定：

  · 命中复现包 + 环境指纹一致 + 入口文件哈希一致  ->  直接回放，**不加载本模块**
  · 没有复现包 / 指纹漂移 / 回放失败              ->  进入本模块

所以在「已经跑通过、现在没变化」的常态路径上，LLM 调用次数恒为 0。

## 协议

单轮单动作的 JSON，不依赖 provider 的 function-calling（任何 OpenAI
兼容网关都能跑，包括只支持纯 chat 的网关）。模型每轮必须输出一个
JSON 对象：

    {"thought": "为什么要这么做", "action": "工具名", "args": {...}}

工具名与参数在 TOOL_DOC 里，未知工具/非法参数都会被拒并回灌一条
observation，让模型自己纠正 —— 不让异常穿透到平台主流程。

## 边界

  · 只能碰「该模型的语料目录」和它的临时输出目录，路径越界直接拒
  · 写入口文件前自动备份，且必须能通过 compile()
  · 轮数、单轮时长、总时长都有上限；到上限就 give_up，不会无限烧
  · 全部动作落 transcript（JSONL），进交付包当审计证据
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 工具文档
TOOL_DOC = """\
可用工具（每轮只能调一个）：

1. list_dir        {"path": "."}
   列出模型目录下的文件（含大小）。path 只能指向模型目录内部。

2. read_text       {"path": "config.json", "max_bytes": 12000}
   读模型目录内的文本文件。二进制/超大文件会被截断。

3. run_py          {"code": "python 代码", "timeout": 300}
   在**昇腾容器内**执行一段 python，工作目录 /workspace/out，
   模型在 /workspace/model（只读），环境变量 ASCEND_RT_VISIBLE_DEVICES 已设好。
   这是你唯一的"动手探测"手段：查类签名、试加载、试前向、打印 shape/dtype。
   torch 的 device 用 npu（npu_shim 已把 .cuda() 重定向）。
   注意：每次 run_py 都是新进程，不保留上一轮的变量。

4. write_entry     {"content": "完整的 entry.py 源码"}
   写入模型的专用入口 entry.py（覆盖前自动备份）。写完后用 run_probe 验证。
   入口约定（必须遵守，平台的判定靠它）：
     · 命令行参数 --model <目录> --out <目录>
     · 成功时在 <out 目录>/entry_result.json 写 {"model":..., "loaded":true, ...}
     · 最后向 stdout 打印一行  VERDICT=LOAD-OK  <耗时>s  device=<设备>
     · 失败时打印 VERDICT=LOAD-FAIL 并打印 ERROR: <原始报错>

5. patch_json      {"path": "config.json", "patch": {"key": "value"}}
   合并写入模型目录里的 json 文件（如 config.json / configuration.json）。

6. run_probe       {}
   用当前 entry.py（或通用探针）在昇腾卡上重跑一次完整探针，
   返回末尾 60 行输出。判定看 VERDICT=LOAD-OK / LOAD-FAIL。

7. finish          {"ok": true, "summary": "...", "root_cause": "...", "fix": "..."}
   结束。ok=true 表示流程已经真的跑通（run_probe 出现过 VERDICT=LOAD-OK）。

规则：
  · 一次只输出一个 JSON 对象，不要输出 Markdown 代码块，不要输出多余解释。
  · 先看证据再动手：不确定类签名/输入格式时就 run_py 打印出来，别猜。
  · 不要修改模型权重文件；不要试图联网下载；不要动 /workspace/model 以外的路径。
  · 同一个错误不要连续试两次同样的改法。
"""

SYSTEM_PROMPT = """\
你是昇腾 NPU（Ascend 910B）模型迁移的现场工程师。你的任务：让给定模型在 NPU 上
真正完成一次前向推理，并给出可复现的专用入口 entry.py。

你有两件武器：
  1. 已知经验（坑库命中、历史修复记录、通用探针试过的路线）；
  2. 在容器里动手验证（run_py）—— 这是你比固定流水线强的地方。

工作方法：
  · 先读证据：目录结构、config.json、上次探针的报错、已有 entry.py（如果有）。
  · 形成假设 -> 用 run_py 做最小验证 -> 再改 entry.py -> 用 run_probe 复验。
  · 报错要按证据分桶：缺依赖 / 设备 API / 模型结构与自定义代码 / 算子 / 资源。
  · 不要为了让它"看起来通过"而伪造 VERDICT；判定只认容器里真的跑出结果。

只输出一个 JSON 对象，格式 {"thought": "...", "action": "...", "args": {...}}。
"""


# ---------------------------------------------------------------- LLM 调用
class LlmError(Exception):
    pass


class LlmClient(object):
    """OpenAI 兼容 chat/completions 的最小客户端（urllib）。"""

    def __init__(self, base_url, api_key, model, timeout=120, max_tokens=1600,
                 temperature=0.2):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "elapsed_s": 0.0}

    def available(self):
        return bool(self.base_url and self.api_key and self.model)

    def chat(self, messages):
        if not self.available():
            raise LlmError("LLM 未配置（base_url / api_key / model 缺一）")
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise LlmError("HTTP %s %s" % (e.code, detail))
        except Exception as e:  # noqa: BLE001
            raise LlmError("%s: %s" % (type(e).__name__, str(e)[:200]))
        self.stats["calls"] += 1
        self.stats["elapsed_s"] = round(self.stats["elapsed_s"] + time.time() - t0, 2)
        u = d.get("usage") or {}
        self.stats["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] += int(u.get("completion_tokens") or 0)
        try:
            msg = d["choices"][0]["message"]
        except Exception:  # noqa: BLE001
            raise LlmError("响应结构异常：%s" % json.dumps(d)[:200])
        # 部分网关把思维链放在 reasoning_content，content 为空时也要能兜住
        content = msg.get("content") or ""
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content, msg


# ---------------------------------------------------------------- 动作解析
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_action(text):
    """从模型输出里抠出一个 JSON 动作对象。

    实测这类网关会：包一层 ```json、前后加解释、把 thinking 也吐出来。
    这里按「最外层花括号配对」扫描，取第一个能解析出 action 的对象。
    """
    if not text:
        raise ValueError("空响应")
    cands = []
    m = _FENCE.search(text)
    if m:
        cands.append(m.group(1).strip())
    cands.append(text.strip())
    # 花括号配对扫描（字符串内的括号不算）
    for s in list(cands):
        depth, start, in_str, esc = 0, -1, False, False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    cands.append(s[start:i + 1])
                    start = -1
    for s in cands:
        try:
            o = json.loads(s)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(o, dict) and o.get("action"):
            return o
    for s in cands:
        try:
            o = json.loads(s)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(o, dict) and ("thought" in o or "args" in o):
            return {"action": "", "args": o,
                    "thought": str(o.get("thought") or "")[:400]}
    raise ValueError("没有可解析的 JSON 动作：%s" % text[:160].replace("\n", " "))


# ---------------------------------------------------------------- 主循环
class AgentLoop(object):
    """单任务 agent：有界、可审计、失败可解释。

    平台侧只需提供 sandbox（工具实现）和 emit（事件回调）。
    """

    MAX_TURNS = int(os.environ.get("MODEL_AGENT_MAX_TURNS", "16"))
    MAX_WALL = float(os.environ.get("MODEL_AGENT_MAX_WALL", "1800"))

    def __init__(self, client, sandbox, emit=None, log=None, max_turns=None,
                 max_wall=None):
        self.client = client
        self.sandbox = sandbox
        self._emit = emit or (lambda *a, **k: None)
        self._log = log or (lambda *a, **k: None)
        self.max_turns = int(max_turns or self.MAX_TURNS)
        self.max_wall = float(max_wall or self.MAX_WALL)
        self.transcript = []
        self.result = None
        self.stop_reason = None

    # -- 内部 --------------------------------------------------
    def _record(self, **kw):
        kw.setdefault("ts", round(time.time(), 3))
        self.transcript.append(kw)
        self._log("AGENT %s" % json.dumps(
            {k: (v if not isinstance(v, str) else v[:200]) for k, v in kw.items()},
            ensure_ascii=False))

    def _step(self, action, args):
        name = (action or "").strip()
        args = args if isinstance(args, dict) else {}
        fn = getattr(self.sandbox, "tool_" + name, None)
        if fn is None:
            return {"ok": False, "error": "未知工具 %r，可用：%s"
                    % (name, "list_dir/read_text/run_py/write_entry/patch_json/"
                             "run_probe/finish")}
        try:
            out = fn(**args)
        except TypeError as e:
            return {"ok": False, "error": "参数不匹配：%s" % e}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        return out if isinstance(out, dict) else {"ok": True, "result": str(out)[:4000]}

    # -- 对外 --------------------------------------------------
    def run(self, task_brief):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_brief + "\n\n" + TOOL_DOC}]
        t0 = time.time()
        self._emit("agent", "running",
                   "Agent 介入：开始诊断（最多 %d 轮，上限 %d 分钟）"
                   % (self.max_turns, int(self.max_wall // 60)))
        for turn in range(1, self.max_turns + 1):
            if time.time() - t0 > self.max_wall:
                self.stop_reason = "wall_timeout"
                break
            try:
                content, raw = self.client.chat(messages)
            except LlmError as e:
                self.stop_reason = "llm_error: %s" % e
                self._record(turn=turn, kind="llm_error", error=str(e)[:300])
                self._emit("agent", "warn", "Agent 调用模型失败：%s" % str(e)[:160])
                break
            try:
                act = parse_action(content)
            except ValueError as e:
                self._record(turn=turn, kind="parse_error",
                             raw=content[:600], error=str(e)[:200])
                self._emit("agent", "warn",
                           "Agent 第 %d 轮输出无法解析，已回灌纠正" % turn)
                messages.append({"role": "assistant", "content": content[:2000]})
                messages.append({"role": "user", "content":
                                 "你上一轮没有输出合法 JSON 动作。只输出一个 JSON 对象，"
                                 '格式 {"thought": "...", "action": "...", "args": {...}}。'})
                continue

            action = str(act.get("action") or "")
            args = act.get("args") or {}
            thought = str(act.get("thought") or "")[:400]
            if thought:
                self._emit("agent", "running", "第 %d 轮 · %s" % (turn, thought[:180]))

            if action == "finish":
                ok = bool(args.get("ok"))
                self.result = {"ok": ok,
                               "summary": str(args.get("summary") or "")[:800],
                               "root_cause": str(args.get("root_cause") or "")[:400],
                               "fix": str(args.get("fix") or "")[:800],
                               "turns": turn}
                self._record(turn=turn, kind="finish", ok=ok,
                             summary=self.result["summary"],
                             root_cause=self.result["root_cause"])
                self.stop_reason = "finished"
                break

            obs = self._step(action, args)
            shown = _brief(obs)
            self._record(turn=turn, kind="action", action=action,
                         args=_safe_args(args), ok=obs.get("ok"),
                         observation=shown[:1200])
            self._emit("agent", "ok" if obs.get("ok") else "warn",
                       "第 %d 轮 · %s → %s" % (turn, action, shown[:220]))
            messages.append({"role": "assistant",
                             "content": json.dumps({"thought": thought,
                                                    "action": action,
                                                    "args": args},
                                                   ensure_ascii=False)[:8000]})
            messages.append({"role": "user", "content":
                             "观察结果：\n" + json.dumps(obs, ensure_ascii=False)[:6000]})
        else:
            self.stop_reason = "max_turns"

        if self.result is None:
            self.result = {"ok": False, "summary":
                           "Agent 未在限定条件内跑通（%s）" % (self.stop_reason or "未知"),
                           "root_cause": "", "fix": "", "turns": len(self.transcript)}
            self._emit("agent", "failed",
                       "Agent 放弃：%s（%d 轮）" % (self.stop_reason, len(self.transcript)))
        elif self.result.get("ok"):
            self._emit("agent", "ok", "Agent 判定已跑通：%s"
                       % (self.result.get("summary") or "")[:180])
        else:
            self._emit("agent", "failed", "Agent 判定未跑通：%s"
                       % (self.result.get("summary") or "")[:180])
        self.result["stop_reason"] = self.stop_reason
        self.result["llm"] = dict(self.client.stats)
        self.result["elapsed_s"] = round(time.time() - t0, 1)
        return self.result

    def dump_jsonl(self, path):
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for row in self.transcript:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return path
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------- 小工具
def _brief(obs):
    if not isinstance(obs, dict):
        return str(obs)[:300]
    if not obs.get("ok"):
        return "失败：" + str(obs.get("error") or "")[:260]
    for k in ("stdout_tail", "output_tail", "content", "result", "summary", "files"):
        v = obs.get(k)
        if v:
            if isinstance(v, str):
                return v[-600:] if len(v) > 600 else v
            return json.dumps(v, ensure_ascii=False)[:600]
    return json.dumps(obs, ensure_ascii=False)[:400]


def _safe_args(args):
    """写 transcript 时把超长内容（比如整份 entry.py）换成长度+哈希。"""
    import hashlib
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = "<%d 字符 sha256=%s>" % (
                len(v), hashlib.sha256(v.encode("utf-8", "replace")).hexdigest()[:16])
        else:
            out[k] = v
    return out
