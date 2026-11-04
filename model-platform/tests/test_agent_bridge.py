#!/usr/bin/env python3
"""离线自测：Agent 介入层 + 路线判定。

这些用例不需要 NPU、不需要网络、不需要数据库，任何一台装了 python3
的机器都能跑：

    cd model-platform && python3 -m unittest discover -s tests -v

覆盖的是「最容易悄悄坏掉」的三块：
  1. 模型输出解析（网关会给 JSON 包各种壳，解析失败=整条链路卡住）
  2. 路线判定（判错了要么白烧 LLM，要么该介入时不介入）
  3. 沙箱路径边界（agent 只能碰自己的模型目录）
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent_bridge  # noqa: E402


class TestParseAction(unittest.TestCase):
    def test_plain_json(self):
        a = agent_bridge.parse_action('{"thought":"t","action":"run_probe","args":{}}')
        self.assertEqual(a["action"], "run_probe")

    def test_fenced_json(self):
        txt = '好的，我先看看：\n```json\n{"action":"list_dir","args":{"path":"."}}\n```\n'
        a = agent_bridge.parse_action(txt)
        self.assertEqual(a["action"], "list_dir")
        self.assertEqual(a["args"]["path"], ".")

    def test_json_inside_prose(self):
        txt = ('分析：这个报错是 decoder_input_ids 缺失。\n'
               '{"thought":"先确认输入","action":"run_py","args":{"code":"print(1)"}}\n'
               '以上就是我的判断。')
        a = agent_bridge.parse_action(txt)
        self.assertEqual(a["action"], "run_py")

    def test_nested_braces_in_string(self):
        txt = '{"action":"write_entry","args":{"content":"d = {\\"a\\": 1}\\nprint(d)"}}'
        a = agent_bridge.parse_action(txt)
        self.assertEqual(a["action"], "write_entry")
        self.assertIn('"a": 1', a["args"]["content"])

    def test_reasoning_leaked_into_content(self):
        # 实测这个网关会把 thinking 混进正文，最后一句话才是动作
        txt = ('我们看一下 config.json。它里面 architectures 是空数组。'
               '那么应该用 AutoModelForCausalLM。\n'
               '{"thought":"直接跑一次探针","action":"run_probe","args":{}}')
        self.assertEqual(agent_bridge.parse_action(txt)["action"], "run_probe")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            agent_bridge.parse_action("")

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            agent_bridge.parse_action("我需要更多信息。")


class TestSafeArgs(unittest.TestCase):
    def test_long_content_hashed(self):
        big = "x" * 5000
        out = agent_bridge._safe_args({"content": big, "path": "a.py"})
        self.assertEqual(out["path"], "a.py")
        self.assertTrue(out["content"].startswith("<5000 字符 sha256="))
        self.assertLess(len(out["content"]), 80)


class TestLlmClientConfig(unittest.TestCase):
    def test_available_flags(self):
        self.assertFalse(agent_bridge.LlmClient("", "k", "m").available())
        self.assertFalse(agent_bridge.LlmClient("http://x", "", "m").available())
        self.assertFalse(agent_bridge.LlmClient("http://x", "k", "").available())
        self.assertTrue(agent_bridge.LlmClient("http://x", "k", "m").available())

    def test_unavailable_raises(self):
        c = agent_bridge.LlmClient("", "", "")
        with self.assertRaises(agent_bridge.LlmError):
            c.chat([{"role": "user", "content": "hi"}])


class _FakeClient(object):
    """按脚本逐轮返回，用来验证主循环的边界，不碰网络。"""

    def __init__(self, script):
        self.script = list(script)
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "elapsed_s": 0.0}
        self.seen = []

    def chat(self, messages):
        self.seen.append(messages)
        self.stats["calls"] += 1
        if not self.script:
            return '{"action":"finish","args":{"ok":false,"summary":"脚本用尽"}}', {}
        return self.script.pop(0), {}


class _Box(object):
    def __init__(self):
        self.called = []

    def tool_run_probe(self):
        self.called.append("run_probe")
        return {"ok": True, "passed": True, "stdout_tail": "VERDICT=LOAD-OK  20.4s"}

    def tool_list_dir(self, path="."):
        self.called.append("list_dir")
        return {"ok": True, "files": ["config.json  1200 B"]}


class TestAgentLoop(unittest.TestCase):
    def _loop(self, script, max_turns=4):
        ev = []
        loop = agent_bridge.AgentLoop(_FakeClient(script), _Box(),
                                      emit=lambda *a: ev.append(a), max_turns=max_turns,
                                      max_wall=60)
        res = loop.run("brief")
        return res, ev

    def test_success_path_records_actions(self):
        script = ['{"action":"list_dir","args":{}}',
                  '{"action":"run_probe","args":{}}',
                  '{"action":"finish","args":{"ok":true,"summary":"跑通了"}}']
        res, ev = self._loop(script)
        self.assertTrue(res["ok"])
        self.assertEqual(res["turns"], 3)
        kinds = [a[1] for a in ev]
        self.assertIn("ok", kinds)
        self.assertIn("running", kinds)

    def test_unknown_tool_is_rejected_not_crashed(self):
        script = ['{"action":"rm_rf_root","args":{}}',
                  '{"action":"finish","args":{"ok":false,"summary":"放弃"}}']
        res, _ = self._loop(script)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stop_reason"], "finished")

    def test_max_turns_stops(self):
        script = ['{"action":"list_dir","args":{}}'] * 10
        res, ev = self._loop(script, max_turns=3)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stop_reason"], "max_turns")
        self.assertIn("failed", [a[1] for a in ev])

    def test_parse_error_is_fed_back(self):
        script = ["我不知道该怎么做。",
                  '{"action":"finish","args":{"ok":true,"summary":"ok"}}']
        res, _ = self._loop(script)
        self.assertTrue(res["ok"])
        # 解析失败那一轮不能算动作轮
        self.assertEqual(res["turns"], 2)


class TestTranscriptFile(unittest.TestCase):
    def test_dump_jsonl(self):
        import tempfile
        loop = agent_bridge.AgentLoop(_FakeClient([]), _Box(), max_turns=1, max_wall=5)
        loop.run("brief")
        with tempfile.TemporaryDirectory() as d:
            p = loop.dump_jsonl(os.path.join(d, "a", "agent.jsonl"))
            self.assertTrue(os.path.isfile(p))
            n = sum(1 for _ in open(p, encoding="utf-8"))
            self.assertGreaterEqual(n, 1)


class TestAssistantEchoKeepsRealArgs(unittest.TestCase):
    """回归：喂回模型的「我自己刚才做了什么」必须带真实参数。

    曾经这里用了脱敏版（长内容换成 `<N 字符 sha256=…>`），结果模型在下一轮
    把那个占位符当成真代码原样再发一次，落盘就成了 SyntaxError
    （实测 yolo_x_layout 第 10 轮踩到）。
    """

    def test_code_present_in_second_call(self):
        code = "print('real code')\n" + "# pad\n" * 200
        client = _FakeClient([
            json.dumps({"action": "run_py", "args": {"code": code}}),
            json.dumps({"action": "finish", "args": {"ok": True, "summary": "ok"}}),
        ])
        box = _Box()
        box.tool_run_py = lambda code, timeout=300: {"ok": True, "stdout_tail": "hi"}
        loop = agent_bridge.AgentLoop(client, box, max_turns=3, max_wall=30)
        loop.run("brief")
        self.assertGreaterEqual(len(client.seen), 2)
        second = json.dumps(client.seen[1], ensure_ascii=False)
        self.assertIn("real code", second)
        self.assertNotIn("字符 sha256=", second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
