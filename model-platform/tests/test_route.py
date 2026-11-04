#!/usr/bin/env python3
"""离线自测：路线判定（回放 vs Agent）+ 沙箱路径边界。

路线判定是这次改造的闸门：
  判成 replay 却跑不通 -> 白跑一次，然后才升级（可接受，但不该常态）
  判成 agent 却能直接跑 -> 白烧 LLM 调用（用户明确要求避免）
所以这几种组合必须逐条锁住。
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import app  # noqa: E402


def _recipe(token=None, **kw):
    r = {"model": "a/b", "route": "transformers.AutoModel", "elapsed_s": 20.4,
         "autodeps": [], "fingerprint": ({"token": token} if token else {})}
    r.update(kw)
    return r


class TestDecideRoute(unittest.TestCase):
    def setUp(self):
        self._orig_recipes = app.load_recipes
        self._orig_fp = app.env_fingerprint
        app.env_fingerprint = lambda force=False: {"token": "FP1", "image_id": "sha256:x"}

    def tearDown(self):
        app.load_recipes = self._orig_recipes
        app.env_fingerprint = self._orig_fp

    def _with(self, recipes):
        app.load_recipes = lambda: recipes

    def test_no_recipe_goes_agent(self):
        self._with({})
        d = app.decide_route("a/b")
        self.assertEqual(d["route"], "agent")
        self.assertIn("无复现包", d["why"])

    def test_fingerprint_match_replays(self):
        self._with({"a/b": _recipe(token="FP1")})
        d = app.decide_route("a/b")
        self.assertEqual(d["route"], "replay")
        self.assertIn("指纹一致", d["why"])

    def test_fingerprint_mismatch_goes_agent(self):
        self._with({"a/b": _recipe(token="FP0")})
        d = app.decide_route("a/b")
        self.assertEqual(d["route"], "agent")
        self.assertIn("不一致", d["why"])

    def test_legacy_recipe_without_fingerprint_replays_once(self):
        # 历史 111 条配方都没有指纹：不能一上来就全判成 agent，
        # 先按原样回放，成了补指纹，不成才升级
        self._with({"a/b": _recipe(token=None)})
        d = app.decide_route("a/b")
        self.assertEqual(d["route"], "replay")
        self.assertTrue(d.get("legacy"))
        self.assertIn("回放一次", d["why"])

    def test_other_model_unaffected(self):
        self._with({"a/b": _recipe(token="FP1")})
        self.assertEqual(app.decide_route("c/d")["route"], "agent")


class TestEnvFingerprintShape(unittest.TestCase):
    def test_token_stable_and_present(self):
        orig = app.npu
        app.npu = lambda cmd, timeout=120: (0, "sha256:abc\n---\ndriver 24.1\n---\nfirmware 7.5\n")
        try:
            app._FP["data"] = None
            fp = app.env_fingerprint(force=True)
            self.assertTrue(fp["token"])
            self.assertEqual(len(fp["token"]), 16)
            again = app.env_fingerprint(force=True)
            self.assertEqual(fp["token"], again["token"])
            # 镜像变了，指纹必须跟着变（否则环境升级后还会盲回放）
            app.npu = lambda cmd, timeout=120: (0, "sha256:def\n---\ndriver 24.1\n---\nfirmware 7.5\n")
            changed = app.env_fingerprint(force=True)
            self.assertNotEqual(fp["token"], changed["token"])
        finally:
            app.npu = orig
            app._FP["data"] = None

    def test_no_image_means_no_token(self):
        orig = app.npu
        app.npu = lambda cmd, timeout=120: (1, "")
        try:
            app._FP["data"] = None
            fp = app.env_fingerprint(force=True)
            self.assertEqual(fp["token"], "")
        finally:
            app.npu = orig
            app._FP["data"] = None


class TestSandboxPathGuard(unittest.TestCase):
    def _box(self):
        return app.AgentSandbox("nlp", "/data/model_corpus/nlp/org__name", 3, "job-x", None)

    def test_relative_ok(self):
        b = self._box()
        self.assertEqual(b._rel("config.json"), "config.json")
        self.assertEqual(b._rel("./sub/a.txt"), "sub/a.txt")

    def test_absolute_inside_ok(self):
        b = self._box()
        self.assertEqual(b._rel("/data/model_corpus/nlp/org__name/entry.py"), "entry.py")

    def test_parent_traversal_rejected(self):
        b = self._box()
        for bad in ("..", "../other", "a/../../b"):
            with self.assertRaises(ValueError):
                b._rel(bad)

    def test_absolute_outside_rejected(self):
        b = self._box()
        for bad in ("/etc/passwd", "/data/model_corpus/nlp/other__x/f.py"):
            with self.assertRaises(ValueError):
                b._rel(bad)

    def test_run_py_rejects_destructive(self):
        b = self._box()
        r = b.tool_run_py("import os; os.system('rm -rf /')")
        self.assertFalse(r["ok"])
        self.assertIn("破坏性", r["error"])

    def test_write_entry_rejects_syntax_error(self):
        b = self._box()
        r = b.tool_write_entry("def f(:\n  pass\n")
        self.assertFalse(r["ok"])
        self.assertIn("语法错误", r["error"])

    def test_write_entry_rejects_oversize(self):
        b = self._box()
        r = b.tool_write_entry("#" * 200001)
        self.assertFalse(r["ok"])
        self.assertIn("过大", r["error"])


class TestAgentConfig(unittest.TestCase):
    def test_defaults_follow_env(self):
        os.environ["MODEL_AGENT_ENABLED"] = "0"
        try:
            cfg = app.agent_config()
            self.assertFalse(cfg["enabled"])
            self.assertEqual(cfg["model"], os.environ.get(
                "MODEL_AGENT_MODEL", "deepseek-v4.1-flash-1017"))
        finally:
            os.environ.pop("MODEL_AGENT_ENABLED", None)

    def test_key_priority_env_first(self):
        os.environ["MODEL_AGENT_KEY"] = "sk-env"
        try:
            self.assertEqual(app._agent_key(), "sk-env")
        finally:
            os.environ.pop("MODEL_AGENT_KEY", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
