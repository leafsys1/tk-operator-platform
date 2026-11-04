"""P0-3 流式并入：输出契约（两段式）+ 增量切分 + 插件回流端点 + 兜底。

全离线：不跑 dsh、不花 token —— 直接喂增量/文本，测「该上屏什么、该落什么」。
对手件：`app/services/prose_relay.py`（切分）、`app/services/chat.py`（解析）、
`app/routers/runs.py`（`POST /api/agent-jobs/{id}/delta`）。
"""

from __future__ import annotations

import json

from app.routers import runs as runs_router
from app.services import agent_runner, chat
from app.services.prose_relay import SPEC_SEP, ProseRelay, split_prose, split_tail


# ---------------------------------------------------------------- 契约解析


def test_parse_output_two_part_contract():
    """新格式：分隔符之前是 prose（= reply），之后是 JSON 产物。"""
    text = f"已确认算子名与精度口径。还差两点：\n1) eps 取多少？\n2) 归约维？\n{SPEC_SEP}\n" + json.dumps(
        {"spec": {"op_name": "rms_norm", "oracle": "torch CPU"},
         "questions": [{"id": "q1", "text": "eps 取多少？"}], "ready": False}, ensure_ascii=False)
    reply, parsed = chat.parse_output(text)
    assert reply.startswith("已确认算子名") and SPEC_SEP not in reply
    assert parsed["spec"]["oracle"] == "torch CPU"
    assert parsed["questions"][0]["id"] == "q1" and parsed["ready"] is False


def test_parse_output_legacy_single_json_still_works():
    """旧格式（整段一个 JSON、reply 在 JSON 里）不被破坏 —— 模型不听话/历史记录都能读。"""
    reply, parsed = chat.parse_output('```json\n{"reply": "先说结论", "spec": {"op_name": "x"}, "ready": true}\n```')
    assert reply == "先说结论" and parsed["spec"]["op_name"] == "x"


def test_parse_output_tail_without_reply_key():
    """新契约的尾段可以只有 spec/questions/ready（没有 reply 键）。"""
    reply, parsed = chat.parse_output(f"我说完了\n{SPEC_SEP}\n" + '{"spec": {}, "questions": [], "ready": true}')
    assert reply == "我说完了" and not parsed["questions"] and parsed["ready"] is True


def test_parse_output_separator_but_broken_json():
    """分隔符在但 JSON 坏了 → 明确报错（不能静默当成功）。"""
    try:
        chat.parse_output(f"说点人话\n{SPEC_SEP}\n{{oops")
    except chat.ChatError as e:
        assert "JSON" in str(e)
    else:  # pragma: no cover
        raise AssertionError("应当抛 ChatError")


def test_prose_only_falls_back_to_plain_text():
    assert chat.prose_only("就是一句普通的话") == "就是一句普通的话"
    assert chat.prose_only("") == ""


# ---------------------------------------------------------------- 增量切分


def test_relay_streams_prose_then_swallows_json_tail():
    r = ProseRelay()
    out = []
    for chunk in ["你好", "，还差", "两点。\n", SPEC_SEP, "\n{", '"spec": {}', "}"]:
        out.append(r.feed(chunk))
    assert "".join(out) == "你好，还差两点。"
    assert r.tail.strip() == '{"spec": {}}'
    assert r.mode == "tail"


def test_relay_holds_back_separator_split_across_chunks():
    """分隔符被切成两段（`<<<SPE` + `C>>>`）时不能把半截当正文喷出去。"""
    r = ProseRelay()
    first = r.feed("正文一")
    second = r.feed("正文二\n<<<SPE")
    third = r.feed("C>>>\n{")
    assert first + second + third == "正文一正文二"
    assert r.mode == "tail"


def test_relay_suppresses_json_first_output():
    """模型退回旧格式（起手就是 JSON）→ 一个字都不上屏，交给解析器取 reply。"""
    r = ProseRelay()
    assert r.feed('```json\n{"reply') == ""
    assert r.feed('": "旧格式", "ready": true}\n```') == ""
    assert r.mode == "suppress" and r.prose == ""
    assert chat.prose_only(r.raw) == "旧格式"


def test_relay_flush_emits_leftover_when_no_separator():
    r = ProseRelay()
    assert r.feed("没有分隔符的一段话") == "没有分隔符的一段话"
    assert r.feed("尾巴") == "尾巴"
    assert r.flush() == "" and r.prose == "没有分隔符的一段话尾巴"


def test_split_helpers():
    assert split_prose(f"人话\n{SPEC_SEP}\n{{}}") == "人话"
    assert split_tail(f"人话\n{SPEC_SEP}\n{{}}") == "{}"
    assert split_prose('{"reply": "x"}') == ""  # 像 JSON 就不是 prose
    assert split_prose("就是人话") == "就是人话"


# ---------------------------------------------------------------- JobState 增量


def test_push_delta_publishes_only_prose():
    st = agent_runner.JobState(7, "p")
    st.push_delta("reason", "想一下")  # 推理增量不上屏（stderr 已经带了）
    st.push_delta("text", "好的，")
    st.push_delta("text", f"问两件事\n{SPEC_SEP}\n{{")
    st.push_delta("text", '"ready": false}')
    kinds = [e.get("type") for e in st.events]
    assert kinds == ["delta", "delta"]
    assert st.prose == "好的，问两件事"
    assert st.text_deltas == 3 and st.reason_deltas == 1


def test_plain_stream_passes_text_verbatim():
    """正式 agent run 没有「prose + JSON 尾」契约：以 `{`/```` ``` ```` 开头也要原样上屏。"""
    st = agent_runner.JobState(70, "p", plain_stream=True)
    st.push_delta("text", '{"answer": "这是普通 JSON 输出，不该被吞"}\n')
    st.push_delta("text", "```py\nprint(1)\n```")
    deltas = [e for e in st.events if e.get("type") == "delta"]
    assert len(deltas) == 2
    assert "".join(e["text"] for e in deltas) == '{"answer": "这是普通 JSON 输出，不该被吞"}\n```py\nprint(1)\n```'


def test_emit_fallback_prose_when_plugin_silent():
    st = agent_runner.JobState(8, "p")
    st.final = f"兜底的一段答复\n{SPEC_SEP}\n" + '{"spec": {}, "questions": [], "ready": false}'
    agent_runner.emit_fallback_prose(st, chat.prose_only)
    assert st.prose == "兜底的一段答复"
    assert st.events[-1]["type"] == "delta" and st.events[-1]["fallback"] is True

    # 插件正常时不该重复补发
    st2 = agent_runner.JobState(9, "p")
    st2.push_delta("text", "逐字的")
    st2.final = "逐字的"
    agent_runner.emit_fallback_prose(st2, chat.prose_only)
    assert len([e for e in st2.events if e.get("fallback")]) == 0


def test_build_command_mounts_plugin_only_when_streaming():
    plain = agent_runner.build_command("任务", session_id="s1")
    assert agent_runner.settings.dsh_stream_patch not in " ".join(plain)
    streamed = agent_runner.build_command("任务", session_id="s1", stream=True)
    assert agent_runner.settings.dsh_stream_patch in " ".join(streamed)
    assert streamed.count("--patch") == 2 and streamed[-2:] == ["--json", "任务"]


def test_delta_env_points_back_at_this_job():
    env = agent_runner.delta_env(42)
    assert env["OPX_RUN_ID"] == "42"
    assert env["OPX_DELTA_URL"].endswith("/api/agent-jobs/42/delta")


# ---------------------------------------------------------------- 插件回流端点


def test_delta_endpoint_feeds_the_job(client):
    c, _Session = client
    st = agent_runner.JobState(4242, "p")
    agent_runner.REGISTRY[4242] = st
    try:
        r = c.post("/api/agent-jobs/4242/delta", json={"run": "4242", "k": "text", "t": "你好"})
        assert r.status_code == 200, r.text
        assert r.json()["text_deltas"] == 1
        assert st.prose == "你好" and st.events[-1]["type"] == "delta"
        # 迟到增量（job 已结束）不报错、也不上屏
        st.done = True
        r2 = c.post("/api/agent-jobs/4242/delta", json={"run": "4242", "k": "text", "t": "晚了"})
        assert r2.status_code == 200 and r2.json()["ok"] is False
        assert st.prose == "你好"
    finally:
        agent_runner.REGISTRY.pop(4242, None)


def test_delta_endpoint_404_for_unknown_job(client):
    c, _Session = client
    r = c.post("/api/agent-jobs/999999/delta", json={"run": "999999", "k": "text", "t": "x"})
    assert r.status_code == 404


def test_job_live_exposes_prose_and_delta_count(client):
    c, _Session = client
    st = agent_runner.JobState(555, "p")
    agent_runner.REGISTRY[555] = st
    try:
        st.push_delta("text", "上屏这段")
        body = c.get("/api/agent-jobs/555/live").json()
        assert body["prose"] == "上屏这段" and body["text_deltas"] == 1
    finally:
        agent_runner.REGISTRY.pop(555, None)
        assert runs_router is not None  # 端点确实来自 runs 路由
