"""P2 优化点搜索/诊断链路的回归测试（审查 1cf4408：M6/M9/M10/M11/M14/M15/M16）。

全部使用临时库 / 临时工作区 / 假 dsh 回调：不跑真实 dsh、不碰真实库（backend/opx.db）、
不启动服务。落库路径（_finish）用假 ``agent_runner.start`` 捕获 on_done，再手工触发。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.models import (AgentJob, IdeaSearch, Loop, OptimizationIdea, Project,
                        Run, SpecVersion, Task)
from app.services import idea_search, loop_budget
from app.services.prose_relay import SPEC_SEP

OP = "elementwise_add"


# ---------------------------------------------------------------- 夹具与小工具


@pytest.fixture(autouse=True)
def _clean_pending():
    """每个用例前后清空模块级运行态，避免假 start 留下的 PENDING 串用例。"""
    idea_search.PENDING.clear()
    yield
    idea_search.PENDING.clear()


@pytest.fixture()
def local_session(db_session, monkeypatch):
    """_finish 用模块级 SessionLocal；测试指到 db_session 的同一个临时库。"""
    factory = sessionmaker(bind=db_session.get_bind(), autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(idea_search, "SessionLocal", factory)
    return factory


def _seed_api(Session, workspace) -> int:
    with Session() as db:
        project = Project(name="ideas", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=OP, op_name=OP, target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return task.id


def _fake_agent_start(monkeypatch) -> dict:
    """把 agent_runner.start 换掉：只捕获 on_done，不真起 dsh。"""
    captured: dict = {}

    def fake_start(job_id, prompt, **kwargs):
        captured.update(job_id=job_id, prompt=prompt, **kwargs)
        return SimpleNamespace(job_id=job_id)

    monkeypatch.setattr(idea_search.agent_runner, "start", fake_start)
    return captured


def _begin_search(db, task_id: int) -> tuple[IdeaSearch, AgentJob]:
    search = IdeaSearch(task_id=task_id, status="running")
    db.add(search)
    db.flush()
    job = AgentJob(task_id=task_id, kind="诊断", status="running")
    db.add(job)
    db.flush()
    search.agent_job_id = job.id
    db.commit()
    return search, job


def _state(final: str = "", *, error=None, exit_code: int | None = 0) -> SimpleNamespace:
    return SimpleNamespace(final=final, error=error, exit_code=exit_code)


# ---------------------------------------------------------------- M6：只读工程强制


def test_readonly_violation_detected_and_rolled_back(
        db_session, seeded, workspace, local_session, monkeypatch):
    """诊断 Agent 改了工作区（修改 + 新增）→ failed + 自动回滚 + 不解析/不入库。"""
    task = seeded["task"]
    captured = _fake_agent_start(monkeypatch)
    search, job = _begin_search(db_session, task.id)

    idea_search.start(search, job, "prompt", workspace, workspace.parent / ".dsh")
    assert idea_search.PENDING[job.id]["search_id"] == search.id

    # 模拟运行中改动：改一个已有文件 + 新增一个文件（只读强制必须两个都抓到）
    (workspace / "src" / "add.cpp").write_text("// hacked by agent\n", encoding="utf-8")
    (workspace / "src" / "extra.cpp").write_text("// new file\n", encoding="utf-8")
    final = "结论\n" + SPEC_SEP + json.dumps(
        {"summary": "s", "ideas": [{"title": "不该被入库"}]}, ensure_ascii=False)
    captured["on_done"](_state(final))

    with local_session() as db:
        row = db.get(IdeaSearch, search.id)
        assert row.status == "failed"
        assert "诊断 Agent 改动了工作区（2 个文件）" in row.summary
        assert "已自动回滚" in row.summary
        result = json.loads(row.result_json)
        assert result["readonly_violation"] is True
        assert result["rollback_failed"] == []
        paths = {item["path"] for item in result["changed_files"]}
        assert paths == {"src/add.cpp", "src/extra.cpp"}
        # 只读违规不解析输出：即使 final 里有合法 ideas 也不能入库
        assert db.scalar(select(func.count()).select_from(OptimizationIdea).where(
            OptimizationIdea.task_id == task.id)) == 0
    # 工作区被还原：改的回原内容、新增的删掉
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == "// kernel\n"
    assert not (workspace / "src" / "extra.cpp").exists()


def test_readonly_violation_rolls_back_even_when_agent_errors(
        db_session, seeded, workspace, local_session, monkeypatch):
    """Agent 自己失败（超时）也照样回滚它改过的文件。"""
    task = seeded["task"]
    captured = _fake_agent_start(monkeypatch)
    search, job = _begin_search(db_session, task.id)
    idea_search.start(search, job, "prompt", workspace, workspace.parent / ".dsh")
    (workspace / "src" / "add.cpp").write_text("// changed then failed\n", encoding="utf-8")
    captured["on_done"](_state("", error="智能体调用超时", exit_code=None))

    with local_session() as db:
        row = db.get(IdeaSearch, search.id)
        assert row.status == "failed"
        assert "工作区" in row.summary and "已自动回滚" in row.summary
    assert (workspace / "src" / "add.cpp").read_text(encoding="utf-8") == "// kernel\n"


def test_finish_success_ingests_ideas(db_session, seeded, workspace, local_session, monkeypatch):
    """无改动 + 正常输出 → ok，思路带 search_id 入库（确保只读分支没有误伤正常路径）。"""
    task = seeded["task"]
    captured = _fake_agent_start(monkeypatch)
    search, job = _begin_search(db_session, task.id)
    idea_search.start(search, job, "prompt", workspace, workspace.parent / ".dsh")
    final = "说明\n" + SPEC_SEP + json.dumps({
        "summary": "找到一条新思路",
        "ideas": [{"title": "缩小 tile", "change_scope": f"ops/{OP}/", "confidence": "high"}],
    }, ensure_ascii=False)
    captured["on_done"](_state(final))

    with local_session() as db:
        row = db.get(IdeaSearch, search.id)
        assert row.status == "ok"
        stats = json.loads(row.result_json)
        assert stats["new"] == 1 and stats["duplicates"] == 0
        idea = db.scalar(select(OptimizationIdea).where(
            OptimizationIdea.task_id == task.id))
        assert idea.search_id == search.id and idea.status == "proposed"


# ---------------------------------------------------------------- M14：_finish 事务


def test_ingest_exception_does_not_leave_new_ideas(
        db_session, seeded, workspace, local_session, monkeypatch):
    """updates.idea_id 非法 → _finish failed 且已 flush 的新思路必须被 rollback（审查 M14）。"""
    task = seeded["task"]
    captured = _fake_agent_start(monkeypatch)
    search, job = _begin_search(db_session, task.id)
    idea_search.start(search, job, "prompt", workspace, workspace.parent / ".dsh")
    final = "说明\n" + SPEC_SEP + json.dumps({
        "summary": "s",
        "ideas": [{"title": "会先 flush 的新思路", "change_scope": f"ops/{OP}/"}],
        "updates": [{"idea_id": "abc", "status": "rejected"}],
    }, ensure_ascii=False)
    captured["on_done"](_state(final))

    with local_session() as db:
        row = db.get(IdeaSearch, search.id)
        assert row.status == "failed"
        assert "诊断结果处理失败" in row.summary
        assert json.loads(row.result_json)["new"] == 0
        assert db.scalar(select(func.count()).select_from(OptimizationIdea).where(
            OptimizationIdea.task_id == task.id)) == 0


# ---------------------------------------------------------------- M15：去重完整性


def test_dedupe_symbol_only_titles_not_collapsed(db_session, seeded):
    """_norm 归一化为空时回退原文：纯符号标题不再互相撞车（审查 M15）。"""
    task = seeded["task"]
    first, dup1 = idea_search.add_user(db_session, task, {"title": "★★★"})
    second, dup2 = idea_search.add_user(db_session, task, {"title": "◆◆◆"})
    assert dup1 is False and dup2 is False and first.id != second.id
    again, dup3 = idea_search.add_user(db_session, task, {"title": "★★★"})
    assert dup3 is True and again.id == first.id
    # 普通标题归一化后等价仍去重
    third, _ = idea_search.add_user(db_session, task, {
        "title": "ABC-优化", "change_scope": f"ops/{OP}/"})
    fourth, dup4 = idea_search.add_user(db_session, task, {
        "title": "abc 优化", "change_scope": f"ops/{OP}/"})
    assert dup4 is True and fourth.id == third.id


def test_dedupe_integrity_error_falls_back_to_duplicate(
        db_session, seeded, monkeypatch):
    """并发在唯一索引上撞车 → rollback 后重查返回 duplicate，而不是 500（审查 M15）。"""
    task = seeded["task"]
    existing, created = idea_search.add_user(db_session, task, {"title": "并发思路"})
    assert created is False

    real_find = idea_search._find_duplicate
    calls = {"n": 0}

    def racing_find(db, task_id, key):
        calls["n"] += 1
        if calls["n"] == 1:  # 模拟并发窗口：预查看不到，flush/commit 才撞唯一索引
            return None
        return real_find(db, task_id, key)

    monkeypatch.setattr(idea_search, "_find_duplicate", racing_find)
    row, duplicate = idea_search.add_user(db_session, task, {"title": "并发思路"})
    assert duplicate is True and row.id == existing.id
    calls["n"] = 0

    search = IdeaSearch(task_id=task.id, status="running")
    db_session.add(search)
    db_session.commit()
    stats = idea_search._ingest(db_session, search,
                                {"ideas": [{"title": "并发思路"}]})
    db_session.commit()
    assert stats["new"] == 0 and stats["duplicates"] == 1
    assert db_session.scalar(select(func.count()).select_from(OptimizationIdea).where(
        OptimizationIdea.task_id == task.id)) == 1


# ---------------------------------------------------------------- M11：串行 / PATCH 约束


def test_serial_blocked_by_open_loop_even_after_patch(client, workspace):
    """串行判定以 Loop 状态为准：PATCH idea.status=rejected 绕不过 open Loop（审查 M11）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    first = c.post(f"/api/tasks/{task_id}/optimization-ideas", json={
        "title": "缩小 tile", "change_scope": f"ops/{OP}/"}).json()["idea"]
    created = c.post(f"/api/optimization-ideas/{first['id']}/round", json={})
    assert created.status_code == 201, created.text
    loop_id = created.json()["loop"]["id"]

    # 旧实现靠 idea.status=validating 判串行 → PATCH 成 rejected 即可绕过；新实现看 Loop。
    patched = c.patch(f"/api/optimization-ideas/{first['id']}", json={"status": "rejected"})
    assert patched.status_code == 200, patched.text
    second = c.post(f"/api/tasks/{task_id}/optimization-ideas", json={
        "title": "减少 DataCopy", "change_scope": f"ops/{OP}/"}).json()["idea"]
    blocked = c.post(f"/api/optimization-ideas/{second['id']}/round", json={})
    assert blocked.status_code == 409
    assert "串行验证" in blocked.text
    with Session() as db:
        assert db.get(Loop, loop_id).status == "open"

    # Loop 关闭（否决）后才允许下一条思路建轮
    assert c.post(f"/api/loops/{loop_id}/reject", json={"reason": "无收益"}).status_code == 200
    assert c.post(f"/api/optimization-ideas/{second['id']}/round", json={}).status_code == 201


def test_patch_status_constraints_and_round_requires_proposed(client, workspace):
    """PATCH 只允许 proposed/rejected/archived；rejected/archived 的思路不能再建轮（审查 M11）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    idea = c.post(f"/api/tasks/{task_id}/optimization-ideas", json={
        "title": "待定思路", "change_scope": f"ops/{OP}/"}).json()["idea"]
    for status in ("selected", "validating", "verified"):
        resp = c.patch(f"/api/optimization-ideas/{idea['id']}", json={"status": status})
        assert resp.status_code == 400, (status, resp.text)
    for status in ("proposed", "rejected", "archived"):
        resp = c.patch(f"/api/optimization-ideas/{idea['id']}", json={"status": status})
        assert resp.status_code == 200, (status, resp.text)
        assert resp.json()["status"] == status

    # 已归档的 idea 不能建轮
    archived = c.patch(f"/api/optimization-ideas/{idea['id']}", json={"status": "archived"})
    assert archived.status_code == 200
    blocked = c.post(f"/api/optimization-ideas/{idea['id']}/round", json={})
    assert blocked.status_code == 409
    assert "proposed" in blocked.text

    # 已否决（未挂 Loop）的 idea 同样不能建轮
    other = c.post(f"/api/tasks/{task_id}/optimization-ideas", json={
        "title": "已被否决的思路", "change_scope": f"ops/{OP}/"}).json()["idea"]
    assert c.patch(f"/api/optimization-ideas/{other['id']}",
                   json={"status": "rejected"}).status_code == 200
    assert c.post(f"/api/optimization-ideas/{other['id']}/round", json={}).status_code == 409


# ---------------------------------------------------------------- M9：运行态健壮性


def _patch_search_deps(monkeypatch):
    """绕过 dsh 可执行文件 / 凭据 / 建 prompt，只测路由状态机。"""
    from app.routers import optimization as optimization_router

    monkeypatch.setenv("OPX_LLM_KEY", "test-only")
    monkeypatch.setattr(optimization_router.shutil, "which", lambda _name: "/tmp/dsh")
    monkeypatch.setattr(idea_search, "build_prompt",
                        lambda db, task, workspace, extra: f"prompt:{extra}")
    started: dict = {}

    def fake_start(search, job, prompt, workspace, dsh_home):
        started.update(search=search.id, job=job.id, prompt=prompt,
                       workspace=str(workspace), dsh_home=str(dsh_home))

    monkeypatch.setattr(idea_search, "start", fake_start)
    return started


def test_zombie_running_search_self_heals(client, workspace, monkeypatch):
    """running 搜索的 AgentJob 已不在跑（failed/缺失）→ 标记 failed 并允许新搜索（审查 M9）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    started = _patch_search_deps(monkeypatch)
    with Session() as db:
        dead = IdeaSearch(task_id=task_id, status="running")
        db.add(dead)
        db.flush()
        dead_job = AgentJob(task_id=task_id, kind="诊断", status="failed")
        db.add(dead_job)
        db.flush()
        dead.agent_job_id = dead_job.id
        orphan = IdeaSearch(task_id=task_id, status="running")  # 连 AgentJob 都没挂上
        db.add(orphan)
        db.commit()
        dead_id, orphan_id = dead.id, orphan.id

    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={"extra": "重点看访存"})
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "running"
    assert started["prompt"] == "prompt:重点看访存"
    with Session() as db:
        assert db.get(IdeaSearch, dead_id).status == "failed"
        assert db.get(IdeaSearch, orphan_id).status == "failed"
        assert db.scalar(select(func.count()).select_from(IdeaSearch).where(
            IdeaSearch.task_id == task_id, IdeaSearch.status == "running")) == 1


def test_live_running_search_still_conflicts(client, workspace, monkeypatch):
    """AgentJob 仍 running（进程真活着）→ 保持 409，不能误杀。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    _patch_search_deps(monkeypatch)
    with Session() as db:
        live = IdeaSearch(task_id=task_id, status="running")
        db.add(live)
        db.flush()
        job = AgentJob(task_id=task_id, kind="诊断", status="running")
        db.add(job)
        db.flush()
        live.agent_job_id = job.id
        db.commit()
        live_id = live.id
    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={})
    assert response.status_code == 409
    assert str(live_id) in response.text
    with Session() as db:
        assert db.get(IdeaSearch, live_id).status == "running"


def test_start_failure_marks_search_and_job_failed(client, workspace, monkeypatch):
    """start 抛错（mkdir/dsh 启动失败等）→ search/job 收尾为 failed + 500，不留 running（审查 M9）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    started = _patch_search_deps(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("dsh 启动失败")

    monkeypatch.setattr(idea_search, "start", boom)
    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={})
    assert response.status_code == 500
    assert "诊断 Agent 启动失败" in response.text
    assert started == {}  # 没走到假 start 的正常记录
    with Session() as db:
        searches = list(db.scalars(select(IdeaSearch).where(IdeaSearch.task_id == task_id)))
        assert searches and all(row.status == "failed" for row in searches)
        assert "启动失败" in (searches[0].summary or "")
        jobs = list(db.scalars(select(AgentJob).where(AgentJob.task_id == task_id)))
        assert jobs and all(job.status == "failed" for job in jobs)


def test_start_search_builds_prompt_in_threadpool(client, workspace, monkeypatch):
    """build_prompt 的同步重 IO 必须走线程池，不能阻塞事件循环（审查 M13/M9）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    from app.routers import optimization as optimization_router

    monkeypatch.setenv("OPX_LLM_KEY", "test-only")
    monkeypatch.setattr(optimization_router.shutil, "which", lambda _name: "/tmp/dsh")
    calls: dict = {}

    def fake_prompt(db, task, workspace, extra):
        calls["extra"] = extra
        return f"prompt:{extra}"

    async def fake_threadpool(fn, *args, **kwargs):
        calls["fn"] = fn
        calls["args"] = args
        return fn(*args, **kwargs)

    monkeypatch.setattr(idea_search, "build_prompt", fake_prompt)
    monkeypatch.setattr(optimization_router, "run_in_threadpool", fake_threadpool)
    monkeypatch.setattr(idea_search, "start",
                        lambda search, job, prompt, ws, home: calls.update(started=prompt))
    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={"extra": "重点"})
    assert response.status_code == 201, response.text
    assert calls["fn"] is fake_prompt
    assert calls["args"][3] == "重点"
    assert calls["started"] == "prompt:重点"


def test_idea_search_extra_length_is_capped(client, workspace):
    """IdeaSearchRequest.extra 有 4000 字上限（审查 M10：prompt 注入面）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    response = c.post(f"/api/tasks/{task_id}/idea-searches", json={"extra": "x" * 4001})
    assert response.status_code == 422


# ---------------------------------------------------------------- M10：证据限额


def test_evidence_pack_bounds_loop_and_idea_payloads(db_session, seeded):
    task = seeded["task"]
    db_session.add(Loop(
        task_id=task.id, kind="perf", status="closed",
        optimize_json=json.dumps({
            "status": "done", "changed": True, "reverted": False, "guard_reverted": False,
            "verification": "pass", "after_sha256": "a" * 64,
            "agent_summary": "摘要" * 20_000,
            "diff": "d" * 100_000,
            "revert": {"blob": "r" * 50_000},
            "agent_output": {"blob": "o" * 50_000},
        }, ensure_ascii=False),
    ))
    for i in range(70):
        db_session.add(OptimizationIdea(
            task_id=task.id, source="agent", title=f"思路 {i}", status="proposed",
            dedupe_key=f"k{i:03d}",
            evidence_json=json.dumps({"blob": "x" * 5000, "idx": i}),
            outcome_json=json.dumps({"blob": "y" * 5000}),
        ))
    db_session.commit()

    evidence = idea_search.evidence_pack(db_session, task)
    assert len(evidence["ideas"]) <= idea_search.MAX_IDEAS
    assert evidence.get("ideas_truncated") == 70 - idea_search.MAX_IDEAS
    for loop in evidence["loops"]:
        optimize = loop["optimize"]
        assert optimize is not None
        assert "diff" not in optimize and "revert" not in optimize and "agent_output" not in optimize
        assert len(json.dumps(optimize, ensure_ascii=False)) <= idea_search.MAX_LOOP_OPTIMIZE_CHARS
    for idea in evidence["ideas"]:
        for key in ("evidence", "outcome"):
            payload = idea.get(key)
            if payload is not None:
                assert len(json.dumps(payload, ensure_ascii=False)) <= idea_search.MAX_IDEA_FIELD_CHARS


def test_evidence_pack_bounds_run_reports(db_session, seeded, tmp_path, monkeypatch):
    task = seeded["task"]
    big_case_list = [{"case_id": f"c{i}", "median_us": i * 1.5, "note": "z" * 20}
                     for i in range(800)]
    (tmp_path / "performance.json").write_text(json.dumps({
        "verdict": "pass", "cases": big_case_list}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "correctness.json").write_text(json.dumps({
        "verdict": "pass", "cases": big_case_list}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({
        "metrics": [{"key": f"m{i}", "value": i, "note": "z" * 20} for i in range(800)]},
        ensure_ascii=False), encoding="utf-8")
    run = Run(task_id=task.id, action="benchmark", status="ok",
              manifest_path=str(tmp_path / "manifest.json"),
              code_snapshot_sha256="0" * 64, criteria_ref="criteria@1", suite_ref="suite@1")
    db_session.add(run)
    db_session.commit()
    monkeypatch.setattr(idea_search.gate, "artifact_paths", lambda _run: {
        "performance": tmp_path / "performance.json",
        "correctness": tmp_path / "correctness.json",
        "metrics": tmp_path / "metrics.json",
    })

    evidence = idea_search.evidence_pack(db_session, task)
    reports = evidence["runs"][0]["reports"]
    serialized = json.dumps(reports, ensure_ascii=False)
    # 4 份报告聚合也要压进单个 run 的上限，且不是原样塞满
    assert len(serialized) <= idea_search.MAX_RUN_REPORT_CHARS
    assert serialized.count('"case_id"') < 800
    assert serialized.count('"m') < 800


def test_bounded_evidence_hard_cap():
    """证据包 JSON 总长硬上限 120k，超出必须截断并打 truncated（审查 M10）。"""
    evidence = {
        "task": {"id": 1, "op_name": OP},
        "runs": [{"id": i, "reports": {"performance": {"blob": "y" * 5000}}}
                 for i in range(16)],
        "loops": [{"id": i, "optimize": {"agent_summary": "z" * 5000}} for i in range(12)],
        "ideas": [{"id": i, "evidence": {"blob": "b" * 5000}} for i in range(60)],
    }
    assert len(json.dumps(evidence, ensure_ascii=False)) > idea_search.MAX_EVIDENCE_CHARS
    bounded = idea_search._bounded_evidence(evidence)
    assert bounded["truncated"] is True
    assert len(json.dumps(bounded, ensure_ascii=False)) <= idea_search.MAX_EVIDENCE_CHARS


def test_build_prompt_caps_extra_and_mentions_truncation(db_session, seeded, workspace):
    task = seeded["task"]
    prompt = idea_search.build_prompt(db_session, task, workspace, "x" * 5000)
    assert "证据包可能被截断" in prompt
    assert "禁止修改任何文件" in prompt
    assert "x" * 4001 not in prompt  # extra 内部也兜底截断到 4000


# ---------------------------------------------------------------- M16：解析严格化


def test_parse_output_requires_ideas_list():
    with pytest.raises(idea_search.IdeaSearchError):
        idea_search._parse_output("说明\n" + SPEC_SEP + '{"summary": "没有 ideas"}')
    with pytest.raises(idea_search.IdeaSearchError):
        idea_search._parse_output("说明\n" + SPEC_SEP + '{"summary": "x", "ideas": "nope"}')
    with pytest.raises(idea_search.IdeaSearchError):
        idea_search._parse_output("说明\n" + SPEC_SEP + "not json")
    with pytest.raises(idea_search.IdeaSearchError):
        idea_search._parse_output("完全没有 JSON 的普通回答")


def test_parse_output_cleans_prose_on_both_paths():
    prose, parsed = idea_search._parse_output(
        "```\n先看证据\n```\n" + SPEC_SEP + '\n{"summary": "s", "ideas": []}\n')
    assert parsed == {"summary": "s", "ideas": []}
    assert "先看证据" in prose
    assert "```" not in prose and SPEC_SEP not in prose

    # 没有 SPEC_SEP：走 fence/裸 JSON 兜底，prose 同样清洗掉 fence
    prose2, parsed2 = idea_search._parse_output(
        '说明文字\n```json\n{"summary":"s","ideas":[{"title":"t"}]}\n```\n')
    assert parsed2["ideas"][0]["title"] == "t"
    assert "说明文字" in prose2
    assert "```" not in prose2 and SPEC_SEP not in prose2

    # 分隔符后还包了 fence（非裸 JSON）：仍走兜底扫描，空 list 合法
    prose3, parsed3 = idea_search._parse_output(
        "说明\n" + SPEC_SEP + '\n```json\n{"summary":"s","ideas":[]}\n```\n')
    assert parsed3["ideas"] == []
    assert "```" not in prose3

    # 兜底路径同样严格：ideas 不是数组要抛错
    with pytest.raises(idea_search.IdeaSearchError):
        idea_search._parse_output('```json\n{"summary":"s","ideas":"nope"}\n```')


# ---------------------------------------------------------------- budget-check 契约


def test_budget_check_passes_reservation_id(client, workspace, monkeypatch):
    """budget-check 端点透传 reservation_id（budget-fix 的 keyword-only 契约）。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    with Session() as db:
        loop = Loop(task_id=task_id, kind="perf", status="open",
                    objective_json=json.dumps({"type": "perf",
                                               "budget": {"runs_per_action": 2}}))
        db.add(loop)
        db.commit()
        loop_id = loop.id

    from app.routers import optimization as optimization_router
    recorded: dict = {}

    def fake_check(db, task, loop, action, *, reservation_id=None):
        recorded.update(db=db, task=task.id, loop=loop.id, action=action,
                        reservation_id=reservation_id)
        return {"enforced": True, "reused": True, "audit_id": 11,
                "limit": 2, "used": 1}

    monkeypatch.setattr(optimization_router.loop_budget, "check_and_audit", fake_check)
    response = c.post(f"/api/loops/{loop_id}/budget-check",
                      json={"action": "benchmark", "reservation_id": 7})
    assert response.status_code == 200, response.text
    assert response.json()["audit_id"] == 11
    assert recorded == {"db": recorded["db"], "task": task_id, "loop": loop_id,
                        "action": "benchmark", "reservation_id": 7}

    response = c.post(f"/api/loops/{loop_id}/budget-check",
                      json={"action": "test", "reservation_id": None})
    assert response.status_code == 200
    assert recorded["reservation_id"] is None

    assert c.post(f"/api/loops/{loop_id}/budget-check",
                  json={"action": "test", "reservation_id": "abc"}).status_code == 400
    assert c.post(f"/api/loops/{loop_id}/budget-check",
                  json={"action": "sleep"}).status_code == 400

    def reject(*args, **kwargs):
        raise loop_budget.BudgetExceeded("预算预留无效或已被使用（fail-closed）")

    monkeypatch.setattr(optimization_router.loop_budget, "check_and_audit", reject)
    response = c.post(f"/api/loops/{loop_id}/budget-check",
                      json={"action": "benchmark", "reservation_id": 99})
    assert response.status_code == 409


def test_budget_check_reservation_works_with_real_service(client, workspace):
    """端点 ↔ 真实 loop_budget 的预留契约：允许审计可 claim 一次，重复/伪造 fail-closed 409。"""
    c, Session = client
    task_id = _seed_api(Session, workspace)
    with Session() as db:
        loop = Loop(task_id=task_id, kind="perf", status="open",
                    objective_json=json.dumps({"type": "perf",
                                               "budget": {"runs_per_action": 2}}))
        db.add(loop)
        db.commit()
        loop_id = loop.id

    reserved = c.post(f"/api/loops/{loop_id}/budget-check", json={"action": "benchmark"})
    assert reserved.status_code == 200, reserved.text
    audit_id = reserved.json().get("audit_id")
    assert isinstance(audit_id, int)
    claimed = c.post(f"/api/loops/{loop_id}/budget-check",
                     json={"action": "benchmark", "reservation_id": audit_id})
    assert claimed.status_code == 200, claimed.text
    assert claimed.json()["reused"] is True and claimed.json()["audit_id"] == audit_id
    # 预留只能消费一次；伪造 id 也一样 fail-closed
    assert c.post(f"/api/loops/{loop_id}/budget-check",
                  json={"action": "benchmark", "reservation_id": audit_id}).status_code == 409
    assert c.post(f"/api/loops/{loop_id}/budget-check",
                  json={"action": "benchmark", "reservation_id": 999999}).status_code == 409
