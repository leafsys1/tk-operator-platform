"""版本存档（savepoints v1）验收：归档 / 解包安全 / 手动存档 / diff / 回滚 / 守卫 /
任务删除 / 迁移往返，外加自动归档挂点与 best-effort 语义。

契约见 `docs/plan-savepoints-v1.md` §8。全部用临时库 / 临时归档根，不碰真实仓库 `archives/`。
"""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import AgentJob, Loop, Project, Run, Savepoint, SpecVersion, Task
from app.routers import runs as runs_router
from app.services import (action_runner, agent_runner, case_confirm, conversations,
                          exec_actions, optimize, optimization, savepoints, task_deletion)

BACKEND = Path(__file__).resolve().parents[1]
OP = "elementwise_add"

#: 契约 §4.1 的 SavepointOut 字段（逐字）
SAVEPOINT_KEYS = {
    "id", "task_id", "label", "note", "kind", "code_snapshot_sha256", "archive_bytes",
    "spec_version_id", "spec_version", "criteria_ref", "suite_ref", "image_digest",
    "runs", "in_sync", "parent_id", "loop_id", "created_at",
}


@pytest.fixture(autouse=True)
def _isolated_archive_root(tmp_path: Path, monkeypatch):
    """每个用例一个独立归档根（内容寻址会跨用例复用同 sha 文件，测试必须隔离）。"""
    root = tmp_path / "archives"
    monkeypatch.setattr(settings, "savepoint_root", str(root))
    return root


def _write_task_files(ws: Path) -> None:
    (ws / "ops" / OP / "op_host").mkdir(parents=True, exist_ok=True)
    (ws / "ops" / OP / "op_host" / "add.asc").write_text("// kernel v1\n", encoding="utf-8")
    (ws / "cases" / OP).mkdir(parents=True, exist_ok=True)
    (ws / "cases" / OP / "cases.json").write_text('{"cases": []}', encoding="utf-8")
    (ws / "tools").mkdir(parents=True, exist_ok=True)
    (ws / "tools" / "measure_curve.py").write_text("# shared v1\n", encoding="utf-8")


def _save(db, task, **kw) -> Savepoint:
    sp = savepoints.create(db, task, **kw)
    db.commit()
    db.refresh(sp)
    return sp


def _tar_info(name: str, *, kind: bytes = tarfile.REGTYPE, linkname: str | None = None,
              data: bytes = b"") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname or ""
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    return info, data


def _make_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if info.type == tarfile.REGTYPE else None)
    return path


# ---------------------------------------------------------------- §8.1 归档往返 / 去重


def test_archive_roundtrip_and_content_addressing(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.cpp").write_text("int a = 1;\n", encoding="utf-8")
    (ws / "src" / "b.h").write_text("// b\n", encoding="utf-8")
    # 所有 skip 目录都放点噪声：归档与 snapshot 的文件集合必须逐字一致
    for skip_dir in (".git", "node_modules", "__pycache__", ".venv", "dist", ".dsh", ".cannbot",
                     ".skills-src", ".opx-chat", "runs"):
        (ws / skip_dir).mkdir()
        (ws / skip_dir / "noise.bin").write_bytes(b"noise")

    path, size, sha = savepoints.ensure_archive(ws)
    sources = savepoints.archive_root() / "sources"
    assert path == sources / f"{sha}.tar.gz"
    assert path.is_file() and size == path.stat().st_size
    # 与 runs.code_snapshot_sha256 同算法
    assert sha == exec_actions.snapshot(ws).sha256
    # skip 集合逐字一致（归档里没有运行时目录）
    with tarfile.open(path) as tf:
        assert sorted(m.name for m in tf) == ["src/a.cpp", "src/b.h"]

    # 解包后重算快照 == 存档 sha
    dest = tmp_path / "unpack"
    savepoints._safe_extract(path, dest)
    assert exec_actions.snapshot(dest).sha256 == sha

    # 同 sha 重复归档只有一份文件（幂等、内容寻址）
    assert savepoints.ensure_archive(ws) == (path, size, sha)
    assert len(list(sources.glob("*.tar.gz"))) == 1


# ---------------------------------------------------------------- §8.2 解包安全


def test_extract_rejects_unsafe_members_and_writes_nothing_outside(tmp_path: Path):
    cases = {
        "parent": _tar_info("../escape.txt", data=b"pwn"),
        "absolute": _tar_info("/tmp/opx-escape-abs.txt", data=b"pwn"),
        "symlink": _tar_info("link", kind=tarfile.SYMTYPE, linkname="/etc/passwd"),
        "hardlink": _tar_info("hard", kind=tarfile.LNKTYPE, linkname="a.txt"),
    }
    for name, member in cases.items():
        archive = _make_tar(tmp_path / f"{name}.tar.gz", [member])
        with pytest.raises(savepoints.SavepointError):
            savepoints._safe_extract(archive, tmp_path / f"out-{name}")
    assert not (tmp_path / "escape.txt").exists()
    assert not Path("/tmp/opx-escape-abs.txt").exists()


def test_diff_rejects_malicious_archive_without_writing_outside(db_session, seeded,
                                                                tmp_path: Path):
    """DB 被篡改成含 `../` 的恶意 tar → diff 拒绝，且不写工作区外。"""
    evil = _make_tar(savepoints.archive_root() / "sources" / f"{'f' * 64}.tar.gz",
                     [_tar_info("../escape2.txt", data=b"pwn")])
    task = seeded["task"]
    sp = Savepoint(task_id=task.id, project_id=task.project_id, label="evil", kind="manual",
                   code_snapshot_sha256="f" * 64, archive_path=str(evil), archive_bytes=1)
    db_session.add(sp)
    db_session.commit()
    with pytest.raises(savepoints.SavepointArchiveError):
        savepoints.diff(db_session, sp, task)
    assert not (tmp_path / "escape2.txt").exists()


# ---------------------------------------------------------------- §8.3 手动创建 + 环境引用


def test_manual_create_env_refs_runs_in_sync_spec_version(db_session, seeded, workspace: Path):
    _write_task_files(workspace)
    task, spec = seeded["task"], seeded["spec"]
    sha = savepoints.current_sha(workspace)

    old_ok = Run(task_id=task.id, action="test", status="ok", code_snapshot_sha256=sha,
                 spec_version_id=spec.id, criteria_ref="custom@2", suite_ref="cases@abc",
                 image_digest="local-id:sha256:img", env_recipe_snapshot="docker run ...",
                 ended_at=datetime.now(timezone.utc) - timedelta(hours=1))
    failed_new = Run(task_id=task.id, action="test", status="failed", code_snapshot_sha256=sha,
                     spec_version_id=spec.id, criteria_ref="failed@1",
                     ended_at=datetime.now(timezone.utc))
    other_sha = Run(task_id=task.id, action="test", status="ok", code_snapshot_sha256="0" * 64)
    db_session.add_all([old_ok, failed_new, other_sha])
    db_session.commit()

    sp = _save(db_session, task, label="M1 通过", note="基线")
    assert sp.kind == "manual" and sp.code_snapshot_sha256 == sha
    # 环境引用取「同 sha 的最新 ok run」
    assert sp.spec_version_id == spec.id and sp.criteria_ref == "custom@2"
    assert sp.suite_ref == "cases@abc" and sp.image_digest == "local-id:sha256:img"
    assert sp.env_recipe_snapshot == "docker run ..."

    payload = savepoints.out(db_session, sp, workspace=workspace)
    assert set(payload) == SAVEPOINT_KEYS
    assert payload["in_sync"] is True and payload["spec_version"] == 1
    assert payload["spec_version_id"] == spec.id
    # runs：按 sha 实时匹配（含 failed），id 升序、最多 20
    assert [r["id"] for r in payload["runs"]] == [old_ok.id, failed_new.id]
    assert payload["runs"][0]["ended_at"].endswith("+00:00")
    assert payload["created_at"].endswith("+00:00")

    # 缺省 label + 列表排序（新→旧）
    sp2 = _save(db_session, task)
    assert sp2.label.startswith("存档 ")
    assert [r["id"] for r in savepoints.list_for_task(db_session, task)] == [sp2.id, sp.id]
    assert savepoints.list_for_task(db_session, task, limit=1)[0]["id"] == sp2.id


# ---------------------------------------------------------------- §8.4 diff


def test_diff_scope_classification_unified_diff_and_truncation(db_session, seeded,
                                                               workspace: Path, monkeypatch):
    _write_task_files(workspace)
    task = seeded["task"]
    sp = _save(db_session, task, label="基线")

    (workspace / "ops" / OP / "op_host" / "add.asc").write_text("// kernel v2\n", encoding="utf-8")
    (workspace / "cases" / OP / "cases.json").unlink()
    (workspace / "ops" / OP / "op_host" / "new.asc").write_text("// new\n", encoding="utf-8")
    (workspace / "tools" / "measure_curve.py").write_text("# shared v2\n", encoding="utf-8")
    (workspace / "tools" / "extra.py").write_text("# extra\n", encoding="utf-8")

    res = savepoints.diff(db_session, sp, task)
    by = {c["path"]: c for c in res["changed"]}
    assert by[f"ops/{OP}/op_host/add.asc"]["status"] == "modified"
    assert by[f"ops/{OP}/op_host/add.asc"]["scope"] == "task"
    assert by[f"cases/{OP}/cases.json"]["status"] == "removed"
    assert by[f"cases/{OP}/cases.json"]["scope"] == "task"
    assert by[f"ops/{OP}/op_host/new.asc"]["status"] == "added"
    assert by[f"ops/{OP}/op_host/new.asc"]["scope"] == "task"
    assert by["tools/measure_curve.py"]["scope"] == "shared"
    assert set(res["outside_scope"]) == {"tools/measure_curve.py", "tools/extra.py"}
    assert res["saved_sha256"] == sp.code_snapshot_sha256
    assert res["current_sha256"] == savepoints.current_sha(workspace)
    assert res["in_sync"] is False
    assert "--- a/" in res["diff"] and "+++ b/" in res["diff"]
    assert res["diff_truncated"] is False
    assert set(res["savepoint"]) == SAVEPOINT_KEYS

    # 超 MAX_DIFF_CHARS → 截断并置 diff_truncated
    monkeypatch.setattr(optimize, "MAX_DIFF_CHARS", 40)
    (workspace / "tools" / "measure_curve.py").write_text(
        "\n".join(f"# line {i}" for i in range(50)), encoding="utf-8")
    assert savepoints.diff(db_session, sp, task)["diff_truncated"] is True

    # op_name 为空：所有路径都算 shared
    task.op_name = None
    db_session.flush()
    res2 = savepoints.diff(db_session, sp, task)
    assert res2["changed"] and all(c["scope"] == "shared" for c in res2["changed"])
    assert set(res2["outside_scope"]) == {c["path"] for c in res2["changed"]}


# ---------------------------------------------------------------- §8.5 回滚（task 范围）


def test_restore_task_scope_restores_task_paths_keeps_shared_and_rescues(db_session, seeded,
                                                                        workspace: Path):
    _write_task_files(workspace)
    task = seeded["task"]
    sp = _save(db_session, task, label="基线")
    task_file = workspace / "ops" / OP / "op_host" / "add.asc"
    cases_file = workspace / "cases" / OP / "cases.json"
    shared_file = workspace / "tools" / "measure_curve.py"
    saved_sha = sp.code_snapshot_sha256

    task_file.write_text("// kernel v2\n", encoding="utf-8")
    cases_file.unlink()
    (workspace / "ops" / OP / "op_host" / "new.asc").write_text("// new\n", encoding="utf-8")
    shared_file.write_text("# shared v2\n", encoding="utf-8")
    before_sha = savepoints.current_sha(workspace)

    res = savepoints.restore(db_session, sp, task, scope="task", note="回基线")
    assert res["restored"] is True and res["in_sync"] is True
    assert res["note"] == "回基线"
    assert res["current_sha256"] == savepoints.current_sha(workspace)
    # 全量口径仍不等：shared 文件留着改动（scope 只回任务范围）
    assert res["current_sha256"] != saved_sha
    # 任务范围还原：改的回来、删的回来、新增的删掉；shared 不动
    assert task_file.read_text(encoding="utf-8") == "// kernel v1\n"
    assert cases_file.read_text(encoding="utf-8") == '{"cases": []}'
    assert not (workspace / "ops" / OP / "op_host" / "new.asc").exists()
    assert shared_file.read_text(encoding="utf-8") == "# shared v2\n"
    assert res["skipped_outside_scope"] == ["tools/measure_curve.py"]
    assert res["changed"] and all(c["scope"] == "task" for c in res["changed"])

    # 急救存档：为回滚哪个存档而做、存的是回滚前的现场
    rescue_out = res["rescue"]
    assert rescue_out is not None and set(rescue_out) == SAVEPOINT_KEYS
    rescue = db_session.get(Savepoint, rescue_out["id"])
    assert rescue.kind == "rescue" and rescue.parent_id == sp.id
    assert rescue.label == f"回滚前自动存档（目标 #{sp.id}）"
    assert rescue.code_snapshot_sha256 == before_sha
    assert Path(rescue.archive_path).is_file()

    # 再查 /diff：任务范围已无差异、全量仍有 shared 差异
    d = savepoints.diff(db_session, sp, task)
    assert d["in_sync"] is False
    assert [c for c in d["changed"] if c["scope"] == "task"] == []
    assert d["outside_scope"] == ["tools/measure_curve.py"]


# ---------------------------------------------------------------- §8.6 回滚（full）


def test_restore_full_scope_restores_shared(db_session, seeded, workspace: Path):
    _write_task_files(workspace)
    task = seeded["task"]
    sp = _save(db_session, task, label="基线")
    shared_file = workspace / "tools" / "measure_curve.py"
    shared_file.write_text("# shared v2\n", encoding="utf-8")
    (workspace / "tools" / "extra.py").write_text("# new shared\n", encoding="utf-8")

    res = savepoints.restore(db_session, sp, task, scope="full")
    assert res["restored"] is True and res["in_sync"] is True
    assert res["current_sha256"] == sp.code_snapshot_sha256
    assert shared_file.read_text(encoding="utf-8") == "# shared v1\n"
    assert not (workspace / "tools" / "extra.py").exists()
    assert res["skipped_outside_scope"] == []
    assert {c["scope"] for c in res["changed"]} == {"shared"}


# ---------------------------------------------------------------- §8.7 守卫


def test_restore_guards_busy_missing_corrupt_and_noop(db_session, seeded, tmp_path: Path,
                                                      workspace: Path):
    _write_task_files(workspace)
    task = seeded["task"]
    sp = _save(db_session, task, label="基线")
    shared_file = workspace / "tools" / "measure_curve.py"
    shared_file.write_text("# changed\n", encoding="utf-8")

    # 运行中的动作作业（本进程 REGISTRY）→ 409
    action_runner.REGISTRY[999] = action_runner.JobState(job_id=999, task_id=task.id)
    try:
        with pytest.raises(savepoints.SavepointBusyError):
            savepoints.restore(db_session, sp, task)
    finally:
        action_runner.REGISTRY.pop(999, None)

    # DB running 的「对话」AgentJob → 409
    job = AgentJob(task_id=task.id, kind="对话", status="running")
    db_session.add(job)
    db_session.commit()
    with pytest.raises(savepoints.SavepointBusyError):
        savepoints.restore(db_session, sp, task)
    # 不写工作区的作业类型（需求澄清）不挡——当前已一致时直接返回 restored=false
    job.kind = "需求澄清"
    shared_file.write_text("# shared v1\n", encoding="utf-8")
    job.status = "ok"
    db_session.commit()
    noop = savepoints.restore(db_session, sp, task)
    assert noop["restored"] is False and noop["rescue"] is None and noop["in_sync"] is True

    # 归档缺失 → 409 语义
    shared_file.write_text("# changed again\n", encoding="utf-8")
    missing = Savepoint(task_id=task.id, project_id=task.project_id, label="缺失",
                        kind="manual", code_snapshot_sha256=sp.code_snapshot_sha256,
                        archive_path=str(tmp_path / "missing.tar.gz"), archive_bytes=1)
    db_session.add(missing)
    db_session.commit()
    with pytest.raises(savepoints.SavepointArchiveError):
        savepoints.restore(db_session, missing, task)

    # 归档内容与 sha 对不上 → 409 语义
    corrupt_path = _make_tar(savepoints.archive_root() / "sources" / f"{'e' * 64}.tar.gz",
                             [_tar_info("a.txt", data=b"x")])
    corrupt = Savepoint(task_id=task.id, project_id=task.project_id, label="损坏",
                        kind="manual", code_snapshot_sha256="e" * 64,
                        archive_path=str(corrupt_path), archive_bytes=1)
    db_session.add(corrupt)
    db_session.commit()
    with pytest.raises(savepoints.SavepointArchiveError):
        savepoints.restore(db_session, corrupt, task)

    # 归档路径越界（DB 被篡改）→ 拒绝读文件
    outside = Savepoint(task_id=task.id, project_id=task.project_id, label="越界",
                        kind="manual", code_snapshot_sha256=sp.code_snapshot_sha256,
                        archive_path="/etc/passwd", archive_bytes=1)
    with pytest.raises(savepoints.SavepointArchiveError):
        savepoints.get_archive_path(outside)

    # 任务无 op_name + scope=task → 400 语义
    task.op_name = None
    db_session.flush()
    with pytest.raises(savepoints.SavepointError, match="op_name"):
        savepoints.restore(db_session, sp, task)
    # scope 非法 → 400 语义
    with pytest.raises(savepoints.SavepointError, match="scope"):
        savepoints.restore(db_session, sp, task, scope="bogus")


def _seed_api(Session, workspace: Path, *, op: str | None = OP, name: str = "p",
              with_loop: bool = False) -> tuple[int, int, int, int | None]:
    with Session() as db:
        project = Project(name=name, workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="elementwise_add", op_name=op,
                    target_chip="910B1")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        loop_id = None
        if with_loop:
            # 自动优化现在只允许 objective 轮次（旧 Loop 只读）：这里建一个可优化轮次。
            objective = {
                "type": "perf", "target_dimension": "all", "target_cases": [],
                "constraints": {"allowed_paths": [f"ops/{op}/"] if op else []},
            }
            loop = Loop(task_id=task.id, kind="perf", scope="分块", hypothesis="增大分块",
                        judge_condition="精度不退化",
                        objective_json=json.dumps(objective, ensure_ascii=False))
            db.add(loop)
            db.flush()
            loop_id = loop.id
        db.commit()
        return project.id, task.id, spec.id, loop_id


def test_api_endpoints_contract_and_status_codes(client, workspace: Path):
    c, Session = client
    _write_task_files(workspace)
    _pid, tid, spec_id, _ = _seed_api(Session, workspace)

    # 手动存档：201 + 契约字段
    r = c.post(f"/api/tasks/{tid}/savepoints", json={"label": "M1 基线", "note": "备注"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body) == SAVEPOINT_KEYS
    assert body["kind"] == "manual" and body["note"] == "备注"
    assert body["spec_version_id"] == spec_id and body["spec_version"] == 1
    assert body["in_sync"] is True and body["runs"] == []
    sp_id, sha = body["id"], body["code_snapshot_sha256"]

    # 列表 / detail / limit 校验
    rows = c.get(f"/api/tasks/{tid}/savepoints").json()
    assert [x["id"] for x in rows] == [sp_id]
    assert c.get(f"/api/tasks/{tid}/savepoints", params={"limit": 0}).status_code == 422
    assert c.get(f"/api/tasks/{tid}/savepoints", params={"limit": 201}).status_code == 422
    assert c.get(f"/api/savepoints/{sp_id}").json()["id"] == sp_id

    # in_sync 是全量口径：工作区一改就 false
    (workspace / "tools" / "measure_curve.py").write_text("# shared v2\n", encoding="utf-8")
    assert c.get(f"/api/tasks/{tid}/savepoints").json()[0]["in_sync"] is False
    diff = c.get(f"/api/savepoints/{sp_id}/diff")
    assert diff.status_code == 200 and diff.json()["in_sync"] is False

    # 回滚：task 范围（shared 不动）→ 再 full → 已一致时 no-op
    res = c.post(f"/api/savepoints/{sp_id}/restore", json={"scope": "task", "note": "回基线"})
    assert res.status_code == 200, res.text
    assert res.json()["restored"] is True and res.json()["in_sync"] is True
    assert res.json()["note"] == "回基线"
    full = c.post(f"/api/savepoints/{sp_id}/restore", json={"scope": "full"})
    assert full.status_code == 200 and full.json()["in_sync"] is True
    assert full.json()["current_sha256"] == sha
    again = c.post(f"/api/savepoints/{sp_id}/restore", json={})
    assert again.status_code == 200
    assert again.json()["restored"] is False and again.json()["rescue"] is None

    # 归档下载：gzip 内容 + 契约文件名
    arch = c.get(f"/api/savepoints/{sp_id}/archive")
    assert arch.status_code == 200 and arch.content[:2] == b"\x1f\x8b"
    assert f"savepoint-{sp_id}-{sha[:12]}.tar.gz" in arch.headers["content-disposition"]

    # 删除：204 只删行，归档文件保留
    assert c.delete(f"/api/savepoints/{sp_id}").status_code == 204
    assert c.get(f"/api/savepoints/{sp_id}").status_code == 404
    assert (savepoints.archive_root() / "sources" / f"{sha}.tar.gz").is_file()

    # 404 / 400 / 409
    assert c.get("/api/savepoints/999999").status_code == 404
    assert c.get("/api/savepoints/999999/diff").status_code == 404
    assert c.get("/api/tasks/999999/savepoints").status_code == 404
    assert c.post("/api/tasks/999999/savepoints", json={}).status_code == 404

    _pid2, tid2, _sid2, _ = _seed_api(Session, workspace, op=None, name="p2")
    sp2 = c.post(f"/api/tasks/{tid2}/savepoints", json={}).json()
    bad = c.post(f"/api/savepoints/{sp2['id']}/restore", json={"scope": "task"})
    assert bad.status_code == 400 and "op_name" in bad.text
    (workspace / "tools" / "measure_curve.py").write_text("# again\n", encoding="utf-8")
    with Session() as db:
        row = db.get(Savepoint, sp2["id"])
        row.archive_path = str(savepoints.archive_root() / "sources" / "missing.tar.gz")
        db.commit()
    gone = c.post(f"/api/savepoints/{sp2['id']}/restore", json={"scope": "full"})
    assert gone.status_code == 409 and "缺失" in gone.text
    assert c.get(f"/api/savepoints/{sp2['id']}/diff").status_code == 409


def test_api_restore_blocked_by_running_agent_job(client, workspace: Path):
    c, Session = client
    _write_task_files(workspace)
    _pid, tid, _sid, _ = _seed_api(Session, workspace)
    sp = c.post(f"/api/tasks/{tid}/savepoints", json={}).json()
    (workspace / "tools" / "measure_curve.py").write_text("# changed\n", encoding="utf-8")
    with Session() as db:
        db.add(AgentJob(task_id=tid, kind="开发", status="running"))
        db.commit()
    r = c.post(f"/api/savepoints/{sp['id']}/restore", json={})
    assert r.status_code == 409 and "运行中的智能体作业" in r.text
    # 任务范围外（需求澄清）不挡
    with Session() as db:
        db.query(AgentJob).filter(AgentJob.task_id == tid).update({"kind": "需求澄清"})
        db.commit()
    ok = c.post(f"/api/savepoints/{sp['id']}/restore", json={})
    assert ok.status_code == 200, ok.text


# ---------------------------------------------------------------- §8.8 任务删除


def test_task_deletion_removes_savepoint_rows_but_keeps_archive(db_session, seeded,
                                                                workspace: Path):
    _write_task_files(workspace)
    task = seeded["task"]
    sp = _save(db_session, task, label="基线")
    archive = Path(sp.archive_path)
    assert task_deletion.counts(db_session, task)["savepoints"] == 1

    task_deletion.delete(db_session, task)
    db_session.expire_all()
    assert db_session.get(Savepoint, sp.id) is None
    # 归档文件内容寻址、可能被别的任务引用 → v1 不随任务删除清理
    assert archive.is_file()


# ---------------------------------------------------------------- §8.9 迁移往返


def test_alembic_roundtrip_includes_savepoints(tmp_path: Path, monkeypatch):
    db = tmp_path / "mig.db"
    monkeypatch.setattr(settings, "db_url", f"sqlite:///{db}")
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def tables() -> set[str]:
        con = sqlite3.connect(db)
        try:
            return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()

    command.upgrade(cfg, "head")
    assert "savepoints" in tables()
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(savepoints)")}
    indexes = {r[1] for r in con.execute("PRAGMA index_list(savepoints)")}
    con.close()
    assert {"id", "project_id", "task_id", "label", "note", "kind", "code_snapshot_sha256",
            "archive_path", "archive_bytes", "spec_version_id", "criteria_ref", "suite_ref",
            "image_digest", "env_recipe_snapshot", "parent_id", "created_at",
            "created_by"} <= cols
    assert {"ix_savepoints_task_id", "ix_savepoints_code_snapshot_sha256"} <= indexes

    command.downgrade(cfg, "base")
    assert "savepoints" not in tables()
    command.upgrade(cfg, "head")
    assert "savepoints" in tables()


# ---------------------------------------------------------------- 自动归档：语义 + 挂点


def test_auto_capture_dedupes_keeps_archives_and_is_best_effort(db_session, seeded,
                                                               workspace: Path, monkeypatch):
    _write_task_files(workspace)
    task = seeded["task"]

    first = savepoints.auto_capture(db_session, task, workspace, label="自动 1")
    db_session.commit()
    assert first is not None and first.kind == "auto"
    # 最新一条 auto 同 sha → 跳过（不建重复行）
    assert savepoints.auto_capture(db_session, task, workspace, label="自动 2") is None

    (workspace / "tools" / "measure_curve.py").write_text("# v2\n", encoding="utf-8")
    second = savepoints.auto_capture(db_session, task, workspace, label="自动 2")
    db_session.commit()
    assert second is not None and second.code_snapshot_sha256 != first.code_snapshot_sha256
    archive1, archive2 = Path(first.archive_path), Path(second.archive_path)

    # MAX_AUTO_KEEP 裁剪：只删行，不删归档
    monkeypatch.setattr(savepoints, "MAX_AUTO_KEEP", 1)
    (workspace / "tools" / "measure_curve.py").write_text("# v3\n", encoding="utf-8")
    third = savepoints.auto_capture(db_session, task, workspace, label="自动 3")
    db_session.commit()
    rows = list(db_session.scalars(
        select(Savepoint).where(Savepoint.task_id == task.id, Savepoint.kind == "auto")))
    assert [r.id for r in rows] == [third.id]
    assert archive1.is_file() and archive2.is_file() and Path(third.archive_path).is_file()

    # best-effort：工作区不存在 / 打包失败都不抛
    assert savepoints.auto_capture(db_session, task, workspace / "nope", label="x") is None
    (workspace / "tools" / "measure_curve.py").write_text("# v4\n", encoding="utf-8")
    assert savepoints.auto_capture(db_session, task, workspace, label="") is not None


def test_hook_action_runner_auto_captures_before_job(db_session, seeded, workspace: Path,
                                                     monkeypatch):
    _write_task_files(workspace)
    task = seeded["task"]
    task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace)
    db_session.commit()
    Session = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(action_runner, "SessionLocal", Session)
    plan = exec_actions.build_plan(db_session, task, "build")
    assert plan.ready, plan.missing

    async def go():
        job = action_runner.start(db_session, task, plan, cmd=["bash", "-c", "true"],
                                  run_ref="rSAVE-1")
        for _ in range(200):
            if job.id not in action_runner.REGISTRY:
                return
            await asyncio.sleep(0.02)
        raise AssertionError("作业未收尾")

    asyncio.run(go())
    db_session.expire_all()
    labels = {r.label for r in db_session.scalars(
        select(Savepoint).where(Savepoint.task_id == task.id))}
    assert "build 执行前自动存档" in labels


def test_hook_conversation_auto_captures_before_send(db_session, seeded, workspace: Path,
                                                     monkeypatch):
    _write_task_files(workspace)
    task = seeded["task"]
    conv = conversations.create(db_session, task.id)
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: None)
    conversations.send(db_session, conv, "改一下 kernel")
    labels = {r.label for r in db_session.scalars(
        select(Savepoint).where(Savepoint.task_id == task.id))}
    assert "对话改前自动存档" in labels


def test_hook_loop_optimize_auto_captures_before_agent(client, workspace: Path, monkeypatch):
    c, Session = client
    _write_task_files(workspace)
    _pid, tid, _sid, loop_id = _seed_api(Session, workspace, with_loop=True)
    monkeypatch.setattr(agent_runner, "start", lambda *a, **kw: None)
    monkeypatch.setattr(runs_router.shutil, "which", lambda _name: "/fake/dsh")
    monkeypatch.setenv("OPX_LLM_KEY", "test-key")
    # 本用例只测“起优化前自动存档”这个挂点；G1 基线门禁另有专门回归测试。
    monkeypatch.setattr(optimization, "require_baseline", lambda db, loop: {"ready": True})

    try:
        r = c.post(f"/api/loops/{loop_id}/optimize", json={})
        assert r.status_code == 201, r.text
        with Session() as db:
            labels = {row.label for row in db.scalars(
                select(Savepoint).where(Savepoint.task_id == tid))}
        assert "自动优化改前自动存档" in labels
    finally:
        # optimize.PENDING 是模块级运行态：假 start 不会触发收尾，别把下一个用例的 Loop 卡成 409
        optimize.PENDING.clear()
