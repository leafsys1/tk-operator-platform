"""P0-3「任务开始」页：草稿生命周期（建草稿 / 存草稿 / 重新加载 / 转正）。

覆盖 `POST /api/tasks`（扩字段）+ `PATCH /api/tasks/{id}`（exclude_unset 语义）+ 列表 `?draft=` 过滤。
对应 `docs/UI-task-start-v1.md` §2.3（2026-10-15 定）。
"""

from __future__ import annotations

from app.models import Project


def _project(Session) -> int:
    with Session() as db:
        p = Project(name="p", workspace_path="/tmp/ws")
        db.add(p)
        db.commit()
        return p.id


def test_draft_create_save_reload_and_promote(client):
    c, Session = client
    pid = _project(Session)

    # 首次输入 → 建草稿（is_draft=1 + 需求原文）
    created = c.post("/api/tasks", json={
        "project_id": pid, "title": "rms_norm", "op_name": "rms_norm",
        "input_form": "自然语言", "is_draft": True,
        "raw_input": "我要开发 rms_norm，做到跟 torch CPU 的实现一一对应",
    })
    assert created.status_code == 201, created.text
    task = created.json()
    tid = task["id"]
    assert task["is_draft"] is True
    assert task["raw_input"].startswith("我要开发 rms_norm")

    # 列表过滤：草稿只在 ?draft=true 出现
    assert any(t["id"] == tid for t in c.get("/api/tasks", params={"draft": "true"}).json())
    assert not any(t["id"] == tid for t in c.get("/api/tasks", params={"draft": "false"}).json())

    # 存草稿：改配置 + 标题，raw_input 不动
    patched = c.patch(f"/api/tasks/{tid}", json={
        "title": "rms_norm-910b", "device_id": "3", "timeout_s": 2400,
        "criteria_ref": "ops-precision-standard@1", "budget_per_action": 1, "fix_rounds_max": 2,
    })
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["device_id"] == "3" and body["timeout_s"] == 2400
    assert body["raw_input"].startswith("我要开发 rms_norm")  # 没传的字段保持原值

    # 重新加载
    reloaded = c.get(f"/api/tasks/{tid}").json()
    assert reloaded["title"] == "rms_norm-910b"
    assert reloaded["budget_per_action"] == 1 and reloaded["fix_rounds_max"] == 2

    # 转正（is_draft=false）→ 挪进正式列表
    promoted = c.patch(f"/api/tasks/{tid}", json={"is_draft": False})
    assert promoted.json()["is_draft"] is False
    assert any(t["id"] == tid for t in c.get("/api/tasks", params={"draft": "false"}).json())
    assert not any(t["id"] == tid for t in c.get("/api/tasks", params={"draft": "true"}).json())


def test_patch_rejects_unknown_host_and_can_clear_override(client):
    c, Session = client
    pid = _project(Session)
    tid = c.post("/api/tasks", json={"project_id": pid, "title": "t", "is_draft": True}).json()["id"]

    bad = c.patch(f"/api/tasks/{tid}", json={"host_id": 999999})
    assert bad.status_code == 400
    assert "host_id" in bad.json()["detail"]

    c.patch(f"/api/tasks/{tid}", json={"device_id": "5", "timeout_s": 900})
    # 显式传 null 才清空（exclude_unset 与 None 是两回事）
    cleared = c.patch(f"/api/tasks/{tid}", json={"device_id": None}).json()
    assert cleared["device_id"] is None
    assert cleared["timeout_s"] == 900  # 没传的保持


def _formal_task(Session) -> int:
    """一个已定版形态的任务（is_draft=False + current_spec_version_id），用于状态机守卫测试。"""
    from app.models import SpecVersion, Task

    with Session() as db:
        p = Project(name="p-formal", workspace_path="/tmp/ws")
        db.add(p)
        db.flush()
        task = Task(project_id=p.id, title="rms_norm", op_name="rms_norm", stage="开发")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        db.commit()
        return task.id


def test_patch_rejects_null_or_blank_unsettable_fields(client):
    """显式 null / 空串 → 422（不是写库时 NOT NULL 500）；stage 只收枚举（审查 #9/#3）。"""
    c, Session = client
    tid = c.post("/api/tasks", json={"project_id": _project(Session), "title": "t",
                                     "is_draft": True}).json()["id"]

    for body in ({"title": None}, {"title": ""}, {"title": "   "},
                 {"is_draft": None}, {"stage": "瞎写的阶段"}):
        r = c.patch(f"/api/tasks/{tid}", json=body)
        assert r.status_code == 422, (body, r.status_code, r.text)
    assert c.get(f"/api/tasks/{tid}").status_code == 200  # 一个坏 PATCH 都没写进去
    # 合法标题仍可改
    assert c.patch(f"/api/tasks/{tid}", json={"title": " 新标题 "}).json()["title"] == "新标题"


def test_patch_keeps_draft_state_machine_closed(client):
    """已有 spec/runs 的任务不能打回草稿；定版后不能再改 op_name（审查 #4/#5）。"""
    c, Session = client
    tid = _formal_task(Session)

    r = c.patch(f"/api/tasks/{tid}", json={"is_draft": True})
    assert r.status_code == 409 and "草稿" in r.json()["detail"]

    r = c.patch(f"/api/tasks/{tid}", json={"op_name": "other_op"})
    assert r.status_code == 409 and "算子名" in r.json()["detail"] and "/specs" in r.json()["detail"]

    # 同一个 op_name 再 PATCH 一次不算改（幂等放行）
    assert c.patch(f"/api/tasks/{tid}", json={"op_name": "rms_norm"}).status_code == 200
    # 纯草稿（无 spec/runs）仍可显式置 is_draft=true
    draft = c.post("/api/tasks", json={"project_id": _project(Session), "title": "d",
                                       "is_draft": True}).json()["id"]
    assert c.patch(f"/api/tasks/{draft}", json={"is_draft": True}).status_code == 200


def test_patch_checks_effective_host_image_pair(client):
    """host/image 交叉校验用「生效值」（请求值 > 任务现值 > 项目默认），不放过跨机组合（审查 #18）。"""
    from app.models import Host, HostImage, Task

    c, Session = client
    with Session() as db:
        h1 = Host(name="h1", ssh_alias="h1")
        h2 = Host(name="h2", ssh_alias="h2")
        db.add_all([h1, h2])
        db.flush()
        i1 = HostImage(host_id=h1.id, image_ref="img1", tag="t")
        i2 = HostImage(host_id=h2.id, image_ref="img2", tag="t")
        db.add_all([i1, i2])
        db.flush()
        project = Project(name="p-pair", workspace_path="/tmp/ws",
                          host_id=h1.id, image_id=i1.id)
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="t")
        db.add(task)
        db.commit()
        ids = {"task": task.id, "h2": h2.id, "i1": i1.id, "i2": i2.id}

    # 只改 host（image 沿用项目默认 i1，在 h1 上）→ 跨机，拒绝
    bad = c.patch(f"/api/tasks/{ids['task']}", json={"host_id": ids["h2"]})
    assert bad.status_code == 400 and "不在执行机" in bad.json()["detail"]
    # 显式给出不匹配的组合 → 拒绝
    bad2 = c.patch(f"/api/tasks/{ids['task']}",
                   json={"host_id": ids["h2"], "image_id": ids["i1"]})
    assert bad2.status_code == 400
    # 同机组合 → 放行
    ok = c.patch(f"/api/tasks/{ids['task']}",
                 json={"host_id": ids["h2"], "image_id": ids["i2"]})
    assert ok.status_code == 200, ok.text
    assert (ok.json()["host_id"], ok.json()["image_id"]) == (ids["h2"], ids["i2"])


def test_op_name_path_traversal_rejected(client):
    """op_name 参与工作区路径拼接，必须挡 `../..` / 绝对路径（任务外页面审查 M15）。"""
    c, Session = client
    pid = _project(Session)
    for bad in ("../evil", "a/b", "..", ".", "/tmp/evil", "a\\b"):
        r = c.post("/api/tasks", json={"project_id": pid, "title": "t", "op_name": bad})
        assert r.status_code == 422, (bad, r.status_code, r.text)

    created = c.post("/api/tasks", json={"project_id": pid, "title": "t", "op_name": "relu"})
    assert created.status_code == 201
    tid = created.json()["id"]
    assert c.patch(f"/api/tasks/{tid}", json={"op_name": "../evil"}).status_code == 422
    # 规格版本写入入口同样校验
    spec = c.post(f"/api/tasks/{tid}/specs", json={"op_name": "../evil", "spec_json": "{}"})
    assert spec.status_code == 422
