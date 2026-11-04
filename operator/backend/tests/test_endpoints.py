"""API 管理（P4）：端点登记 CRUD + 探针（网络假掉）+ 软删/单价/环境变量口径。

凭据纪律：页面 key 只写不读；探针从页面 key 或环境变量取 key，不落响应明文。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from app.config import settings
from app.models import AgentJob, Endpoint, Project, Task
from app.services import endpoint_probe, llm_config


def test_endpoint_crud_and_unique_name(client):
    c, _ = client
    r = c.post("/api/endpoints", json={"name": "gw", "provider": "dsh-llm-pi-ai",
                                       "base_url": "https://example.test/v1", "model": "m1",
                                       "key_ref": "OPX_LLM_KEY"})
    assert r.status_code == 201, r.text
    eid = r.json()["id"]
    assert r.json()["probe_state"] == "unknown"

    # 同名拒绝
    assert c.post("/api/endpoints", json={"name": "gw"}).status_code == 400

    # PATCH
    p = c.patch(f"/api/endpoints/{eid}", json={"model": "m2", "note": "改一下"})
    assert p.status_code == 200 and p.json()["model"] == "m2"

    assert [e["name"] for e in c.get("/api/endpoints").json()] == ["gw"]
    assert c.delete(f"/api/endpoints/{eid}").status_code == 204
    assert c.get("/api/endpoints").json() == []

    # 软删保留行/id：新建端点不会复用旧 id（审查 H2）
    newer = c.post("/api/endpoints", json={"name": "gw2"})
    assert newer.status_code == 201 and newer.json()["id"] > eid


def test_probe_records_state(client, monkeypatch):
    c, Session = client
    with Session() as db:
        db.add(Endpoint(name="gw", provider="dsh-llm-pi-ai", base_url="https://example.test/v1",
                        key_ref="OPX_LLM_KEY"))
        db.commit()
        eid = db.query(Endpoint).first().id

    monkeypatch.setattr(endpoint_probe, "probe",
                        lambda base_url, key_ref, **kw: {"ok": True, "models": ["m1", "m2"], "model_count": 2})
    body = c.post(f"/api/endpoints/{eid}/probe").json()
    assert body["probe_state"] == "ok" and body["probe"]["model_count"] == 2

    with Session() as db:
        row = db.get(Endpoint, eid)
        assert row.probe_state == "ok" and row.last_probe_at is not None


def test_probe_failure_records_fail(client, monkeypatch):
    c, Session = client
    with Session() as db:
        db.add(Endpoint(name="gw", provider="dsh-llm-deepseek", base_url="https://example.test/v1",
                        key_ref="OPX_LLM_KEY"))
        db.commit()
        eid = db.query(Endpoint).first().id

    monkeypatch.setattr(endpoint_probe, "probe",
                        lambda base_url, key_ref, **kw: {"ok": False, "error": "HTTP 401: nope"})
    body = c.post(f"/api/endpoints/{eid}/probe").json()
    assert body["probe_state"] == "fail" and "401" in body["probe"]["error"]


def test_probe_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPX_LLM_KEY", raising=False)
    out = endpoint_probe.probe("https://example.test/v1", "OPX_LLM_KEY")
    assert out["ok"] is False and "OPX_LLM_KEY" in out["error"]


def test_probe_error_body_masks_key(client, monkeypatch):
    """探针错误回显目标响应体时，不能把当前 key 原样带出来（审查 API-7）。"""
    c, Session = client
    with Session() as db:
        row = Endpoint(name="gw", provider="dsh-llm-deepseek", base_url="https://example.test/v1",
                       api_key="sk-secret-ABCDEF")
        db.add(row)
        db.commit()
        eid = row.id
    monkeypatch.setattr(
        endpoint_probe, "probe",
        lambda *a, **kw: {"ok": False, "error": "HTTP 401: {\"error\":\"bad auth: Bearer sk-secret-ABCDEF\"}"},
    )
    body = c.post(f"/api/endpoints/{eid}/probe").json()
    assert "sk-secret-ABCDEF" not in str(body)
    assert "***" in body["probe"]["error"]


def test_endpoint_prices_roundtrip_and_validation(client):
    """单价配置（D2）：CRUD 往返、只配一半算未定价、负数/非有限/超大值拒绝。"""
    c, _ = client
    r = c.post("/api/endpoints", json={
        "name": "priced", "provider": "dsh-llm-deepseek",
        "price_in": 2, "price_out": 8, "price_cache_read": 0.2,
    })
    assert r.status_code == 201, r.text
    body = r.json()
    eid = body["id"]
    assert body["priced"] is True
    assert (body["price_in"], body["price_out"], body["price_cache_read"]) == (2.0, 8.0, 0.2)
    assert body["price_cache_write"] is None

    # 只配一半 = 未定价（别拿半个价算钱）
    half = c.post("/api/endpoints", json={"name": "half", "price_in": 2})
    assert half.json()["priced"] is False

    # 显式置空 = 清掉该项
    cleared = c.patch(f"/api/endpoints/{eid}", json={"price_out": None})
    assert cleared.json()["priced"] is False and cleared.json()["price_out"] is None

    # 负数直接 400
    assert c.patch(f"/api/endpoints/{eid}", json={"price_in": -1}).status_code == 400
    assert c.post("/api/endpoints", json={"name": "bad", "price_in": -0.1}).status_code == 400
    # Infinity / 1e999 / 超大有限值：必须 400（否则金额溢出、消耗页白屏）
    assert c.post("/api/endpoints", content='{"name":"inf","price_in":1e999,"price_out":1}',
                  headers={"Content-Type": "application/json"}).status_code == 400
    assert c.post("/api/endpoints", json={"name": "huge", "price_in": 1e308, "price_out": 1}).status_code == 400
    assert c.patch(f"/api/endpoints/{eid}", json={"price_out": 1e7}).status_code == 400

    # 其它字段校验
    assert c.post("/api/endpoints", json={"name": "pv", "provider": "nope"}).status_code == 400
    assert c.post("/api/endpoints", json={"name": "bu", "base_url": "ftp://x"}).status_code == 400
    assert c.post("/api/endpoints", json={"name": "q", "base_url": "https://x/v1?a=1"}).status_code == 400
    assert c.post("/api/endpoints", json={"name": "ml", "max_model_len": 0}).status_code == 400


def test_endpoint_key_env_sync_matrix(client, monkeypatch):
    """key → 进程环境变量：只对当前端点生效；清空/切换回退启动基线（审查 M6）。"""
    import os
    from app.config import settings

    c, _ = client
    monkeypatch.setattr(llm_config, "_BASELINE_KEY", "env-fallback")
    monkeypatch.setattr(llm_config, "_BASELINE_CAPTURED", True)
    monkeypatch.setenv(settings.llm_key_env, "env-fallback")

    r = c.post("/api/endpoints", json={
        "name": "gw", "provider": "dsh-llm-pi-ai", "base_url": "https://example.test/v1",
        "model": "m1", "key_ref": settings.llm_key_env, "api_key": "sk-from-page",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["has_key"] is True and body["is_active"] is True
    assert "api_key" not in body and "sk-from-page" not in r.text
    assert os.environ[settings.llm_key_env] == "sk-from-page"

    r2 = c.post("/api/endpoints", json={
        "name": "gw2", "provider": "dsh-llm-deepseek", "base_url": "https://example.test/v2",
        "key_ref": settings.llm_key_env, "api_key": "sk-second",
    })
    assert r2.json()["is_active"] is False
    assert os.environ[settings.llm_key_env] == "sk-from-page"

    # 改非当前端点的 key 不得动当前进程 key（审查 M6）
    assert c.patch(f"/api/endpoints/{r2.json()['id']}", json={"api_key": "sk-second-2"}).status_code == 200
    assert os.environ[settings.llm_key_env] == "sk-from-page"

    assert c.post(f"/api/endpoints/{r2.json()['id']}/activate").json()["is_active"] is True
    assert os.environ[settings.llm_key_env] == "sk-second-2"

    # 清空当前端点 key：回退启动基线，而不是删掉环境变量
    cleared = c.patch(f"/api/endpoints/{r2.json()['id']}", json={"api_key": ""})
    assert cleared.json()["has_key"] is False
    assert os.environ[settings.llm_key_env] == "env-fallback"

    # 激活一个无 key 端点：同样回退基线
    r3 = c.post("/api/endpoints", json={"name": "nokey", "provider": "dsh-llm-deepseek"})
    assert c.post(f"/api/endpoints/{r3.json()['id']}/activate").status_code == 200
    assert os.environ[settings.llm_key_env] == "env-fallback"


def test_dsh_patch_generated_from_active_endpoint(client, monkeypatch):
    """dsh 路由 patch 跟着「当前端点」动态生成（baseURL / model / provider 都对）。"""
    c, _ = client
    gen = llm_config.generated_patch_path()

    r = c.post("/api/endpoints", json={
        "name": "DeepSeek 官方", "provider": "dsh-llm-deepseek",
        "base_url": "https://api.deepseek.com", "model": "deepseek-flash",
        "max_model_len": 128000, "api_key": "sk-deepseek-page",
    })
    assert r.status_code == 201, r.text
    assert gen.is_file(), "应生成 dsh 路由 patch"

    data = yaml.safe_load(gen.read_text(encoding="utf-8"))
    assert data[0]["id"] == "llm-pi-ai" and data[1]["id"] == "agent-default-model"
    providers = data[0]["config"]["providers"]
    provider = providers["deepseek"]
    assert provider["baseURL"] == "https://api.deepseek.com"
    assert provider["api"] == "openai-completions"
    assert provider["apiKeyEnv"] == settings.llm_key_env
    assert provider["models"][0]["id"] == "deepseek-flash"
    assert provider["models"][0]["contextWindow"] == 128000
    assert data[1]["config"] == {"provider": "deepseek", "model": "deepseek-flash"}

    # key 与路由同源：dsh 的 key 就是当前端点 key
    assert os.environ[settings.llm_key_env] == "sk-deepseek-page"

    # agent_runner 实际命令里用的是生成 patch
    from app.services import agent_runner
    cmd = agent_runner.build_command("hi")
    assert str(gen) in cmd and str(gen) == cmd[cmd.index("--patch") + 1]

    # 切到另一个端点：patch/provider/model 跟着换
    r2 = c.post("/api/endpoints", json={
        "name": "opencode 网关", "provider": "dsh-llm-pi-ai",
        "base_url": "https://opencode.ai/inference/go/v1", "model": "deepseek-v4.1-flash",
    })
    assert c.post(f"/api/endpoints/{r2.json()['id']}/activate").status_code == 200
    data = yaml.safe_load(gen.read_text(encoding="utf-8"))
    provider = data[0]["config"]["providers"]["pi-ai"]
    assert "opencode.ai" in provider["baseURL"]
    assert "x-opencode-session" in provider["headers"]
    assert data[1]["config"] == {"provider": "pi-ai", "model": "deepseek-v4.1-flash"}


def test_dsh_patch_falls_back_when_endpoint_incomplete(client):
    """端点缺 base_url/model（或没有端点）时不生成 patch，退回静态兜底文件。"""
    c, _ = client
    gen = llm_config.generated_patch_path()
    assert not gen.exists()

    # 只配了名字、缺 base_url/model：不能生成（否则 dsh 拿到半截路由）
    r = c.post("/api/endpoints", json={"name": "empty", "provider": "dsh-llm-deepseek"})
    assert r.status_code == 201, r.text
    assert not gen.exists()
    assert llm_config.dsh_patch_for_run() == (
        Path(settings.repo_root) / settings.dsh_profile_patch
    )


def test_delete_active_endpoint_falls_back_and_history_keeps_price(client):
    """删当前端点：显式顺延下一个 active；历史调用的金额仍按被删端点价目算（审查 H2/M7）。"""
    c, Session = client
    with Session() as db:
        p = Project(name="p"); db.add(p); db.flush()
        t = Task(project_id=p.id, title="t"); db.add(t); db.commit()
        tid = t.id

    r = c.post("/api/endpoints", json={"name": "premium", "provider": "dsh-llm-deepseek",
                                       "price_in": 1000, "price_out": 1000})
    assert r.status_code == 201 and r.json()["is_active"] is True
    old_id = r.json()["id"]
    with Session() as db:
        db.add(AgentJob(task_id=tid, kind="开发", status="ok", tokens_in=1_000_000, endpoint_id=old_id))
        db.commit()
    assert c.get("/api/usage").json()["totals"]["cost"] == 1000.0

    # 第二个端点（未激活）；删掉 active 端点后应显式顺延到它
    r2 = c.post("/api/endpoints", json={"name": "cheap", "provider": "dsh-llm-deepseek",
                                        "price_in": 1, "price_out": 1})
    assert r2.status_code == 201 and r2.json()["is_active"] is False
    assert c.delete(f"/api/endpoints/{old_id}").status_code == 204
    rows = c.get("/api/endpoints").json()
    assert [x["name"] for x in rows] == ["cheap"] and rows[0]["is_active"] is True

    # 历史金额不被新端点改写；新端点也不会复用旧 id
    tot = c.get("/api/usage").json()["totals"]
    assert tot["cost"] == 1000.0 and tot["cost_estimated"] == 0
    assert rows[0]["id"] != old_id


def test_no_active_endpoint_after_deleting_last_then_create_auto_activates(client):
    """删掉最后一个 active 端点后：列表里没有 active；新建端点应自动成为当前（复核回归）。"""
    c, _ = client
    a = c.post("/api/endpoints", json={"name": "a", "provider": "dsh-llm-deepseek"})
    assert a.status_code == 201 and a.json()["is_active"] is True
    assert c.post("/api/endpoints", json={"name": "b", "provider": "dsh-llm-deepseek"}).json()["is_active"] is False

    # 建 C 不会抢 active（B 还没有 active，只有 A 是）；删 A 后 B 顺延
    c.post("/api/endpoints", json={"name": "c", "provider": "dsh-llm-deepseek"})
    assert c.delete(f"/api/endpoints/{a.json()['id']}").status_code == 204
    rows = {x["name"]: x for x in c.get("/api/endpoints").json()}
    assert rows["b"]["is_active"] is True and rows["c"]["is_active"] is False

    # 删掉 active（b）→ c 顺延为 active；再删最后一个 c → 列表空；新建 d 自动 active
    assert c.delete(f"/api/endpoints/{rows['b']['id']}").status_code == 204
    rows = c.get("/api/endpoints").json()
    assert [x["name"] for x in rows] == ["c"] and rows[0]["is_active"] is True
    assert c.delete(f"/api/endpoints/{rows[0]['id']}").status_code == 204
    assert c.get("/api/endpoints").json() == []
    d = c.post("/api/endpoints", json={"name": "d", "provider": "dsh-llm-deepseek"})
    assert d.status_code == 201 and d.json()["is_active"] is True


def test_key_lives_in_secret_store_not_sqlite(client):
    """页面 key 进本机密钥库；DB 的 api_key 列必须为空，只留 secret_ref。"""
    from app.services import secret_store

    c, Session = client
    r = c.post("/api/endpoints", json={
        "name": "with-key", "provider": "dsh-llm-deepseek", "api_key": "sk-page-key",
    })
    assert r.status_code == 201, r.text
    eid = r.json()["id"]
    assert r.json()["has_key"] is True
    assert r.json()["key_backend"] == "file"  # conftest 强制 file 后端
    ref = f"endpoint-{eid}"

    with Session() as db:
        row = db.get(Endpoint, eid)
        assert row.api_key is None  # 不再落库
        assert row.secret_ref == ref
    assert secret_store.get_secret(ref) == "sk-page-key"
    # 接口仍然不回显明文
    assert "sk-page-key" not in c.get("/api/endpoints").text

    # 清空 key → 密钥库删除 + 引用清空
    cleared = c.patch(f"/api/endpoints/{eid}", json={"api_key": ""})
    assert cleared.json()["has_key"] is False and cleared.json()["key_backend"] is None
    assert secret_store.get_secret(ref) is None
    with Session() as db:
        assert db.get(Endpoint, eid).secret_ref is None

    # 重新设置后删除端点 → 密钥也删除（不把 key 留在机器上）
    c.patch(f"/api/endpoints/{eid}", json={"api_key": "sk-second"})
    assert secret_store.get_secret(ref) == "sk-second"
    assert c.delete(f"/api/endpoints/{eid}").status_code == 204
    assert secret_store.get_secret(ref) is None


def test_legacy_plaintext_key_migrates_to_secret_store(client):
    """旧库明文 api_key 启动时搬到密钥库并清空列（llm_config.migrate_legacy_secrets）。"""
    from app.services import llm_config, secret_store

    c, Session = client
    with Session() as db:
        row = Endpoint(name="legacy", provider="dsh-llm-deepseek", api_key="sk-legacy")
        db.add(row)
        db.commit()
        eid = row.id

    with Session() as db:
        report = llm_config.migrate_legacy_secrets(db)
    assert eid in report["migrated"] and report["failed"] == []

    ref = f"endpoint-{eid}"
    assert secret_store.get_secret(ref) == "sk-legacy"
    with Session() as db:
        row = db.get(Endpoint, eid)
        assert row.api_key is None and row.secret_ref == ref
    body = c.get("/api/endpoints").json()
    assert body[0]["has_key"] is True and body[0]["key_backend"] == "file"


def test_endpoint_name_case_insensitive_and_is_active_false_rejected(client):
    """端点名按不区分大小写去重；`is_active=false` 明确拒绝（不能把当前端点置空）。"""
    c, _ = client
    a = c.post("/api/endpoints", json={"name": "Cheap", "provider": "dsh-llm-deepseek"})
    assert a.status_code == 201 and a.json()["is_active"] is True
    assert c.post("/api/endpoints", json={"name": "cheap", "provider": "dsh-llm-deepseek"}).status_code == 400

    b = c.post("/api/endpoints", json={"name": "other", "provider": "dsh-llm-deepseek"})
    assert b.status_code == 201 and b.json()["is_active"] is False
    r = c.patch(f"/api/endpoints/{a.json()['id']}", json={"is_active": False})
    assert r.status_code == 400 and "设为当前" in r.json()["detail"]
    # 当前端点仍然是 a
    rows = {x["name"]: x for x in c.get("/api/endpoints").json()}
    assert rows["Cheap"]["is_active"] is True
