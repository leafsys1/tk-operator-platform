"""HTTP 契约：plan（dry_run）、501 行为、门禁端点、动作清单。"""

from __future__ import annotations

from datetime import datetime, timezone

import hashlib
import json
from pathlib import Path

from app.models import Artifact, Host, HostImage, Project, Run, SpecVersion, Task
from app.services import case_confirm


def _seed_api(Session, workspace: Path):
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
        db.add(host)
        db.flush()
        image = HostImage(host_id=host.id, image_ref="lfx-op-dev-910b", tag="torch210-cann910",
                          is_op_dev=True, env_recipe="docker run --privileged ...",
                          torch_version="2.10.0", digest="sha256:beef")
        db.add(image)
        db.flush()
        project = Project(name="p", workspace_path=str(workspace), host_id=host.id, image_id=image.id,
                          criteria_ref="ops-precision-standard@1")
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title="t", op_name="elementwise_add")
        db.add(task)
        db.flush()
        spec = SpecVersion(task_id=task.id, version=1, spec_json="{}", sha256="0" * 64)
        db.add(spec)
        db.flush()
        task.current_spec_version_id = spec.id
        task.cases_confirmed_at = datetime.now(timezone.utc)  # v2：四动作 plan 要求用例组已确认
        task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace)
        db.commit()
        return task.id, image.id


def test_actions_contract_lists_required_artifacts(client):
    c, _ = client
    body = c.get("/api/actions").json()
    assert body["required_artifacts"]["test"] == ["log", "correctness"]
    by_key = {a["action"]: a for a in body["actions"]}
    assert by_key["benchmark"]["needs_device"] is True
    assert by_key["build"]["needs_device"] is False


def test_plan_dry_run_and_async_start(client, workspace: Path, monkeypatch):
    c, Session = client
    task_id, _ = _seed_api(Session, workspace)
    # 本用例只验 HTTP plan/异步启动契约；smoke 前置由 test_smoke/test_plan 单独覆盖。
    from app.services import exec_actions
    monkeypatch.setattr(exec_actions, "smoke_gate_reason", lambda *_a, **_kw: None)

    plan = c.post(f"/api/tasks/{task_id}/actions", json={"action": "test", "dry_run": True}).json()
    assert plan["ready"] is True, plan.get("missing")
    assert plan["criteria_ref"] == "ops-precision-standard@1"
    assert plan["spec_version"] == 1
    assert plan["snapshot_files"] >= 4
    assert "--action test" in plan["command"]
    # 四动作脚本按 OPX_OP 找 ops/<op>；不传会退回 elementwise_add
    assert "OPX_OP=elementwise_add" in plan["command"]

    # dry_run=false 现在起一条异步作业（这里把真执行换掉，只验接口契约）
    from app.services import action_runner

    class FakeJob:
        id = 99
        run_ref = "rFAKE-99"

    monkeypatch.setattr(action_runner, "start", lambda *a, **kw: FakeJob())
    r = c.post(f"/api/tasks/{task_id}/actions", json={"action": "test", "dry_run": False})
    assert r.status_code == 200
    assert r.json()["action_job_id"] == 99 and r.json()["run_ref"] == "rFAKE-99"


def test_plan_reports_unbound_project(client, workspace: Path):
    c, Session = client
    task_id, _ = _seed_api(Session, workspace)
    with Session() as db:
        db.query(Project).update({"host_id": None, "image_id": None})
        db.commit()
    plan = c.post(f"/api/tasks/{task_id}/actions", json={"action": "build"}).json()
    assert plan["ready"] is False
    assert any("未绑定执行机" in m for m in plan["missing"])


def test_gate_endpoint_over_registered_run(client, workspace: Path):
    c, Session = client
    task_id, image_id = _seed_api(Session, workspace)

    # 造一份真实的 run 目录 + manifest（这里就是 M1 里 nm_test.sh 会写出来的形状）
    run_dir = workspace / "runs" / "r1"
    (run_dir / "out").mkdir(parents=True)
    (run_dir / "out" / "run.log").write_bytes(b"hello\n")
    snap = "a" * 64
    man = {
        "run_id": "r1", "exit_code": 0, "code_snapshot_sha256": snap, "spec_version_id": 1,
        "criteria_ref": "ops-precision-standard@1", "suite_ref": "cases@abc", "image_digest": "sha256:beef",
        "device": "0", "lock_mode": "exclusive",
        "artifacts": [{"type": "log", "path": "out/run.log",
                       "sha256": hashlib.sha256(b"hello\n").hexdigest(), "size_bytes": 6}],
    }
    (run_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")

    created = c.post(f"/api/tasks/{task_id}/runs", json={
        "action": "test", "status": "ok", "exit_code": 0, "image_id": image_id,
        "spec_version_id": 1, "code_snapshot_sha256": snap, "suite_ref": "cases@abc",
        "criteria_ref": "ops-precision-standard@1", "image_digest": "sha256:beef", "device_id": "0",
        "manifest_path": str(run_dir / "manifest.json"), "sync_state": "pulled",
        "artifacts": man["artifacts"],
    })
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]

    gate = c.get(f"/api/runs/{run_id}/gate").json()
    codes = {f["code"] for f in gate["findings"]}
    # 这次执行本身是「记录齐了但没产出 correctness 报告」→ 不该判过，且要说清缺什么
    assert gate["verdict"] == "fail"
    assert "E_REQUIRED_ARTIFACT_MISSING" in codes
    assert c.get("/api/runs/999999/gate").status_code == 404


def test_artifact_paths_cannot_escape_run_directory(client, workspace: Path):
    """对外登记和历史脏数据读取都要拒绝宿主绝对路径 / `..`（审查 H8）。"""
    c, Session = client
    task_id, _image_id = _seed_api(Session, workspace)
    run_dir = workspace / "runs" / "safe"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text('{"artifacts": []}', encoding="utf-8")
    base = {
        "action": "test", "status": "failed", "exit_code": 1,
        "manifest_path": str(manifest),
    }
    absolute = c.post(f"/api/tasks/{task_id}/runs", json={
        **base, "artifacts": [{"type": "log", "path": "/etc/hosts"}],
    })
    assert absolute.status_code == 400
    traversal = c.post(f"/api/tasks/{task_id}/runs", json={
        **base, "artifacts": [{"type": "log", "path": "../secret.txt"}],
    })
    assert traversal.status_code == 400

    # 即使是升级前已经落进 DB 的绝对路径，读取端也必须 fail-closed。
    with Session() as db:
        row = Run(task_id=task_id, action="test", status="failed", exit_code=1,
                  manifest_path=str(manifest))
        db.add(row)
        db.flush()
        art = Artifact(run_id=row.id, type="log", path="/etc/hosts")
        db.add(art)
        db.commit()
        artifact_id = art.id
    assert c.get(f"/api/artifacts/{artifact_id}/text").status_code == 403
    assert c.get(f"/api/artifacts/{artifact_id}/download").status_code == 403


def test_cancel_sends_remote_docker_rm(client, workspace: Path, monkeypatch):
    """取消要传到远端：`docker rm -f opx-<run_ref>`（不是只杀本地会话）。"""
    from types import SimpleNamespace

    from app.services import remote_cancel

    c, Session = client
    task_id, _ = _seed_api(Session, workspace)
    with Session() as db:
        run = Run(task_id=task_id, action="test", status="running", remote_workdir="rCANCEL")
        db.add(run)
        db.commit()
        rid = run.id
    seen: list[list[str]] = []
    monkeypatch.setattr(remote_cancel.subprocess, "run",
                        lambda cmd, **kw: (seen.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr="")))
    body = c.post(f"/api/runs/{rid}/cancel").json()
    assert body["remote"]["ok"] is True
    assert "docker rm -f opx-rCANCEL" in seen[0][-1]


def test_health(client):
    c, _ = client
    assert c.get("/api/health").status_code == 200


def test_run_report_endpoint_reads_machine_report(client, workspace: Path):
    """结果视图靠这个端点在页面上画图：按类型读 run 的机读报告，读到就原样返回。"""
    c, Session = client
    task_id, _ = _seed_api(Session, workspace)

    run_dir = workspace / "runs" / "r1"
    (run_dir / "out").mkdir(parents=True)
    perf = {"spec_version_id": 1, "code_snapshot_sha256": "a" * 64,
            "criteria_ref": "ops-precision-standard@1", "suite_ref": "cases@x",
            "cases": [{"case_id": "bench_1m", "n": 10,
                       "candidate": {"device_us": {"median_us": 1.0}},
                       "baseline": {"device_us": {"median_us": 2.0}}}]}
    body = json.dumps(perf).encode()
    (run_dir / "out" / "performance.json").write_bytes(body)
    man = {"run_id": "r1", "exit_code": 0, "code_snapshot_sha256": "a" * 64, "spec_version_id": 1,
           "criteria_ref": "ops-precision-standard@1", "image_digest": "sha256:beef",
           "artifacts": [{"type": "performance", "path": "out/performance.json",
                          "sha256": hashlib.sha256(body).hexdigest()}]}
    (run_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with Session() as db:
        row = Run(task_id=task_id, action="benchmark", status="ok", exit_code=0,
                  manifest_path=str(run_dir / "manifest.json"), code_snapshot_sha256="a" * 64,
                  criteria_ref="ops-precision-standard@1", image_digest="sha256:beef")
        db.add(row)
        db.commit()
        rid = row.id

    r = c.get(f"/api/runs/{rid}/report?type=performance")
    assert r.status_code == 200
    assert r.json()["report"]["cases"][0]["case_id"] == "bench_1m"
    # 该 run 没有 metrics 产物 → 明确 404（不静默返回空）
    assert c.get(f"/api/runs/{rid}/report?type=metrics").status_code == 404
    assert c.get("/api/runs/999999/report").status_code == 404


def test_plan_prefers_task_host_then_current_test_server(client, workspace: Path):
    """执行机优先级：任务显式 > 当前测试服务器 > 项目默认；镜像也必须落在同一台机器上。"""
    c, Session = client
    task_id, project_image_id = _seed_api(Session, workspace)
    with Session() as db:
        from app.models import Project
        project = db.query(Project).first()
        second = Host(name="tk37", ssh_alias="tk37", cards="8×910B1")
        db.add(second)
        db.flush()
        second_image = HostImage(host_id=second.id, image_ref="img2", tag="t",
                                 is_op_dev=True, is_test=True, env_recipe="docker run --privileged ...",
                                 torch_version="2.10.0", digest="sha256:2222")
        db.add(second_image)
        db.flush()
        second.is_test_server = True
        db.commit()
        second_id, second_image_id = second.id, second_image.id

    # 没指定任务覆盖 → 用当前测试服务器（不是项目默认的第一台）
    plan = c.post(f"/api/tasks/{task_id}/actions", json={"action": "build"}).json()
    assert plan["host_id"] == second_id
    # 项目默认镜像不在当前测试服务器上 → 明确报缺项，而不是悄悄跨机跑
    assert any("不在执行机" in m for m in plan["missing"])

    # 任务显式选第二台的测试镜像 → plan 用任务覆盖
    p = c.patch(f"/api/tasks/{task_id}", json={"host_id": second_id, "image_id": second_image_id})
    assert p.status_code == 200, p.text
    plan2 = c.post(f"/api/tasks/{task_id}/actions", json={"action": "build"}).json()
    assert plan2["host_id"] == second_id and plan2["image_id"] == second_image_id
    assert plan2["ready"] is True, plan2.get("missing")


def test_host_test_server_is_unique_and_image_test_flag(client):
    c, Session = client
    a = c.post("/api/hosts", json={"name": "a", "ssh_alias": "a"}).json()
    b = c.post("/api/hosts", json={"name": "b", "ssh_alias": "b"}).json()
    assert c.post(f"/api/hosts/{b['id']}/set-test").json()["is_test_server"] is True
    hosts = c.get("/api/hosts").json()
    assert [h["is_test_server"] for h in hosts] == [True, False]  # b 排在最前
    assert all(h["is_test_server"] is False for h in hosts if h["id"] != b["id"])

    with Session() as db:
        img = HostImage(host_id=b["id"], image_ref="img", tag="t")
        db.add(img)
        db.commit()
        img_id = img.id
    r = c.patch(f"/api/host-images/{img_id}", json={"is_test": True}).json()
    assert r["is_test"] is True
    assert c.get(f"/api/hosts/{b['id']}/images").json()[0]["is_test"] is True


def test_host_create_requires_ssh_target_and_supports_edit(client):
    """SSH 目标才是必填项；名称可省（默认取目标）；登记后可用 PATCH 改错。"""
    c, _ = client
    assert c.post("/api/hosts", json={"name": "no-target"}).status_code == 400

    created = c.post("/api/hosts", json={"ssh_alias": "user@10.0.0.9", "cred_ref": "~/.ssh/id_host"})
    assert created.status_code == 201, created.text
    h = created.json()
    assert h["name"] == "user@10.0.0.9"  # 名称缺省 = SSH 目标
    assert h["cred_ref"] == "~/.ssh/id_host"  # 只回引用路径，不是私钥内容

    # 名称缺省也参与唯一性
    assert c.post("/api/hosts", json={"ssh_alias": "user@10.0.0.9"}).status_code == 409

    patched = c.patch(f"/api/hosts/{h['id']}", json={"name": "tk37", "ssh_alias": "tk37", "note": "第二台"})
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert (body["name"], body["ssh_alias"], body["note"]) == ("tk37", "tk37", "第二台")

    # 不能把连接目标清空（否则探针/执行都没有落点）
    assert c.patch(f"/api/hosts/{h['id']}", json={"ssh_alias": None, "address": None}).status_code == 400


def test_host_probe_uses_host_ssh_key_ref(client, monkeypatch):
    """探针的私钥跟 host 走（cred_ref），端口从 `user@host:port` 拆出来。"""
    from app.routers import hosts as hosts_router
    from app.services.probe import ProbeResult

    c, _ = client
    h = c.post("/api/hosts", json={"ssh_alias": "tk37:50023", "cred_ref": "~/.ssh/id_host"}).json()
    seen: dict[str, object] = {}

    def fake_probe(alias, key, opts, timeout, port=None):
        seen.update(alias=alias, key=key, port=port)
        return ProbeResult(ok=True, hostname="tk37")

    monkeypatch.setattr(hosts_router, "probe_host", fake_probe)
    r = c.post(f"/api/hosts/{h['id']}/probe")
    assert r.status_code == 200, r.text
    assert seen["alias"] == "tk37"  # 端口不能留在目标里（`user@host:port` 不是合法 ssh 目标）
    assert seen["port"] == 50023
    # `~` 必须在后端展开：值最终会带引号进 ssh 命令行，bash 不会替我们展开
    assert seen["key"] == str(Path("~/.ssh/id_host").expanduser())


def test_host_ssh_key_upload_writes_private_file(client, tmp_path, monkeypatch):
    """页面「选择文件…」走这里：私钥落到本机 key 目录（0600），只回路径不给内容。"""
    from app.config import settings as app_settings

    c, _ = client
    key_dir = tmp_path / "keys"
    monkeypatch.setattr(app_settings, "ssh_key_dir", str(key_dir))
    pem = b"-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END OPENSSH PRIVATE KEY-----\n"

    r = c.post("/api/hosts/ssh-key", files={"file": ("id_ed25519", pem, "application/octet-stream")})
    assert r.status_code == 200, r.text
    body = r.json()
    path = Path(body["path"])
    assert path.parent == key_dir
    assert path.suffix == ".pem"  # 无扩展名的私钥统一存成 .pem
    assert path.read_bytes() == pem
    assert path.stat().st_mode & 0o777 == 0o600
    assert body["name"] == "id_ed25519"
    assert "PRIVATE KEY" not in str(body)  # 不回显私钥内容


def test_host_ssh_key_upload_rejects_non_key(client, tmp_path, monkeypatch):
    from app.config import settings as app_settings

    c, _ = client
    monkeypatch.setattr(app_settings, "ssh_key_dir", str(tmp_path / "keys"))
    r = c.post("/api/hosts/ssh-key", files={"file": ("note.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert not (tmp_path / "keys").exists()


def test_host_ssh_key_upload_rejects_weak_header_and_oversize(client, tmp_path, monkeypatch):
    """只认 PEM/OpenSSH 私钥头；分块读并在 64KB 处截断（hosts #11）。"""
    from app.config import settings as app_settings

    c, _ = client
    monkeypatch.setattr(app_settings, "ssh_key_dir", str(tmp_path / "keys"))
    # 只含 "PRIVATE KEY-----" 子串、不是合法 PEM/OpenSSH 头（旧实现会误收）
    weak = b"this is not a key -----BEGIN CERTIFICATE----- PRIVATE KEY-----\n"
    r = c.post("/api/hosts/ssh-key", files={"file": ("fake.pem", weak, "application/octet-stream")})
    assert r.status_code == 400
    # 超 64KB：边读边限长拒绝（不用先全量读进内存）
    big = b"-----BEGIN OPENSSH PRIVATE KEY-----\n" + b"A" * (70 * 1024)
    r = c.post("/api/hosts/ssh-key", files={"file": ("big.pem", big, "application/octet-stream")})
    assert r.status_code == 400 and "太大" in r.json()["detail"]


def test_delete_host_removes_images_and_promotes_test_server(client):
    c, Session = client
    a = c.post("/api/hosts", json={"ssh_alias": "del-a"}).json()
    b = c.post("/api/hosts", json={"ssh_alias": "del-b"}).json()
    assert a["is_test_server"] is True
    with Session() as db:
        db.add(HostImage(host_id=a["id"], image_ref="img", tag="t"))
        db.commit()

    r = c.delete(f"/api/hosts/{a['id']}")
    assert r.status_code == 204, r.text
    hosts = c.get("/api/hosts").json()
    assert [h["id"] for h in hosts] == [b["id"]]
    assert hosts[0]["is_test_server"] is True  # 当前测试服务器顺延
    assert c.get(f"/api/hosts/{a['id']}/images").json() == []  # 镜像随 host 级联删除

    assert c.delete(f"/api/hosts/{a['id']}").status_code == 404


def test_delete_host_rejects_when_referenced(client):
    c, Session = client
    h = c.post("/api/hosts", json={"ssh_alias": "del-ref"}).json()
    with Session() as db:
        project = Project(name="p-del-ref", workspace_path="/tmp/ws", host_id=h["id"])
        db.add(project)
        db.flush()
        db.add(Task(project_id=project.id, title="t-del-ref", host_id=h["id"]))
        db.commit()

    r = c.delete(f"/api/hosts/{h['id']}")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "仍被引用" in detail and "项目 1" in detail and "任务 1" in detail
    assert any(x["id"] == h["id"] for x in c.get("/api/hosts").json())


def test_host_ssh_target_rejects_option_injection(client):
    """以 `-` 开头 / 含空白的目标会被 ssh 当选项（页面审查 M1）→ create/update 都 400。"""
    c, _ = client
    for field in ("ssh_alias", "address"):
        assert c.post("/api/hosts", json={field: "-oProxyCommand=touch /tmp/pwn"}).status_code == 400
        assert c.post("/api/hosts", json={field: "root@host -oProxyCommand=x"}).status_code == 400
    assert c.get("/api/hosts").json() == []  # 非法登记不留半行

    h = c.post("/api/hosts", json={"ssh_alias": "safe-host"}).json()
    for field in ("ssh_alias", "address"):
        assert c.patch(f"/api/hosts/{h['id']}", json={field: "-oProxyCommand=x"}).status_code == 400
        assert c.patch(f"/api/hosts/{h['id']}", json={field: "root@h\n-oProxyCommand=x"}).status_code == 400
    # 非法 PATCH 不改动原值
    assert c.get("/api/hosts").json()[0]["ssh_alias"] == "safe-host"
