"""服务器探针与镜像清单（task-5：页面审查 pages-hosts.md 的 M8/M9/M10/M11/M12/#10/#12/#13）。

覆盖：
- 探针失败清空旧机派生字段（cards/CANN/docker_ok）并把旧镜像置 stale，probe_error 可回读；
- docker 命令 rc=0 但 0 个镜像 → docker_ok=True（不能显示「docker 不可用」）；
- 镜像清单按 image_id 对账：多张 `<none>` 悬空镜像互不覆盖、远端已删的镜像删除/标记 missing、
  被 Task 引用的镜像保留；
- 镜像 PATCH：元数据可写+截断、op_dev 必须带 recipe、悬空镜像禁止标记；
- SSH 目标选项注入的防御（probe.ssh_exec / devices.check_devices）与 ssh 不存在不冒泡 500；
- sync_image_meta.py 走 services.ssh 同源解析（目标/端口/私钥）。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Host, HostImage, Project, Task
from app.services import probe as probe_mod
from app.services.devices import check_devices
from app.services.probe import ProbeResult, parse_docker_images, parse_npu_smi

#: 能被 parse_npu_smi 识别的输出（芯片行 `| 0  910B1 | OK | …`）
NPU_OK = """npu-smi 23.0.rc1
+---------------------------+
| 0     910B1               | OK            | 0.0         0                 0    / 0             |
| 0                         | 0000:01:00.0  | 0           0 / 0            1000 / 32768          |
+---------------------------+
"""

NPU_WITH_PROCESS_TABLE = """npu-smi 26.0.rc1 Version: 26.0.rc1
| NPU   Name                | Health        | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip                      | Bus-Id        | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 0     910B1               | OK            | 103.2 31 0 / 0 |
| 0                         | 0000:C1:00.0  | 0 0 / 0 44537 / 65536 |
| 1     910B1               | OK            | 101.6 32 0 / 0 |
| 0                         | 0000:82:00.0  | 0 0 / 0 3470 / 65536 |
| NPU     Chip              | Process id    | Process name | Process memory(MB) |
| 0       0                 | 3588388       | VLLMEngineCore | 11868 |
| 0       0                 | 3589701       | python3        | 112 |
"""


def _row(repo: str, tag: str, image_id: str, size: str = "1GB") -> str:
    return f"{repo}\t{tag}\t{image_id}\t2024-01-01 00:00:00 +0800\t{size}"


def _images(*rows: str) -> list[dict]:
    return parse_docker_images("\n".join(rows))


def _create_host(c, target: str = "tk36") -> dict:
    r = c.post("/api/hosts", json={"ssh_alias": target})
    assert r.status_code == 201, r.text
    return r.json()


def _probe(c, monkeypatch, host_id: int, result: ProbeResult) -> None:
    from app.routers import hosts as hosts_router

    monkeypatch.setattr(hosts_router, "probe_host", lambda *a, **kw: result)
    r = c.post(f"/api/hosts/{host_id}/probe")
    assert r.status_code == 200, r.text


def _ok_probe(images: list[dict] | None = None, *, cards: str | None = "8×910B1") -> ProbeResult:
    return ProbeResult(ok=True, hostname="tk36", cards=cards, cann_version="cann-9.2.0",
                       docker_ok=True, npu_raw=NPU_OK, images=list(images or []))


def test_parse_npu_smi_ignores_modern_process_table():
    info = parse_npu_smi(NPU_WITH_PROCESS_TABLE)
    assert info.cards == "2×910B1"
    assert [chip["index"] for chip in info.chips] == [0, 1]
    assert info.hbm_total_mb == 65536
    assert info.hbm_used_mb == 3470


# —— SSH 目标防御 ——

def test_ssh_exec_rejects_option_target_and_missing_ssh(monkeypatch):
    assert probe_mod.ssh_exec("-oProxyCommand=touch /tmp/pwn", "hostname", "", "", 5) == (
        2, "invalid ssh target")
    assert probe_mod.ssh_exec("root@host extra", "hostname", "", "", 5) == (2, "invalid ssh target")

    def boom(*a, **kw):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(probe_mod.subprocess, "run", boom)
    rc, out = probe_mod.ssh_exec("tk36", "hostname", "", "", 5)
    assert rc == 127 and "ssh" in out  # 不冒泡 500（审查 hosts #10）


def test_check_devices_rejects_option_target(monkeypatch):
    def should_not_run(*a, **kw):
        raise AssertionError("非法目标不应该真的执行 ssh")

    monkeypatch.setattr("app.services.devices.subprocess.run", should_not_run)
    report = check_devices("-oProxyCommand=touch /tmp/pwn", "", "", 8)
    assert report.ok is False
    assert report.error == "invalid ssh target"


# —— docker_ok 语义（M8）——

def test_probe_host_docker_zero_images_is_ok(monkeypatch):
    def fake_ssh(alias, command, key, opts, timeout=25, port=None):
        if command == probe_mod.HOSTNAME_CMD:
            return 0, "tk36"
        if command == probe_mod.NPU_SMI_CMD:
            return 0, NPU_OK
        if command == probe_mod.CANN_VER_CMD:
            return 0, "cann-9.2.0"
        return 0, ""  # docker images 成功，但一台机器上 0 个镜像

    monkeypatch.setattr(probe_mod, "ssh_exec", fake_ssh)
    res = probe_mod.probe_host("tk36", "", "", 5)
    assert res.ok is True
    assert res.docker_ok is True  # 0 镜像 ≠ docker 不可用
    assert res.images == []
    assert res.cards == "1×910B1"
    assert not res.errors


def test_probe_host_docker_failure_sets_docker_ok_false(monkeypatch):
    docker_rc = {"value": 0}

    def fake_ssh(alias, command, key, opts, timeout=25, port=None):
        if command == probe_mod.HOSTNAME_CMD:
            return 0, "tk36"
        if command == probe_mod.NPU_SMI_CMD:
            return 0, NPU_OK
        if command == probe_mod.CANN_VER_CMD:
            return 0, ""
        # 远端命令带 `| head`：docker 失败时管道 rc 仍可能是 0，只能靠输出识别
        return docker_rc["value"], "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."

    monkeypatch.setattr(probe_mod, "ssh_exec", fake_ssh)
    res = probe_mod.probe_host("tk36", "", "", 5)
    assert res.docker_ok is False
    assert any("docker images" in e for e in res.errors)
    assert res.images == []

    docker_rc["value"] = 1  # 命令本身失败（例如没有 docker）
    res = probe_mod.probe_host("tk36", "", "", 5)
    assert res.docker_ok is False
    assert any("docker images 失败" in e for e in res.errors)


# —— 探针失败：清派生字段 + 旧镜像 stale（M11 / 审查 #4）——

def test_probe_failure_clears_derived_fields_and_marks_images_stale(client, monkeypatch):
    from app.routers import hosts as hosts_router

    c, Session = client
    h = _create_host(c, "tk37")
    with Session() as db:
        host = db.get(Host, h["id"])
        host.cards = "8×910B1"
        host.cann_version = "cann-9.2.0"
        host.docker_ok = True
        host.probe_state = "ok"
        host.npu_state = "old machine raw"
        db.add(HostImage(host_id=host.id, image_ref="img-a", tag="v1", image_id="sha256:aaa"))
        db.commit()

    monkeypatch.setattr(
        hosts_router, "probe_host",
        lambda *a, **kw: ProbeResult(ok=False, errors=["ssh 失败（rc=255）：Permission denied (publickey)"]),
    )
    r = c.post(f"/api/hosts/{h['id']}/probe")
    assert r.status_code == 502

    body = c.get("/api/hosts").json()[0]
    assert body["probe_state"] == "fail"
    assert body["cards"] is None
    assert body["cann_version"] is None
    assert body["docker_ok"] is False
    assert body["probe_error"] and "Permission denied" in body["probe_error"]
    assert len(body["probe_error"]) <= 500

    images = c.get(f"/api/hosts/{h['id']}/images").json()
    assert [i["scan_state"] for i in images] == ["stale"]


def test_probe_success_replaces_old_machine_cards(client, monkeypatch):
    c, Session = client
    h = _create_host(c, "tk41")
    with Session() as db:
        host = db.get(Host, h["id"])
        host.cards = "8×910B1"
        host.cann_version = "cann-old"
        db.commit()
    _probe(c, monkeypatch, h["id"], _ok_probe([], cards=None))
    body = c.get("/api/hosts").json()[0]
    # 本机真实结果覆盖旧值：npu-smi 没返回卡 → cards 清空；CANN 用本轮扫描值
    assert body["cards"] is None and body["cann_version"] == "cann-9.2.0"
    assert body["probe_state"] == "ok"


# —— 镜像清单对账（M12 / 审查 #5）——

def test_probe_dangling_images_do_not_collide(client, monkeypatch):
    c, _ = client
    h = _create_host(c, "tk39")
    scan = _images(
        _row("<none>", "<none>", "sha256:aaaa", "1000MB"),
        _row("<none>", "<none>", "sha256:bbbb", "2000MB"),
        _row("img-a", "v1", "sha256:cccc"),
    )
    _probe(c, monkeypatch, h["id"], _ok_probe(scan))
    images = c.get(f"/api/hosts/{h['id']}/images").json()
    assert len(images) == 3  # 4 行旧库同键的根因：不能用 (image_ref, tag) 做键
    assert {i["image_id"] for i in images} == {"sha256:aaaa", "sha256:bbbb", "sha256:cccc"}
    assert {i["scan_state"] for i in images} == {"scanned"}

    # 重复探针幂等：不新增行
    _probe(c, monkeypatch, h["id"], _ok_probe(scan))
    assert len(c.get(f"/api/hosts/{h['id']}/images").json()) == 3


def test_probe_reconciles_images_removed_remotely(client, monkeypatch):
    c, Session = client
    h = _create_host(c, "tk40")
    scan1 = _images(
        _row("img-a", "v1", "sha256:aaa"),
        _row("img-b", "v1", "sha256:bbb"),
        _row("img-c", "v1", "sha256:ccc"),
    )
    _probe(c, monkeypatch, h["id"], _ok_probe(scan1))
    ids = {i["image_ref"]: i["id"] for i in c.get(f"/api/hosts/{h['id']}/images").json()}
    assert set(ids) == {"img-a", "img-b", "img-c"}

    with Session() as db:
        # b：有测试标记 → 远端没了也要保留并标 missing
        db.get(HostImage, ids["img-b"]).is_test = True
        # c：被任务引用 → 不能硬删（否则外键悬空），保留并标 missing
        project = Project(name="p-img-ref", workspace_path="/tmp/ws")
        db.add(project)
        db.flush()
        db.add(Task(project_id=project.id, title="t-img-ref", image_id=ids["img-c"]))
        db.commit()

    _probe(c, monkeypatch, h["id"], _ok_probe(_images(_row("img-a", "v1", "sha256:aaa"))))
    images = c.get(f"/api/hosts/{h['id']}/images").json()
    by_ref = {i["image_ref"]: i for i in images}
    # 无标记的 b 被删 → 不在结果里；被引用的 c / 有标记的 b 不在（b 有标记）
    assert set(by_ref) == {"img-a", "img-b", "img-c"}
    assert by_ref["img-a"]["id"] == ids["img-a"]  # 已有行原地更新，不重建
    assert by_ref["img-a"]["scan_state"] == "scanned"
    assert by_ref["img-b"]["scan_state"] == "missing"
    assert by_ref["img-c"]["scan_state"] == "missing"

    # 再把 c 的引用去掉后，下一次探针既无标记也无引用 → 删除
    with Session() as db:
        db.query(Task).delete()
        db.commit()
    _probe(c, monkeypatch, h["id"], _ok_probe(_images(_row("img-a", "v1", "sha256:aaa"))))
    refs = {i["image_ref"] for i in c.get(f"/api/hosts/{h['id']}/images").json()}
    assert refs == {"img-a", "img-b"}  # b 仍带测试标记


def test_probe_without_docker_keeps_images(client, monkeypatch):
    """探针部分成功（卡拿到了但 docker 命令失败）时不能把旧镜像清空/删除。"""
    c, _ = client
    h = _create_host(c, "tk42")
    _probe(c, monkeypatch, h["id"], _ok_probe(_images(_row("img-a", "v1", "sha256:aaa"))))
    partial = ProbeResult(ok=True, hostname="tk36", cards="8×910B1", docker_ok=False,
                          npu_raw=NPU_OK, images=[], errors=["docker images 失败（rc=1）"])
    _probe(c, monkeypatch, h["id"], partial)
    images = c.get(f"/api/hosts/{h['id']}/images").json()
    assert [i["image_ref"] for i in images] == ["img-a"]  # 命令失败 ≠ 0 个镜像
    assert images[0]["scan_state"] == "stale"


# —— 镜像 PATCH（M9 / M10 / #13）——

def _image(c: object, Session, host_id: int, **kwargs) -> int:
    with Session() as db:
        img = HostImage(host_id=host_id, image_ref=kwargs.pop("image_ref", "img"),
                        tag=kwargs.pop("tag", "t"), **kwargs)
        db.add(img)
        db.commit()
        return img.id


def test_patch_image_metadata_editable_and_truncated(client):
    c, Session = client
    h = _create_host(c, "tk43")
    img_id = _image(c, Session, h["id"])

    r = c.patch(f"/api/host-images/{img_id}", json={
        "digest": "  sha256:abc  ", "cann_version": "cann-9.2.0",
        "python_version": "3.10", "torch_version": "2.10.0",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["digest"], body["cann_version"], body["python_version"], body["torch_version"]) == (
        "sha256:abc", "cann-9.2.0", "3.10", "2.10.0")
    assert c.get(f"/api/hosts/{h['id']}/images").json()[0]["torch_version"] == "2.10.0"

    # 超长按列长度截断（digest 200 / cann 120 / python 60 / torch 60）
    body = c.patch(f"/api/host-images/{img_id}", json={
        "digest": "d" * 500, "cann_version": "c" * 500,
        "python_version": "p" * 500, "torch_version": "t" * 500,
    }).json()
    assert [len(body[k]) for k in ("digest", "cann_version", "python_version", "torch_version")] == [
        200, 120, 60, 60]


def test_patch_image_op_dev_requires_recipe(client):
    c, Session = client
    h = _create_host(c, "tk44")
    img_id = _image(c, Session, h["id"])

    assert c.patch(f"/api/host-images/{img_id}", json={"is_op_dev": True}).status_code == 400
    r = c.patch(f"/api/host-images/{img_id}",
                json={"is_op_dev": True, "env_recipe": "  docker run --privileged img  "})
    assert r.status_code == 200, r.text
    assert r.json()["env_recipe"] == "docker run --privileged img"

    # 只改元数据不碰 recipe（页面审查 M9：不能无条件覆盖既有配方）
    r = c.patch(f"/api/host-images/{img_id}", json={"note": "补一句备注"})
    assert r.status_code == 200
    assert r.json()["is_op_dev"] is True and r.json()["env_recipe"] == "docker run --privileged img"

    # 想清空 recipe 就必须同时取消算子开发标记
    assert c.patch(f"/api/host-images/{img_id}", json={"env_recipe": None}).status_code == 400
    r = c.patch(f"/api/host-images/{img_id}", json={"is_op_dev": False, "env_recipe": None})
    assert r.status_code == 200
    assert r.json()["is_op_dev"] is False and r.json()["env_recipe"] is None


def test_patch_dangling_image_cannot_be_marked(client):
    c, Session = client
    h = _create_host(c, "tk45")
    img_id = _image(c, Session, h["id"], image_ref="<none>", tag=None,
                    image_id="sha256:dangling", dangling=True)

    assert c.patch(f"/api/host-images/{img_id}", json={"is_test": True}).status_code == 400
    assert c.patch(f"/api/host-images/{img_id}",
                   json={"is_op_dev": True, "env_recipe": "docker run ..."}).status_code == 400
    # 悬空镜像仍可补元数据（不会误伤编辑）
    r = c.patch(f"/api/host-images/{img_id}", json={"digest": "sha256:d"})
    assert r.status_code == 200 and r.json()["dangling"] is True


# —— sync_image_meta.py：与页面/执行同源解析（#12）——

def test_probe_rejects_concurrent_same_host(client):
    """per-host 探针锁：并发探针返回 409，不重复 SSH/插镜像（hosts #7）。"""
    from app.routers import hosts as hosts_router

    c, _ = client
    host = _create_host(c, "lock-host")
    lock = hosts_router._probe_lock(host["id"])
    assert lock.acquire(blocking=False) is True
    try:
        r = c.post(f"/api/hosts/{host['id']}/probe")
        assert r.status_code == 409 and "探针中" in r.json()["detail"]
    finally:
        lock.release()


def test_probe_partial_errors_are_surfaced(client, monkeypatch):
    """探针部分成功（卡拿到、docker 失败）也要留告警，不能静默（hosts #9）。"""
    c, _ = client
    host = _create_host(c, "partial-host")
    from app.routers import hosts as hosts_router

    result = ProbeResult(ok=True, hostname="tk36", cards="8×910B1", cann_version="cann-9.2.0",
                         docker_ok=False, npu_raw=NPU_OK, images=[],
                         errors=["docker images 失败（rc=1）：permission denied"])
    monkeypatch.setattr(hosts_router, "probe_host", lambda *a, **kw: result)
    r = c.post(f"/api/hosts/{host['id']}/probe")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["probe_state"] == "ok"
    assert "docker images 失败" in (body.get("probe_error") or "")


def test_sync_image_meta_uses_shared_ssh_resolution(tmp_path, monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        sync = importlib.import_module("sync_image_meta")
    finally:
        sys.path.remove(str(scripts_dir))

    engine = create_engine(f"sqlite:///{tmp_path / 'sync.db'}")
    Base.metadata.create_all(engine)
    SyncSession = sessionmaker(bind=engine)
    with SyncSession() as db:
        host = Host(name="tk-port", ssh_alias="root@1.2.3.4:50023", cred_ref="~/.ssh/id_host")
        db.add(host)
        db.flush()
        db.add(HostImage(host_id=host.id, image_ref="img", tag="t"))
        db.commit()

    seen: list[dict] = []

    def fake_ssh(alias, command, key, opts, timeout=25, port=None):
        seen.append({"alias": alias, "key": key, "port": port, "cmd": command})
        return 0, ""

    monkeypatch.setattr(sync, "SessionLocal", SyncSession)
    monkeypatch.setattr(sync, "ssh_exec", fake_ssh)
    monkeypatch.setattr(sys, "argv", ["sync_image_meta.py", "--dry-run"])
    assert sync.main() == 0

    assert seen, "应该至少发一次只读 ssh"
    first = seen[0]
    assert first["alias"] == "root@1.2.3.4"  # 端口不能留在目标里（否则 25s 超时）
    assert first["port"] == 50023
    assert first["key"] == str(Path("~/.ssh/id_host").expanduser())  # host.cred_ref 优先
    assert "docker images" in first["cmd"]  # 只读语义
