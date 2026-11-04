"""项目配置（默认执行机/镜像）PATCH 契约 —— 「把镜像设为项目默认」的入口。"""

from __future__ import annotations

from pathlib import Path

from app.models import Host, HostImage, Project


def test_patch_project_default_image(client, workspace: Path):
    c, Session = client
    with Session() as db:
        h1 = Host(name="h1", ssh_alias="h1")
        db.add(h1)
        db.flush()
        h2 = Host(name="h2", ssh_alias="h2")
        db.add(h2)
        db.flush()
        img1 = HostImage(host_id=h1.id, image_ref="opdev", tag="v1", is_op_dev=True, env_recipe="x")
        db.add(img1)
        db.flush()
        img2 = HostImage(host_id=h2.id, image_ref="opdev", tag="v2", is_op_dev=True, env_recipe="x")
        db.add(img2)
        db.flush()
        p = Project(name="p", workspace_path=str(workspace), host_id=h1.id)
        db.add(p)
        db.commit()
        pid, h2_id, img1_id, img2_id = p.id, h2.id, img1.id, img2.id

    r = c.patch(f"/api/projects/{pid}", json={"image_id": img1_id})
    assert r.status_code == 200, r.text
    assert r.json()["image_id"] == img1_id

    # 镜像不属于项目执行机 → 400（避免配置成永远跑不了的一对）
    r = c.patch(f"/api/projects/{pid}", json={"image_id": img2_id})
    assert r.status_code == 400, r.text
    assert "不在执行机" in r.json()["detail"]

    # 只换执行机也不能留下旧机器上的默认镜像。
    r = c.patch(f"/api/projects/{pid}", json={"host_id": h2_id})
    assert r.status_code == 400, r.text
    assert "不在执行机" in r.json()["detail"]

    # 同一请求成对切换可以成功。
    r = c.patch(f"/api/projects/{pid}", json={"host_id": h2_id, "image_id": img2_id})
    assert r.status_code == 200, r.text
    assert r.json()["host_id"] == h2_id and r.json()["image_id"] == img2_id

    # 显式 null 清空回退（项目无默认镜像）
    r = c.patch(f"/api/projects/{pid}", json={"image_id": None})
    assert r.status_code == 200, r.text
    assert r.json()["image_id"] is None

    # 不存在的项目 / 镜像
    assert c.patch("/api/projects/999", json={"image_id": None}).status_code == 404
    r = c.patch(f"/api/projects/{pid}", json={"image_id": 999})
    assert r.status_code == 400


def test_create_project_validates_host_image_pair(client, workspace: Path):
    c, Session = client
    with Session() as db:
        h1 = Host(name="create-h1", ssh_alias="h1")
        h2 = Host(name="create-h2", ssh_alias="h2")
        db.add_all([h1, h2])
        db.flush()
        image = HostImage(host_id=h1.id, image_ref="opdev", tag="v1", is_op_dev=True,
                          env_recipe="x")
        db.add(image)
        db.commit()
        h1_id, h2_id, image_id = h1.id, h2.id, image.id

    bad = c.post("/api/projects", json={"name": "bad", "workspace_path": str(workspace),
                                        "host_id": h2_id, "image_id": image_id})
    assert bad.status_code == 400
    good = c.post("/api/projects", json={"name": "good", "workspace_path": str(workspace),
                                         "host_id": h1_id, "image_id": image_id})
    assert good.status_code == 201, good.text


def test_patch_project_rejects_duplicate_name(client, workspace: Path):
    c, Session = client
    with Session() as db:
        a = Project(name="a", workspace_path=str(workspace))
        db.add(a)
        b = Project(name="b", workspace_path=str(workspace))
        db.add(b)
        db.commit()
        bid = b.id
    r = c.patch(f"/api/projects/{bid}", json={"name": "a"})
    assert r.status_code == 409, r.text
