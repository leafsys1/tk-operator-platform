"""硬件画像读侧（方案 §13-A11）：服务的列出/读取/按芯片选 + 端点。"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from app.models import Host, HostImage, Project  # noqa: E402
from app.services import chips  # noqa: E402

DOC = {
    "schema": "opx-chip-profile/1",
    "arch": "DAV_2201",
    "npu_arch_flag": "dav-2201",
    "product_family": "Atlas A2",
    "models": ["Ascend910B1", "Ascend910B2"],
    "model_variable": {"hbm_bandwidth_gbps": {"value": 1600}},
    "compute": {"vector_fp32_peak_tflops": {"value": 5.68}},
}


def _write_chip(ws: Path, name: str = "dav-2201", doc: dict | None = None) -> Path:
    d = ws / "ops" / "chips"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc or DOC, allow_unicode=True), encoding="utf-8")
    return p


def test_list_load_and_find(tmp_path: Path):
    _write_chip(tmp_path)
    lst = chips.list_chips(tmp_path)
    assert lst and lst[0]["name"] == "dav-2201"
    assert "Ascend910B1" in lst[0]["models"]

    doc = chips.load_chip(tmp_path, "dav-2201")
    assert doc["compute"]["vector_fp32_peak_tflops"]["value"] == 5.68
    assert chips.load_chip(tmp_path, "missing") is None

    # 按目标芯片选（910B1 → models 命中）
    assert chips.find_chip_for(tmp_path, "910B1")["arch"] == "DAV_2201"
    # 没命中 → 退第一个；空目录 → None
    assert chips.find_chip_for(tmp_path, "310P")["arch"] == "DAV_2201"
    assert chips.find_chip_for(tmp_path / "empty", "910B1") is None
    assert chips.list_chips(tmp_path / "empty") == []


def test_chips_endpoints(client, workspace: Path):
    c, Session = client
    _write_chip(workspace)

    # 端点要求工作区来自 project（与 /api/chips 的实现一致）
    assert c.get("/api/chips").json() == []          # 还没有项目 → 空
    with Session() as db:
        host = Host(name="tk36", ssh_alias="tk36")
        db.add(host)
        db.flush()
        img = HostImage(host_id=host.id, image_ref="x", tag="y")
        db.add(img)
        db.flush()
        db.add(Project(name="p", workspace_path=str(workspace), host_id=host.id, image_id=img.id))
        db.commit()

    body = c.get("/api/chips").json()
    assert body and body[0]["name"] == "dav-2201"
    full = c.get("/api/chips/dav-2201").json()
    assert full["schema"] == "opx-chip-profile/1"
    assert c.get("/api/chips/nope").status_code == 404
