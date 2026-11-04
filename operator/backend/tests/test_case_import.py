"""案例库导入脚本：bundle 校验 / 解包 / 索引（不依赖网络与控制台）。"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_cases.py"
_spec = importlib.util.spec_from_file_location("import_cases", SCRIPT)
import_cases = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(import_cases)


def _zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return path


def _manifest(cases=None, name="demo") -> str:
    man = {"schema": "opx-case-library/1", "name": name, "version": "2026.10"}
    if cases is not None:
        man["cases"] = cases
    return json.dumps(man)


def test_import_valid_bundle(tmp_path: Path):
    z = _zip(tmp_path / "cases.zip", {
        "manifest.json": _manifest([{"op": "relu", "scenario": "激活", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "c1"}]}),
        "cases/relu/README.md": "relu",
        "outside.txt": "should be skipped",
    })
    dest = tmp_path / "lib"
    res = import_cases.import_bundle(z, dest)
    assert res["name"] == "demo" and [c["op"] for c in res["cases"]] == ["relu"]
    assert (dest / "demo" / "cases" / "relu" / "cases.json").is_file()
    assert not (dest / "demo" / "outside.txt").exists()  # 只解 cases/ 与 manifest
    idx = json.loads((dest / "index.json").read_text())
    assert idx["bundles"][0]["name"] == "demo" and idx["bundles"][0]["version"] == "2026.10"


def test_manifest_cases_optional_inferred(tmp_path: Path):
    z = _zip(tmp_path / "cases.zip", {
        "manifest.json": _manifest(),
        "cases/relu/cases.json": json.dumps({"cases": []}),
        "cases/relu/cases.json.bak": "x",
        "cases/softmax/cases.json": json.dumps({"cases": []}),
    })
    res = import_cases.import_bundle(z, tmp_path / "lib")
    assert sorted(c["op"] for c in res["cases"]) == ["relu", "softmax"]


def test_rejects_path_traversal(tmp_path: Path):
    z = _zip(tmp_path / "evil.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": []}),
        "../evil.txt": "x",
    })
    with pytest.raises(import_cases.BundleError, match="不安全"):
        import_cases.import_bundle(z, tmp_path / "lib")


def test_rejects_missing_manifest_or_cases_json(tmp_path: Path):
    z1 = _zip(tmp_path / "a.zip", {"cases/relu/cases.json": "{}"})
    with pytest.raises(import_cases.BundleError, match="manifest"):
        import_cases.import_bundle(z1, tmp_path / "lib")
    z2 = _zip(tmp_path / "b.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/other.txt": "x",
    })
    with pytest.raises(import_cases.BundleError, match="cases.json"):
        import_cases.import_bundle(z2, tmp_path / "lib")


def test_force_overwrites_and_index_dedups(tmp_path: Path):
    dest = tmp_path / "lib"
    z1 = _zip(tmp_path / "v1.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "v1"}]}),
    })
    import_cases.import_bundle(z1, dest)
    with pytest.raises(import_cases.BundleError, match="已存在"):
        import_cases.import_bundle(z1, dest)
    z2 = _zip(tmp_path / "v2.zip", {
        "manifest.json": _manifest([{"op": "relu", "path": "cases/relu"}]),
        "cases/relu/cases.json": json.dumps({"cases": [{"case_id": "v2"}]}),
    })
    import_cases.import_bundle(z2, dest, force=True)
    data = json.loads((dest / "relu" / "cases.json").read_text()) if (dest / "relu" / "cases.json").exists() else None
    # 实际落点：dest/demo/cases/relu/cases.json
    body = json.loads((dest / "demo" / "cases" / "relu" / "cases.json").read_text())
    assert body["cases"][0]["case_id"] == "v2"
    idx = json.loads((dest / "index.json").read_text())
    assert len([b for b in idx["bundles"] if b["name"] == "demo"]) == 1
