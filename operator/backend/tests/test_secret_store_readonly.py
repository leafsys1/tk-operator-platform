"""Credential reads must work in a process without secret-directory write access."""
import json
from pathlib import Path

import pytest
from app.services import secret_store


@pytest.mark.parametrize("indexed", [True, False])
def test_file_secret_reads_without_mkdir_chmod_or_index_write(tmp_path, monkeypatch, indexed):
    root = tmp_path / "secrets"
    monkeypatch.setattr(secret_store.settings, "secret_dir", str(root))
    monkeypatch.setattr(secret_store.settings, "secret_backend", "file")
    secret_store.set_secret("example", "test-placeholder")
    if not indexed:
        (root / "index.json").unlink()
    def denied(*args, **kwargs):
        raise PermissionError("read-only secret store")
    monkeypatch.setattr(Path, "mkdir", denied)
    monkeypatch.setattr(Path, "chmod", denied)
    monkeypatch.setattr(secret_store, "_save_index", denied)
    assert secret_store.get_secret("example") == "test-placeholder"
    assert secret_store.has_secret("example")
    assert secret_store.get_secret("missing") is None
    assert secret_store.backend_for("example") == ("file" if indexed else None)


def test_keychain_read_does_not_modify_secret_directory(tmp_path, monkeypatch):
    root = tmp_path / "secrets"
    root.mkdir()
    (root / "index.json").write_text(json.dumps({"example": "keychain"}))
    monkeypatch.setattr(secret_store.settings, "secret_dir", str(root))
    monkeypatch.setattr(secret_store, "_keychain_read", lambda ref: "test-placeholder")
    def denied(*args, **kwargs):
        raise PermissionError("read-only secret store")
    monkeypatch.setattr(Path, "mkdir", denied)
    monkeypatch.setattr(Path, "chmod", denied)
    assert secret_store.get_secret("example") == "test-placeholder"
