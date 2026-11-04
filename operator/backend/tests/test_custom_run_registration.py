"""Custom commands must not replace formal correctness evidence."""

import io
import json
from pathlib import Path
import urllib.request

import pytest

from test_api import _seed_api


@pytest.mark.parametrize("status,rc,failure", [("ok", 0, None), ("failed", 1, "execution_failed")])
def test_custom_registration_api_accepts_record_only(client, workspace, status, rc, failure):
    c, Session = client
    task_id, _ = _seed_api(Session, workspace)
    response = c.post(f"/api/tasks/{task_id}/runs", json={
        "action": "custom", "status": status, "exit_code": rc, "failure_kind": failure,
    })
    assert response.status_code == 201, response.text
    assert response.json()["action"] == "custom"
    assert response.json()["failure_kind"] == failure
    actions = c.get("/api/actions").json()
    assert "custom" not in {item["action"] for item in actions["actions"]}
    rejected = c.post(f"/api/tasks/{task_id}/actions", json={"action": "custom", "dry_run": True})
    assert rejected.status_code == 400, rejected.text
    assert c.post(f"/api/tasks/{task_id}/runs", json={"action": "arbitrary"}).status_code == 400


@pytest.mark.parametrize("action", [None, "", "custom", "build", "smoke", "test", "benchmark", "profile"])
@pytest.mark.parametrize("rc", [0, 1, 3])
def test_shell_registration_preserves_action(monkeypatch, tmp_path, action, rc):
    script = (Path(__file__).resolve().parents[2] / "scripts/nm_test.sh").read_text()
    registration = script.split('out, rc, api, task, loop = sys.argv[1:6]', 1)[1].split('\nPY\n', 1)[0]
    manifest = {"action": action, "run_id": "isolated-custom-test", "artifacts": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    captured = []

    def capture(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(b'{"id": 1}')

    monkeypatch.setattr(urllib.request, "urlopen", capture)
    source = "import json, os, sys, urllib.request\n" + registration
    exec(compile(source, "nm_test.sh registration", "exec"), {
        "out": str(tmp_path), "rc": str(rc), "api": "http://example.invalid",
        "task": "14", "loop": "",
    })
    assert len(captured) == 1
    assert captured[0]["action"] == (action or "custom")
    assert captured[0]["status"] == ("ok" if rc == 0 else "failed")
    if rc == 3:
        assert captured[0]["failure_kind"] == "env_failed"
    elif rc == 1 and action in (None, "", "custom"):
        assert captured[0]["failure_kind"] == "execution_failed"
    assert captured[0]["remote_workdir"] == manifest["run_id"]
