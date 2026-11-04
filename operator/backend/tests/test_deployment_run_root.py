"""Relocated deployments must use the same run directory in UI and both execution paths."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import threading

from app.config import settings
from app.services import action_runner, exec_actions


def test_relocated_plan_and_both_execution_paths(db_session, seeded, tmp_path, monkeypatch):
    remote_root = '/data/isolated-console/runs'
    shared_locks = '/data/shared-npu-locks'
    monkeypatch.setattr(settings, 'remote_run_root', remote_root)
    monkeypatch.setenv('OPX_REMOTE_ROOT', '/data/stale-parent/runs')
    monkeypatch.setenv('OPX_LOCK_ROOT', shared_locks)
    task = seeded['task']
    plan = exec_actions.build_plan(db_session, task, 'build', check_device=False)
    assert plan.ready, plan.missing
    assert plan.remote_dir == remote_root + '/<run_id>'
    assert f'OPX_REMOTE_ROOT={remote_root}' in shlex.split(plan.command)
    assert remote_root in plan.steps[2] and remote_root in plan.steps[4]
    env = action_runner.build_env(task, plan, None)
    assert env['OPX_REMOTE_ROOT'] == remote_root
    assert env['OPX_LOCK_ROOT'] == shared_locks

    # Exercise the actual workspace wrapper against a local plan response. Its
    # terminal runner is a stub: this is a regression check, never hardware proof.
    root = tmp_path / 'console'
    ws = root / 'workspaces/cann-ops'
    (ws / 'scripts').mkdir(parents=True)
    (root / 'scripts').mkdir()
    source = Path(__file__).resolve().parents[2]
    shutil.copy2(source / 'workspaces/cann-ops/scripts/nm.sh', ws / 'scripts/nm.sh')
    stub = root / 'scripts/nm_test.sh'
    stub.write_text('#!/bin/sh\nprintf "RUN_ROOT=%s\\nLOCK_ROOT=%s\\n" "$OPX_REMOTE_ROOT" "$OPX_LOCK_ROOT"\n')
    stub.chmod(0o755)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == f'/api/tasks/{task.id}/actions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body == {'action': 'build', 'dry_run': True}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(plan.as_dict()).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            ['bash', str(ws / 'scripts/nm.sh'), '--task', str(task.id), '--action', 'build'],
            env={**os.environ, 'OPX_API': f'http://127.0.0.1:{server.server_port}'},
            capture_output=True, text=True, timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stderr + result.stdout
    assert f'RUN_ROOT={remote_root}\nLOCK_ROOT={shared_locks}' in result.stdout
