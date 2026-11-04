"""Exercise the real provider launcher with a fake DSH, never a real model or NPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import shlex

import pytest


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "app"
    workspace = repo / "workspaces/cann-ops"
    work = workspace / ".cannbot/demo/workflow1"
    work.mkdir(parents=True)
    (workspace / ".dsh/agents").mkdir(parents=True)
    (workspace / ".dsh/agents/ops-direct-invoke-developer.md").write_text("Developer role")
    (repo / "backend/.runtime").mkdir(parents=True)
    (repo / "backend/.runtime/dsh-llm.patch.yml").write_text("[]\n")
    profile = tmp_path / "parent-dsh/profiles/headless"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text('{"dsh":{"profile":{"bundles":[]}}}')
    (profile / "cordis.patch.yml").write_text('- id: sandbox-policy\n  config:\n    mode: read-only\n')
    (profile / "cordis.yml").write_text("# parent profile must remain untouched\n[]\n")
    (profile / "cordis.yml").chmod(0o444)
    (repo / "backend/.venv/bin").mkdir(parents=True)
    python = repo / "backend/.venv/bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    (repo / "scripts").mkdir()
    source = Path(__file__).resolve().parents[2] / "scripts"
    for name in ["cannbot", "cannbot_dsh_launcher.py"]:
        shutil.copy2(source / name, repo / "scripts" / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "dsh"
    fake.write_text(f"#!{sys.executable}\n" + '''import json,os,sys,hashlib
from pathlib import Path
Path(os.environ['DSH_PROFILE_DIR'],'cordis.yml').write_text('[]')
args=sys.argv[1:]
patches=[json.loads(Path(args[i+1]).read_text()) for i,a in enumerate(args) if a=='--patch']
print(json.dumps({'cwd':os.getcwd(),'args':args,'patches':patches,
 'work_dir':os.environ.get('WORK_DIR'),
 'key_present':bool(os.environ.get('OPX_LLM_KEY')),
 'key_digest':hashlib.sha256(os.environ.get('OPX_LLM_KEY','').encode()).hexdigest(),
 'home':os.environ.get('DSH_HOME'),'profile':os.environ.get('DSH_PROFILE_DIR'),
 'inherited_session':os.environ.get('DSH_SESSION_ID')}))
''')
    fake.chmod(0o755)
    env = {**os.environ, "OPX_REPO_ROOT": str(repo), "OPX_WORKSPACE_ROOT": str(workspace),
           "OPX_LLM_KEY": "test-only-not-a-real-key",
           "DSH_HOME": str(profile.parents[1]), "DSH_PROFILE_DIR": str(profile),
           "DSH_SESSION_ID": "parent-session-must-not-be-reused",
           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    argv = [str(repo / "scripts/cannbot"), "run", "--format", "json", "--agent",
            "ops-direct-invoke-developer", "Generate the authorized operator only.\nKeep evidence truthful."]
    return repo, workspace, work, env, argv


def test_nested_workflow_applies_granted_scope_and_isolates_runtime(launcher):
    repo, workspace, work, env, argv = launcher
    result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["cwd"] == str(workspace)
    assert data["work_dir"] == str(work)
    assert data["patches"] == [[], [{"id": "sandbox-policy", "config": {
        "mode": "workspace-write", "workspaceRoot": str(workspace),
    }}]]
    assert data["key_present"]
    assert env["OPX_LLM_KEY"] not in result.stdout + result.stderr
    assert str(workspace / ".dsh/agents/ops-direct-invoke-developer.md") in data["args"][-1]
    assert str(work) in data["args"][-1]
    assert (work / ".opx-runtime/dsh-workspace.patch.json").stat().st_mode & 0o777 == 0o600
    child_home = Path(data["home"])
    assert child_home.is_relative_to(work / ".opx-runtime")
    assert data["profile"] == str(child_home / "profiles/headless")
    assert data["inherited_session"] is None
    assert (Path(data["profile"]) / "cordis.patch.yml").read_bytes() == (
        Path(env["DSH_PROFILE_DIR"]) / "cordis.patch.yml"
    ).read_bytes()
    second = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=10)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["home"] != data["home"]
    assert (Path(env["DSH_PROFILE_DIR"]) / "cordis.yml").read_text() == "# parent profile must remain untouched\n[]\n"


def test_missing_inherited_profile_fails_instead_of_resetting_policy(launcher):
    repo, workspace, work, env, argv = launcher
    env.pop("DSH_PROFILE_DIR")
    env.pop("DSH_HOME")
    result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "refusing to change the child policy" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("invalid", [
    "missing_root", "outside_project", "all_projects", "project_root_symlink",
    "outside_cwd", "role_escape", "role_symlink", "patch_escape",
])
def test_launcher_fails_closed_for_invalid_scope(launcher, tmp_path, invalid):
    repo, workspace, work, env, argv = launcher
    outside = tmp_path / "unrelated"
    outside.mkdir()
    if invalid == "missing_root":
        env.pop("OPX_WORKSPACE_ROOT")
    elif invalid == "outside_project":
        env["OPX_WORKSPACE_ROOT"] = str(outside)
    elif invalid == "all_projects":
        env["OPX_WORKSPACE_ROOT"] = str(repo / "workspaces")
    elif invalid == "project_root_symlink":
        moved = tmp_path / "relocated-projects"
        (repo / "workspaces").rename(moved)
        (repo / "workspaces").symlink_to(moved, target_is_directory=True)
    elif invalid == "outside_cwd":
        work = outside
    elif invalid == "role_escape":
        argv[-2] = "../../outside"
    elif invalid == "role_symlink":
        external = tmp_path / "external-role.md"
        external.write_text("External role")
        role = workspace / ".dsh/agents/ops-direct-invoke-developer.md"
        role.unlink()
        role.symlink_to(external)
    elif invalid == "patch_escape":
        (work / ".opx-runtime").symlink_to(outside, target_is_directory=True)
    result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not result.stdout
    assert env["OPX_LLM_KEY"] not in result.stderr
    assert not list(outside.iterdir())


@pytest.mark.parametrize("forward_store_config", [True, False])
def test_sanitized_shell_can_resolve_relocated_store_without_forwarding_key(
    launcher, tmp_path, monkeypatch, forward_store_config,
):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.config import settings
    from app.models import Base, Endpoint
    from app.services import agent_runner, secret_store

    repo, workspace, work, original_env, argv = launcher
    store = tmp_path / "deployment-store"
    monkeypatch.setattr(settings, "secret_dir", str(store))
    monkeypatch.setattr(settings, "secret_backend", "file")
    key = "relocated-test-value-never-log"
    secret_store.set_secret("endpoint-child-test", key)
    db_url = f"sqlite:///{tmp_path / 'child.db'}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Endpoint(name="test", provider="dsh-llm-deepseek", is_active=True,
                        secret_ref="endpoint-child-test"))
        db.commit()
    engine.dispose()
    # Use the actual DB and secret-store modules in the launcher's subprocess.
    (repo / "backend/app").symlink_to(Path(__file__).resolve().parents[1] / "app",
                                      target_is_directory=True)
    env = agent_runner.provider_runtime_env(original_env)
    # Reproduce the shell's sensitive-name filter, including the non-secret
    # OPX_SECRET_DIR that caused the observed server child startup failure.
    env = {k: v for k, v in env.items()
           if not any(word in k.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(OPX_DB_URL=db_url, HOME=str(tmp_path / "empty-home"))
    if not forward_store_config:
        env.pop("OPX_PROVIDER_STORE_DIR")
        env.pop("OPX_PROVIDER_STORE_BACKEND")
    before = {p.name: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=20)
    assert key not in result.stdout + result.stderr
    assert not any(key in arg for arg in argv)
    if forward_store_config:
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["key_digest"] == hashlib.sha256(key.encode()).hexdigest()
        assert data["cwd"] == str(workspace)
        assert data["work_dir"] == str(work)
        assert data["patches"][-1][0]["config"] == {
            "mode": "workspace-write", "workspaceRoot": str(workspace),
        }
    else:
        assert result.returncode == 2
        assert "cannot load the active endpoint key" in result.stderr
        assert not result.stdout
    assert before == {p.name: p.read_bytes() for p in store.rglob("*") if p.is_file()}
