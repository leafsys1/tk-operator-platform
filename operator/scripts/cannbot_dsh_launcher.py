"""Run a workflow child from its explicitly authorized project root.

OPX_WORKSPACE_ROOT is a deployment authorization boundary, not inferred from cwd.
Configure it only for a bundled project explicitly authorized for workspace-write.
This adapter leaves approval policy intact and never promotes staged outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def configured_directory(name: str) -> Path:
    value = os.environ.get(name, "")
    path = Path(value)
    if not value or not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{name} must name an explicitly configured absolute directory")
    return path.resolve()


def prepare_launch(agent: str, prompt: str) -> tuple[list[str], dict[str, str]]:
    repo = configured_directory("OPX_REPO_ROOT")
    workspace = configured_directory("OPX_WORKSPACE_ROOT")
    cwd = Path.cwd().resolve()
    projects = (repo / "workspaces").resolve()
    if not projects.is_relative_to(repo) or workspace == projects or not workspace.is_relative_to(projects):
        raise ValueError("OPX_WORKSPACE_ROOT must be a project below OPX_REPO_ROOT/workspaces")
    if not cwd.is_relative_to(workspace):
        raise ValueError("workflow directory is outside the configured project")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", agent):
        raise ValueError("invalid workflow role name")
    role = (workspace / ".dsh" / "agents" / f"{agent}.md").resolve()
    if not role.is_file() or not role.is_relative_to(workspace):
        raise ValueError("workflow role must exist inside the configured project")
    llm_patch = repo / "backend/.runtime/dsh-llm.patch.yml"
    if not llm_patch.is_file():
        raise ValueError("missing generated DSH model patch")

    # Preserve the harness's absolute WORK_DIR contract. Apply the granted
    # workspace-write mode explicitly: DSH's default is read-only, and a fresh
    # child home does not inherit the parent's per-session grants.
    runtime = cwd / ".opx-runtime"
    if not runtime.resolve().is_relative_to(workspace):
        raise ValueError("runtime patch directory escapes the configured project")
    runtime.mkdir(mode=0o700, exist_ok=True)
    workspace_patch = runtime / "dsh-workspace.patch.json"
    if workspace_patch.is_symlink():
        raise ValueError("runtime workspace patch must not be a symlink")
    patch = [{"id": "sandbox-policy", "config": {
        "mode": "workspace-write", "workspaceRoot": str(workspace),
    }}]
    fd, name = tempfile.mkstemp(prefix="workspace-", suffix=".json", dir=runtime)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(patch, stream, ensure_ascii=False)
            stream.write("\n")
        os.replace(name, workspace_patch)
    finally:
        if os.path.exists(name):
            os.unlink(name)

    env = os.environ.copy()
    # A child must not inherit the parent's writable profile/session directory:
    # DSH bootstraps cordis.yml even for an existing profile. Keep each child's
    # runtime inside the authorized project and preserve the profile overlays.
    inherited_profile = env.get("DSH_PROFILE_DIR")
    if not inherited_profile or Path(inherited_profile).name != "headless":
        inherited_home = env.get("DSH_HOME")
        inherited_profile = str(Path(inherited_home) / "profiles/headless") if inherited_home else ""
    profile = Path(inherited_profile)
    if not inherited_profile or not profile.is_dir() or not (profile / "package.json").is_file():
        raise ValueError("missing inherited headless profile; refusing to change the child policy")
    child_home = Path(tempfile.mkdtemp(prefix=f"{agent}-", dir=runtime))
    child_profile = child_home / "profiles/headless"
    child_profile.mkdir(parents=True)
    for name in ("package.json", "cordis.yml", "cordis.patch.yml", "pnpm-workspace.yaml"):
        source = profile / name
        if source.is_file():
            shutil.copyfile(source, child_profile / name)
    env["DSH_HOME"] = str(child_home)
    env["DSH_PROFILE_DIR"] = str(child_profile)
    env["DSH_PROFILE"] = "headless"
    env.pop("DSH_SESSION_ID", None)
    env["WORK_DIR"] = str(cwd)
    if not env.get("OPX_LLM_KEY"):
        # Shell-tool environments intentionally remove *KEY* and *SECRET*.
        # Restore only non-secret configuration supplied by the console parent;
        # never rename a credential to evade the tool environment filter.
        for name, alias in (
            ("OPX_SECRET_DIR", "OPX_PROVIDER_STORE_DIR"),
            ("OPX_SECRET_BACKEND", "OPX_PROVIDER_STORE_BACKEND"),
        ):
            if not env.get(name) and env.get(alias):
                env[name] = env[alias]
        # Same active endpoint and secret store as API management. The key stays
        # in a private subprocess pipe and the child environment, never argv/logs.
        query = """
from app.db import SessionLocal
from app.services import llm_config
with SessionLocal() as db:
    key = llm_config.endpoint_key(llm_config.active_endpoint(db)) or ""
if not key:
    raise SystemExit("active endpoint has no API key")
print(key, end="")
"""
        try:
            result = subprocess.run(
                [str(repo / "backend/.venv/bin/python"), "-c", query],
                cwd=repo / "backend", env=env, capture_output=True, text=True,
                check=False, timeout=15,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("timed out reading the active endpoint from the local secret store") from None
        if result.returncode or not result.stdout:
            raise ValueError("cannot load the active endpoint key from the local secret store")
        env["OPX_LLM_KEY"] = result.stdout
    role_note = (
        f"You are the workflow child role '{agent}'. Read {role} completely and follow it. "
        f"You are not the PM. The workflow WORK_DIR is {cwd}; preserve that report/verdict scope. "
        f"The session starts at the authorized project workspace {workspace}. "
        "Use absolute WORK_DIR paths for all workflow reports and verdicts.\n\n"
    )
    return [
        "dsh", "--profile", "headless", "--patch", str(llm_patch),
        "--patch", str(workspace_patch), "--json", role_note + prompt,
    ], env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--format", choices=["json"], default="json")
    parser.add_argument("--agent", required=True)
    parser.add_argument("prompt")
    args = parser.parse_args()
    try:
        argv, env = prepare_launch(args.agent, args.prompt)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"cannbot: {exc}\n")
    # DSH uses session.header.cwd ahead of sandbox-policy.workspaceRoot.
    # Starting in the nested workflow directory would silently narrow the grant
    # and make the project's formal ops/tests/docs paths read-only.
    os.chdir(configured_directory("OPX_WORKSPACE_ROOT"))
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
