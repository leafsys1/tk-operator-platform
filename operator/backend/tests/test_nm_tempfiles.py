"""Run the real runner/locker mktemp expressions on the host OS, without SSH/NPU."""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nm_test.sh"


@pytest.mark.parametrize("variable", ["RUNNER", "LOCKER"])
@pytest.mark.parametrize("tmpdir_mode", ["with_space", "empty", "unset"])
def test_runner_tempfiles_are_private_unique_and_portable(tmp_path, variable, tmpdir_mode):
    assignment = re.search(
        rf"^\s*{variable}=\$\(mktemp[^\n]+\)$", SCRIPT.read_text(), re.MULTILINE
    ).group(0).strip()
    env = os.environ.copy()
    expected_parent = Path("/tmp")
    if tmpdir_mode == "with_space":
        expected_parent = tmp_path / "temporary files"
        expected_parent.mkdir()
        env["TMPDIR"] = str(expected_parent)
    elif tmpdir_mode == "empty":
        env["TMPDIR"] = ""
    else:
        env.pop("TMPDIR", None)
    command = "set -eu\n" + (
        assignment + f'\nprintf \'%s\\n\' "${{{variable}}}"\n'
    ) * 2
    proc = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
    paths = [Path(line) for line in proc.stdout.splitlines()]
    try:
        assert proc.returncode == 0, proc.stderr
        assert len(paths) == 2 and paths[0] != paths[1]
        for path in paths:
            assert path.parent.resolve() == expected_parent.resolve()
            assert path.is_file()
            assert path.name.startswith(f"nm_test_{variable.lower()}.")
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
