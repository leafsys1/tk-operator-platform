import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.runtime_lock import sqlite_console_lock


def test_second_process_rejected_and_release_allows_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'console.db'}"
    source = ("from app.services.runtime_lock import sqlite_console_lock\n"
              f"with sqlite_console_lock({url!r}): print('acquired')\n")
    with sqlite_console_lock(url):
        contender = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                                   cwd=Path(__file__).resolve().parents[1])
        assert contender.returncode != 0
        assert "拒绝重复启动" in contender.stderr
    restarted = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[1])
    assert restarted.returncode == 0, restarted.stderr
    assert restarted.stdout.strip() == "acquired"


def test_duplicate_lifespan_does_not_reconcile_live_jobs(tmp_path, monkeypatch):
    from app import main

    url = f"sqlite:///{tmp_path / 'console.db'}"
    monkeypatch.setattr(main.settings, "db_url", url)
    def should_not_open_db():
        pytest.fail("duplicate startup reached database reconciliation")
    monkeypatch.setattr(main, "SessionLocal", should_not_open_db)

    async def start():
        async with main.lifespan(main.app):
            pytest.fail("duplicate startup acquired ownership")

    with sqlite_console_lock(url):
        with pytest.raises(RuntimeError, match="未修改作业状态"):
            asyncio.run(start())
