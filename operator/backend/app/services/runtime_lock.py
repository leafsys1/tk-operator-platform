"""Keep SQLite startup reconciliation exclusive to one live console process."""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import os

from sqlalchemy.engine import make_url


@contextmanager
def sqlite_console_lock(db_url: str):
    url = make_url(db_url)
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        yield
        return
    database = Path(url.database).expanduser().resolve()
    lock_path = database.with_name(database.name + ".console.lock")
    # Keep the inode: unlinking a released lock permits two distinct lock files.
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该 SQLite 数据库已有运行中的控制台；拒绝重复启动，未修改作业状态") from exc
        try:
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid()))
            lock.flush()
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
