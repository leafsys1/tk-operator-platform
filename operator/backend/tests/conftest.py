"""pytest 共享夹具：临时库 + 临时工作区 + TestClient。

为什么要这些（2026-10-15 复查 ②4）：控制台自身一直没有测试，P1 期间 3 个真 bug 全靠手测发现。
这里只覆盖「契约与校验」这类最容易回归的部分：plan 的缺项分支、门禁 fail code、快照算法一致性、501 行为。
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 测试一律不碰真实远端：设备检查默认关闭（build_plan 里的 OPX_DEVICE_CHECK）
os.environ.setdefault("OPX_DEVICE_CHECK", "0")
# 测试不碰真实 Keychain/Secret Service：强制文件后端 + 临时目录（必须在 import settings 前设置）。
os.environ.setdefault("OPX_SECRET_BACKEND", "file")
os.environ.setdefault("OPX_SECRET_DIR", tempfile.mkdtemp(prefix="opx-test-secrets-"))
# dsh 路由 patch 是运行时生成文件：测试指到临时目录，别写进仓库 backend/.runtime/。
os.environ.setdefault("OPX_DSH_GENERATED_PATCH",
                      str(Path(tempfile.mkdtemp(prefix="opx-test-runtime-")) / "dsh-llm.patch.yml"))
# app 的 lifespan 走真实 `SessionLocal`：测试库里不能放真端点，否则会按生产端点生成 patch。
# 指到一个临时库（下面 _app_db_tables 建表），测试不再碰 backend/opx.db。
os.environ.setdefault("OPX_DB_URL",
                      f"sqlite:///{Path(tempfile.mkdtemp(prefix='opx-test-appdb-')) / 'app.db'}")
# 版本存档的源码归档根：指到临时目录，别让测试往仓库 archives/ 写文件。
os.environ.setdefault("OPX_SAVEPOINT_ROOT", tempfile.mkdtemp(prefix="opx-test-savepoints-"))
# 优化报告归档（Step 3）：同样别让测试往仓库 reports/ 写文件。
os.environ.setdefault("OPX_REPORT_ROOT", tempfile.mkdtemp(prefix="opx-test-reports-"))


@pytest.fixture(scope="session", autouse=True)
def _app_db_tables():
    """lifespan 用的 `app.db.engine` 指向临时库，这里先把表建出来。"""
    from app.db import engine
    from app.models import Base

    Base.metadata.create_all(engine)
    yield


@pytest.fixture(autouse=True)
def _isolate_report_root(tmp_path, monkeypatch):
    """每个用例独立的报告归档根：避免不同测试的 loop_id 复用造成串目录/串校验。"""
    from app.config import settings

    monkeypatch.setattr(settings, "report_root", str(tmp_path / "reports"))
    yield


@pytest.fixture(autouse=True)
def _isolate_secret_store():
    """每个用例前清空测试密钥库（ref 如 endpoint-1 会跨用例重复）。"""
    from app.services import llm_config, secret_store

    secret_store.reset_for_tests()
    llm_config.reset_generated_patch_for_tests()
    yield
    secret_store.reset_for_tests()
    llm_config.reset_generated_patch_for_tests()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """一个像样的任务工作区：四个入口脚本 + 用例集目录。"""
    ws = tmp_path / "ws"
    (ws / "cases").mkdir(parents=True)
    for name in ("build.sh", "test.sh", "benchmark.sh", "profile.sh"):
        (ws / name).write_text("#!/usr/bin/env bash\necho stub\n", encoding="utf-8")
    (ws / "cases" / "case_001.json").write_text('{"case_id": "c1"}', encoding="utf-8")
    (ws / "src").mkdir()
    (ws / "src" / "add.cpp").write_text("// kernel\n", encoding="utf-8")
    return ws


@pytest.fixture()
def db_session(tmp_path: Path):
    """一个临时 SQLite 库（直接建表，不走 alembic —— 迁移往返另有 test_migration.py）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


@pytest.fixture()
def seeded(db_session, workspace: Path):
    """项目 + 执行机 + 能跑算子的镜像 + 任务 + 规格版本（对齐 P1.5-a 实测结论）。"""
    from app.models import DEFAULT_CRITERIA_REF, Host, HostImage, Project, SpecVersion, Task
    from app.services import case_confirm

    host = Host(name="tk36", ssh_alias="tk36", cards="8×910B1")
    db_session.add(host)
    db_session.flush()
    image = HostImage(
        host_id=host.id,
        image_ref="lfx-op-dev-910b",
        tag="torch210-cann910",
        is_op_dev=True,
        env_recipe="docker run --privileged ...",
        torch_version="2.10.0+cpu / torch_npu 2.10.0.post4",
        digest="sha256:deadbeef",
    )
    db_session.add(image)
    db_session.flush()
    project = Project(name="CANN 算子开发", workspace_path=str(workspace), host_id=host.id,
                      image_id=image.id, criteria_ref=DEFAULT_CRITERIA_REF)
    db_session.add(project)
    db_session.flush()
    task = Task(project_id=project.id, title="elementwise_add", op_name="elementwise_add",
                target_chip="910B1", input_form="自然语言")
    db_session.add(task)
    db_session.flush()
    spec = SpecVersion(task_id=task.id, version=1, op_name="elementwise_add", spec_json="{}",
                       sha256="0" * 64)
    db_session.add(spec)
    db_session.flush()
    task.current_spec_version_id = spec.id
    # 任务生命周期 v2：确认记录必须同时带当前 cases/ 内容哈希，不能用只有时间戳的脏数据
    # 绕过「确认后用例被修改」门禁。
    task.cases_confirmed_at = datetime.now(timezone.utc)
    task.cases_confirmed_sha256 = case_confirm.cases_suite_sha256(workspace)
    task.cases_confirmed_by = "fixture"
    db_session.commit()
    return {"host": host, "image": image, "project": project, "task": task, "spec": spec}


@pytest.fixture()
def client(tmp_path: Path, workspace: Path, monkeypatch):
    """TestClient + 独立库（依赖覆盖），并把工作区/设备检查按测试需要注入。"""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import get_db
    from app.main import app
    from app.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    # 与生产 `app/db.SessionLocal` 对齐：autoflush=False（软删/顺延这类「先改后查」逻辑
    # 必须自己 flush，否则测试会与线上行为分叉 —— 复核发现过一次端点顺延回归）。
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    monkeypatch.setenv("OPX_DEVICE_CHECK", "0")
    with TestClient(app) as c:
        yield c, Session
    app.dependency_overrides.clear()
