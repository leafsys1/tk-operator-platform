"""Alembic 环境：直接复用 app 的模型与 settings（连接串来自 OPX_DB_URL）。

不要手改 alembic.ini 里的 sqlalchemy.url —— 这里从 settings 覆盖，保证「代码与迁移看同一个库」。
"""

import pathlib
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # backend/

from app.config import settings  # noqa: E402
from app.models import Base  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", settings.db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = settings.db_url
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,  # SQLite 改列需要 batch 模式
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
