from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make pkb_api / pkb_ingestion importable regardless of cwd / PYTHONPATH so
# `uv run alembic ...` works cross-platform without a PYTHONPATH prefix.
_MIGRATIONS_DIR = Path(__file__).resolve().parent
for _src in (
    _MIGRATIONS_DIR.parent / "src",  # apps/api/src
    _MIGRATIONS_DIR.parents[1] / "packages" / "ingestion" / "src",
):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from pkb_api.models import Base  # noqa: E402
from pkb_api.settings import settings  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
