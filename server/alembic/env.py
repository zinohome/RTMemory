"""Alembic env.py with async SQLAlchemy support and pgvector extension."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Alembic can detect them
from app.db.base import Base
from app.db.models import (  # noqa: F401 — ensure models registered on Base
    Space,
    Entity,
    Relation,
    Memory,
    Document,
    Chunk,
    MemorySource,
)

config = context.config

# Allow RTMEM_DATABASE_URL env var to override the URL in alembic.ini
# This is essential for Docker where the database host differs from localhost.
_db_url = os.environ.get("RTMEM_DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations synchronously using the provided connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=False,
    )
    with context.begin_transaction():
        # Ensure pgvector extension is available
        connection.execute(context._proxy.sql("CREATE EXTENSION IF NOT EXISTS vector"))
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — dispatches to async."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()