"""Alembic environment for Reelio Textify coordination."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from reelio.extraction.coordination import CoordinationConfig

config = context.config
target_metadata = None


def _configured_database_url() -> str:
    """Load the coordination migration target from dedicated settings.

    Returns:
        The configured PostgreSQL asyncpg URL.
    """
    return CoordinationConfig().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    context.configure(
        url=_configured_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    """Configure and execute migrations through Alembic's sync bridge.

    Args:
        connection: Synchronous SQLAlchemy connection supplied by ``run_sync``.
    """
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against the configured asynchronous PostgreSQL database."""
    engine = create_async_engine(
        _configured_database_url(),
        poolclass=pool.NullPool,
        hide_parameters=True,
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
