"""Shared pytest fixtures and safe import-time settings defaults."""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

os.environ.setdefault("REELIO_LLM_PROVIDER", "deepseek")
os.environ.setdefault("REELIO_DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("REELIO_TMDB_API_KEY", "test-tmdb-key")
os.environ.setdefault("REELIO_SPOTIFY_CLIENT_ID", "test-spotify-client-id")
os.environ.setdefault("REELIO_SPOTIFY_CLIENT_SECRET", "test-spotify-client-secret")
os.environ.setdefault("REELIO_OPEN_LIBRARY_CONTACT_EMAIL", "test@example.invalid")

from reelio.main import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Yield an HTTP client connected to the application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def coordination_database(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[str]:
    """Yield one isolated PostgreSQL database and remove it after the test."""
    configured_url = os.environ.get("REELIO_TEST_DATABASE_URL")
    if configured_url is None:
        pytest.skip("REELIO_TEST_DATABASE_URL is required for PostgreSQL integration tests")

    admin_url = make_url(configured_url)
    if admin_url.drivername != "postgresql+asyncpg" or admin_url.database is None:
        pytest.fail("REELIO_TEST_DATABASE_URL must be a postgresql+asyncpg admin database")

    database_name = f"reelio_test_{uuid4().hex}"
    database_url = admin_url.set(database=database_name)
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{database_name}"'))

        monkeypatch.setenv("REELIO_COORDINATION_DATABASE_URL", str(database_url))
        monkeypatch.setenv("REELIO_COORDINATION_HMAC_SECRET", "test-coordination-hmac-secret")
        yield str(database_url)
    finally:
        try:
            async with admin_engine.connect() as connection:
                await connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :database_name "
                        "AND pid <> pg_backend_pid()"
                    ),
                    {"database_name": database_name},
                )
                await connection.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
        finally:
            await admin_engine.dispose()


@pytest.fixture
async def migrated_coordination_database(
    coordination_database: str,
) -> AsyncIterator[str]:
    """Yield one isolated PostgreSQL database upgraded through the migration head."""
    configuration = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    await asyncio.to_thread(command.upgrade, configuration, "head")
    yield coordination_database
