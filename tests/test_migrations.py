"""PostgreSQL migration contracts for durable Textify coordination."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid1, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_REVISION = "0002_coordination_expiration"
_TABLE_NAME = "textify_coordination"
_EXPIRATION_INDEX = "textify_coordination_expires_at_id_idx"
_CAPABILITY_PATH_PREFIX = "/api/v1/transcription-jobs/"


def _alembic_config() -> Config:
    """Create Alembic configuration rooted at this repository."""
    return Config(str(Path(__file__).parents[1] / "alembic.ini"))


def _upgrade_head() -> None:
    """Apply the coordination migration chain through its current head."""
    command.upgrade(_alembic_config(), "head")


def _downgrade_base() -> None:
    """Revert every coordination migration."""
    command.downgrade(_alembic_config(), "base")


def _coordination_values(
    *,
    url_identity: bytes | None = None,
    idempotency_key: UUID | None = None,
    capability_link: str | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> dict[str, object]:
    """Build valid pending-state values with optional lifecycle overrides."""
    creation_time = created_at or datetime.now(UTC)
    update_time = updated_at or creation_time
    expiration_time = expires_at or creation_time + timedelta(days=1)
    return {
        "url_identity": url_identity or uuid4().bytes + uuid4().bytes,
        "idempotency_key": idempotency_key or uuid4(),
        "capability_link": capability_link,
        "created_at": creation_time,
        "updated_at": update_time,
        "expires_at": expiration_time,
    }


async def _insert(engine: AsyncEngine, values: dict[str, object]) -> None:
    """Insert one coordination row in its own committed transaction."""
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO textify_coordination "
                "(url_identity, idempotency_key, capability_link, created_at, "
                "updated_at, expires_at) "
                "VALUES (:url_identity, :idempotency_key, :capability_link, "
                ":created_at, :updated_at, :expires_at)"
            ),
            values,
        )


async def _assert_rejected(engine: AsyncEngine, values: dict[str, object]) -> None:
    """Assert PostgreSQL rejects one invalid coordination record."""
    with pytest.raises(IntegrityError):
        await _insert(engine, values)


def _capability_link() -> str:
    """Return one valid bearer Capability Link for migration coverage."""
    return f"{_CAPABILITY_PATH_PREFIX}{uuid4()}"


async def test_coordination_migration_upgrades_downgrades_and_reupgrades(
    coordination_database: str,
) -> None:
    """Create exactly the durable schema, remove it, then restore its revision."""
    await asyncio.to_thread(_upgrade_head)
    engine = create_async_engine(coordination_database)
    try:
        async with engine.connect() as connection:
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            columns = await connection.execute(
                text(
                    "SELECT column_name, data_type, is_nullable, identity_generation "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :table_name "
                    "ORDER BY ordinal_position"
                ),
                {"table_name": _TABLE_NAME},
            )
            constraint_names = await connection.scalars(
                text(
                    "SELECT conname "
                    "FROM pg_constraint "
                    "WHERE conrelid = 'public.textify_coordination'::regclass "
                    "AND contype IN ('p', 'u', 'c')"
                )
            )
            expiration_index = await connection.scalar(
                text(f"SELECT to_regclass('public.{_EXPIRATION_INDEX}')")
            )

        assert revision == _REVISION
        assert expiration_index == _EXPIRATION_INDEX
        assert columns.all() == [
            ("id", "bigint", "NO", "ALWAYS"),
            ("url_identity", "bytea", "NO", None),
            ("idempotency_key", "uuid", "NO", None),
            ("capability_link", "text", "YES", None),
            ("created_at", "timestamp with time zone", "NO", None),
            ("updated_at", "timestamp with time zone", "NO", None),
            ("expires_at", "timestamp with time zone", "NO", None),
        ]
        assert set(constraint_names.all()) == {
            "textify_coordination_pkey",
            "textify_coordination_url_identity_key",
            "textify_coordination_idempotency_key_key",
            "textify_coordination_url_identity_length_check",
            "textify_coordination_idempotency_key_uuid4_check",
            "textify_coordination_capability_state_check",
            "textify_coordination_timestamp_order_check",
        }

        await asyncio.to_thread(_downgrade_base)

        async with engine.connect() as connection:
            table = await connection.scalar(
                text("SELECT to_regclass('public.textify_coordination')")
            )
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            expiration_index = await connection.scalar(
                text(f"SELECT to_regclass('public.{_EXPIRATION_INDEX}')")
            )

        assert table is None
        assert revision is None
        assert expiration_index is None

        await asyncio.to_thread(_upgrade_head)

        async with engine.connect() as connection:
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            table = await connection.scalar(
                text("SELECT to_regclass('public.textify_coordination')")
            )
            expiration_index = await connection.scalar(
                text(f"SELECT to_regclass('public.{_EXPIRATION_INDEX}')")
            )

        assert revision == _REVISION
        assert table == _TABLE_NAME
        assert expiration_index == _EXPIRATION_INDEX
    finally:
        await engine.dispose()


async def test_coordination_migration_enforces_persisted_invariants(
    coordination_database: str,
) -> None:
    """Reject duplicate, malformed, and unordered coordination state in PostgreSQL."""
    await asyncio.to_thread(_upgrade_head)
    engine = create_async_engine(coordination_database)
    created_at = datetime(2026, 10, 14, tzinfo=UTC)
    try:
        pending = _coordination_values(created_at=created_at)
        await _insert(engine, pending)

        assert isinstance(pending["url_identity"], bytes)
        assert len(pending["url_identity"]) == 32
        assert isinstance(pending["idempotency_key"], UUID)
        assert pending["idempotency_key"].version == 4

        await _assert_rejected(
            engine,
            _coordination_values(
                url_identity=pending["url_identity"],
                created_at=created_at + timedelta(seconds=1),
            ),
        )
        await _assert_rejected(
            engine,
            _coordination_values(
                idempotency_key=pending["idempotency_key"],
                created_at=created_at + timedelta(seconds=2),
            ),
        )
        await _assert_rejected(
            engine,
            _coordination_values(
                url_identity=b"not-32-bytes",
                created_at=created_at + timedelta(seconds=3),
            ),
        )
        await _assert_rejected(
            engine,
            _coordination_values(
                idempotency_key=uuid1(),
                created_at=created_at + timedelta(seconds=4),
            ),
        )
        await _assert_rejected(
            engine,
            _coordination_values(
                created_at=created_at,
                updated_at=created_at + timedelta(seconds=1),
            ),
        )

        accepted_created_at = created_at + timedelta(seconds=5)
        accepted_updated_at = accepted_created_at + timedelta(seconds=1)
        accepted_link = _capability_link()
        await _insert(
            engine,
            _coordination_values(
                capability_link=accepted_link,
                created_at=accepted_created_at,
                updated_at=accepted_updated_at,
                expires_at=accepted_updated_at + timedelta(days=1),
            ),
        )

        for malformed_link in (
            "https://textify.example/api/v1/transcription-jobs/"
            "00000000-0000-4000-8000-000000000000",
            "/api/v1/transcription-jobs/00000000-0000-3000-8000-000000000000",
            "/api/v1/transcription-jobs/00000000-0000-4000-7000-000000000000",
            "/api/v1/transcription-jobs/00000000-0000-4000-8000-000000000000/extra",
        ):
            await _assert_rejected(
                engine,
                _coordination_values(
                    capability_link=malformed_link,
                    created_at=accepted_created_at,
                    updated_at=accepted_updated_at,
                    expires_at=accepted_updated_at + timedelta(days=1),
                ),
            )

        await _assert_rejected(
            engine,
            _coordination_values(
                capability_link=_capability_link(),
                created_at=accepted_created_at + timedelta(seconds=2),
                updated_at=accepted_created_at + timedelta(seconds=1),
                expires_at=accepted_created_at + timedelta(seconds=3),
            ),
        )
        await _assert_rejected(
            engine,
            _coordination_values(
                capability_link=_capability_link(),
                created_at=accepted_created_at,
                updated_at=accepted_updated_at,
                expires_at=accepted_updated_at,
            ),
        )
    finally:
        await engine.dispose()
