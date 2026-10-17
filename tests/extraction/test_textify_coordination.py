"""Durable Textify coordination repository contract tests."""

import hmac
from collections.abc import Callable
from datetime import timedelta
from typing import cast

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from reelio.extraction import coordination as coordination_module
from reelio.extraction.coordination import (
    CoordinationConfig,
    CoordinationError,
    PostgresTextifyCoordinationRepository,
    create_postgres_textify_coordination,
)

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CAPABILITY_LINK = "/api/v1/transcription-jobs/00000000-0000-4000-8000-000000000000"
_HMAC_SECRET = "test-coordination-hmac-secret"


class _CloseRecordingEngine:
    def __init__(self) -> None:
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


def _settings(database_url: str) -> CoordinationConfig:
    """Build dedicated test configuration without consulting a dotenv file."""
    constructor = cast(Callable[..., CoordinationConfig], CoordinationConfig)
    return constructor(
        _env_file=None,
        database_url=database_url,
        hmac_secret=_HMAC_SECRET,
    )


async def test_repository_commits_pending_coordination_before_returning(
    migrated_coordination_database: str,
) -> None:
    """Persist only the HMAC URL identity and UUIDv4 before Textify admission."""
    repository = await create_postgres_textify_coordination(
        _settings(migrated_coordination_database)
    )
    try:
        coordination = await repository.get_or_create(_SUBMITTED_URL)
        expected_identity = hmac.digest(
            _HMAC_SECRET.encode("utf-8"),
            _SUBMITTED_URL.encode("utf-8"),
            "sha256",
        )

        assert coordination.id > 0
        assert coordination.url_identity == expected_identity
        assert coordination.idempotency_key.version == 4
        assert coordination.capability_link is None
        assert coordination.created_at.tzinfo is not None
        assert coordination.created_at == coordination.updated_at
        assert coordination.expires_at == coordination.created_at + timedelta(days=1)

        verification_engine = create_async_engine(migrated_coordination_database)
        try:
            async with verification_engine.connect() as connection:
                stored_row = await connection.execute(
                    text(
                        "SELECT url_identity, idempotency_key, capability_link, created_at, "
                        "updated_at, expires_at "
                        "FROM textify_coordination WHERE id = :coordination_id"
                    ),
                    {"coordination_id": coordination.id},
                )
                stored = stored_row.one()
        finally:
            await verification_engine.dispose()

        assert stored == (
            expected_identity,
            coordination.idempotency_key,
            None,
            coordination.created_at,
            coordination.updated_at,
            coordination.expires_at,
        )
    finally:
        await repository.aclose()


async def test_repository_attaches_only_valid_capability_links_after_commit(
    migrated_coordination_database: str,
) -> None:
    """Commit the admitted bearer Capability Link before the caller can poll it."""
    repository = await create_postgres_textify_coordination(
        _settings(migrated_coordination_database)
    )
    try:
        pending = await repository.get_or_create(_SUBMITTED_URL)
        accepted = await repository.attach_capability(pending.id, _CAPABILITY_LINK)

        assert accepted.id == pending.id
        assert accepted.url_identity == pending.url_identity
        assert accepted.idempotency_key == pending.idempotency_key
        assert accepted.capability_link == _CAPABILITY_LINK
        assert accepted.created_at == pending.created_at
        assert accepted.updated_at > pending.updated_at
        assert accepted.expires_at == pending.expires_at

        verification_engine = create_async_engine(migrated_coordination_database)
        try:
            async with verification_engine.connect() as connection:
                stored_link = await connection.scalar(
                    text(
                        "SELECT capability_link FROM textify_coordination "
                        "WHERE id = :coordination_id"
                    ),
                    {"coordination_id": pending.id},
                )
        finally:
            await verification_engine.dispose()

        assert stored_link == _CAPABILITY_LINK
    finally:
        await repository.aclose()


async def test_repository_rejects_invalid_or_missing_capability_attachment(
    migrated_coordination_database: str,
) -> None:
    """Leave pending state unchanged when attachment cannot target one valid row."""
    repository = await create_postgres_textify_coordination(
        _settings(migrated_coordination_database)
    )
    try:
        pending = await repository.get_or_create(_SUBMITTED_URL)

        with pytest.raises(CoordinationError):
            await repository.attach_capability(
                pending.id,
                "https://textify.example/api/v1/transcription-jobs/"
                "00000000-0000-4000-8000-000000000000",
            )
        with pytest.raises(CoordinationError):
            await repository.attach_capability(-1, _CAPABILITY_LINK)

        verification_engine = create_async_engine(migrated_coordination_database)
        try:
            async with verification_engine.connect() as connection:
                stored_link = await connection.scalar(
                    text(
                        "SELECT capability_link FROM textify_coordination "
                        "WHERE id = :coordination_id"
                    ),
                    {"coordination_id": pending.id},
                )
        finally:
            await verification_engine.dispose()

        assert stored_link is None
    finally:
        await repository.aclose()


async def test_repository_factory_disposes_engine_after_schema_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispose the owned engine when PostgreSQL schema verification rejects startup."""
    engine = _CloseRecordingEngine()

    async def fail_verification(
        repository: PostgresTextifyCoordinationRepository,
    ) -> None:
        del repository
        raise CoordinationError("Textify coordination schema revision is unavailable")

    monkeypatch.setattr(
        coordination_module,
        "create_async_engine",
        lambda *args, **kwargs: cast(AsyncEngine, engine),
    )
    monkeypatch.setattr(
        PostgresTextifyCoordinationRepository,
        "_verify_ready",
        fail_verification,
    )

    with pytest.raises(
        CoordinationError,
        match="Textify coordination schema revision is unavailable",
    ):
        await create_postgres_textify_coordination(
            _settings("postgresql+asyncpg://reelio@localhost/reelio")
        )

    assert engine.dispose_calls == 1
