"""Durable PostgreSQL coordination for Textify admissions."""

import hmac
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast
from uuid import UUID, uuid4

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    LargeBinary,
    MetaData,
    Table,
    Text,
    select,
    text,
    update,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import RowMapping, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_COORDINATION_REVISION = "0001_textify_coordination"
_CAPABILITY_LINK_PATTERN = re.compile(
    r"/api/v1/transcription-jobs/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_COORDINATION_TABLE = Table(
    "textify_coordination",
    MetaData(),
    Column("id", BigInteger(), primary_key=True),
    Column("url_identity", LargeBinary(), nullable=False),
    Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
    Column("capability_link", Text(), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)


class CoordinationConfig(BaseSettings):
    """Validate the dedicated PostgreSQL coordination configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_COORDINATION_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        validate_default=True,
    )

    database_url: SecretStr
    hmac_secret: SecretStr
    retention_seconds: int = Field(default=86_400, gt=0)

    @model_validator(mode="after")
    def _validate_database_and_hmac_secret(self) -> "CoordinationConfig":
        """Require an async PostgreSQL endpoint and nonblank HMAC secret.

        Raises:
            ValueError: If the endpoint is incomplete or not PostgreSQL through asyncpg,
                or if the HMAC secret is blank.
        """
        database_url = self.database_url.get_secret_value()
        endpoint = make_url(database_url)
        if (
            endpoint.drivername != "postgresql+asyncpg"
            or endpoint.host is None
            or endpoint.database is None
        ):
            raise ValueError(
                "coordination database URL must use postgresql+asyncpg with a host and database"
            )
        if not self.hmac_secret.get_secret_value().strip():
            raise ValueError("coordination HMAC secret must not be blank")
        return self


class CoordinationError(RuntimeError):
    """Indicate that durable Textify coordination could not preserve its contract."""


@dataclass(frozen=True, slots=True)
class TextifyCoordination:
    """Contain one committed pending or accepted Textify job association."""

    id: int
    url_identity: bytes
    idempotency_key: UUID
    capability_link: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class TextifyCoordinationRepository(Protocol):
    """Persist narrow durable state around Textify job admission."""

    async def get_or_create(self, submitted_url: str) -> TextifyCoordination:
        """Atomically create or retrieve coordination for a submitted URL.

        Args:
            submitted_url: Exact URL submitted for public Source extraction.

        Returns:
            The existing coordination record or newly committed pending record.
        """
        ...

    async def attach_capability(
        self,
        coordination_id: int,
        capability_link: str,
    ) -> TextifyCoordination:
        """Commit Textify's validated bearer Capability Link.

        Args:
            coordination_id: Internal coordination record identifier.
            capability_link: Textify's exact relative polling capability.

        Returns:
            The committed accepted Textify coordination record.

        Raises:
            CoordinationError: If the link is invalid or no single record is updated.
        """
        ...

    async def aclose(self) -> None:
        """Release repository-owned database resources."""
        ...


class PostgresTextifyCoordinationRepository:
    """Persist Textify admission state through one async SQLAlchemy engine."""

    def __init__(
        self,
        engine: AsyncEngine,
        hmac_secret: bytes,
        retention: timedelta,
    ) -> None:
        """Initialize the repository with its owned engine and identity policy.

        Args:
            engine: Engine exclusively owned by this coordination repository.
            hmac_secret: Dedicated secret bytes used to derive URL identity.
            retention: Positive duration during which coordination may remain durable.
        """
        self._engine = engine
        self._hmac_secret = hmac_secret
        self._retention = retention

    async def get_or_create(self, submitted_url: str) -> TextifyCoordination:
        """Create or return durable coordination without persisting the submitted URL.

        Args:
            submitted_url: Exact URL submitted for public Source extraction.

        Returns:
            The existing coordination record or newly committed pending record.
        """
        created_at = datetime.now(UTC)
        url_identity = hmac.digest(
            self._hmac_secret,
            submitted_url.encode("utf-8"),
            "sha256",
        )
        values = {
            "url_identity": url_identity,
            "idempotency_key": uuid4(),
            "capability_link": None,
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": created_at + self._retention,
        }
        async with self._engine.begin() as connection:
            result = await connection.execute(
                postgresql.insert(_COORDINATION_TABLE)
                .values(values)
                .on_conflict_do_nothing(index_elements=[_COORDINATION_TABLE.c.url_identity])
                .returning(*_COORDINATION_TABLE.c)
            )
            row = result.mappings().one_or_none()
            if row is None:
                result = await connection.execute(
                    select(_COORDINATION_TABLE).where(
                        _COORDINATION_TABLE.c.url_identity == url_identity
                    )
                )
                row = result.mappings().one()
        return _to_coordination(row)

    async def attach_capability(
        self,
        coordination_id: int,
        capability_link: str,
    ) -> TextifyCoordination:
        """Commit Textify's validated bearer Capability Link.

        Args:
            coordination_id: Internal coordination record identifier.
            capability_link: Textify's exact relative polling capability.

        Returns:
            The committed accepted Textify coordination record.

        Raises:
            CoordinationError: If the link is invalid or no single record is updated.
        """
        _validate_capability_link(capability_link)
        updated_at = datetime.now(UTC)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                update(_COORDINATION_TABLE)
                .where(_COORDINATION_TABLE.c.id == coordination_id)
                .values(capability_link=capability_link, updated_at=updated_at)
                .returning(*_COORDINATION_TABLE.c)
            )
            records = result.mappings().all()
            if len(records) != 1:
                raise CoordinationError("Textify coordination record could not be updated")
            record = _to_coordination(records[0])
        return record

    async def aclose(self) -> None:
        """Dispose the repository-owned SQLAlchemy engine."""
        await self._engine.dispose()

    async def _verify_ready(self) -> None:
        """Verify writable PostgreSQL access and the exact installed revision.

        Raises:
            CoordinationError: If the database revision does not match this application.
        """
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TEMPORARY TABLE _reelio_coordination_write_probe "
                    "(id integer) ON COMMIT DROP"
                )
            )
            await connection.execute(
                text("INSERT INTO _reelio_coordination_write_probe (id) VALUES (1)")
            )
            revisions = (
                await connection.scalars(text("SELECT version_num FROM alembic_version"))
            ).all()
            if revisions != [_COORDINATION_REVISION]:
                raise CoordinationError("Textify coordination schema revision is unavailable")


def _to_coordination(row: RowMapping) -> TextifyCoordination:
    """Convert one PostgreSQL record into immutable coordination state.

    Args:
        row: Returned row from the dedicated coordination table.

    Returns:
        Immutable Reelio-owned coordination state.
    """
    return TextifyCoordination(
        id=cast(int, row["id"]),
        url_identity=cast(bytes, row["url_identity"]),
        idempotency_key=cast(UUID, row["idempotency_key"]),
        capability_link=cast(str | None, row["capability_link"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        expires_at=cast(datetime, row["expires_at"]),
    )


def _validate_capability_link(capability_link: str) -> None:
    """Reject every link outside Textify's exact polling capability grammar.

    Args:
        capability_link: Candidate relative Textify polling link.

    Raises:
        CoordinationError: If the candidate is not an exact supported capability.
    """
    if _CAPABILITY_LINK_PATTERN.fullmatch(capability_link) is None:
        raise CoordinationError("Textify returned an invalid capability link")


async def create_postgres_textify_coordination(
    settings: CoordinationConfig,
) -> PostgresTextifyCoordinationRepository:
    """Create a verified repository owning one PostgreSQL coordination engine.

    Args:
        settings: Validated dedicated coordination database and HMAC settings.

    Returns:
        A verified repository ready to create and attach Textify coordination.

    Raises:
        CoordinationError: If the installed revision is not the required baseline.
    """
    engine = create_async_engine(
        settings.database_url.get_secret_value(),
        hide_parameters=True,
        pool_pre_ping=True,
    )
    repository = PostgresTextifyCoordinationRepository(
        engine=engine,
        hmac_secret=settings.hmac_secret.get_secret_value().encode("utf-8"),
        retention=timedelta(seconds=settings.retention_seconds),
    )
    try:
        await repository._verify_ready()
    except BaseException:
        await engine.dispose()
        raise
    return repository
