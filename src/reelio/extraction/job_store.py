"""Deterministic in-memory adapter for Source Extraction job storage."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from reelio.database import CURRENT_SCHEMA_REVISION, DatabaseConfig
from reelio.extraction.jobs import (
    EXTRACTION_RESPONSE_VERSION,
    ExtractionJobStateError,
    ExtractionJobStatus,
    ExtractionJobStoreUnavailableError,
    FailedSourceJob,
    QueuedSourceJob,
    RunningSourceJob,
    SourceExtractionJob,
    SucceededSourceJob,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.schemas import ExtractResponse


class InMemoryExtractionJobStore:
    """Store Source Extraction jobs deterministically for tests and local composition."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        """Initialize an empty store with injectable identity and UTC time sources.

        Args:
            clock: UTC clock used for every committed lifecycle timestamp.
            uuid_factory: UUID4 factory used when admitting Source jobs.
        """
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._jobs: dict[UUID, SourceExtractionJob] = {}
        self._owners: dict[UUID, bytes] = {}
        self._admissions: dict[tuple[bytes, UUID], QueuedSourceJob] = {}
        self._lock = asyncio.Lock()

    async def admit_source(
        self,
        source_url: str,
        market: SpotifyMarket | None,
        owner_digest: bytes,
        idempotency_key: UUID,
    ) -> QueuedSourceJob:
        """Commit or replay one owner-scoped queued Source Extraction job.

        Args:
            source_url: Validated public Source URL submitted by the caller.
            market: Optional requested Effective Market.
            owner_digest: HMAC-derived anonymous owner identity.
            idempotency_key: Caller-supplied owner-scoped replay identity.

        Returns:
            The original queued job snapshot for this owner and idempotency key.
        """
        async with self._lock:
            admission_key = (owner_digest, idempotency_key)
            admitted_job = self._admissions.get(admission_key)
            if admitted_job is not None:
                return _copy_queued_job(admitted_job)

            submitted_at = self._clock()
            job = QueuedSourceJob(
                job_id=self._uuid_factory(),
                source_url=source_url,
                market=market,
                submitted_at=submitted_at,
                updated_at=submitted_at,
            )
            self._jobs[job.job_id] = job
            self._owners[job.job_id] = owner_digest
            self._admissions[admission_key] = job
            return _copy_queued_job(job)

    async def get_for_owner(
        self,
        job_id: UUID,
        owner_digest: bytes,
    ) -> SourceExtractionJob | None:
        """Return one immutable Source Extraction job snapshot for its owner.

        Args:
            job_id: Opaque Source Extraction job identity.
            owner_digest: HMAC-derived anonymous owner identity.

        Returns:
            The current snapshot, or None when the job is absent or has another owner.
        """
        async with self._lock:
            if self._owners.get(job_id) != owner_digest:
                return None
            job = self._jobs.get(job_id)
            return _copy_job(job) if job is not None else None

    async def claim_next_source(self) -> RunningSourceJob | None:
        """Atomically claim the oldest queued Source Extraction job.

        Returns:
            The committed running snapshot, or None when no job is queued.
        """
        async with self._lock:
            queued_jobs = (job for job in self._jobs.values() if isinstance(job, QueuedSourceJob))
            queued_job = min(
                queued_jobs,
                default=None,
                key=lambda job: (job.submitted_at, job.job_id.int),
            )
            if queued_job is None:
                return None

            started_at = self._clock()
            running_job = RunningSourceJob(
                job_id=queued_job.job_id,
                source_url=queued_job.source_url,
                market=queued_job.market,
                submitted_at=queued_job.submitted_at,
                updated_at=started_at,
                started_at=started_at,
            )
            self._jobs[running_job.job_id] = running_job
            return running_job

    async def complete_success(
        self,
        job_id: UUID,
        result: ExtractResponse,
    ) -> SucceededSourceJob:
        """Commit a mapped response for one running Source Extraction job.

        Args:
            job_id: Claimed job identity.
            result: Existing public Extraction response produced by the worker.

        Returns:
            The committed succeeded snapshot.

        Raises:
            ExtractionJobStateError: If the job is absent or not running.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, RunningSourceJob):
                raise ExtractionJobStateError(
                    "Source Extraction job must be running before it can succeed"
                )

            finished_at = self._clock()
            succeeded_job = SucceededSourceJob(
                job_id=job.job_id,
                source_url=job.source_url,
                market=job.market,
                submitted_at=job.submitted_at,
                updated_at=finished_at,
                started_at=job.started_at,
                finished_at=finished_at,
                result_version=EXTRACTION_RESPONSE_VERSION,
                result=result.model_copy(deep=True),
            )
            self._jobs[job_id] = succeeded_job
            return SucceededSourceJob(
                job_id=succeeded_job.job_id,
                source_url=succeeded_job.source_url,
                market=succeeded_job.market,
                submitted_at=succeeded_job.submitted_at,
                updated_at=succeeded_job.updated_at,
                started_at=succeeded_job.started_at,
                finished_at=succeeded_job.finished_at,
                result_version=succeeded_job.result_version,
                result=succeeded_job.result.model_copy(deep=True),
            )

    async def aclose(self) -> None:
        """Release resources owned by the in-memory store."""


def _copy_job(job: SourceExtractionJob) -> SourceExtractionJob:
    """Return a caller-isolated immutable snapshot.

    Args:
        job: Stored lifecycle snapshot.

    Returns:
        A snapshot whose successful response cannot mutate stored state.
    """
    if isinstance(job, SucceededSourceJob):
        return SucceededSourceJob(
            job_id=job.job_id,
            source_url=job.source_url,
            market=job.market,
            submitted_at=job.submitted_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result_version=job.result_version,
            result=job.result.model_copy(deep=True),
        )
    return job


def _copy_queued_job(job: QueuedSourceJob) -> QueuedSourceJob:
    """Return the immutable original admission snapshot."""
    return QueuedSourceJob(
        job_id=job.job_id,
        source_url=job.source_url,
        market=job.market,
        submitted_at=job.submitted_at,
        updated_at=job.updated_at,
    )


_EXTRACTION_JOB_TABLE = Table(
    "extraction_job",
    MetaData(),
    Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
    Column("source_url", Text(), nullable=False),
    Column("market", String(length=2), nullable=True),
    Column("status", Text(), nullable=False),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("result_version", Text(), nullable=True),
    Column("result", postgresql.JSONB(), nullable=True),
    Column("error_code", Text(), nullable=True),
    Column("error_message", Text(), nullable=True),
    Column("owner_digest", postgresql.BYTEA(), nullable=False),
    Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
)


class PostgresExtractionJobStore:
    """Persist Source Extraction jobs through an owned async SQLAlchemy engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize the store with an exclusively owned database engine.

        Args:
            engine: Async engine used only for Source Extraction job storage.
        """
        self._engine = engine

    async def admit_source(
        self,
        source_url: str,
        market: SpotifyMarket | None,
        owner_digest: bytes,
        idempotency_key: UUID,
    ) -> QueuedSourceJob:
        """Commit or replay one owner-scoped queued Source Extraction job.

        Args:
            source_url: Validated public Source URL submitted by the caller.
            market: Optional requested Effective Market.
            owner_digest: HMAC-derived anonymous owner identity.
            idempotency_key: Caller-supplied owner-scoped replay identity.

        Returns:
            The original queued job snapshot for this owner and idempotency key.

        Raises:
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot commit admission.
        """
        submitted_at = datetime.now(UTC)
        try:
            async with self._engine.begin() as connection:
                inserted = await connection.execute(
                    postgresql.insert(_EXTRACTION_JOB_TABLE)
                    .values(
                        job_id=uuid4(),
                        source_url=source_url,
                        market=market,
                        owner_digest=owner_digest,
                        idempotency_key=idempotency_key,
                        status=ExtractionJobStatus.QUEUED.value,
                        submitted_at=submitted_at,
                        updated_at=submitted_at,
                    )
                    .on_conflict_do_nothing(
                        index_elements=(
                            _EXTRACTION_JOB_TABLE.c.owner_digest,
                            _EXTRACTION_JOB_TABLE.c.idempotency_key,
                        )
                    )
                    .returning(*_EXTRACTION_JOB_TABLE.c)
                )
                row = inserted.mappings().one_or_none()
                if row is None:
                    replay = await connection.execute(
                        select(_EXTRACTION_JOB_TABLE).where(
                            _EXTRACTION_JOB_TABLE.c.owner_digest == owner_digest,
                            _EXTRACTION_JOB_TABLE.c.idempotency_key == idempotency_key,
                        )
                    )
                    row = replay.mappings().one()
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc
        return _to_queued_admission(row)

    async def get_for_owner(
        self,
        job_id: UUID,
        owner_digest: bytes,
    ) -> SourceExtractionJob | None:
        """Return one current Source Extraction job snapshot for its owner.

        Args:
            job_id: Opaque Source Extraction job identity.
            owner_digest: HMAC-derived anonymous owner identity.

        Returns:
            The current snapshot, or None when no matching job has that identity.

        Raises:
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot complete the read.
        """
        try:
            async with self._engine.connect() as connection:
                result = await connection.execute(
                    select(_EXTRACTION_JOB_TABLE).where(
                        _EXTRACTION_JOB_TABLE.c.job_id == job_id,
                        _EXTRACTION_JOB_TABLE.c.owner_digest == owner_digest,
                    )
                )
                row = result.mappings().one_or_none()
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc
        return _to_source_extraction_job(row) if row is not None else None

    async def claim_next_source(self) -> RunningSourceJob | None:
        """Atomically commit one oldest queued Source job as running.

        Returns:
            The committed running snapshot, or None when no job is queued.

        Raises:
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot complete the claim.
        """
        try:
            async with self._engine.begin() as connection:
                selection = await connection.execute(
                    select(_EXTRACTION_JOB_TABLE)
                    .where(_EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.QUEUED.value)
                    .order_by(
                        _EXTRACTION_JOB_TABLE.c.submitted_at,
                        _EXTRACTION_JOB_TABLE.c.job_id,
                    )
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                queued_row = selection.mappings().one_or_none()
                if queued_row is None:
                    return None

                started_at = datetime.now(UTC)
                result = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(
                        _EXTRACTION_JOB_TABLE.c.job_id == queued_row["job_id"],
                        _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.QUEUED.value,
                    )
                    .values(
                        status=ExtractionJobStatus.RUNNING.value,
                        updated_at=started_at,
                        started_at=started_at,
                    )
                    .returning(*_EXTRACTION_JOB_TABLE.c)
                )
                row = result.mappings().one_or_none()
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc
        if row is None:
            return None
        job = _to_source_extraction_job(row)
        if not isinstance(job, RunningSourceJob):
            raise ExtractionJobStateError(
                "Source Extraction job claim did not return a running job"
            )
        return job

    async def complete_success(
        self,
        job_id: UUID,
        result: ExtractResponse,
    ) -> SucceededSourceJob:
        """Commit one mapped response only for a currently running Source job.

        Args:
            job_id: Claimed job identity.
            result: Existing public Extraction response produced by the worker.

        Returns:
            The committed succeeded job snapshot.

        Raises:
            ExtractionJobStateError: If the job is absent or not running.
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot commit completion.
        """
        finished_at = datetime.now(UTC)
        try:
            async with self._engine.begin() as connection:
                update_result = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(
                        _EXTRACTION_JOB_TABLE.c.job_id == job_id,
                        _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.RUNNING.value,
                    )
                    .values(
                        status=ExtractionJobStatus.SUCCEEDED.value,
                        updated_at=finished_at,
                        finished_at=finished_at,
                        result_version=EXTRACTION_RESPONSE_VERSION,
                        result=result.model_dump(mode="json"),
                    )
                    .returning(*_EXTRACTION_JOB_TABLE.c)
                )
                row = update_result.mappings().one_or_none()
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc
        if row is None:
            raise ExtractionJobStateError(
                "Source Extraction job must be running before it can succeed"
            )
        job = _to_source_extraction_job(row)
        if not isinstance(job, SucceededSourceJob):
            raise ExtractionJobStateError(
                "Source Extraction job completion did not return a succeeded job"
            )
        return job

    async def aclose(self) -> None:
        """Dispose the database engine owned by this job store."""
        await self._engine.dispose()

    async def _verify_ready(self) -> None:
        """Verify writable PostgreSQL access at the application schema revision.

        Raises:
            ExtractionJobStoreUnavailableError: If PostgreSQL is unavailable or its
                schema revision does not match this application.
        """
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    text(
                        "CREATE TEMPORARY TABLE _reelio_extraction_job_write_probe "
                        "(id integer) ON COMMIT DROP"
                    )
                )
                await connection.execute(
                    text("INSERT INTO _reelio_extraction_job_write_probe (id) VALUES (1)")
                )
                revisions = (
                    await connection.scalars(text("SELECT version_num FROM alembic_version"))
                ).all()
                if revisions != [CURRENT_SCHEMA_REVISION]:
                    raise ExtractionJobStoreUnavailableError(
                        "Source Extraction job schema revision is unavailable"
                    )
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc


async def create_production_extraction_job_store() -> PostgresExtractionJobStore:
    """Create one verified production Source Extraction job store.

    Returns:
        A verified PostgreSQL-backed Source Extraction job store.

    Raises:
        ExtractionJobStoreUnavailableError: If PostgreSQL cannot provide the required
            writable schema.
    """
    settings = DatabaseConfig()  # type: ignore[call-arg]
    engine = create_async_engine(
        settings.database_url.get_secret_value(),
        hide_parameters=True,
        pool_pre_ping=True,
    )
    store = PostgresExtractionJobStore(engine)
    try:
        await store._verify_ready()
    except BaseException:
        await engine.dispose()
        raise
    return store


def _to_queued_admission(row: RowMapping) -> QueuedSourceJob:
    """Project one stored admission as its immutable original queued snapshot."""
    market_value = cast(str | None, row["market"])
    return QueuedSourceJob(
        job_id=cast(UUID, row["job_id"]),
        source_url=cast(str, row["source_url"]),
        market=SpotifyMarket(market_value) if market_value is not None else None,
        submitted_at=cast(datetime, row["submitted_at"]),
        updated_at=cast(datetime, row["submitted_at"]),
    )


def _to_source_extraction_job(row: RowMapping) -> SourceExtractionJob:
    """Convert one durable database row into an application-owned job snapshot.

    Args:
        row: Source Extraction job row returned from PostgreSQL.

    Returns:
        The matching immutable lifecycle snapshot.

    Raises:
        ExtractionJobStateError: If the stored row violates lifecycle invariants.
    """
    status = ExtractionJobStatus(cast(str, row["status"]))
    market_value = cast(str | None, row["market"])
    job_id = cast(UUID, row["job_id"])
    source_url = cast(str, row["source_url"])
    market = SpotifyMarket(market_value) if market_value is not None else None
    submitted_at = cast(datetime, row["submitted_at"])
    updated_at = cast(datetime, row["updated_at"])
    if status is ExtractionJobStatus.QUEUED:
        return QueuedSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
        )
    if status is ExtractionJobStatus.RUNNING:
        return RunningSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            started_at=cast(datetime, row["started_at"]),
        )
    if status is ExtractionJobStatus.SUCCEEDED:
        return SucceededSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            started_at=cast(datetime, row["started_at"]),
            finished_at=cast(datetime, row["finished_at"]),
            result_version=cast(str, row["result_version"]),
            result=ExtractResponse.model_validate(row["result"]),
        )
    if status is ExtractionJobStatus.FAILED:
        return FailedSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            started_at=cast(datetime | None, row["started_at"]),
            finished_at=cast(datetime, row["finished_at"]),
            error_code=cast(str, row["error_code"]),
            error_message=cast(str, row["error_message"]),
        )
    raise ExtractionJobStateError("Source Extraction job has an invalid stored status")
