"""Deterministic in-memory adapter for Source Extraction job storage."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    func,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.sql.elements import ColumnElement

from reelio.database import CURRENT_SCHEMA_REVISION, DatabaseConfig
from reelio.extraction.exceptions import INTERNAL_ERROR_CODE, INTERNAL_ERROR_MESSAGE
from reelio.extraction.job_config import ExtractionJobConfig
from reelio.extraction.jobs import (
    EXTRACTION_RESPONSE_VERSION,
    ExtractionJobCapacityExceededError,
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

_DEFAULT_LEASE_LIFETIME = timedelta(seconds=60)


class InMemoryExtractionJobStore:
    """Store Source Extraction jobs deterministically for tests and local composition."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
        lease_token_factory: Callable[[], UUID] = uuid4,
        active_job_cap: int = 10,
        lease_lifetime: timedelta = _DEFAULT_LEASE_LIFETIME,
    ) -> None:
        """Initialize an empty store with injectable identity and UTC time sources.

        Args:
            clock: UTC clock used for every committed lifecycle timestamp.
            uuid_factory: UUID4 factory used when admitting Source jobs.
            lease_token_factory: UUID4 factory used when claiming Source jobs.
            active_job_cap: Maximum queued and running Source jobs admitted at once.
            lease_lifetime: Positive duration granted to every claimed Source job.

        Raises:
            ValueError: If active_job_cap or lease_lifetime is not positive.
        """
        if active_job_cap <= 0:
            raise ValueError("active_job_cap must be positive")
        if lease_lifetime <= timedelta():
            raise ValueError("lease_lifetime must be positive")

        self._clock = clock
        self._uuid_factory = uuid_factory
        self._lease_token_factory = lease_token_factory
        self._active_job_cap = active_job_cap
        self._lease_lifetime = lease_lifetime
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

        Raises:
            ExtractionJobCapacityExceededError: If active job capacity is full.
        """
        async with self._lock:
            admission_key = (owner_digest, idempotency_key)
            admitted_job = self._admissions.get(admission_key)
            if admitted_job is not None:
                return _copy_queued_job(admitted_job)

            active_job_count = sum(
                isinstance(job, QueuedSourceJob | RunningSourceJob) for job in self._jobs.values()
            )
            if active_job_count >= self._active_job_cap:
                raise ExtractionJobCapacityExceededError(
                    "Source Extraction job capacity is temporarily full"
                )

            submitted_at = self._clock()
            job = QueuedSourceJob(
                job_id=self._uuid_factory(),
                source_url=source_url,
                market=market,
                submitted_at=submitted_at,
                updated_at=submitted_at,
                attempt_count=0,
                eligible_at=submitted_at,
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
        """Atomically claim the oldest queued or expired Source Extraction job.

        Returns:
            The committed running snapshot, or None when no job is eligible to run.
        """
        async with self._lock:
            now = self._clock()
            for job_id, job in self._jobs.items():
                if (isinstance(job, QueuedSourceJob) and job.attempt_count == 3) or (
                    isinstance(job, RunningSourceJob)
                    and job.attempt_count == 3
                    and job.lease_expires_at <= now
                ):
                    self._jobs[job_id] = _terminalize_exhausted_attempt(job, now)

            claimable_jobs = (
                job
                for job in self._jobs.values()
                if (
                    isinstance(job, QueuedSourceJob)
                    and job.eligible_at <= now
                    and job.attempt_count < 3
                )
                or (
                    isinstance(job, RunningSourceJob)
                    and job.lease_expires_at <= now
                    and job.attempt_count < 3
                )
            )
            claimable_job = min(
                claimable_jobs,
                default=None,
                key=lambda job: (job.submitted_at, job.job_id.int),
            )
            if claimable_job is None:
                return None

            running_job = RunningSourceJob(
                job_id=claimable_job.job_id,
                source_url=claimable_job.source_url,
                market=claimable_job.market,
                submitted_at=claimable_job.submitted_at,
                updated_at=now,
                attempt_count=claimable_job.attempt_count + 1,
                started_at=now,
                lease_token=self._lease_token_factory(),
                lease_expires_at=now + self._lease_lifetime,
            )
            self._jobs[running_job.job_id] = running_job
            return running_job

    async def renew_source_attempt(self, job_id: UUID, lease_token: UUID) -> bool:
        """Renew one currently owned running Source Extraction attempt."""
        async with self._lock:
            now = self._clock()
            running_job = self._owned_running_job(job_id, lease_token, now)
            if running_job is None:
                return False

            self._jobs[job_id] = RunningSourceJob(
                job_id=running_job.job_id,
                source_url=running_job.source_url,
                market=running_job.market,
                submitted_at=running_job.submitted_at,
                updated_at=now,
                attempt_count=running_job.attempt_count,
                started_at=running_job.started_at,
                lease_token=running_job.lease_token,
                lease_expires_at=now + self._lease_lifetime,
            )
            return True

    async def complete_success(
        self,
        job_id: UUID,
        lease_token: UUID,
        result: ExtractResponse,
    ) -> SucceededSourceJob:
        """Commit a mapped response for one currently owned Source job attempt."""
        async with self._lock:
            finished_at = self._clock()
            job = self._owned_running_job(job_id, lease_token, finished_at)
            if job is None:
                raise ExtractionJobStateError(
                    "Source Extraction job attempt must be owned before it can succeed"
                )

            succeeded_job = SucceededSourceJob(
                job_id=job.job_id,
                source_url=job.source_url,
                market=job.market,
                submitted_at=job.submitted_at,
                updated_at=finished_at,
                attempt_count=job.attempt_count,
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
                attempt_count=succeeded_job.attempt_count,
                started_at=succeeded_job.started_at,
                finished_at=succeeded_job.finished_at,
                result_version=succeeded_job.result_version,
                result=succeeded_job.result.model_copy(deep=True),
            )

    async def complete_failure(
        self,
        job_id: UUID,
        lease_token: UUID,
        error_code: str,
        error_message: str,
    ) -> FailedSourceJob:
        """Commit one failure for a currently owned Source job attempt."""
        async with self._lock:
            finished_at = self._clock()
            job = self._owned_running_job(job_id, lease_token, finished_at)
            if job is None:
                raise ExtractionJobStateError(
                    "Source Extraction job attempt must be owned before it can fail"
                )

            failed_job = FailedSourceJob(
                job_id=job.job_id,
                source_url=job.source_url,
                market=job.market,
                submitted_at=job.submitted_at,
                updated_at=finished_at,
                attempt_count=job.attempt_count,
                started_at=job.started_at,
                finished_at=finished_at,
                error_code=error_code,
                error_message=error_message,
            )
            self._jobs[job_id] = failed_job
            return failed_job

    async def retry_source_attempt(
        self,
        job_id: UUID,
        lease_token: UUID,
        retry_delay: timedelta,
    ) -> bool:
        """Reschedule one currently owned Source attempt after a positive delay.

        Raises:
            ValueError: If retry_delay is not positive.
        """
        if retry_delay <= timedelta():
            raise ValueError("retry_delay must be positive")

        async with self._lock:
            now = self._clock()
            job = self._owned_running_job(job_id, lease_token, now)
            if job is None:
                return False

            self._jobs[job_id] = QueuedSourceJob(
                job_id=job.job_id,
                source_url=job.source_url,
                market=job.market,
                submitted_at=job.submitted_at,
                updated_at=now,
                attempt_count=job.attempt_count,
                eligible_at=now + retry_delay,
            )
            return True

    async def requeue_source_attempt(self, job_id: UUID, lease_token: UUID) -> bool:
        """Return one currently owned Source job attempt to the immediate queue."""
        async with self._lock:
            now = self._clock()
            job = self._owned_running_job(job_id, lease_token, now)
            if job is None:
                return False

            self._jobs[job_id] = QueuedSourceJob(
                job_id=job.job_id,
                source_url=job.source_url,
                market=job.market,
                submitted_at=job.submitted_at,
                updated_at=now,
                attempt_count=job.attempt_count,
                eligible_at=now,
            )
            return True

    def _owned_running_job(
        self,
        job_id: UUID,
        lease_token: UUID,
        now: datetime,
    ) -> RunningSourceJob | None:
        """Return a running job only while its supplied attempt owns a live lease."""
        job = self._jobs.get(job_id)
        if (
            not isinstance(job, RunningSourceJob)
            or job.lease_token != lease_token
            or job.lease_expires_at <= now
        ):
            return None
        return job

    async def aclose(self) -> None:
        """Release resources owned by the in-memory store."""


def _terminalize_exhausted_attempt(
    job: QueuedSourceJob | RunningSourceJob,
    finished_at: datetime,
) -> FailedSourceJob:
    """Return the generic terminal failure for a job that exhausted its attempts."""
    return FailedSourceJob(
        job_id=job.job_id,
        source_url=job.source_url,
        market=job.market,
        submitted_at=job.submitted_at,
        updated_at=finished_at,
        attempt_count=job.attempt_count,
        started_at=job.started_at if isinstance(job, RunningSourceJob) else None,
        finished_at=finished_at,
        error_code=INTERNAL_ERROR_CODE,
        error_message=INTERNAL_ERROR_MESSAGE,
    )


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
            attempt_count=job.attempt_count,
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
        attempt_count=job.attempt_count,
        eligible_at=job.eligible_at,
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
    Column("attempt_count", Integer(), nullable=False),
    Column("eligible_at", DateTime(timezone=True), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("result_version", Text(), nullable=True),
    Column("result", postgresql.JSONB(), nullable=True),
    Column("error_code", Text(), nullable=True),
    Column("error_message", Text(), nullable=True),
    Column("owner_digest", postgresql.BYTEA(), nullable=False),
    Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
    Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
)


class PostgresExtractionJobStore:
    """Persist Source Extraction jobs through an owned async SQLAlchemy engine."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        active_job_cap: int = 10,
        lease_lifetime: timedelta = _DEFAULT_LEASE_LIFETIME,
    ) -> None:
        """Initialize the store with an exclusively owned database engine.

        Args:
            engine: Async engine used only for Source Extraction job storage.
            active_job_cap: Maximum queued and running Source jobs admitted at once.
            lease_lifetime: Positive duration granted to every claimed Source job.

        Raises:
            ValueError: If active_job_cap or lease_lifetime is not positive.
        """
        if active_job_cap <= 0:
            raise ValueError("active_job_cap must be positive")
        if lease_lifetime <= timedelta():
            raise ValueError("lease_lifetime must be positive")

        self._engine = engine
        self._active_job_cap = active_job_cap
        self._lease_lifetime = lease_lifetime

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
            ExtractionJobCapacityExceededError: If active job capacity is full.
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot commit admission.
        """
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    text(
                        "SELECT pg_advisory_xact_lock("
                        "hashtextextended('reelio.extraction_job.admission', 0))"
                    )
                )
                replay = await connection.execute(
                    select(_EXTRACTION_JOB_TABLE).where(
                        _EXTRACTION_JOB_TABLE.c.owner_digest == owner_digest,
                        _EXTRACTION_JOB_TABLE.c.idempotency_key == idempotency_key,
                    )
                )
                replay_row = replay.mappings().one_or_none()
                if replay_row is not None:
                    return _to_queued_admission(replay_row)

                active_job_count = (
                    await connection.execute(
                        select(func.count())
                        .select_from(_EXTRACTION_JOB_TABLE)
                        .where(
                            _EXTRACTION_JOB_TABLE.c.status.in_(
                                (
                                    ExtractionJobStatus.QUEUED.value,
                                    ExtractionJobStatus.RUNNING.value,
                                )
                            )
                        )
                    )
                ).scalar_one()
                if active_job_count >= self._active_job_cap:
                    raise ExtractionJobCapacityExceededError(
                        "Source Extraction job capacity is temporarily full"
                    )

                submitted_at = datetime.now(UTC)
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
                        attempt_count=0,
                        eligible_at=submitted_at,
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
        """Atomically claim the oldest queued or expired Source Extraction job.

        Returns:
            The committed running snapshot, or None when no job is eligible to run.

        Raises:
            ExtractionJobStoreUnavailableError: If PostgreSQL cannot complete the claim.
        """
        try:
            async with self._engine.begin() as connection:
                now = await self._current_time(connection)
                await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(
                        _EXTRACTION_JOB_TABLE.c.attempt_count == 3,
                        or_(
                            _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.QUEUED.value,
                            and_(
                                _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.RUNNING.value,
                                _EXTRACTION_JOB_TABLE.c.lease_expires_at <= now,
                            ),
                        ),
                    )
                    .values(
                        status=ExtractionJobStatus.FAILED.value,
                        updated_at=now,
                        finished_at=now,
                        error_code=INTERNAL_ERROR_CODE,
                        error_message=INTERNAL_ERROR_MESSAGE,
                        eligible_at=None,
                        lease_token=None,
                        lease_expires_at=None,
                    )
                )

                selection = await connection.execute(
                    select(_EXTRACTION_JOB_TABLE)
                    .where(
                        or_(
                            and_(
                                _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.QUEUED.value,
                                _EXTRACTION_JOB_TABLE.c.eligible_at <= now,
                            ),
                            and_(
                                _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.RUNNING.value,
                                _EXTRACTION_JOB_TABLE.c.lease_expires_at <= now,
                            ),
                        ),
                        _EXTRACTION_JOB_TABLE.c.attempt_count < 3,
                    )
                    .order_by(
                        _EXTRACTION_JOB_TABLE.c.submitted_at,
                        _EXTRACTION_JOB_TABLE.c.job_id,
                    )
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                claimed_row = selection.mappings().one_or_none()
                if claimed_row is None:
                    return None

                result = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(
                        _EXTRACTION_JOB_TABLE.c.job_id == claimed_row["job_id"],
                        or_(
                            and_(
                                _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.QUEUED.value,
                                _EXTRACTION_JOB_TABLE.c.eligible_at <= now,
                            ),
                            and_(
                                _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.RUNNING.value,
                                _EXTRACTION_JOB_TABLE.c.lease_expires_at <= now,
                            ),
                        ),
                        _EXTRACTION_JOB_TABLE.c.attempt_count < 3,
                    )
                    .values(
                        status=ExtractionJobStatus.RUNNING.value,
                        updated_at=now,
                        attempt_count=_EXTRACTION_JOB_TABLE.c.attempt_count + 1,
                        eligible_at=None,
                        started_at=now,
                        lease_token=uuid4(),
                        lease_expires_at=now + self._lease_lifetime,
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

    async def renew_source_attempt(self, job_id: UUID, lease_token: UUID) -> bool:
        """Renew one currently owned running Source Extraction attempt."""
        try:
            async with self._engine.begin() as connection:
                now = await self._current_time(connection)
                renewed = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(self._owned_attempt_predicate(job_id, lease_token, now))
                    .values(
                        updated_at=now,
                        lease_expires_at=now + self._lease_lifetime,
                    )
                    .returning(_EXTRACTION_JOB_TABLE.c.job_id)
                )
                return renewed.scalar_one_or_none() is not None
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc

    async def complete_success(
        self,
        job_id: UUID,
        lease_token: UUID,
        result: ExtractResponse,
    ) -> SucceededSourceJob:
        """Commit one mapped response only for a currently owned Source attempt."""
        try:
            async with self._engine.begin() as connection:
                finished_at = await self._current_time(connection)
                update_result = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(self._owned_attempt_predicate(job_id, lease_token, finished_at))
                    .values(
                        status=ExtractionJobStatus.SUCCEEDED.value,
                        updated_at=finished_at,
                        finished_at=finished_at,
                        result_version=EXTRACTION_RESPONSE_VERSION,
                        result=result.model_dump(mode="json"),
                        eligible_at=None,
                        lease_token=None,
                        lease_expires_at=None,
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
                "Source Extraction job attempt must be owned before it can succeed"
            )
        job = _to_source_extraction_job(row)
        if not isinstance(job, SucceededSourceJob):
            raise ExtractionJobStateError(
                "Source Extraction job completion did not return a succeeded job"
            )
        return job

    async def complete_failure(
        self,
        job_id: UUID,
        lease_token: UUID,
        error_code: str,
        error_message: str,
    ) -> FailedSourceJob:
        """Commit one failure only for a currently owned Source attempt."""
        try:
            async with self._engine.begin() as connection:
                finished_at = await self._current_time(connection)
                update_result = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(self._owned_attempt_predicate(job_id, lease_token, finished_at))
                    .values(
                        status=ExtractionJobStatus.FAILED.value,
                        updated_at=finished_at,
                        finished_at=finished_at,
                        error_code=error_code,
                        error_message=error_message,
                        eligible_at=None,
                        lease_token=None,
                        lease_expires_at=None,
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
                "Source Extraction job attempt must be owned before it can fail"
            )
        job = _to_source_extraction_job(row)
        if not isinstance(job, FailedSourceJob):
            raise ExtractionJobStateError(
                "Source Extraction job completion did not return a failed job"
            )
        return job

    async def retry_source_attempt(
        self,
        job_id: UUID,
        lease_token: UUID,
        retry_delay: timedelta,
    ) -> bool:
        """Reschedule one currently owned Source attempt after a positive delay.

        Raises:
            ValueError: If retry_delay is not positive.
        """
        if retry_delay <= timedelta():
            raise ValueError("retry_delay must be positive")

        try:
            async with self._engine.begin() as connection:
                now = await self._current_time(connection)
                rescheduled = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(self._owned_attempt_predicate(job_id, lease_token, now))
                    .values(
                        status=ExtractionJobStatus.QUEUED.value,
                        updated_at=now,
                        started_at=None,
                        eligible_at=now + retry_delay,
                        lease_token=None,
                        lease_expires_at=None,
                    )
                    .returning(_EXTRACTION_JOB_TABLE.c.job_id)
                )
                return rescheduled.scalar_one_or_none() is not None
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc

    async def requeue_source_attempt(self, job_id: UUID, lease_token: UUID) -> bool:
        """Return one currently owned Source attempt to the immediate queue."""
        try:
            async with self._engine.begin() as connection:
                now = await self._current_time(connection)
                requeued = await connection.execute(
                    update(_EXTRACTION_JOB_TABLE)
                    .where(self._owned_attempt_predicate(job_id, lease_token, now))
                    .values(
                        status=ExtractionJobStatus.QUEUED.value,
                        updated_at=now,
                        started_at=None,
                        eligible_at=now,
                        lease_token=None,
                        lease_expires_at=None,
                    )
                    .returning(_EXTRACTION_JOB_TABLE.c.job_id)
                )
                return requeued.scalar_one_or_none() is not None
        except SQLAlchemyError as exc:
            raise ExtractionJobStoreUnavailableError(
                "Source Extraction job storage is temporarily unavailable"
            ) from exc

    async def _current_time(self, connection: AsyncConnection) -> datetime:
        """Read PostgreSQL's transaction authority time."""
        return cast(datetime, await connection.scalar(select(func.now())))

    def _owned_attempt_predicate(
        self,
        job_id: UUID,
        lease_token: UUID,
        now: datetime,
    ) -> ColumnElement[bool]:
        """Build the durable predicate that fences all owned attempt writes."""
        return and_(
            _EXTRACTION_JOB_TABLE.c.job_id == job_id,
            _EXTRACTION_JOB_TABLE.c.status == ExtractionJobStatus.RUNNING.value,
            _EXTRACTION_JOB_TABLE.c.lease_token == lease_token,
            _EXTRACTION_JOB_TABLE.c.lease_expires_at > now,
        )

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
    database_settings = DatabaseConfig()  # type: ignore[call-arg]
    job_settings = ExtractionJobConfig()
    engine = create_async_engine(
        database_settings.database_url.get_secret_value(),
        hide_parameters=True,
        pool_pre_ping=True,
    )
    store = PostgresExtractionJobStore(
        engine,
        active_job_cap=job_settings.active_job_cap,
        lease_lifetime=timedelta(seconds=job_settings.lease_lifetime_seconds),
    )
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
        attempt_count=cast(int, row["attempt_count"]),
        eligible_at=cast(datetime, row["eligible_at"]),
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
    attempt_count = cast(int, row["attempt_count"])
    if status is ExtractionJobStatus.QUEUED:
        return QueuedSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            attempt_count=attempt_count,
            eligible_at=cast(datetime, row["eligible_at"]),
        )
    if status is ExtractionJobStatus.RUNNING:
        return RunningSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            attempt_count=attempt_count,
            started_at=cast(datetime, row["started_at"]),
            lease_token=cast(UUID, row["lease_token"]),
            lease_expires_at=cast(datetime, row["lease_expires_at"]),
        )
    if status is ExtractionJobStatus.SUCCEEDED:
        return SucceededSourceJob(
            job_id=job_id,
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=updated_at,
            attempt_count=attempt_count,
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
            attempt_count=attempt_count,
            started_at=cast(datetime | None, row["started_at"]),
            finished_at=cast(datetime, row["finished_at"]),
            error_code=cast(str, row["error_code"]),
            error_message=cast(str, row["error_message"]),
        )
    raise ExtractionJobStateError("Source Extraction job has an invalid stored status")
