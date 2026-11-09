"""Application-owned Source Extraction job state and storage protocol."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.schemas import ExtractResponse

EXTRACTION_RESPONSE_VERSION = "extraction-response-v1"


class ExtractionJobStatus(StrEnum):
    """Represent durable lifecycle states for one Source Extraction job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExtractionJobStoreUnavailableError(RuntimeError):
    """Indicate that durable Source Extraction job storage is unavailable."""


class ExtractionJobStateError(RuntimeError):
    """Indicate that a Source Extraction job cannot make the requested transition."""


@dataclass(frozen=True, slots=True)
class _SourceExtractionJobSnapshot:
    """Contain immutable Source input shared by every lifecycle snapshot."""

    job_id: UUID
    source_url: str
    market: SpotifyMarket | None
    submitted_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class QueuedSourceJob(_SourceExtractionJobSnapshot):
    """Represent a committed Source Extraction job awaiting a worker claim."""

    status: ExtractionJobStatus = field(
        init=False,
        default=ExtractionJobStatus.QUEUED,
    )


@dataclass(frozen=True, slots=True)
class RunningSourceJob(_SourceExtractionJobSnapshot):
    """Represent one Source Extraction job exclusively claimed by a worker."""

    started_at: datetime
    status: ExtractionJobStatus = field(
        init=False,
        default=ExtractionJobStatus.RUNNING,
    )


@dataclass(frozen=True, slots=True)
class SucceededSourceJob(_SourceExtractionJobSnapshot):
    """Represent a Source Extraction job with its versioned public response."""

    started_at: datetime
    finished_at: datetime
    result_version: str
    result: ExtractResponse
    status: ExtractionJobStatus = field(
        init=False,
        default=ExtractionJobStatus.SUCCEEDED,
    )


@dataclass(frozen=True, slots=True)
class FailedSourceJob(_SourceExtractionJobSnapshot):
    """Represent a terminal Source Extraction failure retained for later projection."""

    started_at: datetime | None
    finished_at: datetime
    error_code: str
    error_message: str
    status: ExtractionJobStatus = field(
        init=False,
        default=ExtractionJobStatus.FAILED,
    )


type SourceExtractionJob = QueuedSourceJob | RunningSourceJob | SucceededSourceJob | FailedSourceJob


class ExtractionJobStore(Protocol):
    """Persist and atomically transition Source Extraction job snapshots."""

    async def admit_source(
        self,
        source_url: str,
        market: SpotifyMarket | None,
    ) -> QueuedSourceJob:
        """Commit one queued Source Extraction job.

        Args:
            source_url: Validated public Source URL submitted by the caller.
            market: Optional requested Effective Market.

        Returns:
            The committed queued job snapshot.

        Raises:
            ExtractionJobStoreUnavailableError: If durable admission cannot commit.
        """
        ...

    async def get(self, job_id: UUID) -> SourceExtractionJob | None:
        """Return one current Source Extraction job snapshot when it exists.

        Args:
            job_id: Opaque Source Extraction job identity.

        Returns:
            The current snapshot, or None when no job has that identity.

        Raises:
            ExtractionJobStoreUnavailableError: If storage cannot complete the read.
        """
        ...

    async def claim_next_source(self) -> RunningSourceJob | None:
        """Atomically claim the oldest queued Source Extraction job.

        Returns:
            The committed running snapshot, or None when no job is queued.

        Raises:
            ExtractionJobStoreUnavailableError: If storage cannot complete the claim.
        """
        ...

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
            ExtractionJobStoreUnavailableError: If storage cannot complete the update.
        """
        ...

    async def aclose(self) -> None:
        """Release resources owned by this job store."""
        ...
