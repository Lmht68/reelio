"""Deterministic contract tests for Source Extraction job storage."""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

import pytest

from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import (
    EXTRACTION_RESPONSE_VERSION,
    ExtractionJobStateError,
    ExtractionJobStatus,
    QueuedSourceJob,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.schemas import ExtractResponse

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_SUBMITTED_AT = datetime(2026, 10, 15, tzinfo=UTC)

_OWNER_A = bytes(range(32))
_OWNER_B = bytes(reversed(range(32)))


def _idempotency_key(number: int) -> UUID:
    """Return one deterministic valid idempotency key."""
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


_DEFAULT_IDEMPOTENCY_KEY = _idempotency_key(1)


def _response() -> ExtractResponse:
    """Build one known successful public extraction response."""
    return ExtractResponse.model_validate(
        {
            "market": "JP",
            "source": {
                "platform": "youtube",
                "video_id": "dQw4w9WgXcQ",
                "url": _SUBMITTED_URL,
                "title": "Job store source",
                "description": "Job store description.",
                "channel": "Job store channel",
                "duration_seconds": 42,
            },
            "transcript": {
                "text": "A job store transcript.",
                "language": "en",
                "method": "youtube_captions",
            },
            "statistics": {
                "movies": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "tv_series": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "tracks": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "music_releases": {
                    "n_mentions": 0,
                    "n_resolved": 0,
                    "n_unresolved": 0,
                },
                "books": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            },
            "results": {
                "movies": [],
                "tv_series": [],
                "tracks": [],
                "music_releases": [],
                "books": [],
            },
        }
    )


def _uuid_factory(*job_ids: UUID) -> Callable[[], UUID]:
    """Return deterministic UUIDs in the supplied order."""
    remaining_job_ids = iter(job_ids)
    return lambda: next(remaining_job_ids)


def _store(*job_ids: UUID) -> InMemoryExtractionJobStore:
    """Build a job store with deterministic admission identity and clock."""
    return InMemoryExtractionJobStore(
        clock=lambda: _SUBMITTED_AT,
        uuid_factory=_uuid_factory(*job_ids),
    )


async def _admit(
    store: InMemoryExtractionJobStore,
    source_url: str = _SUBMITTED_URL,
    *,
    market: SpotifyMarket | None = None,
    owner_digest: bytes = _OWNER_A,
    idempotency_key: UUID = _DEFAULT_IDEMPOTENCY_KEY,
) -> QueuedSourceJob:
    """Admit one Source job with a deterministic anonymous owner."""
    return await store.admit_source(
        source_url,
        market=market,
        owner_digest=owner_digest,
        idempotency_key=idempotency_key,
    )


async def test_admission_returns_one_deterministic_queued_snapshot() -> None:
    """Admit the exact Source input as a committed queued job."""
    job_id = UUID("00000000-0000-4000-8000-000000000002")
    store = _store(job_id)

    admitted = await _admit(store)

    assert admitted.job_id == job_id
    assert admitted.source_url == _SUBMITTED_URL
    assert admitted.market is None
    assert admitted.status is ExtractionJobStatus.QUEUED
    assert admitted.submitted_at == _SUBMITTED_AT
    assert admitted.updated_at == _SUBMITTED_AT
    assert await store.get_for_owner(job_id, _OWNER_A) == admitted


async def test_claims_oldest_queued_job_by_submission_time_then_identifier() -> None:
    """Claim the deterministic oldest queued Source job first."""
    later_job_id = UUID("00000000-0000-4000-8000-000000000002")
    earlier_job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(later_job_id, earlier_job_id)

    await _admit(
        store,
        "https://www.youtube.com/watch?v=later",
        idempotency_key=_idempotency_key(1),
    )
    await _admit(
        store,
        "https://www.youtube.com/watch?v=earlier",
        idempotency_key=_idempotency_key(2),
    )

    claimed = await store.claim_next_source()

    assert claimed is not None
    assert claimed.job_id == earlier_job_id
    assert claimed.status is ExtractionJobStatus.RUNNING
    assert claimed.started_at == _SUBMITTED_AT


async def test_claim_never_returns_one_job_twice() -> None:
    """Move a claimed job out of the queued claim set atomically."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    await _admit(store)

    first_claim = await store.claim_next_source()
    second_claim = await store.claim_next_source()

    assert first_claim is not None
    assert first_claim.job_id == job_id
    assert second_claim is None


async def test_running_job_transitions_to_versioned_succeeded_response() -> None:
    """Persist the existing response only after a running job completes."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    await _admit(store)
    await store.claim_next_source()

    completed = await store.complete_success(job_id, _response())

    assert completed.job_id == job_id
    assert completed.status is ExtractionJobStatus.SUCCEEDED
    assert completed.result_version == EXTRACTION_RESPONSE_VERSION
    assert completed.result.model_dump(mode="json") == _response().model_dump(mode="json")
    stored = await store.get_for_owner(job_id, _OWNER_A)
    assert stored == completed


async def test_completion_rejects_every_nonrunning_job_state() -> None:
    """Prevent a queued job from becoming successful without a claim."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    await _admit(store)

    with pytest.raises(ExtractionJobStateError):
        await store.complete_success(job_id, _response())


async def test_admission_replays_the_original_job_for_the_same_owner_and_key() -> None:
    """Reuse one owner-scoped idempotency key after a lifecycle transition."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)

    admitted = await _admit(store)
    await store.claim_next_source()
    replayed = await _admit(
        store,
        "https://www.youtube.com/watch?v=changed",
    )

    assert replayed == admitted
    assert (await store.get_for_owner(job_id, _OWNER_A)) is not None


async def test_admission_allows_different_owners_to_reuse_one_idempotency_key() -> None:
    """Scope idempotency reuse to one anonymous owner."""
    first_job_id = UUID("00000000-0000-4000-8000-000000000001")
    second_job_id = UUID("00000000-0000-4000-8000-000000000002")
    store = _store(first_job_id, second_job_id)
    idempotency_key = _idempotency_key(1)

    first = await _admit(store, owner_digest=_OWNER_A, idempotency_key=idempotency_key)
    second = await _admit(store, owner_digest=_OWNER_B, idempotency_key=idempotency_key)

    assert first.job_id == first_job_id
    assert second.job_id == second_job_id


async def test_read_returns_a_job_only_for_its_anonymous_owner() -> None:
    """Hide an existing job from a different anonymous owner."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    admitted = await _admit(store)

    assert await store.get_for_owner(job_id, _OWNER_A) == admitted
    assert await store.get_for_owner(job_id, _OWNER_B) is None
