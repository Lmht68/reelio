"""Deterministic contract tests for Source Extraction job storage."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import (
    EXTRACTION_RESPONSE_VERSION,
    AdmittedSourceJob,
    ExtractionJobCapacityExceededError,
    ExtractionJobStateError,
    ExtractionJobStatus,
    FailedSourceJob,
    QueuedSourceJob,
    RunningSourceJob,
    SucceededSourceJob,
    decode_stored_extraction_response,
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


def test_stored_result_decoder_accepts_only_the_current_response_contract() -> None:
    """Reject unsupported versions and malformed durable successful responses."""
    stored_response = _response().model_dump(mode="json")

    decoded_response = decode_stored_extraction_response(
        EXTRACTION_RESPONSE_VERSION,
        stored_response,
    )

    assert decoded_response.model_dump(mode="json") == stored_response
    with pytest.raises(ExtractionJobStateError, match="unsupported stored result version"):
        decode_stored_extraction_response("extraction-response-v2", stored_response)
    with pytest.raises(ExtractionJobStateError, match="malformed stored result"):
        decode_stored_extraction_response(EXTRACTION_RESPONSE_VERSION, {})


def _uuid_factory(*job_ids: UUID) -> Callable[[], UUID]:
    """Return deterministic UUIDs in the supplied order."""
    remaining_job_ids = iter(job_ids)
    return lambda: next(remaining_job_ids)


class _MutableClock:
    """Advance deterministic UTC time between durable store operations."""

    def __init__(self, now: datetime = _SUBMITTED_AT) -> None:
        """Initialize the clock at one known instant."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current UTC instant."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the current instant by an exact number of seconds."""
        self.now += timedelta(seconds=seconds)


def _lease_token(number: int) -> UUID:
    """Return one deterministic UUID4 lease token."""
    return UUID(f"10000000-0000-4000-8000-{number:012d}")


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
) -> AdmittedSourceJob:
    """Admit one Source job with a deterministic anonymous owner."""
    return await store.admit_source(
        source_url,
        market=market,
        owner_digest=owner_digest,
        idempotency_key=idempotency_key,
    )


@pytest.mark.parametrize("active_job_cap", [0, -1])
def test_in_memory_store_rejects_nonpositive_active_job_capacity(active_job_cap: int) -> None:
    """Require a positive direct active job capacity."""
    with pytest.raises(ValueError, match="active_job_cap must be positive"):
        InMemoryExtractionJobStore(active_job_cap=active_job_cap)


async def test_concurrent_admission_caps_active_jobs_and_releases_rejected_keys() -> None:
    """Admit ten active jobs, replay them, then release a rejected key after success."""
    idempotency_keys = tuple(_idempotency_key(number) for number in range(1, 12))
    job_ids = tuple(UUID(f"10000000-0000-4000-8000-{number:012d}") for number in range(1, 12))
    store = InMemoryExtractionJobStore(
        clock=lambda: _SUBMITTED_AT,
        uuid_factory=_uuid_factory(*job_ids),
        active_job_cap=10,
    )

    outcomes = await asyncio.gather(
        *(_admit(store, idempotency_key=idempotency_key) for idempotency_key in idempotency_keys),
        return_exceptions=True,
    )
    admitted = [
        (idempotency_key, outcome)
        for idempotency_key, outcome in zip(idempotency_keys, outcomes, strict=True)
        if isinstance(outcome, AdmittedSourceJob)
    ]
    rejected_keys = [
        idempotency_key
        for idempotency_key, outcome in zip(idempotency_keys, outcomes, strict=True)
        if isinstance(outcome, ExtractionJobCapacityExceededError)
    ]

    assert len(admitted) == 10
    assert len(rejected_keys) == 1

    replayed = await _admit(store, idempotency_key=admitted[0][0])
    assert replayed == admitted[0][1]

    running = await store.claim_next_source()
    assert running is not None
    await store.complete_success(running.job_id, running.lease_token, _response())

    admitted_after_completion = await _admit(store, idempotency_key=rejected_keys[0])
    assert admitted_after_completion.job_id == job_ids[10]


async def test_capacity_rejection_cleans_expired_jobs_before_consuming_identity() -> None:
    """Read the clock for bounded expiration cleanup before checking active capacity."""
    clock_calls = 0
    identity_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        return _SUBMITTED_AT

    def uuid_factory() -> UUID:
        nonlocal identity_calls
        identity_calls += 1
        return UUID("00000000-0000-4000-8000-000000000001")

    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=uuid_factory,
        active_job_cap=1,
    )
    await _admit(store, idempotency_key=_idempotency_key(1))

    with pytest.raises(ExtractionJobCapacityExceededError):
        await _admit(store, idempotency_key=_idempotency_key(2))

    assert clock_calls == 2
    assert identity_calls == 1


async def test_admission_returns_one_deterministic_job_identity() -> None:
    """Return only the committed identity while queued input remains internal."""
    job_id = UUID("00000000-0000-4000-8000-000000000002")
    store = _store(job_id)

    admitted = await _admit(store)
    stored = await store.get_for_owner(job_id, _OWNER_A)

    assert admitted == AdmittedSourceJob(job_id=job_id)
    assert isinstance(stored, QueuedSourceJob)
    assert stored.source_url == _SUBMITTED_URL
    assert stored.market is None
    assert stored.status is ExtractionJobStatus.QUEUED
    assert stored.attempt_count == 0
    assert stored.eligible_at == _SUBMITTED_AT
    assert stored.submitted_at == _SUBMITTED_AT
    assert stored.updated_at == _SUBMITTED_AT


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


async def test_retry_delays_the_next_claim_and_issues_a_fresh_lease() -> None:
    """Keep a retry queued until its eligibility time and increment its attempt."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await _admit(store)
    first_attempt = await store.claim_next_source()

    assert first_attempt is not None
    assert first_attempt.attempt_count == 1
    assert await store.retry_source_attempt(
        job_id,
        first_attempt.lease_token,
        timedelta(seconds=10),
    )
    assert await store.claim_next_source() is None

    clock.advance(10)
    second_attempt = await store.claim_next_source()

    assert second_attempt is not None
    assert second_attempt.attempt_count == 2
    assert second_attempt.lease_token != first_attempt.lease_token


async def test_final_requeued_attempt_becomes_a_terminal_internal_failure() -> None:
    """Prevent graceful requeue from leaving a third attempt active forever."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2), _lease_token(3))).__next__,
    )
    await _admit(store)

    first_attempt = await store.claim_next_source()
    assert first_attempt is not None
    assert await store.retry_source_attempt(
        job_id,
        first_attempt.lease_token,
        timedelta(seconds=1),
    )
    clock.advance(1)

    second_attempt = await store.claim_next_source()
    assert second_attempt is not None
    assert await store.retry_source_attempt(
        job_id,
        second_attempt.lease_token,
        timedelta(seconds=1),
    )
    clock.advance(1)

    third_attempt = await store.claim_next_source()
    assert third_attempt is not None
    assert third_attempt.attempt_count == 3
    assert await store.requeue_source_attempt(job_id, third_attempt.lease_token)

    assert await store.claim_next_source() is None
    stored = await store.get_for_owner(job_id, _OWNER_A)

    assert isinstance(stored, FailedSourceJob)
    assert stored.error_code == "internal_error"
    assert stored.error_message == "An unexpected error occurred."
    assert stored.expires_at == clock.now + timedelta(days=7)
    assert not hasattr(stored, "source_url")
    assert not hasattr(stored, "market")


async def test_expired_final_attempt_becomes_a_terminal_internal_failure() -> None:
    """Prevent an expired third lease from remaining active without another claim."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2), _lease_token(3))).__next__,
    )
    await _admit(store)

    first_attempt = await store.claim_next_source()
    assert first_attempt is not None
    assert await store.retry_source_attempt(
        job_id,
        first_attempt.lease_token,
        timedelta(seconds=1),
    )
    clock.advance(1)

    second_attempt = await store.claim_next_source()
    assert second_attempt is not None
    assert await store.retry_source_attempt(
        job_id,
        second_attempt.lease_token,
        timedelta(seconds=1),
    )
    clock.advance(1)

    third_attempt = await store.claim_next_source()
    assert third_attempt is not None
    clock.advance(60)

    assert await store.claim_next_source() is None
    stored = await store.get_for_owner(job_id, _OWNER_A)

    assert isinstance(stored, FailedSourceJob)
    assert stored.error_code == "internal_error"
    assert stored.error_message == "An unexpected error occurred."
    assert stored.expires_at == clock.now + timedelta(days=7)
    assert not hasattr(stored, "source_url")
    assert not hasattr(stored, "market")


async def test_running_job_transitions_to_versioned_succeeded_response() -> None:
    """Persist the existing response only after a running job completes."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    await _admit(store)
    running = await store.claim_next_source()

    assert running is not None
    completed = await store.complete_success(job_id, running.lease_token, _response())

    assert completed.job_id == job_id
    assert completed.status is ExtractionJobStatus.SUCCEEDED
    assert completed.result_version == EXTRACTION_RESPONSE_VERSION
    assert completed.expires_at == _SUBMITTED_AT + timedelta(days=7)
    assert completed.result.model_dump(mode="json") == _response().model_dump(mode="json")
    assert not hasattr(completed, "source_url")
    assert not hasattr(completed, "market")
    stored = await store.get_for_owner(job_id, _OWNER_A)
    assert stored == completed


async def test_completion_rejects_every_nonrunning_job_state() -> None:
    """Prevent a queued job from becoming successful without a claim."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = _store(job_id)
    await _admit(store)

    with pytest.raises(ExtractionJobStateError):
        await store.complete_success(job_id, _lease_token(1), _response())


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

    owned_job = await store.get_for_owner(job_id, _OWNER_A)

    assert isinstance(owned_job, QueuedSourceJob)
    assert owned_job.job_id == admitted.job_id
    assert await store.get_for_owner(job_id, _OWNER_B) is None


async def test_claimed_attempt_has_a_sixty_second_lease_and_renews_from_now() -> None:
    """Extend one valid attempt lease from the authority time of its heartbeat."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=lambda: _lease_token(1),
    )
    await _admit(store)

    claimed = await store.claim_next_source()

    assert claimed is not None
    assert claimed.lease_token == _lease_token(1)
    assert claimed.lease_expires_at == _SUBMITTED_AT + timedelta(seconds=60)

    clock.advance(15)

    assert await store.renew_source_attempt(job_id, claimed.lease_token) is True
    renewed = await store.get_for_owner(job_id, _OWNER_A)

    assert isinstance(renewed, RunningSourceJob)
    assert renewed.lease_token == claimed.lease_token
    assert renewed.updated_at == _SUBMITTED_AT + timedelta(seconds=15)
    assert renewed.lease_expires_at == _SUBMITTED_AT + timedelta(seconds=75)


async def test_expired_attempt_is_reclaimed_at_its_exact_expiration_boundary() -> None:
    """Replace an expired owner at the exact moment its lease is no longer valid."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await _admit(store)
    original = await store.claim_next_source()

    assert original is not None
    clock.advance(60)

    assert await store.renew_source_attempt(job_id, original.lease_token) is False
    reclaimed = await store.claim_next_source()

    assert reclaimed is not None
    assert reclaimed.job_id == job_id
    assert reclaimed.lease_token == _lease_token(2)
    assert reclaimed.lease_token != original.lease_token
    assert reclaimed.started_at == _SUBMITTED_AT + timedelta(seconds=60)
    assert reclaimed.lease_expires_at == _SUBMITTED_AT + timedelta(seconds=120)


async def test_reclaimed_attempt_rejects_stale_terminal_and_requeue_writes() -> None:
    """Fence every old terminal or requeue write after another worker reclaims a job."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await _admit(store)
    original = await store.claim_next_source()

    assert original is not None
    clock.advance(60)
    reclaimed = await store.claim_next_source()

    assert reclaimed is not None
    with pytest.raises(ExtractionJobStateError):
        await store.complete_success(job_id, original.lease_token, _response())
    with pytest.raises(ExtractionJobStateError):
        await store.complete_failure(
            job_id,
            original.lease_token,
            "pipeline_failed",
            "Source pipeline failed.",
        )
    assert await store.requeue_source_attempt(job_id, original.lease_token) is False
    assert (
        await store.retry_source_attempt(
            job_id,
            original.lease_token,
            timedelta(seconds=1),
        )
        is False
    )

    stored = await store.get_for_owner(job_id, _OWNER_A)
    assert stored == reclaimed


async def test_running_job_transitions_to_expiring_failed_response_without_input() -> None:
    """Retain only one safe failure and its seven-day availability window."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=lambda: _lease_token(1),
    )
    await _admit(store, market=SpotifyMarket("JP"))
    running = await store.claim_next_source()

    assert running is not None
    completed = await store.complete_failure(
        job_id,
        running.lease_token,
        "catalog_provider_failed",
        "Catalog provider request failed.",
    )
    stored = await store.get_for_owner(job_id, _OWNER_A)

    assert completed.expires_at == _SUBMITTED_AT + timedelta(days=7)
    assert not hasattr(completed, "source_url")
    assert not hasattr(completed, "market")
    assert stored == completed


async def test_terminal_jobs_are_hidden_at_the_exact_expiration_boundary() -> None:
    """Keep successful and failed jobs visible through day seven minus one microsecond."""
    succeeded_job_id = UUID("00000000-0000-4000-8000-000000000001")
    failed_job_id = UUID("00000000-0000-4000-8000-000000000002")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=_uuid_factory(succeeded_job_id, failed_job_id),
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await _admit(store, idempotency_key=_idempotency_key(1))
    await _admit(store, idempotency_key=_idempotency_key(2))
    successful_attempt = await store.claim_next_source()
    assert successful_attempt is not None
    await store.complete_success(
        successful_attempt.job_id,
        successful_attempt.lease_token,
        _response(),
    )
    failed_attempt = await store.claim_next_source()
    assert failed_attempt is not None
    await store.complete_failure(
        failed_attempt.job_id,
        failed_attempt.lease_token,
        "catalog_provider_failed",
        "Catalog provider request failed.",
    )

    clock.now += timedelta(days=7) - timedelta(microseconds=1)

    assert isinstance(
        await store.get_for_owner(succeeded_job_id, _OWNER_A),
        SucceededSourceJob,
    )
    assert isinstance(
        await store.get_for_owner(failed_job_id, _OWNER_A),
        FailedSourceJob,
    )

    clock.now += timedelta(microseconds=1)

    assert await store.get_for_owner(succeeded_job_id, _OWNER_A) is None
    assert await store.get_for_owner(failed_job_id, _OWNER_A) is None


async def test_admission_cleanup_deletes_one_ordered_batch_without_touching_active_work() -> None:
    """Delete 100 expired terminals and retire their replay metadata in one admission path."""
    terminal_job_ids = tuple(
        UUID(f"00000000-0000-4000-8000-{number:012d}") for number in range(1, 102)
    )
    active_job_id = UUID("00000000-0000-4000-8000-000000000102")
    replacement_job_id = UUID("00000000-0000-4000-8000-000000000103")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=_uuid_factory(*terminal_job_ids, active_job_id, replacement_job_id),
        lease_token_factory=lambda: _lease_token(1),
    )
    for number, job_id in enumerate(terminal_job_ids, start=1):
        await _admit(store, idempotency_key=_idempotency_key(number))
        attempt = await store.claim_next_source()
        assert attempt is not None
        assert attempt.job_id == job_id
        await store.complete_success(job_id, attempt.lease_token, _response())
    await _admit(store, idempotency_key=_idempotency_key(102))

    clock.now += timedelta(days=7)

    active_attempt = await store.claim_next_source()

    assert active_attempt is not None
    assert active_attempt.job_id == active_job_id
    remaining_terminal_ids = {
        job_id
        for job_id, job in store._jobs.items()
        if isinstance(job, SucceededSourceJob | FailedSourceJob)
    }
    assert remaining_terminal_ids == {terminal_job_ids[-1]}
    assert _idempotency_key(1) not in {
        key for owner, key in store._admissions if owner == _OWNER_A
    }
    assert store._admissions[(_OWNER_A, _idempotency_key(101))] == terminal_job_ids[-1]

    replacement = await _admit(store, idempotency_key=_idempotency_key(1))

    assert replacement.job_id == replacement_job_id
    assert isinstance(store._jobs[active_job_id], RunningSourceJob)
