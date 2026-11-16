"""Behavioral tests for bounded Source Extraction job execution."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import QueuedSourceJob, RunningSourceJob, SucceededSourceJob
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.types import (
    BookResults,
    ExtractionResults,
    MusicResults,
    PipelineResult,
    Platform,
    ScreenWorkResults,
    Source,
    Transcript,
    TranscriptMethod,
    TranscriptPipelineResult,
)
from reelio.extraction.worker import ExtractionJobWorker
from reelio.main import create_app
from tests.extraction.fakes import FakeInterpretationService, FakeResultAggregator

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

_OWNER_DIGEST = b"worker-owner-digest-012345678901"
_IDEMPOTENCY_KEY = UUID("00000000-0000-4000-8000-000000000001")


def _pipeline_result(market: SpotifyMarket) -> PipelineResult:
    """Build one known Source result with no resolved or unresolved mentions."""
    return PipelineResult(
        source=Source(
            platform=Platform.YOUTUBE,
            video_id="dQw4w9WgXcQ",
            url=_SUBMITTED_URL,
            title="Worker source",
            description="Worker description.",
            channel="Worker channel",
            duration_seconds=42,
        ),
        transcript=Transcript(
            text="Worker transcript.",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        ),
        results=ExtractionResults(
            screen_works=ScreenWorkResults(movies=[], tv_series=[]),
            music=MusicResults(tracks=[], music_releases=[]),
            books=BookResults(books=[]),
        ),
        market=market,
    )


class _RecordingPipeline:
    """Record Source pipeline calls and return a prepared pipeline result."""

    def __init__(self, result: PipelineResult) -> None:
        """Initialize a pipeline that returns one known Source result.

        Args:
            result: Result returned for every Source execution.
        """
        self._result = result
        self.calls: list[tuple[str, SpotifyMarket | None]] = []

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Record and return one prepared Source pipeline result."""
        self.calls.append((url, market))
        return self._result

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        """Reject Transcript execution because worker jobs accept Sources only."""
        del transcript_text, market
        raise AssertionError("unexpected transcript pipeline call")

    async def aclose(self) -> None:
        """Release no resources for this deterministic pipeline fake."""


async def test_run_once_returns_false_without_a_queued_job() -> None:
    """Avoid pipeline execution when no Source job is ready to claim."""
    store = InMemoryExtractionJobStore()
    pipeline = _RecordingPipeline(_pipeline_result(SpotifyMarket("US")))
    worker = ExtractionJobWorker(store, pipeline)

    ran = await worker.run_once()

    assert ran is False
    assert pipeline.calls == []


async def test_run_once_delivers_the_exact_source_and_requested_market() -> None:
    """Run a claimed Source job with its preserved input values."""
    store = InMemoryExtractionJobStore(
        uuid_factory=lambda: UUID("00000000-0000-4000-8000-000000000001")
    )
    await store.admit_source(
        _SUBMITTED_URL,
        SpotifyMarket("JP"),
        _OWNER_DIGEST,
        _IDEMPOTENCY_KEY,
    )
    pipeline = _RecordingPipeline(_pipeline_result(SpotifyMarket("JP")))
    worker = ExtractionJobWorker(store, pipeline)

    ran = await worker.run_once()

    assert ran is True
    assert pipeline.calls == [(_SUBMITTED_URL, SpotifyMarket("JP"))]


async def test_run_once_persists_the_existing_mapper_response() -> None:
    """Store the public response mapped from the pipeline's Source result."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    worker = ExtractionJobWorker(store, _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))))

    assert await worker.run_once() is True
    stored = await store.get_for_owner(job_id, _OWNER_DIGEST)

    assert isinstance(stored, SucceededSourceJob)
    assert stored.result.model_dump(mode="json") == {
        "market": "US",
        "source": {
            "platform": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "url": _SUBMITTED_URL,
            "title": "Worker source",
            "description": "Worker description.",
            "channel": "Worker channel",
            "duration_seconds": 42,
        },
        "transcript": {
            "text": "Worker transcript.",
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


async def test_run_once_commits_an_empty_successful_result() -> None:
    """Retain a successful Source result that has no grouped mentions."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    worker = ExtractionJobWorker(store, _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))))

    assert await worker.run_once() is True
    stored = await store.get_for_owner(job_id, _OWNER_DIGEST)

    assert isinstance(stored, SucceededSourceJob)
    assert stored.result.results.movies == []
    assert stored.result.results.tv_series == []
    assert stored.result.results.tracks == []
    assert stored.result.results.music_releases == []
    assert stored.result.results.books == []


class _BlockingPipeline(_RecordingPipeline):
    """Hold one Source execution open to observe worker-instance serialization."""

    def __init__(self, result: PipelineResult) -> None:
        """Initialize concurrency controls around one prepared pipeline result.

        Args:
            result: Result released after the test allows the pipeline to finish.
        """
        super().__init__(result)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.running_calls = 0
        self.overlapped = False

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Block one Source execution and record any forbidden overlap."""
        self.running_calls += 1
        self.overlapped = self.overlapped or self.running_calls > 1
        self.calls.append((url, market))
        self.started.set()
        await self.release.wait()
        self.running_calls -= 1
        return self._result


class _ThreeJobBlockingPipeline(_RecordingPipeline):
    """Block three Source jobs to observe a two-slot scheduler."""

    def __init__(self, result: PipelineResult) -> None:
        """Initialize controls for two blocked jobs followed by one queued job.

        Args:
            result: Result returned after each job is released.
        """
        super().__init__(result)
        self.first_two_started = asyncio.Event()
        self.third_started = asyncio.Event()
        self.release_first_calls = (asyncio.Event(), asyncio.Event())
        self.release_third = asyncio.Event()
        self.running_calls = 0
        self.peak_overlap = 0

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Block each call until its deterministic test release."""
        call_index = len(self.calls)
        self.calls.append((url, market))
        self.running_calls += 1
        self.peak_overlap = max(self.peak_overlap, self.running_calls)
        if call_index == 1:
            self.first_two_started.set()

        if call_index < 2:
            await self.release_first_calls[call_index].wait()
        else:
            self.third_started.set()
            await self.release_third.wait()

        self.running_calls -= 1
        return self._result


class _BlockingSourceTranscriptProvider:
    """Block real pipeline acquisition until the lifecycle test releases it."""

    def __init__(self) -> None:
        """Initialize acquisition synchronization for one known Source."""
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        """Block and return one canonical Source and Transcript.

        Args:
            submitted_url: Exact URL passed from the admitted Source job.

        Returns:
            Canonical Source and its normalized Transcript.
        """
        self.calls.append(submitted_url)
        self.started.set()
        await self.release.wait()
        return (
            Source(
                platform=Platform.YOUTUBE,
                video_id="dQw4w9WgXcQ",
                url=submitted_url,
                title="Lifecycle source",
                description="Lifecycle description.",
                channel="Lifecycle channel",
                duration_seconds=42,
            ),
            Transcript(
                text="Lifecycle transcript.",
                language="en",
                method=TranscriptMethod.YOUTUBE_CAPTIONS,
            ),
        )

    async def aclose(self) -> None:
        """Release no resources for the blocked deterministic provider."""


@pytest.mark.parametrize("concurrency", [0, -1])
def test_worker_rejects_nonpositive_concurrency(concurrency: int) -> None:
    """Require a positive worker slot count."""
    with pytest.raises(ValueError, match="concurrency must be positive"):
        ExtractionJobWorker(
            InMemoryExtractionJobStore(),
            _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))),
            concurrency=concurrency,
        )


async def test_run_until_idle_honors_a_configured_single_slot() -> None:
    """Keep a second job queued until the configured single slot completes."""
    job_ids = (
        UUID("00000000-0000-4000-8000-000000000001"),
        UUID("00000000-0000-4000-8000-000000000002"),
    )
    store = InMemoryExtractionJobStore(uuid_factory=iter(job_ids).__next__)
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    await store.admit_source(
        "https://www.youtube.com/watch?v=second",
        None,
        _OWNER_DIGEST,
        UUID("00000000-0000-4000-8000-000000000002"),
    )
    pipeline = _BlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    worker = ExtractionJobWorker(store, pipeline, concurrency=1)

    worker_run = asyncio.create_task(worker.run_until_idle())
    await pipeline.started.wait()

    second_job = await store.get_for_owner(job_ids[1], _OWNER_DIGEST)
    assert isinstance(second_job, QueuedSourceJob)
    assert pipeline.calls == [(_SUBMITTED_URL, None)]

    pipeline.release.set()

    assert await worker_run == 2
    assert pipeline.overlapped is False


async def test_run_until_idle_uses_two_slots_then_claims_the_third_job() -> None:
    """Hold two jobs running until either completion frees the third queued job."""
    job_ids = tuple(UUID(f"00000000-0000-4000-8000-{number:012d}") for number in range(1, 4))
    store = InMemoryExtractionJobStore(uuid_factory=iter(job_ids).__next__)
    for number in range(1, 4):
        await store.admit_source(
            f"https://www.youtube.com/watch?v=job-{number}",
            None,
            _OWNER_DIGEST,
            UUID(f"10000000-0000-4000-8000-{number:012d}"),
        )
    pipeline = _ThreeJobBlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    worker = ExtractionJobWorker(store, pipeline)

    worker_run = asyncio.create_task(worker.run_until_idle())
    await pipeline.first_two_started.wait()

    third_job = await store.get_for_owner(job_ids[2], _OWNER_DIGEST)
    assert isinstance(third_job, QueuedSourceJob)
    assert pipeline.running_calls == 2
    assert pipeline.peak_overlap == 2

    pipeline.release_first_calls[0].set()
    await pipeline.third_started.wait()

    assert pipeline.peak_overlap == 2

    pipeline.release_first_calls[1].set()
    pipeline.release_third.set()

    assert await worker_run == 3
    for job_id in job_ids:
        completed_job = await store.get_for_owner(job_id, _OWNER_DIGEST)
        assert isinstance(completed_job, SucceededSourceJob)


async def test_public_job_lifecycle_runs_the_real_pipeline_and_mapper() -> None:
    """Expose queued, running, then the mapped successful Extraction response."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    source_provider = _BlockingSourceTranscriptProvider()
    pipeline = ExtractionPipeline(
        source_transcript_provider=source_provider,
        interpretation_service=FakeInterpretationService(),
        result_aggregator=FakeResultAggregator(),
    )
    worker = ExtractionJobWorker(store, pipeline)

    async def pipeline_factory() -> ExtractionPipeline:
        return pipeline

    async def job_store_factory() -> InMemoryExtractionJobStore:
        return store

    application = create_app(
        pipeline_factory=pipeline_factory,
        job_store_factory=job_store_factory,
    )
    expected_result = {
        "market": "JP",
        "source": {
            "platform": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "url": _SUBMITTED_URL,
            "title": "Lifecycle source",
            "description": "Lifecycle description.",
            "channel": "Lifecycle channel",
            "duration_seconds": 42,
        },
        "transcript": {
            "text": "Lifecycle transcript.",
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

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            admitted = await client.post(
                "/api/extraction-jobs",
                json={"url": _SUBMITTED_URL, "market": "JP"},
                headers={"Idempotency-Key": str(_IDEMPOTENCY_KEY)},
            )
            queued = await client.get(f"/api/extraction-jobs/{job_id}")
            worker_run = asyncio.create_task(worker.run_until_idle())
            await source_provider.started.wait()
            running = await client.get(f"/api/extraction-jobs/{job_id}")
            source_provider.release.set()
            assert await worker_run == 1
            succeeded = await client.get(f"/api/extraction-jobs/{job_id}")

    assert admitted.status_code == 202
    assert admitted.json() == {"job_id": str(job_id), "status": "queued"}
    assert queued.json() == {"job_id": str(job_id), "status": "queued"}
    assert running.json() == {"job_id": str(job_id), "status": "running"}
    assert succeeded.json() == {
        "job_id": str(job_id),
        "status": "succeeded",
        "result": expected_result,
    }
    assert source_provider.calls == [_SUBMITTED_URL]
    for response in (queued, running, succeeded):
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"


class _MutableClock:
    """Advance the deterministic authority time used by one in-memory job store."""

    def __init__(self, now: datetime = datetime(2026, 10, 15, tzinfo=UTC)) -> None:
        """Initialize the clock at one known UTC instant."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current authority instant."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance authority time by an exact number of seconds."""
        self.now += timedelta(seconds=seconds)


class _ControlledSleeper:
    """Release individual worker sleeps only when a test permits the heartbeat."""

    def __init__(self) -> None:
        """Initialize call and release queues."""
        self.calls: list[float] = []
        self._calls: asyncio.Queue[None] = asyncio.Queue()
        self._releases: asyncio.Queue[None] = asyncio.Queue()

    async def __call__(self, seconds: float) -> None:
        """Record one requested sleep and wait for an explicit test release."""
        self.calls.append(seconds)
        await self._calls.put(None)
        await self._releases.get()

    async def wait_for_call(self) -> None:
        """Wait until the worker starts another controlled sleep."""
        await self._calls.get()

    def release_one(self) -> None:
        """Release exactly one blocked controlled sleep."""
        self._releases.put_nowait(None)


class _CancellableBlockingPipeline(_RecordingPipeline):
    """Block one Source execution and observe local worker cancellation."""

    def __init__(self, result: PipelineResult) -> None:
        """Initialize cancellation and release synchronization."""
        super().__init__(result)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Block until released or cancelled, then return the prepared result."""
        self.calls.append((url, market))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return self._result


def _lease_token(number: int) -> UUID:
    """Return one deterministic UUID4 lease token."""
    return UUID(f"10000000-0000-4000-8000-{number:012d}")


@pytest.mark.parametrize("heartbeat_interval_seconds", [0, -1, float("inf")])
def test_worker_rejects_invalid_direct_heartbeat_interval(
    heartbeat_interval_seconds: float,
) -> None:
    """Require a finite positive direct heartbeat interval."""
    with pytest.raises(ValueError, match="finite and positive"):
        ExtractionJobWorker(
            InMemoryExtractionJobStore(),
            _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))),
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )


async def test_heartbeat_renews_every_interval_while_the_pipeline_is_active() -> None:
    """Renew a claimed attempt at the configured cadence until pipeline completion."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    pipeline = _CancellableBlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    sleeper = _ControlledSleeper()
    worker = ExtractionJobWorker(
        store,
        pipeline,
        heartbeat_interval_seconds=15.0,
        sleep=sleeper,
    )

    worker_run = asyncio.create_task(worker.run_once())
    await pipeline.started.wait()
    await sleeper.wait_for_call()
    assert sleeper.calls == [15.0]

    sleeper.release_one()
    await sleeper.wait_for_call()
    assert sleeper.calls == [15.0, 15.0]

    pipeline.release.set()

    assert await worker_run is True
    assert sleeper.calls == [15.0, 15.0]


async def test_lease_loss_cancels_pipeline_and_discards_its_result() -> None:
    """Stop local work without a terminal write once the attempt is no longer owned."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=lambda: _lease_token(1),
    )
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    pipeline = _CancellableBlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    sleeper = _ControlledSleeper()
    worker = ExtractionJobWorker(store, pipeline, sleep=sleeper)

    worker_run = asyncio.create_task(worker.run_once())
    await pipeline.started.wait()
    await sleeper.wait_for_call()
    clock.advance(60)
    sleeper.release_one()

    assert await worker_run is False
    assert pipeline.cancelled.is_set()
    stored = await store.get_for_owner(job_id, _OWNER_DIGEST)
    assert isinstance(stored, RunningSourceJob)
    assert stored.lease_token == _lease_token(1)


async def test_shutdown_requeues_only_still_owned_attempts_without_claiming_next_work() -> None:
    """Requeue owned cancellation work and preserve a reclaimed attempt unchanged."""
    first_job_id = UUID("00000000-0000-4000-8000-000000000001")
    second_job_id = UUID("00000000-0000-4000-8000-000000000002")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=iter((first_job_id, second_job_id)).__next__,
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    await store.admit_source(
        "https://www.youtube.com/watch?v=second",
        None,
        _OWNER_DIGEST,
        UUID("00000000-0000-4000-8000-000000000002"),
    )
    pipeline = _CancellableBlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    worker = ExtractionJobWorker(store, pipeline, concurrency=1)

    worker_run = asyncio.create_task(worker.run_until_idle())
    await pipeline.started.wait()
    clock.advance(60)
    reclaimed = await store.claim_next_source()

    assert reclaimed is not None
    assert reclaimed.job_id == first_job_id
    await worker.shutdown()

    assert await worker_run == 0
    assert pipeline.cancelled.is_set()
    first_job = await store.get_for_owner(first_job_id, _OWNER_DIGEST)
    second_job = await store.get_for_owner(second_job_id, _OWNER_DIGEST)
    assert first_job == reclaimed
    assert isinstance(second_job, QueuedSourceJob)


async def test_direct_worker_cancellation_leaves_running_work_for_lease_recovery() -> None:
    """Leave abrupt worker cancellation unfenced until another worker reclaims it."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    clock = _MutableClock()
    store = InMemoryExtractionJobStore(
        clock=clock,
        uuid_factory=lambda: job_id,
        lease_token_factory=iter((_lease_token(1), _lease_token(2))).__next__,
    )
    await store.admit_source(_SUBMITTED_URL, None, _OWNER_DIGEST, _IDEMPOTENCY_KEY)
    first_pipeline = _CancellableBlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    first_worker = ExtractionJobWorker(store, first_pipeline)

    first_run = asyncio.create_task(first_worker.run_until_idle())
    await first_pipeline.started.wait()
    first_run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first_run
    assert first_pipeline.cancelled.is_set()
    running = await store.get_for_owner(job_id, _OWNER_DIGEST)
    assert isinstance(running, RunningSourceJob)
    assert running.lease_token == _lease_token(1)

    clock.advance(60)
    second_pipeline = _RecordingPipeline(_pipeline_result(SpotifyMarket("US")))
    second_worker = ExtractionJobWorker(store, second_pipeline)

    assert await second_worker.run_once() is True
    recovered = await store.get_for_owner(job_id, _OWNER_DIGEST)
    assert isinstance(recovered, SucceededSourceJob)
