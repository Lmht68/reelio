"""Behavioral tests for one-slot Source Extraction job execution."""

import asyncio
from uuid import UUID

from httpx import ASGITransport, AsyncClient

from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import SucceededSourceJob
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
    await store.admit_source(_SUBMITTED_URL, SpotifyMarket("JP"))
    pipeline = _RecordingPipeline(_pipeline_result(SpotifyMarket("JP")))
    worker = ExtractionJobWorker(store, pipeline)

    ran = await worker.run_once()

    assert ran is True
    assert pipeline.calls == [(_SUBMITTED_URL, SpotifyMarket("JP"))]


async def test_run_once_persists_the_existing_mapper_response() -> None:
    """Store the public response mapped from the pipeline's Source result."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    await store.admit_source(_SUBMITTED_URL, None)
    worker = ExtractionJobWorker(store, _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))))

    assert await worker.run_once() is True
    stored = await store.get(job_id)

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
    await store.admit_source(_SUBMITTED_URL, None)
    worker = ExtractionJobWorker(store, _RecordingPipeline(_pipeline_result(SpotifyMarket("US"))))

    assert await worker.run_once() is True
    stored = await store.get(job_id)

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


async def test_concurrent_run_once_calls_do_not_overlap_pipeline_execution() -> None:
    """Serialize calls made concurrently to one one-slot worker instance."""
    store = InMemoryExtractionJobStore(
        uuid_factory=lambda: UUID("00000000-0000-4000-8000-000000000001")
    )
    await store.admit_source(_SUBMITTED_URL, None)
    pipeline = _BlockingPipeline(_pipeline_result(SpotifyMarket("US")))
    worker = ExtractionJobWorker(store, pipeline)

    first_run = asyncio.create_task(worker.run_once())
    await pipeline.started.wait()
    second_run = asyncio.create_task(worker.run_once())
    await asyncio.sleep(0)

    assert pipeline.calls == [(_SUBMITTED_URL, None)]
    pipeline.release.set()

    assert await first_run is True
    assert await second_run is False
    assert pipeline.overlapped is False


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
            )
            queued = await client.get(f"/api/extraction-jobs/{job_id}")
            worker_run = asyncio.create_task(worker.run_once())
            await source_provider.started.wait()
            running = await client.get(f"/api/extraction-jobs/{job_id}")
            source_provider.release.set()
            assert await worker_run is True
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
