"""Application composition and documentation lifecycle tests."""

import pytest
from httpx import ASGITransport, AsyncClient

import reelio.main as main_module
from reelio.config import Environment, app_settings
from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
    BookResults,
    ExtractionResults,
    MusicResults,
    PipelineResult,
    ScreenWorkResults,
    Transcript,
    TranscriptMethod,
    TranscriptPipelineResult,
)
from reelio.main import create_app


class _FakePipeline:
    def __init__(self) -> None:
        self.close_calls = 0
        self.transcript_calls: list[tuple[str, SpotifyMarket | None]] = []

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        raise AssertionError(f"unexpected pipeline call for {url}")

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        self.transcript_calls.append((transcript_text, market))
        return TranscriptPipelineResult(
            transcript=Transcript(
                text=transcript_text,
                language="und",
                method=TranscriptMethod.TEXT_SUBMISSION,
            ),
            results=ExtractionResults(
                screen_works=ScreenWorkResults(movies=[], tv_series=[]),
                music=MusicResults(tracks=[], music_releases=[]),
                books=BookResults(books=[]),
            ),
            market=market or SpotifyMarket("US"),
        )

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeJobStore(InMemoryExtractionJobStore):
    def __init__(self, close_events: list[str] | None = None) -> None:
        super().__init__()
        self.close_calls = 0
        self._close_events = close_events

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_events is not None:
            self._close_events.append("store")


class _CloseRecordingPipeline(_FakePipeline):
    def __init__(self, close_events: list[str]) -> None:
        super().__init__()
        self._close_events = close_events

    async def aclose(self) -> None:
        await super().aclose()
        self._close_events.append("pipeline")


@pytest.mark.parametrize(
    ("environment", "expected_status"),
    [(Environment.PRODUCTION, 404), (Environment.LOCAL, 200)],
)
async def test_docs_are_gated_by_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: Environment,
    expected_status: int,
) -> None:
    """Expose docs only in non-production environments."""
    monkeypatch.setattr(app_settings, "environment", environment)
    application = create_app()
    transport = ASGITransport(app=application)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/docs")

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("environment", "expected_status"),
    [
        (Environment.LOCAL, 200),
        (Environment.STAGING, 200),
        (Environment.PRODUCTION, 404),
    ],
)
async def test_internal_transcript_route_is_gated_by_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: Environment,
    expected_status: int,
) -> None:
    """Expose the internal Transcript route only outside production."""
    monkeypatch.setattr(app_settings, "environment", environment)
    pipeline = _FakePipeline()
    application = create_app()
    application.state.extraction_pipeline = pipeline
    transport = ASGITransport(app=application)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/internal/extractions",
            json={"transcript": "Dune: Part One (2021) was excellent."},
        )

    assert response.status_code == expected_status
    expected_calls = (
        [("Dune: Part One (2021) was excellent.", None)] if expected_status == 200 else []
    )
    assert pipeline.transcript_calls == expected_calls


async def test_injected_resource_factories_own_one_resource_per_lifespan() -> None:
    """Publish both resources only during lifespan and release each on shutdown."""
    pipeline = _FakePipeline()
    store = _FakeJobStore()
    pipeline_calls = 0
    store_calls = 0

    async def pipeline_factory() -> _FakePipeline:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return pipeline

    async def job_store_factory() -> _FakeJobStore:
        nonlocal store_calls
        store_calls += 1
        return store

    application = create_app(
        pipeline_factory=pipeline_factory,
        job_store_factory=job_store_factory,
    )
    assert not hasattr(application.state, "extraction_pipeline")
    assert not hasattr(application.state, "extraction_job_store")

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert application.state.extraction_job_store is store
        assert pipeline_calls == 1
        assert store_calls == 1

    assert not hasattr(application.state, "extraction_pipeline")
    assert not hasattr(application.state, "extraction_job_store")
    assert pipeline.close_calls == 1
    assert store.close_calls == 1

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert application.state.extraction_job_store is store
        assert pipeline_calls == 2
        assert store_calls == 2
    assert pipeline.close_calls == 2
    assert store.close_calls == 2


async def test_pipeline_factory_failure_closes_the_created_store() -> None:
    """Close the job store and publish no state when pipeline construction fails."""
    store = _FakeJobStore()

    async def failing_factory() -> _FakePipeline:
        raise RuntimeError("provider construction failed")

    async def job_store_factory() -> _FakeJobStore:
        return store

    application = create_app(
        pipeline_factory=failing_factory,
        job_store_factory=job_store_factory,
    )
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="provider construction failed"):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")
    assert not hasattr(application.state, "extraction_job_store")
    assert store.close_calls == 1


async def test_production_lifespan_delegates_to_both_public_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use process-independent production factories for lifespan resource ownership."""
    pipeline = _FakePipeline()
    store = _FakeJobStore()
    pipeline_calls = 0
    store_calls = 0

    async def pipeline_factory() -> _FakePipeline:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return pipeline

    async def job_store_factory() -> _FakeJobStore:
        nonlocal store_calls
        store_calls += 1
        return store

    monkeypatch.setattr(
        main_module,
        "create_production_extraction_pipeline",
        pipeline_factory,
    )
    monkeypatch.setattr(
        main_module,
        "create_production_extraction_job_store",
        job_store_factory,
    )
    application = create_app()

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert application.state.extraction_job_store is store
        assert pipeline_calls == 1
        assert store_calls == 1

    assert not hasattr(application.state, "extraction_pipeline")
    assert not hasattr(application.state, "extraction_job_store")
    assert pipeline.close_calls == 1
    assert store.close_calls == 1


async def test_lifespan_closes_pipeline_before_its_job_store() -> None:
    """Close pipeline resources before the store that records their work."""
    close_events: list[str] = []
    pipeline = _CloseRecordingPipeline(close_events)
    store = _FakeJobStore(close_events)
    application = create_app(
        pipeline_factory=lambda: _completed(pipeline),
        job_store_factory=lambda: _completed(store),
    )

    async with application.router.lifespan_context(application):
        pass

    assert close_events == ["pipeline", "store"]


async def _completed[ValueType](value: ValueType) -> ValueType:
    """Return one preconstructed asynchronous factory value.

    Args:
        value: Value returned by the asynchronous factory.

    Returns:
        The same value.
    """
    return value
