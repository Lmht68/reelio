"""Application composition and documentation lifecycle tests."""

import pytest
from httpx import ASGITransport, AsyncClient

import reelio.main as main_module
from reelio.config import Environment, app_settings
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


async def test_injected_pipeline_factory_owns_one_pipeline_per_lifespan() -> None:
    """Store one factory result during lifespan and release it on shutdown."""
    pipeline = _FakePipeline()
    calls = 0

    async def factory() -> _FakePipeline:
        nonlocal calls
        calls += 1
        return pipeline

    application = create_app(pipeline_factory=factory)
    assert not hasattr(application.state, "extraction_pipeline")

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 1

    assert not hasattr(application.state, "extraction_pipeline")
    assert pipeline.close_calls == 1

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 2
    assert pipeline.close_calls == 2


async def test_pipeline_factory_failure_aborts_lifespan() -> None:
    """Propagate provider construction failures during startup."""

    async def failing_factory() -> _FakePipeline:
        raise RuntimeError("provider construction failed")

    application = create_app(pipeline_factory=failing_factory)
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="provider construction failed"):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_delegates_to_the_public_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the process-independent production factory for application lifecycle ownership."""
    pipeline = _FakePipeline()
    calls = 0

    async def factory() -> _FakePipeline:
        nonlocal calls
        calls += 1
        return pipeline

    monkeypatch.setattr(main_module, "create_production_extraction_pipeline", factory)
    application = create_app()

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 1

    assert not hasattr(application.state, "extraction_pipeline")
    assert pipeline.close_calls == 1
