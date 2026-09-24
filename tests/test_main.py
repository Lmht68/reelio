"""Application composition and documentation lifecycle tests."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import NoReturn, cast

import ctranslate2
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from redis.asyncio import Redis

import reelio.extraction.services.transcription.acquisition as transcription_service
import reelio.main as main_module
from reelio.cache import CacheConfig, DisabledCache
from reelio.config import Environment, app_settings
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.interpretation.config import (
    LLMProvider,
    LLMProviderSelectionConfig,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
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


class _FakeProvider:
    def __init__(self) -> None:
        self.provider_name = LLMProvider.OPENAI
        self.model_name = "gpt-5-nano"
        self.close_calls = 0

    async def complete(self, messages: object) -> str:
        raise AssertionError(f"unexpected provider completion: {messages}")

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeScreenWorkResolver:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeBookResolver:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeCache:
    """Record one shared-cache lifecycle owned outside the extraction pipeline."""

    def __init__(self, events: list[str] | None = None) -> None:
        """Initialize the cache close counter and optional lifecycle trace."""
        self.close_calls = 0
        self._events = events

    async def aclose(self) -> None:
        """Record cache closure."""
        self.close_calls += 1
        if self._events is not None:
            self._events.append("cache")


def _transcription_settings(device: str) -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(
        _env_file=None,
        whisper_device=device,
        whisper_model="test-model",
        whisper_compute_type="test-type",
    )


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


def test_cuda_preflight_fails_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unavailable explicit CUDA before constructing WhisperModel."""
    constructor_calls: list[tuple[str, str, str]] = []

    def fake_model(
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> object:
        constructor_calls.append((model_size_or_path, device, compute_type))
        return object()

    monkeypatch.setattr(
        ctranslate2,
        "get_cuda_device_count",
        lambda: 0,
    )
    monkeypatch.setattr(transcription_service, "WhisperModel", fake_model)

    with pytest.raises(
        RuntimeError,
        match=r"^REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available\.$",
    ):
        transcription_service.load_whisper_transcriber(_transcription_settings("cuda"))

    assert constructor_calls == []


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_non_cuda_devices_skip_cuda_preflight(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Pass CPU and auto settings directly to the model constructor."""
    constructor_calls: list[tuple[str, str, str]] = []

    def fake_model(
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> object:
        constructor_calls.append((model_size_or_path, device, compute_type))
        return object()

    monkeypatch.setattr(
        ctranslate2,
        "get_cuda_device_count",
        lambda: pytest.fail("CPU and auto must skip CUDA preflight"),
    )
    monkeypatch.setattr(transcription_service, "WhisperModel", fake_model)

    transcription_service.load_whisper_transcriber(_transcription_settings(device))

    assert constructor_calls == [("test-model", device, "test-type")]


async def test_pipeline_factory_failure_aborts_lifespan() -> None:
    """Propagate model or dependency construction failures during startup."""

    async def failing_factory() -> _FakePipeline:
        raise RuntimeError("model load failed")

    application = create_app(pipeline_factory=failing_factory)
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="model load failed"):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_closes_one_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct one selected provider and close it when production stops."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    provider = _FakeProvider()
    monkeypatch.setenv("REELIO_OPEN_LIBRARY_CONTACT_EMAIL", "test@example.invalid")
    resolver = _FakeScreenWorkResolver()
    book_resolver = _FakeBookResolver()
    book_resolver_factory_calls = 0
    provider_factory_calls = 0
    resolver_factory_calls = 0

    def create_provider(selection: LLMProviderSelectionConfig) -> _FakeProvider:
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        assert selection.llm_provider is LLMProvider.OPENAI
        return provider

    def create_resolver(
        settings: object,
        configured_cache: object,
    ) -> _FakeScreenWorkResolver:
        nonlocal resolver_factory_calls
        del settings, configured_cache
        resolver_factory_calls += 1
        return resolver

    def create_book_resolver(settings: object, cache: object) -> _FakeBookResolver:
        nonlocal book_resolver_factory_calls
        book_resolver_factory_calls += 1
        return book_resolver

    monkeypatch.setattr(main_module, "create_mention_interpretation_provider", create_provider)
    monkeypatch.setattr(main_module, "load_whisper_transcriber", lambda settings: object())
    monkeypatch.setattr(main_module, "create_tmdb_screen_work_resolver", create_resolver)
    monkeypatch.setattr(
        main_module,
        "create_open_library_book_resolver",
        create_book_resolver,
    )
    application = create_app()

    async with application.router.lifespan_context(application):
        assert provider_factory_calls == 1
        assert resolver_factory_calls == 1
        assert book_resolver_factory_calls == 1

    assert provider.close_calls == 1
    assert resolver.close_calls == 1
    assert book_resolver.close_calls == 1


async def test_production_lifespan_closes_provider_after_partial_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close a selected provider when a later production dependency fails."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    provider = _FakeProvider()

    def create_provider(selection: LLMProviderSelectionConfig) -> _FakeProvider:
        assert selection.llm_provider is LLMProvider.OPENAI
        return provider

    def fail_whisper_load(settings: TranscriptionConfig) -> NoReturn:
        raise RuntimeError("Whisper load failed")

    monkeypatch.setattr(
        main_module,
        "create_mention_interpretation_provider",
        create_provider,
    )
    monkeypatch.setattr(main_module, "load_whisper_transcriber", fail_whisper_load)
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="Whisper load failed"):
        await context.__aenter__()
    assert provider.close_calls == 1
    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_closes_resolver_after_aggregation_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close every acquired resource when aggregation setup aborts startup."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REELIO_OPEN_LIBRARY_CONTACT_EMAIL", "test@example.invalid")
    provider = _FakeProvider()
    resolver = _FakeScreenWorkResolver()
    book_resolver = _FakeBookResolver()
    resolver_factory_calls = 0

    def create_provider(selection: LLMProviderSelectionConfig) -> _FakeProvider:
        assert selection.llm_provider is LLMProvider.OPENAI
        return provider

    def create_resolver(
        settings: object,
        configured_cache: object,
    ) -> _FakeScreenWorkResolver:
        nonlocal resolver_factory_calls
        del settings, configured_cache
        resolver_factory_calls += 1
        return resolver

    def fail_aggregation_setup(
        screen_work_resolver: object,
        music_resolver: object,
        configured_book_resolver: object,
    ) -> NoReturn:
        assert screen_work_resolver is resolver
        assert music_resolver is not None
        assert configured_book_resolver is book_resolver
        raise RuntimeError("aggregation setup failed")

    monkeypatch.setattr(main_module, "create_mention_interpretation_provider", create_provider)
    monkeypatch.setattr(main_module, "load_whisper_transcriber", lambda settings: object())
    monkeypatch.setattr(main_module, "create_tmdb_screen_work_resolver", create_resolver)
    monkeypatch.setattr(
        main_module,
        "create_open_library_book_resolver",
        lambda settings, cache: book_resolver,
    )
    monkeypatch.setattr(
        main_module,
        "ExtractionResultAggregator",
        fail_aggregation_setup,
    )
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="aggregation setup failed"):
        await context.__aenter__()

    assert resolver_factory_calls == 1
    assert provider.close_calls == 1
    assert resolver.close_calls == 1
    assert book_resolver.close_calls == 1
    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_rejects_unsupported_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail startup before making an extraction pipeline for an unknown provider."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "unsupported")
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(ValidationError):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")


class _FakeSpotifyCatalog:
    """Record lifecycle ownership for a composed Spotify catalog boundary."""

    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        """Record one catalog shutdown call."""
        self.close_calls += 1


async def test_injected_lifespan_owns_one_spotify_catalog() -> None:
    """Close the injected catalog context after the injected pipeline exits."""
    catalog = _FakeSpotifyCatalog()
    pipeline = _FakePipeline()

    @asynccontextmanager
    async def catalog_factory(
        _settings: SpotifyConfig,
    ) -> AsyncGenerator[_FakeSpotifyCatalog]:
        try:
            yield catalog
        finally:
            await catalog.aclose()

    async def pipeline_factory() -> _FakePipeline:
        return pipeline

    application = create_app(
        pipeline_factory=pipeline_factory,
        spotify_catalog_factory=catalog_factory,
    )

    async with application.router.lifespan_context(application):
        assert application.state.spotify_catalog is catalog

    assert catalog.close_calls == 1
    assert pipeline.close_calls == 1
    assert not hasattr(application.state, "spotify_catalog")


async def test_disabled_production_lifespan_never_constructs_a_redis_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled production composition uses no Redis client factory."""
    monkeypatch.setenv("REELIO_CACHE_ENABLED", "false")
    monkeypatch.delenv("REELIO_CACHE_REDIS_URL", raising=False)
    monkeypatch.delenv("REELIO_CACHE_KEY_SECRET", raising=False)
    redis_factory_calls = 0
    pipeline = _FakePipeline()

    class _FakeSpotifySettings:
        default_market = SpotifyMarket("US")

    @asynccontextmanager
    async def spotify_catalog_factory(
        settings: object,
        cache: object,
    ) -> AsyncGenerator[_FakeSpotifyCatalog]:
        del settings
        assert isinstance(cache, DisabledCache)
        yield _FakeSpotifyCatalog()

    def fail_redis_factory(*args: object, **kwargs: object) -> NoReturn:
        nonlocal redis_factory_calls
        del args, kwargs
        redis_factory_calls += 1
        raise AssertionError("Disabled cache must not construct a Redis client")

    async def create_pipeline(
        default_market: SpotifyMarket,
        spotify_catalog: object,
        cache: object,
    ) -> _FakePipeline:
        del default_market, spotify_catalog
        assert isinstance(cache, DisabledCache)
        return pipeline

    monkeypatch.setattr(main_module, "SpotifyConfig", _FakeSpotifySettings)
    monkeypatch.setattr(main_module, "create_spotify_catalog", spotify_catalog_factory)
    monkeypatch.setattr(main_module, "_create_production_pipeline", create_pipeline)
    monkeypatch.setattr(Redis, "from_url", fail_redis_factory)
    application = create_app()

    async with application.router.lifespan_context(application):
        assert pipeline.close_calls == 0

    assert redis_factory_calls == 0
    assert pipeline.close_calls == 1


async def test_enabled_production_lifespan_shares_cache_and_closes_it_after_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass one enabled cache into TMDB, Spotify, and Open Library before teardown."""
    monkeypatch.setenv("REELIO_CACHE_ENABLED", "true")
    monkeypatch.setenv("REELIO_CACHE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("REELIO_CACHE_KEY_SECRET", "cache-key")
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REELIO_OPEN_LIBRARY_CONTACT_EMAIL", "test@example.invalid")
    events: list[str] = []
    cache = _FakeCache(events)
    provider = _FakeProvider()
    screen_work_resolver = _FakeScreenWorkResolver()

    class _RecordingBookResolver(_FakeBookResolver):
        async def aclose(self) -> None:
            """Record resolver closure before the outer cache closes."""
            await super().aclose()
            events.append("book")

    class _FakeSpotifySettings:
        default_market = SpotifyMarket("US")

    book_resolver = _RecordingBookResolver()
    received_book_caches: list[object] = []
    received_spotify_caches: list[object] = []
    received_tmdb_caches: list[object] = []

    @asynccontextmanager
    async def spotify_catalog_factory(
        settings: object,
        configured_cache: object,
    ) -> AsyncGenerator[_FakeSpotifyCatalog]:
        del settings
        received_spotify_caches.append(configured_cache)
        yield _FakeSpotifyCatalog()

    def create_resolver(
        settings: object,
        configured_cache: object,
    ) -> _FakeScreenWorkResolver:
        del settings
        received_tmdb_caches.append(configured_cache)
        return screen_work_resolver

    def create_configured_cache(settings: CacheConfig) -> _FakeCache:
        assert settings.enabled is True
        return cache

    def create_book_resolver(settings: object, configured_cache: object) -> _FakeBookResolver:
        del settings
        received_book_caches.append(configured_cache)
        return book_resolver

    monkeypatch.setattr(main_module, "SpotifyConfig", _FakeSpotifySettings)
    monkeypatch.setattr(main_module, "create_spotify_catalog", spotify_catalog_factory)
    monkeypatch.setattr(main_module, "create_cache", create_configured_cache)
    monkeypatch.setattr(
        main_module,
        "create_mention_interpretation_provider",
        lambda selection: provider,
    )
    monkeypatch.setattr(main_module, "load_whisper_transcriber", lambda settings: object())
    monkeypatch.setattr(
        main_module,
        "create_tmdb_screen_work_resolver",
        create_resolver,
    )
    monkeypatch.setattr(
        main_module,
        "create_open_library_book_resolver",
        create_book_resolver,
    )
    application = create_app()

    async with application.router.lifespan_context(application):
        assert received_book_caches == [cache]
        assert received_tmdb_caches == [cache]
        assert received_spotify_caches == [cache]
        assert cache.close_calls == 0

    assert book_resolver.close_calls == 1
    assert cache.close_calls == 1
    assert events == ["book", "cache"]


async def test_enabled_cache_closes_once_after_partial_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close one enabled cache when pipeline composition fails before application startup."""
    monkeypatch.setenv("REELIO_CACHE_ENABLED", "true")
    monkeypatch.setenv("REELIO_CACHE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("REELIO_CACHE_KEY_SECRET", "cache-key")
    cache = _FakeCache()

    class _FakeSpotifySettings:
        default_market = SpotifyMarket("US")

    @asynccontextmanager
    async def spotify_catalog_factory(
        settings: object,
        configured_cache: object,
    ) -> AsyncGenerator[_FakeSpotifyCatalog]:
        del settings
        assert configured_cache is cache
        yield _FakeSpotifyCatalog()

    def create_configured_cache(settings: CacheConfig) -> _FakeCache:
        assert settings.enabled is True
        return cache

    async def fail_pipeline(
        default_market: SpotifyMarket,
        spotify_catalog: object,
        configured_cache: object,
    ) -> NoReturn:
        del default_market, spotify_catalog
        assert configured_cache is cache
        raise RuntimeError("pipeline composition failed")

    monkeypatch.setattr(main_module, "SpotifyConfig", _FakeSpotifySettings)
    monkeypatch.setattr(main_module, "create_spotify_catalog", spotify_catalog_factory)
    monkeypatch.setattr(main_module, "create_cache", create_configured_cache)
    monkeypatch.setattr(main_module, "_create_production_pipeline", fail_pipeline)
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="pipeline composition failed"):
        await context.__aenter__()

    assert cache.close_calls == 1
