"""Public production extraction pipeline factory tests."""

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import NoReturn

import pytest

import reelio.extraction.factory as factory_module
from reelio.extraction.factory import create_production_extraction_pipeline
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.interpretation.config import LLMProvider
from reelio.extraction.services.interpretation.types import LLMMessage
from reelio.extraction.types import (
    BookMentions,
    BookResults,
    MovieResult,
    Platform,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
    TranscriptMethod,
)


class _FakeCache:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def get_or_load[ValueT](
        self,
        entry: object,
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        del entry
        return await loader()

    async def aclose(self) -> None:
        self.events.append("cache")


class _FakeProvider:
    provider_name = LLMProvider.OPENAI
    model_name = "factory-test-model"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        del messages
        return '{"movies": [{"title": "Dune: Part One", "year": 2021}], "tv_series": [], "tracks": [], "music_releases": [], "books": []}'

    async def aclose(self) -> None:
        self.events.append("llm")


class _FakeSourceTranscriptProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[str] = []

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        self.calls.append(submitted_url)
        return (
            Source(
                platform=Platform.YOUTUBE,
                video_id="dQw4w9WgXcQ",
                url=submitted_url,
                title="Factory test source",
                description="Factory test description.",
                channel="Factory test channel",
                duration_seconds=42,
            ),
            Transcript(
                text="Dune: Part One (2021) was excellent.",
                language="en",
                method=TranscriptMethod.YOUTUBE_CAPTIONS,
            ),
        )

    async def aclose(self) -> None:
        self.events.append("textify")


class _FakeScreenWorkResolver:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def resolve(self, mentions: ScreenWorkMentions) -> ScreenWorkResults:
        return ScreenWorkResults(
            movies=[
                MovieResult(ResultStatus.UNRESOLVED, movie_mention, None)
                for movie_mention in mentions.movies
            ],
            tv_series=[],
        )

    async def aclose(self) -> None:
        self.events.append("tmdb")


class _FakeBookResolver:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def resolve(self, mentions: BookMentions) -> BookResults:
        assert mentions.books == []
        return BookResults(books=[])

    async def aclose(self) -> None:
        self.events.append("open_library")


class _FakeSpotifyConfig:
    default_market = SpotifyMarket("US")


class _FakeSettings:
    max_source_title_chars = 500
    max_description_chars = 2_000
    max_transcript_language_chars = 64
    max_transcript_chars = 100_000


async def test_factory_runs_source_and_transcript_extraction_with_shared_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create, use, and close one complete pipeline without FastAPI."""
    events: list[str] = []
    cache = _FakeCache(events)
    provider = _FakeProvider(events)
    source_transcript_provider = _FakeSourceTranscriptProvider(events)
    screen_work_resolver = _FakeScreenWorkResolver(events)
    book_resolver = _FakeBookResolver(events)
    received_caches: list[object] = []

    @asynccontextmanager
    async def create_catalog(
        settings: object,
        configured_cache: object,
    ) -> AsyncGenerator[object]:
        assert isinstance(settings, _FakeSpotifyConfig)
        received_caches.append(configured_cache)
        try:
            yield object()
        finally:
            events.append("spotify")

    def create_cache(settings: object) -> _FakeCache:
        assert settings is not None
        return cache

    def create_provider(settings: object) -> _FakeProvider:
        assert isinstance(settings, _FakeSettings)
        return provider

    async def create_textify_provider(
        textify_settings: object,
        coordination_settings: object,
        configured_cache: object,
    ) -> _FakeSourceTranscriptProvider:
        assert isinstance(textify_settings, _FakeSettings)
        assert isinstance(coordination_settings, _FakeSettings)
        received_caches.append(configured_cache)
        return source_transcript_provider

    def create_tmdb_resolver(
        settings: object,
        configured_cache: object,
    ) -> _FakeScreenWorkResolver:
        assert isinstance(settings, _FakeSettings)
        received_caches.append(configured_cache)
        return screen_work_resolver

    def create_book_resolver(
        settings: object,
        configured_cache: object,
    ) -> _FakeBookResolver:
        assert isinstance(settings, _FakeSettings)
        received_caches.append(configured_cache)
        return book_resolver

    monkeypatch.setattr(factory_module, "CacheConfig", lambda environment: _FakeSettings())
    monkeypatch.setattr(factory_module, "create_cache", create_cache)
    monkeypatch.setattr(factory_module, "SpotifyConfig", _FakeSpotifyConfig)
    monkeypatch.setattr(factory_module, "create_spotify_catalog", create_catalog)
    monkeypatch.setattr(factory_module, "InterpretationConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "LLMProviderSelectionConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "create_mention_interpretation_provider", create_provider)
    monkeypatch.setattr(factory_module, "TextifyConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "CoordinationConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_textify_source_transcript_provider",
        create_textify_provider,
    )
    monkeypatch.setattr(factory_module, "TMDBConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "create_tmdb_screen_work_resolver", create_tmdb_resolver)
    monkeypatch.setattr(factory_module, "OpenLibraryConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "create_open_library_book_resolver", create_book_resolver)

    pipeline = await create_production_extraction_pipeline()

    source_result = await pipeline.run("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    transcript_result = await pipeline.run_transcript(
        "Dune: Part One (2021) was excellent.",
        SpotifyMarket("JP"),
    )
    await pipeline.aclose()

    assert source_result.market == SpotifyMarket("US")
    assert source_result.source.video_id == "dQw4w9WgXcQ"
    assert transcript_result.market == SpotifyMarket("JP")
    assert source_transcript_provider.calls == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    assert received_caches == [cache, cache, cache, cache]
    assert events == ["textify", "llm", "tmdb", "open_library", "spotify", "cache"]


class _ClosingResource:
    def __init__(self, name: str, close_events: list[str]) -> None:
        self._name = name
        self._close_events = close_events

    async def aclose(self) -> None:
        self._close_events.append(self._name)


def _configure_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
    attempts: list[str],
    close_events: list[str],
) -> None:
    def create_cache_settings(environment: object) -> _FakeSettings:
        del environment
        return _FakeSettings()

    def create_cache(settings: object) -> _ClosingResource:
        del settings
        attempts.append("cache")
        return _ClosingResource("cache", close_events)

    @asynccontextmanager
    async def create_catalog(
        settings: object,
        cache: object,
    ) -> AsyncGenerator[object]:
        del settings, cache
        attempts.append("spotify")
        if failure_boundary == "spotify":
            raise RuntimeError("spotify construction failed")
        try:
            yield object()
        finally:
            close_events.append("spotify")

    def create_llm_provider(settings: object) -> _ClosingResource:
        del settings
        attempts.append("llm")
        if failure_boundary == "llm":
            raise RuntimeError("llm construction failed")
        return _ClosingResource("llm", close_events)

    async def create_textify_provider(
        textify_settings: object,
        coordination_settings: object,
        cache: object,
    ) -> _ClosingResource:
        del textify_settings, coordination_settings, cache
        attempts.append("textify")
        if failure_boundary == "textify":
            raise RuntimeError("textify construction failed")
        return _ClosingResource("textify", close_events)

    def create_tmdb_resolver(
        settings: object,
        cache: object,
    ) -> _ClosingResource:
        del settings, cache
        attempts.append("tmdb")
        if failure_boundary == "tmdb":
            raise RuntimeError("tmdb construction failed")
        return _ClosingResource("tmdb", close_events)

    def create_book_resolver(
        settings: object,
        cache: object,
    ) -> _ClosingResource:
        del settings, cache
        attempts.append("open_library")
        if failure_boundary == "open_library":
            raise RuntimeError("open_library construction failed")
        return _ClosingResource("open_library", close_events)

    def create_aggregator(
        screen_work_resolver: object,
        music_resolver: object,
        book_resolver: object,
    ) -> object:
        del screen_work_resolver, music_resolver, book_resolver
        attempts.append("aggregator")
        if failure_boundary == "aggregator":
            raise RuntimeError("aggregator construction failed")
        return object()

    def create_pipeline(
        source_transcript_provider: object,
        interpretation_service: object,
        result_aggregator: object,
        default_market: SpotifyMarket,
    ) -> NoReturn:
        del source_transcript_provider, interpretation_service, result_aggregator, default_market
        attempts.append("pipeline")
        raise RuntimeError("pipeline construction failed")

    monkeypatch.setattr(factory_module, "CacheConfig", create_cache_settings)
    monkeypatch.setattr(factory_module, "create_cache", create_cache)
    monkeypatch.setattr(factory_module, "SpotifyConfig", _FakeSpotifyConfig)
    monkeypatch.setattr(factory_module, "create_spotify_catalog", create_catalog)
    monkeypatch.setattr(factory_module, "InterpretationConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "LLMProviderSelectionConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_mention_interpretation_provider",
        create_llm_provider,
    )
    monkeypatch.setattr(factory_module, "TextifyConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "CoordinationConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_textify_source_transcript_provider",
        create_textify_provider,
    )
    monkeypatch.setattr(factory_module, "TMDBConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "create_tmdb_screen_work_resolver", create_tmdb_resolver)
    monkeypatch.setattr(factory_module, "OpenLibraryConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_open_library_book_resolver",
        create_book_resolver,
    )
    monkeypatch.setattr(factory_module, "ExtractionResultAggregator", create_aggregator)
    if failure_boundary == "pipeline":
        monkeypatch.setattr(factory_module, "ExtractionPipeline", create_pipeline)


@pytest.mark.parametrize(
    ("failure_boundary", "expected_attempts", "expected_closes"),
    [
        ("spotify", ["cache", "spotify"], ["cache"]),
        ("llm", ["cache", "spotify", "llm"], ["spotify", "cache"]),
        (
            "textify",
            ["cache", "spotify", "llm", "textify"],
            ["llm", "spotify", "cache"],
        ),
        (
            "tmdb",
            ["cache", "spotify", "llm", "textify", "tmdb"],
            ["textify", "llm", "spotify", "cache"],
        ),
        (
            "open_library",
            ["cache", "spotify", "llm", "textify", "tmdb", "open_library"],
            ["tmdb", "textify", "llm", "spotify", "cache"],
        ),
        (
            "aggregator",
            [
                "cache",
                "spotify",
                "llm",
                "textify",
                "tmdb",
                "open_library",
                "aggregator",
            ],
            ["open_library", "tmdb", "textify", "llm", "spotify", "cache"],
        ),
        (
            "pipeline",
            [
                "cache",
                "spotify",
                "llm",
                "textify",
                "tmdb",
                "open_library",
                "aggregator",
                "pipeline",
            ],
            ["open_library", "tmdb", "textify", "llm", "spotify", "cache"],
        ),
    ],
)
async def test_factory_closes_all_prior_resources_after_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
    expected_attempts: list[str],
    expected_closes: list[str],
) -> None:
    """Propagate construction failures after closing every earlier factory resource."""
    attempts: list[str] = []
    close_events: list[str] = []
    _configure_startup_failure(
        monkeypatch,
        failure_boundary,
        attempts,
        close_events,
    )

    with pytest.raises(RuntimeError, match=f"{failure_boundary} construction failed"):
        await create_production_extraction_pipeline()

    assert attempts == expected_attempts
    assert close_events == expected_closes


async def test_factory_closes_spotify_and_cache_after_core_shutdown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release outer resources even when the complete core pipeline cannot close."""
    close_events: list[str] = []

    @asynccontextmanager
    async def create_catalog(
        settings: object,
        cache: object,
    ) -> AsyncGenerator[object]:
        del settings, cache
        try:
            yield object()
        finally:
            close_events.append("spotify")

    async def create_textify_provider(
        textify_settings: object,
        coordination_settings: object,
        cache: object,
    ) -> _ClosingResource:
        del textify_settings, coordination_settings, cache
        return _ClosingResource("textify", close_events)

    class _FailingCorePipeline:
        def __init__(
            self,
            source_transcript_provider: object,
            interpretation_service: object,
            result_aggregator: object,
            default_market: SpotifyMarket,
        ) -> None:
            del (
                source_transcript_provider,
                interpretation_service,
                result_aggregator,
                default_market,
            )

        async def aclose(self) -> NoReturn:
            close_events.append("core")
            raise RuntimeError("core shutdown failed")

    monkeypatch.setattr(
        factory_module,
        "CacheConfig",
        lambda environment: _FakeSettings(),
    )
    monkeypatch.setattr(
        factory_module,
        "create_cache",
        lambda _: _ClosingResource("cache", close_events),
    )
    monkeypatch.setattr(factory_module, "SpotifyConfig", _FakeSpotifyConfig)
    monkeypatch.setattr(factory_module, "create_spotify_catalog", create_catalog)
    monkeypatch.setattr(factory_module, "InterpretationConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "LLMProviderSelectionConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_mention_interpretation_provider",
        lambda _: _ClosingResource("llm", close_events),
    )
    monkeypatch.setattr(factory_module, "TextifyConfig", _FakeSettings)
    monkeypatch.setattr(factory_module, "CoordinationConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_textify_source_transcript_provider",
        create_textify_provider,
    )
    monkeypatch.setattr(factory_module, "TMDBConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_tmdb_screen_work_resolver",
        lambda *_: _ClosingResource("tmdb", close_events),
    )
    monkeypatch.setattr(factory_module, "OpenLibraryConfig", _FakeSettings)
    monkeypatch.setattr(
        factory_module,
        "create_open_library_book_resolver",
        lambda *_: _ClosingResource("open_library", close_events),
    )
    monkeypatch.setattr(factory_module, "ExtractionPipeline", _FailingCorePipeline)

    pipeline = await create_production_extraction_pipeline()

    with pytest.raises(RuntimeError, match="core shutdown failed"):
        await pipeline.aclose()

    assert close_events == ["core", "spotify", "cache"]
