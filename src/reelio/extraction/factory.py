"""Compose and own production Extraction Pipelines outside HTTP lifecycle state."""

from contextlib import AsyncExitStack

from reelio.cache import AsyncCache, CacheConfig, create_cache
from reelio.config import app_settings
from reelio.extraction.coordination import CoordinationConfig
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.service import ExtractionPipeline, ExtractionPipelineProtocol
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.catalog.spotify import SpotifyCatalog, create_spotify_catalog
from reelio.extraction.services.enrichment.config import OpenLibraryConfig, TMDBConfig
from reelio.extraction.services.enrichment.open_library import (
    create_open_library_book_resolver,
)
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
from reelio.extraction.services.enrichment.spotify import SpotifyMusicResolver
from reelio.extraction.services.enrichment.tmdb import create_tmdb_screen_work_resolver
from reelio.extraction.services.interpretation.config import (
    InterpretationConfig,
    LLMProviderSelectionConfig,
)
from reelio.extraction.services.interpretation.factory import (
    create_mention_interpretation_provider,
)
from reelio.extraction.services.interpretation.service import (
    MentionInterpretationService,
)
from reelio.extraction.services.textify import (
    TextifyConfig,
    create_textify_source_transcript_provider,
)
from reelio.extraction.types import PipelineResult, TranscriptPipelineResult


class _OwnedExtractionPipeline(ExtractionPipelineProtocol):
    """Own one core pipeline and its outer Spotify catalog and shared cache."""

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        outer_resources: AsyncExitStack,
    ) -> None:
        """Store the complete pipeline and resources that outlive its composition."""
        self._pipeline = pipeline
        self._outer_resources = outer_resources

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Delegate Source extraction to the owned core pipeline."""
        return await self._pipeline.run(url, market)

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        """Delegate direct Transcript Submission extraction to the core pipeline."""
        return await self._pipeline.run_transcript(transcript_text, market)

    async def aclose(self) -> None:
        """Close core resources before the Spotify catalog and shared cache."""
        try:
            await self._pipeline.aclose()
        finally:
            await self._outer_resources.aclose()


async def _create_core_pipeline(
    default_market: SpotifyMarket,
    spotify_catalog: SpotifyCatalog,
    cache: AsyncCache,
) -> ExtractionPipeline:
    """Compose the core pipeline and transfer resource ownership only on success."""
    async with AsyncExitStack() as core_resources:
        interpretation_settings = InterpretationConfig()
        llm_provider_selection = LLMProviderSelectionConfig()  # type: ignore[call-arg]
        llm_provider = create_mention_interpretation_provider(llm_provider_selection)
        core_resources.push_async_callback(llm_provider.aclose)

        textify_settings = TextifyConfig()
        coordination_settings = CoordinationConfig()  # type: ignore[call-arg]
        source_transcript_provider = await create_textify_source_transcript_provider(
            textify_settings,
            coordination_settings,
            cache,
        )
        core_resources.push_async_callback(source_transcript_provider.aclose)

        interpretation_service = MentionInterpretationService(
            provider=llm_provider,
            settings=interpretation_settings,
            cache=cache,
        )
        tmdb_settings = TMDBConfig()  # type: ignore[call-arg]
        screen_work_resolver = create_tmdb_screen_work_resolver(tmdb_settings, cache)
        core_resources.push_async_callback(screen_work_resolver.aclose)

        open_library_settings = OpenLibraryConfig()  # type: ignore[call-arg]
        book_resolver = create_open_library_book_resolver(open_library_settings, cache)
        core_resources.push_async_callback(book_resolver.aclose)

        music_resolver = SpotifyMusicResolver(spotify_catalog)
        result_aggregator = ExtractionResultAggregator(
            screen_work_resolver,
            music_resolver,
            book_resolver,
        )
        pipeline = ExtractionPipeline(
            source_transcript_provider=source_transcript_provider,
            interpretation_service=interpretation_service,
            result_aggregator=result_aggregator,
            default_market=default_market,
        )
        core_resources.pop_all()
        return pipeline


async def create_production_extraction_pipeline() -> ExtractionPipelineProtocol:
    """Create one complete production Extraction Pipeline and its shared resources.

    Returns:
        ExtractionPipelineProtocol: One complete, caller-owned Extraction Pipeline.

    Raises:
        Exception: Propagates invalid configuration and adapter construction failures.
    """
    async with AsyncExitStack() as outer_resources:
        cache_settings = CacheConfig(environment=app_settings.environment)
        cache = create_cache(cache_settings)
        outer_resources.push_async_callback(cache.aclose)

        spotify_settings = SpotifyConfig()  # type: ignore[call-arg]
        spotify_catalog = await outer_resources.enter_async_context(
            create_spotify_catalog(spotify_settings, cache)
        )
        pipeline = await _create_core_pipeline(
            spotify_settings.default_market,
            spotify_catalog,
            cache,
        )
        return _OwnedExtractionPipeline(pipeline, outer_resources.pop_all())
