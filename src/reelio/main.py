"""Application composition root and entry point."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from reelio.config import Environment, app_settings
from reelio.extraction.exceptions import (
    ExtractionError,
    extraction_error_handler,
    unhandled_error_handler,
)
from reelio.extraction.router import router as extraction_router
from reelio.extraction.service import ExtractionPipeline, ExtractionPipelineProtocol
from reelio.extraction.services.enrichment.config import tmdb_settings as _tmdb_settings
from reelio.extraction.services.enrichment.tmdb import create_tmdb_movie_resolver
from reelio.extraction.services.interpretation.config import (
    InterpretationConfig,
    LLMProviderSelectionConfig,
)
from reelio.extraction.services.interpretation.factory import (
    create_screen_work_mention_provider,
)
from reelio.extraction.services.interpretation.service import (
    ScreenWorkMentionInterpretationService,
)
from reelio.extraction.services.transcription.acquisition import (
    YouTubeCaptionProvider,
    YtDlpAudioDownloader,
    load_whisper_transcriber,
)
from reelio.extraction.services.transcription.config import (
    transcription_settings as _transcription_settings,
)
from reelio.extraction.services.transcription.inspection import YtDlpMetadataExtractor
from reelio.extraction.services.transcription.service import (
    SourceMetadataService,
    TranscriptionService,
)
from reelio.logging import configure_logging
from reelio.ops import router as ops_router

_DOCS_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}
_PipelineFactory = Callable[[], Awaitable[ExtractionPipelineProtocol]]


async def _create_production_pipeline() -> ExtractionPipelineProtocol:
    """Load production dependencies and compose one extraction pipeline."""
    interpretation_settings = InterpretationConfig()
    llm_provider_selection = LLMProviderSelectionConfig()  # type: ignore[call-arg]
    async with AsyncExitStack() as cleanup:
        llm_provider = create_screen_work_mention_provider(llm_provider_selection)
        cleanup.push_async_callback(llm_provider.aclose)
        transcriber = await asyncio.to_thread(
            load_whisper_transcriber,
            _transcription_settings,
        )
        source_metadata_service = SourceMetadataService(
            extractor=YtDlpMetadataExtractor(
                max_duration_seconds=_transcription_settings.max_video_duration_seconds,
                temp_media_dir=_transcription_settings.temp_media_dir,
            ),
            settings=_transcription_settings,
        )
        transcription_service = TranscriptionService(
            provider=YouTubeCaptionProvider(),
            audio_downloader=YtDlpAudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=_transcription_settings.temp_media_dir,
            semaphore=asyncio.Semaphore(_transcription_settings.whisper_max_concurrent),
        )
        interpretation_service = ScreenWorkMentionInterpretationService(
            provider=llm_provider,
            settings=interpretation_settings,
        )
        movie_resolver = create_tmdb_movie_resolver(_tmdb_settings)
        pipeline = ExtractionPipeline(
            source_metadata_service=source_metadata_service,
            transcription_service=transcription_service,
            interpretation_service=interpretation_service,
            movie_resolver=movie_resolver,
        )
        cleanup.pop_all()
        return pipeline


@asynccontextmanager
async def _managed_lifespan(
    application: FastAPI,
    pipeline_factory: _PipelineFactory,
) -> AsyncGenerator[None]:
    pipeline = await pipeline_factory()
    application.state.extraction_pipeline = pipeline
    try:
        yield
    finally:
        try:
            await pipeline.aclose()
        finally:
            if hasattr(application.state, "extraction_pipeline"):
                del application.state.extraction_pipeline


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown resources.

    Args:
        application: FastAPI application receiving the lifespan context.

    Yields:
        None: While the application is running.
    """
    async with _managed_lifespan(application, _create_production_pipeline):
        yield


def _lifespan_for(
    pipeline_factory: _PipelineFactory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def configured_lifespan(application: FastAPI) -> AsyncGenerator[None]:
        async with _managed_lifespan(application, pipeline_factory):
            yield

    return configured_lifespan


def create_app(pipeline_factory: _PipelineFactory | None = None) -> FastAPI:
    """Build and configure the Reelio FastAPI application.

    Args:
        pipeline_factory: Optional asynchronous factory used to compose the
            lifespan-owned pipeline. The production factory is used when omitted.

    Returns:
        FastAPI: The composed application instance.
    """
    configure_logging(app_settings.log_level)
    application = FastAPI(
        title="Reelio API",
        version="0.1.0",
        default_response_class=JSONResponse,
        openapi_url=("/openapi.json" if app_settings.environment in _DOCS_ENVIRONMENTS else None),
        lifespan=(lifespan if pipeline_factory is None else _lifespan_for(pipeline_factory)),
    )
    application.include_router(ops_router)
    application.include_router(extraction_router)
    application.add_exception_handler(ExtractionError, extraction_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)
    return application


app = create_app()
