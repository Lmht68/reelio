"""Application composition root and entry point."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from reelio.config import Environment, app_settings
from reelio.extraction.exceptions import (
    ExtractionError,
    extraction_error_handler,
    unhandled_error_handler,
)
from reelio.extraction.factory import create_production_extraction_pipeline
from reelio.extraction.router import internal_router
from reelio.extraction.router import router as extraction_router
from reelio.extraction.service import ExtractionPipelineProtocol
from reelio.logging import configure_logging
from reelio.ops import router as ops_router

_DOCS_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}
_INTERNAL_API_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}
_PipelineFactory = Callable[[], Awaitable[ExtractionPipelineProtocol]]


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
    async with _managed_lifespan(application, create_production_extraction_pipeline):
        yield


def _lifespan_for(
    pipeline_factory: _PipelineFactory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def configured_lifespan(application: FastAPI) -> AsyncGenerator[None]:
        async with _managed_lifespan(application, pipeline_factory):
            yield

    return configured_lifespan


def create_app(
    pipeline_factory: _PipelineFactory | None = None,
) -> FastAPI:
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
        lifespan=lifespan if pipeline_factory is None else _lifespan_for(pipeline_factory),
    )
    application.include_router(ops_router)
    application.include_router(extraction_router)
    if app_settings.environment in _INTERNAL_API_ENVIRONMENTS:
        application.include_router(internal_router)
    application.add_exception_handler(ExtractionError, extraction_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)

    return application


app = create_app()
