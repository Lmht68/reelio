"""Application composition root and entry point."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from reelio.config import Environment, app_settings
from reelio.extraction.anonymous_owner import AnonymousOwnerConfig, AnonymousOwnerManager
from reelio.extraction.exceptions import (
    ExtractionError,
    extraction_error_handler,
    unhandled_error_handler,
)
from reelio.extraction.factory import create_production_extraction_pipeline
from reelio.extraction.job_router import router as extraction_job_router
from reelio.extraction.job_store import create_production_extraction_job_store
from reelio.extraction.jobs import ExtractionJobStore
from reelio.extraction.router import internal_router
from reelio.extraction.router import router as extraction_router
from reelio.extraction.service import ExtractionPipelineProtocol
from reelio.logging import configure_logging
from reelio.ops import router as ops_router

_DOCS_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}
_INTERNAL_API_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}
_PipelineFactory = Callable[[], Awaitable[ExtractionPipelineProtocol]]
_JobStoreFactory = Callable[[], Awaitable[ExtractionJobStore]]


@asynccontextmanager
async def _managed_lifespan(
    application: FastAPI,
    pipeline_factory: _PipelineFactory,
    job_store_factory: _JobStoreFactory,
) -> AsyncGenerator[None]:
    async with AsyncExitStack() as exit_stack:
        job_store = await job_store_factory()
        exit_stack.push_async_callback(job_store.aclose)
        pipeline = await pipeline_factory()
        exit_stack.push_async_callback(pipeline.aclose)
        application.state.extraction_job_store = job_store
        application.state.extraction_pipeline = pipeline
        try:
            yield
        finally:
            if hasattr(application.state, "extraction_pipeline"):
                del application.state.extraction_pipeline
            if hasattr(application.state, "extraction_job_store"):
                del application.state.extraction_job_store


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown resources.

    Args:
        application: FastAPI application receiving the lifespan context.

    Yields:
        None: While the application is running.
    """
    async with _managed_lifespan(
        application,
        create_production_extraction_pipeline,
        create_production_extraction_job_store,
    ):
        yield


def _lifespan_for(
    pipeline_factory: _PipelineFactory,
    job_store_factory: _JobStoreFactory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def configured_lifespan(application: FastAPI) -> AsyncGenerator[None]:
        async with _managed_lifespan(application, pipeline_factory, job_store_factory):
            yield

    return configured_lifespan


def create_app(
    pipeline_factory: _PipelineFactory | None = None,
    job_store_factory: _JobStoreFactory | None = None,
    anonymous_owner_manager: AnonymousOwnerManager | None = None,
) -> FastAPI:
    """Build and configure the Reelio FastAPI application.

    Args:
        pipeline_factory: Optional asynchronous factory used to compose the
            lifespan-owned pipeline. The production factory is used when omitted.
        job_store_factory: Optional asynchronous factory used to compose the
            lifespan-owned durable Source Extraction job store. The production factory
            is used when omitted.
        anonymous_owner_manager: Optional owner-cookie manager used by extraction-job
            routes. The environment-configured production manager is used when omitted.

    Returns:
        FastAPI: The composed application instance.
    """
    configured_lifespan = (
        lifespan
        if pipeline_factory is None and job_store_factory is None
        else _lifespan_for(
            pipeline_factory or create_production_extraction_pipeline,
            job_store_factory or create_production_extraction_job_store,
        )
    )
    owner_manager = anonymous_owner_manager or AnonymousOwnerManager(
        AnonymousOwnerConfig()  # type: ignore[call-arg]  # BaseSettings resolves the required secret.
    )
    configure_logging(app_settings.log_level)
    application = FastAPI(
        title="Reelio API",
        version="0.1.0",
        default_response_class=JSONResponse,
        openapi_url=("/openapi.json" if app_settings.environment in _DOCS_ENVIRONMENTS else None),
        lifespan=configured_lifespan,
    )
    application.state.anonymous_owner_manager = owner_manager
    application.include_router(ops_router)
    application.include_router(extraction_router)
    application.include_router(extraction_job_router)
    if app_settings.environment in _INTERNAL_API_ENVIRONMENTS:
        application.include_router(internal_router)
    application.add_exception_handler(ExtractionError, extraction_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)

    return application


app = create_app()
