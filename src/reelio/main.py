"""Application composition root and entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from reelio.config import Environment, app_settings
from reelio.extraction.exceptions import (
    ExtractionError,
    extraction_error_handler,
    unhandled_error_handler,
)
from reelio.extraction.router import router as extraction_router
from reelio.extraction.services.enrichment.config import (  # noqa: F401
    tmdb_settings as _tmdb_settings,
)
from reelio.extraction.services.entities.config import (  # noqa: F401
    llm_settings as _llm_settings,
)
from reelio.extraction.services.transcription.config import (  # noqa: F401
    transcription_settings as _transcription_settings,
)
from reelio.logging import configure_logging
from reelio.ops import router as ops_router

_DOCS_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown resources.

    Args:
        application: FastAPI application receiving the lifespan context.

    Yields:
        None: While the application is running.
    """
    yield


def create_app() -> FastAPI:
    """Build and configure the Reelio FastAPI application.

    Returns:
        FastAPI: The composed application instance.
    """
    configure_logging(app_settings.log_level)
    application = FastAPI(
        title="Reelio API",
        version="0.1.0",
        default_response_class=JSONResponse,
        openapi_url=(
            "/openapi.json" if app_settings.environment in _DOCS_ENVIRONMENTS else None
        ),
        lifespan=lifespan,
    )
    application.include_router(ops_router)
    application.include_router(extraction_router)
    application.add_exception_handler(ExtractionError, extraction_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)
    return application


app = create_app()
