"""Application composition root."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from reelio.api.router import router as api_router
from reelio.config import Environment, app_settings
from reelio.enrichment.config import tmdb_settings as _tmdb_settings  # noqa: F401
from reelio.entities.config import llm_settings as _llm_settings  # noqa: F401
from reelio.logging import configure_logging
from reelio.transcription.config import (  # noqa: F401
    transcription_settings as _transcription_settings,
)

_DOCS_ENVIRONMENTS = {Environment.LOCAL, Environment.STAGING}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
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
        default_response_class=ORJSONResponse,
        openapi_url=(
            "/openapi.json" if app_settings.environment in _DOCS_ENVIRONMENTS else None
        ),
        lifespan=lifespan,
    )
    application.include_router(api_router)
    return application
