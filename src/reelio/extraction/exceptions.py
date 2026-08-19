"""Domain exceptions and HTTP error handlers for extraction."""

import logging
from typing import ClassVar, cast

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Base class for errors raised by the extraction pipeline.

    Attributes:
        code: Stable machine-readable error code for the API response.
        status_code: HTTP status code associated with the error.
    """

    code: ClassVar[str]
    status_code: ClassVar[int]


class InvalidSourceError(ExtractionError):
    """Indicate that the submitted source URL is invalid."""

    code = "invalid_source"
    status_code = 400


class UnsupportedPlatformError(ExtractionError):
    """Indicate that the source platform is not supported."""

    code = "unsupported_platform"
    status_code = 400


class SourceUnavailableError(ExtractionError):
    """Indicate that the source cannot be accessed or found."""

    code = "source_unavailable"
    status_code = 404


class DurationLimitExceededError(ExtractionError):
    """Indicate that the source exceeds the configured duration limit."""

    code = "duration_limit_exceeded"
    status_code = 413


class MetadataProviderError(ExtractionError):
    """Indicate that source metadata could not be retrieved or normalized."""

    code = "metadata_provider_failed"
    status_code = 502


class TranscriptionError(ExtractionError):
    """Indicate that no transcript could be acquired."""

    code = "transcription_failed"
    status_code = 502


class EntityExtractionError(ExtractionError):
    """Indicate that mention interpretation failed."""

    code = "entity_extraction_failed"
    status_code = 502


class EnrichmentError(ExtractionError):
    """Indicate that provider enrichment failed."""

    code = "enrichment_failed"
    status_code = 502


class PipelineTimeoutError(ExtractionError):
    """Indicate that an external pipeline operation timed out."""

    code = "pipeline_timeout"
    status_code = 504


async def extraction_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert an extraction domain error into its public error response.

    Args:
        request: Request that triggered the exception.
        exc: Exception registered as an ``ExtractionError`` handler.

    Returns:
        JSONResponse: Error body and status defined by the domain exception.
    """
    error = cast(ExtractionError, exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": str(error)}},
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Hide unexpected exception details behind a generic error response.

    Args:
        request: Request that triggered the exception.
        exc: Unexpected exception raised while handling the request.

    Returns:
        JSONResponse: Generic internal-error body with HTTP status 500.
    """
    logger.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            }
        },
    )
