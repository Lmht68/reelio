"""Domain exceptions and HTTP error handlers for extraction."""

import logging
from typing import ClassVar, cast

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

INTERNAL_ERROR_CODE = "internal_error"
INTERNAL_ERROR_MESSAGE = "An unexpected error occurred."


class ExtractionError(Exception):
    """Base class for errors raised by the extraction pipeline.

    Attributes:
        code: Stable machine-readable error code for the API response.
        status_code: HTTP status code associated with the error.
        retry_after_seconds: Optional client retry delay included in ``Retry-After``.
    """

    code: ClassVar[str]
    status_code: ClassVar[int]
    retry_after_seconds: ClassVar[int | None] = None
    retryable: ClassVar[bool] = False


def is_retryable_provider_status(status_code: int) -> bool:
    """Return whether one provider response status denotes a transient failure.

    Args:
        status_code: HTTP status returned by an external provider.

    Returns:
        True for rate limiting and server failures, otherwise False.
    """
    return status_code == 429 or 500 <= status_code <= 599


class InvalidSourceError(ExtractionError):
    """Indicate that the submitted source URL is invalid."""

    code = "invalid_source"
    status_code = 400
    retryable: ClassVar[bool] = False


class UnsupportedPlatformError(ExtractionError):
    """Indicate that the source platform is not supported."""

    code = "unsupported_platform"
    status_code = 400
    retryable: ClassVar[bool] = False


class SourceUnavailableError(ExtractionError):
    """Indicate that the source cannot be accessed or found."""

    code = "source_unavailable"
    status_code = 404
    retryable: ClassVar[bool] = False


class DurationLimitExceededError(ExtractionError):
    """Indicate that the source exceeds the configured duration limit."""

    code = "duration_limit_exceeded"
    status_code = 413
    retryable: ClassVar[bool] = False


class InterpretationInputTooLargeError(ExtractionError):
    """Indicate that Interpretation Material exceeds a configured limit."""

    code = "interpretation_input_too_large"
    status_code = 413
    retryable: ClassVar[bool] = False


class MetadataProviderError(ExtractionError):
    """Indicate that source metadata could not be retrieved or normalized."""

    code = "metadata_provider_failed"
    status_code = 502
    retryable: ClassVar[bool] = False


class TranscriptionError(ExtractionError):
    """Indicate that no transcript could be acquired."""

    code = "transcription_failed"
    status_code = 502
    retryable: ClassVar[bool] = False


class RetryableTranscriptionError(TranscriptionError):
    """Indicate a transient Transcript acquisition failure."""

    retryable: ClassVar[bool] = True


class TranscriptionUnavailableError(ExtractionError):
    """Indicate temporary Textify capacity, job-store, or coordination unavailability."""

    code = "transcription_unavailable"
    status_code = 503
    retry_after_seconds = 2
    retryable: ClassVar[bool] = True


class MentionInterpretationError(ExtractionError):
    """Indicate that mention interpretation failed."""

    code = "mention_interpretation_failed"
    status_code = 502
    retryable: ClassVar[bool] = False


class RetryableMentionInterpretationError(MentionInterpretationError):
    """Indicate a transient Mention Interpretation provider failure."""

    retryable: ClassVar[bool] = True


class InvalidLLMResponseError(MentionInterpretationError):
    """Indicate that the LLM returned malformed or invalid structured output."""

    code = "invalid_llm_response"
    retryable: ClassVar[bool] = False


class EnrichmentError(ExtractionError):
    """Indicate that provider enrichment failed."""

    code = "enrichment_failed"
    status_code = 502
    retryable: ClassVar[bool] = False


class RetryableEnrichmentError(EnrichmentError):
    """Indicate a transient enrichment provider failure."""

    retryable: ClassVar[bool] = True


class CatalogProviderError(ExtractionError):
    """Indicate that a core catalog provider request failed."""

    code = "catalog_provider_failed"
    status_code = 502
    retryable: ClassVar[bool] = False


class RetryableCatalogProviderError(CatalogProviderError):
    """Indicate a transient catalog provider failure."""

    retryable: ClassVar[bool] = True


class PipelineTimeoutError(ExtractionError):
    """Indicate that an external pipeline operation timed out."""

    code = "pipeline_timeout"
    status_code = 504
    retryable: ClassVar[bool] = True


async def extraction_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert an extraction domain error into its public error response.

    Args:
        request: Request that triggered the exception.
        exc: Exception registered as an ``ExtractionError`` handler.

    Returns:
        JSONResponse: Error body and status defined by the domain exception.
    """
    error = cast(ExtractionError, exc)
    headers = (
        {"Retry-After": str(error.retry_after_seconds)}
        if error.retry_after_seconds is not None
        else None
    )
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": str(error)}},
        headers=headers,
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
                "code": INTERNAL_ERROR_CODE,
                "message": INTERNAL_ERROR_MESSAGE,
            }
        },
    )
