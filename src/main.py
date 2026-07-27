import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.transcript import (
    TranscriptDownloadError,
    TranscriptError,
    TranscriptInvalidURLError,
    TranscriptNotFoundError,
    TranscriptResult,
    TranscriptService,
    TranscriptTranscriptionError,
    TranscriptUnsupportedPlatformError,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(asctime)s:[%(name)s]:%(message)s")
logger = logging.getLogger(__name__)

# --- Transcript Service (singleton) ---

_transcript_service = TranscriptService(
    whisper_model_size=settings.whisper_model,
    whisper_device=settings.whisper_device,
    whisper_compute_type=settings.whisper_compute_type,
    temp_dir=settings.transcript_temp_dir,
    whisper_max_concurrent=settings.whisper_max_concurrent,
    whisper_max_duration_seconds=settings.whisper_max_duration_seconds,
)


def get_transcript_service() -> TranscriptService:
    """Dependency: returns the singleton TranscriptService."""
    return _transcript_service


# Module-level singleton for FastAPI Depends, avoiding B008 (function call in default).
_transcript_dependency = Depends(get_transcript_service)

# --- Request/Response Models ---


class TranscriptRequest(BaseModel):
    url: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


# --- FastAPI Application ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Reelio")
    # Initialization:
    # - Load ML models
    # - Connect to a database
    # - Warm up caches
    # - Verify external services
    # - Start background tasks

    yield

    logger.info("Shutting down Reelio")
    # Cleanup:
    # - Close database connections
    # - Stop background workers
    # - Release resources


app = FastAPI(title="Reelio", version="0.1.0", lifespan=lifespan)


_STATUS_BY_ERROR: list[tuple[type[TranscriptError], int]] = [
    (TranscriptInvalidURLError, 400),
    (TranscriptUnsupportedPlatformError, 400),
    (TranscriptNotFoundError, 404),
    (TranscriptDownloadError, 502),
    (TranscriptTranscriptionError, 502),
]


@app.exception_handler(TranscriptError)
async def transcript_error_handler(_request: Request, exc: TranscriptError) -> JSONResponse:
    status_code = 500
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            status_code = code
            break
    if status_code >= 500:
        logger.exception("Unexpected transcript endpoint error: %s", exc)
    else:
        logger.warning("Transcript request failed: %s", exc)
    body = ErrorResponse(detail=str(exc), error_type=type(exc).__name__)
    return JSONResponse(status_code=status_code, content=body.model_dump())


# --- Routes ---


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello from Reelio!"}


@app.post(
    "/api/transcript",
    response_model=TranscriptResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL or unsupported platform"},
        404: {"model": ErrorResponse, "description": "No transcript available"},
        502: {"model": ErrorResponse, "description": "Download or transcription failed"},
        500: {"model": ErrorResponse, "description": "Unexpected transcript error"},
    },
)
async def extract_transcript(
    request: TranscriptRequest,
    service: TranscriptService = _transcript_dependency,
) -> TranscriptResult:
    """Extract a transcript from a reel/video URL.

    Supports YouTube (via direct caption fetch) and Facebook, Instagram,
    TikTok (via audio download + Whisper speech-to-text).
    """
    return await service.extract(request.url)
