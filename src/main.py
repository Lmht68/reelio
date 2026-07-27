import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
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

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(asctime)s:[%(name)s]:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Transcript Service (singleton) ---

_transcript_service = TranscriptService(
    whisper_model_size=settings.whisper_model,
    whisper_device=settings.whisper_device,
    whisper_compute_type=settings.whisper_compute_type,
    temp_dir=settings.transcript_temp_dir,
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
    try:
        return await service.extract(request.url)
    except (TranscriptInvalidURLError, TranscriptUnsupportedPlatformError) as exc:
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptNotFoundError as exc:
        logger.warning("Transcript not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TranscriptDownloadError, TranscriptTranscriptionError) as exc:
        logger.exception("Transcript processing failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TranscriptError as exc:
        logger.exception("Unexpected transcript endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
