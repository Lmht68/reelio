"""HTTP routing and domain-to-schema conversion for extraction."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.service import FakePipeline, Pipeline
from reelio.extraction.types import (
    Candidate,
    EnrichedMovie,
    MentionResult,
    PipelineResult,
)

router = APIRouter(prefix="/api", tags=["extraction"])

_fake_pipeline = FakePipeline()


def get_pipeline() -> Pipeline:
    """Provide the extraction pipeline implementation for dependency injection.

    Returns:
        Pipeline: The stateless fake pipeline used during the contract phase.
    """
    return _fake_pipeline


def _to_movie_schema(movie: EnrichedMovie) -> extraction_schemas.Movie:
    return extraction_schemas.Movie(
        title=movie.title,
        year=movie.year,
        directors=movie.directors,
        description=movie.description,
        poster_url=movie.poster_url,
        tmdb_id=movie.tmdb_id,
        tmdb_url=movie.tmdb_url,
        imdb_id=movie.imdb_id,
        imdb_url=movie.imdb_url,
        tmdb_score=movie.tmdb_score,
    )


def _to_candidate_schema(candidate: Candidate) -> extraction_schemas.Candidate:
    movie_data = _to_movie_schema(candidate).model_dump()
    return extraction_schemas.Candidate(
        **movie_data, resolution_score=candidate.resolution_score
    )


def _to_result_schema(result: MentionResult) -> extraction_schemas.Result:
    movie = _to_movie_schema(result.movie) if result.movie is not None else None
    return extraction_schemas.Result(
        status=result.status,
        mentioned_as=result.mentioned_as,
        evidence=result.evidence,
        extraction_confidence=result.extraction_confidence,
        resolution_confidence=result.resolution_confidence,
        movie=movie,
        candidates=[_to_candidate_schema(candidate) for candidate in result.candidates],
    )


def _to_response(result: PipelineResult) -> extraction_schemas.ExtractResponse:
    return extraction_schemas.ExtractResponse(
        source=extraction_schemas.Source(
            platform=result.source.platform,
            video_id=result.source.video_id,
            url=result.source.url,
            title=result.source.title,
            description=result.source.description,
            channel=result.source.channel,
            duration_seconds=result.source.duration_seconds,
        ),
        transcript=extraction_schemas.Transcript(
            text=result.transcript.text,
            language=result.transcript.language,
            method=result.transcript.method,
        ),
        results=[_to_result_schema(item) for item in result.results],
    )


@router.post(
    "/extract",
    status_code=status.HTTP_200_OK,
    summary="Extract mentioned movies from a YouTube video",
    description=(
        "Accept a public YouTube URL and return the normalized Source, the "
        "Transcript with its acquisition method, and one Resolved, Ambiguous, "
        "or Unresolved result per mentioned movie."
    ),
    response_description="Source, transcript, and per-mention results.",
    responses={
        400: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Invalid URL or unsupported platform.",
        },
        404: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Video unavailable, private, or not found.",
        },
        413: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Video exceeds the duration limit.",
        },
        500: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Unexpected internal failure.",
        },
        502: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Metadata, transcript, LLM, or TMDB provider failure.",
        },
        504: {
            "model": extraction_schemas.ErrorResponse,
            "description": "External provider timeout.",
        },
    },
)
async def extract(
    payload: extraction_schemas.ExtractRequest,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
) -> extraction_schemas.ExtractResponse:
    """Extract structured movie mentions from a submitted source URL.

    Args:
        payload: Validated extraction request containing the source URL.
        pipeline: Pipeline implementation supplied by FastAPI dependency injection.

    Returns:
        ExtractResponse: Canonical source, transcript, and mention results.

    Raises:
        ExtractionError: If the pipeline raises an extraction domain error.
    """
    result = await pipeline.run(payload.url)
    return _to_response(result)
