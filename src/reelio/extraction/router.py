"""HTTP routing and domain-to-schema conversion for extraction."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.service import ExtractionPipelineProtocol
from reelio.extraction.types import (
    EnrichedMovie,
    MentionResult,
    MovieMention,
    PipelineResult,
)

router = APIRouter(prefix="/api", tags=["extraction"])


def get_pipeline(request: Request) -> ExtractionPipelineProtocol:
    """Provide the lifespan-owned extraction pipeline.

    Args:
        request: Current request whose application owns the pipeline.

    Returns:
        ExtractionPipelineProtocol: The composed extraction pipeline.
    """
    return cast(ExtractionPipelineProtocol, request.app.state.extraction_pipeline)


def _to_movie_mention_schema(mention: MovieMention) -> extraction_schemas.MovieMentionModel:
    return extraction_schemas.MovieMentionModel(
        title=mention.title,
        year=mention.year,
    )


def _to_movie_schema(movie: EnrichedMovie) -> extraction_schemas.MovieModel:
    return extraction_schemas.MovieModel(
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


def _to_result_schema(result: MentionResult) -> extraction_schemas.ResultModel:
    movie_mention = _to_movie_mention_schema(result.movie_mention) if result.movie_mention is not None else None
    movie = _to_movie_schema(result.movie) if result.movie is not None else None
    return extraction_schemas.ResultModel(
        status=result.status,
        movie_mention=movie_mention,
        movie=movie,
    )


def _to_response(result: PipelineResult) -> extraction_schemas.ExtractResponse:
    return extraction_schemas.ExtractResponse(
        source=extraction_schemas.SourceModel(
            platform=result.source.platform,
            video_id=result.source.video_id,
            url=result.source.url,
            title=result.source.title,
            description=result.source.description,
            channel=result.source.channel,
            duration_seconds=result.source.duration_seconds,
        ),
        transcript=extraction_schemas.TranscriptModel(
            text=result.transcript.text,
            language=result.transcript.language,
            method=result.transcript.method,
        ),
        results=[_to_result_schema(item) for item in result.results],
    )


@router.post(
    "/extract",
    status_code=status.HTTP_200_OK,
    summary="Extract mentioned movies from a supported public video Source",
    description=(
        "Accept a public YouTube, Instagram, Facebook, TikTok, or X video URL "
        "and return the normalized Source, the Transcript with its acquisition "
        "method, and one Resolved, Ambiguous, or Unresolved result per mentioned "
        "movie."
    ),
    response_description="Source, transcript, and per-mention results.",
    responses={
        400: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Invalid URL, unsupported platform, or unsupported Source.",
        },
        404: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Source unavailable, private, or not found.",
        },
        413: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Source exceeds the duration limit.",
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
    pipeline: Annotated[ExtractionPipelineProtocol, Depends(get_pipeline)],
) -> extraction_schemas.ExtractResponse:
    """Extract structured movie mentions from a submitted source URL.

    Args:
        payload: Validated extraction request containing the source URL.
        pipeline: ExtractionPipelineProtocol implementation supplied by FastAPI dependency injection.

    Returns:
        ExtractResponse: Canonical source, transcript, and mention results.

    Raises:
        ExtractionError: If the pipeline raises an extraction domain error.
    """
    result = await pipeline.run(payload.url)
    return _to_response(result)
