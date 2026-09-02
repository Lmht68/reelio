"""HTTP routing and domain-to-schema conversion for extraction."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.service import ExtractionPipelineProtocol
from reelio.extraction.types import (
    EnrichedMovie,
    EnrichedTVSeries,
    MovieMention,
    MovieResult,
    PipelineResult,
    TVSeriesMention,
    TVSeriesResult,
)

_EXTRACT_RESPONSE_EXAMPLE = {
    "source": {
        "platform": "youtube",
        "video_id": "dQw4w9WgXcQ",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Screen Work review",
        "description": "A review mentioning Movies and TV Series.",
        "channel": "Example channel",
        "duration_seconds": 42,
    },
    "transcript": {
        "text": "Dune: Part One and The Last of Us are excellent.",
        "language": "en",
        "method": "youtube_captions",
    },
    "results": {
        "movies": [
            {
                "status": "resolved",
                "movie_mention": {"title": "Dune: Part One", "year": 2021},
                "movie": {
                    "title": "Dune: Part One",
                    "year": 2021,
                    "cast": ["Timothée Chalamet"],
                    "directors": ["Denis Villeneuve"],
                    "description": "Paul Atreides faces his destiny on Arrakis.",
                    "poster_url": "https://image.tmdb.org/t/p/w500/dune.jpg",
                    "tmdb_id": 438631,
                    "tmdb_url": "https://www.themoviedb.org/movie/438631",
                    "imdb_id": "tt1160419",
                    "imdb_url": "https://www.imdb.com/title/tt1160419/",
                    "tmdb_score": 7.8,
                },
            },
            {
                "status": "unresolved",
                "movie_mention": {"title": "Unknown Movie", "year": 2024},
                "movie": None,
            },
        ],
        "tv_series": [
            {
                "status": "resolved",
                "tv_series_mention": {"title": "The Last of Us", "year": 2023},
                "tv_series": {
                    "title": "The Last of Us",
                    "first_air_year": 2023,
                    "last_air_year": 2025,
                    "cast": ["Pedro Pascal", "Bella Ramsey"],
                    "creators": ["Craig Mazin", "Neil Druckmann"],
                    "description": "A smuggler escorts a teenager across a ruined America.",
                    "poster_url": "https://image.tmdb.org/t/p/w500/the-last-of-us.jpg",
                    "tmdb_id": 100088,
                    "tmdb_url": "https://www.themoviedb.org/tv/100088",
                    "imdb_id": "tt3581920",
                    "imdb_url": "https://www.imdb.com/title/tt3581920/",
                    "tmdb_score": 8.6,
                },
            },
            {
                "status": "unresolved",
                "tv_series_mention": {"title": "Unknown TV Series", "year": 2024},
                "tv_series": None,
            },
        ],
    },
}


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


def _to_tv_series_mention_schema(
    mention: TVSeriesMention,
) -> extraction_schemas.TVSeriesMentionModel:
    return extraction_schemas.TVSeriesMentionModel(
        title=mention.title,
        year=mention.year,
    )


def _to_movie_schema(movie: EnrichedMovie) -> extraction_schemas.MovieModel:
    return extraction_schemas.MovieModel(
        title=movie.title,
        year=movie.year,
        cast=movie.cast,
        directors=movie.directors,
        description=movie.description,
        poster_url=movie.poster_url,
        tmdb_id=movie.tmdb_id,
        tmdb_url=movie.tmdb_url,
        imdb_id=movie.imdb_id,
        imdb_url=movie.imdb_url,
        tmdb_score=movie.tmdb_score,
    )


def _to_tv_series_schema(
    tv_series: EnrichedTVSeries,
) -> extraction_schemas.TVSeriesModel:
    return extraction_schemas.TVSeriesModel(
        title=tv_series.title,
        first_air_year=tv_series.first_air_year,
        last_air_year=tv_series.last_air_year,
        cast=tv_series.cast,
        creators=tv_series.creators,
        description=tv_series.description,
        poster_url=tv_series.poster_url,
        tmdb_id=tv_series.tmdb_id,
        tmdb_url=tv_series.tmdb_url,
        imdb_id=tv_series.imdb_id,
        imdb_url=tv_series.imdb_url,
        tmdb_score=tv_series.tmdb_score,
    )


def _to_movie_result_schema(
    result: MovieResult,
) -> extraction_schemas.MovieResultModel:
    movie = _to_movie_schema(result.movie) if result.movie is not None else None
    return extraction_schemas.MovieResultModel(
        status=result.status,
        movie_mention=_to_movie_mention_schema(result.movie_mention),
        movie=movie,
    )


def _to_tv_series_result_schema(
    result: TVSeriesResult,
) -> extraction_schemas.TVSeriesResultModel:
    tv_series = _to_tv_series_schema(result.tv_series) if result.tv_series is not None else None
    return extraction_schemas.TVSeriesResultModel(
        status=result.status,
        tv_series_mention=_to_tv_series_mention_schema(result.tv_series_mention),
        tv_series=tv_series,
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
        results=extraction_schemas.ScreenWorkResultsModel(
            movies=[_to_movie_result_schema(item) for item in result.results.screen_works.movies],
            tv_series=[
                _to_tv_series_result_schema(item) for item in result.results.screen_works.tv_series
            ],
        ),
    )


@router.post(
    "/extract",
    status_code=status.HTTP_200_OK,
    response_model=extraction_schemas.ExtractResponse,
    summary="Extract mentioned Movies and TV Series from a public video Source",
    description=(
        "Accept a public YouTube, Instagram, Facebook, TikTok, or X video URL and "
        "return the normalized Source, the Transcript with its acquisition method, "
        "and grouped Movie and TV Series results. Each list preserves first-reference "
        "order within its kind, with no cross-kind ordering. Resolved TV Series report "
        "their TV First Air Year, an optional final air year where null means "
        "unavailable rather than proof of continuation, Creators from TMDB created_by, "
        "and up to five aggregate cast names in provider order without role filtering "
        "or person deduplication. Any TMDB provider failure fails the complete request."
    ),
    response_description="Source, transcript, and grouped Screen Work Results.",
    responses={
        200: {
            "description": "Grouped Movie and TV Series results.",
            "content": {
                "application/json": {"example": _EXTRACT_RESPONSE_EXAMPLE},
            },
        },
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
            "description": "Source duration or Interpretation Material exceeds its limit.",
        },
        500: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Unexpected internal failure.",
        },
        502: {
            "model": extraction_schemas.ErrorResponse,
            "description": (
                "Metadata, transcript, LLM, or TMDB provider failure. Any TMDB "
                "provider failure fails the complete request."
            ),
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
    """Extract structured Movie and TV Series Mentions from a submitted source URL.

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
