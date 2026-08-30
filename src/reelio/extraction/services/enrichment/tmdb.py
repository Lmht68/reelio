"""Resolve and enrich Movie Mentions through TMDB."""

import asyncio
import logging
from collections.abc import Sequence
from typing import cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reelio.extraction.exceptions import EnrichmentError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import TMDBConfig
from reelio.extraction.types import (
    EnrichedMovie,
    MentionResult,
    MovieMention,
    ResultStatus,
    normalize_screen_work_title,
)

logger = logging.getLogger(__name__)

_ENRICHMENT_ERROR_MESSAGE = "TMDB candidate resolution and enrichment failed."
_ENRICHMENT_TIMEOUT_MESSAGE = "TMDB candidate resolution timed out."
_STAGE = "candidate_resolution"


class _TMDBModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _SearchCandidate(_TMDBModel):
    id: int
    title: str = ""
    original_title: str = ""
    release_date: str = ""


class _SearchResponse(_TMDBModel):
    results: list[_SearchCandidate] = Field(default_factory=list)
    total_pages: int = Field(default=0, ge=0)


class _CrewMember(_TMDBModel):
    name: str
    job: str


class _CastMember(_TMDBModel):
    name: str


class _Credits(_TMDBModel):
    cast: list[_CastMember] = Field(default_factory=list)
    crew: list[_CrewMember] = Field(default_factory=list)


class _AlternativeTitle(_TMDBModel):
    title: str


class _AlternativeTitles(_TMDBModel):
    titles: list[_AlternativeTitle] = Field(default_factory=list)


class _MovieDetails(_TMDBModel):
    id: int
    title: str
    release_date: str = ""
    overview: str = ""
    poster_path: str | None = None
    imdb_id: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    credits: _Credits = Field(default_factory=_Credits)
    alternative_titles: _AlternativeTitles = Field(default_factory=_AlternativeTitles)


class TMDBMovieResolver:
    """Resolve ordered Movie Mentions and attach TMDB-backed metadata."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        image_base_url: str,
    ) -> None:
        """Initialize the resolver with a reusable authenticated client.

        Args:
            client: Lifespan-owned TMDB HTTP client.
            image_base_url: TMDB image URL prefix including the desired size.
        """
        self._client = client
        self._image_base_url = image_base_url.rstrip("/")

    async def resolve(
        self,
        movie_mentions: Sequence[MovieMention],
    ) -> list[MentionResult]:
        """Resolve and enrich Movie Mentions while preserving their order.

        Args:
            movie_mentions: Canonical Movie Mentions in first-reference order.

        Returns:
            list[MentionResult]: One Resolved or Unresolved Result per Movie Mention.

        Raises:
            EnrichmentError: If TMDB fails or returns an invalid response.
            PipelineTimeoutError: If a TMDB request times out.
        """
        if not movie_mentions:
            return []
        return list(await asyncio.gather(*(self._resolve_one(item) for item in movie_mentions)))

    async def aclose(self) -> None:
        """Close the lifespan-owned TMDB client and its connection pool."""
        await self._client.aclose()

    async def _resolve_one(self, movie_mention: MovieMention) -> MentionResult:
        normalized_mention_title = normalize_screen_work_title(movie_mention.title)
        search_response = await self._get_model(
            "search/movie",
            {
                "query": movie_mention.title,
                "include_adult": True,
                "language": "en-US",
                "year": movie_mention.year,
                "page": 1,
            },
            _SearchResponse,
        )

        # Limit to the first three candidates to reduce TMDB requests
        for candidate in search_response.results[:3]:
            candidate_year = _release_year(candidate.release_date)

            if candidate_year is None or candidate_year != movie_mention.year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == normalized_mention_title
                for title in (candidate.title, candidate.original_title)
            )
            # Dont need to check alternative titles if the main title matches
            append_to_response = (
                "credits" if primary_titles_matched else "credits,alternative_titles"
            )
            movie = await self._get_model(
                f"movie/{candidate.id}",
                {
                    "append_to_response": append_to_response,
                    "language": "en-US",
                },
                _MovieDetails,
            )

            if not primary_titles_matched:
                if not any(
                    normalize_screen_work_title(alternative_title.title) == normalized_mention_title
                    for alternative_title in movie.alternative_titles.titles
                ):
                    continue

            return MentionResult(
                status=ResultStatus.RESOLVED,
                movie_mention=movie_mention,
                movie=self._enrich(movie_mention, movie),
            )

        return MentionResult(
            status=ResultStatus.UNRESOLVED,
            movie_mention=movie_mention,
            movie=None,
        )

    async def _get_model[ModelType: BaseModel](
        self,
        path: str,
        params: dict[str, str | int | bool],
        model_type: type[ModelType],
    ) -> ModelType:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error(
                "TMDB request timed out",
                extra={"stage": _STAGE, "reason": "provider_timeout"},
            )
            raise PipelineTimeoutError(_ENRICHMENT_TIMEOUT_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.error(
                "TMDB request failed",
                extra={"stage": _STAGE, "reason": "provider_failure"},
            )
            raise EnrichmentError(_ENRICHMENT_ERROR_MESSAGE) from exc

        try:
            payload = cast(object, response.json())
            return model_type.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            logger.error(
                "TMDB response validation failed",
                extra={"stage": _STAGE, "reason": "invalid_provider_response"},
            )
            raise EnrichmentError(_ENRICHMENT_ERROR_MESSAGE) from exc

    def _enrich(
        self,
        movie_mention: MovieMention,
        movie: _MovieDetails,
    ) -> EnrichedMovie:
        cast_members = [member.name for member in movie.credits.cast[:5]]
        directors = list(
            dict.fromkeys(member.name for member in movie.credits.crew if member.job == "Director")
        )
        poster_url = (
            f"{self._image_base_url}/{movie.poster_path.lstrip('/')}" if movie.poster_path else None
        )
        imdb_id = movie.imdb_id.strip() if movie.imdb_id else None
        return EnrichedMovie(
            title=movie_mention.title,
            year=movie_mention.year,
            cast=cast_members,
            directors=directors,
            description=movie.overview,
            poster_url=poster_url,
            tmdb_id=movie.id,
            tmdb_url=f"https://www.themoviedb.org/movie/{movie.id}",
            imdb_id=imdb_id,
            imdb_url=(f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None),
            tmdb_score=movie.vote_average,
        )


def create_tmdb_movie_resolver(settings: TMDBConfig) -> TMDBMovieResolver:
    """Create a reusable authenticated TMDB Movie Resolver.

    Args:
        settings: Validated TMDB credentials, endpoints, and timeout.

    Returns:
        TMDBMovieResolver: Resolver owning one asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url.rstrip('/')}/",
        headers={
            "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
            "accept": "application/json",
        },
        timeout=settings.request_timeout_seconds,
    )
    return TMDBMovieResolver(client, settings.image_base_url)


def _release_year(release_date: str) -> int | None:
    year = release_date[:4]
    return int(year) if len(year) == 4 and year.isdigit() else None
