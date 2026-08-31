"""Resolve and enrich grouped Screen Work Mentions through TMDB."""

import asyncio
import logging
from typing import cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reelio.extraction.exceptions import EnrichmentError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import TMDBConfig
from reelio.extraction.types import (
    EnrichedMovie,
    EnrichedTVSeries,
    MovieMention,
    MovieResult,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    TVSeriesMention,
    TVSeriesResult,
    normalize_screen_work_title,
)

logger = logging.getLogger(__name__)

_ENRICHMENT_ERROR_MESSAGE = "TMDB candidate resolution and enrichment failed."
_ENRICHMENT_TIMEOUT_MESSAGE = "TMDB candidate resolution timed out."
_STAGE = "candidate_resolution"


class _TMDBModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _MovieSearchCandidate(_TMDBModel):
    id: int
    title: str = ""
    original_title: str = ""
    release_date: str = ""


class _MovieSearchResponse(_TMDBModel):
    results: list[_MovieSearchCandidate] = Field(default_factory=list)
    total_pages: int = Field(default=0, ge=0)


class _CrewMember(_TMDBModel):
    name: str
    job: str


class _CastMember(_TMDBModel):
    name: str


class _MovieCredits(_TMDBModel):
    cast: list[_CastMember] = Field(default_factory=list)
    crew: list[_CrewMember] = Field(default_factory=list)


class _AlternativeTitle(_TMDBModel):
    title: str


class _MovieAlternativeTitles(_TMDBModel):
    titles: list[_AlternativeTitle] = Field(default_factory=list)


class _MovieDetails(_TMDBModel):
    id: int
    title: str
    release_date: str = ""
    overview: str = ""
    poster_path: str | None = None
    imdb_id: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    credits: _MovieCredits = Field(default_factory=_MovieCredits)
    alternative_titles: _MovieAlternativeTitles = Field(default_factory=_MovieAlternativeTitles)


class _TVSearchCandidate(_TMDBModel):
    id: int
    name: str = ""
    original_name: str = ""
    first_air_date: str = ""


class _TVSearchResponse(_TMDBModel):
    results: list[_TVSearchCandidate] = Field(default_factory=list)
    total_pages: int = Field(default=0, ge=0)


class _Creator(_TMDBModel):
    name: str


class _TVAlternativeTitles(_TMDBModel):
    titles: list[_AlternativeTitle] = Field(default_factory=list)


class _TVAggregateCredits(_TMDBModel):
    cast: list[_CastMember] = Field(default_factory=list)


class _TVExternalIDs(_TMDBModel):
    imdb_id: str | None = None


class _TVSeriesDetails(_TMDBModel):
    id: int
    name: str
    aggregate_credits: _TVAggregateCredits
    external_ids: _TVExternalIDs
    status: str = ""
    last_air_date: str | None = None
    created_by: list[_Creator] = Field(default_factory=list)
    overview: str = ""
    poster_path: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    alternative_titles: _TVAlternativeTitles = Field(default_factory=_TVAlternativeTitles)


class TMDBScreenWorkResolver:
    """Resolve ordered Movies and TV Series and attach TMDB-backed metadata."""

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
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        """Resolve grouped Screen Work Mentions while preserving per-kind order.

        Args:
            screen_work_mentions: Canonical mentions grouped by Screen Work kind.

        Returns:
            ScreenWorkResults: Resolved or unresolved Results grouped by kind.

        Raises:
            EnrichmentError: If TMDB fails or returns an invalid response.
            PipelineTimeoutError: If a TMDB request times out.
        """
        movie_results, tv_series_results = await asyncio.gather(
            asyncio.gather(
                *(
                    self._resolve_movie(movie_mention)
                    for movie_mention in screen_work_mentions.movies
                )
            ),
            asyncio.gather(
                *(
                    self._resolve_tv_series(tv_series_mention)
                    for tv_series_mention in screen_work_mentions.tv_series
                )
            ),
        )
        return ScreenWorkResults(
            movies=movie_results,
            tv_series=tv_series_results,
        )

    async def aclose(self) -> None:
        """Close the lifespan-owned TMDB client and its connection pool."""
        await self._client.aclose()

    async def _resolve_movie(self, movie_mention: MovieMention) -> MovieResult:
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
            _MovieSearchResponse,
        )

        # Limit to the first three candidates to reduce TMDB requests
        for candidate in search_response.results[:3]:
            candidate_year = _year_from_date(candidate.release_date)

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

            return MovieResult(
                status=ResultStatus.RESOLVED,
                movie_mention=movie_mention,
                movie=self._enrich_movie(movie_mention, movie),
            )

        return MovieResult(
            status=ResultStatus.UNRESOLVED,
            movie_mention=movie_mention,
            movie=None,
        )

    async def _resolve_tv_series(
        self,
        tv_series_mention: TVSeriesMention,
    ) -> TVSeriesResult:
        normalized_mention_title = normalize_screen_work_title(tv_series_mention.title)
        search_response = await self._get_model(
            "search/tv",
            {
                "query": tv_series_mention.title,
                "include_adult": True,
                "language": "en-US",
                "first_air_date_year": tv_series_mention.year,
                "page": 1,
            },
            _TVSearchResponse,
        )

        for candidate in search_response.results[:3]:
            candidate_year = _year_from_date(candidate.first_air_date)
            if candidate_year is None or candidate_year != tv_series_mention.year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == normalized_mention_title
                for title in (candidate.name, candidate.original_name)
            )

            append_to_response = (
                "aggregate_credits,external_ids"
                if primary_titles_matched
                else "aggregate_credits,alternative_titles,external_ids"
            )
            tv_series = await self._get_model(
                f"tv/{candidate.id}",
                {
                    "append_to_response": append_to_response,
                    "language": "en-US",
                },
                _TVSeriesDetails,
            )

            if not primary_titles_matched:
                if not any(
                    normalize_screen_work_title(alternative_title.title) == normalized_mention_title
                    for alternative_title in tv_series.alternative_titles.titles
                ):
                    continue

            return TVSeriesResult(
                status=ResultStatus.RESOLVED,
                tv_series_mention=tv_series_mention,
                tv_series=self._enrich_tv_series(tv_series_mention, tv_series),
            )

        return TVSeriesResult(
            status=ResultStatus.UNRESOLVED,
            tv_series_mention=tv_series_mention,
            tv_series=None,
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

    def _enrich_movie(
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

    def _enrich_tv_series(
        self,
        tv_series_mention: TVSeriesMention,
        tv_series: _TVSeriesDetails,
    ) -> EnrichedTVSeries:
        cast_members = [member.name for member in tv_series.aggregate_credits.cast[:5]]
        creators = list(dict.fromkeys(creator.name for creator in tv_series.created_by))
        poster_url = (
            f"{self._image_base_url}/{tv_series.poster_path.lstrip('/')}"
            if tv_series.poster_path
            else None
        )
        imdb_id = tv_series.external_ids.imdb_id
        if imdb_id:
            imdb_id = imdb_id.strip()
        last_air_year = (
            _year_from_date(tv_series.last_air_date or "")
            if tv_series.status in {"Ended", "Canceled"}
            else None
        )
        return EnrichedTVSeries(
            title=tv_series_mention.title,
            first_air_year=tv_series_mention.year,
            last_air_year=last_air_year,
            cast=cast_members,
            creators=creators,
            description=tv_series.overview,
            poster_url=poster_url,
            tmdb_id=tv_series.id,
            tmdb_url=f"https://www.themoviedb.org/tv/{tv_series.id}",
            imdb_id=imdb_id,
            imdb_url=(f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None),
            tmdb_score=tv_series.vote_average,
        )


def create_tmdb_screen_work_resolver(settings: TMDBConfig) -> TMDBScreenWorkResolver:
    """Create a reusable authenticated TMDB Screen Work Resolver.

    Args:
        settings: Validated TMDB credentials, endpoints, and timeout.

    Returns:
        TMDBScreenWorkResolver: Resolver owning one asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url.rstrip('/')}/",
        headers={
            "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
            "accept": "application/json",
        },
        timeout=settings.request_timeout_seconds,
    )
    return TMDBScreenWorkResolver(client, settings.image_base_url)


def _year_from_date(release_date: str) -> int | None:
    year = release_date[:4]
    return int(year) if len(year) == 4 and year.isdigit() else None
