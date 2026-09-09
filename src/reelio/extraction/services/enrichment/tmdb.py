"""Resolve and enrich grouped Screen Work Mentions through TMDB."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, cast

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from rapidfuzz import fuzz

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
_CANDIDATE_LIMIT = 3
_FUZZY_TITLE_SCORE_THRESHOLD = 80.0


def _normalize_fuzzy_screen_work_title(title: str) -> str:
    return normalize_screen_work_title(title).casefold()


def _has_fuzzy_title_match(
    normalized_mention_title: str,
    primary_title: str,
    original_title: str,
) -> bool:
    return any(
        normalized_mention_title
        and normalized_provider_title
        and fuzz.ratio(normalized_mention_title, normalized_provider_title)
        > _FUZZY_TITLE_SCORE_THRESHOLD
        for normalized_provider_title in (
            _normalize_fuzzy_screen_work_title(primary_title),
            _normalize_fuzzy_screen_work_title(original_title),
        )
    )


class _TMDBModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _parse_optional_tmdb_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


type _OptionalTMDBDate = Annotated[
    date | None,
    BeforeValidator(_parse_optional_tmdb_date),
]


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


class _MovieDetailsBase(_TMDBModel):
    id: int
    title: str
    overview: str = ""
    poster_path: str | None = None
    imdb_id: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    credits: _MovieCredits = Field(default_factory=_MovieCredits)


class _MovieDetails(_MovieDetailsBase):
    release_date: date
    alternative_titles: _MovieAlternativeTitles = Field(default_factory=_MovieAlternativeTitles)


class _FallbackMovieDetails(_MovieDetailsBase):
    release_date: _OptionalTMDBDate = None


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
    first_air_date: _OptionalTMDBDate = None


@dataclass(slots=True)
class _MovieMatch:
    details: _MovieDetailsBase
    release_year: int


@dataclass(slots=True)
class _TVSeriesMatch:
    details: _TVSeriesDetails
    first_air_year: int


@dataclass(slots=True)
class _MovieResolutionContext:
    mention: MovieMention
    normalized_title: str
    candidates: list[_MovieSearchCandidate] = field(default_factory=list)
    details_by_candidate_id: dict[int, _MovieDetailsBase] = field(default_factory=dict)
    match: _MovieMatch | None = None


@dataclass(slots=True)
class _TVResolutionContext:
    mention: TVSeriesMention
    normalized_title: str
    candidates: list[_TVSearchCandidate] = field(default_factory=list)
    details_by_candidate_id: dict[int, _TVSeriesDetails] = field(default_factory=dict)
    match: _TVSeriesMatch | None = None


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
        movie_contexts = [
            _MovieResolutionContext(
                mention=movie_mention,
                normalized_title=normalize_screen_work_title(movie_mention.title),
            )
            for movie_mention in screen_work_mentions.movies
        ]
        tv_series_contexts = [
            _TVResolutionContext(
                mention=tv_series_mention,
                normalized_title=normalize_screen_work_title(tv_series_mention.title),
            )
            for tv_series_mention in screen_work_mentions.tv_series
        ]
        await asyncio.gather(
            *(self._resolve_movie_strict(context) for context in movie_contexts),
            *(self._resolve_tv_series_strict(context) for context in tv_series_contexts),
        )
        await asyncio.gather(
            *(
                self._resolve_movie_direct_fuzzy(context)
                for context in movie_contexts
                if context.match is None
            ),
            *(
                self._resolve_tv_series_direct_fuzzy(context)
                for context in tv_series_contexts
                if context.match is None
            ),
        )
        return ScreenWorkResults(
            movies=[self._to_movie_result(context) for context in movie_contexts],
            tv_series=[self._to_tv_series_result(context) for context in tv_series_contexts],
        )

    async def aclose(self) -> None:
        """Close the lifespan-owned TMDB client and its connection pool."""
        await self._client.aclose()

    async def _resolve_movie_strict(self, context: _MovieResolutionContext) -> None:
        for search_year in (
            context.mention.year,
            context.mention.year + 1,
            context.mention.year - 1,
        ):
            match = await self._find_movie_in_year(context, search_year)
            if match is not None:
                context.match = match
                return

    async def _find_movie_in_year(
        self,
        context: _MovieResolutionContext,
        search_year: int,
    ) -> _MovieMatch | None:
        search_response = await self._get_model(
            "search/movie",
            {
                "query": context.mention.title,
                "include_adult": True,
                "language": "en-US",
                "year": search_year,
                "page": 1,
            },
            _MovieSearchResponse,
        )
        candidates = search_response.results[:_CANDIDATE_LIMIT]
        context.candidates.extend(candidates)

        for candidate in candidates:
            candidate_year = _year_from_date(candidate.release_date)
            if candidate_year is None or candidate_year != search_year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == context.normalized_title
                for title in (candidate.title, candidate.original_title)
            )
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
            context.details_by_candidate_id[candidate.id] = movie

            if not primary_titles_matched and not any(
                normalize_screen_work_title(alternative_title.title) == context.normalized_title
                for alternative_title in movie.alternative_titles.titles
            ):
                continue

            if abs(movie.release_date.year - context.mention.year) > 1:
                continue

            return _MovieMatch(details=movie, release_year=movie.release_date.year)

        return None

    async def _resolve_tv_series_strict(self, context: _TVResolutionContext) -> None:
        for search_year in (
            context.mention.year,
            context.mention.year + 1,
            context.mention.year - 1,
        ):
            match = await self._find_tv_series_in_year(context, search_year)
            if match is not None:
                context.match = match
                return

    async def _find_tv_series_in_year(
        self,
        context: _TVResolutionContext,
        search_year: int,
    ) -> _TVSeriesMatch | None:
        search_response = await self._get_model(
            "search/tv",
            {
                "query": context.mention.title,
                "include_adult": True,
                "language": "en-US",
                "first_air_date_year": search_year,
                "page": 1,
            },
            _TVSearchResponse,
        )
        candidates = search_response.results[:_CANDIDATE_LIMIT]
        context.candidates.extend(candidates)

        for candidate in candidates:
            candidate_year = _year_from_date(candidate.first_air_date)
            if candidate_year is None or candidate_year != search_year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == context.normalized_title
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
            context.details_by_candidate_id[candidate.id] = tv_series

            if not primary_titles_matched and not any(
                normalize_screen_work_title(alternative_title.title) == context.normalized_title
                for alternative_title in tv_series.alternative_titles.titles
            ):
                continue

            return _TVSeriesMatch(
                details=tv_series,
                first_air_year=candidate_year,
            )

        return None

    async def _resolve_movie_direct_fuzzy(
        self,
        context: _MovieResolutionContext,
    ) -> None:
        normalized_mention_title = _normalize_fuzzy_screen_work_title(context.mention.title)
        for candidate in context.candidates:
            if not _has_fuzzy_title_match(
                normalized_mention_title,
                candidate.title,
                candidate.original_title,
            ):
                continue

            movie = context.details_by_candidate_id.get(candidate.id)
            if movie is None:
                movie = await self._get_model(
                    f"movie/{candidate.id}",
                    {
                        "append_to_response": "credits",
                        "language": "en-US",
                    },
                    _FallbackMovieDetails,
                )
                context.details_by_candidate_id[candidate.id] = movie

            release_date = cast(
                _MovieDetails | _FallbackMovieDetails,
                movie,
            ).release_date
            if release_date is None or abs(release_date.year - context.mention.year) > 1:
                continue

            context.match = _MovieMatch(
                details=movie,
                release_year=release_date.year,
            )
            return

    async def _resolve_tv_series_direct_fuzzy(
        self,
        context: _TVResolutionContext,
    ) -> None:
        normalized_mention_title = _normalize_fuzzy_screen_work_title(context.mention.title)
        for candidate in context.candidates:
            if not _has_fuzzy_title_match(
                normalized_mention_title,
                candidate.name,
                candidate.original_name,
            ):
                continue

            tv_series = context.details_by_candidate_id.get(candidate.id)
            if tv_series is None:
                tv_series = await self._get_model(
                    f"tv/{candidate.id}",
                    {
                        "append_to_response": "aggregate_credits,external_ids",
                        "language": "en-US",
                    },
                    _TVSeriesDetails,
                )
                context.details_by_candidate_id[candidate.id] = tv_series

            if (
                tv_series.first_air_date is None
                or abs(tv_series.first_air_date.year - context.mention.year) > 1
            ):
                continue

            context.match = _TVSeriesMatch(
                details=tv_series,
                first_air_year=tv_series.first_air_date.year,
            )
            return

    def _to_movie_result(self, context: _MovieResolutionContext) -> MovieResult:
        if context.match is None:
            return MovieResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=context.mention,
                movie=None,
            )
        return MovieResult(
            status=ResultStatus.RESOLVED,
            movie_mention=context.mention,
            movie=self._enrich_movie(
                context.match.details,
                context.match.release_year,
            ),
        )

    def _to_tv_series_result(self, context: _TVResolutionContext) -> TVSeriesResult:
        if context.match is None:
            return TVSeriesResult(
                status=ResultStatus.UNRESOLVED,
                tv_series_mention=context.mention,
                tv_series=None,
            )
        return TVSeriesResult(
            status=ResultStatus.RESOLVED,
            tv_series_mention=context.mention,
            tv_series=self._enrich_tv_series(
                context.match.details,
                context.match.first_air_year,
            ),
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
        movie: _MovieDetailsBase,
        release_year: int,
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
            title=movie.title,
            year=release_year,
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
        tv_series: _TVSeriesDetails,
        first_air_year: int,
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
            title=tv_series.name,
            first_air_year=first_air_year,
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
