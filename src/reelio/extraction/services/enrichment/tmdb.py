"""Resolve and enrich grouped Screen Work Mentions through TMDB."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, cast

import httpx
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from rapidfuzz import fuzz

from reelio.cache import AsyncCache, CacheCodec, CacheCodecError, CacheEntry
from reelio.cache.interface import JsonObject
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
_DIRECT_SEARCH_CANDIDATE_LIMIT = 3
_FUZZY_TITLE_SCORE_THRESHOLD = 80.0
_EMPTY_SEARCH_TTL_SECONDS = 900
_POSITIVE_SEARCH_TTL_SECONDS = 21_600
_DETAIL_TTL_SECONDS = 86_400
_TMDB_ADAPTER_CONTRACT_VERSION = "tmdb-screen-work-v1"


def _normalize_fuzzy_screen_work_title(title: str) -> str:
    return normalize_screen_work_title(title).casefold()


def _screen_work_search_fragments(title: str) -> tuple[str, ...]:
    tokens = normalize_screen_work_title(title).split()
    if len(tokens) <= 1:
        return ()

    suffix = " ".join(tokens[1:])
    prefix = " ".join(tokens[:-1])
    return (suffix,) if suffix == prefix else (suffix, prefix)


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
    """Ignore unknown fields at the TMDB provider boundary."""

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


class _MovieSearchCandidateModel(_TMDBModel):
    """Model one Movie Search Candidate returned by TMDB."""

    id: int
    title: str = ""
    original_title: str = ""
    release_date: str = ""


class _MovieSearchResponseModel(_TMDBModel):
    """Model the TMDB Movie Search response."""

    results: list[_MovieSearchCandidateModel] = Field(default_factory=list)


class _CrewMemberModel(_TMDBModel):
    """Model one TMDB Movie Crew credit."""

    name: str
    job: str


class _CastMemberModel(_TMDBModel):
    """Model one TMDB cast credit."""

    name: str


class _MovieCreditsModel(_TMDBModel):
    """Model TMDB Movie credits."""

    cast: list[_CastMemberModel] = Field(default_factory=list)
    crew: list[_CrewMemberModel] = Field(default_factory=list)


class _AlternativeTitleModel(_TMDBModel):
    """Model one TMDB alternative title."""

    title: str


class _MovieAlternativeTitlesModel(_TMDBModel):
    """Model TMDB Movie alternative titles."""

    titles: list[_AlternativeTitleModel] = Field(default_factory=list)


class _MovieDetailModel(_TMDBModel):
    """Model common TMDB Movie detail fields used by Reelio."""

    id: int
    title: str
    overview: str = ""
    poster_path: str | None = None
    imdb_id: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    credits: _MovieCreditsModel = Field(default_factory=_MovieCreditsModel)


class _RequiredMovieDetailModel(_MovieDetailModel):
    """Model Movie details that must include a valid release date."""

    release_date: date
    alternative_titles: _MovieAlternativeTitlesModel = Field(
        default_factory=_MovieAlternativeTitlesModel
    )


class _OptionalMovieDetailModel(_MovieDetailModel):
    """Model Movie details whose release date may be unavailable."""

    release_date: _OptionalTMDBDate = None


class _TVSeriesSearchCandidateModel(_TMDBModel):
    """Model one TV Series Search Candidate returned by TMDB."""

    id: int
    name: str = ""
    original_name: str = ""
    first_air_date: str = ""


class _TVSeriesSearchResponseModel(_TMDBModel):
    """Model the TMDB TV Series Search response."""

    results: list[_TVSeriesSearchCandidateModel] = Field(default_factory=list)


class _CreatorModel(_TMDBModel):
    """Model one TMDB TV Series Creator."""

    name: str


class _TVAlternativeTitlesModel(_TMDBModel):
    """Model TMDB TV Series alternative titles."""

    titles: list[_AlternativeTitleModel] = Field(default_factory=list)


class _TVAggregateCreditsModel(_TMDBModel):
    """Model TMDB TV Series aggregate credits."""

    cast: list[_CastMemberModel] = Field(default_factory=list)


class _TVExternalIDsModel(_TMDBModel):
    """Model TMDB TV Series external IDs."""

    imdb_id: str | None = None


class _TVSeriesDetailModel(_TMDBModel):
    """Model TMDB TV Series details required for Reelio resolution."""

    id: int
    name: str
    aggregate_credits: _TVAggregateCreditsModel
    external_ids: _TVExternalIDsModel
    status: str = ""
    last_air_date: str | None = None
    created_by: list[_CreatorModel] = Field(default_factory=list)
    overview: str = ""
    poster_path: str | None = None
    vote_average: float = Field(default=0.0, ge=0, le=10)
    alternative_titles: _TVAlternativeTitlesModel = Field(default_factory=_TVAlternativeTitlesModel)
    first_air_date: _OptionalTMDBDate = None


@dataclass(frozen=True, slots=True)
class _MovieSearchCandidate:
    """Contain one normalized Movie Search Candidate."""

    tmdb_id: int
    title: str
    original_title: str
    release_year: int | None


@dataclass(frozen=True, slots=True)
class _TVSeriesSearchCandidate:
    """Contain one normalized TV Series Search Candidate."""

    tmdb_id: int
    name: str
    original_name: str
    first_air_year: int | None


@dataclass(frozen=True, slots=True)
class _MovieDetailCandidate:
    """Contain normalized Movie detail metadata used by Reelio."""

    tmdb_id: int
    title: str
    release_year: int | None
    alternative_titles: tuple[str, ...]
    cast: tuple[str, ...]
    directors: tuple[str, ...]
    description: str
    poster_url: str | None
    imdb_id: str | None
    tmdb_score: float


@dataclass(frozen=True, slots=True)
class _TVSeriesDetailCandidate:
    """Contain normalized TV Series detail metadata used by Reelio."""

    tmdb_id: int
    name: str
    first_air_year: int | None
    last_air_year: int | None
    alternative_titles: tuple[str, ...]
    cast: tuple[str, ...]
    creators: tuple[str, ...]
    description: str
    poster_url: str | None
    imdb_id: str | None
    tmdb_score: float


class _TMDBCacheModel(BaseModel):
    """Forbid unrecognized or coercible values in TMDB cache payloads."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _MovieSearchCandidateCacheModel(_TMDBCacheModel):
    """Contain one cached normalized Movie Search Candidate."""

    tmdb_id: StrictInt
    title: StrictStr
    original_title: StrictStr
    release_year: StrictInt | None


class _MovieSearchCacheModel(_TMDBCacheModel):
    """Contain one bounded cached Movie Search operation."""

    candidates: list[_MovieSearchCandidateCacheModel] = Field(max_length=3)


class _TVSeriesSearchCandidateCacheModel(_TMDBCacheModel):
    """Contain one cached normalized TV Series Search Candidate."""

    tmdb_id: StrictInt
    name: StrictStr
    original_name: StrictStr
    first_air_year: StrictInt | None


class _TVSeriesSearchCacheModel(_TMDBCacheModel):
    """Contain one bounded cached TV Series Search operation."""

    candidates: list[_TVSeriesSearchCandidateCacheModel] = Field(max_length=3)


class _MovieDetailCacheModel(_TMDBCacheModel):
    """Contain one cached normalized Movie detail operation."""

    tmdb_id: StrictInt
    title: StrictStr
    release_year: StrictInt | None
    alternative_titles: list[StrictStr]
    cast: list[StrictStr] = Field(max_length=5)
    directors: list[StrictStr]
    description: StrictStr
    poster_url: StrictStr | None
    imdb_id: StrictStr | None
    tmdb_score: StrictFloat = Field(ge=0, le=10)


class _TVSeriesDetailCacheModel(_TMDBCacheModel):
    """Contain one cached normalized TV Series detail operation."""

    tmdb_id: StrictInt
    name: StrictStr
    first_air_year: StrictInt | None
    last_air_year: StrictInt | None
    alternative_titles: list[StrictStr]
    cast: list[StrictStr] = Field(max_length=5)
    creators: list[StrictStr]
    description: StrictStr
    poster_url: StrictStr | None
    imdb_id: StrictStr | None
    tmdb_score: StrictFloat = Field(ge=0, le=10)


class _MovieSearchCacheCodec:
    """Encode bounded normalized Movie Search Candidates."""

    version = "movie-search-v1"

    def encode(self, value: tuple[_MovieSearchCandidate, ...]) -> JsonObject:
        """Encode normalized Movie Search Candidates."""
        return cast(
            JsonObject,
            _MovieSearchCacheModel(
                candidates=[
                    _MovieSearchCandidateCacheModel(
                        tmdb_id=candidate.tmdb_id,
                        title=candidate.title,
                        original_title=candidate.original_title,
                        release_year=candidate.release_year,
                    )
                    for candidate in value
                ]
            ).model_dump(mode="json"),
        )

    def decode(self, payload: JsonObject) -> tuple[_MovieSearchCandidate, ...]:
        """Decode strict normalized Movie Search Candidates.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _MovieSearchCacheModel.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached TMDB Movie Search value") from exc
        return tuple(
            _MovieSearchCandidate(
                tmdb_id=candidate.tmdb_id,
                title=candidate.title,
                original_title=candidate.original_title,
                release_year=candidate.release_year,
            )
            for candidate in cached_value.candidates
        )


class _TVSeriesSearchCacheCodec:
    """Encode bounded normalized TV Series Search Candidates."""

    version = "tv-series-search-v1"

    def encode(self, value: tuple[_TVSeriesSearchCandidate, ...]) -> JsonObject:
        """Encode normalized TV Series Search Candidates."""
        return cast(
            JsonObject,
            _TVSeriesSearchCacheModel(
                candidates=[
                    _TVSeriesSearchCandidateCacheModel(
                        tmdb_id=candidate.tmdb_id,
                        name=candidate.name,
                        original_name=candidate.original_name,
                        first_air_year=candidate.first_air_year,
                    )
                    for candidate in value
                ]
            ).model_dump(mode="json"),
        )

    def decode(self, payload: JsonObject) -> tuple[_TVSeriesSearchCandidate, ...]:
        """Decode strict normalized TV Series Search Candidates.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _TVSeriesSearchCacheModel.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached TMDB TV Series Search value") from exc
        return tuple(
            _TVSeriesSearchCandidate(
                tmdb_id=candidate.tmdb_id,
                name=candidate.name,
                original_name=candidate.original_name,
                first_air_year=candidate.first_air_year,
            )
            for candidate in cached_value.candidates
        )


class _MovieDetailCacheCodec:
    """Encode normalized Movie detail Candidates."""

    version = "movie-detail-v1"

    def __init__(self, *, require_release_year: bool) -> None:
        """Set whether this operation requires a normalized release year."""
        self._require_release_year = require_release_year

    def encode(self, value: _MovieDetailCandidate) -> JsonObject:
        """Encode a normalized Movie detail Candidate."""
        if self._require_release_year and value.release_year is None:
            raise CacheCodecError("Required TMDB Movie detail lacks a release year")

        return cast(
            JsonObject,
            _MovieDetailCacheModel(
                tmdb_id=value.tmdb_id,
                title=value.title,
                release_year=value.release_year,
                alternative_titles=list(value.alternative_titles),
                cast=list(value.cast),
                directors=list(value.directors),
                description=value.description,
                poster_url=value.poster_url,
                imdb_id=value.imdb_id,
                tmdb_score=value.tmdb_score,
            ).model_dump(mode="json"),
        )

    def decode(self, payload: JsonObject) -> _MovieDetailCandidate:
        """Decode a strict normalized Movie detail Candidate.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _MovieDetailCacheModel.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached TMDB Movie detail value") from exc
        if self._require_release_year and cached_value.release_year is None:
            raise CacheCodecError("Required cached TMDB Movie detail lacks a release year")

        return _MovieDetailCandidate(
            tmdb_id=cached_value.tmdb_id,
            title=cached_value.title,
            release_year=cached_value.release_year,
            alternative_titles=tuple(cached_value.alternative_titles),
            cast=tuple(cached_value.cast),
            directors=tuple(cached_value.directors),
            description=cached_value.description,
            poster_url=cached_value.poster_url,
            imdb_id=cached_value.imdb_id,
            tmdb_score=cached_value.tmdb_score,
        )


class _TVSeriesDetailCacheCodec:
    """Encode normalized TV Series detail Candidates."""

    version = "tv-series-detail-v1"

    def encode(self, value: _TVSeriesDetailCandidate) -> JsonObject:
        """Encode a normalized TV Series detail Candidate."""
        return cast(
            JsonObject,
            _TVSeriesDetailCacheModel(
                tmdb_id=value.tmdb_id,
                name=value.name,
                first_air_year=value.first_air_year,
                last_air_year=value.last_air_year,
                alternative_titles=list(value.alternative_titles),
                cast=list(value.cast),
                creators=list(value.creators),
                description=value.description,
                poster_url=value.poster_url,
                imdb_id=value.imdb_id,
                tmdb_score=value.tmdb_score,
            ).model_dump(mode="json"),
        )

    def decode(self, payload: JsonObject) -> _TVSeriesDetailCandidate:
        """Decode a strict normalized TV Series detail Candidate.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _TVSeriesDetailCacheModel.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached TMDB TV Series detail value") from exc
        return _TVSeriesDetailCandidate(
            tmdb_id=cached_value.tmdb_id,
            name=cached_value.name,
            first_air_year=cached_value.first_air_year,
            last_air_year=cached_value.last_air_year,
            alternative_titles=tuple(cached_value.alternative_titles),
            cast=tuple(cached_value.cast),
            creators=tuple(cached_value.creators),
            description=cached_value.description,
            poster_url=cached_value.poster_url,
            imdb_id=cached_value.imdb_id,
            tmdb_score=cached_value.tmdb_score,
        )


_MOVIE_SEARCH_CACHE_CODEC = _MovieSearchCacheCodec()
_TV_SERIES_SEARCH_CACHE_CODEC = _TVSeriesSearchCacheCodec()
_MOVIE_REQUIRED_DETAIL_CACHE_CODEC = _MovieDetailCacheCodec(require_release_year=True)
_MOVIE_OPTIONAL_DETAIL_CACHE_CODEC = _MovieDetailCacheCodec(require_release_year=False)
_TV_SERIES_DETAIL_CACHE_CODEC = _TVSeriesDetailCacheCodec()


def _parse_movie_search(value: object) -> tuple[_MovieSearchCandidate, ...]:
    """Validate and normalize one TMDB Movie Search response."""
    response = _MovieSearchResponseModel.model_validate(value)
    return tuple(
        _MovieSearchCandidate(
            tmdb_id=candidate.id,
            title=candidate.title,
            original_title=candidate.original_title,
            release_year=_year_from_date(candidate.release_date),
        )
        for candidate in response.results[:_DIRECT_SEARCH_CANDIDATE_LIMIT]
    )


def _parse_tv_series_search(value: object) -> tuple[_TVSeriesSearchCandidate, ...]:
    """Validate and normalize one TMDB TV Series Search response."""
    response = _TVSeriesSearchResponseModel.model_validate(value)
    return tuple(
        _TVSeriesSearchCandidate(
            tmdb_id=candidate.id,
            name=candidate.name,
            original_name=candidate.original_name,
            first_air_year=_year_from_date(candidate.first_air_date),
        )
        for candidate in response.results[:_DIRECT_SEARCH_CANDIDATE_LIMIT]
    )


def _parse_required_movie_detail(
    value: object,
    image_base_url: str,
) -> _MovieDetailCandidate:
    """Validate and normalize Movie details requiring a release date."""
    detail = _RequiredMovieDetailModel.model_validate(value)
    return _to_movie_detail_candidate(
        detail,
        detail.release_date.year,
        tuple(title.title for title in detail.alternative_titles.titles),
        image_base_url,
    )


def _parse_optional_movie_detail(
    value: object,
    image_base_url: str,
) -> _MovieDetailCandidate:
    """Validate and normalize Movie details whose release date may be unavailable."""
    detail = _OptionalMovieDetailModel.model_validate(value)
    return _to_movie_detail_candidate(
        detail,
        detail.release_date.year if detail.release_date is not None else None,
        (),
        image_base_url,
    )


def _to_movie_detail_candidate(
    detail: _MovieDetailModel,
    release_year: int | None,
    alternative_titles: tuple[str, ...],
    image_base_url: str,
) -> _MovieDetailCandidate:
    """Translate validated Movie provider data into an immutable Candidate."""
    return _MovieDetailCandidate(
        tmdb_id=detail.id,
        title=detail.title,
        release_year=release_year,
        alternative_titles=alternative_titles,
        cast=tuple(member.name for member in detail.credits.cast[:5]),
        directors=tuple(
            dict.fromkeys(member.name for member in detail.credits.crew if member.job == "Director")
        ),
        description=detail.overview,
        poster_url=_normalized_poster_url(image_base_url, detail.poster_path),
        imdb_id=_normalized_imdb_id(detail.imdb_id),
        tmdb_score=detail.vote_average,
    )


def _parse_tv_series_detail(
    value: object,
    image_base_url: str,
) -> _TVSeriesDetailCandidate:
    """Validate and normalize one TMDB TV Series detail response."""
    detail = _TVSeriesDetailModel.model_validate(value)
    return _TVSeriesDetailCandidate(
        tmdb_id=detail.id,
        name=detail.name,
        first_air_year=(detail.first_air_date.year if detail.first_air_date is not None else None),
        last_air_year=(
            _year_from_date(detail.last_air_date or "")
            if detail.status in {"Ended", "Canceled"}
            else None
        ),
        alternative_titles=tuple(title.title for title in detail.alternative_titles.titles),
        cast=tuple(member.name for member in detail.aggregate_credits.cast[:5]),
        creators=tuple(dict.fromkeys(creator.name for creator in detail.created_by)),
        description=detail.overview,
        poster_url=_normalized_poster_url(image_base_url, detail.poster_path),
        imdb_id=_normalized_imdb_id(detail.external_ids.imdb_id),
        tmdb_score=detail.vote_average,
    )


def _normalized_poster_url(image_base_url: str, poster_path: str | None) -> str | None:
    """Return the normalized poster URL when TMDB provides a poster path."""
    return f"{image_base_url.rstrip('/')}/{poster_path.lstrip('/')}" if poster_path else None


def _normalized_imdb_id(imdb_id: str | None) -> str | None:
    """Return a stripped TMDB IMDb ID while preserving existing blank semantics."""
    return imdb_id.strip() if imdb_id else None


@dataclass(slots=True)
class _MovieMatch:
    details: _MovieDetailCandidate
    release_year: int


@dataclass(slots=True)
class _TVSeriesMatch:
    details: _TVSeriesDetailCandidate
    first_air_year: int


@dataclass(slots=True)
class _MovieResolutionContext:
    mention: MovieMention
    normalized_title: str
    candidates: list[_MovieSearchCandidate] = field(default_factory=list)
    details_by_candidate_id: dict[int, _MovieDetailCandidate] = field(default_factory=dict)
    match: _MovieMatch | None = None


@dataclass(slots=True)
class _TVResolutionContext:
    mention: TVSeriesMention
    normalized_title: str
    candidates: list[_TVSeriesSearchCandidate] = field(default_factory=list)
    details_by_candidate_id: dict[int, _TVSeriesDetailCandidate] = field(default_factory=dict)
    match: _TVSeriesMatch | None = None


class TMDBScreenWorkResolver:
    """Resolve ordered Movies and TV Series and attach TMDB-backed metadata."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        image_base_url: str,
        cache: AsyncCache,
    ) -> None:
        """Initialize the resolver with a reusable HTTP client and borrowed cache.

        Args:
            client: Lifespan-owned TMDB HTTP client.
            image_base_url: TMDB image URL prefix including the desired size.
            cache: Shared cache borrowed from the application lifespan.
        """
        self._client = client
        self._image_base_url = image_base_url.rstrip("/")
        self._cache = cache

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
        await asyncio.gather(
            *(
                self._resolve_movie_fragment_search(context)
                for context in movie_contexts
                if context.match is None
            ),
            *(
                self._resolve_tv_series_fragment_search(context)
                for context in tv_series_contexts
                if context.match is None
            ),
        )
        return ScreenWorkResults(
            movies=[self._to_movie_result(context) for context in movie_contexts],
            tv_series=[self._to_tv_series_result(context) for context in tv_series_contexts],
        )

    async def aclose(self) -> None:
        """Close the resolver-owned TMDB HTTP client without closing the cache."""
        await self._client.aclose()

    async def _search_movies(
        self,
        query: str,
        year: int | None,
    ) -> tuple[_MovieSearchCandidate, ...]:
        params: dict[str, str | int | bool] = {
            "query": query,
            "include_adult": True,
            "language": "en-US",
            "page": 1,
        }
        if year is not None:
            params["year"] = year
        return await self._get_cached_value(
            "search/movie",
            params,
            self._cache_entry(
                "search/movie",
                params,
                "movie-search",
                _MOVIE_SEARCH_CACHE_CODEC,
                _search_ttl_seconds,
            ),
            _parse_movie_search,
        )

    async def _search_tv_series(
        self,
        query: str,
        year: int | None,
    ) -> tuple[_TVSeriesSearchCandidate, ...]:
        params: dict[str, str | int | bool] = {
            "query": query,
            "include_adult": True,
            "language": "en-US",
            "page": 1,
        }
        if year is not None:
            params["first_air_date_year"] = year
        return await self._get_cached_value(
            "search/tv",
            params,
            self._cache_entry(
                "search/tv",
                params,
                "tv-series-search",
                _TV_SERIES_SEARCH_CACHE_CODEC,
                _search_ttl_seconds,
            ),
            _parse_tv_series_search,
        )

    async def _get_movie_details(
        self,
        tmdb_id: int,
        include_alternative_titles: bool,
        require_release_date: bool,
    ) -> _MovieDetailCandidate:
        append_to_response = (
            "credits,alternative_titles" if include_alternative_titles else "credits"
        )
        path = f"movie/{tmdb_id}"
        params: dict[str, str | int | bool] = {
            "append_to_response": append_to_response,
            "language": "en-US",
        }
        if require_release_date:
            return await self._get_cached_value(
                path,
                params,
                self._cache_entry(
                    path,
                    params,
                    "movie-detail-required-release-date",
                    _MOVIE_REQUIRED_DETAIL_CACHE_CODEC,
                    _detail_ttl_seconds,
                    include_image_base_url=True,
                ),
                lambda value: _parse_required_movie_detail(value, self._image_base_url),
            )
        return await self._get_cached_value(
            path,
            params,
            self._cache_entry(
                path,
                params,
                "movie-detail-optional-release-date",
                _MOVIE_OPTIONAL_DETAIL_CACHE_CODEC,
                _detail_ttl_seconds,
                include_image_base_url=True,
            ),
            lambda value: _parse_optional_movie_detail(value, self._image_base_url),
        )

    async def _get_tv_series_details(
        self,
        tmdb_id: int,
        include_alternative_titles: bool,
    ) -> _TVSeriesDetailCandidate:
        append_to_response = (
            "aggregate_credits,alternative_titles,external_ids"
            if include_alternative_titles
            else "aggregate_credits,external_ids"
        )
        path = f"tv/{tmdb_id}"
        params: dict[str, str | int | bool] = {
            "append_to_response": append_to_response,
            "language": "en-US",
        }
        return await self._get_cached_value(
            path,
            params,
            self._cache_entry(
                path,
                params,
                "tv-series-detail",
                _TV_SERIES_DETAIL_CACHE_CODEC,
                _detail_ttl_seconds,
                include_image_base_url=True,
            ),
            lambda value: _parse_tv_series_detail(value, self._image_base_url),
        )

    def _cache_entry[ValueT](
        self,
        path: str,
        params: dict[str, str | int | bool],
        value_kind: str,
        codec: CacheCodec[ValueT],
        ttl_seconds: Callable[[ValueT], int],
        *,
        include_image_base_url: bool = False,
    ) -> CacheEntry[ValueT]:
        identity_parameters: JsonObject = dict(params)
        identity: JsonObject = {
            "path": path,
            "params": identity_parameters,
            "value_kind": value_kind,
            "adapter_contract_version": _TMDB_ADAPTER_CONTRACT_VERSION,
        }
        if include_image_base_url:
            identity["image_base_url"] = self._image_base_url
        return CacheEntry(
            layer="provider:tmdb",
            key_version="v1",
            identity=identity,
            codec=codec,
            ttl_seconds=ttl_seconds,
            wait_timeout_seconds=1.0,
        )

    async def _get_cached_value[ValueT](
        self,
        path: str,
        params: dict[str, str | int | bool],
        entry: CacheEntry[ValueT],
        parser: Callable[[object], ValueT],
    ) -> ValueT:
        return await self._cache.get_or_load(
            entry,
            lambda: self._load_provider_value(path, params, parser),
        )

    async def _load_provider_value[ValueT](
        self,
        path: str,
        params: dict[str, str | int | bool],
        parser: Callable[[object], ValueT],
    ) -> ValueT:
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
            return parser(cast(object, response.json()))
        except (ValidationError, ValueError) as exc:
            logger.error(
                "TMDB response validation failed",
                extra={"stage": _STAGE, "reason": "invalid_provider_response"},
            )
            raise EnrichmentError(_ENRICHMENT_ERROR_MESSAGE) from exc

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
        candidates = await self._search_movies(context.mention.title, search_year)
        context.candidates.extend(candidates)

        for candidate in candidates:
            if candidate.release_year is None or candidate.release_year != search_year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == context.normalized_title
                for title in (candidate.title, candidate.original_title)
            )
            movie = await self._get_movie_details(
                candidate.tmdb_id,
                include_alternative_titles=not primary_titles_matched,
                require_release_date=True,
            )
            context.details_by_candidate_id[candidate.tmdb_id] = movie

            if not primary_titles_matched and not any(
                normalize_screen_work_title(alternative_title) == context.normalized_title
                for alternative_title in movie.alternative_titles
            ):
                continue

            if movie.release_year is None or abs(movie.release_year - context.mention.year) > 1:
                continue

            return _MovieMatch(details=movie, release_year=movie.release_year)

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
        candidates = await self._search_tv_series(context.mention.title, search_year)
        context.candidates.extend(candidates)

        for candidate in candidates:
            if candidate.first_air_year is None or candidate.first_air_year != search_year:
                continue

            primary_titles_matched = any(
                normalize_screen_work_title(title) == context.normalized_title
                for title in (candidate.name, candidate.original_name)
            )
            tv_series = await self._get_tv_series_details(
                candidate.tmdb_id,
                include_alternative_titles=not primary_titles_matched,
            )
            context.details_by_candidate_id[candidate.tmdb_id] = tv_series

            if not primary_titles_matched and not any(
                normalize_screen_work_title(alternative_title) == context.normalized_title
                for alternative_title in tv_series.alternative_titles
            ):
                continue

            return _TVSeriesMatch(
                details=tv_series,
                first_air_year=candidate.first_air_year,
            )

        return None

    async def _resolve_movie_direct_fuzzy(
        self,
        context: _MovieResolutionContext,
    ) -> None:
        context.match = await self._find_movie_fuzzy_match(
            context,
            context.candidates,
        )

    async def _find_movie_fuzzy_match(
        self,
        context: _MovieResolutionContext,
        candidates: list[_MovieSearchCandidate] | tuple[_MovieSearchCandidate, ...],
    ) -> _MovieMatch | None:
        normalized_mention_title = _normalize_fuzzy_screen_work_title(context.mention.title)
        for candidate in candidates:
            if not _has_fuzzy_title_match(
                normalized_mention_title,
                candidate.title,
                candidate.original_title,
            ):
                continue

            movie = context.details_by_candidate_id.get(candidate.tmdb_id)
            if movie is None:
                movie = await self._get_movie_details(
                    candidate.tmdb_id,
                    include_alternative_titles=False,
                    require_release_date=False,
                )
                context.details_by_candidate_id[candidate.tmdb_id] = movie

            if movie.release_year is None or abs(movie.release_year - context.mention.year) > 1:
                continue

            return _MovieMatch(
                details=movie,
                release_year=movie.release_year,
            )

        return None

    async def _resolve_tv_series_direct_fuzzy(
        self,
        context: _TVResolutionContext,
    ) -> None:
        context.match = await self._find_tv_series_fuzzy_match(
            context,
            context.candidates,
        )

    async def _find_tv_series_fuzzy_match(
        self,
        context: _TVResolutionContext,
        candidates: list[_TVSeriesSearchCandidate] | tuple[_TVSeriesSearchCandidate, ...],
    ) -> _TVSeriesMatch | None:
        normalized_mention_title = _normalize_fuzzy_screen_work_title(context.mention.title)
        for candidate in candidates:
            if not _has_fuzzy_title_match(
                normalized_mention_title,
                candidate.name,
                candidate.original_name,
            ):
                continue

            tv_series = context.details_by_candidate_id.get(candidate.tmdb_id)
            if tv_series is None:
                tv_series = await self._get_tv_series_details(
                    candidate.tmdb_id,
                    include_alternative_titles=False,
                )
                context.details_by_candidate_id[candidate.tmdb_id] = tv_series

            if (
                tv_series.first_air_year is None
                or abs(tv_series.first_air_year - context.mention.year) > 1
            ):
                continue

            return _TVSeriesMatch(
                details=tv_series,
                first_air_year=tv_series.first_air_year,
            )

        return None

    async def _resolve_movie_fragment_search(
        self,
        context: _MovieResolutionContext,
    ) -> None:
        for fragment in _screen_work_search_fragments(context.mention.title):
            match = await self._find_movie_fuzzy_match(
                context,
                await self._search_movies(fragment, None),
            )
            if match is not None:
                context.match = match
                return

    async def _resolve_tv_series_fragment_search(
        self,
        context: _TVResolutionContext,
    ) -> None:
        for fragment in _screen_work_search_fragments(context.mention.title):
            match = await self._find_tv_series_fuzzy_match(
                context,
                await self._search_tv_series(fragment, None),
            )
            if match is not None:
                context.match = match
                return

    def _to_movie_result(self, context: _MovieResolutionContext) -> MovieResult:
        if context.match is None:
            return MovieResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=context.mention,
                movie=None,
            )
        movie = context.match.details
        return MovieResult(
            status=ResultStatus.RESOLVED,
            movie_mention=context.mention,
            movie=EnrichedMovie(
                title=movie.title,
                year=context.match.release_year,
                cast=list(movie.cast),
                directors=list(movie.directors),
                description=movie.description,
                poster_url=movie.poster_url,
                tmdb_id=movie.tmdb_id,
                tmdb_url=f"https://www.themoviedb.org/movie/{movie.tmdb_id}",
                imdb_id=movie.imdb_id,
                imdb_url=(
                    f"https://www.imdb.com/title/{movie.imdb_id}/" if movie.imdb_id else None
                ),
                tmdb_score=movie.tmdb_score,
            ),
        )

    def _to_tv_series_result(self, context: _TVResolutionContext) -> TVSeriesResult:
        if context.match is None:
            return TVSeriesResult(
                status=ResultStatus.UNRESOLVED,
                tv_series_mention=context.mention,
                tv_series=None,
            )
        tv_series = context.match.details
        return TVSeriesResult(
            status=ResultStatus.RESOLVED,
            tv_series_mention=context.mention,
            tv_series=EnrichedTVSeries(
                title=tv_series.name,
                first_air_year=context.match.first_air_year,
                last_air_year=tv_series.last_air_year,
                cast=list(tv_series.cast),
                creators=list(tv_series.creators),
                description=tv_series.description,
                poster_url=tv_series.poster_url,
                tmdb_id=tv_series.tmdb_id,
                tmdb_url=f"https://www.themoviedb.org/tv/{tv_series.tmdb_id}",
                imdb_id=tv_series.imdb_id,
                imdb_url=(
                    f"https://www.imdb.com/title/{tv_series.imdb_id}/"
                    if tv_series.imdb_id
                    else None
                ),
                tmdb_score=tv_series.tmdb_score,
            ),
        )


def create_tmdb_screen_work_resolver(
    settings: TMDBConfig,
    cache: AsyncCache,
) -> TMDBScreenWorkResolver:
    """Create a reusable authenticated TMDB Screen Work Resolver.

    Args:
        settings: Validated TMDB credentials, endpoints, and timeout.
        cache: Shared cache borrowed from the application lifespan.

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
    return TMDBScreenWorkResolver(client, settings.image_base_url, cache)


def _search_ttl_seconds(value: tuple[object, ...]) -> int:
    """Return the positive or empty TMDB Search freshness contract."""
    return _POSITIVE_SEARCH_TTL_SECONDS if value else _EMPTY_SEARCH_TTL_SECONDS


def _detail_ttl_seconds(value: object) -> int:
    """Return the fixed TMDB detail freshness contract."""
    del value
    return _DETAIL_TTL_SECONDS


def _year_from_date(release_date: str) -> int | None:
    year = release_date[:4]
    return int(year) if len(year) == 4 and year.isdigit() else None
