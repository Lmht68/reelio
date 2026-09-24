"""Spotify Client Credentials catalog adapter."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Annotated, Literal, NoReturn, Self, cast

import httpx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    field_validator,
)

from reelio.cache import (
    AsyncCache,
    CacheCodec,
    CacheCodecError,
    CacheSkip,
    CacheWrite,
    RetainedCacheValue,
    RevalidatingCacheEntry,
)
from reelio.cache.interface import JsonObject
from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.catalog.types import (
    AlbumCandidate,
    ImageCandidate,
    TrackCandidate,
)
from reelio.extraction.types import AlbumType, ArtistCredit

logger = logging.getLogger(__name__)

_CATALOG_ERROR_MESSAGE = "Spotify catalog request failed."
_CATALOG_TIMEOUT_MESSAGE = "Spotify catalog request timed out."
_SEARCH_LIMIT = 3
_EMPTY_SEARCH_FRESHNESS_CAP_SECONDS = 900
_SEARCH_PHYSICAL_RETENTION_SECONDS = 21_600
_SEARCH_ADAPTER_CONTRACT_VERSION = "spotify-search-v1"
_STAGE = "spotify_catalog"
_NON_BLANK_TEXT = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_RELEASE_YEAR_PATTERN = re.compile(r"^[0-9]{4}$")
_RELEASE_MONTH_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}$")
_RELEASE_DAY_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def _validate_release_date(release_date: str) -> str:
    """Require Spotify's supported calendar-valid release-date formats."""
    if _RELEASE_YEAR_PATTERN.fullmatch(release_date) is not None:
        return release_date
    if _RELEASE_MONTH_PATTERN.fullmatch(release_date) is not None:
        date_value = f"{release_date}-01"
    elif _RELEASE_DAY_PATTERN.fullmatch(release_date) is not None:
        date_value = release_date
    else:
        raise ValueError("release_date must use YYYY, YYYY-MM, or YYYY-MM-DD")

    try:
        date.fromisoformat(date_value)
    except ValueError as exc:
        raise ValueError("release_date is not a valid calendar date") from exc
    return release_date


def _validate_cached_nonblank_text(value: str) -> str:
    """Reject cache fields that would create an invalid normalized Candidate."""
    if not value.strip():
        raise ValueError("Cached text must not be blank")
    return value


type _CachedNonBlankText = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_cached_nonblank_text),
]


class _SpotifyModel(BaseModel):
    """Base model for private Spotify response DTOs."""

    model_config = ConfigDict(extra="ignore")


class _SpotifyCacheModel(BaseModel):
    """Forbid unrecognized or coercible fields in a Redis cache payload."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _CachedArtistCredit(_SpotifyCacheModel):
    """Contain one normalized Spotify Artist Credit in a cache value."""

    spotify_artist_id: _CachedNonBlankText
    name: _CachedNonBlankText


class _CachedImageCandidate(_SpotifyCacheModel):
    """Contain one normalized image without provider response fields."""

    url: HttpUrl
    width: int | None = Field(ge=0)
    height: int | None = Field(ge=0)


class _CachedAlbumCandidate(_SpotifyCacheModel):
    """Contain one normalized Spotify Album Candidate in a cache value."""

    spotify_album_id: _CachedNonBlankText
    spotify_url: HttpUrl
    title: _CachedNonBlankText
    artists: list[_CachedArtistCredit] = Field(min_length=1)
    release_date: _CachedNonBlankText
    album_type: AlbumType
    images: list[_CachedImageCandidate]

    @field_validator("release_date")
    @classmethod
    def _validate_release_date(cls, release_date: str) -> str:
        """Require the same calendar-valid release-date forms as provider parsing."""
        return _validate_release_date(release_date)


class _CachedTrackCandidate(_SpotifyCacheModel):
    """Contain one normalized playable Spotify Track Candidate in a cache value."""

    spotify_track_id: _CachedNonBlankText
    spotify_url: HttpUrl
    title: _CachedNonBlankText
    artists: list[_CachedArtistCredit] = Field(min_length=1)
    album: _CachedAlbumCandidate


class _CachedSearchValue(_SpotifyCacheModel):
    """Contain persisted validator and provider lifetime metadata for one search."""

    etag: str | None
    freshness_lifetime_seconds: int = Field(ge=0)

    @field_validator("etag")
    @classmethod
    def _validate_etag(cls, etag: str | None) -> str | None:
        """Require persisted validators to contain a non-whitespace byte."""
        if etag is not None and not etag.strip():
            raise ValueError("Cached ETag must not be blank")
        return etag


class _CachedTrackSearchValue(_CachedSearchValue):
    """Contain the bounded normalized Track Candidate cache value."""

    candidates: list[_CachedTrackCandidate] = Field(max_length=3)


class _CachedAlbumSearchValue(_CachedSearchValue):
    """Contain the bounded normalized Album Candidate cache value."""

    candidates: list[_CachedAlbumCandidate] = Field(max_length=3)


class _SpotifyExternalUrls(_SpotifyModel):
    """Validate Spotify direct links needed by application candidates."""

    spotify: HttpUrl


class _SpotifyArtist(_SpotifyModel):
    """Validate one provider artist credit."""

    id: _NON_BLANK_TEXT
    name: _NON_BLANK_TEXT


class _SpotifyImage(_SpotifyModel):
    """Validate one provider-hosted album image."""

    url: HttpUrl
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)


class _SpotifyAlbum(_SpotifyModel):
    """Validate the shared Album shape in Spotify Track and Album responses."""

    id: _NON_BLANK_TEXT
    name: _NON_BLANK_TEXT
    artists: list[_SpotifyArtist] = Field(min_length=1)
    external_urls: _SpotifyExternalUrls
    release_date: str = Field(min_length=1)
    album_type: AlbumType
    images: list[_SpotifyImage] = Field(default_factory=list)

    @field_validator("release_date")
    @classmethod
    def _validate_release_date(cls, release_date: str) -> str:
        """Require Spotify's supported release-date formats."""
        return _validate_release_date(release_date)


class _SpotifyTrack(_SpotifyModel):
    """Validate one Spotify Track returned for the requested market."""

    id: _NON_BLANK_TEXT
    name: _NON_BLANK_TEXT
    artists: list[_SpotifyArtist] = Field(min_length=1)
    external_urls: _SpotifyExternalUrls
    album: _SpotifyAlbum


class _SpotifyPaging[ItemType: BaseModel](_SpotifyModel):
    """Validate the ordered Spotify search items container."""

    items: list[ItemType]


class _TrackSearchResponse(_SpotifyModel):
    """Validate the Track search payload required by the adapter."""

    tracks: _SpotifyPaging[_SpotifyTrack]


class _AlbumSearchResponse(_SpotifyModel):
    """Validate the Album search payload required by the adapter."""

    albums: _SpotifyPaging[_SpotifyAlbum]


class _TokenResponse(_SpotifyModel):
    """Validate a non-empty Client Credentials access token response."""

    access_token: _NON_BLANK_TEXT
    expires_in: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class _SpotifySearchCacheValue[CandidateT]:
    """Contain normalized search Candidates and provider revalidation metadata."""

    candidates: tuple[CandidateT, ...]
    etag: str | None
    freshness_lifetime_seconds: int | None


class _TrackSearchCacheCodec:
    """Encode and decode normalized Spotify Track Search cache values."""

    version = "spotify-track-search-v1"

    def encode(self, value: _SpotifySearchCacheValue[TrackCandidate]) -> JsonObject:
        """Encode a validated normalized Track Search result for retention.

        Raises:
            CacheCodecError: If no provider lifetime is available for persistence.
        """
        freshness_lifetime_seconds = value.freshness_lifetime_seconds
        if freshness_lifetime_seconds is None:
            raise CacheCodecError("Spotify cache values require a provider freshness lifetime")
        return _strict_cache_encode(
            _CachedTrackSearchValue,
            {
                "candidates": [
                    _encode_track_candidate(candidate) for candidate in value.candidates
                ],
                "etag": value.etag,
                "freshness_lifetime_seconds": freshness_lifetime_seconds,
            },
        )

    def decode(self, payload: JsonObject) -> _SpotifySearchCacheValue[TrackCandidate]:
        """Decode one strict normalized Track Search cache value.

        Raises:
            CacheCodecError: If the cached value is malformed or incompatible.
        """
        try:
            cached_value = _CachedTrackSearchValue.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached Spotify Track Search value") from exc
        return _SpotifySearchCacheValue(
            candidates=tuple(
                _decode_track_candidate(candidate) for candidate in cached_value.candidates
            ),
            etag=cached_value.etag,
            freshness_lifetime_seconds=cached_value.freshness_lifetime_seconds,
        )


class _AlbumSearchCacheCodec:
    """Encode and decode normalized Spotify Album Search cache values."""

    version = "spotify-album-search-v1"

    def encode(self, value: _SpotifySearchCacheValue[AlbumCandidate]) -> JsonObject:
        """Encode a validated normalized Album Search result for retention.

        Raises:
            CacheCodecError: If no provider lifetime is available for persistence.
        """
        freshness_lifetime_seconds = value.freshness_lifetime_seconds
        if freshness_lifetime_seconds is None:
            raise CacheCodecError("Spotify cache values require a provider freshness lifetime")
        return _strict_cache_encode(
            _CachedAlbumSearchValue,
            {
                "candidates": [
                    _encode_album_candidate(candidate) for candidate in value.candidates
                ],
                "etag": value.etag,
                "freshness_lifetime_seconds": freshness_lifetime_seconds,
            },
        )

    def decode(self, payload: JsonObject) -> _SpotifySearchCacheValue[AlbumCandidate]:
        """Decode one strict normalized Album Search cache value.

        Raises:
            CacheCodecError: If the cached value is malformed or incompatible.
        """
        try:
            cached_value = _CachedAlbumSearchValue.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached Spotify Album Search value") from exc
        return _SpotifySearchCacheValue(
            candidates=tuple(
                _decode_album_candidate(candidate) for candidate in cached_value.candidates
            ),
            etag=cached_value.etag,
            freshness_lifetime_seconds=cached_value.freshness_lifetime_seconds,
        )


_TRACK_SEARCH_CACHE_CODEC = _TrackSearchCacheCodec()
_ALBUM_SEARCH_CACHE_CODEC = _AlbumSearchCacheCodec()


def _strict_cache_encode(
    model_type: type[_SpotifyCacheModel],
    payload: object,
) -> JsonObject:
    """Validate and serialize one normalized cache payload without provider DTO fields.

    Raises:
        CacheCodecError: If the normalized value violates the persisted cache contract.
    """
    try:
        cached_value = model_type.model_validate(payload)
    except ValidationError as exc:
        raise CacheCodecError("Invalid normalized Spotify cache value") from exc
    return cast(JsonObject, cached_value.model_dump(mode="json"))


def _encode_artist_credit(artist: ArtistCredit) -> dict[str, object]:
    """Serialize one normalized Artist Credit."""
    return {
        "spotify_artist_id": artist.spotify_artist_id,
        "name": artist.name,
    }


def _encode_image_candidate(image: ImageCandidate) -> dict[str, object]:
    """Serialize one normalized image."""
    return {
        "url": image.url,
        "width": image.width,
        "height": image.height,
    }


def _encode_album_candidate(album: AlbumCandidate) -> dict[str, object]:
    """Serialize one normalized Album Candidate."""
    return {
        "spotify_album_id": album.spotify_album_id,
        "spotify_url": album.spotify_url,
        "title": album.title,
        "artists": [_encode_artist_credit(artist) for artist in album.artists],
        "release_date": album.release_date,
        "album_type": album.album_type,
        "images": [_encode_image_candidate(image) for image in album.images],
    }


def _encode_track_candidate(track: TrackCandidate) -> dict[str, object]:
    """Serialize one normalized playable Track Candidate."""
    return {
        "spotify_track_id": track.spotify_track_id,
        "spotify_url": track.spotify_url,
        "title": track.title,
        "artists": [_encode_artist_credit(artist) for artist in track.artists],
        "album": _encode_album_candidate(track.album),
    }


def _decode_artist_credit(artist: _CachedArtistCredit) -> ArtistCredit:
    """Reconstruct one immutable normalized Artist Credit."""
    return ArtistCredit(
        spotify_artist_id=artist.spotify_artist_id,
        name=artist.name,
    )


def _decode_image_candidate(image: _CachedImageCandidate) -> ImageCandidate:
    """Reconstruct one immutable normalized image."""
    return ImageCandidate(
        url=str(image.url),
        width=image.width,
        height=image.height,
    )


def _decode_album_candidate(album: _CachedAlbumCandidate) -> AlbumCandidate:
    """Reconstruct one immutable normalized Album Candidate."""
    return AlbumCandidate(
        spotify_album_id=album.spotify_album_id,
        spotify_url=str(album.spotify_url),
        title=album.title,
        artists=tuple(_decode_artist_credit(artist) for artist in album.artists),
        release_date=album.release_date,
        album_type=album.album_type,
        images=tuple(_decode_image_candidate(image) for image in album.images),
    )


def _decode_track_candidate(track: _CachedTrackCandidate) -> TrackCandidate:
    """Reconstruct one immutable normalized playable Track Candidate."""
    return TrackCandidate(
        spotify_track_id=track.spotify_track_id,
        spotify_url=str(track.spotify_url),
        title=track.title,
        artists=tuple(_decode_artist_credit(artist) for artist in track.artists),
        album=_decode_album_candidate(track.album),
    )


@dataclass(frozen=True, slots=True)
class _SearchResponseCachePolicy:
    """Contain parsed provider cache metadata without inventing freshness."""

    state: Literal["omitted", "usable", "unusable"]
    freshness_lifetime_seconds: int | None
    age_seconds: int


@dataclass(frozen=True, slots=True)
class _SpotifySearchRequest[CandidateT]:
    """Contain one cache-loader search operation and its optional retained value."""

    query: str
    market: SpotifyMarket
    item_type: Literal["track", "album"]
    retained_value: RetainedCacheValue[_SpotifySearchCacheValue[CandidateT]] | None
    parse_candidates: Callable[[object], tuple[CandidateT, ...]]


def _spotify_search_cache_entry[CandidateT](
    query: str,
    market: SpotifyMarket,
    item_type: Literal["track", "album"],
    codec: CacheCodec[_SpotifySearchCacheValue[CandidateT]],
) -> RevalidatingCacheEntry[_SpotifySearchCacheValue[CandidateT]]:
    """Describe one exact Spotify Search operation for the shared cache."""
    return RevalidatingCacheEntry(
        layer="provider:spotify",
        key_version="v1",
        identity={
            "operation": "search",
            "query": query,
            "item_type": item_type,
            "market": str(market),
            "adapter_contract_version": _SEARCH_ADAPTER_CONTRACT_VERSION,
        },
        codec=codec,
        wait_timeout_seconds=1.0,
    )


def _parse_track_candidates(payload: object) -> tuple[TrackCandidate, ...]:
    """Validate and normalize up to three ordered Track Search items."""
    response = _TrackSearchResponse.model_validate(payload)
    return tuple(_to_track_candidate(item) for item in response.tracks.items[:_SEARCH_LIMIT])


def _parse_album_candidates(payload: object) -> tuple[AlbumCandidate, ...]:
    """Validate and normalize up to three ordered Album Search items."""
    response = _AlbumSearchResponse.model_validate(payload)
    return tuple(_to_album_candidate(item) for item in response.albums.items[:_SEARCH_LIMIT])


class SpotifyCatalog:
    """Supply typed, market-aware Spotify Track and Album Candidates.

    The adapter owns Client Credentials authentication, token reuse, normalized search
    caching, conditional provider revalidation, and credential-safe error translation.

    Args:
        client: Lifespan-owned HTTP client for Spotify API and token requests.
        settings: Validated Spotify credentials and request settings.
        cache: Lifespan-owned shared cache borrowed for normalized search operations.
        clock: Monotonic clock used for token lifetime and retry deadlines.
        sleep: Awaitable delay used only for a bounded ``Retry-After`` retry.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: SpotifyConfig,
        cache: AsyncCache,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize the lifespan-owned adapter without requesting a token."""
        self._client = client
        self._client_id = settings.client_id
        self._client_secret = settings.client_secret
        self._token_url = settings.token_url
        self._request_timeout_seconds = settings.request_timeout_seconds
        self._token_expiry_skew_seconds = settings.token_expiry_skew_seconds
        self._cache = cache
        self._clock = clock
        self._sleep = sleep
        self._access_token: str | None = None
        self._token_refresh_at = 0.0
        self._token_lock = asyncio.Lock()

    async def search_tracks(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[TrackCandidate, ...]:
        """Search Spotify Tracks in one effective market.

        Args:
            query: Spotify search query constructed by the Track resolver.
            market: Required ISO 3166-1 alpha-2 effective market.

        Returns:
            Up to three provider-ordered typed Track Candidates.

        Raises:
            CatalogProviderError: If Spotify authentication, request, or response validation fails.
            PipelineTimeoutError: If the complete catalog operation exceeds its timeout.
        """
        entry = _spotify_search_cache_entry(query, market, "track", _TRACK_SEARCH_CACHE_CODEC)

        async def load(
            retained_value: RetainedCacheValue[_SpotifySearchCacheValue[TrackCandidate]] | None,
        ) -> (
            CacheWrite[_SpotifySearchCacheValue[TrackCandidate]]
            | CacheSkip[_SpotifySearchCacheValue[TrackCandidate]]
        ):
            return await self._search(
                _SpotifySearchRequest(
                    query=query,
                    market=market,
                    item_type="track",
                    retained_value=retained_value,
                    parse_candidates=_parse_track_candidates,
                )
            )

        cache_value = await self._cache.get_or_load_revalidating(entry, load)
        return cache_value.candidates

    async def search_albums(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[AlbumCandidate, ...]:
        """Search Spotify Albums in one effective market.

        Args:
            query: Spotify search query constructed by the Music Release resolver.
            market: Required ISO 3166-1 alpha-2 effective market.

        Returns:
            Up to three provider-ordered typed Album Candidates.

        Raises:
            CatalogProviderError: If Spotify authentication, request, or response validation fails.
            PipelineTimeoutError: If the complete catalog operation exceeds its timeout.
        """
        entry = _spotify_search_cache_entry(query, market, "album", _ALBUM_SEARCH_CACHE_CODEC)

        async def load(
            retained_value: RetainedCacheValue[_SpotifySearchCacheValue[AlbumCandidate]] | None,
        ) -> (
            CacheWrite[_SpotifySearchCacheValue[AlbumCandidate]]
            | CacheSkip[_SpotifySearchCacheValue[AlbumCandidate]]
        ):
            return await self._search(
                _SpotifySearchRequest(
                    query=query,
                    market=market,
                    item_type="album",
                    retained_value=retained_value,
                    parse_candidates=_parse_album_candidates,
                )
            )

        cache_value = await self._cache.get_or_load_revalidating(entry, load)
        return cache_value.candidates

    async def __aenter__(self) -> Self:
        """Return this lifespan-owned catalog adapter."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the adapter when its managed lifespan exits."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the lifespan-owned HTTP client without closing the shared cache."""
        await self._client.aclose()

    async def _search[CandidateT](
        self,
        search_request: _SpotifySearchRequest[CandidateT],
    ) -> (
        CacheWrite[_SpotifySearchCacheValue[CandidateT]]
        | CacheSkip[_SpotifySearchCacheValue[CandidateT]]
    ):
        """Load or conditionally revalidate one normalized Spotify Search operation."""
        deadline = self._clock() + self._request_timeout_seconds
        try:
            async with asyncio.timeout(self._request_timeout_seconds):
                access_token = await self._get_access_token()
                response = await self._get_with_single_rate_limit_retry(
                    search_request,
                    access_token,
                    deadline,
                )
                if response.status_code == httpx.codes.NOT_MODIFIED:
                    return self._revalidated_cache_result(search_request, response)
                response.raise_for_status()
                candidates = search_request.parse_candidates(cast(object, response.json()))
                policy = _search_response_cache_policy(response)
                etag = _usable_response_etag(response)
                if policy.state != "usable":
                    return _uncacheable_search_result(candidates, etag)
                return _cacheable_search_result(
                    candidates,
                    etag,
                    cast(int, policy.freshness_lifetime_seconds),
                )
        except TimeoutError as exc:
            self._raise_timeout(exc)
        except httpx.TimeoutException as exc:
            self._raise_timeout(exc)
        except httpx.HTTPError as exc:
            self._raise_provider_error("http_failure", exc)
        except (ValidationError, ValueError) as exc:
            self._raise_provider_error("invalid_provider_response", exc)

    def _revalidated_cache_result[CandidateT](
        self,
        search_request: _SpotifySearchRequest[CandidateT],
        response: httpx.Response,
    ) -> (
        CacheWrite[_SpotifySearchCacheValue[CandidateT]]
        | CacheSkip[_SpotifySearchCacheValue[CandidateT]]
    ):
        """Rebuild cache metadata after a representation-validating 304 response."""
        retained_value = search_request.retained_value
        if retained_value is None or retained_value.value.etag is None:
            raise ValueError("Spotify returned an unsolicited Not Modified response")
        etag = _matching_304_etag(response, retained_value.value.etag)
        policy = _search_response_cache_policy(response)
        if policy.state == "unusable":
            return _uncacheable_search_result(retained_value.value.candidates, etag)
        if policy.state == "usable":
            return _cacheable_search_result(
                retained_value.value.candidates,
                etag,
                cast(int, policy.freshness_lifetime_seconds),
            )

        previous_lifetime = retained_value.value.freshness_lifetime_seconds
        if previous_lifetime is None:
            raise ValueError("Retained Spotify value omitted its provider lifetime")
        return _cacheable_search_result(
            retained_value.value.candidates,
            etag,
            max(0, previous_lifetime - policy.age_seconds),
        )

    async def _get_access_token(self) -> str:
        async with self._token_lock:
            if self._access_token is not None and self._clock() < self._token_refresh_at:
                return self._access_token

            response = await self._client.post(
                self._token_url,
                data={"grant_type": "client_credentials"},
                auth=httpx.BasicAuth(
                    self._client_id.get_secret_value(),
                    self._client_secret.get_secret_value(),
                ),
            )
            response.raise_for_status()
            token = self._validate_response(cast(object, response.json()), _TokenResponse)
            self._access_token = token.access_token
            safe_lifetime = max(0.0, token.expires_in - self._token_expiry_skew_seconds)
            self._token_refresh_at = self._clock() + safe_lifetime
            return self._access_token

    async def _get_with_single_rate_limit_retry[CandidateT](
        self,
        search_request: _SpotifySearchRequest[CandidateT],
        access_token: str,
        deadline: float,
    ) -> httpx.Response:
        """Request one Search operation and preserve validation headers through one retry."""
        request_params: dict[str, str | int] = {
            "q": search_request.query,
            "type": search_request.item_type,
            "market": search_request.market,
            "offset": 0,
            "limit": _SEARCH_LIMIT,
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        retained_value = search_request.retained_value
        if retained_value is not None and retained_value.value.etag is not None:
            headers["If-None-Match"] = retained_value.value.etag
        response = await self._client.get(
            "search",
            params=request_params,
            headers=headers,
        )
        if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
            return response

        retry_after = _retry_after_seconds(response)
        if retry_after > deadline - self._clock():
            self._raise_provider_error("rate_limit_exceeds_timeout")

        await self._sleep(retry_after)
        response = await self._client.get(
            "search",
            params=request_params,
            headers=headers,
        )
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            self._raise_provider_error("rate_limit_retry_exhausted")
        return response

    def _validate_response[ModelType: BaseModel](
        self,
        payload: object,
        model_type: type[ModelType],
    ) -> ModelType:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            self._raise_provider_error("invalid_provider_response", exc)

    def _raise_timeout(self, exc: Exception) -> NoReturn:
        logger.error(
            "Spotify catalog request timed out",
            extra={"stage": _STAGE, "reason": "provider_timeout"},
        )
        raise PipelineTimeoutError(_CATALOG_TIMEOUT_MESSAGE) from exc

    def _raise_provider_error(self, reason: str, exc: Exception | None = None) -> NoReturn:
        logger.error(
            "Spotify catalog request failed",
            extra={"stage": _STAGE, "reason": reason},
        )
        if exc is None:
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE)
        raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc


def create_spotify_catalog(settings: SpotifyConfig, cache: AsyncCache) -> SpotifyCatalog:
    """Create a lifespan-owned Spotify Catalog adapter.

    Args:
        settings: Validated Spotify credentials and request settings.
        cache: Lifespan-owned shared cache borrowed for normalized Spotify Search.

    Returns:
        SpotifyCatalog backed by one reusable asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url}/",
        headers={"accept": "application/json"},
        timeout=settings.request_timeout_seconds,
    )
    return SpotifyCatalog(client, settings, cache)


def _uncacheable_search_result[CandidateT](
    candidates: tuple[CandidateT, ...],
    etag: str | None,
) -> CacheSkip[_SpotifySearchCacheValue[CandidateT]]:
    """Return a successful Search result without retaining unsupported freshness metadata."""
    return CacheSkip(
        _SpotifySearchCacheValue(
            candidates=candidates,
            etag=etag,
            freshness_lifetime_seconds=None,
        )
    )


def _cacheable_search_result[CandidateT](
    candidates: tuple[CandidateT, ...],
    etag: str | None,
    freshness_lifetime_seconds: int,
) -> (
    CacheWrite[_SpotifySearchCacheValue[CandidateT]]
    | CacheSkip[_SpotifySearchCacheValue[CandidateT]]
):
    """Build serving and physical durations from one usable provider lifetime."""
    serving_freshness_cap = (
        _EMPTY_SEARCH_FRESHNESS_CAP_SECONDS
        if not candidates
        else _SEARCH_PHYSICAL_RETENTION_SECONDS
    )
    serving_freshness_seconds = min(freshness_lifetime_seconds, serving_freshness_cap)
    if serving_freshness_seconds == 0 and etag is None:
        return _uncacheable_search_result(candidates, etag)
    retention_seconds = (
        _SEARCH_PHYSICAL_RETENTION_SECONDS if etag is not None else serving_freshness_seconds
    )
    return CacheWrite(
        _SpotifySearchCacheValue(
            candidates=candidates,
            etag=etag,
            freshness_lifetime_seconds=freshness_lifetime_seconds,
        ),
        freshness_seconds=serving_freshness_seconds,
        retention_seconds=retention_seconds,
    )


def _search_response_cache_policy(response: httpx.Response) -> _SearchResponseCachePolicy:
    """Parse explicit Spotify Cache-Control and Age fields into a storage decision."""
    age_seconds = _response_age_seconds(response)
    if age_seconds is None:
        return _SearchResponseCachePolicy("unusable", None, 0)

    cache_control_fields = response.headers.get_list("cache-control")
    if not cache_control_fields:
        return _SearchResponseCachePolicy("omitted", None, age_seconds)
    directives = [
        _cache_control_directive(member)
        for cache_control_field in cache_control_fields
        for member in _split_cache_control_members(cache_control_field)
    ]
    directive_names = {name for name, _ in directives}
    if "no-store" in directive_names or "private" in directive_names:
        return _SearchResponseCachePolicy("unusable", None, age_seconds)
    if "no-cache" in directive_names:
        return _SearchResponseCachePolicy("usable", 0, age_seconds)

    s_maxage_values = [value for name, value in directives if name == "s-maxage"]
    if s_maxage_values:
        freshness_lifetime_seconds = _single_cache_lifetime(s_maxage_values)
    else:
        max_age_values = [value for name, value in directives if name == "max-age"]
        freshness_lifetime_seconds = _single_cache_lifetime(max_age_values)
    if freshness_lifetime_seconds is None:
        return _SearchResponseCachePolicy("unusable", None, age_seconds)
    return _SearchResponseCachePolicy(
        "usable",
        max(0, freshness_lifetime_seconds - age_seconds),
        age_seconds,
    )


def _response_age_seconds(response: httpx.Response) -> int | None:
    """Return one valid nonnegative Age field, or None when it is malformed."""
    age_fields = response.headers.get_list("age")
    if not age_fields:
        return 0
    if len(age_fields) != 1:
        return None
    age_value = age_fields[0].strip()
    if re.fullmatch(r"[0-9]+", age_value) is None:
        return None
    return int(age_value)


def _split_cache_control_members(cache_control_field: str) -> tuple[str, ...]:
    """Split Cache-Control members on commas outside quoted strings."""
    members: list[str] = []
    member_start = 0
    in_quotes = False
    escaped = False
    for index, character in enumerate(cache_control_field):
        if in_quotes and escaped:
            escaped = False
            continue
        if in_quotes and character == "\\":
            escaped = True
            continue
        if character == '"':
            in_quotes = not in_quotes
            continue
        if character == "," and not in_quotes:
            members.append(cache_control_field[member_start:index])
            member_start = index + 1
    members.append(cache_control_field[member_start:])
    return tuple(members)


def _cache_control_directive(member: str) -> tuple[str, str | None]:
    """Normalize a directive name and retain its exact trimmed optional value."""
    name, separator, value = member.partition("=")
    return name.strip().lower(), value.strip() if separator else None


def _single_cache_lifetime(values: list[str | None]) -> int | None:
    """Return exactly one unquoted nonnegative decimal lifetime, if present."""
    if len(values) != 1:
        return None
    value = values[0]
    if value is None or re.fullmatch(r"[0-9]+", value) is None:
        return None
    return int(value)


def _usable_response_etag(response: httpx.Response) -> str | None:
    """Return one nonblank ETag without stripping or normalizing its header value."""
    etag_fields = response.headers.get_list("etag")
    if len(etag_fields) != 1 or not etag_fields[0].strip():
        return None
    return etag_fields[0]


def _matching_304_etag(response: httpx.Response, retained_etag: str) -> str:
    """Require optional 304 ETag metadata to exactly match the retained validator."""
    etag_fields = response.headers.get_list("etag")
    if not etag_fields:
        return retained_etag
    if len(etag_fields) != 1 or not etag_fields[0].strip() or etag_fields[0] != retained_etag:
        raise ValueError("Spotify 304 ETag does not match retained validator")
    return retained_etag


def _retry_after_seconds(response: httpx.Response) -> float:
    """Parse a non-negative Spotify Retry-After delay in seconds."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        raise ValueError("Spotify rate limit response omitted Retry-After")
    try:
        delay = float(retry_after)
    except ValueError as exc:
        raise ValueError("Spotify Retry-After is not numeric") from exc
    if delay < 0:
        raise ValueError("Spotify Retry-After must not be negative")
    return delay


def _to_track_candidate(track: _SpotifyTrack) -> TrackCandidate:
    """Translate a private Spotify Track DTO into an application candidate."""
    return TrackCandidate(
        spotify_track_id=track.id,
        spotify_url=str(track.external_urls.spotify),
        title=track.name,
        artists=tuple(_to_artist_credit(artist) for artist in track.artists),
        album=_to_album_candidate(track.album),
    )


def _to_album_candidate(album: _SpotifyAlbum) -> AlbumCandidate:
    """Translate a private Spotify Album DTO into an application candidate."""
    return AlbumCandidate(
        spotify_album_id=album.id,
        spotify_url=str(album.external_urls.spotify),
        title=album.name,
        artists=tuple(_to_artist_credit(artist) for artist in album.artists),
        release_date=album.release_date,
        album_type=album.album_type,
        images=tuple(
            ImageCandidate(
                url=str(image.url),
                width=image.width,
                height=image.height,
            )
            for image in album.images
        ),
    )


def _to_artist_credit(artist: _SpotifyArtist) -> ArtistCredit:
    """Translate one private Spotify Artist DTO into an application credit."""
    return ArtistCredit(spotify_artist_id=artist.id, name=artist.name)
