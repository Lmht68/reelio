"""Spotify Client Credentials catalog adapter."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import date
from types import TracebackType
from typing import Annotated, Literal, NoReturn, Self, cast

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    model_validator,
)

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.catalog.types import (
    AlbumCandidate,
    ArtistCredit,
    ImageCandidate,
    TrackCandidate,
)

logger = logging.getLogger(__name__)

_CATALOG_ERROR_MESSAGE = "Spotify catalog request failed."
_CATALOG_TIMEOUT_MESSAGE = "Spotify catalog request timed out."
_STAGE = "spotify_catalog"
_NON_BLANK_TEXT = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_RELEASE_YEAR_PATTERN = re.compile(r"^\d{4}$")


class _SpotifyModel(BaseModel):
    """Base model for private Spotify response DTOs."""

    model_config = ConfigDict(extra="ignore")


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
    release_date: _NON_BLANK_TEXT
    release_date_precision: Literal["year", "month", "day"]
    album_type: Literal["album", "single", "compilation"]
    images: list[_SpotifyImage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_release_date(self) -> _SpotifyAlbum:
        """Require a release-date value compatible with its reported precision."""
        if self.release_date_precision == "year":
            is_valid = _RELEASE_YEAR_PATTERN.fullmatch(self.release_date) is not None
        else:
            date_value = (
                self.release_date
                if self.release_date_precision == "day"
                else f"{self.release_date}-01"
            )
            try:
                date.fromisoformat(date_value)
            except ValueError:
                is_valid = False
            else:
                is_valid = True
        if not is_valid:
            raise ValueError("release_date does not match release_date_precision")
        return self


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


class SpotifyCatalog:
    """Supply typed, market-aware Spotify Track and Album Candidates.

    The adapter owns Client Credentials authentication, token reuse, provider request
    construction, response validation, relinking behavior, and credential-safe error
    translation for its lifespan.

    Args:
        client: Lifespan-owned HTTP client for Spotify API and token requests.
        settings: Validated Spotify credentials and request settings.
        clock: Monotonic clock used for token lifetime and retry deadlines.
        sleep: Awaitable delay used only for a bounded ``Retry-After`` retry.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: SpotifyConfig,
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
        payload = await self._search(query, market, "track")
        response = self._validate_response(payload, _TrackSearchResponse)
        return tuple(_to_track_candidate(item) for item in response.tracks.items[:3])

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
        payload = await self._search(query, market, "album")
        response = self._validate_response(payload, _AlbumSearchResponse)
        return tuple(_to_album_candidate(item) for item in response.albums.items[:3])

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
        """Close the lifespan-owned HTTP client and its connection pool."""
        await self._client.aclose()

    async def _search(
        self,
        query: str,
        market: SpotifyMarket,
        item_type: Literal["track", "album"],
    ) -> object:
        deadline = self._clock() + self._request_timeout_seconds
        try:
            async with asyncio.timeout(self._request_timeout_seconds):
                access_token = await self._get_access_token()
                response = await self._get_with_single_rate_limit_retry(
                    query,
                    market,
                    item_type,
                    access_token,
                    deadline,
                )
                response.raise_for_status()
                return cast(object, response.json())
        except TimeoutError as exc:
            self._raise_timeout(exc)
        except httpx.TimeoutException as exc:
            self._raise_timeout(exc)
        except httpx.HTTPError as exc:
            self._raise_provider_error("http_failure", exc)
        except (ValidationError, ValueError) as exc:
            self._raise_provider_error("invalid_provider_response", exc)

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

    async def _get_with_single_rate_limit_retry(
        self,
        query: str,
        market: SpotifyMarket,
        item_type: Literal["track", "album"],
        access_token: str,
        deadline: float,
    ) -> httpx.Response:
        response = await self._client.get(
            "search",
            params={"q": query, "type": item_type, "market": market, "limit": 3},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
            return response

        retry_after = _retry_after_seconds(response)
        if retry_after > deadline - self._clock():
            self._raise_provider_error("rate_limit_exceeds_timeout")

        await self._sleep(retry_after)
        response = await self._client.get(
            "search",
            params={"q": query, "type": item_type, "market": market, "limit": 3},
            headers={"Authorization": f"Bearer {access_token}"},
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


def create_spotify_catalog(settings: SpotifyConfig) -> SpotifyCatalog:
    """Create a lifespan-owned Spotify Catalog adapter.

    Args:
        settings: Validated Spotify credentials and request settings.

    Returns:
        SpotifyCatalog backed by one reusable asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url}/",
        headers={"accept": "application/json"},
        timeout=settings.request_timeout_seconds,
    )
    return SpotifyCatalog(client, settings)


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
        release_date_precision=album.release_date_precision,
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
