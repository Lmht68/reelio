"""Spotify catalog boundary contract tests."""

from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import ValidationError

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.catalog.spotify import SpotifyCatalog

_MARKET = SpotifyMarket("JP")


def _settings(**values: object) -> SpotifyConfig:
    """Build Spotify settings without reading repository environment files."""
    settings_type = cast(Callable[..., SpotifyConfig], SpotifyConfig)
    settings_values: dict[str, object] = {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "base_url": "https://api.spotify.test/v1",
        "token_url": "https://accounts.spotify.test/api/token",
    }
    settings_values.update(values)
    return settings_type(_env_file=None, **settings_values)


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    """Build a mock-transport Spotify client."""
    return httpx.AsyncClient(
        base_url="https://api.spotify.test/v1/",
        transport=handler,
    )


def _track_payload(track_id: str) -> dict[str, object]:
    """Return a valid relinked Spotify Track response payload."""
    return {
        "id": track_id,
        "name": "Kiki's Delivery Service",
        "artists": [{"id": "artist-1", "name": "Yumi Arai"}],
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
        "linked_from": {"id": "original-track"},
        "album": {
            "id": "album-1",
            "name": "Kiki's Delivery Service",
            "artists": [{"id": "artist-1", "name": "Yumi Arai"}],
            "external_urls": {"spotify": "https://open.spotify.com/album/album-1"},
            "release_date": "1989-04-25",
            "release_date_precision": "day",
            "album_type": "album",
            "images": [
                {
                    "url": "https://i.scdn.co/image/cover",
                    "width": 640,
                    "height": 640,
                }
            ],
        },
    }


async def test_catalog_reuses_token_and_returns_playable_track_candidate() -> None:
    """Translate a relinked market Track without exposing provider DTO fields."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "accounts.spotify.test":
            assert request.method == "POST"
            assert request.url.path == "/api/token"
            assert request.content == b"grant_type=client_credentials"
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )

        assert request.method == "GET"
        assert request.url.path == "/v1/search"
        assert request.headers["authorization"] == "Bearer access-token"
        assert dict(request.url.params) == {
            "q": "Kiki's Delivery Service Yumi Arai",
            "type": "track",
            "market": "JP",
            "limit": "3",
        }
        return httpx.Response(
            200,
            json={"tracks": {"items": [_track_payload("playable-track")]}},
        )

    client = _client(httpx.MockTransport(handle))
    catalog = SpotifyCatalog(client, _settings())

    first_candidates = await catalog.search_tracks("Kiki's Delivery Service Yumi Arai", _MARKET)
    second_candidates = await catalog.search_tracks("Kiki's Delivery Service Yumi Arai", _MARKET)

    assert len(first_candidates) == 1
    assert second_candidates == first_candidates
    candidate = first_candidates[0]
    assert candidate.spotify_track_id == "playable-track"
    assert candidate.spotify_url == "https://open.spotify.com/track/playable-track"
    assert candidate.title == "Kiki's Delivery Service"
    assert candidate.artists[0].spotify_artist_id == "artist-1"
    assert candidate.artists[0].name == "Yumi Arai"
    assert candidate.album.spotify_album_id == "album-1"
    assert candidate.album.release_date == "1989-04-25"
    assert candidate.album.release_date_precision == "day"
    assert candidate.album.album_type == "album"
    assert candidate.album.images[0].url == "https://i.scdn.co/image/cover"
    assert not hasattr(candidate, "linked_from")
    assert len([request for request in requests if request.method == "POST"]) == 1
    assert len([request for request in requests if request.method == "GET"]) == 2

    await catalog.aclose()
    assert client.is_closed is True


class _FakeClock:
    """Advance a deterministic monotonic clock through adapter retry delays."""

    def __init__(self) -> None:
        self.value = 0.0
        self.delays: list[float] = []

    def __call__(self) -> float:
        """Return the current deterministic monotonic time."""
        return self.value

    async def sleep(self, delay: float) -> None:
        """Record and advance by one requested retry delay."""
        self.delays.append(delay)
        self.value += delay


async def test_catalog_retries_one_bounded_rate_limit_with_a_fake_clock() -> None:
    """Honor one fitting Spotify Retry-After without wall-clock waiting."""
    search_requests = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal search_requests
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        search_requests += 1
        if search_requests == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json={"tracks": {"items": [_track_payload("playable-track")]}},
        )

    clock = _FakeClock()
    catalog = SpotifyCatalog(
        _client(httpx.MockTransport(handle)),
        _settings(request_timeout_seconds=5.0),
        clock=clock,
        sleep=clock.sleep,
    )

    candidates = await catalog.search_tracks("Kiki", _MARKET)

    assert candidates[0].spotify_track_id == "playable-track"
    assert search_requests == 2
    assert clock.delays == [2.0]
    await catalog.aclose()


@pytest.mark.parametrize("retry_after", ["20", "not-a-delay"])
async def test_catalog_rejects_rate_limits_that_cannot_be_retried(
    retry_after: str,
) -> None:
    """Treat unbounded or malformed rate-limit retries as core-provider failures."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        return httpx.Response(429, headers={"Retry-After": retry_after})

    clock = _FakeClock()
    catalog = SpotifyCatalog(
        _client(httpx.MockTransport(handle)),
        _settings(request_timeout_seconds=5.0),
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(CatalogProviderError, match="Spotify catalog request failed"):
        await catalog.search_tracks("Kiki", _MARKET)

    assert clock.delays == []
    await catalog.aclose()


async def test_catalog_rejects_second_rate_limit_without_returning_candidates() -> None:
    """Treat a second Spotify rate rejection as a typed operational failure."""
    search_requests = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal search_requests
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        search_requests += 1
        return httpx.Response(429, headers={"Retry-After": "1"})

    clock = _FakeClock()
    catalog = SpotifyCatalog(
        _client(httpx.MockTransport(handle)),
        _settings(request_timeout_seconds=5.0),
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(CatalogProviderError, match="Spotify catalog request failed"):
        await catalog.search_tracks("Kiki", _MARKET)

    assert search_requests == 2
    assert clock.delays == [1.0]
    await catalog.aclose()


async def test_catalog_refreshes_token_at_its_safe_pre_expiry_boundary() -> None:
    """Reuse a token before, but not at, the configured safe expiry boundary."""
    token_requests = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests
        if request.url.host == "accounts.spotify.test":
            token_requests += 1
            return httpx.Response(
                200,
                json={"access_token": f"access-token-{token_requests}", "expires_in": 10},
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    clock = _FakeClock()
    catalog = SpotifyCatalog(
        _client(httpx.MockTransport(handle)),
        _settings(token_expiry_skew_seconds=3.0),
        clock=clock,
        sleep=clock.sleep,
    )

    await catalog.search_tracks("Kiki", _MARKET)
    clock.value = 6.9
    await catalog.search_tracks("Kiki", _MARKET)
    clock.value = 7.0
    await catalog.search_tracks("Kiki", _MARKET)

    assert token_requests == 2
    await catalog.aclose()


async def test_catalog_returns_ordered_album_candidates_and_empty_track_searches() -> None:
    """Preserve successful empty searches apart from provider failures."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        if request.url.params["type"] == "track":
            return httpx.Response(200, json={"tracks": {"items": []}})
        assert dict(request.url.params) == {
            "q": "Kiki's Delivery Service",
            "type": "album",
            "market": "JP",
            "limit": "3",
        }
        albums = [_track_payload(f"track-{position}")["album"] for position in range(4)]
        return httpx.Response(200, json={"albums": {"items": albums}})

    catalog = SpotifyCatalog(_client(httpx.MockTransport(handle)), _settings())

    no_tracks = await catalog.search_tracks("No Match", _MARKET)
    albums = await catalog.search_albums("Kiki's Delivery Service", _MARKET)

    assert no_tracks == ()
    assert [album.spotify_album_id for album in albums] == ["album-1", "album-1", "album-1"]
    assert len(albums) == 3
    await catalog.aclose()


async def test_catalog_maps_authentication_rejection_without_logging_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep Client Credentials and token-bearing details out of typed failures and logs."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request)

    catalog = SpotifyCatalog(_client(httpx.MockTransport(handle)), _settings())

    with pytest.raises(CatalogProviderError, match="Spotify catalog request failed"):
        await catalog.search_tracks("Kiki", _MARKET)

    logged_messages = "\\n".join(record.getMessage() for record in caplog.records)
    assert "test-client-id" not in logged_messages
    assert "test-client-secret" not in logged_messages
    assert "access-token" not in logged_messages
    await catalog.aclose()


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection failed"),
        httpx.ReadTimeout("request timed out"),
    ],
)
async def test_catalog_translates_network_failures_to_typed_errors(
    failure: httpx.RequestError,
) -> None:
    """Translate network failures without returning a completed no-match result."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        raise failure

    catalog = SpotifyCatalog(_client(httpx.MockTransport(handle)), _settings())
    expected_error = (
        PipelineTimeoutError
        if isinstance(failure, httpx.TimeoutException)
        else CatalogProviderError
    )

    with pytest.raises(expected_error):
        await catalog.search_tracks("Kiki", _MARKET)

    await catalog.aclose()


async def test_catalog_rejects_http_and_malformed_search_responses() -> None:
    """Reject non-success and structurally incomplete catalog responses."""
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json={"tracks": {"items": [{"id": "missing-fields"}]}}),
        ]
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        return next(responses)

    catalog = SpotifyCatalog(_client(httpx.MockTransport(handle)), _settings())

    for query in ("first", "second"):
        with pytest.raises(CatalogProviderError, match="Spotify catalog request failed"):
            await catalog.search_tracks(query, _MARKET)

    await catalog.aclose()


def test_spotify_configuration_rejects_missing_or_unsafe_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail startup validation without echoing credential or token-bearing configuration."""
    monkeypatch.delenv("REELIO_SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("REELIO_SPOTIFY_CLIENT_SECRET", raising=False)

    with pytest.raises(ValidationError, match="REELIO_SPOTIFY_CLIENT_ID"):
        SpotifyConfig(_env_file=None)  # type: ignore[call-arg]

    with pytest.raises(ValidationError) as error:
        _settings(
            client_secret="spotify-client-secret",
            token_url="https://account:token-secret@spotify.test/api/token?access_token=secret",
        )

    assert "spotify-client-secret" not in str(error.value)
    assert "token-secret" not in str(error.value)
    assert "access_token=secret" not in str(error.value)


@pytest.mark.parametrize("market", ["us", "USA", "U1", " U"])
def test_spotify_configuration_rejects_invalid_default_market(market: str) -> None:
    """Require the configured default market to use uppercase alpha-two syntax."""
    with pytest.raises(ValidationError):
        _settings(default_market=market)


def test_spotify_configuration_defaults_to_us_market() -> None:
    """Provide US as the validated market when no override is configured."""
    assert _settings().default_market == "US"


async def test_catalog_rejects_impossible_month_precision_release_dates() -> None:
    """Treat calendar-invalid required Spotify release data as a provider failure."""
    track = _track_payload("playable-track")
    album = track["album"]
    assert isinstance(album, dict)
    album["release_date"] = "2024-13"
    album["release_date_precision"] = "month"

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "access-token", "expires_in": 3600},
            )
        return httpx.Response(200, json={"tracks": {"items": [track]}})

    catalog = SpotifyCatalog(_client(httpx.MockTransport(handle)), _settings())

    with pytest.raises(CatalogProviderError, match="Spotify catalog request failed"):
        await catalog.search_tracks("Kiki", _MARKET)

    await catalog.aclose()
