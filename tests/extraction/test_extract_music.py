"""In-process HTTP coverage for Spotify-backed Music extraction."""

import json
from collections.abc import Callable, Iterator, Sequence
from typing import cast

import httpx
import pytest

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.router import get_pipeline
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.catalog.config import SpotifyConfig
from reelio.extraction.services.catalog.spotify import SpotifyCatalog
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
from reelio.extraction.services.enrichment.spotify import SpotifyMusicResolver
from reelio.extraction.services.interpretation.config import (
    InterpretationConfig,
    LLMProvider,
)
from reelio.extraction.services.interpretation.service import MentionInterpretationService
from reelio.extraction.services.interpretation.types import LLMMessage
from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import Platform, Source, Transcript, TranscriptMethod
from reelio.main import app
from tests.extraction.fakes import FakeScreenWorkResolver

_CANONICAL_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Isolate the application-level extraction pipeline override."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _MetadataService:
    """Return deterministic inspected Source metadata for endpoint coverage."""

    async def inspect(self, submitted_url: str) -> InspectedSource:
        """Return a canonical Source for the submitted endpoint URL."""
        assert submitted_url == _CANONICAL_URL
        return InspectedSource(
            source=Source(
                platform=Platform.YOUTUBE,
                video_id="dQw4w9WgXcQ",
                url=submitted_url,
                title="Music review",
                description="A review of One More Time.",
                channel="Example channel",
                duration_seconds=42,
            )
        )


class _TranscriptionService:
    """Return deterministic transcript material for endpoint coverage."""

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: PreparedAudio | None = None,
    ) -> Transcript:
        """Return the transcript consumed by deterministic interpretation."""
        assert source.url == submitted_url
        assert prepared_audio is None
        return Transcript(
            text="One More Time by Daft Punk is the standout track.",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        )


class _InterpretationProvider:
    """Return one strict Track interpretation response without network I/O."""

    def __init__(self) -> None:
        """Initialize a provider with observable completion calls."""
        self.calls: list[tuple[LLMMessage, ...]] = []
        self.closed = False

    @property
    def provider_name(self) -> LLMProvider:
        """Return a stable test provider identity."""
        return LLMProvider.DEEPSEEK

    @property
    def model_name(self) -> str:
        """Return a stable test model identity."""
        return "deterministic-track-provider"

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return one response with a Track and a Music Release Mention."""
        self.calls.append(tuple(messages))
        return json.dumps(
            {
                "movies": [],
                "tv_series": [],
                "tracks": [
                    {
                        "track_title": "One More Time",
                        "artists": ["Daft Punk"],
                        "release_title": "Discovery",
                        "release_year": 2001,
                    }
                ],
                "music_releases": [
                    {
                        "release_title": "Discovery",
                        "artists": ["Daft Punk"],
                        "release_year": 2001,
                    }
                ],
            }
        )

    async def aclose(self) -> None:
        """Record pipeline-owned interpretation-provider closure."""
        self.closed = True


def _spotify_settings() -> SpotifyConfig:
    """Build Spotify settings without repository environment files."""
    settings_type = cast(Callable[..., SpotifyConfig], SpotifyConfig)
    return settings_type(
        _env_file=None,
        client_id="test-client-id",
        client_secret="test-client-secret",
        base_url="https://api.spotify.test/v1",
        token_url="https://accounts.spotify.test/api/token",
    )


def _interpretation_settings() -> InterpretationConfig:
    """Build interpretation limits without repository environment files."""
    settings_type = cast(Callable[..., InterpretationConfig], InterpretationConfig)
    return settings_type(_env_file=None)


def _spotify_images(image_set: str) -> list[dict[str, int | str]]:
    """Return provider-ordered Spotify image payloads for one Album."""
    return [
        {
            "url": f"https://i.scdn.co/image/{image_set}-primary",
            "width": 640,
            "height": 640,
        },
        {
            "url": f"https://i.scdn.co/image/{image_set}-secondary",
            "width": 300,
            "height": 300,
        },
    ]


def _spotify_track_payload(include_images: bool) -> dict[str, object]:
    """Return one relinked Spotify Track payload matching the interpreted Track."""
    return {
        "id": "playable-track",
        "name": "One More Time",
        "artists": [{"id": "4tZwfgrHOc3mvqYlEYSvVi", "name": "Daft Punk"}],
        "external_urls": {"spotify": "https://open.spotify.com/track/playable-track"},
        "linked_from": {"id": "original-track"},
        "album": {
            "id": "attached-album",
            "name": "Discovery",
            "artists": [{"id": "4tZwfgrHOc3mvqYlEYSvVi", "name": "Daft Punk"}],
            "external_urls": {"spotify": "https://open.spotify.com/album/attached-album"},
            "release_date": "2001-02-26",
            "release_date_precision": "day",
            "album_type": "album",
            "images": _spotify_images("attached") if include_images else [],
        },
    }


def _spotify_album_payload(include_images: bool) -> dict[str, object]:
    """Return one Spotify Album payload matching the interpreted Music Release."""
    return {
        "id": "direct-album",
        "name": "Discovery",
        "artists": [{"id": "4tZwfgrHOc3mvqYlEYSvVi", "name": "Daft Punk"}],
        "external_urls": {"spotify": "https://open.spotify.com/album/direct-album"},
        "release_date": "2001-02-26",
        "release_date_precision": "day",
        "album_type": "album",
        "images": _spotify_images("direct") if include_images else [],
    }


@pytest.mark.parametrize(
    ("include_images", "attached_cover_url", "direct_cover_url"),
    [
        (
            True,
            "https://i.scdn.co/image/attached-primary",
            "https://i.scdn.co/image/direct-primary",
        ),
        (False, None, None),
    ],
    ids=["provider-images", "no-provider-images"],
)
async def test_extract_resolves_music_through_mocked_spotify_catalog(
    include_images: bool,
    attached_cover_url: str | None,
    direct_cover_url: str | None,
) -> None:
    """Expose a resolved Track and Music Release through the in-process HTTP endpoint."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "accounts.spotify.test":
            assert request.method == "POST"
            return httpx.Response(
                200,
                json={"access_token": "test-access-token", "expires_in": 3600},
            )

        assert request.method == "GET"
        assert request.url.path == "/v1/search"
        assert request.headers["authorization"] == "Bearer test-access-token"
        params = dict(request.url.params)
        assert params["market"] == "JP"
        assert params["limit"] == "3"
        if params["type"] == "track":
            assert params["q"] == ("track:One More Time artist:Daft Punk album:Discovery year:2001")
            return httpx.Response(
                200,
                json={"tracks": {"items": [_spotify_track_payload(include_images)]}},
            )
        assert params["type"] == "album"
        assert params["q"] == "album:Discovery artist:Daft Punk year:2001"
        return httpx.Response(
            200,
            json={"albums": {"items": [_spotify_album_payload(include_images)]}},
        )

    http_client = httpx.AsyncClient(
        base_url="https://api.spotify.test/v1/",
        transport=httpx.MockTransport(handle),
    )
    catalog = SpotifyCatalog(http_client, _spotify_settings())
    interpretation_provider = _InterpretationProvider()
    pipeline = ExtractionPipeline(
        _MetadataService(),
        _TranscriptionService(),
        MentionInterpretationService(interpretation_provider, _interpretation_settings()),
        ExtractionResultAggregator(
            FakeScreenWorkResolver(),
            SpotifyMusicResolver(catalog),
        ),
        SpotifyMarket("US"),
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/extract",
                json={"url": _CANONICAL_URL, "market": "JP"},
            )
    finally:
        await pipeline.aclose()
        await catalog.aclose()

    assert response.status_code == 200
    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert len(interpretation_provider.calls) == 1
    assert len(requests) == 3
    assert sum(request.method == "POST" for request in requests) == 1
    assert sum(request.method == "GET" for request in requests) == 2
    assert response.json()["market"] == "JP"
    assert set(response.json()["results"]) == {
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
    }
    assert response.json()["results"]["movies"] == []
    assert response.json()["results"]["tv_series"] == []
    assert response.json()["results"]["tracks"] == [
        {
            "status": "resolved",
            "track_mention": {
                "track_title": "One More Time",
                "artists": ["Daft Punk"],
                "release_title": "Discovery",
                "release_year": 2001,
            },
            "track": {
                "track_title": "One More Time",
                "artists": [
                    {
                        "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                        "name": "Daft Punk",
                    }
                ],
                "spotify_track_id": "playable-track",
                "spotify_url": "https://open.spotify.com/track/playable-track",
                "preferred_music_release": {
                    "release_title": "Discovery",
                    "artists": [
                        {
                            "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                            "name": "Daft Punk",
                        }
                    ],
                    "release_date": "2001-02-26",
                    "release_date_precision": "day",
                    "album_type": "album",
                    "spotify_album_id": "attached-album",
                    "spotify_url": "https://open.spotify.com/album/attached-album",
                    "cover_url": attached_cover_url,
                },
                "cover_url": attached_cover_url,
            },
        }
    ]
    assert response.json()["results"]["music_releases"] == [
        {
            "status": "resolved",
            "music_release_mention": {
                "release_title": "Discovery",
                "artists": ["Daft Punk"],
                "release_year": 2001,
            },
            "music_release": {
                "release_title": "Discovery",
                "artists": [
                    {
                        "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                        "name": "Daft Punk",
                    }
                ],
                "release_date": "2001-02-26",
                "release_date_precision": "day",
                "album_type": "album",
                "spotify_album_id": "direct-album",
                "spotify_url": "https://open.spotify.com/album/direct-album",
                "cover_url": direct_cover_url,
            },
        }
    ]
