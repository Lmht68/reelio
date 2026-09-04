"""In-process HTTP coverage for Spotify-backed Track extraction."""

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
from reelio.extraction.services.enrichment.spotify import SpotifyTrackResolver
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
        """Return one response with a single release-contextualized Track Mention."""
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
                "music_releases": [],
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


def _spotify_track_payload() -> dict[str, object]:
    """Return one Spotify Track payload that exactly matches the interpreted Track."""
    return {
        "id": "0DiWol3AO6WpXZgp0goxAV",
        "name": "One More Time",
        "artists": [{"id": "4tZwfgrHOc3mvqYlEYSvVi", "name": "Daft Punk"}],
        "external_urls": {"spotify": "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV"},
        "album": {
            "id": "2noRn2Aes5aoNVsU6iWThc",
            "name": "Discovery",
            "artists": [{"id": "4tZwfgrHOc3mvqYlEYSvVi", "name": "Daft Punk"}],
            "external_urls": {"spotify": "https://open.spotify.com/album/2noRn2Aes5aoNVsU6iWThc"},
            "release_date": "2001-02-26",
            "release_date_precision": "day",
            "album_type": "album",
            "images": [],
        },
    }


async def test_extract_resolves_a_track_through_mocked_spotify_catalog() -> None:
    """Expose a resolved Track through the in-process HTTP endpoint without live I/O."""
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
        assert dict(request.url.params) == {
            "q": "track:One More Time artist:Daft Punk album:Discovery year:2001",
            "type": "track",
            "market": "JP",
            "limit": "3",
        }
        return httpx.Response(
            200,
            json={"tracks": {"items": [_spotify_track_payload()]}},
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
            SpotifyTrackResolver(catalog),
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
    assert len(requests) == 2
    assert response.json()["market"] == "JP"
    assert set(response.json()["results"]) == {"movies", "tv_series", "tracks"}
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
                "spotify_track_id": "0DiWol3AO6WpXZgp0goxAV",
                "spotify_url": "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
            },
        }
    ]
