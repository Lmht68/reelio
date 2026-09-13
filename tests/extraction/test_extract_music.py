"""In-process HTTP coverage for Spotify-backed Music extraction."""

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, cast

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
from reelio.extraction.types import (
    AlbumType,
    Platform,
    Source,
    Transcript,
    TranscriptMethod,
)
from reelio.main import app
from tests.extraction.fakes import FakeBookResolver, FakeScreenWorkResolver

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
    """Return one configured strict interpretation response without network I/O."""

    def __init__(self, response: dict[str, object]) -> None:
        """Initialize a provider with one observable response.

        Args:
            response: Strict interpretation response returned from every completion.
        """
        self._response = response
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
        """Return the configured strict interpretation response."""
        self.calls.append(tuple(messages))
        return json.dumps(self._response)

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


def _spotify_artist_payloads(artist_names: Sequence[str]) -> list[dict[str, str]]:
    """Return Spotify Artist Credit payloads in the supplied order."""
    return [
        {"id": f"artist-{index}", "name": artist_name}
        for index, artist_name in enumerate(artist_names)
    ]


def _spotify_track_payload(
    *,
    spotify_track_id: str = "playable-track",
    title: str = "One More Time",
    artist_names: Sequence[str] = ("Daft Punk",),
    attached_album_id: str = "attached-album",
    attached_album_title: str = "Discovery",
    attached_album_artist_names: Sequence[str] = ("Daft Punk",),
    attached_album_release_date: str = "2001",
    attached_album_type: AlbumType = "album",
    include_images: bool = True,
) -> dict[str, object]:
    """Return one relinked Spotify Track payload with configurable music metadata."""
    return {
        "id": spotify_track_id,
        "name": title,
        "artists": _spotify_artist_payloads(artist_names),
        "external_urls": {"spotify": f"https://open.spotify.com/track/{spotify_track_id}"},
        "linked_from": {"id": "original-track"},
        "album": {
            "id": attached_album_id,
            "name": attached_album_title,
            "artists": _spotify_artist_payloads(attached_album_artist_names),
            "external_urls": {"spotify": f"https://open.spotify.com/album/{attached_album_id}"},
            "release_date": attached_album_release_date,
            "album_type": attached_album_type,
            "images": _spotify_images("attached") if include_images else [],
        },
    }


def _spotify_album_payload(
    *,
    spotify_album_id: str = "direct-album",
    title: str = "Discovery",
    artist_names: Sequence[str] = ("Daft Punk",),
    release_date: str = "2001-02",
    album_type: AlbumType = "album",
    include_images: bool = True,
) -> dict[str, object]:
    """Return one Spotify Album payload with configurable music metadata."""
    return {
        "id": spotify_album_id,
        "name": title,
        "artists": _spotify_artist_payloads(artist_names),
        "external_urls": {"spotify": f"https://open.spotify.com/album/{spotify_album_id}"},
        "release_date": release_date,
        "album_type": album_type,
        "images": _spotify_images("direct") if include_images else [],
    }


def _interpretation_response(
    *,
    tracks: list[dict[str, object]],
    music_releases: list[dict[str, object]],
) -> dict[str, object]:
    """Return a complete strict interpretation response with supplied Music Mentions."""
    return {
        "movies": [],
        "tv_series": [],
        "tracks": tracks,
        "music_releases": music_releases,
        "books": [],
    }


async def _post_extract(
    interpretation_response: dict[str, object],
    spotify_transport: httpx.AsyncBaseTransport,
) -> tuple[httpx.Response, _InterpretationProvider, httpx.AsyncClient]:
    """Run the real extraction pipeline through the HTTP endpoint and close its owners."""
    http_client = httpx.AsyncClient(
        base_url="https://api.spotify.test/v1/",
        transport=spotify_transport,
    )
    catalog = SpotifyCatalog(http_client, _spotify_settings())
    interpretation_provider = _InterpretationProvider(interpretation_response)
    pipeline = ExtractionPipeline(
        _MetadataService(),
        _TranscriptionService(),
        MentionInterpretationService(interpretation_provider, _interpretation_settings()),
        ExtractionResultAggregator(
            FakeScreenWorkResolver(),
            SpotifyMusicResolver(catalog),
            FakeBookResolver(),
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

    return response, interpretation_provider, http_client


@dataclass(frozen=True)
class _ExactResolutionScenario:
    """Define one endpoint-visible exact Music resolution outcome."""

    interpretation_response: dict[str, object]
    search_type: Literal["track", "album"]
    search_query: str
    candidates: tuple[dict[str, object], ...]
    result_list_key: Literal["tracks", "music_releases"]
    mention_key: Literal["track_mention", "music_release_mention"]
    entity_key: Literal["track", "music_release"]
    mention: dict[str, object]
    expected_status: Literal["resolved", "unresolved"]
    expected_spotify_id: str | None


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
    """Expose exact shared-credit Music resolution through the in-process endpoint."""
    requests: list[httpx.Request] = []
    track_mention = {
        "track_title": "One More Time",
        "artists": ["Daft Punk", "Romanthony"],
        "release_title": "Discovery",
        "release_year": 2001,
    }
    music_release_mention = {
        "release_title": "Discovery",
        "artists": ["Daft Punk", "Romanthony"],
        "release_year": 2001,
    }
    interpretation_response = _interpretation_response(
        tracks=[track_mention],
        music_releases=[music_release_mention],
    )

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
        if params["type"] == "track":
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            _spotify_track_payload(
                                title="One More Time (2011 Remaster)",
                                attached_album_title="Discovery (Deluxe Edition)",
                                artist_names=("  romanthony  ", "Additional Artist"),
                                attached_album_artist_names=(
                                    "DAFT PUNK",
                                    "Additional Artist",
                                ),
                                attached_album_release_date="2025",
                                include_images=include_images,
                            )
                        ]
                    }
                },
            )
        assert params["type"] == "album"
        return httpx.Response(
            200,
            json={
                "albums": {
                    "items": [
                        _spotify_album_payload(
                            title="Discovery (Bonus Edition)",
                            artist_names=("Additional Artist", "DAFT PUNK"),
                            release_date="2025-02",
                            include_images=include_images,
                        )
                    ]
                }
            },
        )

    response, interpretation_provider, http_client = await _post_extract(
        interpretation_response,
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert len(interpretation_provider.calls) == 1
    assert len(requests) == 3
    assert sum(request.method == "POST" for request in requests) == 1
    search_parameters = {
        request.url.params["type"]: dict(request.url.params)
        for request in requests
        if request.method == "GET"
    }
    assert search_parameters == {
        "track": {
            "q": "track:One More Time artist:Daft Punk",
            "type": "track",
            "market": "JP",
            "offset": "0",
            "limit": "3",
        },
        "album": {
            "q": "album:Discovery artist:Daft Punk",
            "type": "album",
            "market": "JP",
            "offset": "0",
            "limit": "3",
        },
    }

    response_body = response.json()
    assert response_body["market"] == "JP"
    assert response_body["results"]["movies"] == []
    assert response_body["results"]["tv_series"] == []
    track_result = response_body["results"]["tracks"][0]
    assert track_result["status"] == "resolved"
    assert track_result["track_mention"] == track_mention
    assert track_result["track"] == {
        "track_title": "One More Time (2011 Remaster)",
        "artists": [
            {"spotify_artist_id": "artist-0", "name": "romanthony"},
            {"spotify_artist_id": "artist-1", "name": "Additional Artist"},
        ],
        "spotify_track_id": "playable-track",
        "spotify_url": "https://open.spotify.com/track/playable-track",
        "preferred_music_release": {
            "release_title": "Discovery (Deluxe Edition)",
            "artists": [
                {"spotify_artist_id": "artist-0", "name": "DAFT PUNK"},
                {"spotify_artist_id": "artist-1", "name": "Additional Artist"},
            ],
            "release_date": "2025",
            "album_type": "album",
            "spotify_album_id": "attached-album",
            "spotify_url": "https://open.spotify.com/album/attached-album",
            "cover_url": attached_cover_url,
        },
        "cover_url": attached_cover_url,
    }
    music_release_result = response_body["results"]["music_releases"][0]
    assert music_release_result["status"] == "resolved"
    assert music_release_result["music_release_mention"] == music_release_mention
    assert music_release_result["music_release"] == {
        "release_title": "Discovery (Bonus Edition)",
        "artists": [
            {"spotify_artist_id": "artist-0", "name": "Additional Artist"},
            {"spotify_artist_id": "artist-1", "name": "DAFT PUNK"},
        ],
        "release_date": "2025-02",
        "album_type": "album",
        "spotify_album_id": "direct-album",
        "spotify_url": "https://open.spotify.com/album/direct-album",
        "cover_url": direct_cover_url,
    }


@pytest.mark.parametrize(
    "scenario",
    [
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[
                    {
                        "track_title": "Standalone",
                        "artists": ["Artist One"],
                        "release_title": None,
                        "release_year": None,
                    }
                ],
                music_releases=[],
            ),
            search_type="track",
            search_query="track:Standalone artist:Artist One",
            candidates=(
                _spotify_track_payload(
                    spotify_track_id="standalone",
                    title="Standalone",
                    artist_names=("Artist One",),
                    attached_album_title="Unrelated Album",
                ),
            ),
            result_list_key="tracks",
            mention_key="track_mention",
            entity_key="track",
            mention={
                "track_title": "Standalone",
                "artists": ["Artist One"],
                "release_title": None,
                "release_year": None,
            },
            expected_status="resolved",
            expected_spotify_id="standalone",
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[
                    {
                        "track_title": "Context Song",
                        "artists": ["Artist One"],
                        "release_title": "Context Album",
                        "release_year": 1999,
                    }
                ],
                music_releases=[],
            ),
            search_type="track",
            search_query="track:Context Song artist:Artist One",
            candidates=(
                _spotify_track_payload(
                    spotify_track_id="wrong-context",
                    title="Context Song",
                    artist_names=("Artist One",),
                    attached_album_title="Other Album",
                ),
                _spotify_track_payload(
                    spotify_track_id="exact-context",
                    title="Context Song",
                    artist_names=("Artist One",),
                    attached_album_title="Context Album",
                    attached_album_release_date="2025",
                ),
            ),
            result_list_key="tracks",
            mention_key="track_mention",
            entity_key="track",
            mention={
                "track_title": "Context Song",
                "artists": ["Artist One"],
                "release_title": "Context Album",
                "release_year": 1999,
            },
            expected_status="resolved",
            expected_spotify_id="exact-context",
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[
                    {
                        "track_title": "Context Song",
                        "artists": ["Artist One"],
                        "release_title": "Context Album",
                        "release_year": None,
                    }
                ],
                music_releases=[],
            ),
            search_type="track",
            search_query="track:Context Song artist:Artist One",
            candidates=(
                _spotify_track_payload(
                    spotify_track_id="wrong-context",
                    title="Context Song",
                    artist_names=("Artist One",),
                    attached_album_title="Other Album",
                ),
            ),
            result_list_key="tracks",
            mention_key="track_mention",
            entity_key="track",
            mention={
                "track_title": "Context Song",
                "artists": ["Artist One"],
                "release_title": "Context Album",
                "release_year": None,
            },
            expected_status="unresolved",
            expected_spotify_id=None,
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[],
                music_releases=[
                    {
                        "release_title": "Compilation",
                        "artists": ["Artist One"],
                        "release_year": 2000,
                    }
                ],
            ),
            search_type="album",
            search_query="album:Compilation artist:Artist One",
            candidates=(
                _spotify_album_payload(
                    spotify_album_id="compilation",
                    title="Compilation",
                    artist_names=("Artist One",),
                    album_type="compilation",
                ),
            ),
            result_list_key="music_releases",
            mention_key="music_release_mention",
            entity_key="music_release",
            mention={
                "release_title": "Compilation",
                "artists": ["Artist One"],
                "release_year": 2000,
            },
            expected_status="resolved",
            expected_spotify_id="compilation",
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[
                    {
                        "track_title": "No Shared Artist",
                        "artists": ["Artist One"],
                        "release_title": None,
                        "release_year": None,
                    }
                ],
                music_releases=[],
            ),
            search_type="track",
            search_query="track:No Shared Artist artist:Artist One",
            candidates=(
                _spotify_track_payload(
                    title="No Shared Artist",
                    artist_names=("Other Artist",),
                ),
            ),
            result_list_key="tracks",
            mention_key="track_mention",
            entity_key="track",
            mention={
                "track_title": "No Shared Artist",
                "artists": ["Artist One"],
                "release_title": None,
                "release_year": None,
            },
            expected_status="unresolved",
            expected_spotify_id=None,
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[],
                music_releases=[
                    {
                        "release_title": "Exact Album",
                        "artists": ["Artist One"],
                        "release_year": None,
                    }
                ],
            ),
            search_type="album",
            search_query="album:Exact Album artist:Artist One",
            candidates=(
                _spotify_album_payload(
                    title="Different Album",
                    artist_names=("Artist One",),
                ),
            ),
            result_list_key="music_releases",
            mention_key="music_release_mention",
            entity_key="music_release",
            mention={
                "release_title": "Exact Album",
                "artists": ["Artist One"],
                "release_year": None,
            },
            expected_status="unresolved",
            expected_spotify_id=None,
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[
                    {
                        "track_title": "First Exact",
                        "artists": ["Artist One"],
                        "release_title": None,
                        "release_year": None,
                    }
                ],
                music_releases=[],
            ),
            search_type="track",
            search_query="track:First Exact artist:Artist One",
            candidates=(
                _spotify_track_payload(
                    spotify_track_id="first-exact",
                    title="First Exact",
                    artist_names=("Artist One",),
                ),
                _spotify_track_payload(
                    spotify_track_id="second-exact",
                    title="First Exact",
                    artist_names=("Artist One",),
                ),
            ),
            result_list_key="tracks",
            mention_key="track_mention",
            entity_key="track",
            mention={
                "track_title": "First Exact",
                "artists": ["Artist One"],
                "release_title": None,
                "release_year": None,
            },
            expected_status="resolved",
            expected_spotify_id="first-exact",
        ),
        _ExactResolutionScenario(
            interpretation_response=_interpretation_response(
                tracks=[],
                music_releases=[
                    {
                        "release_title": "Bounded Album",
                        "artists": ["Artist One"],
                        "release_year": None,
                    }
                ],
            ),
            search_type="album",
            search_query="album:Bounded Album artist:Artist One",
            candidates=(
                _spotify_album_payload(title="Wrong One", artist_names=("Artist One",)),
                _spotify_album_payload(title="Wrong Two", artist_names=("Artist One",)),
                _spotify_album_payload(title="Wrong Three", artist_names=("Artist One",)),
                _spotify_album_payload(
                    spotify_album_id="fourth-exact",
                    title="Bounded Album",
                    artist_names=("Artist One",),
                ),
            ),
            result_list_key="music_releases",
            mention_key="music_release_mention",
            entity_key="music_release",
            mention={
                "release_title": "Bounded Album",
                "artists": ["Artist One"],
                "release_year": None,
            },
            expected_status="unresolved",
            expected_spotify_id=None,
        ),
    ],
    ids=[
        "track-without-context",
        "track-context-uses-later-exact-candidate",
        "track-context-unresolved",
        "direct-compilation",
        "no-shared-artist",
        "below-fuzzy-threshold",
        "first-exact-candidate",
        "fourth-candidate-is-ignored",
    ],
)
async def test_extract_applies_the_bounded_music_resolution_matrix(
    scenario: _ExactResolutionScenario,
) -> None:
    """Expose bounded Music resolution and the first-three Candidate limit."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "test-access-token", "expires_in": 3600},
            )

        assert request.method == "GET"
        assert dict(request.url.params) == {
            "q": scenario.search_query,
            "type": scenario.search_type,
            "market": "JP",
            "offset": "0",
            "limit": "3",
        }
        result_key = "tracks" if scenario.search_type == "track" else "albums"
        return httpx.Response(
            200,
            json={result_key: {"items": list(scenario.candidates)}},
        )

    response, interpretation_provider, http_client = await _post_extract(
        scenario.interpretation_response,
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert len(interpretation_provider.calls) == 1
    assert sum(request.method == "POST" for request in requests) == 1
    assert sum(request.method == "GET" for request in requests) == 1
    result = response.json()["results"][scenario.result_list_key][0]
    assert result[scenario.mention_key] == scenario.mention
    assert result["status"] == scenario.expected_status
    if scenario.expected_spotify_id is None:
        assert result[scenario.entity_key] is None
    else:
        spotify_id_key = (
            "spotify_track_id" if scenario.search_type == "track" else "spotify_album_id"
        )
        assert result[scenario.entity_key][spotify_id_key] == scenario.expected_spotify_id


async def _extract_direct_music_release(
    release_title: str,
    candidates: tuple[dict[str, object], ...],
    *,
    artists: Sequence[str] = ("Daft Punk",),
    release_year: int | None = 2001,
) -> dict[str, object]:
    """Resolve one direct Music Release through the in-process extraction endpoint."""
    music_release_mention: dict[str, object] = {
        "release_title": release_title,
        "artists": list(artists),
        "release_year": release_year,
    }
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
            "q": f"album:{release_title} artist:{artists[0]}",
            "type": "album",
            "market": "JP",
            "offset": "0",
            "limit": "3",
        }
        return httpx.Response(200, json={"albums": {"items": list(candidates)}})

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(tracks=[], music_releases=[music_release_mention]),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert len(interpretation_provider.calls) == 1
    assert sum(request.method == "POST" for request in requests) == 1
    assert sum(request.method == "GET" for request in requests) == 1
    result = response.json()["results"]["music_releases"][0]
    assert result["music_release_mention"] == music_release_mention
    return cast(dict[str, object], result)


async def _extract_track_version(
    track_title: str,
    candidates: tuple[dict[str, object], ...],
    *,
    artists: Sequence[str] = ("Daft Punk",),
    release_title: str | None = None,
    release_year: int | None = None,
) -> dict[str, object]:
    """Resolve one Track through the in-process extraction endpoint."""
    track_mention: dict[str, object] = {
        "track_title": track_title,
        "artists": list(artists),
        "release_title": release_title,
        "release_year": release_year,
    }
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
            "q": f"track:{track_title} artist:{artists[0]}",
            "type": "track",
            "market": "JP",
            "offset": "0",
            "limit": "3",
        }
        return httpx.Response(200, json={"tracks": {"items": list(candidates)}})

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(tracks=[track_mention], music_releases=[]),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert len(interpretation_provider.calls) == 1
    assert sum(request.method == "POST" for request in requests) == 1
    assert sum(request.method == "GET" for request in requests) == 1
    result = response.json()["results"]["tracks"][0]
    assert result["track_mention"] == track_mention
    return cast(dict[str, object], result)


_MUSIC_RELEASE_EDITION_CASES: tuple[tuple[str, str], ...] = (
    ("Discovery (Remaster)", "remaster"),
    ("Discovery [Remastered]", "remastered"),
    ("Discovery - Remastered Version", "remastered-version"),
    ("Discovery: 2011 Remaster", "year-remaster"),
    ("Discovery (Remastered 2011)", "remastered-year"),
    ("Discovery [2011 Remastered Version]", "year-remastered-version"),
    ("Discovery - Deluxe", "deluxe"),
    ("Discovery: Deluxe Edition", "deluxe-edition"),
    ("Discovery (sUPER   dELUXE   Edition)", "super-deluxe"),
    ("Discovery (Extended)", "extended"),
    ("Discovery [Extended Version]", "extended-version"),
    ("Discovery - Extended Edition", "extended-edition"),
    ("Discovery [Expanded Edition]", "expanded"),
    ("Discovery - Special Edition", "special"),
    ("Discovery: Anniversary Edition", "anniversary"),
    ("Discovery (1st Anniversary)", "first-anniversary"),
    ("Discovery [2nd Anniversary Edition]", "second-anniversary"),
    ("Discovery - 3rd Anniversary", "third-anniversary"),
    ("Discovery: 20th Anniversary Edition", "twentieth-anniversary"),
    ("Discovery (Reissue)", "reissue"),
    ("Discovery [Reissued]", "reissued"),
    ("Discovery - Bonus Edition", "bonus-edition"),
    ("Discovery: Bonus Version", "bonus-version"),
    ("Discovery (Bonus Track)", "bonus-track"),
    ("Discovery [Bonus Track Version]", "bonus-track-version"),
)


@pytest.mark.parametrize(
    ("candidate_title", "spotify_album_id"),
    _MUSIC_RELEASE_EDITION_CASES,
)
async def test_extract_resolves_every_supported_music_release_edition_designation(
    candidate_title: str,
    spotify_album_id: str,
) -> None:
    """Resolve every controlled Music Release edition across supported boundaries."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id=spotify_album_id,
                title=candidate_title,
            ),
        ),
    )

    assert result["status"] == "resolved"
    music_release = cast(dict[str, object], result["music_release"])
    assert music_release["spotify_album_id"] == spotify_album_id


@pytest.mark.parametrize(
    ("candidate_title", "spotify_track_id"),
    (
        ("One More Time (Remaster)", "track-remaster"),
        ("One More Time [Remastered]", "track-remastered"),
        ("One More Time - Remastered Version", "track-remastered-version"),
        ("One More Time: 2011 Remaster", "track-year-remaster"),
        ("One More Time (Remastered 2011)", "track-remastered-year"),
        (
            "One More Time [2011 Remastered Version]",
            "track-year-remastered-version",
        ),
        ("One More Time - Bonus Edition", "track-bonus-edition"),
        ("One More Time: Bonus Version", "track-bonus-version"),
        ("One More Time (Bonus Track)", "track-bonus-track"),
        ("One More Time [Bonus Track Version]", "track-bonus-track-version"),
    ),
)
async def test_extract_resolves_every_supported_track_version(
    candidate_title: str,
    spotify_track_id: str,
) -> None:
    """Resolve every controlled Track version without release context."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id=spotify_track_id,
                title=candidate_title,
                attached_album_title="Unrelated Release",
            ),
        ),
    )

    assert result["status"] == "resolved"
    track = cast(dict[str, object], result["track"])
    assert track["spotify_track_id"] == spotify_track_id


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    (
        ("One More Time", "One More Time (Remaster)"),
        ("One More Time (Remaster)", "One More Time"),
        ("One More Time (Remaster)", "One More Time (Bonus Track Version)"),
    ),
)
async def test_extract_resolves_equivalent_track_versions_symmetrically(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Resolve bare and equivalent Track versions in either direction."""
    result = await _extract_track_version(
        mention_title,
        (_spotify_track_payload(title=candidate_title),),
    )

    assert result["status"] == "resolved"
    assert result["track"] is not None


async def test_extract_strips_stacked_track_versions_from_the_right() -> None:
    """Resolve stacked Track version segments across supported boundaries."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title="One More Time: Bonus Track [2011 Remaster]",
            ),
        ),
    )

    assert result["status"] == "resolved"
    assert result["track"] is not None


async def test_extract_resolves_louis_prima_controlled_composite_medley() -> None:
    """Resolve Louis Prima's composite Medley with Spotify-owned metadata."""
    result = await _extract_track_version(
        "Just a Gigolo / I Ain’t Got Nobody",
        (
            _spotify_track_payload(
                spotify_track_id="louis-prima-medley",
                title="Just A Gigolo / I Ain't Got Nobody - Medley / Remastered 2002",
                artist_names=("Louis Prima",),
                attached_album_id="louis-prima-wildest",
                attached_album_title="The Wildest!",
                attached_album_artist_names=("Louis Prima",),
                attached_album_release_date="2002-10-22",
            ),
        ),
        artists=("Louis Prima",),
    )

    assert result == {
        "status": "resolved",
        "track_mention": {
            "track_title": "Just a Gigolo / I Ain’t Got Nobody",
            "artists": ["Louis Prima"],
            "release_title": None,
            "release_year": None,
        },
        "track": {
            "track_title": ("Just A Gigolo / I Ain't Got Nobody - Medley / Remastered 2002"),
            "artists": [{"spotify_artist_id": "artist-0", "name": "Louis Prima"}],
            "spotify_track_id": "louis-prima-medley",
            "spotify_url": "https://open.spotify.com/track/louis-prima-medley",
            "preferred_music_release": {
                "release_title": "The Wildest!",
                "artists": [
                    {"spotify_artist_id": "artist-0", "name": "Louis Prima"},
                ],
                "release_date": "2002-10-22",
                "album_type": "album",
                "spotify_album_id": "louis-prima-wildest",
                "spotify_url": ("https://open.spotify.com/album/louis-prima-wildest"),
                "cover_url": "https://i.scdn.co/image/attached-primary",
            },
            "cover_url": "https://i.scdn.co/image/attached-primary",
        },
    }


async def test_extract_resolves_explicit_compound_medley_with_retained_designation() -> None:
    """Resolve an explicit composite Medley when the Candidate retains Medley."""
    result = await _extract_track_version(
        "First Work / Second Work - Medley",
        (
            _spotify_track_payload(
                spotify_track_id="retained-compound-medley",
                title="First Work/Second Work - Remastered 2002/Medley",
            ),
        ),
    )

    assert result["status"] == "resolved"
    track = cast(dict[str, object], result["track"])
    assert track["spotify_track_id"] == "retained-compound-medley"
    assert track["track_title"] == "First Work/Second Work - Remastered 2002/Medley"


@pytest.mark.parametrize(
    "component",
    (
        "Live",
        "2024 Remix",
        "Acoustic",
        "Instrumental",
        "Radio Edit",
        "Karaoke",
        "A Tribute Performance",
        "Deluxe Edition",
        "Unknown Version",
    ),
)
async def test_extract_rejects_ineligible_compound_medley_component(
    component: str,
) -> None:
    """Reject a composite Medley containing blocked or unrecognized material."""
    result = await _extract_track_version(
        "First Work / Second Work",
        (
            _spotify_track_payload(
                title=f"First Work / Second Work - Medley / {component}",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    (
        (
            "Single Work",
            "Single Work - Medley / Remastered 2002",
        ),
        (
            "First Work / Second Work",
            "First Work / Different Work - Medley / Remastered 2002",
        ),
    ),
)
async def test_extract_rejects_invalid_composite_medley_structure(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Reject non-composite and changed-base composite Medley Candidates."""
    result = await _extract_track_version(
        mention_title,
        (_spotify_track_payload(title=candidate_title),),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


async def test_extract_blocks_fuzzy_similarity_for_changed_medley_base() -> None:
    """Reject a changed Medley base that would otherwise exceed fuzzy similarity."""
    result = await _extract_track_version(
        "A Gigolo / I Ain't Got Nobody",
        (
            _spotify_track_payload(
                title="Just a Gigolo / I Ain't Got Someboody - Medley",
                artist_names=("Louis Prima",),
            ),
        ),
        artists=("Louis Prima",),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


async def test_extract_requires_candidate_to_retain_explicit_medley() -> None:
    """Reject a bare Candidate for a Mention that explicitly names a Medley."""
    result = await _extract_track_version(
        "Just a Gigolo / I Ain't Got Nobody - Medley",
        (
            _spotify_track_payload(
                title="Just a Gigolo / I Ain't Got Nobody",
                artist_names=("Louis Prima",),
            ),
        ),
        artists=("Louis Prima",),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


@pytest.mark.parametrize(
    ("candidate_title", "spotify_album_id"),
    _MUSIC_RELEASE_EDITION_CASES,
)
async def test_extract_resolves_track_with_equivalent_music_release_context(
    candidate_title: str,
    spotify_album_id: str,
) -> None:
    """Resolve exact Track titles with every controlled attached release edition."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id=f"track-{spotify_album_id}",
                title="One More Time",
                attached_album_id=f"release-{spotify_album_id}",
                attached_album_title=candidate_title,
                attached_album_release_date="2025",
            ),
        ),
        release_title="Discovery",
        release_year=1999,
    )

    assert result["status"] == "resolved"
    track = cast(dict[str, object], result["track"])
    assert track["spotify_track_id"] == f"track-{spotify_album_id}"


@pytest.mark.parametrize(
    (
        "candidate_title",
        "attached_album_title",
        "attached_album_type",
        "expected_status",
    ),
    (
        (
            "One More Time (Remaster)",
            "Discovery",
            "album",
            "resolved",
        ),
        (
            "One More Time",
            "Discovery (Extended Version)",
            "single",
            "resolved",
        ),
        (
            "One More Time (Remaster)",
            "Discovery (Deluxe Edition)",
            "album",
            "resolved",
        ),
        (
            "One More",
            "Discovery",
            "album",
            "unresolved",
        ),
        (
            "One More Time (Remaster)",
            "Homework (Deluxe Edition)",
            "album",
            "unresolved",
        ),
        (
            "One More Time (Remaster)",
            "Discovery (Live) (Deluxe Edition)",
            "album",
            "unresolved",
        ),
        (
            "One More Time",
            "Discovery (Deluxe Edition)",
            "compilation",
            "unresolved",
        ),
    ),
)
async def test_extract_requires_equivalent_track_and_music_release_context(
    candidate_title: str,
    attached_album_title: str,
    attached_album_type: AlbumType,
    expected_status: Literal["resolved", "unresolved"],
) -> None:
    """Require controlled Track and attached Music Release equivalence."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title=candidate_title,
                attached_album_title=attached_album_title,
                attached_album_type=attached_album_type,
            ),
        ),
        release_title="Discovery",
        release_year=1999,
    )

    assert result["status"] == expected_status
    assert (result["track"] is not None) is (expected_status == "resolved")


async def test_extract_ignores_attached_release_without_release_context() -> None:
    """Resolve a Track version without querying or matching its attached Album."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title="One More Time (Remaster)",
                attached_album_title="Unrelated Release",
            ),
        ),
    )

    assert result["status"] == "resolved"
    track = cast(dict[str, object], result["track"])
    preferred_music_release = cast(dict[str, object], track["preferred_music_release"])
    assert preferred_music_release["release_title"] == "Unrelated Release"


async def test_extract_checks_exact_track_titles_before_versions() -> None:
    """Choose an exact Track after an earlier equivalent Track version."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id="earlier-equivalent",
                title="One More Time (Remaster)",
            ),
            _spotify_track_payload(
                spotify_track_id="later-exact",
                title="One More Time",
            ),
        ),
    )

    track = cast(dict[str, object], result["track"])
    assert result["status"] == "resolved"
    assert track["spotify_track_id"] == "later-exact"


async def test_extract_keeps_artist_filter_and_provider_order_for_track_versions() -> None:
    """Choose the first artist-eligible equivalent Track version."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id="ineligible",
                title="One More Time (Remaster)",
                artist_names=("Other Artist",),
            ),
            _spotify_track_payload(
                spotify_track_id="first-equivalent",
                title="One More Time (Bonus Track)",
            ),
            _spotify_track_payload(
                spotify_track_id="second-equivalent",
                title="One More Time (Remastered)",
            ),
        ),
    )

    track = cast(dict[str, object], result["track"])
    assert result["status"] == "resolved"
    assert track["spotify_track_id"] == "first-equivalent"


async def test_extract_limits_equivalent_track_versions_to_three_candidates() -> None:
    """Ignore an equivalent Track version after the existing candidate bound."""
    result = await _extract_track_version(
        "Bounded Track",
        (
            _spotify_track_payload(title="Wrong One"),
            _spotify_track_payload(title="Wrong Two"),
            _spotify_track_payload(title="Wrong Three"),
            _spotify_track_payload(
                spotify_track_id="fourth-equivalent",
                title="Bounded Track (Remaster)",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


@pytest.mark.parametrize(
    "candidate_title",
    (
        "One More Time (Deluxe)",
        "One More Time (Deluxe Edition)",
        "One More Time (Super Deluxe Edition)",
        "One More Time (Expanded Edition)",
        "One More Time (Special Edition)",
        "One More Time (Anniversary Edition)",
        "One More Time (1st Anniversary)",
        "One More Time (1st Anniversary Edition)",
        "One More Time (Reissue)",
        "One More Time (Reissued)",
        "One More Time (Extended)",
        "One More Time (Extended Version)",
        "One More Time (Extended Edition)",
    ),
)
async def test_extract_rejects_release_only_track_version_designations(
    candidate_title: str,
) -> None:
    """Reject release-only editions as independently qualifying Track versions."""
    result = await _extract_track_version(
        "One More Time",
        (_spotify_track_payload(title=candidate_title),),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


@pytest.mark.parametrize(
    "blocked_segment",
    (
        "Live at Wembley",
        "2024 Remix",
        "Acoustic",
        "Instrumental",
        "Radio-Edit",
        "Karaoke",
        "A Tribute Performance",
    ),
)
async def test_extract_rejects_blocked_track_version_material(
    blocked_segment: str,
) -> None:
    """Reject blocked Track material before a recognized version suffix."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title=f"One More Time ({blocked_segment}) (Remaster)",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


@pytest.mark.parametrize(
    "candidate_title",
    (
        "One More Times (Remaster)",
        "One More Time (Archive Notes) (Remaster)",
        "One More Time (Remaster Version)",
        "One More Time (11 Remaster)",
        "One More Time (12345 Remaster)",
        "One More Time (Remaster Extra)",
        "One More Time Remaster",
        "Remastered One More Time",
    ),
)
async def test_extract_keeps_track_title_differences_below_fuzzy_threshold_unresolved(
    candidate_title: str,
) -> None:
    """Leave below-threshold Track titles unresolved."""
    result = await _extract_track_version(
        "One More Time",
        (_spotify_track_payload(title=candidate_title),),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


async def test_extract_preserves_arbitrary_track_subtitles_during_version_matching() -> None:
    """Retain equal unrecognized Track subtitles while removing a later version."""
    result = await _extract_track_version(
        "One More Time (Archive Notes)",
        (_spotify_track_payload(title="One More Time (Archive Notes) (Remaster)"),),
    )

    assert result["status"] == "resolved"
    assert result["track"] is not None


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    [
        ("Discovery", "Discovery (Deluxe Edition)"),
        ("Discovery (Deluxe Edition)", "Discovery"),
        ("Discovery [Expanded Edition]", "Discovery - 2011 Remaster"),
        ("Discovery (Extended Edition)", "Discovery"),
    ],
)
async def test_extract_resolves_equivalent_music_release_editions_symmetrically(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Resolve bare and equivalent-edition Music Release titles in either direction."""
    result = await _extract_direct_music_release(
        mention_title,
        (_spotify_album_payload(title=candidate_title),),
    )

    assert result["status"] == "resolved"
    assert result["music_release"] is not None


@pytest.mark.parametrize(
    "candidate_title",
    [
        "Discovery: Deluxe Edition [2011 Remaster]",
        "Discovery - Deluxe Edition: 2011 Remaster",
    ],
)
async def test_extract_strips_stacked_music_release_editions_from_the_right(
    candidate_title: str,
) -> None:
    """Resolve stacked Edition segments regardless of mixed supported boundaries."""
    result = await _extract_direct_music_release(
        "Discovery",
        (_spotify_album_payload(title=candidate_title),),
    )

    assert result["status"] == "resolved"
    assert result["music_release"] is not None


async def test_extract_checks_exact_music_release_titles_before_editions() -> None:
    """Choose an exact Music Release after an earlier equivalent-edition Candidate."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="earlier-equivalent",
                title="Discovery (Deluxe Edition)",
            ),
            _spotify_album_payload(
                spotify_album_id="later-exact",
                title="Discovery",
            ),
        ),
    )

    music_release = cast(dict[str, object], result["music_release"])
    assert result["status"] == "resolved"
    assert music_release["spotify_album_id"] == "later-exact"


async def test_extract_keeps_provider_order_for_equivalent_music_release_editions() -> None:
    """Choose the first eligible equivalent edition after an ineligible Candidate."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="ineligible",
                title="Other Album",
            ),
            _spotify_album_payload(
                spotify_album_id="first-edition",
                title="Discovery (Deluxe Edition)",
            ),
            _spotify_album_payload(
                spotify_album_id="second-edition",
                title="Discovery (Expanded Edition)",
            ),
        ),
    )

    music_release = cast(dict[str, object], result["music_release"])
    assert result["status"] == "resolved"
    assert music_release["spotify_album_id"] == "first-edition"


@pytest.mark.parametrize(
    ("candidate_title", "album_type", "expected_status"),
    [
        ("Discovery (Deluxe Edition)", "album", "resolved"),
        ("Discovery (Deluxe Edition)", "single", "resolved"),
        ("Discovery (Deluxe Edition)", "compilation", "unresolved"),
        ("Discovery (Extended)", "single", "resolved"),
        ("Discovery (Extended)", "compilation", "unresolved"),
        ("Discovery [Extended Version]", "single", "resolved"),
        ("Discovery [Extended Version]", "compilation", "unresolved"),
        ("Discovery - Extended Edition", "single", "resolved"),
        ("Discovery - Extended Edition", "compilation", "unresolved"),
    ],
)
async def test_extract_limits_equivalent_music_release_editions_to_albums_and_singles(
    candidate_title: str,
    album_type: AlbumType,
    expected_status: Literal["resolved", "unresolved"],
) -> None:
    """Accept only Album and Single Candidates during equivalent-edition fallback."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                title=candidate_title,
                album_type=album_type,
            ),
        ),
    )

    assert result["status"] == expected_status
    assert (result["music_release"] is not None) is (expected_status == "resolved")


@pytest.mark.parametrize(
    "candidate_title",
    [
        "Discoveries (Deluxe Edition)",
        "Discovery (Deluxe Edition Extra)",
        "Discovery (11 Remaster)",
        "Discovery (12345 Remaster)",
        "Discovery (20 Anniversary)",
        "Discovery (Twentieth Anniversary Edition)",
        "Discovery (Original Motion Picture Soundtrack)",
        "Discovery Deluxe Edition",
    ],
)
async def test_extract_keeps_music_release_title_differences_below_fuzzy_threshold_unresolved(
    candidate_title: str,
) -> None:
    """Leave below-threshold Music Release titles unresolved."""
    result = await _extract_direct_music_release(
        "Discovery",
        (_spotify_album_payload(title=candidate_title),),
    )

    assert result["status"] == "unresolved"
    assert result["music_release"] is None


@pytest.mark.parametrize(
    "blocked_segment",
    [
        "Live at Wembley",
        "2024 Remix",
        "Acoustic",
        "Instrumental",
        "Radio-Edit",
        "Karaoke",
        "A Tribute to Discovery",
        "Greatest Hits",
    ],
)
async def test_extract_rejects_blocked_music_release_edition_material(
    blocked_segment: str,
) -> None:
    """Reject blocked trailing material even when an edition suffix is recognized."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                title=f"Discovery ({blocked_segment}) (Deluxe Edition)",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["music_release"] is None


async def test_extract_scans_past_unrecognized_segments_for_blocked_edition_material() -> None:
    """Reject blocked material left of an unrecognized trailing title segment."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                title="Discovery (Live) (Archive Notes) (Deluxe Edition)",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["music_release"] is None


async def test_extract_preserves_the_public_result_shape_for_equivalent_editions() -> None:
    """Return provider-backed Music Release metadata without edition-match fields."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="provider-edition",
                title="Discovery (Deluxe Edition)",
                artist_names=("Daft Punk", "Provider Guest"),
                release_date="2025-02-26",
                album_type="album",
            ),
        ),
    )

    assert result == {
        "status": "resolved",
        "music_release_mention": {
            "release_title": "Discovery",
            "artists": ["Daft Punk"],
            "release_year": 2001,
        },
        "music_release": {
            "release_title": "Discovery (Deluxe Edition)",
            "artists": [
                {"spotify_artist_id": "artist-0", "name": "Daft Punk"},
                {"spotify_artist_id": "artist-1", "name": "Provider Guest"},
            ],
            "release_date": "2025-02-26",
            "album_type": "album",
            "spotify_album_id": "provider-edition",
            "spotify_url": "https://open.spotify.com/album/provider-edition",
            "cover_url": "https://i.scdn.co/image/direct-primary",
        },
    }


async def test_extract_resolves_the_forever_story_to_first_extended_release() -> None:
    """Choose the first matching Extended Music Release after Artist Credit filtering."""
    result = await _extract_direct_music_release(
        "The Forever Story",
        (
            _spotify_album_payload(
                spotify_album_id="wrong-artist",
                title="The Forever Story (Extended Version)",
                artist_names=("Other Artist",),
            ),
            _spotify_album_payload(
                spotify_album_id="first-extended",
                title="The Forever Story (Extended Version)",
                artist_names=("JID",),
                release_date="2022-10-31",
                album_type="album",
            ),
            _spotify_album_payload(
                spotify_album_id="later-extended",
                title="The Forever Story (Extended Version)",
                artist_names=("JID",),
            ),
        ),
        artists=("JID",),
        release_year=2022,
    )

    assert result == {
        "status": "resolved",
        "music_release_mention": {
            "release_title": "The Forever Story",
            "artists": ["JID"],
            "release_year": 2022,
        },
        "music_release": {
            "release_title": "The Forever Story (Extended Version)",
            "artists": [{"spotify_artist_id": "artist-0", "name": "JID"}],
            "release_date": "2022-10-31",
            "album_type": "album",
            "spotify_album_id": "first-extended",
            "spotify_url": "https://open.spotify.com/album/first-extended",
            "cover_url": "https://i.scdn.co/image/direct-primary",
        },
    }


async def test_extract_resolves_fuzzy_track_with_music_identity_normalization() -> None:
    """Resolve a normalized above-threshold Track title without release context."""
    result = await _extract_track_version(
        "AMÉLIE DREAM",
        (
            _spotify_track_payload(
                spotify_track_id="normalized-fuzzy-track",
                title="Amélie Dreams",
                artist_names=("DAFT PUNK",),
                attached_album_id="unrelated-release",
                attached_album_title="Unrelated Release",
                attached_album_artist_names=("Provider Artist",),
                attached_album_release_date="2025-02-26",
            ),
        ),
    )

    assert result == {
        "status": "resolved",
        "track_mention": {
            "track_title": "AMÉLIE DREAM",
            "artists": ["Daft Punk"],
            "release_title": None,
            "release_year": None,
        },
        "track": {
            "track_title": "Amélie Dreams",
            "artists": [{"spotify_artist_id": "artist-0", "name": "DAFT PUNK"}],
            "spotify_track_id": "normalized-fuzzy-track",
            "spotify_url": "https://open.spotify.com/track/normalized-fuzzy-track",
            "preferred_music_release": {
                "release_title": "Unrelated Release",
                "artists": [
                    {"spotify_artist_id": "artist-0", "name": "Provider Artist"},
                ],
                "release_date": "2025-02-26",
                "album_type": "album",
                "spotify_album_id": "unrelated-release",
                "spotify_url": "https://open.spotify.com/album/unrelated-release",
                "cover_url": "https://i.scdn.co/image/attached-primary",
            },
            "cover_url": "https://i.scdn.co/image/attached-primary",
        },
    }


async def test_extract_prefers_title_normalized_extended_track_context_to_earlier_fuzzy_candidate() -> (
    None
):
    """Choose edition-aware title identity over an earlier fuzzy Track Candidate."""
    result = await _extract_track_version(
        "‘Til I Can’t / Part I",
        (
            _spotify_track_payload(
                spotify_track_id="earlier-fuzzy",
                title="Til I Can / Part I",
                attached_album_id="earlier-fuzzy-release",
                attached_album_title="Artists Choice/Volume 1",
            ),
            _spotify_track_payload(
                spotify_track_id="normalized-extended",
                title="'Til I Can't/Part I",
                attached_album_id="normalized-extended-release",
                attached_album_title="Artist's Choice/Volume 1 (Extended Version)",
                attached_album_release_date="2025-02-26",
            ),
        ),
        release_title="Artist’s Choice / Volume 1",
        release_year=1999,
    )

    assert result == {
        "status": "resolved",
        "track_mention": {
            "track_title": "‘Til I Can’t / Part I",
            "artists": ["Daft Punk"],
            "release_title": "Artist’s Choice / Volume 1",
            "release_year": 1999,
        },
        "track": {
            "track_title": "'Til I Can't/Part I",
            "artists": [{"spotify_artist_id": "artist-0", "name": "Daft Punk"}],
            "spotify_track_id": "normalized-extended",
            "spotify_url": "https://open.spotify.com/track/normalized-extended",
            "preferred_music_release": {
                "release_title": "Artist's Choice/Volume 1 (Extended Version)",
                "artists": [{"spotify_artist_id": "artist-0", "name": "Daft Punk"}],
                "release_date": "2025-02-26",
                "album_type": "album",
                "spotify_album_id": "normalized-extended-release",
                "spotify_url": ("https://open.spotify.com/album/normalized-extended-release"),
                "cover_url": "https://i.scdn.co/image/attached-primary",
            },
            "cover_url": "https://i.scdn.co/image/attached-primary",
        },
    }


async def test_extract_resolves_fuzzy_music_release_with_provider_metadata() -> None:
    """Resolve an above-threshold Music Release with provider-owned values."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="fuzzy-release",
                title="Discoverd",
                artist_names=("DAFT PUNK",),
                release_date="2025-02-26",
            ),
        ),
    )

    assert result == {
        "status": "resolved",
        "music_release_mention": {
            "release_title": "Discovery",
            "artists": ["Daft Punk"],
            "release_year": 2001,
        },
        "music_release": {
            "release_title": "Discoverd",
            "artists": [{"spotify_artist_id": "artist-0", "name": "DAFT PUNK"}],
            "release_date": "2025-02-26",
            "album_type": "album",
            "spotify_album_id": "fuzzy-release",
            "spotify_url": "https://open.spotify.com/album/fuzzy-release",
            "cover_url": "https://i.scdn.co/image/direct-primary",
        },
    }


async def test_extract_prefers_title_normalized_extended_release_to_earlier_fuzzy_candidate() -> (
    None
):
    """Choose edition-aware title identity over an earlier fuzzy Music Release."""
    result = await _extract_direct_music_release(
        "Artist’s Choice / Volume 1",
        (
            _spotify_album_payload(
                spotify_album_id="earlier-fuzzy",
                title="Artists Choice/Volume 1",
            ),
            _spotify_album_payload(
                spotify_album_id="normalized-extended",
                title="Artist's Choice/Volume 1 (Extended Edition)",
            ),
        ),
    )

    music_release = cast(dict[str, object], result["music_release"])
    assert result["status"] == "resolved"
    assert result["music_release_mention"] == {
        "release_title": "Artist’s Choice / Volume 1",
        "artists": ["Daft Punk"],
        "release_year": 2001,
    }
    assert music_release["spotify_album_id"] == "normalized-extended"
    assert music_release["release_title"] == "Artist's Choice/Volume 1 (Extended Edition)"


async def test_extract_rejects_fuzzy_title_at_exact_threshold() -> None:
    """Leave a Track unresolved when normalized title similarity is exactly 80."""
    result = await _extract_track_version(
        "abcdefghij",
        (_spotify_track_payload(title="abcdXYghij"),),
    )

    assert result == {
        "status": "unresolved",
        "track_mention": {
            "track_title": "abcdefghij",
            "artists": ["Daft Punk"],
            "release_title": None,
            "release_year": None,
        },
        "track": None,
    }


async def test_extract_requires_artist_credit_match_for_fuzzy_resolution() -> None:
    """Reject an above-threshold Track title without an Artist Credit Match."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title="One More Tiem",
                artist_names=("Daft Punks",),
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


async def test_extract_resolves_father_and_son_with_candidate_artist_alias() -> None:
    """Resolve a Track when the Candidate credit contains a spaced-slash alias."""
    result = await _extract_track_version(
        "Father and Son",
        (
            _spotify_track_payload(
                spotify_track_id="father-and-son",
                title="Father and Son",
                artist_names=("Yusuf / Cat Stevens", "Additional Artist"),
                attached_album_id="tea-for-the-tillerman",
                attached_album_title="Tea for the Tillerman",
                attached_album_artist_names=("Yusuf / Cat Stevens",),
                attached_album_release_date="1970",
                attached_album_type="album",
            ),
        ),
        artists=("Cat Stevens",),
    )

    assert result == {
        "status": "resolved",
        "track_mention": {
            "track_title": "Father and Son",
            "artists": ["Cat Stevens"],
            "release_title": None,
            "release_year": None,
        },
        "track": {
            "track_title": "Father and Son",
            "artists": [
                {"spotify_artist_id": "artist-0", "name": "Yusuf / Cat Stevens"},
                {"spotify_artist_id": "artist-1", "name": "Additional Artist"},
            ],
            "spotify_track_id": "father-and-son",
            "spotify_url": "https://open.spotify.com/track/father-and-son",
            "preferred_music_release": {
                "release_title": "Tea for the Tillerman",
                "artists": [
                    {"spotify_artist_id": "artist-0", "name": "Yusuf / Cat Stevens"},
                ],
                "release_date": "1970",
                "album_type": "album",
                "spotify_album_id": "tea-for-the-tillerman",
                "spotify_url": ("https://open.spotify.com/album/tea-for-the-tillerman"),
                "cover_url": "https://i.scdn.co/image/attached-primary",
            },
            "cover_url": "https://i.scdn.co/image/attached-primary",
        },
    }


async def test_extract_resolves_music_release_with_candidate_artist_alias() -> None:
    """Resolve a Music Release with a normalized spaced-slash Candidate alias."""
    result = await _extract_direct_music_release(
        "Tea for the Tillerman",
        (
            _spotify_album_payload(
                spotify_album_id="tea-for-the-tillerman",
                title="Tea for the Tillerman",
                artist_names=("Yusuf / CAT   STEVENS", "Additional Artist"),
                release_date="1970",
                album_type="album",
            ),
        ),
        artists=("Cat Stevens",),
        release_year=1970,
    )

    assert result == {
        "status": "resolved",
        "music_release_mention": {
            "release_title": "Tea for the Tillerman",
            "artists": ["Cat Stevens"],
            "release_year": 1970,
        },
        "music_release": {
            "release_title": "Tea for the Tillerman",
            "artists": [
                {"spotify_artist_id": "artist-0", "name": "Yusuf / CAT   STEVENS"},
                {"spotify_artist_id": "artist-1", "name": "Additional Artist"},
            ],
            "release_date": "1970",
            "album_type": "album",
            "spotify_album_id": "tea-for-the-tillerman",
            "spotify_url": "https://open.spotify.com/album/tea-for-the-tillerman",
            "cover_url": "https://i.scdn.co/image/direct-primary",
        },
    }


async def test_extract_preserves_complete_spaced_slash_artist_credit_equality() -> None:
    """Resolve a normalized-equal complete Candidate credit before alias splitting."""
    result = await _extract_direct_music_release(
        "Alias Collection",
        (
            _spotify_album_payload(
                title="Alias Collection",
                artist_names=("YUSUF / CAT STEVENS",),
            ),
        ),
        artists=("Yusuf / Cat Stevens",),
    )

    assert result == {
        "status": "resolved",
        "music_release_mention": {
            "release_title": "Alias Collection",
            "artists": ["Yusuf / Cat Stevens"],
            "release_year": 2001,
        },
        "music_release": {
            "release_title": "Alias Collection",
            "artists": [
                {"spotify_artist_id": "artist-0", "name": "YUSUF / CAT STEVENS"},
            ],
            "release_date": "2001-02",
            "album_type": "album",
            "spotify_album_id": "direct-album",
            "spotify_url": "https://open.spotify.com/album/direct-album",
            "cover_url": "https://i.scdn.co/image/direct-primary",
        },
    }


@pytest.mark.parametrize(
    ("mention_artist", "candidate_artist"),
    (
        ("Yusuf / Cat Stevens", "Cat Stevens"),
        ("Cat Stevens", "Yusuf/Cat Stevens"),
        ("AC", "AC/DC"),
        ("Cat Stevens", "Yusuf & Cat Stevens"),
        ("Cat Stevens", "Yusuf and Cat Stevens"),
        ("Cat Stevens", "Yusuf, Cat Stevens"),
        ("Stevens", "Yusuf / Cat Stevens"),
        ("Cat Stevens", "Cat Steven"),
        ("Cat Stevens", "Stevens Cat"),
        ("Cat Stevens", "Kat Stevens"),
    ),
)
async def test_extract_rejects_non_alias_artist_credit_forms(
    mention_artist: str,
    candidate_artist: str,
) -> None:
    """Reject relations outside complete Candidate-credit spaced-slash aliases."""
    result = await _extract_direct_music_release(
        "Alias Collection",
        (
            _spotify_album_payload(
                title="Alias Collection",
                artist_names=(candidate_artist,),
            ),
        ),
        artists=(mention_artist,),
    )

    assert result == {
        "status": "unresolved",
        "music_release_mention": {
            "release_title": "Alias Collection",
            "artists": [mention_artist],
            "release_year": 2001,
        },
        "music_release": None,
    }


@pytest.mark.parametrize(
    ("mention_artist", "candidate_artist"),
    (
        ("D’Angelo", "D'Angelo"),
        ("Artist / One", "Artist/One"),
    ),
)
async def test_extract_keeps_title_identity_out_of_artist_credit_matching(
    mention_artist: str,
    candidate_artist: str,
) -> None:
    """Reject title-equivalent releases with distinct normalized Artist Credits."""
    result = await _extract_direct_music_release(
        "Artist’s Choice / Volume 1",
        (
            _spotify_album_payload(
                title="Artist's Choice/Volume 1",
                artist_names=(candidate_artist,),
            ),
        ),
        artists=(mention_artist,),
    )

    assert result["status"] == "unresolved"
    assert result["music_release"] is None


@pytest.mark.parametrize(
    ("candidate_track_title", "candidate_release_title", "expected_status"),
    (
        ("One More Tiem", "Discoverd", "resolved"),
        ("One More Tiem", "Other Album", "unresolved"),
        ("One More", "Discoverd", "unresolved"),
    ),
)
async def test_extract_requires_every_explicit_track_context_title_to_pass_fuzzy_threshold(
    candidate_track_title: str,
    candidate_release_title: str,
    expected_status: Literal["resolved", "unresolved"],
) -> None:
    """Require above-threshold Track and attached Music Release titles."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                title=candidate_track_title,
                attached_album_title=candidate_release_title,
                attached_album_release_date="2025-02-26",
            ),
        ),
        release_title="Discovery",
        release_year=1999,
    )

    assert result["status"] == expected_status
    if expected_status == "unresolved":
        assert result["track"] is None
        return
    track = cast(dict[str, object], result["track"])
    preferred_music_release = cast(dict[str, object], track["preferred_music_release"])
    assert preferred_music_release["release_date"] == "2025-02-26"


@pytest.mark.parametrize(
    ("later_title", "expected_spotify_track_id"),
    (
        ("One More Time", "later-exact"),
        ("One More Time (Remaster)", "later-edition"),
    ),
)
async def test_extract_prefers_earlier_track_resolution_stages_to_fuzzy(
    later_title: str,
    expected_spotify_track_id: str,
) -> None:
    """Choose a later exact or edition Track before an earlier fuzzy Candidate."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id="earlier-fuzzy",
                title="One More Tiem",
            ),
            _spotify_track_payload(
                spotify_track_id=expected_spotify_track_id,
                title=later_title,
            ),
        ),
    )

    track = cast(dict[str, object], result["track"])
    assert result["status"] == "resolved"
    assert track["spotify_track_id"] == expected_spotify_track_id


@pytest.mark.parametrize(
    ("later_title", "expected_spotify_album_id"),
    (
        ("Discovery", "later-exact"),
        ("Discovery (Deluxe Edition)", "later-edition"),
    ),
)
async def test_extract_prefers_earlier_music_release_resolution_stages_to_fuzzy(
    later_title: str,
    expected_spotify_album_id: str,
) -> None:
    """Choose a later exact or edition release before an earlier fuzzy Candidate."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="earlier-fuzzy",
                title="Discoverd",
            ),
            _spotify_album_payload(
                spotify_album_id=expected_spotify_album_id,
                title=later_title,
            ),
        ),
    )

    music_release = cast(dict[str, object], result["music_release"])
    assert result["status"] == "resolved"
    assert music_release["spotify_album_id"] == expected_spotify_album_id


async def test_extract_keeps_provider_order_for_fuzzy_track_candidates() -> None:
    """Choose the first provider-ordered above-threshold fuzzy Track."""
    result = await _extract_track_version(
        "One More Time",
        (
            _spotify_track_payload(
                spotify_track_id="first-fuzzy",
                title="The One More Time",
            ),
            _spotify_track_payload(
                spotify_track_id="second-fuzzy",
                title="One More Times",
            ),
        ),
    )

    track = cast(dict[str, object], result["track"])
    assert result["status"] == "resolved"
    assert track["spotify_track_id"] == "first-fuzzy"


async def test_extract_keeps_provider_order_for_fuzzy_music_release_candidates() -> None:
    """Choose the first provider-ordered above-threshold fuzzy Music Release."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                spotify_album_id="first-fuzzy",
                title="The Discovery",
            ),
            _spotify_album_payload(
                spotify_album_id="second-fuzzy",
                title="Discoveryy",
            ),
        ),
    )

    music_release = cast(dict[str, object], result["music_release"])
    assert result["status"] == "resolved"
    assert music_release["spotify_album_id"] == "first-fuzzy"


async def test_extract_ignores_fourth_fuzzy_track_candidate_without_another_search() -> None:
    """Leave a fourth above-threshold Track outside the bounded provider result set."""
    result = await _extract_track_version(
        "Bounded Track",
        (
            _spotify_track_payload(title="Wrong One"),
            _spotify_track_payload(title="Wrong Two"),
            _spotify_track_payload(title="Wrong Three"),
            _spotify_track_payload(
                spotify_track_id="fourth-fuzzy",
                title="Bounded Trak",
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["track"] is None


async def test_extract_allows_recording_version_material_in_fuzzy_track_resolution() -> None:
    """Resolve complete above-threshold Track titles containing remix material."""
    result = await _extract_track_version(
        "One More Time (Remix)",
        (_spotify_track_payload(title="One More Tiem (Remix)"),),
    )

    assert result["status"] == "resolved"
    assert result["track"] is not None


async def test_extract_allows_edition_material_in_fuzzy_music_release_resolution() -> None:
    """Resolve a fuzzy compilation title containing an edition designation."""
    result = await _extract_direct_music_release(
        "Discovery (Deluxe Edition)",
        (
            _spotify_album_payload(
                title="Discoverd (Deluxe Edition)",
                album_type="compilation",
            ),
        ),
    )

    assert result["status"] == "resolved"
    music_release = cast(dict[str, object], result["music_release"])
    assert music_release["album_type"] == "compilation"


async def test_extract_returns_atomic_catalog_failure_without_music_results() -> None:
    """Map a Spotify search failure to the existing provider-error response."""
    interpretation_response = _interpretation_response(
        tracks=[
            {
                "track_title": "One More Time",
                "artists": ["Daft Punk"],
                "release_title": None,
                "release_year": None,
            }
        ],
        music_releases=[],
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "accounts.spotify.test":
            return httpx.Response(
                200,
                json={"access_token": "test-access-token", "expires_in": 3600},
            )
        assert request.method == "GET"
        return httpx.Response(503)

    response, interpretation_provider, http_client = await _post_extract(
        interpretation_response,
        httpx.MockTransport(handle),
    )

    assert interpretation_provider.closed is True
    assert http_client.is_closed is True
    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "catalog_provider_failed",
            "message": "Spotify catalog request failed.",
        }
    }
