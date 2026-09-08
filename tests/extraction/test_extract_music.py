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
        "track_title": "One More Time",
        "artists": [
            {"spotify_artist_id": "artist-0", "name": "romanthony"},
            {"spotify_artist_id": "artist-1", "name": "Additional Artist"},
        ],
        "spotify_track_id": "playable-track",
        "spotify_url": "https://open.spotify.com/track/playable-track",
        "preferred_music_release": {
            "release_title": "Discovery",
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
        "release_title": "Discovery",
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
                    title="Exact Albums",
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
        "near-title",
        "first-exact-candidate",
        "fourth-candidate-is-ignored",
    ],
)
async def test_extract_applies_the_exact_music_resolution_matrix(
    scenario: _ExactResolutionScenario,
) -> None:
    """Expose exact-only Music resolution and the first-three Candidate bound."""
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
) -> dict[str, object]:
    """Resolve one direct Music Release through the in-process extraction endpoint."""
    music_release_mention = {
        "release_title": release_title,
        "artists": ["Daft Punk"],
        "release_year": 2001,
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
            "q": f"album:{release_title} artist:Daft Punk",
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


@pytest.mark.parametrize(
    ("candidate_title", "spotify_album_id"),
    [
        ("Discovery (Remaster)", "remaster"),
        ("Discovery [Remastered]", "remastered"),
        ("Discovery - Remastered Version", "remastered-version"),
        ("Discovery: 2011 Remaster", "year-remaster"),
        ("Discovery (Remastered 2011)", "remastered-year"),
        ("Discovery [2011 Remastered Version]", "year-remastered-version"),
        ("Discovery - Deluxe", "deluxe"),
        ("Discovery: Deluxe Edition", "deluxe-edition"),
        ("Discovery (sUPER   dELUXE   Edition)", "super-deluxe"),
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
    ],
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
    ("mention_title", "candidate_title"),
    [
        ("Discovery", "Discovery (Deluxe Edition)"),
        ("Discovery (Deluxe Edition)", "Discovery"),
        ("Discovery [Expanded Edition]", "Discovery - 2011 Remaster"),
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
    ("album_type", "expected_status"),
    [
        ("album", "resolved"),
        ("single", "resolved"),
        ("compilation", "unresolved"),
    ],
)
async def test_extract_limits_equivalent_music_release_editions_to_albums_and_singles(
    album_type: AlbumType,
    expected_status: Literal["resolved", "unresolved"],
) -> None:
    """Accept only Album and Single Candidates during equivalent-edition fallback."""
    result = await _extract_direct_music_release(
        "Discovery",
        (
            _spotify_album_payload(
                title="Discovery (Deluxe Edition)",
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
async def test_extract_rejects_uncontrolled_music_release_title_differences(
    candidate_title: str,
) -> None:
    """Leave non-equivalent Music Release titles unresolved without fuzzy matching."""
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
