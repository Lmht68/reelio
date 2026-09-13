"""In-process HTTP coverage for Open Library-backed Book Work extraction."""

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from time import monotonic
from typing import cast

import httpx
import pytest

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.router import get_pipeline
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.enrichment.open_library import OpenLibraryBookResolver
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
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
from tests.extraction.fakes import FakeMusicResolver, FakeScreenWorkResolver

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
                title="Book review",
                description="A review of Pride and Prejudice.",
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
            text="Pride and Prejudice by Jane Austen is a classic.",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        )


class _InterpretationProvider:
    """Return one configured strict interpretation response without network I/O."""

    def __init__(self, response: dict[str, object]) -> None:
        """Initialize a provider with one observable response."""
        self._response = response
        self.calls: list[tuple[LLMMessage, ...]] = []

    @property
    def provider_name(self) -> LLMProvider:
        """Return a stable test provider identity."""
        return LLMProvider.DEEPSEEK

    @property
    def model_name(self) -> str:
        """Return a stable test model identity."""
        return "deterministic-book-provider"

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return the configured strict interpretation response."""
        self.calls.append(tuple(messages))
        return json.dumps(self._response)

    async def aclose(self) -> None:
        """Satisfy the pipeline-owned interpretation-provider lifecycle."""
        return None


def _interpretation_settings() -> InterpretationConfig:
    settings_type = cast(Callable[..., InterpretationConfig], InterpretationConfig)
    return settings_type(_env_file=None)


def _interpretation_response(books: list[dict[str, object]]) -> dict[str, object]:
    """Return a complete strict interpretation response with supplied Book Mentions."""
    return {
        "movies": [],
        "tv_series": [],
        "tracks": [],
        "music_releases": [],
        "books": books,
    }


async def _post_extract(
    interpretation_response: dict[str, object],
    open_library_transport: httpx.AsyncBaseTransport,
) -> tuple[httpx.Response, _InterpretationProvider, httpx.AsyncClient]:
    """Run the real Book extraction pipeline through the HTTP endpoint and close owners."""
    http_client = httpx.AsyncClient(
        base_url="https://openlibrary.test/",
        transport=open_library_transport,
        headers={"User-Agent": "Reelio (test@example.invalid)"},
    )
    book_resolver = OpenLibraryBookResolver(
        http_client,
        3.0,
        monotonic,
        asyncio.sleep,
    )
    interpretation_provider = _InterpretationProvider(interpretation_response)
    pipeline = ExtractionPipeline(
        _MetadataService(),
        _TranscriptionService(),
        MentionInterpretationService(interpretation_provider, _interpretation_settings()),
        ExtractionResultAggregator(
            FakeScreenWorkResolver(),
            FakeMusicResolver(),
            book_resolver,
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

    return response, interpretation_provider, http_client


def _search_response(*candidates: dict[str, object]) -> dict[str, object]:
    """Build one minimal Open Library Search response."""
    return {"docs": list(candidates)}


async def test_extract_returns_resolved_and_unresolved_exact_book_works() -> None:
    """Expose deduplicated exact Book Work results in first-reference order."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params["title"] == "Pride and Prejudice":
            return httpx.Response(
                200,
                json=_search_response(
                    {
                        "key": "/works/OL66554W",
                        "title": "Pride and Prejudice",
                        "author_key": ["OL21594A"],
                        "author_name": ["Jane Austen"],
                    }
                ),
            )
        return httpx.Response(200, json=_search_response())

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(
            [
                {"title": "Pride and Prejudice", "authors": ["Jane Austen"]},
                {"title": "Pride and Prejudice", "authors": ["Jane Austen"]},
                {"title": "Unknown Book", "authors": []},
            ]
        ),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert len(interpretation_provider.calls) == 1
    assert http_client.is_closed is True
    assert [request.url.params["title"] for request in requests] == [
        "Pride and Prejudice",
        "Unknown Book",
    ]
    payload = response.json()
    assert payload["market"] == "JP"
    assert payload["results"]["movies"] == []
    assert payload["results"]["tv_series"] == []
    assert payload["results"]["tracks"] == []
    assert payload["results"]["music_releases"] == []
    assert payload["statistics"]["books"] == {
        "n_mentions": 2,
        "n_resolved": 1,
        "n_unresolved": 1,
    }
    assert payload["results"]["books"] == [
        {
            "status": "resolved",
            "book_mention": {
                "title": "Pride and Prejudice",
                "authors": ["Jane Austen"],
            },
            "book": {
                "title": "Pride and Prejudice",
                "authors": [
                    {
                        "open_library_author_id": "OL21594A",
                        "name": "Jane Austen",
                        "open_library_url": "https://openlibrary.org/authors/OL21594A",
                    }
                ],
                "open_library_work_id": "OL66554W",
                "open_library_url": "https://openlibrary.org/works/OL66554W",
            },
        },
        {
            "status": "unresolved",
            "book_mention": {"title": "Unknown Book", "authors": []},
            "book": None,
        },
    ]


async def test_extract_maps_open_library_failure_to_catalog_provider_error() -> None:
    """Map an Open Library provider failure through the stable public error envelope."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    response, _, _ = await _post_extract(
        _interpretation_response([{"title": "Pride and Prejudice", "authors": ["Jane Austen"]}]),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "catalog_provider_failed",
            "message": "Open Library catalog request failed.",
        }
    }


async def test_extract_maps_open_library_timeout_to_pipeline_timeout() -> None:
    """Map an Open Library timeout through the stable public error envelope."""

    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    response, _, _ = await _post_extract(
        _interpretation_response([{"title": "Pride and Prejudice", "authors": ["Jane Austen"]}]),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 504
    assert response.json() == {
        "error": {
            "code": "pipeline_timeout",
            "message": "Open Library catalog request timed out.",
        }
    }
