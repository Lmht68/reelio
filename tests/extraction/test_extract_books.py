"""In-process HTTP coverage for Open Library-backed Book Work extraction."""

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from time import monotonic
from typing import cast

import httpx
import pytest

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.router import get_pipeline
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.enrichment.config import OpenLibraryConfig
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
_WORK_SEARCH_FIELDS = (
    "key,title,alternative_title,author_key,author_name,"
    "author_alternative_name,editions,editions.key,editions.title,cover_i,"
    "cover_edition_key"
)
_EDITION_SEARCH_FIELDS = (
    "key,editions,editions.key,editions.title,editions.format,"
    "editions.publish_year,editions.cover_i"
)


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

    def __init__(self, text: str, language: str) -> None:
        """Initialize deterministic transcript material for one request."""
        self._text = text
        self._language = language

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
            text=self._text,
            language=self._language,
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        )


class _AbsentPreferredEditionTransport(httpx.AsyncBaseTransport):
    """Return nullable Edition Search data around existing resolver fixtures."""

    def __init__(self, delegate: httpx.AsyncBaseTransport) -> None:
        """Initialize the transport wrapping legacy Work-resolution responses."""
        self._delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Return absent preferred Editions or delegate Work-resolution requests."""
        if _is_preferred_edition_search(request):
            return httpx.Response(200, json={"docs": []})
        return await self._delegate.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped provider transport."""
        await self._delegate.aclose()


def _is_preferred_edition_search(request: httpx.Request) -> bool:
    """Return whether the provider request selects a Work's preferred Edition."""
    return request.url.path == "/search.json" and "q" in request.url.params


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


def _open_library_settings(**values: object) -> OpenLibraryConfig:
    settings_type = cast(Callable[..., OpenLibraryConfig], OpenLibraryConfig)
    return settings_type(
        _env_file=None,
        contact_email="catalog-contact@example.invalid",
        **values,
    )


class _FakeClock:
    """Advance deterministic monotonic time through resolver request waits."""

    def __init__(self) -> None:
        """Initialize the deterministic monotonic clock."""
        self.value = 0.0

    def __call__(self) -> float:
        """Return the current deterministic monotonic time."""
        return self.value

    async def sleep(self, delay_seconds: float) -> None:
        """Advance deterministic monotonic time by a requested delay."""
        self.value += delay_seconds


async def _post_extract(
    interpretation_response: dict[str, object],
    open_library_transport: httpx.AsyncBaseTransport,
    *,
    transcript_text: str = "Pride and Prejudice by Jane Austen is a classic.",
    transcript_language: str = "en",
    default_missing_preferred_edition: bool = True,
    clock: _FakeClock | None = None,
) -> tuple[httpx.Response, _InterpretationProvider, httpx.AsyncClient]:
    """Run the real Book extraction pipeline through the HTTP endpoint and close owners."""
    catalog_transport = (
        _AbsentPreferredEditionTransport(open_library_transport)
        if default_missing_preferred_edition
        else open_library_transport
    )
    http_client = httpx.AsyncClient(
        base_url="https://openlibrary.test/",
        transport=catalog_transport,
        headers={"User-Agent": "Reelio (test@example.invalid)"},
    )
    book_resolver = OpenLibraryBookResolver(
        http_client,
        _open_library_settings(),
        clock=clock or monotonic,
        sleep=clock.sleep if clock is not None else asyncio.sleep,
    )
    interpretation_provider = _InterpretationProvider(interpretation_response)
    pipeline = ExtractionPipeline(
        _MetadataService(),
        _TranscriptionService(transcript_text, transcript_language),
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
                "/api/extractions",
                json={"url": _CANONICAL_URL, "market": "JP"},
            )
    finally:
        await pipeline.aclose()

    return response, interpretation_provider, http_client


def _search_response(*candidates: dict[str, object]) -> dict[str, object]:
    """Build one minimal Open Library Search response."""
    return {"docs": list(candidates)}


def _work_record(work_id: str) -> dict[str, object]:
    """Build one Open Library terminal Work identity response."""
    return {"type": {"key": "/type/work"}, "key": f"/works/{work_id}"}


def _terminal_work_response(request: httpx.Request) -> httpx.Response:
    """Return a terminal Work identity record for a Work lookup request."""
    work_id = request.url.path.removeprefix("/works/").removesuffix(".json")
    return httpx.Response(200, json=_work_record(work_id))


async def test_extract_returns_resolved_and_unresolved_exact_book_works() -> None:
    """Expose deduplicated exact Book Work results in first-reference order."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
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
    searches = [request for request in requests if request.url.path == "/search.json"]
    assert [request.url.params["title"] for request in searches] == [
        "Pride and Prejudice",
        "Unknown Book",
    ]
    assert searches[0].url.params["author"] == "Jane Austen"
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
                "edition": None,
                "cover_url": None,
                "cover_edition_id": None,
            },
        },
        {
            "status": "unresolved",
            "book_mention": {"title": "Unknown Book", "authors": []},
            "book": None,
        },
    ]


async def test_extract_resolves_long_ships_subtitle_equivalence_through_http() -> None:
    """Resolve one subtitle-equivalent Work and retain an ambiguous main title."""

    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            assert request.url.params["q"] in {
                "key:/works/OL1W AND language:eng",
                "key:/works/OL1W",
            }
            return httpx.Response(200, json=_search_response())
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if request.url.params["title"] == "The Long Ships":
            assert request.url.params["author"] == "Frans G. Bengtsson"
            return httpx.Response(
                200,
                json=_search_response(
                    {
                        "key": "/works/OL1W",
                        "title": "The Long Ships: A Saga of the Viking Age",
                        "author_key": ["OL1A"],
                        "author_name": ["Frans G. Bengtsson"],
                    }
                ),
            )
        if request.url.params["title"] == "Shared Main":
            assert "author" not in request.url.params
            return httpx.Response(
                200,
                json=_search_response(
                    {
                        "key": "/works/OL2W",
                        "title": "Shared Main: First",
                    },
                    {
                        "key": "/works/OL3W",
                        "title": "Shared Main - Second",
                    },
                ),
            )
        raise AssertionError(f"unexpected Open Library request: {request.url}")

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(
            [
                {
                    "title": "The Long Ships",
                    "authors": ["Frans G. Bengtsson"],
                },
                {"title": "Shared Main", "authors": []},
            ]
        ),
        httpx.MockTransport(handle),
        transcript_text=("The Long Ships by Frans G. Bengtsson appears before Shared Main."),
        default_missing_preferred_edition=False,
    )

    assert response.status_code == 200
    assert len(interpretation_provider.calls) == 1
    assert http_client.is_closed is True
    assert len(requests) == 7
    work_searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and not _is_preferred_edition_search(request)
    ]
    assert Counter(
        tuple(sorted(request.url.params.items())) for request in work_searches
    ) == Counter(
        {
            (
                ("author", "Frans G. Bengtsson"),
                ("fields", _WORK_SEARCH_FIELDS),
                ("limit", "5"),
                ("title", "The Long Ships"),
            ): 1,
            (
                ("fields", _WORK_SEARCH_FIELDS),
                ("limit", "5"),
                ("title", "Shared Main"),
            ): 1,
        }
    )
    assert Counter(
        request.url.path for request in requests if request.url.path.startswith("/works/")
    ) == Counter(
        {
            "/works/OL1W.json": 1,
            "/works/OL2W.json": 1,
            "/works/OL3W.json": 1,
        }
    )
    preferred_edition_searches = [
        request for request in requests if _is_preferred_edition_search(request)
    ]
    assert Counter(
        tuple(sorted(request.url.params.items())) for request in preferred_edition_searches
    ) == Counter(
        {
            (
                ("fields", _EDITION_SEARCH_FIELDS),
                ("limit", "1"),
                ("q", "key:/works/OL1W AND language:eng"),
            ): 1,
            (
                ("fields", _EDITION_SEARCH_FIELDS),
                ("limit", "1"),
                ("q", "key:/works/OL1W"),
            ): 1,
        }
    )

    payload = response.json()
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
                "title": "The Long Ships",
                "authors": ["Frans G. Bengtsson"],
            },
            "book": {
                "title": "The Long Ships: A Saga of the Viking Age",
                "authors": [
                    {
                        "open_library_author_id": "OL1A",
                        "name": "Frans G. Bengtsson",
                        "open_library_url": "https://openlibrary.org/authors/OL1A",
                    }
                ],
                "open_library_work_id": "OL1W",
                "open_library_url": "https://openlibrary.org/works/OL1W",
                "edition": None,
                "cover_url": None,
                "cover_edition_id": None,
            },
        },
        {
            "status": "unresolved",
            "book_mention": {"title": "Shared Main", "authors": []},
            "book": None,
        },
    ]


async def test_extract_resolves_fuzzy_authorful_books_and_leaves_authorless_ambiguity() -> None:
    """Expose bounded resolver policy through the production-shaped HTTP endpoint."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        title = request.url.params["title"]
        if title == "Pride and Prejudce":
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
        if title == "Shared Title":
            return httpx.Response(
                200,
                json=_search_response(
                    {
                        "key": "/works/OL1W",
                        "title": "Shared Title",
                        "author_key": ["OL1A"],
                        "author_name": ["First Author"],
                    },
                    {
                        "key": "/works/OL2W",
                        "title": "Shared Title",
                        "author_key": ["OL2A"],
                        "author_name": ["Second Author"],
                    },
                ),
            )
        raise AssertionError(f"unexpected Search request: {request.url}")

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(
            [
                {"title": "Pride and Prejudce", "authors": ["Jane Austen"]},
                {"title": "Shared Title", "authors": []},
            ]
        ),
        httpx.MockTransport(handle),
    )

    assert response.status_code == 200
    assert len(interpretation_provider.calls) == 1
    assert http_client.is_closed is True
    searches = [request for request in requests if request.url.path == "/search.json"]
    assert [
        (request.url.params["title"], request.url.params.get("author")) for request in searches
    ] == [
        ("Pride and Prejudce", "Jane Austen"),
        ("Shared Title", None),
    ]
    payload = response.json()
    assert payload["statistics"]["books"] == {
        "n_mentions": 2,
        "n_resolved": 1,
        "n_unresolved": 1,
    }
    assert payload["results"]["books"] == [
        {
            "status": "resolved",
            "book_mention": {
                "title": "Pride and Prejudce",
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
                "edition": None,
                "cover_url": None,
                "cover_edition_id": None,
            },
        },
        {
            "status": "unresolved",
            "book_mention": {"title": "Shared Title", "authors": []},
            "book": None,
        },
    ]


async def test_extract_retries_open_library_5xx_before_returning_unresolved_book() -> None:
    """Retry a transient catalog failure through the production-shaped endpoint."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if len(dispatch_times) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_search_response())

    response, _, http_client = await _post_extract(
        _interpretation_response([{"title": "Unknown Book", "authors": []}]),
        httpx.MockTransport(handle),
        clock=clock,
    )

    assert response.status_code == 200
    assert dispatch_times == pytest.approx([0.0, 1 / 3])
    assert response.json()["statistics"]["books"] == {
        "n_mentions": 1,
        "n_resolved": 0,
        "n_unresolved": 1,
    }
    assert http_client.is_closed is True


async def test_extract_maps_open_library_failure_to_catalog_provider_error() -> None:
    """Map an exhausted Open Library retry through the stable public error envelope."""
    clock = _FakeClock()
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    response, _, http_client = await _post_extract(
        _interpretation_response([{"title": "Pride and Prejudice", "authors": ["Jane Austen"]}]),
        httpx.MockTransport(handle),
        clock=clock,
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "catalog_provider_failed",
            "message": "Open Library catalog request failed.",
        }
    }
    assert len(requests) == 2
    assert http_client.is_closed is True


async def test_extract_maps_open_library_timeout_to_pipeline_timeout() -> None:
    """Map an Open Library timeout through the stable public error envelope."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    response, _, http_client = await _post_extract(
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
    assert len(requests) == 1
    assert http_client.is_closed is True


async def test_extract_exposes_provider_preferred_editions_independent_of_source_signals() -> None:
    """Keep Edition selection English-first and Work-ID-only through the HTTP endpoint."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            query = request.url.params["q"]
            if query == "key:/works/OL1W AND language:eng":
                return httpx.Response(
                    200,
                    json={
                        "docs": [
                            {
                                "key": "/works/OL1W",
                                "editions": {
                                    "docs": [
                                        {
                                            "key": "/books/OL1001M",
                                            "title": "Search Edition Title",
                                            "format": ["Print"],
                                            "publish_year": [1995],
                                            "cover_i": 101,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                )
            if query == "key:/works/OL2W AND language:eng":
                return httpx.Response(
                    200,
                    json={
                        "docs": [
                            {
                                "key": "/works/OL2W",
                                "editions": {
                                    "docs": [
                                        {
                                            "key": "/books/OL1002M",
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                )
            raise AssertionError(f"unexpected preferred Edition Search: {request.url}")
        if request.url.path == "/books/OL1001M.json":
            return httpx.Response(
                200,
                json={
                    "key": "/books/OL1001M",
                    "title": "Provider Edition Title",
                    "publishers": ["Example Press", " Example Press "],
                    "isbn_10": ["0123456789", "not-an-isbn"],
                    "isbn_13": ["9780123456786", "9780123456786"],
                    "publish_date": "1995",
                    "covers": [0, 901],
                },
            )
        if request.url.path == "/books/OL1002M.json":
            return httpx.Response(
                200,
                json={
                    "key": "/books/OL1002M",
                },
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        title = request.url.params["title"]
        work_id = {
            "Interpreted First Mention": "OL1W",
            "Interpreted Second Mention": "OL2W",
        }[title]
        cover_id, cover_edition_key = {
            "OL1W": (801, "OL2001M"),
            "OL2W": (802, "OL2002M"),
        }[work_id]
        return httpx.Response(
            200,
            json=_search_response(
                {
                    "key": f"/works/{work_id}",
                    "title": title,
                    "cover_i": cover_id,
                    "cover_edition_key": cover_edition_key,
                }
            ),
        )

    response, interpretation_provider, http_client = await _post_extract(
        _interpretation_response(
            [
                {"title": "Interpreted First Mention", "authors": []},
                {"title": "Interpreted Second Mention", "authors": []},
            ]
        ),
        httpx.MockTransport(handle),
        transcript_text=(
            "Japanese narration discusses a French paperback audiobook from 2001 with "
            "ISBN 9780123456786."
        ),
        transcript_language="ja",
        default_missing_preferred_edition=False,
    )

    assert response.status_code == 200
    assert len(interpretation_provider.calls) == 1
    assert http_client.is_closed is True
    payload = response.json()
    assert payload["market"] == "JP"
    assert payload["transcript"]["language"] == "ja"
    first_book = payload["results"]["books"][0]
    assert first_book["status"] == "resolved"
    assert first_book["book_mention"]["title"] == "Interpreted First Mention"
    assert first_book["book"]["title"] == "Interpreted First Mention"
    assert first_book["book"]["edition"] == {
        "title": "Provider Edition Title",
        "publication_year": 1995,
        "publishers": ["Example Press", " Example Press "],
        "isbn_10": ["0123456789", "not-an-isbn"],
        "isbn_13": ["9780123456786", "9780123456786"],
        "open_library_edition_id": "OL1001M",
        "open_library_url": "https://openlibrary.org/books/OL1001M",
        "cover_url": "https://covers.openlibrary.org/b/id/901-L.jpg",
    }
    assert first_book["book"]["cover_url"] == "https://covers.openlibrary.org/b/id/901-L.jpg"
    assert first_book["book"]["cover_edition_id"] == "OL1001M"
    assert first_book["book"]["edition"]["title"] != first_book["book_mention"]["title"]
    assert first_book["book"]["edition"]["title"] != first_book["book"]["title"]
    second_book = payload["results"]["books"][1]
    assert second_book["status"] == "resolved"
    assert second_book["book_mention"]["title"] == "Interpreted Second Mention"
    assert second_book["book"]["edition"] == {
        "title": None,
        "publication_year": None,
        "publishers": [],
        "isbn_10": [],
        "isbn_13": [],
        "open_library_edition_id": "OL1002M",
        "open_library_url": "https://openlibrary.org/books/OL1002M",
        "cover_url": None,
    }
    assert second_book["book"]["cover_url"] == "https://covers.openlibrary.org/b/id/802-L.jpg"
    assert second_book["book"]["cover_edition_id"] == "OL2002M"
    assert {
        request.url.params["q"] for request in requests if _is_preferred_edition_search(request)
    } == {
        "key:/works/OL1W AND language:eng",
        "key:/works/OL2W AND language:eng",
    }
    for request in requests:
        if not _is_preferred_edition_search(request):
            continue
        assert dict(request.url.params) == {
            "q": request.url.params["q"],
            "fields": _EDITION_SEARCH_FIELDS,
            "limit": "1",
        }
        assert "JP" not in str(request.url)
        assert "audiobook" not in request.url.params["q"].casefold()
        assert "isbn" not in request.url.params["q"].casefold()
    work_searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and not _is_preferred_edition_search(request)
    ]
    assert {tuple(sorted(request.url.params.items())) for request in work_searches} == {
        (
            ("fields", _WORK_SEARCH_FIELDS),
            ("limit", "5"),
            ("title", "Interpreted First Mention"),
        ),
        (
            ("fields", _WORK_SEARCH_FIELDS),
            ("limit", "5"),
            ("title", "Interpreted Second Mention"),
        ),
    }
