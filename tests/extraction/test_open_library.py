"""Open Library Book Work resolver contract tests."""

import asyncio
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import OpenLibraryConfig
from reelio.extraction.services.enrichment.open_library import (
    OpenLibraryBookResolver,
    create_open_library_book_resolver,
)
from reelio.extraction.types import AuthorCredit, BookMention, BookMentions, ResultStatus


def _settings(**values: object) -> OpenLibraryConfig:
    settings_type = cast(Callable[..., OpenLibraryConfig], OpenLibraryConfig)
    return settings_type(
        _env_file=None,
        contact_email="catalog-contact@example.invalid",
        **values,
    )


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://openlibrary.org/",
        transport=handler,
        headers={"User-Agent": "Reelio (catalog-contact@example.invalid)"},
    )


def _mentions(*books: BookMention) -> BookMentions:
    return BookMentions(books=list(books))


def _candidate(
    key: str,
    title: str,
    author_keys: list[str] | None = None,
    author_names: list[str] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {"key": key, "title": title}
    if author_keys is not None:
        candidate["author_key"] = author_keys
    if author_names is not None:
        candidate["author_name"] = author_names
    return candidate


def _search_response(*candidates: dict[str, object]) -> dict[str, object]:
    return {"docs": list(candidates)}


class _FakeClock:
    """Advance deterministic monotonic time through rate-limit waits."""

    def __init__(self) -> None:
        self.value = 0.0
        self.delays: list[float] = []

    def __call__(self) -> float:
        """Return the current deterministic monotonic time."""
        return self.value

    async def sleep(self, delay_seconds: float) -> None:
        """Record and advance one deterministic request delay."""
        self.delays.append(delay_seconds)
        self.value += delay_seconds


def _resolver(
    handler: httpx.AsyncBaseTransport,
    clock: _FakeClock | None = None,
) -> OpenLibraryBookResolver:
    fake_clock = clock or _FakeClock()
    return OpenLibraryBookResolver(
        _client(handler),
        3.0,
        fake_clock,
        fake_clock.sleep,
    )


async def test_resolver_matches_normalized_title_and_any_primary_author() -> None:
    """Resolve an exact title when any complete Author Credit matches."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_search_response(
                _candidate(
                    "/works/OL1W",
                    "Pride and Prejudice",
                    ["OL2A"],
                    ["Different Author"],
                ),
                _candidate(
                    "OL3W",
                    "  PRIDE AND PREJUDICE  ",
                    ["/authors/OL4A", "OL5A"],
                    ["Jane Austen", "Coauthor"],
                ),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    mention = BookMention(
        title="  Pride and Prejudice  ",
        authors=[AuthorCredit(name="JANE AUSTEN")],
    )

    results = await resolver.resolve(_mentions(mention))

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/search.json"
    assert dict(request.url.params) == {
        "title": "  Pride and Prejudice  ",
        "fields": "key,title,author_key,author_name",
        "limit": "5",
    }
    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book_mention is mention
    assert result.book is not None
    assert result.book.title == "  PRIDE AND PREJUDICE  "
    assert [author.name for author in result.book.authors] == ["Jane Austen", "Coauthor"]
    assert [author.open_library_author_id for author in result.book.authors] == ["OL4A", "OL5A"]
    assert [author.open_library_url for author in result.book.authors] == [
        "https://openlibrary.org/authors/OL4A",
        "https://openlibrary.org/authors/OL5A",
    ]
    assert result.book.open_library_work_id == "OL3W"
    assert result.book.open_library_url == "https://openlibrary.org/works/OL3W"
    await resolver.aclose()


async def test_resolver_selects_first_provider_ranked_authorful_exact_candidate() -> None:
    """Preserve Open Library provider order after exact title and Author matching."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_search_response(
                _candidate("OL10W", "Dune", ["OL11A"], ["Frank Herbert"]),
                _candidate("OL12W", "Dune", ["OL13A"], ["Frank Herbert"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(title="Dune", authors=[AuthorCredit(name="Frank Herbert")]),
        )
    )

    assert results.books[0].book is not None
    assert results.books[0].book.open_library_work_id == "OL10W"
    await resolver.aclose()


async def test_resolver_returns_unresolved_for_author_mismatch_and_ambiguous_authorless_title() -> (
    None
):
    """Avoid identity claims when exact title Candidates remain ineligible or ambiguous."""

    async def handle(request: httpx.Request) -> httpx.Response:
        title = request.url.params["title"]
        if title == "Dune":
            return httpx.Response(
                200,
                json=_search_response(
                    _candidate("OL1W", "Dune", ["OL2A"], ["Frank Herbert"]),
                ),
            )
        return httpx.Response(
            200,
            json=_search_response(
                _candidate("OL3W", "Shared Title"),
                _candidate("OL4W", "Shared Title"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    author_mismatch = BookMention(title="Dune", authors=[AuthorCredit(name="Brian Herbert")])
    ambiguous = BookMention(title="Shared Title", authors=[])

    results = await resolver.resolve(_mentions(author_mismatch, ambiguous))

    assert [result.status for result in results.books] == [
        ResultStatus.UNRESOLVED,
        ResultStatus.UNRESOLVED,
    ]
    assert [result.book_mention for result in results.books] == [author_mismatch, ambiguous]
    assert all(result.book is None for result in results.books)
    await resolver.aclose()


async def test_resolver_resolves_unique_authorless_title_with_empty_provider_authorship() -> None:
    """Preserve an authorless Mention and provider's absent authorship without synthesis."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_response(_candidate("OL9W", "Anonymous Work")))

    resolver = _resolver(httpx.MockTransport(handle))
    mention = BookMention(title="Anonymous Work", authors=[])

    results = await resolver.resolve(_mentions(mention))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book_mention is mention
    assert result.book is not None
    assert result.book.authors == []
    await resolver.aclose()


async def test_resolver_preserves_concurrent_mention_result_order() -> None:
    """Return ordered results even when independent requests complete out of order."""

    async def handle(request: httpx.Request) -> httpx.Response:
        title = request.url.params["title"]
        if title == "First":
            await asyncio.sleep(0)
        work_id = "OL1W" if title == "First" else "OL2W"
        return httpx.Response(200, json=_search_response(_candidate(work_id, title)))

    resolver = _resolver(httpx.MockTransport(handle))
    first = BookMention(title="First", authors=[])
    second = BookMention(title="Second", authors=[])

    results = await resolver.resolve(_mentions(first, second))

    assert [result.book_mention for result in results.books] == [first, second]
    assert [result.book.open_library_work_id for result in results.books if result.book] == [
        "OL1W",
        "OL2W",
    ]
    await resolver.aclose()


async def test_resolver_skips_requests_for_empty_mentions_and_closes_client() -> None:
    """Return empty results without request tasks and release the owned client."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    resolver = _resolver(httpx.MockTransport(handle))

    results = await resolver.resolve(_mentions())

    assert results.books == []
    assert requests == []
    await resolver.aclose()
    assert resolver._client.is_closed is True


async def test_resolver_spaces_concurrent_requests_at_three_per_second() -> None:
    """Serialize dispatch timing while allowing one task per Book Work Mention."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        title = request.url.params["title"]
        return httpx.Response(
            200,
            json=_search_response(_candidate(f"OL{len(dispatch_times)}W", title)),
        )

    resolver = _resolver(httpx.MockTransport(handle), clock)

    await resolver.resolve(
        _mentions(
            BookMention(title="One", authors=[]),
            BookMention(title="Two", authors=[]),
            BookMention(title="Three", authors=[]),
        )
    )

    assert dispatch_times == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert clock.delays == pytest.approx([1 / 3, 1 / 3])
    await resolver.aclose()


@pytest.mark.parametrize(
    "response_payload",
    [
        {"docs": [{"title": "Missing Work ID"}]},
        _search_response(_candidate("bad-work", "Invalid Work ID")),
        _search_response(
            _candidate("OL1W", "Mismatched Authors", ["OL2A"], None),
        ),
        _search_response(
            _candidate("OL1W", "Invalid Author ID", ["bad-author"], ["Name"]),
        ),
    ],
)
async def test_resolver_maps_invalid_provider_responses_to_catalog_failure(
    response_payload: dict[str, object],
) -> None:
    """Reject malformed provider data without returning a partial result."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_payload)

    resolver = _resolver(httpx.MockTransport(handle))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


async def test_resolver_maps_http_failures_and_timeouts_to_typed_errors() -> None:
    """Translate provider HTTP failures and timeouts without leaking provider details."""
    request_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(500)
        raise httpx.ReadTimeout("timed out", request=request)

    resolver = _resolver(httpx.MockTransport(handle))
    mentions = _mentions(BookMention(title="Target", authors=[]))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(mentions)
    with pytest.raises(PipelineTimeoutError, match="Open Library catalog request timed out"):
        await resolver.resolve(mentions)

    await resolver.aclose()


async def test_resolver_maps_malformed_json_and_transport_errors_to_catalog_failure() -> None:
    """Translate unsafe provider response encodings and transport errors consistently."""
    request_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, content=b"{not-json")
        raise httpx.ConnectError("connection failed", request=request)

    resolver = _resolver(httpx.MockTransport(handle))
    mentions = _mentions(BookMention(title="Target", authors=[]))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(mentions)
    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(mentions)

    await resolver.aclose()


async def test_factory_configures_identifying_contact_and_owned_client() -> None:
    """Build one client with the configured endpoint, timeout, and User-Agent."""
    resolver = create_open_library_book_resolver(
        _settings(base_url="https://catalog.example", request_timeout_seconds=4.5),
    )

    assert str(resolver._client.base_url) == "https://catalog.example/"
    assert resolver._client.headers["User-Agent"] == "Reelio (catalog-contact@example.invalid)"
    assert resolver._client.timeout.connect == 4.5
    await resolver.aclose()
