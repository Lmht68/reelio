"""Open Library Book Work resolver contract tests."""

import asyncio
from collections.abc import Callable
from datetime import date
from typing import cast

import httpx
import pytest

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import OpenLibraryConfig
from reelio.extraction.services.enrichment.open_library import (
    OpenLibraryBookResolver,
    create_open_library_book_resolver,
)
from reelio.extraction.types import (
    AuthorCredit,
    BookMention,
    BookMentions,
    BookResults,
    ResultStatus,
)

_WORK_SEARCH_FIELDS = (
    "key,title,alternative_title,author_key,author_name,"
    "author_alternative_name,editions,editions.key,editions.title,cover_i,"
    "cover_edition_key"
)
_EDITION_SEARCH_FIELDS = (
    "key,editions,editions.key,editions.title,editions.format,"
    "editions.publish_year,editions.cover_i"
)


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


class _AbsentPreferredEditionTransport(httpx.AsyncBaseTransport):
    """Return nullable Edition Search data around an existing Work test transport."""

    def __init__(self, delegate: httpx.AsyncBaseTransport) -> None:
        """Initialize the transport that supplies default missing Edition data."""
        self._delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Return no Edition selection or delegate the Work-resolution request."""
        if _is_preferred_edition_search(request):
            return httpx.Response(200, json=_preferred_edition_search_response())
        return await self._delegate.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped transport with the resolver-owned client."""
        await self._delegate.aclose()


def _is_preferred_edition_search(request: httpx.Request) -> bool:
    """Return whether a request is the Work-ID-based preferred Edition Search."""
    return request.url.path == "/search.json" and "q" in request.url.params


def _mentions(*books: BookMention) -> BookMentions:
    return BookMentions(books=list(books))


def _candidate(
    key: str,
    title: str,
    author_keys: list[str] | None = None,
    author_names: list[str] | None = None,
    *,
    alternative_titles: list[str] | None = None,
    author_aliases: list[str] | None = None,
    edition_titles: list[str] | None = None,
    cover_id: int | None = None,
    cover_edition_key: str | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {"key": key, "title": title}
    if author_keys is not None:
        candidate["author_key"] = author_keys
    if author_names is not None:
        candidate["author_name"] = author_names
    if alternative_titles is not None:
        candidate["alternative_title"] = alternative_titles
    if author_aliases is not None:
        candidate["author_alternative_name"] = author_aliases
    if edition_titles is not None:
        candidate["editions"] = {"docs": [{"title": value} for value in edition_titles]}
    if cover_id is not None:
        candidate["cover_i"] = cover_id
    if cover_edition_key is not None:
        candidate["cover_edition_key"] = cover_edition_key
    return candidate


def _work_search_response(*candidates: dict[str, object]) -> dict[str, object]:
    return {"docs": list(candidates)}


def _selected_edition(
    edition_id: str,
    *,
    title: str | None = None,
    formats: list[str] | None = None,
    publication_years: list[int] | None = None,
    cover_id: int | None = None,
) -> dict[str, object]:
    edition: dict[str, object] = {"key": f"/books/{edition_id}"}
    if title is not None:
        edition["title"] = title
    if formats is not None:
        edition["format"] = formats
    if publication_years is not None:
        edition["publish_year"] = publication_years
    if cover_id is not None:
        edition["cover_i"] = cover_id
    return edition


def _preferred_edition_search_response(
    work_id: str | None = None,
    *editions: dict[str, object],
) -> dict[str, object]:
    if work_id is None:
        return {"docs": []}
    return {
        "docs": [
            {
                "key": f"/works/{work_id}",
                "editions": {"docs": list(editions)},
            }
        ]
    }


def _edition_record(
    edition_id: str,
    *,
    title: str | None = None,
    publishers: list[str] | None = None,
    isbn_10: list[str] | None = None,
    isbn_13: list[str] | None = None,
    publish_date: str | None = None,
    physical_format: str | None = None,
    covers: list[int] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {"key": f"/books/{edition_id}"}
    if title is not None:
        record["title"] = title
    if publishers is not None:
        record["publishers"] = publishers
    if isbn_10 is not None:
        record["isbn_10"] = isbn_10
    if isbn_13 is not None:
        record["isbn_13"] = isbn_13
    if publish_date is not None:
        record["publish_date"] = publish_date
    if physical_format is not None:
        record["physical_format"] = physical_format
    if covers is not None:
        record["covers"] = covers
    return record


def _work_record(work_id: str) -> dict[str, object]:
    return {"type": {"key": "/type/work"}, "key": f"/works/{work_id}"}


def _redirect_record(work_id: str) -> dict[str, object]:
    return {"type": {"key": "/type/redirect"}, "location": f"/works/{work_id}"}


def _terminal_work_response(request: httpx.Request) -> httpx.Response:
    work_id = request.url.path.removeprefix("/works/").removesuffix(".json")
    return httpx.Response(200, json=_work_record(work_id))


def _assert_single_mention_cover_requests(
    requests: list[httpx.Request],
    selected_edition_id: str,
) -> None:
    assert [request.url.path for request in requests] == [
        "/search.json",
        "/works/OL1W.json",
        "/search.json",
        f"/books/{selected_edition_id}.json",
    ]
    assert all(request.url.host == "openlibrary.org" for request in requests)
    assert not any(request.url.host == "covers.openlibrary.org" for request in requests)
    assert not any(request.url.path.startswith("/covers/") for request in requests)
    assert not any(request.url.path.startswith("/editions/") for request in requests)
    assert [request.url.path for request in requests if request.url.path.startswith("/books/")] == [
        f"/books/{selected_edition_id}.json"
    ]
    assert not any("isbn" in str(request.url).casefold() for request in requests)


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
    *,
    default_missing_preferred_edition: bool = True,
) -> OpenLibraryBookResolver:
    fake_clock = clock or _FakeClock()
    transport = (
        _AbsentPreferredEditionTransport(handler) if default_missing_preferred_edition else handler
    )
    return OpenLibraryBookResolver(
        _client(transport),
        3.0,
        fake_clock,
        fake_clock.sleep,
    )


def _resolved_work_id(results: BookResults) -> str:
    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book is not None
    return result.book.open_library_work_id


async def test_resolver_uses_author_constrained_exact_search_without_fallback() -> None:
    """Resolve an exact authorful Candidate through one constrained Search window."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
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

    searches = [request for request in requests if request.url.path == "/search.json"]
    assert len(searches) == 1
    assert dict(searches[0].url.params) == {
        "title": "  Pride and Prejudice  ",
        "author": "JANE AUSTEN",
        "fields": _WORK_SEARCH_FIELDS,
        "limit": "5",
    }
    assert [request.url.path for request in requests[1:]] == [
        "/works/OL1W.json",
        "/works/OL3W.json",
    ]
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


async def test_resolver_falls_back_once_after_constrained_window_has_no_match() -> None:
    """Run one title-only Search only after constrained matching cannot resolve."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" in request.url.params:
            return httpx.Response(200, json=_work_search_response())
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Dune", ["OL2A"], ["Frank Herbert"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Dune", authors=[AuthorCredit(name="Frank Herbert")]))
    )

    searches = [request for request in requests if request.url.path == "/search.json"]
    assert [dict(request.url.params) for request in searches] == [
        {
            "title": "Dune",
            "author": "Frank Herbert",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        },
        {"title": "Dune", "fields": _WORK_SEARCH_FIELDS, "limit": "5"},
    ]
    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_does_not_fallback_when_constrained_fuzzy_match_passes() -> None:
    """Keep a safe constrained fuzzy match without broadening the Search."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Pride and Prejudce",
                    ["OL2A"],
                    ["Jane Austen"],
                ),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="Pride and Prejudice",
                authors=[AuthorCredit(name="Jane Austen")],
            )
        )
    )

    assert len([request for request in requests if request.url.path == "/search.json"]) == 1
    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_limits_each_search_window_to_five_candidates() -> None:
    """Ignore a sixth Search Candidate even when it would otherwise resolve."""
    requests: list[httpx.Request] = []
    constrained_candidates = [
        _candidate(f"OL{index}W", f"Other {index}", ["OL1A"], ["Author"]) for index in range(1, 6)
    ]
    constrained_candidates.append(_candidate("OL6W", "Target", ["OL1A"], ["Author"]))

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" in request.url.params:
            return httpx.Response(200, json={"docs": constrained_candidates})
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target", authors=[AuthorCredit(name="Author")]))
    )

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert [request.url.path for request in requests if request.url.path.startswith("/works/")] == [
        "/works/OL1W.json",
        "/works/OL2W.json",
        "/works/OL3W.json",
        "/works/OL4W.json",
        "/works/OL5W.json",
    ]
    assert len([request for request in requests if request.url.path == "/search.json"]) == 2
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_title", "candidate"),
    [
        (
            "Canonical Work",
            _candidate("OL1W", "Canonical Work", ["OL1A"], ["Author"]),
        ),
        (
            "Titre traduit",
            _candidate(
                "OL1W",
                "Canonical Work",
                ["OL1A"],
                ["Author"],
                alternative_titles=["Titre traduit"],
            ),
        ),
        (
            "Selected Edition Title",
            _candidate(
                "OL1W",
                "Canonical Work",
                ["OL1A"],
                ["Author"],
                edition_titles=["Selected Edition Title", "Ignored Edition Title"],
            ),
        ),
    ],
)
async def test_resolver_uses_primary_alternative_or_selected_edition_titles(
    mention_title: str,
    candidate: dict[str, object],
) -> None:
    """Match bounded identity titles while enriching only primary Work metadata."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(candidate))

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title=mention_title, authors=[AuthorCredit(name="Author")]))
    )

    result = results.books[0]
    assert result.book is not None
    assert result.book.title == "Canonical Work"
    assert result.book.open_library_work_id == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_author", "provider_author", "alias"),
    [
        ("Eric Arthur Blair", "George Orwell", "Eric Arthur Blair"),
        ("WHO", "World Health Organization", "WHO"),
    ],
)
async def test_resolver_accepts_exact_author_aliases(
    mention_author: str,
    provider_author: str,
    alias: str,
) -> None:
    """Treat exact personal and organizational aliases as eligible authorship."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Target",
                    ["OL1A"],
                    [provider_author],
                    author_aliases=[alias],
                )
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target", authors=[AuthorCredit(name=mention_author)]))
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_accepts_one_matching_author_among_multiple_credits() -> None:
    """Require one exact Author Credit rather than complete authorship equality."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Target", ["OL1A"], ["Matching Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="Target",
                authors=[
                    AuthorCredit(name="Matching Author"),
                    AuthorCredit(name="Unmatched Author"),
                ],
            )
        )
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_author", "provider_author"),
    [
        ("George Orwel", "George Orwell"),
        ("World Health Organisaton", "World Health Organization"),
    ],
)
async def test_resolver_rejects_similar_author_names(
    mention_author: str,
    provider_author: str,
) -> None:
    """Never use fuzzy personal or organization matching for authorship eligibility."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" not in request.url.params:
            return httpx.Response(200, json=_work_search_response())
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Target", ["OL1A"], [provider_author]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target", authors=[AuthorCredit(name=mention_author)]))
    )

    assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.aclose()


async def test_resolver_prefers_exact_title_before_higher_ranked_fuzzy_title() -> None:
    """Exhaust exact Candidates before considering fuzzy title scores."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Target Workx", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Target Work", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target Work", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL2W"
    await resolver.aclose()


async def test_resolver_rejects_fuzzy_score_at_the_strict_threshold() -> None:
    """Reject a title ratio of exactly 80.0 before title-only fallback."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" not in request.url.params:
            return httpx.Response(200, json=_work_search_response())
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefxyij", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="abcdefghij", authors=[AuthorCredit(name="Author")]))
    )

    assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.aclose()


async def test_resolver_accepts_fuzzy_score_above_the_strict_threshold() -> None:
    """Accept a 90.0 title ratio when exact verification has no match."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefgxij", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="abcdefghij", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_selects_the_strongest_fuzzy_title_score() -> None:
    """Choose the highest candidate score rather than provider rank for fuzzy titles."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefgxij", ["OL1A"], ["Author"]),
                _candidate("OL2W", "abcdefghijk", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="abcdefghij", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL2W"
    await resolver.aclose()


async def test_resolver_keeps_provider_order_for_equal_fuzzy_scores() -> None:
    """Keep the earlier provider Candidate when safe fuzzy scores tie."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefgxij", ["OL1A"], ["Author"]),
                _candidate("OL2W", "abcdefxhij", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="abcdefghij", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_resolves_unique_authorless_exact_title() -> None:
    """Resolve one exact canonical Candidate without author metadata synthesis."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Anonymous Work")))

    resolver = _resolver(httpx.MockTransport(handle))
    mention = BookMention(title="Anonymous Work", authors=[])
    results = await resolver.resolve(_mentions(mention))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book_mention is mention
    assert result.book is not None
    assert result.book.authors == []
    await resolver.aclose()


async def test_resolver_leaves_authorless_exact_candidates_ambiguous() -> None:
    """Retain ambiguity when two distinct canonical Works exactly match a title."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Shared Title", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Shared Title", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Shared Title", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.aclose()


async def test_resolver_does_not_fuzzily_resolve_authorless_mentions() -> None:
    """Reject a sole fuzzy Candidate for a Book Mention without authorship."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "abcdefghijk")))

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.aclose()


async def test_resolver_deduplicates_repeated_raw_ids_before_candidate_verification() -> None:
    """Keep first Search metadata for a repeated raw Work ID before matching."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" not in request.url.params:
            return httpx.Response(200, json=_work_search_response())
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Earlier Candidate", ["OL1A"], ["Author"]),
                _candidate("OL1W", "Target", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target", authors=[AuthorCredit(name="Author")]))
    )

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert [request.url.path for request in requests if request.url.path.startswith("/works/")] == [
        "/works/OL1W.json"
    ]
    await resolver.aclose()


async def test_resolver_canonicalizes_redirects_before_candidate_deduplication() -> None:
    """Emit a terminal Work ID and retain first Search metadata across redirect aliases."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(200, json=_redirect_record("OL3W"))
        if request.url.path == "/works/OL2W.json":
            return httpx.Response(200, json=_redirect_record("OL3W"))
        if request.url.path == "/works/OL3W.json":
            return httpx.Response(200, json=_work_record("OL3W"))
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Target", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Target", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Target", authors=[AuthorCredit(name="Author")]))
    )

    result = results.books[0]
    assert result.book is not None
    assert result.book.title == "Target"
    assert result.book.open_library_work_id == "OL3W"
    assert result.book.open_library_url == "https://openlibrary.org/works/OL3W"
    assert [request.url.path for request in requests] == [
        "/search.json",
        "/works/OL1W.json",
        "/works/OL3W.json",
        "/works/OL2W.json",
        "/works/OL3W.json",
    ]
    await resolver.aclose()


async def test_resolver_keeps_distinct_canonical_ids_with_identical_metadata() -> None:
    """Do not collapse distinct Works based on equal titles or author names."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Shared Title", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Shared Title", ["OL1A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Shared Title", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.aclose()


async def test_resolver_collapses_resolved_results_by_canonical_work_id() -> None:
    """Preserve the first differently worded Mention for one canonical Work."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Canonical Title",
                    ["OL1A"],
                    ["Author"],
                    alternative_titles=["Original Title", "Translated Title"],
                ),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    first_mention = BookMention(
        title="Original Title",
        authors=[AuthorCredit(name="Author")],
    )
    second_mention = BookMention(
        title="Translated Title",
        authors=[AuthorCredit(name="Author")],
    )
    results = await resolver.resolve(_mentions(first_mention, second_mention))

    assert [result.book_mention for result in results.books] == [first_mention]
    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_preserves_concurrent_mention_result_order() -> None:
    """Return ordered unresolved and resolved results after concurrent resolution."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        title = request.url.params["title"]
        if title == "First":
            await asyncio.sleep(0)
        work_id = "OL1W" if title == "First" else "OL2W"
        return httpx.Response(200, json=_work_search_response(_candidate(work_id, title)))

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


async def test_resolver_spaces_search_work_and_edition_requests_at_three_per_second() -> None:
    """Apply the shared dispatch limiter to all Open Library request paths."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if _is_preferred_edition_search(request):
            return httpx.Response(200, json=_preferred_edition_search_response())
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        title = request.url.params["title"]
        work_id = {"One": "OL1W", "Two": "OL2W", "Three": "OL3W"}[title]
        return httpx.Response(200, json=_work_search_response(_candidate(work_id, title)))

    resolver = _resolver(
        httpx.MockTransport(handle),
        clock,
        default_missing_preferred_edition=False,
    )

    await resolver.resolve(
        _mentions(
            BookMention(title="One", authors=[]),
            BookMention(title="Two", authors=[]),
            BookMention(title="Three", authors=[]),
        )
    )

    assert dispatch_times == pytest.approx([index / 3 for index in range(12)])
    assert clock.delays == pytest.approx([1 / 3] * 11)
    await resolver.aclose()


async def test_resolver_selects_first_english_relevance_edition() -> None:
    """Load only the first English Search-selected Edition and preserve raw metadata."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            if request.url.params["q"].endswith("language:eng"):
                return httpx.Response(
                    200,
                    json=_preferred_edition_search_response(
                        "OL1W",
                        _selected_edition(
                            "OL101M",
                            title="Search Edition Title",
                            formats=["Print"],
                            publication_years=[1995],
                            cover_id=101,
                        ),
                        _selected_edition(
                            "OL102M",
                            title="Ignored Relevance Result",
                            formats=["Audio"],
                            publication_years=[2000],
                            cover_id=102,
                        ),
                    ),
                )
            raise AssertionError(f"unexpected fallback Edition Search: {request.url}")
        if request.url.path == "/books/OL101M.json":
            return httpx.Response(
                200,
                json=_edition_record(
                    "OL101M",
                    title="Direct Edition Title",
                    publishers=["Penguin", "Penguin", " PENGUIN "],
                    isbn_10=["0141439513", "not-an-isbn", " 0141439513 "],
                    isbn_13=["9780141439518", "9780141439518", " 9780141439518 "],
                    publish_date="1995",
                    covers=[0, -1, 321],
                ),
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Provider Work",
                    ["OL1A"],
                    ["Author"],
                    cover_id=802,
                    cover_edition_key="/books/OL902M",
                ),
            ),
        )

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(
        _mentions(BookMention(title="Provider Work", authors=[AuthorCredit(name="Author")]))
    )

    result = results.books[0]
    assert result.book is not None
    assert result.book.edition is not None
    assert result.book.edition.title == "Direct Edition Title"
    assert result.book.edition.publication_year == 1995
    assert result.book.edition.publishers == ["Penguin", "Penguin", " PENGUIN "]
    assert result.book.edition.isbn_10 == ["0141439513", "not-an-isbn", " 0141439513 "]
    assert result.book.edition.isbn_13 == [
        "9780141439518",
        "9780141439518",
        " 9780141439518 ",
    ]
    assert result.book.edition.open_library_edition_id == "OL101M"
    assert result.book.edition.open_library_url == "https://openlibrary.org/books/OL101M"
    assert result.book.edition.cover_url == "https://covers.openlibrary.org/b/id/321-L.jpg"
    assert result.book.cover_url == "https://covers.openlibrary.org/b/id/321-L.jpg"
    assert result.book.cover_edition_id == "OL101M"
    edition_searches = [request for request in requests if _is_preferred_edition_search(request)]
    assert [dict(request.url.params) for request in edition_searches] == [
        {
            "q": "key:/works/OL1W AND language:eng",
            "fields": _EDITION_SEARCH_FIELDS,
            "limit": "1",
        }
    ]
    assert all("sort" not in request.url.params for request in edition_searches)
    _assert_single_mention_cover_requests(requests, "OL101M")
    await resolver.aclose()


@pytest.mark.parametrize(
    ("cover_edition_key", "expected_cover_edition_id"),
    [
        ("OL902M", "OL902M"),
        (None, None),
    ],
)
async def test_resolver_uses_work_cover_fallback_without_reassigning_selected_edition(
    cover_edition_key: str | None,
    expected_cover_edition_id: str | None,
) -> None:
    """Retain a coverless selected Edition while exposing Work Search artwork."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            assert request.url.params["q"] == "key:/works/OL1W AND language:eng"
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL901M"),
                ),
            )
        if request.url.path == "/books/OL901M.json":
            return httpx.Response(200, json=_edition_record("OL901M"))
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Target",
                    cover_id=802,
                    cover_edition_key=cover_edition_key,
                )
            ),
        )

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book is not None
    assert result.book.edition is not None
    assert result.book.edition.open_library_edition_id == "OL901M"
    assert result.book.edition.cover_url is None
    assert result.book.cover_url == "https://covers.openlibrary.org/b/id/802-L.jpg"
    assert result.book.cover_edition_id == expected_cover_edition_id
    assert [
        dict(request.url.params) for request in requests if _is_preferred_edition_search(request)
    ] == [
        {
            "q": "key:/works/OL1W AND language:eng",
            "fields": _EDITION_SEARCH_FIELDS,
            "limit": "1",
        }
    ]
    _assert_single_mention_cover_requests(requests, "OL901M")
    await resolver.aclose()


async def test_resolver_falls_back_to_any_language_after_no_english_edition() -> None:
    """Select an unrestricted relevance result only when English Search has none."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            query = request.url.params["q"]
            if query.endswith("language:eng"):
                return httpx.Response(200, json=_preferred_edition_search_response("OL1W"))
            assert query == "key:/works/OL1W"
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL201M", title="Fallback Edition"),
                ),
            )
        if request.url.path == "/books/OL201M.json":
            return httpx.Response(200, json=_edition_record("OL201M"))
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is not None
    assert results.books[0].book.edition.open_library_edition_id == "OL201M"
    assert [
        request.url.params["q"] for request in requests if _is_preferred_edition_search(request)
    ] == [
        "key:/works/OL1W AND language:eng",
        "key:/works/OL1W",
    ]
    await resolver.aclose()


@pytest.mark.parametrize(
    "edition_search_response",
    [
        _preferred_edition_search_response(),
        {"docs": [{"key": "/works/OL1W"}]},
        _preferred_edition_search_response("OL1W"),
    ],
)
async def test_resolver_retains_resolved_work_when_preferred_edition_is_absent(
    edition_search_response: dict[str, object],
) -> None:
    """Return nullable Edition data without changing a resolved Work outcome."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            return httpx.Response(200, json=edition_search_response)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book is not None
    assert result.book.edition is None
    assert not any(request.url.path.startswith("/books/") for request in requests)
    await resolver.aclose()


async def test_resolver_uses_first_nonblank_nested_edition_title_for_work_identity() -> None:
    """Ignore blank optional nested Edition titles during Work Candidate matching."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                {
                    "key": "/works/OL1W",
                    "title": "Canonical Work",
                    "author_key": ["OL1A"],
                    "author_name": ["Author"],
                    "editions": {"docs": [{"title": "  "}, {"title": "Mention Edition Title"}]},
                }
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="Mention Edition Title",
                authors=[AuthorCredit(name="Author")],
            )
        )
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("search_formats", "physical_format"),
    [
        (["Print"], "Hardcover"),
        (["Ebook"], "EPUB"),
        (None, None),
        (["   "], "  "),
        (["Microform"], "Loose-leaf"),
    ],
)
async def test_resolver_accepts_non_audio_edition_formats(
    search_formats: list[str] | None,
    physical_format: str | None,
) -> None:
    """Keep print, ebook, absent, blank, and unknown Edition formats eligible."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL301M", formats=search_formats),
                ),
            )
        if request.url.path == "/books/OL301M.json":
            return httpx.Response(
                200,
                json=_edition_record("OL301M", physical_format=physical_format),
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is not None
    assert [
        request.url.params["q"] for request in requests if _is_preferred_edition_search(request)
    ] == ["key:/works/OL1W AND language:eng"]
    await resolver.aclose()


@pytest.mark.parametrize(
    "audio_format",
    ["Audio", "AUDIOBOOK", "Audio   Book", "Sound Recording", "Cassette", "MP3"],
)
async def test_resolver_falls_back_after_explicit_english_audiobook(
    audio_format: str,
) -> None:
    """Reject an English audiobook and load only the unrestricted relevance result."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            query = request.url.params["q"]
            if query.endswith("language:eng"):
                return httpx.Response(
                    200,
                    json=_preferred_edition_search_response(
                        "OL1W",
                        _selected_edition("OL401M", formats=[audio_format]),
                    ),
                )
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL402M", formats=["Print"]),
                ),
            )
        if request.url.path == "/books/OL402M.json":
            return httpx.Response(200, json=_edition_record("OL402M"))
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is not None
    assert results.books[0].book.edition.open_library_edition_id == "OL402M"
    assert [request.url.path for request in requests if request.url.path.startswith("/books/")] == [
        "/books/OL402M.json"
    ]
    await resolver.aclose()


async def test_resolver_returns_no_edition_for_unrestricted_audiobook() -> None:
    """Do not load or enumerate another Edition after an audio fallback result."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            if request.url.params["q"].endswith("language:eng"):
                return httpx.Response(200, json=_preferred_edition_search_response())
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL501M", formats=["Audiobook"]),
                ),
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is None
    assert not any(request.url.path.startswith("/books/") for request in requests)
    assert not any("/editions" in request.url.path for request in requests)
    await resolver.aclose()


async def test_resolver_does_not_reload_an_english_edition_rejected_as_audio() -> None:
    """Avoid a repeated detail request when the fallback selects the rejected ID."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL601M", formats=["Print"]),
                ),
            )
        if request.url.path == "/books/OL601M.json":
            return httpx.Response(
                200,
                json=_edition_record("OL601M", physical_format="Sound Recording"),
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is None
    assert [request.url.path for request in requests if request.url.path.startswith("/books/")] == [
        "/books/OL601M.json"
    ]
    assert [
        request.url.params["q"] for request in requests if _is_preferred_edition_search(request)
    ] == [
        "key:/works/OL1W AND language:eng",
        "key:/works/OL1W",
    ]
    await resolver.aclose()


async def test_resolver_returns_sparse_selected_edition_metadata() -> None:
    """Return valid Edition identity with nullable optional provider metadata."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL701M", cover_id=0),
                ),
            )
        if request.url.path == "/books/OL701M.json":
            return httpx.Response(200, json=_edition_record("OL701M", covers=[0, -1]))
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book is not None
    assert result.book.edition is not None
    assert result.book.edition.title is None
    assert result.book.edition.publication_year is None
    assert result.book.edition.publishers == []
    assert result.book.edition.isbn_10 == []
    assert result.book.edition.isbn_13 == []
    assert result.book.edition.cover_url is None
    assert result.book.cover_url is None
    assert result.book.cover_edition_id is None
    _assert_single_mention_cover_requests(requests, "OL701M")
    await resolver.aclose()


@pytest.mark.parametrize(
    ("numeric_years", "publish_date", "expected_year"),
    [
        ([1990], "Reprinted in 2024", 1990),
        ([date.today().year], "First published in 1990", date.today().year),
        ([date.today().year + 1], "First published in 1990", None),
        (None, "First published in 1990", 1990),
        (None, None, None),
        (None, "", None),
        (None, "undated", None),
        (None, "1890, reissued 1990", None),
        (None, f"First published in {date.today().year + 1}", None),
    ],
)
async def test_resolver_derives_publication_year_from_provider_metadata(
    numeric_years: list[int] | None,
    publish_date: str | None,
    expected_year: int | None,
) -> None:
    """Apply numeric precedence and conservative free-form publication year rules."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL801M", publication_years=numeric_years),
                ),
            )
        if request.url.path == "/books/OL801M.json":
            return httpx.Response(
                200,
                json=_edition_record("OL801M", publish_date=publish_date),
            )
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].book is not None
    assert results.books[0].book.edition is not None
    assert results.books[0].book.edition.publication_year == expected_year
    await resolver.aclose()


@pytest.mark.parametrize(
    "preferred_edition_response",
    [
        _preferred_edition_search_response(
            "OL1W",
            {"key": "/books/not-an-edition"},
        ),
        _preferred_edition_search_response(
            "OL1W",
            {"key": "/books/OL901M", "format": "not-a-list"},
        ),
        {"docs": [{"key": "/works/not-a-work", "editions": {"docs": []}}]},
    ],
)
async def test_resolver_maps_invalid_preferred_edition_search_to_catalog_failure(
    preferred_edition_response: dict[str, object],
) -> None:
    """Reject malformed selected Edition Search data without a partial Work result."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if _is_preferred_edition_search(request):
            return httpx.Response(200, json=preferred_edition_response)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


@pytest.mark.parametrize(
    "edition_payload",
    [
        {"key": "/books/OL999M"},
        {"key": "/books/OL902M", "publishers": "not-a-list"},
        {"key": "/books/OL902M", "isbn_10": "not-a-list"},
    ],
)
async def test_resolver_maps_invalid_selected_edition_record_to_catalog_failure(
    edition_payload: dict[str, object],
) -> None:
    """Reject mismatched Edition identity and malformed direct Edition fields."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL902M"),
                ),
            )
        if request.url.path == "/books/OL902M.json":
            return httpx.Response(200, json=edition_payload)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


@pytest.mark.parametrize(
    "response_payload",
    [
        {"docs": [{"title": "Missing Work ID"}]},
        _work_search_response(_candidate("bad-work", "Invalid Work ID")),
        _work_search_response(
            _candidate("OL1W", "Mismatched Authors", ["OL2A"], None),
        ),
        _work_search_response(
            _candidate("OL1W", "Invalid Author ID", ["bad-author"], ["Name"]),
        ),
        {
            "docs": [
                {
                    "key": "OL1W",
                    "title": "Invalid alternative title",
                    "alternative_title": "not-a-list",
                }
            ]
        },
        {
            "docs": [
                {
                    "key": "OL1W",
                    "title": "Invalid author alias",
                    "author_alternative_name": ["Alias", ["wrong-shape"]],
                }
            ]
        },
        {
            "docs": [
                {
                    "key": "OL1W",
                    "title": "Invalid nested Edition title",
                    "editions": {"docs": [{"title": ["wrong-shape"]}]},
                }
            ]
        },
        _work_search_response(
            _candidate(
                "OL1W",
                "Target",
                cover_id=802,
                cover_edition_key="not-an-edition",
            )
        ),
    ],
)
async def test_resolver_maps_invalid_search_responses_to_catalog_failure(
    response_payload: dict[str, object],
) -> None:
    """Reject malformed Search fields without returning a partial result."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_payload)

    resolver = _resolver(httpx.MockTransport(handle))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


@pytest.mark.parametrize(
    "work_payload",
    [
        {"type": {"key": "/type/work"}},
        {"type": {"key": "/type/redirect"}},
        {"type": {"key": "/type/redirect"}, "location": "bad-work"},
        {"type": {"key": "/type/unknown"}, "key": "/works/OL1W"},
    ],
)
async def test_resolver_maps_invalid_work_identity_records_to_catalog_failure(
    work_payload: dict[str, object],
) -> None:
    """Reject malformed terminal Work records and redirects safely."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return httpx.Response(200, json=work_payload)
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(httpx.MockTransport(handle))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


async def test_resolver_maps_cyclic_work_redirects_to_catalog_failure() -> None:
    """Stop cyclic Work redirects without guessing a canonical identity."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(200, json=_redirect_record("OL2W"))
        if request.url.path == "/works/OL2W.json":
            return httpx.Response(200, json=_redirect_record("OL1W"))
        return httpx.Response(200, json=_work_search_response(_candidate("OL1W", "Target")))

    resolver = _resolver(httpx.MockTransport(handle))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    await resolver.aclose()


async def test_resolver_retries_one_transient_transport_failure() -> None:
    """Retry one transient connection failure before surfacing a provider outage."""
    request_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle))

    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert request_count == 2
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
    assert request_count == 2

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
    assert request_count == 3

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
