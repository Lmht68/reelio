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

    def advance(self, seconds: float) -> None:
        """Advance the deterministic monotonic clock without recording a sleep."""
        self.value += seconds

    async def sleep(self, delay_seconds: float) -> None:
        """Record and advance one deterministic request delay."""
        self.delays.append(delay_seconds)
        self.value += delay_seconds


def _resolver(
    handler: httpx.AsyncBaseTransport,
    clock: _FakeClock | None = None,
    *,
    settings: OpenLibraryConfig | None = None,
    default_missing_preferred_edition: bool = True,
) -> OpenLibraryBookResolver:
    fake_clock = clock or _FakeClock()
    transport = (
        _AbsentPreferredEditionTransport(handler) if default_missing_preferred_edition else handler
    )
    return OpenLibraryBookResolver(
        _client(transport),
        settings or _settings(),
        clock=fake_clock,
        sleep=fake_clock.sleep,
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


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    [
        ("Main", "Main:Subtitle"),
        ("Main", "Main :Subtitle"),
        ("Main", "Main: Subtitle"),
        ("Main", "Main : Subtitle"),
        ("Main", "Main - Subtitle"),
        ("Main", "Main – Subtitle"),
        ("Main: Subtitle", "Main"),
        ("  MAIN  ", " main : subtitle "),
    ],
)
async def test_resolver_equates_unique_exact_main_titles_with_one_subtitle(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Resolve equivalent main titles when exactly one title has a subtitle."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", candidate_title)),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    mention = BookMention(title=mention_title, authors=[])

    results = await resolver.resolve(_mentions(mention))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book_mention is mention
    assert result.book is not None
    assert result.book.title == candidate_title
    assert result.book.open_library_work_id == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    [
        ("Main", "Main - First: Second"),
        ("Main", "Main: First - Second"),
        ("Catch-22", "Catch-22: Subtitle"),
    ],
)
async def test_resolver_uses_leftmost_book_subtitle_separator(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Resolve from the leftmost valid boundary while retaining intrinsic hyphens."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", candidate_title)),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title=mention_title, authors=[])))

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_title", "candidate_title"),
    [
        ("Main", "Main-Subtitle"),
        ("Main", "Main -Subtitle"),
        ("Main", "Main- Subtitle"),
        ("Main", "Main–Subtitle"),
        ("Main", "Main –Subtitle"),
        ("Main", "Main– Subtitle"),
        ("Main", "Main:"),
        ("Main", ": Subtitle"),
        ("Main: Mention subtitle", "Main: Candidate subtitle"),
    ],
)
async def test_resolver_rejects_non_equivalent_book_subtitle_boundaries(
    mention_title: str,
    candidate_title: str,
) -> None:
    """Reject malformed boundaries and pairs with subtitles on both titles."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", candidate_title)),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title=mention_title, authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.UNRESOLVED
    assert result.book is None
    await resolver.aclose()


@pytest.mark.parametrize(
    ("candidate", "expected_title"),
    [
        (_candidate("OL1W", "Main: Subtitle"), "Main: Subtitle"),
        (
            _candidate(
                "OL1W",
                "Provider Primary Title",
                alternative_titles=["Main: Subtitle"],
            ),
            "Provider Primary Title",
        ),
        (
            _candidate(
                "OL1W",
                "Provider Primary Title",
                edition_titles=["Main: Subtitle", "Ignored Edition Title"],
            ),
            "Provider Primary Title",
        ),
    ],
)
async def test_resolver_uses_all_candidate_title_sources_for_subtitle_equivalence(
    candidate: dict[str, object],
    expected_title: str,
) -> None:
    """Use Work, alternative Work, and selected Edition Candidate titles."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(200, json=_work_search_response(candidate))

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Main", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.book is not None
    assert result.book.title == expected_title
    assert result.book.open_library_work_id == "OL1W"
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_author", "provider_author", "author_alias"),
    [
        ("Provider Author", "Provider Author", None),
        ("Provider Alias", "Provider Author", "Provider Alias"),
    ],
)
async def test_resolver_requires_exact_provider_author_credit_for_subtitle_equivalence(
    mention_author: str,
    provider_author: str,
    author_alias: str | None,
) -> None:
    """Require an exact provider name or alias before authorful equivalence."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Main: Subtitle",
                    ["OL1A"],
                    [provider_author],
                    author_aliases=[author_alias] if author_alias is not None else None,
                )
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="Main",
                authors=[AuthorCredit(name=mention_author)],
            )
        )
    )

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_excludes_mismatched_author_from_subtitle_equivalence() -> None:
    """Resolve only the exact-author Candidate among shared-main-title Works."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Shared Main: First", ["OL1A"], ["Other Author"]),
                _candidate("OL2W", "Shared Main - Second", ["OL2A"], ["Matching Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="Shared Main",
                authors=[AuthorCredit(name="Matching Author")],
            )
        )
    )

    assert _resolved_work_id(results) == "OL2W"
    await resolver.aclose()


async def test_resolver_counts_one_work_once_across_matching_candidate_titles() -> None:
    """Resolve one canonical Work even when several of its titles qualify."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Shared Main: First",
                    alternative_titles=["Shared Main - Second"],
                    edition_titles=["Shared Main – Third"],
                )
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Shared Main", authors=[])))

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_leaves_distinct_exact_main_title_works_ambiguous() -> None:
    """Leave an authorless Mention unresolved when two Works qualify."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Shared Main: First"),
                _candidate("OL2W", "Shared Main - Second"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Shared Main", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.UNRESOLVED
    assert result.book is None
    await resolver.aclose()


@pytest.mark.parametrize(
    ("canonicalize_together", "expected_work_id"),
    [(True, "OL3W"), (False, None)],
)
async def test_resolver_counts_subtitle_matches_by_canonical_work_id(
    canonicalize_together: bool,
    expected_work_id: str | None,
) -> None:
    """Resolve redirect aliases once and keep distinct canonical Works ambiguous."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            work_id = request.url.path.removeprefix("/works/").removesuffix(".json")
            if canonicalize_together and work_id in {"OL1W", "OL2W"}:
                return httpx.Response(200, json=_redirect_record("OL3W"))
            return httpx.Response(200, json=_work_record(work_id))
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Shared Main: First"),
                _candidate("OL2W", "Shared Main - Second"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="Shared Main", authors=[])))

    result = results.books[0]
    if expected_work_id is None:
        assert result.status is ResultStatus.UNRESOLVED
        assert result.book is None
    else:
        assert _resolved_work_id(results) == expected_work_id
    await resolver.aclose()


@pytest.mark.parametrize(
    ("candidate_title", "expected_work_id"),
    [
        ("abcdefgxij: A deliberately long subtitle", "OL1W"),
        ("abcdefxyij: A deliberately long subtitle", None),
    ],
)
async def test_resolver_applies_strict_fuzzy_main_title_threshold(
    candidate_title: str,
    expected_work_id: str | None,
) -> None:
    """Resolve only main-title fuzzy scores strictly above the threshold."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", candidate_title)),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    result = results.books[0]
    if expected_work_id is None:
        assert result.status is ResultStatus.UNRESOLVED
        assert result.book is None
    else:
        assert _resolved_work_id(results) == expected_work_id
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_title", "candidate_title", "expected_work_id"),
    [
        ("abcdef", "abcdeg: A deliberately long subtitle", "OL1W"),
        ("abcde", "abcdex: A deliberately long subtitle", None),
    ],
)
async def test_resolver_requires_six_normalized_characters_for_fuzzy_main_titles(
    mention_title: str,
    candidate_title: str,
    expected_work_id: str | None,
) -> None:
    """Require six normalized main-title code points before fuzzy scoring."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", candidate_title)),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title=mention_title, authors=[])))

    result = results.books[0]
    if expected_work_id is None:
        assert result.status is ResultStatus.UNRESOLVED
        assert result.book is None
    else:
        assert _resolved_work_id(results) == expected_work_id
    await resolver.aclose()


@pytest.mark.parametrize(
    ("exact_candidate_titles", "expected_work_id"),
    [
        ((("OL1W", "abcdefghij: Exact subtitle"),), "OL1W"),
        (
            (
                ("OL1W", "abcdefghij: Exact subtitle"),
                ("OL3W", "abcdefghij - Second exact subtitle"),
            ),
            None,
        ),
    ],
)
async def test_resolver_applies_exact_main_title_precedence_before_fuzzy_main_titles(
    exact_candidate_titles: tuple[tuple[str, str], ...],
    expected_work_id: str | None,
) -> None:
    """Apply exact equivalence before fuzzy resolution without breaking ambiguity."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        exact_candidates = [
            _candidate(work_id, candidate_title)
            for work_id, candidate_title in exact_candidate_titles
        ]
        return httpx.Response(
            200,
            json=_work_search_response(
                *exact_candidates,
                _candidate("OL2W", "abcdefgxij: Approximate subtitle"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    result = results.books[0]
    if expected_work_id is None:
        assert result.status is ResultStatus.UNRESOLVED
        assert result.book is None
    else:
        assert _resolved_work_id(results) == expected_work_id
    await resolver.aclose()


async def test_resolver_leaves_multiple_fuzzy_main_title_works_ambiguous() -> None:
    """Do not rank distinct fuzzy main-title Candidates by score or source order."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefgxij: First"),
                _candidate("OL2W", "abcdefghix: Second"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    result = results.books[0]
    assert result.status is ResultStatus.UNRESOLVED
    assert result.book is None
    await resolver.aclose()


async def test_resolver_counts_one_work_once_across_fuzzy_candidate_titles() -> None:
    """Count one canonical Work once when several of its titles are fuzzy matches."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "abcdefgxij: First",
                    alternative_titles=["abcdefghix - Second"],
                    edition_titles=["abcdefgzij – Third"],
                )
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    assert _resolved_work_id(results) == "OL1W"
    await resolver.aclose()


async def test_resolver_excludes_author_mismatches_from_fuzzy_main_title_ambiguity() -> None:
    """Exclude mismatched Authors before fuzzy main-title ambiguity evaluation."""

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
                    "abcdefgxij: A deliberately long subtitle",
                    ["OL1A"],
                    ["Matching Author"],
                ),
                _candidate(
                    "OL2W",
                    "abcdefghix: Another deliberately long subtitle",
                    ["OL2A"],
                    ["Mismatched Author"],
                ),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="abcdefghij",
                authors=[AuthorCredit(name="Matching Author")],
            )
        )
    )

    assert _resolved_work_id(results) == "OL1W"
    searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and "q" not in request.url.params
    ]
    assert [dict(request.url.params) for request in searches] == [
        {
            "title": "abcdefghij",
            "author": "Matching Author",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        }
    ]
    await resolver.aclose()


async def test_resolver_falls_back_after_ambiguous_constrained_fuzzy_main_titles() -> None:
    """Use the title-only window after constrained fuzzy main-title ambiguity."""

    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" in request.url.params:
            return httpx.Response(
                200,
                json=_work_search_response(
                    _candidate(
                        "OL1W",
                        "abcdefgxij: First constrained subtitle",
                        ["OL1A"],
                        ["Matching Author"],
                    ),
                    _candidate(
                        "OL2W",
                        "abcdefghix: Second constrained subtitle",
                        ["OL2A"],
                        ["Matching Author"],
                    ),
                ),
            )
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL3W",
                    "abcdefgzij: A title-only fallback subtitle",
                    ["OL3A"],
                    ["Matching Author"],
                )
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(
            BookMention(
                title="abcdefghij",
                authors=[AuthorCredit(name="Matching Author")],
            )
        )
    )

    assert _resolved_work_id(results) == "OL3W"
    searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and "q" not in request.url.params
    ]
    assert [dict(request.url.params) for request in searches] == [
        {
            "title": "abcdefghij",
            "author": "Matching Author",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        },
        {
            "title": "abcdefghij",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        },
    ]
    await resolver.aclose()


async def test_resolver_counts_fuzzy_main_title_redirect_aliases_once() -> None:
    """Resolve multiple fuzzy redirect aliases as one canonical Work."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            work_id = request.url.path.removeprefix("/works/").removesuffix(".json")
            if work_id in {"OL1W", "OL2W"}:
                return httpx.Response(200, json=_redirect_record("OL3W"))
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "abcdefgxij: First"),
                _candidate("OL2W", "abcdefghix: Second"),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(_mentions(BookMention(title="abcdefghij", authors=[])))

    assert len(results.books) == 1
    assert _resolved_work_id(results) == "OL3W"
    await resolver.aclose()


async def test_resolver_prefers_complete_title_exact_match_over_subtitle_equivalence() -> None:
    """Keep complete-title exact verification ahead of subtitle equivalence."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Main: Subtitle", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Main", ["OL2A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Main", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL2W"
    await resolver.aclose()


async def test_resolver_prefers_complete_title_fuzzy_match_over_subtitle_equivalence() -> None:
    """Keep complete-title fuzzy verification ahead of subtitle equivalence."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate("OL1W", "Main: Subtitle", ["OL1A"], ["Author"]),
                _candidate("OL2W", "Mainx", ["OL2A"], ["Author"]),
            ),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Main", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL2W"
    await resolver.aclose()


async def test_resolver_stops_after_constrained_subtitle_equivalence() -> None:
    """Avoid a title-only Search after the constrained subtitle stage resolves."""

    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL1W", "Main: Subtitle", ["OL1A"], ["Author"])),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Main", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL1W"
    searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and "q" not in request.url.params
    ]
    assert [dict(request.url.params) for request in searches] == [
        {
            "title": "Main",
            "author": "Author",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        }
    ]
    await resolver.aclose()


@pytest.mark.parametrize(
    "constrained_candidates",
    [
        [
            _candidate("OL1W", "Main: First", ["OL1A"], ["Author"]),
            _candidate("OL2W", "Main - Second", ["OL2A"], ["Author"]),
        ],
        [_candidate("OL1W", "Unrelated Work", ["OL1A"], ["Author"])],
    ],
)
async def test_resolver_falls_back_after_failed_or_ambiguous_constrained_subtitle_stage(
    constrained_candidates: list[dict[str, object]],
) -> None:
    """Issue one title-only Search after a constrained subtitle stage cannot resolve."""

    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        if "author" in request.url.params:
            return httpx.Response(
                200,
                json=_work_search_response(*constrained_candidates),
            )
        return httpx.Response(
            200,
            json=_work_search_response(_candidate("OL3W", "Main: Fallback", ["OL3A"], ["Author"])),
        )

    resolver = _resolver(httpx.MockTransport(handle))
    results = await resolver.resolve(
        _mentions(BookMention(title="Main", authors=[AuthorCredit(name="Author")]))
    )

    assert _resolved_work_id(results) == "OL3W"
    searches = [
        request
        for request in requests
        if request.url.path == "/search.json" and "q" not in request.url.params
    ]
    assert [dict(request.url.params) for request in searches] == [
        {
            "title": "Main",
            "author": "Author",
            "fields": _WORK_SEARCH_FIELDS,
            "limit": "5",
        },
        {"title": "Main", "fields": _WORK_SEARCH_FIELDS, "limit": "5"},
    ]
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
        if request_count <= 2:
            return httpx.Response(500)
        raise httpx.ReadTimeout("timed out", request=request)

    resolver = _resolver(httpx.MockTransport(handle))
    mentions = _mentions(BookMention(title="Target", authors=[]))

    with pytest.raises(CatalogProviderError, match="Open Library catalog request failed"):
        await resolver.resolve(mentions)
    with pytest.raises(PipelineTimeoutError, match="Open Library catalog request timed out"):
        await resolver.resolve(mentions)
    assert request_count == 3

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


async def test_resolver_caches_empty_search_until_ttl_expiry() -> None:
    """Reuse a valid empty Search until its fixed 24-hour cache entry expires."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle), clock)
    mentions = _mentions(BookMention(title="No Match", authors=[]))

    initial_results = await resolver.resolve(mentions)
    clock.advance(86_399.999)
    cached_results = await resolver.resolve(mentions)
    clock.advance(0.001)
    expired_results = await resolver.resolve(mentions)

    assert all(
        results.books[0].status is ResultStatus.UNRESOLVED
        for results in (initial_results, cached_results, expired_results)
    )
    assert dispatch_times == pytest.approx([0.0, 86_400.0])
    await resolver.aclose()


async def test_resolver_keeps_recent_search_cache_entries_when_capacity_is_reached() -> None:
    """Evict the least-recently-used successful Search cache entry at capacity."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle))

    for index in range(100):
        results = await resolver.resolve(_mentions(BookMention(title=f"Title {index}", authors=[])))
        assert results.books[0].status is ResultStatus.UNRESOLVED
    await resolver.resolve(_mentions(BookMention(title="Title 0", authors=[])))
    await resolver.resolve(_mentions(BookMention(title="Title 100", authors=[])))
    await resolver.resolve(_mentions(BookMention(title="Title 1", authors=[])))

    assert len(requests) == 102
    assert [request.url.params["title"] for request in requests[-2:]] == [
        "Title 100",
        "Title 1",
    ]
    await resolver.aclose()


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
async def test_resolver_retries_transient_http_failures_once(
    status_code: int,
) -> None:
    """Retry each retryable HTTP status once through the shared dispatch limiter."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if len(dispatch_times) == 1:
            return httpx.Response(status_code)
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle), clock)

    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert dispatch_times == pytest.approx([0.0, 1 / 3])
    await resolver.aclose()


async def test_resolver_retries_transient_transport_failure_once() -> None:
    """Retry one non-timeout transport failure through the shared dispatch limiter."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if len(dispatch_times) == 1:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle), clock)

    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert dispatch_times == pytest.approx([0.0, 1 / 3])
    await resolver.aclose()


async def test_resolver_honors_fitting_retry_after_before_retrying() -> None:
    """Honor a fitting 429 Retry-After delay before the one allowed retry."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if len(dispatch_times) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle), clock)

    results = await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert dispatch_times == pytest.approx([0.0, 2.0])
    await resolver.aclose()


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
async def test_resolver_stops_after_second_transient_http_failure(
    status_code: int,
) -> None:
    """Return the stable provider error after exactly two transient HTTP failures."""
    dispatch_times: list[float] = []
    clock = _FakeClock()

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        headers = {"Retry-After": "0"} if status_code == 429 else {}
        return httpx.Response(status_code, headers=headers)

    resolver = _resolver(httpx.MockTransport(handle), clock)

    with pytest.raises(
        CatalogProviderError,
        match=r"^Open Library catalog request failed\.$",
    ):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert dispatch_times == pytest.approx([0.0, 1 / 3])
    await resolver.aclose()


@pytest.mark.parametrize(
    ("headers", "settings"),
    [
        (None, _settings()),
        ({"Retry-After": "invalid"}, _settings()),
        ({"Retry-After": "-1"}, _settings()),
        ({"Retry-After": "2"}, _settings(request_timeout_seconds=1.0)),
    ],
)
async def test_resolver_rejects_unusable_retry_after(
    headers: dict[str, str] | None,
    settings: OpenLibraryConfig,
) -> None:
    """Fail a 429 without retrying when its Retry-After cannot fit the deadline."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers=headers)

    resolver = _resolver(httpx.MockTransport(handle), settings=settings)

    with pytest.raises(
        CatalogProviderError,
        match=r"^Open Library catalog request failed\.$",
    ):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert len(requests) == 1
    await resolver.aclose()


async def test_resolver_does_not_retry_nontransient_http_or_timeout_failures() -> None:
    """Map a 4xx and a timeout after one request without leaking provider details."""
    http_requests: list[httpx.Request] = []

    async def handle_http(request: httpx.Request) -> httpx.Response:
        http_requests.append(request)
        return httpx.Response(404)

    http_resolver = _resolver(httpx.MockTransport(handle_http))
    with pytest.raises(
        CatalogProviderError,
        match=r"^Open Library catalog request failed\.$",
    ):
        await http_resolver.resolve(_mentions(BookMention(title="Target", authors=[])))
    assert len(http_requests) == 1
    await http_resolver.aclose()

    timeout_requests: list[httpx.Request] = []

    async def handle_timeout(request: httpx.Request) -> httpx.Response:
        timeout_requests.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    timeout_resolver = _resolver(httpx.MockTransport(handle_timeout))
    with pytest.raises(
        PipelineTimeoutError,
        match=r"^Open Library catalog request timed out\.$",
    ):
        await timeout_resolver.resolve(_mentions(BookMention(title="Target", authors=[])))
    assert len(timeout_requests) == 1
    await timeout_resolver.aclose()


async def test_resolver_retries_after_malformed_response_without_caching_failure() -> None:
    """Fetch a later valid Search after a malformed response fails validation."""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"docs": [{"title": "Missing Work ID"}]})
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(httpx.MockTransport(handle))
    mentions = _mentions(BookMention(title="Target", authors=[]))

    with pytest.raises(
        CatalogProviderError,
        match=r"^Open Library catalog request failed\.$",
    ):
        await resolver.resolve(mentions)
    results = await resolver.resolve(mentions)

    assert results.books[0].status is ResultStatus.UNRESOLVED
    assert len(requests) == 2
    await resolver.aclose()


async def test_resolver_timeout_prevents_retry_beyond_lookup_deadline() -> None:
    """Return the timeout error before a delayed retry can reach its limiter slot."""
    clock = _FakeClock()
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    resolver = _resolver(
        httpx.MockTransport(handle),
        clock,
        settings=_settings(request_timeout_seconds=0.1),
    )

    with pytest.raises(
        PipelineTimeoutError,
        match=r"^Open Library catalog request timed out\.$",
    ):
        await resolver.resolve(_mentions(BookMention(title="Target", authors=[])))

    assert len(requests) == 1
    await resolver.aclose()


async def test_resolver_coalesces_provider_rich_identical_lookups() -> None:
    """Share each provider request while allocating independent public result lists."""
    arrivals: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    gates: dict[tuple[str, str], asyncio.Event] = {}
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        request_key = (
            request.url.path,
            request.url.params.get("q") or request.url.params.get("title") or "",
        )
        requests.append(request)
        gate = gates.setdefault(request_key, asyncio.Event())
        await arrivals.put(request_key)
        await gate.wait()
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(200, json=_work_record("OL1W"))
        if request.url.path == "/books/OL101M.json":
            return httpx.Response(
                200,
                json=_edition_record(
                    "OL101M",
                    title="Provider Edition",
                    publishers=["Publisher"],
                    isbn_10=["0123456789"],
                    isbn_13=["9780123456786"],
                    covers=[101],
                ),
            )
        if _is_preferred_edition_search(request):
            return httpx.Response(
                200,
                json=_preferred_edition_search_response(
                    "OL1W",
                    _selected_edition("OL101M", title="Provider Edition", cover_id=101),
                ),
            )
        return httpx.Response(
            200,
            json=_work_search_response(
                _candidate(
                    "OL1W",
                    "Provider Work",
                    ["OL1A"],
                    ["Provider Author"],
                    cover_id=802,
                    cover_edition_key="/books/OL802M",
                )
            ),
        )

    resolver = _resolver(
        httpx.MockTransport(handle),
        default_missing_preferred_edition=False,
    )
    mentions = _mentions(
        BookMention(title="Provider Work", authors=[AuthorCredit(name="Provider Author")])
    )
    first_lookup = asyncio.create_task(resolver.resolve(mentions))
    second_lookup = asyncio.create_task(resolver.resolve(mentions))

    expected_request_keys = [
        ("/search.json", "Provider Work"),
        ("/works/OL1W.json", ""),
        ("/search.json", "key:/works/OL1W AND language:eng"),
        ("/books/OL101M.json", ""),
    ]
    for expected_request_key in expected_request_keys:
        assert await arrivals.get() == expected_request_key
        await asyncio.sleep(0)
        assert arrivals.empty()
        gates[expected_request_key].set()

    first_results, second_results = await asyncio.gather(first_lookup, second_lookup)
    first_book = first_results.books[0].book
    second_book = second_results.books[0].book
    assert first_book is not None
    assert second_book is not None
    assert first_book.title == second_book.title == "Provider Work"
    assert first_book.authors == second_book.authors
    assert first_book.authors is not second_book.authors
    assert first_book.authors[0].open_library_author_id == "OL1A"
    assert first_book.edition is not None
    assert second_book.edition is not None
    assert first_book.edition.publishers == second_book.edition.publishers == ["Publisher"]
    assert first_book.edition.publishers is not second_book.edition.publishers
    assert first_book.edition.isbn_10 == second_book.edition.isbn_10 == ["0123456789"]
    assert first_book.edition.isbn_13 == second_book.edition.isbn_13 == ["9780123456786"]
    assert (
        first_book.cover_url
        == second_book.cover_url
        == ("https://covers.openlibrary.org/b/id/101-L.jpg")
    )
    assert first_book.cover_edition_id == second_book.cover_edition_id == "OL101M"
    assert len(requests) == 4

    cached_results = await resolver.resolve(mentions)
    cached_book = cached_results.books[0].book
    assert cached_book is not None
    assert cached_book.authors == first_book.authors
    assert cached_book.authors is not first_book.authors
    assert cached_book.edition is not None
    assert cached_book.edition.publishers == first_book.edition.publishers
    assert cached_book.edition.publishers is not first_book.edition.publishers
    assert len(requests) == 4
    await resolver.aclose()


async def test_resolver_shares_coalesced_timeout_and_retries_later() -> None:
    """Give coalesced waiters one timeout and retry with a fresh provider request."""
    request_started = asyncio.Event()
    requests: list[httpx.Request] = []
    clock = _FakeClock()

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            request_started.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=_work_search_response())

    resolver = _resolver(
        httpx.MockTransport(handle),
        clock,
        settings=_settings(request_timeout_seconds=0.01),
    )
    mentions = _mentions(BookMention(title="Target", authors=[]))
    first_lookup = asyncio.create_task(resolver.resolve(mentions))
    second_lookup = asyncio.create_task(resolver.resolve(mentions))

    await request_started.wait()
    outcomes = await asyncio.gather(first_lookup, second_lookup, return_exceptions=True)

    assert all(isinstance(outcome, PipelineTimeoutError) for outcome in outcomes)
    assert len(requests) == 1
    clock.advance(1 / 3)
    later_results = await resolver.resolve(mentions)
    assert later_results.books[0].status is ResultStatus.UNRESOLVED
    assert len(requests) == 2
    await resolver.aclose()


class _TrackingTransport(httpx.AsyncBaseTransport):
    """Block one request until resolver shutdown closes this transport."""

    def __init__(self) -> None:
        """Initialize observable request and closure state."""
        self.request_started = asyncio.Event()
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Block the request until resolver shutdown cancels its producer task."""
        self.request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("The tracked request should be cancelled before responding.")

    async def aclose(self) -> None:
        """Record closure by the resolver-owned HTTP client."""
        self.closed = True


async def test_resolver_aclose_cancels_pending_producer_and_closes_transport() -> None:
    """Cancel and drain an in-flight shielded lookup before closing the owned client."""
    transport = _TrackingTransport()
    resolver = _resolver(transport)
    lookup = asyncio.create_task(
        resolver.resolve(_mentions(BookMention(title="Target", authors=[])))
    )

    await transport.request_started.wait()
    await resolver.aclose()

    with pytest.raises(asyncio.CancelledError):
        await lookup
    assert transport.closed is True
    assert resolver._client.is_closed is True
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
