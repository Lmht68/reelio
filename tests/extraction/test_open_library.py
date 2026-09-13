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
from reelio.extraction.types import (
    AuthorCredit,
    BookMention,
    BookMentions,
    BookResults,
    ResultStatus,
)

_SEARCH_FIELDS = (
    "key,title,alternative_title,author_key,author_name,"
    "author_alternative_name,editions,editions.key,editions.title"
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
    return candidate


def _search_response(*candidates: dict[str, object]) -> dict[str, object]:
    return {"docs": list(candidates)}


def _work_record(work_id: str) -> dict[str, object]:
    return {"type": {"key": "/type/work"}, "key": f"/works/{work_id}"}


def _redirect_record(work_id: str) -> dict[str, object]:
    return {"type": {"key": "/type/redirect"}, "location": f"/works/{work_id}"}


def _terminal_work_response(request: httpx.Request) -> httpx.Response:
    work_id = request.url.path.removeprefix("/works/").removesuffix(".json")
    return httpx.Response(200, json=_work_record(work_id))


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

    searches = [request for request in requests if request.url.path == "/search.json"]
    assert len(searches) == 1
    assert dict(searches[0].url.params) == {
        "title": "  Pride and Prejudice  ",
        "author": "JANE AUSTEN",
        "fields": _SEARCH_FIELDS,
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
            return httpx.Response(200, json=_search_response())
        return httpx.Response(
            200,
            json=_search_response(
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
            "fields": _SEARCH_FIELDS,
            "limit": "5",
        },
        {"title": "Dune", "fields": _SEARCH_FIELDS, "limit": "5"},
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
            json=_search_response(
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
        return httpx.Response(200, json=_search_response())

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
        return httpx.Response(200, json=_search_response(candidate))

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
            json=_search_response(
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
            json=_search_response(
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
            return httpx.Response(200, json=_search_response())
        return httpx.Response(
            200,
            json=_search_response(
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
            json=_search_response(
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
            return httpx.Response(200, json=_search_response())
        return httpx.Response(
            200,
            json=_search_response(
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
            json=_search_response(
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
            json=_search_response(
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
            json=_search_response(
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
        return httpx.Response(200, json=_search_response(_candidate("OL1W", "Anonymous Work")))

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
            json=_search_response(
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
        return httpx.Response(200, json=_search_response(_candidate("OL1W", "abcdefghijk")))

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
            return httpx.Response(200, json=_search_response())
        return httpx.Response(
            200,
            json=_search_response(
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
            json=_search_response(
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
            json=_search_response(
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
            json=_search_response(
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


async def test_resolver_spaces_search_and_work_requests_at_three_per_second() -> None:
    """Apply the shared dispatch limiter to all Search and Work identity requests."""
    clock = _FakeClock()
    dispatch_times: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        dispatch_times.append(clock())
        if request.url.path.startswith("/works/"):
            return _terminal_work_response(request)
        title = request.url.params["title"]
        work_id = {"One": "OL1W", "Two": "OL2W", "Three": "OL3W"}[title]
        return httpx.Response(200, json=_search_response(_candidate(work_id, title)))

    resolver = _resolver(httpx.MockTransport(handle), clock)

    await resolver.resolve(
        _mentions(
            BookMention(title="One", authors=[]),
            BookMention(title="Two", authors=[]),
            BookMention(title="Three", authors=[]),
        )
    )

    assert dispatch_times == pytest.approx([index / 3 for index in range(6)])
    assert clock.delays == pytest.approx([1 / 3] * 5)
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
        return httpx.Response(200, json=_search_response(_candidate("OL1W", "Target")))

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
        return httpx.Response(200, json=_search_response(_candidate("OL1W", "Target")))

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
