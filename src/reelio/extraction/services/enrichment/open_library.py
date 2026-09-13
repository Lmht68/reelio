"""Resolve exact Book Work Mentions through Open Library Search."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from time import monotonic
from typing import cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rapidfuzz import fuzz

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import OpenLibraryConfig
from reelio.extraction.types import (
    BookMention,
    BookMentions,
    BookResult,
    BookResults,
    EnrichedAuthorCredit,
    EnrichedBookWork,
    ResultStatus,
    normalize_book_identity,
    normalize_book_text,
)

logger = logging.getLogger(__name__)

_CATALOG_ERROR_MESSAGE = "Open Library catalog request failed."
_TIMEOUT_ERROR_MESSAGE = "Open Library catalog request timed out."
_STAGE = "book_work_resolution"
_SEARCH_LIMIT = 5
_FUZZY_TITLE_SCORE_THRESHOLD = 80.0
_SEARCH_FIELDS = (
    "key,title,alternative_title,author_key,author_name,"
    "author_alternative_name,editions,editions.key,editions.title"
)
_WORK_ID_PATTERN = re.compile(r"(?:/works/)?(OL[0-9]+W)")
_AUTHOR_ID_PATTERN = re.compile(r"(?:/authors/)?(OL[0-9]+A)")


class _OpenLibraryModel(BaseModel):
    """Ignore unknown Open Library fields at the resolver boundary."""

    model_config = ConfigDict(extra="ignore")


class _OpenLibrarySearchEditionModel(_OpenLibraryModel):
    """Model the query-selected Edition title used only for identity matching."""

    title: str = Field(min_length=1)


class _OpenLibrarySearchEditionsModel(_OpenLibraryModel):
    """Model nested query-selected Edition Search fields."""

    docs: list[_OpenLibrarySearchEditionModel]


class _OpenLibrarySearchCandidateModel(_OpenLibraryModel):
    """Model the Search fields needed for exact Book Work resolution."""

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    author_key: list[str] | None = None
    author_name: list[str] | None = None
    alternative_title: list[str] | None = None
    author_alternative_name: list[str] | None = None
    editions: _OpenLibrarySearchEditionsModel | None = None


class _OpenLibraryWorkTypeModel(_OpenLibraryModel):
    """Model the type discriminator on a Work identity response."""

    key: str = Field(min_length=1)


class _OpenLibraryWorkRecordModel(_OpenLibraryModel):
    """Model a terminal Work record or a redirect in Work identity resolution."""

    type: _OpenLibraryWorkTypeModel
    key: str | None = None
    location: str | None = None


class _OpenLibrarySearchResponseModel(_OpenLibraryModel):
    """Model an Open Library Search response with its required candidate list."""

    docs: list[_OpenLibrarySearchCandidateModel]


@dataclass(frozen=True, slots=True)
class _OpenLibraryCandidate:
    """Contain bounded Work metadata used for Book Work matching."""

    title: str
    authors: list[EnrichedAuthorCredit]
    open_library_work_id: str
    candidate_titles: tuple[str, ...]
    author_aliases: tuple[str, ...]


class OpenLibraryBookResolver:
    """Resolve Book Work Mentions with exact Open Library Search matches."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        requests_per_second: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        """Initialize a resolver with injectable HTTP and monotonic timing boundaries.

        Args:
            client: Reusable Open Library HTTP client owned by this resolver.
            requests_per_second: Maximum resolver-wide request dispatch rate.
            clock: Monotonic clock used to schedule request dispatches.
            sleep: Awaitable delay used to enforce the dispatch interval.
        """
        self._client = client
        self._request_interval_seconds = 1 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._request_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def resolve(self, book_mentions: BookMentions) -> BookResults:
        """Resolve Book Work Mentions concurrently while preserving Mention order.

        Args:
            book_mentions: Canonical Book Work Mentions in first-reference order.

        Returns:
            BookResults: Resolved or unresolved Book Work Results in Mention order.

        Raises:
            CatalogProviderError: If Open Library fails or returns invalid provider data.
            PipelineTimeoutError: If an Open Library request times out.
        """
        if not book_mentions.books:
            return BookResults(books=[])

        results = await asyncio.gather(
            *(self._resolve_mention(book_mention) for book_mention in book_mentions.books)
        )
        return BookResults(books=_drop_duplicate_resolved_book_works(list(results)))

    async def aclose(self) -> None:
        """Close the resolver-owned Open Library HTTP client."""
        await self._client.aclose()

    async def _resolve_mention(self, book_mention: BookMention) -> BookResult:
        raw_to_canonical_work_ids: dict[str, str] = {}
        seen_work_ids: set[str] = set()
        if book_mention.authors:
            candidates = await self._canonicalize_candidates(
                await self._search(
                    book_mention.title,
                    author=book_mention.authors[0].name,
                ),
                raw_to_canonical_work_ids,
                seen_work_ids,
            )
            candidate = _select_authorful_candidate(book_mention, candidates)
            if candidate is None:
                candidates = await self._canonicalize_candidates(
                    await self._search(book_mention.title),
                    raw_to_canonical_work_ids,
                    seen_work_ids,
                )
                candidate = _select_authorful_candidate(book_mention, candidates)
        else:
            candidates = await self._canonicalize_candidates(
                await self._search(book_mention.title),
                raw_to_canonical_work_ids,
                seen_work_ids,
            )
            candidate = _select_unique_exact_candidate(book_mention, candidates)

        if candidate is None:
            return BookResult(
                status=ResultStatus.UNRESOLVED,
                book_mention=book_mention,
                book=None,
            )
        return BookResult(
            status=ResultStatus.RESOLVED,
            book_mention=book_mention,
            book=EnrichedBookWork(
                title=candidate.title,
                authors=candidate.authors,
                open_library_work_id=candidate.open_library_work_id,
                open_library_url=(
                    f"https://openlibrary.org/works/{candidate.open_library_work_id}"
                ),
            ),
        )

    async def _search(
        self,
        title: str,
        author: str | None = None,
    ) -> list[_OpenLibraryCandidate]:
        params: dict[str, str | int] = {
            "title": title,
            "fields": _SEARCH_FIELDS,
            "limit": _SEARCH_LIMIT,
        }
        if author is not None:
            params["author"] = author

        response = await self._get_catalog_response("/search.json", params)

        try:
            response_model = _OpenLibrarySearchResponseModel.model_validate(
                cast(object, response.json())
            )
            return [_to_candidate(candidate) for candidate in response_model.docs[:_SEARCH_LIMIT]]
        except (ValidationError, ValueError) as exc:
            logger.error(
                "Open Library catalog response validation failed",
                extra={"stage": _STAGE, "reason": "invalid_provider_response"},
            )
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc

    async def _get_catalog_response(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        try:
            await self._await_request_turn()
            response = await self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error(
                "Open Library catalog request timed out",
                extra={"stage": _STAGE, "reason": "timeout"},
            )
            raise PipelineTimeoutError(_TIMEOUT_ERROR_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.error(
                "Open Library catalog request failed",
                extra={"stage": _STAGE, "reason": "provider_failure"},
            )
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc
        return response

    async def _canonicalize_candidates(
        self,
        candidates: list[_OpenLibraryCandidate],
        raw_to_canonical_work_ids: dict[str, str],
        seen_work_ids: set[str],
    ) -> list[_OpenLibraryCandidate]:
        canonical_candidates: list[_OpenLibraryCandidate] = []
        for candidate in candidates:
            raw_work_id = candidate.open_library_work_id
            canonical_work_id = raw_to_canonical_work_ids.get(raw_work_id)
            if canonical_work_id is None:
                canonical_work_id = await self._resolve_canonical_work_id(raw_work_id)
                raw_to_canonical_work_ids[raw_work_id] = canonical_work_id
            if canonical_work_id in seen_work_ids:
                continue
            seen_work_ids.add(canonical_work_id)
            canonical_candidates.append(replace(candidate, open_library_work_id=canonical_work_id))
        return canonical_candidates

    async def _resolve_canonical_work_id(self, work_id: str) -> str:
        seen_redirect_ids: set[str] = set()
        current_work_id = work_id
        while current_work_id not in seen_redirect_ids:
            seen_redirect_ids.add(current_work_id)
            response = await self._get_catalog_response(f"/works/{current_work_id}.json")
            try:
                record = _OpenLibraryWorkRecordModel.model_validate(cast(object, response.json()))
                if record.type.key == "/type/work":
                    if record.key is None:
                        raise ValueError("Open Library Work record is missing its key")
                    return _parse_provider_id(record.key, _WORK_ID_PATTERN, "Work")
                if record.type.key == "/type/redirect":
                    if record.location is None:
                        raise ValueError("Open Library Work redirect is missing its location")
                    current_work_id = _parse_provider_id(
                        record.location,
                        _WORK_ID_PATTERN,
                        "Work",
                    )
                    continue
                raise ValueError("Open Library Work record has an invalid type")
            except (ValidationError, ValueError) as exc:
                logger.error(
                    "Open Library catalog response validation failed",
                    extra={"stage": _STAGE, "reason": "invalid_provider_response"},
                )
                raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc
        logger.error(
            "Open Library catalog response validation failed",
            extra={"stage": _STAGE, "reason": "invalid_provider_response"},
        )
        raise CatalogProviderError(_CATALOG_ERROR_MESSAGE)

    async def _await_request_turn(self) -> None:
        async with self._request_lock:
            delay_seconds = self._next_request_at - self._clock()
            if delay_seconds > 0:
                await self._sleep(delay_seconds)
            self._next_request_at = self._clock() + self._request_interval_seconds


def create_open_library_book_resolver(
    settings: OpenLibraryConfig,
) -> OpenLibraryBookResolver:
    """Create a reusable Open Library Book Work resolver.

    Args:
        settings: Validated Open Library contact, endpoint, timeout, and request rate.

    Returns:
        OpenLibraryBookResolver: Resolver owning one asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url}/",
        headers={"User-Agent": f"Reelio ({settings.contact_email})"},
        timeout=settings.request_timeout_seconds,
    )
    return OpenLibraryBookResolver(
        client,
        settings.requests_per_second,
        monotonic,
        asyncio.sleep,
    )


def _to_candidate(
    candidate_model: _OpenLibrarySearchCandidateModel,
) -> _OpenLibraryCandidate:
    work_id = _parse_provider_id(candidate_model.key, _WORK_ID_PATTERN, "Work")
    author_keys = candidate_model.author_key
    author_names = candidate_model.author_name
    if author_keys is None and author_names is None:
        authors: list[EnrichedAuthorCredit] = []
    elif author_keys is None or author_names is None or len(author_keys) != len(author_names):
        raise ValueError("Open Library candidate author arrays must be paired")
    else:
        authors = []
        for author_id, author_name in zip(author_keys, author_names, strict=True):
            open_library_author_id = _parse_provider_id(
                author_id,
                _AUTHOR_ID_PATTERN,
                "Author",
            )
            authors.append(
                EnrichedAuthorCredit(
                    open_library_author_id=open_library_author_id,
                    name=_require_provider_text(author_name, "author name"),
                    open_library_url=(f"https://openlibrary.org/authors/{open_library_author_id}"),
                )
            )
    primary_title = _require_provider_text(candidate_model.title, "title")
    alternative_titles = tuple(
        _require_provider_text(title, "alternative title")
        for title in candidate_model.alternative_title or []
    )
    selected_edition_title: tuple[str, ...] = ()
    if candidate_model.editions is not None and candidate_model.editions.docs:
        selected_edition_title = (
            _require_provider_text(
                candidate_model.editions.docs[0].title,
                "query-selected Edition title",
            ),
        )
    author_aliases = tuple(
        _require_provider_text(author_alias, "author alternative name")
        for author_alias in candidate_model.author_alternative_name or []
    )
    return _OpenLibraryCandidate(
        title=primary_title,
        authors=authors,
        open_library_work_id=work_id,
        candidate_titles=_unique_candidate_titles(
            primary_title,
            *alternative_titles,
            *selected_edition_title,
        ),
        author_aliases=author_aliases,
    )


def _unique_candidate_titles(*titles: str) -> tuple[str, ...]:
    """Preserve the first title for each normalized form within one Candidate."""

    seen_title_forms: set[str] = set()
    unique_titles: list[str] = []
    for title in titles:
        normalized_title = normalize_book_identity(title)
        if normalized_title in seen_title_forms:
            continue
        seen_title_forms.add(normalized_title)
        unique_titles.append(title)
    return tuple(unique_titles)


def _parse_provider_id(value: str, pattern: re.Pattern[str], kind: str) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"Open Library {kind} ID is invalid")
    return match.group(1)


def _require_provider_text(value: str, field_name: str) -> str:
    if not normalize_book_text(value):
        raise ValueError(f"Open Library {field_name} is blank")
    return value


def _select_unique_exact_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    """Return the sole exact Candidate for an authorless Book Mention."""

    normalized_title = normalize_book_identity(book_mention.title)
    exact_candidates = [
        candidate
        for candidate in candidates
        if _candidate_has_exact_title(candidate, normalized_title)
    ]
    return exact_candidates[0] if len(exact_candidates) == 1 else None


def _select_authorful_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    """Return an exact-first, bounded fuzzy title match for an authorful Mention."""

    normalized_title = normalize_book_identity(book_mention.title)
    eligible_candidates = [
        candidate for candidate in candidates if _is_author_eligible(candidate, book_mention)
    ]
    for candidate in eligible_candidates:
        if _candidate_has_exact_title(candidate, normalized_title):
            return candidate

    selected_candidate: _OpenLibraryCandidate | None = None
    selected_score = _FUZZY_TITLE_SCORE_THRESHOLD
    for candidate in eligible_candidates:
        candidate_score = max(
            fuzz.ratio(normalized_title, normalize_book_identity(candidate_title))
            for candidate_title in candidate.candidate_titles
        )
        if candidate_score > selected_score:
            selected_candidate = candidate
            selected_score = candidate_score
    return selected_candidate


def _candidate_has_exact_title(
    candidate: _OpenLibraryCandidate,
    normalized_title: str,
) -> bool:
    """Return whether a Candidate has an exact normalized Work title."""

    return any(
        normalize_book_identity(candidate_title) == normalized_title
        for candidate_title in candidate.candidate_titles
    )


def _is_author_eligible(
    candidate: _OpenLibraryCandidate,
    book_mention: BookMention,
) -> bool:
    """Return whether any complete Mention Author Credit matches provider names."""

    normalized_mention_author_names = {
        normalize_book_identity(author_credit.name) for author_credit in book_mention.authors
    }
    normalized_candidate_author_names = {
        normalize_book_identity(author_credit.name) for author_credit in candidate.authors
    }
    normalized_candidate_author_names.update(
        normalize_book_identity(author_alias) for author_alias in candidate.author_aliases
    )
    return bool(normalized_mention_author_names & normalized_candidate_author_names)


def _drop_duplicate_resolved_book_works(
    results: list[BookResult],
) -> list[BookResult]:
    """Drop later resolved results for an already returned canonical Work ID."""

    returned_work_ids: set[str] = set()
    deduplicated_results: list[BookResult] = []
    for result in results:
        if result.book is None:
            deduplicated_results.append(result)
            continue
        if result.book.open_library_work_id in returned_work_ids:
            continue
        returned_work_ids.add(result.book.open_library_work_id)
        deduplicated_results.append(result)
    return deduplicated_results
