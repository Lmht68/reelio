"""Resolve exact Book Work Mentions through Open Library Search."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
_WORK_ID_PATTERN = re.compile(r"(?:/works/)?(OL[0-9]+W)")
_AUTHOR_ID_PATTERN = re.compile(r"(?:/authors/)?(OL[0-9]+A)")


class _OpenLibraryModel(BaseModel):
    """Ignore unknown Open Library fields at the resolver boundary."""

    model_config = ConfigDict(extra="ignore")


class _OpenLibrarySearchCandidateModel(_OpenLibraryModel):
    """Model the Search fields needed for exact Book Work resolution."""

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    author_key: list[str] | None = None
    author_name: list[str] | None = None


class _OpenLibrarySearchResponseModel(_OpenLibraryModel):
    """Model an Open Library Search response with its required candidate list."""

    docs: list[_OpenLibrarySearchCandidateModel]


@dataclass(frozen=True, slots=True)
class _OpenLibraryCandidate:
    """Contain validated Work metadata used for exact matching."""

    title: str
    authors: list[EnrichedAuthorCredit]
    open_library_work_id: str


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
        return BookResults(books=list(results))

    async def aclose(self) -> None:
        """Close the resolver-owned Open Library HTTP client."""
        await self._client.aclose()

    async def _resolve_mention(self, book_mention: BookMention) -> BookResult:
        candidates = await self._search(book_mention.title)
        candidate = _select_exact_candidate(book_mention, candidates)
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

    async def _search(self, title: str) -> list[_OpenLibraryCandidate]:
        try:
            await self._await_request_turn()
            response = await self._client.get(
                "/search.json",
                params={
                    "title": title,
                    "fields": "key,title,author_key,author_name",
                    "limit": _SEARCH_LIMIT,
                },
            )
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

        try:
            response_model = _OpenLibrarySearchResponseModel.model_validate(
                cast(object, response.json())
            )
            return [_to_candidate(candidate) for candidate in response_model.docs]
        except (ValidationError, ValueError) as exc:
            logger.error(
                "Open Library catalog response validation failed",
                extra={"stage": _STAGE, "reason": "invalid_provider_response"},
            )
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc

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
    return _OpenLibraryCandidate(
        title=_require_provider_text(candidate_model.title, "title"),
        authors=authors,
        open_library_work_id=work_id,
    )


def _parse_provider_id(value: str, pattern: re.Pattern[str], kind: str) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"Open Library {kind} ID is invalid")
    return match.group(1)


def _require_provider_text(value: str, field_name: str) -> str:
    if not normalize_book_text(value):
        raise ValueError(f"Open Library {field_name} is blank")
    return value


def _select_exact_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    normalized_title = normalize_book_identity(book_mention.title)
    title_matches = [
        candidate
        for candidate in candidates
        if normalize_book_identity(candidate.title) == normalized_title
    ]
    if not book_mention.authors:
        return title_matches[0] if len(title_matches) == 1 else None

    normalized_author_names = {
        normalize_book_identity(author_credit.name) for author_credit in book_mention.authors
    }
    for candidate in title_matches:
        if any(
            normalize_book_identity(author_credit.name) in normalized_author_names
            for author_credit in candidate.authors
        ):
            return candidate
    return None
