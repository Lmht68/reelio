"""Resolve exact Book Work Mentions through Open Library Search."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from time import monotonic
from typing import Annotated, Literal, NoReturn, cast

import httpx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)
from rapidfuzz import fuzz

from reelio.cache import AsyncCache, CacheCodec, CacheCodecError, CacheEntry
from reelio.cache.interface import JsonObject
from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import OpenLibraryConfig
from reelio.extraction.types import (
    BookEdition,
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
_SEARCH_LIMIT = 3
_EMPTY_SEARCH_TTL_SECONDS = 900
_POSITIVE_SEARCH_TTL_SECONDS = 21_600
_DETAIL_TTL_SECONDS = 86_400

type _CatalogRequestKey = tuple[str, tuple[tuple[str, str], ...]]

_FUZZY_MAIN_TITLE_MIN_LENGTH = 6
_FUZZY_TITLE_SCORE_THRESHOLD = 80.0
_WORK_SEARCH_FIELDS = (
    "key,title,alternative_title,author_key,author_name,"
    "author_alternative_name,editions,editions.key,editions.title,cover_i,"
    "cover_edition_key"
)
_EDITION_SEARCH_FIELDS = (
    "key,editions,editions.key,editions.title,editions.format,"
    "editions.publish_year,editions.cover_i"
)
_WORK_ID_PATTERN = re.compile(r"(?:/works/)?(OL[0-9]+W)")
_AUTHOR_ID_PATTERN = re.compile(r"(?:/authors/)?(OL[0-9]+A)")
_EDITION_ID_PATTERN = re.compile(r"(?:/books/)?(OL[0-9]+M)")
_PUBLICATION_YEAR_PATTERN = re.compile(r"(?<!\d)\d{4}(?!\d)")
_AUDIOBOOK_FORMAT_PATTERN = re.compile(
    r"(?<!\w)(?:audio|audiobook|audio book|sound recording|cassette|mp3)(?!\w)"
)
_BOOK_SUBTITLE_SEPARATOR_PATTERN = re.compile(r":|\s+-\s+|\s+\N{EN DASH}\s+")


class _OpenLibraryModel(BaseModel):
    """Ignore unknown Open Library fields at the resolver boundary."""

    model_config = ConfigDict(extra="ignore", strict=True)


class _OpenLibraryCacheModel(BaseModel):
    """Forbid unrecognized or coercible values in an Open Library cache payload."""

    model_config = ConfigDict(extra="forbid", strict=True)


def _validate_cached_nonblank_text(value: str) -> str:
    """Reject whitespace-only cached values excluded by provider parsing."""
    if not value.strip():
        raise ValueError("Cached text must not be blank")
    return value


type _CachedNonBlankText = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_cached_nonblank_text),
]
type _CachedWorkId = Annotated[str, Field(pattern=r"^OL[0-9]+W$")]
type _CachedAuthorId = Annotated[str, Field(pattern=r"^OL[0-9]+A$")]
type _CachedEditionId = Annotated[str, Field(pattern=r"^OL[0-9]+M$")]
type _CachedCoverUrl = Annotated[
    str,
    Field(pattern=r"^https://covers\.openlibrary\.org/b/id/[1-9][0-9]*-L\.jpg$"),
]


class _CachedAuthorCredit(_OpenLibraryCacheModel):
    """Contain one normalized Open Library Author value for a cached Candidate."""

    open_library_author_id: _CachedAuthorId
    name: _CachedNonBlankText


class _CachedCandidate(_OpenLibraryCacheModel):
    """Contain one normalized Candidate value for a cached Work Search."""

    title: _CachedNonBlankText
    authors: list[_CachedAuthorCredit]
    open_library_work_id: _CachedWorkId
    candidate_titles: list[_CachedNonBlankText] = Field(min_length=1)
    author_aliases: list[_CachedNonBlankText]
    cover_url: _CachedCoverUrl | None
    cover_edition_id: _CachedEditionId | None


class _CachedSearchCandidates(_OpenLibraryCacheModel):
    """Contain the bounded normalized Candidate list for one Work Search."""

    candidates: list[_CachedCandidate]


class _SearchCandidatesCacheCodec:
    """Encode normalized Work Search results independent of provider response shape."""

    version = "work-search-v1"

    def encode(self, value: tuple[_OpenLibraryCandidate, ...]) -> JsonObject:
        """Encode normalized Candidates after provider validation."""
        return cast(
            JsonObject,
            {
                "candidates": [
                    {
                        "title": candidate.title,
                        "authors": [
                            {
                                "open_library_author_id": author.open_library_author_id,
                                "name": author.name,
                            }
                            for author in candidate.authors
                        ],
                        "open_library_work_id": candidate.open_library_work_id,
                        "candidate_titles": list(candidate.candidate_titles),
                        "author_aliases": list(candidate.author_aliases),
                        "cover_url": candidate.cover_url,
                        "cover_edition_id": candidate.cover_edition_id,
                    }
                    for candidate in value
                ]
            },
        )

    def decode(self, payload: JsonObject) -> tuple[_OpenLibraryCandidate, ...]:
        """Decode one strict normalized Candidate payload.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _CachedSearchCandidates.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached Open Library Work Search value") from exc
        return tuple(
            _OpenLibraryCandidate(
                title=candidate.title,
                authors=tuple(
                    EnrichedAuthorCredit(
                        open_library_author_id=author.open_library_author_id,
                        name=author.name,
                        open_library_url=(
                            f"https://openlibrary.org/authors/{author.open_library_author_id}"
                        ),
                    )
                    for author in candidate.authors
                ),
                open_library_work_id=candidate.open_library_work_id,
                candidate_titles=tuple(candidate.candidate_titles),
                author_aliases=tuple(candidate.author_aliases),
                cover_url=candidate.cover_url,
                cover_edition_id=candidate.cover_edition_id,
            )
            for candidate in cached_value.candidates
        )


class _CachedSelectedEdition(_OpenLibraryCacheModel):
    """Contain one normalized preferred Edition selection."""

    open_library_edition_id: _CachedEditionId
    title: _CachedNonBlankText | None
    formats: list[str]
    publication_year: int | None
    cover_id: int | None


class _CachedSelectedEditionSearch(_OpenLibraryCacheModel):
    """Contain nullable normalized preferred Edition Search output."""

    selected_edition: _CachedSelectedEdition | None


class _SelectedEditionSearchCacheCodec:
    """Encode normalized preferred Edition Search results."""

    version = "preferred-edition-search-v1"

    def encode(self, value: _OpenLibrarySelectedEdition | None) -> JsonObject:
        """Encode a normalized preferred Edition selection after provider validation."""
        if value is None:
            return {"selected_edition": None}
        return cast(
            JsonObject,
            {
                "selected_edition": {
                    "open_library_edition_id": value.open_library_edition_id,
                    "title": value.title,
                    "formats": list(value.formats),
                    "publication_year": value.publication_year,
                    "cover_id": value.cover_id,
                }
            },
        )

    def decode(self, payload: JsonObject) -> _OpenLibrarySelectedEdition | None:
        """Decode one strict nullable preferred Edition selection.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _CachedSelectedEditionSearch.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError(
                "Invalid cached Open Library preferred Edition Search value"
            ) from exc
        selected_edition = cached_value.selected_edition
        if selected_edition is None:
            return None
        return _OpenLibrarySelectedEdition(
            open_library_edition_id=selected_edition.open_library_edition_id,
            title=selected_edition.title,
            formats=tuple(selected_edition.formats),
            publication_year=selected_edition.publication_year,
            cover_id=selected_edition.cover_id,
        )


class _CachedTerminalWorkTarget(_OpenLibraryCacheModel):
    """Contain a cached terminal Work identity target."""

    kind: Literal["terminal"]
    work_id: _CachedWorkId


class _CachedRedirectWorkTarget(_OpenLibraryCacheModel):
    """Contain a cached Work redirect target."""

    kind: Literal["redirect"]
    work_id: _CachedWorkId


type _CachedWorkTarget = Annotated[
    _CachedTerminalWorkTarget | _CachedRedirectWorkTarget,
    Field(discriminator="kind"),
]
_CACHED_WORK_TARGET_ADAPTER: TypeAdapter[_CachedWorkTarget] = TypeAdapter(_CachedWorkTarget)


class _WorkTargetCacheCodec:
    """Encode normalized terminal Work and redirect operations."""

    version = "work-detail-v1"

    def encode(self, value: _OpenLibraryWorkTarget) -> JsonObject:
        """Encode a normalized Work target after provider validation."""
        if isinstance(value, _TerminalWorkTarget):
            return {"kind": "terminal", "work_id": value.work_id}
        return {"kind": "redirect", "work_id": value.work_id}

    def decode(self, payload: JsonObject) -> _OpenLibraryWorkTarget:
        """Decode one discriminated strict Work target.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _CACHED_WORK_TARGET_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached Open Library Work detail value") from exc
        if isinstance(cached_value, _CachedTerminalWorkTarget):
            return _TerminalWorkTarget(cached_value.work_id)
        return _RedirectWorkTarget(cached_value.work_id)


class _CachedEditionRecord(_OpenLibraryCacheModel):
    """Contain one normalized Open Library Edition detail record."""

    open_library_edition_id: _CachedEditionId
    title: str | None
    publishers: list[str]
    isbn_10: list[str]
    isbn_13: list[str]
    publish_date: str | None
    physical_format: str | None
    covers: list[int]


class _EditionRecordCacheCodec:
    """Encode normalized Open Library Edition detail records."""

    version = "edition-detail-v1"

    def encode(self, value: _OpenLibraryEditionRecord) -> JsonObject:
        """Encode a normalized Edition record after provider validation."""
        return cast(
            JsonObject,
            {
                "open_library_edition_id": value.open_library_edition_id,
                "title": value.title,
                "publishers": list(value.publishers),
                "isbn_10": list(value.isbn_10),
                "isbn_13": list(value.isbn_13),
                "publish_date": value.publish_date,
                "physical_format": value.physical_format,
                "covers": list(value.covers),
            },
        )

    def decode(self, payload: JsonObject) -> _OpenLibraryEditionRecord:
        """Decode one strict normalized Edition detail record.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        try:
            cached_value = _CachedEditionRecord.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid cached Open Library Edition detail value") from exc
        return _OpenLibraryEditionRecord(
            open_library_edition_id=cached_value.open_library_edition_id,
            title=cached_value.title,
            publishers=tuple(cached_value.publishers),
            isbn_10=tuple(cached_value.isbn_10),
            isbn_13=tuple(cached_value.isbn_13),
            publish_date=cached_value.publish_date,
            physical_format=cached_value.physical_format,
            covers=tuple(cached_value.covers),
        )


_SEARCH_CANDIDATES_CACHE_CODEC = _SearchCandidatesCacheCodec()
_SELECTED_EDITION_SEARCH_CACHE_CODEC = _SelectedEditionSearchCacheCodec()
_WORK_TARGET_CACHE_CODEC = _WorkTargetCacheCodec()
_EDITION_RECORD_CACHE_CODEC = _EditionRecordCacheCodec()


class _OpenLibrarySearchEditionModel(_OpenLibraryModel):
    """Model nested Open Library Search Edition metadata."""

    key: str | None = None
    title: str | None = None
    format: list[str] | None = None
    publish_year: list[int] | None = None
    cover_i: int | None = None


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
    cover_i: int | None = None
    cover_edition_key: str | None = None


class _OpenLibraryPreferredEditionWorkModel(_OpenLibraryModel):
    """Model one Work result used only for preferred Edition selection."""

    key: str = Field(min_length=1)
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


class _OpenLibraryPreferredEditionSearchResponseModel(_OpenLibraryModel):
    """Model a Search response for one relevance-selected Work Edition."""

    docs: list[_OpenLibraryPreferredEditionWorkModel]


class _OpenLibraryEditionRecordModel(_OpenLibraryModel):
    """Model one direct Open Library Edition record."""

    key: str = Field(min_length=1)
    title: str | None = None
    publishers: list[str] | None = None
    isbn_10: list[str] | None = None
    isbn_13: list[str] | None = None
    publish_date: str | None = None
    physical_format: str | None = None
    covers: list[int] | None = None


@dataclass(frozen=True, slots=True)
class _TerminalWorkTarget:
    """Contain the canonical target of a terminal Open Library Work record."""

    work_id: str


@dataclass(frozen=True, slots=True)
class _RedirectWorkTarget:
    """Contain the canonical target of an Open Library Work redirect."""

    work_id: str


type _OpenLibraryWorkTarget = _TerminalWorkTarget | _RedirectWorkTarget


@dataclass(frozen=True, slots=True)
class _OpenLibraryCandidate:
    """Contain bounded Work metadata used for Book Work matching."""

    title: str
    authors: tuple[EnrichedAuthorCredit, ...]
    open_library_work_id: str
    candidate_titles: tuple[str, ...]
    author_aliases: tuple[str, ...]
    cover_url: str | None
    cover_edition_id: str | None


@dataclass(frozen=True, slots=True)
class _BookTitleParts:
    """Contain the normalized main title and optional subtitle of a Book title."""

    main_title: str
    subtitle: str | None


@dataclass(frozen=True, slots=True)
class _OpenLibrarySelectedEdition:
    """Contain Search metadata for one relevance-selected Open Library Edition."""

    open_library_edition_id: str
    title: str | None
    formats: tuple[str, ...]
    publication_year: int | None
    cover_id: int | None


@dataclass(frozen=True, slots=True)
class _OpenLibraryEditionRecord:
    """Contain immutable provider metadata from one Open Library Edition record."""

    open_library_edition_id: str
    title: str | None
    publishers: tuple[str, ...]
    isbn_10: tuple[str, ...]
    isbn_13: tuple[str, ...]
    publish_date: str | None
    physical_format: str | None
    covers: tuple[int, ...]


class OpenLibraryBookResolver:
    """Resolve Book Work Mentions with complete-title and subtitle-aware matches."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: OpenLibraryConfig,
        cache: AsyncCache,
        *,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize a resolver with injectable HTTP, cache, and timing boundaries.

        Args:
            client: Reusable Open Library HTTP client owned by this resolver.
            settings: Validated Open Library lookup timeout and dispatch rate.
            cache: Shared cache borrowed from the application lifespan.
            clock: Monotonic clock used to schedule requests and lookup deadlines.
            sleep: Awaitable delay used to enforce dispatch and retry waits.
        """
        self._client = client
        self._request_timeout_seconds = settings.request_timeout_seconds
        self._request_interval_seconds = 1 / settings.requests_per_second
        self._cache = cache
        self._clock = clock
        self._sleep = sleep
        self._lookup_lock = asyncio.Lock()
        self._in_flight_lookups: dict[_CatalogRequestKey, asyncio.Task[object]] = {}
        self._closed = False
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
        """Close the resolver-owned Open Library HTTP client and pending lookups."""
        async with self._lookup_lock:
            if self._closed:
                return
            self._closed = True
            pending_lookups = tuple(self._in_flight_lookups.values())

        for pending_lookup in pending_lookups:
            pending_lookup.cancel()
        if pending_lookups:
            await asyncio.gather(*pending_lookups, return_exceptions=True)

        async with self._lookup_lock:
            self._in_flight_lookups.clear()
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
            candidate = _select_authorless_candidate(book_mention, candidates)

        if candidate is None:
            return BookResult(
                status=ResultStatus.UNRESOLVED,
                book_mention=book_mention,
                book=None,
            )
        edition = await self._select_preferred_edition(candidate.open_library_work_id)
        cover_url: str | None
        cover_edition_id: str | None
        if edition is not None and edition.cover_url is not None:
            cover_url = edition.cover_url
            cover_edition_id = edition.open_library_edition_id
        else:
            cover_url = candidate.cover_url
            cover_edition_id = candidate.cover_edition_id

        return BookResult(
            status=ResultStatus.RESOLVED,
            book_mention=book_mention,
            book=EnrichedBookWork(
                title=candidate.title,
                authors=list(candidate.authors),
                open_library_work_id=candidate.open_library_work_id,
                open_library_url=(
                    f"https://openlibrary.org/works/{candidate.open_library_work_id}"
                ),
                edition=edition,
                cover_url=cover_url,
                cover_edition_id=cover_edition_id,
            ),
        )

    async def _search(
        self,
        title: str,
        author: str | None = None,
    ) -> list[_OpenLibraryCandidate]:
        params: dict[str, str | int] = {
            "title": title,
            "fields": _WORK_SEARCH_FIELDS,
            "limit": _SEARCH_LIMIT,
        }
        if author is not None:
            params["author"] = author
        candidates = await self._get_catalog_value(
            "/search.json",
            _to_search_candidates,
            _catalog_cache_entry(
                "/search.json",
                params,
                _SEARCH_CANDIDATES_CACHE_CODEC,
                _search_candidates_ttl_seconds,
            ),
            params,
        )
        return list(candidates)

    async def _select_preferred_edition(self, work_id: str) -> BookEdition | None:
        english_selected_edition = await self._search_preferred_edition(
            work_id,
            require_english=True,
        )
        if english_selected_edition is not None and not _is_explicit_audiobook(
            english_selected_edition.formats
        ):
            english_edition = await self._load_selected_edition(english_selected_edition)
            if english_edition is not None:
                return english_edition
            rejected_edition_id = english_selected_edition.open_library_edition_id
        elif english_selected_edition is not None:
            rejected_edition_id = english_selected_edition.open_library_edition_id
        else:
            rejected_edition_id = None

        fallback_selected_edition = await self._search_preferred_edition(
            work_id,
            require_english=False,
        )
        if (
            fallback_selected_edition is None
            or fallback_selected_edition.open_library_edition_id == rejected_edition_id
            or _is_explicit_audiobook(fallback_selected_edition.formats)
        ):
            return None
        return await self._load_selected_edition(fallback_selected_edition)

    async def _search_preferred_edition(
        self,
        work_id: str,
        *,
        require_english: bool,
    ) -> _OpenLibrarySelectedEdition | None:
        query = f"key:/works/{work_id}"
        if require_english:
            query = f"{query} AND language:eng"
        params: dict[str, str | int] = {
            "q": query,
            "fields": _EDITION_SEARCH_FIELDS,
            "limit": 1,
        }
        return await self._get_catalog_value(
            "/search.json",
            lambda value: _to_selected_edition(value, work_id),
            _catalog_cache_entry(
                "/search.json",
                params,
                _SELECTED_EDITION_SEARCH_CACHE_CODEC,
                _selected_edition_ttl_seconds,
            ),
            params,
        )

    async def _load_selected_edition(
        self,
        selected_edition: _OpenLibrarySelectedEdition,
    ) -> BookEdition | None:
        path = f"/books/{selected_edition.open_library_edition_id}.json"
        record = await self._get_catalog_value(
            path,
            lambda value: _to_edition_record(
                value,
                selected_edition.open_library_edition_id,
            ),
            _catalog_cache_entry(
                path,
                None,
                _EDITION_RECORD_CACHE_CODEC,
                _detail_ttl_seconds,
            ),
        )
        formats = (*selected_edition.formats,)
        if record.physical_format is not None:
            formats += (record.physical_format,)
        if _is_explicit_audiobook(formats):
            return None
        cover_id = next(
            (cover_id for cover_id in record.covers if cover_id > 0),
            selected_edition.cover_id,
        )
        return BookEdition(
            title=(
                _optional_provider_text(record.title)
                or _optional_provider_text(selected_edition.title)
            ),
            publication_year=_publication_year(
                selected_edition.publication_year,
                record.publish_date,
            ),
            publishers=list(record.publishers),
            isbn_10=list(record.isbn_10),
            isbn_13=list(record.isbn_13),
            open_library_edition_id=record.open_library_edition_id,
            open_library_url=(f"https://openlibrary.org/books/{record.open_library_edition_id}"),
            cover_url=_cover_url(cover_id),
        )

    async def _get_catalog_value[ValueT](
        self,
        path: str,
        parser: Callable[[object], ValueT],
        cache_entry: CacheEntry[ValueT],
        params: dict[str, str | int] | None = None,
    ) -> ValueT:
        request_key = _catalog_request_key(path, params)
        async with self._lookup_lock:
            if self._closed:
                raise RuntimeError("Open Library resolver is closed.")
            lookup = self._in_flight_lookups.get(request_key)
            if lookup is None:
                lookup = cast(
                    asyncio.Task[object],
                    asyncio.create_task(
                        self._load_catalog_value(
                            request_key,
                            path,
                            parser,
                            cache_entry,
                            params,
                        )
                    ),
                )
                lookup.add_done_callback(_retrieve_task_exception)
                self._in_flight_lookups[request_key] = lookup
        return cast(ValueT, await asyncio.shield(lookup))

    async def _load_catalog_value[ValueT](
        self,
        request_key: _CatalogRequestKey,
        path: str,
        parser: Callable[[object], ValueT],
        cache_entry: CacheEntry[ValueT],
        params: dict[str, str | int] | None,
    ) -> ValueT:
        async def load_provider_value() -> ValueT:
            """Load and parse a provider value without cache ownership."""
            return await self._load_provider_value(path, parser, params)

        try:
            return await self._cache.get_or_load(cache_entry, load_provider_value)
        finally:
            async with self._lookup_lock:
                self._in_flight_lookups.pop(request_key, None)

    async def _load_provider_value[ValueT](
        self,
        path: str,
        parser: Callable[[object], ValueT],
        params: dict[str, str | int] | None,
    ) -> ValueT:
        try:
            deadline = self._clock() + self._request_timeout_seconds
            async with asyncio.timeout(self._request_timeout_seconds):
                response = await self._get_catalog_response(path, params, deadline)
                return parser(cast(object, response.json()))
        except (TimeoutError, httpx.TimeoutException) as exc:
            logger.error(
                "Open Library catalog request timed out",
                extra={"stage": _STAGE, "reason": "provider_timeout"},
            )
            raise PipelineTimeoutError(_TIMEOUT_ERROR_MESSAGE) from exc
        except (ValidationError, ValueError) as exc:
            logger.error(
                "Open Library catalog response validation failed",
                extra={"stage": _STAGE, "reason": "invalid_provider_response"},
            )
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exc

    async def _get_catalog_response(
        self,
        path: str,
        params: dict[str, str | int] | None,
        deadline: float,
    ) -> httpx.Response:
        for attempt in range(2):
            await self._await_request_turn(deadline)
            try:
                response = await self._client.get(path, params=params)
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as exc:
                if attempt == 1:
                    self._raise_catalog_failure("retry_exhausted", exc)
                continue

            if response.is_success:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 1:
                    self._raise_catalog_failure("retry_exhausted")
                retry_after_header = response.headers.get("Retry-After")
                if response.status_code == 429 and retry_after_header is None:
                    self._raise_catalog_failure("http_failure")
                if retry_after_header is not None:
                    try:
                        retry_after = _retry_after_seconds(response)
                    except ValueError as exc:
                        self._raise_catalog_failure("http_failure", exc)
                    if retry_after > deadline - self._clock():
                        self._raise_catalog_failure("retry_after_exceeds_timeout")
                    await self._sleep(retry_after)
                continue
            self._raise_catalog_failure("http_failure")
        raise AssertionError("Open Library request attempts were not exhausted.")

    async def _canonicalize_candidates(
        self,
        candidates: Sequence[_OpenLibraryCandidate],
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
            path = f"/works/{current_work_id}.json"
            target = await self._get_catalog_value(
                path,
                _to_work_target,
                _catalog_cache_entry(
                    path,
                    None,
                    _WORK_TARGET_CACHE_CODEC,
                    _detail_ttl_seconds,
                ),
            )
            if isinstance(target, _TerminalWorkTarget):
                return target.work_id
            current_work_id = target.work_id
        logger.error(
            "Open Library catalog response validation failed",
            extra={"stage": _STAGE, "reason": "invalid_provider_response"},
        )
        raise CatalogProviderError(_CATALOG_ERROR_MESSAGE)

    async def _await_request_turn(self, deadline: float) -> None:
        async with self._request_lock:
            current_time = self._clock()
            if current_time >= deadline:
                raise TimeoutError
            delay_seconds = self._next_request_at - current_time
            if delay_seconds > 0:
                if delay_seconds > deadline - current_time:
                    raise TimeoutError
                await self._sleep(delay_seconds)
            current_time = self._clock()
            if current_time >= deadline:
                raise TimeoutError
            self._next_request_at = current_time + self._request_interval_seconds

    def _raise_catalog_failure(
        self,
        reason: str,
        exception: Exception | None = None,
    ) -> NoReturn:
        logger.error(
            "Open Library catalog request failed",
            extra={"stage": _STAGE, "reason": reason},
        )
        if exception is None:
            raise CatalogProviderError(_CATALOG_ERROR_MESSAGE)
        raise CatalogProviderError(_CATALOG_ERROR_MESSAGE) from exception


def create_open_library_book_resolver(
    settings: OpenLibraryConfig,
    cache: AsyncCache,
) -> OpenLibraryBookResolver:
    """Create a reusable Open Library Book Work resolver.

    Args:
        settings: Validated Open Library contact, endpoint, timeout, and request rate.
        cache: Shared cache borrowed from the application lifespan.

    Returns:
        OpenLibraryBookResolver: Resolver owning one asynchronous HTTP client.
    """
    client = httpx.AsyncClient(
        base_url=f"{settings.base_url}/",
        headers={"User-Agent": f"Reelio ({settings.contact_email})"},
        timeout=settings.request_timeout_seconds,
    )
    return OpenLibraryBookResolver(client, settings, cache)


def _catalog_request_key(
    path: str,
    params: dict[str, str | int] | None,
) -> _CatalogRequestKey:
    return (
        path,
        tuple(sorted((key, str(value)) for key, value in (params or {}).items())),
    )


def _catalog_cache_entry[ValueT](
    path: str,
    params: dict[str, str | int] | None,
    codec: CacheCodec[ValueT],
    ttl_seconds: Callable[[ValueT], int],
) -> CacheEntry[ValueT]:
    """Describe one normalized Open Library operation for shared caching."""
    identity_parameters: JsonObject = {
        name: str(value) for name, value in sorted((params or {}).items())
    }
    return CacheEntry(
        layer="provider:open-library",
        key_version="v1",
        identity={"path": path, "params": identity_parameters},
        codec=codec,
        ttl_seconds=ttl_seconds,
    )


def _search_candidates_ttl_seconds(value: tuple[_OpenLibraryCandidate, ...]) -> int:
    """Return the positive or negative Work Search freshness contract."""
    return _POSITIVE_SEARCH_TTL_SECONDS if value else _EMPTY_SEARCH_TTL_SECONDS


def _selected_edition_ttl_seconds(value: _OpenLibrarySelectedEdition | None) -> int:
    """Return the positive or negative preferred Edition Search freshness contract."""
    return _POSITIVE_SEARCH_TTL_SECONDS if value is not None else _EMPTY_SEARCH_TTL_SECONDS


def _detail_ttl_seconds(value: object) -> int:
    """Return the fixed Work and Edition detail freshness contract."""
    del value
    return _DETAIL_TTL_SECONDS


def _retrieve_task_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _retry_after_seconds(response: httpx.Response) -> float:
    retry_after_header = response.headers["Retry-After"]
    try:
        retry_after = float(retry_after_header)
    except ValueError as exc:
        raise ValueError("Open Library Retry-After is invalid") from exc
    if not math.isfinite(retry_after) or retry_after < 0:
        raise ValueError("Open Library Retry-After is invalid")
    return retry_after


def _to_search_candidates(value: object) -> tuple[_OpenLibraryCandidate, ...]:
    response_model = _OpenLibrarySearchResponseModel.model_validate(value)
    return tuple(_to_candidate(candidate) for candidate in response_model.docs[:_SEARCH_LIMIT])


def _to_selected_edition(
    value: object,
    work_id: str,
) -> _OpenLibrarySelectedEdition | None:
    response_model = _OpenLibraryPreferredEditionSearchResponseModel.model_validate(value)
    if not response_model.docs:
        return None
    selected_work = response_model.docs[0]
    selected_work_id = _parse_provider_id(
        selected_work.key,
        _WORK_ID_PATTERN,
        "Work",
    )
    if selected_work_id != work_id:
        raise ValueError("Open Library preferred Edition Work ID does not match")
    if selected_work.editions is None or not selected_work.editions.docs:
        return None
    selected_edition = selected_work.editions.docs[0]
    if selected_edition.key is None:
        raise ValueError("Open Library Edition ID is missing")
    return _OpenLibrarySelectedEdition(
        open_library_edition_id=_parse_provider_id(
            selected_edition.key,
            _EDITION_ID_PATTERN,
            "Edition",
        ),
        title=_optional_provider_text(selected_edition.title),
        formats=tuple(selected_edition.format or []),
        publication_year=(
            selected_edition.publish_year[0] if selected_edition.publish_year else None
        ),
        cover_id=selected_edition.cover_i,
    )


def _to_edition_record(value: object, expected_edition_id: str) -> _OpenLibraryEditionRecord:
    record = _OpenLibraryEditionRecordModel.model_validate(value)
    record_edition_id = _parse_provider_id(
        record.key,
        _EDITION_ID_PATTERN,
        "Edition",
    )
    if record_edition_id != expected_edition_id:
        raise ValueError("Open Library Edition record ID does not match")
    return _OpenLibraryEditionRecord(
        open_library_edition_id=record_edition_id,
        title=record.title,
        publishers=tuple(record.publishers or []),
        isbn_10=tuple(record.isbn_10 or []),
        isbn_13=tuple(record.isbn_13 or []),
        publish_date=record.publish_date,
        physical_format=record.physical_format,
        covers=tuple(record.covers or []),
    )


def _to_work_target(value: object) -> _OpenLibraryWorkTarget:
    record = _OpenLibraryWorkRecordModel.model_validate(value)
    if record.type.key == "/type/work":
        if record.key is None:
            raise ValueError("Open Library Work record is missing its key")
        return _TerminalWorkTarget(_parse_provider_id(record.key, _WORK_ID_PATTERN, "Work"))
    if record.type.key == "/type/redirect":
        if record.location is None:
            raise ValueError("Open Library Work redirect is missing its location")
        return _RedirectWorkTarget(_parse_provider_id(record.location, _WORK_ID_PATTERN, "Work"))
    raise ValueError("Open Library Work record has an invalid type")


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
    if candidate_model.editions is not None:
        for edition in candidate_model.editions.docs:
            edition_title = _optional_provider_text(edition.title)
            if edition_title is not None:
                selected_edition_title = (edition_title,)
                break
    author_aliases = tuple(
        _require_provider_text(author_alias, "author alternative name")
        for author_alias in candidate_model.author_alternative_name or []
    )
    cover_url = _cover_url(candidate_model.cover_i)
    cover_edition_id = (
        _parse_provider_id(
            candidate_model.cover_edition_key,
            _EDITION_ID_PATTERN,
            "Edition",
        )
        if cover_url is not None and candidate_model.cover_edition_key is not None
        else None
    )
    return _OpenLibraryCandidate(
        title=primary_title,
        authors=tuple(authors),
        open_library_work_id=work_id,
        candidate_titles=_unique_candidate_titles(
            primary_title,
            *alternative_titles,
            *selected_edition_title,
        ),
        author_aliases=author_aliases,
        cover_url=cover_url,
        cover_edition_id=cover_edition_id,
    )


def _cover_url(cover_id: int | None) -> str | None:
    """Return the large HTTPS Open Library cover URL for a positive cover ID."""

    if cover_id is None or cover_id <= 0:
        return None
    return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"


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


def _optional_provider_text(value: str | None) -> str | None:
    """Return provider text unchanged when it contains non-whitespace content."""

    if value is None or not normalize_book_text(value):
        return None
    return value


def _is_explicit_audiobook(formats: Iterable[str]) -> bool:
    """Return whether provider format metadata explicitly identifies an audiobook."""

    return any(
        _AUDIOBOOK_FORMAT_PATTERN.search(" ".join(format_value.split()).casefold()) is not None
        for format_value in formats
    )


def _publication_year(numeric_year: int | None, publication_date: str | None) -> int | None:
    """Return a non-future provider publication year when it is unambiguous."""

    current_year = date.today().year
    if numeric_year is not None:
        return numeric_year if numeric_year <= current_year else None
    if publication_date is None:
        return None
    years = _PUBLICATION_YEAR_PATTERN.findall(publication_date)
    if len(years) != 1:
        return None
    publication_year = int(years[0])
    return publication_year if publication_year <= current_year else None


def _select_authorless_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    """Return a uniquely verified Candidate for an authorless Book Mention."""

    normalized_title = normalize_book_identity(book_mention.title)
    exact_candidates = [
        candidate
        for candidate in candidates
        if _candidate_has_exact_title(candidate, normalized_title)
    ]
    if len(exact_candidates) == 1:
        return exact_candidates[0]
    return _select_subtitle_equivalent_candidate(book_mention, candidates)


def _select_subtitle_equivalent_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    """Return a unique Candidate with exact or bounded fuzzy main-title equivalence."""

    mention_title_parts = _book_title_parts(book_mention.title)
    normalized_mention_main_title = normalize_book_identity(mention_title_parts.main_title)
    mention_has_subtitle = mention_title_parts.subtitle is not None
    exact_main_title_candidates = [
        candidate
        for candidate in candidates
        if _candidate_has_exact_main_title_equivalence(
            candidate,
            normalized_mention_main_title,
            mention_has_subtitle,
        )
    ]
    if len(exact_main_title_candidates) == 1:
        return exact_main_title_candidates[0]
    if exact_main_title_candidates:
        return None

    fuzzy_main_title_candidates = [
        candidate
        for candidate in candidates
        if _candidate_has_fuzzy_main_title_equivalence(
            candidate,
            normalized_mention_main_title,
            mention_has_subtitle,
        )
    ]
    return fuzzy_main_title_candidates[0] if len(fuzzy_main_title_candidates) == 1 else None


def _select_authorful_candidate(
    book_mention: BookMention,
    candidates: list[_OpenLibraryCandidate],
) -> _OpenLibraryCandidate | None:
    """Return a complete-title or subtitle-aware match for an authorful Mention."""

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
    if selected_candidate is not None:
        return selected_candidate
    return _select_subtitle_equivalent_candidate(book_mention, eligible_candidates)


def _candidate_has_exact_title(
    candidate: _OpenLibraryCandidate,
    normalized_title: str,
) -> bool:
    """Return whether a Candidate has an exact normalized complete Work title."""

    return any(
        normalize_book_identity(candidate_title) == normalized_title
        for candidate_title in candidate.candidate_titles
    )


def _book_title_parts(title: str) -> _BookTitleParts:
    """Split a Book title at its leftmost recognized nonblank subtitle boundary."""

    for separator_match in _BOOK_SUBTITLE_SEPARATOR_PATTERN.finditer(title):
        main_title = normalize_book_text(title[: separator_match.start()])
        subtitle = normalize_book_text(title[separator_match.end() :])
        if main_title and subtitle:
            return _BookTitleParts(main_title=main_title, subtitle=subtitle)
    return _BookTitleParts(main_title=normalize_book_text(title), subtitle=None)


def _candidate_has_exact_main_title_equivalence(
    candidate: _OpenLibraryCandidate,
    normalized_mention_main_title: str,
    mention_has_subtitle: bool,
) -> bool:
    """Return whether a Candidate title has exact Book main-title equivalence."""

    return any(
        (candidate_title_parts.subtitle is not None) != mention_has_subtitle
        and normalize_book_identity(candidate_title_parts.main_title)
        == normalized_mention_main_title
        for candidate_title in candidate.candidate_titles
        for candidate_title_parts in (_book_title_parts(candidate_title),)
    )


def _candidate_has_fuzzy_main_title_equivalence(
    candidate: _OpenLibraryCandidate,
    normalized_mention_main_title: str,
    mention_has_subtitle: bool,
) -> bool:
    """Return whether a Candidate title has bounded fuzzy main-title equivalence."""

    if len(normalized_mention_main_title) < _FUZZY_MAIN_TITLE_MIN_LENGTH:
        return False
    return any(
        fuzz.ratio(normalized_mention_main_title, normalized_candidate_main_title)
        > _FUZZY_TITLE_SCORE_THRESHOLD
        for candidate_title in candidate.candidate_titles
        for candidate_title_parts in (_book_title_parts(candidate_title),)
        if (candidate_title_parts.subtitle is not None) != mention_has_subtitle
        for normalized_candidate_main_title in (
            normalize_book_identity(candidate_title_parts.main_title),
        )
        if len(normalized_candidate_main_title) >= _FUZZY_MAIN_TITLE_MIN_LENGTH
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
