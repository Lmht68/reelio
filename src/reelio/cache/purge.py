"""Fail-closed administrative Redis cache purges by provider namespace."""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from reelio.cache.config import CacheConfig
from reelio.config import Environment

logger = logging.getLogger(__name__)

_NAMESPACE_PATTERN = re.compile(r"[a-z0-9:-]+")
_DEFAULT_SCAN_COUNT = 100
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 1.0


class PurgeProvider(StrEnum):
    """Identify the cache provider scope selected by an operator."""

    SPOTIFY = "spotify"
    TMDB = "tmdb"
    OPEN_LIBRARY = "open-library"
    SOURCE = "source"


class PurgeNamespaceCategory(StrEnum):
    """Identify one stable visible cache layer eligible for purge."""

    EXACT_OPERATION = "exact-operation"
    SUBMITTED_URL_ALIAS = "submitted-url-alias"
    SOURCE_METADATA = "source-metadata"
    TRANSCRIPT = "transcript"
    MENTION_INTERPRETATION = "mention-interpretation"


@dataclass(frozen=True, slots=True)
class NamespacePurgeResult:
    """Report one completed visible cache-layer purge."""

    namespace_category: PurgeNamespaceCategory
    deleted_count: int
    duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderPurgeResult:
    """Report all completed cache-layer purges for one provider scope."""

    provider: PurgeProvider
    environment: Environment
    namespaces: tuple[NamespacePurgeResult, ...]


class ProviderPurgeError(RuntimeError):
    """Report a safe, partial provider purge failure without Redis details."""

    def __init__(
        self,
        provider: PurgeProvider,
        environment: Environment,
        namespace_category: PurgeNamespaceCategory,
        deleted_count: int,
    ) -> None:
        """Initialize a failure with only operator-safe purge context.

        Args:
            provider: Explicit provider scope requested by the operator.
            environment: Explicit deployment environment selected by the operator.
            namespace_category: Visible layer whose purge could not complete.
            deleted_count: Matching keys unlinked before the failure.
        """
        super().__init__("Provider cache purge did not complete.")
        self.provider = provider
        self.environment = environment
        self.namespace_category = namespace_category
        self.deleted_count = deleted_count


class _PurgeRedisClient(Protocol):
    """Describe the narrow Redis command surface needed for fail-closed purges."""

    def scan(
        self,
        cursor: int,
        match: str | None = None,
        count: int | None = None,
    ) -> Awaitable[tuple[int | bytes, Sequence[bytes | str]]]:
        """Scan one cursor page for keys matching a Redis glob."""
        ...

    def unlink(self, *names: bytes | str) -> Awaitable[int]:
        """Asynchronously delete the supplied keys."""
        ...

    def aclose(self) -> Awaitable[None]:
        """Close the owned Redis client."""
        ...


_PROVIDER_CATEGORIES: dict[PurgeProvider, tuple[PurgeNamespaceCategory, ...]] = {
    PurgeProvider.SPOTIFY: (PurgeNamespaceCategory.EXACT_OPERATION,),
    PurgeProvider.TMDB: (PurgeNamespaceCategory.EXACT_OPERATION,),
    PurgeProvider.OPEN_LIBRARY: (PurgeNamespaceCategory.EXACT_OPERATION,),
    PurgeProvider.SOURCE: (
        PurgeNamespaceCategory.SUBMITTED_URL_ALIAS,
        PurgeNamespaceCategory.SOURCE_METADATA,
        PurgeNamespaceCategory.TRANSCRIPT,
        PurgeNamespaceCategory.MENTION_INTERPRETATION,
    ),
}

_STATIC_CATEGORY_LAYERS: dict[PurgeNamespaceCategory, str] = {
    PurgeNamespaceCategory.SUBMITTED_URL_ALIAS: "source:alias",
    PurgeNamespaceCategory.SOURCE_METADATA: "source:metadata",
    PurgeNamespaceCategory.TRANSCRIPT: "source:transcript",
    PurgeNamespaceCategory.MENTION_INTERPRETATION: "source:interpretation",
}


class RedisProviderPurger:
    """Incrementally unlink one provider's visible cache layers from Redis."""

    def __init__(
        self,
        client: _PurgeRedisClient,
        namespace: str,
        environment: Environment,
        *,
        scan_count: int = _DEFAULT_SCAN_COUNT,
        command_timeout_seconds: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize an owner of one Redis client and safe environment namespace.

        Args:
            client: Redis-compatible client to scan, unlink, and close.
            namespace: Validated visible namespace derived from the environment.
            environment: Deployment environment represented by the namespace.
            scan_count: Maximum keys Redis should inspect per scan page.
            command_timeout_seconds: Maximum duration for each Redis command.
            monotonic: Monotonic clock used to measure category duration.

        Raises:
            ValueError: If namespace syntax or command bounds are unsafe.
        """
        if not _NAMESPACE_PATTERN.fullmatch(namespace):
            raise ValueError(
                "Purge namespace must contain only lowercase letters, digits, hyphens, and colons"
            )
        if scan_count <= 0:
            raise ValueError("Purge scan count must be positive")
        if command_timeout_seconds <= 0:
            raise ValueError("Purge command timeout must be positive")
        self._client = client
        self._namespace = namespace
        self._environment = environment
        self._scan_count = scan_count
        self._command_timeout_seconds = command_timeout_seconds
        self._monotonic = monotonic
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def purge(self, provider: PurgeProvider) -> ProviderPurgeResult:
        """Purge all cache layers selected by one explicit provider scope.

        Args:
            provider: Catalog provider or all-Source scope selected by the operator.

        Returns:
            Successful purge counts and durations for every selected visible layer.

        Raises:
            ProviderPurgeError: If Redis scanning or unlinking cannot complete.
            RuntimeError: If the purger has already been closed.
        """
        if self._closed:
            raise RuntimeError("Provider purger is closed")
        namespace_results: list[NamespacePurgeResult] = []
        for namespace_category in _PROVIDER_CATEGORIES[provider]:
            namespace_results.append(await self._purge_category(provider, namespace_category))
        return ProviderPurgeResult(provider, self._environment, tuple(namespace_results))

    async def aclose(self) -> None:
        """Close the owned Redis client at most once."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._client.aclose()

    async def _purge_category(
        self,
        provider: PurgeProvider,
        namespace_category: PurgeNamespaceCategory,
    ) -> NamespacePurgeResult:
        """Scan and unlink every version and lease key in one visible layer."""
        started_at = self._monotonic()
        deleted_count = 0
        cursor = 0
        match_pattern = f"{self._namespace}:{_layer_for(provider, namespace_category)}:*"
        try:
            while True:
                next_cursor, keys = await self._scan_page(cursor, match_pattern)
                cursor = int(next_cursor)
                if keys:
                    deleted_count += await self._unlink_keys(keys)
                if cursor == 0:
                    break
        except (TimeoutError, RedisError) as error:
            duration_ms = _duration_ms(started_at, self._monotonic)
            _emit_purge_event(
                provider,
                self._environment,
                namespace_category,
                duration_ms,
                "failed",
                deleted_count,
            )
            raise ProviderPurgeError(
                provider,
                self._environment,
                namespace_category,
                deleted_count,
            ) from error
        duration_ms = _duration_ms(started_at, self._monotonic)
        _emit_purge_event(
            provider,
            self._environment,
            namespace_category,
            duration_ms,
            "completed",
            deleted_count,
        )
        return NamespacePurgeResult(namespace_category, deleted_count, duration_ms)

    async def _scan_page(
        self,
        cursor: int,
        match_pattern: str,
    ) -> tuple[int | bytes, Sequence[bytes | str]]:
        """Run one bounded Redis SCAN command."""
        async with asyncio.timeout(self._command_timeout_seconds):
            return await self._client.scan(
                cursor=cursor,
                match=match_pattern,
                count=self._scan_count,
            )

    async def _unlink_keys(self, keys: Sequence[bytes | str]) -> int:
        """Run one bounded Redis UNLINK command for opaque key material."""
        async with asyncio.timeout(self._command_timeout_seconds):
            return await self._client.unlink(*keys)


def create_provider_purger(settings: CacheConfig) -> RedisProviderPurger:
    """Create a lazy Redis provider purger from validated cache settings.

    Args:
        settings: Enabled cache settings containing the Redis endpoint and environment.

    Returns:
        A purger owning a lazy Redis client for one environment namespace.

    Raises:
        ValueError: If settings lack the validated Redis endpoint required for purge.
    """
    redis_url = settings.redis_url
    if redis_url is None:
        raise ValueError("Provider purge requires a Redis URL")
    client = Redis.from_url(redis_url.get_secret_value(), decode_responses=False)
    return RedisProviderPurger(
        cast(_PurgeRedisClient, client),
        settings.namespace,
        settings.environment,
    )


def _layer_for(
    provider: PurgeProvider,
    namespace_category: PurgeNamespaceCategory,
) -> str:
    """Return the stable visible cache layer selected by one category."""
    if namespace_category is PurgeNamespaceCategory.EXACT_OPERATION:
        return f"provider:{provider.value}"
    return _STATIC_CATEGORY_LAYERS[namespace_category]


def _duration_ms(started_at: float, monotonic: Callable[[], float]) -> float:
    """Return a rounded nonnegative monotonic duration in milliseconds."""
    return round((monotonic() - started_at) * 1_000, 3)


def _emit_purge_event(
    provider: PurgeProvider,
    environment: Environment,
    namespace_category: PurgeNamespaceCategory,
    duration_ms: float,
    outcome: str,
    deleted_count: int,
) -> None:
    """Emit one privacy-safe structured category outcome without Redis material."""
    logger.info(
        "provider cache purge",
        extra={
            "provider": provider.value,
            "environment": environment.value,
            "namespace_category": namespace_category.value,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "deleted_count": deleted_count,
        },
    )
