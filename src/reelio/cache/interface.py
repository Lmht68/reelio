"""Application-owned asynchronous shared-cache abstractions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class CacheCodecError(ValueError):
    """Signal that a cached payload cannot reconstruct an owned application value."""


class CacheCodec[ValueT](Protocol):
    """Encode and decode one versioned application-owned cached value."""

    version: str

    def encode(self, value: ValueT) -> JsonObject:
        """Encode an application value into a JSON object."""
        ...

    def decode(self, payload: JsonObject) -> ValueT:
        """Decode a validated JSON object into an application value.

        Raises:
            CacheCodecError: If the payload is malformed or incompatible.
        """
        ...


def _validate_wait_timeout(wait_timeout_seconds: float) -> None:
    """Reject coordination deadlines that cannot bound waiting."""
    if not isfinite(wait_timeout_seconds) or wait_timeout_seconds <= 0:
        raise ValueError("Cache wait timeout must be finite and positive")


@dataclass(frozen=True, slots=True)
class CacheEntry[ValueT]:
    """Describe one versioned shared-cache operation.

    Attributes:
        layer: Stable cache layer literal.
        key_version: Stable cache-key shape version.
        identity: Canonical operation identity used only as HMAC input.
        codec: Application-owned cache value codec.
        ttl_seconds: Function selecting a positive expiry for a loaded value.
        wait_timeout_seconds: Maximum waiter poll duration before an unowned load.
    """

    layer: str
    key_version: str
    identity: JsonObject
    codec: CacheCodec[ValueT]
    ttl_seconds: Callable[[ValueT], int]
    wait_timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate the bounded miss-coordination policy."""
        _validate_wait_timeout(self.wait_timeout_seconds)


@dataclass(frozen=True, slots=True)
class RevalidatingCacheEntry[ValueT]:
    """Describe a versioned operation that can conditionally revalidate retained values.

    Attributes:
        layer: Stable cache layer literal.
        key_version: Stable cache-key shape version.
        identity: Canonical operation identity used only as HMAC input.
        codec: Application-owned cache value codec.
        wait_timeout_seconds: Maximum waiter poll duration before an unowned load.
    """

    layer: str
    key_version: str
    identity: JsonObject
    codec: CacheCodec[ValueT]
    wait_timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate the bounded miss-coordination policy."""
        _validate_wait_timeout(self.wait_timeout_seconds)


@dataclass(frozen=True, slots=True)
class CacheWrite[ValueT]:
    """Describe a value and its serving and physical-retention durations.

    Attributes:
        value: Normalized application-owned value to cache.
        freshness_seconds: Nonnegative duration for which the value is a cache hit.
        retention_seconds: Positive physical duration for conditional revalidation.
    """

    value: ValueT
    freshness_seconds: int
    retention_seconds: int

    def __post_init__(self) -> None:
        """Validate exact cache durations before an owned write."""
        if type(self.freshness_seconds) is not int or self.freshness_seconds < 0:
            raise ValueError("Cache freshness must be a nonnegative integer")
        if type(self.retention_seconds) is not int or self.retention_seconds <= 0:
            raise ValueError("Cache retention must be a positive integer")
        if self.freshness_seconds > self.retention_seconds:
            raise ValueError("Cache freshness cannot exceed retention")


@dataclass(frozen=True, slots=True)
class CacheSkip[ValueT]:
    """Describe a loader value that must not remain in the shared cache.

    Attributes:
        value: Application-owned value to return without retention.
    """

    value: ValueT


@dataclass(frozen=True, slots=True)
class RetainedCacheValue[ValueT]:
    """Expose a physically retained value only to its revalidation loader.

    Attributes:
        value: Normalized application-owned retained value.
    """

    value: ValueT


class AsyncCache(Protocol):
    """Read through versioned cache boundary owned outside its consumers."""

    async def get_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Return a cached value or invoke the loader exactly once for a miss."""
        ...

    async def get_or_load_revalidating[ValueT](
        self,
        entry: RevalidatingCacheEntry[ValueT],
        loader: Callable[
            [RetainedCacheValue[ValueT] | None],
            Awaitable[CacheWrite[ValueT] | CacheSkip[ValueT]],
        ],
    ) -> ValueT:
        """Return a fresh value or load against an optional retained value."""
        ...

    async def aclose(self) -> None:
        """Release cache-owned resources."""
        ...
