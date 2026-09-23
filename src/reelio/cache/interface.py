"""Application-owned asynchronous shared-cache abstractions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class CacheEntry[ValueT]:
    """Describe one versioned shared-cache operation.

    Attributes:
        layer: Stable cache layer literal.
        key_version: Stable cache-key shape version.
        identity: Canonical operation identity used only as HMAC input.
        codec: Application-owned cache value codec.
        ttl_seconds: Function selecting a positive expiry for a loaded value.
    """

    layer: str
    key_version: str
    identity: JsonObject
    codec: CacheCodec[ValueT]
    ttl_seconds: Callable[[ValueT], int]


class AsyncCache(Protocol):
    """Read through versioned cache boundary owned outside its consumers."""

    async def get_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Return a cached value or invoke the loader exactly once for a miss."""
        ...

    async def aclose(self) -> None:
        """Release cache-owned resources."""
        ...
