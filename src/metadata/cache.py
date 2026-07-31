import math
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import cast

from cachetools import TTLCache

from src.entities.models import EntityType
from src.metadata.models import EntityMetadata

CacheKey = tuple[str, EntityType, str, str]


@dataclass(frozen=True, slots=True)
class CacheLookup:
    hit: bool
    value: EntityMetadata | None = None


_NO_MATCH = object()


class InMemoryTTLCache:
    """Bounded in-memory TTL cache for entity metadata results.

    Stores positive hits as EntityMetadata and negative hits as an internal sentinel.
    Not thread-safe; safe only while confined to a single event loop and thread.
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and greater than zero")
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self._store: MutableMapping[CacheKey, object] = TTLCache(
            maxsize=max_entries,
            ttl=ttl_seconds,
            timer=timer,
        )

    def get(self, key: CacheKey) -> CacheLookup:
        try:
            raw = self._store[key]
        except KeyError:
            return CacheLookup(hit=False)
        if raw is _NO_MATCH:
            return CacheLookup(hit=True, value=None)
        return CacheLookup(hit=True, value=cast(EntityMetadata, raw))

    def set(self, key: CacheKey, value: EntityMetadata | None) -> None:
        self._store[key] = _NO_MATCH if value is None else value
