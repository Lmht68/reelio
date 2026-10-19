"""Application-owned optional shared-cache boundary."""

from reelio.cache.config import CacheConfig
from reelio.cache.disabled import DisabledCache
from reelio.cache.interface import (
    AsyncCache,
    CacheCodec,
    CacheCodecError,
    CacheEntry,
    CacheLoadResult,
    CacheSkip,
    CacheWrite,
    RetainedCacheValue,
    RevalidatingCacheEntry,
)
from reelio.cache.redis import RedisCache, create_cache

__all__ = [
    "AsyncCache",
    "CacheCodec",
    "CacheCodecError",
    "CacheConfig",
    "CacheEntry",
    "CacheLoadResult",
    "CacheSkip",
    "CacheWrite",
    "DisabledCache",
    "RedisCache",
    "RetainedCacheValue",
    "RevalidatingCacheEntry",
    "create_cache",
]
