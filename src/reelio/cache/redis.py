"""Redis-backed implementation of the application-owned async cache boundary."""

import asyncio
import hashlib
import hmac
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from reelio.cache.config import CacheConfig
from reelio.cache.disabled import DisabledCache
from reelio.cache.interface import AsyncCache, CacheCodecError, CacheEntry, JsonObject

logger = logging.getLogger(__name__)

_ENVELOPE_VERSION = 1
_CACHE_MISS = object()
_LAYER_PATTERN = re.compile(r"[a-z][a-z0-9-]*(?::[a-z][a-z0-9-]*)*")
_KEY_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")


class _RedisClient(Protocol):
    """Contain the minimal Redis command surface owned by this cache."""

    def get(self, name: str) -> Awaitable[bytes | str | None]:
        """Load one raw cache payload."""
        ...

    def set(self, name: str, value: bytes, ex: int) -> Awaitable[object]:
        """Store one raw cache payload with an expiry."""
        ...

    def aclose(self) -> Awaitable[None]:
        """Close the client and its owned pool."""
        ...


class RedisCache:
    """Read through versioned values using one lazily connected Redis client."""

    def __init__(self, client: _RedisClient, namespace: str, key_secret: bytes) -> None:
        """Initialize a cache around one client owned by this instance.

        Args:
            client: Redis client whose pool this cache closes.
            namespace: Fixed environment-derived visible key namespace.
            key_secret: Dedicated HMAC key for private operation identities.
        """
        self._client = client
        self._namespace = namespace
        self._key_secret = key_secret
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def get_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Return a decoded cache hit or load and best-effort store a fresh value.

        Args:
            entry: Versioned cache operation and codec definition.
            loader: Awaitable operation producing the value for a cache miss.

        Returns:
            A decoded cached value or the loader's original value.
        """
        try:
            key = self._cache_key(entry)
        except (TypeError, ValueError):
            _log_cache_failure(entry.layer, "key_build_failed")
            return await loader()

        try:
            raw_payload = await self._client.get(key)
        except RedisError:
            _log_cache_failure(entry.layer, "read_failed")
        else:
            decoded_value = _decode_cached_value(raw_payload, entry)
            if decoded_value is not _CACHE_MISS:
                return cast(ValueT, decoded_value)

        value = await loader()
        await self._store(entry, key, value)
        return value

    async def aclose(self) -> None:
        """Close the owned Redis client at most once."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._client.aclose()

    def _cache_key[ValueT](self, entry: CacheEntry[ValueT]) -> str:
        """Build the HMAC-obscured visible Redis key for one cache entry.

        Args:
            entry: Cache operation descriptor containing stable key metadata.

        Returns:
            Stable Redis key without raw operation identity values.

        Raises:
            ValueError: If cache layer or version literals are unsafe.
            TypeError: If the operation identity is not canonical JSON.
        """
        if not _LAYER_PATTERN.fullmatch(entry.layer):
            raise ValueError("Cache layer must be a stable literal")
        if not _KEY_VERSION_PATTERN.fullmatch(entry.key_version):
            raise ValueError("Cache key version must be a stable literal")
        identity_bytes = json.dumps(
            entry.identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        digest = hmac.new(self._key_secret, identity_bytes, hashlib.sha256).hexdigest()
        return f"{self._namespace}:{entry.layer}:{entry.key_version}:{digest}"

    async def _store[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        value: ValueT,
    ) -> None:
        """Best-effort encode and persist one successfully loaded cache value."""
        try:
            ttl_seconds = entry.ttl_seconds(value)
            if type(ttl_seconds) is not int or ttl_seconds <= 0:
                raise ValueError("Cache TTL must be a positive integer")
            encoded_value = entry.codec.encode(value)
            envelope = {
                "envelope_version": _ENVELOPE_VERSION,
                "value_version": entry.codec.version,
                "value": encoded_value,
            }
            payload = json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        except (CacheCodecError, TypeError, ValueError):
            _log_cache_failure(entry.layer, "encode_failed")
            return

        try:
            await self._client.set(key, payload, ex=ttl_seconds)
        except RedisError:
            _log_cache_failure(entry.layer, "write_failed")


def create_cache(settings: CacheConfig) -> AsyncCache:
    """Create the configured cache without connecting to Redis during startup.

    Args:
        settings: Validated shared-cache configuration.

    Returns:
        DisabledCache when disabled, otherwise one RedisCache owning a lazy client.
    """
    if not settings.enabled:
        return DisabledCache()

    redis_url = settings.redis_url
    key_secret = settings.key_secret
    if redis_url is None or key_secret is None:
        raise ValueError("Enabled cache settings must include Redis URL and key secret")
    client = Redis.from_url(redis_url.get_secret_value(), decode_responses=False)
    return RedisCache(client, settings.namespace, key_secret.get_secret_value().encode())


def _decode_cached_value[ValueT](
    raw_payload: bytes | str | None,
    entry: CacheEntry[ValueT],
) -> ValueT | object:
    """Decode a matching cache envelope, returning an internal miss sentinel on failure."""
    if raw_payload is None:
        return _CACHE_MISS
    try:
        envelope = json.loads(raw_payload)
        if not isinstance(envelope, dict) or set(envelope) != {
            "envelope_version",
            "value_version",
            "value",
        }:
            return _CACHE_MISS
        if (
            envelope["envelope_version"] != _ENVELOPE_VERSION
            or envelope["value_version"] != entry.codec.version
            or not isinstance(envelope["value"], dict)
        ):
            return _CACHE_MISS
        return entry.codec.decode(cast(JsonObject, envelope["value"]))
    except (CacheCodecError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _CACHE_MISS


def _log_cache_failure(layer: str, outcome: str) -> None:
    """Log only cache layer and failure outcome, never a key or identity."""
    logger.warning(
        "Shared cache operation failed", extra={"cache_layer": layer, "outcome": outcome}
    )
