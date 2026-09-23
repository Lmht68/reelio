"""Behavioral contracts for the application-owned shared-cache boundary."""

import json
from collections.abc import Awaitable

import pytest

from reelio.cache import CacheCodecError, CacheEntry, DisabledCache, RedisCache
from reelio.cache.interface import JsonObject
from tests.cache.fakes import FakeRedis


class _Clock:
    """Advance deterministic monotonic time for cache expiry tests."""

    def __init__(self) -> None:
        """Initialize time at zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """Return the current deterministic time."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Advance the deterministic time.

        Args:
            seconds: Positive or zero simulated seconds to add.
        """
        self.value += seconds


class _OptionalTextCodec:
    """Strict codec for nullable text values."""

    version = "text-v1"

    def encode(self, value: str | None) -> JsonObject:
        """Encode nullable text into a strict payload.

        Args:
            value: Nullable text cache value.

        Returns:
            JSON payload containing the text value.
        """
        return {"text": value}

    def decode(self, payload: JsonObject) -> str | None:
        """Decode nullable text only from an exact strict payload.

        Args:
            payload: Parsed cached JSON payload.

        Returns:
            Nullable text value.

        Raises:
            CacheCodecError: If payload does not contain exactly nullable text.
        """
        value = payload.get("text")
        if set(payload) != {"text"} or value is not None and not isinstance(value, str):
            raise CacheCodecError("Expected nullable text payload")
        return value


class _StringListCodec:
    """Strict codec for a mutable collection allocation contract."""

    version = "list-v1"

    def encode(self, value: list[str]) -> JsonObject:
        """Encode one list of strings.

        Args:
            value: String collection to persist.

        Returns:
            JSON payload with copied string collection.
        """
        return {"items": list(value)}

    def decode(self, payload: JsonObject) -> list[str]:
        """Decode one strict list of strings with an independent list allocation.

        Args:
            payload: Parsed cached JSON payload.

        Returns:
            Newly allocated string list.

        Raises:
            CacheCodecError: If the payload is not a string list.
        """
        value = payload.get("items")
        if (
            set(payload) != {"items"}
            or not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
        ):
            raise CacheCodecError("Expected string list payload")
        return [item for item in value if isinstance(item, str)]


def _text_entry(
    identity: JsonObject,
    *,
    ttl_seconds: int = 10,
) -> CacheEntry[str | None]:
    """Create one deterministic nullable-text cache operation.

    Args:
        identity: JSON operation identity.
        ttl_seconds: Fixed positive expiry for loaded values.

    Returns:
        Cache descriptor using the nullable-text codec.
    """
    return CacheEntry(
        layer="provider:open-library",
        key_version="v1",
        identity=identity,
        codec=_OptionalTextCodec(),
        ttl_seconds=lambda _: ttl_seconds,
    )


def _redis_cache(clock: _Clock) -> tuple[RedisCache, FakeRedis]:
    """Create a cache and its observable fake client.

    Args:
        clock: Deterministic expiry clock.

    Returns:
        Cache and client sharing the supplied clock.
    """
    redis = FakeRedis(clock)
    return RedisCache(redis, "reelio:local", b"test-cache-key"), redis


async def test_disabled_cache_loads_once_without_persisting_nullable_value() -> None:
    """Disabled caching preserves one loader call even when it returns None."""
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return None

    cache = DisabledCache()
    entry = _text_entry({"path": "/search.json"})

    assert await cache.get_or_load(entry, load) is None
    assert calls == 1
    await cache.aclose()


async def test_redis_cache_reuses_fresh_nullable_values_without_loader_calls() -> None:
    """A successful cached None remains distinct from a cache miss."""
    clock = _Clock()
    cache, _ = _redis_cache(clock)
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return None

    entry = _text_entry({"path": "/search.json"})

    assert await cache.get_or_load(entry, load) is None
    assert await cache.get_or_load(entry, load) is None
    assert calls == 1


async def test_redis_cache_canonicalizes_equivalent_identity_and_isolates_distinct_identity() -> (
    None
):
    """Object ordering shares one HMAC key while distinct identities never collide."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)
    calls = 0

    async def load_first() -> str | None:
        nonlocal calls
        calls += 1
        return "first"

    async def load_second() -> str | None:
        nonlocal calls
        calls += 1
        return "second"

    first = _text_entry({"path": "/search.json", "params": {"title": "Dune", "limit": "3"}})
    equivalent = _text_entry({"params": {"limit": "3", "title": "Dune"}, "path": "/search.json"})
    distinct = _text_entry({"path": "/search.json", "params": {"title": "Foundation"}})

    assert await cache.get_or_load(first, load_first) == "first"
    assert await cache.get_or_load(equivalent, load_second) == "first"
    assert await cache.get_or_load(distinct, load_second) == "second"
    assert calls == 2
    assert len(redis.keys) == 2


async def test_redis_cache_expires_entries_at_the_exact_ttl_boundary() -> None:
    """Freshness ends exactly when Redis expiry reaches the configured TTL."""
    clock = _Clock()
    cache, _ = _redis_cache(clock)
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return f"value-{calls}"

    entry = _text_entry({"path": "/search.json"}, ttl_seconds=10)

    assert await cache.get_or_load(entry, load) == "value-1"
    clock.advance(9.999)
    assert await cache.get_or_load(entry, load) == "value-1"
    clock.advance(0.001)
    assert await cache.get_or_load(entry, load) == "value-2"
    assert calls == 2


@pytest.mark.parametrize(
    "envelope",
    [
        {"envelope_version": 2, "value_version": "text-v1", "value": {"text": "bad"}},
        {"envelope_version": 1, "value_version": "other-v1", "value": {"text": "bad"}},
        {"envelope_version": 1, "value_version": "text-v1", "value": {"text": 1}},
    ],
)
async def test_redis_cache_reloads_incompatible_or_invalid_payloads(
    envelope: JsonObject,
) -> None:
    """Envelope and codec incompatibilities are misses that successful loads overwrite."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})

    async def load() -> str | None:
        return "fresh"

    key = cache._cache_key(entry)
    await redis.put_raw(key, json.dumps(envelope).encode())

    assert await cache.get_or_load(entry, load) == "fresh"
    assert await redis.get(key) == (
        b'{"envelope_version":1,"value":{"text":"fresh"},"value_version":"text-v1"}'
    )


async def test_redis_cache_decodes_independent_collection_allocations() -> None:
    """Cache hits reconstruct values rather than retaining mutable Python collections."""
    clock = _Clock()
    cache, _ = _redis_cache(clock)
    entry = CacheEntry(
        layer="provider:open-library",
        key_version="v1",
        identity={"path": "/search.json"},
        codec=_StringListCodec(),
        ttl_seconds=lambda _: 10,
    )

    first = await cache.get_or_load(entry, lambda: _completed(["author"]))
    first.append("mutated")
    cached = await cache.get_or_load(entry, lambda: _completed(["unexpected"]))

    assert cached == ["author"]
    assert cached is not first


async def test_redis_cache_fails_open_for_read_and_write_errors() -> None:
    """Redis command failures preserve the normal loader value and later provider behavior."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return "loaded"

    redis.fail_reads = True
    assert await cache.get_or_load(entry, load) == "loaded"
    redis.fail_reads = False
    redis.fail_writes = True
    write_failure_entry = _text_entry({"path": "/books/OL1M.json"})
    assert await cache.get_or_load(write_failure_entry, load) == "loaded"
    assert calls == 2


async def test_redis_cache_propagates_loader_errors_without_writing() -> None:
    """Provider errors remain unchanged and never create cache entries."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})

    async def load() -> str | None:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await cache.get_or_load(entry, load)
    assert redis.keys == ()


async def test_redis_cache_obscures_identity_from_visible_key_prefix() -> None:
    """Visible keys expose stable metadata and a digest, never raw lookup content."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)
    raw_title = "A Private Title"
    entry = _text_entry({"path": "/search.json", "params": {"title": raw_title}})

    assert await cache.get_or_load(entry, lambda: _completed("cached")) == "cached"

    assert redis.keys[0].startswith("reelio:local:provider:open-library:v1:")
    assert raw_title not in redis.keys[0]
    assert redis.keys[0].rsplit(":", maxsplit=1)[1].isalnum()
    assert len(redis.keys[0].rsplit(":", maxsplit=1)[1]) == 64


async def test_redis_cache_closes_its_owned_client_once() -> None:
    """Cache shutdown is idempotent even when called repeatedly."""
    clock = _Clock()
    cache, redis = _redis_cache(clock)

    await cache.aclose()
    await cache.aclose()

    assert redis.close_calls == 1


def _completed[ValueT](value: ValueT) -> Awaitable[ValueT]:
    """Return one already-completed awaitable value.

    Args:
        value: Value to resolve.

    Returns:
        Awaitable resolving to value.
    """

    async def completed() -> ValueT:
        return value

    return completed()
