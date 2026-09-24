"""Behavioral contracts for shared-cache coordination and failure isolation."""

import asyncio
import json
import logging
from collections.abc import Awaitable
from itertools import count
from math import inf, nan
from typing import cast

import pytest

from reelio.cache import (
    CacheCodecError,
    CacheEntry,
    CacheSkip,
    CacheWrite,
    DisabledCache,
    RedisCache,
    RevalidatingCacheEntry,
)
from reelio.cache.interface import JsonObject, RetainedCacheValue
from reelio.cache.redis import _CachePolicy, _CacheRuntime
from tests.cache.fakes import FakeRedis, ManualClock, ManualSleeper


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


class _TextCodec:
    """Strict codec for non-null text values."""

    version = "text-v1"

    def encode(self, value: str) -> JsonObject:
        """Encode text into a strict payload."""
        return {"text": value}

    def decode(self, payload: JsonObject) -> str:
        """Decode text only from an exact strict payload."""
        value = payload.get("text")
        if set(payload) != {"text"} or not isinstance(value, str):
            raise CacheCodecError("Expected text payload")
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
    wait_timeout_seconds: float = 1.0,
) -> CacheEntry[str | None]:
    """Create one deterministic nullable-text cache operation.

    Args:
        identity: JSON operation identity.
        ttl_seconds: Fixed positive expiry for loaded values.
        wait_timeout_seconds: Bounded layer-specific wait duration.

    Returns:
        Cache descriptor using the nullable-text codec.
    """
    return CacheEntry(
        layer="provider:open-library",
        key_version="v1",
        identity=identity,
        codec=_OptionalTextCodec(),
        ttl_seconds=lambda _: ttl_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
    )


def _revalidating_text_entry(
    identity: JsonObject,
    *,
    wait_timeout_seconds: float = 1.0,
) -> RevalidatingCacheEntry[str]:
    """Create one deterministic revalidating text-cache operation."""
    return RevalidatingCacheEntry(
        layer="provider:spotify",
        key_version="v1",
        identity=identity,
        codec=_TextCodec(),
        wait_timeout_seconds=wait_timeout_seconds,
    )


def _cache_runtime(clock: ManualClock, sleeper: ManualSleeper, prefix: str) -> _CacheRuntime:
    """Create deterministic time and unique lease tokens for one cache instance."""
    tokens = count()
    return _CacheRuntime(clock, sleeper.sleep, lambda: f"{prefix}-{next(tokens)}")


def _redis_cache(
    clock: ManualClock,
    *,
    redis: FakeRedis | None = None,
    sleeper: ManualSleeper | None = None,
    policy: _CachePolicy | None = None,
    token_prefix: str = "token",
) -> tuple[RedisCache, FakeRedis, ManualSleeper]:
    """Create one observable cache instance using shared deterministic infrastructure."""
    selected_sleeper = sleeper or ManualSleeper(clock)
    selected_redis = redis or FakeRedis(clock)
    cache = RedisCache(
        selected_redis,
        "reelio:local",
        b"test-cache-key",
        policy=policy,
        runtime=_cache_runtime(clock, selected_sleeper, token_prefix),
    )
    return cache, selected_redis, selected_sleeper


async def _settle() -> None:
    """Yield enough event-loop turns for tasks to register their next deterministic wait."""
    for _ in range(3):
        await asyncio.sleep(0)


async def _wait_for_command(redis: FakeRedis, operation: str, expected_count: int) -> None:
    """Wait for a cache task to issue a known number of fake Redis commands."""
    for _ in range(20):
        if redis.command_counts[operation] >= expected_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {expected_count} {operation} commands")


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


async def test_disabled_cache_revalidates_without_retaining() -> None:
    """Disabled caching supplies no retained value and unwraps either loader result."""
    cache = DisabledCache()
    entry = _revalidating_text_entry({"operation": "search"})
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheWrite("loaded", freshness_seconds=0, retention_seconds=1)

    assert await cache.get_or_load_revalidating(entry, load) == "loaded"
    assert retained_values == [None]
    await cache.aclose()


@pytest.mark.parametrize(
    ("freshness_seconds", "retention_seconds"),
    [
        (-1, 1),
        (True, 1),
        (0, 0),
        (0, True),
        (2, 1),
    ],
)
def test_cache_write_rejects_invalid_exact_durations(
    freshness_seconds: int,
    retention_seconds: int,
) -> None:
    """Serving freshness must be integral, nonnegative, and within retention."""
    with pytest.raises(ValueError):
        CacheWrite(
            "value",
            freshness_seconds=freshness_seconds,
            retention_seconds=retention_seconds,
        )


@pytest.mark.parametrize("wait_timeout_seconds", [0.0, -0.1, nan, inf, -inf])
def test_cache_entry_rejects_unbounded_wait_policy(wait_timeout_seconds: float) -> None:
    """Every cache layer requires a finite positive coordination deadline."""
    with pytest.raises(ValueError, match="finite and positive"):
        _text_entry({"path": "/search.json"}, wait_timeout_seconds=wait_timeout_seconds)


@pytest.mark.parametrize("wait_timeout_seconds", [0.0, -0.1, nan, inf, -inf])
def test_revalidating_cache_entry_rejects_unbounded_wait_policy(
    wait_timeout_seconds: float,
) -> None:
    """Revalidating operations keep the ordinary bounded coordination invariant."""
    with pytest.raises(ValueError, match="finite and positive"):
        _revalidating_text_entry(
            {"operation": "search"},
            wait_timeout_seconds=wait_timeout_seconds,
        )


async def test_revalidating_cache_classifies_fresh_and_retained_at_exact_boundary() -> None:
    """Freshness ends at the PTTL boundary and exposes retained data only to loaders."""
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def first_load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheWrite("first", freshness_seconds=5, retention_seconds=10)

    async def second_load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheWrite("second", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, first_load) == "first"
    clock.advance(4.999)
    assert await cache.get_or_load_revalidating(entry, second_load) == "first"
    clock.advance(0.001)
    assert await cache.get_or_load_revalidating(entry, second_load) == "second"
    assert retained_values == [None, RetainedCacheValue("first")]


async def test_revalidating_cache_treats_zero_pttl_as_retained() -> None:
    """An atomic PTTL of zero remains eligible for one conditional revalidation."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def write_first(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("first", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, write_first) == "first"
    key = cache._cache_key(entry)
    redis.set_pttl_override(key, 0)
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def skip_retained(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheSkip("revalidated")

    assert await cache.get_or_load_revalidating(entry, skip_retained) == "revalidated"
    assert retained_values == [RetainedCacheValue("first")]
    assert redis.raw_value(key) is None


@pytest.mark.parametrize("pttl_milliseconds", [-1, -2, -3, 10_001])
async def test_revalidating_cache_heals_invalid_atomic_pttl_before_loading(
    pttl_milliseconds: int,
) -> None:
    """Impossible physical-retention states cannot reach loaders as retained values."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def write_first(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("first", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, write_first) == "first"
    key = cache._cache_key(entry)
    redis.set_pttl_override(key, pttl_milliseconds)
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def skip_corrupt(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheSkip("loaded")

    assert await cache.get_or_load_revalidating(entry, skip_corrupt) == "loaded"
    assert retained_values == [None]
    assert redis.command_counts["corrupt_delete"] == 1
    assert redis.raw_value(key) is None


async def test_revalidating_cache_expires_physical_retention_after_zero_boundary() -> None:
    """Values expire physically after their retained PTTL reaches zero and advances."""
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def write_first(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("first", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, write_first) == "first"
    clock.advance(10.001)
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def skip_expired(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheSkip("cold")

    assert await cache.get_or_load_revalidating(entry, skip_expired) == "cold"
    assert retained_values == [None]


async def test_revalidating_cache_read_failure_loads_without_retention_or_write() -> None:
    """A failed atomic read supplies no retained data and makes that invocation read-only."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})
    redis.fail_operations.add("read")
    retained_values: list[RetainedCacheValue[str] | None] = []

    async def load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheWrite("loaded", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, load) == "loaded"
    assert retained_values == [None]
    assert redis.raw_value(cache._cache_key(entry)) is None


async def test_invalid_ordinary_ttl_returns_value_without_caching() -> None:
    """Ordinary TTL validation stays on the guarded write path and remains fail-open."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/invalid-ttl"}, ttl_seconds=0)

    assert await cache.get_or_load(entry, lambda: _completed("provider-value")) == "provider-value"
    assert redis.raw_value(cache._cache_key(entry)) is None


async def test_revalidating_cache_persists_the_exact_v2_freshness_envelope() -> None:
    """Persist only the V2 envelope fields required for cross-worker revalidation."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("cached", freshness_seconds=5, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, load) == "cached"

    raw_payload = redis.raw_value(cache._cache_key(entry))
    assert raw_payload is not None
    assert json.loads(raw_payload) == {
        "envelope_version": 2,
        "value_version": "text-v1",
        "value": {"text": "cached"},
        "freshness_seconds": 5,
        "retention_seconds": 10,
    }


async def test_retained_loader_error_preserves_value_for_a_later_revalidation() -> None:
    """A provider error releases ownership without deleting retained data or serving it."""
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def first_load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("first", freshness_seconds=0, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, first_load) == "first"

    async def fail_load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value == RetainedCacheValue("first")
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await cache.get_or_load_revalidating(entry, fail_load)

    retained_values: list[RetainedCacheValue[str] | None] = []

    async def recover_load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        retained_values.append(retained_value)
        return CacheSkip("recovered")

    assert await cache.get_or_load_revalidating(entry, recover_load) == "recovered"
    assert retained_values == [RetainedCacheValue("first")]


async def test_owner_rereads_newer_fill_before_calling_its_loader() -> None:
    """A lease owner must not reload when another worker fills before its ownership read."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})
    key = cache._cache_key(entry)
    redis.block("lease_acquire")
    loader_calls = 0

    async def load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        nonlocal loader_calls
        loader_calls += 1
        return CacheWrite("unexpected", freshness_seconds=5, retention_seconds=10)

    task = asyncio.create_task(cache.get_or_load_revalidating(entry, load))
    await redis.started("lease_acquire").wait()
    await redis.put_raw(
        key,
        json.dumps(
            {
                "envelope_version": 2,
                "value_version": "text-v1",
                "value": {"text": "newer"},
                "freshness_seconds": 5,
                "retention_seconds": 10,
            }
        ).encode(),
        ex=10,
    )
    redis.unblock("lease_acquire")

    assert await task == "newer"
    assert loader_calls == 0


async def test_cache_skip_does_not_discard_after_lease_ownership_is_lost() -> None:
    """A stale CacheSkip token cannot delete retained data after another owner takes over."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _revalidating_text_entry({"operation": "search"})

    async def seed(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value is None
        return CacheWrite("retained", freshness_seconds=0, retention_seconds=10)

    assert await cache.get_or_load_revalidating(entry, seed) == "retained"
    loader_started = asyncio.Event()
    release_loader = asyncio.Event()

    async def skip(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str] | CacheSkip[str]:
        assert retained_value == RetainedCacheValue("retained")
        loader_started.set()
        await release_loader.wait()
        return CacheSkip("provider-result")

    task = asyncio.create_task(cache.get_or_load_revalidating(entry, skip))
    await loader_started.wait()
    lease_key = cache._lease_key(cache._cache_key(entry))
    await redis.set(lease_key, "replacement-owner", px=30_000)
    release_loader.set()

    assert await task == "provider-result"
    assert redis.raw_value(cache._cache_key(entry)) is not None


async def test_redis_cache_reuses_fresh_nullable_values_without_loader_calls() -> None:
    """A successful cached None remains distinct from a cache miss."""
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
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
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
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
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
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
    "raw_payload",
    [
        b"{",
        b"\xff",
        json.dumps({"envelope_version": 2, "value_version": "text-v1", "value": {}}).encode(),
        json.dumps({"envelope_version": 1, "value_version": "other-v1", "value": {}}).encode(),
        json.dumps({"envelope_version": 1, "value_version": "text-v1", "value": []}).encode(),
        json.dumps(
            {"envelope_version": 1, "value_version": "text-v1", "value": {"text": 1}}
        ).encode(),
    ],
)
async def test_redis_cache_heals_corrupt_payloads_before_filling(raw_payload: bytes) -> None:
    """Malformed envelopes become a fresh load after bounded conditional deletion."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    key = cache._cache_key(entry)
    await redis.put_raw(key, raw_payload)
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return "fresh"

    assert await cache.get_or_load(entry, load) == "fresh"
    assert await cache.get_or_load(entry, load) == "fresh"
    assert calls == 1
    assert redis.command_counts["corrupt_delete"] == 1


async def test_corruption_delete_failure_returns_loader_value_without_overwrite() -> None:
    """Unavailable corruption cleanup fails open without replacing unverified bytes."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    key = cache._cache_key(entry)
    raw_payload = b"{"
    await redis.put_raw(key, raw_payload)
    redis.fail_operations.add("corrupt_delete")

    assert await cache.get_or_load(entry, lambda: _completed("fresh")) == "fresh"
    assert redis.raw_value(key) == raw_payload


async def test_redis_cache_decodes_independent_collection_allocations() -> None:
    """Cache hits reconstruct values rather than retaining mutable Python collections."""
    clock = ManualClock()
    cache, _, _ = _redis_cache(clock)
    entry = CacheEntry(
        layer="provider:open-library",
        key_version="v1",
        identity={"path": "/search.json"},
        codec=_StringListCodec(),
        ttl_seconds=lambda _: 10,
        wait_timeout_seconds=1.0,
    )

    first = await cache.get_or_load(entry, lambda: _completed(["author"]))
    first.append("mutated")
    cached = await cache.get_or_load(entry, lambda: _completed(["unexpected"]))

    assert cached == ["author"]
    assert cached is not first


async def test_redis_cache_keeps_identity_out_of_visible_data_and_lease_keys() -> None:
    """Data and lease keys contain stable metadata and a digest, not private identity."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    raw_title = "A Private Title"
    entry = _text_entry({"path": "/search.json", "params": {"title": raw_title}})

    assert await cache.get_or_load(entry, lambda: _completed("cached")) == "cached"

    data_key = redis.keys[0]
    assert data_key.startswith("reelio:local:provider:open-library:v1:")
    assert raw_title not in data_key
    assert len(data_key.rsplit(":", maxsplit=1)[1]) == 64
    assert raw_title not in cache._lease_key(data_key)


async def test_redis_cache_coalesces_one_miss_across_instances() -> None:
    """Two cache instances share one expensive load and leave a reusable fresh value."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    first_cache, _, _ = _redis_cache(
        clock,
        redis=redis,
        sleeper=sleeper,
        token_prefix="first",
    )
    second_cache, _, _ = _redis_cache(
        clock,
        redis=redis,
        sleeper=sleeper,
        token_prefix="second",
    )
    entry = _text_entry({"path": "/search.json"})
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()
    first_calls = 0
    second_calls = 0

    async def first_loader() -> str | None:
        nonlocal first_calls
        first_calls += 1
        owner_started.set()
        await release_owner.wait()
        return "shared"

    async def second_loader() -> str | None:
        nonlocal second_calls
        second_calls += 1
        return "unexpected"

    first_task = asyncio.create_task(first_cache.get_or_load(entry, first_loader))
    await owner_started.wait()
    second_task = asyncio.create_task(second_cache.get_or_load(entry, second_loader))
    await _wait_for_command(redis, "lease_acquire", 2)
    release_owner.set()
    assert await first_task == "shared"

    sleeper.advance(0.1)
    await _settle()
    assert await second_task == "shared"
    assert await second_cache.get_or_load(entry, second_loader) == "shared"
    assert first_calls == 1
    assert second_calls == 0


async def test_active_owner_renews_before_the_original_lease_expires() -> None:
    """Renewal retains sole ownership past the initial 30-second lease window."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    cache, redis, _ = _redis_cache(clock, sleeper=sleeper, token_prefix="owner")
    entry = _text_entry({"path": "/search.json"})
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner_loader() -> str | None:
        owner_started.set()
        await release_owner.wait()
        return "owner"

    owner_task = asyncio.create_task(cache.get_or_load(entry, owner_loader))
    await owner_started.wait()
    await _settle()
    lease_key = cache._lease_key(cache._cache_key(entry))

    sleeper.advance(10.0)
    await _settle()
    sleeper.advance(10.0)
    await _settle()
    sleeper.advance(10.001)
    await _settle()

    assert redis.command_counts["lease_renew"] == 3
    assert redis.raw_value(lease_key) == b"owner-0"

    release_owner.set()
    assert await owner_task == "owner"


async def test_waiter_loads_without_writing_after_its_layer_deadline() -> None:
    """A held owner cannot make a waiter exceed its bounded unowned fallback deadline."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    owner_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="owner")
    waiter_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="waiter")
    entry = _text_entry({"path": "/search.json"}, wait_timeout_seconds=0.2)
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()
    waiter_calls = 0

    async def owner_loader() -> str | None:
        owner_started.set()
        await release_owner.wait()
        return "owner"

    async def waiter_loader() -> str | None:
        nonlocal waiter_calls
        waiter_calls += 1
        return "waiter"

    owner_task = asyncio.create_task(owner_cache.get_or_load(entry, owner_loader))
    await owner_started.wait()
    waiter_task = asyncio.create_task(waiter_cache.get_or_load(entry, waiter_loader))
    await _wait_for_command(redis, "lease_acquire", 2)

    sleeper.advance(0.2)
    await _settle()
    assert await waiter_task == "waiter"
    assert waiter_calls == 1
    assert redis.command_counts["ownership_write"] == 0

    release_owner.set()
    assert await owner_task == "owner"


async def test_lost_owner_returns_its_value_without_overwriting_a_newer_fill() -> None:
    """An expired former owner cannot replace a newer value after another owner fills."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    first_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="first")
    second_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="second")
    entry = _text_entry({"path": "/search.json"})
    first_started = asyncio.Event()
    finish_first = asyncio.Event()

    async def first_loader() -> str | None:
        first_started.set()
        await finish_first.wait()
        return "first-result"

    first_task = asyncio.create_task(first_cache.get_or_load(entry, first_loader))
    await first_started.wait()
    await _settle()

    sleeper.advance(30.001)
    await _settle()
    assert redis.command_counts["lease_renew"] == 1

    assert (
        await second_cache.get_or_load(entry, lambda: _completed("second-result"))
        == "second-result"
    )
    finish_first.set()
    assert await first_task == "first-result"
    assert (
        await second_cache.get_or_load(entry, lambda: _completed("unexpected")) == "second-result"
    )


async def test_old_token_release_does_not_delete_a_new_owner_lease() -> None:
    """A stale loader error cannot release another owner's replacement lease token."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    policy = _CachePolicy(renew_interval_seconds=100.0)
    first_cache, _, _ = _redis_cache(
        clock,
        redis=redis,
        sleeper=sleeper,
        policy=policy,
        token_prefix="first",
    )
    second_cache, _, _ = _redis_cache(
        clock,
        redis=redis,
        sleeper=sleeper,
        policy=policy,
        token_prefix="second",
    )
    entry = _text_entry({"path": "/search.json"})
    first_started = asyncio.Event()
    fail_first = asyncio.Event()
    second_started = asyncio.Event()
    finish_second = asyncio.Event()

    async def first_loader() -> str | None:
        first_started.set()
        await fail_first.wait()
        raise RuntimeError("provider failed")

    async def second_loader() -> str | None:
        second_started.set()
        await finish_second.wait()
        return "second"

    first_task = asyncio.create_task(first_cache.get_or_load(entry, first_loader))
    await first_started.wait()
    await _settle()
    sleeper.advance(30.001)

    second_task = asyncio.create_task(second_cache.get_or_load(entry, second_loader))
    await second_started.wait()
    lease_key = first_cache._lease_key(first_cache._cache_key(entry))
    fail_first.set()
    with pytest.raises(RuntimeError, match="provider failed"):
        await first_task
    assert redis.raw_value(lease_key) == b"second-0"

    finish_second.set()
    assert await second_task == "second"


@pytest.mark.parametrize("operation", ["lease_acquire", "ownership_write"])
async def test_redis_command_failures_remain_fail_open(operation: str) -> None:
    """Acquisition and ownership-write failures return the original loader value."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    redis.fail_operations.add(operation)

    assert await cache.get_or_load(entry, lambda: _completed("loaded")) == "loaded"
    assert redis.raw_value(cache._cache_key(entry)) is None


async def test_wait_read_failure_returns_the_waiter_loader_value() -> None:
    """A Redis failure during contention rechecks abandons coordination immediately."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    owner_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="owner")
    waiter_cache, _, _ = _redis_cache(clock, redis=redis, sleeper=sleeper, token_prefix="waiter")
    entry = _text_entry({"path": "/search.json"})
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner_loader() -> str | None:
        owner_started.set()
        await release_owner.wait()
        return "owner"

    owner_task = asyncio.create_task(owner_cache.get_or_load(entry, owner_loader))
    await owner_started.wait()
    waiter_task = asyncio.create_task(waiter_cache.get_or_load(entry, lambda: _completed("waiter")))
    await _wait_for_command(redis, "lease_acquire", 2)
    redis.fail_operations.add("read")

    sleeper.advance(0.1)
    await _settle()
    assert await waiter_task == "waiter"

    release_owner.set()
    assert await owner_task == "owner"


async def test_renewal_failure_leaves_the_loaded_value_uncached() -> None:
    """A failed renewal does not surface and prevents an unverified conditional fill."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    cache, redis, _ = _redis_cache(clock, sleeper=sleeper)
    entry = _text_entry({"path": "/search.json"})
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner_loader() -> str | None:
        owner_started.set()
        await release_owner.wait()
        return "owner"

    owner_task = asyncio.create_task(cache.get_or_load(entry, owner_loader))
    await owner_started.wait()
    await _settle()
    redis.fail_operations.add("lease_renew")
    sleeper.advance(10.0)
    await _settle()
    release_owner.set()

    assert await owner_task == "owner"
    assert redis.raw_value(cache._cache_key(entry)) is None


async def test_loader_error_is_preserved_and_releases_its_lease() -> None:
    """Loader errors escape unchanged, release their token, and allow a later fill."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})

    async def fail() -> str | None:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await cache.get_or_load(entry, fail)
    assert redis.keys == ()

    assert await cache.get_or_load(entry, lambda: _completed("recovered")) == "recovered"
    assert redis.raw_value(cache._cache_key(entry)) is not None


async def test_blocked_redis_command_respects_the_bounded_command_timeout() -> None:
    """A blocked cache read becomes a normal loader result when its command bound expires."""
    clock = ManualClock()
    policy = _CachePolicy(command_timeout_seconds=0.01)
    cache, redis, _ = _redis_cache(clock, policy=policy)
    redis.block("read")
    entry = _text_entry({"path": "/search.json"})

    assert await cache.get_or_load(entry, lambda: _completed("loaded")) == "loaded"
    assert redis.command_counts["read"] == 1
    assert _CachePolicy().command_timeout_seconds == 1.0


async def test_oversize_envelopes_are_returned_without_being_cached() -> None:
    """Values whose complete envelopes exceed one MiB are recomputed on every request."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    value = "x" * _CachePolicy().maximum_envelope_bytes
    calls = 0

    async def load() -> str | None:
        nonlocal calls
        calls += 1
        return value

    assert await cache.get_or_load(entry, load) == value
    assert await cache.get_or_load(entry, load) == value
    assert calls == 2
    assert redis.raw_value(cache._cache_key(entry)) is None


async def test_circuit_bypasses_then_recovers_through_one_probe() -> None:
    """One failure opens the circuit, a failed probe reopens it, and one probe recovers."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)
    entry = _text_entry({"path": "/search.json"})
    redis.fail_operations.add("read")

    assert await cache.get_or_load(entry, lambda: _completed("first")) == "first"
    assert redis.command_counts["read"] == 1
    redis.fail_operations.clear()
    assert await cache.get_or_load(entry, lambda: _completed("bypassed")) == "bypassed"
    assert redis.command_counts["read"] == 1

    clock.advance(30.0)
    redis.fail_operations.add("read")
    assert await cache.get_or_load(entry, lambda: _completed("failed-probe")) == "failed-probe"
    assert redis.command_counts["read"] == 2
    redis.fail_operations.clear()
    assert await cache.get_or_load(entry, lambda: _completed("still-bypassed")) == "still-bypassed"
    assert redis.command_counts["read"] == 2

    clock.advance(30.0)
    redis.block("read")
    probe_task = asyncio.create_task(cache.get_or_load(entry, lambda: _completed("recovered")))
    await _wait_for_command(redis, "read", 3)
    bypass_task = asyncio.create_task(
        cache.get_or_load(entry, lambda: _completed("concurrent-bypass"))
    )
    assert await bypass_task == "concurrent-bypass"
    assert redis.command_counts["read"] == 3
    redis.unblock("read")
    assert await probe_task == "recovered"
    assert await cache.get_or_load(entry, lambda: _completed("unexpected")) == "recovered"


async def test_aclose_cancels_renewal_before_closing_redis() -> None:
    """Shutdown drains a live renewal task and prevents it from extending an owned lease."""
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    cache, redis, _ = _redis_cache(clock, sleeper=sleeper)
    entry = _text_entry({"path": "/search.json"})
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner_loader() -> str | None:
        owner_started.set()
        await release_owner.wait()
        return "owner"

    owner_task = asyncio.create_task(cache.get_or_load(entry, owner_loader))
    await owner_started.wait()
    await _settle()
    lease_key = cache._lease_key(cache._cache_key(entry))

    await cache.aclose()
    sleeper.advance(10.0)
    await _settle()
    assert redis.close_calls == 1
    assert redis.command_counts["lease_renew"] == 0
    assert redis.raw_value(lease_key) == b"token-0"

    release_owner.set()
    assert await owner_task == "owner"


async def test_cache_close_is_idempotent() -> None:
    """Repeated cache shutdown closes the owned client once."""
    clock = ManualClock()
    cache, redis, _ = _redis_cache(clock)

    await cache.aclose()
    await cache.aclose()

    assert redis.close_calls == 1


async def test_cache_events_are_structured_and_private(caplog: pytest.LogCaptureFixture) -> None:
    """All cache event categories avoid identities, values, keys, and lease tokens."""
    caplog.set_level(logging.INFO, logger="reelio.cache.redis")
    clock = ManualClock()
    sleeper = ManualSleeper(clock)
    redis = FakeRedis(clock)
    cache, _, _ = _redis_cache(
        clock,
        redis=redis,
        sleeper=sleeper,
        token_prefix="PRIVATE-TOKEN",
    )
    base_entry = _text_entry({"private": "PRIVATE-IDENTITY"}, wait_timeout_seconds=0.1)

    assert (
        await cache.get_or_load(base_entry, lambda: _completed("PRIVATE-VALUE")) == "PRIVATE-VALUE"
    )
    assert await cache.get_or_load(base_entry, lambda: _completed("unexpected")) == "PRIVATE-VALUE"

    hold_owner = asyncio.Event()
    owner_started = asyncio.Event()

    async def blocked_loader() -> str | None:
        owner_started.set()
        await hold_owner.wait()
        return "owner"

    wait_entry = _text_entry({"private": "wait"}, wait_timeout_seconds=0.1)
    owner_task = asyncio.create_task(cache.get_or_load(wait_entry, blocked_loader))
    await owner_started.wait()
    waiter_task = asyncio.create_task(cache.get_or_load(wait_entry, lambda: _completed("waiter")))
    await _wait_for_command(redis, "lease_acquire", 3)
    sleeper.advance(0.1)
    await _settle()
    assert await waiter_task == "waiter"
    hold_owner.set()
    assert await owner_task == "owner"

    corrupt_entry = _text_entry({"private": "corrupt"})
    await redis.put_raw(cache._cache_key(corrupt_entry), b"{")
    assert await cache.get_or_load(corrupt_entry, lambda: _completed("fresh")) == "fresh"

    oversize_entry = _text_entry({"private": "oversize"})
    oversized_value = "x" * _CachePolicy().maximum_envelope_bytes
    assert (
        await cache.get_or_load(oversize_entry, lambda: _completed(oversized_value))
        == oversized_value
    )

    lease_loss_entry = _text_entry({"private": "lease-loss"})
    release_loss_owner = asyncio.Event()
    loss_owner_started = asyncio.Event()

    async def lease_loss_loader() -> str | None:
        loss_owner_started.set()
        await release_loss_owner.wait()
        return "old"

    lease_loss_task = asyncio.create_task(cache.get_or_load(lease_loss_entry, lease_loss_loader))
    await loss_owner_started.wait()
    lease_key = cache._lease_key(cache._cache_key(lease_loss_entry))
    await redis.set(lease_key, "replacement", px=30_000)
    release_loss_owner.set()
    assert await lease_loss_task == "old"

    redis.fail_operations.add("read")
    failed_entry = _text_entry({"private": "failure"})
    assert await cache.get_or_load(failed_entry, lambda: _completed("failed-open")) == "failed-open"
    redis.fail_operations.clear()
    clock.advance(30.0)
    recovered_entry = _text_entry({"private": "recovered"})
    assert await cache.get_or_load(recovered_entry, lambda: _completed("recovered")) == "recovered"

    event_names = {
        cast(str, record.__dict__["cache_event"])
        for record in caplog.records
        if record.name == "reelio.cache.redis"
    }
    assert {
        "hit",
        "miss",
        "fill",
        "wait",
        "corruption",
        "oversize_skip",
        "fail_open",
        "lease_loss",
        "circuit_open",
        "circuit_probe",
        "circuit_closed",
    } <= event_names
    record_text = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    for private_marker in (
        "PRIVATE-IDENTITY",
        "PRIVATE-VALUE",
        "PRIVATE-TOKEN",
        "replacement",
        "reelio:local:provider:open-library:v1:",
    ):
        assert private_marker not in record_text


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
