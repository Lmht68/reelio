"""Real Redis-compatible cache, purge, outage, and lifespan smoke coverage."""

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from multiprocessing import get_context
from types import SimpleNamespace
from typing import Protocol, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from redis.asyncio import Redis

import reelio.main as main_module
from reelio.cache import (
    CacheCodecError,
    CacheConfig,
    CacheEntry,
    DisabledCache,
    RedisCache,
    create_cache,
)
from reelio.cache.interface import JsonObject
from reelio.cache.purge import PurgeProvider, RedisProviderPurger, _PurgeRedisClient
from reelio.cache.redis import (
    _ATOMIC_READ_SCRIPT,
    _CORRUPTION_DELETE_SCRIPT,
    _LEASE_FILL_SCRIPT,
    _LEASE_RELEASE_SCRIPT,
    _LEASE_RENEW_SCRIPT,
    _OWNED_DISCARD_SCRIPT,
    _RedisClient,
)
from reelio.config import Environment
from reelio.extraction.market import SpotifyMarket

pytestmark = pytest.mark.redis_smoke


class _TextCodec:
    """Encode the simple string values persisted by real Redis smoke tests."""

    version = "redis-smoke-text-v1"

    def encode(self, value: str) -> JsonObject:
        """Encode one text value into its exact JSON payload."""
        return {"text": value}

    def decode(self, payload: JsonObject) -> str:
        """Decode an exact text payload.

        Raises:
            CacheCodecError: If the payload structure is not valid.
        """
        value = payload.get("text")
        if set(payload) != {"text"} or not isinstance(value, str):
            raise CacheCodecError("Expected text payload")
        return value


class _WorkerQueue(Protocol):
    """Describe the narrow spawned-worker result queue surface."""

    def put(self, value: tuple[str, int]) -> None:
        """Publish one worker result to the parent process."""
        ...


class _TracingRedisClient:
    """Record final client closure while delegating Redis operations unchanged."""

    def __init__(self, client: Redis, events: list[str]) -> None:
        """Initialize one transparent Redis client wrapper.

        Args:
            client: Real Redis client whose close operation is observed.
            events: Ordered lifecycle event sink.
        """
        self._client = client
        self._events = events
        self.close_calls = 0

    def __getattr__(self, name: str) -> object:
        """Delegate all non-lifecycle Redis operations to the wrapped client."""
        return getattr(self._client, name)

    async def aclose(self) -> None:
        """Record and close the wrapped Redis client once per cache request."""
        self.close_calls += 1
        self._events.append("client")
        await self._client.aclose()


class _LifecycleProvider:
    """Record provider teardown performed by the lightweight pipeline substitute."""

    def __init__(self, events: list[str]) -> None:
        """Initialize the lifecycle event sink."""
        self._events = events

    async def aclose(self) -> None:
        """Record provider shutdown."""
        self._events.append("provider")


class _LifecyclePipeline:
    """Close its provider before reporting pipeline teardown."""

    def __init__(self, provider: _LifecycleProvider, events: list[str]) -> None:
        """Initialize the provider and pipeline event sink."""
        self._provider = provider
        self._events = events

    async def aclose(self) -> None:
        """Close provider-owned resources before the pipeline completes shutdown."""
        await self._provider.aclose()
        self._events.append("pipeline")


def _redis_url_or_skip() -> str:
    """Return the disposable real Redis endpoint or skip this smoke scenario."""
    redis_url = os.environ.get("REELIO_TEST_REDIS_URL")
    if redis_url is None:
        pytest.skip("REELIO_TEST_REDIS_URL is not configured")
    assert redis_url is not None
    return redis_url


def _unavailable_redis_url_or_skip() -> str:
    """Return the bounded unavailable endpoint or skip the outage scenario."""
    redis_url = os.environ.get("REELIO_TEST_REDIS_UNAVAILABLE_URL")
    if redis_url is None:
        pytest.skip("REELIO_TEST_REDIS_UNAVAILABLE_URL is not configured")
    assert redis_url is not None
    return redis_url


def _smoke_settings(redis_url: str) -> CacheConfig:
    """Create enabled local cache settings for one disposable Redis endpoint."""
    return CacheConfig(
        enabled=True,
        environment=Environment.LOCAL,
        redis_url=SecretStr(redis_url),
        key_secret=SecretStr("redis-smoke-cache-secret"),
    )


def _smoke_namespace(settings: CacheConfig) -> str:
    """Create one UUID-isolated namespace beneath the validated cache namespace."""
    return f"{settings.namespace}:smoke:{uuid4().hex}"


def _entry(layer: str, identity: str) -> CacheEntry[str]:
    """Create one ordinary visible-layer cache entry with a stable smoke identity."""
    return CacheEntry(
        layer=layer,
        key_version="v1",
        identity={"identity": identity},
        codec=_TextCodec(),
        ttl_seconds=lambda _: 60,
        wait_timeout_seconds=1.0,
    )


def _redis_cache(client: Redis, namespace: str) -> RedisCache:
    """Create a real Redis cache that owns its client."""
    return RedisCache(
        cast(_RedisClient, client),
        namespace,
        b"redis-smoke-cache-secret",
    )


def _purger(client: Redis, namespace: str) -> RedisProviderPurger:
    """Create a provider purger owning a dedicated real Redis client."""
    return RedisProviderPurger(
        cast(_PurgeRedisClient, client),
        namespace,
        Environment.LOCAL,
    )


async def _cleanup_namespace(client: Redis, namespace: str) -> None:
    """Incrementally unlink all UUID-scoped smoke keys without KEYS or flushes."""
    cursor = 0
    while True:
        cursor, keys = await client.scan(cursor=cursor, match=f"{namespace}:*", count=100)
        if keys:
            await client.unlink(*keys)
        if cursor == 0:
            return


async def _load(cache: RedisCache, entry: CacheEntry[str], value: str) -> tuple[str, int]:
    """Load one entry while reporting whether its loader executed."""
    loader_calls = 0

    async def loader() -> str:
        nonlocal loader_calls
        loader_calls += 1
        return value

    return await cache.get_or_load(entry, loader), loader_calls


def _restart_cache_worker(
    redis_url: str,
    namespace: str,
    expect_cached: bool,
    output_queue: _WorkerQueue,
) -> None:
    """Fill or reuse one cache value from a spawned process.

    Args:
        redis_url: Disposable Redis server endpoint shared by both workers.
        namespace: UUID-isolated cache namespace shared by both workers.
        expect_cached: Whether invoking the loader indicates an error.
        output_queue: Parent-owned queue receiving value and loader invocation count.
    """

    async def run() -> tuple[str, int]:
        client = Redis.from_url(redis_url, decode_responses=False)
        cache = _redis_cache(client, namespace)
        loader_calls = 0

        async def loader() -> str:
            nonlocal loader_calls
            loader_calls += 1
            if expect_cached:
                raise AssertionError("The restarted cache loader must not run")
            return "retained"

        try:
            return await cache.get_or_load(
                _entry("provider:spotify", "restart"), loader
            ), loader_calls
        finally:
            await cache.aclose()

    try:
        result = asyncio.run(run())
    except Exception:
        output_queue.put(("worker-error", -1))
        return
    output_queue.put(result)


async def test_redis_lease_scripts_enforce_token_ownership() -> None:
    """Exercise acquire, expiry, renewal, conditional fill, and release against Redis."""
    redis_url = _redis_url_or_skip()
    client = Redis.from_url(redis_url, decode_responses=False)
    key_prefix = f"reelio:smoke:{uuid4().hex}"
    data_key = f"{key_prefix}:data"
    lease_key = f"{key_prefix}:lease"
    abandoned_lease_key = f"{key_prefix}:abandoned"
    release_lease_key = f"{key_prefix}:release"
    retained_data_key = f"{key_prefix}:retained"
    retained_lease_key = f"{key_prefix}:retained-lease"
    owner_token = "owner-token"
    other_token = "other-token"
    payload = b'{"value":"fresh"}'
    renew = client.register_script(_LEASE_RENEW_SCRIPT)
    release = client.register_script(_LEASE_RELEASE_SCRIPT)
    fill = client.register_script(_LEASE_FILL_SCRIPT)
    delete_corrupt = client.register_script(_CORRUPTION_DELETE_SCRIPT)
    discard = client.register_script(_OWNED_DISCARD_SCRIPT)
    atomic_read = client.register_script(_ATOMIC_READ_SCRIPT)

    try:
        assert await atomic_read(keys=[data_key], args=[]) == [0, -2]
        assert await client.set(data_key, payload, ex=60) is True
        atomic_result = await atomic_read(keys=[data_key], args=[])
        assert atomic_result[0:2] == [1, payload]
        assert isinstance(atomic_result[2], int)
        assert 0 <= atomic_result[2] <= 60_000
        await client.unlink(data_key)

        assert await client.set(lease_key, owner_token, nx=True, px=200) is True
        assert await client.set(lease_key, other_token, nx=True, px=200) is None

        assert await client.set(abandoned_lease_key, owner_token, nx=True, px=50) is True
        await asyncio.sleep(0.075)
        assert await client.get(abandoned_lease_key) is None

        await asyncio.sleep(0.12)
        assert await renew(keys=[lease_key], args=[owner_token, 200]) == 1
        assert await renew(keys=[lease_key], args=[other_token, 200]) == 0
        await asyncio.sleep(0.11)
        assert await client.get(lease_key) == owner_token.encode()

        assert await fill(keys=[data_key, lease_key], args=[other_token, payload, 60]) == 0
        assert await client.get(data_key) is None
        assert await client.get(lease_key) == owner_token.encode()
        assert await fill(keys=[data_key, lease_key], args=[owner_token, payload, 60]) == 1
        assert await client.get(data_key) == payload
        assert await client.get(lease_key) is None

        assert await client.set(release_lease_key, owner_token, nx=True, px=200) is True
        assert await release(keys=[release_lease_key], args=[other_token]) == 0
        assert await client.get(release_lease_key) == owner_token.encode()
        assert await release(keys=[release_lease_key], args=[owner_token]) == 1
        assert await client.get(release_lease_key) is None

        assert await client.set(data_key, b"corrupt", ex=60) is True
        assert await delete_corrupt(keys=[data_key], args=[b"different"]) == 0
        assert await client.get(data_key) == b"corrupt"
        assert await delete_corrupt(keys=[data_key], args=[b"corrupt"]) == 1
        assert await client.get(data_key) is None

        assert await client.set(retained_data_key, payload, ex=60) is True
        assert await client.set(retained_lease_key, owner_token, nx=True, px=200) is True
        assert await discard(keys=[retained_data_key, retained_lease_key], args=[other_token]) == 0
        assert await client.get(retained_data_key) == payload
        assert await client.get(retained_lease_key) == owner_token.encode()
        assert await discard(keys=[retained_data_key, retained_lease_key], args=[owner_token]) == 1
        assert await client.get(retained_data_key) is None
        assert await client.get(retained_lease_key) is None
    finally:
        await client.unlink(
            data_key,
            lease_key,
            abandoned_lease_key,
            release_lease_key,
            retained_data_key,
            retained_lease_key,
        )
        await client.aclose()


async def test_real_provider_purge_isolates_layers_and_forces_post_purge_loads() -> None:
    """Purge live multi-page provider and Source keys without touching other layers."""
    redis_url = _redis_url_or_skip()
    settings = _smoke_settings(redis_url)
    namespace = _smoke_namespace(settings)
    cache_client = Redis.from_url(redis_url, decode_responses=False)
    spotify_purger_client = Redis.from_url(redis_url, decode_responses=False)
    source_purger_client = Redis.from_url(redis_url, decode_responses=False)
    cleanup_client = Redis.from_url(redis_url, decode_responses=False)
    cache = _redis_cache(cache_client, namespace)
    spotify_purger = _purger(spotify_purger_client, namespace)
    source_purger = _purger(source_purger_client, namespace)
    spotify_entries = [_entry("provider:spotify", f"spotify-{index}") for index in range(64)]
    source_entries = [
        _entry(layer, f"{layer}-{identity}")
        for layer in (
            "source:alias",
            "source:metadata",
            "source:transcript",
            "source:interpretation",
        )
        for identity in range(1)
    ]
    preserved_entries = [
        _entry("provider:tmdb", "preserved-tmdb"),
        _entry("provider:open-library", "preserved-open-library"),
    ]

    try:
        for index, entry in enumerate(spotify_entries):
            assert (await _load(cache, entry, f"spotify-{index}"))[1] == 1
            assert await cache_client.set(
                cache._lease_key(cache._cache_key(entry)),
                f"spotify-lease-{index}",
                px=60_000,
            )
        for index, entry in enumerate(source_entries):
            assert (await _load(cache, entry, f"source-{index}"))[1] == 1
            assert await cache_client.set(
                cache._lease_key(cache._cache_key(entry)),
                f"source-lease-{index}",
                px=60_000,
            )
        for index, entry in enumerate(preserved_entries):
            assert (await _load(cache, entry, f"preserved-{index}"))[1] == 1

        preflight_cursor, _ = await cache_client.scan(
            cursor=0,
            match=f"{namespace}:provider:spotify:*",
            count=1,
        )
        assert preflight_cursor != 0

        spotify_result = await spotify_purger.purge(PurgeProvider.SPOTIFY)
        source_result = await source_purger.purge(PurgeProvider.SOURCE)
        await spotify_purger.aclose()
        await source_purger.aclose()

        assert spotify_result.namespaces[0].deleted_count == 128
        assert tuple(result.deleted_count for result in source_result.namespaces) == (2, 2, 2, 2)
        for entry in spotify_entries + source_entries:
            cache_key = cache._cache_key(entry)
            assert await cleanup_client.get(cache_key) is None
            assert await cleanup_client.get(cache._lease_key(cache_key)) is None
        for entry in preserved_entries:
            assert await cleanup_client.get(cache._cache_key(entry)) is not None

        for index, entry in enumerate(spotify_entries + source_entries):
            value, loader_calls = await _load(cache, entry, f"reloaded-{index}")
            assert value == f"reloaded-{index}"
            assert loader_calls == 1
        for index, entry in enumerate(preserved_entries):
            value, loader_calls = await _load(cache, entry, f"preserved-{index}")
            assert value == f"preserved-{index}"
            assert loader_calls == 0
    finally:
        await _cleanup_namespace(cleanup_client, namespace)
        await cache.aclose()
        await spotify_purger.aclose()
        await source_purger.aclose()
        await cleanup_client.aclose()


async def test_real_cache_disabled_mode_and_enabled_reuse() -> None:
    """Avoid live contact while disabled, then reuse one live Redis value when enabled."""
    redis_url = _redis_url_or_skip()
    disabled_cache = create_cache(CacheConfig(enabled=False, environment=Environment.LOCAL))
    assert isinstance(disabled_cache, DisabledCache)
    await disabled_cache.aclose()

    settings = _smoke_settings(redis_url)
    namespace = _smoke_namespace(settings)
    client = Redis.from_url(redis_url, decode_responses=False)
    cleanup_client = Redis.from_url(redis_url, decode_responses=False)
    cache = _redis_cache(client, namespace)
    entry = _entry("provider:tmdb", "enabled-reuse")

    try:
        assert await _load(cache, entry, "retained") == ("retained", 1)
        assert await _load(cache, entry, "unexpected") == ("retained", 0)
    finally:
        await _cleanup_namespace(cleanup_client, namespace)
        await cache.aclose()
        await cleanup_client.aclose()


async def test_real_cache_reuses_value_after_spawned_process_restart() -> None:
    """Reuse a live value across two spawned processes sharing namespace and HMAC secret."""
    redis_url = _redis_url_or_skip()
    namespace = _smoke_namespace(_smoke_settings(redis_url))
    cleanup_client = Redis.from_url(redis_url, decode_responses=False)
    context = get_context("spawn")
    output_queue = context.Queue()

    try:
        first_worker = context.Process(
            target=_restart_cache_worker,
            args=(redis_url, namespace, False, output_queue),
        )
        first_worker.start()
        first_worker.join(timeout=10)
        assert first_worker.exitcode == 0
        assert cast(tuple[str, int], output_queue.get(timeout=10)) == ("retained", 1)

        second_worker = context.Process(
            target=_restart_cache_worker,
            args=(redis_url, namespace, True, output_queue),
        )
        second_worker.start()
        second_worker.join(timeout=10)
        assert second_worker.exitcode == 0
        assert cast(tuple[str, int], output_queue.get(timeout=10)) == ("retained", 0)
    finally:
        await _cleanup_namespace(cleanup_client, namespace)
        await cleanup_client.aclose()


async def test_real_cache_fails_open_when_redis_is_unavailable() -> None:
    """Return normal loader output when the configured Redis endpoint refuses connections."""
    unavailable_redis_url = _unavailable_redis_url_or_skip()
    namespace = _smoke_namespace(_smoke_settings(_redis_url_or_skip()))
    client = Redis.from_url(unavailable_redis_url, decode_responses=False)
    cache = _redis_cache(client, namespace)
    entry = _entry("provider:open-library", "unavailable")

    try:
        assert await _load(cache, entry, "loaded-without-redis") == ("loaded-without-redis", 1)
    finally:
        await cache.aclose()


async def test_real_lifespan_closes_pipeline_provider_before_cache_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close lightweight pipeline resources before exactly one real cache/client close."""
    redis_url = _redis_url_or_skip()
    settings = _smoke_settings(redis_url)
    namespace = _smoke_namespace(settings)
    events: list[str] = []
    raw_cache_client = Redis.from_url(redis_url, decode_responses=False)
    cleanup_client = Redis.from_url(redis_url, decode_responses=False)
    tracing_client = _TracingRedisClient(raw_cache_client, events)
    cache = _redis_cache(cast(Redis, tracing_client), namespace)
    original_cache_close = cache.aclose

    async def close_cache() -> None:
        events.append("cache")
        await original_cache_close()

    monkeypatch.setattr(cache, "aclose", close_cache)
    provider = _LifecycleProvider(events)
    pipeline = _LifecyclePipeline(provider, events)

    @asynccontextmanager
    async def catalog_context(
        settings: object,
        configured_cache: object,
    ) -> AsyncGenerator[object]:
        del settings, configured_cache
        yield object()

    async def create_pipeline(
        default_market: SpotifyMarket,
        spotify_catalog: object,
        configured_cache: RedisCache,
    ) -> _LifecyclePipeline:
        del default_market, spotify_catalog
        assert configured_cache is cache
        assert await _load(configured_cache, _entry("provider:spotify", "lifespan"), "ready") == (
            "ready",
            1,
        )
        return pipeline

    monkeypatch.setattr(main_module, "CacheConfig", lambda environment: settings)
    monkeypatch.setattr(main_module, "create_cache", lambda _: cache)
    monkeypatch.setattr(
        main_module,
        "SpotifyConfig",
        lambda: SimpleNamespace(default_market=SpotifyMarket("US")),
    )
    monkeypatch.setattr(main_module, "create_spotify_catalog", catalog_context)
    monkeypatch.setattr(main_module, "_create_production_pipeline", create_pipeline)
    application: FastAPI = main_module.create_app()

    try:
        async with application.router.lifespan_context(application):
            assert application.state.extraction_pipeline is pipeline

        assert events == ["provider", "pipeline", "cache", "client"]
        assert events.count("cache") == 1
        assert tracing_client.close_calls == 1
        assert not cache._renewal_tasks
    finally:
        await _cleanup_namespace(cleanup_client, namespace)
        await cache.aclose()
        await cleanup_client.aclose()
