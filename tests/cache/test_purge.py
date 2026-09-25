"""Behavioral contracts for fail-closed provider cache purges."""

import logging
from collections.abc import Callable
from typing import cast

import pytest

import reelio.cache.cli as cli_module
from reelio.cache import CacheCodecError, CacheEntry, CacheWrite, RedisCache, RevalidatingCacheEntry
from reelio.cache.config import CacheConfig
from reelio.cache.interface import JsonObject, RetainedCacheValue
from reelio.cache.purge import (
    ProviderPurgeError,
    PurgeNamespaceCategory,
    PurgeProvider,
    RedisProviderPurger,
)
from reelio.cache.redis import _CacheRuntime
from reelio.config import Environment
from tests.cache.fakes import FakeRedis, ManualClock

type _Descriptor = CacheEntry[str] | RevalidatingCacheEntry[str]


class _TextCodec:
    """Encode and validate the small string values used by purge tests."""

    version = "purge-text-v1"

    def encode(self, value: str) -> JsonObject:
        """Encode one text value into an exact JSON payload."""
        return {"text": value}

    def decode(self, payload: JsonObject) -> str:
        """Decode one exact text payload.

        Raises:
            CacheCodecError: If the payload has an unexpected shape.
        """
        value = payload.get("text")
        if set(payload) != {"text"} or not isinstance(value, str):
            raise CacheCodecError("Expected text payload")
        return value


class _FailingLaterScanRedis(FakeRedis):
    """Inject a scan failure after the first provider key was unlinked."""

    async def unlink(self, *names: bytes | str) -> int:
        """Delete one page, then fail the following scan command."""
        deleted_count = await super().unlink(*names)
        self.fail_operations.add("scan")
        return deleted_count


class _RecordingPurger:
    """Record command invocation and close behavior without a Redis connection."""

    def __init__(self, error: ProviderPurgeError | None = None) -> None:
        """Initialize an optional safe purge failure."""
        self.error = error
        self.providers: list[PurgeProvider] = []
        self.close_calls = 0

    async def purge(self, provider: PurgeProvider) -> None:
        """Record the selected provider or raise the configured safe failure."""
        self.providers.append(provider)
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        """Record one command cleanup call."""
        self.close_calls += 1


def _cache(redis: FakeRedis, clock: ManualClock) -> RedisCache:
    """Create one Redis cache using the deterministic fake client."""
    return RedisCache(
        redis,
        "reelio:local",
        b"PRIVATE-CREDENTIAL",
        runtime=_CacheRuntime(clock, _sleep, _token),
    )


def _token() -> str:
    """Return a fixed lease token that must never reach purge telemetry."""
    return "PRIVATE-TOKEN"


async def _sleep(_: float) -> None:
    """Yield once for cache paths that need a controllable sleeper."""
    return None


def _ordinary_entry(layer: str, identity: JsonObject) -> CacheEntry[str]:
    """Create one ordinary cache descriptor in a visible purge layer."""
    return CacheEntry(
        layer=layer,
        key_version="v1",
        identity=identity,
        codec=_TextCodec(),
        ttl_seconds=lambda _: 60,
        wait_timeout_seconds=1.0,
    )


def _revalidating_entry(layer: str, identity: JsonObject) -> RevalidatingCacheEntry[str]:
    """Create one revalidating descriptor in a visible provider layer."""
    return RevalidatingCacheEntry(
        layer=layer,
        key_version="v1",
        identity=identity,
        codec=_TextCodec(),
        wait_timeout_seconds=1.0,
    )


def _catalog_entries(provider: PurgeProvider, count: int) -> list[_Descriptor]:
    """Create current catalog-layer descriptor shapes for one selected provider."""
    layer = f"provider:{provider.value}"
    entries: list[_Descriptor] = []
    for index in range(count):
        identity: JsonObject = {
            "operation": "search",
            "query": f"PRIVATE-QUERY-{provider.value}-{index}",
            "market": "US",
        }
        entry = (
            _revalidating_entry(layer, identity)
            if provider is PurgeProvider.SPOTIFY
            else _ordinary_entry(layer, identity)
        )
        entries.append(entry)
    return entries


def _source_entries(count_per_layer: int) -> list[CacheEntry[str]]:
    """Create Source alias, metadata, Transcript, and interpretation descriptors."""
    entries: list[CacheEntry[str]] = []
    layers = (
        "source:alias",
        "source:metadata",
        "source:transcript",
        "source:interpretation",
    )
    for layer in layers:
        for index in range(count_per_layer):
            identity: JsonObject = {
                "platform": "youtube" if index % 2 == 0 else "tiktok",
                "llm_provider": "openai" if index % 2 == 0 else "deepseek",
                "source_url": f"https://PRIVATE-URL.invalid/{layer}/{index}",
                "transcript": f"PRIVATE-TRANSCRIPT-{layer}-{index}",
            }
            entries.append(_ordinary_entry(layer, identity))
    return entries


async def _store(cache: RedisCache, entry: _Descriptor, value: str) -> str:
    """Populate an ordinary or revalidating descriptor through its public cache seam."""
    if isinstance(entry, CacheEntry):
        return await cache.get_or_load(entry, lambda: _completed(value))

    async def load(
        retained_value: RetainedCacheValue[str] | None,
    ) -> CacheWrite[str]:
        del retained_value
        return CacheWrite(value, freshness_seconds=60, retention_seconds=60)

    return await cache.get_or_load_revalidating(entry, load)


async def _assert_hit(cache: RedisCache, entry: _Descriptor, expected_value: str) -> None:
    """Assert a descriptor is retained and does not invoke a replacement loader."""
    loader_calls = 0

    async def load() -> str:
        nonlocal loader_calls
        loader_calls += 1
        return "unexpected"

    if isinstance(entry, CacheEntry):
        result = await cache.get_or_load(entry, load)
    else:

        async def revalidating_load(
            retained_value: RetainedCacheValue[str] | None,
        ) -> CacheWrite[str]:
            del retained_value
            nonlocal loader_calls
            loader_calls += 1
            return CacheWrite("unexpected", freshness_seconds=60, retention_seconds=60)

        result = await cache.get_or_load_revalidating(entry, revalidating_load)
    assert result == expected_value
    assert loader_calls == 0


async def _assert_miss(cache: RedisCache, entry: _Descriptor, replacement: str) -> None:
    """Assert a descriptor invokes its loader after the matching purge."""
    loader_calls = 0

    async def load() -> str:
        nonlocal loader_calls
        loader_calls += 1
        return replacement

    if isinstance(entry, CacheEntry):
        result = await cache.get_or_load(entry, load)
    else:

        async def revalidating_load(
            retained_value: RetainedCacheValue[str] | None,
        ) -> CacheWrite[str]:
            del retained_value
            nonlocal loader_calls
            loader_calls += 1
            return CacheWrite(replacement, freshness_seconds=60, retention_seconds=60)

        result = await cache.get_or_load_revalidating(entry, revalidating_load)
    assert result == replacement
    assert loader_calls == 1


async def _store_lease(redis: FakeRedis, cache: RedisCache, entry: _Descriptor) -> None:
    """Seed a matching coordination lease so purge must remove write authority."""
    cache_key = cache._cache_key(entry)
    assert await redis.set(cache._lease_key(cache_key), "PRIVATE-TOKEN", px=60_000)


async def _completed(value: str) -> str:
    """Return one value from an already-completed loader."""
    return value


@pytest.mark.parametrize(
    "provider",
    [PurgeProvider.SPOTIFY, PurgeProvider.TMDB, PurgeProvider.OPEN_LIBRARY],
)
async def test_catalog_purge_removes_only_selected_provider_data_and_leases(
    provider: PurgeProvider,
) -> None:
    """Purge one multi-page catalog layer without disturbing other cache layers."""
    clock = ManualClock()
    redis = FakeRedis(clock)
    cache = _cache(redis, clock)
    selected_entries = _catalog_entries(provider, 3)
    other_providers = tuple(
        candidate
        for candidate in (
            PurgeProvider.SPOTIFY,
            PurgeProvider.TMDB,
            PurgeProvider.OPEN_LIBRARY,
        )
        if candidate is not provider
    )
    preserved_catalog_entries: list[_Descriptor] = [
        entry for candidate in other_providers for entry in _catalog_entries(candidate, 1)
    ]
    preserved_source_entries = _source_entries(1)
    preserved_entries = preserved_catalog_entries.copy()
    preserved_entries.extend(preserved_source_entries)

    for index, entry in enumerate(selected_entries):
        assert await _store(cache, entry, f"selected-{index}") == f"selected-{index}"
        await _store_lease(redis, cache, entry)
    for index, entry in enumerate(preserved_entries):
        assert await _store(cache, entry, f"preserved-{index}") == f"preserved-{index}"

    result = await RedisProviderPurger(
        redis,
        "reelio:local",
        Environment.LOCAL,
        scan_count=1,
    ).purge(provider)

    assert result.namespaces[0].namespace_category is PurgeNamespaceCategory.EXACT_OPERATION
    assert result.namespaces[0].deleted_count == 6
    assert redis.command_counts["scan"] > 1
    assert redis.command_counts["unlink"] >= 1
    for entry in selected_entries:
        assert cache._cache_key(entry) not in redis.keys
        assert cache._lease_key(cache._cache_key(entry)) not in redis.keys
        await _assert_miss(cache, entry, "selected-reloaded")
    for index, entry in enumerate(preserved_entries):
        await _assert_hit(cache, entry, f"preserved-{index}")


async def test_source_purge_removes_all_source_categories_and_preserves_catalog_entries() -> None:
    """Purge every Source-derived layer across platform and LLM identities."""
    clock = ManualClock()
    redis = FakeRedis(clock)
    cache = _cache(redis, clock)
    source_entries = _source_entries(2)
    catalog_entries: list[_Descriptor] = [
        entry
        for provider in (
            PurgeProvider.SPOTIFY,
            PurgeProvider.TMDB,
            PurgeProvider.OPEN_LIBRARY,
        )
        for entry in _catalog_entries(provider, 1)
    ]

    for index, source_entry in enumerate(source_entries):
        assert await _store(cache, source_entry, f"source-{index}") == f"source-{index}"
        await _store_lease(redis, cache, source_entry)
    for index, catalog_entry in enumerate(catalog_entries):
        assert await _store(cache, catalog_entry, f"catalog-{index}") == f"catalog-{index}"

    result = await RedisProviderPurger(
        redis,
        "reelio:local",
        Environment.LOCAL,
        scan_count=1,
    ).purge(PurgeProvider.SOURCE)

    assert tuple(item.namespace_category for item in result.namespaces) == (
        PurgeNamespaceCategory.SUBMITTED_URL_ALIAS,
        PurgeNamespaceCategory.SOURCE_METADATA,
        PurgeNamespaceCategory.TRANSCRIPT,
        PurgeNamespaceCategory.MENTION_INTERPRETATION,
    )
    assert tuple(item.deleted_count for item in result.namespaces) == (4, 4, 4, 4)
    for source_entry in source_entries:
        assert cache._cache_key(source_entry) not in redis.keys
        assert cache._lease_key(cache._cache_key(source_entry)) not in redis.keys
        await _assert_miss(cache, source_entry, "source-reloaded")
    for index, catalog_entry in enumerate(catalog_entries):
        await _assert_hit(cache, catalog_entry, f"catalog-{index}")


async def test_purge_reports_zero_for_empty_categories_and_advances_empty_pages() -> None:
    """Return empty success and continue after a nonterminal empty scan page."""
    clock = ManualClock()
    redis = FakeRedis(clock)
    empty_purger = RedisProviderPurger(redis, "reelio:local", Environment.LOCAL)

    empty_catalog = await empty_purger.purge(PurgeProvider.TMDB)
    empty_source = await empty_purger.purge(PurgeProvider.SOURCE)

    assert empty_catalog.namespaces[0].deleted_count == 0
    assert all(item.deleted_count == 0 for item in empty_source.namespaces)
    assert redis.command_counts["unlink"] == 0

    cache = _cache(redis, clock)
    entries = _catalog_entries(PurgeProvider.TMDB, 2)
    for index, entry in enumerate(entries):
        await _store(cache, entry, f"value-{index}")
    redis.empty_scan_page_indexes.add(0)

    result = await RedisProviderPurger(
        redis,
        "reelio:local",
        Environment.LOCAL,
        scan_count=1,
    ).purge(PurgeProvider.TMDB)

    assert result.namespaces[0].deleted_count == 2
    assert redis.command_counts["scan"] >= 4
    assert all(cache._cache_key(entry) not in redis.keys for entry in entries)


async def test_purge_failure_reports_safe_partial_count_without_false_success() -> None:
    """Return fixed errors for injected scan and unlink failures after partial work."""
    clock = ManualClock()
    scan_redis = FakeRedis(clock)
    scan_redis.fail_operations.add("scan")
    scan_purger = RedisProviderPurger(scan_redis, "reelio:local", Environment.LOCAL)

    with pytest.raises(ProviderPurgeError) as scan_error:
        await scan_purger.purge(PurgeProvider.SPOTIFY)

    assert str(scan_error.value) == "Provider cache purge did not complete."
    assert scan_error.value.provider is PurgeProvider.SPOTIFY
    assert scan_error.value.environment is Environment.LOCAL
    assert scan_error.value.namespace_category is PurgeNamespaceCategory.EXACT_OPERATION
    assert scan_error.value.deleted_count == 0

    unlink_redis = FakeRedis(clock)
    unlink_redis.fail_operations.add("unlink")
    unlink_cache = _cache(unlink_redis, clock)
    unlink_entry = _catalog_entries(PurgeProvider.TMDB, 1)[0]
    await _store(unlink_cache, unlink_entry, "value")
    unlink_purger = RedisProviderPurger(unlink_redis, "reelio:local", Environment.LOCAL)

    with pytest.raises(ProviderPurgeError) as unlink_error:
        await unlink_purger.purge(PurgeProvider.TMDB)

    assert unlink_error.value.namespace_category is PurgeNamespaceCategory.EXACT_OPERATION
    assert unlink_error.value.deleted_count == 0

    partial_redis = _FailingLaterScanRedis(clock)
    partial_cache = _cache(partial_redis, clock)
    for index, entry in enumerate(_catalog_entries(PurgeProvider.OPEN_LIBRARY, 2)):
        await _store(partial_cache, entry, f"value-{index}")
    partial_purger = RedisProviderPurger(
        partial_redis,
        "reelio:local",
        Environment.LOCAL,
        scan_count=1,
    )

    with pytest.raises(ProviderPurgeError) as partial_error:
        await partial_purger.purge(PurgeProvider.OPEN_LIBRARY)

    assert partial_error.value.namespace_category is PurgeNamespaceCategory.EXACT_OPERATION
    assert partial_error.value.deleted_count == 1


async def test_purge_events_exclude_private_cache_material(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit only provider, environment, category, duration, outcome, and count."""
    caplog.set_level(logging.INFO, logger="reelio.cache.purge")
    clock = ManualClock()
    redis = FakeRedis(clock)
    cache = _cache(redis, clock)
    private_entry = _ordinary_entry(
        "provider:spotify",
        {
            "credential": "PRIVATE-CREDENTIAL",
            "digest": "PRIVATE-DIGEST",
            "url": "https://PRIVATE-URL.invalid/?query=PRIVATE-QUERY",
            "transcript": "PRIVATE-TRANSCRIPT",
        },
    )
    assert await _store(cache, private_entry, "PRIVATE-PAYLOAD") == "PRIVATE-PAYLOAD"
    await redis.put_raw(
        "reelio:local:provider:spotify:v1:PRIVATE-DIGEST",
        b"PRIVATE-PAYLOAD",
    )

    await RedisProviderPurger(redis, "reelio:local", Environment.LOCAL).purge(PurgeProvider.SPOTIFY)

    records = [record for record in caplog.records if record.name == "reelio.cache.purge"]
    assert len(records) == 1
    assert {
        key
        for key in records[0].__dict__
        if key
        in {
            "provider",
            "environment",
            "namespace_category",
            "duration_ms",
            "outcome",
            "deleted_count",
        }
    } == {
        "provider",
        "environment",
        "namespace_category",
        "duration_ms",
        "outcome",
        "deleted_count",
    }
    record_text = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in records)
    for private_marker in (
        "PRIVATE-CREDENTIAL",
        "PRIVATE-DIGEST",
        "PRIVATE-URL",
        "PRIVATE-QUERY",
        "PRIVATE-TRANSCRIPT",
        "PRIVATE-PAYLOAD",
    ):
        assert private_marker not in record_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--environment", "local"],
        ["--provider", "source"],
    ],
)
def test_cli_requires_each_flag_before_constructing_settings_or_client(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """Reject either omitted required flag before settings or Redis construction."""
    constructed: list[str] = []

    def fail_settings(**values: object) -> object:
        constructed.append("settings")
        raise AssertionError(values)

    def fail_client(_: object) -> object:
        constructed.append("client")
        raise AssertionError("Redis client must not be constructed")

    monkeypatch.setattr(cli_module, "CacheConfig", fail_settings)
    monkeypatch.setattr(cli_module, "create_provider_purger", fail_client)

    with pytest.raises(SystemExit) as error:
        cli_module.main(argv)

    assert error.value.code == 2
    assert constructed == []


def test_cli_uses_explicit_environment_and_closes_successful_purger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override process environment for namespace validation and close successful work."""
    _use_environment_only_cache_config(monkeypatch)
    monkeypatch.setenv("REELIO_ENVIRONMENT", "production")
    monkeypatch.setenv("REELIO_CACHE_REDIS_URL", "redis://localhost:6399/15")
    monkeypatch.setenv("REELIO_CACHE_KEY_SECRET", "PRIVATE-CREDENTIAL")
    purger = _RecordingPurger()
    seen_environments: list[Environment] = []

    def create_purger(settings: object) -> _RecordingPurger:
        seen_environments.append(cast(CacheConfig, settings).environment)
        return purger

    monkeypatch.setattr(cli_module, "create_provider_purger", create_purger)

    assert cli_module.main(["--provider", "source", "--environment", "local"]) == 0
    assert seen_environments == [Environment.LOCAL]
    assert purger.providers == [PurgeProvider.SOURCE]
    assert purger.close_calls == 1


def test_cli_sanitizes_configuration_and_redis_failure_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return failure without credentials, URLs, query text, or Redis exception detail."""
    _use_environment_only_cache_config(monkeypatch)
    monkeypatch.setenv(
        "REELIO_CACHE_REDIS_URL",
        "redis://PRIVATE-CREDENTIAL@localhost:6399/15?query=PRIVATE-QUERY",
    )
    monkeypatch.setenv("REELIO_CACHE_KEY_SECRET", "  ")

    assert cli_module.main(["--provider", "source", "--environment", "local"]) == 1

    configuration_output = capsys.readouterr().err
    assert "Provider cache purge configuration is invalid." in configuration_output
    assert "PRIVATE-CREDENTIAL" not in configuration_output
    assert "PRIVATE-QUERY" not in configuration_output

    monkeypatch.setenv("REELIO_CACHE_KEY_SECRET", "PRIVATE-CREDENTIAL")
    unavailable_error = ProviderPurgeError(
        PurgeProvider.SOURCE,
        Environment.LOCAL,
        PurgeNamespaceCategory.TRANSCRIPT,
        3,
    )
    failing_purger = _RecordingPurger(unavailable_error)
    monkeypatch.setattr(cli_module, "create_provider_purger", lambda _: failing_purger)

    assert cli_module.main(["--provider", "source", "--environment", "local"]) == 1

    unavailable_output = capsys.readouterr().err
    assert (
        "Provider cache purge did not complete. Check Redis availability and retry."
        in unavailable_output
    )
    assert "PRIVATE-CREDENTIAL" not in unavailable_output
    assert "PRIVATE-QUERY" not in unavailable_output
    assert failing_purger.close_calls == 1


def _use_environment_only_cache_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent CLI cache settings from reading a developer-local dotenv file."""
    constructor = cast(Callable[..., CacheConfig], CacheConfig)

    def construct_cache_config(**values: object) -> CacheConfig:
        return constructor(_env_file=None, **values)

    monkeypatch.setattr(cli_module, "CacheConfig", construct_cache_config)
