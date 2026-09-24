"""Redis-backed implementation of the application-owned async cache boundary."""

import asyncio
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from reelio.cache.config import CacheConfig
from reelio.cache.disabled import DisabledCache
from reelio.cache.interface import AsyncCache, CacheCodecError, CacheEntry, JsonObject

logger = logging.getLogger(__name__)

_ENVELOPE_VERSION = 1
_LAYER_PATTERN = re.compile(r"[a-z][a-z0-9-]*(?::[a-z][a-z0-9-]*)*")
_KEY_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")

_LEASE_RENEW_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("PEXPIRE", KEYS[1], ARGV[2])
end
return 0
"""
_LEASE_RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""
_LEASE_FILL_SCRIPT = """
if redis.call("GET", KEYS[2]) == ARGV[1] then
    redis.call("SET", KEYS[1], ARGV[2], "EX", ARGV[3])
    return redis.call("DEL", KEYS[2])
end
return 0
"""
_CORRUPTION_DELETE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("UNLINK", KEYS[1])
end
return 0
"""


type _ScriptArgument = bytes | str | int


class _RedisScript(Protocol):
    """Describe the asynchronous callable returned by Redis script registration."""

    def __call__(
        self,
        *,
        keys: list[str],
        args: list[_ScriptArgument],
    ) -> Awaitable[object]:
        """Execute the registered script with positional Redis keys and arguments."""
        ...


class _RedisClient(Protocol):
    """Contain the minimal Redis command surface owned by this cache."""

    def get(self, name: str) -> Awaitable[bytes | str | None]:
        """Load one raw cache payload."""
        ...

    def set(
        self,
        name: str,
        value: bytes | str,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> Awaitable[bool | None]:
        """Store a value with optional expiry and not-exists semantics."""
        ...

    def register_script(self, script: str) -> _RedisScript:
        """Register a Lua script without eagerly connecting to Redis."""
        ...

    def aclose(self) -> Awaitable[None]:
        """Close the client and its owned pool."""
        ...


@dataclass(frozen=True, slots=True)
class _CachePolicy:
    """Contain fixed coordination and cache-isolation limits."""

    lease_ttl_milliseconds: int = 30_000
    renew_interval_seconds: float = 10.0
    circuit_open_seconds: float = 30.0
    command_timeout_seconds: float = 1.0
    wait_poll_seconds: float = 0.1
    maximum_envelope_bytes: int = 1_048_576


@dataclass(frozen=True, slots=True)
class _CacheRuntime:
    """Contain injected time, sleeping, and token behavior for cache coordination."""

    monotonic: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]
    token_factory: Callable[[], str]


@dataclass(frozen=True, slots=True)
class _Scripts:
    """Contain lazily registered token-owned Redis scripts."""

    renew: _RedisScript
    release: _RedisScript
    fill: _RedisScript
    delete_corrupt: _RedisScript


@dataclass(frozen=True, slots=True)
class _CacheRead[ValueT]:
    """Classify one raw data-key read without collapsing corruption into absence."""

    state: Literal["absent", "valid", "corrupt"]
    value: ValueT | None = None
    raw_payload: bytes | str | None = None
    corruption_reason: str | None = None


@dataclass(slots=True)
class _LeaseOwnership:
    """Track one token-owned lease across renewal and loader completion."""

    token: str
    lost: bool = False


class _CommandUnavailable:
    """Mark a Redis operation that failed or was bypassed by the circuit."""


class _UnownedLoad:
    """Mark a corruption race that must not proceed to an unverified cache write."""


_COMMAND_UNAVAILABLE = _CommandUnavailable()
_UNOWNED_LOAD = _UnownedLoad()
_PRODUCTION_POLICY = _CachePolicy()
_PRODUCTION_RUNTIME = _CacheRuntime(time.monotonic, asyncio.sleep, lambda: secrets.token_hex(32))


class RedisCache:
    """Coordinate token-owned shared cache fills while keeping cache failures local."""

    def __init__(
        self,
        client: _RedisClient,
        namespace: str,
        key_secret: bytes,
        *,
        policy: _CachePolicy | None = None,
        runtime: _CacheRuntime | None = None,
    ) -> None:
        """Initialize a cache around one client owned by this instance.

        Args:
            client: Redis client whose pool this cache closes.
            namespace: Fixed environment-derived visible key namespace.
            key_secret: Dedicated HMAC key for private operation identities.
            policy: Private coordination limits, primarily for deterministic tests.
            runtime: Private clock, sleeper, and token source, primarily for tests.
        """
        self._client = client
        self._namespace = namespace
        self._key_secret = key_secret
        self._policy = policy or _PRODUCTION_POLICY
        self._runtime = runtime or _PRODUCTION_RUNTIME
        self._close_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._circuit_lock = asyncio.Lock()
        self._closed = False
        self._renewal_tasks: set[asyncio.Task[None]] = set()
        self._scripts: _Scripts | None = None
        self._circuit_open_until: float | None = None
        self._circuit_probe_in_flight = False

    async def get_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Return a decoded cache hit or an uncoordinated loader result.

        Args:
            entry: Versioned operation descriptor including the waiter deadline.
            loader: Awaitable operation producing the value for a cache miss.

        Returns:
            A decoded cached value or the loader's original value.
        """
        try:
            key = self._cache_key(entry)
        except (TypeError, ValueError):
            _emit_cache_event(entry.layer, "fail_open", "key_build_failed", warning=True)
            return await loader()

        if await self._is_closed():
            _emit_cache_event(entry.layer, "fail_open", "cache_closed", warning=True)
            return await loader()

        cache_read = await self._read_cached_value(entry, key, operation="read")
        if isinstance(cache_read, _CommandUnavailable):
            return await loader()
        if cache_read.state == "valid":
            _emit_cache_event(entry.layer, "hit")
            return cast(ValueT, cache_read.value)
        if cache_read.state == "corrupt":
            healed_cache_read = await self._heal_corruption(entry, key, cache_read)
            if isinstance(healed_cache_read, (_CommandUnavailable, _UnownedLoad)):
                return await loader()
            if healed_cache_read.state == "valid":
                _emit_cache_event(entry.layer, "hit")
                return cast(ValueT, healed_cache_read.value)

        _emit_cache_event(entry.layer, "miss")
        lease_key = self._lease_key(key)
        ownership = await self._try_acquire_lease(entry, lease_key)
        if isinstance(ownership, _CommandUnavailable):
            return await loader()
        if ownership is not None:
            return await self._load_as_owner(entry, key, lease_key, ownership, loader)

        return await self._wait_or_load(entry, key, lease_key, loader)

    async def aclose(self) -> None:
        """Cancel renewal work and close the owned Redis client at most once."""
        async with self._close_lock:
            async with self._lifecycle_lock:
                if self._closed:
                    return
                self._closed = True
                renewal_tasks = tuple(self._renewal_tasks)
                self._renewal_tasks.clear()

            for renewal_task in renewal_tasks:
                renewal_task.cancel()
            if renewal_tasks:
                await asyncio.gather(*renewal_tasks, return_exceptions=True)
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

    async def _wait_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        lease_key: str,
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Poll a contended fill for one bounded layer deadline before loading unowned."""
        _emit_cache_event(entry.layer, "wait", "started")
        deadline = self._runtime.monotonic() + entry.wait_timeout_seconds
        while True:
            remaining_seconds = deadline - self._runtime.monotonic()
            if remaining_seconds <= 0:
                break
            await self._runtime.sleep(min(self._policy.wait_poll_seconds, remaining_seconds))
            if self._runtime.monotonic() >= deadline:
                break

            cache_read = await self._read_cached_value(entry, key, operation="wait_read")
            if isinstance(cache_read, _CommandUnavailable):
                return await loader()
            if cache_read.state == "valid":
                _emit_cache_event(entry.layer, "hit")
                return cast(ValueT, cache_read.value)
            if cache_read.state == "corrupt":
                healed_cache_read = await self._heal_corruption(entry, key, cache_read)
                if isinstance(healed_cache_read, (_CommandUnavailable, _UnownedLoad)):
                    return await loader()
                if healed_cache_read.state == "valid":
                    _emit_cache_event(entry.layer, "hit")
                    return cast(ValueT, healed_cache_read.value)

            ownership = await self._try_acquire_lease(entry, lease_key)
            if isinstance(ownership, _CommandUnavailable):
                return await loader()
            if ownership is not None:
                return await self._load_as_owner(entry, key, lease_key, ownership, loader)

        _emit_cache_event(entry.layer, "wait", "deadline")
        return await loader()

    async def _load_as_owner[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        lease_key: str,
        ownership: _LeaseOwnership,
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Run one loader under a lease and fill only while the token remains current."""
        renewal_task = await self._start_renewal(entry, lease_key, ownership)
        if renewal_task is None:
            return await loader()

        try:
            loaded_value = await loader()
        except BaseException:
            await self._stop_renewal(renewal_task)
            await self._release_lease(entry, lease_key, ownership)
            raise

        await self._stop_renewal(renewal_task)
        if ownership.lost or await self._is_closed():
            return loaded_value
        await self._store_if_owned(entry, key, lease_key, ownership, loaded_value)
        return loaded_value

    async def _start_renewal[ValueT](
        self,
        entry: CacheEntry[ValueT],
        lease_key: str,
        ownership: _LeaseOwnership,
    ) -> asyncio.Task[None] | None:
        """Create and register one renewal task unless shutdown has already started."""
        async with self._lifecycle_lock:
            if self._closed:
                return None
            renewal_task = asyncio.create_task(
                self._renew_lease(entry, lease_key, ownership),
                name="reelio-cache-lease-renewal",
            )
            self._renewal_tasks.add(renewal_task)
            renewal_task.add_done_callback(self._discard_renewal_task)
            return renewal_task

    async def _stop_renewal(self, renewal_task: asyncio.Task[None]) -> None:
        """Cancel and drain one renewal task that may already be shutdown-drained."""
        if not renewal_task.done():
            renewal_task.cancel()
        await asyncio.gather(renewal_task, return_exceptions=True)

    async def _renew_lease[ValueT](
        self,
        entry: CacheEntry[ValueT],
        lease_key: str,
        ownership: _LeaseOwnership,
    ) -> None:
        """Extend one lease until cancellation or token ownership becomes unverifiable."""
        while True:
            await self._runtime.sleep(self._policy.renew_interval_seconds)
            if await self._is_closed():
                return
            result = await self._run_redis_command(
                entry,
                "lease_renew",
                lambda: self._scripts_for_client().renew(
                    keys=[lease_key],
                    args=[ownership.token, self._policy.lease_ttl_milliseconds],
                ),
            )
            if result is _COMMAND_UNAVAILABLE:
                ownership.lost = True
                return
            if not bool(result):
                ownership.lost = True
                _emit_cache_event(entry.layer, "lease_loss", "lease_renew_lost", warning=True)
                return

    async def _store_if_owned[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        lease_key: str,
        ownership: _LeaseOwnership,
        value: ValueT,
    ) -> None:
        """Serialize once and atomically fill only while the owner token still matches."""
        try:
            ttl_seconds = entry.ttl_seconds(value)
            if type(ttl_seconds) is not int or ttl_seconds <= 0:
                raise ValueError("Cache TTL must be a positive integer")
            encoded_value = entry.codec.encode(value)
            if not isinstance(encoded_value, dict):
                raise TypeError("Cache codec must encode a JSON object")
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
            _emit_cache_event(entry.layer, "fail_open", "encode_failed", warning=True)
            await self._release_lease(entry, lease_key, ownership)
            return

        if len(payload) > self._policy.maximum_envelope_bytes:
            _emit_cache_event(entry.layer, "oversize_skip")
            await self._release_lease(entry, lease_key, ownership)
            return

        result = await self._run_redis_command(
            entry,
            "ownership_write",
            lambda: self._scripts_for_client().fill(
                keys=[key, lease_key],
                args=[ownership.token, payload, ttl_seconds],
            ),
        )
        if result is _COMMAND_UNAVAILABLE:
            return
        if not bool(result):
            _emit_cache_event(entry.layer, "lease_loss", "ownership_write_lost", warning=True)
            return
        _emit_cache_event(entry.layer, "fill")

    async def _release_lease[ValueT](
        self,
        entry: CacheEntry[ValueT],
        lease_key: str,
        ownership: _LeaseOwnership,
    ) -> None:
        """Best-effort release only the current lease token outside cache shutdown."""
        if await self._is_closed():
            return
        result = await self._run_redis_command(
            entry,
            "lease_release",
            lambda: self._scripts_for_client().release(
                keys=[lease_key],
                args=[ownership.token],
            ),
        )
        if result is not _COMMAND_UNAVAILABLE and not bool(result):
            _emit_cache_event(entry.layer, "lease_loss", "lease_release_lost", warning=True)

    async def _read_cached_value[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        *,
        operation: Literal["read", "wait_read"],
    ) -> _CacheRead[ValueT] | _CommandUnavailable:
        """Read and classify one raw cache payload under the command bound."""
        raw_payload = await self._run_redis_command(
            entry,
            operation,
            lambda: self._client.get(key),
        )
        if isinstance(raw_payload, _CommandUnavailable):
            return _COMMAND_UNAVAILABLE
        return _decode_cached_value(raw_payload, entry)

    async def _heal_corruption[ValueT](
        self,
        entry: CacheEntry[ValueT],
        key: str,
        cache_read: _CacheRead[ValueT],
    ) -> _CacheRead[ValueT] | _CommandUnavailable | _UnownedLoad:
        """Conditionally unlink corrupt bytes without deleting a concurrent repair."""
        for cleanup_attempt in range(2):
            raw_payload = cache_read.raw_payload
            corruption_reason = cache_read.corruption_reason
            if raw_payload is None or corruption_reason is None:
                return _UNOWNED_LOAD
            _emit_cache_event(entry.layer, "corruption", corruption_reason, warning=True)
            raw_bytes = _as_raw_bytes(raw_payload)
            cleanup_result = await self._run_redis_command(
                entry,
                "corrupt_delete",
                partial(self._delete_corrupt_payload, key, raw_bytes),
            )
            if isinstance(cleanup_result, _CommandUnavailable):
                return _COMMAND_UNAVAILABLE
            if bool(cleanup_result):
                return _CacheRead("absent")

            refreshed_cache_read = await self._read_cached_value(entry, key, operation="read")
            if isinstance(refreshed_cache_read, _CommandUnavailable):
                return _COMMAND_UNAVAILABLE
            if refreshed_cache_read.state != "corrupt":
                return refreshed_cache_read
            cache_read = refreshed_cache_read
            if cleanup_attempt == 1:
                return _UNOWNED_LOAD
        return _UNOWNED_LOAD

    async def _delete_corrupt_payload(self, key: str, raw_bytes: bytes) -> object:
        """Run one compare-raw-payload-and-UNLINK script invocation."""
        script_args: list[_ScriptArgument] = [raw_bytes]
        return await self._scripts_for_client().delete_corrupt(
            keys=[key],
            args=script_args,
        )

    async def _try_acquire_lease[ValueT](
        self,
        entry: CacheEntry[ValueT],
        lease_key: str,
    ) -> _LeaseOwnership | None | _CommandUnavailable:
        """Acquire a token-owned lease or report ordinary existing-owner contention."""
        token = self._runtime.token_factory()
        acquired = await self._run_redis_command(
            entry,
            "lease_acquire",
            lambda: self._client.set(
                lease_key,
                token,
                px=self._policy.lease_ttl_milliseconds,
                nx=True,
            ),
        )
        if isinstance(acquired, _CommandUnavailable):
            return _COMMAND_UNAVAILABLE
        return _LeaseOwnership(token) if acquired is True else None

    async def _run_redis_command[ValueT, ResultT](
        self,
        entry: CacheEntry[ValueT],
        operation: str,
        command: Callable[[], Awaitable[ResultT]],
    ) -> ResultT | _CommandUnavailable:
        """Run one bounded Redis command unless shutdown or circuit state bypasses it."""
        if await self._is_closed():
            return _COMMAND_UNAVAILABLE
        is_probe = await self._admit_redis_command(entry.layer)
        if is_probe is None:
            return _COMMAND_UNAVAILABLE
        try:
            async with asyncio.timeout(self._policy.command_timeout_seconds):
                result = await command()
        except TimeoutError:
            await self._open_circuit(entry.layer, "command_timeout")
            _emit_cache_event(entry.layer, "fail_open", "command_timeout", warning=True)
            return _COMMAND_UNAVAILABLE
        except RedisError:
            failure_outcome = f"{operation}_failed"
            await self._open_circuit(entry.layer, failure_outcome)
            _emit_cache_event(entry.layer, "fail_open", failure_outcome, warning=True)
            return _COMMAND_UNAVAILABLE
        if is_probe:
            await self._close_circuit_after_probe(entry.layer)
        return result

    async def _admit_redis_command(self, layer: str) -> bool | None:
        """Admit one normal command, one recovery probe, or a circuit bypass."""
        async with self._circuit_lock:
            if self._circuit_open_until is None:
                return False
            if self._runtime.monotonic() < self._circuit_open_until:
                _emit_cache_event(layer, "fail_open", "circuit_open", warning=True)
                return None
            if self._circuit_probe_in_flight:
                _emit_cache_event(layer, "fail_open", "circuit_open", warning=True)
                return None
            self._circuit_probe_in_flight = True
        _emit_cache_event(layer, "circuit_probe", "started")
        return True

    async def _open_circuit(self, layer: str, outcome: str) -> None:
        """Open or refresh the local circuit after one bounded Redis failure."""
        async with self._circuit_lock:
            self._circuit_open_until = self._runtime.monotonic() + self._policy.circuit_open_seconds
            self._circuit_probe_in_flight = False
        _emit_cache_event(layer, "circuit_open", outcome, warning=True)

    async def _close_circuit_after_probe(self, layer: str) -> None:
        """Close only the still-designated recovery probe's circuit state."""
        circuit_closed = False
        async with self._circuit_lock:
            if self._circuit_probe_in_flight:
                self._circuit_open_until = None
                self._circuit_probe_in_flight = False
                circuit_closed = True
        if circuit_closed:
            _emit_cache_event(layer, "circuit_closed", "recovered")

    async def _is_closed(self) -> bool:
        """Return whether shutdown has started under the lifecycle serialization lock."""
        async with self._lifecycle_lock:
            return self._closed

    def _scripts_for_client(self) -> _Scripts:
        """Register token-owned scripts lazily without opening a Redis connection."""
        if self._scripts is None:
            self._scripts = _Scripts(
                renew=self._client.register_script(_LEASE_RENEW_SCRIPT),
                release=self._client.register_script(_LEASE_RELEASE_SCRIPT),
                fill=self._client.register_script(_LEASE_FILL_SCRIPT),
                delete_corrupt=self._client.register_script(_CORRUPTION_DELETE_SCRIPT),
            )
        return self._scripts

    def _lease_key(self, key: str) -> str:
        """Derive the coordination key from an already HMAC-obscured data key."""
        return f"{key}:lease"

    def _discard_renewal_task(self, renewal_task: asyncio.Task[None]) -> None:
        """Remove completed renewal work and retrieve unexpected task failures."""
        self._renewal_tasks.discard(renewal_task)
        if not renewal_task.cancelled():
            renewal_task.exception()


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
    client = Redis.from_url(
        redis_url.get_secret_value(),
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=False,
        decode_responses=False,
    )
    return RedisCache(
        cast(_RedisClient, client),
        settings.namespace,
        key_secret.get_secret_value().encode(),
    )


def _decode_cached_value[ValueT](
    raw_payload: bytes | str | None,
    entry: CacheEntry[ValueT],
) -> _CacheRead[ValueT]:
    """Classify a raw cache value as absent, valid, or safely recoverable corruption."""
    if raw_payload is None:
        return _CacheRead("absent")
    try:
        payload_text = raw_payload.decode() if isinstance(raw_payload, bytes) else raw_payload
    except UnicodeDecodeError:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="malformed_utf8",
        )
    try:
        envelope = json.loads(payload_text)
    except json.JSONDecodeError:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="malformed_json",
        )
    if not isinstance(envelope, dict) or set(envelope) != {
        "envelope_version",
        "value_version",
        "value",
    }:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="envelope_shape",
        )
    if envelope["envelope_version"] != _ENVELOPE_VERSION:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="envelope_version",
        )
    if envelope["value_version"] != entry.codec.version:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="codec_version",
        )
    encoded_value = envelope["value"]
    if not isinstance(encoded_value, dict):
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="value_shape",
        )
    try:
        decoded_value = entry.codec.decode(cast(JsonObject, encoded_value))
    except (CacheCodecError, TypeError, ValueError):
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="codec_validation",
        )
    return _CacheRead("valid", value=decoded_value)


def _as_raw_bytes(raw_payload: bytes | str) -> bytes:
    """Return exact Redis wire bytes for conditional corruption cleanup."""
    return raw_payload if isinstance(raw_payload, bytes) else raw_payload.encode()


def _emit_cache_event(
    layer: str,
    event: str,
    outcome: str | None = None,
    *,
    warning: bool = False,
) -> None:
    """Emit one privacy-safe structured cache event without operation material."""
    extra: dict[str, str] = {"cache_layer": layer, "cache_event": event}
    if outcome is not None:
        extra["outcome"] = outcome
    log_method = logger.warning if warning else logger.info
    log_method("shared cache event", extra=extra)
