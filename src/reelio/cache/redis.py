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
from reelio.cache.interface import (
    AsyncCache,
    CacheCodecError,
    CacheEntry,
    CacheSkip,
    CacheWrite,
    JsonObject,
    RetainedCacheValue,
    RevalidatingCacheEntry,
)

logger = logging.getLogger(__name__)

_ENVELOPE_VERSION = 2
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
_OWNED_DISCARD_SCRIPT = """
-- OWNED_DISCARD
if redis.call("GET", KEYS[2]) == ARGV[1] then
    redis.call("UNLINK", KEYS[1])
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
_ATOMIC_READ_SCRIPT = """
local raw_payload = redis.call("GET", KEYS[1])
if not raw_payload then
    return {0, -2}
end
return {1, raw_payload, redis.call("PTTL", KEYS[1])}
"""


type _ScriptArgument = bytes | str | int
type _CacheDescriptor[ValueT] = CacheEntry[ValueT] | RevalidatingCacheEntry[ValueT]


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
    """Contain lazily registered cache coordination scripts."""

    renew: _RedisScript
    release: _RedisScript
    fill: _RedisScript
    discard: _RedisScript
    delete_corrupt: _RedisScript
    read: _RedisScript


@dataclass(frozen=True, slots=True)
class _CacheRead[ValueT]:
    """Classify one Redis data-key read without collapsing corruption into absence."""

    state: Literal["absent", "fresh", "retained", "corrupt"]
    value: ValueT | None = None
    raw_payload: bytes | str | None = None
    corruption_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _OrdinaryCacheValue[ValueT]:
    """Adapt ordinary TTL callers to the shared revalidating state machine."""

    value: ValueT


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

type _LoadOutcome[ValueT] = _OrdinaryCacheValue[ValueT] | CacheWrite[ValueT] | CacheSkip[ValueT]
type _RevalidatingLoader[ValueT] = Callable[
    [RetainedCacheValue[ValueT] | None], Awaitable[_LoadOutcome[ValueT]]
]


class RedisCache:
    """Coordinate token-owned fills and conditional revalidation through one cache seam."""

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
        """Return a fresh ordinary-TTL value or invoke its loader.

        Args:
            entry: Versioned cache operation descriptor.
            loader: Awaitable operation producing a required value for a miss.

        Returns:
            Decoded cache value or the loader's original value.
        """

        async def revalidating_loader(
            retained_value: RetainedCacheValue[ValueT] | None,
        ) -> _OrdinaryCacheValue[ValueT]:
            del retained_value
            return _OrdinaryCacheValue(await loader())

        return await self._get_or_load(entry, revalidating_loader)

    async def get_or_load_revalidating[ValueT](
        self,
        entry: RevalidatingCacheEntry[ValueT],
        loader: Callable[
            [RetainedCacheValue[ValueT] | None],
            Awaitable[CacheWrite[ValueT] | CacheSkip[ValueT]],
        ],
    ) -> ValueT:
        """Return a fresh value or revalidate a physically retained normalized value.

        Args:
            entry: Versioned cache descriptor with shared operation identity.
            loader: Loader receiving only the latest retained normalized value, if any.

        Returns:
            Decoded fresh value or the value produced by the loader.
        """
        return await self._get_or_load(entry, loader)

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

    async def _get_or_load[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        loader: _RevalidatingLoader[ValueT],
    ) -> ValueT:
        """Execute one cache operation while hiding lease and read-state mechanics."""
        try:
            key = self._cache_key(entry)
        except (TypeError, ValueError):
            _emit_cache_event(entry.layer, "fail_open", "key_build_failed", warning=True)
            return await self._load_without_cache(loader)

        if await self._is_closed():
            _emit_cache_event(entry.layer, "fail_open", "cache_closed", warning=True)
            return await self._load_without_cache(loader)

        cache_read = await self._read_and_heal(entry, key, operation="read")
        if isinstance(cache_read, (_CommandUnavailable, _UnownedLoad)):
            return await self._load_without_cache(loader)
        if cache_read.state == "fresh":
            _emit_cache_event(entry.layer, "hit")
            return cast(ValueT, cache_read.value)

        _emit_cache_event(entry.layer, "miss")
        lease_key = self._lease_key(key)
        ownership = await self._try_acquire_lease(entry, lease_key)
        if isinstance(ownership, _CommandUnavailable):
            return await self._load_without_cache(loader)
        if ownership is not None:
            return await self._load_as_owner(entry, key, lease_key, ownership, loader)

        return await self._wait_or_load(entry, key, lease_key, loader)

    def _cache_key[ValueT](self, entry: _CacheDescriptor[ValueT]) -> str:
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
        entry: _CacheDescriptor[ValueT],
        key: str,
        lease_key: str,
        loader: _RevalidatingLoader[ValueT],
    ) -> ValueT:
        """Poll a contended fill, then make one final ownership attempt before fallback."""
        _emit_cache_event(entry.layer, "wait", "started")
        deadline = self._runtime.monotonic() + entry.wait_timeout_seconds
        while True:
            remaining_seconds = deadline - self._runtime.monotonic()
            if remaining_seconds <= 0:
                break
            await self._runtime.sleep(min(self._policy.wait_poll_seconds, remaining_seconds))
            if self._runtime.monotonic() >= deadline:
                break

            cache_read = await self._read_and_heal(entry, key, operation="wait_read")
            if isinstance(cache_read, (_CommandUnavailable, _UnownedLoad)):
                return await self._load_without_cache(loader)
            if cache_read.state == "fresh":
                _emit_cache_event(entry.layer, "hit")
                return cast(ValueT, cache_read.value)

            ownership = await self._try_acquire_lease(entry, lease_key)
            if isinstance(ownership, _CommandUnavailable):
                return await self._load_without_cache(loader)
            if ownership is not None:
                return await self._load_as_owner(entry, key, lease_key, ownership, loader)

        _emit_cache_event(entry.layer, "wait", "deadline")
        ownership = await self._try_acquire_lease(entry, lease_key)
        if isinstance(ownership, _CommandUnavailable):
            return await self._load_without_cache(loader)
        if ownership is not None:
            return await self._load_as_owner(entry, key, lease_key, ownership, loader)

        final_cache_read = await self._read_and_heal(entry, key, operation="wait_read")
        if isinstance(final_cache_read, (_CommandUnavailable, _UnownedLoad)):
            return await self._load_without_cache(loader)
        if final_cache_read.state == "fresh":
            _emit_cache_event(entry.layer, "hit")
            return cast(ValueT, final_cache_read.value)
        return await self._load_without_cache(loader, final_cache_read)

    async def _load_as_owner[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        key: str,
        lease_key: str,
        ownership: _LeaseOwnership,
        loader: _RevalidatingLoader[ValueT],
    ) -> ValueT:
        """Re-read under a lease, then load and write only while that lease remains owned."""
        renewal_task = await self._start_renewal(entry, lease_key, ownership)
        if renewal_task is None:
            return await self._load_without_cache(loader)

        ownership_read = await self._read_and_heal(entry, key, operation="read")
        if isinstance(ownership_read, (_CommandUnavailable, _UnownedLoad)):
            await self._stop_renewal(renewal_task)
            await self._release_lease(entry, lease_key, ownership)
            return await self._load_without_cache(loader)
        if ownership_read.state == "fresh":
            await self._stop_renewal(renewal_task)
            await self._release_lease(entry, lease_key, ownership)
            _emit_cache_event(entry.layer, "hit")
            return cast(ValueT, ownership_read.value)

        retained_value = self._retained_value(ownership_read)
        try:
            load_outcome = await loader(retained_value)
        except BaseException:
            await self._stop_renewal(renewal_task)
            await self._release_lease(entry, lease_key, ownership)
            raise

        await self._stop_renewal(renewal_task)
        if ownership.lost or await self._is_closed():
            return load_outcome.value
        if isinstance(load_outcome, CacheSkip):
            await self._discard_if_owned(entry, key, lease_key, ownership)
            return load_outcome.value
        await self._store_if_owned(entry, key, lease_key, ownership, load_outcome)
        return load_outcome.value

    async def _load_without_cache[ValueT](
        self,
        loader: _RevalidatingLoader[ValueT],
        cache_read: _CacheRead[ValueT] | None = None,
    ) -> ValueT:
        """Invoke a loader without write authority and expose only current retained data."""
        return (await loader(self._retained_value(cache_read))).value

    def _retained_value[ValueT](
        self,
        cache_read: _CacheRead[ValueT] | None,
    ) -> RetainedCacheValue[ValueT] | None:
        """Wrap a still-retained value without exposing absent or corrupt cache states."""
        if cache_read is None or cache_read.state != "retained":
            return None
        return RetainedCacheValue(cast(ValueT, cache_read.value))

    async def _start_renewal[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
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
        entry: _CacheDescriptor[ValueT],
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
        entry: _CacheDescriptor[ValueT],
        key: str,
        lease_key: str,
        ownership: _LeaseOwnership,
        load_outcome: _OrdinaryCacheValue[ValueT] | CacheWrite[ValueT],
    ) -> None:
        """Serialize once and atomically fill only while the owner token still matches."""
        try:
            cache_write = self._cache_write(entry, load_outcome)
            encoded_value = entry.codec.encode(cache_write.value)
            if not isinstance(encoded_value, dict):
                raise TypeError("Cache codec must encode a JSON object")
            envelope = {
                "envelope_version": _ENVELOPE_VERSION,
                "value_version": entry.codec.version,
                "value": encoded_value,
                "freshness_seconds": cache_write.freshness_seconds,
                "retention_seconds": cache_write.retention_seconds,
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
                args=[ownership.token, payload, cache_write.retention_seconds],
            ),
        )
        if result is _COMMAND_UNAVAILABLE:
            return
        if not bool(result):
            _emit_cache_event(entry.layer, "lease_loss", "ownership_write_lost", warning=True)
            return
        _emit_cache_event(entry.layer, "fill")

    def _cache_write[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        load_outcome: _OrdinaryCacheValue[ValueT] | CacheWrite[ValueT],
    ) -> CacheWrite[ValueT]:
        """Adapt ordinary TTL values without moving validation outside the guarded write."""
        if isinstance(load_outcome, CacheWrite):
            return load_outcome
        if not isinstance(entry, CacheEntry):
            raise TypeError("Revalidating loaders must return CacheWrite or CacheSkip")
        ttl_seconds = entry.ttl_seconds(load_outcome.value)
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("Cache TTL must be a positive integer")
        return CacheWrite(
            load_outcome.value,
            freshness_seconds=ttl_seconds,
            retention_seconds=ttl_seconds,
        )

    async def _discard_if_owned[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        key: str,
        lease_key: str,
        ownership: _LeaseOwnership,
    ) -> None:
        """Remove retained data only while the loader still owns the coordinating lease."""
        result = await self._run_redis_command(
            entry,
            "ownership_discard",
            lambda: self._scripts_for_client().discard(
                keys=[key, lease_key],
                args=[ownership.token],
            ),
        )
        if result is _COMMAND_UNAVAILABLE:
            return
        if not bool(result):
            _emit_cache_event(entry.layer, "lease_loss", "ownership_discard_lost", warning=True)
            return
        _emit_cache_event(entry.layer, "discard")

    async def _release_lease[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
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

    async def _read_and_heal[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        key: str,
        *,
        operation: Literal["read", "wait_read"],
    ) -> _CacheRead[ValueT] | _CommandUnavailable | _UnownedLoad:
        """Run the atomic read and conditionally remove malformed retained data."""
        cache_read = await self._read_cached_value(entry, key, operation=operation)
        if isinstance(cache_read, _CommandUnavailable) or cache_read.state != "corrupt":
            return cache_read
        return await self._heal_corruption(entry, key, cache_read, operation=operation)

    async def _read_cached_value[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        key: str,
        *,
        operation: Literal["read", "wait_read"],
    ) -> _CacheRead[ValueT] | _CommandUnavailable:
        """Atomically read raw bytes and Redis physical retention under the command bound."""
        atomic_read = await self._run_redis_command(
            entry,
            operation,
            lambda: self._scripts_for_client().read(keys=[key], args=[]),
        )
        if isinstance(atomic_read, _CommandUnavailable):
            return _COMMAND_UNAVAILABLE
        return _decode_atomic_read(atomic_read, entry)

    async def _heal_corruption[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
        key: str,
        cache_read: _CacheRead[ValueT],
        *,
        operation: Literal["read", "wait_read"],
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

            refreshed_cache_read = await self._read_cached_value(entry, key, operation=operation)
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
        return await self._scripts_for_client().delete_corrupt(
            keys=[key],
            args=[raw_bytes],
        )

    async def _try_acquire_lease[ValueT](
        self,
        entry: _CacheDescriptor[ValueT],
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
        entry: _CacheDescriptor[ValueT],
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
        """Register coordination scripts lazily without opening a Redis connection."""
        if self._scripts is None:
            self._scripts = _Scripts(
                renew=self._client.register_script(_LEASE_RENEW_SCRIPT),
                release=self._client.register_script(_LEASE_RELEASE_SCRIPT),
                fill=self._client.register_script(_LEASE_FILL_SCRIPT),
                discard=self._client.register_script(_OWNED_DISCARD_SCRIPT),
                delete_corrupt=self._client.register_script(_CORRUPTION_DELETE_SCRIPT),
                read=self._client.register_script(_ATOMIC_READ_SCRIPT),
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
        settings: Validated application cache configuration.

    Returns:
        DisabledCache when disabled, otherwise one RedisCache owning a lazy client.
    """
    if not settings.enabled:
        return DisabledCache()

    redis_url = settings.redis_url
    key_secret = settings.key_secret
    if redis_url is None or key_secret is None:
        raise ValueError("Enabled cache requires Redis URL and key secret")
    client = Redis.from_url(redis_url.get_secret_value(), decode_responses=False)
    return RedisCache(
        cast(_RedisClient, client),
        settings.namespace,
        key_secret.get_secret_value().encode(),
    )


def _decode_atomic_read[ValueT](
    atomic_result: object,
    entry: _CacheDescriptor[ValueT],
) -> _CacheRead[ValueT]:
    """Validate the exact atomic Redis read result before decoding cached bytes."""
    if not isinstance(atomic_result, (list, tuple)):
        return _CacheRead("corrupt", corruption_reason="atomic_read_shape")
    if (
        len(atomic_result) == 2
        and type(atomic_result[0]) is int
        and atomic_result[0] == 0
        and type(atomic_result[1]) is int
        and atomic_result[1] == -2
    ):
        return _CacheRead("absent")
    if (
        len(atomic_result) == 3
        and type(atomic_result[0]) is int
        and atomic_result[0] == 1
        and isinstance(atomic_result[1], (bytes, str))
    ):
        return _decode_cached_value(atomic_result[1], atomic_result[2], entry)
    return _CacheRead("corrupt", corruption_reason="atomic_read_shape")


def _decode_cached_value[ValueT](
    raw_payload: bytes | str,
    pttl_milliseconds: object,
    entry: _CacheDescriptor[ValueT],
) -> _CacheRead[ValueT]:
    """Classify one envelope by strict metadata and Redis-owned physical retention."""
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
        "freshness_seconds",
        "retention_seconds",
    }:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="envelope_shape",
        )
    if (
        type(envelope["envelope_version"]) is not int
        or envelope["envelope_version"] != _ENVELOPE_VERSION
    ):
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="envelope_version",
        )
    if (
        not isinstance(envelope["value_version"], str)
        or envelope["value_version"] != entry.codec.version
    ):
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="codec_version",
        )
    freshness_seconds = envelope["freshness_seconds"]
    retention_seconds = envelope["retention_seconds"]
    if (
        type(freshness_seconds) is not int
        or freshness_seconds < 0
        or type(retention_seconds) is not int
        or retention_seconds <= 0
        or freshness_seconds > retention_seconds
    ):
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="envelope_retention",
        )
    if type(pttl_milliseconds) is not int:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="pttl_shape",
        )
    if pttl_milliseconds == -1:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="pttl_persistent",
        )
    if pttl_milliseconds == -2:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="pttl_absent",
        )
    if pttl_milliseconds < 0:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="pttl_negative",
        )
    if pttl_milliseconds > retention_seconds * 1_000:
        return _CacheRead(
            "corrupt",
            raw_payload=raw_payload,
            corruption_reason="pttl_above_retention",
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
    fresh_threshold_milliseconds = (retention_seconds - freshness_seconds) * 1_000
    state: Literal["fresh", "retained"]
    state = "fresh" if pttl_milliseconds > fresh_threshold_milliseconds else "retained"
    return _CacheRead(state, value=decoded_value)


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
