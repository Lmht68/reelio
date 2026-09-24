"""Deterministic Redis-compatible fake for shared-cache behavior tests."""

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis.exceptions import RedisError

type _ScriptArgument = bytes | str | int


@dataclass(frozen=True, slots=True)
class _StoredValue:
    """Contain raw Redis bytes and their optional deterministic expiry."""

    value: bytes
    expires_at: float | None


class ManualClock:
    """Advance deterministic monotonic time shared by cache instances."""

    def __init__(self) -> None:
        """Initialize time at zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """Return the current deterministic time."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Advance deterministic time by a nonnegative number of seconds.

        Args:
            seconds: Simulated seconds to add.

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            raise ValueError("Manual clock cannot move backwards")
        self.value += seconds


class ManualSleeper:
    """Block until tests explicitly advance the shared deterministic clock."""

    def __init__(self, clock: ManualClock) -> None:
        """Initialize sleeper state for one manual clock.

        Args:
            clock: Clock determining each pending sleep deadline.
        """
        self._clock = clock
        self._pending: list[tuple[float, asyncio.Future[None]]] = []

    async def sleep(self, seconds: float) -> None:
        """Wait until advance reaches the requested deterministic deadline.

        Args:
            seconds: Nonnegative simulated delay.

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            raise ValueError("Manual sleep duration cannot be negative")
        future = asyncio.get_running_loop().create_future()
        self._pending.append((self._clock() + seconds, future))
        self._wake_ready()
        await future

    def advance(self, seconds: float) -> None:
        """Advance time and release every sleep whose deadline has arrived.

        Args:
            seconds: Nonnegative simulated seconds to add.
        """
        self._clock.advance(seconds)
        self._wake_ready()

    def _wake_ready(self) -> None:
        ready_futures = [
            future
            for deadline, future in self._pending
            if deadline <= self._clock() and not future.done()
        ]
        self._pending = [
            (deadline, future)
            for deadline, future in self._pending
            if deadline > self._clock() and not future.done()
        ]
        for future in ready_futures:
            future.set_result(None)


class FakeRedis:
    """Store raw Redis values with expiry, scripts, failures, and command gates."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Initialize an empty Redis-compatible store.

        Args:
            clock: Deterministic monotonic clock used to expire entries.
        """
        self._clock = clock
        self._values: dict[str, _StoredValue] = {}
        self._pttl_overrides: dict[str, int] = {}
        self.fail_operations: set[str] = set()
        self.command_calls: list[str] = []
        self._blocks: dict[str, asyncio.Event] = {}
        self._started: dict[str, asyncio.Event] = {}
        self.close_calls = 0

    @property
    def keys(self) -> tuple[str, ...]:
        """Return current unexpired raw Redis keys."""
        self._discard_expired()
        return tuple(self._values)

    @property
    def command_counts(self) -> Counter[str]:
        """Return how often each Redis-compatible operation was invoked."""
        return Counter(self.command_calls)

    @property
    def fail_reads(self) -> bool:
        """Return whether all cache data reads currently fail."""
        return "read" in self.fail_operations

    @fail_reads.setter
    def fail_reads(self, enabled: bool) -> None:
        self._set_legacy_failure("read", enabled)

    @property
    def fail_writes(self) -> bool:
        """Return whether token-owned cache fills currently fail."""
        return "ownership_write" in self.fail_operations

    @fail_writes.setter
    def fail_writes(self, enabled: bool) -> None:
        self._set_legacy_failure("ownership_write", enabled)

    async def get(self, name: str) -> bytes | None:
        """Return one unexpired raw value.

        Args:
            name: Redis key to load.

        Returns:
            Stored bytes, or None when the key is absent.

        Raises:
            RedisError: If get failure injection is enabled.
        """
        await self._before("read")
        return self._get_raw(name)

    async def set(
        self,
        name: str,
        value: bytes | str,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Store raw bytes with Redis-compatible NX, EX, and PX behavior.

        Args:
            name: Redis key to store.
            value: Raw value or UTF-8 text value.
            ex: Optional whole-second positive expiry.
            px: Optional millisecond positive expiry.
            nx: Whether to write only when no unexpired key exists.

        Returns:
            True when stored, or False for a failed NX condition.

        Raises:
            RedisError: If the matching command failure is injected.
            ValueError: If expiry arguments are invalid.
        """
        operation = "lease_acquire" if nx else "set"
        await self._before(operation)
        self._discard_expired()
        if nx and name in self._values:
            return False
        self._values[name] = _StoredValue(
            _as_bytes(value),
            self._expiry_from_arguments(ex=ex, px=px),
        )
        return True

    def register_script(
        self,
        script: str,
    ) -> Callable[..., Awaitable[object]]:
        """Return a lazy script callable with the matching Redis operation semantics.

        Args:
            script: Lua source registered by the cache.

        Returns:
            Awaitable callable that atomically applies the requested script behavior.
        """
        operation = _script_operation(script)

        async def execute(*, keys: list[str], args: list[_ScriptArgument]) -> object:
            await self._before(operation)
            self._discard_expired(expire_at_boundary=operation != "read")
            return self._execute_script(operation, keys, args)

        return execute

    async def aclose(self) -> None:
        """Record one client closure."""
        self.close_calls += 1

    async def put_raw(self, name: str, value: bytes, ex: int | None = 60) -> None:
        """Seed raw bytes for envelope compatibility tests.

        Args:
            name: Redis key to seed.
            value: Raw bytes to seed.
            ex: Optional positive expiry in seconds.

        Raises:
            ValueError: If expiry is invalid.
        """
        self._values[name] = _StoredValue(value, self._expiry_from_arguments(ex=ex, px=None))
        self._pttl_overrides.pop(name, None)

    def raw_value(self, name: str) -> bytes | None:
        """Return raw bytes directly without recording a Redis command.

        Args:
            name: Redis key to inspect.

        Returns:
            Stored raw bytes, or None when absent or expired.
        """
        return self._get_raw(name)

    def set_pttl_override(self, name: str, pttl_milliseconds: int) -> None:
        """Force one atomic-read PTTL result without changing the stored raw payload.

        Args:
            name: Existing data key whose reported PTTL should be replaced.
            pttl_milliseconds: Exact millisecond result returned by atomic reads.

        Raises:
            KeyError: If no unexpired raw value is stored for the key.
        """
        if self._get_raw(name) is None:
            raise KeyError(name)
        self._pttl_overrides[name] = pttl_milliseconds

    def block(self, operation: str) -> None:
        """Make an operation wait until unblock is called.

        Args:
            operation: Redis-compatible operation name to block.
        """
        self._blocks[operation] = asyncio.Event()

    def unblock(self, operation: str) -> None:
        """Release all commands blocked for one operation.

        Args:
            operation: Redis-compatible operation name to unblock.
        """
        blocker = self._blocks.pop(operation, None)
        if blocker is not None:
            blocker.set()

    def started(self, operation: str) -> asyncio.Event:
        """Return an event set when an operation begins.

        Args:
            operation: Redis-compatible operation name to observe.

        Returns:
            Event set before failure injection or gate waiting.
        """
        return self._started.setdefault(operation, asyncio.Event())

    async def _before(self, operation: str) -> None:
        self.command_calls.append(operation)
        self.started(operation).set()
        if operation in self.fail_operations:
            raise RedisError(f"Injected {operation} failure")
        blocker = self._blocks.get(operation)
        if blocker is not None:
            await blocker.wait()
        if operation in self.fail_operations:
            raise RedisError(f"Injected {operation} failure")

    def _execute_script(
        self,
        operation: str,
        keys: list[str],
        args: list[_ScriptArgument],
    ) -> object:
        if operation == "read":
            return self._atomic_read(keys[0])
        if operation == "lease_renew":
            return self._renew_lease(keys[0], _as_bytes(args[0]), int(args[1]))
        if operation == "lease_release":
            return self._release_lease(keys[0], _as_bytes(args[0]))
        if operation == "ownership_write":
            return self._fill_if_owned(
                data_key=keys[0],
                lease_key=keys[1],
                token=_as_bytes(args[0]),
                payload=_as_bytes(args[1]),
                ttl_seconds=int(args[2]),
            )
        if operation == "ownership_discard":
            return self._discard_if_owned(
                data_key=keys[0],
                lease_key=keys[1],
                token=_as_bytes(args[0]),
            )
        if operation == "corrupt_delete":
            return self._delete_if_raw_matches(keys[0], _as_bytes(args[0]))
        raise ValueError(f"Unsupported script operation: {operation}")

    def _renew_lease(self, lease_key: str, token: bytes, ttl_milliseconds: int) -> int:
        current_lease = self._get_raw(lease_key)
        if current_lease != token:
            return 0
        if ttl_milliseconds <= 0:
            raise ValueError("Lease TTL must be positive")
        self._values[lease_key] = _StoredValue(
            token,
            self._clock() + ttl_milliseconds / 1_000,
        )
        return 1

    def _release_lease(self, lease_key: str, token: bytes) -> int:
        if self._get_raw(lease_key) != token:
            return 0
        del self._values[lease_key]
        self._pttl_overrides.pop(lease_key, None)
        return 1

    def _fill_if_owned(
        self,
        *,
        data_key: str,
        lease_key: str,
        token: bytes,
        payload: bytes,
        ttl_seconds: int,
    ) -> int:
        if self._get_raw(lease_key) != token:
            return 0
        if ttl_seconds <= 0:
            raise ValueError("Data TTL must be positive")
        self._values[data_key] = _StoredValue(payload, self._clock() + ttl_seconds)
        self._pttl_overrides.pop(data_key, None)
        del self._values[lease_key]
        self._pttl_overrides.pop(lease_key, None)
        return 1

    def _discard_if_owned(self, *, data_key: str, lease_key: str, token: bytes) -> int:
        if self._get_raw(lease_key) != token:
            return 0
        self._values.pop(data_key, None)
        self._pttl_overrides.pop(data_key, None)
        del self._values[lease_key]
        self._pttl_overrides.pop(lease_key, None)
        return 1

    def _delete_if_raw_matches(self, key: str, raw_payload: bytes) -> int:
        if self._get_raw(key) != raw_payload:
            return 0
        del self._values[key]
        self._pttl_overrides.pop(key, None)
        return 1

    def _atomic_read(self, name: str) -> list[bytes | int]:
        stored_value = self._values.get(name)
        if stored_value is None:
            return [0, -2]
        return [1, stored_value.value, self._pttl_milliseconds(name, stored_value)]

    def _pttl_milliseconds(self, name: str, stored_value: _StoredValue) -> int:
        override = self._pttl_overrides.get(name)
        if override is not None:
            return override
        if stored_value.expires_at is None:
            return -1
        remaining_seconds = max(0.0, stored_value.expires_at - self._clock())
        return int(round(remaining_seconds * 1_000))

    def _expiry_from_arguments(self, *, ex: int | None, px: int | None) -> float | None:
        if ex is not None and px is not None:
            raise ValueError("Only one Redis expiry argument may be supplied")
        if ex is not None:
            if ex <= 0:
                raise ValueError("Redis EX expiry must be positive")
            return self._clock() + ex
        if px is not None:
            if px <= 0:
                raise ValueError("Redis PX expiry must be positive")
            return self._clock() + px / 1_000
        return None

    def _get_raw(self, name: str) -> bytes | None:
        self._discard_expired()
        stored_value = self._values.get(name)
        return None if stored_value is None else stored_value.value

    def _discard_expired(self, *, expire_at_boundary: bool = True) -> None:
        expired_keys = [
            key
            for key, stored_value in self._values.items()
            if stored_value.expires_at is not None
            and (
                stored_value.expires_at <= self._clock()
                if expire_at_boundary
                else stored_value.expires_at < self._clock()
            )
        ]
        for key in expired_keys:
            del self._values[key]
            self._pttl_overrides.pop(key, None)

    def _set_legacy_failure(self, operation: str, enabled: bool) -> None:
        if enabled:
            self.fail_operations.add(operation)
            return
        self.fail_operations.discard(operation)


def _as_bytes(value: _ScriptArgument) -> bytes:
    """Convert fake Redis text or raw script arguments to wire bytes."""
    if isinstance(value, bytes):
        return value
    return str(value).encode()


def _script_operation(script: str) -> str:
    """Identify one cache script from its stable Redis command vocabulary."""
    if "PTTL" in script:
        return "read"
    if "OWNED_DISCARD" in script:
        return "ownership_discard"
    if "PEXPIRE" in script:
        return "lease_renew"
    if '"EX"' in script:
        return "ownership_write"
    if "UNLINK" in script:
        return "corrupt_delete"
    if '"DEL"' in script:
        return "lease_release"
    raise ValueError("Unsupported Redis script")
