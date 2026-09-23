"""Deterministic raw-byte Redis fake for shared-cache behavior tests."""

from collections.abc import Callable

from redis.exceptions import RedisError


class FakeRedis:
    """Store raw Redis values with deterministic expiry and controllable command failures."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Initialize an empty fake Redis store.

        Args:
            clock: Deterministic monotonic clock used to expire entries.
        """
        self._clock = clock
        self._values: dict[str, tuple[bytes, float]] = {}
        self.fail_reads = False
        self.fail_writes = False
        self.close_calls = 0

    @property
    def keys(self) -> tuple[str, ...]:
        """Return current unexpired raw Redis keys."""
        self._discard_expired()
        return tuple(self._values)

    async def get(self, name: str) -> bytes | None:
        """Return one unexpired raw value unless reads are configured to fail.

        Args:
            name: Redis key to retrieve.

        Returns:
            Stored raw bytes or None for a miss.

        Raises:
            RedisError: If read failure injection is enabled.
        """
        if self.fail_reads:
            raise RedisError("Injected read failure")
        self._discard_expired()
        stored_value = self._values.get(name)
        return None if stored_value is None else stored_value[0]

    async def set(self, name: str, value: bytes, ex: int) -> bool:
        """Store raw bytes until the supplied positive expiry unless writes fail.

        Args:
            name: Redis key to store.
            value: Raw bytes to store.
            ex: Positive expiry in seconds.

        Returns:
            True after storing the value.

        Raises:
            RedisError: If write failure injection is enabled.
        """
        if self.fail_writes:
            raise RedisError("Injected write failure")
        self._values[name] = (value, self._clock() + ex)
        return True

    async def aclose(self) -> None:
        """Record one client closure."""
        self.close_calls += 1

    async def put_raw(self, name: str, value: bytes, ex: int = 60) -> None:
        """Seed raw bytes for envelope compatibility tests.

        Args:
            name: Redis key to seed.
            value: Raw bytes to seed.
            ex: Positive expiry in seconds.
        """
        self._values[name] = (value, self._clock() + ex)

    def _discard_expired(self) -> None:
        expired_keys = [
            key for key, (_, expires_at) in self._values.items() if self._clock() >= expires_at
        ]
        for key in expired_keys:
            del self._values[key]
