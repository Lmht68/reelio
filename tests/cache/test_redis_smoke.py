"""Real Redis-compatible lease-script smoke coverage."""

import asyncio
import os
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from reelio.cache.redis import (
    _ATOMIC_READ_SCRIPT,
    _CORRUPTION_DELETE_SCRIPT,
    _LEASE_FILL_SCRIPT,
    _LEASE_RELEASE_SCRIPT,
    _LEASE_RENEW_SCRIPT,
    _OWNED_DISCARD_SCRIPT,
)

pytestmark = pytest.mark.redis_smoke


async def test_redis_lease_scripts_enforce_token_ownership() -> None:
    """Exercise acquire, expiry, renewal, conditional fill, and release against Redis."""
    redis_url = os.environ.get("REELIO_TEST_REDIS_URL")
    if redis_url is None:
        pytest.skip("REELIO_TEST_REDIS_URL is not configured")

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
