"""HTTP contract tests for the health endpoint."""

from httpx import AsyncClient


async def test_health_returns_ok(client: AsyncClient) -> None:
    """Return the stable healthy status payload."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_root_endpoint_is_not_exposed(client: AsyncClient) -> None:
    """Keep the removed hello-world endpoint unavailable."""
    response = await client.get("/")

    assert response.status_code == 404
