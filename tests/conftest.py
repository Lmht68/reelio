"""Shared pytest fixtures and safe import-time settings defaults."""

import os
from collections.abc import AsyncIterator

os.environ.setdefault("REELIO_LLM_PROVIDER", "deepseek")
os.environ.setdefault("REELIO_DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("REELIO_TMDB_API_KEY", "test-tmdb-key")

import pytest
from httpx import ASGITransport, AsyncClient

from reelio.main import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Yield an HTTP client connected to the application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
