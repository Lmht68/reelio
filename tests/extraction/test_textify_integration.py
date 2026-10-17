"""End-to-end HTTP coverage for Reelio's Textify-backed extraction path."""

import hmac
from collections.abc import Awaitable, Callable
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from reelio.config import Environment, app_settings
from reelio.extraction.coordination import (
    CoordinationConfig,
    create_postgres_textify_coordination,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.textify import TextifySourceTranscriptProvider
from reelio.main import create_app
from tests.extraction.fakes import FakeInterpretationService, FakeResultAggregator

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CAPABILITY_LINK = "/api/v1/transcription-jobs/00000000-0000-4000-8000-000000000000"
_HMAC_SECRET = "test-coordination-hmac-secret"


class _TextifyContractServer:
    """Serve the narrow Textify lifecycle consumed by Reelio."""

    def __init__(self, transcript_method: str, active_polls: int = 0) -> None:
        """Configure one successful Textify job lifecycle.

        Args:
            transcript_method: Textify transcript acquisition method to return.
            active_polls: Number of processing responses before successful completion.
        """
        self.application = FastAPI()
        self.transcript_method = transcript_method
        self.active_polls = active_polls
        self.request_paths: list[str] = []
        self.admission_bodies: list[object] = []
        self.idempotency_keys: list[str] = []
        self._poll_count = 0
        self.application.add_api_route(
            "/api/v1/transcription-jobs",
            self._admit,
            methods=["POST"],
        )
        self.application.add_api_route(
            "/api/v1/transcription-jobs/{job_id}",
            self._inspect,
            methods=["GET"],
        )

    async def _admit(self, request: Request) -> JSONResponse:
        """Accept a Textify job with a canonical capability link."""
        self.request_paths.append(request.url.path)
        self.admission_bodies.append(await request.json())
        idempotency_key = request.headers.get("Idempotency-Key")
        assert idempotency_key is not None
        self.idempotency_keys.append(idempotency_key)
        return JSONResponse(
            status_code=202,
            content={"status": "queued", "links": {"self": _CAPABILITY_LINK}},
            headers={"Location": _CAPABILITY_LINK, "Retry-After": "0"},
        )

    async def _inspect(self, request: Request, job_id: str) -> JSONResponse:
        """Return an active or completed job for the admitted capability."""
        assert job_id == _CAPABILITY_LINK.rsplit("/", maxsplit=1)[1]
        self.request_paths.append(request.url.path)
        if self._poll_count < self.active_polls:
            self._poll_count += 1
            return JSONResponse(
                content={"status": "processing", "links": {"self": _CAPABILITY_LINK}},
                headers={"Retry-After": "0"},
            )
        return JSONResponse(
            content={
                "status": "finished",
                "outcome": "succeeded",
                "links": {"self": _CAPABILITY_LINK},
                "result": {
                    "source": {
                        "platform": "youtube",
                        "video_id": "dQw4w9WgXcQ",
                        "url": _SUBMITTED_URL,
                        "title": "Textify integration video",
                        "description": "A Textify integration description.",
                        "channel": "Textify integration channel",
                        "duration_seconds": 42,
                    },
                    "transcript": {
                        "method": self.transcript_method,
                        "language": "en",
                        "text": "Textify integration transcript.",
                    },
                },
            }
        )


def _coordination_settings(database_url: str) -> CoordinationConfig:
    """Create coordination settings without consulting environment files."""
    constructor = cast(Callable[..., CoordinationConfig], CoordinationConfig)
    return constructor(
        _env_file=None,
        database_url=database_url,
        hmac_secret=_HMAC_SECRET,
    )


async def _create_pipeline(
    database_url: str,
    server: _TextifyContractServer,
    sleeper: Callable[[int], Awaitable[None]],
) -> ExtractionPipeline:
    """Compose a real coordination-backed pipeline against a Textify ASGI server."""
    coordination_repository = await create_postgres_textify_coordination(
        _coordination_settings(database_url)
    )
    textify_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.application),
        base_url="http://textify.test",
        follow_redirects=False,
    )
    source_transcript_provider = TextifySourceTranscriptProvider(
        client=textify_client,
        coordination_repository=coordination_repository,
        sleeper=sleeper,
    )
    return ExtractionPipeline(
        source_transcript_provider=source_transcript_provider,
        interpretation_service=FakeInterpretationService(),
        result_aggregator=FakeResultAggregator(),
        default_market=SpotifyMarket("US"),
    )


async def _stored_coordination(database_url: str) -> tuple[bytes, UUID, str]:
    """Read the durable Textify state written by one public extraction request."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text(
                        "SELECT url_identity, idempotency_key, capability_link "
                        "FROM textify_coordination"
                    )
                )
            ).one()
    finally:
        await engine.dispose()
    return cast(tuple[bytes, UUID, str], tuple(row))


async def _pipeline_factory(pipeline: ExtractionPipeline) -> ExtractionPipeline:
    """Return one already-composed pipeline to the application lifespan."""
    return pipeline


async def test_public_extraction_uses_textify_whisper_and_durable_coordination(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a public extraction after durable admission and Textify completion."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper", active_polls=1)
    waits: list[int] = []

    async def sleeper(seconds: int) -> None:
        waits.append(seconds)

    pipeline = await _create_pipeline(migrated_coordination_database, server, sleeper)
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            public_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )
            transcript_response = await client.post(
                "/api/internal/extractions",
                json={"transcript": "Submitted transcript text."},
            )

    assert public_response.status_code == 200
    assert public_response.json()["source"] == {
        "platform": "youtube",
        "video_id": "dQw4w9WgXcQ",
        "url": _SUBMITTED_URL,
        "title": "Textify integration video",
        "description": "A Textify integration description.",
        "channel": "Textify integration channel",
        "duration_seconds": 42,
    }
    assert public_response.json()["transcript"] == {
        "method": "whisper",
        "language": "en",
        "text": "Textify integration transcript.",
    }
    assert transcript_response.status_code == 200
    assert transcript_response.json()["transcript"] == {
        "method": "text_submission",
        "language": "und",
        "text": "Submitted transcript text.",
    }
    assert waits == [0, 0]
    assert server.request_paths == [
        "/api/v1/transcription-jobs",
        _CAPABILITY_LINK,
        _CAPABILITY_LINK,
    ]
    assert server.admission_bodies == [{"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]}]
    assert len(server.idempotency_keys) == 1
    assert server.idempotency_keys[0] == server.idempotency_keys[0].lower()
    assert UUID(server.idempotency_keys[0]).version == 4

    url_identity, idempotency_key, capability_link = await _stored_coordination(
        migrated_coordination_database
    )
    assert url_identity == hmac.digest(
        _HMAC_SECRET.encode("utf-8"),
        _SUBMITTED_URL.encode("utf-8"),
        "sha256",
    )
    assert idempotency_key == UUID(server.idempotency_keys[0])
    assert capability_link == _CAPABILITY_LINK


async def test_repeated_public_extraction_resumes_existing_textify_job(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse durable coordination when a caller repeats an exact submitted URL."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper")

    async def sleeper(seconds: int) -> None:
        assert seconds == 0

    pipeline = await _create_pipeline(migrated_coordination_database, server, sleeper)
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            first_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )
            second_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert len(server.idempotency_keys) == 1


async def test_public_extraction_preserves_textify_youtube_captions(
    migrated_coordination_database: str,
) -> None:
    """Return Textify's YouTube-caption method without local transcription fallback."""
    server = _TextifyContractServer("youtube_captions")

    async def sleeper(seconds: int) -> None:
        assert seconds == 0

    pipeline = await _create_pipeline(migrated_coordination_database, server, sleeper)
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})

    assert response.status_code == 200
    assert response.json()["transcript"]["method"] == "youtube_captions"
    assert server.request_paths == ["/api/v1/transcription-jobs", _CAPABILITY_LINK]
