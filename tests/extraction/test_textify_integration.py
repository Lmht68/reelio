"""End-to-end HTTP coverage for Reelio's Textify-backed extraction path."""

import asyncio
import hmac
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.engine import make_url
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
_HMAC_SECRET = "test-coordination-hmac-secret"

_TextifyTerminalBehavior = Literal["succeeded", "failed", "cancelled", "job_not_found"]

_StoredCoordination = tuple[bytes, UUID, str | None]


@dataclass(slots=True)
class _TextifyJob:
    """Store one idempotently admitted Textify job's observable lifecycle."""

    request_body: object
    capability_link: str
    active_polls_remaining: int
    terminal_behavior: _TextifyTerminalBehavior


class _TextifyContractServer:
    """Serve the narrow Textify lifecycle consumed by Reelio."""

    def __init__(
        self,
        transcript_method: str,
        active_polls: int = 0,
        concurrent_admissions: int = 0,
        terminal_behaviors: tuple[_TextifyTerminalBehavior, ...] = (),
    ) -> None:
        """Configure ordered idempotent Textify job lifecycles.

        Args:
            transcript_method: Textify transcript acquisition method to return.
            active_polls: Per-job processing responses before terminal completion.
            concurrent_admissions: Number of admissions to release together.
            terminal_behaviors: Terminal behavior for each newly admitted job.
        """
        self.application = FastAPI()
        self.transcript_method = transcript_method
        self.active_polls = active_polls
        self.concurrent_admissions = concurrent_admissions
        self.terminal_behaviors = terminal_behaviors
        self.request_paths: list[str] = []
        self.admission_bodies: list[object] = []
        self.idempotency_keys: list[str] = []
        self.cancellation_paths: list[str] = []
        self.inspection_statuses: list[str] = []
        self._admission_release = asyncio.Event()
        self._jobs_by_idempotency_key: dict[str, _TextifyJob] = {}
        self._jobs_by_capability_link: dict[str, _TextifyJob] = {}
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
        self.application.add_api_route(
            "/api/v1/transcription-jobs/{job_id}/cancellation",
            self._cancel,
            methods=["PUT"],
        )

    @property
    def capability_links(self) -> list[str]:
        """Return accepted polling capabilities in admission order."""
        return [job.capability_link for job in self._jobs_by_idempotency_key.values()]

    @property
    def job_count(self) -> int:
        """Return the number of durably admitted fake Textify jobs."""
        return len(self._jobs_by_idempotency_key)

    async def _admit(self, request: Request) -> JSONResponse:
        """Idempotently accept a Textify job with a canonical capability link."""
        self.request_paths.append(request.url.path)
        request_body = await request.json()
        self.admission_bodies.append(request_body)
        idempotency_key = request.headers.get("Idempotency-Key")
        assert idempotency_key is not None
        self.idempotency_keys.append(idempotency_key)
        job = self._jobs_by_idempotency_key.get(idempotency_key)
        if job is None:
            capability_link = f"/api/v1/transcription-jobs/{uuid4()}"
            terminal_behavior = (
                self.terminal_behaviors[len(self._jobs_by_idempotency_key)]
                if len(self._jobs_by_idempotency_key) < len(self.terminal_behaviors)
                else "succeeded"
            )
            job = _TextifyJob(
                request_body=request_body,
                capability_link=capability_link,
                active_polls_remaining=self.active_polls,
                terminal_behavior=terminal_behavior,
            )
            self._jobs_by_idempotency_key[idempotency_key] = job
            self._jobs_by_capability_link[capability_link] = job
        else:
            assert job.request_body == request_body

        if self.concurrent_admissions:
            if len(self.idempotency_keys) == self.concurrent_admissions:
                self._admission_release.set()
            await self._admission_release.wait()

        return JSONResponse(
            status_code=202,
            content={"status": "queued", "links": self._links(job.capability_link)},
            headers={"Location": job.capability_link, "Retry-After": "0"},
        )

    async def _inspect(self, request: Request, job_id: str) -> JSONResponse:
        """Return an active, terminal, or missing job for the admitted capability."""
        capability_link = request.url.path
        assert job_id == capability_link.rsplit("/", maxsplit=1)[1]
        job = self._jobs_by_capability_link[capability_link]
        self.request_paths.append(capability_link)
        if job.active_polls_remaining > 0:
            job.active_polls_remaining -= 1
            self.inspection_statuses.append("processing")
            return JSONResponse(
                content={
                    "status": "processing",
                    "links": self._links(capability_link),
                },
                headers={"Retry-After": "0"},
            )
        if job.terminal_behavior == "job_not_found":
            self.inspection_statuses.append("job_not_found")
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "job_not_found",
                        "message": "The transcription job was not found.",
                    }
                },
            )
        self.inspection_statuses.append("finished")
        if job.terminal_behavior in {"failed", "cancelled"}:
            return JSONResponse(
                content={
                    "status": "finished",
                    "outcome": job.terminal_behavior,
                    "links": self._links(capability_link),
                    "error": {
                        "code": f"job_{job.terminal_behavior}",
                        "message": f"The transcription job {job.terminal_behavior}.",
                    },
                }
            )
        return JSONResponse(
            content={
                "status": "finished",
                "outcome": "succeeded",
                "links": self._links(capability_link),
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

    async def _cancel(self, request: Request, job_id: str) -> Response:
        """Record a cancellation request without changing the fake job state."""
        del job_id
        self.cancellation_paths.append(request.url.path)
        return Response(status_code=204)

    @staticmethod
    def _links(capability_link: str) -> dict[str, str]:
        """Build Textify's polling and additive cancellation capability links."""
        return {
            "self": capability_link,
            "cancel": f"{capability_link}/cancellation",
        }


class _ResponseLostAdmissionTransport(httpx.AsyncBaseTransport):
    """Drop exactly one already-processed Textify admission response."""

    def __init__(self, application: FastAPI) -> None:
        """Wrap an ASGI Textify application for ambiguous-admission coverage."""
        self._transport = httpx.ASGITransport(app=application)
        self._discard_next_admission = True

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward requests while discarding the first Textify admission response."""
        response = await self._transport.handle_async_request(request)
        if (
            self._discard_next_admission
            and request.method == "POST"
            and request.url.path == "/api/v1/transcription-jobs"
        ):
            self._discard_next_admission = False
            await response.aclose()
            raise httpx.ReadError(
                "Textify admission response was lost",
                request=request,
            )
        return response

    async def aclose(self) -> None:
        """Close the wrapped ASGI Textify transport."""
        await self._transport.aclose()


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
    textify_transport: httpx.AsyncBaseTransport | None = None,
) -> ExtractionPipeline:
    """Compose a real coordination-backed pipeline against a Textify ASGI server."""
    coordination_repository = await create_postgres_textify_coordination(
        _coordination_settings(database_url)
    )
    transport = (
        httpx.ASGITransport(app=server.application)
        if textify_transport is None
        else textify_transport
    )
    textify_client = httpx.AsyncClient(
        transport=transport,
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


async def _stored_coordinations(
    database_url: str,
) -> list[_StoredCoordination]:
    """Read durable Textify identities and links from public extraction requests."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT url_identity, idempotency_key, capability_link "
                        "FROM textify_coordination ORDER BY id"
                    )
                )
            ).all()
    finally:
        await engine.dispose()
    return [cast(_StoredCoordination, tuple(row)) for row in rows]


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
    capability_link = server.capability_links[0]
    assert server.request_paths == [
        "/api/v1/transcription-jobs",
        capability_link,
        capability_link,
    ]
    assert server.admission_bodies == [{"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]}]
    assert len(server.idempotency_keys) == 1
    assert server.idempotency_keys[0] == server.idempotency_keys[0].lower()
    assert UUID(server.idempotency_keys[0]).version == 4
    assert server.job_count == 1
    assert server.cancellation_paths == []

    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    url_identity, idempotency_key, stored_capability_link = stored_coordinations[0]
    assert url_identity == hmac.digest(
        _HMAC_SECRET.encode("utf-8"),
        _SUBMITTED_URL.encode("utf-8"),
        "sha256",
    )
    assert idempotency_key == UUID(server.idempotency_keys[0])
    assert stored_capability_link == capability_link


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
    assert server.job_count == 1


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
    assert server.request_paths == [
        "/api/v1/transcription-jobs",
        server.capability_links[0],
    ]


async def test_concurrent_public_extractions_share_one_textify_job(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Share one accepted Textify job across concurrent Reelio applications."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer(
        "faster_whisper",
        concurrent_admissions=2,
    )

    async def sleeper(seconds: int) -> None:
        assert seconds == 0
        await asyncio.sleep(0)

    first_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        sleeper,
    )
    second_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        sleeper,
    )
    first_application = create_app(pipeline_factory=lambda: _pipeline_factory(first_pipeline))
    second_application = create_app(pipeline_factory=lambda: _pipeline_factory(second_pipeline))

    async with (
        first_application.router.lifespan_context(first_application),
        second_application.router.lifespan_context(second_application),
    ):
        first_transport = httpx.ASGITransport(app=first_application)
        second_transport = httpx.ASGITransport(app=second_application)
        async with (
            httpx.AsyncClient(
                transport=first_transport,
                base_url="http://reelio.test",
            ) as first_client,
            httpx.AsyncClient(
                transport=second_transport,
                base_url="http://reelio.test",
            ) as second_client,
        ):
            first_response, second_response = await asyncio.gather(
                first_client.post("/api/extractions", json={"url": _SUBMITTED_URL}),
                second_client.post("/api/extractions", json={"url": _SUBMITTED_URL}),
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    _, idempotency_key, capability_link = stored_coordinations[0]
    assert capability_link == server.capability_links[0]
    assert server.job_count == 1
    assert len(server.idempotency_keys) == 2
    assert set(server.idempotency_keys) == {str(idempotency_key)}
    assert server.cancellation_paths == []


async def test_ambiguous_admission_reuses_the_durable_textify_identity(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume a response-lost admission through its committed idempotency key."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper")

    async def sleeper(seconds: int) -> None:
        assert seconds == 0

    lost_response_transport = _ResponseLostAdmissionTransport(server.application)
    first_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        sleeper,
        lost_response_transport,
    )
    first_application = create_app(pipeline_factory=lambda: _pipeline_factory(first_pipeline))

    async with first_application.router.lifespan_context(first_application):
        transport = httpx.ASGITransport(app=first_application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            first_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert first_response.status_code == 502
    assert first_response.json()["error"]["code"] == "transcription_failed"
    first_stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(first_stored_coordinations) == 1
    first_url_identity, first_idempotency_key, first_capability_link = first_stored_coordinations[0]
    assert first_capability_link is None

    resumed_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        sleeper,
    )
    resumed_application = create_app(pipeline_factory=lambda: _pipeline_factory(resumed_pipeline))

    async with resumed_application.router.lifespan_context(resumed_application):
        transport = httpx.ASGITransport(app=resumed_application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            resumed_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert resumed_response.status_code == 200
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert stored_coordinations == [
        (
            first_url_identity,
            first_idempotency_key,
            server.capability_links[0],
        )
    ]
    assert server.admission_bodies == [
        {"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]},
        {"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]},
    ]
    assert server.idempotency_keys == [str(first_idempotency_key)] * 2
    assert server.job_count == 1
    assert server.cancellation_paths == []


async def test_restart_after_request_cancellation_resumes_the_active_textify_job(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume an attached job after caller cancellation and application shutdown."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper", active_polls=1)
    first_poll_sleep_started = asyncio.Event()

    async def blocking_sleeper(seconds: int) -> None:
        assert seconds == 0
        first_poll_sleep_started.set()
        await asyncio.Event().wait()

    first_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        blocking_sleeper,
    )
    first_application = create_app(pipeline_factory=lambda: _pipeline_factory(first_pipeline))

    async with first_application.router.lifespan_context(first_application):
        transport = httpx.ASGITransport(app=first_application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            first_request = asyncio.create_task(
                client.post("/api/extractions", json={"url": _SUBMITTED_URL})
            )
            await first_poll_sleep_started.wait()
            stored_coordinations = await _stored_coordinations(migrated_coordination_database)
            assert len(stored_coordinations) == 1
            assert stored_coordinations[0][2] == server.capability_links[0]
            first_request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_request

    resumed_waits: list[int] = []

    async def resumed_sleeper(seconds: int) -> None:
        resumed_waits.append(seconds)

    resumed_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        resumed_sleeper,
    )
    resumed_application = create_app(pipeline_factory=lambda: _pipeline_factory(resumed_pipeline))

    async with resumed_application.router.lifespan_context(resumed_application):
        transport = httpx.ASGITransport(app=resumed_application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            resumed_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert resumed_response.status_code == 200
    assert resumed_waits == [0, 0]
    assert server.inspection_statuses == ["processing", "finished"]
    assert server.request_paths == [
        "/api/v1/transcription-jobs",
        server.capability_links[0],
        server.capability_links[0],
    ]
    assert server.job_count == 1
    assert len(server.idempotency_keys) == 1
    assert server.cancellation_paths == []


async def test_distinct_exact_urls_receive_distinct_textify_coordination_identities(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep distinct submitted URL strings separate despite one Source projection."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    shared_source_url = f"{_SUBMITTED_URL}&feature=shared"
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
                json={"url": shared_source_url},
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["source"]["platform"] == "youtube"
    assert second_response.json()["source"]["platform"] == "youtube"
    assert first_response.json()["source"]["video_id"] == "dQw4w9WgXcQ"
    assert second_response.json()["source"]["video_id"] == "dQw4w9WgXcQ"
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 2
    expected_identities = {
        hmac.digest(
            _HMAC_SECRET.encode("utf-8"),
            submitted_url.encode("utf-8"),
            "sha256",
        )
        for submitted_url in (_SUBMITTED_URL, shared_source_url)
    }
    assert {url_identity for url_identity, _, _ in stored_coordinations} == (expected_identities)
    assert all(len(url_identity) == 32 for url_identity, _, _ in stored_coordinations)
    assert len({idempotency_key for _, idempotency_key, _ in stored_coordinations}) == 2
    assert server.job_count == 2
    assert len(set(server.idempotency_keys)) == 2
    assert server.cancellation_paths == []


@pytest.mark.parametrize("terminal_behavior", ["failed", "cancelled"])
async def test_public_extraction_retires_terminal_textify_coordination(
    migrated_coordination_database: str,
    terminal_behavior: Literal["failed", "cancelled"],
) -> None:
    """Retire failed or cancelled Textify coordination before a later admission."""
    server = _TextifyContractServer(
        "faster_whisper",
        terminal_behaviors=(terminal_behavior,),
    )

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
            first_response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})
            stored_after_terminal = await _stored_coordinations(migrated_coordination_database)
            second_response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})

    assert first_response.status_code == 502
    assert first_response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }
    assert stored_after_terminal == []
    assert second_response.status_code == 200
    assert server.job_count == 2
    assert len(server.idempotency_keys) == 2
    assert UUID(server.idempotency_keys[0]).version == 4
    assert UUID(server.idempotency_keys[1]).version == 4
    assert server.idempotency_keys[0] != server.idempotency_keys[1]


async def test_public_extraction_retires_stale_capability_without_resubmitting(
    migrated_coordination_database: str,
) -> None:
    """Return temporary unavailability before a later request admits replacement work."""
    server = _TextifyContractServer(
        "faster_whisper",
        terminal_behaviors=("job_not_found",),
    )

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
            stale_response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})
            stored_after_stale_response = await _stored_coordinations(
                migrated_coordination_database
            )
            first_capability_link = server.capability_links[0]
            replacement_response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert stale_response.status_code == 503
    assert stale_response.headers["Retry-After"] == "2"
    assert stale_response.json() == {
        "error": {
            "code": "transcription_unavailable",
            "message": "Transcription is temporarily unavailable.",
        }
    }
    assert stored_after_stale_response == []
    assert server.request_paths[:2] == [
        "/api/v1/transcription-jobs",
        first_capability_link,
    ]
    assert server.admission_bodies[:1] == [
        {"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]}
    ]
    assert replacement_response.status_code == 200
    assert server.job_count == 2
    assert len(server.idempotency_keys) == 2
    assert UUID(server.idempotency_keys[0]).version == 4
    assert UUID(server.idempotency_keys[1]).version == 4
    assert server.idempotency_keys[0] != server.idempotency_keys[1]


async def _insert_coordination_records(
    database_url: str,
    coordination_records: list[dict[str, object]],
) -> None:
    """Insert test-only durable coordination records with explicit expiration."""
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO textify_coordination "
                    "(url_identity, idempotency_key, capability_link, created_at, "
                    "updated_at, expires_at) "
                    "VALUES (:url_identity, :idempotency_key, :capability_link, "
                    ":created_at, :updated_at, :expires_at)"
                ),
                coordination_records,
            )
    finally:
        await engine.dispose()


async def test_public_extraction_replaces_expired_matching_coordination_after_bounded_cleanup(
    migrated_coordination_database: str,
) -> None:
    """Clear one targeted expired record without draining more than one cleanup batch."""
    server = _TextifyContractServer("faster_whisper")

    async def sleeper(seconds: int) -> None:
        assert seconds == 0

    expiration_start = datetime.now(UTC) - timedelta(hours=1)
    unrelated_records: list[dict[str, object]] = []
    for index in range(101):
        expires_at = expiration_start + timedelta(seconds=index)
        created_at = expires_at - timedelta(seconds=1)
        unrelated_url = f"https://example.invalid/expired/{index}"
        unrelated_records.append(
            {
                "url_identity": hmac.digest(
                    _HMAC_SECRET.encode("utf-8"),
                    unrelated_url.encode("utf-8"),
                    "sha256",
                ),
                "idempotency_key": uuid4(),
                "capability_link": None,
                "created_at": created_at,
                "updated_at": created_at,
                "expires_at": expires_at,
            }
        )
    stale_idempotency_key = uuid4()
    stale_expires_at = expiration_start + timedelta(seconds=102)
    stale_created_at = stale_expires_at - timedelta(seconds=1)
    target_identity = hmac.digest(
        _HMAC_SECRET.encode("utf-8"),
        _SUBMITTED_URL.encode("utf-8"),
        "sha256",
    )
    await _insert_coordination_records(
        migrated_coordination_database,
        [
            *unrelated_records,
            {
                "url_identity": target_identity,
                "idempotency_key": stale_idempotency_key,
                "capability_link": None,
                "created_at": stale_created_at,
                "updated_at": stale_created_at,
                "expires_at": stale_expires_at,
            },
        ],
    )

    pipeline = await _create_pipeline(migrated_coordination_database, server, sleeper)
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})

    verification_engine = create_async_engine(migrated_coordination_database)
    try:
        async with verification_engine.connect() as connection:
            expired_count = await connection.scalar(
                text(
                    "SELECT count(*) FROM textify_coordination "
                    "WHERE expires_at <= CURRENT_TIMESTAMP"
                )
            )
            replacement = cast(
                tuple[UUID, datetime, datetime],
                (
                    await connection.execute(
                        text(
                            "SELECT idempotency_key, created_at, expires_at "
                            "FROM textify_coordination WHERE url_identity = :url_identity"
                        ),
                        {"url_identity": target_identity},
                    )
                ).one(),
            )
    finally:
        await verification_engine.dispose()

    replacement_idempotency_key, replacement_created_at, replacement_expires_at = replacement
    assert response.status_code == 200
    assert expired_count == 1
    assert replacement_idempotency_key != stale_idempotency_key
    assert replacement_idempotency_key.version == 4
    assert replacement_expires_at == replacement_created_at + timedelta(days=1)
    assert replacement_expires_at - replacement_created_at <= timedelta(days=1)
    assert server.idempotency_keys == [str(replacement_idempotency_key)]


async def test_public_extraction_returns_unavailable_before_textify_when_postgres_is_lost(
    migrated_coordination_database: str,
) -> None:
    """Fail closed when the verified coordination database disappears at runtime."""
    server = _TextifyContractServer("faster_whisper")

    async def sleeper(seconds: int) -> None:
        assert seconds == 0

    pipeline = await _create_pipeline(migrated_coordination_database, server, sleeper)
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))
    coordination_url = make_url(migrated_coordination_database)
    database_name = coordination_url.database
    assert database_name is not None
    admin_url = make_url(os.environ["REELIO_TEST_DATABASE_URL"])
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with application.router.lifespan_context(application):
            async with admin_engine.connect() as connection:
                await connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :database_name "
                        "AND pid <> pg_backend_pid()"
                    ),
                    {"database_name": database_name},
                )
                await connection.execute(text(f'DROP DATABASE "{database_name}"'))

            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://reelio.test",
            ) as client:
                response = await client.post("/api/extractions", json={"url": _SUBMITTED_URL})
    finally:
        await admin_engine.dispose()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert response.json() == {
        "error": {
            "code": "transcription_unavailable",
            "message": "Transcription is temporarily unavailable.",
        }
    }
    assert server.request_paths == []
    assert server.idempotency_keys == []
