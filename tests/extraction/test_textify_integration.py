"""End-to-end HTTP coverage for Reelio's Textify-backed extraction path."""

import asyncio
import hmac
import json
import logging
import os
import time
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
from reelio.extraction.services.textify import (
    TextifyConfig,
    TextifySourceTranscriptProvider,
)
from reelio.main import create_app
from tests.extraction.fakes import FakeInterpretationService, FakeResultAggregator

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_HMAC_SECRET = "test-coordination-hmac-secret"

_TextifyTerminalBehavior = Literal["succeeded", "failed", "cancelled", "job_not_found"]


@dataclass(frozen=True, slots=True)
class _TextifyHttpError:
    """Describe one documented or malformed Textify HTTP error."""

    status_code: int
    code: str
    retry_after: int | None = None
    message: str = "Textify provider error."


_StoredCoordination = tuple[bytes, UUID, str | None]


@dataclass(slots=True)
class _TextifyJob:
    """Store one idempotently admitted Textify job's observable lifecycle."""

    request_body: object
    public_id: UUID
    capability_link: str
    submitted_at: datetime
    active_polls_remaining: int
    terminal_behavior: _TextifyTerminalBehavior
    terminal_error_code: str


class _TextifyContractServer:
    """Serve the version-one Textify lifecycle consumed by Reelio."""

    def __init__(
        self,
        transcript_method: str,
        active_polls: int = 0,
        concurrent_admissions: int = 0,
        terminal_behaviors: tuple[_TextifyTerminalBehavior, ...] = (),
        terminal_error_codes: tuple[str, ...] = (),
        admission_errors: tuple[_TextifyHttpError, ...] = (),
        polling_errors: tuple[_TextifyHttpError, ...] = (),
        active_retry_after: int = 0,
        response_transform: Callable[[str, dict[str, object]], object] | None = None,
        source_url: str = _SUBMITTED_URL,
        transcript_text: str = "Textify integration transcript.",
        terminal_error_message: str = "Textify terminal failure.",
    ) -> None:
        """Configure ordered idempotent Textify job lifecycles.

        Args:
            transcript_method: Textify transcript acquisition method to return.
            active_polls: Per-job processing responses before terminal completion.
            concurrent_admissions: Number of admissions to release together.
            terminal_behaviors: Terminal behavior for each newly admitted job.
            terminal_error_codes: Failed-job code for each newly admitted job.
            admission_errors: Ordered errors returned before any job admission.
            polling_errors: Ordered errors returned before lifecycle inspection.
            active_retry_after: Retry instruction for active lifecycle responses.
            response_transform: Optional test-only response mutation.
            source_url: Source URL returned by successful jobs.
            transcript_text: Transcript text returned by successful jobs.
            terminal_error_message: Failed-job provider message.
        """
        self.application = FastAPI()
        self.transcript_method = transcript_method
        self.active_polls = active_polls
        self.concurrent_admissions = concurrent_admissions
        self.terminal_behaviors = terminal_behaviors
        self.terminal_error_codes = terminal_error_codes
        self.active_retry_after = active_retry_after
        self.response_transform = response_transform
        self.source_url = source_url
        self.transcript_text = transcript_text
        self.terminal_error_message = terminal_error_message
        self.request_paths: list[str] = []
        self.admission_bodies: list[object] = []
        self.admission_body_bytes: list[bytes] = []
        self.idempotency_keys: list[str] = []
        self.cancellation_paths: list[str] = []
        self.inspection_statuses: list[str] = []
        self._admission_errors = list(admission_errors)
        self._polling_errors = list(polling_errors)
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
        request_body_bytes = await request.body()
        request_body = json.loads(request_body_bytes)
        self.admission_bodies.append(request_body)
        self.admission_body_bytes.append(request_body_bytes)
        idempotency_key = request.headers.get("Idempotency-Key")
        assert idempotency_key is not None
        self.idempotency_keys.append(idempotency_key)
        if self._admission_errors:
            return self._error_response(
                "admission_error",
                self._admission_errors.pop(0),
            )

        job = self._jobs_by_idempotency_key.get(idempotency_key)
        if job is None:
            job_index = len(self._jobs_by_idempotency_key)
            public_id = uuid4()
            capability_link = f"/api/v1/transcription-jobs/{public_id}"
            terminal_behavior = (
                self.terminal_behaviors[job_index]
                if job_index < len(self.terminal_behaviors)
                else "succeeded"
            )
            terminal_error_code = (
                self.terminal_error_codes[job_index]
                if job_index < len(self.terminal_error_codes)
                else "transcription_failed"
            )
            job = _TextifyJob(
                request_body=request_body,
                public_id=public_id,
                capability_link=capability_link,
                submitted_at=datetime(2026, 10, 14, tzinfo=UTC),
                active_polls_remaining=self.active_polls,
                terminal_behavior=terminal_behavior,
                terminal_error_code=terminal_error_code,
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
            content=self._content(
                "admission",
                {
                    "id": str(job.public_id),
                    "status": "queued",
                    "submitted_at": self._timestamp(job.submitted_at),
                    "links": self._active_links(job.capability_link),
                },
            ),
            headers={"Location": job.capability_link, "Retry-After": "0"},
        )

    async def _inspect(self, request: Request, job_id: str) -> JSONResponse:
        """Return an active, terminal, or missing job for the admitted capability."""
        capability_link = request.url.path
        assert job_id == capability_link.rsplit("/", maxsplit=1)[1]
        job = self._jobs_by_capability_link[capability_link]
        self.request_paths.append(capability_link)
        if self._polling_errors:
            return self._error_response("polling_error", self._polling_errors.pop(0))
        if job.active_polls_remaining > 0:
            job.active_polls_remaining -= 1
            self.inspection_statuses.append("processing")
            return JSONResponse(
                content=self._content(
                    "processing",
                    {
                        "id": str(job.public_id),
                        "status": "processing",
                        "submitted_at": self._timestamp(job.submitted_at),
                        "started_at": self._timestamp(job.submitted_at + timedelta(seconds=1)),
                        "cancellation_requested": False,
                        "links": self._active_links(capability_link),
                    },
                ),
                headers={"Retry-After": str(self.active_retry_after)},
            )
        if job.terminal_behavior == "job_not_found":
            self.inspection_statuses.append("job_not_found")
            return self._error_response(
                "polling_error",
                _TextifyHttpError(404, "job_not_found"),
            )
        self.inspection_statuses.append("finished")
        if job.terminal_behavior == "failed":
            return JSONResponse(
                content=self._content(
                    "failed",
                    {
                        "id": str(job.public_id),
                        "status": "finished",
                        "outcome": "failed",
                        "submitted_at": self._timestamp(job.submitted_at),
                        "started_at": None,
                        "finished_at": self._timestamp(job.submitted_at + timedelta(seconds=2)),
                        "error": {
                            "code": job.terminal_error_code,
                            "message": self.terminal_error_message,
                        },
                        "links": self._finished_links(capability_link),
                    },
                )
            )
        if job.terminal_behavior == "cancelled":
            return JSONResponse(
                content=self._content(
                    "cancelled",
                    {
                        "id": str(job.public_id),
                        "status": "finished",
                        "outcome": "cancelled",
                        "submitted_at": self._timestamp(job.submitted_at),
                        "started_at": None,
                        "finished_at": self._timestamp(job.submitted_at + timedelta(seconds=2)),
                        "links": self._finished_links(capability_link),
                    },
                )
            )
        return JSONResponse(
            content=self._content(
                "succeeded",
                {
                    "id": str(job.public_id),
                    "status": "finished",
                    "outcome": "succeeded",
                    "submitted_at": self._timestamp(job.submitted_at),
                    "started_at": self._timestamp(job.submitted_at + timedelta(seconds=1)),
                    "finished_at": self._timestamp(job.submitted_at + timedelta(seconds=2)),
                    "links": self._finished_links(capability_link),
                    "result": {
                        "source": {
                            "platform": "youtube",
                            "video_id": "dQw4w9WgXcQ",
                            "url": self.source_url,
                            "title": "Textify integration video",
                            "description": "A Textify integration description.",
                            "channel": "Textify integration channel",
                            "duration_seconds": 42,
                        },
                        "transcript": {
                            "method": self.transcript_method,
                            "language": "en",
                            "text": self.transcript_text,
                        },
                    },
                },
            )
        )

    async def _cancel(self, request: Request, job_id: str) -> Response:
        """Record a cancellation request without changing the fake job state."""
        del job_id
        self.cancellation_paths.append(request.url.path)
        return Response(status_code=204)

    def _error_response(
        self,
        phase: str,
        error: _TextifyHttpError,
    ) -> JSONResponse:
        """Build one configurable Textify error response."""
        headers = {"Retry-After": str(error.retry_after)} if error.retry_after is not None else None
        return JSONResponse(
            status_code=error.status_code,
            content=self._content(
                phase,
                {"error": {"code": error.code, "message": error.message}},
            ),
            headers=headers,
        )

    def _content(self, phase: str, payload: dict[str, object]) -> object:
        """Apply an optional test-only response transformation."""
        if self.response_transform is None:
            return payload
        return self.response_transform(phase, payload)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        """Serialize one UTC lifecycle timestamp in the Textify wire format."""
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _active_links(capability_link: str) -> dict[str, str]:
        """Build active polling and cancellation capabilities."""
        return {
            "self": capability_link,
            "cancel": f"{capability_link}/cancellation",
        }

    @staticmethod
    def _finished_links(capability_link: str) -> dict[str, str]:
        """Build a retained terminal polling capability."""
        return {"self": capability_link}


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


class _RetryClock:
    """Advance deterministic monotonic time through injected retry sleeps."""

    def __init__(self) -> None:
        """Initialize a zero-based clock with no observed waits."""
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        """Return the current deterministic monotonic time."""
        return self.now

    async def sleep(self, seconds: float) -> None:
        """Record and advance one requested retry delay."""
        self.waits.append(seconds)
        self.now += seconds


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Raise ordered transport errors before forwarding to a Textify application."""

    def __init__(
        self,
        application: FastAPI,
        error_types: list[type[httpx.TransportError]],
    ) -> None:
        """Initialize a forwarding transport with deterministic initial failures."""
        self._transport = httpx.ASGITransport(app=application)
        self._error_types = error_types

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Raise the next configured error or forward the Textify request."""
        if self._error_types:
            error_type = self._error_types.pop(0)
            raise error_type("Scripted Textify transport failure", request=request)
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped ASGI transport."""
        await self._transport.aclose()


def _coordination_settings(database_url: str) -> CoordinationConfig:
    """Create coordination settings without consulting environment files."""
    constructor = cast(Callable[..., CoordinationConfig], CoordinationConfig)
    return constructor(
        _env_file=None,
        database_url=database_url,
        hmac_secret=_HMAC_SECRET,
    )


def _textify_settings(**values: object) -> TextifyConfig:
    """Create Textify settings without reading environment files."""
    constructor = cast(Callable[..., TextifyConfig], TextifyConfig)
    return constructor(_env_file=None, **values)


async def _create_pipeline(
    database_url: str,
    server: _TextifyContractServer,
    sleeper: Callable[[float], Awaitable[None]],
    textify_transport: httpx.AsyncBaseTransport | None = None,
    textify_settings: TextifyConfig | None = None,
    clock: Callable[[], float] = time.monotonic,
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
        settings=textify_settings or _textify_settings(),
        clock=clock,
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


async def _request_public_extraction(
    pipeline: ExtractionPipeline,
    submitted_url: str = _SUBMITTED_URL,
    adapter_log_handler: logging.Handler | None = None,
) -> httpx.Response:
    """Exercise one public Extraction request through a lifespan-owned pipeline.

    Args:
        pipeline: Extraction pipeline owned by the test application lifespan.
        submitted_url: Source URL sent through the public Extraction route.
        adapter_log_handler: Optional handler attached after application logging setup.

    Returns:
        Public Extraction HTTP response.
    """
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))
    adapter_logger = logging.getLogger("reelio.extraction.services.textify")
    if adapter_log_handler is not None:
        adapter_logger.addHandler(adapter_log_handler)
    try:
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://reelio.test",
            ) as client:
                return await client.post("/api/extractions", json={"url": submitted_url})
    finally:
        if adapter_log_handler is not None:
            adapter_logger.removeHandler(adapter_log_handler)


class _MalformedPollingResponseTransport(httpx.AsyncBaseTransport):
    """Replace the first Textify polling response with malformed JSON."""

    def __init__(self, application: FastAPI) -> None:
        """Initialize a forwarding transport with one malformed poll response."""
        self._transport = httpx.ASGITransport(app=application)
        self._replace_next_poll = True

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward admission then return malformed bytes for the first poll."""
        if self._replace_next_poll and request.method == "GET":
            self._replace_next_poll = False
            return httpx.Response(
                status_code=200,
                content=b"{",
                request=request,
            )
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped ASGI transport."""
        await self._transport.aclose()


def _remove_admission_timestamp(
    phase: str,
    payload: dict[str, object],
) -> object:
    """Remove one required field from an otherwise valid admission response."""
    if phase == "admission":
        del payload["submitted_at"]
    return payload


def _replace_terminal_outcome(
    phase: str,
    payload: dict[str, object],
) -> object:
    """Return an unsupported terminal outcome variant."""
    if phase == "succeeded":
        payload["outcome"] = "unknown"
    return payload


def _replace_terminal_status(
    phase: str,
    payload: dict[str, object],
) -> object:
    """Return an unsupported lifecycle status variant."""
    if phase == "succeeded":
        payload["status"] = "unknown"
    return payload


def _replace_admission_self_link(
    phase: str,
    payload: dict[str, object],
) -> object:
    """Return an unsafe admission polling capability."""
    if phase == "admission":
        links = cast(dict[str, object], payload["links"])
        links["self"] = "/api/v1/transcription-jobs/private-self-link-sentinel"
    return payload


def _replace_admission_cancel_link(
    phase: str,
    payload: dict[str, object],
) -> object:
    """Return an unsafe admission cancellation capability."""
    if phase == "admission":
        links = cast(dict[str, object], payload["links"])
        links["cancel"] = "/api/v1/transcription-jobs/private-cancel-link-sentinel"
    return payload


def _add_response_fields(phase: str, payload: dict[str, object]) -> object:
    """Add unknown fields at every consumed Textify model level."""
    payload["provider_addition"] = {"nested": "ignored"}
    links = payload.get("links")
    if isinstance(links, dict):
        links["provider_addition"] = "ignored"
    if phase != "succeeded":
        return payload
    result = cast(dict[str, object], payload["result"])
    result["provider_addition"] = "ignored"
    source = cast(dict[str, object], result["source"])
    source["provider_addition"] = "ignored"
    transcript = cast(dict[str, object], result["transcript"])
    transcript["provider_addition"] = "ignored"
    return payload


async def test_public_extraction_uses_textify_whisper_and_durable_coordination(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a public extraction after durable admission and Textify completion."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper", active_polls=1)
    waits: list[float] = []

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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


async def test_ambiguous_admission_retries_the_durable_textify_identity(
    migrated_coordination_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a response-lost admission with its committed idempotency key."""
    monkeypatch.setattr(app_settings, "environment", Environment.LOCAL)
    server = _TextifyContractServer("faster_whisper")
    waits: list[float] = []

    async def sleeper(seconds: float) -> None:
        waits.append(seconds)

    lost_response_transport = _ResponseLostAdmissionTransport(server.application)
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        sleeper,
        lost_response_transport,
    )
    application = create_app(pipeline_factory=lambda: _pipeline_factory(pipeline))

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://reelio.test",
        ) as client:
            response = await client.post(
                "/api/extractions",
                json={"url": _SUBMITTED_URL},
            )

    assert response.status_code == 200
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    _, idempotency_key, capability_link = stored_coordinations[0]
    assert capability_link == server.capability_links[0]
    assert waits == [1.0, 0.0]
    assert server.admission_bodies == [
        {"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]},
        {"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]},
    ]
    assert server.admission_body_bytes[0] == server.admission_body_bytes[1]
    assert server.idempotency_keys == [str(idempotency_key)] * 2
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

    async def blocking_sleeper(seconds: float) -> None:
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

    resumed_waits: list[float] = []

    async def resumed_sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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

    async def sleeper(seconds: float) -> None:
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


async def test_public_extraction_recovers_admission_capacity_within_budget(
    migrated_coordination_database: str,
) -> None:
    """Retry capacity rejections through the public Extraction route."""
    server = _TextifyContractServer(
        "faster_whisper",
        admission_errors=(
            _TextifyHttpError(503, "transcription_capacity_exceeded", retry_after=2),
            _TextifyHttpError(503, "transcription_capacity_exceeded", retry_after=3),
        ),
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        textify_settings=_textify_settings(
            admission_capacity_retry_budget_seconds=10.0,
        ),
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 200
    assert clock.waits == [2.0, 3.0, 0.0]
    assert (
        server.admission_bodies == [{"url": _SUBMITTED_URL, "exclude": ["transcript.segments"]}] * 3
    )
    assert server.admission_body_bytes[0] == server.admission_body_bytes[1]
    assert server.admission_body_bytes[1] == server.admission_body_bytes[2]
    assert len(set(server.idempotency_keys)) == 1
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] == server.capability_links[0]


async def test_public_extraction_stops_capacity_retry_before_budget_overrun(
    migrated_coordination_database: str,
) -> None:
    """Return Reelio's retryable contract without sleeping beyond capacity budget."""
    server = _TextifyContractServer(
        "faster_whisper",
        admission_errors=(
            _TextifyHttpError(503, "transcription_capacity_exceeded", retry_after=11),
        ),
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        textify_settings=_textify_settings(
            admission_capacity_retry_budget_seconds=10.0,
        ),
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert response.json() == {
        "error": {
            "code": "transcription_unavailable",
            "message": "Transcription is temporarily unavailable.",
        }
    }
    assert clock.waits == []
    assert server.job_count == 0
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] is None


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_code"),
    [
        (httpx.ReadTimeout, 504, "pipeline_timeout"),
        (httpx.ConnectError, 502, "transcription_failed"),
    ],
)
async def test_public_extraction_preserves_pending_admission_after_transport_exhaustion(
    migrated_coordination_database: str,
    error_type: type[httpx.TransportError],
    expected_status: int,
    expected_code: str,
) -> None:
    """Keep pending coordination after exhausted Textify communication retries."""
    server = _TextifyContractServer("faster_whisper")
    clock = _RetryClock()
    failing_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        _ScriptedTransport(server.application, [error_type, error_type]),
        _textify_settings(transport_retry_budget_seconds=2.0),
        clock,
    )

    failed_response = await _request_public_extraction(failing_pipeline)

    assert failed_response.status_code == expected_status
    assert failed_response.json()["error"]["code"] == expected_code
    assert clock.waits == [1.0, 1.0]
    pending_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(pending_coordinations) == 1
    assert pending_coordinations[0][2] is None

    recovery_clock = _RetryClock()
    recovery_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        recovery_clock.sleep,
        clock=recovery_clock,
    )
    recovery_response = await _request_public_extraction(recovery_pipeline)

    assert recovery_response.status_code == 200
    recovered_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert recovered_coordinations[0][:2] == pending_coordinations[0][:2]
    assert recovered_coordinations[0][2] == server.capability_links[0]


async def test_public_extraction_recovers_after_polling_job_store_outage(
    migrated_coordination_database: str,
) -> None:
    """Keep accepted coordination while Textify job storage is unavailable."""
    server = _TextifyContractServer(
        "faster_whisper",
        polling_errors=(_TextifyHttpError(503, "job_store_unavailable"),),
    )
    first_clock = _RetryClock()
    first_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        first_clock.sleep,
        clock=first_clock,
    )

    outage_response = await _request_public_extraction(first_pipeline)

    assert outage_response.status_code == 503
    assert outage_response.headers["Retry-After"] == "2"
    accepted_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(accepted_coordinations) == 1
    assert accepted_coordinations[0][2] == server.capability_links[0]

    recovery_clock = _RetryClock()
    recovery_pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        recovery_clock.sleep,
        clock=recovery_clock,
    )
    recovery_response = await _request_public_extraction(recovery_pipeline)

    assert recovery_response.status_code == 200
    assert await _stored_coordinations(migrated_coordination_database) == (accepted_coordinations)
    assert server.job_count == 1
    assert len(server.idempotency_keys) == 1


async def test_public_extraction_waits_for_reachable_job_beyond_transport_budget(
    migrated_coordination_database: str,
) -> None:
    """Continue active polling after cumulative waits exceed transport retry budget."""
    server = _TextifyContractServer(
        "faster_whisper",
        active_polls=31,
        active_retry_after=1,
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        textify_settings=_textify_settings(transport_retry_budget_seconds=30.0),
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 200
    assert clock.now == 31.0
    assert clock.waits == [0.0, *([1.0] * 31)]
    assert server.inspection_statuses == ["processing"] * 31 + ["finished"]


@pytest.mark.parametrize(
    ("provider_code", "expected_status", "expected_code", "expected_message"),
    [
        ("invalid_url", 400, "invalid_source", "Invalid source URL."),
        ("unsupported_media", 400, "invalid_source", "Invalid source URL."),
        (
            "unsupported_platform",
            400,
            "unsupported_platform",
            "Only YouTube, Instagram, Facebook, TikTok, and X URLs are supported.",
        ),
        ("unsupported_content", 404, "source_unavailable", "Source is unavailable."),
        (
            "video_too_long",
            413,
            "duration_limit_exceeded",
            "Video exceeds the configured duration limit.",
        ),
        (
            "invalid_media_duration",
            502,
            "metadata_provider_failed",
            "Unable to retrieve source metadata.",
        ),
        (
            "metadata_retrieval_failed",
            502,
            "metadata_provider_failed",
            "Unable to retrieve source metadata.",
        ),
        (
            "audio_download_failed",
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            "no_usable_transcript",
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            "transcription_failed",
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            "worker_interrupted",
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            "internal_error",
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            "metadata_timeout",
            504,
            "pipeline_timeout",
            "Source metadata acquisition timed out.",
        ),
        (
            "queue_timeout",
            504,
            "pipeline_timeout",
            "Transcript acquisition timed out.",
        ),
        (
            "audio_download_timeout",
            504,
            "pipeline_timeout",
            "Transcript acquisition timed out.",
        ),
        (
            "transcription_timeout",
            504,
            "pipeline_timeout",
            "Transcript acquisition timed out.",
        ),
        (
            "transcription_capacity_exceeded",
            503,
            "transcription_unavailable",
            "Transcription is temporarily unavailable.",
        ),
        (
            "job_store_unavailable",
            503,
            "transcription_unavailable",
            "Transcription is temporarily unavailable.",
        ),
    ],
)
async def test_public_extraction_maps_terminal_textify_error_codes(
    migrated_coordination_database: str,
    provider_code: str,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """Translate terminal Textify codes without exposing provider wording."""
    private_provider_message = "private Textify terminal error sentinel"
    server = _TextifyContractServer(
        "faster_whisper",
        terminal_behaviors=("failed",),
        terminal_error_codes=(provider_code,),
        terminal_error_message=private_provider_message,
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == expected_status
    assert response.json() == {
        "error": {
            "code": expected_code,
            "message": expected_message,
        }
    }
    if expected_status == 503:
        assert response.headers["Retry-After"] == "2"
    assert private_provider_message not in response.text
    assert await _stored_coordinations(migrated_coordination_database) == []


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_code", "expected_message"),
    [
        (
            _TextifyHttpError(400, "invalid_url"),
            400,
            "invalid_source",
            "Invalid source URL.",
        ),
        (
            _TextifyHttpError(400, "unsupported_platform"),
            400,
            "unsupported_platform",
            "Only YouTube, Instagram, Facebook, TikTok, and X URLs are supported.",
        ),
        (
            _TextifyHttpError(503, "job_store_unavailable"),
            503,
            "transcription_unavailable",
            "Transcription is temporarily unavailable.",
        ),
        (
            _TextifyHttpError(422, "invalid_request"),
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            _TextifyHttpError(409, "idempotency_conflict"),
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
        (
            _TextifyHttpError(500, "internal_error"),
            502,
            "transcription_failed",
            "Transcript is unavailable for this video.",
        ),
    ],
)
async def test_public_extraction_maps_documented_admission_errors(
    migrated_coordination_database: str,
    provider_error: _TextifyHttpError,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """Preserve Reelio errors for documented Textify admission rejections."""
    server = _TextifyContractServer(
        "faster_whisper",
        admission_errors=(provider_error,),
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == expected_status
    assert response.json() == {
        "error": {
            "code": expected_code,
            "message": expected_message,
        }
    }
    assert provider_error.message not in response.text
    if expected_status == 503:
        assert response.headers["Retry-After"] == "2"
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] is None


@pytest.mark.parametrize(
    "provider_error",
    [
        _TextifyHttpError(422, "invalid_request"),
        _TextifyHttpError(500, "internal_error"),
    ],
)
async def test_public_extraction_keeps_accepted_coordination_for_polling_failures(
    migrated_coordination_database: str,
    provider_error: _TextifyHttpError,
) -> None:
    """Leave accepted coordination intact after non-terminal polling failures."""
    server = _TextifyContractServer(
        "faster_whisper",
        polling_errors=(provider_error,),
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] == server.capability_links[0]


@pytest.mark.parametrize(
    ("response_transform", "has_capability_link"),
    [
        (_remove_admission_timestamp, False),
        (_replace_terminal_outcome, True),
        (_replace_terminal_status, True),
        (_replace_admission_self_link, False),
        (_replace_admission_cancel_link, False),
    ],
)
async def test_public_extraction_rejects_malformed_textify_lifecycle_responses(
    migrated_coordination_database: str,
    response_transform: Callable[[str, dict[str, object]], object],
    has_capability_link: bool,
) -> None:
    """Reject missing fields, unsupported variants, and unsafe capabilities."""
    server = _TextifyContractServer(
        "faster_whisper",
        response_transform=response_transform,
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert (stored_coordinations[0][2] is not None) is has_capability_link


async def test_public_extraction_rejects_malformed_textify_json(
    migrated_coordination_database: str,
) -> None:
    """Fail closed on malformed JSON while retaining accepted coordination."""
    server = _TextifyContractServer("faster_whisper")
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        _MalformedPollingResponseTransport(server.application),
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] == server.capability_links[0]


async def test_public_extraction_rejects_invalid_terminal_error_context(
    migrated_coordination_database: str,
) -> None:
    """Fail closed when an admission-only error is embedded in a failed job."""
    server = _TextifyContractServer(
        "faster_whisper",
        terminal_behaviors=("failed",),
        terminal_error_codes=("invalid_request",),
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "transcription_failed"
    stored_coordinations = await _stored_coordinations(migrated_coordination_database)
    assert len(stored_coordinations) == 1
    assert stored_coordinations[0][2] == server.capability_links[0]


async def test_public_extraction_ignores_additive_textify_response_fields(
    migrated_coordination_database: str,
) -> None:
    """Accept forward-compatible additions while retaining strict known fields."""
    server = _TextifyContractServer(
        "faster_whisper",
        response_transform=_add_response_fields,
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )

    response = await _request_public_extraction(pipeline)

    assert response.status_code == 200
    assert response.json()["source"]["url"] == _SUBMITTED_URL
    assert response.json()["transcript"] == {
        "method": "whisper",
        "language": "en",
        "text": "Textify integration transcript.",
    }


async def test_public_extraction_excludes_private_textify_data_from_errors_and_logs(
    migrated_coordination_database: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep private provider payload values out of public and adapter log records."""
    submitted_url_sentinel = "submitted-url-private-sentinel"
    source_url_sentinel = "returned-source-url-private-sentinel"
    transcript_sentinel = "successful-transcript-private-sentinel"
    error_message_sentinel = "terminal-error-private-sentinel"
    self_link_sentinel = "self-capability-private-sentinel"
    cancel_link_sentinel = "cancel-capability-private-sentinel"

    def response_transform(phase: str, payload: dict[str, object]) -> object:
        if phase != "processing":
            return payload
        links = cast(dict[str, object], payload["links"])
        links["self"] = f"/api/v1/transcription-jobs/{self_link_sentinel}"
        links["cancel"] = f"/api/v1/transcription-jobs/{cancel_link_sentinel}"
        payload["error"] = {
            "code": "internal_error",
            "message": error_message_sentinel,
        }
        payload["result"] = {
            "source": {"url": f"https://provider.test/{source_url_sentinel}"},
            "transcript": {"text": transcript_sentinel},
        }
        return payload

    server = _TextifyContractServer(
        "faster_whisper",
        active_polls=1,
        response_transform=response_transform,
    )
    clock = _RetryClock()
    pipeline = await _create_pipeline(
        migrated_coordination_database,
        server,
        clock.sleep,
        clock=clock,
    )
    submitted_url = f"https://www.youtube.com/watch?v=dQw4w9WgXcQ&private={submitted_url_sentinel}"

    response = await _request_public_extraction(
        pipeline,
        submitted_url,
        adapter_log_handler=caplog.handler,
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }
    adapter_records = [
        record for record in caplog.records if record.name == "reelio.extraction.services.textify"
    ]
    assert adapter_records
    for private_value in (
        submitted_url_sentinel,
        source_url_sentinel,
        transcript_sentinel,
        error_message_sentinel,
        self_link_sentinel,
        cancel_link_sentinel,
    ):
        assert private_value not in response.text
        assert all(private_value not in record.getMessage() for record in adapter_records)
        assert all(private_value not in str(record.__dict__) for record in adapter_records)
