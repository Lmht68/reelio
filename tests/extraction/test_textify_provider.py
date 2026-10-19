"""Textify source-and-transcript provider HTTP contract tests."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import httpx
import pytest

from reelio.cache import DisabledCache
from reelio.extraction.coordination import TextifyCoordination
from reelio.extraction.exceptions import TranscriptionError
from reelio.extraction.services.textify import (
    TextifyConfig,
    TextifySourceTranscriptProvider,
)
from reelio.extraction.types import Platform, TranscriptMethod

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CAPABILITY_LINK = "/api/v1/transcription-jobs/00000000-0000-4000-8000-000000000000"
_SUBMITTED_AT = "2026-10-14T00:00:00Z"
_STARTED_AT = "2026-10-14T00:00:01Z"
_FINISHED_AT = "2026-10-14T00:00:02Z"


def _textify_settings(**values: object) -> TextifyConfig:
    """Create Textify settings without reading environment files."""
    constructor = cast(Callable[..., TextifyConfig], TextifyConfig)
    return constructor(_env_file=None, **values)


class _CoordinationRepository:
    def __init__(self) -> None:
        now = datetime(2026, 10, 14, tzinfo=UTC)
        self.pending = TextifyCoordination(
            id=7,
            url_identity=b"x" * 32,
            idempotency_key=uuid4(),
            capability_link=None,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=1),
        )
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    async def get_or_create(self, submitted_url: str) -> TextifyCoordination:
        self.calls.append(("get_or_create", submitted_url))
        return self.pending

    async def attach_capability(
        self,
        coordination: TextifyCoordination,
        capability_link: str,
    ) -> TextifyCoordination:
        self.calls.append(("attach_capability", (coordination, capability_link)))
        return TextifyCoordination(
            id=coordination.id,
            url_identity=coordination.url_identity,
            idempotency_key=coordination.idempotency_key,
            capability_link=capability_link,
            created_at=coordination.created_at,
            updated_at=coordination.created_at + timedelta(seconds=1),
            expires_at=coordination.expires_at,
        )

    async def delete(self, coordination: TextifyCoordination) -> bool:
        """Record retirement of one exact Textify coordination snapshot."""
        self.calls.append(("delete", coordination))
        return True

    async def aclose(self) -> None:
        self.closed = True


async def test_provider_submits_commits_and_polls_the_stored_capability() -> None:
    """Acquire Reelio values after durable admission and active polling waits."""
    repository = _CoordinationRepository()
    requests: list[httpx.Request] = []
    waits: list[float] = []

    async def sleeper(seconds: float) -> None:
        waits.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert repository.calls == [("get_or_create", _SUBMITTED_URL)]
            assert request.url.path == "/api/v1/transcription-jobs"
            assert json.loads(request.content) == {
                "url": _SUBMITTED_URL,
                "exclude": ["transcript.segments"],
            }
            assert request.headers["Idempotency-Key"] == str(repository.pending.idempotency_key)
            return httpx.Response(
                202,
                headers={"Location": _CAPABILITY_LINK, "Retry-After": "2"},
                json={
                    "id": "00000000-0000-4000-8000-000000000000",
                    "status": "queued",
                    "submitted_at": _SUBMITTED_AT,
                    "links": {
                        "self": _CAPABILITY_LINK,
                        "cancel": f"{_CAPABILITY_LINK}/cancellation",
                    },
                    "additive_field": "accepted",
                },
            )
        if len(requests) == 2:
            assert repository.calls == [
                ("get_or_create", _SUBMITTED_URL),
                ("attach_capability", (repository.pending, _CAPABILITY_LINK)),
            ]
            assert request.url.path == _CAPABILITY_LINK
            return httpx.Response(
                200,
                headers={"Retry-After": "3"},
                json={
                    "id": "00000000-0000-4000-8000-000000000000",
                    "status": "processing",
                    "submitted_at": _SUBMITTED_AT,
                    "started_at": _STARTED_AT,
                    "cancellation_requested": False,
                    "links": {
                        "self": _CAPABILITY_LINK,
                        "cancel": f"{_CAPABILITY_LINK}/cancellation",
                    },
                    "additive_field": "active",
                },
            )
        assert request.url.path == _CAPABILITY_LINK
        return httpx.Response(
            200,
            json={
                "id": "00000000-0000-4000-8000-000000000000",
                "status": "finished",
                "outcome": "succeeded",
                "submitted_at": _SUBMITTED_AT,
                "started_at": _STARTED_AT,
                "finished_at": _FINISHED_AT,
                "links": {"self": _CAPABILITY_LINK},
                "result": {
                    "source": {
                        "platform": "youtube",
                        "video_id": "dQw4w9WgXcQ",
                        "url": _SUBMITTED_URL,
                        "title": "Video title",
                        "description": "Video description",
                        "channel": "Video channel",
                        "duration_seconds": 42,
                    },
                    "transcript": {
                        "method": "faster_whisper",
                        "language": "en",
                        "text": "Normalized transcript text.",
                    },
                },
                "additive_field": "finished",
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://textify.test/",
        follow_redirects=False,
    )
    provider = TextifySourceTranscriptProvider(
        client=client,
        coordination_repository=repository,
        settings=_textify_settings(),
        cache=DisabledCache(),
        sleeper=sleeper,
    )

    source, transcript = await provider.acquire(_SUBMITTED_URL)

    assert waits == [2.0, 3.0]
    assert [request.url.path for request in requests] == [
        "/api/v1/transcription-jobs",
        _CAPABILITY_LINK,
        _CAPABILITY_LINK,
    ]
    assert source.platform is Platform.YOUTUBE
    assert source.video_id == "dQw4w9WgXcQ"
    assert transcript.method is TranscriptMethod.WHISPER
    assert transcript.language == "en"
    assert transcript.text == "Normalized transcript text."

    await provider.aclose()

    assert repository.closed is True
    assert client.is_closed is True


async def test_provider_rejects_admission_that_does_not_repeat_its_capability() -> None:
    """Fail safely before polling when Textify admission headers and JSON disagree."""
    repository = _CoordinationRepository()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            202,
            headers={
                "Location": _CAPABILITY_LINK,
                "Retry-After": "2",
            },
            json={
                "id": "11111111-1111-4111-8111-111111111111",
                "status": "queued",
                "submitted_at": _SUBMITTED_AT,
                "links": {
                    "self": "/api/v1/transcription-jobs/11111111-1111-4111-8111-111111111111",
                    "cancel": "/api/v1/transcription-jobs/"
                    "11111111-1111-4111-8111-111111111111/cancellation",
                },
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://textify.test/",
        follow_redirects=False,
    )
    provider = TextifySourceTranscriptProvider(
        client=client,
        coordination_repository=repository,
        settings=_textify_settings(),
        cache=DisabledCache(),
        sleeper=lambda seconds: pytest.fail(f"unexpected poll wait: {seconds}"),
    )

    with pytest.raises(TranscriptionError, match="^Transcript is unavailable for this video\\.$"):
        await provider.acquire(_SUBMITTED_URL)

    assert repository.calls == [("get_or_create", _SUBMITTED_URL)]

    await provider.aclose()
