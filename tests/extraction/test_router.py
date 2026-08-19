"""HTTP contract tests for the extraction endpoint."""

from collections.abc import Callable, Iterator
from typing import cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    EnrichmentError,
    EntityExtractionError,
    ExtractionError,
    InvalidSourceError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.router import get_pipeline
from reelio.extraction.schemas import ExtractResponse
from reelio.extraction.service import FakePipeline, Pipeline
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.service import SourceMetadataService
from reelio.extraction.types import PipelineResult, ResultStatus
from reelio.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Clear dependency overrides before and after every extraction test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _RaisingPipeline:
    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def run(self, url: str) -> PipelineResult:
        raise self._exception


_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _MetadataExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> dict[str, object]:
        self.calls.append(canonical_url)
        return {
            "id": _VIDEO_ID,
            "title": "Router test video",
            "description": "A complete router test description.",
            "channel": "Router test channel",
            "duration": 42.2,
        }


def _settings() -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(_env_file=None)


def _install_pipeline(application: FastAPI, pipeline: Pipeline) -> None:
    application.dependency_overrides[get_pipeline] = lambda: pipeline


async def test_extract_returns_schema_valid_response(client: AsyncClient) -> None:
    """Return real source metadata and all deterministic result branches."""
    metadata_extractor = _MetadataExtractor()
    pipeline = FakePipeline(
        SourceMetadataService(
            extractor=metadata_extractor,
            settings=_settings(),
        )
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == "youtube"
    assert payload.source.video_id == _VIDEO_ID
    assert payload.source.url == _CANONICAL_URL
    assert payload.source.title == "Router test video"
    assert payload.source.description == "A complete router test description."
    assert payload.source.channel == "Router test channel"
    assert payload.source.duration_seconds == 43
    assert metadata_extractor.calls == [_CANONICAL_URL]
    assert payload.transcript.method == "youtube_captions"
    assert payload.transcript.text
    assert {result.status for result in payload.results} == {
        ResultStatus.RESOLVED,
        ResultStatus.AMBIGUOUS,
        ResultStatus.UNRESOLVED,
    }
    assert all(len(result.mentioned_as) >= 1 for result in payload.results)
    assert all(len(result.evidence) >= 1 for result in payload.results)

    ambiguous = next(
        result for result in payload.results if result.status is ResultStatus.AMBIGUOUS
    )
    assert ambiguous.movie is None
    assert len(ambiguous.candidates) == 3
    assert len({candidate.resolution_score for candidate in ambiguous.candidates}) == 3

    unresolved = next(
        result for result in payload.results if result.status is ResultStatus.UNRESOLVED
    )
    assert unresolved.resolution_confidence is None
    assert unresolved.movie is None
    assert unresolved.candidates == []


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (InvalidSourceError("invalid source"), 400, "invalid_source"),
        (
            UnsupportedPlatformError("unsupported platform"),
            400,
            "unsupported_platform",
        ),
        (SourceUnavailableError("source unavailable"), 404, "source_unavailable"),
        (
            DurationLimitExceededError("duration limit exceeded"),
            413,
            "duration_limit_exceeded",
        ),
        (
            MetadataProviderError("Unable to retrieve YouTube metadata."),
            502,
            "metadata_provider_failed",
        ),
        (TranscriptionError("transcription failed"), 502, "transcription_failed"),
        (
            EntityExtractionError("entity extraction failed"),
            502,
            "entity_extraction_failed",
        ),
        (EnrichmentError("enrichment failed"), 502, "enrichment_failed"),
        (PipelineTimeoutError("pipeline timed out"), 504, "pipeline_timeout"),
    ],
)
async def test_extraction_errors_map_to_contract(
    client: AsyncClient,
    exception: ExtractionError,
    expected_status: int,
    expected_code: str,
) -> None:
    """Map every extraction domain error to its stable HTTP contract."""
    _install_pipeline(app, _RaisingPipeline(exception))

    response = await client.post(
        "/api/extract",
        json={"url": "https://www.youtube.com/watch?v=anything"},
    )

    assert response.status_code == expected_status
    assert response.json() == {
        "error": {"code": expected_code, "message": str(exception)}
    }


@pytest.mark.parametrize("payload", [{}, {"url": 123}, {"url": ""}])
async def test_malformed_requests_keep_fastapi_422_contract(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    """Keep FastAPI validation responses for malformed request bodies."""
    response = await client.post("/api/extract", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


async def test_unhandled_failures_do_not_leak_internals() -> None:
    """Return a generic 500 response when the pipeline raises unexpectedly."""
    _install_pipeline(
        app,
        _RaisingPipeline(RuntimeError("sensitive database path /var/reelio/secret.db")),
    )
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extract",
            json={"url": "https://www.youtube.com/watch?v=anything"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred.",
        }
    }
    assert "sensitive database path" not in response.text


async def test_extract_is_documented_in_openapi(client: AsyncClient) -> None:
    """Document the endpoint request and configured error responses."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/extract"]["post"]
    responses = operation["responses"]
    assert {"200", "400", "404", "413", "500", "502", "504", "422"} <= set(responses)
    for status_code in ("400", "404", "413", "500", "502", "504"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/ExtractRequest"}
    source_properties = document["components"]["schemas"]["Source"]["properties"]
    assert {
        "platform",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "duration_seconds",
    } <= set(source_properties)
