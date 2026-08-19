"""HTTP contract tests for the extraction endpoint."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    EnrichmentError,
    EntityExtractionError,
    ExtractionError,
    InvalidSourceError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.router import get_pipeline
from reelio.extraction.schemas import ExtractResponse
from reelio.extraction.service import Pipeline
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


def _install_pipeline(application: FastAPI, pipeline: Pipeline) -> None:
    application.dependency_overrides[get_pipeline] = lambda: pipeline


async def test_extract_returns_schema_valid_response(client: AsyncClient) -> None:
    """Return all result branches in a schema-valid response."""
    response = await client.post(
        "/api/extract",
        json={"url": "https://www.youtube.com/watch?v=anything"},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == "youtube"
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
    operation = response.json()["paths"]["/api/extract"]["post"]
    responses = operation["responses"]
    assert {"200", "400", "404", "413", "500", "502", "504", "422"} <= set(responses)
    for status_code in ("400", "404", "413", "500", "502", "504"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/ExtractRequest"}
