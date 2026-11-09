"""ASGI contracts for durable Source Extraction job admission and status."""

from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from reelio.extraction.job_router import router
from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import ExtractionJobStoreUnavailableError
from reelio.extraction.schemas import ExtractResponse

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _application(store: object) -> FastAPI:
    """Build one ASGI application with an injected Source job store."""
    application = FastAPI()
    application.state.extraction_job_store = store
    application.include_router(router)
    return application


def _response() -> ExtractResponse:
    """Build one known existing Extraction response for success projection."""
    return ExtractResponse.model_validate(
        {
            "market": "JP",
            "source": {
                "platform": "youtube",
                "video_id": "dQw4w9WgXcQ",
                "url": _SUBMITTED_URL,
                "title": "Job router source",
                "description": "Job router description.",
                "channel": "Job router channel",
                "duration_seconds": 42,
            },
            "transcript": {
                "text": "A job router transcript.",
                "language": "en",
                "method": "youtube_captions",
            },
            "statistics": {
                "movies": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "tv_series": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "tracks": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
                "music_releases": {
                    "n_mentions": 0,
                    "n_resolved": 0,
                    "n_unresolved": 0,
                },
                "books": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            },
            "results": {
                "movies": [],
                "tv_series": [],
                "tracks": [],
                "music_releases": [],
                "books": [],
            },
        }
    )


async def test_submission_validates_the_existing_source_request() -> None:
    """Reject invalid public Source input before admission."""
    application = _application(InMemoryExtractionJobStore())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/extraction-jobs", json={"url": ""})

    assert response.status_code == 422


async def test_submission_commits_and_exposes_a_queued_job() -> None:
    """Return the committed job identity, status location, and polling headers."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    application = _application(store)
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        admitted = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL, "market": "JP"},
        )
        observed = await client.get(f"/api/extraction-jobs/{job_id}")

    assert admitted.status_code == 202
    assert admitted.json() == {"job_id": str(job_id), "status": "queued"}
    assert admitted.headers["Location"] == f"/api/extraction-jobs/{job_id}"
    assert admitted.headers["Cache-Control"] == "no-store"
    assert admitted.headers["Retry-After"] == "1"
    assert observed.status_code == 200
    assert observed.json() == {"job_id": str(job_id), "status": "queued"}
    assert observed.headers["Cache-Control"] == "no-store"
    assert observed.headers["Retry-After"] == "1"


async def test_status_projects_only_queued_running_and_succeeded_fields() -> None:
    """Hide durable input and metadata while preserving the exact success result."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    store = InMemoryExtractionJobStore(uuid_factory=lambda: job_id)
    queued = await store.admit_source(_SUBMITTED_URL, market=None)
    application = _application(store)
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        queued_response = await client.get(f"/api/extraction-jobs/{queued.job_id}")
        await store.claim_next_source()
        running_response = await client.get(f"/api/extraction-jobs/{queued.job_id}")
        await store.complete_success(queued.job_id, _response())
        succeeded_response = await client.get(f"/api/extraction-jobs/{queued.job_id}")

    assert queued_response.json() == {"job_id": str(job_id), "status": "queued"}
    assert running_response.json() == {"job_id": str(job_id), "status": "running"}
    assert succeeded_response.json() == {
        "job_id": str(job_id),
        "status": "succeeded",
        "result": _response().model_dump(mode="json"),
    }
    assert succeeded_response.headers["Cache-Control"] == "no-store"
    assert "Retry-After" not in succeeded_response.headers
    private_fields = {
        "source_url",
        "market",
        "submitted_at",
        "updated_at",
        "started_at",
        "finished_at",
        "result_version",
        "error_code",
        "error_message",
    }
    for response in (queued_response, running_response, succeeded_response):
        assert private_fields.isdisjoint(response.json())
        assert response.headers["Cache-Control"] == "no-store"


async def test_missing_job_returns_the_no_store_error_envelope() -> None:
    """Distinguish an absent opaque job identity without cacheable error state."""
    application = _application(InMemoryExtractionJobStore())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/extraction-jobs/00000000-0000-4000-8000-000000000001")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "extraction_job_not_found",
            "message": "Extraction job was not found.",
        }
    }
    assert response.headers["Cache-Control"] == "no-store"
    assert "Retry-After" not in response.headers


class _UnavailableStore:
    """Model one durable store outage at the public HTTP seam."""

    async def admit_source(self, source_url: str, market: object) -> object:
        """Refuse admission while the durable authority is unavailable."""
        del source_url, market
        raise ExtractionJobStoreUnavailableError("database unavailable")

    async def get(self, job_id: UUID) -> object:
        """Refuse polling while the durable authority is unavailable."""
        del job_id
        raise ExtractionJobStoreUnavailableError("database unavailable")


async def test_store_unavailability_returns_the_retryable_extraction_envelope() -> None:
    """Never select an in-memory fallback when durable storage is unavailable."""
    application = _application(_UnavailableStore())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        admission_response = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
        )
        status_response = await client.get(
            "/api/extraction-jobs/00000000-0000-4000-8000-000000000001"
        )

    expected_body = {
        "error": {
            "code": "extraction_unavailable",
            "message": "Extraction is temporarily unavailable.",
        }
    }
    for response in (admission_response, status_response):
        assert response.status_code == 503
        assert response.json() == expected_body
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Retry-After"] == "2"
