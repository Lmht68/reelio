"""ASGI contracts for durable Source Extraction job admission and status."""

import hmac
from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from reelio.config import Environment
from reelio.extraction.anonymous_owner import (
    AnonymousOwnerConfig,
    AnonymousOwnerManager,
)
from reelio.extraction.job_router import router
from reelio.extraction.job_store import InMemoryExtractionJobStore
from reelio.extraction.jobs import ExtractionJobStoreUnavailableError, QueuedSourceJob
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.schemas import ExtractResponse

_SUBMITTED_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

_OWNER_COOKIE_NAME = "reelio_anonymous_owner"
_OWNER_SECRET = "router-owner-hmac-secret"
_IDEMPOTENCY_KEY = UUID("00000000-0000-4000-8000-000000000001")
_SECOND_IDEMPOTENCY_KEY = UUID("00000000-0000-4000-8000-000000000002")


def _headers(idempotency_key: UUID = _IDEMPOTENCY_KEY) -> dict[str, str]:
    """Build one valid public idempotency header."""
    return {"Idempotency-Key": str(idempotency_key)}


def _owner_manager(
    environment: Environment = Environment.LOCAL,
) -> AnonymousOwnerManager:
    """Build a deterministic owner manager for one ASGI application."""
    return AnonymousOwnerManager(
        AnonymousOwnerConfig(
            hmac_secret=SecretStr(_OWNER_SECRET),
            environment=environment,
        )
    )


def _application(
    store: object,
    *,
    owner_manager: AnonymousOwnerManager | None = None,
) -> FastAPI:
    """Build one ASGI application with injected job-store and owner dependencies."""
    application = FastAPI()
    application.state.extraction_job_store = store
    application.state.anonymous_owner_manager = owner_manager or _owner_manager()
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
        response = await client.post(
            "/api/extraction-jobs",
            json={"url": ""},
            headers=_headers(),
        )

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
            headers=_headers(),
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
    application = _application(store)
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        admitted = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        queued_response = await client.get(f"/api/extraction-jobs/{job_id}")
        await store.claim_next_source()
        running_response = await client.get(f"/api/extraction-jobs/{job_id}")
        await store.complete_success(job_id, _response())
        succeeded_response = await client.get(f"/api/extraction-jobs/{job_id}")

    assert admitted.status_code == 202
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

    async def admit_source(
        self,
        source_url: str,
        market: SpotifyMarket | None,
        owner_digest: bytes,
        idempotency_key: UUID,
    ) -> object:
        """Refuse admission while the durable authority is unavailable."""
        del source_url, market, owner_digest, idempotency_key
        raise ExtractionJobStoreUnavailableError("database unavailable")

    async def get_for_owner(self, job_id: UUID, owner_digest: bytes) -> object:
        """Refuse polling while the durable authority is unavailable."""
        del job_id, owner_digest
        raise ExtractionJobStoreUnavailableError("database unavailable")


async def test_store_unavailability_returns_the_retryable_extraction_envelope() -> None:
    """Never select an in-memory fallback when durable storage is unavailable."""
    application = _application(_UnavailableStore())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        admission_response = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
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


class _RecordingStore:
    """Record ownership inputs received through the public admission route."""

    def __init__(self) -> None:
        """Initialize an empty admission recording."""
        self.owner_digests: list[bytes] = []
        self.idempotency_keys: list[UUID] = []

    async def admit_source(
        self,
        source_url: str,
        market: SpotifyMarket | None,
        owner_digest: bytes,
        idempotency_key: UUID,
    ) -> QueuedSourceJob:
        """Record one admission and return a committed queued snapshot."""
        self.owner_digests.append(owner_digest)
        self.idempotency_keys.append(idempotency_key)
        submitted_at = datetime(2026, 10, 15, tzinfo=UTC)
        return QueuedSourceJob(
            job_id=UUID("00000000-0000-4000-8000-000000000001"),
            source_url=source_url,
            market=market,
            submitted_at=submitted_at,
            updated_at=submitted_at,
        )


async def test_submission_requires_a_uuid_idempotency_header() -> None:
    """Reject a missing or malformed idempotency header before admission."""
    application = _application(InMemoryExtractionJobStore())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/api/extraction-jobs", json={"url": _SUBMITTED_URL})
        malformed = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers={"Idempotency-Key": "not-a-uuid"},
        )

    assert missing.status_code == 422
    assert malformed.status_code == 422


async def test_admission_issues_an_httponly_owner_cookie_and_passes_only_its_digest() -> None:
    """Derive the recorded storage identity from a cryptographically shaped cookie."""
    store = _RecordingStore()
    application = _application(store)
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        owner_token = client.cookies[_OWNER_COOKIE_NAME]

    assert response.status_code == 202
    assert len(owner_token) == 43
    assert all(
        character.isascii() and (character.isalnum() or character in "-_")
        for character in owner_token
    )
    assert response.headers["set-cookie"].startswith(f"{_OWNER_COOKIE_NAME}=")
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "Max-Age=2592000" in response.headers["set-cookie"]
    assert "Path=/" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert "Secure" not in response.headers["set-cookie"]
    assert store.owner_digests == [
        hmac.digest(_OWNER_SECRET.encode(), owner_token.encode("ascii"), "sha256")
    ]
    assert store.idempotency_keys == [_IDEMPOTENCY_KEY]


async def test_admission_sets_secure_owner_cookies_outside_local() -> None:
    """Mark anonymous owner cookies Secure in non-local deployments."""
    application = _application(
        _RecordingStore(),
        owner_manager=_owner_manager(Environment.PRODUCTION),
    )
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )

    assert "Secure" in response.headers["set-cookie"]


async def test_same_owner_replay_returns_one_job_identity() -> None:
    """Reuse the original admission when one browser retries the same key."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    application = _application(InMemoryExtractionJobStore(uuid_factory=lambda: job_id))
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        replay = await client.post(
            "/api/extraction-jobs",
            json={"url": "https://www.youtube.com/watch?v=changed"},
            headers=_headers(),
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()


async def test_independent_browser_profiles_isolate_jobs_and_share_not_found_body() -> None:
    """Allow same-key admissions per profile while hiding another profile's job."""
    first_job_id = UUID("00000000-0000-4000-8000-000000000001")
    second_job_id = UUID("00000000-0000-4000-8000-000000000002")
    application = _application(
        InMemoryExtractionJobStore(uuid_factory=iter((first_job_id, second_job_id)).__next__)
    )
    transport = ASGITransport(app=application)

    async with (
        AsyncClient(transport=transport, base_url="http://test") as profile_a,
        AsyncClient(transport=transport, base_url="http://test") as profile_b,
    ):
        first = await profile_a.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        readable = await profile_a.get(f"/api/extraction-jobs/{first_job_id}")
        second = await profile_b.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        unauthorized = await profile_b.get(f"/api/extraction-jobs/{first_job_id}")
        missing = await profile_b.get("/api/extraction-jobs/00000000-0000-4000-8000-000000000003")

    assert first.status_code == 202
    assert readable.status_code == 200
    assert second.status_code == 202
    assert second.json()["job_id"] == str(second_job_id)
    assert unauthorized.status_code == 404
    assert missing.status_code == 404
    assert unauthorized.content == missing.content


async def test_malformed_owner_cookie_uses_the_shared_not_found_response() -> None:
    """Hide a valid job when the browser presents an invalid owner credential."""
    job_id = UUID("00000000-0000-4000-8000-000000000001")
    application = _application(InMemoryExtractionJobStore(uuid_factory=lambda: job_id))
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as owner:
        admitted = await owner.post(
            "/api/extraction-jobs",
            json={"url": _SUBMITTED_URL},
            headers=_headers(),
        )
        expected_not_found = await owner.get(
            "/api/extraction-jobs/00000000-0000-4000-8000-000000000002"
        )

    async with AsyncClient(transport=transport, base_url="http://test") as malformed:
        malformed.cookies.set(_OWNER_COOKIE_NAME, "not-a-canonical-owner-token")
        response = await malformed.get(f"/api/extraction-jobs/{job_id}")

    assert admitted.status_code == 202
    assert response.status_code == 404
    assert response.content == expected_not_found.content
