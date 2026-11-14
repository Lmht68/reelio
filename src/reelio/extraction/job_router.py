"""Admission and status routing for durable Source Extraction jobs."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Header, Request, status
from fastapi.responses import JSONResponse

from reelio.extraction import job_schemas
from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.anonymous_owner import (
    ANONYMOUS_OWNER_COOKIE_NAME,
    AnonymousOwnerManager,
)
from reelio.extraction.jobs import (
    ExtractionJobStore,
    ExtractionJobStoreUnavailableError,
    FailedSourceJob,
    QueuedSourceJob,
    RunningSourceJob,
    SucceededSourceJob,
)

router = APIRouter(prefix="/api", tags=["extraction jobs"])

_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_POLL_HEADERS = {**_NO_STORE_HEADERS, "Retry-After": "1"}
_UNAVAILABLE_HEADERS = {**_NO_STORE_HEADERS, "Retry-After": "2"}


def get_job_store(request: Request) -> ExtractionJobStore:
    """Provide the lifespan-owned Source Extraction job store.

    Args:
        request: Current request whose application owns the job store.

    Returns:
        The application-owned durable Source Extraction job store.
    """
    return cast(ExtractionJobStore, request.app.state.extraction_job_store)


JobStoreDependency = Annotated[ExtractionJobStore, Depends(get_job_store)]


def get_anonymous_owner_manager(request: Request) -> AnonymousOwnerManager:
    """Provide the application-owned anonymous owner-cookie manager.

    Args:
        request: Current request whose application owns the manager.

    Returns:
        The manager that resolves and issues anonymous job owners.
    """
    return cast(AnonymousOwnerManager, request.app.state.anonymous_owner_manager)


AnonymousOwnerManagerDependency = Annotated[
    AnonymousOwnerManager,
    Depends(get_anonymous_owner_manager),
]
OwnerCookie = Annotated[str | None, Cookie(alias=ANONYMOUS_OWNER_COOKIE_NAME)]
IdempotencyKeyHeader = Annotated[UUID, Header(alias="Idempotency-Key")]


@router.post(
    "/extraction-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=job_schemas.AcceptedSourceExtractionJob,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Durable Source Extraction job storage is temporarily unavailable.",
        },
    },
)
async def admit_source_extraction_job(
    payload: extraction_schemas.ExtractRequest,
    idempotency_key: IdempotencyKeyHeader,
    store: JobStoreDependency,
    owner_manager: AnonymousOwnerManagerDependency,
    owner_cookie: OwnerCookie = None,
) -> JSONResponse:
    """Commit an owner-scoped queued Source Extraction job.

    Args:
        payload: Validated Source URL and optional requested Effective Market.
        idempotency_key: Required caller-supplied owner-scoped replay identity.
        store: Durable job store supplied by application lifespan ownership.
        owner_manager: Cookie owner resolver supplied by application state.
        owner_cookie: Optional existing HttpOnly anonymous owner cookie.

    Returns:
        An accepted queued-job projection with no-store polling headers and a refreshed
        owner cookie.
    """
    owner = owner_manager.resolve_or_issue(owner_cookie)
    try:
        job = await store.admit_source(
            payload.url,
            payload.market,
            owner.digest,
            idempotency_key,
        )
    except ExtractionJobStoreUnavailableError:
        response = _unavailable_response()
        owner_manager.set_cookie(response, owner)
        return response

    accepted = job_schemas.AcceptedSourceExtractionJob(
        job_id=job.job_id,
        status="queued",
    )
    response = JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=accepted.model_dump(mode="json"),
        headers={
            **_POLL_HEADERS,
            "Location": f"/api/extraction-jobs/{job.job_id}",
        },
    )
    owner_manager.set_cookie(response, owner)
    return response


@router.get(
    "/extraction-jobs/{job_id}",
    response_model=job_schemas.SourceExtractionJobStatusResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": extraction_schemas.ErrorResponse,
            "description": "The Source Extraction job does not exist.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Durable Source Extraction job storage is temporarily unavailable.",
        },
    },
)
async def get_source_extraction_job(
    job_id: UUID,
    store: JobStoreDependency,
    owner_manager: AnonymousOwnerManagerDependency,
    owner_cookie: OwnerCookie = None,
) -> JSONResponse:
    """Project a Source Extraction job only for its anonymous owner.

    Args:
        job_id: Opaque Source Extraction job identifier.
        store: Durable job store supplied by application lifespan ownership.
        owner_manager: Cookie owner resolver supplied by application state.
        owner_cookie: Optional existing HttpOnly anonymous owner cookie.

    Returns:
        A no-store queued, running, or succeeded public job projection.

    Raises:
        RuntimeError: If a failed state is found before its public projection exists.
    """
    owner = owner_manager.resolve_existing(owner_cookie)
    if owner is None:
        return _not_found_response()
    try:
        job = await store.get_for_owner(job_id, owner.digest)
    except ExtractionJobStoreUnavailableError:
        return _unavailable_response()

    if job is None:
        return _not_found_response()
    if isinstance(job, QueuedSourceJob):
        return _response(
            job_schemas.QueuedSourceExtractionJob(
                job_id=job.job_id,
                status="queued",
            ),
            _POLL_HEADERS,
        )
    if isinstance(job, RunningSourceJob):
        return _response(
            job_schemas.RunningSourceExtractionJob(
                job_id=job.job_id,
                status="running",
            ),
            _POLL_HEADERS,
        )
    if isinstance(job, SucceededSourceJob):
        return _response(
            job_schemas.SucceededSourceExtractionJob(
                job_id=job.job_id,
                status="succeeded",
                result=job.result,
            ),
            _NO_STORE_HEADERS,
        )
    if isinstance(job, FailedSourceJob):
        raise RuntimeError("Source Extraction job failed before failure projection exists")
    raise RuntimeError("Source Extraction job has an unsupported public state")


def _response(
    body: job_schemas.SourceExtractionJobStatusResponse,
    headers: dict[str, str],
) -> JSONResponse:
    """Serialize one validated public job projection with explicit cache headers.

    Args:
        body: Validated nonterminal or succeeded public projection.
        headers: Cache and polling headers for the projection state.

    Returns:
        JSON response containing only the public job projection.
    """
    return JSONResponse(content=body.model_dump(mode="json"), headers=headers)


def _not_found_response() -> JSONResponse:
    """Return the shared unobservable response for absent and unauthorized jobs."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "code": "extraction_job_not_found",
                "message": "Extraction job was not found.",
            }
        },
        headers=_NO_STORE_HEADERS,
    )


def _unavailable_response() -> JSONResponse:
    """Return the shared safe response for durable job-store outages.

    Returns:
        Retryable no-store extraction-unavailable error response.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": {
                "code": "extraction_unavailable",
                "message": "Extraction is temporarily unavailable.",
            }
        },
        headers=_UNAVAILABLE_HEADERS,
    )
