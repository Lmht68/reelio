"""HTTP routes for the Reelio API."""

from fastapi import APIRouter, status

from reelio.api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check application health",
    description="Return the application liveness status.",
    response_description="The application is healthy.",
    tags=["health"],
)
async def health() -> HealthResponse:
    """Return the application's liveness status.

    Returns:
        HealthResponse: The stable healthy status payload.
    """
    return HealthResponse()
