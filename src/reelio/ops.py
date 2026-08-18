"""Operational HTTP endpoints for the application."""

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Represent a successful application health check."""

    status: Literal["ok"] = "ok"


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
