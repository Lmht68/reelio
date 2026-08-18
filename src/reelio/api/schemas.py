"""API response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Represent a successful application health check."""

    status: Literal["ok"] = "ok"
