"""Public HTTP schemas for durable Source Extraction jobs."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from reelio.extraction.schemas import ExtractResponse


class AcceptedSourceExtractionJob(BaseModel):
    """Represent a newly committed Source Extraction job admission."""

    job_id: UUID
    status: Literal["queued"]


class QueuedSourceExtractionJob(BaseModel):
    """Represent a Source Extraction job waiting for one worker claim."""

    job_id: UUID
    status: Literal["queued"]


class RunningSourceExtractionJob(BaseModel):
    """Represent a Source Extraction job being processed by one worker."""

    job_id: UUID
    status: Literal["running"]


class SucceededSourceExtractionJob(BaseModel):
    """Represent one Source Extraction job with its unchanged public result."""

    job_id: UUID
    status: Literal["succeeded"]
    result: ExtractResponse


type SourceExtractionJobStatusResponse = Annotated[
    QueuedSourceExtractionJob | RunningSourceExtractionJob | SucceededSourceExtractionJob,
    Field(discriminator="status"),
]
