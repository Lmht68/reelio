"""Public HTTP schemas for durable Source Extraction jobs."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from reelio.extraction.schemas import ErrorDetail, ExtractResponse


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


class FailedSourceExtractionJob(BaseModel):
    """Represent one terminal Source Extraction job failure safely."""

    job_id: UUID
    status: Literal["failed"]
    error: ErrorDetail


type SourceExtractionJobStatusResponse = Annotated[
    QueuedSourceExtractionJob
    | RunningSourceExtractionJob
    | SucceededSourceExtractionJob
    | FailedSourceExtractionJob,
    Field(discriminator="status"),
]
