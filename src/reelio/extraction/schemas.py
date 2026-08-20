"""Pydantic models for the extraction HTTP contract."""

from pydantic import BaseModel, Field

from reelio.extraction.types import Platform, ResultStatus, TranscriptMethod


class ExtractRequest(BaseModel):
    """Request body for extracting movie mentions from a source URL."""

    url: str = Field(
        min_length=1,
        max_length=2048,
        examples=[
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.instagram.com/reel/ABC123",
            "https://www.facebook.com/reel/123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://x.com/creator/status/123456789",
        ],
    )


class Source(BaseModel):
    """Canonical source identity and video metadata returned to the caller."""

    platform: Platform
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


class Transcript(BaseModel):
    """Transcript text and acquisition metadata returned to the caller."""

    text: str
    language: str
    method: TranscriptMethod


class Movie(BaseModel):
    """Provider-enriched movie metadata in an extraction response."""

    title: str
    year: int | None
    directors: list[str]
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float = Field(ge=0, le=10)


class Candidate(Movie):
    """Enriched movie candidate with its resolver score."""

    resolution_score: float = Field(ge=0, le=1)


class Result(BaseModel):
    """One interpreted mention and its resolution outcome."""

    status: ResultStatus
    mentioned_as: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    extraction_confidence: float = Field(ge=0, le=1)
    resolution_confidence: float | None = Field(ge=0, le=1)
    movie: Movie | None
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


class ExtractResponse(BaseModel):
    """Successful extraction response containing source and movie results."""

    source: Source
    transcript: Transcript
    results: list[Result]


class ErrorDetail(BaseModel):
    """Machine-readable error code and human-readable message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Consistent error response body for extraction failures."""

    error: ErrorDetail
