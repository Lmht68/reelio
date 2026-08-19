"""Domain types shared by extraction services and the API adapter."""

from dataclasses import dataclass
from enum import StrEnum


class Platform(StrEnum):
    """Content platforms supported by the extraction context."""

    YOUTUBE = "youtube"


class TranscriptMethod(StrEnum):
    """Methods used to acquire a normalized transcript."""

    YOUTUBE_CAPTIONS = "youtube_captions"
    WHISPER = "whisper"


class ResultStatus(StrEnum):
    """Resolution outcomes for an interpreted mention."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass
class Source:
    """Identify the source content and its canonical platform URL.

    Attributes:
        platform: Platform hosting the source.
        video_id: Stable external identifier for the source.
        url: Canonical URL reconstructed from the platform identity.
    """

    platform: Platform
    video_id: str
    url: str


@dataclass
class Transcript:
    """Contain the normalized text interpreted by the extraction pipeline.

    Attributes:
        text: Full normalized transcript text.
        language: Detected or requested transcript language.
        method: Acquisition method that produced the text.
    """

    text: str
    language: str
    method: TranscriptMethod


@dataclass
class EnrichedMovie:
    """Contain provider-verified metadata for a movie entity.

    Attributes:
        title: Canonical movie title.
        year: Release year when the provider supplies one.
        directors: Provider-verified director names.
        description: Short provider-supplied movie description.
        poster_url: Provider image URL when available.
        tmdb_id: TMDB movie identifier.
        tmdb_url: Canonical TMDB movie URL.
        imdb_id: Provider-verified IMDb identifier when available.
        imdb_url: Canonical IMDb URL when an IMDb identifier exists.
        tmdb_score: TMDB vote average on a zero-to-ten scale.
    """

    title: str
    year: int | None
    directors: list[str]
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float


@dataclass
class Candidate(EnrichedMovie):
    """Contain an enriched movie candidate and its resolution score.

    Attributes:
        resolution_score: Resolver score on a zero-to-one scale.
    """

    resolution_score: float


@dataclass
class MentionResult:
    """Represent one interpreted mention and its resolution outcome.

    Attributes:
        status: Resolution outcome for the mention.
        mentioned_as: Phrases used for the same mention.
        evidence: Source passages supporting the mention.
        extraction_confidence: LLM confidence on a zero-to-one scale.
        resolution_confidence: Resolver confidence, or ``None`` when unresolved.
        movie: Resolved movie, or ``None`` when no unique movie was selected.
        candidates: Enriched alternatives for an ambiguous mention.
    """

    status: ResultStatus
    mentioned_as: list[str]
    evidence: list[str]
    extraction_confidence: float
    resolution_confidence: float | None
    movie: EnrichedMovie | None
    candidates: list[Candidate]


@dataclass
class PipelineResult:
    """Contain the structured output of an end-to-end extraction pipeline.

    Attributes:
        source: Canonical identity of the submitted source.
        transcript: Full transcript used for mention interpretation.
        results: One result for each interpreted movie mention.
    """

    source: Source
    transcript: Transcript
    results: list[MentionResult]
