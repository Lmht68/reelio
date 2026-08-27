"""Domain types shared by extraction services and the API adapter."""

import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

_MAX_FUTURE_RELEASE_YEARS = 2


def maximum_movie_release_year() -> int:
    """Return the latest accepted Movie Release Year.

    Returns:
        int: Current year plus the two-year confirmed-forthcoming horizon.
    """
    return date.today().year + _MAX_FUTURE_RELEASE_YEARS


def normalize_movie_title(title: str) -> str:
    """Produce the domain comparison form for a Canonical Movie Title.

    Args:
        title: Canonical or provider-backed title to normalize.

    Returns:
        str: NFC-normalized title with leading, trailing, and repeated
        whitespace removed.
    """
    return " ".join(unicodedata.normalize("NFC", title).split())


class Platform(StrEnum):
    """Content platforms supported by the extraction context."""

    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    X = "x"


class TranscriptMethod(StrEnum):
    """Methods used to acquire a normalized transcript."""

    YOUTUBE_CAPTIONS = "youtube_captions"
    WHISPER = "whisper"


class ResultStatus(StrEnum):
    """Resolution outcomes for an interpreted mention."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass
class Source:
    """Contain canonical source identity and normalized video metadata.

    Attributes:
        platform: Platform hosting the source.
        video_id: Stable external identifier for the source.
        url: Validated canonical URL returned or reconstructed for the source.
        title: Provider-supplied video title.
        description: Complete provider-supplied video description.
        channel: Provider-supplied channel name.
        duration_seconds: Video duration rounded up to whole seconds.
    """

    platform: Platform
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


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
class MovieMention:
    """Contain a Movie Mention interpreted from complete Interpretation Material.

    Attributes:
        title: Canonical Movie Title.
        year: Movie Release Year.
    """

    title: str
    year: int


@dataclass
class EnrichedMovie:
    """Contain provider-verified metadata for a movie entity.

    Attributes:
        title: Canonical movie title.
        year: Release year.
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
    year: int
    directors: list[str]
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float


@dataclass
class MentionResult:
    """Represent one interpreted mention and its resolution outcome.

    Attributes:
        status: Current resolution state of the Movie Mention.
        movie_mention: Canonical title and Movie Release Year.
        movie: Enriched Entity, or ``None`` before resolution or after no match.
    """

    status: ResultStatus
    movie_mention: MovieMention
    movie: EnrichedMovie | None


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
