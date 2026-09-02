"""Domain types for extraction and Screen Work identity primitives."""

import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

MINIMUM_SCREEN_WORK_MENTION_YEAR = 1888
_MAX_FUTURE_SCREEN_WORK_MENTION_YEARS = 2


def maximum_screen_work_mention_year() -> int:
    """Return the latest accepted Screen Work Mention year.

    Returns:
        int: Current year plus the two-year confirmed-forthcoming horizon.
    """
    return date.today().year + _MAX_FUTURE_SCREEN_WORK_MENTION_YEARS


def normalize_screen_work_title(title: str) -> str:
    """Produce the domain comparison form for a Canonical Movie or TV Series title.

    Args:
        title: Canonical or provider-backed Movie or TV Series title to normalize.

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
class TVSeriesMention:
    """Contain a TV Series Mention interpreted from Interpretation Material.

    Attributes:
        title: Canonical TV Series Title.
        year: TV First Air Year.
    """

    title: str
    year: int


@dataclass
class ScreenWorkMentions:
    """Contain ordered Screen Work Mentions grouped by kind.

    List position preserves first-reference order within each Screen Work kind.
    There is no cross-kind ordering.

    Attributes:
        movies: Canonical Movie Mentions in first-reference order.
        tv_series: Canonical TV Series Mentions in first-reference order.
    """

    movies: list[MovieMention]
    tv_series: list[TVSeriesMention]


@dataclass
class ExtractionMentions:
    """Contain interpreted mentions grouped by service scope.

    Attributes:
        screen_works: Ordered Screen Work Mentions grouped by kind.
    """

    screen_works: ScreenWorkMentions


@dataclass
class EnrichedMovie:
    """Contain provider-verified metadata for a movie entity.

    Attributes:
        title: Canonical movie title.
        year: Release year.
        cast: Up to five provider-ordered cast member names.
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
    cast: list[str]
    directors: list[str]
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float


@dataclass
class EnrichedTVSeries:
    """Contain provider-verified metadata for a TV Series entity.

    Attributes:
        title: Canonical TV Series title.
        first_air_year: Verified first air year.
        last_air_year: Final air year when available from a completed series.
        cast: Up to five provider-ordered aggregate cast member names.
        creators: Provider-verified creator names.
        description: Short provider-supplied TV Series description.
        poster_url: Provider image URL when available.
        tmdb_id: TMDB TV Series identifier.
        tmdb_url: Canonical TMDB TV Series URL.
        imdb_id: Provider-verified IMDb identifier when available.
        imdb_url: Canonical IMDb URL when an IMDb identifier exists.
        tmdb_score: TMDB vote average on a zero-to-ten scale.
    """

    title: str
    first_air_year: int
    last_air_year: int | None
    cast: list[str]
    creators: list[str]
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float


@dataclass
class MovieResult:
    """Represent one Movie Mention and its resolution outcome.

    Attributes:
        status: Current resolution state of the Movie Mention.
        movie_mention: Canonical title and Movie Release Year.
        movie: Enriched Movie, or ``None`` when unresolved.
    """

    status: ResultStatus
    movie_mention: MovieMention
    movie: EnrichedMovie | None


@dataclass
class TVSeriesResult:
    """Represent one TV Series Mention and its resolution outcome.

    Attributes:
        status: Current resolution state of the TV Series Mention.
        tv_series_mention: Canonical title and TV First Air Year.
        tv_series: Enriched TV Series, or ``None`` when unresolved.
    """

    status: ResultStatus
    tv_series_mention: TVSeriesMention
    tv_series: EnrichedTVSeries | None


@dataclass
class ScreenWorkResults:
    """Contain ordered Screen Work Results grouped by kind.

    List position preserves first-reference order within each Screen Work kind.
    There is no cross-kind ordering.

    Attributes:
        movies: Movie Results in first-reference order.
        tv_series: TV Series Results in first-reference order.
    """

    movies: list[MovieResult]
    tv_series: list[TVSeriesResult]


@dataclass
class ExtractionResults:
    """Contain resolved results grouped by service scope.

    Attributes:
        screen_works: Ordered Screen Work Results grouped by kind.
    """

    screen_works: ScreenWorkResults


@dataclass
class PipelineResult:
    """Contain the structured output of an end-to-end extraction pipeline.

    Attributes:
        source: Canonical identity of the submitted source.
        transcript: Full transcript used for mention interpretation.
        results: Resolved results grouped by service scope.
    """

    source: Source
    transcript: Transcript
    results: ExtractionResults
