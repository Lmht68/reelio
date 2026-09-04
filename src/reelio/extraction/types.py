"""Domain types for extraction and identity primitives."""

import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from reelio.extraction.market import SpotifyMarket

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


def normalize_music_text(text: str) -> str:
    """Normalize music text for display without changing case.

    Args:
        text: Music title or artist name to normalize.

    Returns:
        str: NFC-normalized text with leading, trailing, and repeated whitespace
        removed.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_music_identity(text: str) -> str:
    """Normalize music text for case-insensitive identity comparison.

    Args:
        text: Music title or artist name to normalize.

    Returns:
        str: Display-normalized text with Unicode case folding applied.
    """
    return normalize_music_text(text).casefold()


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


@dataclass(frozen=True, slots=True)
class ArtistCredit:
    """Identify one Spotify artist credit in display order.

    Attributes:
        spotify_artist_id: Spotify Artist identifier.
        name: Spotify-authoritative credited artist name.
    """

    spotify_artist_id: str
    name: str


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
class TrackMention:
    """Contain a released Track Mention interpreted from source material.

    Attributes:
        track_title: Complete recording title.
        artists: Ordered credited artist names.
        release_title: Album or single title when explicitly stated.
        release_year: Release year when explicitly stated.
    """

    track_title: str
    artists: list[str]
    release_title: str | None
    release_year: int | None


@dataclass
class MusicReleaseMention:
    """Contain a Music Release Mention interpreted from source material.

    Attributes:
        release_title: Album, EP, single, or compilation title.
        artists: Ordered credited artist names.
        release_year: Release year when explicitly stated.
    """

    release_title: str
    artists: list[str]
    release_year: int | None


@dataclass
class MusicMentions:
    """Contain ordered Music Mentions grouped by resolvable kind.

    Attributes:
        tracks: Released Track Mentions in first-reference order.
        music_releases: Raw Music Release Mentions in first-reference order.
    """

    tracks: list[TrackMention]
    music_releases: list[MusicReleaseMention]


@dataclass
class ExtractionMentions:
    """Contain interpreted mentions grouped by service scope.

    Attributes:
        screen_works: Ordered Screen Work Mentions grouped by kind.
        music: Ordered Music Mentions grouped by kind.
    """

    screen_works: ScreenWorkMentions
    music: MusicMentions


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
class EnrichedTrack:
    """Contain Spotify-verified metadata for a playable Track.

    Attributes:
        track_title: Spotify-authoritative Track title.
        artists: Ordered Spotify Track artist credits.
        spotify_track_id: Playable Spotify Track identifier for the effective market.
        spotify_url: Direct Spotify URL for the playable Track.
    """

    track_title: str
    artists: list[ArtistCredit]
    spotify_track_id: str
    spotify_url: str


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
class TrackResult:
    """Represent one Track Mention and its resolution outcome.

    Attributes:
        status: Current resolution state of the Track Mention.
        track_mention: Canonical Track Mention.
        track: Enriched Track, or ``None`` when unresolved.
    """

    status: ResultStatus
    track_mention: TrackMention
    track: EnrichedTrack | None


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
class MusicResults:
    """Contain ordered Music Results grouped by public result kind.

    Attributes:
        tracks: Track Results in first-reference order.
    """

    tracks: list[TrackResult]


@dataclass
class ExtractionResults:
    """Contain resolved results grouped by service scope.

    Attributes:
        screen_works: Ordered Screen Work Results grouped by kind.
        music: Ordered Music Results grouped by kind.
    """

    screen_works: ScreenWorkResults
    music: MusicResults


@dataclass
class PipelineResult:
    """Contain the structured output of an end-to-end extraction pipeline.

    Attributes:
        source: Canonical identity of the submitted source.
        transcript: Full transcript used for mention interpretation.
        results: Resolved results grouped by service scope.
        market: Effective Spotify market used for catalog resolution.
    """

    source: Source
    transcript: Transcript
    results: ExtractionResults
    market: SpotifyMarket
