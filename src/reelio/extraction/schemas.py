"""Pydantic models for the extraction HTTP contract."""

from pydantic import BaseModel, Field

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import Platform, ResultStatus, TranscriptMethod


class ExtractRequest(BaseModel):
    """Request one extraction with an optional Spotify market."""

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

    market: SpotifyMarket | None = Field(
        default=None,
        description="Optional ISO 3166-1 alpha-2 Spotify market.",
        examples=["US", "JP"],
    )


class SourceModel(BaseModel):
    """Canonical source identity and video metadata returned to the caller."""

    platform: Platform
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


class TranscriptModel(BaseModel):
    """Transcript text and acquisition metadata returned to the caller."""

    text: str
    language: str
    method: TranscriptMethod


class MovieModel(BaseModel):
    """Provider-enriched movie metadata in an extraction response."""

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
    tmdb_score: float = Field(ge=0, le=10)


class MovieMentionModel(BaseModel):
    """Title and release year of a movie mention interpreted from a transcript."""

    title: str
    year: int


class MovieResultModel(BaseModel):
    """One interpreted Movie Mention and its resolution outcome."""

    status: ResultStatus
    movie_mention: MovieMentionModel
    movie: MovieModel | None


class TVSeriesModel(BaseModel):
    """Provider-enriched TV Series metadata in an extraction response."""

    title: str
    first_air_year: int = Field(description="TV First Air Year.")
    last_air_year: int | None = Field(
        description=(
            "Final air year when available. Null means unavailable rather than proof "
            "that the TV Series continues."
        )
    )
    cast: list[str] = Field(
        description=(
            "First five aggregate cast names in provider response order without role "
            "filtering or person deduplication."
        )
    )
    creators: list[str] = Field(
        description=(
            "Creator names from TMDB created_by, preserving provider order after "
            "duplicate-name removal."
        )
    )
    description: str
    poster_url: str | None
    tmdb_id: int
    tmdb_url: str
    imdb_id: str | None
    imdb_url: str | None
    tmdb_score: float = Field(
        ge=0,
        le=10,
        description="TMDB vote average on a zero-to-ten scale.",
    )


class TVSeriesMentionModel(BaseModel):
    """Title and first air year of a TV Series Mention from a transcript."""

    title: str
    year: int = Field(description="TV First Air Year.")


class TVSeriesResultModel(BaseModel):
    """One interpreted TV Series Mention and its resolution outcome."""

    status: ResultStatus
    tv_series_mention: TVSeriesMentionModel
    tv_series: TVSeriesModel | None


class ArtistCreditModel(BaseModel):
    """One Spotify artist credit in provider display order."""

    spotify_artist_id: str
    name: str


class TrackMentionModel(BaseModel):
    """One interpreted Track Mention with explicit optional release context."""

    track_title: str
    artists: list[str]
    release_title: str | None
    release_year: int | None


class TrackModel(BaseModel):
    """Spotify-verified metadata for one playable Track."""

    track_title: str
    artists: list[ArtistCreditModel]
    spotify_track_id: str
    spotify_url: str


class TrackResultModel(BaseModel):
    """One interpreted Track Mention and its resolution outcome."""

    status: ResultStatus
    track_mention: TrackMentionModel
    track: TrackModel | None


class ExtractionResultsModel(BaseModel):
    """Resolved results grouped in independent first-reference order."""

    movies: list[MovieResultModel]
    tv_series: list[TVSeriesResultModel]
    tracks: list[TrackResultModel]


class ExtractResponse(BaseModel):
    """Successful extraction response containing grouped results and effective market."""

    market: SpotifyMarket = Field(
        description="Effective ISO 3166-1 alpha-2 Spotify market.",
    )
    source: SourceModel
    transcript: TranscriptModel
    results: ExtractionResultsModel


class ErrorDetail(BaseModel):
    """Machine-readable error code and human-readable message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Consistent error response body for extraction failures."""

    error: ErrorDetail
