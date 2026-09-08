"""HTTP contract tests for the extraction endpoint."""

import asyncio
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import NoReturn, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from reelio.extraction.exceptions import (
    CatalogProviderError,
    DurationLimitExceededError,
    EnrichmentError,
    ExtractionError,
    InterpretationInputTooLargeError,
    InvalidLLMResponseError,
    InvalidSourceError,
    MetadataProviderError,
    MovieMentionInterpretationError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.router import get_pipeline
from reelio.extraction.schemas import ExtractResponse
from reelio.extraction.service import ExtractionPipeline, ExtractionPipelineProtocol
from reelio.extraction.services.transcription.acquisition import (
    WhisperResult,
    _WhisperProviderFailure,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.inspection import ExtractedMetadata
from reelio.extraction.services.transcription.service import (
    SourceMetadataService,
    TranscriptionService,
)
from reelio.extraction.types import (
    ArtistCredit,
    EnrichedMovie,
    EnrichedMusicRelease,
    EnrichedTrack,
    EnrichedTVSeries,
    ExtractionMentions,
    ExtractionResults,
    MovieMention,
    MovieResult,
    MusicMentions,
    MusicReleaseMention,
    MusicReleaseResult,
    MusicResults,
    PipelineResult,
    Platform,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    TrackMention,
    TrackResult,
    Transcript,
    TranscriptMethod,
    TVSeriesMention,
    TVSeriesResult,
)
from reelio.main import app
from tests.extraction.fakes import (
    FakeInterpretationService as _FakeInterpretationService,
)
from tests.extraction.fakes import FakeResultAggregator as _FakeResultAggregator


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Clear dependency overrides before and after every extraction test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _RaisingPipeline:
    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        raise self._exception

    async def aclose(self) -> None:
        return None


_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"
_DEFAULT_MARKET = SpotifyMarket("US")


class _MetadataExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        return ExtractedMetadata(
            {
                "id": _VIDEO_ID,
                "title": "Router test video",
                "description": "A complete router test description.",
                "channel": "Router test channel",
                "duration": 42.2,
            }
        )


class _SocialMetadataExtractor:
    def __init__(
        self,
        metadata: dict[str, object],
    ) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        return ExtractedMetadata(self.metadata)


def _settings() -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(_env_file=None)


class _CaptionTrack:
    def __init__(self, language_code: str, segments: Sequence[str]) -> None:
        self.language_code = language_code
        self.is_generated = False
        self._segments = segments

    def fetch_segments(self) -> Sequence[str]:
        return self._segments


class _CaptionProvider:
    def __init__(self, tracks: Sequence[_CaptionTrack]) -> None:
        self._tracks = tracks

    def list_tracks(self, video_id: str) -> Sequence[_CaptionTrack]:
        return self._tracks


class _AudioDownloader:
    def download(self, source: object, destination: Path) -> Path:
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        return audio_path


class _FailingWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> NoReturn:
        raise _WhisperProviderFailure("model failure")


class _FixedWhisperTranscriber:
    def __init__(self, result: WhisperResult) -> None:
        self.result = result

    def transcribe(self, audio_path: Path) -> WhisperResult:
        return self.result


class _BlockingWhisperTranscriber:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test worker was not released")
        return WhisperResult(text="concurrent speech", language="en", segment_count=1)


def _transcription_service(provider: _CaptionProvider) -> TranscriptionService:
    return TranscriptionService(
        provider=provider,
        audio_downloader=_AudioDownloader(),
        transcriber=_FailingWhisperTranscriber(),
        temp_media_dir=_settings().temp_media_dir,
        semaphore=asyncio.Semaphore(1),
    )


def _enriched_movie(movie_mention: MovieMention) -> EnrichedMovie:
    return EnrichedMovie(
        title=movie_mention.title,
        year=movie_mention.year,
        cast=[
            "Timothée Chalamet",
            "Rebecca Ferguson",
            "Oscar Isaac",
            "Josh Brolin",
            "Stellan Skarsgård",
        ],
        directors=["Denis Villeneuve"],
        description="Paul Atreides faces his destiny on Arrakis.",
        poster_url="https://image.tmdb.org/t/p/w500/dune.jpg",
        tmdb_id=438631,
        tmdb_url="https://www.themoviedb.org/movie/438631",
        imdb_id="tt1160419",
        imdb_url="https://www.imdb.com/title/tt1160419/",
        tmdb_score=7.8,
    )


def _enriched_tv_series(tv_series_mention: TVSeriesMention) -> EnrichedTVSeries:
    return EnrichedTVSeries(
        title=tv_series_mention.title,
        first_air_year=tv_series_mention.year,
        last_air_year=None,
        cast=["Pedro Pascal", "Bella Ramsey"],
        creators=["Craig Mazin", "Neil Druckmann"],
        description="A smuggler escorts a teenager across a ruined America.",
        poster_url="https://image.tmdb.org/t/p/w500/the-last-of-us.jpg",
        tmdb_id=100088,
        tmdb_url="https://www.themoviedb.org/tv/100088",
        imdb_id="tt3581920",
        imdb_url="https://www.imdb.com/title/tt3581920/",
        tmdb_score=8.6,
    )


def _enriched_track(track_mention: TrackMention) -> EnrichedTrack:
    preferred_music_release = EnrichedMusicRelease(
        release_title="Discovery",
        artists=[
            ArtistCredit(
                spotify_artist_id="4tZwfgrHOc3mvqYlEYSvVi",
                name=track_mention.artists[0],
            )
        ],
        release_date="2001-02-26",
        album_type="album",
        spotify_album_id="track-album",
        spotify_url="https://open.spotify.com/album/track-album",
        cover_url="https://i.scdn.co/image/track-cover",
    )
    return EnrichedTrack(
        track_title=track_mention.track_title,
        artists=[
            ArtistCredit(
                spotify_artist_id="4tZwfgrHOc3mvqYlEYSvVi",
                name=track_mention.artists[0],
            )
        ],
        spotify_track_id="0DiWol3AO6WpXZgp0goxAV",
        spotify_url="https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
        preferred_music_release=preferred_music_release,
        cover_url=preferred_music_release.cover_url,
    )


def _enriched_music_release(
    music_release_mention: MusicReleaseMention,
) -> EnrichedMusicRelease:
    return EnrichedMusicRelease(
        release_title=music_release_mention.release_title,
        artists=[
            ArtistCredit(
                spotify_artist_id="4tZwfgrHOc3mvqYlEYSvVi",
                name=music_release_mention.artists[0],
            )
        ],
        release_date="2001-02-26",
        album_type="album",
        spotify_album_id="direct-album",
        spotify_url="https://open.spotify.com/album/direct-album",
        cover_url="https://i.scdn.co/image/direct-cover",
    )


def _pipeline(
    metadata_service: SourceMetadataService,
    transcription_service: TranscriptionService,
    mentions: ScreenWorkMentions | None = None,
    results: ScreenWorkResults | None = None,
    music_mentions: MusicMentions | None = None,
    music_results: MusicResults | None = None,
) -> ExtractionPipeline:
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    interpreted_screen_works = (
        mentions
        if mentions is not None
        else ScreenWorkMentions(movies=[movie_mention], tv_series=[])
    )
    screen_work_results = (
        results
        if results is not None
        else ScreenWorkResults(
            movies=[
                MovieResult(
                    status=ResultStatus.RESOLVED,
                    movie_mention=interpreted_movie_mention,
                    movie=_enriched_movie(interpreted_movie_mention),
                )
                for interpreted_movie_mention in interpreted_screen_works.movies
            ],
            tv_series=[
                TVSeriesResult(
                    status=ResultStatus.UNRESOLVED,
                    tv_series_mention=interpreted_tv_series_mention,
                    tv_series=None,
                )
                for interpreted_tv_series_mention in interpreted_screen_works.tv_series
            ],
        )
    )
    interpreted_music = (
        music_mentions
        if music_mentions is not None
        else MusicMentions(tracks=[], music_releases=[])
    )
    resolved_music = (
        music_results
        if music_results is not None
        else MusicResults(
            tracks=[
                TrackResult(
                    status=ResultStatus.UNRESOLVED,
                    track_mention=track_mention,
                    track=None,
                )
                for track_mention in interpreted_music.tracks
            ],
            music_releases=[
                MusicReleaseResult(
                    status=ResultStatus.UNRESOLVED,
                    music_release_mention=music_release_mention,
                    music_release=None,
                )
                for music_release_mention in interpreted_music.music_releases
            ],
        )
    )
    return ExtractionPipeline(
        metadata_service,
        transcription_service,
        _FakeInterpretationService(
            ExtractionMentions(
                screen_works=interpreted_screen_works,
                music=interpreted_music,
            )
        ),
        _FakeResultAggregator(
            ExtractionResults(
                screen_works=screen_work_results,
                music=resolved_music,
            )
        ),
    )


def _install_pipeline(application: FastAPI, pipeline: ExtractionPipelineProtocol) -> None:
    application.dependency_overrides[get_pipeline] = lambda: pipeline


async def test_extract_returns_resolved_and_unresolved_screen_work_and_music_results(
    client: AsyncClient,
) -> None:
    """Return grouped results and their complete resolution statistics."""
    metadata_extractor = _MetadataExtractor()
    resolved_movie_mention = MovieMention(title="Dune: Part One", year=2021)
    unresolved_movie_mention = MovieMention(title="Unknown Movie", year=2024)
    resolved_tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    unresolved_tv_series_mention = TVSeriesMention(title="Unknown TV Series", year=2024)
    mentions = ScreenWorkMentions(
        movies=[resolved_movie_mention, unresolved_movie_mention],
        tv_series=[resolved_tv_series_mention, unresolved_tv_series_mention],
    )
    results = ScreenWorkResults(
        movies=[
            MovieResult(
                status=ResultStatus.RESOLVED,
                movie_mention=resolved_movie_mention,
                movie=_enriched_movie(resolved_movie_mention),
            ),
            MovieResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=unresolved_movie_mention,
                movie=None,
            ),
        ],
        tv_series=[
            TVSeriesResult(
                status=ResultStatus.RESOLVED,
                tv_series_mention=resolved_tv_series_mention,
                tv_series=_enriched_tv_series(resolved_tv_series_mention),
            ),
            TVSeriesResult(
                status=ResultStatus.UNRESOLVED,
                tv_series_mention=unresolved_tv_series_mention,
                tv_series=None,
            ),
        ],
    )
    resolved_track_mention = TrackMention(
        track_title="One More Time",
        artists=["Daft Punk"],
        release_title="Discovery",
        release_year=2001,
    )
    unresolved_track_mention = TrackMention(
        track_title="Unknown Track",
        artists=["Unknown Artist"],
        release_title=None,
        release_year=None,
    )
    resolved_music_release_mention = MusicReleaseMention(
        release_title="Discovery",
        artists=["Daft Punk"],
        release_year=2001,
    )
    unresolved_music_release_mention = MusicReleaseMention(
        release_title="Unknown Album",
        artists=["Unknown Artist"],
        release_year=None,
    )
    music_mentions = MusicMentions(
        tracks=[resolved_track_mention, unresolved_track_mention],
        music_releases=[
            resolved_music_release_mention,
            unresolved_music_release_mention,
        ],
    )
    music_results = MusicResults(
        tracks=[
            TrackResult(
                status=ResultStatus.RESOLVED,
                track_mention=resolved_track_mention,
                track=_enriched_track(resolved_track_mention),
            ),
            TrackResult(
                status=ResultStatus.UNRESOLVED,
                track_mention=unresolved_track_mention,
                track=None,
            ),
        ],
        music_releases=[
            MusicReleaseResult(
                status=ResultStatus.RESOLVED,
                music_release_mention=resolved_music_release_mention,
                music_release=_enriched_music_release(resolved_music_release_mention),
            ),
            MusicReleaseResult(
                status=ResultStatus.UNRESOLVED,
                music_release_mention=unresolved_music_release_mention,
                music_release=None,
            ),
        ],
    )
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=metadata_extractor,
            settings=_settings(),
        ),
        _transcription_service(
            _CaptionProvider([_CaptionTrack("en-GB", ["Router", "caption text."])])
        ),
        mentions,
        results,
        music_mentions=music_mentions,
        music_results=music_results,
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == "youtube"
    assert payload.source.video_id == _VIDEO_ID
    assert payload.source.url == _CANONICAL_URL
    assert payload.source.title == "Router test video"
    assert payload.source.description == "A complete router test description."
    assert payload.source.channel == "Router test channel"
    assert payload.source.duration_seconds == 43
    assert metadata_extractor.calls == [_CANONICAL_URL]
    assert payload.transcript.language == "en-GB"
    assert payload.transcript.method == "youtube_captions"
    assert payload.transcript.text == "Router caption text."

    raw_response = response.json()
    assert list(raw_response) == ["market", "source", "transcript", "statistics", "results"]
    raw_statistics = raw_response["statistics"]
    assert raw_statistics == {
        "movies": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tv_series": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tracks": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "music_releases": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    }
    raw_results = raw_response["results"]
    assert [item["status"] for item in raw_results["movies"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["tv_series"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["tracks"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["music_releases"]] == [
        "resolved",
        "unresolved",
    ]
    assert raw_results["movies"][1]["movie"] is None
    assert "movie" in raw_results["movies"][1]
    assert raw_results["tv_series"][1]["tv_series"] is None
    assert "tv_series" in raw_results["tv_series"][1]
    assert raw_results["tracks"][1]["track"] is None
    assert "track" in raw_results["tracks"][1]
    assert raw_results["music_releases"][1]["music_release"] is None
    assert "music_release" in raw_results["music_releases"][1]
    assert set(raw_results["tracks"][0]["track"]) == {
        "track_title",
        "artists",
        "spotify_track_id",
        "spotify_url",
        "preferred_music_release",
        "cover_url",
    }
    assert set(raw_results["tracks"][0]["track"]["preferred_music_release"]) == {
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    }
    assert raw_results["tracks"][0]["track"]["cover_url"] == ("https://i.scdn.co/image/track-cover")
    assert raw_results["music_releases"][0]["music_release_mention"] == {
        "release_title": "Discovery",
        "artists": ["Daft Punk"],
        "release_year": 2001,
    }
    assert set(raw_results["music_releases"][0]["music_release"]) == {
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    }
    assert set(raw_results["tv_series"][0]["tv_series"]) == {
        "title",
        "first_air_year",
        "last_air_year",
        "cast",
        "creators",
        "description",
        "poster_url",
        "tmdb_id",
        "tmdb_url",
        "imdb_id",
        "imdb_url",
        "tmdb_score",
    }

    resolved_movie = payload.results.movies[0]
    assert resolved_movie.status is ResultStatus.RESOLVED
    assert resolved_movie.movie_mention.title == resolved_movie_mention.title
    assert resolved_movie.movie_mention.year == resolved_movie_mention.year
    assert resolved_movie.movie is not None
    assert resolved_movie.movie.tmdb_id == 438631
    assert resolved_movie.movie.cast == [
        "Timothée Chalamet",
        "Rebecca Ferguson",
        "Oscar Isaac",
        "Josh Brolin",
        "Stellan Skarsgård",
    ]
    assert resolved_movie.movie.directors == ["Denis Villeneuve"]
    assert payload.results.movies[1].movie is None

    resolved_tv_series = payload.results.tv_series[0]
    assert resolved_tv_series.status is ResultStatus.RESOLVED
    assert resolved_tv_series.tv_series_mention.title == resolved_tv_series_mention.title
    assert resolved_tv_series.tv_series_mention.year == resolved_tv_series_mention.year
    assert resolved_tv_series.tv_series is not None
    assert resolved_tv_series.tv_series.title == "The Last of Us"
    assert resolved_tv_series.tv_series.first_air_year == 2023
    assert resolved_tv_series.tv_series.last_air_year is None
    assert resolved_tv_series.tv_series.cast == ["Pedro Pascal", "Bella Ramsey"]
    assert resolved_tv_series.tv_series.creators == ["Craig Mazin", "Neil Druckmann"]
    assert (
        resolved_tv_series.tv_series.description
        == "A smuggler escorts a teenager across a ruined America."
    )
    assert (
        resolved_tv_series.tv_series.poster_url
        == "https://image.tmdb.org/t/p/w500/the-last-of-us.jpg"
    )
    assert resolved_tv_series.tv_series.tmdb_id == 100088
    assert resolved_tv_series.tv_series.tmdb_url == "https://www.themoviedb.org/tv/100088"
    assert resolved_tv_series.tv_series.imdb_id == "tt3581920"
    assert resolved_tv_series.tv_series.imdb_url == "https://www.imdb.com/title/tt3581920/"
    assert resolved_tv_series.tv_series.tmdb_score == 8.6
    assert payload.results.tv_series[1].tv_series is None
    resolved_track = payload.results.tracks[0]
    assert resolved_track.status is ResultStatus.RESOLVED
    assert resolved_track.track_mention.track_title == resolved_track_mention.track_title
    assert resolved_track.track_mention.artists == resolved_track_mention.artists
    assert resolved_track.track_mention.release_title == resolved_track_mention.release_title
    assert resolved_track.track_mention.release_year == resolved_track_mention.release_year
    assert resolved_track.track is not None
    assert resolved_track.track.track_title == "One More Time"
    assert resolved_track.track.artists[0].spotify_artist_id == "4tZwfgrHOc3mvqYlEYSvVi"
    assert resolved_track.track.artists[0].name == "Daft Punk"
    assert resolved_track.track.spotify_track_id == "0DiWol3AO6WpXZgp0goxAV"
    assert (
        resolved_track.track.spotify_url == "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV"
    )
    assert resolved_track.track.preferred_music_release.spotify_album_id == "track-album"
    assert (
        resolved_track.track.preferred_music_release.cover_url
        == "https://i.scdn.co/image/track-cover"
    )
    assert resolved_track.track.cover_url == resolved_track.track.preferred_music_release.cover_url
    assert payload.results.tracks[1].track is None
    resolved_music_release = payload.results.music_releases[0]
    assert resolved_music_release.status is ResultStatus.RESOLVED
    assert (
        resolved_music_release.music_release_mention.release_title
        == resolved_music_release_mention.release_title
    )
    assert (
        resolved_music_release.music_release_mention.artists
        == resolved_music_release_mention.artists
    )
    assert (
        resolved_music_release.music_release_mention.release_year
        == resolved_music_release_mention.release_year
    )
    assert resolved_music_release.music_release is not None
    assert resolved_music_release.music_release.release_title == "Discovery"
    assert resolved_music_release.music_release.artists[0].spotify_artist_id == (
        "4tZwfgrHOc3mvqYlEYSvVi"
    )
    assert resolved_music_release.music_release.artists[0].name == "Daft Punk"
    assert resolved_music_release.music_release.release_date == "2001-02-26"
    assert resolved_music_release.music_release.album_type == "album"
    assert resolved_music_release.music_release.spotify_album_id == "direct-album"
    assert (
        resolved_music_release.music_release.spotify_url
        == "https://open.spotify.com/album/direct-album"
    )
    assert resolved_music_release.music_release.cover_url == "https://i.scdn.co/image/direct-cover"
    assert payload.results.music_releases[1].status is ResultStatus.UNRESOLVED
    assert (
        payload.results.music_releases[1].music_release_mention.release_title
        == unresolved_music_release_mention.release_title
    )
    assert (
        payload.results.music_releases[1].music_release_mention.artists
        == unresolved_music_release_mention.artists
    )
    assert (
        payload.results.music_releases[1].music_release_mention.release_year
        == unresolved_music_release_mention.release_year
    )
    assert payload.results.music_releases[1].music_release is None


async def test_extract_maps_unavailable_captions_to_502(
    client: AsyncClient,
) -> None:
    """Map a valid Source with no usable captions to Transcript Unavailable."""
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        _transcription_service(_CaptionProvider([])),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }


@pytest.mark.parametrize(
    ("mentions", "expected_movies", "expected_tv_series"),
    [
        (ScreenWorkMentions(movies=[], tv_series=[]), [], []),
        (
            ScreenWorkMentions(
                movies=[MovieMention(title="Dune: Part One", year=2021)],
                tv_series=[],
            ),
            ["Dune: Part One"],
            [],
        ),
        (
            ScreenWorkMentions(
                movies=[],
                tv_series=[
                    TVSeriesMention(title="The Last of Us", year=2023),
                    TVSeriesMention(title="Arcane", year=2021),
                ],
            ),
            [],
            ["The Last of Us", "Arcane"],
        ),
        (
            ScreenWorkMentions(
                movies=[MovieMention(title="Dune: Part One", year=2021)],
                tv_series=[
                    TVSeriesMention(title="Arcane", year=2021),
                    TVSeriesMention(title="The Last of Us", year=2023),
                ],
            ),
            ["Dune: Part One"],
            ["Arcane", "The Last of Us"],
        ),
    ],
    ids=["empty", "movie-only", "tv-only", "mixed"],
)
async def test_extract_groups_screen_work_results(
    client: AsyncClient,
    mentions: ScreenWorkMentions,
    expected_movies: list[str],
    expected_tv_series: list[str],
) -> None:
    """Serialize always-present independently ordered result lists, including Tracks."""
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        _transcription_service(_CaptionProvider([_CaptionTrack("en", ["Grouped", "results."])])),
        mentions,
    )
    _install_pipeline(app, pipeline)

    response = await client.post("/api/extract", json={"url": _CANONICAL_URL})

    assert response.status_code == 200
    results = response.json()["results"]
    assert set(results) == {"movies", "tv_series", "tracks", "music_releases"}
    assert [item["movie_mention"]["title"] for item in results["movies"]] == expected_movies
    assert [
        item["tv_series_mention"]["title"] for item in results["tv_series"]
    ] == expected_tv_series
    assert results["tracks"] == []
    assert results["music_releases"] == []
    assert all(set(item) == {"status", "movie_mention", "movie"} for item in results["movies"])
    assert all(
        set(item) == {"status", "tv_series_mention", "tv_series"} for item in results["tv_series"]
    )
    assert all(item["tv_series"] is None for item in results["tv_series"])


async def test_extract_returns_whisper_transcript(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Serialize a successful Whisper Transcript through the unchanged schema."""
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        TranscriptionService(
            provider=_CaptionProvider([]),
            audio_downloader=_AudioDownloader(),
            transcriber=_FixedWhisperTranscriber(
                WhisperResult(
                    text="Spoken audio text.",
                    language="en",
                    segment_count=2,
                )
            ),
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)
    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.transcript.text == "Spoken audio text."
    assert payload.transcript.language == "en"
    assert payload.transcript.method == "whisper"


@pytest.mark.parametrize(
    (
        "submitted_url",
        "provider_url",
        "canonical_url",
        "extractor_key",
        "video_id",
        "platform",
    ),
    [
        (
            "https://www.instagram.com/reel/ABC123",
            "https://www.instagram.com/reel/ABC123",
            "https://www.instagram.com/reel/ABC123",
            "Instagram",
            "ABC123",
            "instagram",
        ),
        (
            "https://www.facebook.com/reel/123456789",
            "https://www.facebook.com/reel/123456789",
            "https://www.facebook.com/reel/123456789",
            "FacebookReel",
            "123456789",
            "facebook",
        ),
        (
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "TikTok",
            "1234567890123456789",
            "tiktok",
        ),
        (
            "https://twitter.com/creator/status/1234567890123456789",
            "https://twitter.com/creator/status/1234567890123456789",
            "https://twitter.com/creator/status/1234567890123456789",
            "Twitter",
            "1234567890123456789",
            "x",
        ),
    ],
)
async def test_social_sources_serialize_unchanged_response_schema(
    client: AsyncClient,
    tmp_path: Path,
    submitted_url: str,
    provider_url: str,
    canonical_url: str,
    extractor_key: str,
    video_id: str,
    platform: str,
) -> None:
    """Serialize every social Source with a direct Whisper Transcript."""
    extractor = _SocialMetadataExtractor(
        {
            "id": video_id,
            "extractor_key": extractor_key,
            "webpage_url": canonical_url,
            "title": "Social router video",
            "description": "Social router description",
            "channel": "Social router channel",
            "duration": 42.2,
            "formats": [{"vcodec": "avc1"}],
        }
    )
    pipeline = _pipeline(
        SourceMetadataService(extractor=extractor, settings=_settings()),
        TranscriptionService(
            provider=_CaptionProvider([_CaptionTrack("en", ["must", "not", "run"])]),
            audio_downloader=_AudioDownloader(),
            transcriber=_FixedWhisperTranscriber(
                WhisperResult(
                    text="Social router speech.",
                    language="en",
                    segment_count=1,
                )
            ),
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)

    response = await client.post("/api/extract", json={"url": submitted_url})

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == platform
    assert payload.source.video_id == video_id
    assert payload.source.url == canonical_url
    assert payload.source.title == "Social router video"
    assert payload.source.description == "Social router description"
    assert payload.source.channel == "Social router channel"
    assert payload.source.duration_seconds == 43
    assert payload.transcript.method == "whisper"
    assert payload.transcript.text == "Social router speech."
    assert extractor.calls == [provider_url]


async def test_concurrent_whisper_http_requests_queue_and_succeed(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Queue concurrent endpoint fallbacks behind one shared service semaphore."""
    transcriber = _BlockingWhisperTranscriber()
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        TranscriptionService(
            provider=_CaptionProvider([]),
            audio_downloader=_AudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)

    first = asyncio.create_task(client.post("/api/extract", json={"url": _CANONICAL_URL}))
    assert await asyncio.to_thread(transcriber.started.wait, 5)
    second = asyncio.create_task(client.post("/api/extract", json={"url": _CANONICAL_URL}))
    await asyncio.sleep(0)

    assert transcriber.calls == 1

    transcriber.release.set()
    first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert transcriber.calls == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (InvalidSourceError("invalid source"), 400, "invalid_source"),
        (
            UnsupportedPlatformError("unsupported platform"),
            400,
            "unsupported_platform",
        ),
        (SourceUnavailableError("source unavailable"), 404, "source_unavailable"),
        (
            DurationLimitExceededError("duration limit exceeded"),
            413,
            "duration_limit_exceeded",
        ),
        (
            InterpretationInputTooLargeError("interpretation input too large"),
            413,
            "interpretation_input_too_large",
        ),
        (
            MetadataProviderError("Unable to retrieve YouTube metadata."),
            502,
            "metadata_provider_failed",
        ),
        (TranscriptionError("transcription failed"), 502, "transcription_failed"),
        (
            MovieMentionInterpretationError("movie mention interpretation failed"),
            502,
            "movie_mention_interpretation_failed",
        ),
        (
            InvalidLLMResponseError("invalid provider response"),
            502,
            "invalid_llm_response",
        ),
        (EnrichmentError("enrichment failed"), 502, "enrichment_failed"),
        (
            CatalogProviderError("Spotify catalog request failed."),
            502,
            "catalog_provider_failed",
        ),
        (PipelineTimeoutError("pipeline timed out"), 504, "pipeline_timeout"),
    ],
)
async def test_extraction_errors_map_to_contract(
    client: AsyncClient,
    exception: ExtractionError,
    expected_status: int,
    expected_code: str,
) -> None:
    """Map every extraction domain error to its stable HTTP contract."""
    _install_pipeline(app, _RaisingPipeline(exception))

    response = await client.post(
        "/api/extract",
        json={"url": "https://www.youtube.com/watch?v=anything"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"error": {"code": expected_code, "message": str(exception)}}


@pytest.mark.parametrize("payload", [{}, {"url": 123}, {"url": ""}])
async def test_malformed_requests_keep_fastapi_422_contract(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    """Keep FastAPI validation responses for malformed request bodies."""
    _install_pipeline(app, _RaisingPipeline(RuntimeError("unused")))
    response = await client.post("/api/extract", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


async def test_unhandled_failures_do_not_leak_internals() -> None:
    """Return a generic 500 response when the pipeline raises unexpectedly."""
    _install_pipeline(
        app,
        _RaisingPipeline(RuntimeError("sensitive database path /var/reelio/secret.db")),
    )
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extract",
            json={"url": "https://www.youtube.com/watch?v=anything"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred.",
        }
    }
    assert "sensitive database path" not in response.text


async def test_extract_is_documented_in_openapi(client: AsyncClient) -> None:
    """Document result statistics, grouped Music resolution, TV metadata, and failures."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/extract"]["post"]
    responses = operation["responses"]
    description = operation["description"]
    assert "offset zero and limit three" in description
    assert "first ordered Artist Credit only" in description
    assert "at least one Artist Credit matches any Mention Artist Credit" in description
    assert "exact title equality across every artist-eligible Candidate" in description
    assert "before reusing the same bounded artist-eligible Candidate sequence" in description
    assert "Equivalent Track Versions" in description
    assert "complete trailing remaster and bonus designations" in description
    assert "release-only Track variants" in description
    assert "explicit Music Release context" in description
    assert "attached Album title that is exact or equivalent" in description
    assert "without release context, the attached Album does not constrain matching" in description
    assert "same bounded artist-eligible Candidate sequence" in description
    assert "without another Spotify search" in description
    assert "right to left" in description
    assert "parentheses, brackets, a spaced hyphen, or colon" in description
    assert "album or single Candidates" in description
    assert "blocked trailing segments" in description
    assert "ordinary Resolved Result" in description
    assert "Release year does not participate in retrieval or Candidate verification" in description
    assert "fuzzy" not in description.lower()
    assert "90 percent" not in description.lower()
    assert "every ordered Artist Credit" not in description
    assert "release year must match" not in description.lower()
    assert {"200", "400", "404", "413", "500", "502", "504", "422"} <= set(responses)
    for status_code in ("400", "404", "413", "500", "502", "504"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/ExtractRequest"}
    schemas = document["components"]["schemas"]
    extract_request = schemas["ExtractRequest"]
    assert extract_request["required"] == ["url"]
    request_market = extract_request["properties"]["market"]
    assert request_market["examples"] == ["US", "JP"]
    assert request_market["anyOf"][0]["pattern"] == "^[A-Z]{2}$"
    source_properties = schemas["SourceModel"]["properties"]
    assert {
        "platform",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "duration_seconds",
    } <= set(source_properties)
    response_schema = responses["200"]["content"]["application/json"]["schema"]
    assert response_schema == {"$ref": "#/components/schemas/ExtractResponse"}

    example = responses["200"]["content"]["application/json"]["example"]
    assert example["market"] == "US"
    assert example["statistics"] == {
        "movies": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tv_series": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tracks": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "music_releases": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    }
    assert set(example["results"]) == {
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
    }
    resolved_tv_series_example = example["results"]["tv_series"][0]
    assert resolved_tv_series_example["status"] == "resolved"
    assert set(resolved_tv_series_example["tv_series"]) == {
        "title",
        "first_air_year",
        "last_air_year",
        "cast",
        "creators",
        "description",
        "poster_url",
        "tmdb_id",
        "tmdb_url",
        "imdb_id",
        "imdb_url",
        "tmdb_score",
    }
    unresolved_tv_series_example = example["results"]["tv_series"][1]
    assert unresolved_tv_series_example["status"] == "unresolved"
    assert unresolved_tv_series_example.get("tv_series") is None
    resolved_track_example = example["results"]["tracks"][0]
    assert resolved_track_example["status"] == "resolved"
    assert resolved_track_example["track_mention"] == {
        "track_title": "One More Time",
        "artists": ["Daft Punk"],
        "release_title": "Discovery",
        "release_year": 2001,
    }
    assert resolved_track_example["track"]["track_title"] == ("One More Time (2011 Remaster)")
    assert (
        resolved_track_example["track"]["track_title"]
        != resolved_track_example["track_mention"]["track_title"]
    )
    assert set(resolved_track_example["track"]) == {
        "track_title",
        "artists",
        "spotify_track_id",
        "spotify_url",
        "preferred_music_release",
        "cover_url",
    }
    assert set(resolved_track_example["track"]["preferred_music_release"]) == {
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    }
    assert resolved_track_example["track"]["preferred_music_release"]["release_title"] == (
        "Discovery (Deluxe Edition)"
    )
    assert (
        resolved_track_example["track"]["preferred_music_release"]["release_title"]
        != resolved_track_example["track_mention"]["release_title"]
    )
    assert (
        resolved_track_example["track"]["cover_url"]
        == resolved_track_example["track"]["preferred_music_release"]["cover_url"]
    )
    unresolved_track_example = example["results"]["tracks"][1]
    assert unresolved_track_example["status"] == "unresolved"
    assert unresolved_track_example.get("track") is None
    resolved_music_release_example = example["results"]["music_releases"][0]
    assert resolved_music_release_example["status"] == "resolved"
    assert resolved_music_release_example["music_release_mention"] == {
        "release_title": "Discovery",
        "artists": ["Daft Punk"],
        "release_year": 2001,
    }
    assert set(resolved_music_release_example["music_release"]) == {
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    }
    assert resolved_music_release_example["music_release"]["release_title"] == (
        "Discovery (Deluxe Edition)"
    )
    assert (
        resolved_music_release_example["music_release"]["release_title"]
        != resolved_music_release_example["music_release_mention"]["release_title"]
    )
    assert (
        resolved_music_release_example["music_release"]["cover_url"]
        == "https://i.scdn.co/image/discovery-cover"
    )
    unresolved_music_release_example = example["results"]["music_releases"][1]
    assert unresolved_music_release_example["status"] == "unresolved"
    assert (
        unresolved_music_release_example["music_release_mention"]["release_title"]
        == "Unknown Album"
    )
    assert unresolved_music_release_example["music_release_mention"]["artists"] == [
        "Unknown Artist"
    ]
    assert unresolved_music_release_example.get("music_release") is None

    assert "ResultModel" not in schemas
    assert all("cover" not in schema_name.casefold() for schema_name in schemas)
    extract_response_schema = schemas["ExtractResponse"]
    extract_response_properties = extract_response_schema["properties"]
    assert extract_response_schema["required"] == [
        "market",
        "source",
        "transcript",
        "statistics",
        "results",
    ]
    assert extract_response_properties["market"]["pattern"] == "^[A-Z]{2}$"
    assert extract_response_properties["statistics"] == {
        "$ref": "#/components/schemas/ExtractionStatisticsModel"
    }
    assert extract_response_properties["results"] == {
        "$ref": "#/components/schemas/ExtractionResultsModel"
    }
    extraction_results = schemas["ExtractionResultsModel"]
    assert extraction_results["required"] == [
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
    ]
    assert extraction_results["properties"]["movies"] == {
        "items": {"$ref": "#/components/schemas/MovieResultModel"},
        "type": "array",
        "title": "Movies",
    }
    assert extraction_results["properties"]["tv_series"] == {
        "items": {"$ref": "#/components/schemas/TVSeriesResultModel"},
        "type": "array",
        "title": "Tv Series",
    }
    assert extraction_results["properties"]["tracks"] == {
        "items": {"$ref": "#/components/schemas/TrackResultModel"},
        "type": "array",
        "title": "Tracks",
    }
    assert extraction_results["properties"]["music_releases"] == {
        "items": {"$ref": "#/components/schemas/MusicReleaseResultModel"},
        "type": "array",
        "title": "Music Releases",
    }
    extraction_statistics = schemas["ExtractionStatisticsModel"]
    assert extraction_statistics["required"] == [
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
    ]
    assert extraction_statistics["properties"] == {
        "movies": {"$ref": "#/components/schemas/ResultCountsModel"},
        "tv_series": {"$ref": "#/components/schemas/ResultCountsModel"},
        "tracks": {"$ref": "#/components/schemas/ResultCountsModel"},
        "music_releases": {"$ref": "#/components/schemas/ResultCountsModel"},
    }
    result_counts = schemas["ResultCountsModel"]
    assert result_counts["required"] == ["n_mentions", "n_resolved", "n_unresolved"]
    for count_name in result_counts["required"]:
        assert result_counts["properties"][count_name]["minimum"] == 0

    movie_result_schema = schemas["MovieResultModel"]
    assert {"status", "movie_mention", "movie"} <= set(movie_result_schema["required"])
    assert movie_result_schema["properties"]["movie_mention"] == {
        "$ref": "#/components/schemas/MovieMentionModel"
    }
    tv_series_result_schema = schemas["TVSeriesResultModel"]
    assert {"status", "tv_series_mention", "tv_series"} <= set(tv_series_result_schema["required"])
    assert tv_series_result_schema["properties"]["tv_series_mention"] == {
        "$ref": "#/components/schemas/TVSeriesMentionModel"
    }
    assert tv_series_result_schema["properties"]["tv_series"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/TVSeriesModel"},
            {"type": "null"},
        ],
    }
    track_result_schema = schemas["TrackResultModel"]
    assert {"status", "track_mention", "track"} <= set(track_result_schema["required"])
    assert track_result_schema["properties"]["track_mention"] == {
        "$ref": "#/components/schemas/TrackMentionModel"
    }
    assert track_result_schema["properties"]["track"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/TrackModel"},
            {"type": "null"},
        ],
    }
    music_release_result_schema = schemas["MusicReleaseResultModel"]
    assert {"status", "music_release_mention", "music_release"} <= set(
        music_release_result_schema["required"]
    )
    assert music_release_result_schema["properties"]["music_release_mention"] == {
        "$ref": "#/components/schemas/MusicReleaseMentionModel"
    }
    assert music_release_result_schema["properties"]["music_release"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/MusicReleaseModel"},
            {"type": "null"},
        ],
    }
    music_release_mention_schema = schemas["MusicReleaseMentionModel"]
    assert music_release_mention_schema["required"] == [
        "release_title",
        "artists",
        "release_year",
    ]
    music_release_schema = schemas["MusicReleaseModel"]
    assert music_release_schema["required"] == [
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    ]
    assert music_release_schema["properties"]["artists"] == {
        "items": {"$ref": "#/components/schemas/ArtistCreditModel"},
        "type": "array",
        "title": "Artists",
    }
    assert set(music_release_schema["properties"]["album_type"]["enum"]) == {
        "album",
        "single",
        "compilation",
    }

    tv_series_schema = schemas["TVSeriesModel"]
    assert {
        "title",
        "first_air_year",
        "last_air_year",
        "cast",
        "creators",
        "description",
        "poster_url",
        "tmdb_id",
        "tmdb_url",
        "imdb_id",
        "imdb_url",
        "tmdb_score",
    } <= set(tv_series_schema["required"])
    tv_series_properties = tv_series_schema["properties"]
    assert tv_series_properties["tmdb_score"]["minimum"] == 0
    assert tv_series_properties["tmdb_score"]["maximum"] == 10
    assert (
        tv_series_properties["tmdb_score"]["description"]
        == "TMDB vote average on a zero-to-ten scale."
    )
    assert tv_series_properties["first_air_year"]["description"] == "TV First Air Year."
    assert "unavailable rather than proof" in tv_series_properties["last_air_year"]["description"]
    assert "First five aggregate cast" in tv_series_properties["cast"]["description"]
    assert "created_by" in tv_series_properties["creators"]["description"]

    movie_mention_schema = schemas["MovieMentionModel"]
    assert {"title", "year"} <= set(movie_mention_schema["required"])
    tv_series_mention_schema = schemas["TVSeriesMentionModel"]
    assert {"title", "year"} <= set(tv_series_mention_schema["required"])
    assert tv_series_mention_schema["properties"]["year"]["description"] == "TV First Air Year."
    movie_schema = schemas["MovieModel"]
    assert {"year", "tmdb_score"} <= set(movie_schema["required"])
    track_mention_schema = schemas["TrackMentionModel"]
    assert track_mention_schema["required"] == [
        "track_title",
        "artists",
        "release_title",
        "release_year",
    ]
    track_schema = schemas["TrackModel"]
    assert track_schema["required"] == [
        "track_title",
        "artists",
        "spotify_track_id",
        "spotify_url",
        "preferred_music_release",
        "cover_url",
    ]
    assert track_schema["properties"]["artists"] == {
        "items": {"$ref": "#/components/schemas/ArtistCreditModel"},
        "type": "array",
        "title": "Artists",
    }
    music_release_cover = music_release_schema["properties"]["cover_url"]
    assert music_release_cover["anyOf"] == [{"type": "string"}, {"type": "null"}]
    track_cover = track_schema["properties"]["cover_url"]
    assert track_cover["anyOf"] == [{"type": "string"}, {"type": "null"}]
    preferred_music_release = track_schema["properties"]["preferred_music_release"]
    assert preferred_music_release.get(
        "$ref"
    ) == "#/components/schemas/MusicReleaseModel" or preferred_music_release.get("allOf") == [
        {"$ref": "#/components/schemas/MusicReleaseModel"}
    ]
    artist_credit_schema = schemas["ArtistCreditModel"]
    assert artist_credit_schema["required"] == ["spotify_artist_id", "name"]
    assert set(document["components"]["schemas"]["Platform"]["enum"]) == {
        "youtube",
        "instagram",
        "facebook",
        "tiktok",
        "x",
    }
    assert "YouTube, Instagram, Facebook, TikTok, or X" in operation["description"]
    assert "first-reference order" in operation["description"]
    assert "Each statistics category counts returned results" in operation["description"]
    assert "Track Results retain their interpreted Track Mention" in operation["description"]
    assert (
        "Music Release Results retain their interpreted Music Release Mention"
        in operation["description"]
    )
    assert "provider-reported release date" in operation["description"]
    assert "preferred_music_release is the Spotify Album attached" in operation["description"]
    assert "first provider-ordered Spotify-hosted Album image URL" in operation["description"]
    assert "artwork link-back" in operation["description"]
    assert "worldwide-edition" in operation["description"]
    assert "Music Releases" in operation["summary"]
    assert (
        "Any TMDB or Spotify provider failure fails the complete request."
        in operation["description"]
    )
    assert (
        "Any TMDB or Spotify provider failure fails the complete request."
        in responses["502"]["description"]
    )


class _MarketPipeline:
    """Return a deterministic Pipeline Result while recording effective-market input."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, SpotifyMarket | None]] = []

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Record the request market and return an otherwise empty extraction result."""
        self.calls.append((url, market))
        return PipelineResult(
            source=Source(
                platform=Platform.YOUTUBE,
                video_id=_VIDEO_ID,
                url=url,
                title="Market test",
                description="",
                channel="Channel",
                duration_seconds=1,
            ),
            transcript=Transcript(
                text="",
                language="en",
                method=TranscriptMethod.YOUTUBE_CAPTIONS,
            ),
            results=ExtractionResults(
                screen_works=ScreenWorkResults(movies=[], tv_series=[]),
                music=MusicResults(tracks=[], music_releases=[]),
            ),
            market=market or _DEFAULT_MARKET,
        )

    async def aclose(self) -> None:
        """Satisfy the lifespan-owned pipeline protocol."""


async def test_extract_validates_forwards_and_exposes_the_effective_market(
    client: AsyncClient,
) -> None:
    """Use the supplied market or configured US default through the HTTP contract."""
    pipeline = _MarketPipeline()
    _install_pipeline(app, pipeline)

    explicit_response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL, "market": "JP"},
    )
    default_response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )
    invalid_response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL, "market": "jp"},
    )

    assert explicit_response.status_code == 200
    assert explicit_response.json()["market"] == "JP"
    assert explicit_response.json()["statistics"] == {
        "movies": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "tv_series": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "tracks": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "music_releases": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
    }
    assert default_response.status_code == 200
    assert default_response.json()["market"] == "US"
    assert invalid_response.status_code == 422
    assert pipeline.calls == [(_CANONICAL_URL, "JP"), (_CANONICAL_URL, None)]
