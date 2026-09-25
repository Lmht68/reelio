"""HTTP contract tests for the extraction endpoint."""

import asyncio
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from itertools import count
from pathlib import Path
from typing import NoReturn, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from yt_dlp.utils import YoutubeDLError

import reelio.extraction.services.transcription.service as transcription_service_module
from reelio.cache import DisabledCache, RedisCache
from reelio.cache.redis import _CacheRuntime
from reelio.extraction.exceptions import (
    CatalogProviderError,
    DurationLimitExceededError,
    EnrichmentError,
    ExtractionError,
    InterpretationInputTooLargeError,
    InvalidLLMResponseError,
    InvalidSourceError,
    MentionInterpretationError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.router import get_pipeline
from reelio.extraction.schemas import ExtractResponse, TranscriptExtractResponse
from reelio.extraction.service import ExtractionPipeline, ExtractionPipelineProtocol
from reelio.extraction.services.interpretation.config import (
    InterpretationConfig,
    LLMProvider,
)
from reelio.extraction.services.interpretation.service import (
    MentionInterpretationService,
)
from reelio.extraction.services.interpretation.types import LLMMessage
from reelio.extraction.services.transcription.acquisition import (
    WhisperResult,
    _WhisperProviderFailure,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.inspection import ExtractedMetadata
from reelio.extraction.services.transcription.service import (
    CachedTranscriptionService,
    SourceMetadataService,
    TranscriptionService,
)
from reelio.extraction.types import (
    ArtistCredit,
    AuthorCredit,
    BookEdition,
    BookMention,
    BookMentions,
    BookResult,
    BookResults,
    EnrichedAuthorCredit,
    EnrichedBookWork,
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
    TranscriptPipelineResult,
    TVSeriesMention,
    TVSeriesResult,
)
from reelio.main import app
from tests.cache.fakes import FakeRedis, ManualClock
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

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        raise self._exception

    async def aclose(self) -> None:
        return None


_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"
_DEFAULT_MARKET = SpotifyMarket("US")


class _RecordingSourceMetadataService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def inspect(self, submitted_url: str) -> NoReturn:
        self.calls.append(submitted_url)
        raise AssertionError("submitted transcripts must not inspect a Source")


class _RecordingTranscriptionService:
    def __init__(self) -> None:
        self.calls: list[tuple[Source, str]] = []

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: object | None = None,
    ) -> NoReturn:
        self.calls.append((source, submitted_url))
        raise AssertionError("submitted transcripts must not acquire media")


class _RecordingLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[LLMMessage, ...]] = []
        self.closed = False
        self.provider_name = LLMProvider.DEEPSEEK
        self.model_name = "recording-model"

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        self.calls.append(tuple(messages))
        return self.response

    async def aclose(self) -> None:
        self.closed = True


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


class _OutcomeMetadataExtractor:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        if not self._outcomes:
            raise AssertionError("Unexpected metadata extraction")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ExtractedMetadata(cast(dict[str, object], outcome))


def _router_cache(clock: ManualClock) -> tuple[RedisCache, FakeRedis]:
    tokens = count()
    redis = FakeRedis(clock)
    cache = RedisCache(
        redis,
        "reelio:test",
        b"router-source-cache-test-key",
        runtime=_CacheRuntime(clock, asyncio.sleep, lambda: f"router-{next(tokens)}"),
    )
    return cache, redis


def _social_transcription_service(tmp_path: Path) -> TranscriptionService:
    return TranscriptionService(
        provider=_CaptionProvider([]),
        audio_downloader=_AudioDownloader(),
        transcriber=_FixedWhisperTranscriber(
            WhisperResult(
                text="Social cache transcript.",
                language="en",
                segment_count=1,
            )
        ),
        temp_media_dir=tmp_path,
        semaphore=asyncio.Semaphore(1),
    )


def _settings() -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(_env_file=None)


def _interpretation_settings(**values: object) -> InterpretationConfig:
    settings_type = cast(Callable[..., InterpretationConfig], InterpretationConfig)
    return settings_type(_env_file=None, **values)


class _CaptionTrack:
    def __init__(self, language_code: str, segments: Sequence[str]) -> None:
        self.language_code = language_code
        self.is_generated = False
        self._segments = segments
        self.fetch_calls = 0

    def fetch_segments(self) -> Sequence[str]:
        self.fetch_calls += 1
        return self._segments


class _CaptionProvider:
    def __init__(self, tracks: Sequence[_CaptionTrack]) -> None:
        self._tracks = tracks
        self.calls: list[str] = []

    def list_tracks(self, video_id: str) -> Sequence[_CaptionTrack]:
        self.calls.append(video_id)
        return self._tracks


class _AudioDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download(self, source_url: str, destination: Path) -> Path:
        self.calls.append((source_url, destination))
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        return audio_path


class _FailingWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> NoReturn:
        raise _WhisperProviderFailure("model failure")


class _FixedWhisperTranscriber:
    def __init__(self, result: WhisperResult) -> None:
        self.result = result
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.calls.append(audio_path)
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


def _enriched_book(book_mention: BookMention) -> EnrichedBookWork:
    return EnrichedBookWork(
        title=book_mention.title,
        authors=[
            EnrichedAuthorCredit(
                open_library_author_id="OL21594A",
                name=author_credit.name,
                open_library_url="https://openlibrary.org/authors/OL21594A",
            )
            for author_credit in book_mention.authors
        ],
        open_library_work_id="OL66554W",
        open_library_url="https://openlibrary.org/works/OL66554W",
        edition=BookEdition(
            title="Pride and Prejudice: A Collector's Edition",
            publication_year=1813,
            publishers=["T. Egerton", " T. Egerton "],
            isbn_10=["0141439513", "not-an-isbn"],
            isbn_13=["9780141439518", "9780141439518"],
            open_library_edition_id="OL12345M",
            open_library_url="https://openlibrary.org/books/OL12345M",
            cover_url="https://covers.openlibrary.org/b/id/12345-L.jpg",
        ),
        cover_url="https://covers.openlibrary.org/b/id/67890-L.jpg",
        cover_edition_id="OL67890M",
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
    music_release = EnrichedMusicRelease(
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
        music_release=music_release,
        cover_url=music_release.cover_url,
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
    transcript_service: TranscriptionService | CachedTranscriptionService,
    mentions: ScreenWorkMentions | None = None,
    results: ScreenWorkResults | None = None,
    music_mentions: MusicMentions | None = None,
    music_results: MusicResults | None = None,
    book_mentions: BookMentions | None = None,
    book_results: BookResults | None = None,
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
    interpreted_books = book_mentions if book_mentions is not None else BookMentions(books=[])
    resolved_books = (
        book_results
        if book_results is not None
        else BookResults(
            books=[
                BookResult(
                    status=ResultStatus.UNRESOLVED,
                    book_mention=book_mention,
                    book=None,
                )
                for book_mention in interpreted_books.books
            ]
        )
    )

    return ExtractionPipeline(
        metadata_service,
        transcript_service,
        _FakeInterpretationService(
            ExtractionMentions(
                screen_works=interpreted_screen_works,
                music=interpreted_music,
                books=interpreted_books,
            )
        ),
        _FakeResultAggregator(
            ExtractionResults(
                screen_works=screen_work_results,
                music=resolved_music,
                books=resolved_books,
            )
        ),
    )


def _install_pipeline(application: FastAPI, pipeline: ExtractionPipelineProtocol) -> None:
    application.dependency_overrides[get_pipeline] = lambda: pipeline


async def test_extract_transcript_interprets_normalized_text_without_source(
    client: AsyncClient,
) -> None:
    """Interpret submitted text, enrich its Mention, and omit a fabricated Source."""
    submitted_text = "Dune: Part One (2021) was excellent."
    source_metadata_service = _RecordingSourceMetadataService()
    transcription_service = _RecordingTranscriptionService()
    llm_provider = _RecordingLLMProvider(
        json.dumps(
            {
                "movies": [{"title": "Dune: Part One", "year": 2021}],
                "tv_series": [],
                "tracks": [],
                "music_releases": [],
                "books": [],
            }
        )
    )
    pipeline = ExtractionPipeline(
        source_metadata_service,
        transcription_service,
        MentionInterpretationService(
            llm_provider,
            _interpretation_settings(),
            DisabledCache(),
        ),
        _FakeResultAggregator(),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/internal/extractions",
        json={"transcript": f"  {submitted_text}\t", "market": "JP"},
    )

    assert response.status_code == 200
    raw_response = response.json()
    payload = TranscriptExtractResponse.model_validate(raw_response)
    assert raw_response["transcript"] == {
        "text": submitted_text,
        "language": "und",
        "method": "text_submission",
    }
    assert raw_response["market"] == "JP"
    assert "source" not in raw_response
    assert set(raw_response["results"]) == {
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
        "books",
    }
    assert payload.statistics.movies.n_mentions == 1
    assert payload.statistics.movies.n_resolved == 0
    assert payload.statistics.movies.n_unresolved == 1
    assert len(payload.results.movies) == 1
    movie_result = payload.results.movies[0]
    assert movie_result.status is ResultStatus.UNRESOLVED
    assert movie_result.movie_mention.title == "Dune: Part One"
    assert movie_result.movie_mention.year == 2021
    assert movie_result.movie is None
    assert payload.results.tv_series == []
    assert payload.results.tracks == []
    assert payload.results.music_releases == []
    assert payload.results.books == []
    assert source_metadata_service.calls == []
    assert transcription_service.calls == []
    assert len(llm_provider.calls) == 1
    assert json.loads(llm_provider.calls[0][1].content) == {
        "source_title": "",
        "source_description": "",
        "transcript_language": "und",
        "transcript": submitted_text,
    }


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
    resolved_book_mention = BookMention(
        title="Pride and Prejudice",
        authors=[AuthorCredit(name="Jane Austen")],
    )
    unresolved_book_mention = BookMention(
        title="Unknown Book",
        authors=[AuthorCredit(name="Unknown Author")],
    )
    interpreted_books = BookMentions(books=[resolved_book_mention, unresolved_book_mention])
    resolved_books = BookResults(
        books=[
            BookResult(
                status=ResultStatus.RESOLVED,
                book_mention=resolved_book_mention,
                book=_enriched_book(resolved_book_mention),
            ),
            BookResult(
                status=ResultStatus.UNRESOLVED,
                book_mention=unresolved_book_mention,
                book=None,
            ),
        ]
    )
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=metadata_extractor,
            settings=_settings(),
            cache=DisabledCache(),
        ),
        _transcription_service(
            _CaptionProvider([_CaptionTrack("en-GB", ["Router", "caption text."])])
        ),
        mentions,
        results,
        music_mentions=music_mentions,
        music_results=music_results,
        book_mentions=interpreted_books,
        book_results=resolved_books,
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extractions",
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
        "books": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    }
    raw_results = raw_response["results"]
    assert [item["status"] for item in raw_results["movies"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["tv_series"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["tracks"]] == ["resolved", "unresolved"]
    assert [item["status"] for item in raw_results["music_releases"]] == [
        "resolved",
        "unresolved",
    ]
    assert [item["status"] for item in raw_results["books"]] == ["resolved", "unresolved"]
    assert raw_results["movies"][1]["movie"] is None
    assert "movie" in raw_results["movies"][1]
    assert raw_results["tv_series"][1]["tv_series"] is None
    assert "tv_series" in raw_results["tv_series"][1]
    assert raw_results["tracks"][1]["track"] is None
    assert "track" in raw_results["tracks"][1]
    assert raw_results["music_releases"][1]["music_release"] is None
    assert "music_release" in raw_results["music_releases"][1]
    assert raw_results["books"][1]["book"] is None
    assert "book" in raw_results["books"][1]
    assert set(raw_results["tracks"][0]["track"]) == {
        "track_title",
        "artists",
        "spotify_track_id",
        "spotify_url",
        "music_release",
        "cover_url",
    }
    assert set(raw_results["tracks"][0]["track"]["music_release"]) == {
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
    assert set(raw_results["books"][0]["book"]) == {
        "title",
        "authors",
        "open_library_work_id",
        "open_library_url",
        "edition",
        "cover_url",
        "cover_edition_id",
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
    assert resolved_track.track.music_release.spotify_album_id == "track-album"
    assert resolved_track.track.music_release.cover_url == "https://i.scdn.co/image/track-cover"
    assert resolved_track.track.cover_url == resolved_track.track.music_release.cover_url
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
    resolved_book = payload.results.books[0]
    assert resolved_book.status is ResultStatus.RESOLVED
    assert resolved_book.book_mention.title == resolved_book_mention.title
    assert resolved_book.book_mention.authors == ["Jane Austen"]
    assert resolved_book.book is not None
    assert resolved_book.book.title == "Pride and Prejudice"
    assert resolved_book.book.authors[0].open_library_author_id == "OL21594A"
    assert resolved_book.book.authors[0].name == "Jane Austen"
    assert (
        resolved_book.book.authors[0].open_library_url == "https://openlibrary.org/authors/OL21594A"
    )
    assert resolved_book.book.open_library_work_id == "OL66554W"
    assert resolved_book.book.open_library_url == "https://openlibrary.org/works/OL66554W"
    assert resolved_book.book.cover_url == "https://covers.openlibrary.org/b/id/67890-L.jpg"
    assert resolved_book.book.cover_edition_id == "OL67890M"
    assert resolved_book.book.edition is not None
    assert resolved_book.book.edition.title == "Pride and Prejudice: A Collector's Edition"
    assert resolved_book.book.edition.publication_year == 1813
    assert resolved_book.book.edition.publishers == ["T. Egerton", " T. Egerton "]
    assert resolved_book.book.edition.isbn_10 == ["0141439513", "not-an-isbn"]
    assert resolved_book.book.edition.isbn_13 == ["9780141439518", "9780141439518"]
    assert resolved_book.book.edition.open_library_edition_id == "OL12345M"
    assert resolved_book.book.edition.open_library_url == "https://openlibrary.org/books/OL12345M"
    assert resolved_book.book.edition.cover_url == "https://covers.openlibrary.org/b/id/12345-L.jpg"
    assert payload.results.books[1].status is ResultStatus.UNRESOLVED
    assert payload.results.books[1].book_mention.title == unresolved_book_mention.title
    assert payload.results.books[1].book_mention.authors == ["Unknown Author"]
    assert payload.results.books[1].book is None


async def test_extract_maps_unavailable_captions_to_502(
    client: AsyncClient,
) -> None:
    """Map a valid Source with no usable captions to Transcript Unavailable."""
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
            cache=DisabledCache(),
        ),
        _transcription_service(_CaptionProvider([])),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extractions",
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
            cache=DisabledCache(),
        ),
        _transcription_service(_CaptionProvider([_CaptionTrack("en", ["Grouped", "results."])])),
        mentions,
    )
    _install_pipeline(app, pipeline)

    response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})

    assert response.status_code == 200
    results = response.json()["results"]
    assert set(results) == {"movies", "tv_series", "tracks", "music_releases", "books"}
    assert [item["movie_mention"]["title"] for item in results["movies"]] == expected_movies
    assert [
        item["tv_series_mention"]["title"] for item in results["tv_series"]
    ] == expected_tv_series
    assert results["tracks"] == []
    assert results["music_releases"] == []
    assert results["books"] == []
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
            cache=DisabledCache(),
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
        "/api/extractions",
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
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=DisabledCache(),
        ),
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

    response = await client.post("/api/extractions", json={"url": submitted_url})

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
            cache=DisabledCache(),
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

    first = asyncio.create_task(client.post("/api/extractions", json={"url": _CANONICAL_URL}))
    assert await asyncio.to_thread(transcriber.started.wait, 5)
    second = asyncio.create_task(client.post("/api/extractions", json={"url": _CANONICAL_URL}))
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
            MentionInterpretationError("mention interpretation failed"),
            502,
            "mention_interpretation_failed",
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
        "/api/extractions",
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
    response = await client.post("/api/extractions", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "payload",
    [{}, {"transcript": 123}, {"transcript": ""}, {"transcript": " \t\n"}],
)
async def test_malformed_transcript_requests_skip_the_pipeline(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    """Reject invalid submitted Transcript payloads before pipeline invocation."""
    pipeline = _MarketPipeline()
    _install_pipeline(app, pipeline)

    response = await client.post("/api/internal/extractions", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()
    assert pipeline.transcript_calls == []


async def test_oversized_submitted_transcript_returns_existing_413_contract(
    client: AsyncClient,
) -> None:
    """Keep Interpretation Material size enforcement in the interpretation service."""
    source_metadata_service = _RecordingSourceMetadataService()
    transcription_service = _RecordingTranscriptionService()
    llm_provider = _RecordingLLMProvider(
        json.dumps(
            {
                "movies": [],
                "tv_series": [],
                "tracks": [],
                "music_releases": [],
                "books": [],
            }
        )
    )
    pipeline = ExtractionPipeline(
        source_metadata_service,
        transcription_service,
        MentionInterpretationService(
            llm_provider,
            _interpretation_settings(max_transcript_chars=5),
            DisabledCache(),
        ),
        _FakeResultAggregator(),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/internal/extractions",
        json={"transcript": "123456"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "interpretation_input_too_large",
            "message": "Interpretation Material exceeds the configured limit.",
        }
    }
    assert source_metadata_service.calls == []
    assert transcription_service.calls == []
    assert llm_provider.calls == []


async def test_unhandled_failures_do_not_leak_internals() -> None:
    """Return a generic 500 response when the pipeline raises unexpectedly."""
    _install_pipeline(
        app,
        _RaisingPipeline(RuntimeError("sensitive database path /var/reelio/secret.db")),
    )
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extractions",
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
    """Document result statistics, grouped Music and Book resolution, and failures."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/extractions"]["post"]
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
    assert "identity verification" in description
    assert "null edition, cover_url, and cover_edition_id" in description
    assert "Result has book set to null" in description
    assert "fuzzy" not in description.lower()
    assert "90 percent" not in description.lower()
    assert "every ordered Artist Credit" not in description
    assert "release year must match" not in description.lower()
    assert {"200", "400", "404", "413", "500", "502", "504", "422"} <= set(responses)
    for status_code in ("400", "404", "413", "500", "502", "504"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}
    assert "catalog_provider_failed" in responses["502"]["description"]
    assert "Open Library catalog request failed." in responses["502"]["description"]
    assert "pipeline_timeout" in responses["504"]["description"]
    assert "Open Library catalog request timed out." in responses["504"]["description"]

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
        "books": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    }
    assert set(example["results"]) == {
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
        "books",
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
        "music_release",
        "cover_url",
    }
    assert set(resolved_track_example["track"]["music_release"]) == {
        "release_title",
        "artists",
        "release_date",
        "album_type",
        "spotify_album_id",
        "spotify_url",
        "cover_url",
    }
    assert resolved_track_example["track"]["music_release"]["release_title"] == (
        "Discovery (Deluxe Edition)"
    )
    assert (
        resolved_track_example["track"]["music_release"]["release_title"]
        != resolved_track_example["track_mention"]["release_title"]
    )
    assert (
        resolved_track_example["track"]["cover_url"]
        == resolved_track_example["track"]["music_release"]["cover_url"]
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
    resolved_book_example = example["results"]["books"][0]
    assert resolved_book_example["status"] == "resolved"
    assert resolved_book_example["book_mention"] == {
        "title": "Pride and Prejudice",
        "authors": ["Jane Austen"],
    }
    assert resolved_book_example["book"] == {
        "title": "Pride and Prejudice",
        "authors": [
            {
                "open_library_author_id": "OL21594A",
                "name": "Jane Austen",
                "open_library_url": "https://openlibrary.org/authors/OL21594A",
            }
        ],
        "open_library_work_id": "OL66554W",
        "open_library_url": "https://openlibrary.org/works/OL66554W",
        "edition": {
            "title": "Pride and Prejudice: A Collector's Edition",
            "publication_year": 1813,
            "publishers": ["T. Egerton"],
            "isbn_10": ["0141439513"],
            "isbn_13": ["9780141439518"],
            "open_library_edition_id": "OL12345M",
            "open_library_url": "https://openlibrary.org/books/OL12345M",
            "cover_url": "https://covers.openlibrary.org/b/id/12345-L.jpg",
        },
        "cover_url": "https://covers.openlibrary.org/b/id/12345-L.jpg",
        "cover_edition_id": "OL12345M",
    }
    unresolved_book_example = example["results"]["books"][1]
    assert unresolved_book_example["status"] == "unresolved"
    assert unresolved_book_example["book_mention"] == {
        "title": "Unknown Book",
        "authors": [],
    }
    assert unresolved_book_example.get("book") is None

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
        "books",
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
    assert extraction_results["properties"]["books"] == {
        "items": {"$ref": "#/components/schemas/BookResultModel"},
        "type": "array",
        "title": "Books",
    }
    extraction_statistics = schemas["ExtractionStatisticsModel"]
    assert extraction_statistics["required"] == [
        "movies",
        "tv_series",
        "tracks",
        "music_releases",
        "books",
    ]
    assert extraction_statistics["properties"] == {
        "movies": {"$ref": "#/components/schemas/ResultCountsModel"},
        "tv_series": {"$ref": "#/components/schemas/ResultCountsModel"},
        "tracks": {"$ref": "#/components/schemas/ResultCountsModel"},
        "music_releases": {"$ref": "#/components/schemas/ResultCountsModel"},
        "books": {"$ref": "#/components/schemas/ResultCountsModel"},
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
    book_result_schema = schemas["BookResultModel"]
    assert book_result_schema["required"] == ["status", "book_mention", "book"]
    assert book_result_schema["properties"]["book_mention"] == {
        "$ref": "#/components/schemas/BookMentionModel"
    }
    assert book_result_schema["properties"]["book"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/BookModel"},
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
        "music_release",
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
    music_release = track_schema["properties"]["music_release"]
    assert music_release.get(
        "$ref"
    ) == "#/components/schemas/MusicReleaseModel" or music_release.get("allOf") == [
        {"$ref": "#/components/schemas/MusicReleaseModel"}
    ]
    artist_credit_schema = schemas["ArtistCreditModel"]
    assert artist_credit_schema["required"] == ["spotify_artist_id", "name"]
    book_mention_schema = schemas["BookMentionModel"]
    assert book_mention_schema["required"] == ["title", "authors"]
    assert book_mention_schema["properties"]["authors"] == {
        "items": {"type": "string"},
        "type": "array",
        "title": "Authors",
    }
    book_schema = schemas["BookModel"]
    assert book_schema["required"] == [
        "title",
        "authors",
        "open_library_work_id",
        "open_library_url",
        "edition",
        "cover_url",
        "cover_edition_id",
    ]
    assert book_schema["properties"]["authors"] == {
        "items": {"$ref": "#/components/schemas/EnrichedAuthorCreditModel"},
        "type": "array",
        "title": "Authors",
    }
    assert book_schema["properties"]["edition"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/BookEditionModel"},
            {"type": "null"},
        ]
    }
    assert book_schema["properties"]["cover_url"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert book_schema["properties"]["cover_edition_id"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    book_edition_schema = schemas["BookEditionModel"]
    assert book_edition_schema["required"] == [
        "title",
        "publication_year",
        "publishers",
        "isbn_10",
        "isbn_13",
        "open_library_edition_id",
        "open_library_url",
        "cover_url",
    ]
    assert book_edition_schema["properties"]["publishers"] == {
        "items": {"type": "string"},
        "type": "array",
        "title": "Publishers",
    }
    assert book_edition_schema["properties"]["isbn_10"] == {
        "items": {"type": "string"},
        "type": "array",
        "title": "Isbn 10",
    }
    assert book_edition_schema["properties"]["isbn_13"] == {
        "items": {"type": "string"},
        "type": "array",
        "title": "Isbn 13",
    }
    assert book_edition_schema["properties"]["title"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert book_edition_schema["properties"]["publication_year"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert book_edition_schema["properties"]["cover_url"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    enriched_author_credit_schema = schemas["EnrichedAuthorCreditModel"]
    assert enriched_author_credit_schema["required"] == [
        "open_library_author_id",
        "name",
        "open_library_url",
    ]
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
    assert "music_release is the Spotify Album attached" in operation["description"]
    assert "first provider-ordered Spotify-hosted Album image URL" in operation["description"]
    assert "artwork link-back" in operation["description"]
    assert "worldwide-edition" in operation["description"]
    assert "Music Releases" in operation["summary"]
    assert "Book Works" in operation["summary"]
    assert "Book Work Results retain their interpreted Book Mention" in operation["description"]
    assert "Open Library Work title" in operation["description"]
    assert (
        "Book Work cover_url first uses the selected Edition's own cover"
        in operation["description"]
    )
    assert "the fallback never populates edition.cover_url" in operation["description"]
    assert "Book Work cover fields without another request" in operation["description"]
    assert "English-first Open Library relevance" in operation["description"]
    assert (
        "unrestricted fallback for missing or audiobook English results" in operation["description"]
    )
    assert "independent of Effective Market and Source Edition signals" in operation["description"]
    assert (
        "Any TMDB, Spotify, or Open Library provider failure fails the complete request."
        in operation["description"]
    )
    assert (
        "Any TMDB, Spotify, or Open Library provider failure fails the complete request."
        in responses["502"]["description"]
    )
    internal_operation = document["paths"]["/api/internal/extractions"]["post"]
    internal_responses = internal_operation["responses"]
    assert set(internal_responses) == {"200", "413", "422", "500", "502", "504"}
    for status_code in ("413", "500", "502", "504"):
        schema = internal_responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}
    internal_request_schema = internal_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert internal_request_schema == {"$ref": "#/components/schemas/TranscriptExtractRequest"}
    transcript_extract_request = schemas["TranscriptExtractRequest"]
    assert transcript_extract_request["required"] == ["transcript"]
    transcript_market = transcript_extract_request["properties"]["market"]
    assert transcript_market["examples"] == ["US", "JP"]
    assert transcript_market["anyOf"][0]["pattern"] == "^[A-Z]{2}$"
    internal_response_schema = internal_responses["200"]["content"]["application/json"]["schema"]
    assert internal_response_schema == {"$ref": "#/components/schemas/TranscriptExtractResponse"}
    transcript_extract_response = schemas["TranscriptExtractResponse"]
    assert transcript_extract_response["required"] == [
        "market",
        "transcript",
        "statistics",
        "results",
    ]
    assert "source" not in transcript_extract_response["properties"]
    assert "source" not in internal_responses["200"]["content"]["application/json"]["example"]


class _MarketPipeline:
    """Return a deterministic Pipeline Result while recording effective-market input."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, SpotifyMarket | None]] = []
        self.transcript_calls: list[tuple[str, SpotifyMarket | None]] = []

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
                books=BookResults(books=[]),
            ),
            market=market or _DEFAULT_MARKET,
        )

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        """Record submitted transcript input and return an empty extraction result."""
        self.transcript_calls.append((transcript_text, market))
        return TranscriptPipelineResult(
            transcript=Transcript(
                text=transcript_text,
                language="und",
                method=TranscriptMethod.TEXT_SUBMISSION,
            ),
            results=ExtractionResults(
                screen_works=ScreenWorkResults(movies=[], tv_series=[]),
                music=MusicResults(tracks=[], music_releases=[]),
                books=BookResults(books=[]),
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
        "/api/extractions",
        json={"url": _CANONICAL_URL, "market": "JP"},
    )
    default_response = await client.post(
        "/api/extractions",
        json={"url": _CANONICAL_URL},
    )
    invalid_response = await client.post(
        "/api/extractions",
        json={"url": _CANONICAL_URL, "market": "jp"},
    )

    assert explicit_response.status_code == 200
    assert explicit_response.json()["market"] == "JP"
    assert explicit_response.json()["statistics"] == {
        "movies": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "tv_series": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "tracks": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "music_releases": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        "books": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
    }
    assert default_response.status_code == 200
    assert default_response.json()["market"] == "US"
    assert invalid_response.status_code == 422
    assert pipeline.calls == [(_CANONICAL_URL, "JP"), (_CANONICAL_URL, None)]


def _youtube_metadata() -> dict[str, object]:
    return {
        "id": _VIDEO_ID,
        "title": "Router test video",
        "description": "A complete router test description.",
        "channel": "Router test channel",
        "duration": 42.2,
    }


async def test_reuses_source_metadata_across_repeated_http_submissions(
    client: AsyncClient,
) -> None:
    """Return two successful Source responses after only one provider inspection."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        _transcription_service(
            _CaptionProvider([_CaptionTrack("en", ["Cached source transcript."])])
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        first_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        second_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.json()["source"] == second_response.json()["source"]
        assert extractor.calls == [_CANONICAL_URL]
    finally:
        await cache.aclose()


async def test_converges_submitted_aliases_through_provider_identity(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Serve accepted Instagram aliases from one provider-authoritative Source."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    canonical_url = "https://www.instagram.com/reel/ABC_123"
    extractor = _SocialMetadataExtractor(
        {
            "id": "ABC_123",
            "extractor_key": "Instagram",
            "webpage_url": canonical_url,
            "title": "Instagram cache video",
            "description": "Instagram cache description",
            "channel": "Instagram cache channel",
            "duration": 42.2,
            "formats": [{"vcodec": "avc1"}],
        }
    )
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        _social_transcription_service(tmp_path),
    )
    _install_pipeline(app, pipeline)

    try:
        submitted_response = await client.post(
            "/api/extractions",
            json={"url": submitted_url},
        )
        canonical_response = await client.post(
            "/api/extractions",
            json={"url": canonical_url},
        )

        assert submitted_response.status_code == 200
        assert canonical_response.status_code == 200
        assert submitted_response.json()["source"] == canonical_response.json()["source"]
        assert len(extractor.calls) == 1
    finally:
        await cache.aclose()


async def test_source_metadata_reuse_is_market_independent(
    client: AsyncClient,
) -> None:
    """Keep Source reuse independent from each requested effective market."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        _transcription_service(
            _CaptionProvider([_CaptionTrack("en", ["Market independent source."])])
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        us_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "US"},
        )
        jp_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "JP"},
        )

        assert us_response.status_code == 200
        assert jp_response.status_code == 200
        assert us_response.json()["market"] == "US"
        assert jp_response.json()["market"] == "JP"
        assert us_response.json()["source"] == jp_response.json()["source"]
        assert extractor.calls == [_CANONICAL_URL]
    finally:
        await cache.aclose()


@pytest.mark.parametrize(
    ("first_error", "expected_status", "expected_code", "expected_message"),
    [
        (
            MetadataProviderError("provider failure"),
            502,
            "metadata_provider_failed",
            "Unable to retrieve source metadata.",
        ),
        (
            YoutubeDLError("Video unavailable"),
            404,
            "source_unavailable",
            "Source is unavailable.",
        ),
    ],
    ids=["provider", "unavailable"],
)
async def test_source_inspection_errors_are_not_cached(
    client: AsyncClient,
    first_error: Exception,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """Expose the first typed Source error and succeed on a later HTTP retry."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([first_error, _youtube_metadata()])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        _transcription_service(_CaptionProvider([_CaptionTrack("en", ["Retry succeeds."])])),
    )
    _install_pipeline(app, pipeline)

    try:
        failed_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        successful_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL},
        )

        assert failed_response.status_code == expected_status
        assert failed_response.json() == {
            "error": {"code": expected_code, "message": expected_message}
        }
        assert successful_response.status_code == 200
        assert extractor.calls == [_CANONICAL_URL, _CANONICAL_URL]
    finally:
        await cache.aclose()


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_code", "expected_message"),
    [
        (
            MetadataProviderError("provider failure"),
            502,
            "metadata_provider_failed",
            "Unable to retrieve source metadata.",
        ),
        (
            YoutubeDLError("Video unavailable"),
            404,
            "source_unavailable",
            "Source is unavailable.",
        ),
    ],
    ids=["provider", "unavailable"],
)
async def test_cache_read_failure_preserves_typed_source_errors(
    client: AsyncClient,
    provider_error: Exception,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """Never serve stale Source metadata or cache-specific errors during read outages."""
    clock = ManualClock()
    cache, redis = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata(), provider_error])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        _transcription_service(_CaptionProvider([_CaptionTrack("en", ["Warm source cache."])])),
    )
    _install_pipeline(app, pipeline)

    try:
        warm_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        redis.fail_operations.add("read")
        failed_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})

        assert warm_response.status_code == 200
        assert failed_response.status_code == expected_status
        assert failed_response.json() == {
            "error": {"code": expected_code, "message": expected_message}
        }
        assert extractor.calls == [_CANONICAL_URL, _CANONICAL_URL]
    finally:
        await cache.aclose()


async def test_endpoint_reuses_caption_transcript_without_repeating_acquisition(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Reuse one Caption Transcript through the public Source extraction endpoint."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    track = _CaptionTrack("en", ["Cached Caption Transcript."])
    provider = _CaptionProvider([track])
    downloader = _AudioDownloader()
    transcriber = _FixedWhisperTranscriber(
        WhisperResult(text="unused Whisper", language="en", segment_count=1)
    )
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=provider,
                audio_downloader=downloader,
                transcriber=transcriber,
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        first_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        second_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})

        expected_transcript = {
            "text": "Cached Caption Transcript.",
            "language": "en",
            "method": "youtube_captions",
        }
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.json()["transcript"] == expected_transcript
        assert second_response.json()["transcript"] == expected_transcript
        assert extractor.calls == [_CANONICAL_URL]
        assert provider.calls == [_VIDEO_ID]
        assert track.fetch_calls == 1
        assert downloader.calls == []
        assert transcriber.calls == []
    finally:
        await cache.aclose()


async def test_endpoint_reuses_whisper_transcript_without_repeating_acquisition(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Reuse one social-video Whisper Transcript through the public endpoint."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    submitted_url = "https://www.instagram.com/reel/ABC123"
    extractor = _SocialMetadataExtractor(
        {
            "id": "ABC123",
            "extractor_key": "Instagram",
            "webpage_url": submitted_url,
            "title": "Whisper cache router video",
            "description": "Whisper cache router description",
            "channel": "Whisper cache router channel",
            "duration": 42.2,
            "formats": [{"vcodec": "avc1"}],
        }
    )
    provider = _CaptionProvider([_CaptionTrack("en", ["unused Caption"])])
    downloader = _AudioDownloader()
    transcriber = _FixedWhisperTranscriber(
        WhisperResult(
            text="  Cached Whisper Transcript.  ",
            language="fr",
            segment_count=2,
        )
    )
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=provider,
                audio_downloader=downloader,
                transcriber=transcriber,
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        first_response = await client.post("/api/extractions", json={"url": submitted_url})
        second_response = await client.post("/api/extractions", json={"url": submitted_url})

        expected_transcript = {
            "text": "Cached Whisper Transcript.",
            "language": "fr",
            "method": "whisper",
        }
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.json()["transcript"] == expected_transcript
        assert second_response.json()["transcript"] == expected_transcript
        assert extractor.calls == [submitted_url]
        assert provider.calls == []
        assert len(downloader.calls) == 1
        assert downloader.calls[0][0] == submitted_url
        assert len(transcriber.calls) == 1
    finally:
        await cache.aclose()


async def test_endpoint_reuses_transcript_across_markets_while_enriching_each_market(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Keep Transcript reuse independent from market-sensitive downstream work."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    track = _CaptionTrack("en", ["Market-independent Transcript."])
    provider = _CaptionProvider([track])
    interpretation_service = _FakeInterpretationService()
    result_aggregator = _FakeResultAggregator()
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=provider,
                audio_downloader=_AudioDownloader(),
                transcriber=_FixedWhisperTranscriber(
                    WhisperResult(text="unused Whisper", language="en", segment_count=1)
                ),
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
        interpretation_service,
        result_aggregator,
    )
    _install_pipeline(app, pipeline)

    try:
        us_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "US"},
        )
        jp_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "JP"},
        )

        assert us_response.status_code == 200
        assert jp_response.status_code == 200
        assert us_response.json()["transcript"] == jp_response.json()["transcript"]
        assert extractor.calls == [_CANONICAL_URL]
        assert provider.calls == [_VIDEO_ID]
        assert track.fetch_calls == 1
        assert len(interpretation_service.calls) == 2
        assert len(result_aggregator.calls) == 2
        assert result_aggregator.markets == [SpotifyMarket("US"), SpotifyMarket("JP")]
    finally:
        await cache.aclose()


async def test_endpoint_reuses_mention_interpretation_across_effective_markets(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Reuse Mention interpretation while aggregating each effective market."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    caption_track = _CaptionTrack("en", ["Market-independent mention content."])
    caption_provider = _CaptionProvider([caption_track])
    llm_provider = _RecordingLLMProvider(
        json.dumps(
            {
                "movies": [
                    {"title": "Dune: Part One", "year": 2021},
                    {"title": "Arrival", "year": 2016},
                ],
                "tv_series": [{"title": "Fargo", "year": 2014}],
                "tracks": [
                    {
                        "track_title": "Track One",
                        "artists": ["Artist One"],
                        "release_title": "Release One",
                        "release_year": 2020,
                    }
                ],
                "music_releases": [
                    {
                        "release_title": "Release One",
                        "artists": ["Artist One"],
                        "release_year": 2020,
                    }
                ],
                "books": [{"title": "Book One", "authors": ["Author One"]}],
            }
        )
    )
    interpretation_service = MentionInterpretationService(
        llm_provider,
        _interpretation_settings(),
        cache,
    )
    result_aggregator = _FakeResultAggregator()
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=caption_provider,
                audio_downloader=_AudioDownloader(),
                transcriber=_FixedWhisperTranscriber(
                    WhisperResult(text="unused Whisper", language="en", segment_count=1)
                ),
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
        interpretation_service,
        result_aggregator,
        SpotifyMarket("US"),
    )
    _install_pipeline(app, pipeline)

    try:
        us_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "US"},
        )
        jp_response = await client.post(
            "/api/extractions",
            json={"url": _CANONICAL_URL, "market": "JP"},
        )

        assert us_response.status_code == 200
        assert jp_response.status_code == 200
        assert us_response.json()["results"] == jp_response.json()["results"]
        assert len(llm_provider.calls) == 1
        assert caption_provider.calls == [_VIDEO_ID]
        assert caption_track.fetch_calls == 1
        assert [movie.title for movie in result_aggregator.calls[1].screen_works.movies] == [
            "Dune: Part One",
            "Arrival",
        ]
        assert [
            tv_series.title for tv_series in result_aggregator.calls[1].screen_works.tv_series
        ] == ["Fargo"]
        assert [track.track_title for track in result_aggregator.calls[1].music.tracks] == [
            "Track One"
        ]
        assert [
            music_release.release_title
            for music_release in result_aggregator.calls[1].music.music_releases
        ] == ["Release One"]
        assert [book.title for book in result_aggregator.calls[1].books.books] == ["Book One"]
        assert result_aggregator.calls[1] == result_aggregator.calls[0]
        assert result_aggregator.calls[1] is not result_aggregator.calls[0]
        assert result_aggregator.markets == [SpotifyMarket("US"), SpotifyMarket("JP")]
    finally:
        await cache.aclose()


async def test_direct_transcript_submissions_do_not_use_video_transcript_cache(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Keep direct text submissions isolated from cached video-derived Transcripts."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    track = _CaptionTrack("en", ["Shared Transcript Text."])
    provider = _CaptionProvider([track])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=provider,
                audio_downloader=_AudioDownloader(),
                transcriber=_FixedWhisperTranscriber(
                    WhisperResult(text="unused Whisper", language="en", segment_count=1)
                ),
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        before_warm = await client.post(
            "/api/internal/extractions",
            json={"transcript": "Shared Transcript Text."},
        )
        video_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        after_warm = await client.post(
            "/api/internal/extractions",
            json={"transcript": "Shared Transcript Text."},
        )

        expected_submission = {
            "text": "Shared Transcript Text.",
            "language": "und",
            "method": "text_submission",
        }
        assert before_warm.status_code == 200
        assert before_warm.json()["transcript"] == expected_submission
        assert video_response.status_code == 200
        assert video_response.json()["transcript"]["method"] == "youtube_captions"
        assert after_warm.status_code == 200
        assert after_warm.json()["transcript"] == expected_submission
        assert extractor.calls == [_CANONICAL_URL]
        assert provider.calls == [_VIDEO_ID]
        assert track.fetch_calls == 1
    finally:
        await cache.aclose()


async def test_endpoint_transcript_contract_bump_preserves_source_metadata_reuse(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reacquire only the Transcript after its acquisition-contract version changes."""
    clock = ManualClock()
    cache, _ = _router_cache(clock)
    extractor = _OutcomeMetadataExtractor([_youtube_metadata()])
    track = _CaptionTrack("en", ["Transcript contract v1."])
    provider = _CaptionProvider([track])
    pipeline = _pipeline(
        SourceMetadataService(
            extractor=extractor,
            settings=_settings(),
            cache=cache,
        ),
        CachedTranscriptionService(
            TranscriptionService(
                provider=provider,
                audio_downloader=_AudioDownloader(),
                transcriber=_FixedWhisperTranscriber(
                    WhisperResult(text="unused Whisper", language="en", segment_count=1)
                ),
                temp_media_dir=tmp_path,
                semaphore=asyncio.Semaphore(1),
            ),
            cache,
        ),
    )
    _install_pipeline(app, pipeline)

    try:
        first_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})
        track._segments = ["Transcript contract v2."]
        monkeypatch.setattr(
            transcription_service_module,
            "_TRANSCRIPT_ACQUISITION_CONTRACT_VERSION",
            "transcript-acquisition-v2",
        )
        second_response = await client.post("/api/extractions", json={"url": _CANONICAL_URL})

        assert first_response.status_code == 200
        assert first_response.json()["transcript"]["text"] == "Transcript contract v1."
        assert second_response.status_code == 200
        assert second_response.json()["transcript"]["text"] == "Transcript contract v2."
        assert extractor.calls == [_CANONICAL_URL]
        assert provider.calls == [_VIDEO_ID, _VIDEO_ID]
        assert track.fetch_calls == 2
    finally:
        await cache.aclose()
