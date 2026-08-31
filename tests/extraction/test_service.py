"""Pipeline orchestration contract tests."""

import logging
import shutil
from pathlib import Path

import pytest

import reelio.extraction.services.transcription.inspection as transcription_inspection
from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    EnrichmentError,
    InvalidSourceError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
)
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    MovieMention,
    MovieResult,
    Platform,
    ResultStatus,
    ScreenWorkMentions,
    Source,
    Transcript,
    TranscriptMethod,
    TVSeriesMention,
)
from tests.extraction.fakes import (
    FakeInterpretationService as _FakeInterpretationService,
)
from tests.extraction.fakes import (
    FakeMovieResolver as _FakeMovieResolver,
)

_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _FakeSourceMetadataService:
    def __init__(
        self,
        source: Source | None = None,
        error: Exception | None = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> None:
        self.source = source
        self.prepared_audio = prepared_audio
        self.error = error
        self.calls: list[str] = []

    async def inspect(self, submitted_url: str) -> InspectedSource:
        self.calls.append(submitted_url)
        if self.error is not None:
            raise self.error
        assert self.source is not None
        return InspectedSource(self.source, self.prepared_audio)


class _FakeTranscriptionService:
    def __init__(self, transcript: Transcript, error: Exception | None = None) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[Source, str]] = []

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: object | None = None,
    ) -> Transcript:
        self.calls.append((source, submitted_url))
        if self.error is not None:
            raise self.error
        return self.transcript


def _pipeline(
    metadata_service: _FakeSourceMetadataService,
    transcription_service: _FakeTranscriptionService,
    interpretation_service: _FakeInterpretationService | None = None,
    movie_resolver: _FakeMovieResolver | None = None,
) -> ExtractionPipeline:
    return ExtractionPipeline(
        metadata_service,
        transcription_service,
        interpretation_service or _FakeInterpretationService(),
        movie_resolver or _FakeMovieResolver(),
    )


def _source() -> Source:
    return Source(
        platform=Platform.YOUTUBE,
        video_id=_VIDEO_ID,
        url=_CANONICAL_URL,
        title="Pipeline test video",
        description="Pipeline test description.",
        channel="Pipeline test channel",
        duration_seconds=42,
    )


def _transcript() -> Transcript:
    return Transcript(
        text="A real pipeline transcript.",
        language="en-GB",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    )


async def test_pipeline_returns_empty_grouped_results_and_calls_movie_resolver_once() -> None:
    """Resolve an empty Movie sequence and return both required result lists."""
    source = _source()
    transcript = _transcript()
    metadata_service = _FakeSourceMetadataService(source)
    transcription_service = _FakeTranscriptionService(transcript)
    interpretation_service = _FakeInterpretationService(ScreenWorkMentions(movies=[], tv_series=[]))
    movie_resolver = _FakeMovieResolver()
    pipeline = _pipeline(
        metadata_service,
        transcription_service,
        interpretation_service,
        movie_resolver,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result.source is source
    assert result.transcript is transcript
    assert metadata_service.calls == [_CANONICAL_URL]
    assert transcription_service.calls == [(source, _CANONICAL_URL)]
    assert interpretation_service.calls == [(source, transcript)]
    assert movie_resolver.calls == [()]
    assert result.results.movies == []
    assert result.results.tv_series == []


async def test_pipeline_returns_ordered_movie_results_unchanged() -> None:
    """Pass only Movie Mentions to the resolver and preserve its result objects."""
    movie_mentions = [
        MovieMention(title="Dune: Part One", year=2021),
        MovieMention(title="Che: Part One", year=2008),
    ]
    movie_results = [
        MovieResult(ResultStatus.UNRESOLVED, movie_mentions[0], None),
        MovieResult(ResultStatus.UNRESOLVED, movie_mentions[1], None),
    ]
    interpretation_service = _FakeInterpretationService(
        ScreenWorkMentions(movies=movie_mentions, tv_series=[])
    )
    movie_resolver = _FakeMovieResolver(results=movie_results)
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
        interpretation_service,
        movie_resolver,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert movie_resolver.calls == [tuple(movie_mentions)]
    assert result.results.movies == movie_results
    assert result.results.movies[0] is movie_results[0]
    assert result.results.movies[1] is movie_results[1]
    assert result.results.tv_series == []


async def test_pipeline_projects_tv_only_mentions_without_movie_resolution_input() -> None:
    """Return ordered unresolved TV Series Results while resolving an empty Movie list."""
    tv_series_mentions = [
        TVSeriesMention(title="The Last of Us", year=2023),
        TVSeriesMention(title="Arcane", year=2021),
    ]
    interpretation_service = _FakeInterpretationService(
        ScreenWorkMentions(movies=[], tv_series=tv_series_mentions)
    )
    movie_resolver = _FakeMovieResolver()
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
        interpretation_service,
        movie_resolver,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert movie_resolver.calls == [()]
    assert result.results.movies == []
    assert [item.status for item in result.results.tv_series] == [
        ResultStatus.UNRESOLVED,
        ResultStatus.UNRESOLVED,
    ]
    assert [item.tv_series_mention for item in result.results.tv_series] == tv_series_mentions
    assert result.results.tv_series[0].tv_series_mention is tv_series_mentions[0]
    assert result.results.tv_series[1].tv_series_mention is tv_series_mentions[1]
    assert all(item.tv_series is None for item in result.results.tv_series)


async def test_pipeline_groups_mixed_interpretation_results() -> None:
    """Keep Movie and TV Series ordering independent in one pipeline result."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    movie_result = MovieResult(ResultStatus.UNRESOLVED, movie_mention, None)
    interpretation_service = _FakeInterpretationService(
        ScreenWorkMentions(
            movies=[movie_mention],
            tv_series=[tv_series_mention],
        )
    )
    movie_resolver = _FakeMovieResolver(results=[movie_result])
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
        interpretation_service,
        movie_resolver,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert movie_resolver.calls == [(movie_mention,)]
    assert result.results.movies == [movie_result]
    assert result.results.tv_series[0].status is ResultStatus.UNRESOLVED
    assert result.results.tv_series[0].tv_series_mention is tv_series_mention
    assert result.results.tv_series[0].tv_series is None


async def test_pipeline_preserves_whisper_transcript_method() -> None:
    """Pass a Whisper Transcript through orchestration unchanged."""
    transcript = Transcript(
        text="Spoken audio transcript.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(transcript),
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result.transcript is transcript
    assert result.transcript.method is TranscriptMethod.WHISPER


async def test_pipeline_is_platform_agnostic_for_social_whisper_sources() -> None:
    """Pass a social Source and Whisper Transcript through unchanged."""
    source = Source(
        platform=Platform.INSTAGRAM,
        video_id="ABC123",
        url="https://www.instagram.com/reel/ABC123",
        title="Social pipeline video",
        description="Social pipeline description.",
        channel="Social pipeline channel",
        duration_seconds=42,
    )
    transcript = Transcript(
        text="Social pipeline speech.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    metadata_service = _FakeSourceMetadataService(source)
    transcription_service = _FakeTranscriptionService(transcript)
    pipeline = _pipeline(metadata_service, transcription_service)

    result = await pipeline.run(source.url)

    assert result.source is source
    assert result.transcript is transcript
    assert metadata_service.calls == [source.url]
    assert transcription_service.calls == [(source, source.url)]


@pytest.mark.parametrize(
    "source_error",
    [
        InvalidSourceError("invalid source"),
        SourceUnavailableError("source unavailable"),
        DurationLimitExceededError("duration limit exceeded"),
    ],
)
async def test_pipeline_does_not_acquire_transcript_after_source_rejection(
    source_error: Exception,
) -> None:
    """Stop before transcription when Source inspection rejects the input."""
    metadata_service = _FakeSourceMetadataService(error=source_error)
    transcription_service = _FakeTranscriptionService(_transcript())
    pipeline = _pipeline(metadata_service, transcription_service)

    with pytest.raises(type(source_error)) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is source_error
    assert transcription_service.calls == []


@pytest.mark.parametrize(
    "transcription_error",
    [
        TranscriptionError("Transcript is unavailable for this video."),
        PipelineTimeoutError("Transcript acquisition timed out."),
    ],
)
async def test_pipeline_propagates_transcription_errors_unchanged(
    transcription_error: Exception,
) -> None:
    """Propagate transcription domain errors without HTTP-layer translation."""
    metadata_service = _FakeSourceMetadataService(_source())
    transcription_service = _FakeTranscriptionService(_transcript(), error=transcription_error)
    pipeline = _pipeline(metadata_service, transcription_service)

    with pytest.raises(type(transcription_error)) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is transcription_error
    assert transcription_service.calls == [(_source(), _CANONICAL_URL)]


async def test_pipeline_propagates_resolution_errors_unchanged() -> None:
    """Propagate TMDB resolution failures without changing their domain policy."""
    resolution_error = EnrichmentError("TMDB candidate resolution failed.")
    movie_resolver = _FakeMovieResolver(error=resolution_error)
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
        _FakeInterpretationService(
            ScreenWorkMentions(
                movies=[MovieMention(title="Dune: Part One", year=2021)],
                tv_series=[],
            )
        ),
        movie_resolver,
    )

    with pytest.raises(EnrichmentError) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is resolution_error
    assert movie_resolver.calls == [
        (MovieMention(title="Dune: Part One", year=2021),),
    ]


async def test_pipeline_closes_interpretation_and_resolution_modules() -> None:
    """Delegate lifespan cleanup to interpretation and resolution modules."""
    interpretation_service = _FakeInterpretationService()
    movie_resolver = _FakeMovieResolver()
    pipeline = _pipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
        interpretation_service,
        movie_resolver,
    )

    await pipeline.aclose()

    assert interpretation_service.closed is True
    assert movie_resolver.closed is True


@pytest.mark.parametrize(
    "transcription_error",
    [None, TranscriptionError("Transcript is unavailable for this video.")],
)
async def test_pipeline_cleans_inspection_audio_after_transcript_stage(
    tmp_path: Path,
    transcription_error: TranscriptionError | None,
) -> None:
    """Remove inspection media after transcript success and failure."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()
    audio_path = request_directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    metadata_service = _FakeSourceMetadataService(
        _source(),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    transcription_service = _FakeTranscriptionService(
        _transcript(),
        error=transcription_error,
    )
    pipeline = _pipeline(metadata_service, transcription_service)

    if transcription_error is None:
        await pipeline.run(_CANONICAL_URL)
    else:
        with pytest.raises(TranscriptionError):
            await pipeline.run(_CANONICAL_URL)

    assert not request_directory.exists()


async def test_pipeline_preserves_transcription_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Keep the transcript failure when temporary media cleanup also fails."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()
    audio_path = request_directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    transcription_error = TranscriptionError("Transcript is unavailable for this video.")
    pipeline = _pipeline(
        _FakeSourceMetadataService(
            _source(),
            prepared_audio=PreparedAudio(audio_path, request_directory),
        ),
        _FakeTranscriptionService(_transcript(), error=transcription_error),
    )

    def raise_cleanup_error(path: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(shutil, "rmtree", raise_cleanup_error)

    with (
        caplog.at_level(logging.ERROR, logger=transcription_inspection.__name__),
        pytest.raises(TranscriptionError) as error,
    ):
        await pipeline.run(_CANONICAL_URL)

    assert error.value is transcription_error
    assert request_directory.exists()
    assert any(record.getMessage() == "temporary media cleanup failed" for record in caplog.records)
