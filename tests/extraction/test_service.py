"""Pipeline orchestration contract tests."""

import logging
import shutil
from pathlib import Path

import pytest

import reelio.extraction.services.transcription.inspection as transcription_inspection
from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
)
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    MentionResult,
    Platform,
    ResultStatus,
    Source,
    Transcript,
    TranscriptMethod,
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


async def test_pipeline_returns_real_source_and_transcript_with_placeholders() -> None:
    """Combine inspected Source and acquired Transcript with placeholder results."""
    source = _source()
    transcript = _transcript()
    metadata_service = _FakeSourceMetadataService(source)
    transcription_service = _FakeTranscriptionService(transcript)
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

    result = await pipeline.run(_CANONICAL_URL)

    assert result.source is source
    assert result.transcript is transcript
    assert metadata_service.calls == [_CANONICAL_URL]
    assert transcription_service.calls == [(source, _CANONICAL_URL)]
    assert [item.status for item in result.results] == [
        ResultStatus.RESOLVED,
        ResultStatus.UNRESOLVED,
    ]
    assert result.results[0].movie_mention.title == "Dune: Part Two"
    assert result.results[0].movie is not None
    assert result.results[0].movie.title == "Dune: Part Two"
    assert result.results[1].movie_mention.title == "Che"
    assert result.results[1].movie is None


async def test_pipeline_preserves_whisper_transcript_method() -> None:
    """Pass a Whisper Transcript through orchestration unchanged."""
    transcript = Transcript(
        text="Spoken audio transcript.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    pipeline = ExtractionPipeline(
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
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

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
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

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
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

    with pytest.raises(type(transcription_error)) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is transcription_error
    assert transcription_service.calls == [(_source(), _CANONICAL_URL)]


async def test_pipeline_result_keeps_resolved_and_unresolved_placeholder_results() -> None:
    """Keep the existing resolved and unresolved placeholder result shapes."""
    pipeline = ExtractionPipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
    )

    result = await pipeline.run(_CANONICAL_URL)

    resolved, unresolved = result.results
    assert isinstance(resolved, MentionResult)
    assert resolved.movie_mention.title == "Dune: Part Two"
    assert resolved.movie is not None
    assert unresolved.movie_mention.title == "Che"
    assert unresolved.movie is None


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
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

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
    pipeline = ExtractionPipeline(
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
