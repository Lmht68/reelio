"""Pipeline orchestration contract tests."""

import pytest

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
)
from reelio.extraction.service import ExtractionPipeline
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
    ) -> None:
        self.source = source
        self.error = error
        self.calls: list[str] = []

    async def inspect(self, submitted_url: str) -> Source:
        self.calls.append(submitted_url)
        if self.error is not None:
            raise self.error
        assert self.source is not None
        return self.source


class _FakeTranscriptionService:
    def __init__(self, transcript: Transcript, error: Exception | None = None) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[Source] = []

    async def acquire(self, source: Source) -> Transcript:
        self.calls.append(source)
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
    assert transcription_service.calls == [source]
    assert [item.status for item in result.results] == [
        ResultStatus.RESOLVED,
        ResultStatus.AMBIGUOUS,
        ResultStatus.UNRESOLVED,
    ]
    assert result.results[0].movie is not None
    assert result.results[0].movie.title == "Dune: Part Two"
    assert result.results[1].candidates[0].title == "Dune"
    assert result.results[2].mentioned_as == ["that 90s space movie"]


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
    transcription_service = _FakeTranscriptionService(
        _transcript(), error=transcription_error
    )
    pipeline = ExtractionPipeline(metadata_service, transcription_service)

    with pytest.raises(type(transcription_error)) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is transcription_error
    assert transcription_service.calls == [_source()]


async def test_pipeline_result_keeps_each_placeholder_result_branch() -> None:
    """Keep the existing resolved, ambiguous, and unresolved result shapes."""
    pipeline = ExtractionPipeline(
        _FakeSourceMetadataService(_source()),
        _FakeTranscriptionService(_transcript()),
    )

    result = await pipeline.run(_CANONICAL_URL)

    resolved, ambiguous, unresolved = result.results
    assert isinstance(resolved, MentionResult)
    assert resolved.movie is not None
    assert resolved.candidates == []
    assert ambiguous.movie is None
    assert len(ambiguous.candidates) == 3
    assert unresolved.movie is None
    assert unresolved.candidates == []
