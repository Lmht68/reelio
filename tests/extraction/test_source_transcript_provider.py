"""Local Source-and-Transcript provider contract tests."""

import asyncio
import logging
import shutil
from pathlib import Path

import pytest

import reelio.extraction.services.transcription.inspection as transcription_inspection
from reelio.extraction.exceptions import InvalidSourceError, TranscriptionError
from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import (
    InspectedSource,
    LocalSourceTranscriptProvider,
)
from reelio.extraction.types import Platform, Source, Transcript, TranscriptMethod

_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _SourceMetadataInspector:
    def __init__(
        self,
        source: Source | None = None,
        error: Exception | None = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> None:
        self.source = source
        self.error = error
        self.prepared_audio = prepared_audio
        self.calls: list[str] = []

    async def inspect(self, submitted_url: str) -> InspectedSource:
        self.calls.append(submitted_url)
        if self.error is not None:
            raise self.error
        assert self.source is not None
        return InspectedSource(self.source, self.prepared_audio)


class _TranscriptAcquirer:
    def __init__(
        self,
        transcript: Transcript,
        error: Exception | None = None,
    ) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[Source, str, PreparedAudio | None]] = []

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: PreparedAudio | None = None,
    ) -> Transcript:
        self.calls.append((source, submitted_url, prepared_audio))
        if self.error is not None:
            raise self.error
        return self.transcript


def _source() -> Source:
    return Source(
        platform=Platform.YOUTUBE,
        video_id=_VIDEO_ID,
        url=_CANONICAL_URL,
        title="Provider test video",
        description="Provider test description.",
        channel="Provider test channel",
        duration_seconds=42,
    )


def _transcript() -> Transcript:
    return Transcript(
        text="A provider transcript.",
        language="en",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    )


async def test_local_provider_acquires_source_and_transcript_for_submitted_url() -> None:
    """Return the acquired domain values after one local inspection and acquisition."""
    source = _source()
    transcript = _transcript()
    source_inspector = _SourceMetadataInspector(source)
    transcript_acquirer = _TranscriptAcquirer(transcript)
    provider = LocalSourceTranscriptProvider(source_inspector, transcript_acquirer)

    acquired_source, acquired_transcript = await provider.acquire(_CANONICAL_URL)

    assert acquired_source is source
    assert acquired_transcript is transcript
    assert source_inspector.calls == [_CANONICAL_URL]
    assert transcript_acquirer.calls == [(source, _CANONICAL_URL, None)]


async def test_local_provider_does_not_acquire_transcript_when_inspection_fails() -> None:
    """Propagate inspection failures before the Transcript acquisition stage."""
    source_error = InvalidSourceError("invalid source")
    source_inspector = _SourceMetadataInspector(error=source_error)
    transcript_acquirer = _TranscriptAcquirer(_transcript())
    provider = LocalSourceTranscriptProvider(source_inspector, transcript_acquirer)

    with pytest.raises(InvalidSourceError) as error:
        await provider.acquire(_CANONICAL_URL)

    assert error.value is source_error
    assert transcript_acquirer.calls == []


@pytest.mark.parametrize(
    "transcription_error",
    [None, TranscriptionError("Transcript is unavailable for this video.")],
)
async def test_local_provider_removes_inspection_media_after_acquisition(
    tmp_path: Path,
    transcription_error: TranscriptionError | None,
) -> None:
    """Remove request-scoped inspection media after Transcript success and failure."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()
    audio_path = request_directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    source_inspector = _SourceMetadataInspector(
        _source(),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    transcript_acquirer = _TranscriptAcquirer(
        _transcript(),
        error=transcription_error,
    )
    provider = LocalSourceTranscriptProvider(source_inspector, transcript_acquirer)

    if transcription_error is None:
        await provider.acquire(_CANONICAL_URL)
    else:
        with pytest.raises(TranscriptionError) as error:
            await provider.acquire(_CANONICAL_URL)
        assert error.value is transcription_error

    assert not request_directory.exists()


async def test_local_provider_preserves_transcription_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Keep a Transcript failure when request-scoped media cleanup also fails."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()
    audio_path = request_directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    transcription_error = TranscriptionError("Transcript is unavailable for this video.")
    source_inspector = _SourceMetadataInspector(
        _source(),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    transcript_acquirer = _TranscriptAcquirer(
        _transcript(),
        error=transcription_error,
    )
    provider = LocalSourceTranscriptProvider(source_inspector, transcript_acquirer)

    def raise_cleanup_error(path: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(shutil, "rmtree", raise_cleanup_error)

    with (
        caplog.at_level(logging.ERROR, logger=transcription_inspection.__name__),
        pytest.raises(TranscriptionError) as error,
    ):
        await provider.acquire(_CANONICAL_URL)

    assert error.value is transcription_error
    assert request_directory.exists()
    assert any(record.getMessage() == "temporary media cleanup failed" for record in caplog.records)


class _CancellingTranscriptAcquirer:
    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: PreparedAudio | None = None,
    ) -> Transcript:
        raise asyncio.CancelledError


async def test_local_provider_removes_inspection_media_after_cancellation(
    tmp_path: Path,
) -> None:
    """Remove request-scoped inspection media when Transcript acquisition cancels."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()
    audio_path = request_directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    source_inspector = _SourceMetadataInspector(
        _source(),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    provider = LocalSourceTranscriptProvider(
        source_inspector,
        _CancellingTranscriptAcquirer(),
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.acquire(_CANONICAL_URL)

    assert not request_directory.exists()
