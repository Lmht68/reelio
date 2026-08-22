"""Acquire Caption and Whisper Transcripts for validated Sources."""

import asyncio
import logging
import tempfile
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import ctranslate2
import yt_dlp
from faster_whisper import WhisperModel
from requests.exceptions import RequestException, Timeout
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    YouTubeTranscriptApi,
)
from yt_dlp.utils import DownloadError, YoutubeDLError

from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.inspection import _is_timeout_exception
from reelio.extraction.types import Transcript, TranscriptMethod

logger = logging.getLogger(__name__)

_YTDLP_OPTIONS: Final[dict[str, object]] = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "format": "bestaudio/best",
    # "postprocessors": [
    #     {
    #         "key": "FFmpegExtractAudio",
    #         "preferredcodec": "mp3",
    #         "preferredquality": "192",
    #     }
    # ],
}
_AUDIO_OUTPUT_TEMPLATE: Final[str] = "audio.%(ext)s"
_WHISPER_TEMP_PREFIX: Final[str] = "reelio-whisper-"


class _CaptionProviderFailure(Exception):
    """Represent an ordinary failure at the caption provider boundary."""


class _CaptionProviderTimeout(Exception):
    """Represent a timeout at the caption provider boundary."""


class _WhisperProviderFailure(Exception):
    """Represent an ordinary failure at the Whisper provider boundary."""


class _WhisperProviderTimeout(Exception):
    """Represent a terminal timeout at the Whisper provider boundary."""


def _log_acquisition_error(event: str, reason: str) -> None:
    """Log a safe, structured reason for an acquisition failure."""
    logger.debug(
        event,
        extra={
            "stage": "transcription",
            "reason": reason,
        },
    )


_CAPTION_EXTERNAL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    CouldNotRetrieveTranscript,
    ElementTree.ParseError,
    RequestException,
    AttributeError,
    IndexError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_WHISPER_EXTERNAL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    AttributeError,
    EOFError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class CaptionTrack(Protocol):
    """Expose the selection metadata and text for one Caption Track."""

    @property
    def language_code(self) -> str:
        """Return the track's original BCP 47 language code."""
        ...

    @property
    def is_generated(self) -> bool:
        """Return whether the provider generated the track automatically."""
        ...

    def fetch_segments(self) -> Sequence[str]:
        """Fetch the original timed-text segments without timestamps.

        Returns:
            Sequence[str]: Text content in provider segment order.

        Raises:
            _CaptionProviderFailure: If the provider payload is unusable.
            _CaptionProviderTimeout: If the provider request times out.
        """
        ...


class CaptionProvider(Protocol):
    """List Caption Tracks for a validated Source."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Return tracks in the provider's insertion order.

        Args:
            video_id: Stable external video identity.

        Returns:
            Sequence[CaptionTrack]: Available Caption Tracks.

        Raises:
            _CaptionProviderFailure: If track listing fails.
            _CaptionProviderTimeout: If track listing times out.
        """
        ...


@dataclass(frozen=True, slots=True)
class WhisperResult:
    """Contain normalized text and metadata from one Whisper operation."""

    text: str
    language: str
    segment_count: int


class AudioDownloader(Protocol):
    """Download one Source's native best-audio representation."""

    def download(self, source_url: str, destination: Path) -> Path:
        """Download audio into the provided request directory.

        Args:
            source_url: Validated provider URL to download.
            destination: Existing private request directory.

        Returns:
            Path: Completed audio file path owned by ``destination``.

        Raises:
            _WhisperProviderFailure: If the download result is unusable.
            _WhisperProviderTimeout: If the provider times out.
        """
        ...


class WhisperTranscriber(Protocol):
    """Transcribe one local audio file with a preloaded model."""

    def transcribe(self, audio_path: Path) -> WhisperResult:
        """Transcribe the provided local audio file.

        Args:
            audio_path: Validated completed audio file.

        Returns:
            WhisperResult: Normalized text and detected language.

        Raises:
            _WhisperProviderFailure: If model inference or output validation fails.
        """
        ...


class _LibrarySnippet(Protocol):
    """Expose the text field used from one provider snippet."""

    text: str


class _LibraryFetchedTranscript(Protocol):
    """Expose the iterable snippets returned by the provider."""

    def __iter__(self) -> Iterator[_LibrarySnippet]:
        """Return snippets in provider segment order."""
        ...


class _LibraryTranscript(Protocol):
    """Expose the provider track fields used by the adapter."""

    language_code: str
    is_generated: bool

    def fetch(self, preserve_formatting: bool = False) -> object:
        """Fetch provider timed-text data."""
        ...


class _YouTubeCaptionTrack:
    """Adapt one youtube-transcript-api track to CaptionTrack."""

    def __init__(self, transcript: _LibraryTranscript) -> None:
        try:
            language_code = transcript.language_code
            is_generated = transcript.is_generated
        except (AttributeError, TypeError, ValueError) as exc:
            _log_acquisition_error("caption provider error", "invalid_track_metadata")
            raise _CaptionProviderFailure from exc
        if not isinstance(language_code, str) or not isinstance(is_generated, bool):
            _log_acquisition_error("caption provider error", "invalid_track_metadata")
            raise _CaptionProviderFailure
        self._transcript = transcript
        self._language_code = language_code
        self._is_generated = is_generated

    @property
    def language_code(self) -> str:
        """Return the provider's original language code."""
        return self._language_code

    @property
    def is_generated(self) -> bool:
        """Return the provider's generated-track flag."""
        return self._is_generated

    def fetch_segments(self) -> Sequence[str]:
        """Fetch segment text with provider formatting disabled.

        Returns:
            Sequence[str]: Provider segment text in original order.

        Raises:
            _CaptionProviderFailure: If the provider payload is malformed.
            _CaptionProviderTimeout: If the provider request times out.
        """
        try:
            fetched = cast(
                _LibraryFetchedTranscript,
                self._transcript.fetch(preserve_formatting=False),
            )
            segments = tuple(snippet.text for snippet in fetched)
            if not all(isinstance(text, str) for text in segments):
                _log_acquisition_error(
                    "caption provider error",
                    "invalid_segment_payload",
                )
                raise _CaptionProviderFailure
            return segments
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_fetch_timeout")
            raise _CaptionProviderTimeout from exc
        except _CaptionProviderFailure:
            raise
        except _CAPTION_EXTERNAL_FAILURES as exc:
            _log_acquisition_error("caption provider error", "track_fetch_failed")
            raise _CaptionProviderFailure from exc


class YouTubeCaptionProvider:
    """Adapt youtube-transcript-api to the Reelio CaptionProvider contract."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """List Caption Tracks using one provider client instance.

        Args:
            video_id: Stable external video identity.

        Returns:
            Sequence[CaptionTrack]: Wrapped provider tracks in provider order.

        Raises:
            _CaptionProviderFailure: If the provider payload cannot be adapted.
            _CaptionProviderTimeout: If the listing request times out.
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = cast(Iterable[_LibraryTranscript], api.list(video_id))
            wrapped_tracks: list[CaptionTrack] = []
            for track in transcript_list:
                try:
                    wrapped_tracks.append(_YouTubeCaptionTrack(track))
                except _CaptionProviderFailure:
                    logger.debug(
                        "caption track unavailable",
                        extra={
                            "stage": "transcription",
                            "reason": "invalid_track_metadata",
                        },
                    )
                    continue
            return tuple(wrapped_tracks)
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_listing_timeout")
            raise _CaptionProviderTimeout from exc
        except _CAPTION_EXTERNAL_FAILURES as exc:
            _log_acquisition_error("caption provider error", "track_listing_failed")
            raise _CaptionProviderFailure from exc


class _WhisperSegment(Protocol):
    """Expose the text field returned by faster-whisper."""

    text: str


class _WhisperInfo(Protocol):
    """Expose detected language returned by faster-whisper."""

    language: str


class _WhisperModel(Protocol):
    """Expose the faster-whisper method used by the adapter."""

    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        vad_filter: bool,
        temperature: float,
        condition_on_previous_text: bool,
        initial_prompt: str,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]:
        """Return a lazy segment iterator and transcription metadata."""
        ...


class YtDlpAudioDownloader:
    """Download native best audio into a private request directory."""

    def download(self, source_url: str, destination: Path) -> Path:
        """Download one Source's native best-audio representation.

        Args:
            source_url: Validated provider URL to download.
            destination: Existing private request directory.

        Returns:
            Path: Validated completed audio file inside ``destination``.

        Raises:
            _WhisperProviderFailure: If yt-dlp returns unusable output.
            _WhisperProviderTimeout: If yt-dlp reports a typed timeout.
        """
        request_directory = destination.resolve()
        options = {
            **_YTDLP_OPTIONS,
            "outtmpl": str(request_directory / _AUDIO_OUTPUT_TEMPLATE),
        }
        try:
            with yt_dlp.YoutubeDL(options) as youtube_dl:
                raw_info = youtube_dl.extract_info(source_url, download=True)
                if not isinstance(raw_info, Mapping) or "entries" in raw_info:
                    _log_acquisition_error("whisper provider error", "invalid_download_result")
                    raise _WhisperProviderFailure
                prepared_path = youtube_dl.prepare_filename(raw_info)
        except _WhisperProviderFailure:
            raise
        except DownloadError as exc:
            if _is_timeout_exception(exc):
                _log_acquisition_error(
                    "whisper provider timeout",
                    "audio_download_timeout",
                )
                raise _WhisperProviderTimeout from exc
            _log_acquisition_error("whisper provider error", "audio_download_failed=DownloadError")
            raise _WhisperProviderFailure from exc
        except YoutubeDLError as exc:
            _log_acquisition_error("whisper provider error", "audio_download_failed=YoutubeDLError")
            raise _WhisperProviderFailure from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _log_acquisition_error("whisper provider error", "invalid_download_result")
            raise _WhisperProviderFailure from exc

        if not isinstance(prepared_path, (str, Path)):
            _log_acquisition_error("whisper provider error", "invalid_download_path")
            raise _WhisperProviderFailure
        completed_path = Path(prepared_path).resolve()
        if completed_path.parent != request_directory:
            _log_acquisition_error(
                "whisper provider error",
                "audio_path_outside_request_directory",
            )
            raise _WhisperProviderFailure
        if not completed_path.is_file():
            _log_acquisition_error("whisper provider error", "audio_file_missing")
            raise _WhisperProviderFailure
        return completed_path


class FasterWhisperTranscriber:
    """Adapt one preloaded faster-whisper model to WhisperTranscriber."""

    def __init__(self, model: object, settings: TranscriptionConfig) -> None:
        """Initialize the adapter around an already-loaded model.

        Args:
            model: Preloaded faster-whisper model instance.
            settings: Environment-backed options for model inference.
        """
        self._model = model
        self._settings = settings

    def transcribe(self, audio_path: Path) -> WhisperResult:
        """Transcribe and normalize one local audio file.

        Args:
            audio_path: Validated completed audio file.

        Returns:
            WhisperResult: Normalized text, detected language, and segment count.

        Raises:
            _WhisperProviderFailure: If model output or inference is unusable.
        """
        segment_count = 0
        try:
            settings = self._settings
            segments, info = cast(_WhisperModel, self._model).transcribe(
                str(audio_path),
                beam_size=settings.whisper_beam_size,
                vad_filter=settings.whisper_vad_filter,
                temperature=settings.whisper_temperature,
                condition_on_previous_text=settings.whisper_cond_on_prev_txt,
                initial_prompt=settings.whisper_initial_prompt,
            )

            def segment_texts() -> Iterator[str]:
                nonlocal segment_count
                for segment in segments:
                    segment_count += 1
                    text = segment.text
                    if not isinstance(text, str):
                        logger.debug(
                            "invalid segment text type",
                            extra={"stage": "transcription"},
                        )
                        raise TypeError
                    yield text

            text = _normalize_segments(segment_texts())
            language = info.language
            if not isinstance(language, str) or not language.strip() or not text:
                logger.debug("invalid language type", extra={"stage": "transcription"})
                raise ValueError
        except _WHISPER_EXTERNAL_FAILURES as exc:
            _log_acquisition_error("whisper provider error", "transcription_failed")
            raise _WhisperProviderFailure from exc

        return WhisperResult(
            text=text,
            language=language,
            segment_count=segment_count,
        )


def load_whisper_transcriber(
    settings: TranscriptionConfig,
) -> FasterWhisperTranscriber:
    """Load one configured faster-whisper model and wrap it.

    Args:
        settings: Environment-backed transcription settings.

    Returns:
        FasterWhisperTranscriber: Adapter around the loaded model.

    Raises:
        RuntimeError: If explicit CUDA configuration has no CUDA device.
        Exception: If faster-whisper cannot load or download the model.
    """
    if settings.whisper_device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        _log_acquisition_error("whisper provider error", "cuda_device_unavailable")
        raise RuntimeError("REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available.")
    model = WhisperModel(
        model_size_or_path=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return FasterWhisperTranscriber(model, settings)


async def _finish_cancelled_worker(
    worker: asyncio.Task[Transcript],
) -> None:
    """Wait for a shielded native worker before releasing its semaphore."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    try:
        worker.result()
    except (_WhisperProviderFailure, _WhisperProviderTimeout):
        return
    except Exception:
        logger.exception(
            "Whisper worker failed after request cancellation",
            extra={
                "stage": "transcription",
                "reason": "unexpected_worker_failure",
            },
        )


def _validate_audio_path(audio_path: Path, request_directory: Path) -> Path:
    if not isinstance(audio_path, Path):
        _log_acquisition_error("whisper provider error", "invalid_audio_path_type")
        raise _WhisperProviderFailure
    resolved_path = audio_path.resolve()
    if resolved_path.parent != request_directory:
        _log_acquisition_error(
            "whisper provider error",
            "audio_path_outside_request_directory",
        )
        raise _WhisperProviderFailure
    if not resolved_path.is_file():
        _log_acquisition_error("whisper provider error", "audio_file_missing")
        raise _WhisperProviderFailure
    return resolved_path


def _rank_caption_tracks(
    tracks: Sequence[CaptionTrack],
) -> tuple[CaptionTrack, ...]:
    buckets: list[list[CaptionTrack]] = [[], [], [], [], [], []]
    for track in tracks:
        language_code = track.language_code.casefold()
        is_english = language_code == "en" or language_code.startswith("en-")
        if is_english:
            if track.is_generated:
                bucket = 2 if language_code == "en" else 3
            else:
                bucket = 0 if language_code == "en" else 1
        else:
            bucket = 5 if track.is_generated else 4
        buckets[bucket].append(track)
    return tuple(track for bucket in buckets for track in bucket)


def _normalize_segments(segments: Iterable[str]) -> str:
    tokens: list[str] = []
    for segment in segments:
        if not isinstance(segment, str):
            raise TypeError
        tokens.extend(segment.split())
    return " ".join(tokens)


def acquire_transcript(
    provider: CaptionProvider,
    video_id: str,
) -> Transcript | None:
    try:
        tracks = provider.list_tracks(video_id)
        ranked_tracks = _rank_caption_tracks(tracks)
    except _CaptionProviderTimeout:
        raise
    except _CaptionProviderFailure:
        raise
    except (Timeout, TimeoutError) as exc:
        _log_acquisition_error("caption provider timeout", "track_listing_timeout")
        raise _CaptionProviderTimeout from exc
    except _CAPTION_EXTERNAL_FAILURES as exc:
        _log_acquisition_error("caption provider error", "track_listing_failed")
        raise _CaptionProviderFailure from exc

    for track in ranked_tracks:
        try:
            segments = track.fetch_segments()
            segment_count = len(segments)
            transcript_text = _normalize_segments(segments)
            if not transcript_text:
                continue
            language = track.language_code
            if not isinstance(language, str) or not language.strip():
                raise TypeError
        except _CaptionProviderTimeout:
            raise
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_fetch_timeout")
            raise _CaptionProviderTimeout from exc
        except _CAPTION_EXTERNAL_FAILURES:
            logger.debug(
                "caption track unavailable",
                extra={
                    "stage": "transcription",
                    "reason": "track_fetch_failed",
                },
            )
            continue

        method = TranscriptMethod.YOUTUBE_CAPTIONS
        logger.debug(
            "transcript acquired",
            extra={
                "stage": "transcription",
                "transcript_text": transcript_text,
                "language": language,
                "method": method.value,
                "segment_count": segment_count,
            },
        )
        return Transcript(
            text=transcript_text,
            language=language,
            method=method,
        )

    return None


async def acquire_whisper(
    source_url: str,
    audio_downloader: AudioDownloader,
    transcriber: WhisperTranscriber,
    temp_media_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Transcript:
    """Run one complete Whisper operation under the shared semaphore.

    Args:
        source_url: Validated provider URL whose audio should be downloaded.
        audio_downloader: Synchronous native-audio adapter.
        transcriber: Synchronous preloaded Whisper adapter.
        temp_media_dir: Root directory for request-scoped media.
        semaphore: Application-lifetime Whisper concurrency gate.

    Returns:
        Transcript: Normalized Whisper transcript.

    Raises:
        _WhisperProviderFailure: If download or model output is unusable.
        _WhisperProviderTimeout: If download or model inference times out.
        asyncio.CancelledError: If the request is cancelled.
    """
    await semaphore.acquire()
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(
                _acquire_whisper_sync,
                source_url,
                audio_downloader,
                transcriber,
                temp_media_dir,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _finish_cancelled_worker(worker)
            raise
    finally:
        semaphore.release()


def _acquire_whisper_sync(
    source_url: str,
    audio_downloader: AudioDownloader,
    transcriber: WhisperTranscriber,
    temp_media_dir: Path,
) -> Transcript:
    temp_media_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=_WHISPER_TEMP_PREFIX,
        dir=str(temp_media_dir),
    ) as request_directory_name:
        request_directory = Path(request_directory_name).resolve()
        try:
            downloaded_path = audio_downloader.download(source_url, request_directory)
        except _WhisperProviderTimeout:
            raise
        except _WhisperProviderFailure:
            raise
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("whisper provider timeout", "audio_download_timeout")
            raise _WhisperProviderTimeout from exc

        audio_path = _validate_audio_path(downloaded_path, request_directory)
        audio_size_bytes = audio_path.stat().st_size
        try:
            result = transcriber.transcribe(audio_path)
        except _WhisperProviderTimeout:
            raise
        except _WhisperProviderFailure:
            raise
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("whisper provider timeout", "transcription_timeout")
            raise _WhisperProviderTimeout from exc

        if (
            not isinstance(result, WhisperResult)
            or not isinstance(result.text, str)
            or not isinstance(result.language, str)
            or not result.language.strip()
            or result.segment_count <= 0
        ):
            _log_acquisition_error(
                "whisper provider error",
                "invalid_transcription_result",
            )
            raise _WhisperProviderFailure
        transcript_text = _normalize_segments((result.text,))
        if not transcript_text:
            _log_acquisition_error(
                "whisper provider error",
                "empty_transcription_result",
            )
            raise _WhisperProviderFailure

        method = TranscriptMethod.WHISPER
        logger.debug(
            "transcript acquired",
            extra={
                "stage": "transcription",
                "transcript_text": transcript_text,
                "language": result.language,
                "method": method.value,
                "segment_count": result.segment_count,
                "audio_path": str(audio_path),
                "audio_size_bytes": audio_size_bytes,
            },
        )
        return Transcript(
            text=transcript_text,
            language=result.language,
            method=method,
        )
