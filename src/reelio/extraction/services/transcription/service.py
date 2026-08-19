"""Inspect YouTube Sources and acquire Caption or Whisper Transcripts."""

import asyncio
import logging
import math
import re
import socket
import tempfile
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Final, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import ctranslate2  # type: ignore[import-untyped]
import yt_dlp  # type: ignore[import-untyped]
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from requests.exceptions import RequestException, Timeout
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    YouTubeTranscriptApi,
)
from yt_dlp.utils import DownloadError, YoutubeDLError  # type: ignore[import-untyped]

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.types import Platform, Source, Transcript, TranscriptMethod

logger = logging.getLogger(__name__)

_ALLOWED_YOUTUBE_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)
_PATH_BASED_FORMS: Final[frozenset[str]] = frozenset({"shorts", "embed", "live"})
_VIDEO_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]{11}")
_CANONICAL_URL_TEMPLATE: Final[str] = "https://www.youtube.com/watch?v={video_id}"
_YTDLP_OPTIONS: Final[dict[str, object]] = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "ignoreconfig": True,
}
_UNAVAILABLE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "private video",
        "this video is private",
        "video is private",
        "video unavailable",
        "video is unavailable",
        "this video is unavailable",
        "video not found",
        "video is not available",
        "this video is not available",
        "video has been removed",
        "this video has been removed",
        "video was removed",
        "video has been deleted",
        "this video has been deleted",
        "account has been terminated",
        "not available in your country",
        "not available in your region",
        "blocked in your country",
        "geo-restricted",
        "geo restricted",
        "geo-blocked",
        "geoblocked",
        "country restriction",
        "age-restricted",
        "age restricted",
        "confirm your age",
        "login required",
        "sign in required",
        "log in to confirm",
        "sign in to confirm",
        "requires login",
        "requires you to sign in",
    }
)
_INVALID_SOURCE_MESSAGE: Final[str] = "Invalid YouTube URL."
_UNSUPPORTED_PLATFORM_MESSAGE: Final[str] = "Only YouTube URLs are supported."
_SOURCE_UNAVAILABLE_MESSAGE: Final[str] = "YouTube video is unavailable."
_METADATA_PROVIDER_MESSAGE: Final[str] = "Unable to retrieve YouTube metadata."
_REDACTED_LOG_VALUE: Final[str] = "[REDACTED]"
_SENSITIVE_QUERY_PARTS: Final[frozenset[str]] = frozenset(
    {"api", "apikey", "auth", "authorization", "key", "password", "secret", "token"}
)

_MISSING = object()
_TRANSCRIPT_UNAVAILABLE_MESSAGE: Final[str] = (
    "Transcript is unavailable for this video."
)
_TRANSCRIPT_TIMEOUT_MESSAGE: Final[str] = "Transcript acquisition timed out."
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


def _is_timeout_exception(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, (Timeout, TimeoutError, socket.timeout)):
            return True
        if isinstance(candidate, DownloadError):
            exc_info = getattr(candidate, "exc_info", None)
            if isinstance(exc_info, tuple) and len(exc_info) > 1:
                nested = exc_info[1]
                if isinstance(nested, BaseException):
                    pending.append(nested)
        for attribute in ("__cause__", "__context__", "reason"):
            nested = getattr(candidate, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


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

    def download(self, source: Source, destination: Path) -> Path:
        """Download audio into the provided request directory.

        Args:
            source: Canonical YouTube Source to download.
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
            raise _CaptionProviderFailure from exc
        if not isinstance(language_code, str) or not isinstance(is_generated, bool):
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
                raise _CaptionProviderFailure
            return segments
        except (Timeout, TimeoutError) as exc:
            raise _CaptionProviderTimeout from exc
        except _CaptionProviderFailure:
            raise
        except _CAPTION_EXTERNAL_FAILURES as exc:
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
                        extra={"stage": "transcription"},
                    )
                    continue
            return tuple(wrapped_tracks)
        except (Timeout, TimeoutError) as exc:
            raise _CaptionProviderTimeout from exc
        except _CAPTION_EXTERNAL_FAILURES as exc:
            raise _CaptionProviderFailure from exc


class MetadataExtractor(Protocol):
    """Retrieve raw metadata for one canonical video URL."""

    def extract(self, canonical_url: str) -> Mapping[str, object]:
        """Return provider metadata without downloading media.

        Args:
            canonical_url: Canonical watch URL for one video.

        Returns:
            Mapping[str, object]: Raw provider metadata.
        """
        ...


class YtDlpMetadataExtractor:
    """Retrieve one video's metadata through yt-dlp."""

    def extract(self, canonical_url: str) -> Mapping[str, object]:
        """Extract metadata without downloading the video.

        Args:
            canonical_url: Canonical watch URL for one video.

        Returns:
            Mapping[str, object]: Raw yt-dlp metadata.

        Raises:
            MetadataProviderError: If yt-dlp returns a non-mapping value.
        """
        with yt_dlp.YoutubeDL(_YTDLP_OPTIONS) as youtube_dl:
            raw_metadata = youtube_dl.extract_info(canonical_url, download=False)

        if not isinstance(raw_metadata, Mapping):
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
        return cast(Mapping[str, object], raw_metadata)


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
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]:
        """Return a lazy segment iterator and transcription metadata."""
        ...


class YtDlpAudioDownloader:
    """Download native best audio into a private request directory."""

    def download(self, source: Source, destination: Path) -> Path:
        """Download one Source's native best-audio representation.

        Args:
            source: Canonical YouTube Source to download.
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
            "format": "bestaudio/best",
            "outtmpl": str(request_directory / _AUDIO_OUTPUT_TEMPLATE),
        }
        try:
            with yt_dlp.YoutubeDL(options) as youtube_dl:
                raw_info = youtube_dl.extract_info(source.url, download=True)
                if not isinstance(raw_info, Mapping) or "entries" in raw_info:
                    raise _WhisperProviderFailure
                prepared_path = youtube_dl.prepare_filename(raw_info)
        except _WhisperProviderFailure:
            raise
        except DownloadError as exc:
            if _is_timeout_exception(exc):
                raise _WhisperProviderTimeout from exc
            raise _WhisperProviderFailure from exc
        except YoutubeDLError as exc:
            raise _WhisperProviderFailure from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _WhisperProviderFailure from exc

        if not isinstance(prepared_path, (str, Path)):
            raise _WhisperProviderFailure
        completed_path = Path(prepared_path).resolve()
        if completed_path.parent != request_directory or not completed_path.is_file():
            raise _WhisperProviderFailure
        return completed_path


class FasterWhisperTranscriber:
    """Adapt one preloaded faster-whisper model to WhisperTranscriber."""

    def __init__(self, model: object) -> None:
        """Initialize the adapter around an already-loaded model.

        Args:
            model: Preloaded faster-whisper model instance.
        """
        self._model = model

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
            segments, info = cast(_WhisperModel, self._model).transcribe(
                str(audio_path)
            )

            def segment_texts() -> Iterator[str]:
                nonlocal segment_count
                for segment in segments:
                    segment_count += 1
                    text = segment.text
                    if not isinstance(text, str):
                        raise TypeError
                    yield text

            text = _normalize_segments(segment_texts())
            language = info.language
            if not isinstance(language, str) or not language.strip() or not text:
                raise ValueError
        except _WHISPER_EXTERNAL_FAILURES as exc:
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
        raise RuntimeError(
            "REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available."
        )
    model = WhisperModel(
        model_size_or_path=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return FasterWhisperTranscriber(model)


class SourceMetadataService:
    """Validate one YouTube URL and return normalized source metadata."""

    def __init__(
        self,
        extractor: MetadataExtractor,
        settings: TranscriptionConfig,
    ) -> None:
        """Initialize the service with provider and duration-limit dependencies.

        Args:
            extractor: Synchronous provider adapter for metadata retrieval.
            settings: Transcription settings containing the duration limit.
        """
        self._extractor = extractor
        self._settings = settings

    async def inspect(self, submitted_url: str) -> Source:
        """Validate a URL, retrieve metadata, enforce duration, and return a Source.

        Args:
            submitted_url: URL submitted by the API caller.

        Returns:
            Source: Canonical YouTube identity and normalized metadata.

        Raises:
            InvalidSourceError: If the URL does not identify one YouTube video.
            UnsupportedPlatformError: If the URL belongs to another host.
            SourceUnavailableError: If the provider reports inaccessible content.
            MetadataProviderError: If provider access or metadata normalization fails.
            DurationLimitExceededError: If the video exceeds the configured limit.
        """
        platform, video_id, canonical_url = _canonicalize_url(submitted_url)

        try:
            raw_metadata = await asyncio.to_thread(
                self._extractor.extract,
                canonical_url,
            )
        except MetadataProviderError as exc:
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc
        except YoutubeDLError as exc:
            if _is_unavailable_error(str(exc)):
                raise SourceUnavailableError(_SOURCE_UNAVAILABLE_MESSAGE) from exc
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc

        if not isinstance(raw_metadata, Mapping):
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)

        title, description, channel, duration_seconds = _normalize_metadata(
            raw_metadata,
            video_id,
        )
        source = Source(
            platform=platform,
            video_id=video_id,
            url=canonical_url,
            title=title,
            description=description,
            channel=channel,
            duration_seconds=duration_seconds,
        )
        logger.debug(
            "source metadata normalized",
            extra={
                "stage": "transcription",
                "submitted_url": _safe_submitted_url(submitted_url),
                "platform": source.platform.value,
                "video_id": source.video_id,
                "canonical_url": source.url,
                "title": _REDACTED_LOG_VALUE,
                "title_length": len(source.title),
                "description": _REDACTED_LOG_VALUE,
                "description_length": len(source.description),
                "channel": _REDACTED_LOG_VALUE,
                "channel_length": len(source.channel),
                "duration_seconds": source.duration_seconds,
            },
        )

        if duration_seconds > self._settings.max_video_duration_seconds:
            raise DurationLimitExceededError(
                "Video exceeds the configured duration limit of "
                f"{self._settings.max_video_duration_seconds} seconds."
            )
        return source


class TranscriptionService:
    """Acquire a Transcript from captions with a Whisper fallback."""

    def __init__(
        self,
        provider: CaptionProvider,
        audio_downloader: AudioDownloader,
        transcriber: WhisperTranscriber,
        temp_media_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize caption and Whisper acquisition dependencies.

        Args:
            provider: Synchronous provider boundary for Caption Tracks.
            audio_downloader: Synchronous native-audio download adapter.
            transcriber: Preloaded synchronous Whisper adapter.
            temp_media_dir: Root directory for request-scoped media.
            semaphore: Application-lifetime Whisper concurrency gate.
        """
        self._provider = provider
        self._audio_downloader = audio_downloader
        self._transcriber = transcriber
        self._temp_media_dir = temp_media_dir
        self._semaphore = semaphore

    async def acquire(self, source: Source) -> Transcript:
        """Acquire a normalized Transcript for a validated Source.

        Args:
            source: Validated Source whose video ID identifies provider data.

        Returns:
            Transcript: Caption or Whisper text and acquisition metadata.

        Raises:
            TranscriptionError: If captions and Whisper produce no Transcript.
            PipelineTimeoutError: If the terminal Whisper path times out.
        """
        try:
            transcript = await asyncio.to_thread(
                _acquire_transcript,
                self._provider,
                source.video_id,
            )
        except _CaptionProviderFailure, _CaptionProviderTimeout:
            transcript = None

        if transcript is not None:
            return transcript

        try:
            return await self._acquire_whisper(source)
        except _WhisperProviderTimeout as exc:
            raise PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE) from exc
        except _WhisperProviderFailure as exc:
            raise TranscriptionError(_TRANSCRIPT_UNAVAILABLE_MESSAGE) from exc

    async def _acquire_whisper(self, source: Source) -> Transcript:
        await self._semaphore.acquire()
        try:
            worker = asyncio.create_task(
                asyncio.to_thread(self._acquire_whisper_sync, source)
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _finish_cancelled_worker(worker)
                raise
        finally:
            self._semaphore.release()

    def _acquire_whisper_sync(self, source: Source) -> Transcript:
        self._temp_media_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=_WHISPER_TEMP_PREFIX,
            dir=str(self._temp_media_dir),
        ) as request_directory_name:
            request_directory = Path(request_directory_name).resolve()
            try:
                downloaded_path = self._audio_downloader.download(
                    source,
                    request_directory,
                )
            except _WhisperProviderTimeout:
                raise
            except _WhisperProviderFailure:
                raise
            except (Timeout, TimeoutError) as exc:
                raise _WhisperProviderTimeout from exc
            audio_path = _validate_audio_path(downloaded_path, request_directory)
            audio_size_bytes = audio_path.stat().st_size
            try:
                result = self._transcriber.transcribe(audio_path)
            except _WhisperProviderTimeout:
                raise
            except _WhisperProviderFailure:
                raise
            except (Timeout, TimeoutError) as exc:
                raise _WhisperProviderTimeout from exc
            if (
                not isinstance(result, WhisperResult)
                or not isinstance(result.text, str)
                or not isinstance(result.language, str)
                or not result.language.strip()
                or result.segment_count <= 0
            ):
                raise _WhisperProviderFailure
            transcript_text = _normalize_segments((result.text,))
            if not transcript_text:
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
    except _WhisperProviderFailure, _WhisperProviderTimeout:
        return
    except Exception:
        logger.exception("Whisper worker failed after request cancellation")


def _validate_audio_path(audio_path: Path, request_directory: Path) -> Path:
    if not isinstance(audio_path, Path):
        raise _WhisperProviderFailure
    resolved_path = audio_path.resolve()
    if resolved_path.parent != request_directory or not resolved_path.is_file():
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


def _acquire_transcript(
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
        raise _CaptionProviderTimeout from exc
    except _CAPTION_EXTERNAL_FAILURES as exc:
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
            raise _CaptionProviderTimeout from exc
        except _CAPTION_EXTERNAL_FAILURES:
            logger.debug(
                "caption track unavailable",
                extra={"stage": "transcription"},
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


def _canonicalize_url(submitted_url: str) -> tuple[Platform, str, str]:
    if not submitted_url or _contains_control_character(submitted_url):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if _contains_malformed_percent_encoding(submitted_url):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    try:
        parsed_url = urlsplit(submitted_url)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
    except ValueError as exc:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE) from exc

    if parsed_url.scheme.casefold() != "https" or hostname is None:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if username is not None or password is not None:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if port is not None or _authority_has_port(parsed_url.netloc):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    host = hostname.casefold()
    if _looks_like_youtube_host_trick(host):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if host not in _ALLOWED_YOUTUBE_HOSTS:
        raise UnsupportedPlatformError(_UNSUPPORTED_PLATFORM_MESSAGE)

    try:
        query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    except ValueError as exc:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE) from exc

    path = parsed_url.path
    if "%" in path:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    path_segments = _path_segments(path)
    video_id = _video_id_for_path(host, path_segments, query_pairs)
    if _VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    return (
        Platform.YOUTUBE,
        video_id,
        _CANONICAL_URL_TEMPLATE.format(video_id=video_id),
    )


def _contains_control_character(value: str) -> bool:
    return any(character.isspace() or ord(character) < 0x20 for character in value)


def _contains_malformed_percent_encoding(value: str) -> bool:
    for index, character in enumerate(value):
        if character == "%":
            if index + 2 >= len(value):
                return True
            if not all(
                hex_digit in "0123456789abcdefABCDEF"
                for hex_digit in value[index + 1 : index + 3]
            ):
                return True
    return False


def _authority_has_port(netloc: str) -> bool:
    authority = netloc.rsplit("@", maxsplit=1)[-1]
    if authority.startswith("["):
        closing_bracket = authority.find("]")
        if closing_bracket == -1:
            return True
        return authority[closing_bracket + 1 :].startswith(":")
    return ":" in authority


def _looks_like_youtube_host_trick(host: str) -> bool:
    if host in _ALLOWED_YOUTUBE_HOSTS:
        return False
    for base_host in ("youtube.com", "youtu.be"):
        if (
            host.startswith(f"{base_host}.")
            or host.endswith(f".{base_host}")
            or f".{base_host}." in host
        ):
            return True
    return False


def _path_segments(path: str) -> list[str]:
    if not path or not path.startswith("/"):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    segments = path[1:].split("/")
    if any(not segment for segment in segments):
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    return segments


def _video_id_for_path(
    host: str,
    path_segments: list[str],
    query_pairs: list[tuple[str, str]],
) -> str:
    video_query_values = [value for key, value in query_pairs if key == "v"]
    if len(video_query_values) > 1:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    if host == "youtu.be":
        if len(path_segments) != 1:
            raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
        video_id = path_segments[0]
        if video_query_values and video_query_values[0] not in {"", video_id}:
            raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    elif path_segments[0] == "watch":
        if len(path_segments) != 1 or len(video_query_values) != 1:
            raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
        video_id = video_query_values[0]
    elif path_segments[0] in _PATH_BASED_FORMS:
        if len(path_segments) != 2:
            raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
        video_id = path_segments[1]
        if video_query_values and video_query_values[0] not in {"", video_id}:
            raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    else:
        raise InvalidSourceError(_INVALID_SOURCE_MESSAGE)

    return video_id


def _normalize_metadata(
    metadata: Mapping[str, object],
    video_id: str,
) -> tuple[str, str, str, int]:
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)

    description_value = metadata.get("description")
    if description_value is None:
        description = ""
    elif isinstance(description_value, str):
        description = description_value
    else:
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)

    channel_value = _optional_text(metadata, "channel")
    uploader_value = _optional_text(metadata, "uploader")
    channel = channel_value or uploader_value or ""

    if "id" in metadata and metadata["id"] != video_id:
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)

    duration_value = metadata.get("duration", _MISSING)
    if isinstance(duration_value, bool) or not isinstance(
        duration_value,
        (Real, Decimal),
    ):
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
    try:
        if duration_value < 0 or not math.isfinite(duration_value):
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
        duration_seconds = cast(int, math.ceil(duration_value))
    except (OverflowError, TypeError, ValueError) as exc:
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc

    return title, description, channel, duration_seconds


def _optional_text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
    return value


def _is_unavailable_error(message: str) -> bool:
    normalized_message = message.casefold()
    return any(marker in normalized_message for marker in _UNAVAILABLE_MARKERS)


def _safe_submitted_url(submitted_url: str) -> str:
    try:
        parsed_url = urlsplit(submitted_url)
        query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    except ValueError:
        return _REDACTED_LOG_VALUE

    has_sensitive_query = any(_is_sensitive_query_key(key) for key, _ in query_pairs)
    if not parsed_url.fragment and not has_sensitive_query:
        return submitted_url

    safe_pairs = [
        (key, _REDACTED_LOG_VALUE if _is_sensitive_query_key(key) else value)
        for key, value in query_pairs
    ]
    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            urlencode(safe_pairs),
            "",
        )
    )


def _is_sensitive_query_key(key: str) -> bool:
    normalized_key = key.casefold().replace("-", "_")
    return normalized_key in _SENSITIVE_QUERY_PARTS or any(
        part in normalized_key for part in ("api_key", "token", "secret", "password")
    )
