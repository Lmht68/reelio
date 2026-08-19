"""Inspect YouTube Sources and acquire Caption Transcripts."""

import asyncio
import logging
import math
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from numbers import Real
from typing import Final, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yt_dlp  # type: ignore[import-untyped]
from requests.exceptions import RequestException, Timeout
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    YouTubeTranscriptApi,
)
from yt_dlp.utils import YoutubeDLError  # type: ignore[import-untyped]

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


class _CaptionProviderFailure(Exception):
    """Represent an ordinary failure at the caption provider boundary."""


class _CaptionProviderTimeout(Exception):
    """Represent a timeout at the caption provider boundary."""


_CAPTION_EXTERNAL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    CouldNotRetrieveTranscript,
    ElementTree.ParseError,
    RequestException,
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
)
_CAPTION_TRACK_FAILURES: Final[tuple[type[BaseException], ...]] = (
    _CaptionProviderFailure,
    *_CAPTION_EXTERNAL_FAILURES,
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
        except Timeout as exc:
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
        except Timeout as exc:
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
    """Acquire a Transcript from an injected CaptionProvider."""

    def __init__(self, provider: CaptionProvider) -> None:
        """Initialize the service with a caption provider.

        Args:
            provider: Synchronous provider boundary for Caption Tracks.
        """
        self._provider = provider

    async def acquire(self, source: Source) -> Transcript:
        """Acquire the first usable Transcript for a validated Source.

        Args:
            source: Validated Source whose video ID identifies the provider data.

        Returns:
            Transcript: Normalized caption text and acquisition metadata.

        Raises:
            TranscriptionError: If no Caption Track produces usable text.
            PipelineTimeoutError: If listing or fetching captions times out.
        """
        try:
            transcript = await asyncio.to_thread(
                _acquire_transcript,
                self._provider,
                source.video_id,
            )
        except _CaptionProviderTimeout as exc:
            raise PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE) from exc
        except _CaptionProviderFailure as exc:
            raise TranscriptionError(_TRANSCRIPT_UNAVAILABLE_MESSAGE) from exc

        if transcript is None:
            raise TranscriptionError(_TRANSCRIPT_UNAVAILABLE_MESSAGE)
        return transcript


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


def _normalize_segments(segments: Sequence[str]) -> str:
    tokens: list[str] = []
    for segment in segments:
        tokens.extend(segment.split())
    return " ".join(tokens)


def _acquire_transcript(
    provider: CaptionProvider,
    video_id: str,
) -> Transcript | None:
    try:
        tracks = provider.list_tracks(video_id)
    except _CaptionProviderTimeout:
        raise
    except _CaptionProviderFailure:
        raise
    except Timeout as exc:
        raise _CaptionProviderTimeout from exc
    except _CAPTION_EXTERNAL_FAILURES as exc:
        raise _CaptionProviderFailure from exc

    for track in _rank_caption_tracks(tracks):
        try:
            segments = track.fetch_segments()
        except _CaptionProviderTimeout:
            raise
        except Timeout as exc:
            raise _CaptionProviderTimeout from exc
        except _CAPTION_TRACK_FAILURES:
            logger.debug(
                "caption track unavailable",
                extra={"stage": "transcription"},
            )
            continue

        segment_count = len(segments)
        transcript_text = _normalize_segments(segments)
        if not transcript_text:
            continue

        language = track.language_code
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
