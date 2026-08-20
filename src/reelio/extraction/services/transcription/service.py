"""Orchestrate Source inspection and Transcript acquisition."""

import asyncio
import logging
from pathlib import Path

from requests.exceptions import RequestException, Timeout
from yt_dlp.utils import YoutubeDLError  # type: ignore[import-untyped]

import reelio.extraction.services.transcription.acquisition as acquisition
import reelio.extraction.services.transcription.inspection as inspection
from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.types import Platform, Source, Transcript

logger = logging.getLogger(__name__)

_SOURCE_UNAVAILABLE_MESSAGE = "Source is unavailable."
_METADATA_PROVIDER_MESSAGE = "Unable to retrieve source metadata."
_METADATA_TIMEOUT_MESSAGE = "Source metadata acquisition timed out."
_TRANSCRIPT_UNAVAILABLE_MESSAGE = "Transcript is unavailable for this video."
_TRANSCRIPT_TIMEOUT_MESSAGE = "Transcript acquisition timed out."
_REDACTED_LOG_VALUE = "[REDACTED]"


class SourceMetadataService:
    """Inspect one submitted Source URL and normalize provider metadata."""

    def __init__(
        self,
        extractor: inspection.MetadataExtractor,
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
            Source: Canonical identity and normalized metadata.

        Raises:
            inspection.InvalidSourceError: If the URL or processed shape is invalid.
            inspection.UnsupportedPlatformError: If the URL uses another host.
            SourceUnavailableError: If the provider reports inaccessible content.
            MetadataProviderError: If provider access or metadata is malformed.
            DurationLimitExceededError: If the Source exceeds the configured limit.
            PipelineTimeoutError: If typed metadata access times out.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        try:
            raw_metadata = await asyncio.to_thread(
                self._extractor.extract,
                submitted.provider_url,
            )
        except MetadataProviderError as exc:
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc
        except (Timeout, TimeoutError) as exc:
            raise PipelineTimeoutError(_METADATA_TIMEOUT_MESSAGE) from exc
        except RequestException as exc:
            if inspection._is_timeout_exception(exc):
                raise PipelineTimeoutError(_METADATA_TIMEOUT_MESSAGE) from exc
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc
        except YoutubeDLError as exc:
            if inspection._is_timeout_exception(exc):
                raise PipelineTimeoutError(_METADATA_TIMEOUT_MESSAGE) from exc
            if inspection._is_unavailable_error(str(exc)):
                raise SourceUnavailableError(_SOURCE_UNAVAILABLE_MESSAGE) from exc
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc
        except (
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if inspection._is_timeout_exception(exc):
                raise PipelineTimeoutError(_METADATA_TIMEOUT_MESSAGE) from exc
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE) from exc

        normalized = inspection.normalize_processed_metadata(
            raw_metadata,
            submitted,
        )
        source = Source(
            platform=submitted.platform,
            video_id=normalized.video_id,
            url=normalized.canonical_url,
            title=normalized.title,
            description=normalized.description,
            channel=normalized.channel,
            duration_seconds=normalized.duration_seconds,
        )
        logger.debug(
            "source metadata normalized",
            extra={
                "stage": "transcription",
                "submitted_url": inspection.safe_submitted_url(submitted_url),
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

        if source.duration_seconds > self._settings.max_video_duration_seconds:
            raise DurationLimitExceededError(
                "Video exceeds the configured duration limit of "
                f"{self._settings.max_video_duration_seconds} seconds."
            )
        return source


class TranscriptionService:
    """Acquire Caption or Whisper Transcripts for validated Sources."""

    def __init__(
        self,
        provider: acquisition.CaptionProvider,
        audio_downloader: acquisition.AudioDownloader,
        transcriber: acquisition.WhisperTranscriber,
        temp_media_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize caption and Whisper acquisition dependencies.

        Args:
            provider: Synchronous provider boundary for YouTube Caption Tracks.
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
            source: Validated Source whose identity identifies provider data.

        Returns:
            Transcript: Caption or Whisper text and acquisition metadata.

        Raises:
            TranscriptionError: If no usable Transcript can be acquired.
            PipelineTimeoutError: If the terminal Whisper path times out.
        """
        if source.platform is Platform.YOUTUBE:
            try:
                transcript = await asyncio.to_thread(
                    acquisition.acquire_transcript,
                    self._provider,
                    source.video_id,
                )
            except (
                acquisition._CaptionProviderFailure,
                acquisition._CaptionProviderTimeout,
            ):
                transcript = None
            if transcript is not None:
                return transcript

        try:
            return await acquisition.acquire_whisper(
                source,
                self._audio_downloader,
                self._transcriber,
                self._temp_media_dir,
                self._semaphore,
            )
        except acquisition._WhisperProviderTimeout as exc:
            raise PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE) from exc
        except acquisition._WhisperProviderFailure as exc:
            raise TranscriptionError(_TRANSCRIPT_UNAVAILABLE_MESSAGE) from exc
