"""Orchestrate Source inspection and Transcript acquisition."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError
from requests.exceptions import RequestException, Timeout
from yt_dlp.utils import YoutubeDLError

import reelio.extraction.services.transcription.acquisition as acquisition
import reelio.extraction.services.transcription.inspection as inspection
from reelio.cache import AsyncCache, CacheCodecError, CacheEntry
from reelio.cache.interface import JsonObject
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
from reelio.extraction.types import Platform, Source, SourceIdentity, Transcript

logger = logging.getLogger(__name__)

_SOURCE_UNAVAILABLE_MESSAGE = "Source is unavailable."
_METADATA_PROVIDER_MESSAGE = "Unable to retrieve source metadata."
_METADATA_TIMEOUT_MESSAGE = "Source metadata acquisition timed out."
_TRANSCRIPT_UNAVAILABLE_MESSAGE = "Transcript is unavailable for this video."
_TRANSCRIPT_TIMEOUT_MESSAGE = "Transcript acquisition timed out."


_SOURCE_CACHE_TTL_SECONDS = 2_592_000
_SOURCE_CACHE_WAIT_TIMEOUT_SECONDS = 1.0
_SOURCE_ALIAS_CONTRACT_VERSION = "source-alias-v1"
_SOURCE_METADATA_CONTRACT_VERSION = "source-metadata-v1"


class _SourceAliasPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    platform: str
    external_content_id: str


class _SourceMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    platform: str
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


class _SourceAliasCodec:
    """Serialize Source identities for one platform-scoped URL alias."""

    version = _SOURCE_ALIAS_CONTRACT_VERSION

    def __init__(self, expected_platform: Platform) -> None:
        self._expected_platform = expected_platform

    def encode(self, value: SourceIdentity) -> JsonObject:
        """Encode one platform-matching Source identity."""
        _validate_source_identity(value, self._expected_platform)
        payload = _SourceAliasPayload(
            platform=value.platform.value,
            external_content_id=value.external_content_id,
        )
        return {
            "platform": payload.platform,
            "external_content_id": payload.external_content_id,
        }

    def decode(self, payload: JsonObject) -> SourceIdentity:
        """Decode one exact platform-matching Source identity."""
        try:
            decoded = _SourceAliasPayload.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid Source alias cache value") from exc
        source_identity = SourceIdentity(
            platform=self._expected_platform,
            external_content_id=decoded.external_content_id,
        )
        _validate_source_identity(source_identity, self._expected_platform)
        if decoded.platform != self._expected_platform.value:
            raise CacheCodecError("Source alias platform does not match its cache key")
        return source_identity


class _SourceMetadataCodec:
    """Serialize normalized Source metadata for one expected Source identity."""

    version = _SOURCE_METADATA_CONTRACT_VERSION

    def __init__(self, expected_identity: SourceIdentity) -> None:
        _validate_source_identity(expected_identity, expected_identity.platform)
        self._expected_identity = expected_identity

    def encode(self, value: Source) -> JsonObject:
        """Encode one Source whose identity matches this cache key."""
        _validate_source(value, self._expected_identity)
        payload = _SourceMetadataPayload(
            platform=value.platform.value,
            video_id=value.video_id,
            url=value.url,
            title=value.title,
            description=value.description,
            channel=value.channel,
            duration_seconds=value.duration_seconds,
        )
        return {
            "platform": payload.platform,
            "video_id": payload.video_id,
            "url": payload.url,
            "title": payload.title,
            "description": payload.description,
            "channel": payload.channel,
            "duration_seconds": payload.duration_seconds,
        }

    def decode(self, payload: JsonObject) -> Source:
        """Decode one exact Source metadata payload into a new allocation."""
        try:
            decoded = _SourceMetadataPayload.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid Source metadata cache value") from exc
        source = Source(
            platform=self._expected_identity.platform,
            video_id=decoded.video_id,
            url=decoded.url,
            title=decoded.title,
            description=decoded.description,
            channel=decoded.channel,
            duration_seconds=decoded.duration_seconds,
        )
        if decoded.platform != self._expected_identity.platform.value:
            raise CacheCodecError("Source platform does not match its cache key")
        _validate_source(source, self._expected_identity)
        return source


def _validate_source_identity(
    source_identity: SourceIdentity,
    expected_platform: Platform,
) -> None:
    if (
        not isinstance(source_identity, SourceIdentity)
        or source_identity.platform is not expected_platform
    ):
        raise CacheCodecError("Source identity does not match its cache key")
    _validate_content_id(source_identity.external_content_id)


def _validate_source(source: Source, expected_identity: SourceIdentity) -> None:
    if not isinstance(source, Source):
        raise CacheCodecError("Expected Source cache value")
    if source.platform is not expected_identity.platform:
        raise CacheCodecError("Source platform does not match its cache key")
    _validate_content_id(source.video_id)
    if source.video_id != expected_identity.external_content_id:
        raise CacheCodecError("Source ID does not match its cache key")
    _validate_source_url(source.url, expected_identity.platform)
    for text in (source.title, source.description, source.channel):
        if not isinstance(text, str):
            raise CacheCodecError("Source text must be a string")
    if type(source.duration_seconds) is not int or source.duration_seconds < 0:
        raise CacheCodecError("Source duration must be a nonnegative integer")


def _source_identity(source: Source) -> SourceIdentity:
    """Return the cache identity represented by normalized Source metadata."""
    return SourceIdentity(source.platform, source.video_id)


def _validate_content_id(content_id: str) -> None:
    if (
        not isinstance(content_id, str)
        or not content_id.strip()
        or inspection._contains_control_character(content_id)
    ):
        raise CacheCodecError("Source ID must be nonblank and control-character-free")


def _validate_source_url(url: str, expected_platform: Platform) -> None:
    if not isinstance(url, str):
        raise CacheCodecError("Source URL must be a string")
    try:
        classified_url = inspection.classify_submitted_url(url)
    except (InvalidSourceError, UnsupportedPlatformError) as exc:
        raise CacheCodecError("Source URL is not a valid platform URL") from exc
    if classified_url.platform is not expected_platform:
        raise CacheCodecError("Source URL platform does not match its cache key")


def _source_alias_entry(
    platform: Platform,
    normalized_url: str,
) -> CacheEntry[SourceIdentity]:
    return CacheEntry(
        layer="source:alias",
        key_version="v1",
        identity={
            "platform": platform.value,
            "normalized_url": normalized_url,
            "contract_version": _SOURCE_ALIAS_CONTRACT_VERSION,
        },
        codec=_SourceAliasCodec(platform),
        ttl_seconds=_source_alias_ttl_seconds,
        wait_timeout_seconds=_SOURCE_CACHE_WAIT_TIMEOUT_SECONDS,
    )


def _source_metadata_entry(source_identity: SourceIdentity) -> CacheEntry[Source]:
    return CacheEntry(
        layer="source:metadata",
        key_version="v1",
        identity={
            "platform": source_identity.platform.value,
            "external_content_id": source_identity.external_content_id,
            "contract_version": _SOURCE_METADATA_CONTRACT_VERSION,
        },
        codec=_SourceMetadataCodec(source_identity),
        ttl_seconds=_source_metadata_ttl_seconds,
        wait_timeout_seconds=_SOURCE_CACHE_WAIT_TIMEOUT_SECONDS,
    )


def _source_alias_ttl_seconds(_: SourceIdentity) -> int:
    return _SOURCE_CACHE_TTL_SECONDS


def _source_metadata_ttl_seconds(_: Source) -> int:
    return _SOURCE_CACHE_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class InspectedSource:
    """Contain a normalized Source and optional inspection-stage audio."""

    source: Source
    prepared_audio: inspection.PreparedAudio | None = None

    def cleanup(self) -> None:
        """Delete temporary audio created while inspecting the Source."""
        if self.prepared_audio is not None:
            self.prepared_audio.cleanup()


@dataclass(slots=True)
class _InspectionOwnership:
    """Own inspection-stage audio until a successful Source return transfers it."""

    inspected_source: InspectedSource | None = None

    def capture(self, inspected_source: InspectedSource) -> None:
        """Retain one current-request inspection and its optional prepared audio."""
        if self.inspected_source is not None:
            inspected_source.cleanup()
            raise RuntimeError("Source inspection ran more than once for one request")
        self.inspected_source = inspected_source

    def cleanup(self) -> None:
        """Release currently owned prepared audio after an orchestration failure."""
        if self.inspected_source is not None:
            self.inspected_source.cleanup()
            self.inspected_source = None

    def transfer(self, source: Source) -> InspectedSource:
        """Transfer only current-request prepared audio with a resolved Source."""
        prepared_audio = (
            None if self.inspected_source is None else self.inspected_source.prepared_audio
        )
        self.inspected_source = None
        return InspectedSource(source, prepared_audio)


class SourceMetadataService:
    """Inspect one submitted Source URL and normalize provider metadata."""

    def __init__(
        self,
        extractor: inspection.MetadataExtractor,
        settings: TranscriptionConfig,
        cache: AsyncCache,
    ) -> None:
        """Initialize the service with provider, cache, and policy dependencies.

        Args:
            extractor: Synchronous provider adapter for metadata retrieval.
            settings: Transcription settings containing the duration limit.
            cache: Shared Source metadata cache borrowed from the application lifespan.
        """
        self._extractor = extractor
        self._settings = settings
        self._cache = cache

    async def inspect(self, submitted_url: str) -> InspectedSource:
        """Validate a URL, retrieve metadata, enforce duration, and return it.

        Args:
            submitted_url: URL submitted by the API caller.

        Returns:
            InspectedSource: Canonical Source metadata and optional prepared audio.

        Raises:
            inspection.InvalidSourceError: If the URL or processed shape is invalid.
            inspection.UnsupportedPlatformError: If the URL uses another host.
            SourceUnavailableError: If the provider reports inaccessible content.
            MetadataProviderError: If provider access or metadata is malformed.
            DurationLimitExceededError: If the Source exceeds the configured limit.
            PipelineTimeoutError: If typed metadata access times out.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        ownership = _InspectionOwnership()
        try:
            if submitted.platform is Platform.YOUTUBE:
                source = await self._resolve_youtube_source(submitted, ownership)
            else:
                source = await self._resolve_social_source(submitted, ownership)
            inspected_source = ownership.inspected_source
            if submitted.platform is not Platform.YOUTUBE and inspected_source is not None:
                await self._register_social_aliases(submitted, inspected_source.source)
            self._ensure_duration_within_limit(source)
            return ownership.transfer(source)
        except BaseException:
            ownership.cleanup()
            raise

    async def _resolve_youtube_source(
        self,
        submitted: inspection.SubmittedSource,
        ownership: _InspectionOwnership,
    ) -> Source:
        """Load one local YouTube identity directly from Source metadata."""
        local_video_id = submitted.local_youtube_video_id
        if local_video_id is None:
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
        expected_identity = SourceIdentity(Platform.YOUTUBE, local_video_id)

        async def loader() -> Source:
            return await self._capture_provider_inspection(
                submitted,
                ownership,
                expected_identity,
            )

        return await self._cache.get_or_load(_source_metadata_entry(expected_identity), loader)

    async def _resolve_social_source(
        self,
        submitted: inspection.SubmittedSource,
        ownership: _InspectionOwnership,
    ) -> Source:
        """Resolve a provider-authoritative social Source through a URL alias."""
        source_from_alias_fill: Source | None = None

        async def alias_loader() -> SourceIdentity:
            nonlocal source_from_alias_fill
            inspected_source = await self._capture_provider_inspection(
                submitted,
                ownership,
            )
            source_identity = _source_identity(inspected_source)

            async def metadata_loader() -> Source:
                return inspected_source

            source_from_alias_fill = await self._cache.get_or_load(
                _source_metadata_entry(source_identity),
                metadata_loader,
            )
            return source_identity

        source_identity = await self._cache.get_or_load(
            _source_alias_entry(submitted.platform, submitted.normalized_url),
            alias_loader,
        )
        if source_from_alias_fill is not None:
            return source_from_alias_fill
        return await self._load_source_for_alias(
            source_identity,
            submitted,
            ownership,
        )

    async def _load_source_for_alias(
        self,
        alias_identity: SourceIdentity,
        submitted: inspection.SubmittedSource,
        ownership: _InspectionOwnership,
    ) -> Source:
        """Load metadata for an alias and recover from stale identity hints."""

        async def loader() -> Source:
            inspected_source = await self._capture_provider_inspection(
                submitted,
                ownership,
            )
            inspected_identity = _source_identity(inspected_source)
            if inspected_identity == alias_identity:
                return inspected_source

            async def actual_metadata_loader() -> Source:
                return inspected_source

            return await self._cache.get_or_load(
                _source_metadata_entry(inspected_identity),
                actual_metadata_loader,
            )

        return await self._cache.get_or_load(_source_metadata_entry(alias_identity), loader)

    async def _capture_provider_inspection(
        self,
        submitted: inspection.SubmittedSource,
        ownership: _InspectionOwnership,
        expected_identity: SourceIdentity | None = None,
    ) -> Source:
        """Inspect once and retain only this request's prepared audio."""
        inspected_source = await self._inspect_provider(submitted)
        inspected_identity = _source_identity(inspected_source.source)
        if expected_identity is not None and inspected_identity != expected_identity:
            inspected_source.cleanup()
            raise MetadataProviderError(_METADATA_PROVIDER_MESSAGE)
        ownership.capture(inspected_source)
        return inspected_source.source

    def _ensure_duration_within_limit(self, source: Source) -> None:
        """Enforce the current request duration policy for a normalized Source."""
        if source.duration_seconds <= self._settings.max_video_duration_seconds:
            return
        raise DurationLimitExceededError(
            "Video exceeds the configured duration limit of "
            f"{self._settings.max_video_duration_seconds} seconds."
        )

    async def _register_social_aliases(
        self,
        submitted: inspection.SubmittedSource,
        source: Source,
    ) -> None:
        """Register this inspection's submitted and canonical social URL aliases."""
        source_identity = _source_identity(source)
        try:
            canonical_submitted = inspection.classify_submitted_url(source.url)
        except (InvalidSourceError, UnsupportedPlatformError):
            self._log_alias_registration(source_identity.platform, "canonical_invalid")
            return
        if canonical_submitted.platform is not source_identity.platform:
            self._log_alias_registration(source_identity.platform, "canonical_platform_mismatch")
            return

        normalized_aliases = [submitted.normalized_url]
        if canonical_submitted.normalized_url != submitted.normalized_url:
            normalized_aliases.append(canonical_submitted.normalized_url)

        async def register_alias(normalized_url: str) -> SourceIdentity:
            async def loader() -> SourceIdentity:
                return source_identity

            return await self._cache.get_or_load(
                _source_alias_entry(source_identity.platform, normalized_url),
                loader,
            )

        registered_identities = await asyncio.gather(
            *(register_alias(normalized_url) for normalized_url in normalized_aliases)
        )
        for registered_identity in registered_identities:
            outcome = "registered" if registered_identity == source_identity else "conflict"
            self._log_alias_registration(source_identity.platform, outcome)

    def _log_alias_registration(self, platform: Platform, outcome: str) -> None:
        """Emit Source alias registration telemetry without identity material."""
        logger.debug(
            "source alias registration",
            extra={
                "stage": "transcription",
                "platform": platform.value,
                "outcome": outcome,
            },
        )

    async def _inspect_provider(
        self,
        submitted: inspection.SubmittedSource,
    ) -> InspectedSource:
        """Retrieve, normalize, and validate one provider Source inspection."""
        try:
            extracted_metadata = await asyncio.to_thread(
                self._extractor.extract,
                submitted.provider_url,
            )
        except inspection._MetadataDurationLimitExceeded as exc:
            raise DurationLimitExceededError(
                "Video exceeds the configured duration limit of "
                f"{self._settings.max_video_duration_seconds} seconds."
            ) from exc
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

        try:
            normalized = inspection.normalize_processed_metadata(
                extracted_metadata.metadata,
                submitted,
            )
        except (InvalidSourceError, MetadataProviderError):
            if extracted_metadata.prepared_audio is not None:
                extracted_metadata.prepared_audio.cleanup()
            raise
        source = Source(
            platform=normalized.source_identity.platform,
            video_id=normalized.source_identity.external_content_id,
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
                "platform": source.platform.value,
                "title_length": len(source.title),
                "description_length": len(source.description),
                "channel_length": len(source.channel),
                "duration_seconds": source.duration_seconds,
            },
        )

        if source.duration_seconds > self._settings.max_video_duration_seconds:
            if extracted_metadata.prepared_audio is not None:
                extracted_metadata.prepared_audio.cleanup()
            raise DurationLimitExceededError(
                "Video exceeds the configured duration limit of "
                f"{self._settings.max_video_duration_seconds} seconds."
            )
        return InspectedSource(source, extracted_metadata.prepared_audio)


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

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: inspection.PreparedAudio | None = None,
    ) -> Transcript:
        """Acquire a normalized Transcript for a validated Source.

        Args:
            source: Validated Source whose identity identifies provider data.
            submitted_url: Validated URL supplied by the API caller.
            prepared_audio: Audio downloaded during metadata inspection, when available.

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

        download_url = (
            submitted_url if source.platform is Platform.TIKTOK else source.url
        )  # TikTok canonical URL does not work with yt-dlp audio download
        try:
            return await acquisition.acquire_whisper(
                download_url,
                self._audio_downloader,
                self._transcriber,
                self._temp_media_dir,
                prepared_audio,
                self._semaphore,
            )
        except acquisition._WhisperProviderTimeout as exc:
            raise PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE) from exc
        except acquisition._WhisperProviderFailure as exc:
            raise TranscriptionError(_TRANSCRIPT_UNAVAILABLE_MESSAGE) from exc
