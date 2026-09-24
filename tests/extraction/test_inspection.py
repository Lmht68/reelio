"""Deterministic tests for source validation and metadata acquisition."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import wave
from collections.abc import Callable, Mapping
from decimal import Decimal
from itertools import count
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
import yt_dlp
from requests.exceptions import Timeout
from yt_dlp.utils import DownloadError, YoutubeDLError

import reelio.extraction.services.transcription.inspection as transcription_inspection
from reelio.cache import CacheEntry, DisabledCache, RedisCache
from reelio.cache.redis import _CacheRuntime
from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    UnsupportedPlatformError,
)
from reelio.extraction.services.transcription import service as transcription_service
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.inspection import (
    ExtractedMetadata,
    PreparedAudio,
    YtDlpMetadataExtractor,
)
from reelio.extraction.services.transcription.service import SourceMetadataService
from reelio.extraction.types import Platform, Source, SourceIdentity
from tests.cache.fakes import FakeRedis, ManualClock

_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _FakeExtractor:
    def __init__(
        self,
        metadata: object,
        error: Exception | None = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> None:
        self.metadata = metadata
        self.error = error
        self.prepared_audio = prepared_audio
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        if self.error is not None:
            raise self.error
        return ExtractedMetadata(
            cast(Mapping[str, object], self.metadata),
            self.prepared_audio,
        )


def _metadata(**overrides: object) -> dict[str, object]:
    metadata = {
        "id": _VIDEO_ID,
        "title": "Example video",
        "description": "A complete description.",
        "channel": "Example channel",
        "duration": 12.0,
    }
    metadata.update(overrides)
    return metadata


def _settings(max_duration: int = 1800) -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(
        _env_file=None,
        max_video_duration_seconds=max_duration,
    )


def _service(
    extractor: _FakeExtractor,
    max_duration: int = 1800,
) -> SourceMetadataService:
    return SourceMetadataService(
        extractor=extractor,
        settings=_settings(max_duration),
        cache=DisabledCache(),
    )


class _SequenceExtractor:
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
        return ExtractedMetadata(cast(Mapping[str, object], outcome))


def _cached_service(
    extractor: transcription_inspection.MetadataExtractor,
    cache: RedisCache,
    max_duration: int = 1800,
) -> SourceMetadataService:
    return SourceMetadataService(
        extractor=extractor,
        settings=_settings(max_duration),
        cache=cache,
    )


def _redis_source_cache(clock: ManualClock) -> tuple[RedisCache, FakeRedis]:
    """Build one deterministic shared cache for Source-layer tests."""
    tokens = count()
    redis = FakeRedis(clock)
    cache = RedisCache(
        redis,
        "reelio:test",
        b"source-cache-test-key",
        runtime=_CacheRuntime(clock, asyncio.sleep, lambda: f"source-{next(tokens)}"),
    )
    return cache, redis


def _cached_value(
    cache: RedisCache,
    redis: FakeRedis,
    entry: CacheEntry[Source],
) -> dict[str, object]:
    cache_key = cache._cache_key(entry)
    raw_value = redis.raw_value(cache_key)
    assert raw_value is not None
    return cast(dict[str, object], json.loads(raw_value)["value"])


@pytest.mark.parametrize(
    "submitted_url",
    [
        _CANONICAL_URL,
        f"https://youtube.com/watch?v={_VIDEO_ID}",
        f"https://m.youtube.com/watch?v={_VIDEO_ID}",
        f"https://music.youtube.com/watch?v={_VIDEO_ID}",
        f"https://www.youtube.com/shorts/{_VIDEO_ID}",
        f"https://www.youtube.com/embed/{_VIDEO_ID}",
        f"https://www.youtube.com/live/{_VIDEO_ID}",
        f"https://youtu.be/{_VIDEO_ID}",
        f"https://www.youtube.com/watch?v={_VIDEO_ID}&list=playlist&si=tracking#fragment",
        f"https://www.youtube.com/shorts/{_VIDEO_ID}/?v={_VIDEO_ID}&foo=bar#fragment",
    ],
)
async def test_youtube_url_forms_normalize_to_one_source_identity(
    submitted_url: str,
) -> None:
    """Normalize every accepted YouTube URL form to one canonical identity."""
    extractor = _FakeExtractor(_metadata())

    source = (await _service(extractor).inspect(submitted_url)).source

    assert source.platform is Platform.YOUTUBE
    assert source.video_id == _VIDEO_ID
    assert source.url == _CANONICAL_URL
    assert extractor.calls == [_CANONICAL_URL]


@pytest.mark.parametrize(
    ("submitted_url", "exception_type"),
    [
        (f"http://www.youtube.com/watch?v={_VIDEO_ID}", InvalidSourceError),
        (f"www.youtube.com/watch?v={_VIDEO_ID}", InvalidSourceError),
        ("https://", InvalidSourceError),
        (
            f"https://user:password@www.youtube.com/watch?v={_VIDEO_ID}",
            InvalidSourceError,
        ),
        (
            f"https://www.youtube.com:443/watch?v={_VIDEO_ID}",
            InvalidSourceError,
        ),
        (
            f"https://www.youtube.com.evil.example/watch?v={_VIDEO_ID}",
            InvalidSourceError,
        ),
        ("https://www.youtube.com/watch?v=short", InvalidSourceError),
        ("https://www.youtube.com/watch?v=invalid!id", InvalidSourceError),
        (f"https://www.youtube.com/watch?V={_VIDEO_ID}", InvalidSourceError),
        ("https://www.youtube.com/watch", InvalidSourceError),
        (
            f"https://www.youtube.com/watch?v={_VIDEO_ID}&v={_VIDEO_ID}",
            InvalidSourceError,
        ),
        (
            f"https://www.youtube.com/shorts/{_VIDEO_ID}?v=otherid12345",
            InvalidSourceError,
        ),
        (
            f"https://youtu.be/{_VIDEO_ID}?v=otherid12345",
            InvalidSourceError,
        ),
        (
            f"https://www.youtube.com/shorts/{_VIDEO_ID}/extra",
            InvalidSourceError,
        ),
        ("https://www.youtube.com/playlist?list=playlist", InvalidSourceError),
        ("https://www.youtube.com/channel/channel-id", InvalidSourceError),
        ("https://www.youtube.com/c/channel-name", InvalidSourceError),
        ("https://www.youtube.com/user/user-name", InvalidSourceError),
        ("https://www.youtube.com/@channel", InvalidSourceError),
        ("https://www.youtube.com/feed/subscriptions", InvalidSourceError),
        (f"https://www.youtube.com/shorts/%2F{_VIDEO_ID}", InvalidSourceError),
        (f"https://www.youtube.com/watch?v={_VIDEO_ID}%ZZ", InvalidSourceError),
    ],
)
async def test_invalid_youtube_urls_raise_invalid_source(
    submitted_url: str,
    exception_type: type[InvalidSourceError],
) -> None:
    """Reject malformed or non-video YouTube URLs before provider access."""
    extractor = _FakeExtractor(_metadata())

    with pytest.raises(exception_type, match="Invalid source URL"):
        await _service(extractor).inspect(submitted_url)

    assert extractor.calls == []


@pytest.mark.parametrize(
    "submitted_url",
    [
        "https://vimeo.com/123456789",
        "https://[2001:db8::1]/video",
    ],
)
async def test_non_youtube_hosts_raise_unsupported_platform(
    submitted_url: str,
) -> None:
    """Classify syntactically valid non-YouTube hosts as unsupported."""
    extractor = _FakeExtractor(_metadata())

    with pytest.raises(
        UnsupportedPlatformError,
        match="Only YouTube, Instagram, Facebook, TikTok, and X URLs are supported",
    ):
        await _service(extractor).inspect(submitted_url)

    assert extractor.calls == []


async def test_metadata_fields_are_normalized_into_source() -> None:
    """Map provider metadata to the complete canonical Source contract."""
    description = "A complete description that must not be truncated."
    extractor = _FakeExtractor(
        _metadata(
            title="A real title",
            description=description,
            channel="Preferred channel",
            uploader="Fallback uploader",
            duration=12.1,
        )
    )

    source = (await _service(extractor).inspect(_CANONICAL_URL)).source

    assert source.platform is Platform.YOUTUBE
    assert source.video_id == _VIDEO_ID
    assert source.url == _CANONICAL_URL
    assert source.title == "A real title"
    assert source.description == description
    assert source.channel == "Preferred channel"
    assert source.duration_seconds == 13


@pytest.mark.parametrize(
    ("metadata_overrides", "expected_description", "expected_channel"),
    [
        ({"description": None, "channel": None, "uploader": None}, "", ""),
        (
            {"channel": "", "uploader": "Uploader channel"},
            "A complete description.",
            "Uploader channel",
        ),
    ],
)
async def test_missing_metadata_uses_empty_or_fallback_values(
    metadata_overrides: dict[str, object],
    expected_description: str,
    expected_channel: str,
) -> None:
    """Use documented fallback values for optional provider metadata."""
    extractor = _FakeExtractor(_metadata(**metadata_overrides))

    source = (await _service(extractor).inspect(_CANONICAL_URL)).source

    assert source.description == expected_description
    assert source.channel == expected_channel


async def test_provider_id_is_optional_when_url_identity_is_valid() -> None:
    """Treat the validated URL ID as authoritative when provider ID is absent."""
    metadata = _metadata()
    del metadata["id"]
    extractor = _FakeExtractor(metadata)

    source = (await _service(extractor).inspect(_CANONICAL_URL)).source

    assert source.video_id == _VIDEO_ID


async def test_youtube_preserves_finite_live_status_metadata() -> None:
    """Keep legacy YouTube metadata behavior outside social finite checks."""
    extractor = _FakeExtractor(_metadata(live_status="was_live"))

    source = (await _service(extractor).inspect(_CANONICAL_URL)).source

    assert source.platform is Platform.YOUTUBE
    assert source.video_id == _VIDEO_ID


@pytest.mark.parametrize(
    "provider_metadata",
    [
        _metadata(id=None),
        _metadata(id="differentid"),
        _metadata(title=""),
        _metadata(title=123),
        _metadata(description=123),
        _metadata(channel=123),
        _metadata(uploader=123),
        _metadata(duration=None),
        _metadata(duration=-1),
        _metadata(duration=math.nan),
        _metadata(duration=math.inf),
        _metadata(duration=True),
        _metadata(duration="12"),
        ["not metadata"],
    ],
)
async def test_invalid_provider_metadata_maps_to_stable_error(
    provider_metadata: object,
) -> None:
    """Hide malformed provider metadata behind the stable 502 domain error."""
    extractor = _FakeExtractor(provider_metadata)

    with pytest.raises(
        MetadataProviderError,
        match="Unable to retrieve source metadata",
    ) as error:
        await _service(extractor).inspect(_CANONICAL_URL)

    assert "differentid" not in str(error.value)
    assert "not metadata" not in str(error.value)


async def test_missing_duration_maps_to_metadata_provider_error() -> None:
    """Reject provider results that omit the required duration."""
    metadata = _metadata()
    del metadata["duration"]
    extractor = _FakeExtractor(metadata)

    with pytest.raises(MetadataProviderError, match="Unable to retrieve source metadata"):
        await _service(extractor).inspect(_CANONICAL_URL)


async def test_duration_equal_to_limit_is_allowed() -> None:
    """Allow a video whose ceiled duration exactly equals the configured limit."""
    extractor = _FakeExtractor(_metadata(duration=1800.0))

    source = (await _service(extractor, max_duration=1800).inspect(_CANONICAL_URL)).source

    assert source.duration_seconds == 1800


async def test_decimal_duration_preserves_fraction_before_ceiling() -> None:
    """Ceil precise numeric durations without float precision loss."""
    extractor = _FakeExtractor(_metadata(duration=Decimal("1800.0000000000000001")))

    with pytest.raises(DurationLimitExceededError):
        await _service(extractor, max_duration=1800).inspect(_CANONICAL_URL)


@pytest.mark.parametrize("duration", [1800.1, 1801.0])
async def test_duration_above_limit_is_rejected_after_metadata(
    duration: float,
) -> None:
    """Reject over-limit videos after one metadata call and before later stages."""
    extractor = _FakeExtractor(_metadata(duration=duration))

    with pytest.raises(
        DurationLimitExceededError,
        match="Video exceeds the configured duration limit of 1800 seconds",
    ):
        await _service(extractor, max_duration=1800).inspect(_CANONICAL_URL)

    assert extractor.calls == [_CANONICAL_URL]


@pytest.mark.parametrize(
    "provider_message",
    [
        "Private video. Sign in if you've been granted access.",
        "Video unavailable. This video has been removed.",
        "This video is not available in your country.",
        "This video is age-restricted. Sign in to confirm your age.",
        "This content is geo-restricted.",
        "Login required to view this video.",
        "The video has been deleted by the uploader.",
    ],
)
async def test_unavailable_provider_failures_map_to_not_found(
    provider_message: str,
) -> None:
    """Classify inaccessible YouTube content as a stable 404 error."""
    extractor = _FakeExtractor(
        _metadata(),
        error=YoutubeDLError(provider_message),
    )

    with pytest.raises(
        SourceUnavailableError,
        match="Source is unavailable",
    ) as error:
        await _service(extractor).inspect(_CANONICAL_URL)

    assert provider_message not in str(error.value)


async def test_unknown_provider_failure_maps_to_stable_bad_gateway() -> None:
    """Classify unknown yt-dlp failures without exposing provider details."""
    provider_message = "unexpected provider failure with api-key=secret-value"
    extractor = _FakeExtractor(
        _metadata(),
        error=YoutubeDLError(provider_message),
    )

    with pytest.raises(
        MetadataProviderError,
        match="Unable to retrieve source metadata",
    ) as error:
        await _service(extractor).inspect(_CANONICAL_URL)

    assert provider_message not in str(error.value)


class _FakeYoutubeDL:
    def __init__(self, options: object, result: object) -> None:
        self.options = options
        self.result = result
        self.extract_calls: list[tuple[str, bool]] = []
        self.media_download_calls = 0

    def __enter__(self) -> _FakeYoutubeDL:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    def extract_info(self, canonical_url: str, *, download: bool) -> object:
        self.extract_calls.append((canonical_url, download))
        return self.result


class _RetryingFakeYoutubeDL(_FakeYoutubeDL):
    def __init__(self, outcomes: list[object]) -> None:
        super().__init__({}, outcomes[-1])
        self._outcomes = iter(outcomes)

    def extract_info(self, canonical_url: str, *, download: bool) -> object:
        self.extract_calls.append((canonical_url, download))
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _patch_youtube_dl(
    monkeypatch: pytest.MonkeyPatch,
    fake_youtube_dl: _FakeYoutubeDL,
) -> None:
    def make_youtube_dl(options: object) -> _FakeYoutubeDL:
        fake_youtube_dl.options = options
        return fake_youtube_dl

    monkeypatch.setattr(yt_dlp, "YoutubeDL", make_youtube_dl)


def test_yt_dlp_adapter_uses_metadata_only_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass the canonical URL and metadata-only options to yt-dlp exactly once."""
    fake_youtube_dl = _FakeYoutubeDL({}, _metadata())
    _patch_youtube_dl(monkeypatch, fake_youtube_dl)

    metadata = YtDlpMetadataExtractor(1800, tmp_path).extract(_CANONICAL_URL)

    assert metadata.metadata["title"] == "Example video"
    assert fake_youtube_dl.options == {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": False,
        "ignoreconfig": True,
    }
    assert fake_youtube_dl.extract_calls == [(_CANONICAL_URL, False)]
    assert fake_youtube_dl.media_download_calls == 0


def test_yt_dlp_adapter_retries_only_metadata_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retry transient yt-dlp retrieval failures without repeating validation."""
    fake_youtube_dl = _RetryingFakeYoutubeDL(
        [YoutubeDLError("transient failure")] * 4 + [_metadata()]
    )
    _patch_youtube_dl(monkeypatch, fake_youtube_dl)

    metadata = YtDlpMetadataExtractor(1800, tmp_path).extract(_CANONICAL_URL)

    assert metadata.metadata["title"] == "Example video"
    assert fake_youtube_dl.extract_calls == [(_CANONICAL_URL, False)] * 5


def test_yt_dlp_adapter_rejects_non_mapping_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a provider response that cannot supply named metadata fields."""
    fake_youtube_dl = _FakeYoutubeDL({}, ["not metadata"])
    _patch_youtube_dl(monkeypatch, fake_youtube_dl)

    with pytest.raises(MetadataProviderError, match="Unable to retrieve source metadata"):
        YtDlpMetadataExtractor(1800, tmp_path).extract(_CANONICAL_URL)
    assert fake_youtube_dl.extract_calls == [(_CANONICAL_URL, False)]


@pytest.mark.parametrize(
    ("audio_duration", "expected_duration"),
    [(60.0, 60), (61.0, None)],
)
async def test_missing_metadata_duration_uses_probed_audio_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    audio_duration: float,
    expected_duration: int | None,
) -> None:
    """Use the downloaded audio duration when both yt-dlp results omit it."""
    metadata_without_duration = _metadata(duration=None)
    downloaded_metadata = dict(metadata_without_duration)
    del downloaded_metadata["duration"]
    metadata_youtube_dl = _FakeYoutubeDL({}, metadata_without_duration)

    class _AudioYoutubeDL:
        def __init__(self, options: object) -> None:
            self.options = cast(dict[str, object], options)
            self.extract_calls: list[tuple[str, bool]] = []
            self.prepared_path = Path(cast(str, self.options["outtmpl"]).replace("%(ext)s", "m4a"))

        def __enter__(self) -> _AudioYoutubeDL:
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def extract_info(self, canonical_url: str, *, download: bool) -> object:
            self.extract_calls.append((canonical_url, download))
            match_filter = cast(
                Callable[[dict[str, object]], str | None],
                self.options["match_filter"],
            )
            assert match_filter(downloaded_metadata) is None
            self.prepared_path.write_bytes(b"audio")
            return downloaded_metadata

        def prepare_filename(self, metadata: object) -> Path:
            return self.prepared_path

    audio_youtube_dl: _AudioYoutubeDL | None = None

    def make_youtube_dl(options: object) -> _FakeYoutubeDL | _AudioYoutubeDL:
        nonlocal audio_youtube_dl
        if isinstance(options, dict) and "match_filter" in options:
            audio_youtube_dl = _AudioYoutubeDL(options)
            return audio_youtube_dl
        metadata_youtube_dl.options = options
        return metadata_youtube_dl

    monkeypatch.setattr(yt_dlp, "YoutubeDL", make_youtube_dl)
    monkeypatch.setattr(
        transcription_inspection,
        "_probe_audio_duration",
        lambda audio_path: audio_duration,
        raising=False,
    )

    metadata_service = SourceMetadataService(
        YtDlpMetadataExtractor(60, tmp_path),
        _settings(60),
        DisabledCache(),
    )
    if expected_duration is None:
        with pytest.raises(DurationLimitExceededError):
            await metadata_service.inspect(_CANONICAL_URL)
        assert list(tmp_path.iterdir()) == []
        return

    inspected = await metadata_service.inspect(_CANONICAL_URL)

    assert inspected.source.duration_seconds == expected_duration
    assert metadata_youtube_dl.extract_calls == [(_CANONICAL_URL, False)]
    assert audio_youtube_dl is not None
    assert audio_youtube_dl.extract_calls == [(_CANONICAL_URL, True)]
    match_filter = cast(
        Callable[[dict[str, object]], str | None],
        audio_youtube_dl.options["match_filter"],
    )
    with pytest.raises(yt_dlp.utils.DownloadCancelled):
        match_filter({"duration": 61})
    inspected.cleanup()


async def test_duration_filter_cancellation_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map the first duration-filter cancellation to the configured limit."""
    metadata_without_duration = _metadata()
    del metadata_without_duration["duration"]
    metadata_youtube_dl = _FakeYoutubeDL({}, metadata_without_duration)

    class _DurationLimitYoutubeDL:
        def __init__(self, options: object) -> None:
            self.options = cast(dict[str, object], options)
            self.extract_calls: list[tuple[str, bool]] = []

        def __enter__(self) -> _DurationLimitYoutubeDL:
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def extract_info(self, canonical_url: str, *, download: bool) -> object:
            self.extract_calls.append((canonical_url, download))
            match_filter = cast(
                Callable[[dict[str, object]], str | None],
                self.options["match_filter"],
            )
            match_filter({"duration": 61})
            raise AssertionError("duration filter did not cancel yt-dlp")

    duration_limit_youtube_dl: _DurationLimitYoutubeDL | None = None

    def make_youtube_dl(options: object) -> _FakeYoutubeDL | _DurationLimitYoutubeDL:
        nonlocal duration_limit_youtube_dl
        if isinstance(options, dict) and "match_filter" in options:
            duration_limit_youtube_dl = _DurationLimitYoutubeDL(options)
            return duration_limit_youtube_dl
        metadata_youtube_dl.options = options
        return metadata_youtube_dl

    monkeypatch.setattr(yt_dlp, "YoutubeDL", make_youtube_dl)

    with pytest.raises(DurationLimitExceededError):
        await SourceMetadataService(
            YtDlpMetadataExtractor(60, tmp_path),
            _settings(60),
            DisabledCache(),
        ).inspect(_CANONICAL_URL)

    assert metadata_youtube_dl.extract_calls == [(_CANONICAL_URL, False)]
    assert duration_limit_youtube_dl is not None
    assert duration_limit_youtube_dl.extract_calls == [(_CANONICAL_URL, True)]
    assert list(tmp_path.iterdir()) == []


def test_audio_duration_probe_reads_wav_duration(tmp_path: Path) -> None:
    """Read a generated WAV duration through the real PyAV boundary."""
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x00\x00" * 8_000)

    assert transcription_inspection._probe_audio_duration(audio_path) == pytest.approx(1.0)


def test_audio_duration_probe_maps_invalid_audio_to_metadata_error(tmp_path: Path) -> None:
    """Hide unreadable audio details behind the stable metadata provider error."""
    audio_path = tmp_path / "invalid-audio"
    audio_path.write_bytes(b"not audio")

    with pytest.raises(MetadataProviderError, match="Unable to retrieve source metadata"):
        transcription_inspection._probe_audio_duration(audio_path)


def test_prepared_audio_cleanup_logs_nonfatal_os_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Report a media cleanup failure without raising from cleanup."""
    request_directory = tmp_path / "inspection"
    request_directory.mkdir()

    def raise_cleanup_error(path: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(shutil, "rmtree", raise_cleanup_error)

    with caplog.at_level(logging.ERROR, logger=transcription_inspection.__name__):
        PreparedAudio(request_directory / "audio.m4a", request_directory).cleanup()

    record = next(
        item for item in caplog.records if item.getMessage() == "temporary media cleanup failed"
    )
    assert record.__dict__["stage"] == "transcription"
    assert record.__dict__["reason"] == "temporary_media_cleanup_failed"


def test_failed_audio_fallback_logs_nonfatal_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Preserve audio fallback failure when its temporary directory cannot delete."""
    metadata_without_duration = _metadata()
    del metadata_without_duration["duration"]
    metadata_youtube_dl = _FakeYoutubeDL({}, metadata_without_duration)
    audio_youtube_dl = _FakeYoutubeDL({}, {"entries": []})

    def make_youtube_dl(options: object) -> _FakeYoutubeDL:
        if isinstance(options, dict) and "match_filter" in options:
            audio_youtube_dl.options = options
            return audio_youtube_dl
        metadata_youtube_dl.options = options
        return metadata_youtube_dl

    def raise_cleanup_error(path: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", make_youtube_dl)
    monkeypatch.setattr(shutil, "rmtree", raise_cleanup_error)

    with (
        caplog.at_level(logging.ERROR, logger=transcription_inspection.__name__),
        pytest.raises(MetadataProviderError),
    ):
        YtDlpMetadataExtractor(60, tmp_path).extract(_CANONICAL_URL)

    assert any(record.getMessage() == "temporary media cleanup failed" for record in caplog.records)


@pytest.mark.parametrize(
    ("provider_metadata", "expected_reason"),
    [
        (_metadata(title=123), "invalid_title"),
        (_metadata(duration=None), "invalid_duration_type"),
        (
            {key: value for key, value in _metadata().items() if key != "duration"},
            "missing_duration",
        ),
    ],
)
async def test_metadata_provider_log_identifies_failure_reason(
    provider_metadata: object,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log a stable reason while keeping the public provider error generic."""
    extractor = _FakeExtractor(provider_metadata)

    with (
        caplog.at_level(
            logging.DEBUG,
            logger=transcription_inspection.__name__,
        ),
        pytest.raises(MetadataProviderError),
    ):
        await _service(extractor).inspect(_CANONICAL_URL)

    record = next(item for item in caplog.records if item.getMessage() == "metadata provider error")
    assert record.__dict__["stage"] == "transcription"
    assert record.__dict__["reason"] == expected_reason


@pytest.mark.parametrize(
    ("submitted_url", "error_type", "event", "expected_reason"),
    [
        (
            "https://www.youtube.com/watch?v=short",
            InvalidSourceError,
            "invalid source error",
            "invalid_video_id",
        ),
        (
            "https://vimeo.com/123456789",
            UnsupportedPlatformError,
            "unsupported platform error",
            "unsupported_host",
        ),
    ],
)
async def test_inspection_error_log_identifies_failure_reason(
    submitted_url: str,
    error_type: type[InvalidSourceError | UnsupportedPlatformError],
    event: str,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log a stable reason for each public inspection error type."""
    with (
        caplog.at_level(
            logging.DEBUG,
            logger=transcription_inspection.__name__,
        ),
        pytest.raises(error_type),
    ):
        await _service(_FakeExtractor(_metadata())).inspect(submitted_url)

    record = next(item for item in caplog.records if item.getMessage() == event)
    assert record.__dict__["stage"] == "transcription"
    assert record.__dict__["reason"] == expected_reason


async def test_debug_event_excludes_source_identifiers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Record Source sizes without retaining Source identity or text values."""
    extractor = _FakeExtractor(
        _metadata(
            title="Logged title",
            description="Logged description",
            channel="Logged channel",
            duration=12.1,
        )
    )

    with caplog.at_level(logging.DEBUG, logger=transcription_service.__name__):
        await _service(extractor).inspect(_CANONICAL_URL)

    record = next(
        item for item in caplog.records if item.getMessage() == "source metadata normalized"
    )
    fields = record.__dict__
    assert fields["stage"] == "transcription"
    assert fields["platform"] == "youtube"
    assert fields["title_length"] == len("Logged title")
    assert fields["description_length"] == len("Logged description")
    assert fields["channel_length"] == len("Logged channel")
    assert fields["duration_seconds"] == 13
    for sensitive_field in (
        "submitted_url",
        "video_id",
        "canonical_url",
        "title",
        "description",
        "channel",
    ):
        assert sensitive_field not in fields
    assert "Logged title" not in caplog.text
    assert "Logged description" not in caplog.text
    assert "Logged channel" not in caplog.text


async def test_debug_event_redacts_sensitive_submitted_query_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Avoid logging secrets in submitted URL queries or fragments."""
    submitted_url = f"{_CANONICAL_URL}&api_key=secret-value#fragment-secret"
    extractor = _FakeExtractor(_metadata())

    with caplog.at_level(logging.DEBUG, logger=transcription_service.__name__):
        await _service(extractor).inspect(submitted_url)

    record = next(
        item for item in caplog.records if item.getMessage() == "source metadata normalized"
    )
    assert "secret-value" not in str(record.__dict__)
    assert "fragment-secret" not in str(record.__dict__)


_SOCIAL_ACCEPTED_FORMS = [
    (
        "https://www.instagram.com/p/ABC_123/",
        Platform.INSTAGRAM,
        "https://www.instagram.com/p/ABC_123",
        "https://www.instagram.com/reel/ABC_123",
        "Instagram",
        "ABC_123",
    ),
    (
        "https://instagram.com/tv/ABC-123?utm_source=test",
        Platform.INSTAGRAM,
        "https://instagram.com/tv/ABC-123",
        "https://www.instagram.com/reel/ABC-123",
        "Instagram",
        "ABC-123",
    ),
    (
        "https://www.instagram.com/reel/ABC123",
        Platform.INSTAGRAM,
        "https://www.instagram.com/reel/ABC123",
        "https://www.instagram.com/reel/ABC123",
        "Instagram",
        "ABC123",
    ),
    (
        "https://www.instagram.com/reels/ABC123",
        Platform.INSTAGRAM,
        "https://www.instagram.com/reels/ABC123",
        "https://www.instagram.com/reel/ABC123",
        "Instagram",
        "ABC123",
    ),
    (
        "https://www.facebook.com/watch?v=123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/watch?v=123456789",
        "https://www.facebook.com/watch?v=123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://www.facebook.com/reel/123456789?mibextid=test",
        Platform.FACEBOOK,
        "https://www.facebook.com/reel/123456789",
        "https://www.facebook.com/reel/123456789",
        "FacebookReel",
        "123456789",
    ),
    (
        "https://www.facebook.com/owner/videos/123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/owner/videos/123456789",
        "https://www.facebook.com/owner/videos/123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://www.facebook.com/owner/videos/clip-title/123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/owner/videos/clip-title/123456789",
        "https://www.facebook.com/owner/videos/clip-title/123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://www.facebook.com/owner/posts/123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/owner/posts/123456789",
        "https://www.facebook.com/owner/posts/123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://www.facebook.com/video.php?v=123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/video.php?v=123456789",
        "https://www.facebook.com/video.php?v=123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://www.facebook.com/video/video.php?v=123456789",
        Platform.FACEBOOK,
        "https://www.facebook.com/video/video.php?v=123456789",
        "https://www.facebook.com/video/video.php?v=123456789",
        "Facebook",
        "123456789",
    ),
    (
        "https://fb.watch/abc_123/",
        Platform.FACEBOOK,
        "https://fb.watch/abc_123",
        "https://www.facebook.com/reel/123456789",
        "FacebookReel",
        "123456789",
    ),
    (
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        Platform.TIKTOK,
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "TikTok",
        "1234567890123456789",
    ),
    (
        "https://www.tiktok.com/embed/1234567890123456789",
        Platform.TIKTOK,
        "https://www.tiktok.com/embed/1234567890123456789",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "TikTok",
        "1234567890123456789",
    ),
    (
        "https://vm.tiktok.com/ZMabc123/",
        Platform.TIKTOK,
        "https://vm.tiktok.com/ZMabc123",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "TikTok",
        "1234567890123456789",
    ),
    (
        "https://vt.tiktok.com/ZMabc123",
        Platform.TIKTOK,
        "https://vt.tiktok.com/ZMabc123",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "TikTok",
        "1234567890123456789",
    ),
    (
        "https://www.tiktok.com/t/ZMabc123?lang=en",
        Platform.TIKTOK,
        "https://www.tiktok.com/t/ZMabc123",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "TikTok",
        "1234567890123456789",
    ),
    (
        "https://x.com/creator/status/1234567890123456789",
        Platform.X,
        "https://x.com/creator/status/1234567890123456789",
        "https://x.com/creator/status/1234567890123456789",
        "Twitter",
        "1234567890123456789",
    ),
    (
        "https://twitter.com/creator/status/1234567890123456789/video/2",
        Platform.X,
        "https://twitter.com/creator/status/1234567890123456789/video/2",
        "https://twitter.com/creator/status/1234567890123456789",
        "Twitter",
        "1234567890123456789",
    ),
    (
        "https://x.com/i/web/status/1234567890123456789",
        Platform.X,
        "https://x.com/i/web/status/1234567890123456789",
        "https://x.com/i/web/status/1234567890123456789",
        "Twitter",
        "1234567890123456789",
    ),
    (
        "https://x.com/statuses/1234567890123456789",
        Platform.X,
        "https://x.com/statuses/1234567890123456789",
        "https://x.com/statuses/1234567890123456789",
        "Twitter",
        "1234567890123456789",
    ),
    (
        "https://t.co/abc_123",
        Platform.X,
        "https://t.co/abc_123",
        "https://twitter.com/creator/status/1234567890123456789",
        "Twitter",
        "1234567890123456789",
    ),
]


def _social_metadata(
    extractor_key: str,
    video_id: str,
    canonical_url: str,
    **overrides: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "id": video_id,
        "extractor_key": extractor_key,
        "webpage_url": canonical_url,
        "title": "Social video",
        "description": "Social description",
        "channel": "Social channel",
        "duration": 12.1,
        "formats": [{"format_id": "video", "vcodec": "avc1"}],
    }
    metadata.update(overrides)
    return metadata


@pytest.mark.parametrize(
    (
        "submitted_url",
        "platform",
        "provider_url",
        "canonical_url",
        "extractor_key",
        "video_id",
    ),
    _SOCIAL_ACCEPTED_FORMS,
)
async def test_social_forms_normalize_and_reach_provider_once(
    submitted_url: str,
    platform: Platform,
    provider_url: str,
    canonical_url: str,
    extractor_key: str,
    video_id: str,
) -> None:
    """Accept every allowlisted social form and preserve provider identity."""
    extractor = _FakeExtractor(_social_metadata(extractor_key, video_id, canonical_url))

    source = (await _service(extractor).inspect(submitted_url)).source

    assert source.platform is platform
    assert source.video_id == video_id
    assert source.url == canonical_url
    assert extractor.calls == [provider_url]


@pytest.mark.parametrize(
    "submitted_url",
    [
        "https://www.instagram.com/",
        "https://www.instagram.com/stories/creator/123",
        "https://www.facebook.com/creator",
        "https://www.facebook.com/groups/123",
        "https://www.facebook.com/watch?v=123&v=123",
        "https://www.tiktok.com/@creator",
        "https://www.tiktok.com/live",
        "https://www.tiktok.com/@creator/video/not-numeric",
        "https://x.com/creator/status/123/video/0",
        "https://x.com/spaces/123",
        "https://x.com/creator/status/123/extra",
        "https://vm.tiktok.com/one/two",
    ],
)
async def test_social_unsupported_shapes_fail_before_provider_access(
    submitted_url: str,
) -> None:
    """Reject profiles, collections, and malformed video paths locally."""
    extractor = _FakeExtractor(_social_metadata("Instagram", "ABC123", ""))

    with pytest.raises(InvalidSourceError, match="Invalid source URL"):
        await _service(extractor).inspect(submitted_url)

    assert extractor.calls == []


@pytest.mark.parametrize(
    "submitted_url",
    [
        "https://user:password@www.instagram.com/reel/ABC123",
        "https://www.facebook.com:443/reel/123456789",
        "https://www.tiktok.com/@creator/video/1234567890123456789%ZZ",
        "https://www.tiktok.com/@creator/video/1234567890123456789/extra",
        "https://x.com.evil.example/creator/status/123456789",
        "https://evil.x.com/creator/status/123456789",
        "http://x.com/creator/status/123456789",
    ],
)
async def test_social_url_security_rejections_do_not_call_provider(
    submitted_url: str,
) -> None:
    """Reject unsafe authorities, schemes, encodings, and deceptive hosts."""
    extractor = _FakeExtractor(_social_metadata("Twitter", "123456789", ""))

    with pytest.raises(InvalidSourceError, match="Invalid source URL"):
        await _service(extractor).inspect(submitted_url)

    assert extractor.calls == []


@pytest.mark.parametrize(
    ("submitted_url", "error_type"),
    [
        ("https://vimeo.com/123456789", UnsupportedPlatformError),
        ("https://example.com/video/123", UnsupportedPlatformError),
    ],
)
async def test_unsupported_hosts_remain_distinct(
    submitted_url: str,
    error_type: type[UnsupportedPlatformError],
) -> None:
    """Keep valid unrelated hosts distinct from invalid supported shapes."""
    extractor = _FakeExtractor(_social_metadata("Twitter", "123456789", ""))

    with pytest.raises(error_type, match="Only YouTube"):
        await _service(extractor).inspect(submitted_url)

    assert extractor.calls == []


@pytest.mark.parametrize(
    ("platform", "extractor_key", "canonical_url", "video_id"),
    [
        (
            Platform.INSTAGRAM,
            "instagram",
            "https://www.instagram.com/reel/ABC123",
            "ABC123",
        ),
        (
            Platform.FACEBOOK,
            "FacebookExtra",
            "https://www.facebook.com/reel/123456789",
            "123456789",
        ),
        (
            Platform.TIKTOK,
            "TikTokExtra",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "1234567890123456789",
        ),
        (
            Platform.X,
            "Twitter",
            "https://www.instagram.com/reel/ABC123",
            "123456789",
        ),
    ],
)
async def test_social_extractor_or_canonical_mismatch_is_invalid(
    platform: Platform,
    extractor_key: str,
    canonical_url: str,
    video_id: str,
) -> None:
    """Require exact extractor identity and same-platform canonical URLs."""
    submitted_url = {
        Platform.INSTAGRAM: "https://www.instagram.com/reel/ABC123",
        Platform.FACEBOOK: "https://www.facebook.com/reel/123456789",
        Platform.TIKTOK: "https://www.tiktok.com/@creator/video/1234567890123456789",
        Platform.X: "https://x.com/creator/status/123456789",
    }[platform]
    expected_key = {
        Platform.INSTAGRAM: "Instagram",
        Platform.FACEBOOK: "FacebookReel",
        Platform.TIKTOK: "TikTok",
        Platform.X: "Twitter",
    }[platform]
    metadata = _social_metadata(expected_key, video_id, canonical_url)
    if extractor_key != expected_key:
        metadata["extractor_key"] = extractor_key
    else:
        metadata["webpage_url"] = canonical_url
    extractor = _FakeExtractor(metadata)

    with pytest.raises(InvalidSourceError, match="Invalid source URL"):
        await _service(extractor).inspect(submitted_url)


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"entries": []},
        {"_type": "playlist"},
        {"is_live": True},
        {"live_status": "is_live"},
        {"formats": []},
        {"formats": [{"vcodec": "none"}]},
        {"formats": [{"vcodec": "none", "acodec": "mp4a"}]},
    ],
)
async def test_social_non_finite_or_non_video_metadata_is_invalid(
    metadata_overrides: dict[str, object],
) -> None:
    """Reject collections, live states, and results without video formats."""
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            "ABC123",
            "https://www.instagram.com/reel/ABC123",
            **metadata_overrides,
        )
    )

    with pytest.raises(InvalidSourceError, match="Invalid source URL"):
        await _service(extractor).inspect("https://www.instagram.com/reel/ABC123")


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"id": None},
        {"webpage_url": None},
        {"formats": [{"vcodec": 123}]},
        {"formats": [object()]},
        {"duration": None},
        {"duration": "12"},
    ],
)
async def test_social_malformed_metadata_maps_to_502(
    metadata_overrides: dict[str, object],
) -> None:
    """Hide malformed social metadata behind the stable provider error."""
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            "ABC123",
            "https://www.instagram.com/reel/ABC123",
            **metadata_overrides,
        )
    )

    with pytest.raises(
        MetadataProviderError,
        match="Unable to retrieve source metadata",
    ):
        await _service(extractor).inspect("https://www.instagram.com/reel/ABC123")


async def test_social_optional_metadata_uses_normalized_fallbacks() -> None:
    """Normalize missing social title, description, and channel values."""
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            "ABC123",
            "https://www.instagram.com/reel/ABC123",
            title=" ",
            description=None,
            channel=None,
            uploader="Fallback creator",
        )
    )

    source = (await _service(extractor).inspect("https://www.instagram.com/reel/ABC123")).source

    assert source.title == ""
    assert source.description == ""
    assert source.channel == "Fallback creator"


@pytest.mark.parametrize("duration", [1800.0, 1800.1])
@pytest.mark.parametrize(
    ("submitted_url", "extractor_key", "video_id", "canonical_url"),
    [
        (
            "https://www.instagram.com/reel/ABC123",
            "Instagram",
            "ABC123",
            "https://www.instagram.com/reel/ABC123",
        ),
        (
            "https://www.facebook.com/reel/123456789",
            "FacebookReel",
            "123456789",
            "https://www.facebook.com/reel/123456789",
        ),
        (
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "TikTok",
            "1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
        ),
        (
            "https://x.com/creator/status/123456789",
            "Twitter",
            "987654321",
            "https://x.com/creator/status/123456789",
        ),
    ],
)
async def test_social_duration_limit_is_applied_after_metadata(
    submitted_url: str,
    extractor_key: str,
    video_id: str,
    canonical_url: str,
    duration: float,
) -> None:
    """Apply the configured duration limit to every social Source."""
    extractor = _FakeExtractor(
        _social_metadata(
            extractor_key,
            video_id,
            canonical_url,
            duration=duration,
        )
    )

    if duration == 1800.0:
        source = (await _service(extractor, max_duration=1800).inspect(submitted_url)).source
        assert source.duration_seconds == 1800
    else:
        with pytest.raises(DurationLimitExceededError):
            await _service(extractor, max_duration=1800).inspect(submitted_url)
    assert len(extractor.calls) == 1


async def test_typed_metadata_timeout_maps_to_504_without_message_matching() -> None:
    """Map a nested typed provider timeout without inspecting its text."""
    nested_timeout = Timeout("provider detail")
    provider_timeout = DownloadError(
        "ordinary detail",
        exc_info=(Timeout, nested_timeout, cast(TracebackType, None)),
    )
    extractor = _FakeExtractor(_metadata(), error=provider_timeout)

    with pytest.raises(
        PipelineTimeoutError,
        match="Source metadata acquisition timed out",
    ):
        await _service(extractor).inspect(_CANONICAL_URL)


@pytest.mark.parametrize(
    ("submitted_url", "provider_message"),
    [
        (
            "https://www.instagram.com/reel/ABC123",
            "This post is unavailable.",
        ),
        (
            "https://www.facebook.com/reel/123456789",
            "This post is unavailable.",
        ),
        (
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "This video is not available in your country.",
        ),
        (
            "https://x.com/creator/status/123456789",
            "This post is unavailable.",
        ),
    ],
)
async def test_social_unavailable_provider_failures_map_to_not_found(
    submitted_url: str,
    provider_message: str,
) -> None:
    """Map inaccessible social Sources to the stable 404 domain error."""
    extractor = _FakeExtractor(
        _metadata(),
        error=YoutubeDLError(provider_message),
    )

    with pytest.raises(
        SourceUnavailableError,
        match="Source is unavailable",
    ) as error:
        await _service(extractor).inspect(submitted_url)

    assert provider_message not in str(error.value)
    assert extractor.calls


@pytest.mark.parametrize(
    "submitted_url",
    [
        "https://www.instagram.com/reel/ABC123",
        "https://www.facebook.com/reel/123456789",
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        "https://x.com/creator/status/123456789",
    ],
)
async def test_social_typed_metadata_timeout_maps_to_504(
    submitted_url: str,
) -> None:
    """Map nested typed metadata timeouts for every social platform."""
    nested_timeout = Timeout("provider detail")
    provider_timeout = DownloadError(
        "ordinary detail",
        exc_info=(Timeout, nested_timeout, cast(TracebackType, None)),
    )
    extractor = _FakeExtractor(_metadata(), error=provider_timeout)

    with pytest.raises(
        PipelineTimeoutError,
        match="Source metadata acquisition timed out",
    ):
        await _service(extractor).inspect(submitted_url)


async def test_social_video_formats_allow_audio_entries_with_one_video() -> None:
    """Accept provider format lists containing audio-only entries."""
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            "ABC123",
            "https://www.instagram.com/reel/ABC123",
            formats=[
                {"vcodec": "none"},
                {"acodec": "mp4a.40.2"},
                {"vcodec": "avc1", "acodec": "none"},
            ],
        )
    )

    source = (await _service(extractor).inspect("https://www.instagram.com/reel/ABC123")).source

    assert source.video_id == "ABC123"


async def test_youtube_metadata_cache_hit_skips_provider_and_omits_prepared_audio(
    tmp_path: Path,
) -> None:
    """Reuse normalized YouTube metadata without sharing request audio ownership."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    audio_path = request_directory / "audio.webm"
    audio_path.write_bytes(b"audio")
    extractor = _FakeExtractor(
        _metadata(extractor_only="not cacheable"),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    service = _cached_service(extractor, cache)
    first = None
    second = None

    try:
        first = await service.inspect(_CANONICAL_URL)
        second = await service.inspect(_CANONICAL_URL)

        assert extractor.calls == [_CANONICAL_URL]
        assert second.source == first.source
        assert second.source is not first.source
        assert second.prepared_audio is None
        assert _cached_value(
            cache,
            redis,
            transcription_service._source_metadata_entry(
                SourceIdentity(Platform.YOUTUBE, _VIDEO_ID)
            ),
        ) == {
            "platform": "youtube",
            "video_id": _VIDEO_ID,
            "url": _CANONICAL_URL,
            "title": "Example video",
            "description": "A complete description.",
            "channel": "Example channel",
            "duration_seconds": 12,
        }
    finally:
        if first is not None:
            first.cleanup()
        if second is not None:
            second.cleanup()
        await cache.aclose()


async def test_social_cache_payloads_exclude_raw_metadata_and_prepared_audio(
    tmp_path: Path,
) -> None:
    """Persist only Source fields and alias identity values for social inspections."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    normalized_submitted_url = "https://www.instagram.com/p/ABC_123"
    canonical_url = "https://www.instagram.com/reel/ABC_123"
    source_identity = SourceIdentity(Platform.INSTAGRAM, "ABC_123")
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    audio_path = request_directory / "audio.webm"
    audio_path.write_bytes(b"audio")
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            source_identity.external_content_id,
            canonical_url,
            extractor_only="raw-extractor-sentinel",
        ),
        prepared_audio=PreparedAudio(audio_path, request_directory),
    )
    service = _cached_service(extractor, cache)
    inspected = None

    try:
        inspected = await service.inspect(submitted_url)
        alias_entry = transcription_service._source_alias_entry(
            Platform.INSTAGRAM,
            normalized_submitted_url,
        )
        alias_key = cache._cache_key(alias_entry)
        alias_raw_value = redis.raw_value(alias_key)
        assert alias_raw_value is not None
        assert json.loads(alias_raw_value)["value"] == {
            "platform": "instagram",
            "external_content_id": "ABC_123",
        }
        metadata_value = _cached_value(
            cache,
            redis,
            transcription_service._source_metadata_entry(source_identity),
        )
        assert set(metadata_value) == {
            "platform",
            "video_id",
            "url",
            "title",
            "description",
            "channel",
            "duration_seconds",
        }
        cached_payloads = [
            redis.raw_value(cache_key) for cache_key in redis.keys if ":source:" in cache_key
        ]
        assert all(payload is not None for payload in cached_payloads)
        cached_source_data = b"".join(
            payload for payload in cached_payloads if payload is not None
        ).decode()
        assert "raw-extractor-sentinel" not in cached_source_data
        assert str(audio_path) not in cached_source_data
    finally:
        if inspected is not None:
            inspected.cleanup()
        await cache.aclose()


async def test_source_metadata_cache_expires_at_the_exact_ttl_boundary() -> None:
    """Serve Source metadata until, but not at, the configured freshness boundary."""
    clock = ManualClock()
    cache, _ = _redis_source_cache(clock)
    extractor = _FakeExtractor(_metadata())
    service = _cached_service(extractor, cache)

    try:
        await service.inspect(_CANONICAL_URL)
        clock.advance(2_591_999.999)
        await service.inspect(_CANONICAL_URL)
        assert len(extractor.calls) == 1

        clock.advance(0.001)
        await service.inspect(_CANONICAL_URL)
        assert len(extractor.calls) == 2
    finally:
        await cache.aclose()


async def test_cached_source_rechecks_the_current_duration_limit() -> None:
    """Apply the current duration policy even when metadata is freshly cached."""
    clock = ManualClock()
    cache, _ = _redis_source_cache(clock)
    warm_extractor = _FakeExtractor(_metadata(duration=30.0))
    restricted_extractor = _FakeExtractor(_metadata(duration=30.0))
    warming_service = _cached_service(warm_extractor, cache, max_duration=60)
    restricted_service = _cached_service(restricted_extractor, cache, max_duration=15)

    try:
        await warming_service.inspect(_CANONICAL_URL)
        with pytest.raises(DurationLimitExceededError):
            await restricted_service.inspect(_CANONICAL_URL)
        assert restricted_extractor.calls == []
    finally:
        await cache.aclose()


@pytest.mark.parametrize(
    ("first_outcome", "expected_error"),
    [
        (MetadataProviderError("provider failed"), MetadataProviderError),
        (YoutubeDLError("Video unavailable"), SourceUnavailableError),
        (Timeout("provider timed out"), PipelineTimeoutError),
        (_metadata(id=object()), MetadataProviderError),
        (_metadata(duration=1800.1), DurationLimitExceededError),
    ],
    ids=["provider", "unavailable", "timeout", "malformed", "duration"],
)
async def test_failed_source_inspections_are_not_cached(
    first_outcome: object,
    expected_error: type[Exception],
) -> None:
    """Allow a retry to inspect successfully after every typed first failure."""
    clock = ManualClock()
    cache, _ = _redis_source_cache(clock)
    extractor = _SequenceExtractor([first_outcome, _metadata()])
    service = _cached_service(extractor, cache)

    try:
        with pytest.raises(expected_error):
            await service.inspect(_CANONICAL_URL)
        recovered = await service.inspect(_CANONICAL_URL)

        assert recovered.source.video_id == _VIDEO_ID
        assert extractor.calls == [_CANONICAL_URL, _CANONICAL_URL]
    finally:
        await cache.aclose()


async def test_corrupt_source_metadata_is_deleted_and_reloaded() -> None:
    """Heal a corrupt metadata entry before loading a fresh normalized Source."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    extractor = _FakeExtractor(_metadata())
    service = _cached_service(extractor, cache)
    metadata_entry = transcription_service._source_metadata_entry(
        SourceIdentity(Platform.YOUTUBE, _VIDEO_ID)
    )

    try:
        await service.inspect(_CANONICAL_URL)
        await redis.put_raw(
            cache._cache_key(metadata_entry),
            json.dumps(
                {
                    "envelope_version": 2,
                    "value_version": "source-metadata-v1",
                    "value": {
                        "platform": "youtube",
                        "video_id": _VIDEO_ID,
                        "url": _CANONICAL_URL,
                        "title": "Example video",
                        "description": "A complete description.",
                        "channel": "Example channel",
                        "duration_seconds": 12,
                        "unexpected": True,
                    },
                    "freshness_seconds": 2_592_000,
                    "retention_seconds": 2_592_000,
                }
            ).encode(),
        )

        recovered = await service.inspect(_CANONICAL_URL)
        assert recovered.source.video_id == _VIDEO_ID
        assert len(extractor.calls) == 2
        assert _cached_value(cache, redis, metadata_entry)["video_id"] == _VIDEO_ID
    finally:
        await cache.aclose()


async def test_alias_without_metadata_reinspects_instead_of_constructing_source() -> None:
    """Treat a retained social alias as an identity hint until metadata is available."""
    clock = ManualClock()
    cache, _ = _redis_source_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    source_identity = SourceIdentity(Platform.INSTAGRAM, "ABC_123")
    extractor = _FakeExtractor(
        _social_metadata(
            "Instagram",
            source_identity.external_content_id,
            "https://www.instagram.com/reel/ABC_123",
        )
    )
    service = _cached_service(extractor, cache)

    async def alias_loader() -> SourceIdentity:
        return source_identity

    try:
        await cache.get_or_load(
            transcription_service._source_alias_entry(
                Platform.INSTAGRAM,
                "https://www.instagram.com/p/ABC_123",
            ),
            alias_loader,
        )
        inspected = await service.inspect(submitted_url)

        assert inspected.source.video_id == source_identity.external_content_id
        assert extractor.calls == ["https://www.instagram.com/p/ABC_123"]
    finally:
        await cache.aclose()


async def test_conflicting_alias_never_replaces_or_returns_stale_identity() -> None:
    """Retain the first alias writer while recovering metadata under the new identity."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    source_a = SourceIdentity(Platform.INSTAGRAM, "source-a")
    source_b = SourceIdentity(Platform.INSTAGRAM, "source-b")
    canonical_url = "https://www.instagram.com/reel/ABC_123"
    extractor = _SequenceExtractor(
        [
            _social_metadata("Instagram", source_a.external_content_id, canonical_url),
            _social_metadata("Instagram", source_b.external_content_id, canonical_url),
            _social_metadata("Instagram", source_b.external_content_id, canonical_url),
        ]
    )
    service = _cached_service(extractor, cache)
    source_a_entry = transcription_service._source_metadata_entry(source_a)
    source_b_entry = transcription_service._source_metadata_entry(source_b)

    try:
        await service.inspect(submitted_url)
        await redis.put_raw(cache._cache_key(source_a_entry), b"{")

        recovered = await service.inspect(submitted_url)
        assert recovered.source.video_id == source_b.external_content_id
        assert redis.raw_value(cache._cache_key(source_a_entry)) is None
        assert _cached_value(cache, redis, source_b_entry)["video_id"] == "source-b"

        safe_retry = await service.inspect(submitted_url)
        assert safe_retry.source.video_id == source_b.external_content_id
        assert len(extractor.calls) == 3
    finally:
        await cache.aclose()


@pytest.mark.parametrize("operation", ["read", "ownership_write"])
@pytest.mark.parametrize(
    ("provider_error", "expected_error"),
    [
        (None, None),
        (YoutubeDLError("Video unavailable"), SourceUnavailableError),
    ],
    ids=["success", "unavailable"],
)
async def test_source_cache_failures_preserve_provider_outcomes(
    operation: str,
    provider_error: Exception | None,
    expected_error: type[Exception] | None,
) -> None:
    """Fail open on Source cache outages without changing provider outcomes."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    extractor = _FakeExtractor(_metadata(), error=provider_error)
    service = _cached_service(extractor, cache)
    redis.fail_operations.add(operation)

    try:
        if expected_error is None:
            assert (await service.inspect(_CANONICAL_URL)).source.video_id == _VIDEO_ID
        else:
            with pytest.raises(expected_error):
                await service.inspect(_CANONICAL_URL)
        assert extractor.calls == [_CANONICAL_URL]
    finally:
        await cache.aclose()


async def test_alias_registration_failure_preserves_successful_inspection() -> None:
    """Keep a successful social Source when only canonical alias registration fails."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    source_a = SourceIdentity(Platform.INSTAGRAM, "source-a")
    source_b = SourceIdentity(Platform.INSTAGRAM, "source-b")
    canonical_url = "https://www.instagram.com/reel/ABC_123"
    source_b_value = Source(
        platform=Platform.INSTAGRAM,
        video_id=source_b.external_content_id,
        url=canonical_url,
        title="Source B",
        description="Description B",
        channel="Channel B",
        duration_seconds=12,
    )

    async def alias_loader() -> SourceIdentity:
        return source_a

    async def metadata_loader() -> Source:
        return source_b_value

    try:
        await cache.get_or_load(
            transcription_service._source_alias_entry(
                Platform.INSTAGRAM,
                "https://www.instagram.com/p/ABC_123",
            ),
            alias_loader,
        )
        await cache.get_or_load(
            transcription_service._source_metadata_entry(source_b),
            metadata_loader,
        )
        redis.fail_operations.add("ownership_write")
        extractor = _FakeExtractor(
            _social_metadata("Instagram", source_b.external_content_id, canonical_url)
        )
        service = _cached_service(extractor, cache)

        inspected = await service.inspect(submitted_url)
        assert inspected.source == source_b_value
        assert extractor.calls == ["https://www.instagram.com/p/ABC_123"]
    finally:
        await cache.aclose()


async def test_source_cache_keys_and_logs_exclude_source_sensitive_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep Source identities and text out of cache keys and structured logging."""
    clock = ManualClock()
    cache, redis = _redis_source_cache(clock)
    submitted_url = "https://www.instagram.com/p/ABC_123/"
    canonical_url = "https://www.instagram.com/reel/ABC_123"
    external_content_id = "private-content-id"
    title = "Private title"
    description = "Private description"
    channel = "Private channel"
    service = _cached_service(
        _FakeExtractor(
            _social_metadata(
                "Instagram",
                external_content_id,
                canonical_url,
                title=title,
                description=description,
                channel=channel,
            )
        ),
        cache,
    )

    try:
        with caplog.at_level(logging.DEBUG):
            await service.inspect(submitted_url)
        sensitive_values = (
            submitted_url,
            canonical_url,
            external_content_id,
            title,
            description,
            channel,
        )
        for cache_key in redis.keys:
            assert ":source:" in cache_key
            assert all(value not in cache_key for value in sensitive_values)
        for record in caplog.records:
            if record.name in {
                transcription_service.__name__,
                "reelio.cache.redis",
            }:
                assert all(value not in str(record.__dict__) for value in sensitive_values)
    finally:
        await cache.aclose()
