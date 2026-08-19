"""Deterministic tests for source validation and metadata acquisition."""

import logging
import math
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import cast

import pytest
import yt_dlp  # type: ignore[import-untyped]
from yt_dlp.utils import YoutubeDLError  # type: ignore[import-untyped]

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    MetadataProviderError,
    SourceUnavailableError,
    UnsupportedPlatformError,
)
from reelio.extraction.services.transcription import service as transcription_service
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.service import (
    SourceMetadataService,
    YtDlpMetadataExtractor,
)
from reelio.extraction.types import Platform

_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _FakeExtractor:
    def __init__(self, metadata: object, error: Exception | None = None) -> None:
        self.metadata = metadata
        self.error = error
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> Mapping[str, object]:
        self.calls.append(canonical_url)
        if self.error is not None:
            raise self.error
        return cast(Mapping[str, object], self.metadata)


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
    return SourceMetadataService(extractor=extractor, settings=_settings(max_duration))


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

    source = await _service(extractor).inspect(submitted_url)

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

    with pytest.raises(exception_type, match="Invalid YouTube URL"):
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
        match="Only YouTube URLs are supported",
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

    source = await _service(extractor).inspect(_CANONICAL_URL)

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

    source = await _service(extractor).inspect(_CANONICAL_URL)

    assert source.description == expected_description
    assert source.channel == expected_channel


async def test_provider_id_is_optional_when_url_identity_is_valid() -> None:
    """Treat the validated URL ID as authoritative when provider ID is absent."""
    metadata = _metadata()
    del metadata["id"]
    extractor = _FakeExtractor(metadata)

    source = await _service(extractor).inspect(_CANONICAL_URL)

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
        match="Unable to retrieve YouTube metadata",
    ) as error:
        await _service(extractor).inspect(_CANONICAL_URL)

    assert "differentid" not in str(error.value)
    assert "not metadata" not in str(error.value)


async def test_missing_duration_maps_to_metadata_provider_error() -> None:
    """Reject provider results that omit the required duration."""
    metadata = _metadata()
    del metadata["duration"]
    extractor = _FakeExtractor(metadata)

    with pytest.raises(
        MetadataProviderError, match="Unable to retrieve YouTube metadata"
    ):
        await _service(extractor).inspect(_CANONICAL_URL)


async def test_duration_equal_to_limit_is_allowed() -> None:
    """Allow a video whose ceiled duration exactly equals the configured limit."""
    extractor = _FakeExtractor(_metadata(duration=1800.0))

    source = await _service(extractor, max_duration=1800).inspect(_CANONICAL_URL)

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
        match="YouTube video is unavailable",
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
        match="Unable to retrieve YouTube metadata",
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
) -> None:
    """Pass the canonical URL and metadata-only options to yt-dlp exactly once."""
    fake_youtube_dl = _FakeYoutubeDL({}, _metadata())
    _patch_youtube_dl(monkeypatch, fake_youtube_dl)

    metadata = YtDlpMetadataExtractor().extract(_CANONICAL_URL)

    assert metadata["title"] == "Example video"
    assert fake_youtube_dl.options == {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ignoreconfig": True,
    }
    assert fake_youtube_dl.extract_calls == [(_CANONICAL_URL, False)]
    assert fake_youtube_dl.media_download_calls == 0


def test_yt_dlp_adapter_rejects_non_mapping_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a provider response that cannot supply named metadata fields."""
    fake_youtube_dl = _FakeYoutubeDL({}, ["not metadata"])
    _patch_youtube_dl(monkeypatch, fake_youtube_dl)

    with pytest.raises(
        MetadataProviderError, match="Unable to retrieve YouTube metadata"
    ):
        YtDlpMetadataExtractor().extract(_CANONICAL_URL)


async def test_debug_event_contains_normalized_source_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Record normalized Source fields as structured DEBUG attributes."""
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
        item
        for item in caplog.records
        if item.getMessage() == "source metadata normalized"
    )
    fields = record.__dict__
    assert fields["stage"] == "transcription"
    assert fields["submitted_url"] == _CANONICAL_URL
    assert fields["platform"] == "youtube"
    assert fields["video_id"] == _VIDEO_ID
    assert fields["canonical_url"] == _CANONICAL_URL
    assert fields["title"] == "[REDACTED]"
    assert fields["title_length"] == len("Logged title")
    assert fields["description"] == "[REDACTED]"
    assert fields["description_length"] == len("Logged description")
    assert fields["channel"] == "[REDACTED]"
    assert fields["channel_length"] == len("Logged channel")
    assert fields["duration_seconds"] == 13


async def test_debug_event_redacts_sensitive_submitted_query_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Avoid logging secrets in submitted URL queries or fragments."""
    submitted_url = f"{_CANONICAL_URL}&api_key=secret-value#fragment-secret"
    extractor = _FakeExtractor(_metadata())

    with caplog.at_level(logging.DEBUG, logger=transcription_service.__name__):
        await _service(extractor).inspect(submitted_url)

    record = next(
        item
        for item in caplog.records
        if item.getMessage() == "source metadata normalized"
    )
    assert "secret-value" not in str(record.__dict__)
    assert "fragment-secret" not in str(record.__dict__)
    assert "%5BREDACTED%5D" in record.__dict__["submitted_url"]
