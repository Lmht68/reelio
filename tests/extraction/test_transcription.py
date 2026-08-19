"""Deterministic tests for source validation and metadata acquisition."""

import asyncio
import logging
import math
import tempfile
import threading
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import ClassVar, cast

import pytest
import yt_dlp  # type: ignore[import-untyped]
from requests.exceptions import Timeout
from youtube_transcript_api import CouldNotRetrieveTranscript
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
from reelio.extraction.services.transcription import service as transcription_service
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.service import (
    AudioDownloader,
    FasterWhisperTranscriber,
    SourceMetadataService,
    TranscriptionService,
    WhisperResult,
    WhisperTranscriber,
    YouTubeCaptionProvider,
    YtDlpAudioDownloader,
    YtDlpMetadataExtractor,
    _WhisperProviderFailure,
    _WhisperProviderTimeout,
)
from reelio.extraction.types import Platform, Source, Transcript, TranscriptMethod

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


class _FakeCaptionTrack:
    def __init__(
        self,
        language_code: str,
        is_generated: bool,
        segments: Sequence[str],
        error: Exception | None = None,
    ) -> None:
        self.language_code = language_code
        self.is_generated = is_generated
        self._segments = segments
        self._error = error
        self.fetch_calls = 0

    def fetch_segments(self) -> Sequence[str]:
        self.fetch_calls += 1
        if self._error is not None:
            raise self._error
        return self._segments


class _FakeCaptionProvider:
    def __init__(
        self,
        tracks: Sequence[_FakeCaptionTrack],
        error: Exception | None = None,
    ) -> None:
        self._tracks = tracks
        self._error = error
        self.calls: list[str] = []

    def list_tracks(self, video_id: str) -> Sequence[_FakeCaptionTrack]:
        self.calls.append(video_id)
        if self._error is not None:
            raise self._error
        return self._tracks


def _source() -> Source:
    return Source(
        platform=Platform.YOUTUBE,
        video_id=_VIDEO_ID,
        url=_CANONICAL_URL,
        title="Caption test video",
        description="A caption test source.",
        channel="Caption test channel",
        duration_seconds=42,
    )


async def test_caption_service_returns_selected_caption_track() -> None:
    """Return normalized text and metadata from an available caption track."""
    provider = _FakeCaptionProvider(
        [_FakeCaptionTrack("en", False, ["Hello", "world."])]
    )

    transcript = await _transcription_service(provider).acquire(_source())

    assert transcript == Transcript(
        text="Hello world.",
        language="en",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    )
    assert provider.calls == [_VIDEO_ID]


def _caption_tracks(
    *definitions: tuple[str, bool, Sequence[str]],
) -> list[_FakeCaptionTrack]:
    return [
        _FakeCaptionTrack(language, is_generated, segments)
        for language, is_generated, segments in definitions
    ]


async def test_manual_exact_english_wins_regardless_of_provider_order() -> None:
    """Prefer manual exact English over every lower-ranked track."""
    tracks = _caption_tracks(
        ("de", True, ["generated German"]),
        ("en-US", False, ["regional manual English"]),
        ("en", True, ["generated English"]),
        ("en", False, ["manual exact English"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "en"
    assert transcript.text == "manual exact English"
    assert [track.fetch_calls for track in tracks] == [0, 0, 0, 1]


async def test_manual_regional_english_wins_generated_exact_english() -> None:
    """Prefer manual regional English over generated exact English."""
    tracks = _caption_tracks(
        ("en", True, ["generated exact"]),
        ("en-GB", False, ["manual regional"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "en-GB"
    assert transcript.text == "manual regional"


@pytest.mark.parametrize(
    ("generated", "expected_language"),
    [(False, "en"), (True, "en")],
)
async def test_exact_english_wins_regional_english_within_track_kind(
    generated: bool,
    expected_language: str,
) -> None:
    """Prefer exact English over regional English within one track kind."""
    tracks = _caption_tracks(
        ("en-US", generated, ["regional"]),
        ("en", generated, ["exact"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == expected_language
    assert transcript.text == "exact"


async def test_regional_english_preserves_provider_order() -> None:
    """Preserve provider order among equivalent regional English tracks."""
    tracks = _caption_tracks(
        ("en-GB", False, ["British"]),
        ("en-AU", False, ["Australian"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "en-GB"
    assert transcript.text == "British"


async def test_manual_other_language_wins_generated_other_language() -> None:
    """Prefer manual non-English tracks over generated non-English tracks."""
    tracks = _caption_tracks(
        ("fr", True, ["generated French"]),
        ("de", False, ["manual German"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "de"
    assert transcript.text == "manual German"


async def test_other_language_tracks_preserve_provider_order() -> None:
    """Preserve provider order among equivalent non-English tracks."""
    tracks = _caption_tracks(
        ("fr", False, ["French"]),
        ("de", False, ["German"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "fr"
    assert transcript.text == "French"


async def test_english_classification_accepts_bcp47_regional_codes_only() -> None:
    """Treat en-* as English without treating malformed aliases as English."""
    tracks = _caption_tracks(
        ("en_US", False, ["underscore"]),
        ("english", False, ["word"]),
        ("eng", False, ["three-letter"]),
        ("en-CA", True, ["regional English"]),
    )

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "en-CA"
    assert transcript.text == "regional English"


async def test_failed_higher_ranked_track_falls_through() -> None:
    """Continue to a lower-ranked track after a provider fetch failure."""
    tracks = [
        _FakeCaptionTrack("en", False, [], error=ValueError("provider detail")),
        _FakeCaptionTrack("de", False, ["usable fallback"]),
    ]

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "de"
    assert transcript.text == "usable fallback"
    assert [track.fetch_calls for track in tracks] == [1, 1]


async def test_empty_higher_ranked_track_falls_through() -> None:
    """Continue to a lower-ranked track after empty normalized text."""
    tracks = [
        _FakeCaptionTrack("en", False, [" \t", "\n"]),
        _FakeCaptionTrack("fr", False, ["usable fallback"]),
    ]

    transcript = await _transcription_service(_FakeCaptionProvider(tracks)).acquire(
        _source()
    )

    assert transcript.language == "fr"
    assert transcript.text == "usable fallback"


async def test_segment_whitespace_normalizes_to_plain_text() -> None:
    """Collapse Unicode whitespace while preserving text and segment order."""
    provider = _FakeCaptionProvider(
        [
            _FakeCaptionTrack(
                "en",
                False,
                ["  Hello\tworld  ", "\n", "Ça va?  déjà."],
            )
        ]
    )

    transcript = await _transcription_service(provider).acquire(_source())

    assert transcript.text == "Hello world Ça va? déjà."


@pytest.mark.parametrize(
    "tracks",
    [
        [],
        [_FakeCaptionTrack("en", False, [], error=ValueError("provider detail"))],
        [_FakeCaptionTrack("en", False, [" \t", "\n"])],
    ],
)
async def test_unusable_caption_tracks_raise_transcription_error(
    tracks: list[_FakeCaptionTrack],
) -> None:
    """Represent missing or unusable captions as Transcript Unavailable."""
    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(_FakeCaptionProvider(tracks)).acquire(_source())


async def test_listing_timeout_falls_back_to_whisper(tmp_path: Path) -> None:
    """Fall back to Whisper when Caption Track listing times out."""
    provider = _FakeCaptionProvider([], error=Timeout("sensitive provider detail"))
    downloader = _FakeAudioDownloader()
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="Recovered speech.", language="en", segment_count=1)
    )

    transcript = await _transcription_service(
        provider,
        audio_downloader=downloader,
        transcriber=transcriber,
        temp_media_dir=tmp_path,
    ).acquire(_source())

    assert transcript.method is TranscriptMethod.WHISPER
    assert transcript.text == "Recovered speech."
    assert len(downloader.calls) == 1


async def test_track_timeout_stops_ranked_fallback_and_uses_whisper(
    tmp_path: Path,
) -> None:
    """Stop ranked caption traversal and fall back after a track timeout."""
    tracks = [
        _FakeCaptionTrack("en", False, [], error=Timeout("provider detail")),
        _FakeCaptionTrack("de", False, ["must not be fetched"]),
    ]
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="Recovered speech.", language="en", segment_count=1)
    )

    transcript = await _transcription_service(
        _FakeCaptionProvider(tracks),
        audio_downloader=_FakeAudioDownloader(),
        transcriber=transcriber,
        temp_media_dir=tmp_path,
    ).acquire(_source())

    assert transcript.method is TranscriptMethod.WHISPER
    assert [track.fetch_calls for track in tracks] == [1, 0]


async def test_successful_acquisition_logs_required_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log complete successful Transcript metadata at DEBUG."""
    provider = _FakeCaptionProvider(
        [_FakeCaptionTrack("en", False, ["Hello", "world."])]
    )

    with caplog.at_level(logging.DEBUG, logger=transcription_service.__name__):
        await _transcription_service(provider).acquire(_source())

    record = next(
        item for item in caplog.records if item.getMessage() == "transcript acquired"
    )
    assert record.__dict__["stage"] == "transcription"
    assert record.__dict__["transcript_text"] == "Hello world."
    assert record.__dict__["language"] == "en"
    assert record.__dict__["method"] == "youtube_captions"
    assert record.__dict__["segment_count"] == 2


async def test_failed_and_empty_tracks_do_not_log_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Avoid successful acquisition logs for failed or empty tracks."""
    tracks = [
        _FakeCaptionTrack("en", False, [], error=ValueError("provider detail")),
        _FakeCaptionTrack("de", False, [" \t"]),
    ]

    with (
        caplog.at_level(logging.DEBUG, logger=transcription_service.__name__),
        pytest.raises(TranscriptionError),
    ):
        await _transcription_service(_FakeCaptionProvider(tracks)).acquire(_source())

    assert not any(
        item.getMessage() == "transcript acquired" for item in caplog.records
    )
    assert any(
        item.getMessage() == "caption track unavailable" for item in caplog.records
    )


class _LibrarySnippet:
    def __init__(self, text: object) -> None:
        self.text = text


class _LibraryFetchedTranscript:
    def __init__(self, snippets: Sequence[_LibrarySnippet]) -> None:
        self._snippets = tuple(snippets)

    def __iter__(self) -> Iterator[_LibrarySnippet]:
        return iter(self._snippets)


class _LibraryTrack:
    def __init__(
        self,
        language_code: str,
        is_generated: bool,
        snippets: Sequence[_LibrarySnippet],
        error: Exception | None = None,
    ) -> None:
        self.language_code = language_code
        self.is_generated = is_generated
        self._snippets = snippets
        self._error = error
        self.fetch_formatting: list[bool] = []
        self.translate_calls = 0
        self.fetch_thread_ids: list[int] = []

    def fetch(self, preserve_formatting: bool = False) -> _LibraryFetchedTranscript:
        self.fetch_formatting.append(preserve_formatting)
        self.fetch_thread_ids.append(threading.get_ident())
        if self._error is not None:
            raise self._error
        return _LibraryFetchedTranscript(self._snippets)

    def translate(self, language_code: str) -> _LibraryTrack:
        self.translate_calls += 1
        raise AssertionError(f"unexpected translation to {language_code}")


class _FakeCaptionApi:
    instances: ClassVar[list[_FakeCaptionApi]] = []
    tracks: ClassVar[Sequence[_LibraryTrack]] = ()

    def __init__(self) -> None:
        self.list_video_ids: list[str] = []
        self.list_thread_ids: list[int] = []
        self.__class__.instances.append(self)

    def list(self, video_id: str) -> Sequence[_LibraryTrack]:
        self.list_video_ids.append(video_id)
        self.list_thread_ids.append(threading.get_ident())
        return self.tracks


def _install_caption_api(
    monkeypatch: pytest.MonkeyPatch,
    tracks: Sequence[_LibraryTrack],
) -> None:
    _FakeCaptionApi.instances.clear()
    _FakeCaptionApi.tracks = tracks
    monkeypatch.setattr(transcription_service, "YouTubeTranscriptApi", _FakeCaptionApi)


def test_youtube_caption_adapter_uses_video_id_and_preserves_track_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass only the video ID and expose original track metadata."""
    tracks = [
        _LibraryTrack("en-GB", False, [_LibrarySnippet("Hello")]),
        _LibraryTrack("de", True, [_LibrarySnippet("Hallo")]),
    ]
    _install_caption_api(monkeypatch, tracks)

    wrapped_tracks = YouTubeCaptionProvider().list_tracks(_VIDEO_ID)

    assert len(_FakeCaptionApi.instances) == 1
    assert _FakeCaptionApi.instances[0].list_video_ids == [_VIDEO_ID]
    assert [(track.language_code, track.is_generated) for track in wrapped_tracks] == [
        ("en-GB", False),
        ("de", True),
    ]


async def test_caption_listing_and_fetch_share_one_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one API instance and all caption operations on one worker thread."""
    track = _LibraryTrack("en", False, [_LibrarySnippet("Hello")])
    _install_caption_api(monkeypatch, [track])

    transcript = await _transcription_service(YouTubeCaptionProvider()).acquire(
        _source()
    )

    api = _FakeCaptionApi.instances[0]
    assert transcript.text == "Hello"
    assert len(_FakeCaptionApi.instances) == 1
    assert api.list_thread_ids == track.fetch_thread_ids


def test_youtube_caption_adapter_disables_provider_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch snippets with provider formatting disabled."""
    track = _LibraryTrack("en", False, [_LibrarySnippet("Hello")])
    _install_caption_api(monkeypatch, [track])

    wrapped_track = YouTubeCaptionProvider().list_tracks(_VIDEO_ID)[0]

    assert wrapped_track.fetch_segments() == ("Hello",)
    assert track.fetch_formatting == [False]


def test_caption_acquisition_never_requests_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the selected original track without requesting translation."""
    track = _LibraryTrack("en", False, [_LibrarySnippet("Original")])
    _install_caption_api(monkeypatch, [track])

    wrapped_track = YouTubeCaptionProvider().list_tracks(_VIDEO_ID)[0]

    assert wrapped_track.fetch_segments() == ("Original",)
    assert track.translate_calls == 0


@pytest.mark.parametrize(
    "provider_error",
    [
        CouldNotRetrieveTranscript(_VIDEO_ID),
        ElementTree.ParseError("malformed caption payload"),
    ],
)
async def test_caption_provider_failures_use_stable_transcription_error(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: Exception,
) -> None:
    """Hide provider exception details behind Transcript Unavailable."""
    _install_caption_api(
        monkeypatch,
        [_LibraryTrack("en", False, [], error=provider_error)],
    )

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ) as error:
        await _transcription_service(YouTubeCaptionProvider()).acquire(_source())

    assert _VIDEO_ID not in str(error.value)
    assert "malformed caption payload" not in str(error.value)


async def test_missing_caption_metadata_maps_to_stable_transcription_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hide missing provider track data behind Transcript Unavailable."""
    _install_caption_api(monkeypatch, [cast(_LibraryTrack, object())])

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(YouTubeCaptionProvider()).acquire(_source())


async def test_malformed_caption_track_falls_through_to_valid_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue past malformed track metadata to a valid Caption Track."""
    _install_caption_api(
        monkeypatch,
        [
            cast(_LibraryTrack, object()),
            _LibraryTrack("en", False, [_LibrarySnippet("usable")]),
        ],
    )

    transcript = await _transcription_service(YouTubeCaptionProvider()).acquire(
        _source()
    )

    assert transcript.text == "usable"


async def test_caption_provider_timeout_falls_back_to_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fall back to Whisper after a caption fetch timeout."""
    _install_caption_api(
        monkeypatch,
        [_LibraryTrack("en", False, [], error=Timeout("provider timeout"))],
    )
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="Recovered speech.", language="en", segment_count=1)
    )

    transcript = await _transcription_service(
        YouTubeCaptionProvider(),
        audio_downloader=_FakeAudioDownloader(),
        transcriber=transcriber,
        temp_media_dir=tmp_path,
    ).acquire(_source())

    assert transcript.method is TranscriptMethod.WHISPER
    assert transcript.text == "Recovered speech."


class _FakeAudioDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[Source, Path]] = []

    def download(self, source: Source, destination: Path) -> Path:
        self.calls.append((source, destination))
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        return audio_path


class _FakeWhisperTranscriber:
    def __init__(self, result: WhisperResult) -> None:
        self.result = result
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.calls.append(audio_path)
        return self.result


class _UnavailableWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> WhisperResult:
        raise _WhisperProviderFailure("model failure")


def _transcription_service(
    provider: _FakeCaptionProvider | YouTubeCaptionProvider,
    audio_downloader: AudioDownloader | None = None,
    transcriber: WhisperTranscriber | None = None,
    temp_media_dir: Path | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> TranscriptionService:
    return TranscriptionService(
        provider=provider,
        audio_downloader=(
            audio_downloader if audio_downloader is not None else _FakeAudioDownloader()
        ),
        transcriber=(
            transcriber if transcriber is not None else _UnavailableWhisperTranscriber()
        ),
        temp_media_dir=(
            temp_media_dir
            if temp_media_dir is not None
            else Path(tempfile.gettempdir()) / "reelio-test-transcription"
        ),
        semaphore=semaphore if semaphore is not None else asyncio.Semaphore(1),
    )


async def test_captionless_source_falls_back_to_whisper(tmp_path: Path) -> None:
    """Acquire normalized Whisper text when no Caption Track is usable."""
    downloader = _FakeAudioDownloader()
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="Hello from audio.", language="en", segment_count=2)
    )
    service = TranscriptionService(
        provider=_FakeCaptionProvider([]),
        audio_downloader=downloader,
        transcriber=transcriber,
        temp_media_dir=tmp_path,
        semaphore=asyncio.Semaphore(1),
    )

    transcript = await service.acquire(_source())

    assert transcript == Transcript(
        text="Hello from audio.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    assert len(downloader.calls) == 1
    assert len(transcriber.calls) == 1
    assert list(tmp_path.iterdir()) == []


class _TimeoutWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> WhisperResult:
        raise Timeout("whisper timeout")


class _PartialDownloadFailure:
    def download(self, source: Source, destination: Path) -> Path:
        partial_path = destination / "audio.part"
        partial_path.write_bytes(b"partial")
        raise _WhisperProviderFailure("download provider detail")


async def test_caption_timeout_then_whisper_failure_maps_to_502(
    tmp_path: Path,
) -> None:
    """Map ordinary dual acquisition failure to Transcript Unavailable."""
    provider = _FakeCaptionProvider([], error=Timeout("caption timeout"))

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            provider,
            audio_downloader=_FakeAudioDownloader(),
            transcriber=_UnavailableWhisperTranscriber(),
            temp_media_dir=tmp_path,
        ).acquire(_source())


async def test_caption_timeout_then_whisper_timeout_maps_to_504(
    tmp_path: Path,
) -> None:
    """Map terminal Whisper timeout after a caption timeout to HTTP 504."""
    provider = _FakeCaptionProvider([], error=Timeout("caption timeout"))

    with pytest.raises(
        PipelineTimeoutError,
        match=r"^Transcript acquisition timed out\.$",
    ):
        await _transcription_service(
            provider,
            audio_downloader=_FakeAudioDownloader(),
            transcriber=_TimeoutWhisperTranscriber(),
            temp_media_dir=tmp_path,
        ).acquire(_source())


@pytest.mark.parametrize("text", ["", " \t\n"])
async def test_empty_whisper_output_maps_to_transcription_error(
    tmp_path: Path,
    text: str,
) -> None:
    """Reject empty or whitespace-only Whisper output."""
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text=text, language="en", segment_count=1)
    )

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_FakeAudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=tmp_path,
        ).acquire(_source())


async def test_zero_segment_whisper_output_maps_to_transcription_error(
    tmp_path: Path,
) -> None:
    """Reject a non-empty Whisper payload that reports zero segments."""
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="speech", language="en", segment_count=0)
    )

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_FakeAudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=tmp_path,
        ).acquire(_source())


async def test_whisper_success_logs_audio_fields_and_cleans_media(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Log the complete Whisper acquisition record before cleanup."""
    transcriber = _FakeWhisperTranscriber(
        WhisperResult(text="Hello from audio.", language="en", segment_count=2)
    )

    with caplog.at_level(logging.DEBUG, logger=transcription_service.__name__):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_FakeAudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=tmp_path,
        ).acquire(_source())

    record = next(
        item for item in caplog.records if item.getMessage() == "transcript acquired"
    )
    assert record.__dict__["transcript_text"] == "Hello from audio."
    assert record.__dict__["language"] == "en"
    assert record.__dict__["method"] == "whisper"
    assert record.__dict__["segment_count"] == 2
    assert record.__dict__["audio_size_bytes"] == 5
    assert not Path(record.__dict__["audio_path"]).exists()
    assert list(tmp_path.iterdir()) == []


async def test_download_failure_removes_partial_request_media(tmp_path: Path) -> None:
    """Remove partial audio and its request directory after download failure."""
    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_PartialDownloadFailure(),
            transcriber=_UnavailableWhisperTranscriber(),
            temp_media_dir=tmp_path,
        ).acquire(_source())

    assert list(tmp_path.iterdir()) == []


async def test_transcription_failure_removes_completed_audio(tmp_path: Path) -> None:
    """Remove completed audio and its request directory after model failure."""
    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_FakeAudioDownloader(),
            transcriber=_UnavailableWhisperTranscriber(),
            temp_media_dir=tmp_path,
        ).acquire(_source())

    assert list(tmp_path.iterdir()) == []


class _FakeWhisperModel:
    def __init__(self, texts: Sequence[object], language: object) -> None:
        self._texts = texts
        self._language = language
        self.transcribe_calls = 0
        self.yielded_segments = 0

    def transcribe(
        self,
        audio: str,
    ) -> tuple[Iterator[object], SimpleNamespace]:
        self.transcribe_calls += 1

        def segments() -> Iterator[object]:
            for text in self._texts:
                self.yielded_segments += 1
                yield SimpleNamespace(text=text)

        return segments(), SimpleNamespace(language=self._language)


async def test_faster_whisper_adapter_exhausts_segments_once() -> None:
    """Consume the lazy faster-whisper segment iterator exactly once."""
    model = _FakeWhisperModel(
        ["  Hello\tworld  ", "\nÇa va? déjà."],
        "fr",
    )

    result = FasterWhisperTranscriber(model).transcribe(Path("audio.webm"))

    assert result == WhisperResult(
        text="Hello world Ça va? déjà.",
        language="fr",
        segment_count=2,
    )
    assert model.transcribe_calls == 1
    assert model.yielded_segments == 2


@pytest.mark.parametrize(
    ("texts", "language"),
    [(["hello"], ""), ([object()], "en")],
)
async def test_malformed_whisper_output_maps_to_transcription_error(
    tmp_path: Path,
    texts: Sequence[object],
    language: object,
) -> None:
    """Hide malformed Whisper payloads behind Transcript Unavailable."""
    model = _FakeWhisperModel(texts, language)

    with pytest.raises(
        TranscriptionError,
        match=r"^Transcript is unavailable for this video\.$",
    ):
        await _transcription_service(
            _FakeCaptionProvider([]),
            audio_downloader=_FakeAudioDownloader(),
            transcriber=FasterWhisperTranscriber(model),
            temp_media_dir=tmp_path,
        ).acquire(_source())


class _FakeDownloadYoutubeDL:
    def __init__(
        self,
        options: object,
        result: object,
        prepared_path: str,
        error: Exception | None = None,
    ) -> None:
        self.options = options
        self.result = result
        self.prepared_path = prepared_path
        self.error = error
        self.extract_calls: list[tuple[str, bool]] = []

    def __enter__(self) -> _FakeDownloadYoutubeDL:
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
        if self.error is not None:
            raise self.error
        Path(self.prepared_path).write_bytes(b"audio")
        return self.result

    def prepare_filename(self, info: Mapping[str, object]) -> str:
        return self.prepared_path


def test_yt_dlp_audio_adapter_requests_native_audio_and_validates_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Request native best audio and accept only the completed local file."""
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    prepared_path = str(request_directory / "audio.webm")
    fake = _FakeDownloadYoutubeDL({}, {"id": _VIDEO_ID}, prepared_path)

    def make_youtube_dl(options: object) -> _FakeDownloadYoutubeDL:
        fake.options = options
        return fake

    monkeypatch.setattr(yt_dlp, "YoutubeDL", make_youtube_dl)

    result = YtDlpAudioDownloader().download(_source(), request_directory)

    assert result == Path(prepared_path)
    assert fake.extract_calls == [(_CANONICAL_URL, True)]
    assert fake.options["format"] == "bestaudio/best"  # type: ignore[index]
    assert "postprocessors" not in fake.options  # type: ignore[operator]
    assert str(request_directory) in fake.options["outtmpl"]  # type: ignore[index]


def test_yt_dlp_audio_adapter_rejects_out_of_directory_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a completed provider path outside the private request directory."""
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    outside_path = tmp_path / "outside.webm"
    fake = _FakeDownloadYoutubeDL({}, {"id": _VIDEO_ID}, str(outside_path))
    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda options: fake)

    with pytest.raises(_WhisperProviderFailure):
        YtDlpAudioDownloader().download(_source(), request_directory)


def test_yt_dlp_audio_adapter_rejects_multi_entry_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject yt-dlp metadata that represents more than one video."""
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    fake = _FakeDownloadYoutubeDL(
        {},
        {"entries": [{"id": _VIDEO_ID}, {"id": "another-video"}]},
        str(request_directory / "audio.webm"),
    )
    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda options: fake)

    with pytest.raises(_WhisperProviderFailure):
        YtDlpAudioDownloader().download(_source(), request_directory)


def test_yt_dlp_audio_adapter_preserves_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map nested typed yt-dlp timeout information to the timeout signal."""
    request_directory = tmp_path / "request"
    request_directory.mkdir()
    provider_timeout = Timeout("provider timeout")
    error = DownloadError(
        "redacted provider detail",
        exc_info=(Timeout, provider_timeout, cast(TracebackType, None)),
    )
    fake = _FakeDownloadYoutubeDL(
        {},
        {"id": _VIDEO_ID},
        str(request_directory / "audio.webm"),
        error=error,
    )
    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda options: fake)

    with pytest.raises(_WhisperProviderTimeout):
        YtDlpAudioDownloader().download(_source(), request_directory)


class _BlockingWhisperTranscriber:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.calls.append(audio_path)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test worker was not released")
        return WhisperResult(
            text="serialized speech",
            language="en",
            segment_count=1,
        )


async def test_whisper_fallback_queues_behind_one_semaphore(
    tmp_path: Path,
) -> None:
    """Serialize concurrent Whisper fallbacks and let both complete."""
    transcriber = _BlockingWhisperTranscriber()
    service = _transcription_service(
        _FakeCaptionProvider([]),
        audio_downloader=_FakeAudioDownloader(),
        transcriber=transcriber,
        temp_media_dir=tmp_path,
    )

    first = asyncio.create_task(service.acquire(_source()))
    assert await asyncio.to_thread(transcriber.started.wait, 5)
    second = asyncio.create_task(service.acquire(_source()))
    await asyncio.sleep(0)

    assert len(transcriber.calls) == 1

    transcriber.release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.text == "serialized speech"
    assert second_result.text == "serialized speech"
    assert len(transcriber.calls) == 2
    assert list(tmp_path.iterdir()) == []


async def test_cancellation_while_queued_starts_no_fallback(tmp_path: Path) -> None:
    """Cancel a queued request without creating media or starting a worker."""
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    downloader = _FakeAudioDownloader()
    service = _transcription_service(
        _FakeCaptionProvider([]),
        audio_downloader=downloader,
        transcriber=_FakeWhisperTranscriber(
            WhisperResult(text="unused", language="en", segment_count=1)
        ),
        temp_media_dir=tmp_path,
        semaphore=semaphore,
    )

    queued = asyncio.create_task(service.acquire(_source()))
    await asyncio.sleep(0)
    queued.cancel()

    with pytest.raises(asyncio.CancelledError):
        await queued
    semaphore.release()

    assert downloader.calls == []
    assert list(tmp_path.iterdir()) == []


async def test_active_cancellation_waits_for_cleanup_before_next_worker(
    tmp_path: Path,
) -> None:
    """Finish canceled native work before releasing the shared semaphore."""
    transcriber = _BlockingWhisperTranscriber()
    service = _transcription_service(
        _FakeCaptionProvider([]),
        audio_downloader=_FakeAudioDownloader(),
        transcriber=transcriber,
        temp_media_dir=tmp_path,
    )

    first = asyncio.create_task(service.acquire(_source()))
    assert await asyncio.to_thread(transcriber.started.wait, 5)
    first.cancel()
    await asyncio.sleep(0)

    assert not first.done()
    assert len(list(tmp_path.iterdir())) == 1

    second = asyncio.create_task(service.acquire(_source()))
    await asyncio.sleep(0)
    assert len(transcriber.calls) == 1

    transcriber.release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    second_result = await second

    assert second_result.text == "serialized speech"
    assert len(transcriber.calls) == 2
    assert list(tmp_path.iterdir()) == []
