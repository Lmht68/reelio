"""HTTP contract tests for the extraction endpoint."""

import asyncio
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import NoReturn, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    EnrichmentError,
    EntityExtractionError,
    ExtractionError,
    InvalidSourceError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    UnsupportedPlatformError,
)
from reelio.extraction.router import get_pipeline
from reelio.extraction.schemas import ExtractResponse
from reelio.extraction.service import ExtractionPipeline, ExtractionPipelineProtocol
from reelio.extraction.services.transcription.acquisition import (
    WhisperResult,
    _WhisperProviderFailure,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.services.transcription.inspection import ExtractedMetadata
from reelio.extraction.services.transcription.service import (
    SourceMetadataService,
    TranscriptionService,
)
from reelio.extraction.types import PipelineResult, ResultStatus
from reelio.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Clear dependency overrides before and after every extraction test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _RaisingPipeline:
    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def run(self, url: str) -> PipelineResult:
        raise self._exception


_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _MetadataExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        return ExtractedMetadata(
            {
                "id": _VIDEO_ID,
                "title": "Router test video",
                "description": "A complete router test description.",
                "channel": "Router test channel",
                "duration": 42.2,
            }
        )


class _SocialMetadataExtractor:
    def __init__(
        self,
        metadata: dict[str, object],
    ) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        self.calls.append(canonical_url)
        return ExtractedMetadata(self.metadata)


def _settings() -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(_env_file=None)


class _CaptionTrack:
    def __init__(self, language_code: str, segments: Sequence[str]) -> None:
        self.language_code = language_code
        self.is_generated = False
        self._segments = segments

    def fetch_segments(self) -> Sequence[str]:
        return self._segments


class _CaptionProvider:
    def __init__(self, tracks: Sequence[_CaptionTrack]) -> None:
        self._tracks = tracks

    def list_tracks(self, video_id: str) -> Sequence[_CaptionTrack]:
        return self._tracks


class _AudioDownloader:
    def download(self, source: object, destination: Path) -> Path:
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        return audio_path


class _FailingWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> NoReturn:
        raise _WhisperProviderFailure("model failure")


class _FixedWhisperTranscriber:
    def __init__(self, result: WhisperResult) -> None:
        self.result = result

    def transcribe(self, audio_path: Path) -> WhisperResult:
        return self.result


class _BlockingWhisperTranscriber:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test worker was not released")
        return WhisperResult(text="concurrent speech", language="en", segment_count=1)


def _transcription_service(provider: _CaptionProvider) -> TranscriptionService:
    return TranscriptionService(
        provider=provider,
        audio_downloader=_AudioDownloader(),
        transcriber=_FailingWhisperTranscriber(),
        temp_media_dir=_settings().temp_media_dir,
        semaphore=asyncio.Semaphore(1),
    )


def _install_pipeline(application: FastAPI, pipeline: ExtractionPipelineProtocol) -> None:
    application.dependency_overrides[get_pipeline] = lambda: pipeline


async def test_extract_returns_schema_valid_response(client: AsyncClient) -> None:
    """Return real source metadata and all deterministic result branches."""
    metadata_extractor = _MetadataExtractor()
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=metadata_extractor,
            settings=_settings(),
        ),
        _transcription_service(
            _CaptionProvider([_CaptionTrack("en-GB", ["Router", "caption text."])])
        ),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == "youtube"
    assert payload.source.video_id == _VIDEO_ID
    assert payload.source.url == _CANONICAL_URL
    assert payload.source.title == "Router test video"
    assert payload.source.description == "A complete router test description."
    assert payload.source.channel == "Router test channel"
    assert payload.source.duration_seconds == 43
    assert metadata_extractor.calls == [_CANONICAL_URL]
    assert payload.transcript.language == "en-GB"
    assert payload.transcript.method == "youtube_captions"
    assert payload.transcript.text == "Router caption text."
    assert {result.status for result in payload.results} == {
        ResultStatus.RESOLVED,
        ResultStatus.AMBIGUOUS,
        ResultStatus.UNRESOLVED,
    }
    assert all(len(result.mentioned_as) >= 1 for result in payload.results)
    assert all(len(result.evidence) >= 1 for result in payload.results)

    ambiguous = next(
        result for result in payload.results if result.status is ResultStatus.AMBIGUOUS
    )
    assert ambiguous.movie is None
    assert len(ambiguous.candidates) == 3
    assert len({candidate.resolution_score for candidate in ambiguous.candidates}) == 3

    unresolved = next(
        result for result in payload.results if result.status is ResultStatus.UNRESOLVED
    )
    assert unresolved.resolution_confidence is None
    assert unresolved.movie is None
    assert unresolved.candidates == []


async def test_extract_maps_unavailable_captions_to_502(
    client: AsyncClient,
) -> None:
    """Map a valid Source with no usable captions to Transcript Unavailable."""
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        _transcription_service(_CaptionProvider([])),
    )
    _install_pipeline(app, pipeline)

    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Transcript is unavailable for this video.",
        }
    }


async def test_extract_returns_whisper_transcript(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Serialize a successful Whisper Transcript through the unchanged schema."""
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        TranscriptionService(
            provider=_CaptionProvider([]),
            audio_downloader=_AudioDownloader(),
            transcriber=_FixedWhisperTranscriber(
                WhisperResult(
                    text="Spoken audio text.",
                    language="en",
                    segment_count=2,
                )
            ),
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)
    response = await client.post(
        "/api/extract",
        json={"url": _CANONICAL_URL},
    )

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.transcript.text == "Spoken audio text."
    assert payload.transcript.language == "en"
    assert payload.transcript.method == "whisper"


@pytest.mark.parametrize(
    (
        "submitted_url",
        "provider_url",
        "canonical_url",
        "extractor_key",
        "video_id",
        "platform",
    ),
    [
        (
            "https://www.instagram.com/reel/ABC123",
            "https://www.instagram.com/reel/ABC123",
            "https://www.instagram.com/reel/ABC123",
            "Instagram",
            "ABC123",
            "instagram",
        ),
        (
            "https://www.facebook.com/reel/123456789",
            "https://www.facebook.com/reel/123456789",
            "https://www.facebook.com/reel/123456789",
            "FacebookReel",
            "123456789",
            "facebook",
        ),
        (
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "TikTok",
            "1234567890123456789",
            "tiktok",
        ),
        (
            "https://twitter.com/creator/status/1234567890123456789",
            "https://twitter.com/creator/status/1234567890123456789",
            "https://twitter.com/creator/status/1234567890123456789",
            "Twitter",
            "1234567890123456789",
            "x",
        ),
    ],
)
async def test_social_sources_serialize_unchanged_response_schema(
    client: AsyncClient,
    tmp_path: Path,
    submitted_url: str,
    provider_url: str,
    canonical_url: str,
    extractor_key: str,
    video_id: str,
    platform: str,
) -> None:
    """Serialize every social Source with a direct Whisper Transcript."""
    extractor = _SocialMetadataExtractor(
        {
            "id": video_id,
            "extractor_key": extractor_key,
            "webpage_url": canonical_url,
            "title": "Social router video",
            "description": "Social router description",
            "channel": "Social router channel",
            "duration": 42.2,
            "formats": [{"vcodec": "avc1"}],
        }
    )
    pipeline = ExtractionPipeline(
        SourceMetadataService(extractor=extractor, settings=_settings()),
        TranscriptionService(
            provider=_CaptionProvider([_CaptionTrack("en", ["must", "not", "run"])]),
            audio_downloader=_AudioDownloader(),
            transcriber=_FixedWhisperTranscriber(
                WhisperResult(
                    text="Social router speech.",
                    language="en",
                    segment_count=1,
                )
            ),
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)

    response = await client.post("/api/extract", json={"url": submitted_url})

    assert response.status_code == 200
    payload = ExtractResponse.model_validate(response.json())
    assert payload.source.platform == platform
    assert payload.source.video_id == video_id
    assert payload.source.url == canonical_url
    assert payload.source.title == "Social router video"
    assert payload.source.description == "Social router description"
    assert payload.source.channel == "Social router channel"
    assert payload.source.duration_seconds == 43
    assert payload.transcript.method == "whisper"
    assert payload.transcript.text == "Social router speech."
    assert extractor.calls == [provider_url]


async def test_concurrent_whisper_http_requests_queue_and_succeed(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Queue concurrent endpoint fallbacks behind one shared service semaphore."""
    transcriber = _BlockingWhisperTranscriber()
    pipeline = ExtractionPipeline(
        SourceMetadataService(
            extractor=_MetadataExtractor(),
            settings=_settings(),
        ),
        TranscriptionService(
            provider=_CaptionProvider([]),
            audio_downloader=_AudioDownloader(),
            transcriber=transcriber,
            temp_media_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
        ),
    )
    _install_pipeline(app, pipeline)

    first = asyncio.create_task(client.post("/api/extract", json={"url": _CANONICAL_URL}))
    assert await asyncio.to_thread(transcriber.started.wait, 5)
    second = asyncio.create_task(client.post("/api/extract", json={"url": _CANONICAL_URL}))
    await asyncio.sleep(0)

    assert transcriber.calls == 1

    transcriber.release.set()
    first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert transcriber.calls == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (InvalidSourceError("invalid source"), 400, "invalid_source"),
        (
            UnsupportedPlatformError("unsupported platform"),
            400,
            "unsupported_platform",
        ),
        (SourceUnavailableError("source unavailable"), 404, "source_unavailable"),
        (
            DurationLimitExceededError("duration limit exceeded"),
            413,
            "duration_limit_exceeded",
        ),
        (
            MetadataProviderError("Unable to retrieve YouTube metadata."),
            502,
            "metadata_provider_failed",
        ),
        (TranscriptionError("transcription failed"), 502, "transcription_failed"),
        (
            EntityExtractionError("entity extraction failed"),
            502,
            "entity_extraction_failed",
        ),
        (EnrichmentError("enrichment failed"), 502, "enrichment_failed"),
        (PipelineTimeoutError("pipeline timed out"), 504, "pipeline_timeout"),
    ],
)
async def test_extraction_errors_map_to_contract(
    client: AsyncClient,
    exception: ExtractionError,
    expected_status: int,
    expected_code: str,
) -> None:
    """Map every extraction domain error to its stable HTTP contract."""
    _install_pipeline(app, _RaisingPipeline(exception))

    response = await client.post(
        "/api/extract",
        json={"url": "https://www.youtube.com/watch?v=anything"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"error": {"code": expected_code, "message": str(exception)}}


@pytest.mark.parametrize("payload", [{}, {"url": 123}, {"url": ""}])
async def test_malformed_requests_keep_fastapi_422_contract(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    """Keep FastAPI validation responses for malformed request bodies."""
    _install_pipeline(app, _RaisingPipeline(RuntimeError("unused")))
    response = await client.post("/api/extract", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


async def test_unhandled_failures_do_not_leak_internals() -> None:
    """Return a generic 500 response when the pipeline raises unexpectedly."""
    _install_pipeline(
        app,
        _RaisingPipeline(RuntimeError("sensitive database path /var/reelio/secret.db")),
    )
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/extract",
            json={"url": "https://www.youtube.com/watch?v=anything"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred.",
        }
    }
    assert "sensitive database path" not in response.text


async def test_extract_is_documented_in_openapi(client: AsyncClient) -> None:
    """Document the endpoint request and configured error responses."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/extract"]["post"]
    responses = operation["responses"]
    assert {"200", "400", "404", "413", "500", "502", "504", "422"} <= set(responses)
    for status_code in ("400", "404", "413", "500", "502", "504"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/ExtractRequest"}
    source_properties = document["components"]["schemas"]["Source"]["properties"]
    assert {
        "platform",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "duration_seconds",
    } <= set(source_properties)
    assert set(document["components"]["schemas"]["Platform"]["enum"]) == {
        "youtube",
        "instagram",
        "facebook",
        "tiktok",
        "x",
    }
    assert "YouTube, Instagram, Facebook, TikTok, or X" in operation["description"]
