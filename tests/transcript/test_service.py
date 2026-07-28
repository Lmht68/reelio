import pytest

from src.transcript.exceptions import (
    TranscriptInvalidURLError,
    TranscriptNotFoundError,
    TranscriptUnsupportedPlatformError,
)
from src.transcript.models import Platform, Transcript
from src.transcript.providers.whisper import WhisperProvider
from src.transcript.providers.youtube import YouTubeProvider
from src.transcript.service import TranscriptService


class TestTranscriptServiceInit:
    def test_creates_providers(self):
        service = TranscriptService()
        assert isinstance(service._youtube_provider, YouTubeProvider)
        assert isinstance(service._whisper_provider, WhisperProvider)

    def test_passes_whisper_config(self):
        service = TranscriptService(
            whisper_model_size="tiny",
            whisper_device="cuda",
            whisper_compute_type="float16",
            temp_dir="/tmp/test",
        )
        assert service._whisper_provider._model_size == "tiny"
        assert service._whisper_provider._device == "cuda"
        assert service._whisper_provider._compute_type == "float16"
        assert service._whisper_provider._temp_dir == "/tmp/test"

    def test_passes_whisper_limit_config(self):
        service = TranscriptService(
            whisper_max_concurrent=5,
            whisper_max_duration_seconds=60,
        )
        assert service._whisper_provider._max_concurrent == 5
        assert service._whisper_provider._max_duration_seconds == 60


class TestTranscriptServiceGetProvider:
    def test_youtube_routes_to_youtube_provider(self):
        service = TranscriptService()
        provider = service._get_provider(Platform.YOUTUBE)
        assert isinstance(provider, YouTubeProvider)

    def test_other_platforms_route_to_whisper(self):
        service = TranscriptService()
        for platform in [Platform.INSTAGRAM, Platform.FACEBOOK, Platform.TIKTOK, Platform.X, Platform.THREADS]:
            provider = service._get_provider(platform)
            assert isinstance(provider, WhisperProvider)

    def test_unknown_routes_to_whisper(self):
        service = TranscriptService()
        provider = service._get_provider(Platform.UNKNOWN)
        assert isinstance(provider, WhisperProvider)


class TestTranscriptServiceExtract:
    @pytest.fixture
    def mock_extract(self, mocker):
        """Create mock extract methods that return synthetic results."""

        async def youtube_extract(url: str) -> Transcript:
            return Transcript(
                full_text="YouTube transcript",
                language="en",
            )

        async def whisper_extract(url: str) -> Transcript:
            return Transcript(
                full_text="Whisper transcript",
                language="en",
            )

        return youtube_extract, whisper_extract

    @pytest.mark.anyio
    async def test_extract_youtube(self, mocker, mock_extract, sample_youtube_url):
        youtube_extract, whisper_extract = mock_extract
        mocker.patch.object(YouTubeProvider, "extract", side_effect=youtube_extract)

        service = TranscriptService()
        result = await service.extract(sample_youtube_url)

        assert result.platform == Platform.YOUTUBE
        assert result.transcript.full_text == "YouTube transcript"

    @pytest.mark.anyio
    async def test_extract_instagram(self, mocker, mock_extract, sample_instagram_url):
        youtube_extract, whisper_extract = mock_extract
        mocker.patch.object(WhisperProvider, "extract", side_effect=whisper_extract)

        service = TranscriptService()
        result = await service.extract(sample_instagram_url)

        assert result.platform == Platform.INSTAGRAM
        assert result.transcript.full_text == "Whisper transcript"

    @pytest.mark.anyio
    async def test_extract_invalid_url(self):
        service = TranscriptService()
        with pytest.raises(TranscriptInvalidURLError):
            await service.extract("not-a-valid-url")

    @pytest.mark.anyio
    async def test_extract_unsupported_platform(self):
        service = TranscriptService()
        with pytest.raises(TranscriptUnsupportedPlatformError):
            await service.extract("https://vimeo.com/123456")

    @pytest.mark.anyio
    async def test_extract_empty_url(self):
        service = TranscriptService()
        with pytest.raises(TranscriptInvalidURLError):
            await service.extract("")

    @pytest.mark.anyio
    async def test_extract_propagates_provider_error(self, mocker, sample_youtube_url):
        mocker.patch.object(
            YouTubeProvider,
            "extract",
            side_effect=TranscriptNotFoundError("No transcript"),
        )

        service = TranscriptService()
        with pytest.raises(TranscriptNotFoundError):
            await service.extract(sample_youtube_url)
