import logging

from src.transcript.factory import detect_platform_strict, validate_url
from src.transcript.models import Platform, TranscriptResult
from src.transcript.providers.base import TranscriptProvider
from src.transcript.providers.whisper import WhisperProvider
from src.transcript.providers.youtube import YouTubeProvider

logger = logging.getLogger(__name__)


class TranscriptService:
    """High-level service for extracting transcripts from any supported video URL."""

    def __init__(
        self,
        whisper_model_size: str = "base",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        temp_dir: str | None = None,
        whisper_max_concurrent: int = 2,
        whisper_max_duration_seconds: int = 600,
    ):
        self._youtube_provider = YouTubeProvider()
        self._whisper_provider = WhisperProvider(
            model_size=whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
            temp_dir=temp_dir,
            max_concurrent=whisper_max_concurrent,
            max_duration_seconds=whisper_max_duration_seconds,
        )

    def _get_provider(self, platform: Platform) -> TranscriptProvider:
        """Route a platform to its appropriate provider."""
        if platform == Platform.YOUTUBE:
            return self._youtube_provider
        # All other supported platforms use Whisper (download + STT)
        return self._whisper_provider

    async def extract(self, url: str) -> TranscriptResult:
        """Extract a transcript from the given video URL.

        Automatically detects the platform and uses the appropriate provider.
        """
        validated_url = validate_url(url)
        platform = detect_platform_strict(validated_url)
        provider = self._get_provider(platform)

        logger.info(
            "Extracting transcript: platform=%s, url=%s",
            platform.value,
            validated_url,
        )
        transcript = await provider.extract(validated_url)

        return TranscriptResult(transcript=transcript, platform=platform, source_url=validated_url)
