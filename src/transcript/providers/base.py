from abc import ABC, abstractmethod

from src.transcript.models import TranscriptResult


class TranscriptProvider(ABC):
    """Interface that every transcript platform provider must implement."""

    @abstractmethod
    async def extract(self, url: str) -> TranscriptResult:
        """Extract a transcript from the given video URL.

        Args:
            url: The full URL of the video/reel.

        Returns:
            A TranscriptResult containing the full text, optional segments,
            detected language, and platform metadata.

        Raises:
            TranscriptNotFoundError: No transcript exists for this video.
            TranscriptDownloadError: Failed to download audio.
            TranscriptTranscriptionError: STT processing failed.
            TranscriptInvalidURLError: The URL is not valid.
        """
        ...
