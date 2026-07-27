from abc import ABC, abstractmethod

from src.transcript.models import Transcript


class TranscriptProvider(ABC):
    """Interface that every transcript platform provider must implement."""

    @abstractmethod
    async def extract(self, url: str) -> Transcript:
        """Extract a transcript from the given video URL.

        Args:
            url: The full URL of the video/reel.

        Returns:
            A Transcript containing the full text and detected language

        Raises:
            TranscriptNotFoundError: No transcript exists for this video.
            TranscriptDownloadError: Failed to download audio.
            TranscriptTranscriptionError: STT processing failed.
            TranscriptInvalidURLError: The URL is not valid.
        """
        ...
