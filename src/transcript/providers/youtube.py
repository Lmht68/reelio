import asyncio
import logging
import re
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.transcript.exceptions import (
    TranscriptInvalidURLError,
    TranscriptNotFoundError,
)
from src.transcript.models import Transcript
from src.transcript.providers.base import TranscriptProvider

logger = logging.getLogger(__name__)

# Regex patterns for extracting video IDs from YouTube URLs.
# The (?<!\w) lookbehind ensures we match youtube.com as a proper domain,
# not as a substring of another domain like "notyoutube.com".
_YOUTUBE_ID_PATTERNS = [
    re.compile(
        r"(?:https?://)?(?<!\w)(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?<!\w)(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?<!\w)(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]{11})"
    ),
    re.compile(r"(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})"),
]


class YouTubeProvider(TranscriptProvider):
    """Extracts transcripts from YouTube videos using youtube-transcript-api."""

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Parse a YouTube URL and return the 11-character video ID.

        Raises TranscriptInvalidURLError if no valid ID is found.
        """
        for pattern in _YOUTUBE_ID_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        raise TranscriptInvalidURLError(
            f"Could not extract YouTube video ID from URL: {url}"
        )

    async def extract(self, url: str) -> Transcript:
        video_id = self.extract_video_id(url)

        try:
            transcript_data = await asyncio.to_thread(
                self._fetch_transcript, video_id
            )
        except VideoUnavailable as exc:
            raise TranscriptNotFoundError(
                f"YouTube video is unavailable: {video_id}"
            ) from exc
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            raise TranscriptNotFoundError(
                f"No transcript available for YouTube video: {video_id}"
            ) from exc

        full_text = " ".join(
            item.text.replace("\n", " ").strip() for item in transcript_data
        )

        return Transcript(
            full_text=full_text,
            language="en",
        )

    @staticmethod
    def _fetch_transcript(video_id: str) -> Any:
        """Fetch transcript data synchronously for the given video ID.

        Uses the instance-based API: list() -> find_transcript() -> fetch().
        """
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript(["en"])
        return transcript.fetch()
