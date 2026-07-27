from unittest.mock import MagicMock

import pytest
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
from src.transcript.providers.youtube import YouTubeProvider


class TestYouTubeProviderExtractVideoID:
    @pytest.mark.parametrize("url,expected_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abc123def45", "abc123def45"),
        ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30", "dQw4w9WgXcQ"),
        ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ])
    def test_extract_video_id(self, url, expected_id):
        assert YouTubeProvider.extract_video_id(url) == expected_id

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/",
        "https://youtube.com/channel/UC123456789",
        "https://www.youtube.com/playlist?list=PL123456789",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
    ])
    def test_extract_video_id_invalid(self, url):
        with pytest.raises(TranscriptInvalidURLError):
            YouTubeProvider.extract_video_id(url)


class TestYouTubeProviderExtract:
    @pytest.mark.anyio
    async def test_extract_success(
        self, mock_youtube_api, sample_youtube_url, sample_transcript_snippets
    ):
        provider = YouTubeProvider()
        result = await provider.extract(sample_youtube_url)

        assert isinstance(result, Transcript)
        assert result.language == "en"
        assert result.full_text == (
            "Hello everyone Today we are going to talk about "
            "my top three book recommendations"
        )

    @pytest.mark.anyio
    async def test_extract_video_unavailable(self, mocker, sample_youtube_url):
        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            side_effect=VideoUnavailable("test_video_id"),
        )
        provider = YouTubeProvider()
        with pytest.raises(TranscriptNotFoundError) as exc_info:
            await provider.extract(sample_youtube_url)
        assert "unavailable" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_extract_transcripts_disabled(self, mocker, sample_youtube_url):
        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            side_effect=TranscriptsDisabled("test_video_id"),
        )
        provider = YouTubeProvider()
        with pytest.raises(TranscriptNotFoundError):
            await provider.extract(sample_youtube_url)

    @pytest.mark.anyio
    async def test_extract_no_transcript_found(self, mocker, sample_youtube_url):
        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            side_effect=NoTranscriptFound(
                "test_video_id",
                ["en"],
                MagicMock(),
            ),
        )
        provider = YouTubeProvider()
        with pytest.raises(TranscriptNotFoundError):
            await provider.extract(sample_youtube_url)

    @pytest.mark.anyio
    async def test_extract_newlines_are_stripped(
        self, mocker, sample_youtube_url
    ):
        """Verify that newline characters are removed from full_text."""
        s1 = MagicMock()
        s1.text = "Hello\nworld"
        s1.start = 0.0
        s1.duration = 2.0

        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            return_value=[s1],
        )
        provider = YouTubeProvider()
        result = await provider.extract(sample_youtube_url)
        assert "\n" not in result.full_text
        assert result.full_text == "Hello world"
