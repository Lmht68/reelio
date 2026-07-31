from unittest.mock import MagicMock

import pytest
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
)

from src.transcript.exceptions import (
    TranscriptDownloadError,
    TranscriptInvalidURLError,
    TranscriptNotFoundError,
)
from src.transcript.models import Transcript
from src.transcript.providers.youtube import YouTubeProvider


class TestYouTubeProviderExtractVideoID:
    @pytest.mark.parametrize(
        "url,expected_id",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/abc123def45", "abc123def45"),
            ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30", "dQw4w9WgXcQ"),
            ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?si=abc123&v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_extract_video_id(self, url, expected_id):
        assert YouTubeProvider.extract_video_id(url) == expected_id

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/",
            "https://youtube.com/channel/UC123456789",
            "https://www.youtube.com/playlist?list=PL123456789",
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
            "https://notyoutu.be/dQw4w9WgXcQ",
            "https://evil-youtube.com/watch?v=dQw4w9WgXcQ",
        ],
    )
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
            "Hello everyone Today we are going to talk about my top three book recommendations"
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
    async def test_extract_newlines_are_stripped(self, mocker, sample_youtube_url):
        """Verify that newline characters are removed from full_text."""
        s1 = MagicMock()
        s1.text = "Hello\nworld"
        s1.start = 0.0
        s1.duration = 2.0

        fetched = MagicMock()
        fetched.language_code = "en"
        fetched.__iter__.return_value = [s1]

        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            return_value=fetched,
        )
        provider = YouTubeProvider()
        result = await provider.extract(sample_youtube_url)
        assert "\n" not in result.full_text
        assert result.full_text == "Hello world"


class TestSelectTranscript:
    """Tests for YouTubeProvider._select_transcript selection policy."""

    @staticmethod
    def _make_real_tl(*language_codes: str):
        """Build a real object mimicking TranscriptList with find_transcript that can raise."""
        transcripts = []
        for code in language_codes:
            t = MagicMock()
            t.language_code = code
            transcripts.append(t)

        class _FakeTranscriptList:
            def __init__(self):
                self._transcripts = transcripts

            def __iter__(self):
                return iter(self._transcripts)

            def find_transcript(self, language_codes):
                raise NoTranscriptFound("vid", language_codes, self)

        return _FakeTranscriptList(), transcripts

    def test_exact_en_present(self):
        """When exact 'en' is available, find_transcript returns it directly."""
        tl = MagicMock()
        t_en = MagicMock()
        t_en.language_code = "en"
        tl.find_transcript.return_value = t_en
        result = YouTubeProvider._select_transcript(tl)
        tl.find_transcript.assert_called_once_with(["en"])
        assert result == t_en

    def test_only_en_variant_available(self):
        """When only en-US is available, fallback picks it."""
        tl, transcripts = self._make_real_tl("en-US", "fr")
        result = YouTubeProvider._select_transcript(tl)
        assert result.language_code == "en-US"

    def test_no_english_returns_first(self):
        """When no English at all, returns first available transcript."""
        tl, transcripts = self._make_real_tl("fr", "de")
        result = YouTubeProvider._select_transcript(tl)
        assert result.language_code == "fr"

    def test_empty_list_reraises(self):
        """When transcript list is empty, NoTranscriptFound propagates."""
        tl, _ = self._make_real_tl()
        with pytest.raises(NoTranscriptFound):
            YouTubeProvider._select_transcript(tl)


class TestYouTubeProviderExtractErrorMapping:
    """Parametrized tests that each youtube-transcript-api error maps correctly."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "error,expected_exc,expected_msg_fragment",
        [
            (VideoUnavailable("vid"), TranscriptNotFoundError, "unavailable"),
            (VideoUnplayable("vid", "reason", []), TranscriptNotFoundError, "unavailable"),
            (AgeRestricted("vid"), TranscriptNotFoundError, "unavailable"),
            (TranscriptsDisabled("vid"), TranscriptNotFoundError, "No transcript"),
            (
                NoTranscriptFound("vid", ["en"], MagicMock()),
                TranscriptNotFoundError,
                "No transcript",
            ),
            (InvalidVideoId("vid"), TranscriptInvalidURLError, "Invalid YouTube video ID"),
            (
                CouldNotRetrieveTranscript("vid"),
                TranscriptDownloadError,
                "Failed to fetch transcript",
            ),
            (
                RequestBlocked("vid"),
                TranscriptDownloadError,
                "Failed to fetch transcript",
            ),
            (
                IpBlocked("vid"),
                TranscriptDownloadError,
                "Failed to fetch transcript",
            ),
            (
                PoTokenRequired("vid"),
                TranscriptDownloadError,
                "Failed to fetch transcript",
            ),
            (
                YouTubeRequestFailed("vid", MagicMock()),
                TranscriptDownloadError,
                "Failed to fetch transcript",
            ),
        ],
    )
    async def test_error_mapping(
        self, mocker, sample_youtube_url, error, expected_exc, expected_msg_fragment
    ):
        mocker.patch.object(YouTubeProvider, "_fetch_transcript", side_effect=error)
        provider = YouTubeProvider()
        with pytest.raises(expected_exc) as exc_info:
            await provider.extract(sample_youtube_url)
        assert expected_msg_fragment in str(exc_info.value)

    @pytest.mark.anyio
    async def test_extract_reports_actual_language(self, mocker, sample_youtube_url):
        s1 = MagicMock()
        s1.text = "Xin chao"
        s1.start = 0.0
        s1.duration = 2.0

        fetched = MagicMock()
        fetched.language_code = "vi"
        fetched.__iter__.return_value = [s1]

        mocker.patch.object(
            YouTubeProvider,
            "_fetch_transcript",
            return_value=fetched,
        )
        provider = YouTubeProvider()
        result = await provider.extract(sample_youtube_url)
        assert result.language == "vi"
