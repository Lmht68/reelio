from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_youtube_url() -> str:
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def sample_youtube_shorts_url() -> str:
    return "https://www.youtube.com/shorts/abc123def45"


@pytest.fixture
def sample_instagram_url() -> str:
    return "https://www.instagram.com/reel/CxAbCdEfGhI/"


@pytest.fixture
def sample_facebook_url() -> str:
    return "https://www.facebook.com/reel/123456789/"


@pytest.fixture
def sample_tiktok_url() -> str:
    return "https://www.tiktok.com/@user/video/123456789"


@pytest.fixture
def sample_transcript_snippets() -> list:
    """Synthetic YouTube-style FetchedTranscriptSnippet-like objects."""
    s1 = MagicMock()
    s1.text = "Hello everyone"
    s1.start = 0.0
    s1.duration = 2.0

    s2 = MagicMock()
    s2.text = "Today we are going to talk about"
    s2.start = 2.0
    s2.duration = 3.0

    s3 = MagicMock()
    s3.text = "my top three book recommendations"
    s3.start = 5.0
    s3.duration = 3.5

    return [s1, s2, s3]


@pytest.fixture
def mock_youtube_api(mocker, sample_transcript_snippets):
    """Pre-configured mock for YouTubeTranscriptApi._fetch_transcript."""
    mock = mocker.patch.object(
        __import__(
            "src.transcript.providers.youtube", fromlist=["YouTubeProvider"]
        ).YouTubeProvider,
        "_fetch_transcript",
        return_value=sample_transcript_snippets,
    )
    return mock
