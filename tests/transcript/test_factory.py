import pytest

from src.transcript.exceptions import (
    TranscriptInvalidURLError,
    TranscriptUnsupportedPlatformError,
)
from src.transcript.factory import detect_platform, detect_platform_strict, validate_url
from src.transcript.models import Platform


class TestValidateURL:
    def test_valid_url(self):
        result = validate_url("https://www.youtube.com/watch?v=test")
        assert result == "https://www.youtube.com/watch?v=test"

    def test_empty_url(self):
        with pytest.raises(TranscriptInvalidURLError):
            validate_url("")

    def test_whitespace_only_url(self):
        with pytest.raises(TranscriptInvalidURLError):
            validate_url("   ")

    def test_strips_surrounding_whitespace(self):
        result = validate_url("  https://youtube.com/watch?v=test  ")
        assert result == "https://youtube.com/watch?v=test"

    def test_missing_scheme(self):
        with pytest.raises(TranscriptInvalidURLError):
            validate_url("youtube.com/watch?v=test")

    def test_unsupported_scheme(self):
        with pytest.raises(TranscriptInvalidURLError):
            validate_url("ftp://youtube.com/watch?v=test")

    def test_url_too_long(self):
        long_url = "https://youtube.com/" + "a" * 3000
        with pytest.raises(TranscriptInvalidURLError):
            validate_url(long_url)

    def test_url_at_length_limit(self):
        # "https://youtube.com/" is 20 chars, so we can add 2028 "a" chars
        url = "https://youtube.com/" + "a" * (2048 - 20)
        result = validate_url(url)
        assert result == url


class TestDetectPlatform:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://youtu.be/dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://www.youtube.com/shorts/abc123def45", Platform.YOUTUBE),
            ("https://youtube.com/embed/dQw4w9WgXcQ", Platform.YOUTUBE),
            ("http://youtube.com/watch?v=test", Platform.YOUTUBE),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://www.instagram.com/reel/CxAbCdEfGhI/", Platform.INSTAGRAM),
            ("https://instagram.com/p/CxAbCdEfGhI/", Platform.INSTAGRAM),
            ("https://www.instagram.com/tv/CxAbCdEfGhI/", Platform.INSTAGRAM),
            ("https://www.facebook.com/reel/123456789/", Platform.FACEBOOK),
            ("https://facebook.com/watch/123456789/", Platform.FACEBOOK),
            ("https://www.facebook.com/share/v/abc123/", Platform.FACEBOOK),
            ("https://fb.watch/abc123/", Platform.FACEBOOK),
            ("https://m.facebook.com/reel/123456789/", Platform.FACEBOOK),
            ("https://www.tiktok.com/@user/video/123456789", Platform.TIKTOK),
            ("https://tiktok.com/@user/video/123456789", Platform.TIKTOK),
            ("https://vm.tiktok.com/abc123/", Platform.TIKTOK),
        ],
    )
    def test_detect_platform(self, url, expected):
        assert detect_platform(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://vimeo.com/123456",
            "https://www.dailymotion.com/video/abc123",
            "https://example.com/video",
            "https://twitter.com/user/status/123",
            "https://notyoutu.be/dQw4w9WgXcQ",
            "https://evil-youtube.com/watch?v=dQw4w9WgXcQ",
            "https://notfb.watch/abc/",
            "https://xvm.tiktok.com/abc/",
        ],
    )
    def test_unknown_platform(self, url):
        assert detect_platform(url) == Platform.UNKNOWN


class TestDetectPlatformStrict:
    def test_known_platform(self):
        result = detect_platform_strict("https://www.youtube.com/watch?v=test")
        assert result == Platform.YOUTUBE

    def test_unknown_platform_raises(self):
        with pytest.raises(TranscriptUnsupportedPlatformError) as exc_info:
            detect_platform_strict("https://vimeo.com/123456")
        assert "vimeo.com" in str(exc_info.value)
