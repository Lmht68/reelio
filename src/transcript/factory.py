import re
import urllib.parse
from functools import lru_cache

from src.transcript.exceptions import (
    TranscriptInvalidURLError,
    TranscriptUnsupportedPlatformError,
)
from src.transcript.models import Platform

# Platform detection rules: (Platform, [regex_patterns]).
# The (?<!\w) lookbehind ensures we match domains as proper hostnames,
# not as substrings of other domains (e.g. "notyoutube.com" should not match).
_PLATFORM_PATTERNS: list[tuple[Platform, list[str]]] = [
    (
        Platform.YOUTUBE,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.|m\.|music\.)?youtube\.com/",
            r"(?:https?://)?(?<![\w-])youtu\.be/",
        ],
    ),
    (
        Platform.INSTAGRAM,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.)?instagram\.com/(?:reel|p|tv)/",
        ],
    ),
    (
        Platform.FACEBOOK,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.|m\.)?facebook\.com/(?:reel|watch|share)/",
            r"(?:https?://)?(?<![\w-])fb\.watch/",
        ],
    ),
    (
        Platform.TIKTOK,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.)?tiktok\.com/@",
            r"(?:https?://)?(?<![\w-])vm\.tiktok\.com/",
        ],
    ),
    (
        Platform.X,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.|mobile\.)?(?:twitter|x)\.com/\w+/status/",
        ],
    ),
    (
        Platform.THREADS,
        [
            r"(?:https?://)?(?<![\w-])(?:www\.)?threads\.com/",
        ],
    ),
]

MAX_URL_LENGTH = 2048


def validate_url(url: str) -> str:
    """Basic URL validation. Returns the stripped URL if valid, raises otherwise."""
    url = url.strip()
    if not url:
        raise TranscriptInvalidURLError("URL must not be empty")

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise TranscriptInvalidURLError(f"Invalid URL: {url}")
    if parsed.scheme not in ("http", "https"):
        raise TranscriptInvalidURLError(f"Unsupported URL scheme '{parsed.scheme}': {url}")
    if len(url) > MAX_URL_LENGTH:
        raise TranscriptInvalidURLError(
            f"URL exceeds maximum length of {MAX_URL_LENGTH} characters"
        )
    return url


@lru_cache(maxsize=128)
def detect_platform(url: str) -> Platform:
    """Detect the video platform from a URL.

    Uses an LRU cache to avoid re-parsing the same URL patterns repeatedly.
    """
    for platform, patterns in _PLATFORM_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return platform
    return Platform.UNKNOWN


def detect_platform_strict(url: str) -> Platform:
    """Detect platform and raise if unsupported."""
    platform = detect_platform(url)
    if platform == Platform.UNKNOWN:
        raise TranscriptUnsupportedPlatformError(f"Unsupported video platform for URL: {url}")
    return platform
