class TranscriptError(Exception):
    """Base exception for all transcript-related errors."""

    def __init__(self, message: str, *, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class TranscriptUnsupportedPlatformError(TranscriptError):
    """Raised when the URL does not match any supported platform pattern."""


class TranscriptInvalidURLError(TranscriptError):
    """Raised when the URL is malformed or not a valid URL."""


class TranscriptNotFoundError(TranscriptError):
    """Raised when no transcript is available for the given video."""


class TranscriptDownloadError(TranscriptError):
    """Raised when audio download fails (network error, geo-restriction, etc.)."""


class TranscriptTranscriptionError(TranscriptError):
    """Raised when the Whisper speech-to-text process fails."""
