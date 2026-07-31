class MetadataError(Exception):
    """Base exception for all metadata-related errors."""

    def __init__(self, message: str, *, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class MetadataProviderError(MetadataError):
    """Raised when a metadata provider encounters a transport, timeout, status, rate-limit,
    JSON, or incompatible schema failure."""
