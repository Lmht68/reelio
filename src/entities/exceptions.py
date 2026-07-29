class EntityError(Exception):
    """Base exception for all entity-related errors."""

    def __init__(self, message: str, *, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class EntityExtractionError(EntityError):
    """Raised when LLM entity extraction fails after one retry."""
