class EntityError(Exception):
    """Base exception for all entity-related errors."""

    def __init__(self, message: str, *, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class EntityConfigurationError(EntityError):
    """Raised when required LLM configuration is absent."""


class EntityInputTooLongError(EntityError):
    """Raised when transcript text exceeds the configured character limit."""


class EntityExtractionError(EntityError):
    """Raised when LLM entity extraction fails after one retry."""
