from abc import ABC, abstractmethod

from src.entities.models import Entity


class EntityExtractor(ABC):
    """Interface that every entity extractor must implement."""

    @abstractmethod
    async def extract(self, text: str, language: str | None) -> list[Entity]:
        """Extract entertainment entities from transcript text.

        Args:
            text: The transcript text.
            language: The transcript language code, e.g. 'en', if known.

        Returns:
            A list of extracted entities, possibly empty.

        Raises:
            EntityExtractionError: Extraction failed after one retry.
        """
        ...

    async def aclose(self) -> None:
        """Release resources held by the extractor. Default: nothing to release."""
        return None
