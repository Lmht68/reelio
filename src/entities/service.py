import logging
import time
import unicodedata

from src.entities.extractors.base import EntityExtractor
from src.entities.models import Entity, EntityType

logger = logging.getLogger(__name__)


class EntityService:
    def __init__(self, extractor: EntityExtractor):
        self._extractor = extractor

    async def extract(self, text: str, language: str | None) -> list[Entity]:
        start = time.perf_counter()
        entities = await self._extractor.extract(text, language)
        result = self._dedupe(entities)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Entity extraction completed: %d entities in %.1f ms", len(result), elapsed_ms)
        return result

    async def aclose(self) -> None:
        await self._extractor.aclose()

    @staticmethod
    def _dedupe(entities: list[Entity]) -> list[Entity]:
        seen: set[tuple[EntityType, str | None, str | None]] = set()
        result: list[Entity] = []
        for entity in entities:
            key = (entity.type, _normalize(entity.name), _normalize(entity.context))
            if key not in seen:
                seen.add(key)
                result.append(entity)
        return result


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    return normalized or None
