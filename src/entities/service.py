import logging

from src.entities.extractors.base import EntityExtractor
from src.entities.models import Entity, EntityType

logger = logging.getLogger(__name__)


class EntityService:
    def __init__(self, extractor: EntityExtractor):
        self._extractor = extractor

    async def extract(self, text: str, language: str | None) -> list[Entity]:
        entities = await self._extractor.extract(text, language)
        return self._dedupe(entities)

    @staticmethod
    def _dedupe(entities: list[Entity]) -> list[Entity]:
        seen: set[tuple[EntityType, str]] = set()
        result: list[Entity] = []
        for entity in entities:
            key = (entity.type, entity.name.casefold().strip())
            if key not in seen:
                seen.add(key)
                result.append(entity)
        return result
