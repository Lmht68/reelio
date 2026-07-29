from src.entities.extractors.base import EntityExtractor
from src.entities.models import Entity, EntityType
from src.entities.service import EntityService


class FakeExtractor(EntityExtractor):
    def __init__(self, entities: list[Entity]):
        self._entities = entities

    async def extract(self, text: str, language: str | None) -> list[Entity]:
        return self._entities


class TestEntityService:
    async def test_case_insensitive_dedup(self):
        entities = [
            Entity(name="Dune", type=EntityType.BOOK),
            Entity(name=" DUNE ", type=EntityType.BOOK),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 1
        assert result[0].name == "Dune"

    async def test_first_occurrence_context_preserved(self):
        entities = [
            Entity(name="Inception", type=EntityType.MOVIE, context="2010"),
            Entity(name="inception", type=EntityType.MOVIE, context="Christopher Nolan"),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 1
        assert result[0].name == "Inception"
        assert result[0].context == "2010"

    async def test_empty_extractor_output(self):
        service = EntityService(FakeExtractor([]))
        result = await service.extract("text", None)
        assert result == []
