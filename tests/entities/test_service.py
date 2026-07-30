import logging
from unittest.mock import AsyncMock

from src.entities.extractors.base import EntityExtractor
from src.entities.models import Entity, EntityType
from src.entities.service import EntityService, _normalize


class FakeExtractor(EntityExtractor):
    def __init__(self, entities: list[Entity]):
        self._entities = entities

    async def extract(self, text: str, language: str | None) -> list[Entity]:
        return self._entities


class TestNormalize:
    def test_none_returns_none(self) -> None:
        assert _normalize(None) is None

    def test_blank_returns_none(self) -> None:
        assert _normalize("   ") is None

    def test_casefold(self) -> None:
        assert _normalize("DUNE") == "dune"

    def test_whitespace_collapse(self) -> None:
        assert _normalize("  Frank   Herbert  ") == "frank herbert"

    def test_unicode_normalization(self) -> None:
        # 'Amélie' composed vs decomposed forms
        composed = "Am\u00e9lie"
        decomposed = "Ame\u0301lie"
        assert _normalize(composed) == _normalize(decomposed)

    def test_emoji_not_normalized_away(self) -> None:
        assert _normalize("\U0001f600") == "\U0001f600"


class TestEntityService:
    # -- Dedup: case-insensitive --

    async def test_case_insensitive_dedup(self) -> None:
        entities = [
            Entity(name="Dune", type=EntityType.BOOK),
            Entity(name=" DUNE ", type=EntityType.BOOK),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 1
        assert result[0].name == "Dune"

    # -- Dedup: unicode case + whitespace --

    async def test_unicode_case_whitespace_dedup(self) -> None:
        entities = [
            Entity(name="Amélie", type=EntityType.MOVIE),
            Entity(name=" AMÉLIE ", type=EntityType.MOVIE),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 1
        assert result[0].name == "Amélie"

    # -- Dedup: distinct contexts preserved --

    async def test_distinct_contexts_preserved(self) -> None:
        entities = [
            Entity(name="Heat", type=EntityType.MOVIE, context="1986"),
            Entity(name="Heat", type=EntityType.MOVIE, context="1995"),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 2
        assert {e.context for e in result} == {"1986", "1995"}

    # -- Dedup: blank context equals no context --

    async def test_blank_context_equals_none(self) -> None:
        entities = [
            Entity(name="Dune", type=EntityType.BOOK, context=None),
            Entity(name="Dune", type=EntityType.BOOK, context="   "),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 1
        assert result[0].context is None

    # -- Dedup: no-context song vs artist-context song stay separate --

    async def test_different_contexts_stay_separate(self) -> None:
        entities = [
            Entity(name="Bohemian Rhapsody", type=EntityType.SONG),
            Entity(name="Bohemian Rhapsody", type=EntityType.SONG, context="Queen"),
        ]
        service = EntityService(FakeExtractor(entities))
        result = await service.extract("text", None)
        assert len(result) == 2

    # -- Empty extractor --

    async def test_empty_extractor_output(self) -> None:
        service = EntityService(FakeExtractor([]))
        result = await service.extract("text", None)
        assert result == []

    # -- aclose delegation --

    async def test_aclose_delegates(self) -> None:
        extractor = AsyncMock(spec=EntityExtractor)
        service = EntityService(extractor)
        await service.aclose()
        extractor.aclose.assert_awaited_once_with()

    # -- Logging --

    async def test_log_contains_only_count_and_timing(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger="src.entities.service")
        entities = [Entity(name="Dune", type=EntityType.BOOK)]
        service = EntityService(FakeExtractor(entities))
        await service.extract("a secret transcript about Dune", None)
        assert any("1 entities in" in record.message for record in caplog.records)
        # Verify transcript text not logged
        assert "secret transcript" not in caplog.text
        # Verify entity names not logged
        assert "Dune" not in caplog.text
