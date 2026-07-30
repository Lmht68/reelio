import pytest
from pydantic import ValidationError

from src.entities.models import Entity, EntityType

_VALID_MINIMAL = [
    (EntityType.MOVIE, "Dune"),
    (EntityType.DIRECTOR, "Denis Villeneuve"),
    (EntityType.SONG, "Bohemian Rhapsody"),
    (EntityType.ALBUM, "A Night at the Opera"),
    (EntityType.ARTIST, "Queen"),
    (EntityType.BOOK, "Dune"),
    (EntityType.AUTHOR, "Frank Herbert"),
]


class TestEntity:
    @pytest.mark.parametrize("entity_type,name", _VALID_MINIMAL)
    def test_valid_minimal_per_type(self, entity_type: EntityType, name: str) -> None:
        entity = Entity(name=name, type=entity_type)
        assert entity.name == name
        assert entity.type == entity_type
        assert entity.context is None

    def test_name_stripped(self) -> None:
        entity = Entity(name="  Dune  ", type=EntityType.BOOK)
        assert entity.name == "Dune"

    def test_blank_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            Entity(name="   ", type=EntityType.BOOK)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            Entity(name="", type=EntityType.BOOK)

    def test_name_300_chars_accepted(self) -> None:
        name = "A" * 300
        entity = Entity(name=name, type=EntityType.BOOK)
        assert entity.name == name

    def test_name_301_chars_raises(self) -> None:
        name = "A" * 301
        with pytest.raises(ValidationError):
            Entity(name=name, type=EntityType.BOOK)

    def test_context_stripped(self) -> None:
        entity = Entity(name="Dune", type=EntityType.BOOK, context="  Frank Herbert  ")
        assert entity.context == "Frank Herbert"

    def test_blank_context_becomes_none(self) -> None:
        entity = Entity(name="Dune", type=EntityType.BOOK, context="   ")
        assert entity.context is None

    def test_blank_context_serializes_as_null(self) -> None:
        entity = Entity(name="Dune", type=EntityType.BOOK, context="   ")
        assert entity.model_dump(mode="json")["context"] is None

    def test_context_300_chars_accepted(self) -> None:
        context = "A" * 300
        entity = Entity(name="Dune", type=EntityType.BOOK, context=context)
        assert entity.context == context

    def test_context_301_chars_raises(self) -> None:
        context = "A" * 301
        with pytest.raises(ValidationError):
            Entity(name="Dune", type=EntityType.BOOK, context=context)

    def test_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Entity(name="Dune", type=EntityType.BOOK, bogus=1)  # type: ignore[call-arg]

    def test_name_not_stripped_if_not_string(self) -> None:
        # Non-string name should still fail validation on type, not on our validator
        with pytest.raises(ValidationError):
            Entity(name=123, type=EntityType.BOOK)  # type: ignore[arg-type]
