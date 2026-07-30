from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityType(StrEnum):
    MOVIE = "movie"
    DIRECTOR = "director"
    SONG = "song"
    ALBUM = "album"
    ARTIST = "artist"
    BOOK = "book"
    AUTHOR = "author"


class Entity(BaseModel):
    """A single extracted entertainment entity."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Name of the work or person as explicitly stated in the transcript, "
            "or its unambiguous normalized form"
        ),
    )
    type: EntityType
    context: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Explicit disambiguator supplied by the transcript: artist for songs/albums, "
            "author for books, year for movies"
        ),
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("context", mode="before")
    @classmethod
    def _strip_context(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value
