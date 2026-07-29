from enum import StrEnum

from pydantic import BaseModel, Field


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

    name: str = Field(description="Canonical name of the work or person")
    type: EntityType
    context: str | None = Field(
        default=None,
        description="Disambiguating hint: artist for songs/albums, author for books, year for movies",
    )
