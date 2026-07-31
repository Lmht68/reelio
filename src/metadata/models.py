from typing import Annotated, Literal, Self

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints, model_validator

from src.entities.models import Entity

HttpsUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"])]


class MovieMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["movie"] = "movie"
    provider: Literal["tmdb"] = "tmdb"
    provider_id: str
    title: str
    year: int | None = None
    overview: str | None = None
    poster_url: HttpsUrl | None = None
    imdb_id: str | None = None
    url: HttpsUrl


class DirectorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["director"] = "director"
    provider: Literal["tmdb"] = "tmdb"
    provider_id: str
    name: str
    known_for: list[str] = Field(default_factory=list)
    image_url: HttpsUrl | None = None
    url: HttpsUrl


class SongMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["song"] = "song"
    provider: Literal["spotify"] = "spotify"
    provider_id: str
    title: str
    artists: list[str] = Field(default_factory=list)
    album: str | None = None
    image_url: HttpsUrl | None = None
    url: HttpsUrl


class AlbumMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["album"] = "album"
    provider: Literal["spotify"] = "spotify"
    provider_id: str
    title: str
    artists: list[str] = Field(default_factory=list)
    year: int | None = None
    image_url: HttpsUrl | None = None
    url: HttpsUrl


class ArtistMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["artist"] = "artist"
    provider: Literal["spotify"] = "spotify"
    provider_id: str
    name: str
    genres: list[str] = Field(default_factory=list)
    image_url: HttpsUrl | None = None
    url: HttpsUrl


class BookMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["book"] = "book"
    provider: Literal["google_books"] = "google_books"
    provider_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    description: str | None = None
    thumbnail_url: HttpsUrl | None = None
    isbn: str | None = None
    url: HttpsUrl


class AuthorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["author"] = "author"
    provider: Literal["google_books"] = "google_books"
    provider_id: str
    name: str
    known_for: list[str] = Field(default_factory=list)
    url: HttpsUrl


EntityMetadata = Annotated[
    MovieMetadata
    | DirectorMetadata
    | SongMetadata
    | AlbumMetadata
    | ArtistMetadata
    | BookMetadata
    | AuthorMetadata,
    Field(discriminator="kind"),
]


class EnrichedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: Entity
    metadata: EntityMetadata | None = None

    @model_validator(mode="after")
    def _metadata_kind_matches_entity_type(self) -> Self:
        if self.metadata is None:
            return self
        if self.metadata.kind != self.entity.type.value:
            raise ValueError("metadata kind must match entity type")
        return self
