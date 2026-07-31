import pytest
from pydantic import AnyUrl, ValidationError

from src.entities.models import Entity, EntityType
from src.metadata.models import (
    AlbumMetadata,
    ArtistMetadata,
    AuthorMetadata,
    BookMetadata,
    DirectorMetadata,
    EnrichedEntity,
    MovieMetadata,
    SongMetadata,
)

# ── parametrized round-trip table ──────────────────────────────────────────

METADATA_CASES = [
    (
        EntityType.MOVIE,
        MovieMetadata(
            provider_id="550",
            title="Fight Club",
            year=1999,
            overview="An insomniac office worker...",
            poster_url=AnyUrl("https://image.tmdb.org/t/p/w500/posters/fc.jpg"),
            imdb_id="tt0137523",
            url=AnyUrl("https://www.themoviedb.org/movie/550"),
        ),
        {
            "kind",
            "provider",
            "provider_id",
            "title",
            "year",
            "overview",
            "poster_url",
            "imdb_id",
            "url",
        },
    ),
    (
        EntityType.DIRECTOR,
        DirectorMetadata(
            provider_id="7467",
            name="David Fincher",
            known_for=["Fight Club", "Seven"],
            image_url=AnyUrl("https://image.tmdb.org/t/p/w500/director.jpg"),
            url=AnyUrl("https://www.themoviedb.org/person/7467"),
        ),
        {"kind", "provider", "provider_id", "name", "known_for", "image_url", "url"},
    ),
    (
        EntityType.SONG,
        SongMetadata(
            provider_id="1ABC",
            title="Bohemian Rhapsody",
            artists=["Queen"],
            album="A Night at the Opera",
            image_url=AnyUrl("https://i.scdn.co/image/song.jpg"),
            url=AnyUrl("https://open.spotify.com/track/1ABC"),
        ),
        {"kind", "provider", "provider_id", "title", "artists", "album", "image_url", "url"},
    ),
    (
        EntityType.ALBUM,
        AlbumMetadata(
            provider_id="2DEF",
            title="A Night at the Opera",
            artists=["Queen"],
            year=1975,
            image_url=AnyUrl("https://i.scdn.co/image/album.jpg"),
            url=AnyUrl("https://open.spotify.com/album/2DEF"),
        ),
        {"kind", "provider", "provider_id", "title", "artists", "year", "image_url", "url"},
    ),
    (
        EntityType.ARTIST,
        ArtistMetadata(
            provider_id="3GHI",
            name="Queen",
            genres=["rock", "classic rock"],
            image_url=AnyUrl("https://i.scdn.co/image/artist.jpg"),
            url=AnyUrl("https://open.spotify.com/artist/3GHI"),
        ),
        {"kind", "provider", "provider_id", "name", "genres", "image_url", "url"},
    ),
    (
        EntityType.BOOK,
        BookMetadata(
            provider_id="abc123",
            title="Dune",
            authors=["Frank Herbert"],
            year=1965,
            description="A science fiction novel.",
            thumbnail_url=AnyUrl("https://books.google.com/books/content?id=abc123"),
            isbn="9780441013593",
            url=AnyUrl("https://books.google.com/books?id=abc123"),
        ),
        {
            "kind",
            "provider",
            "provider_id",
            "title",
            "authors",
            "year",
            "description",
            "thumbnail_url",
            "isbn",
            "url",
        },
    ),
    (
        EntityType.AUTHOR,
        AuthorMetadata(
            provider_id="frank herbert",
            name="Frank Herbert",
            known_for=["Dune"],
            url=AnyUrl("https://books.google.com/books?q=inauthor:%22Frank+Herbert%22"),
        ),
        {"kind", "provider", "provider_id", "name", "known_for", "url"},
    ),
]


@pytest.mark.parametrize(
    ("entity_type", "metadata", "expected_keys"),
    METADATA_CASES,
)
def test_round_trip(entity_type, metadata, expected_keys):
    entity = Entity(
        name=metadata.name if hasattr(metadata, "name") else metadata.title,
        type=entity_type,
    )
    enriched = EnrichedEntity(entity=entity, metadata=metadata)

    dumped = enriched.model_dump(mode="json")
    meta_dumped = dumped["metadata"]

    # literal discriminator and provider
    assert meta_dumped["kind"] == metadata.kind
    assert meta_dumped["provider"] == metadata.provider

    # exact key set
    assert set(meta_dumped.keys()) == expected_keys

    # list keys present and never None
    list_keys = {"known_for", "artists", "authors", "genres"}
    for lk in list_keys & expected_keys:
        assert isinstance(meta_dumped[lk], list)

    # round-trip
    revalidated = EnrichedEntity.model_validate(dumped)
    assert revalidated.entity.type == entity_type
    assert revalidated.metadata is not None
    if revalidated.metadata is not None:
        assert revalidated.metadata.kind == metadata.kind


# ── discriminated union rejection ───────────────────────────────────────────


def test_rejects_unknown_discriminator():
    with pytest.raises(ValidationError, match="kind"):
        EnrichedEntity.model_validate(
            {
                "entity": {"name": "x", "type": "movie"},
                "metadata": {
                    "kind": "unknown_kind",
                    "provider": "tmdb",
                    "provider_id": "1",
                    "title": "x",
                    "url": "https://example.com/movie",
                },
            }
        )


def test_rejects_missing_discriminator():
    with pytest.raises(ValidationError):
        EnrichedEntity.model_validate(
            {
                "entity": {"name": "x", "type": "movie"},
                "metadata": {
                    "provider": "tmdb",
                    "provider_id": "1",
                    "title": "x",
                    "url": "https://example.com/movie",
                },
            }
        )


# ── wrong provider literal on each variant ──────────────────────────────────


@pytest.mark.parametrize(
    ("metadata_cls", "default_kwargs"),
    [
        (
            MovieMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/movie")},
        ),
        (
            DirectorMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/dir")},
        ),
        (
            SongMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/song")},
        ),
        (
            AlbumMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/album")},
        ),
        (
            ArtistMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/artist")},
        ),
        (
            BookMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/book")},
        ),
        (
            AuthorMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/author")},
        ),
    ],
)
def test_rejects_wrong_provider_literal(metadata_cls, default_kwargs):
    with pytest.raises(ValidationError, match="provider"):
        metadata_cls(**{**default_kwargs, "provider": "wrong_provider"})


# ── reject unknown fields on each metadata class ─────────────────────────────


@pytest.mark.parametrize(
    ("metadata_cls", "default_kwargs"),
    [
        (
            MovieMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/movie")},
        ),
        (
            DirectorMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/dir")},
        ),
        (
            SongMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/song")},
        ),
        (
            AlbumMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/album")},
        ),
        (
            ArtistMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/artist")},
        ),
        (
            BookMetadata,
            {"provider_id": "1", "title": "x", "url": AnyUrl("https://example.com/book")},
        ),
        (
            AuthorMetadata,
            {"provider_id": "1", "name": "x", "url": AnyUrl("https://example.com/author")},
        ),
    ],
)
def test_rejects_unknown_fields(metadata_cls, default_kwargs):
    with pytest.raises(ValidationError, match="extra"):
        metadata_cls(**{**default_kwargs, "extra_field": "intruder"})


def test_rejects_unknown_fields_on_enriched_entity():
    movie = MovieMetadata(
        provider_id="1",
        title="x",
        url=AnyUrl("https://example.com/movie"),
    )
    entity = Entity(name="x", type=EntityType.MOVIE)
    with pytest.raises(ValidationError, match="extra"):
        EnrichedEntity(entity=entity, metadata=movie, extra="nope")  # type: ignore[call-arg]


# ── URL validation ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metadata_cls", "default_kwargs"),
    [
        (MovieMetadata, {"provider_id": "1", "title": "x"}),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}),
        (SongMetadata, {"provider_id": "1", "title": "x"}),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}),
        (BookMetadata, {"provider_id": "1", "title": "x"}),
        (AuthorMetadata, {"provider_id": "1", "name": "x"}),
    ],
)
def test_rejects_malformed_url(metadata_cls, default_kwargs):
    with pytest.raises(ValidationError):
        metadata_cls(**{**default_kwargs, "url": "not-a-url"})


def test_rejects_http_url_on_movie():
    with pytest.raises(ValidationError):
        MovieMetadata(
            provider_id="1",
            title="x",
            url="https://example.com/movie",
            poster_url="http://example.com/poster.jpg",
        )


@pytest.mark.parametrize(
    ("metadata_cls", "default_kwargs", "url_field"),
    [
        (MovieMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (MovieMetadata, {"provider_id": "1", "title": "x"}, "poster_url"),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}, "image_url"),
        (SongMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (SongMetadata, {"provider_id": "1", "title": "x"}, "image_url"),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}, "image_url"),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}, "image_url"),
        (BookMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (BookMetadata, {"provider_id": "1", "title": "x"}, "thumbnail_url"),
        (AuthorMetadata, {"provider_id": "1", "name": "x"}, "url"),
    ],
)
def test_rejects_http_url(metadata_cls, default_kwargs, url_field):
    kwargs = {**default_kwargs}
    # Set all URL fields to valid HTTPS, then override the target to HTTP
    for f in metadata_cls.model_fields:
        if f.endswith("_url") or f == "url":
            kwargs[f] = "https://example.com/valid"
    kwargs[url_field] = "http://example.com/insecure"
    with pytest.raises(ValidationError):
        metadata_cls(**kwargs)


# ── list immutability ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "metadata_cls_factory",
    [
        lambda: DirectorMetadata(
            provider_id="1",
            name="x",
            known_for=["a"],
            url=AnyUrl("https://example.com/d"),
        ),
        lambda: SongMetadata(
            provider_id="1",
            title="x",
            artists=["a"],
            url=AnyUrl("https://example.com/s"),
        ),
        lambda: AlbumMetadata(
            provider_id="1",
            title="x",
            artists=["a"],
            url=AnyUrl("https://example.com/a"),
        ),
        lambda: ArtistMetadata(
            provider_id="1",
            name="x",
            genres=["a"],
            url=AnyUrl("https://example.com/ar"),
        ),
        lambda: BookMetadata(
            provider_id="1",
            title="x",
            authors=["a"],
            url=AnyUrl("https://example.com/b"),
        ),
        lambda: AuthorMetadata(
            provider_id="1",
            name="x",
            known_for=["a"],
            url=AnyUrl("https://example.com/au"),
        ),
    ],
)
def test_list_isolation(metadata_cls_factory):
    a = metadata_cls_factory()
    b = metadata_cls_factory()
    # Mutate one list
    list_field = None
    for field_name in ("known_for", "artists", "authors", "genres"):
        if hasattr(a, field_name):
            list_field = field_name
            break
    assert list_field is not None
    getattr(a, list_field).append("extra")
    assert "extra" not in getattr(b, list_field)


# ── entity/metadata kind mismatch ────────────────────────────────────────────


def test_rejects_kind_mismatch():
    movie = MovieMetadata(
        provider_id="1",
        title="x",
        url=AnyUrl("https://example.com/movie"),
    )
    entity = Entity(name="x", type=EntityType.BOOK)
    with pytest.raises(ValueError, match="metadata kind must match entity type"):
        EnrichedEntity(entity=entity, metadata=movie)


@pytest.mark.parametrize("entity_type", list(EntityType))
def test_accepts_absent_metadata(entity_type):
    entity = Entity(name="x", type=entity_type)
    enriched = EnrichedEntity(entity=entity)
    assert enriched.metadata is None


# ── ArtistMetadata serializes name ───────────────────────────────────────────


def test_artist_metadata_serializes_name():
    artist = ArtistMetadata(
        provider_id="3GHI",
        name="Queen",
        url=AnyUrl("https://open.spotify.com/artist/3GHI"),
    )
    dumped = artist.model_dump(mode="json")
    assert dumped["name"] == "Queen"


# ── BookMetadata has no Amazon/affiliate keys ───────────────────────────────


def test_book_metadata_no_amazon_keys():
    book = BookMetadata(
        provider_id="abc123",
        title="Dune",
        url=AnyUrl("https://books.google.com/books?id=abc123"),
    )
    dumped = book.model_dump(mode="json")
    assert "amazon_id" not in dumped
    assert "affiliate" not in dumped


def test_book_metadata_rejects_amazon_field():
    with pytest.raises(ValidationError, match="extra"):
        BookMetadata(
            provider_id="abc123",
            title="Dune",
            url=AnyUrl("https://books.google.com/books?id=abc123"),
            amazon_id="B000",  # type: ignore[call-arg]
        )


# ── AuthorMetadata provider_id sample ────────────────────────────────────────


def test_author_metadata_provider_id_frank_herbert():
    author = AuthorMetadata(
        provider_id="frank herbert",
        name="Frank Herbert",
        url=AnyUrl("https://books.google.com/books?q=inauthor:%22Frank+Herbert%22"),
    )
    assert author.provider_id == "frank herbert"


# ── prevalidated AnyUrl bypass detection ──────────────────────────────────────


@pytest.mark.parametrize(
    ("metadata_cls", "required_kwargs", "url_field"),
    [
        (MovieMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (MovieMetadata, {"provider_id": "1", "title": "x"}, "poster_url"),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}, "image_url"),
        (SongMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (SongMetadata, {"provider_id": "1", "title": "x"}, "image_url"),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}, "image_url"),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}, "image_url"),
        (BookMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (BookMetadata, {"provider_id": "1", "title": "x"}, "thumbnail_url"),
        (AuthorMetadata, {"provider_id": "1", "name": "x"}, "url"),
    ],
)
def test_rejects_prevalidated_http_url(metadata_cls, required_kwargs, url_field):
    """A preconstructed HTTP AnyUrl must not bypass the HTTPS-only validator."""
    kwargs = {**required_kwargs}
    for f in metadata_cls.model_fields:
        if f.endswith("_url") or f == "url":
            kwargs[f] = "https://example.com/valid"
    kwargs[url_field] = AnyUrl("http://example.com/insecure")
    with pytest.raises(ValidationError):
        metadata_cls(**kwargs)


@pytest.mark.parametrize(
    ("metadata_cls", "required_kwargs", "url_field"),
    [
        (MovieMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (DirectorMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (SongMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (AlbumMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (ArtistMetadata, {"provider_id": "1", "name": "x"}, "url"),
        (BookMetadata, {"provider_id": "1", "title": "x"}, "url"),
        (AuthorMetadata, {"provider_id": "1", "name": "x"}, "url"),
    ],
)
def test_rejects_prevalidated_non_web_url(metadata_cls, required_kwargs, url_field):
    """A preconstructed non-web-scheme AnyUrl must be rejected."""
    kwargs = {**required_kwargs}
    for f in metadata_cls.model_fields:
        if f.endswith("_url") or f == "url":
            kwargs[f] = "https://example.com/valid"
    kwargs[url_field] = AnyUrl("ftp://example.com/file")
    with pytest.raises(ValidationError):
        metadata_cls(**kwargs)


# ── omitted defaults serialise correctly ─────────────────────────────────────


OMITTED_DEFAULTS_CASES = [
    (
        MovieMetadata,
        {"provider_id": "1", "title": "x", "url": "https://example.com/movie"},
        {"year", "overview", "poster_url", "imdb_id"},
        [],
    ),
    (
        DirectorMetadata,
        {"provider_id": "1", "name": "x", "url": "https://example.com/dir"},
        {"image_url"},
        ["known_for"],
    ),
    (
        SongMetadata,
        {"provider_id": "1", "title": "x", "url": "https://example.com/song"},
        {"album", "image_url"},
        ["artists"],
    ),
    (
        AlbumMetadata,
        {"provider_id": "1", "title": "x", "url": "https://example.com/album"},
        {"year", "image_url"},
        ["artists"],
    ),
    (
        ArtistMetadata,
        {"provider_id": "1", "name": "x", "url": "https://example.com/artist"},
        {"image_url"},
        ["genres"],
    ),
    (
        BookMetadata,
        {"provider_id": "1", "title": "x", "url": "https://example.com/book"},
        {"year", "description", "thumbnail_url", "isbn"},
        ["authors"],
    ),
    (
        AuthorMetadata,
        {"provider_id": "1", "name": "x", "url": "https://example.com/author"},
        set(),
        ["known_for"],
    ),
]


@pytest.mark.parametrize(
    ("metadata_cls", "required_kwargs", "nullable_keys", "list_keys"),
    OMITTED_DEFAULTS_CASES,
)
def test_omitted_defaults_serialize(metadata_cls, required_kwargs, nullable_keys, list_keys):
    """Omitted optional scalars serialize as null; omitted lists serialize as empty arrays."""
    m = metadata_cls(**required_kwargs)
    dumped = m.model_dump(mode="json")
    for key in nullable_keys:
        assert dumped[key] is None, f"{key} should be null"
    for key in list_keys:
        assert dumped[key] == [], f"{key} should be empty list"


@pytest.mark.parametrize(
    ("metadata_cls", "required_kwargs", "nullable_keys", "list_keys"),
    OMITTED_DEFAULTS_CASES,
)
def test_omitted_list_default_isolation(metadata_cls, required_kwargs, nullable_keys, list_keys):
    """Mutating one instance's omitted default list must not affect another instance."""
    if not list_keys:
        pytest.skip("no list fields on this variant")
    a = metadata_cls(**required_kwargs)
    b = metadata_cls(**required_kwargs)
    list_field = list_keys[0]
    getattr(a, list_field).append("extra")
    assert "extra" not in getattr(b, list_field)
