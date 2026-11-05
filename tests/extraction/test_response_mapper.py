"""Public Extraction response mapper tests."""

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.response_mapper import (
    map_extraction_response,
    map_transcript_extraction_response,
)
from reelio.extraction.types import (
    ArtistCredit,
    AuthorCredit,
    BookEdition,
    BookMention,
    BookResult,
    BookResults,
    EnrichedAuthorCredit,
    EnrichedBookWork,
    EnrichedMovie,
    EnrichedMusicRelease,
    EnrichedTrack,
    EnrichedTVSeries,
    ExtractionResults,
    MovieMention,
    MovieResult,
    MusicReleaseMention,
    MusicReleaseResult,
    MusicResults,
    PipelineResult,
    Platform,
    ResultStatus,
    ScreenWorkResults,
    Source,
    TrackMention,
    TrackResult,
    Transcript,
    TranscriptMethod,
    TranscriptPipelineResult,
    TVSeriesMention,
    TVSeriesResult,
)


def test_source_mapper_preserves_the_existing_empty_response_contract() -> None:
    """Map a source result without HTTP request or application state."""
    result = PipelineResult(
        source=Source(
            platform=Platform.YOUTUBE,
            video_id="dQw4w9WgXcQ",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Mapper test source",
            description="Mapper test description.",
            channel="Mapper test channel",
            duration_seconds=42,
        ),
        transcript=Transcript(
            text="A mapper test transcript.",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        ),
        results=ExtractionResults(
            screen_works=ScreenWorkResults(movies=[], tv_series=[]),
            music=MusicResults(tracks=[], music_releases=[]),
            books=BookResults(books=[]),
        ),
        market=SpotifyMarket("JP"),
    )

    response = map_extraction_response(result)

    assert response.model_dump(mode="json") == {
        "market": "JP",
        "source": {
            "platform": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Mapper test source",
            "description": "Mapper test description.",
            "channel": "Mapper test channel",
            "duration_seconds": 42,
        },
        "transcript": {
            "text": "A mapper test transcript.",
            "language": "en",
            "method": "youtube_captions",
        },
        "statistics": {
            "movies": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            "tv_series": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            "tracks": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            "music_releases": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
            "books": {"n_mentions": 0, "n_resolved": 0, "n_unresolved": 0},
        },
        "results": {
            "movies": [],
            "tv_series": [],
            "tracks": [],
            "music_releases": [],
            "books": [],
        },
    }


def _representative_results() -> ExtractionResults:
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    unresolved_movie_mention = MovieMention(title="Unknown Movie", year=2024)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    unresolved_tv_series_mention = TVSeriesMention(title="Unknown TV Series", year=2024)
    track_mention = TrackMention(
        track_title="One More Time",
        artists=["Daft Punk"],
        release_title="Discovery",
        release_year=2001,
    )
    unresolved_track_mention = TrackMention(
        track_title="Unknown Track",
        artists=["Unknown Artist"],
        release_title=None,
        release_year=None,
    )
    music_release_mention = MusicReleaseMention(
        release_title="Discovery",
        artists=["Daft Punk"],
        release_year=2001,
    )
    unresolved_music_release_mention = MusicReleaseMention(
        release_title="Unknown Album",
        artists=["Unknown Artist"],
        release_year=None,
    )
    book_mention = BookMention(
        title="Pride and Prejudice",
        authors=[AuthorCredit(name="Jane Austen")],
    )
    unresolved_book_mention = BookMention(title="Unknown Book", authors=[])
    artist_credit = ArtistCredit(
        spotify_artist_id="4tZwfgrHOc3mvqYlEYSvVi",
        name="Daft Punk",
    )
    track_release = EnrichedMusicRelease(
        release_title="Discovery",
        artists=[artist_credit],
        release_date="2001-02-26",
        album_type="album",
        spotify_album_id="track-album",
        spotify_url="https://open.spotify.com/album/track-album",
        cover_url=None,
    )
    return ExtractionResults(
        screen_works=ScreenWorkResults(
            movies=[
                MovieResult(
                    status=ResultStatus.RESOLVED,
                    movie_mention=movie_mention,
                    movie=EnrichedMovie(
                        title="Dune: Part One",
                        year=2021,
                        cast=["Timothée Chalamet"],
                        directors=["Denis Villeneuve"],
                        description="Paul Atreides faces his destiny on Arrakis.",
                        poster_url=None,
                        tmdb_id=438631,
                        tmdb_url="https://www.themoviedb.org/movie/438631",
                        imdb_id=None,
                        imdb_url=None,
                        tmdb_score=7.8,
                    ),
                ),
                MovieResult(
                    status=ResultStatus.UNRESOLVED,
                    movie_mention=unresolved_movie_mention,
                    movie=None,
                ),
            ],
            tv_series=[
                TVSeriesResult(
                    status=ResultStatus.RESOLVED,
                    tv_series_mention=tv_series_mention,
                    tv_series=EnrichedTVSeries(
                        title="The Last of Us",
                        first_air_year=2023,
                        last_air_year=None,
                        cast=["Pedro Pascal", "Bella Ramsey"],
                        creators=["Craig Mazin", "Neil Druckmann"],
                        description="A smuggler escorts a teenager across a ruined America.",
                        poster_url=None,
                        tmdb_id=100088,
                        tmdb_url="https://www.themoviedb.org/tv/100088",
                        imdb_id="tt3581920",
                        imdb_url="https://www.imdb.com/title/tt3581920/",
                        tmdb_score=8.6,
                    ),
                ),
                TVSeriesResult(
                    status=ResultStatus.UNRESOLVED,
                    tv_series_mention=unresolved_tv_series_mention,
                    tv_series=None,
                ),
            ],
        ),
        music=MusicResults(
            tracks=[
                TrackResult(
                    status=ResultStatus.RESOLVED,
                    track_mention=track_mention,
                    track=EnrichedTrack(
                        track_title="One More Time (2011 Remaster)",
                        artists=[artist_credit],
                        spotify_track_id="0DiWol3AO6WpXZgp0goxAV",
                        spotify_url="https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
                        music_release=track_release,
                        cover_url=None,
                    ),
                ),
                TrackResult(
                    status=ResultStatus.UNRESOLVED,
                    track_mention=unresolved_track_mention,
                    track=None,
                ),
            ],
            music_releases=[
                MusicReleaseResult(
                    status=ResultStatus.RESOLVED,
                    music_release_mention=music_release_mention,
                    music_release=EnrichedMusicRelease(
                        release_title="Discovery (Deluxe Edition)",
                        artists=[artist_credit],
                        release_date="2001-02-26",
                        album_type="album",
                        spotify_album_id="direct-album",
                        spotify_url="https://open.spotify.com/album/direct-album",
                        cover_url="https://i.scdn.co/image/direct-cover",
                    ),
                ),
                MusicReleaseResult(
                    status=ResultStatus.UNRESOLVED,
                    music_release_mention=unresolved_music_release_mention,
                    music_release=None,
                ),
            ],
        ),
        books=BookResults(
            books=[
                BookResult(
                    status=ResultStatus.RESOLVED,
                    book_mention=book_mention,
                    book=EnrichedBookWork(
                        title="Pride and Prejudice",
                        authors=[
                            EnrichedAuthorCredit(
                                open_library_author_id="OL21594A",
                                name="Jane Austen",
                                open_library_url=("https://openlibrary.org/authors/OL21594A"),
                            )
                        ],
                        open_library_work_id="OL66554W",
                        open_library_url="https://openlibrary.org/works/OL66554W",
                        edition=BookEdition(
                            title=None,
                            publication_year=None,
                            publishers=[],
                            isbn_10=[],
                            isbn_13=[],
                            open_library_edition_id="OL12345M",
                            open_library_url="https://openlibrary.org/books/OL12345M",
                            cover_url=None,
                        ),
                        cover_url=None,
                        cover_edition_id=None,
                    ),
                ),
                BookResult(
                    status=ResultStatus.UNRESOLVED,
                    book_mention=unresolved_book_mention,
                    book=None,
                ),
            ]
        ),
    )


def _expected_statistics() -> dict[str, dict[str, int]]:
    return {
        "movies": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tv_series": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tracks": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "music_releases": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "books": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    }


def _expected_results() -> dict[str, list[object]]:
    return {
        "movies": [
            {
                "status": "resolved",
                "movie_mention": {"title": "Dune: Part One", "year": 2021},
                "movie": {
                    "title": "Dune: Part One",
                    "year": 2021,
                    "cast": ["Timothée Chalamet"],
                    "directors": ["Denis Villeneuve"],
                    "description": "Paul Atreides faces his destiny on Arrakis.",
                    "poster_url": None,
                    "tmdb_id": 438631,
                    "tmdb_url": "https://www.themoviedb.org/movie/438631",
                    "imdb_id": None,
                    "imdb_url": None,
                    "tmdb_score": 7.8,
                },
            },
            {
                "status": "unresolved",
                "movie_mention": {"title": "Unknown Movie", "year": 2024},
                "movie": None,
            },
        ],
        "tv_series": [
            {
                "status": "resolved",
                "tv_series_mention": {"title": "The Last of Us", "year": 2023},
                "tv_series": {
                    "title": "The Last of Us",
                    "first_air_year": 2023,
                    "last_air_year": None,
                    "cast": ["Pedro Pascal", "Bella Ramsey"],
                    "creators": ["Craig Mazin", "Neil Druckmann"],
                    "description": "A smuggler escorts a teenager across a ruined America.",
                    "poster_url": None,
                    "tmdb_id": 100088,
                    "tmdb_url": "https://www.themoviedb.org/tv/100088",
                    "imdb_id": "tt3581920",
                    "imdb_url": "https://www.imdb.com/title/tt3581920/",
                    "tmdb_score": 8.6,
                },
            },
            {
                "status": "unresolved",
                "tv_series_mention": {"title": "Unknown TV Series", "year": 2024},
                "tv_series": None,
            },
        ],
        "tracks": [
            {
                "status": "resolved",
                "track_mention": {
                    "track_title": "One More Time",
                    "artists": ["Daft Punk"],
                    "release_title": "Discovery",
                    "release_year": 2001,
                },
                "track": {
                    "track_title": "One More Time (2011 Remaster)",
                    "artists": [
                        {
                            "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                            "name": "Daft Punk",
                        }
                    ],
                    "spotify_track_id": "0DiWol3AO6WpXZgp0goxAV",
                    "spotify_url": "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
                    "music_release": {
                        "release_title": "Discovery",
                        "artists": [
                            {
                                "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                                "name": "Daft Punk",
                            }
                        ],
                        "release_date": "2001-02-26",
                        "album_type": "album",
                        "spotify_album_id": "track-album",
                        "spotify_url": "https://open.spotify.com/album/track-album",
                        "cover_url": None,
                    },
                    "cover_url": None,
                },
            },
            {
                "status": "unresolved",
                "track_mention": {
                    "track_title": "Unknown Track",
                    "artists": ["Unknown Artist"],
                    "release_title": None,
                    "release_year": None,
                },
                "track": None,
            },
        ],
        "music_releases": [
            {
                "status": "resolved",
                "music_release_mention": {
                    "release_title": "Discovery",
                    "artists": ["Daft Punk"],
                    "release_year": 2001,
                },
                "music_release": {
                    "release_title": "Discovery (Deluxe Edition)",
                    "artists": [
                        {
                            "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                            "name": "Daft Punk",
                        }
                    ],
                    "release_date": "2001-02-26",
                    "album_type": "album",
                    "spotify_album_id": "direct-album",
                    "spotify_url": "https://open.spotify.com/album/direct-album",
                    "cover_url": "https://i.scdn.co/image/direct-cover",
                },
            },
            {
                "status": "unresolved",
                "music_release_mention": {
                    "release_title": "Unknown Album",
                    "artists": ["Unknown Artist"],
                    "release_year": None,
                },
                "music_release": None,
            },
        ],
        "books": [
            {
                "status": "resolved",
                "book_mention": {
                    "title": "Pride and Prejudice",
                    "authors": ["Jane Austen"],
                },
                "book": {
                    "title": "Pride and Prejudice",
                    "authors": [
                        {
                            "open_library_author_id": "OL21594A",
                            "name": "Jane Austen",
                            "open_library_url": "https://openlibrary.org/authors/OL21594A",
                        }
                    ],
                    "open_library_work_id": "OL66554W",
                    "open_library_url": "https://openlibrary.org/works/OL66554W",
                    "edition": {
                        "title": None,
                        "publication_year": None,
                        "publishers": [],
                        "isbn_10": [],
                        "isbn_13": [],
                        "open_library_edition_id": "OL12345M",
                        "open_library_url": "https://openlibrary.org/books/OL12345M",
                        "cover_url": None,
                    },
                    "cover_url": None,
                    "cover_edition_id": None,
                },
            },
            {
                "status": "unresolved",
                "book_mention": {"title": "Unknown Book", "authors": []},
                "book": None,
            },
        ],
    }


def test_source_mapper_preserves_complete_resolved_and_unresolved_results() -> None:
    """Map every public result kind, its order, nested values, and resolution counts."""
    result = PipelineResult(
        source=Source(
            platform=Platform.YOUTUBE,
            video_id="dQw4w9WgXcQ",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Mapper test source",
            description="Mapper test description.",
            channel="Mapper test channel",
            duration_seconds=42,
        ),
        transcript=Transcript(
            text="A mapper test transcript.",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        ),
        results=_representative_results(),
        market=SpotifyMarket("JP"),
    )

    payload = map_extraction_response(result).model_dump(mode="json")

    assert payload == {
        "market": "JP",
        "source": {
            "platform": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Mapper test source",
            "description": "Mapper test description.",
            "channel": "Mapper test channel",
            "duration_seconds": 42,
        },
        "transcript": {
            "text": "A mapper test transcript.",
            "language": "en",
            "method": "youtube_captions",
        },
        "statistics": _expected_statistics(),
        "results": _expected_results(),
    }


def test_transcript_mapper_preserves_complete_contract_without_source() -> None:
    """Map a direct Transcript Submission without fabricating a Source field."""
    result = TranscriptPipelineResult(
        transcript=Transcript(
            text="Dune: Part One (2021) was excellent.",
            language="und",
            method=TranscriptMethod.TEXT_SUBMISSION,
        ),
        results=_representative_results(),
        market=SpotifyMarket("JP"),
    )

    payload = map_transcript_extraction_response(result).model_dump(mode="json")

    assert "source" not in payload
    assert payload == {
        "market": "JP",
        "transcript": {
            "text": "Dune: Part One (2021) was excellent.",
            "language": "und",
            "method": "text_submission",
        },
        "statistics": _expected_statistics(),
        "results": _expected_results(),
    }
