"""HTTP routing and domain-to-schema conversion for extraction."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.response_mapper import (
    map_extraction_response,
    map_transcript_extraction_response,
)
from reelio.extraction.service import ExtractionPipelineProtocol

_EXTRACT_RESPONSE_EXAMPLE = {
    "market": "US",
    "source": {
        "platform": "youtube",
        "video_id": "dQw4w9WgXcQ",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Screen Work, Music, and Book Work review",
        "description": "A review mentioning Movies, TV Series, Music, and Book Works.",
        "channel": "Example channel",
        "duration_seconds": 42,
    },
    "transcript": {
        "text": (
            "Dune: Part One, The Last of Us, One More Time, Discovery, and "
            "Pride and Prejudice are excellent."
        ),
        "language": "en",
        "method": "youtube_captions",
    },
    "statistics": {
        "movies": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tv_series": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "tracks": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "music_releases": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
        "books": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    },
    "results": {
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
                    "poster_url": "https://image.tmdb.org/t/p/w500/dune.jpg",
                    "tmdb_id": 438631,
                    "tmdb_url": "https://www.themoviedb.org/movie/438631",
                    "imdb_id": "tt1160419",
                    "imdb_url": "https://www.imdb.com/title/tt1160419/",
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
                    "last_air_year": 2025,
                    "cast": ["Pedro Pascal", "Bella Ramsey"],
                    "creators": ["Craig Mazin", "Neil Druckmann"],
                    "description": "A smuggler escorts a teenager across a ruined America.",
                    "poster_url": "https://image.tmdb.org/t/p/w500/the-last-of-us.jpg",
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
                        "release_title": "Discovery (Deluxe Edition)",
                        "artists": [
                            {
                                "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                                "name": "Daft Punk",
                            }
                        ],
                        "release_date": "2001-02-26",
                        "album_type": "album",
                        "spotify_album_id": "2noRn2Aes5aoNVsU6iWThc",
                        "spotify_url": "https://open.spotify.com/album/2noRn2Aes5aoNVsU6iWThc",
                        "cover_url": "https://i.scdn.co/image/discovery-cover",
                    },
                    "cover_url": "https://i.scdn.co/image/discovery-cover",
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
                    "spotify_album_id": "2noRn2Aes5aoNVsU6iWThc",
                    "spotify_url": "https://open.spotify.com/album/2noRn2Aes5aoNVsU6iWThc",
                    "cover_url": "https://i.scdn.co/image/discovery-cover",
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
                        "title": "Pride and Prejudice: A Collector's Edition",
                        "publication_year": 1813,
                        "publishers": ["T. Egerton"],
                        "isbn_10": ["0141439513"],
                        "isbn_13": ["9780141439518"],
                        "open_library_edition_id": "OL12345M",
                        "open_library_url": "https://openlibrary.org/books/OL12345M",
                        "cover_url": "https://covers.openlibrary.org/b/id/12345-L.jpg",
                    },
                    "cover_url": "https://covers.openlibrary.org/b/id/12345-L.jpg",
                    "cover_edition_id": "OL12345M",
                },
            },
            {
                "status": "unresolved",
                "book_mention": {
                    "title": "Unknown Book",
                    "authors": [],
                },
                "book": None,
            },
        ],
    },
}


_TRANSCRIPT_EXTRACT_RESPONSE_EXAMPLE = {
    key: value for key, value in _EXTRACT_RESPONSE_EXAMPLE.items() if key != "source"
}


router = APIRouter(prefix="/api", tags=["extraction"])
internal_router = APIRouter(prefix="/api/internal", tags=["internal extraction"])


def get_pipeline(request: Request) -> ExtractionPipelineProtocol:
    """Provide the lifespan-owned extraction pipeline.

    Args:
        request: Current request whose application owns the pipeline.

    Returns:
        ExtractionPipelineProtocol: The composed extraction pipeline.
    """
    return cast(ExtractionPipelineProtocol, request.app.state.extraction_pipeline)


@router.post(
    "/extractions",
    status_code=status.HTTP_200_OK,
    response_model=extraction_schemas.ExtractResponse,
    summary="Extract mentioned Movies, TV Series, Tracks, Music Releases, and Book Works from a public video Source",
    description=(
        "Textify backs Source inspection and Transcript acquisition; callers still receive "
        "Reelio's synchronous response and Reelio-owned errors. "
        "Accept a public YouTube, Instagram, Facebook, TikTok, or X video URL and "
        "return the normalized Source, the Transcript with its acquisition method, "
        "the effective Spotify market, category-specific result statistics, and "
        "grouped Movie, TV Series, Track, Music Release, and Book Work results. Each "
        "statistics category counts returned results, Resolved Results, and Unresolved "
        "Results. "
        "Each list preserves first-reference order within its kind, with no cross-kind "
        "ordering. The optional market must use "
        "uppercase ISO 3166-1 alpha-2 syntax; an omitted market uses configured "
        "US. Resolved TV Series report their TV First Air Year, an optional final "
        "air year where null means unavailable rather than proof of continuation, "
        "Creators from TMDB created_by, and up to five aggregate cast names in "
        "provider order without role filtering or person deduplication. Track "
        "Results retain their interpreted Track Mention and expose Spotify's "
        "canonical Track title, ordered artist credits, playable Track ID, and "
        "direct URL after a verified match. A resolved Track's "
        "music_release is the Spotify Album attached to its accepted "
        "Track Candidate. Mention release context constrains Track Candidate "
        "matching, while the preferred Music Release always comes from that "
        "accepted attached Album. Track and Music Release cover_url values are "
        "the first provider-ordered Spotify-hosted Album image URL, or null when "
        "Spotify returns no Album images; a Track's cover_url equals its preferred "
        "Music Release cover_url, and the Album spotify_url is its artwork "
        "link-back. Music Release Results retain their interpreted Music Release "
        "Mention and expose one independently matched, market-specific Spotify "
        "Album identity, provider-reported release date, album type, and direct "
        "URL after a verified match; the contract makes no worldwide-edition, "
        "sibling-release, release-family, inferred-subtype, or "
        "earliest-worldwide-date claims. Book Work Results retain their interpreted "
        "Book Mention and expose the Open Library Work title, provider-ordered Author "
        "Credits, Work ID, canonical URL, and nullable provider-preferred Edition "
        "after identity verification. A resolved Book Work may have null edition, "
        "cover_url, and cover_edition_id fields when Open Library lacks eligible "
        "Edition or cover metadata. Book Work cover_url first uses the selected "
        "Edition's own cover, then accepted Work Search cover metadata. A fallback "
        "cover_edition_id is exposed when Open Library identifies its source Edition; "
        "the fallback never populates edition.cover_url. Missing artwork returns null "
        "Book Work cover fields without another request. An unresolved Book Work "
        "Result has book set to null. Edition selection uses English-first Open "
        "Library relevance, with one unrestricted fallback for missing or audiobook "
        "English results. Missing or audiobook Editions remain null, and selection is "
        "independent of Effective Market and Source Edition signals. Each Track "
        "Mention and Music Release "
        "Mention causes one Spotify search in the effective market at offset zero "
        "and limit three, using the Mention title and first ordered Artist Credit "
        "only. Candidates remain eligible when at least one Artist Credit matches "
        "any Mention Artist Credit under Music Identity Normalization, regardless "
        "of credit order, count, or unmatched additions. Reelio checks exact title "
        "equality across every artist-eligible Candidate in provider order before "
        "reusing the same bounded artist-eligible Candidate sequence without "
        "another Spotify search for controlled Equivalent Track Versions or "
        "Equivalent Music Release Editions. Equivalent Track Versions accept only "
        "complete trailing remaster and bonus designations, including four-digit "
        "remaster years. Deluxe, expanded, special, anniversary, and reissue "
        "designations are release-only Track variants and do not independently "
        "qualify a Track title. With explicit Music Release context, an accepted "
        "Track Candidate must have an attached Album title that is exact or "
        "equivalent under the complete Music Release Edition grammar; without "
        "release context, the attached Album does not constrain matching. Both "
        "fallbacks remove recognized trailing segments right to left from "
        "parentheses, brackets, a spaced hyphen, or colon boundaries before "
        "requiring equal normalized base titles. Track equivalence is symmetric. "
        "The Music Release grammar additionally accepts deluxe, expanded, special, "
        "anniversary, and reissue designations. Equivalent Music Release fallback "
        "accepts only album or single Candidates, excluding compilation Candidates. "
        "Both fallbacks reject blocked trailing segments containing live, remix, "
        "acoustic, instrumental, radio edit, karaoke, or tribute material; Music "
        "Release comparison also rejects greatest hits material. An Equivalent "
        "Track Version or Equivalent Music Release Edition returns the ordinary "
        "Resolved Result with Spotify identity and metadata unchanged. Release year "
        "does not participate in retrieval or Candidate verification, and a "
        "completed search without an exact or eligible equivalent Candidate returns "
        "an Unresolved Result. Any TMDB, Spotify, or Open Library provider failure "
        "fails the complete request."
    ),
    response_description=(
        "Effective market, Source, transcript, category-specific result statistics, "
        "and grouped Movie, TV Series, Track, Music Release, and Book Work results."
    ),
    responses={
        200: {
            "description": "Result statistics and grouped Movie, TV Series, Track, Music Release, and Book Work results.",
            "content": {
                "application/json": {"example": _EXTRACT_RESPONSE_EXAMPLE},
            },
        },
        400: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Invalid URL, unsupported platform, or unsupported Source.",
        },
        404: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Source unavailable, private, or not found.",
        },
        413: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Source duration or Interpretation Material exceeds its limit.",
        },
        500: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Unexpected internal failure.",
        },
        502: {
            "model": extraction_schemas.ErrorResponse,
            "description": (
                "Source or Transcript provider transport or contract failure. Any TMDB, "
                "Spotify, or Open Library provider failure fails the complete request. Open "
                "Library failures are atomic with catalog_provider_failed and message Open "
                "Library catalog request failed."
            ),
        },
        503: {
            "model": extraction_schemas.ErrorResponse,
            "description": (
                "Source-and-Transcript acquisition or durable coordination is temporarily "
                "unavailable."
            ),
            "headers": {
                "Retry-After": {
                    "description": "Whole seconds to wait before retrying temporary unavailability.",
                    "schema": {"type": "integer"},
                },
            },
        },
        504: {
            "model": extraction_schemas.ErrorResponse,
            "description": (
                "External provider timeout, including exhausted Source-and-Transcript "
                "provider transport retries. Open Library timeouts are atomic with "
                "pipeline_timeout and message Open Library catalog request timed out."
            ),
        },
    },
)
async def extract(
    payload: extraction_schemas.ExtractRequest,
    pipeline: Annotated[ExtractionPipelineProtocol, Depends(get_pipeline)],
) -> extraction_schemas.ExtractResponse:
    """Extract structured Movie, TV Series, Track, Music Release, and Book Work Mentions.

    Args:
        payload: Validated extraction request containing source URL and optional market.
        pipeline: ExtractionPipelineProtocol implementation supplied by FastAPI dependency injection.

    Returns:
        ExtractResponse: Effective market, canonical source, transcript, and results.

    Raises:
        ExtractionError: If the pipeline raises an extraction domain error.
    """
    result = await pipeline.run(payload.url, payload.market)
    return map_extraction_response(result)


@internal_router.post(
    "/extractions",
    status_code=status.HTTP_200_OK,
    response_model=extraction_schemas.TranscriptExtractResponse,
    summary="Extract mentioned works from submitted Transcript text",
    description=(
        "Accept Transcript text supplied directly for internal testing and return the "
        "normalized Transcript, the effective Spotify market, category-specific result "
        "statistics, and grouped Movie, TV Series, Track, Music Release, and Book Work "
        "results. This endpoint has no Source and skips Source inspection and Transcript "
        "acquisition. The optional market must use uppercase ISO 3166-1 alpha-2 syntax; "
        "an omitted market uses configured US."
    ),
    response_description=(
        "Effective market, normalized Transcript, category-specific result statistics, "
        "and grouped Movie, TV Series, Track, Music Release, and Book Work results."
    ),
    responses={
        200: {
            "description": (
                "Result statistics and grouped Movie, TV Series, Track, Music Release, "
                "and Book Work results."
            ),
            "content": {
                "application/json": {
                    "example": _TRANSCRIPT_EXTRACT_RESPONSE_EXAMPLE,
                }
            },
        },
        413: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Interpretation Material exceeds its configured limit.",
        },
        500: {
            "model": extraction_schemas.ErrorResponse,
            "description": "Unexpected internal failure.",
        },
        502: {
            "model": extraction_schemas.ErrorResponse,
            "description": (
                "LLM, TMDB, Spotify, or Open Library provider failure. Any TMDB, "
                "Spotify, or Open Library provider failure fails the complete request."
            ),
        },
        504: {
            "model": extraction_schemas.ErrorResponse,
            "description": "External provider timeout.",
        },
    },
)
async def extract_transcript(
    payload: extraction_schemas.TranscriptExtractRequest,
    pipeline: Annotated[ExtractionPipelineProtocol, Depends(get_pipeline)],
) -> extraction_schemas.TranscriptExtractResponse:
    """Extract structured mentions from submitted Transcript text.

    Args:
        payload: Validated Transcript and optional market for internal extraction.
        pipeline: Extraction pipeline supplied by FastAPI dependency injection.

    Returns:
        TranscriptExtractResponse: Effective market, Transcript, and grouped results.

    Raises:
        ExtractionError: If the pipeline raises an extraction domain error.
    """
    result = await pipeline.run_transcript(payload.transcript, payload.market)
    return map_transcript_extraction_response(result)
