"""HTTP routing and domain-to-schema conversion for extraction."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.service import ExtractionPipelineProtocol
from reelio.extraction.types import (
    ArtistCredit,
    EnrichedMovie,
    EnrichedMusicRelease,
    EnrichedTrack,
    EnrichedTVSeries,
    MovieMention,
    MovieResult,
    MusicReleaseMention,
    MusicReleaseResult,
    PipelineResult,
    TrackMention,
    TrackResult,
    TVSeriesMention,
    TVSeriesResult,
)

_EXTRACT_RESPONSE_EXAMPLE = {
    "market": "US",
    "source": {
        "platform": "youtube",
        "video_id": "dQw4w9WgXcQ",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Screen Work and Music review",
        "description": "A review mentioning Movies, TV Series, and Music.",
        "channel": "Example channel",
        "duration_seconds": 42,
    },
    "transcript": {
        "text": "Dune: Part One, The Last of Us, One More Time, and Discovery are excellent.",
        "language": "en",
        "method": "youtube_captions",
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
                    "track_title": "One More Time",
                    "artists": [
                        {
                            "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                            "name": "Daft Punk",
                        }
                    ],
                    "spotify_track_id": "0DiWol3AO6WpXZgp0goxAV",
                    "spotify_url": "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
                    "preferred_music_release": {
                        "release_title": "Discovery",
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
    },
}


router = APIRouter(prefix="/api", tags=["extraction"])


def get_pipeline(request: Request) -> ExtractionPipelineProtocol:
    """Provide the lifespan-owned extraction pipeline.

    Args:
        request: Current request whose application owns the pipeline.

    Returns:
        ExtractionPipelineProtocol: The composed extraction pipeline.
    """
    return cast(ExtractionPipelineProtocol, request.app.state.extraction_pipeline)


def _to_movie_mention_schema(mention: MovieMention) -> extraction_schemas.MovieMentionModel:
    return extraction_schemas.MovieMentionModel(
        title=mention.title,
        year=mention.year,
    )


def _to_tv_series_mention_schema(
    mention: TVSeriesMention,
) -> extraction_schemas.TVSeriesMentionModel:
    return extraction_schemas.TVSeriesMentionModel(
        title=mention.title,
        year=mention.year,
    )


def _to_track_mention_schema(
    mention: TrackMention,
) -> extraction_schemas.TrackMentionModel:
    return extraction_schemas.TrackMentionModel(
        track_title=mention.track_title,
        artists=mention.artists,
        release_title=mention.release_title,
        release_year=mention.release_year,
    )


def _to_artist_credit_schema(
    artist_credit: ArtistCredit,
) -> extraction_schemas.ArtistCreditModel:
    return extraction_schemas.ArtistCreditModel(
        spotify_artist_id=artist_credit.spotify_artist_id,
        name=artist_credit.name,
    )


def _to_movie_schema(movie: EnrichedMovie) -> extraction_schemas.MovieModel:
    return extraction_schemas.MovieModel(
        title=movie.title,
        year=movie.year,
        cast=movie.cast,
        directors=movie.directors,
        description=movie.description,
        poster_url=movie.poster_url,
        tmdb_id=movie.tmdb_id,
        tmdb_url=movie.tmdb_url,
        imdb_id=movie.imdb_id,
        imdb_url=movie.imdb_url,
        tmdb_score=movie.tmdb_score,
    )


def _to_tv_series_schema(
    tv_series: EnrichedTVSeries,
) -> extraction_schemas.TVSeriesModel:
    return extraction_schemas.TVSeriesModel(
        title=tv_series.title,
        first_air_year=tv_series.first_air_year,
        last_air_year=tv_series.last_air_year,
        cast=tv_series.cast,
        creators=tv_series.creators,
        description=tv_series.description,
        poster_url=tv_series.poster_url,
        tmdb_id=tv_series.tmdb_id,
        tmdb_url=tv_series.tmdb_url,
        imdb_id=tv_series.imdb_id,
        imdb_url=tv_series.imdb_url,
        tmdb_score=tv_series.tmdb_score,
    )


def _to_music_release_schema(
    music_release: EnrichedMusicRelease,
) -> extraction_schemas.MusicReleaseModel:
    return extraction_schemas.MusicReleaseModel(
        release_title=music_release.release_title,
        artists=[_to_artist_credit_schema(artist) for artist in music_release.artists],
        release_date=music_release.release_date,
        album_type=music_release.album_type,
        spotify_album_id=music_release.spotify_album_id,
        spotify_url=music_release.spotify_url,
        cover_url=music_release.cover_url,
    )


def _to_track_schema(track: EnrichedTrack) -> extraction_schemas.TrackModel:
    return extraction_schemas.TrackModel(
        track_title=track.track_title,
        artists=[_to_artist_credit_schema(artist) for artist in track.artists],
        spotify_track_id=track.spotify_track_id,
        spotify_url=track.spotify_url,
        preferred_music_release=_to_music_release_schema(track.preferred_music_release),
        cover_url=track.cover_url,
    )


def _to_movie_result_schema(
    result: MovieResult,
) -> extraction_schemas.MovieResultModel:
    movie = _to_movie_schema(result.movie) if result.movie is not None else None
    return extraction_schemas.MovieResultModel(
        status=result.status,
        movie_mention=_to_movie_mention_schema(result.movie_mention),
        movie=movie,
    )


def _to_tv_series_result_schema(
    result: TVSeriesResult,
) -> extraction_schemas.TVSeriesResultModel:
    tv_series = _to_tv_series_schema(result.tv_series) if result.tv_series is not None else None
    return extraction_schemas.TVSeriesResultModel(
        status=result.status,
        tv_series_mention=_to_tv_series_mention_schema(result.tv_series_mention),
        tv_series=tv_series,
    )


def _to_track_result_schema(
    result: TrackResult,
) -> extraction_schemas.TrackResultModel:
    track = _to_track_schema(result.track) if result.track is not None else None
    return extraction_schemas.TrackResultModel(
        status=result.status,
        track_mention=_to_track_mention_schema(result.track_mention),
        track=track,
    )


def _to_music_release_mention_schema(
    mention: MusicReleaseMention,
) -> extraction_schemas.MusicReleaseMentionModel:
    return extraction_schemas.MusicReleaseMentionModel(
        release_title=mention.release_title,
        artists=mention.artists,
        release_year=mention.release_year,
    )


def _to_music_release_result_schema(
    result: MusicReleaseResult,
) -> extraction_schemas.MusicReleaseResultModel:
    music_release = (
        _to_music_release_schema(result.music_release) if result.music_release is not None else None
    )
    return extraction_schemas.MusicReleaseResultModel(
        status=result.status,
        music_release_mention=_to_music_release_mention_schema(result.music_release_mention),
        music_release=music_release,
    )


def _to_response(result: PipelineResult) -> extraction_schemas.ExtractResponse:
    return extraction_schemas.ExtractResponse(
        market=result.market,
        source=extraction_schemas.SourceModel(
            platform=result.source.platform,
            video_id=result.source.video_id,
            url=result.source.url,
            title=result.source.title,
            description=result.source.description,
            channel=result.source.channel,
            duration_seconds=result.source.duration_seconds,
        ),
        transcript=extraction_schemas.TranscriptModel(
            text=result.transcript.text,
            language=result.transcript.language,
            method=result.transcript.method,
        ),
        results=extraction_schemas.ExtractionResultsModel(
            movies=[_to_movie_result_schema(item) for item in result.results.screen_works.movies],
            tv_series=[
                _to_tv_series_result_schema(item) for item in result.results.screen_works.tv_series
            ],
            tracks=[_to_track_result_schema(item) for item in result.results.music.tracks],
            music_releases=[
                _to_music_release_result_schema(item)
                for item in result.results.music.music_releases
            ],
        ),
    )


@router.post(
    "/extract",
    status_code=status.HTTP_200_OK,
    response_model=extraction_schemas.ExtractResponse,
    summary="Extract mentioned Movies, TV Series, Tracks, and Music Releases from a public video Source",
    description=(
        "Accept a public YouTube, Instagram, Facebook, TikTok, or X video URL and "
        "return the normalized Source, the Transcript with its acquisition method, "
        "the effective Spotify market, and grouped Movie, TV Series, Track, and "
        "Music Release results. Each list preserves first-reference order within "
        "its kind, with no cross-kind ordering. The optional market must use "
        "uppercase ISO 3166-1 alpha-2 syntax; an omitted market uses configured "
        "US. Resolved TV Series report their TV First Air Year, an optional final "
        "air year where null means unavailable rather than proof of continuation, "
        "Creators from TMDB created_by, and up to five aggregate cast names in "
        "provider order without role filtering or person deduplication. Track "
        "Results retain their interpreted Track Mention and expose Spotify's "
        "canonical Track title, ordered artist credits, playable Track ID, and "
        "direct URL after a verified match. A resolved Track's "
        "preferred_music_release is the Spotify Album attached to its accepted "
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
        "earliest-worldwide-date claims. Each Track Mention and Music Release "
        "Mention causes one Spotify search in the effective market at offset zero "
        "and limit three, using the Mention title and first ordered Artist Credit "
        "only. Candidates remain eligible when at least one Artist Credit matches "
        "any Mention Artist Credit under Music Identity Normalization, regardless "
        "of credit order, count, or unmatched additions. Reelio selects the first "
        "provider-ordered eligible Track Candidate whose Track title, or Track title "
        "plus explicit attached Music Release title when supplied, matches exactly "
        "under Music Identity Normalization. For a direct Music Release, Reelio "
        "first checks exact title equality across every artist-eligible Candidate, "
        "including compilation Candidates. Only when no exact Candidate appears "
        "does it reuse the same bounded artist-eligible Candidate sequence in "
        "provider order for controlled Equivalent Music Release Editions without "
        "another Spotify search. Equivalent editions remove recognized trailing "
        "remaster, deluxe, expanded, special, anniversary, reissue, and bonus "
        "designations, including four-digit remaster years and numeric ordinal "
        "anniversaries, right to left from parentheses, brackets, a spaced hyphen, "
        "or colon boundaries before requiring equal normalized base titles. The "
        "fallback accepts only album or single Candidates, excludes compilation "
        "Candidates, and rejects blocked trailing segments containing live, remix, "
        "acoustic, instrumental, radio edit, karaoke, tribute, or greatest hits "
        "material. An equivalent match returns the ordinary Resolved Result with "
        "Spotify identity and metadata unchanged. Release year does not participate "
        "in retrieval or Candidate verification, and a completed search without an "
        "exact or eligible equivalent Candidate returns an Unresolved Result. Any "
        "TMDB or Spotify provider failure fails the complete request."
    ),
    response_description=(
        "Effective market, Source, transcript, and grouped Movie, TV Series, "
        "Track, and Music Release results."
    ),
    responses={
        200: {
            "description": ("Grouped Movie, TV Series, Track, and Music Release results."),
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
                "Metadata, transcript, LLM, TMDB, or Spotify provider failure. Any "
                "TMDB or Spotify provider failure fails the complete request."
            ),
        },
        504: {
            "model": extraction_schemas.ErrorResponse,
            "description": "External provider timeout.",
        },
    },
)
async def extract(
    payload: extraction_schemas.ExtractRequest,
    pipeline: Annotated[ExtractionPipelineProtocol, Depends(get_pipeline)],
) -> extraction_schemas.ExtractResponse:
    """Extract structured Movie, TV Series, Track, and Music Release Mentions from a source URL.

    Args:
        payload: Validated extraction request containing source URL and optional market.
        pipeline: ExtractionPipelineProtocol implementation supplied by FastAPI dependency injection.

    Returns:
        ExtractResponse: Effective market, canonical source, transcript, and results.

    Raises:
        ExtractionError: If the pipeline raises an extraction domain error.
    """
    result = await pipeline.run(payload.url, payload.market)
    return _to_response(result)
