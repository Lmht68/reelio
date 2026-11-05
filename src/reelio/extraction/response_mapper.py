"""Map Extraction domain results to the public response schemas."""

from collections.abc import Sequence

from reelio.extraction import schemas as extraction_schemas
from reelio.extraction.types import (
    ArtistCredit,
    BookEdition,
    BookMention,
    BookResult,
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
    PipelineResult,
    ResultStatus,
    TrackMention,
    TrackResult,
    Transcript,
    TranscriptPipelineResult,
    TVSeriesMention,
    TVSeriesResult,
)

_ExtractionResult = MovieResult | TVSeriesResult | TrackResult | MusicReleaseResult | BookResult


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
        music_release=_to_music_release_schema(track.music_release),
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


def _to_book_mention_schema(
    mention: BookMention,
) -> extraction_schemas.BookMentionModel:
    return extraction_schemas.BookMentionModel(
        title=mention.title,
        authors=[author_credit.name for author_credit in mention.authors],
    )


def _to_enriched_author_credit_schema(
    author_credit: EnrichedAuthorCredit,
) -> extraction_schemas.EnrichedAuthorCreditModel:
    return extraction_schemas.EnrichedAuthorCreditModel(
        open_library_author_id=author_credit.open_library_author_id,
        name=author_credit.name,
        open_library_url=author_credit.open_library_url,
    )


def _to_book_edition_schema(
    edition: BookEdition,
) -> extraction_schemas.BookEditionModel:
    return extraction_schemas.BookEditionModel(
        title=edition.title,
        publication_year=edition.publication_year,
        publishers=edition.publishers,
        isbn_10=edition.isbn_10,
        isbn_13=edition.isbn_13,
        open_library_edition_id=edition.open_library_edition_id,
        open_library_url=edition.open_library_url,
        cover_url=edition.cover_url,
    )


def _to_book_schema(book: EnrichedBookWork) -> extraction_schemas.BookModel:
    return extraction_schemas.BookModel(
        title=book.title,
        authors=[
            _to_enriched_author_credit_schema(author_credit) for author_credit in book.authors
        ],
        open_library_work_id=book.open_library_work_id,
        open_library_url=book.open_library_url,
        edition=(_to_book_edition_schema(book.edition) if book.edition is not None else None),
        cover_url=book.cover_url,
        cover_edition_id=book.cover_edition_id,
    )


def _to_book_result_schema(
    result: BookResult,
) -> extraction_schemas.BookResultModel:
    book = _to_book_schema(result.book) if result.book is not None else None
    return extraction_schemas.BookResultModel(
        status=result.status,
        book_mention=_to_book_mention_schema(result.book_mention),
        book=book,
    )


def _to_result_counts_schema(
    results: Sequence[_ExtractionResult],
) -> extraction_schemas.ResultCountsModel:
    n_mentions = len(results)
    n_resolved = sum(item.status is ResultStatus.RESOLVED for item in results)
    return extraction_schemas.ResultCountsModel(
        n_mentions=n_mentions,
        n_resolved=n_resolved,
        n_unresolved=n_mentions - n_resolved,
    )


def _to_transcript_model(transcript: Transcript) -> extraction_schemas.TranscriptModel:
    return extraction_schemas.TranscriptModel(
        text=transcript.text,
        language=transcript.language,
        method=transcript.method,
    )


def _to_statistics_model(
    results: ExtractionResults,
) -> extraction_schemas.ExtractionStatisticsModel:
    return extraction_schemas.ExtractionStatisticsModel(
        movies=_to_result_counts_schema(results.screen_works.movies),
        tv_series=_to_result_counts_schema(results.screen_works.tv_series),
        tracks=_to_result_counts_schema(results.music.tracks),
        music_releases=_to_result_counts_schema(results.music.music_releases),
        books=_to_result_counts_schema(results.books.books),
    )


def _to_results_model(
    results: ExtractionResults,
) -> extraction_schemas.ExtractionResultsModel:
    return extraction_schemas.ExtractionResultsModel(
        movies=[_to_movie_result_schema(item) for item in results.screen_works.movies],
        tv_series=[_to_tv_series_result_schema(item) for item in results.screen_works.tv_series],
        tracks=[_to_track_result_schema(item) for item in results.music.tracks],
        music_releases=[
            _to_music_release_result_schema(item) for item in results.music.music_releases
        ],
        books=[_to_book_result_schema(item) for item in results.books.books],
    )


def map_extraction_response(result: PipelineResult) -> extraction_schemas.ExtractResponse:
    """Map one Source Extraction result to the existing public response schema.

    Args:
        result: Complete pipeline result for a submitted Source.

    Returns:
        ExtractResponse: Public response containing Source, Transcript, and Results.
    """
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
        transcript=_to_transcript_model(result.transcript),
        statistics=_to_statistics_model(result.results),
        results=_to_results_model(result.results),
    )


def map_transcript_extraction_response(
    result: TranscriptPipelineResult,
) -> extraction_schemas.TranscriptExtractResponse:
    """Map one direct Transcript Submission result to the public response schema.

    Args:
        result: Complete pipeline result for submitted Transcript text.

    Returns:
        TranscriptExtractResponse: Public response excluding a Source.
    """
    return extraction_schemas.TranscriptExtractResponse(
        market=result.market,
        transcript=_to_transcript_model(result.transcript),
        statistics=_to_statistics_model(result.results),
        results=_to_results_model(result.results),
    )
