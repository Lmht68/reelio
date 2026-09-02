"""Screen Work identity primitive contract tests."""

from datetime import date

from reelio.extraction.types import (
    MINIMUM_SCREEN_WORK_MENTION_YEAR,
    EnrichedTVSeries,
    ExtractionMentions,
    ExtractionResults,
    MovieMention,
    MovieResult,
    PipelineResult,
    Platform,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
    TranscriptMethod,
    TVSeriesMention,
    TVSeriesResult,
    maximum_screen_work_mention_year,
    normalize_screen_work_title,
)


def test_normalize_screen_work_title_canonicalizes_unicode_and_whitespace() -> None:
    """Normalize equivalent Screen Work title spellings to one identity."""
    assert (
        normalize_screen_work_title("  AME\u0301LIE:\tLe Fabuleux\nDestin!  ")
        == "AMÉLIE: Le Fabuleux Destin!"
    )


def test_screen_work_mention_year_policy() -> None:
    """Allow Screen Work Mention years from 1888 through two future years."""
    assert MINIMUM_SCREEN_WORK_MENTION_YEAR == 1888
    assert maximum_screen_work_mention_year() == date.today().year + 2


def test_extraction_domain_types_preserve_nested_screen_work_identity() -> None:
    """Retain exact nested Screen Work containers through pipeline output."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    screen_work_mentions = ScreenWorkMentions(
        movies=[movie_mention],
        tv_series=[tv_series_mention],
    )
    mentions = ExtractionMentions(screen_works=screen_work_mentions)
    movie_result = MovieResult(
        status=ResultStatus.UNRESOLVED,
        movie_mention=movie_mention,
        movie=None,
    )
    enriched_tv_series = EnrichedTVSeries(
        title=tv_series_mention.title,
        first_air_year=tv_series_mention.year,
        last_air_year=None,
        cast=["Pedro Pascal"],
        creators=["Craig Mazin"],
        description="A post-apocalyptic drama.",
        poster_url=None,
        tmdb_id=100088,
        tmdb_url="https://www.themoviedb.org/tv/100088",
        imdb_id=None,
        imdb_url=None,
        tmdb_score=8.6,
    )
    tv_series_result = TVSeriesResult(
        status=ResultStatus.RESOLVED,
        tv_series_mention=tv_series_mention,
        tv_series=enriched_tv_series,
    )
    screen_work_results = ScreenWorkResults(
        movies=[movie_result],
        tv_series=[tv_series_result],
    )
    results = ExtractionResults(screen_works=screen_work_results)
    pipeline_result = PipelineResult(
        source=Source(
            platform=Platform.YOUTUBE,
            video_id="source",
            url="https://example.com/source",
            title="Source",
            description="Description",
            channel="Channel",
            duration_seconds=1,
        ),
        transcript=Transcript(
            text="Transcript",
            language="en",
            method=TranscriptMethod.YOUTUBE_CAPTIONS,
        ),
        results=results,
    )

    assert pipeline_result.results is results
    assert pipeline_result.results.screen_works is screen_work_results
    assert pipeline_result.results.screen_works.movies == [movie_result]
    assert pipeline_result.results.screen_works.tv_series == [tv_series_result]
    assert mentions.screen_works is screen_work_mentions
    assert mentions.screen_works.movies == [movie_mention]
    assert mentions.screen_works.tv_series == [tv_series_mention]
