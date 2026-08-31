"""Screen Work identity primitive contract tests."""

from datetime import date

from reelio.extraction.types import (
    MINIMUM_SCREEN_WORK_MENTION_YEAR,
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


def test_grouped_screen_work_domain_types_preserve_kind_and_field_placement() -> None:
    """Keep Movie and TV Series mentions and results in their grouped fields."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    mentions = ScreenWorkMentions(
        movies=[movie_mention],
        tv_series=[tv_series_mention],
    )
    movie_result = MovieResult(
        status=ResultStatus.UNRESOLVED,
        movie_mention=movie_mention,
        movie=None,
    )
    tv_series_result = TVSeriesResult(
        status=ResultStatus.UNRESOLVED,
        tv_series_mention=tv_series_mention,
        tv_series=None,
    )
    results = ScreenWorkResults(
        movies=[movie_result],
        tv_series=[tv_series_result],
    )
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

    assert pipeline_result.results.movies == [movie_result]
    assert pipeline_result.results.tv_series == [tv_series_result]
    assert mentions.movies == [movie_mention]
    assert mentions.tv_series == [tv_series_mention]
