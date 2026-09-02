"""Extraction result aggregation contract tests."""

import pytest

from reelio.extraction.exceptions import EnrichmentError, PipelineTimeoutError
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
from reelio.extraction.types import (
    EnrichedMovie,
    ExtractionMentions,
    MovieMention,
    MovieResult,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    TVSeriesMention,
    TVSeriesResult,
)
from tests.extraction.fakes import FakeScreenWorkResolver


@pytest.mark.parametrize(
    "screen_work_mentions",
    [
        ScreenWorkMentions(movies=[], tv_series=[]),
        ScreenWorkMentions(
            movies=[MovieMention(title="Dune: Part One", year=2021)],
            tv_series=[],
        ),
        ScreenWorkMentions(
            movies=[],
            tv_series=[TVSeriesMention(title="The Last of Us", year=2023)],
        ),
        ScreenWorkMentions(
            movies=[MovieMention(title="Dune: Part One", year=2021)],
            tv_series=[TVSeriesMention(title="The Last of Us", year=2023)],
        ),
    ],
)
async def test_aggregator_delegates_the_exact_nested_screen_work_mentions(
    screen_work_mentions: ScreenWorkMentions,
) -> None:
    """Pass each grouped Screen Work input unchanged to its resolver once."""
    resolved_screen_works = ScreenWorkResults(movies=[], tv_series=[])
    resolver = FakeScreenWorkResolver(results=resolved_screen_works)
    aggregator = ExtractionResultAggregator(resolver)

    results = await aggregator.aggregate(ExtractionMentions(screen_works=screen_work_mentions))

    assert resolver.calls == [screen_work_mentions]
    assert resolver.calls[0] is screen_work_mentions
    assert results.screen_works is resolved_screen_works


async def test_aggregator_preserves_nested_results_and_kind_list_identity() -> None:
    """Wrap the resolver result without copying, reordering, or changing values."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    movie_results = [
        MovieResult(
            status=ResultStatus.RESOLVED,
            movie_mention=movie_mention,
            movie=EnrichedMovie(
                title="Dune: Part One",
                year=2021,
                cast=["Timothée Chalamet"],
                directors=["Denis Villeneuve"],
                description="A science-fiction epic.",
                poster_url=None,
                tmdb_id=438631,
                tmdb_url="https://www.themoviedb.org/movie/438631",
                imdb_id=None,
                imdb_url=None,
                tmdb_score=7.8,
            ),
        )
    ]
    tv_series_results = [
        TVSeriesResult(
            status=ResultStatus.UNRESOLVED,
            tv_series_mention=tv_series_mention,
            tv_series=None,
        )
    ]
    resolved_screen_works = ScreenWorkResults(
        movies=movie_results,
        tv_series=tv_series_results,
    )
    aggregator = ExtractionResultAggregator(FakeScreenWorkResolver(results=resolved_screen_works))

    results = await aggregator.aggregate(
        ExtractionMentions(
            screen_works=ScreenWorkMentions(
                movies=[movie_mention],
                tv_series=[tv_series_mention],
            )
        )
    )

    assert results.screen_works is resolved_screen_works
    assert results.screen_works.movies is movie_results
    assert results.screen_works.tv_series is tv_series_results
    assert results.screen_works.movies[0] is movie_results[0]
    assert results.screen_works.tv_series[0] is tv_series_results[0]


@pytest.mark.parametrize(
    "resolver_error",
    [
        EnrichmentError("TMDB candidate resolution failed."),
        PipelineTimeoutError("TMDB candidate resolution timed out."),
    ],
)
async def test_aggregator_propagates_resolver_errors_without_partial_results(
    resolver_error: Exception,
) -> None:
    """Preserve resolver failure identity and do not return partial grouped results."""
    aggregator = ExtractionResultAggregator(FakeScreenWorkResolver(error=resolver_error))

    with pytest.raises(type(resolver_error)) as error:
        await aggregator.aggregate(
            ExtractionMentions(screen_works=ScreenWorkMentions(movies=[], tv_series=[]))
        )

    assert error.value is resolver_error


class _ClosingScreenWorkResolver:
    def __init__(self) -> None:
        self.close_calls = 0

    async def resolve(
        self,
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        raise AssertionError(f"unexpected resolution: {screen_work_mentions}")

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_aggregator_closes_the_screen_work_resolver_once() -> None:
    """Delegate the pipeline's single shutdown call to the nested resolver."""
    resolver = _ClosingScreenWorkResolver()
    aggregator = ExtractionResultAggregator(resolver)

    await aggregator.aclose()

    assert resolver.close_calls == 1
