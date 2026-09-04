"""Extraction result aggregation contract tests."""

import pytest

from reelio.extraction.exceptions import (
    CatalogProviderError,
    EnrichmentError,
    PipelineTimeoutError,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
from reelio.extraction.types import (
    EnrichedMovie,
    ExtractionMentions,
    MovieMention,
    MovieResult,
    MusicMentions,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    TrackMention,
    TrackResult,
    TVSeriesMention,
    TVSeriesResult,
)
from tests.extraction.fakes import FakeScreenWorkResolver, FakeTrackResolver

_MARKET = SpotifyMarket("JP")


async def test_aggregator_resolves_both_scopes_with_the_effective_market() -> None:
    """Pass grouped mentions unchanged to both resolvers and preserve their results."""
    screen_work_mentions = ScreenWorkMentions(
        movies=[MovieMention(title="Dune: Part One", year=2021)],
        tv_series=[TVSeriesMention(title="The Last of Us", year=2023)],
    )
    track_mention = TrackMention(
        track_title="One More Time",
        artists=["Daft Punk"],
        release_title=None,
        release_year=None,
    )
    music_mentions = MusicMentions(
        tracks=[track_mention],
        music_releases=[],
    )
    resolved_screen_works = ScreenWorkResults(movies=[], tv_series=[])
    resolved_tracks = [
        TrackResult(
            status=ResultStatus.UNRESOLVED,
            track_mention=track_mention,
            track=None,
        )
    ]
    screen_work_resolver = FakeScreenWorkResolver(results=resolved_screen_works)
    track_resolver = FakeTrackResolver(results=resolved_tracks)
    aggregator = ExtractionResultAggregator(screen_work_resolver, track_resolver)

    results = await aggregator.aggregate(
        ExtractionMentions(
            screen_works=screen_work_mentions,
            music=music_mentions,
        ),
        _MARKET,
    )

    assert screen_work_resolver.calls == [screen_work_mentions]
    assert track_resolver.calls == [(music_mentions.tracks, _MARKET)]
    assert track_resolver.calls[0][0] is music_mentions.tracks
    assert results.screen_works is resolved_screen_works
    assert results.music.tracks is resolved_tracks


async def test_aggregator_preserves_nested_results_and_kind_list_identity() -> None:
    """Wrap resolver results without copying, reordering, or changing values."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    track_mention = TrackMention(
        track_title="One More Time",
        artists=["Daft Punk"],
        release_title=None,
        release_year=None,
    )
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
    track_results = [
        TrackResult(
            status=ResultStatus.UNRESOLVED,
            track_mention=track_mention,
            track=None,
        )
    ]
    resolved_screen_works = ScreenWorkResults(
        movies=movie_results,
        tv_series=tv_series_results,
    )
    aggregator = ExtractionResultAggregator(
        FakeScreenWorkResolver(results=resolved_screen_works),
        FakeTrackResolver(results=track_results),
    )

    results = await aggregator.aggregate(
        ExtractionMentions(
            screen_works=ScreenWorkMentions(
                movies=[movie_mention],
                tv_series=[tv_series_mention],
            ),
            music=MusicMentions(tracks=[track_mention], music_releases=[]),
        ),
        _MARKET,
    )

    assert results.screen_works is resolved_screen_works
    assert results.screen_works.movies is movie_results
    assert results.screen_works.tv_series is tv_series_results
    assert results.music.tracks is track_results


@pytest.mark.parametrize(
    "resolver_error",
    [
        EnrichmentError("TMDB candidate resolution failed."),
        CatalogProviderError("Spotify catalog request failed."),
        PipelineTimeoutError("provider request timed out."),
    ],
)
async def test_aggregator_propagates_resolver_errors_without_partial_results(
    resolver_error: Exception,
) -> None:
    """Propagate either resolver failure without constructing partial grouped results."""
    screen_work_resolver = FakeScreenWorkResolver()
    track_resolver = FakeTrackResolver()
    if isinstance(resolver_error, EnrichmentError):
        screen_work_resolver = FakeScreenWorkResolver(error=resolver_error)
    else:
        track_resolver = FakeTrackResolver(error=resolver_error)
    aggregator = ExtractionResultAggregator(screen_work_resolver, track_resolver)

    with pytest.raises(type(resolver_error)) as error:
        await aggregator.aggregate(
            ExtractionMentions(
                screen_works=ScreenWorkMentions(movies=[], tv_series=[]),
                music=MusicMentions(tracks=[], music_releases=[]),
            ),
            _MARKET,
        )

    assert error.value is resolver_error


class _ClosingScreenWorkResolver:
    """Record the aggregator-owned Screen Work resolver lifecycle."""

    def __init__(self) -> None:
        """Initialize an uncalled resolver with an observable close count."""
        self.close_calls = 0

    async def resolve(
        self,
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        """Fail if aggregation unexpectedly invokes this resolver."""
        raise AssertionError(f"unexpected resolution: {screen_work_mentions}")

    async def aclose(self) -> None:
        """Record one close request."""
        self.close_calls += 1


async def test_aggregator_closes_only_the_screen_work_resolver() -> None:
    """Leave lifespan-owned Spotify catalog closure outside the aggregator."""
    resolver = _ClosingScreenWorkResolver()
    aggregator = ExtractionResultAggregator(resolver, FakeTrackResolver())

    await aggregator.aclose()

    assert resolver.close_calls == 1
