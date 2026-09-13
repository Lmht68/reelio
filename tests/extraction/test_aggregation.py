"""Extraction result aggregation contract tests."""

import asyncio

import pytest

from reelio.extraction.exceptions import (
    CatalogProviderError,
    EnrichmentError,
    PipelineTimeoutError,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.enrichment.service import ExtractionResultAggregator
from reelio.extraction.types import (
    BookMention,
    BookMentions,
    BookResult,
    BookResults,
    EnrichedMovie,
    ExtractionMentions,
    MovieMention,
    MovieResult,
    MusicMentions,
    MusicReleaseMention,
    MusicReleaseResult,
    MusicResults,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    TrackMention,
    TrackResult,
    TVSeriesMention,
    TVSeriesResult,
)
from tests.extraction.fakes import FakeBookResolver, FakeMusicResolver, FakeScreenWorkResolver

_MARKET = SpotifyMarket("JP")


async def test_aggregator_resolves_all_scopes_with_the_effective_market() -> None:
    """Pass grouped Mentions unchanged to every resolver and preserve their Results."""
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
    music_release_mention = MusicReleaseMention(
        release_title="Discovery",
        artists=["Daft Punk"],
        release_year=2001,
    )
    music_mentions = MusicMentions(
        tracks=[track_mention],
        music_releases=[music_release_mention],
    )
    book_mention = BookMention(title="Pride and Prejudice", authors=[])
    book_mentions = BookMentions(books=[book_mention])
    resolved_screen_works = ScreenWorkResults(movies=[], tv_series=[])
    resolved_tracks = [
        TrackResult(
            status=ResultStatus.UNRESOLVED,
            track_mention=track_mention,
            track=None,
        )
    ]
    resolved_music_releases = [
        MusicReleaseResult(
            status=ResultStatus.UNRESOLVED,
            music_release_mention=music_release_mention,
            music_release=None,
        )
    ]
    resolved_music = MusicResults(
        tracks=resolved_tracks,
        music_releases=resolved_music_releases,
    )
    resolved_books = BookResults(
        books=[
            BookResult(
                status=ResultStatus.UNRESOLVED,
                book_mention=book_mention,
                book=None,
            )
        ]
    )
    screen_work_resolver = FakeScreenWorkResolver(results=resolved_screen_works)
    music_resolver = FakeMusicResolver(results=resolved_music)
    book_resolver = FakeBookResolver(results=resolved_books)
    aggregator = ExtractionResultAggregator(
        screen_work_resolver,
        music_resolver,
        book_resolver,
    )
    mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=music_mentions,
        books=book_mentions,
    )

    results = await aggregator.aggregate(mentions, _MARKET)

    assert screen_work_resolver.calls == [screen_work_mentions]
    assert music_resolver.calls == [(music_mentions, _MARKET)]
    assert music_resolver.calls[0][0] is music_mentions
    assert book_resolver.calls == [book_mentions]
    assert results.screen_works is resolved_screen_works
    assert results.music is resolved_music
    assert results.books is resolved_books
    assert results.music.tracks is resolved_tracks
    assert results.music.music_releases is resolved_music_releases


async def test_aggregator_preserves_nested_results_and_kind_list_identity() -> None:
    """Wrap resolver Results without copying, reordering, or changing values."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    track_mention = TrackMention(
        track_title="One More Time",
        artists=["Daft Punk"],
        release_title=None,
        release_year=None,
    )
    music_release_mention = MusicReleaseMention(
        release_title="Discovery",
        artists=["Daft Punk"],
        release_year=2001,
    )
    book_mention = BookMention(title="Pride and Prejudice", authors=[])
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
    music_release_results = [
        MusicReleaseResult(
            status=ResultStatus.UNRESOLVED,
            music_release_mention=music_release_mention,
            music_release=None,
        )
    ]
    book_results = [
        BookResult(
            status=ResultStatus.UNRESOLVED,
            book_mention=book_mention,
            book=None,
        )
    ]
    resolved_screen_works = ScreenWorkResults(
        movies=movie_results,
        tv_series=tv_series_results,
    )
    resolved_music = MusicResults(
        tracks=track_results,
        music_releases=music_release_results,
    )
    resolved_books = BookResults(books=book_results)
    aggregator = ExtractionResultAggregator(
        FakeScreenWorkResolver(results=resolved_screen_works),
        FakeMusicResolver(results=resolved_music),
        FakeBookResolver(results=resolved_books),
    )

    results = await aggregator.aggregate(
        ExtractionMentions(
            screen_works=ScreenWorkMentions(
                movies=[movie_mention],
                tv_series=[tv_series_mention],
            ),
            music=MusicMentions(
                tracks=[track_mention],
                music_releases=[music_release_mention],
            ),
            books=BookMentions(books=[book_mention]),
        ),
        _MARKET,
    )

    assert results.screen_works is resolved_screen_works
    assert results.screen_works.movies is movie_results
    assert results.screen_works.tv_series is tv_series_results
    assert results.music is resolved_music
    assert results.music.tracks is track_results
    assert results.music.music_releases is music_release_results
    assert results.books is resolved_books
    assert results.books.books is book_results


async def test_aggregator_starts_all_scope_resolvers_concurrently() -> None:
    """Start independent Screen Work, Music, and Book Work resolution together."""
    started: set[str] = set()
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_release(scope: str) -> None:
        started.add(scope)
        if len(started) == 3:
            all_started.set()
        await release.wait()

    class _ScreenWorkResolver:
        async def resolve(self, mentions: ScreenWorkMentions) -> ScreenWorkResults:
            await wait_for_release("screen")
            return ScreenWorkResults(movies=[], tv_series=[])

        async def aclose(self) -> None:
            return None

    class _MusicResolver:
        async def resolve(
            self,
            mentions: MusicMentions,
            market: SpotifyMarket,
        ) -> MusicResults:
            await wait_for_release("music")
            return MusicResults(tracks=[], music_releases=[])

    class _BookResolver:
        async def resolve(self, mentions: BookMentions) -> BookResults:
            await wait_for_release("books")
            return BookResults(books=[])

        async def aclose(self) -> None:
            return None

    aggregator = ExtractionResultAggregator(
        _ScreenWorkResolver(),
        _MusicResolver(),
        _BookResolver(),
    )
    aggregate_task = asyncio.create_task(
        aggregator.aggregate(
            ExtractionMentions(
                screen_works=ScreenWorkMentions(movies=[], tv_series=[]),
                music=MusicMentions(tracks=[], music_releases=[]),
                books=BookMentions(books=[]),
            ),
            _MARKET,
        )
    )

    await asyncio.wait_for(all_started.wait(), timeout=1)
    assert started == {"screen", "music", "books"}
    release.set()
    await aggregate_task


@pytest.mark.parametrize(
    "resolver_error",
    [
        EnrichmentError("TMDB candidate resolution failed."),
        CatalogProviderError("Open Library catalog request failed."),
        PipelineTimeoutError("Open Library catalog request timed out."),
    ],
)
async def test_aggregator_propagates_resolver_errors_without_partial_results(
    resolver_error: Exception,
) -> None:
    """Propagate any resolver failure without constructing partial grouped results."""
    screen_work_resolver = FakeScreenWorkResolver()
    book_resolver = FakeBookResolver()
    if isinstance(resolver_error, EnrichmentError):
        screen_work_resolver = FakeScreenWorkResolver(error=resolver_error)
    else:
        book_resolver = FakeBookResolver(error=resolver_error)
    aggregator = ExtractionResultAggregator(
        screen_work_resolver,
        FakeMusicResolver(),
        book_resolver,
    )

    with pytest.raises(type(resolver_error)) as error:
        await aggregator.aggregate(
            ExtractionMentions(
                screen_works=ScreenWorkMentions(movies=[], tv_series=[]),
                music=MusicMentions(tracks=[], music_releases=[]),
                books=BookMentions(books=[]),
            ),
            _MARKET,
        )

    assert error.value is resolver_error


class _ClosingScreenWorkResolver:
    """Record aggregator-owned Screen Work resolver lifecycle."""

    def __init__(self, error: Exception | None = None) -> None:
        self.close_calls = 0
        self.error = error

    async def resolve(self, mentions: ScreenWorkMentions) -> ScreenWorkResults:
        raise AssertionError(f"unexpected resolution: {mentions}")

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.error is not None:
            raise self.error


class _ClosingBookResolver:
    """Record aggregator-owned Book Work resolver lifecycle."""

    def __init__(self) -> None:
        self.close_calls = 0

    async def resolve(self, mentions: BookMentions) -> BookResults:
        raise AssertionError(f"unexpected resolution: {mentions}")

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_aggregator_closes_screen_and_book_resolvers_only() -> None:
    """Leave lifespan-owned Spotify catalog closure outside the aggregator."""
    screen_resolver = _ClosingScreenWorkResolver()
    book_resolver = _ClosingBookResolver()
    aggregator = ExtractionResultAggregator(
        screen_resolver,
        FakeMusicResolver(),
        book_resolver,
    )

    await aggregator.aclose()

    assert screen_resolver.close_calls == 1
    assert book_resolver.close_calls == 1


async def test_aggregator_closes_books_when_screen_work_shutdown_fails() -> None:
    """Release Book Work resources in the Screen Work resolver's finally path."""
    screen_error = RuntimeError("screen resolver close failed")
    screen_resolver = _ClosingScreenWorkResolver(error=screen_error)
    book_resolver = _ClosingBookResolver()
    aggregator = ExtractionResultAggregator(
        screen_resolver,
        FakeMusicResolver(),
        book_resolver,
    )

    with pytest.raises(RuntimeError, match="screen resolver close failed"):
        await aggregator.aclose()

    assert screen_resolver.close_calls == 1
    assert book_resolver.close_calls == 1
