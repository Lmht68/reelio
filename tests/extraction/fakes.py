"""Shared configurable test doubles for extraction pipeline modules."""

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
    BookMentions,
    BookResult,
    BookResults,
    ExtractionMentions,
    ExtractionResults,
    MovieResult,
    MusicMentions,
    MusicReleaseResult,
    MusicResults,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    TrackResult,
    Transcript,
    TVSeriesResult,
)


class FakeInterpretationService:
    """Provide deterministic generalized interpretation for pipeline tests."""

    def __init__(
        self,
        mentions: ExtractionMentions | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned mentions or a raised exception.

        Args:
            mentions: Extraction Mentions returned by ``interpret``.
            error: Exception raised by ``interpret`` when provided.
        """
        self.mentions = (
            mentions
            if mentions is not None
            else ExtractionMentions(
                screen_works=ScreenWorkMentions(movies=[], tv_series=[]),
                music=MusicMentions(tracks=[], music_releases=[]),
                books=BookMentions(books=[]),
            )
        )
        self.error = error
        self.calls: list[tuple[Source, Transcript]] = []
        self.closed = False

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> ExtractionMentions:
        """Record Interpretation Material and return configured mentions.

        Args:
            source: Source supplied by the extraction pipeline.
            transcript: Transcript supplied by the extraction pipeline.

        Returns:
            ExtractionMentions: Configured mentions grouped by service scope.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append((source, transcript))
        if self.error is not None:
            raise self.error
        return self.mentions

    async def aclose(self) -> None:
        """Record release of interpretation resources."""
        self.closed = True


def _unresolved_screen_work_results(
    screen_work_mentions: ScreenWorkMentions,
) -> ScreenWorkResults:
    return ScreenWorkResults(
        movies=[
            MovieResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=movie_mention,
                movie=None,
            )
            for movie_mention in screen_work_mentions.movies
        ],
        tv_series=[
            TVSeriesResult(
                status=ResultStatus.UNRESOLVED,
                tv_series_mention=tv_series_mention,
                tv_series=None,
            )
            for tv_series_mention in screen_work_mentions.tv_series
        ],
    )


def _unresolved_music_results(music_mentions: MusicMentions) -> MusicResults:
    return MusicResults(
        tracks=[
            TrackResult(
                status=ResultStatus.UNRESOLVED,
                track_mention=track_mention,
                track=None,
            )
            for track_mention in music_mentions.tracks
        ],
        music_releases=[
            MusicReleaseResult(
                status=ResultStatus.UNRESOLVED,
                music_release_mention=music_release_mention,
                music_release=None,
            )
            for music_release_mention in music_mentions.music_releases
        ],
    )


def _unresolved_book_results(book_mentions: BookMentions) -> BookResults:
    return BookResults(
        books=[
            BookResult(
                status=ResultStatus.UNRESOLVED,
                book_mention=book_mention,
                book=None,
            )
            for book_mention in book_mentions.books
        ]
    )


class FakeScreenWorkResolver:
    """Provide deterministic grouped Screen Work resolution for pipeline tests."""

    def __init__(
        self,
        results: ScreenWorkResults | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned results or a raised exception.

        Args:
            results: Results returned by ``resolve``. When omitted, every supplied
                Screen Work Mention remains unresolved.
            error: Exception raised by ``resolve`` when provided.
        """
        self.results = results
        self.error = error
        self.calls: list[ScreenWorkMentions] = []
        self.closed = False

    async def resolve(
        self,
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        """Record and resolve the supplied Screen Work Mentions.

        Args:
            screen_work_mentions: Grouped ordered Mentions from interpretation.

        Returns:
            ScreenWorkResults: Configured or default unresolved grouped Results.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append(screen_work_mentions)
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return _unresolved_screen_work_results(screen_work_mentions)

    async def aclose(self) -> None:
        """Record release of resolution resources."""
        self.closed = True


class FakeBookResolver:
    """Provide deterministic Book Work resolution for aggregation tests."""

    def __init__(
        self,
        results: BookResults | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned Book Work Results or a raised exception.

        Args:
            results: Results returned by ``resolve`` when provided.
            error: Exception raised by ``resolve`` when provided.
        """
        self.results = results
        self.error = error
        self.calls: list[BookMentions] = []
        self.closed = False

    async def resolve(self, book_mentions: BookMentions) -> BookResults:
        """Record and resolve Book Work Mentions."""
        self.calls.append(book_mentions)
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return _unresolved_book_results(book_mentions)

    async def aclose(self) -> None:
        """Record release of resolution resources."""
        self.closed = True


class FakeMusicResolver:
    """Provide deterministic grouped Music resolution for aggregation tests."""

    def __init__(
        self,
        results: MusicResults | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned Music Results or a raised exception.

        Args:
            results: Results returned by ``resolve`` when provided.
            error: Exception raised by ``resolve`` when provided.
        """
        self.results = results
        self.error = error
        self.calls: list[tuple[MusicMentions, SpotifyMarket]] = []

    async def resolve(
        self,
        music_mentions: MusicMentions,
        market: SpotifyMarket,
    ) -> MusicResults:
        """Record and resolve grouped Music Mentions for one effective market."""
        self.calls.append((music_mentions, market))
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return _unresolved_music_results(music_mentions)


class FakeResultAggregator:
    """Provide deterministic generalized aggregation for pipeline tests."""

    def __init__(
        self,
        results: ExtractionResults | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned results or a raised exception.

        Args:
            results: Results returned by ``aggregate``. When omitted, every supplied
                Screen Work Mention remains unresolved.
            error: Exception raised by ``aggregate`` when provided.
        """
        self.results = results
        self.error = error
        self.calls: list[ExtractionMentions] = []
        self.markets: list[SpotifyMarket] = []
        self.closed = False

    async def aggregate(
        self,
        mentions: ExtractionMentions,
        market: SpotifyMarket,
    ) -> ExtractionResults:
        """Record and aggregate supplied mentions for one effective market.

        Args:
            mentions: Mentions grouped by service scope.
            market: Effective Spotify market used for Track resolution.

        Returns:
            ExtractionResults: Configured or default grouped unresolved Results.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append(mentions)
        self.markets.append(market)
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return ExtractionResults(
            screen_works=_unresolved_screen_work_results(mentions.screen_works),
            music=_unresolved_music_results(mentions.music),
            books=_unresolved_book_results(mentions.books),
        )

    async def aclose(self) -> None:
        """Record release of aggregation resources."""
        self.closed = True
