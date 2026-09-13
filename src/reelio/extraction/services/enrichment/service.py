"""Aggregate resolved Extraction Results across service scopes."""

import asyncio
from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
    BookMentions,
    BookResults,
    ExtractionMentions,
    ExtractionResults,
    MusicMentions,
    MusicResults,
    ScreenWorkMentions,
    ScreenWorkResults,
)


class _ScreenWorkResolver(Protocol):
    """Resolve Screen Work Mentions with provider-backed enrichment."""

    async def resolve(
        self,
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        """Return resolved Screen Work Results for grouped mentions."""
        ...

    async def aclose(self) -> None:
        """Release provider-owned resources."""
        ...


class _BookResolver(Protocol):
    """Resolve Book Work Mentions with provider-backed enrichment."""

    async def resolve(self, book_mentions: BookMentions) -> BookResults:
        """Return resolved Book Work Results for ordered Mentions."""
        ...

    async def aclose(self) -> None:
        """Release provider-owned resources."""
        ...


class _MusicResolver(Protocol):
    """Resolve grouped Music Mentions with Spotify-backed enrichment."""

    async def resolve(
        self,
        music_mentions: MusicMentions,
        market: SpotifyMarket,
    ) -> MusicResults:
        """Return grouped Music Results for one effective market."""
        ...


class ExtractionResultAggregator:
    """Coordinate result-kind resolution for one extraction pipeline.

    Args:
        screen_work_resolver: Resolver for Movie and TV Series mentions.
        music_resolver: Resolver for Spotify Track and Music Release mentions.
        book_resolver: Resolver for Open Library Book Work mentions.
    """

    def __init__(
        self,
        screen_work_resolver: _ScreenWorkResolver,
        music_resolver: _MusicResolver,
        book_resolver: _BookResolver,
    ) -> None:
        """Initialize aggregation with resolvers for each service scope.

        Args:
            screen_work_resolver: Resolver for Movie and TV Series mentions.
            music_resolver: Resolver for Spotify Track and Music Release mentions.
            book_resolver: Resolver for Open Library Book Work mentions.
        """
        self._screen_work_resolver = screen_work_resolver
        self._music_resolver = music_resolver
        self._book_resolver = book_resolver

    async def aggregate(
        self,
        mentions: ExtractionMentions,
        market: SpotifyMarket,
    ) -> ExtractionResults:
        """Resolve every mention kind into grouped extraction results atomically.

        Args:
            mentions: Interpreted mentions grouped by service scope.
            market: Effective Spotify market used for Spotify resolution.

        Returns:
            ExtractionResults: Results grouped by service scope.
        """
        resolved_screen_works, resolved_music, resolved_books = await asyncio.gather(
            self._screen_work_resolver.resolve(mentions.screen_works),
            self._music_resolver.resolve(mentions.music, market),
            self._book_resolver.resolve(mentions.books),
        )
        return ExtractionResults(
            screen_works=resolved_screen_works,
            music=resolved_music,
            books=resolved_books,
        )

    async def aclose(self) -> None:
        """Release resources owned by Screen Work and Book Work resolvers."""
        try:
            await self._screen_work_resolver.aclose()
        finally:
            await self._book_resolver.aclose()
