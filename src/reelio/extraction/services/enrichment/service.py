"""Aggregate resolved Extraction Results across service scopes."""

import asyncio
from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
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
    """

    def __init__(
        self,
        screen_work_resolver: _ScreenWorkResolver,
        music_resolver: _MusicResolver,
    ) -> None:
        """Initialize aggregation with resolvers for each service scope.
 badsd
        Args:
            screen_work_resolver: Resolver for Movie and TV Series mentions.
            music_resolver: Resolver for Spotify Track and Music Release mentions.
        """
        self._screen_work_resolver = screen_work_resolver
        self._music_resolver = music_resolver

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
        resolved_screen_works, resolved_music = await asyncio.gather(
            self._screen_work_resolver.resolve(mentions.screen_works),
            self._music_resolver.resolve(mentions.music, market),
        )
        return ExtractionResults(
            screen_works=resolved_screen_works,
            music=resolved_music,
        )

    async def aclose(self) -> None:
        """Release resources owned by kind-specific resolvers."""
        await self._screen_work_resolver.aclose()
