"""Aggregate resolved Extraction Results across service scopes."""

import asyncio
from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
    ExtractionMentions,
    ExtractionResults,
    MusicResults,
    ScreenWorkMentions,
    ScreenWorkResults,
    TrackMention,
    TrackResult,
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


class _TrackResolver(Protocol):
    """Resolve Track Mentions with Spotify-backed enrichment."""

    async def resolve(
        self,
        track_mentions: list[TrackMention],
        market: SpotifyMarket,
    ) -> list[TrackResult]:
        """Return ordered Track Results for one effective market."""
        ...


class ExtractionResultAggregator:
    """Coordinate result-kind resolution for one extraction pipeline.

    Args:
        screen_work_resolver: Resolver for Movie and TV Series mentions.
        track_resolver: Resolver for Spotify Track Mentions.
    """

    def __init__(
        self,
        screen_work_resolver: _ScreenWorkResolver,
        track_resolver: _TrackResolver,
    ) -> None:
        """Initialize aggregation with resolvers for each public result kind.

        Args:
            screen_work_resolver: Resolver for Movie and TV Series mentions.
            track_resolver: Resolver for Spotify Track Mentions.
        """
        self._screen_work_resolver = screen_work_resolver
        self._track_resolver = track_resolver

    async def aggregate(
        self,
        mentions: ExtractionMentions,
        market: SpotifyMarket,
    ) -> ExtractionResults:
        """Resolve every mention kind into grouped extraction results atomically.

        Args:
            mentions: Interpreted mentions grouped by service scope.
            market: Effective Spotify market used for Track resolution.

        Returns:
            ExtractionResults: Results grouped by service scope.
        """
        resolved_screen_works, resolved_tracks = await asyncio.gather(
            self._screen_work_resolver.resolve(mentions.screen_works),
            self._track_resolver.resolve(mentions.music.tracks, market),
        )
        return ExtractionResults(
            screen_works=resolved_screen_works,
            music=MusicResults(tracks=resolved_tracks),
        )

    async def aclose(self) -> None:
        """Release resources owned by kind-specific resolvers."""
        await self._screen_work_resolver.aclose()
