"""Aggregate resolved Extraction Results across service scopes."""

from typing import Protocol

from reelio.extraction.types import (
    ExtractionMentions,
    ExtractionResults,
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


class ExtractionResultAggregator:
    """Coordinate result-kind resolution for one extraction pipeline.

    Args:
        screen_work_resolver: Resolver for Movie and TV Series mentions.
    """

    def __init__(self, screen_work_resolver: _ScreenWorkResolver) -> None:
        """Initialize aggregation with the Screen Work resolver.

        Args:
            screen_work_resolver: Resolver for Movie and TV Series mentions.
        """
        self._screen_work_resolver = screen_work_resolver

    async def aggregate(self, mentions: ExtractionMentions) -> ExtractionResults:
        """Resolve every mention kind into grouped extraction results.

        Args:
            mentions: Interpreted mentions grouped by service scope.

        Returns:
            ExtractionResults: Results grouped by service scope.
        """
        resolved_screen_works = await self._screen_work_resolver.resolve(mentions.screen_works)
        return ExtractionResults(screen_works=resolved_screen_works)

    async def aclose(self) -> None:
        """Release resources owned by kind-specific resolvers."""
        await self._screen_work_resolver.aclose()
