"""Shared configurable test doubles for extraction pipeline modules."""

from collections.abc import Sequence

from reelio.extraction.types import (
    MovieMention,
    MovieResult,
    ResultStatus,
    ScreenWorkMentions,
    Source,
    Transcript,
)


class FakeInterpretationService:
    """Provide deterministic Screen Work Mention interpretation for pipeline tests."""

    def __init__(
        self,
        mentions: ScreenWorkMentions | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned Screen Work Mentions or a raised exception.

        Args:
            mentions: Screen Work Mentions returned by ``interpret``.
            error: Exception raised by ``interpret`` when provided.
        """
        self.mentions = (
            mentions if mentions is not None else ScreenWorkMentions(movies=[], tv_series=[])
        )
        self.error = error
        self.calls: list[tuple[Source, Transcript]] = []
        self.closed = False

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> ScreenWorkMentions:
        """Record Interpretation Material and return configured Screen Work Mentions.

        Args:
            source: Source supplied by the extraction pipeline.
            transcript: Transcript supplied by the extraction pipeline.

        Returns:
            ScreenWorkMentions: Configured grouped Screen Work Mentions.

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


class FakeMovieResolver:
    """Provide deterministic candidate resolution for pipeline tests."""

    def __init__(
        self,
        results: Sequence[MovieResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure returned results or a raised exception.

        Args:
            results: Results returned by ``resolve``. When omitted, every supplied
                Movie Mention remains unresolved.
            error: Exception raised by ``resolve`` when provided.
        """
        self.results = None if results is None else list(results)
        self.error = error
        self.calls: list[tuple[MovieMention, ...]] = []
        self.closed = False

    async def resolve(
        self,
        movie_mentions: Sequence[MovieMention],
    ) -> list[MovieResult]:
        """Record and resolve the supplied Movie Mentions.

        Args:
            movie_mentions: Ordered Movie Mentions from interpretation.

        Returns:
            list[MovieResult]: Configured or default unresolved results.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append(tuple(movie_mentions))
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return [
            MovieResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=movie_mention,
                movie=None,
            )
            for movie_mention in movie_mentions
        ]

    async def aclose(self) -> None:
        """Record release of resolution resources."""
        self.closed = True
