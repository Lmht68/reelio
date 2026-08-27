"""Shared configurable test doubles for extraction pipeline modules."""

from collections.abc import Sequence

from reelio.extraction.types import (
    MentionResult,
    MovieMention,
    ResultStatus,
    Source,
    Transcript,
)


class FakeInterpretationService:
    """Provide deterministic Movie Mention interpretation for pipeline tests."""

    def __init__(
        self,
        movie_mentions: Sequence[MovieMention] = (),
        error: Exception | None = None,
    ) -> None:
        """Configure returned Movie Mentions or a raised exception.

        Args:
            movie_mentions: Movie Mentions returned by ``interpret``.
            error: Exception raised by ``interpret`` when provided.
        """
        self.movie_mentions = list(movie_mentions)
        self.error = error
        self.calls: list[tuple[Source, Transcript]] = []
        self.closed = False

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> list[MovieMention]:
        """Record Interpretation Material and return configured Movie Mentions.

        Args:
            source: Source supplied by the extraction pipeline.
            transcript: Transcript supplied by the extraction pipeline.

        Returns:
            list[MovieMention]: Configured Movie Mentions.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append((source, transcript))
        if self.error is not None:
            raise self.error
        return self.movie_mentions

    async def aclose(self) -> None:
        """Record release of interpretation resources."""
        self.closed = True


class FakeMovieResolver:
    """Provide deterministic candidate resolution for pipeline tests."""

    def __init__(
        self,
        results: Sequence[MentionResult] | None = None,
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
    ) -> list[MentionResult]:
        """Record and resolve the supplied Movie Mentions.

        Args:
            movie_mentions: Ordered Movie Mentions from interpretation.

        Returns:
            list[MentionResult]: Configured or default unresolved results.

        Raises:
            Exception: Configured error when one was provided.
        """
        self.calls.append(tuple(movie_mentions))
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return [
            MentionResult(
                status=ResultStatus.UNRESOLVED,
                movie_mention=movie_mention,
                movie=None,
            )
            for movie_mention in movie_mentions
        ]

    async def aclose(self) -> None:
        """Record release of resolution resources."""
        self.closed = True
