"""Shared configurable test doubles for extraction pipeline modules."""

from reelio.extraction.types import (
    MovieResult,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
    TVSeriesResult,
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

    async def aclose(self) -> None:
        """Record release of resolution resources."""
        self.closed = True
