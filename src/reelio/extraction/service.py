"""Orchestrate Source-to-Enriched-Entity extraction."""

from collections.abc import Sequence
from typing import Protocol

from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    MovieMention,
    MovieResult,
    PipelineResult,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
    TVSeriesResult,
)


class ExtractionPipelineProtocol(Protocol):
    """Define the end-to-end extraction pipeline boundary."""

    async def run(self, url: str) -> PipelineResult:
        """Extract structured Screen Work Mentions from a submitted source URL.

        Args:
            url: Source URL submitted by the API caller.

        Returns:
            PipelineResult: Canonical source, transcript, and mention results.

        Raises:
            ExtractionError: If a pipeline stage fails with a domain error.
        """
        ...

    async def aclose(self) -> None:
        """Release lifespan-owned pipeline resources."""
        ...


class _SourceMetadataInspector(Protocol):
    """Inspect one submitted URL into a Source and request-scoped resources."""

    async def inspect(self, submitted_url: str) -> InspectedSource:
        """Return validated Source metadata and temporary inspection resources."""
        ...


class _TranscriptAcquirer(Protocol):
    """Acquire a Transcript for one validated Source."""

    async def acquire(
        self,
        source: Source,
        submitted_url: str,
        prepared_audio: PreparedAudio | None = None,
    ) -> Transcript:
        """Return a normalized Transcript using the validated submitted URL."""
        ...


class _ScreenWorkMentionInterpreter(Protocol):
    """Interpret ordered Screen Work Mentions from one Source and Transcript."""

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> ScreenWorkMentions:
        """Return canonical mentions in first-reference order within each kind."""
        ...

    async def aclose(self) -> None:
        """Release interpretation provider resources."""
        ...


class _MovieResolver(Protocol):
    """Resolve and enrich ordered Movie Mentions against provider candidates."""

    async def resolve(
        self,
        movie_mentions: Sequence[MovieMention],
    ) -> list[MovieResult]:
        """Return one Resolved or Unresolved Result per Movie Mention."""
        ...

    async def aclose(self) -> None:
        """Release resolution provider resources."""
        ...


class ExtractionPipeline:
    """Orchestrate Source-to-Enriched-Entity extraction stages."""

    def __init__(
        self,
        source_metadata_service: _SourceMetadataInspector,
        transcription_service: _TranscriptAcquirer,
        interpretation_service: _ScreenWorkMentionInterpreter,
        movie_resolver: _MovieResolver,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_metadata_service: Service that validates and inspects Sources.
            transcription_service: Service that acquires Transcripts.
            interpretation_service: Service that interprets Screen Work Mentions.
            movie_resolver: Module that resolves and enriches Movie Mentions.
        """
        self._source_metadata_service = source_metadata_service
        self._transcription_service = transcription_service
        self._interpretation_service = interpretation_service
        self._movie_resolver = movie_resolver

    async def run(self, url: str) -> PipelineResult:
        """Produce Resolved or Unresolved Results for one submitted Source.

        Args:
            url: Source URL submitted by the API caller.

        Returns:
            PipelineResult: Source, Transcript, and per-mention resolution Results.

        Raises:
            ExtractionError: If any pipeline stage fails with a domain error.
        """
        inspected = await self._source_metadata_service.inspect(url)
        try:
            transcript = await self._transcription_service.acquire(
                inspected.source,
                url,
                inspected.prepared_audio,
            )
        finally:
            inspected.cleanup()

        interpreted = await self._interpretation_service.interpret(
            inspected.source,
            transcript,
        )
        movie_results = await self._movie_resolver.resolve(interpreted.movies)
        tv_series_results = [
            TVSeriesResult(
                status=ResultStatus.UNRESOLVED,
                tv_series_mention=mention,
                tv_series=None,
            )
            for mention in interpreted.tv_series
        ]
        return PipelineResult(
            source=inspected.source,
            transcript=transcript,
            results=ScreenWorkResults(
                movies=movie_results,
                tv_series=tv_series_results,
            ),
        )

    async def aclose(self) -> None:
        """Release lifespan-owned interpretation and resolution resources."""
        try:
            await self._interpretation_service.aclose()
        finally:
            await self._movie_resolver.aclose()
