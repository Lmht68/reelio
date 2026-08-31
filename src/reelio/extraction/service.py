"""Orchestrate Source-to-Enriched-Entity extraction."""

from typing import Protocol

from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    PipelineResult,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
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


class _ScreenWorkResolver(Protocol):
    """Resolve and enrich grouped Screen Work Mentions against provider candidates."""

    async def resolve(
        self,
        screen_work_mentions: ScreenWorkMentions,
    ) -> ScreenWorkResults:
        """Return one Resolved or Unresolved Result per Screen Work Mention."""
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
        screen_work_resolver: _ScreenWorkResolver,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_metadata_service: Service that validates and inspects Sources.
            transcription_service: Service that acquires Transcripts.
            interpretation_service: Service that interprets Screen Work Mentions.
            screen_work_resolver: Module that resolves and enriches Screen Work Mentions.
        """
        self._source_metadata_service = source_metadata_service
        self._transcription_service = transcription_service
        self._interpretation_service = interpretation_service
        self._screen_work_resolver = screen_work_resolver

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
        screen_work_results = await self._screen_work_resolver.resolve(interpreted)
        return PipelineResult(
            source=inspected.source,
            transcript=transcript,
            results=screen_work_results,
        )

    async def aclose(self) -> None:
        """Release lifespan-owned interpretation and resolution resources."""
        try:
            await self._interpretation_service.aclose()
        finally:
            await self._screen_work_resolver.aclose()
