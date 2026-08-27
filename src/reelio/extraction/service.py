"""Orchestrate Source inspection, transcription, and Movie Mention interpretation."""

from typing import Protocol

from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    MentionResult,
    MovieMention,
    PipelineResult,
    ResultStatus,
    Source,
    Transcript,
)


class ExtractionPipelineProtocol(Protocol):
    """Define the end-to-end extraction pipeline boundary."""

    async def run(self, url: str) -> PipelineResult:
        """Extract structured movie mentions from a submitted source URL.

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


class _MovieMentionInterpreter(Protocol):
    """Interpret ordered Movie Mentions from one Source and Transcript."""

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> list[MovieMention]:
        """Return canonical Movie Mentions in first-reference order."""
        ...

    async def aclose(self) -> None:
        """Release interpretation provider resources."""
        ...


class ExtractionPipeline:
    """Orchestrate the implemented movie extraction stages."""

    def __init__(
        self,
        source_metadata_service: _SourceMetadataInspector,
        transcription_service: _TranscriptAcquirer,
        interpretation_service: _MovieMentionInterpreter,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_metadata_service: Service that validates and inspects Sources.
            transcription_service: Service that acquires Transcripts.
            interpretation_service: Service that interprets Movie Mentions.
        """
        self._source_metadata_service = source_metadata_service
        self._transcription_service = transcription_service
        self._interpretation_service = interpretation_service

    async def run(self, url: str) -> PipelineResult:
        """Inspect a Source, acquire its Transcript, and interpret Movie Mentions.

        Args:
            url: Source URL submitted by the API caller.

        Returns:
            PipelineResult: Source, Transcript, and Unresolved Results.

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

        movie_mentions = await self._interpretation_service.interpret(
            inspected.source,
            transcript,
        )
        return PipelineResult(
            source=inspected.source,
            transcript=transcript,
            results=[
                MentionResult(
                    status=ResultStatus.UNRESOLVED,
                    movie_mention=movie_mention,
                    movie=None,
                )
                for movie_mention in movie_mentions
            ],
        )

    async def aclose(self) -> None:
        """Release lifespan-owned interpretation resources."""
        await self._interpretation_service.aclose()
