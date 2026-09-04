"""Orchestrate Source-to-Enriched-Entity extraction."""

from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    ExtractionMentions,
    ExtractionResults,
    PipelineResult,
    Source,
    Transcript,
)

_DEFAULT_MARKET = SpotifyMarket("US")


class ExtractionPipelineProtocol(Protocol):
    """Define the end-to-end extraction pipeline boundary."""

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Extract structured mentions and results from a source URL.

        Args:
            url: Source URL submitted by the API caller.
            market: Optional validated Spotify market from the API caller.

        Returns:
            PipelineResult: Canonical source, transcript, and grouped results.

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


class _MentionInterpreter(Protocol):
    """Interpret grouped mentions from one Source and Transcript."""

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> ExtractionMentions:
        """Return canonical mentions grouped by service scope."""
        ...

    async def aclose(self) -> None:
        """Release interpretation provider resources."""
        ...


class _ResultAggregator(Protocol):
    """Resolve and enrich grouped mentions across service scopes."""

    async def aggregate(
        self,
        mentions: ExtractionMentions,
        market: SpotifyMarket,
    ) -> ExtractionResults:
        """Return one Resolved or Unresolved Result per interpreted mention."""
        ...

    async def aclose(self) -> None:
        """Release aggregation resources."""
        ...


class ExtractionPipeline:
    """Orchestrate Source-to-Enriched-Entity extraction stages."""

    def __init__(
        self,
        source_metadata_service: _SourceMetadataInspector,
        transcription_service: _TranscriptAcquirer,
        interpretation_service: _MentionInterpreter,
        result_aggregator: _ResultAggregator,
        default_market: SpotifyMarket = _DEFAULT_MARKET,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_metadata_service: Service that validates and inspects Sources.
            transcription_service: Service that acquires Transcripts.
            interpretation_service: Service that interprets grouped mentions.
            result_aggregator: Module that resolves and enriches grouped mentions.
            default_market: Validated fallback Spotify market for omitted requests.
        """
        self._source_metadata_service = source_metadata_service
        self._transcription_service = transcription_service
        self._interpretation_service = interpretation_service
        self._result_aggregator = result_aggregator
        self._default_market = default_market

    async def run(
        self,
        url: str,
        market: SpotifyMarket | None = None,
    ) -> PipelineResult:
        """Produce Resolved or Unresolved Results for one submitted Source.

        Args:
            url: Source URL submitted by the API caller.
            market: Optional validated Spotify market from the API caller.

        Returns:
            PipelineResult: Source, Transcript, effective market, and Results.

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
        effective_market = self._default_market if market is None else market
        results = await self._result_aggregator.aggregate(interpreted, effective_market)
        return PipelineResult(
            source=inspected.source,
            transcript=transcript,
            results=results,
            market=effective_market,
        )

    async def aclose(self) -> None:
        """Release lifespan-owned interpretation and aggregation resources."""
        try:
            await self._interpretation_service.aclose()
        finally:
            await self._result_aggregator.aclose()
