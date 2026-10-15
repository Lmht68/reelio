"""Orchestrate Source-to-Enriched-Entity extraction."""

from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.types import (
    ExtractionMentions,
    ExtractionResults,
    InterpretationMaterial,
    PipelineResult,
    Source,
    Transcript,
    TranscriptMethod,
    TranscriptPipelineResult,
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

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        """Extract structured mentions and results from submitted transcript text.

        Args:
            transcript_text: Normalized transcript text submitted by the API caller.
            market: Optional validated Spotify market from the API caller.

        Returns:
            TranscriptPipelineResult: Transcript, effective market, and grouped results.

        Raises:
            ExtractionError: If a pipeline stage fails with a domain error.
        """
        ...

    async def aclose(self) -> None:
        """Release lifespan-owned pipeline resources."""
        ...


class SourceTranscriptProvider(Protocol):
    """Acquire a Source and Transcript from one submitted URL."""

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        """Return the normalized Source and Transcript for a submitted URL."""
        ...


class _MentionInterpreter(Protocol):
    """Interpret grouped mentions from complete Interpretation Material."""

    async def interpret(
        self,
        material: InterpretationMaterial,
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
        source_transcript_provider: SourceTranscriptProvider,
        interpretation_service: _MentionInterpreter,
        result_aggregator: _ResultAggregator,
        default_market: SpotifyMarket = _DEFAULT_MARKET,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_transcript_provider: Provider that acquires Sources and Transcripts.
            interpretation_service: Service that interprets grouped mentions.
            result_aggregator: Module that resolves and enriches grouped mentions.
            default_market: Validated fallback Spotify market for omitted requests.
        """
        self._source_transcript_provider = source_transcript_provider
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
        source, transcript = await self._source_transcript_provider.acquire(url)
        material = InterpretationMaterial(
            source_title=source.title,
            source_description=source.description,
            transcript=transcript,
        )
        results, effective_market = await self._interpret_and_aggregate(material, market)
        return PipelineResult(
            source=source,
            transcript=transcript,
            results=results,
            market=effective_market,
        )

    async def run_transcript(
        self,
        transcript_text: str,
        market: SpotifyMarket | None = None,
    ) -> TranscriptPipelineResult:
        """Produce resolved or unresolved results for submitted transcript text.

        Args:
            transcript_text: Normalized transcript text submitted by the API caller.
            market: Optional validated Spotify market from the API caller.

        Returns:
            TranscriptPipelineResult: Transcript, effective market, and Results.

        Raises:
            ExtractionError: If any pipeline stage fails with a domain error.
        """
        transcript = Transcript(
            text=transcript_text,
            language="und",
            method=TranscriptMethod.TEXT_SUBMISSION,
        )
        material = InterpretationMaterial(
            source_title="",
            source_description="",
            transcript=transcript,
        )
        results, effective_market = await self._interpret_and_aggregate(material, market)
        return TranscriptPipelineResult(
            transcript=transcript,
            results=results,
            market=effective_market,
        )

    async def _interpret_and_aggregate(
        self,
        material: InterpretationMaterial,
        market: SpotifyMarket | None,
    ) -> tuple[ExtractionResults, SpotifyMarket]:
        interpreted = await self._interpretation_service.interpret(material)
        effective_market = self._default_market if market is None else market
        results = await self._result_aggregator.aggregate(interpreted, effective_market)
        return results, effective_market

    async def aclose(self) -> None:
        """Release lifespan-owned interpretation and aggregation resources."""
        try:
            await self._interpretation_service.aclose()
        finally:
            await self._result_aggregator.aclose()
