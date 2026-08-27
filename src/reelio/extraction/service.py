"""Orchestrate source inspection, transcription, and placeholder result stages."""

from dataclasses import replace
from typing import Final, Protocol

from reelio.extraction.services.transcription.inspection import PreparedAudio
from reelio.extraction.services.transcription.service import InspectedSource
from reelio.extraction.types import (
    EnrichedMovie,
    MentionResult,
    MovieMention,
    PipelineResult,
    Platform,
    ResultStatus,
    Source,
    Transcript,
    TranscriptMethod,
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


class ExtractionPipeline:
    """Orchestrate implemented stages while later results remain placeholders."""

    def __init__(
        self,
        source_metadata_service: _SourceMetadataInspector,
        transcription_service: _TranscriptAcquirer,
    ) -> None:
        """Initialize the pipeline with explicit stage services.

        Args:
            source_metadata_service: Service that validates and inspects Sources.
            transcription_service: Service that acquires Transcripts.
        """
        self._source_metadata_service = source_metadata_service
        self._transcription_service = transcription_service

    async def run(self, url: str) -> PipelineResult:
        """Inspect the Source, acquire its Transcript, and retain placeholders.

        Args:
            url: Source URL submitted by the API caller.

        Returns:
            PipelineResult: Real Source and Transcript with placeholder results.

        Raises:
            ExtractionError: If Source inspection or Transcript acquisition fails.
        """
        inspected = await self._source_metadata_service.inspect(url)
        try:
            transcript = await self._transcription_service.acquire(
                inspected.source,
                url,
                inspected.prepared_audio,
            )
            return replace(
                _PLACEHOLDER_RESULT,
                source=inspected.source,
                transcript=transcript,
            )
        finally:
            inspected.cleanup()


_PLACEHOLDER_RESULT: Final[PipelineResult] = PipelineResult(
    source=Source(
        platform=Platform.YOUTUBE,
        video_id="f4k3v1de0id",
        url="https://www.youtube.com/watch?v=f4k3v1de0id",
        title="Fake Dune discussion",
        description="A deterministic transcript contract fixture.",
        channel="Reelio test channel",
        duration_seconds=120,
    ),
    transcript=Transcript(
        text=(
            "Dune Part Two blew me away, it reminds me of Che by Steven Soderbergh"
        ),
        language="en",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    ),
    results=[
        MentionResult(
            status=ResultStatus.RESOLVED,
            movie_mention=MovieMention(title="Dune: Part Two", year=2024),
            movie=EnrichedMovie(
                title="Dune: Part Two",
                year=2024,
                directors=["Denis Villeneuve"],
                description=(
                    "Paul Atreides unites with Chani and the Fremen while seeking "
                    "revenge against the conspirators who destroyed his family."
                ),
                poster_url=("https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"),
                tmdb_id=693134,
                tmdb_url="https://www.themoviedb.org/movie/693134",
                imdb_id="tt15239678",
                imdb_url="https://www.imdb.com/title/tt15239678",
                tmdb_score=8.1,
            ),
        ),
        MentionResult(
            status=ResultStatus.UNRESOLVED,
            movie_mention=MovieMention(title="Che", year=2008),
            movie=None,
        ),
    ],
)
