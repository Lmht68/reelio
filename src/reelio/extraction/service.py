"""Orchestrate source inspection, transcription, and placeholder result stages."""

from dataclasses import replace
from typing import Final, Protocol

from reelio.extraction.types import (
    Candidate,
    EnrichedMovie,
    MentionResult,
    PipelineResult,
    Platform,
    ResultStatus,
    Source,
    Transcript,
    TranscriptMethod,
)


class Pipeline(Protocol):
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
    """Inspect one submitted URL into a canonical Source."""

    async def inspect(self, submitted_url: str) -> Source:
        """Return validated Source metadata."""
        ...


class _TranscriptAcquirer(Protocol):
    """Acquire a Transcript for one validated Source."""

    async def acquire(self, source: Source, submitted_url: str) -> Transcript:
        """Return a normalized Transcript using the validated submitted URL."""


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
        source = await self._source_metadata_service.inspect(url)
        transcript = await self._transcription_service.acquire(source, url)
        return replace(_PLACEHOLDER_RESULT, source=source, transcript=transcript)


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
            "Dune Part Two blew me away, and the original Dune still holds up, "
            "but that 90s space movie everyone quotes was overrated."
        ),
        language="en",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    ),
    results=[
        MentionResult(
            status=ResultStatus.RESOLVED,
            mentioned_as=["Dune Part Two"],
            evidence=["Dune Part Two blew me away"],
            extraction_confidence=0.95,
            resolution_confidence=0.85,
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
            candidates=[],
        ),
        MentionResult(
            status=ResultStatus.AMBIGUOUS,
            mentioned_as=["Dune"],
            evidence=["the original Dune still holds up"],
            extraction_confidence=0.9,
            resolution_confidence=0.55,
            movie=None,
            candidates=[
                Candidate(
                    title="Dune",
                    year=2021,
                    directors=["Denis Villeneuve"],
                    description=(
                        "A gifted young man travels to the most dangerous planet "
                        "in the universe to secure his family's future."
                    ),
                    poster_url=("https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"),
                    tmdb_id=438631,
                    tmdb_url="https://www.themoviedb.org/movie/438631",
                    imdb_id="tt1160419",
                    imdb_url="https://www.imdb.com/title/tt1160419",
                    tmdb_score=8.0,
                    resolution_score=0.55,
                ),
                Candidate(
                    title="Dune",
                    year=1984,
                    directors=["David Lynch"],
                    description=(
                        "A noble family becomes embroiled in a war for control of "
                        "a valuable desert planet."
                    ),
                    poster_url=("https://image.tmdb.org/t/p/w500/rK0ah8i2A4B5Yjz8b5dL5f4XQ3a.jpg"),
                    tmdb_id=841,
                    tmdb_url="https://www.themoviedb.org/movie/841",
                    imdb_id="tt0087182",
                    imdb_url="https://www.imdb.com/title/tt0087182",
                    tmdb_score=6.3,
                    resolution_score=0.45,
                ),
                Candidate(
                    title="Jodorowsky's Dune",
                    year=2013,
                    directors=["Frank Pavich"],
                    description=(
                        "Filmmaker Alejandro Jodorowsky recounts his ambitious "
                        "attempt to adapt Dune for the screen."
                    ),
                    poster_url=None,
                    tmdb_id=241256,
                    tmdb_url="https://www.themoviedb.org/movie/241256",
                    imdb_id=None,
                    imdb_url=None,
                    tmdb_score=7.5,
                    resolution_score=0.40,
                ),
            ],
        ),
        MentionResult(
            status=ResultStatus.UNRESOLVED,
            mentioned_as=["that 90s space movie"],
            evidence=["that 90s space movie everyone quotes was overrated"],
            extraction_confidence=0.4,
            resolution_confidence=None,
            movie=None,
            candidates=[],
        ),
    ],
)
