"""Pipeline orchestration contract tests."""

import pytest

from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    EnrichmentError,
    InvalidSourceError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
)
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.service import ExtractionPipeline
from reelio.extraction.types import (
    BookMentions,
    BookResults,
    ExtractionMentions,
    ExtractionResults,
    InterpretationMaterial,
    MovieMention,
    MovieResult,
    MusicMentions,
    MusicResults,
    Platform,
    ResultStatus,
    ScreenWorkMentions,
    ScreenWorkResults,
    Source,
    Transcript,
    TranscriptMethod,
    TranscriptPipelineResult,
    TVSeriesMention,
    TVSeriesResult,
)
from tests.extraction.fakes import (
    FakeInterpretationService as _FakeInterpretationService,
)
from tests.extraction.fakes import FakeResultAggregator as _FakeResultAggregator

_VIDEO_ID = "dQw4w9WgXcQ"
_CANONICAL_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


class _FakeSourceTranscriptProvider:
    def __init__(
        self,
        source: Source,
        transcript: Transcript,
        error: Exception | None = None,
        call_order: list[str] | None = None,
    ) -> None:
        self.source = source
        self.transcript = transcript
        self.error = error
        self.call_order = call_order
        self.calls: list[str] = []
        self.closed = False

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        self.calls.append(submitted_url)
        if self.call_order is not None:
            self.call_order.append("source_transcript_acquisition")
        if self.error is not None:
            raise self.error
        return self.source, self.transcript

    async def aclose(self) -> None:
        self.closed = True


class _OrderedInterpretationService(_FakeInterpretationService):
    def __init__(self, call_order: list[str]) -> None:
        super().__init__()
        self._call_order = call_order

    async def interpret(self, material: InterpretationMaterial) -> ExtractionMentions:
        self._call_order.append("interpretation")
        return await super().interpret(material)


_DEFAULT_MARKET = SpotifyMarket("US")


def _pipeline(
    source_transcript_provider: _FakeSourceTranscriptProvider,
    interpretation_service: _FakeInterpretationService | None = None,
    result_aggregator: _FakeResultAggregator | None = None,
    default_market: SpotifyMarket = _DEFAULT_MARKET,
) -> ExtractionPipeline:
    return ExtractionPipeline(
        source_transcript_provider,
        interpretation_service or _FakeInterpretationService(),
        result_aggregator or _FakeResultAggregator(),
        default_market,
    )


def _source() -> Source:
    return Source(
        platform=Platform.YOUTUBE,
        video_id=_VIDEO_ID,
        url=_CANONICAL_URL,
        title="Pipeline test video",
        description="Pipeline test description.",
        channel="Pipeline test channel",
        duration_seconds=42,
    )


def _transcript() -> Transcript:
    return Transcript(
        text="A real pipeline transcript.",
        language="en-GB",
        method=TranscriptMethod.YOUTUBE_CAPTIONS,
    )


async def test_pipeline_acquires_source_and_transcript_before_interpretation() -> None:
    """Acquire complete material once before invoking interpretation."""
    call_order: list[str] = []
    source = _source()
    transcript = _transcript()
    source_transcript_provider = _FakeSourceTranscriptProvider(
        source,
        transcript,
        call_order=call_order,
    )
    interpretation_service = _OrderedInterpretationService(call_order)
    pipeline = _pipeline(source_transcript_provider, interpretation_service)

    result = await pipeline.run(_CANONICAL_URL)

    assert result.source is source
    assert result.transcript is transcript
    assert source_transcript_provider.calls == [_CANONICAL_URL]
    assert interpretation_service.calls == [
        InterpretationMaterial(
            source_title=source.title,
            source_description=source.description,
            transcript=transcript,
        )
    ]
    assert call_order == ["source_transcript_acquisition", "interpretation"]


async def test_pipeline_runs_submitted_transcript_without_source_stages() -> None:
    """Interpret and aggregate submitted Transcript text without source-side work."""
    submitted_text = "Dune: Part One (2021) was excellent."
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    mentions = ExtractionMentions(
        screen_works=ScreenWorkMentions(movies=[movie_mention], tv_series=[]),
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    results = ExtractionResults(
        screen_works=ScreenWorkResults(
            movies=[MovieResult(ResultStatus.UNRESOLVED, movie_mention, None)],
            tv_series=[],
        ),
        music=MusicResults(tracks=[], music_releases=[]),
        books=BookResults(books=[]),
    )
    source_transcript_provider = _FakeSourceTranscriptProvider(
        _source(),
        _transcript(),
    )
    interpretation_service = _FakeInterpretationService(mentions)
    result_aggregator = _FakeResultAggregator(results=results)
    pipeline = _pipeline(
        source_transcript_provider,
        interpretation_service,
        result_aggregator,
    )

    result = await pipeline.run_transcript(submitted_text)

    expected_transcript = Transcript(
        text=submitted_text,
        language="und",
        method=TranscriptMethod.TEXT_SUBMISSION,
    )
    assert isinstance(result, TranscriptPipelineResult)
    assert result.transcript == expected_transcript
    assert result.market == _DEFAULT_MARKET
    assert result.results is results
    assert source_transcript_provider.calls == []
    assert interpretation_service.calls == [
        InterpretationMaterial(
            source_title="",
            source_description="",
            transcript=expected_transcript,
        )
    ]
    assert result_aggregator.calls == [mentions]
    assert result_aggregator.markets == [_DEFAULT_MARKET]


async def test_pipeline_returns_empty_grouped_results_and_aggregates_once() -> None:
    """Pass empty generalized mentions to aggregation once and retain empty results."""
    source = _source()
    transcript = _transcript()
    source_transcript_provider = _FakeSourceTranscriptProvider(source, transcript)
    screen_work_mentions = ScreenWorkMentions(movies=[], tv_series=[])
    extraction_mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    interpretation_service = _FakeInterpretationService(extraction_mentions)
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        source_transcript_provider,
        interpretation_service,
        result_aggregator,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result.source is source
    assert result.transcript is transcript
    assert source_transcript_provider.calls == [_CANONICAL_URL]
    assert interpretation_service.calls == [
        InterpretationMaterial(
            source_title=source.title,
            source_description=source.description,
            transcript=transcript,
        )
    ]
    assert result_aggregator.calls == [extraction_mentions]
    assert result_aggregator.calls[0] is extraction_mentions
    assert result.results.screen_works.movies == []
    assert result.results.screen_works.tv_series == []


async def test_pipeline_returns_ordered_movie_results_unchanged() -> None:
    """Pass grouped Movie Mentions through and preserve returned result objects."""
    movie_mentions = [
        MovieMention(title="Dune: Part One", year=2021),
        MovieMention(title="Che: Part One", year=2008),
    ]
    movie_results = [
        MovieResult(ResultStatus.UNRESOLVED, movie_mentions[0], None),
        MovieResult(ResultStatus.UNRESOLVED, movie_mentions[1], None),
    ]
    screen_work_mentions = ScreenWorkMentions(movies=movie_mentions, tv_series=[])
    screen_work_results = ScreenWorkResults(movies=movie_results, tv_series=[])
    extraction_mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    extraction_results = ExtractionResults(
        screen_works=screen_work_results,
        music=MusicResults(tracks=[], music_releases=[]),
        books=BookResults(books=[]),
    )
    interpretation_service = _FakeInterpretationService(extraction_mentions)
    result_aggregator = _FakeResultAggregator(results=extraction_results)
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(_source(), _transcript()),
        interpretation_service,
        result_aggregator,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result_aggregator.calls == [extraction_mentions]
    assert result.results is extraction_results
    assert result.results.screen_works.movies == movie_results
    assert result.results.screen_works.movies[0] is movie_results[0]
    assert result.results.screen_works.movies[1] is movie_results[1]
    assert result.results.screen_works.tv_series == []


async def test_pipeline_returns_tv_only_results_from_aggregator() -> None:
    """Pass TV-only Mentions to aggregation without pipeline placeholders."""
    tv_series_mentions = [
        TVSeriesMention(title="The Last of Us", year=2023),
        TVSeriesMention(title="Arcane", year=2021),
    ]
    screen_work_mentions = ScreenWorkMentions(movies=[], tv_series=tv_series_mentions)
    extraction_mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    interpretation_service = _FakeInterpretationService(extraction_mentions)
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(_source(), _transcript()),
        interpretation_service,
        result_aggregator,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result_aggregator.calls == [extraction_mentions]
    assert result.results.screen_works.movies == []
    assert [item.status for item in result.results.screen_works.tv_series] == [
        ResultStatus.UNRESOLVED,
        ResultStatus.UNRESOLVED,
    ]
    assert [
        item.tv_series_mention for item in result.results.screen_works.tv_series
    ] == tv_series_mentions
    assert result.results.screen_works.tv_series[0].tv_series_mention is tv_series_mentions[0]
    assert result.results.screen_works.tv_series[1].tv_series_mention is tv_series_mentions[1]
    assert all(item.tv_series is None for item in result.results.screen_works.tv_series)


async def test_pipeline_groups_mixed_interpretation_results() -> None:
    """Keep Movie and TV Series ordering independent in grouped aggregator results."""
    movie_mention = MovieMention(title="Dune: Part One", year=2021)
    tv_series_mention = TVSeriesMention(title="The Last of Us", year=2023)
    movie_result = MovieResult(ResultStatus.UNRESOLVED, movie_mention, None)
    tv_series_result = TVSeriesResult(ResultStatus.UNRESOLVED, tv_series_mention, None)
    screen_work_mentions = ScreenWorkMentions(
        movies=[movie_mention],
        tv_series=[tv_series_mention],
    )
    screen_work_results = ScreenWorkResults(
        movies=[movie_result],
        tv_series=[tv_series_result],
    )
    extraction_mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    extraction_results = ExtractionResults(
        screen_works=screen_work_results,
        music=MusicResults(tracks=[], music_releases=[]),
        books=BookResults(books=[]),
    )
    interpretation_service = _FakeInterpretationService(extraction_mentions)
    result_aggregator = _FakeResultAggregator(results=extraction_results)
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(_source(), _transcript()),
        interpretation_service,
        result_aggregator,
    )

    result = await pipeline.run(_CANONICAL_URL)

    assert result_aggregator.calls == [extraction_mentions]
    assert result.results is extraction_results
    assert result.results.screen_works.movies == [movie_result]
    assert result.results.screen_works.tv_series == [tv_series_result]


async def test_pipeline_preserves_whisper_transcript_method() -> None:
    """Pass a Whisper Transcript through orchestration unchanged."""
    transcript = Transcript(
        text="Spoken audio transcript.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    pipeline = _pipeline(_FakeSourceTranscriptProvider(_source(), transcript))

    result = await pipeline.run(_CANONICAL_URL)

    assert result.transcript is transcript
    assert result.transcript.method is TranscriptMethod.WHISPER


async def test_pipeline_is_platform_agnostic_for_social_whisper_sources() -> None:
    """Pass a social Source and Whisper Transcript through unchanged."""
    source = Source(
        platform=Platform.INSTAGRAM,
        video_id="ABC123",
        url="https://www.instagram.com/reel/ABC123",
        title="Social pipeline video",
        description="Social pipeline description.",
        channel="Social pipeline channel",
        duration_seconds=42,
    )
    transcript = Transcript(
        text="Social pipeline speech.",
        language="en",
        method=TranscriptMethod.WHISPER,
    )
    source_transcript_provider = _FakeSourceTranscriptProvider(source, transcript)
    pipeline = _pipeline(source_transcript_provider)

    result = await pipeline.run(source.url)

    assert result.source is source
    assert result.transcript is transcript
    assert source_transcript_provider.calls == [source.url]


@pytest.mark.parametrize(
    "provider_error",
    [
        InvalidSourceError("invalid source"),
        SourceUnavailableError("source unavailable"),
        DurationLimitExceededError("duration limit exceeded"),
        TranscriptionError("Transcript is unavailable for this video."),
        PipelineTimeoutError("Transcript acquisition timed out."),
    ],
)
async def test_pipeline_propagates_provider_errors_unchanged(
    provider_error: Exception,
) -> None:
    """Propagate Source-and-Transcript provider errors without translation."""
    source_transcript_provider = _FakeSourceTranscriptProvider(
        _source(),
        _transcript(),
        error=provider_error,
    )
    pipeline = _pipeline(source_transcript_provider)

    with pytest.raises(type(provider_error)) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is provider_error
    assert source_transcript_provider.calls == [_CANONICAL_URL]


async def test_pipeline_propagates_aggregation_errors_unchanged() -> None:
    """Propagate TMDB resolution failures without changing their domain policy."""
    aggregation_error = EnrichmentError("TMDB candidate resolution failed.")
    screen_work_mentions = ScreenWorkMentions(
        movies=[MovieMention(title="Dune: Part One", year=2021)],
        tv_series=[TVSeriesMention(title="The Last of Us", year=2023)],
    )
    extraction_mentions = ExtractionMentions(
        screen_works=screen_work_mentions,
        music=MusicMentions(tracks=[], music_releases=[]),
        books=BookMentions(books=[]),
    )
    result_aggregator = _FakeResultAggregator(error=aggregation_error)
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(_source(), _transcript()),
        _FakeInterpretationService(extraction_mentions),
        result_aggregator,
    )

    with pytest.raises(EnrichmentError) as error:
        await pipeline.run(_CANONICAL_URL)

    assert error.value is aggregation_error
    assert result_aggregator.calls == [extraction_mentions]


async def test_pipeline_closes_source_provider_interpretation_and_aggregation_modules() -> None:
    """Delegate lifespan cleanup to every pipeline-owned module."""
    source_transcript_provider = _FakeSourceTranscriptProvider(_source(), _transcript())
    interpretation_service = _FakeInterpretationService()
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        source_transcript_provider,
        interpretation_service,
        result_aggregator,
    )

    await pipeline.aclose()

    assert source_transcript_provider.closed is True
    assert interpretation_service.closed is True
    assert result_aggregator.closed is True


class _FailingCloseInterpretationService(_FakeInterpretationService):
    async def aclose(self) -> None:
        raise RuntimeError("interpretation close failed")


async def test_pipeline_closes_aggregation_when_interpretation_close_fails() -> None:
    """Close aggregation when the first shutdown operation raises."""
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(_source(), _transcript()),
        _FailingCloseInterpretationService(),
        result_aggregator,
    )

    with pytest.raises(RuntimeError, match="interpretation close failed"):
        await pipeline.aclose()

    assert result_aggregator.closed is True


class _FailingCloseSourceTranscriptProvider(_FakeSourceTranscriptProvider):
    async def aclose(self) -> None:
        raise RuntimeError("source provider close failed")


async def test_pipeline_closes_later_modules_when_provider_close_fails() -> None:
    """Close later modules when Source-and-Transcript cleanup raises."""
    interpretation_service = _FakeInterpretationService()
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        _FailingCloseSourceTranscriptProvider(_source(), _transcript()),
        interpretation_service,
        result_aggregator,
    )

    with pytest.raises(RuntimeError, match="source provider close failed"):
        await pipeline.aclose()

    assert interpretation_service.closed is True
    assert result_aggregator.closed is True


async def test_pipeline_uses_configured_market_until_the_request_overrides_it() -> None:
    """Expose the configured market unless the API supplies an explicit one."""
    source = _source()
    transcript = _transcript()
    result_aggregator = _FakeResultAggregator()
    pipeline = _pipeline(
        _FakeSourceTranscriptProvider(source, transcript),
        result_aggregator=result_aggregator,
        default_market=SpotifyMarket("CA"),
    )

    default_result = await pipeline.run(_CANONICAL_URL)
    explicit_result = await pipeline.run(_CANONICAL_URL, SpotifyMarket("JP"))

    assert default_result.market == "CA"
    assert explicit_result.market == "JP"
    assert result_aggregator.markets == [SpotifyMarket("CA"), SpotifyMarket("JP")]
