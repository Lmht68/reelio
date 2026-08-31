"""Interpret bounded Source material into ordered Screen Work Mentions."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from pydantic import ValidationError

from reelio.extraction.exceptions import (
    InterpretationInputTooLargeError,
    InvalidLLMResponseError,
    MovieMentionInterpretationError,
    PipelineTimeoutError,
)
from reelio.extraction.services.interpretation.config import (
    InterpretationConfig,
    LLMProvider,
)
from reelio.extraction.services.interpretation.prompt import (
    build_interpretation_material,
    build_system_prompt,
)
from reelio.extraction.services.interpretation.schemas import ScreenWorkInterpretationResponse
from reelio.extraction.services.interpretation.types import LLMMessage
from reelio.extraction.types import (
    MovieMention,
    ScreenWorkMentions,
    Source,
    Transcript,
    TVSeriesMention,
    normalize_screen_work_title,
)

logger = logging.getLogger(__name__)

_INPUT_LIMIT_MESSAGE = "Interpretation Material exceeds the configured limit."
_INVALID_RESPONSE_MESSAGE = "The LLM returned an invalid Movie Mention response."
_STAGE = "screen_work_mention_interpretation"


class ScreenWorkMentionProvider(Protocol):
    """Define the provider boundary used by Screen Work Mention interpretation."""

    @property
    def provider_name(self) -> LLMProvider:
        """Return the provider identity safe for structured logging."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identity safe for structured logging."""
        ...

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return structured response content for trusted-role messages.

        Args:
            messages: Trusted instructions and bounded Interpretation Material.

        Returns:
            str: Raw structured response content.

        Raises:
            MovieMentionInterpretationError: If the provider request fails.
            PipelineTimeoutError: If the provider request times out.
        """
        ...

    async def aclose(self) -> None:
        """Release provider-owned network resources."""
        ...


class ScreenWorkMentionInterpretationService:
    """Validate Interpretation Material and produce canonical Screen Work Mentions."""

    def __init__(
        self,
        provider: ScreenWorkMentionProvider,
        settings: InterpretationConfig,
    ) -> None:
        """Initialize interpretation with an LLM provider and validated limits.

        Args:
            provider: Provider-neutral structured completion adapter.
            settings: Interpretation Material size limits.
        """
        self._provider = provider
        self._settings = settings
        self._system_prompt = build_system_prompt()

    async def interpret(
        self,
        source: Source,
        transcript: Transcript,
    ) -> ScreenWorkMentions:
        """Interpret ordered, deduplicated Screen Work Mentions from a Transcript.

        Args:
            source: Canonical Source whose metadata supports interpretation.
            transcript: Complete normalized Transcript to interpret.

        Returns:
            ScreenWorkMentions: Canonical mentions in first-reference order within
                each Screen Work kind.
        Raises:
            InterpretationInputTooLargeError: If any Interpretation Material field
                exceeds its configured limit.
            InvalidLLMResponseError: If the provider returns malformed or invalid JSON.
            MovieMentionInterpretationError: If the provider request fails.
            PipelineTimeoutError: If the provider request times out.
        """
        self._validate_input_limits(source, transcript)
        messages = (
            LLMMessage(role="system", content=self._system_prompt),
            LLMMessage(
                role="user",
                content=build_interpretation_material(
                    source.title,
                    source.description,
                    transcript.language,
                    transcript.text,
                ),
            ),
        )
        started_at = perf_counter()
        try:
            response_content = await self._provider.complete(messages)
        except (MovieMentionInterpretationError, PipelineTimeoutError) as exc:
            logger.error(
                "screen work mention interpretation provider request failed",
                extra={
                    "stage": _STAGE,
                    "reason": exc.code,
                    "provider": self._provider.provider_name.value,
                    "model": self._provider.model_name,
                    "duration_ms": _duration_ms(started_at),
                },
            )
            raise

        try:
            response = ScreenWorkInterpretationResponse.model_validate_json(response_content)
        except ValidationError as exc:
            logger.error(
                "screen work mention interpretation response validation failed",
                extra={
                    "stage": _STAGE,
                    "reason": "invalid_provider_response",
                    "provider": self._provider.provider_name.value,
                    "model": self._provider.model_name,
                    "duration_ms": _duration_ms(started_at),
                },
            )
            raise InvalidLLMResponseError(_INVALID_RESPONSE_MESSAGE) from exc

        mentions = _deduplicate(response)
        logger.debug(
            "screen work mention interpretation completed",
            extra={
                "stage": _STAGE,
                "duration_ms": _duration_ms(started_at),
                "movie_mention_count": len(mentions.movies),
                "tv_series_mention_count": len(mentions.tv_series),
            },
        )
        return mentions

    async def aclose(self) -> None:
        """Close the lifespan-owned interpretation provider."""
        await self._provider.aclose()

    def _validate_input_limits(self, source: Source, transcript: Transcript) -> None:
        limits = (
            ("source_title_too_large", len(source.title), self._settings.max_source_title_chars),
            (
                "source_description_too_large",
                len(source.description),
                self._settings.max_description_chars,
            ),
            (
                "transcript_language_too_large",
                len(transcript.language),
                self._settings.max_transcript_language_chars,
            ),
            (
                "transcript_too_large",
                len(transcript.text),
                self._settings.max_transcript_chars,
            ),
        )
        for reason, actual_size, maximum_size in limits:
            if actual_size <= maximum_size:
                continue
            logger.error(
                "screen work mention interpretation input rejected",
                extra={"stage": _STAGE, "reason": reason},
            )
            raise InterpretationInputTooLargeError(_INPUT_LIMIT_MESSAGE)


def _deduplicate(response: ScreenWorkInterpretationResponse) -> ScreenWorkMentions:
    seen_movie_identities: set[tuple[str, int]] = set()
    movie_mentions: list[MovieMention] = []
    for movie in response.movies:
        title = normalize_screen_work_title(movie.title)
        identity = (title, movie.year)
        if identity in seen_movie_identities:
            continue
        seen_movie_identities.add(identity)
        movie_mentions.append(MovieMention(title=title, year=movie.year))

    seen_tv_series_identities: set[tuple[str, int]] = set()
    tv_series_mentions: list[TVSeriesMention] = []
    for tv_series in response.tv_series:
        title = normalize_screen_work_title(tv_series.title)
        identity = (title, tv_series.year)
        if identity in seen_tv_series_identities:
            continue
        seen_tv_series_identities.add(identity)
        tv_series_mentions.append(TVSeriesMention(title=title, year=tv_series.year))

    return ScreenWorkMentions(
        movies=movie_mentions,
        tv_series=tv_series_mentions,
    )


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1_000, 3)
