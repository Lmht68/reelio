"""Interpret bounded Source material into grouped Extraction Mentions."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol, cast

from pydantic import ValidationError

from reelio.cache import AsyncCache, CacheCodecError, CacheEntry
from reelio.cache.interface import JsonObject
from reelio.extraction.exceptions import (
    InterpretationInputTooLargeError,
    InvalidLLMResponseError,
    MentionInterpretationError,
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
from reelio.extraction.services.interpretation.schemas import InterpretationResponse
from reelio.extraction.services.interpretation.types import LLMMessage
from reelio.extraction.types import (
    AuthorCredit,
    BookMention,
    BookMentions,
    ExtractionMentions,
    InterpretationMaterial,
    MovieMention,
    MusicMentions,
    MusicReleaseMention,
    ScreenWorkMentions,
    TrackMention,
    TranscriptMethod,
    TVSeriesMention,
    normalize_book_identity,
    normalize_music_identity,
    normalize_music_title_identity,
    normalize_screen_work_title,
)

logger = logging.getLogger(__name__)

_INPUT_LIMIT_MESSAGE = "Interpretation Material exceeds the configured limit."
_INVALID_RESPONSE_MESSAGE = "The LLM returned an invalid mention interpretation response."
_STAGE = "mention_interpretation"

_INTERPRETATION_CACHE_TTL_SECONDS = 2_592_000
_INTERPRETATION_CACHE_WAIT_TIMEOUT_SECONDS = 1.0
_INTERPRETATION_PROMPT_VERSION = "mention-interpretation-prompt-v1"
_INTERPRETATION_SCHEMA_VERSION = "mention-interpretation-schema-v1"


class MentionInterpretationProvider(Protocol):
    """Define the provider boundary used by mention interpretation."""

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
            MentionInterpretationError: If the provider request fails.
            PipelineTimeoutError: If the provider request times out.
        """
        ...

    async def aclose(self) -> None:
        """Release provider-owned network resources."""
        ...


class _InterpretationMentionsCodec:
    """Serialize validated ordered Mention collections."""

    def __init__(self, schema_version: str) -> None:
        self.version = schema_version

    def encode(self, value: ExtractionMentions) -> JsonObject:
        """Encode validated Mention collections without interpretation inputs."""
        try:
            response = InterpretationResponse.model_validate(
                {
                    "movies": [
                        {"title": movie.title, "year": movie.year}
                        for movie in value.screen_works.movies
                    ],
                    "tv_series": [
                        {"title": tv_series.title, "year": tv_series.year}
                        for tv_series in value.screen_works.tv_series
                    ],
                    "tracks": [
                        {
                            "track_title": track.track_title,
                            "artists": track.artists,
                            "release_title": track.release_title,
                            "release_year": track.release_year,
                        }
                        for track in value.music.tracks
                    ],
                    "music_releases": [
                        {
                            "release_title": music_release.release_title,
                            "artists": music_release.artists,
                            "release_year": music_release.release_year,
                        }
                        for music_release in value.music.music_releases
                    ],
                    "books": [
                        {
                            "title": book.title,
                            "authors": [author.name for author in book.authors],
                        }
                        for book in value.books.books
                    ],
                }
            )
        except (AttributeError, TypeError, ValidationError) as exc:
            raise CacheCodecError("Invalid Mention interpretation cache value") from exc
        return cast(JsonObject, response.model_dump(mode="json"))

    def decode(self, payload: JsonObject) -> ExtractionMentions:
        """Decode a fresh, validated Mention collection allocation."""
        try:
            response = InterpretationResponse.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid Mention interpretation cache value") from exc
        return _deduplicate(response)


def _interpretation_entry(
    material: InterpretationMaterial,
    provider_name: LLMProvider,
    model_name: str,
    prompt_version: str,
    schema_version: str,
) -> CacheEntry[ExtractionMentions]:
    return CacheEntry(
        layer="source:interpretation",
        key_version="v1",
        identity={
            "source_title": material.source_title,
            "source_description": material.source_description,
            "transcript_language": material.transcript.language,
            "transcript_text": material.transcript.text,
            "provider": provider_name.value,
            "model": model_name,
            "prompt_version": prompt_version,
            "interpretation_schema_version": schema_version,
        },
        codec=_InterpretationMentionsCodec(schema_version),
        ttl_seconds=_interpretation_ttl_seconds,
        wait_timeout_seconds=_INTERPRETATION_CACHE_WAIT_TIMEOUT_SECONDS,
    )


def _interpretation_ttl_seconds(_: ExtractionMentions) -> int:
    return _INTERPRETATION_CACHE_TTL_SECONDS


class MentionInterpretationService:
    """Validate Interpretation Material and produce canonical grouped mentions."""

    def __init__(
        self,
        provider: MentionInterpretationProvider,
        settings: InterpretationConfig,
        cache: AsyncCache,
    ) -> None:
        """Initialize interpretation with an LLM provider, limits, and shared cache.

        Args:
            provider: Provider-neutral structured completion adapter.
            settings: Interpretation Material size limits.
            cache: Shared application-lifespan cache for reusable interpretations.
        """
        self._provider = provider
        self._settings = settings
        self._cache = cache
        self._prompt_version = _INTERPRETATION_PROMPT_VERSION
        self._schema_version = _INTERPRETATION_SCHEMA_VERSION
        self._system_prompt = build_system_prompt()

    async def interpret(
        self,
        material: InterpretationMaterial,
    ) -> ExtractionMentions:
        """Interpret ordered, deduplicated mentions from Interpretation Material.

        Args:
            material: Complete normalized source context and Transcript to interpret.

        Returns:
            ExtractionMentions: Canonical mentions grouped by service scope.

        Raises:
            InterpretationInputTooLargeError: If any Interpretation Material field
                exceeds its configured limit.
            InvalidLLMResponseError: If the provider returns malformed or invalid JSON.
            MentionInterpretationError: If the provider request fails.
            PipelineTimeoutError: If the provider request times out.
        """
        self._validate_input_limits(material)
        if material.transcript.method is TranscriptMethod.TEXT_SUBMISSION:
            return await self._interpret_uncached(material)

        async def loader() -> ExtractionMentions:
            return await self._interpret_uncached(material)

        entry = _interpretation_entry(
            material,
            self._provider.provider_name,
            self._provider.model_name,
            self._prompt_version,
            self._schema_version,
        )
        return await self._cache.get_or_load(entry, loader)

    async def _interpret_uncached(
        self,
        material: InterpretationMaterial,
    ) -> ExtractionMentions:
        messages = (
            LLMMessage(role="system", content=self._system_prompt),
            LLMMessage(
                role="user",
                content=build_interpretation_material(
                    material.source_title,
                    material.source_description,
                    material.transcript.language,
                    material.transcript.text,
                ),
            ),
        )
        started_at = perf_counter()
        try:
            response_content = await self._provider.complete(messages)
        except (MentionInterpretationError, PipelineTimeoutError) as exc:
            logger.error(
                "mention interpretation provider request failed",
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
            response = InterpretationResponse.model_validate_json(response_content)
        except ValidationError as exc:
            logger.error(
                "mention interpretation response validation failed",
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
            "mention interpretation completed",
            extra={
                "stage": _STAGE,
                "duration_ms": _duration_ms(started_at),
                "movie_mention_count": len(mentions.screen_works.movies),
                "tv_series_mention_count": len(mentions.screen_works.tv_series),
                "track_mention_count": len(mentions.music.tracks),
                "music_release_mention_count": len(mentions.music.music_releases),
                "book_mention_count": len(mentions.books.books),
            },
        )
        return mentions

    async def aclose(self) -> None:
        """Close the lifespan-owned interpretation provider."""
        await self._provider.aclose()

    def _validate_input_limits(self, material: InterpretationMaterial) -> None:
        limits = (
            (
                "source_title_too_large",
                len(material.source_title),
                self._settings.max_source_title_chars,
            ),
            (
                "source_description_too_large",
                len(material.source_description),
                self._settings.max_description_chars,
            ),
            (
                "transcript_language_too_large",
                len(material.transcript.language),
                self._settings.max_transcript_language_chars,
            ),
            (
                "transcript_too_large",
                len(material.transcript.text),
                self._settings.max_transcript_chars,
            ),
        )
        for reason, actual_size, maximum_size in limits:
            if actual_size <= maximum_size:
                continue
            logger.error(
                "mention interpretation input rejected",
                extra={"stage": _STAGE, "reason": reason},
            )
            raise InterpretationInputTooLargeError(_INPUT_LIMIT_MESSAGE)


def _deduplicate(response: InterpretationResponse) -> ExtractionMentions:
    seen_movie_identities: set[tuple[str, int]] = set()
    movie_mentions: list[MovieMention] = []
    for movie in response.movies:
        title = normalize_screen_work_title(movie.title)
        movie_identity = (title, movie.year)
        if movie_identity in seen_movie_identities:
            continue
        seen_movie_identities.add(movie_identity)
        movie_mentions.append(MovieMention(title=title, year=movie.year))

    seen_tv_series_identities: set[tuple[str, int]] = set()
    tv_series_mentions: list[TVSeriesMention] = []
    for tv_series in response.tv_series:
        title = normalize_screen_work_title(tv_series.title)
        tv_series_identity = (title, tv_series.year)
        if tv_series_identity in seen_tv_series_identities:
            continue
        seen_tv_series_identities.add(tv_series_identity)
        tv_series_mentions.append(TVSeriesMention(title=title, year=tv_series.year))

    seen_track_identities: set[tuple[str, tuple[str, ...]]] = set()
    track_mentions: list[TrackMention] = []
    for track in response.tracks:
        track_identity = (
            normalize_music_title_identity(track.track_title),
            tuple(normalize_music_identity(artist) for artist in track.artists),
        )
        if track_identity in seen_track_identities:
            continue
        seen_track_identities.add(track_identity)
        track_mentions.append(
            TrackMention(
                track_title=track.track_title,
                artists=track.artists,
                release_title=track.release_title,
                release_year=track.release_year,
            )
        )

    seen_music_release_identities: set[tuple[str, tuple[str, ...]]] = set()
    music_release_mentions: list[MusicReleaseMention] = []
    for music_release in response.music_releases:
        music_release_identity = (
            normalize_music_title_identity(music_release.release_title),
            tuple(normalize_music_identity(artist) for artist in music_release.artists),
        )
        if music_release_identity in seen_music_release_identities:
            continue
        seen_music_release_identities.add(music_release_identity)
        music_release_mentions.append(
            MusicReleaseMention(
                release_title=music_release.release_title,
                artists=music_release.artists,
                release_year=music_release.release_year,
            )
        )

    seen_book_identities: set[tuple[str, tuple[str, ...]]] = set()
    book_mentions: list[BookMention] = []
    for book in response.books:
        book_identity = (
            normalize_book_identity(book.title),
            tuple(normalize_book_identity(author) for author in book.authors),
        )
        if book_identity in seen_book_identities:
            continue
        seen_book_identities.add(book_identity)
        book_mentions.append(
            BookMention(
                title=book.title,
                authors=[AuthorCredit(name=author) for author in book.authors],
            )
        )

    return ExtractionMentions(
        screen_works=ScreenWorkMentions(
            movies=movie_mentions,
            tv_series=tv_series_mentions,
        ),
        music=MusicMentions(
            tracks=track_mentions,
            music_releases=music_release_mentions,
        ),
        books=BookMentions(books=book_mentions),
    )


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1_000, 3)
