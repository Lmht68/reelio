"""Strict response schemas for mention interpretation."""

import unicodedata
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reelio.extraction.types import (
    MINIMUM_SCREEN_WORK_MENTION_YEAR,
    maximum_screen_work_mention_year,
    normalize_music_text,
    normalize_screen_work_title,
)


def _normalize_music_field(value: str, field_name: str) -> str:
    """Validate and normalize one music text field.

    Args:
        value: Music text supplied by the interpretation provider.
        field_name: Schema field used in validation messages.

    Returns:
        str: NFC-normalized text with collapsed whitespace.

    Raises:
        ValueError: If the source text is blank or contains a control character.
    """
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    normalized_value = normalize_music_text(value)
    if not normalized_value:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    return normalized_value


class InterpretedScreenWorkMention(BaseModel):
    """Validate one canonical Screen Work title and year."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1)
    year: int = Field(ge=MINIMUM_SCREEN_WORK_MENTION_YEAR)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        """Normalize serialization whitespace without changing title semantics.

        Args:
            value: Canonical title returned by the provider.

        Returns:
            str: NFC-normalized title with collapsed whitespace.

        Raises:
            ValueError: If the normalized title is empty or contains control characters.
        """
        normalized_title = normalize_screen_work_title(value)
        if not normalized_title:
            raise ValueError("title must contain non-whitespace characters")
        if any(unicodedata.category(character) == "Cc" for character in normalized_title):
            raise ValueError("title must not contain control characters")
        return normalized_title

    @field_validator("year")
    @classmethod
    def validate_screen_work_year(cls, value: int) -> int:
        """Reject implausible future Screen Work years.

        Args:
            value: Screen Work year returned by the provider.

        Returns:
            int: The validated Screen Work year.

        Raises:
            ValueError: If the year is more than two years in the future.
        """
        maximum_screen_work_year = maximum_screen_work_mention_year()
        if value > maximum_screen_work_year:
            raise ValueError(f"year must be no later than {maximum_screen_work_year}")
        return value


class _InterpretedMusicMention(BaseModel):
    """Share strict validation for interpreted Track and Music Release Mentions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("artists", check_fields=False)
    @classmethod
    def normalize_artists(cls, value: list[str]) -> list[str]:
        """Normalize ordered artist names.

        Args:
            value: Artist names returned by the interpretation provider.

        Returns:
            list[str]: Normalized nonblank artist names in their original order.
        """
        return [_normalize_music_field(artist, "artists") for artist in value]

    @field_validator("release_title", check_fields=False)
    @classmethod
    def normalize_optional_release_title(cls, value: str | None) -> str | None:
        """Normalize an explicitly supplied Music Release title.

        Args:
            value: Optional Music Release title from the interpretation provider.

        Returns:
            str | None: Normalized title, or ``None`` when omitted by evidence.
        """
        if value is None:
            return None
        return _normalize_music_field(value, "release_title")

    @field_validator("release_year", check_fields=False)
    @classmethod
    def validate_optional_release_year(cls, value: int | None) -> int | None:
        """Validate an explicitly supplied Music Release year.

        Args:
            value: Optional Music Release year from the interpretation provider.

        Returns:
            int | None: A positive released year, or ``None`` when not explicit.

        Raises:
            ValueError: If the year is not positive or is in the future.
        """
        if value is None:
            return None
        current_year = date.today().year
        if value <= 0 or value > current_year:
            raise ValueError(f"release_year must be from 1 through {current_year}")
        return value


class InterpretedTrackMention(_InterpretedMusicMention):
    """Validate one released Track Mention from interpretation material."""

    track_title: str = Field(min_length=1)
    artists: list[str] = Field(min_length=1)
    release_title: str | None
    release_year: int | None

    @field_validator("track_title")
    @classmethod
    def normalize_track_title(cls, value: str) -> str:
        """Normalize the complete recording title.

        Args:
            value: Track title returned by the interpretation provider.

        Returns:
            str: Normalized nonblank Track title.
        """
        return _normalize_music_field(value, "track_title")


class InterpretedMusicReleaseMention(_InterpretedMusicMention):
    """Validate one independently named Music Release Mention."""

    release_title: str = Field(min_length=1)
    artists: list[str] = Field(min_length=1)
    release_year: int | None

    @field_validator("release_title")
    @classmethod
    def normalize_release_title(cls, value: str) -> str:
        """Normalize the complete Music Release title.

        Args:
            value: Music Release title returned by the interpretation provider.

        Returns:
            str: Normalized nonblank Music Release title.
        """
        return _normalize_music_field(value, "release_title")


class InterpretationResponse(BaseModel):
    """Validate the complete structured Mention interpretation response."""

    model_config = ConfigDict(extra="forbid", strict=True)

    movies: list[InterpretedScreenWorkMention]
    tv_series: list[InterpretedScreenWorkMention]
    tracks: list[InterpretedTrackMention]
    music_releases: list[InterpretedMusicReleaseMention]
