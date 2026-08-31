"""Strict response schemas for Screen Work Mention interpretation."""

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reelio.extraction.types import (
    MINIMUM_SCREEN_WORK_MENTION_YEAR,
    maximum_screen_work_mention_year,
    normalize_screen_work_title,
)


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


class ScreenWorkInterpretationResponse(BaseModel):
    """Validate the complete structured Screen Work Mention response."""

    model_config = ConfigDict(extra="forbid", strict=True)

    movies: list[InterpretedScreenWorkMention]
    tv_series: list[InterpretedScreenWorkMention]
