"""Strict DeepSeek response schemas for Movie Mention interpretation."""

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reelio.extraction.types import maximum_movie_release_year, normalize_movie_title

_EARLIEST_MOVIE_YEAR = 1888


class InterpretedMovie(BaseModel):
    """Validate one canonical movie title and release year from DeepSeek."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1)
    year: int = Field(ge=_EARLIEST_MOVIE_YEAR)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        """Normalize serialization whitespace without changing title semantics.

        Args:
            value: Canonical title returned by DeepSeek.

        Returns:
            str: NFC-normalized title with collapsed whitespace.

        Raises:
            ValueError: If the normalized title is empty or contains control characters.
        """
        normalized_title = normalize_movie_title(value)
        if not normalized_title:
            raise ValueError("title must contain non-whitespace characters")
        if any(unicodedata.category(character) == "Cc" for character in normalized_title):
            raise ValueError("title must not contain control characters")
        return normalized_title

    @field_validator("year")
    @classmethod
    def validate_release_year(cls, value: int) -> int:
        """Reject implausible future release years.

        Args:
            value: Release year returned by DeepSeek.

        Returns:
            int: The validated release year.

        Raises:
            ValueError: If the year is more than two years in the future.
        """
        maximum_release_year = maximum_movie_release_year()
        if value > maximum_release_year:
            raise ValueError(f"year must be no later than {maximum_release_year}")
        return value


class MovieInterpretationResponse(BaseModel):
    """Validate the complete structured response returned by DeepSeek."""

    model_config = ConfigDict(extra="forbid", strict=True)

    movies: list[InterpretedMovie]
