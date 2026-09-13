"""Environment configuration for provider-backed extraction enrichment."""

import unicodedata
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TMDBConfig(BaseSettings):
    """Validate the TMDB read-access token and endpoint."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: SecretStr = Field(validation_alias="REELIO_TMDB_API_KEY")
    base_url: str = Field(
        default="https://api.themoviedb.org/3",
        min_length=1,
        validation_alias="REELIO_TMDB_BASE_URL",
    )
    image_base_url: str = Field(
        default="https://image.tmdb.org/t/p/w500",
        min_length=1,
        validation_alias="REELIO_TMDB_IMAGE_BASE_URL",
    )
    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias="REELIO_TMDB_REQUEST_TIMEOUT_SECONDS",
    )


class OpenLibraryConfig(BaseSettings):
    """Validate Open Library contact, endpoint, timeout, and request rate."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    contact_email: str = Field(validation_alias="REELIO_OPEN_LIBRARY_CONTACT_EMAIL")
    base_url: str = Field(
        default="https://openlibrary.org",
        validation_alias="REELIO_OPEN_LIBRARY_BASE_URL",
    )
    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias="REELIO_OPEN_LIBRARY_REQUEST_TIMEOUT_SECONDS",
    )
    requests_per_second: float = Field(
        default=3.0,
        gt=0,
        le=3,
        validation_alias="REELIO_OPEN_LIBRARY_REQUESTS_PER_SECOND",
    )

    @field_validator("contact_email")
    @classmethod
    def _normalize_contact_email(cls, value: str) -> str:
        """Reject blank or control-character-bearing contact information."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("must not be blank")
        if any(unicodedata.category(character) == "Cc" for character in normalized_value):
            raise ValueError("must not contain control characters")
        return normalized_value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """Require a credential-free absolute HTTP endpoint."""
        endpoint = urlparse(value)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.netloc
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("must be an absolute HTTP URL without credentials or query")
        return value.rstrip("/")


tmdb_settings = TMDBConfig()  # type: ignore[call-arg]
