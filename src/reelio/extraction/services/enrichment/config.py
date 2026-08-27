"""TMDB-enrichment environment configuration."""

from pydantic import Field, SecretStr
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


tmdb_settings = TMDBConfig()  # type: ignore[call-arg]
