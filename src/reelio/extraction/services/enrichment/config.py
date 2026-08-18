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
    base_url: str = "https://api.themoviedb.org/3"


tmdb_settings = TMDBConfig()  # type: ignore[call-arg]
