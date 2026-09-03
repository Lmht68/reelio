"""Spotify catalog environment configuration."""

from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reelio.extraction.market import SpotifyMarket


class SpotifyConfig(BaseSettings):
    """Validate Spotify Client Credentials and catalog request settings."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    client_id: SecretStr = Field(validation_alias="REELIO_SPOTIFY_CLIENT_ID")
    client_secret: SecretStr = Field(validation_alias="REELIO_SPOTIFY_CLIENT_SECRET")
    default_market: SpotifyMarket = Field(
        default=SpotifyMarket("US"),
        validation_alias="REELIO_SPOTIFY_DEFAULT_MARKET",
    )
    base_url: str = Field(
        default="https://api.spotify.com/v1",
        validation_alias="REELIO_SPOTIFY_BASE_URL",
    )
    token_url: str = Field(
        default="https://accounts.spotify.com/api/token",
        validation_alias="REELIO_SPOTIFY_TOKEN_URL",
    )
    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias="REELIO_SPOTIFY_REQUEST_TIMEOUT_SECONDS",
    )
    token_expiry_skew_seconds: float = Field(
        default=30.0,
        ge=0,
        validation_alias="REELIO_SPOTIFY_TOKEN_EXPIRY_SKEW_SECONDS",
    )

    @field_validator("client_id", "client_secret")
    @classmethod
    def _reject_blank_secret(cls, value: SecretStr) -> SecretStr:
        """Reject credentials that cannot authenticate a Spotify request."""
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("base_url", "token_url")
    @classmethod
    def _validate_endpoint(cls, value: str) -> str:
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
