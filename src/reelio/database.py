"""Shared durable PostgreSQL configuration and schema revision."""

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

CURRENT_SCHEMA_REVISION = "0003_extraction_job"


class DatabaseConfig(BaseSettings):
    """Validate the PostgreSQL endpoint shared by durable application resources."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_COORDINATION_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        validate_default=True,
    )

    database_url: SecretStr

    @model_validator(mode="after")
    def _validate_database_url(self) -> "DatabaseConfig":
        """Require one complete asynchronous PostgreSQL endpoint.

        Returns:
            The validated database settings.

        Raises:
            ValueError: If the endpoint is incomplete or does not use asyncpg.
        """
        endpoint = make_url(self.database_url.get_secret_value())
        if (
            endpoint.drivername != "postgresql+asyncpg"
            or endpoint.host is None
            or endpoint.database is None
        ):
            raise ValueError("database URL must use postgresql+asyncpg with a host and database")
        return self
