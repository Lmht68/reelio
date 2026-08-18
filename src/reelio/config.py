"""Application-wide environment configuration."""

from enum import StrEnum
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environments supported by the application."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class AppConfig(BaseSettings):
    """Validate settings shared by the application composition root."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Environment = Environment.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


app_settings = AppConfig()
