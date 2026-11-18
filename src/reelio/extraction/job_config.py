"""Environment-backed configuration for Source Extraction job execution."""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionJobConfig(BaseSettings):
    """Validate bounded Source Extraction worker and admission capacity settings."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_EXTRACTION_",
        env_file=".env",
        extra="ignore",
    )

    worker_concurrency: int = Field(default=2, gt=0)
    active_job_cap: int = Field(default=10, gt=0)
    lease_lifetime_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    heartbeat_interval_seconds: float = Field(default=15.0, gt=0, allow_inf_nan=False)
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_heartbeat_interval(self) -> "ExtractionJobConfig":
        """Require every configured heartbeat to precede lease expiration."""
        if self.heartbeat_interval_seconds >= self.lease_lifetime_seconds:
            raise ValueError("heartbeat interval must be strictly shorter than lease lifetime")
        return self
