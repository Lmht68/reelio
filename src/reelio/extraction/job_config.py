"""Environment-backed configuration for Source Extraction job execution."""

from pydantic import Field
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
