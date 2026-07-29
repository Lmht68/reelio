from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-pro"
    llm_api_key: str = ""

    # Whisper / Transcript
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    transcript_temp_dir: str | None = None
    whisper_max_concurrent: int = Field(default=2, gt=0)
    whisper_max_duration_seconds: int = Field(default=600, gt=0)

    # Entity extraction (LLM)
    llm_timeout_seconds: float = Field(default=60, gt=0)
    entity_max_transcript_chars: int = Field(default=12000, gt=0)


settings = Settings()
