"""Transcription-domain environment configuration."""

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranscriptionConfig(BaseSettings):
    """Validate settings used by transcription services."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    max_video_duration_seconds: int = Field(default=1800, gt=0)
    temp_media_dir: Path = Path(tempfile.gettempdir()) / "reelio"
    whisper_model: str = Field(default="large-v3-turbo", min_length=1)
    whisper_device: Literal["cuda", "cpu", "auto"] = "cuda"
    whisper_compute_type: str = Field(default="float16", min_length=1)


transcription_settings = TranscriptionConfig()
