"""Behavioral tests for environment-backed configuration."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from reelio.config import AppConfig, Environment
from reelio.enrichment.config import TMDBConfig
from reelio.entities.config import LLMConfig, LLMProviderName
from reelio.transcription.config import TranscriptionConfig


def _without_dotenv[SettingsType: BaseSettings](
    settings_type: type[SettingsType],
    **values: object,
) -> SettingsType:
    """Instantiate a settings class without reading the repository .env file."""
    constructor = cast(Callable[..., SettingsType], settings_type)
    return constructor(_env_file=None, **values)


def test_tmdb_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the TMDB bearer token when settings are instantiated."""
    monkeypatch.delenv("REELIO_TMDB_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_TMDB_API_KEY"):
        _without_dotenv(TMDBConfig)


def test_deepseek_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the active DeepSeek key for the default provider."""
    monkeypatch.delenv("REELIO_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("REELIO_LLM_PROVIDER", raising=False)

    with pytest.raises(ValidationError, match="REELIO_DEEPSEEK_API_KEY"):
        _without_dotenv(LLMConfig)


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the active OpenAI key when OpenAI is selected."""
    monkeypatch.delenv("REELIO_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("REELIO_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REELIO_LLM_PROVIDER", raising=False)

    with pytest.raises(ValidationError, match="REELIO_OPENAI_API_KEY"):
        _without_dotenv(LLMConfig, provider=LLMProviderName.OPENAI)


def test_inactive_provider_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow an inactive provider key to be absent."""
    monkeypatch.delenv("REELIO_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REELIO_LLM_PROVIDER", raising=False)

    settings = _without_dotenv(LLMConfig, deepseek_api_key="x")

    assert settings.provider is LLMProviderName.DEEPSEEK
    assert settings.openai_api_key is None


def test_configuration_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the stable Phase 1 defaults from clean process settings."""
    for variable in (
        "REELIO_ENVIRONMENT",
        "REELIO_LOG_LEVEL",
        "REELIO_MAX_VIDEO_DURATION_SECONDS",
        "REELIO_TEMP_MEDIA_DIR",
        "REELIO_WHISPER_MODEL",
        "REELIO_WHISPER_DEVICE",
        "REELIO_WHISPER_COMPUTE_TYPE",
        "REELIO_LLM_PROVIDER",
        "REELIO_DEEPSEEK_BASE_URL",
        "REELIO_DEEPSEEK_MODEL",
        "REELIO_OPENAI_API_KEY",
        "REELIO_OPENAI_MODEL",
        "REELIO_MAX_TRANSCRIPT_CHARS",
        "REELIO_MAX_DESCRIPTION_CHARS",
        "REELIO_TMDB_BASE_URL",
    ):
        monkeypatch.delenv(variable, raising=False)

    app_settings = _without_dotenv(AppConfig)
    transcription_settings = _without_dotenv(TranscriptionConfig)
    llm_settings = _without_dotenv(LLMConfig, deepseek_api_key="x")
    tmdb_settings = _without_dotenv(TMDBConfig, api_key="x")

    assert app_settings.environment is Environment.LOCAL
    assert app_settings.log_level == "INFO"
    assert transcription_settings.max_video_duration_seconds == 1800
    assert (
        transcription_settings.temp_media_dir == Path(tempfile.gettempdir()) / "reelio"
    )
    assert transcription_settings.whisper_model == "large-v3-turbo"
    assert transcription_settings.whisper_device == "cuda"
    assert transcription_settings.whisper_compute_type == "float16"
    assert llm_settings.deepseek_model == "deepseek-v4-flash"
    assert llm_settings.openai_model == "gpt-4o-mini"
    assert llm_settings.max_transcript_chars == 100_000
    assert llm_settings.max_description_chars == 2_000
    assert tmdb_settings.base_url == "https://api.themoviedb.org/3"


def test_video_duration_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respect a positive duration supplied by the process environment."""
    monkeypatch.setenv("REELIO_MAX_VIDEO_DURATION_SECONDS", "60")

    settings = _without_dotenv(TranscriptionConfig)

    assert settings.max_video_duration_seconds == 60


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject provider names outside the supported enum."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "anthropic")

    with pytest.raises(ValidationError):
        _without_dotenv(LLMConfig, deepseek_api_key="x")


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject log levels outside the supported literal values."""
    monkeypatch.setenv("REELIO_LOG_LEVEL", "CHATTY")

    with pytest.raises(ValidationError):
        _without_dotenv(AppConfig)
