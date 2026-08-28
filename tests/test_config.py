"""Behavioral tests for environment-backed configuration."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from reelio.config import AppConfig, Environment
from reelio.extraction.services.enrichment.config import TMDBConfig
from reelio.extraction.services.interpretation.config import (
    DeepSeekConfig,
    InterpretationConfig,
    LLMProvider,
    OpenAIConfig,
    LLMProviderSelectionConfig,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig


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


def test_provider_selection_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject startup configuration that omits the LLM provider selection."""
    monkeypatch.delenv("REELIO_LLM_PROVIDER", raising=False)

    with pytest.raises(ValidationError, match="REELIO_LLM_PROVIDER"):
        _without_dotenv(LLMProviderSelectionConfig)


@pytest.mark.parametrize(
    "value",
    ["", " openai", "openai ", "OpenAI", "DEEPSEEK", "unsupported"],
)
def test_provider_selection_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Accept only exact lowercase configured provider identifiers."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", value)

    with pytest.raises(ValidationError):
        _without_dotenv(LLMProviderSelectionConfig)


@pytest.mark.parametrize(
    ("value", "expected_provider"),
    [("openai", LLMProvider.OPENAI), ("deepseek", LLMProvider.DEEPSEEK)],
)
def test_provider_selection_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected_provider: LLMProvider,
) -> None:
    """Parse each supported provider identity without normalization."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", value)

    settings = _without_dotenv(LLMProviderSelectionConfig)

    assert settings.llm_provider is expected_provider


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require OpenAI credentials when OpenAI configuration is selected."""
    monkeypatch.delenv("REELIO_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_OPENAI_API_KEY"):
        _without_dotenv(OpenAIConfig)


def test_deepseek_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require DeepSeek credentials when DeepSeek configuration is selected."""
    monkeypatch.delenv("REELIO_DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_DEEPSEEK_API_KEY"):
        _without_dotenv(DeepSeekConfig)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_openai_rejects_blank_api_key(api_key: str) -> None:
    """Reject empty and whitespace-only OpenAI credentials."""
    with pytest.raises(ValidationError, match="must not be blank"):
        _without_dotenv(OpenAIConfig, api_key=api_key)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_deepseek_rejects_blank_api_key(api_key: str) -> None:
    """Reject empty and whitespace-only DeepSeek credentials."""
    with pytest.raises(ValidationError, match="must not be blank"):
        _without_dotenv(DeepSeekConfig, api_key=api_key)


def test_openai_rejects_unsupported_reasoning_effort() -> None:
    """Reject unsupported OpenAI reasoning effort during startup validation."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            OpenAIConfig,
            api_key="openai-key",
            reasoning_effort="unsupported",
        )


def test_openai_configuration_ignores_inactive_deepseek_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate OpenAI without reading an invalid inactive DeepSeek configuration."""
    monkeypatch.setenv("REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "not-a-number")

    settings = _without_dotenv(OpenAIConfig, api_key="openai-key")

    assert settings.model == "gpt-5-mini"


def test_deepseek_configuration_ignores_inactive_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate DeepSeek without reading an invalid inactive OpenAI configuration."""
    monkeypatch.setenv("REELIO_OPENAI_MAX_RETRIES", "not-an-integer")

    settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")

    assert settings.model == "deepseek-v4-flash"


def test_configuration_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the stable defaults from clean process settings."""
    for variable in (
        "REELIO_ENVIRONMENT",
        "REELIO_LOG_LEVEL",
        "REELIO_MAX_VIDEO_DURATION_SECONDS",
        "REELIO_TEMP_MEDIA_DIR",
        "REELIO_WHISPER_MODEL",
        "REELIO_WHISPER_DEVICE",
        "REELIO_WHISPER_COMPUTE_TYPE",
        "REELIO_OPENAI_MODEL",
        "REELIO_OPENAI_REASONING_EFFORT",
        "REELIO_OPENAI_REQUEST_TIMEOUT_SECONDS",
        "REELIO_OPENAI_MAX_OUTPUT_TOKENS",
        "REELIO_OPENAI_MAX_RETRIES",
        "REELIO_DEEPSEEK_BASE_URL",
        "REELIO_DEEPSEEK_MODEL",
        "REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS",
        "REELIO_DEEPSEEK_TEMPERATURE",
        "REELIO_DEEPSEEK_MAX_OUTPUT_TOKENS",
        "REELIO_DEEPSEEK_MAX_RETRIES",
        "REELIO_MAX_SOURCE_TITLE_CHARS",
        "REELIO_MAX_DESCRIPTION_CHARS",
        "REELIO_MAX_TRANSCRIPT_LANGUAGE_CHARS",
        "REELIO_MAX_TRANSCRIPT_CHARS",
        "REELIO_TMDB_BASE_URL",
        "REELIO_TMDB_IMAGE_BASE_URL",
        "REELIO_TMDB_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)

    app_settings = _without_dotenv(AppConfig)
    transcription_settings = _without_dotenv(TranscriptionConfig)
    interpretation_settings = _without_dotenv(InterpretationConfig)
    openai_settings = _without_dotenv(OpenAIConfig, api_key="openai-key")
    deepseek_settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")
    tmdb_settings = _without_dotenv(TMDBConfig, api_key="tmdb-key")

    assert app_settings.environment is Environment.LOCAL
    assert app_settings.log_level == "INFO"
    assert transcription_settings.max_video_duration_seconds == 1800
    assert transcription_settings.temp_media_dir == Path(tempfile.gettempdir()) / "reelio"
    assert transcription_settings.whisper_model == "large-v3-turbo"
    assert transcription_settings.whisper_device == "cuda"
    assert transcription_settings.whisper_compute_type == "float16"
    assert transcription_settings.whisper_beam_size == 1
    assert transcription_settings.whisper_vad_filter is True
    assert transcription_settings.whisper_temperature == 0.0
    assert transcription_settings.whisper_cond_on_prev_txt is True
    assert transcription_settings.whisper_initial_prompt == ""
    assert openai_settings.model == "gpt-5-mini"
    assert openai_settings.reasoning_effort == "low"
    assert openai_settings.request_timeout_seconds == 60.0
    assert openai_settings.max_output_tokens == 8_192
    assert openai_settings.max_retries == 2
    assert deepseek_settings.base_url == "https://api.deepseek.com"
    assert deepseek_settings.model == "deepseek-v4-flash"
    assert deepseek_settings.request_timeout_seconds == 60.0
    assert deepseek_settings.temperature == 0.0
    assert deepseek_settings.max_output_tokens == 8_192
    assert deepseek_settings.max_retries == 2
    assert interpretation_settings.max_source_title_chars == 500
    assert interpretation_settings.max_description_chars == 2_000
    assert interpretation_settings.max_transcript_language_chars == 64
    assert interpretation_settings.max_transcript_chars == 100_000
    assert tmdb_settings.base_url == "https://api.themoviedb.org/3"
    assert tmdb_settings.image_base_url == "https://image.tmdb.org/t/p/w500"
    assert tmdb_settings.request_timeout_seconds == 10.0


def test_tmdb_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse TMDB endpoint, image, and timeout settings."""
    monkeypatch.setenv("REELIO_TMDB_BASE_URL", "https://tmdb.test/3")
    monkeypatch.setenv("REELIO_TMDB_IMAGE_BASE_URL", "https://images.test/w342")
    monkeypatch.setenv("REELIO_TMDB_REQUEST_TIMEOUT_SECONDS", "4.5")

    settings = _without_dotenv(TMDBConfig, api_key="tmdb-key")

    assert settings.base_url == "https://tmdb.test/3"
    assert settings.image_base_url == "https://images.test/w342"
    assert settings.request_timeout_seconds == 4.5


def test_video_duration_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respect a positive duration supplied by the process environment."""
    monkeypatch.setenv("REELIO_MAX_VIDEO_DURATION_SECONDS", "60")

    settings = _without_dotenv(TranscriptionConfig)

    assert settings.max_video_duration_seconds == 60


def test_whisper_transcription_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse Whisper inference options from environment variables."""
    monkeypatch.setenv("REELIO_WHISPER_BEAM_SIZE", "7")
    monkeypatch.setenv("REELIO_WHISPER_VAD_FILTER", "false")
    monkeypatch.setenv("REELIO_WHISPER_TEMPERATURE", "0.25")
    monkeypatch.setenv("REELIO_WHISPER_COND_ON_PREV_TXT", "false")
    monkeypatch.setenv("REELIO_WHISPER_INITIAL_PROMPT", "Use proper names.")

    settings = _without_dotenv(TranscriptionConfig)

    assert settings.whisper_beam_size == 7
    assert settings.whisper_vad_filter is False
    assert settings.whisper_temperature == 0.25
    assert settings.whisper_cond_on_prev_txt is False
    assert settings.whisper_initial_prompt == "Use proper names."


def test_openai_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse OpenAI request options from provider-specific environment variables."""
    monkeypatch.setenv("REELIO_OPENAI_MODEL", "gpt-5.1")
    monkeypatch.setenv("REELIO_OPENAI_REASONING_EFFORT", "medium")
    monkeypatch.setenv("REELIO_OPENAI_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REELIO_OPENAI_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("REELIO_OPENAI_MAX_RETRIES", "3")

    settings = _without_dotenv(OpenAIConfig, api_key="openai-key")

    assert settings.model == "gpt-5.1"
    assert settings.reasoning_effort == "medium"
    assert settings.request_timeout_seconds == 12.5
    assert settings.max_output_tokens == 4_096
    assert settings.max_retries == 3


def test_deepseek_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse DeepSeek request options from provider-specific environment variables."""
    monkeypatch.setenv("REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REELIO_DEEPSEEK_TEMPERATURE", "0.1")
    monkeypatch.setenv("REELIO_DEEPSEEK_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("REELIO_DEEPSEEK_MAX_RETRIES", "3")

    settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")

    assert settings.request_timeout_seconds == 12.5
    assert settings.temperature == 0.1
    assert settings.max_output_tokens == 4_096
    assert settings.max_retries == 3


def test_interpretation_material_limit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse Interpretation Material limits from environment variables."""
    monkeypatch.setenv("REELIO_MAX_SOURCE_TITLE_CHARS", "100")
    monkeypatch.setenv("REELIO_MAX_DESCRIPTION_CHARS", "1000")
    monkeypatch.setenv("REELIO_MAX_TRANSCRIPT_LANGUAGE_CHARS", "20")
    monkeypatch.setenv("REELIO_MAX_TRANSCRIPT_CHARS", "50000")

    settings = _without_dotenv(InterpretationConfig)

    assert settings.max_source_title_chars == 100
    assert settings.max_description_chars == 1_000
    assert settings.max_transcript_language_chars == 20
    assert settings.max_transcript_chars == 50_000


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject log levels outside the supported literal values."""
    monkeypatch.setenv("REELIO_LOG_LEVEL", "CHATTY")

    with pytest.raises(ValidationError):
        _without_dotenv(AppConfig)
