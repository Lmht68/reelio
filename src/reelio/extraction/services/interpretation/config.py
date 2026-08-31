"""Configuration models for Screen Work Mention interpretation."""

from enum import StrEnum
from typing import Annotated

from openai.types.shared_params import ReasoningEffort
from pydantic import AfterValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Identify an LLM provider supported for Screen Work Mention interpretation."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"


def _reject_blank_api_key(value: SecretStr) -> SecretStr:
    if not value.get_secret_value().strip():
        raise ValueError("API key must not be blank")
    return value


type _NonBlankSecret = Annotated[SecretStr, AfterValidator(_reject_blank_api_key)]


class _ReelioSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


class LLMProviderSelectionConfig(_ReelioSettings):
    """Validate the LLM provider selected for Screen Work Mention interpretation."""

    llm_provider: LLMProvider = Field(validation_alias="REELIO_LLM_PROVIDER")


class InterpretationConfig(_ReelioSettings):
    """Validate Interpretation Material limits."""

    max_source_title_chars: int = Field(default=500, gt=0)
    max_description_chars: int = Field(default=2_000, gt=0)
    max_transcript_language_chars: int = Field(default=64, gt=0)
    max_transcript_chars: int = Field(default=100_000, gt=0)


class _ProviderRequestConfig(_ReelioSettings):
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_output_tokens: int = Field(default=8_192, gt=0)
    max_retries: int = Field(default=2, ge=0)


class OpenAIConfig(_ProviderRequestConfig):
    """Validate OpenAI credentials and request options."""

    model_config = SettingsConfigDict(env_prefix="REELIO_OPENAI_")

    api_key: _NonBlankSecret = Field(validation_alias="REELIO_OPENAI_API_KEY")
    model: str = Field(default="gpt-5-nano", min_length=1)
    reasoning_effort: ReasoningEffort = "low"


class DeepSeekConfig(_ProviderRequestConfig):
    """Validate DeepSeek credentials and request options."""

    model_config = SettingsConfigDict(env_prefix="REELIO_DEEPSEEK_")

    api_key: _NonBlankSecret = Field(validation_alias="REELIO_DEEPSEEK_API_KEY")
    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    model: str = Field(default="deepseek-v4-flash", min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
