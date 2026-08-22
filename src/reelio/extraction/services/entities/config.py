"""Entity-enrichment LLM configuration."""

from enum import StrEnum
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    """LLM providers supported by the entity enrichment pipeline."""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"


class LLMConfig(BaseSettings):
    """Validate provider credentials and LLM limits."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    provider: LLMProviderName = Field(
        default=LLMProviderName.DEEPSEEK,
        validation_alias="REELIO_LLM_PROVIDER",
    )
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    max_transcript_chars: int = Field(default=100_000, gt=0)
    max_description_chars: int = Field(default=2_000, gt=0)

    @model_validator(mode="after")
    def require_active_provider_key(self) -> Self:
        """Require credentials for the selected LLM provider.

        Raises:
            ValueError: If the selected provider has no API key.
        """
        if self.provider is LLMProviderName.DEEPSEEK and self.deepseek_api_key is None:
            raise ValueError(
                "REELIO_DEEPSEEK_API_KEY is required when REELIO_LLM_PROVIDER=deepseek"
            )
        if self.provider is LLMProviderName.OPENAI and self.openai_api_key is None:
            raise ValueError("REELIO_OPENAI_API_KEY is required when REELIO_LLM_PROVIDER=openai")
        return self


llm_settings = LLMConfig()
