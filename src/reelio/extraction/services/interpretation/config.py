"""DeepSeek configuration for Movie Mention interpretation."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InterpretationConfig(BaseSettings):
    """Validate DeepSeek credentials, request options, and input limits."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    deepseek_api_key: SecretStr = Field(validation_alias="REELIO_DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        min_length=1,
    )
    deepseek_model: str = Field(default="deepseek-v4-flash", min_length=1)
    deepseek_request_timeout_seconds: float = Field(default=60.0, gt=0)
    deepseek_temperature: float = Field(default=0.0, ge=0, le=2)
    deepseek_max_output_tokens: int = Field(default=8_192, gt=0)
    max_source_title_chars: int = Field(default=500, gt=0)
    max_description_chars: int = Field(default=2_000, gt=0)
    max_transcript_language_chars: int = Field(default=64, gt=0)
    max_transcript_chars: int = Field(default=100_000, gt=0)

    @field_validator("deepseek_api_key")
    @classmethod
    def reject_blank_api_key(cls, value: SecretStr) -> SecretStr:
        """Reject an empty or whitespace-only DeepSeek API key.

        Args:
            value: Secret value loaded from application settings.

        Returns:
            SecretStr: The validated API key.

        Raises:
            ValueError: If the API key contains no non-whitespace characters.
        """
        if not value.get_secret_value().strip():
            raise ValueError("REELIO_DEEPSEEK_API_KEY must not be blank")
        return value


interpretation_settings = InterpretationConfig()  # type: ignore[call-arg]
