"""OpenAI-compatible DeepSeek adapter for Movie Mention interpretation."""

from collections.abc import Sequence
from typing import cast

from openai import APIError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from reelio.extraction.exceptions import (
    MovieMentionInterpretationError,
    PipelineTimeoutError,
)
from reelio.extraction.services.interpretation.config import (
    DeepSeekConfig,
    LLMProvider,
)
from reelio.extraction.services.interpretation.types import LLMMessage

_PROVIDER_ERROR_MESSAGE = "Movie Mention interpretation provider failed."
_PROVIDER_TIMEOUT_MESSAGE = "Movie Mention interpretation timed out."


class DeepSeekProvider:
    """Generate JSON Movie Mention interpretations through DeepSeek."""

    def __init__(self, client: AsyncOpenAI, settings: DeepSeekConfig) -> None:
        """Initialize the adapter with a lifespan-owned client and settings.

        Args:
            client: OpenAI SDK client configured for DeepSeek.
            settings: Validated DeepSeek request settings.
        """
        self._client = client
        self._settings = settings

    @property
    def provider_name(self) -> LLMProvider:
        """Return the provider identity safe for structured logging."""
        return LLMProvider.DEEPSEEK

    @property
    def model_name(self) -> str:
        """Return the configured model identity safe for structured logging."""
        return self._settings.model

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return one DeepSeek JSON completion.

        Args:
            messages: Trusted-role messages containing instructions and bounded data.

        Returns:
            str: Raw JSON content, or an empty string when no content was returned.

        Raises:
            PipelineTimeoutError: If the provider exhausts its timeout retries.
            MovieMentionInterpretationError: If the provider request otherwise fails.
        """
        provider_messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": message.role, "content": message.content} for message in messages],
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=provider_messages,
                response_format={"type": "json_object"},
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_output_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APITimeoutError as exc:
            raise PipelineTimeoutError(_PROVIDER_TIMEOUT_MESSAGE) from exc
        except APIError as exc:
            raise MovieMentionInterpretationError(_PROVIDER_ERROR_MESSAGE) from exc

        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def aclose(self) -> None:
        """Close the lifespan-owned DeepSeek client and its connection pool."""
        await self._client.close()


def create_deepseek_provider(settings: DeepSeekConfig) -> DeepSeekProvider:
    """Create a reusable DeepSeek adapter from validated application settings.

    Args:
        settings: DeepSeek credentials, endpoint, and generation options.

    Returns:
        DeepSeekProvider: Provider adapter owning one asynchronous client.
    """
    client = AsyncOpenAI(
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
    return DeepSeekProvider(client, settings)
