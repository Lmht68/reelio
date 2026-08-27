"""OpenAI-compatible DeepSeek adapter for Movie Mention interpretation."""

from collections.abc import Sequence
from typing import cast

from openai import APIError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from reelio.extraction.exceptions import (
    MovieMentionInterpretationError,
    PipelineTimeoutError,
)
from reelio.extraction.services.interpretation.config import InterpretationConfig
from reelio.extraction.services.interpretation.types import LLMMessage

_PROVIDER_ERROR_MESSAGE = "Movie Mention interpretation provider failed."
_PROVIDER_TIMEOUT_MESSAGE = "Movie Mention interpretation timed out."


class DeepSeekProvider:
    """Generate JSON movie interpretations through a reusable DeepSeek client."""

    def __init__(self, client: AsyncOpenAI, settings: InterpretationConfig) -> None:
        """Initialize the adapter with a lifespan-owned client and settings.

        Args:
            client: OpenAI-compatible asynchronous DeepSeek client.
            settings: Validated model and generation settings.
        """
        self._client = client
        self._settings = settings

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
                model=self._settings.deepseek_model,
                messages=provider_messages,
                response_format={"type": "json_object"},
                temperature=self._settings.deepseek_temperature,
                max_tokens=self._settings.deepseek_max_output_tokens,
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


def create_deepseek_provider(settings: InterpretationConfig) -> DeepSeekProvider:
    """Create a reusable DeepSeek adapter from validated application settings.

    Args:
        settings: DeepSeek credentials, endpoint, timeout, and model options.

    Returns:
        DeepSeekProvider: Provider adapter owning one asynchronous client.
    """
    client = AsyncOpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        timeout=settings.deepseek_request_timeout_seconds,
    )
    return DeepSeekProvider(client, settings)
