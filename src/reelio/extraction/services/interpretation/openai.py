"""OpenAI Responses adapter for Screen Work Mention interpretation."""

from collections.abc import Sequence
from typing import cast

from openai import APIError, APITimeoutError, AsyncOpenAI
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam
from openai.types.shared_params import Reasoning

from reelio.extraction.exceptions import (
    MovieMentionInterpretationError,
    PipelineTimeoutError,
)
from reelio.extraction.services.interpretation.config import LLMProvider, OpenAIConfig
from reelio.extraction.services.interpretation.schemas import ScreenWorkInterpretationResponse
from reelio.extraction.services.interpretation.types import LLMMessage

_PROVIDER_ERROR_MESSAGE = "Movie Mention interpretation provider failed."
_PROVIDER_TIMEOUT_MESSAGE = "Movie Mention interpretation timed out."
_RESPONSE_SCHEMA_NAME = "screen_work_mention_interpretation"
_OFFICIAL_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    """Generate JSON Screen Work Mention interpretations through OpenAI."""

    def __init__(self, client: AsyncOpenAI, settings: OpenAIConfig) -> None:
        """Initialize the adapter with a lifespan-owned client and settings.

        Args:
            client: Asynchronous OpenAI client.
            settings: Validated OpenAI request settings.
        """
        self._client = client
        self._settings = settings

    @property
    def provider_name(self) -> LLMProvider:
        """Return the provider identity safe for structured logging."""
        return LLMProvider.OPENAI

    @property
    def model_name(self) -> str:
        """Return the configured model identity safe for structured logging."""
        return self._settings.model

    async def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return one strict JSON response from OpenAI.

        Args:
            messages: Trusted-role messages containing instructions and bounded data.

        Returns:
            str: Structured JSON response content.

        Raises:
            PipelineTimeoutError: If the provider exhausts its timeout retries.
            MovieMentionInterpretationError: If the provider rejects, truncates, or
                otherwise fails the request.
        """
        provider_messages = cast(
            ResponseInputParam,
            [{"role": message.role, "content": message.content} for message in messages],
        )
        response_format = cast(
            ResponseTextConfigParam,
            {
                "format": {
                    "type": "json_schema",
                    "name": _RESPONSE_SCHEMA_NAME,
                    "schema": ScreenWorkInterpretationResponse.model_json_schema(),
                    "strict": True,
                }
            },
        )
        reasoning = cast(Reasoning, {"effort": self._settings.reasoning_effort})
        try:
            response = await self._client.responses.create(
                model=self._settings.model,
                input=provider_messages,
                reasoning=reasoning,
                max_output_tokens=self._settings.max_output_tokens,
                store=False,
                text=response_format,
            )
        except APITimeoutError as exc:
            raise PipelineTimeoutError(_PROVIDER_TIMEOUT_MESSAGE) from exc
        except APIError as exc:
            raise MovieMentionInterpretationError(_PROVIDER_ERROR_MESSAGE) from exc

        if response.status != "completed" or not response.output_text:
            raise MovieMentionInterpretationError(_PROVIDER_ERROR_MESSAGE)
        return response.output_text

    async def aclose(self) -> None:
        """Close the lifespan-owned OpenAI client and its connection pool."""
        await self._client.close()


def create_openai_provider(settings: OpenAIConfig) -> OpenAIProvider:
    """Create a reusable OpenAI adapter from validated application settings.

    Args:
        settings: OpenAI credentials and request options.

    Returns:
        OpenAIProvider: Provider adapter owning one asynchronous client.
    """
    client = AsyncOpenAI(
        api_key=settings.api_key.get_secret_value(),
        base_url=_OFFICIAL_BASE_URL,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
    return OpenAIProvider(client, settings)
