"""Construct the selected Movie Mention interpretation provider."""

from typing import assert_never

from reelio.extraction.services.interpretation.config import (
    DeepSeekConfig,
    LLMProvider,
    LLMProviderSelectionConfig,
    OpenAIConfig,
)
from reelio.extraction.services.interpretation.deepseek import create_deepseek_provider
from reelio.extraction.services.interpretation.openai import create_openai_provider
from reelio.extraction.services.interpretation.service import MovieMentionProvider


def create_movie_mention_provider(
    selection: LLMProviderSelectionConfig,
) -> MovieMentionProvider:
    """Construct the configured lifespan-owned Movie Mention provider.

    Args:
        selection: Validated fixed provider selection for this application lifespan.

    Returns:
        MovieMentionProvider: The selected native provider adapter.

    Raises:
        ValidationError: If the selected provider configuration is invalid.
        RuntimeError: If construction of the selected provider client fails.
    """
    match selection.llm_provider:
        case LLMProvider.OPENAI:
            return create_openai_provider(OpenAIConfig())  # type: ignore[call-arg]
        case LLMProvider.DEEPSEEK:
            return create_deepseek_provider(DeepSeekConfig())  # type: ignore[call-arg]
        case unsupported_provider:
            assert_never(unsupported_provider)
