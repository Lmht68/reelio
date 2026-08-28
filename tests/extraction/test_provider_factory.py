"""Selected Movie Mention provider factory tests."""

import pytest

import reelio.extraction.services.interpretation.factory as provider_factory
from reelio.extraction.services.interpretation.config import (
    DeepSeekConfig,
    OpenAIConfig,
    LLMProviderSelectionConfig,
)


class _FakeProvider:
    async def complete(self, messages: object) -> str:
        """Reject unexpected completion calls during factory tests."""
        raise AssertionError("unexpected provider completion")

    async def aclose(self) -> None:
        """Release no resources for factory tests."""


def test_factory_constructs_only_selected_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore invalid DeepSeek settings when OpenAI is selected at startup."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REELIO_OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "not-a-number")
    provider = _FakeProvider()
    captured_settings: list[OpenAIConfig] = []

    def create_openai(settings: OpenAIConfig) -> _FakeProvider:
        captured_settings.append(settings)
        return provider

    monkeypatch.setattr(provider_factory, "create_openai_provider", create_openai)
    monkeypatch.setattr(
        provider_factory,
        "create_deepseek_provider",
        lambda settings: pytest.fail(f"unexpected DeepSeek configuration: {settings}"),
    )

    selected_provider = provider_factory.create_movie_mention_provider(LLMProviderSelectionConfig())

    assert selected_provider is provider
    assert captured_settings[0].model == "gpt-5-mini"


def test_factory_constructs_only_selected_deepseek_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore invalid OpenAI settings when DeepSeek is selected at startup."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("REELIO_DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("REELIO_OPENAI_MAX_RETRIES", "not-an-integer")
    provider = _FakeProvider()
    captured_settings: list[DeepSeekConfig] = []

    def create_deepseek(settings: DeepSeekConfig) -> _FakeProvider:
        captured_settings.append(settings)
        return provider

    monkeypatch.setattr(provider_factory, "create_deepseek_provider", create_deepseek)
    monkeypatch.setattr(
        provider_factory,
        "create_openai_provider",
        lambda settings: pytest.fail(f"unexpected OpenAI configuration: {settings}"),
    )

    selected_provider = provider_factory.create_movie_mention_provider(LLMProviderSelectionConfig())

    assert selected_provider is provider
    assert captured_settings[0].model == "deepseek-v4-flash"


def test_factory_does_not_fallback_after_selected_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate selected provider construction failure without trying DeepSeek."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REELIO_OPENAI_API_KEY", "openai-key")

    def fail_openai(settings: OpenAIConfig) -> _FakeProvider:
        raise RuntimeError(f"OpenAI construction failed for {settings.model}")

    monkeypatch.setattr(provider_factory, "create_openai_provider", fail_openai)
    monkeypatch.setattr(
        provider_factory,
        "create_deepseek_provider",
        lambda settings: pytest.fail(f"unexpected fallback: {settings}"),
    )

    with pytest.raises(RuntimeError, match="OpenAI construction failed"):
        provider_factory.create_movie_mention_provider(LLMProviderSelectionConfig())
