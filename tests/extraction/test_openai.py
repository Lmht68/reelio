"""OpenAI Responses adapter contract tests."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from openai import APIError, APITimeoutError, AsyncOpenAI

import reelio.extraction.services.interpretation.openai as openai_adapter
from reelio.extraction.exceptions import (
    MovieMentionInterpretationError,
    PipelineTimeoutError,
)
from reelio.extraction.services.interpretation.config import OpenAIConfig
from reelio.extraction.services.interpretation.openai import (
    OpenAIProvider,
    create_openai_provider,
)
from reelio.extraction.services.interpretation.schemas import MovieInterpretationResponse
from reelio.extraction.services.interpretation.types import LLMMessage


class _FakeResponses:
    def __init__(
        self,
        response: SimpleNamespace | None = None,
        error: APIError | None = None,
    ) -> None:
        self.response = response or SimpleNamespace(status="completed", output_text="")
        self.error = error
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> SimpleNamespace:
        """Record one Responses API request and return its configured result."""
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class _FakeOpenAIClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses
        self.closed = False

    async def close(self) -> None:
        """Record client closure."""
        self.closed = True


def _settings(**values: object) -> OpenAIConfig:
    settings_type = cast(Callable[..., OpenAIConfig], OpenAIConfig)
    return settings_type(_env_file=None, api_key="test-key", **values)


def test_openai_provider_constructor_uses_configured_client_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build one OpenAI client with the configured credential, timeout, and retries."""
    fake_client = _FakeOpenAIClient(_FakeResponses())
    client_options: list[dict[str, object]] = []

    def create_client(**options: object) -> _FakeOpenAIClient:
        client_options.append(options)
        return fake_client

    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", create_client)

    provider = create_openai_provider(
        _settings(
            request_timeout_seconds=12.5,
            max_retries=3,
        )
    )

    assert provider.provider_name == "openai"
    assert client_options == [
        {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "timeout": 12.5,
            "max_retries": 3,
        }
    ]


async def test_openai_adapter_sends_strict_responses_request_and_closes_client() -> None:
    """Map trusted messages to a private strict Structured Outputs request."""
    response_json = '{"movies":[{"title":"Dune: Part One","year":2021}]}'
    fake_responses = _FakeResponses(SimpleNamespace(status="completed", output_text=response_json))
    fake_client = _FakeOpenAIClient(fake_responses)
    provider = OpenAIProvider(cast(AsyncOpenAI, fake_client), _settings())
    messages = [LLMMessage(role="system", content="Return JSON")]

    content = await provider.complete(messages)
    await provider.aclose()

    assert content == response_json
    assert fake_responses.kwargs == {
        "model": "gpt-5-mini",
        "input": [{"role": "system", "content": "Return JSON"}],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 8_192,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "movie_mention_interpretation",
                "schema": MovieInterpretationResponse.model_json_schema(),
                "strict": True,
            }
        },
    }
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-5-mini"
    assert fake_client.closed is True


async def test_openai_adapter_maps_refusal_response() -> None:
    """Hide a Responses API refusal behind a provider-neutral failure."""
    refusal_response = SimpleNamespace(
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="unable to comply")],
            )
        ],
    )
    fake_client = _FakeOpenAIClient(_FakeResponses(refusal_response))
    provider = OpenAIProvider(cast(AsyncOpenAI, fake_client), _settings())

    with pytest.raises(MovieMentionInterpretationError):
        await provider.complete([])


async def test_openai_adapter_maps_incomplete_response() -> None:
    """Hide an incomplete Responses API result behind a provider-neutral failure."""
    fake_client = _FakeOpenAIClient(
        _FakeResponses(SimpleNamespace(status="incomplete", output_text=""))
    )
    provider = OpenAIProvider(cast(AsyncOpenAI, fake_client), _settings())

    with pytest.raises(MovieMentionInterpretationError):
        await provider.complete([])


async def test_openai_adapter_maps_missing_output() -> None:
    """Hide a completed Responses API result without structured output."""
    fake_client = _FakeOpenAIClient(
        _FakeResponses(SimpleNamespace(status="completed", output_text=""))
    )
    provider = OpenAIProvider(cast(AsyncOpenAI, fake_client), _settings())

    with pytest.raises(MovieMentionInterpretationError):
        await provider.complete([])


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (
            APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses")),
            PipelineTimeoutError,
        ),
        (
            APIError(
                "provider error",
                httpx.Request("POST", "https://api.openai.com/v1/responses"),
                body=None,
            ),
            MovieMentionInterpretationError,
        ),
    ],
)
async def test_openai_adapter_maps_sdk_failures(
    error: APIError,
    expected_error: type[Exception],
) -> None:
    """Translate OpenAI SDK exceptions to provider-neutral extraction failures."""
    fake_client = _FakeOpenAIClient(_FakeResponses(error=error))
    provider = OpenAIProvider(cast(AsyncOpenAI, fake_client), _settings())

    with pytest.raises(expected_error):
        await provider.complete([])
