"""Application composition and documentation lifecycle tests."""

from collections.abc import Callable
from typing import NoReturn, cast

import ctranslate2
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import reelio.extraction.services.transcription.acquisition as transcription_service
import reelio.main as main_module
from reelio.config import Environment, app_settings
from reelio.extraction.services.interpretation.config import (
    LLMProvider,
    LLMProviderSelectionConfig,
)
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.types import PipelineResult
from reelio.main import create_app


class _FakePipeline:
    def __init__(self) -> None:
        self.close_calls = 0

    async def run(self, url: str) -> PipelineResult:
        raise AssertionError(f"unexpected pipeline call for {url}")

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeProvider:
    def __init__(self) -> None:
        self.provider_name = LLMProvider.OPENAI
        self.model_name = "gpt-5-nano"
        self.close_calls = 0

    async def complete(self, messages: object) -> str:
        raise AssertionError(f"unexpected provider completion: {messages}")

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeScreenWorkResolver:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


def _transcription_settings(device: str) -> TranscriptionConfig:
    settings_type = cast(Callable[..., TranscriptionConfig], TranscriptionConfig)
    return settings_type(
        _env_file=None,
        whisper_device=device,
        whisper_model="test-model",
        whisper_compute_type="test-type",
    )


@pytest.mark.parametrize(
    ("environment", "expected_status"),
    [(Environment.PRODUCTION, 404), (Environment.LOCAL, 200)],
)
async def test_docs_are_gated_by_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: Environment,
    expected_status: int,
) -> None:
    """Expose docs only in non-production environments."""
    monkeypatch.setattr(app_settings, "environment", environment)
    application = create_app()
    transport = ASGITransport(app=application)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/docs")

    assert response.status_code == expected_status


async def test_injected_pipeline_factory_owns_one_pipeline_per_lifespan() -> None:
    """Store one factory result during lifespan and release it on shutdown."""
    pipeline = _FakePipeline()
    calls = 0

    async def factory() -> _FakePipeline:
        nonlocal calls
        calls += 1
        return pipeline

    application = create_app(pipeline_factory=factory)
    assert not hasattr(application.state, "extraction_pipeline")

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 1

    assert not hasattr(application.state, "extraction_pipeline")
    assert pipeline.close_calls == 1

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 2
    assert pipeline.close_calls == 2


def test_cuda_preflight_fails_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unavailable explicit CUDA before constructing WhisperModel."""
    constructor_calls: list[tuple[str, str, str]] = []

    def fake_model(
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> object:
        constructor_calls.append((model_size_or_path, device, compute_type))
        return object()

    monkeypatch.setattr(
        ctranslate2,
        "get_cuda_device_count",
        lambda: 0,
    )
    monkeypatch.setattr(transcription_service, "WhisperModel", fake_model)

    with pytest.raises(
        RuntimeError,
        match=r"^REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available\.$",
    ):
        transcription_service.load_whisper_transcriber(_transcription_settings("cuda"))

    assert constructor_calls == []


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_non_cuda_devices_skip_cuda_preflight(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Pass CPU and auto settings directly to the model constructor."""
    constructor_calls: list[tuple[str, str, str]] = []

    def fake_model(
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> object:
        constructor_calls.append((model_size_or_path, device, compute_type))
        return object()

    monkeypatch.setattr(
        ctranslate2,
        "get_cuda_device_count",
        lambda: pytest.fail("CPU and auto must skip CUDA preflight"),
    )
    monkeypatch.setattr(transcription_service, "WhisperModel", fake_model)

    transcription_service.load_whisper_transcriber(_transcription_settings(device))

    assert constructor_calls == [("test-model", device, "test-type")]


async def test_pipeline_factory_failure_aborts_lifespan() -> None:
    """Propagate model or dependency construction failures during startup."""

    async def failing_factory() -> _FakePipeline:
        raise RuntimeError("model load failed")

    application = create_app(pipeline_factory=failing_factory)
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="model load failed"):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_closes_one_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct one selected provider and close it when production stops."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    provider = _FakeProvider()
    resolver = _FakeScreenWorkResolver()
    provider_factory_calls = 0

    def create_provider(selection: LLMProviderSelectionConfig) -> _FakeProvider:
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        assert selection.llm_provider is LLMProvider.OPENAI
        return provider

    monkeypatch.setattr(main_module, "create_screen_work_mention_provider", create_provider)
    monkeypatch.setattr(main_module, "load_whisper_transcriber", lambda settings: object())
    monkeypatch.setattr(
        main_module,
        "create_tmdb_screen_work_resolver",
        lambda settings: resolver,
    )
    application = create_app()

    async with application.router.lifespan_context(application):
        assert provider_factory_calls == 1

    assert provider.close_calls == 1
    assert resolver.close_calls == 1


async def test_production_lifespan_closes_provider_after_partial_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close a selected provider when a later production dependency fails."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "openai")
    provider = _FakeProvider()

    def create_provider(selection: LLMProviderSelectionConfig) -> _FakeProvider:
        assert selection.llm_provider is LLMProvider.OPENAI
        return provider

    def fail_whisper_load(settings: TranscriptionConfig) -> NoReturn:
        raise RuntimeError("Whisper load failed")

    monkeypatch.setattr(
        main_module,
        "create_screen_work_mention_provider",
        create_provider,
    )
    monkeypatch.setattr(main_module, "load_whisper_transcriber", fail_whisper_load)
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(RuntimeError, match="Whisper load failed"):
        await context.__aenter__()
    assert provider.close_calls == 1
    assert not hasattr(application.state, "extraction_pipeline")


async def test_production_lifespan_rejects_unsupported_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail startup before making an extraction pipeline for an unknown provider."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", "unsupported")
    application = create_app()
    context = application.router.lifespan_context(application)

    with pytest.raises(ValidationError):
        await context.__aenter__()

    assert not hasattr(application.state, "extraction_pipeline")
