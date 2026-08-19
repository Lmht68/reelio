"""Application composition and documentation lifecycle tests."""

from collections.abc import Callable
from typing import cast

import ctranslate2  # type: ignore[import-untyped]
import pytest
from httpx import ASGITransport, AsyncClient

from reelio.config import Environment, app_settings
from reelio.extraction.services.transcription import service as transcription_service
from reelio.extraction.services.transcription.config import TranscriptionConfig
from reelio.extraction.types import PipelineResult
from reelio.main import create_app


class _FakePipeline:
    async def run(self, url: str) -> PipelineResult:
        raise AssertionError(f"unexpected pipeline call for {url}")


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

    async with application.router.lifespan_context(application):
        assert application.state.extraction_pipeline is pipeline
        assert calls == 2


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
