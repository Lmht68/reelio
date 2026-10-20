"""Opt-in checks against an operator-configured Reelio and Textify deployment."""

import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import ValidationError

from reelio.extraction.schemas import ExtractResponse

_REELIO_REPOSITORY_ROOT = Path(__file__).parents[1]
_TEXTIFY_PROJECT_NAME = "textify"
_REELIO_PROJECT_NAME = "reelio"
_LIVE_BASE_URL = os.environ.get("REELIO_LIVE_BASE_URL", "http://127.0.0.1:8181")
_LIVE_TEXTIFY_BASE_URL = os.environ.get(
    "REELIO_LIVE_TEXTIFY_BASE_URL",
    "http://127.0.0.1:8182",
)
_LIVE_SOURCE_URL = os.environ.get("REELIO_LIVE_SOURCE_URL")
_LIVE_RESTART_SOURCE_URL = os.environ.get("REELIO_LIVE_RESTART_SOURCE_URL")
_LIVE_TEXTIFY_PROJECT_DIRECTORY = os.environ.get("REELIO_LIVE_TEXTIFY_PROJECT_DIRECTORY")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_LIVE_SOURCE_URL and _LIVE_RESTART_SOURCE_URL and _LIVE_TEXTIFY_PROJECT_DIRECTORY),
        reason=(
            "REELIO_LIVE_SOURCE_URL, REELIO_LIVE_RESTART_SOURCE_URL, and "
            "REELIO_LIVE_TEXTIFY_PROJECT_DIRECTORY are required for live Compose checks."
        ),
    ),
]


def _required_live_value(value: str | None, environment_name: str) -> str:
    """Return one required live-test input after the module-level skip guard."""
    assert value is not None, f"{environment_name} must be configured for live checks"
    return value


def _textify_project_directory() -> Path:
    """Return the configured Textify Compose project directory."""
    return Path(
        _required_live_value(
            _LIVE_TEXTIFY_PROJECT_DIRECTORY,
            "REELIO_LIVE_TEXTIFY_PROJECT_DIRECTORY",
        )
    )


def _compose(
    project_directory: Path,
    project_name: str,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Run one Docker Compose command without exposing command output on failure."""
    return subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(project_directory),
            "--project-name",
            project_name,
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project_directory,
    )


def _require_compose_success(
    completed: subprocess.CompletedProcess[str],
    operation: str,
) -> None:
    """Fail with operation context while retaining potentially private command output."""
    if completed.returncode != 0:
        pytest.fail(f"{operation} failed with exit code {completed.returncode}")


def _reelio_compose(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Docker Compose against the Reelio project root."""
    return _compose(_REELIO_REPOSITORY_ROOT, _REELIO_PROJECT_NAME, *arguments)


def _textify_compose(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Docker Compose against the configured Textify project as textify."""
    return _compose(_textify_project_directory(), _TEXTIFY_PROJECT_NAME, *arguments)


def _compose_service_records() -> dict[str, dict[str, object]]:
    """Return rendered runtime records keyed by their Compose service name."""
    completed = _reelio_compose("ps", "-a", "--format", "json")
    _require_compose_success(completed, "Inspecting Reelio Compose services")

    records_value = cast(object, json.loads(completed.stdout))
    assert isinstance(records_value, list)
    service_records: dict[str, dict[str, object]] = {}
    for record in records_value:
        assert isinstance(record, dict)
        service_name = record["Service"]
        assert isinstance(service_name, str)
        service_records[service_name] = cast(dict[str, object], record)
    return service_records


def _assert_service_state(
    service_record: Mapping[str, object],
    expected_state: str,
    expected_health: str | None = None,
) -> None:
    """Require one safe Compose lifecycle and optional health state."""
    assert service_record["State"] == expected_state
    if expected_health is not None:
        assert service_record["Health"] == expected_health


def _count_rows(
    compose_command: Callable[..., subprocess.CompletedProcess[str]],
    query: str,
) -> int:
    """Run a count-only PostgreSQL query inside one Compose PostgreSQL service."""
    completed = compose_command(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "$1"',
        "postgres",
        query,
    )
    _require_compose_success(completed, "Running private PostgreSQL count query")
    count_text = completed.stdout.strip()
    assert count_text.isdecimal()
    return int(count_text)


def _reelio_coordination_count(*, attached_only: bool) -> int:
    """Count Reelio coordination rows without returning a private value."""
    predicate = " WHERE capability_link IS NOT NULL" if attached_only else ""
    return _count_rows(
        _reelio_compose,
        f"SELECT count(*) FROM textify_coordination{predicate}",
    )


def _textify_job_count(*, active_only: bool) -> int:
    """Count Textify jobs without returning job identifiers or source values."""
    predicate = " WHERE status IN ('queued', 'processing')" if active_only else ""
    return _count_rows(
        _textify_compose,
        f"SELECT count(*) FROM transcription_job{predicate}",
    )


async def _health_payload(base_url: str) -> object:
    """Fetch one deployed health response without exposing a response body on failure."""
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(5.0, connect=5.0),
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    try:
        return cast(object, response.json())
    except json.JSONDecodeError:
        pytest.fail("Deployed health endpoint did not return JSON")


async def _is_reelio_healthy() -> bool:
    """Return whether Reelio currently satisfies its liveness contract."""
    try:
        async with httpx.AsyncClient(
            base_url=_LIVE_BASE_URL,
            timeout=httpx.Timeout(5.0, connect=5.0),
        ) as client:
            response = await client.get("/health")
        return response.status_code == 200 and response.json() == {"status": "ok"}
    except (httpx.HTTPError, json.JSONDecodeError):
        return False


async def _wait_for_reelio_health() -> None:
    """Wait briefly for the restarted Reelio API to regain its liveness contract."""
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if await _is_reelio_healthy():
            return
        await asyncio.sleep(1.0)

    pytest.fail("Reelio API did not regain /health within 60 seconds after restart")


def _assert_no_textify_private_fields(payload: object) -> None:
    """Reject Textify job and Capability Link fields at every public response level."""
    forbidden_field_names = {
        "capability_link",
        "cancel",
        "job",
        "job_id",
        "lifecycle",
        "links",
        "self",
    }
    if isinstance(payload, Mapping):
        for field_name, field_value in payload.items():
            assert field_name not in forbidden_field_names
            _assert_no_textify_private_fields(field_value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_textify_private_fields(item)


def _payload_digest(value: object) -> str:
    """Hash a private payload for equality assertions without exposing its content."""
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


async def _extract(source_url: str) -> tuple[ExtractResponse, dict[str, object]]:
    """Request and validate one public extraction without exposing private response data."""
    async with httpx.AsyncClient(
        base_url=_LIVE_BASE_URL,
        timeout=httpx.Timeout(2_700.0, connect=10.0),
    ) as client:
        response = await client.post(
            "/api/extractions",
            json={"url": source_url, "market": "US"},
        )

    assert response.status_code == 200
    try:
        payload_value = cast(object, response.json())
    except json.JSONDecodeError:
        pytest.fail("Public extraction endpoint did not return JSON")
    assert isinstance(payload_value, dict)
    payload = cast(dict[str, object], payload_value)
    assert set(payload) == {"market", "source", "transcript", "statistics", "results"}
    _assert_no_textify_private_fields(payload)
    try:
        return ExtractResponse.model_validate(payload), payload
    except ValidationError:
        pytest.fail("Public extraction response did not satisfy ExtractResponse")


async def _assert_deployment_topology() -> None:
    """Assert deployed health gates and private API-to-Textify reachability."""
    service_records = await asyncio.to_thread(_compose_service_records)
    assert set(service_records) == {"api", "migrate", "postgres", "redis"}
    _assert_service_state(service_records["api"], "running", "healthy")
    _assert_service_state(service_records["postgres"], "running", "healthy")
    _assert_service_state(service_records["redis"], "running", "healthy")
    _assert_service_state(service_records["migrate"], "exited")
    assert service_records["migrate"]["ExitCode"] in {0, "0"}

    assert await _health_payload(_LIVE_BASE_URL) == {"status": "ok"}
    assert await _health_payload(_LIVE_TEXTIFY_BASE_URL) == {"status": "ok"}

    connectivity_check = await asyncio.to_thread(
        _reelio_compose,
        "exec",
        "-T",
        "api",
        "python",
        "-c",
        (
            "import sys, urllib.request; "
            "response = urllib.request.urlopen('http://textify:8182/openapi.json', timeout=5); "
            "sys.exit(0 if response.status == 200 else 1)"
        ),
    )
    _require_compose_success(connectivity_check, "Checking Reelio-to-Textify connectivity")


async def _wait_for_restart_window(
    baseline_job_count: int,
    baseline_attached_coordination_count: int,
) -> None:
    """Wait for an attached active Textify job or fail before a completed cache hit."""
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        job_count = await asyncio.to_thread(_textify_job_count, active_only=False)
        active_job_count = await asyncio.to_thread(_textify_job_count, active_only=True)
        attached_coordination_count = await asyncio.to_thread(
            _reelio_coordination_count,
            attached_only=True,
        )
        if job_count > baseline_job_count and active_job_count == 0:
            pytest.fail(
                "The configured restart workload completed before Reelio could restart. "
                "Use a longer fresh uncached captionless source."
            )
        if (
            job_count == baseline_job_count + 1
            and active_job_count >= 1
            and attached_coordination_count == baseline_attached_coordination_count + 1
        ):
            return
        await asyncio.sleep(1.0)

    pytest.fail(
        "Could not observe one attached active Textify job before the restart deadline. "
        "Use a longer fresh uncached captionless source."
    )


async def _wait_for_coordination_cleanup(baseline_coordination_count: int) -> None:
    """Wait for successful completion to retire the active Reelio coordination row."""
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        coordination_count = await asyncio.to_thread(
            _reelio_coordination_count,
            attached_only=False,
        )
        if coordination_count == baseline_coordination_count:
            return
        await asyncio.sleep(1.0)

    pytest.fail("Reelio did not clean up the completed PostgreSQL coordination row")


@pytest.mark.asyncio
async def test_live_compose_topology() -> None:
    """Prove long-running services, migration completion, and private connectivity."""
    await _assert_deployment_topology()


@pytest.mark.asyncio
async def test_live_source_extraction_reuses_exact_url_cache() -> None:
    """Return one public extraction, then reuse its exact URL without Textify API access."""
    await _assert_deployment_topology()
    source_url = _required_live_value(_LIVE_SOURCE_URL, "REELIO_LIVE_SOURCE_URL")

    recreated_redis = await asyncio.to_thread(
        _reelio_compose,
        "up",
        "-d",
        "--force-recreate",
        "--wait",
        "redis",
    )
    _require_compose_success(recreated_redis, "Recreating disposable Reelio Redis")

    baseline_job_count = await asyncio.to_thread(_textify_job_count, active_only=False)
    first_result, _ = await _extract(source_url)
    assert await asyncio.to_thread(_textify_job_count, active_only=False) == baseline_job_count + 1

    textify_api_stopped = False
    try:
        stopped_api = await asyncio.to_thread(_textify_compose, "stop", "api")
        _require_compose_success(stopped_api, "Stopping Textify API for cache verification")
        textify_api_stopped = True

        cached_result, _ = await _extract(source_url)
    finally:
        if textify_api_stopped:
            restarted_api = await asyncio.to_thread(_textify_compose, "start", "api")
            _require_compose_success(
                restarted_api, "Restarting Textify API after cache verification"
            )

    assert _payload_digest(first_result.source.model_dump(mode="json")) == _payload_digest(
        cached_result.source.model_dump(mode="json")
    )
    assert _payload_digest(first_result.transcript.model_dump(mode="json")) == _payload_digest(
        cached_result.transcript.model_dump(mode="json")
    )
    assert await asyncio.to_thread(_textify_job_count, active_only=False) == baseline_job_count + 1


@pytest.mark.asyncio
async def test_live_restart_resumes_active_textify_job_from_postgres() -> None:
    """Restart only Reelio API while a Textify job is active without admitting another job."""
    await _assert_deployment_topology()
    restart_source_url = _required_live_value(
        _LIVE_RESTART_SOURCE_URL,
        "REELIO_LIVE_RESTART_SOURCE_URL",
    )

    baseline_job_count = await asyncio.to_thread(_textify_job_count, active_only=False)
    baseline_attached_coordination_count = await asyncio.to_thread(
        _reelio_coordination_count,
        attached_only=True,
    )
    baseline_coordination_count = await asyncio.to_thread(
        _reelio_coordination_count,
        attached_only=False,
    )

    async with httpx.AsyncClient(
        base_url=_LIVE_BASE_URL,
        timeout=httpx.Timeout(2_700.0, connect=10.0),
    ) as client:
        initial_request = asyncio.create_task(
            client.post(
                "/api/extractions",
                json={"url": restart_source_url, "market": "US"},
            )
        )
        try:
            await _wait_for_restart_window(
                baseline_job_count,
                baseline_attached_coordination_count,
            )
            restarted_api = await asyncio.to_thread(_reelio_compose, "restart", "api")
            _require_compose_success(restarted_api, "Restarting Reelio API")
            await _wait_for_reelio_health()
            assert (
                await asyncio.to_thread(_reelio_coordination_count, attached_only=True)
                == baseline_attached_coordination_count + 1
            )
        finally:
            initial_request.cancel()
            with suppress(asyncio.CancelledError, httpx.HTTPError):
                await initial_request

    await _extract(restart_source_url)
    assert await asyncio.to_thread(_textify_job_count, active_only=False) == baseline_job_count + 1
    await _wait_for_coordination_cleanup(baseline_coordination_count)
