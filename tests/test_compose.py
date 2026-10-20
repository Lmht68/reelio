"""Credential-free contract coverage for the local Compose deployment."""

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[1]
_COMPOSE_FILE = _REPOSITORY_ROOT / "compose.yaml"
_REQUIRED_ENVIRONMENT = {
    "REELIO_POSTGRES_DB": "reelio",
    "REELIO_POSTGRES_USER": "reelio",
    "REELIO_POSTGRES_PASSWORD": "url-safe-postgres-password",
    "REELIO_COORDINATION_HMAC_SECRET": "coordination-hmac-secret",
    "REELIO_CACHE_KEY_SECRET": "cache-key-secret",
    "REELIO_LLM_PROVIDER": "deepseek",
    "REELIO_TMDB_API_KEY": "tmdb-api-key",
    "REELIO_SPOTIFY_CLIENT_ID": "spotify-client-id",
    "REELIO_SPOTIFY_CLIENT_SECRET": "spotify-client-secret",
    "REELIO_OPEN_LIBRARY_CONTACT_EMAIL": "operator@example.invalid",
}


def _compose_is_available() -> bool:
    """Return whether the local Docker Compose CLI can render configurations."""
    if shutil.which("docker") is None:
        return False

    completed = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


_COMPOSE_IS_AVAILABLE = _compose_is_available()
pytestmark = [
    pytest.mark.compose,
    pytest.mark.skipif(
        not _COMPOSE_IS_AVAILABLE,
        reason="Docker Compose CLI is unavailable",
    ),
]


def _render_compose(overrides: Mapping[str, str] | None = None) -> dict[str, object]:
    """Render Compose JSON with sentinel inputs isolated from host credentials."""
    environment_values = dict(_REQUIRED_ENVIRONMENT)
    if overrides is not None:
        environment_values.update(overrides)

    process_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("REELIO_") and name not in {"COMPOSE_FILE", "COMPOSE_ENV_FILES"}
    }
    with TemporaryDirectory() as directory:
        environment_file = Path(directory) / ".env"
        environment_file.write_text(
            "".join(f"{name}={value}\n" for name, value in environment_values.items()),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "docker",
                "compose",
                "--project-directory",
                str(_REPOSITORY_ROOT),
                "--env-file",
                str(environment_file),
                "-f",
                str(_COMPOSE_FILE),
                "config",
                "--format",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=_REPOSITORY_ROOT,
            env=process_environment,
        )

    if completed.returncode != 0:
        pytest.fail(f"docker compose config failed: {completed.stderr}")

    rendered = cast(object, json.loads(completed.stdout))
    assert isinstance(rendered, dict)
    return cast(dict[str, object], rendered)


def _compose_config(
    environment_values: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run Compose configuration with selected sentinel variables omitted."""
    process_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("REELIO_") and name not in {"COMPOSE_FILE", "COMPOSE_ENV_FILES"}
    }
    with TemporaryDirectory() as directory:
        environment_file = Path(directory) / ".env"
        environment_file.write_text(
            "".join(f"{name}={value}\n" for name, value in environment_values.items()),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                "docker",
                "compose",
                "--project-directory",
                str(_REPOSITORY_ROOT),
                "--env-file",
                str(environment_file),
                "-f",
                str(_COMPOSE_FILE),
                "config",
                "--format",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=_REPOSITORY_ROOT,
            env=process_environment,
        )


def _service(rendered: Mapping[str, object], service_name: str) -> dict[str, object]:
    """Return one rendered service after verifying the Compose document shape."""
    services = rendered["services"]
    assert isinstance(services, dict)
    service = services[service_name]
    assert isinstance(service, dict)
    return cast(dict[str, object], service)


def _healthcheck(service: Mapping[str, object]) -> dict[str, object]:
    """Return a service healthcheck after verifying its rendered shape."""
    healthcheck = service["healthcheck"]
    assert isinstance(healthcheck, dict)
    return cast(dict[str, object], healthcheck)


def _depends_on(service: Mapping[str, object], dependency: str, condition: str) -> None:
    """Require one health or completion gate in rendered Compose dependencies."""
    dependencies = service["depends_on"]
    assert isinstance(dependencies, dict)
    dependency_config = dependencies[dependency]
    assert isinstance(dependency_config, dict)
    assert dependency_config["condition"] == condition


def _assert_loopback_api_port(service: Mapping[str, object], published_port: int) -> None:
    """Require the sole API publication to target loopback and container port 8181."""
    ports = service["ports"]
    assert isinstance(ports, list)
    assert len(ports) == 1
    port = ports[0]
    if isinstance(port, str):
        assert port == f"127.0.0.1:{published_port}:8181"
        return

    assert isinstance(port, dict)
    assert port["host_ip"] == "127.0.0.1"
    assert port["published"] == str(published_port)
    assert port["target"] == 8181


def _assert_postgres_data_mount(service: Mapping[str, object]) -> None:
    """Require PostgreSQL's explicit named-volume data mount."""
    volumes = service["volumes"]
    assert isinstance(volumes, list)
    assert len(volumes) == 1
    volume = volumes[0]
    if isinstance(volume, str):
        assert volume == "postgres-data:/var/lib/postgresql/data"
        return

    assert isinstance(volume, dict)
    assert volume["source"] == "postgres-data"
    assert volume["target"] == "/var/lib/postgresql/data"
    assert volume["type"] == "volume"


def test_compose_renders_local_deployment_contract() -> None:
    """Render the complete private local deployment without real credentials."""
    rendered = _render_compose()

    services = rendered["services"]
    assert isinstance(services, dict)
    assert set(services) == {"api", "migrate", "postgres", "redis"}

    api = _service(rendered, "api")
    migrate = _service(rendered, "migrate")
    postgres = _service(rendered, "postgres")
    redis = _service(rendered, "redis")

    for service in (api, migrate):
        build = service["build"]
        assert isinstance(build, dict)
        assert Path(cast(str, build["context"])).resolve() == _REPOSITORY_ROOT
        assert service["image"] == "reelio:local"

    assert migrate["command"] == ["alembic", "upgrade", "head"]
    assert migrate["restart"] == "no"
    _depends_on(migrate, "postgres", "service_healthy")
    migrate_environment = migrate["environment"]
    assert isinstance(migrate_environment, dict)
    assert set(migrate_environment) == {
        "REELIO_COORDINATION_DATABASE_URL",
        "REELIO_COORDINATION_HMAC_SECRET",
    }
    assert migrate_environment["REELIO_COORDINATION_DATABASE_URL"] == (
        "postgresql+asyncpg://reelio:url-safe-postgres-password@postgres:5432/reelio"
    )

    assert api["restart"] == "unless-stopped"
    _depends_on(api, "postgres", "service_healthy")
    _depends_on(api, "redis", "service_healthy")
    _depends_on(api, "migrate", "service_completed_successfully")
    _assert_loopback_api_port(api, 8181)
    api_networks = api["networks"]
    assert isinstance(api_networks, dict)
    assert set(api_networks) == {"default", "reelio-textify"}
    api_environment = api["environment"]
    assert isinstance(api_environment, dict)
    assert api_environment["REELIO_ENVIRONMENT"] == "local"
    assert api_environment["REELIO_CACHE_ENABLED"] == "true"
    assert api_environment["REELIO_CACHE_REDIS_URL"] == "redis://redis:6379/0"
    assert api_environment["REELIO_TEXTIFY_ORIGIN"] == "http://textify:8182"

    assert postgres["image"] == "postgres:17"
    assert postgres["restart"] == "unless-stopped"
    assert not postgres.get("ports")
    _assert_postgres_data_mount(postgres)
    postgres_healthcheck = _healthcheck(postgres)
    assert postgres_healthcheck["test"] == [
        "CMD-SHELL",
        "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}",
    ]
    assert postgres_healthcheck["interval"] == "10s"
    assert postgres_healthcheck["timeout"] == "5s"
    assert postgres_healthcheck["retries"] == 5
    assert postgres_healthcheck["start_period"] == "30s"

    assert redis["restart"] == "unless-stopped"
    assert not redis.get("ports")
    assert not redis.get("volumes")
    assert "mem_limit" not in redis
    assert redis["command"] == [
        "redis-server",
        "--maxmemory",
        "512mb",
        "--maxmemory-policy",
        "allkeys-lfu",
        "--save",
        "",
        "--appendonly",
        "no",
    ]
    redis_healthcheck = _healthcheck(redis)
    assert redis_healthcheck["test"] == ["CMD", "redis-cli", "ping"]
    assert redis_healthcheck["interval"] == "10s"
    assert redis_healthcheck["timeout"] == "5s"
    assert redis_healthcheck["retries"] == 5
    assert redis_healthcheck["start_period"] == "5s"
    for private_service in (postgres, redis, migrate):
        private_networks = private_service.get("networks", {})
        assert isinstance(private_networks, dict)
        assert "reelio-textify" not in private_networks

    networks = rendered["networks"]
    assert isinstance(networks, dict)
    textify_network = networks["reelio-textify"]
    assert isinstance(textify_network, dict)
    assert textify_network["external"] is True
    assert textify_network["name"] == "reelio-textify"

    volumes = rendered["volumes"]
    assert isinstance(volumes, dict)
    postgres_data = volumes["postgres-data"]
    assert isinstance(postgres_data, dict)
    assert postgres_data["name"] == "reelio-postgres-data"


def test_compose_renders_api_host_port_and_image_tag_overrides() -> None:
    """Render loopback publication and a retained image tag selected for rollback."""
    rendered = _render_compose(
        {
            "REELIO_API_HOST_PORT": "18181",
            "REELIO_IMAGE_TAG": "previous-release",
        }
    )

    _assert_loopback_api_port(_service(rendered, "api"), 18181)
    assert _service(rendered, "api")["image"] == "reelio:previous-release"
    assert _service(rendered, "migrate")["image"] == "reelio:previous-release"


@pytest.mark.parametrize("missing_name", _REQUIRED_ENVIRONMENT)
def test_compose_rejects_each_unconditional_required_variable(
    missing_name: str,
) -> None:
    """Reject Compose rendering when one unconditional variable is absent."""
    environment_values = {
        name: value for name, value in _REQUIRED_ENVIRONMENT.items() if name != missing_name
    }

    completed = _compose_config(environment_values)

    assert completed.returncode != 0
    assert missing_name in completed.stderr
