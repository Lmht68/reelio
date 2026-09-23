"""Environment-backed configuration for the optional shared cache."""

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reelio.config import Environment


class CacheConfig(BaseSettings):
    """Validate the explicitly enabled Redis-compatible shared-cache configuration.

    Redis settings are deliberately ignored unless caching is enabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="REELIO_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    enabled: bool = Field(default=False, validation_alias="REELIO_CACHE_ENABLED")
    redis_url: SecretStr | None = Field(
        default=None,
        validation_alias="REELIO_CACHE_REDIS_URL",
    )
    key_secret: SecretStr | None = Field(
        default=None,
        validation_alias="REELIO_CACHE_KEY_SECRET",
    )
    environment: Environment = Field(
        default=Environment.LOCAL,
        validation_alias="REELIO_ENVIRONMENT",
    )

    @property
    def namespace(self) -> str:
        """Return the environment-derived fixed Redis key namespace."""
        return f"reelio:{self.environment.value}"

    @model_validator(mode="after")
    def _validate_enabled_settings(self) -> "CacheConfig":
        """Require secure endpoint and HMAC secret only for enabled caching.

        Raises:
            ValueError: If enabled settings are incomplete or insecure.
        """
        if not self.enabled:
            return self

        redis_url = _required_secret(self.redis_url, "Redis URL")
        _required_secret(self.key_secret, "cache key secret")
        _validate_redis_url(redis_url, self.environment)
        return self


def _required_secret(secret: SecretStr | None, name: str) -> str:
    """Return a nonblank secret value.

    Args:
        secret: Optional setting supplied by Pydantic.
        name: Safe configuration field name for validation errors.

    Returns:
        The original nonblank secret text.

    Raises:
        ValueError: If the setting is absent or whitespace only.
    """
    if secret is None or not secret.get_secret_value().strip():
        raise ValueError(f"{name} must not be blank when caching is enabled")
    return secret.get_secret_value()


def _validate_redis_url(redis_url: str, environment: Environment) -> None:
    """Validate Redis endpoint syntax and environment-specific transport safety.

    Args:
        redis_url: Secret Redis connection endpoint.
        environment: Application deployment environment.

    Raises:
        ValueError: If the endpoint is invalid or weakens required transport security.
    """
    endpoint = urlparse(redis_url)
    if endpoint.scheme not in {"redis", "rediss"} or endpoint.fragment:
        raise ValueError("Redis URL must use redis or rediss without a fragment")
    if not endpoint.netloc or endpoint.hostname is None:
        raise ValueError("Redis URL must be an absolute endpoint with a host")

    try:
        port = endpoint.port
    except ValueError as exc:
        raise ValueError("Redis URL must contain a valid port") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("Redis URL must contain a valid port")

    hostname = endpoint.hostname.casefold()
    if endpoint.scheme == "redis":
        is_local_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if environment is not Environment.LOCAL or not is_local_loopback:
            raise ValueError("redis URLs are allowed only for local loopback endpoints")
        return

    _validate_tls_query_overrides(parse_qsl(endpoint.query, keep_blank_values=True))


def _validate_tls_query_overrides(query: Iterable[tuple[str, str]]) -> None:
    """Reject URL options that weaken redis-py's secure TLS defaults.

    Args:
        query: Parsed Redis URL query parameters.

    Raises:
        ValueError: If certificate validation or hostname checking is weakened.
    """
    for name, value in query:
        if name == "ssl_cert_reqs" and value.casefold() != "required":
            raise ValueError("Redis TLS certificate verification must be required")
        if name == "ssl_check_hostname" and value.casefold() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise ValueError("Redis TLS hostname checking must be enabled")
