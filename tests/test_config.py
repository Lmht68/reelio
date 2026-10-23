"""Behavioral tests for environment-backed configuration."""

from collections.abc import Callable
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from reelio.cache import CacheConfig
from reelio.config import AppConfig, Environment
from reelio.extraction.coordination import CoordinationConfig
from reelio.extraction.services.enrichment.config import OpenLibraryConfig, TMDBConfig
from reelio.extraction.services.interpretation.config import (
    DeepSeekConfig,
    InterpretationConfig,
    LLMProvider,
    LLMProviderSelectionConfig,
    OpenAIConfig,
)
from reelio.extraction.services.textify import TextifyConfig


def _without_dotenv[SettingsType: BaseSettings](
    settings_type: type[SettingsType],
    **values: object,
) -> SettingsType:
    """Instantiate a settings class without reading the repository .env file."""
    constructor = cast(Callable[..., SettingsType], settings_type)
    return constructor(_env_file=None, **values)


def test_tmdb_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the TMDB bearer token when settings are instantiated."""
    monkeypatch.delenv("REELIO_TMDB_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_TMDB_API_KEY"):
        _without_dotenv(TMDBConfig)


def test_open_library_requires_nonblank_safe_contact_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require a safe operational contact and credential-free HTTP endpoint."""
    monkeypatch.delenv("REELIO_OPEN_LIBRARY_CONTACT_EMAIL", raising=False)

    with pytest.raises(ValidationError, match="REELIO_OPEN_LIBRARY_CONTACT_EMAIL"):
        _without_dotenv(OpenLibraryConfig)
    for contact_email in (
        "",
        "  ",
        "catalog@example.invalid\u0000",
        "catalog\u0085@example.invalid",
    ):
        with pytest.raises(ValidationError):
            _without_dotenv(OpenLibraryConfig, contact_email=contact_email)
    for base_url in (
        "ftp://openlibrary.org",
        "https://user:password@openlibrary.org",
        "https://openlibrary.org/search?title=Dune",
        "https://openlibrary.org/#fragment",
    ):
        with pytest.raises(ValidationError):
            _without_dotenv(
                OpenLibraryConfig,
                contact_email="catalog@example.invalid",
                base_url=base_url,
            )


def test_open_library_configuration_defaults_and_valid_overrides() -> None:
    """Provide the exact conservative Open Library request defaults."""
    defaults = _without_dotenv(
        OpenLibraryConfig,
        contact_email="  catalog@example.invalid  ",
    )
    overrides = _without_dotenv(
        OpenLibraryConfig,
        contact_email="catalog@example.invalid",
        base_url="https://catalog.example/",
        request_timeout_seconds=4.5,
        requests_per_second=2.5,
    )

    assert defaults.contact_email == "catalog@example.invalid"
    assert defaults.base_url == "https://openlibrary.org"
    assert defaults.request_timeout_seconds == 10.0
    assert defaults.requests_per_second == 3.0
    assert overrides.base_url == "https://catalog.example"
    assert overrides.request_timeout_seconds == 4.5
    assert overrides.requests_per_second == 2.5


@pytest.mark.parametrize("requests_per_second", [0, -1, 3.1])
def test_open_library_rejects_invalid_request_rates(requests_per_second: float) -> None:
    """Reject nonpositive rates and values above the Open Library ceiling."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            OpenLibraryConfig,
            contact_email="catalog@example.invalid",
            requests_per_second=requests_per_second,
        )


def test_provider_selection_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject startup configuration that omits the LLM provider selection."""
    monkeypatch.delenv("REELIO_LLM_PROVIDER", raising=False)

    with pytest.raises(ValidationError, match="REELIO_LLM_PROVIDER"):
        _without_dotenv(LLMProviderSelectionConfig)


@pytest.mark.parametrize(
    "value",
    ["", " openai", "openai ", "OpenAI", "DEEPSEEK", "unsupported"],
)
def test_provider_selection_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Accept only exact lowercase configured provider identifiers."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", value)

    with pytest.raises(ValidationError):
        _without_dotenv(LLMProviderSelectionConfig)


@pytest.mark.parametrize(
    ("value", "expected_provider"),
    [("openai", LLMProvider.OPENAI), ("deepseek", LLMProvider.DEEPSEEK)],
)
def test_provider_selection_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected_provider: LLMProvider,
) -> None:
    """Parse each supported provider identity without normalization."""
    monkeypatch.setenv("REELIO_LLM_PROVIDER", value)

    settings = _without_dotenv(LLMProviderSelectionConfig)

    assert settings.llm_provider is expected_provider


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require OpenAI credentials when OpenAI configuration is selected."""
    monkeypatch.delenv("REELIO_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_OPENAI_API_KEY"):
        _without_dotenv(OpenAIConfig)


def test_deepseek_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require DeepSeek credentials when DeepSeek configuration is selected."""
    monkeypatch.delenv("REELIO_DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="REELIO_DEEPSEEK_API_KEY"):
        _without_dotenv(DeepSeekConfig)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_openai_rejects_blank_api_key(api_key: str) -> None:
    """Reject empty and whitespace-only OpenAI credentials."""
    with pytest.raises(ValidationError, match="must not be blank"):
        _without_dotenv(OpenAIConfig, api_key=api_key)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_deepseek_rejects_blank_api_key(api_key: str) -> None:
    """Reject empty and whitespace-only DeepSeek credentials."""
    with pytest.raises(ValidationError, match="must not be blank"):
        _without_dotenv(DeepSeekConfig, api_key=api_key)


def test_openai_rejects_unsupported_reasoning_effort() -> None:
    """Reject unsupported OpenAI reasoning effort during startup validation."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            OpenAIConfig,
            api_key="openai-key",
            reasoning_effort="unsupported",
        )


def test_openai_configuration_ignores_inactive_deepseek_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate OpenAI without reading an invalid inactive DeepSeek configuration."""
    monkeypatch.setenv("REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.delenv("REELIO_OPENAI_MODEL", raising=False)

    settings = _without_dotenv(OpenAIConfig, api_key="openai-key")

    assert settings.model == "gpt-5-nano"


def test_deepseek_configuration_ignores_inactive_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate DeepSeek without reading an invalid inactive OpenAI configuration."""
    monkeypatch.setenv("REELIO_OPENAI_MAX_RETRIES", "not-an-integer")

    settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")

    assert settings.model == "deepseek-flash"


def test_configuration_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the stable defaults from clean process settings."""
    for variable in (
        "REELIO_ENVIRONMENT",
        "REELIO_LOG_LEVEL",
        "REELIO_OPENAI_MODEL",
        "REELIO_OPENAI_REASONING_EFFORT",
        "REELIO_OPENAI_REQUEST_TIMEOUT_SECONDS",
        "REELIO_OPENAI_MAX_OUTPUT_TOKENS",
        "REELIO_OPENAI_MAX_RETRIES",
        "REELIO_DEEPSEEK_BASE_URL",
        "REELIO_DEEPSEEK_MODEL",
        "REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS",
        "REELIO_DEEPSEEK_TEMPERATURE",
        "REELIO_DEEPSEEK_MAX_OUTPUT_TOKENS",
        "REELIO_DEEPSEEK_MAX_RETRIES",
        "REELIO_MAX_SOURCE_TITLE_CHARS",
        "REELIO_MAX_DESCRIPTION_CHARS",
        "REELIO_MAX_TRANSCRIPT_LANGUAGE_CHARS",
        "REELIO_MAX_TRANSCRIPT_CHARS",
        "REELIO_TMDB_BASE_URL",
        "REELIO_TMDB_IMAGE_BASE_URL",
        "REELIO_TMDB_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)

    app_settings = _without_dotenv(AppConfig)
    interpretation_settings = _without_dotenv(InterpretationConfig)
    openai_settings = _without_dotenv(OpenAIConfig, api_key="openai-key")
    deepseek_settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")
    tmdb_settings = _without_dotenv(TMDBConfig, api_key="tmdb-key")

    assert app_settings.environment is Environment.LOCAL
    assert app_settings.log_level == "INFO"
    assert openai_settings.model == "gpt-5-nano"
    assert openai_settings.reasoning_effort == "low"
    assert openai_settings.request_timeout_seconds == 60.0
    assert openai_settings.max_output_tokens == 8_192
    assert openai_settings.max_retries == 2
    assert deepseek_settings.base_url == "https://api.deepseek.com"
    assert deepseek_settings.model == "deepseek-flash"
    assert deepseek_settings.request_timeout_seconds == 60.0
    assert deepseek_settings.temperature == 0.0
    assert deepseek_settings.max_output_tokens == 8_192
    assert deepseek_settings.max_retries == 2
    assert interpretation_settings.max_source_title_chars == 500
    assert interpretation_settings.max_description_chars == 2_000
    assert interpretation_settings.max_transcript_language_chars == 64
    assert interpretation_settings.max_transcript_chars == 100_000
    assert tmdb_settings.base_url == "https://api.themoviedb.org/3"
    assert tmdb_settings.image_base_url == "https://image.tmdb.org/t/p/w500"
    assert tmdb_settings.request_timeout_seconds == 10.0


def test_tmdb_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse TMDB endpoint, image, and timeout settings."""
    monkeypatch.setenv("REELIO_TMDB_BASE_URL", "https://tmdb.test/3")
    monkeypatch.setenv("REELIO_TMDB_IMAGE_BASE_URL", "https://images.test/w342")
    monkeypatch.setenv("REELIO_TMDB_REQUEST_TIMEOUT_SECONDS", "4.5")

    settings = _without_dotenv(TMDBConfig, api_key="tmdb-key")

    assert settings.base_url == "https://tmdb.test/3"
    assert settings.image_base_url == "https://images.test/w342"
    assert settings.request_timeout_seconds == 4.5


def test_openai_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse OpenAI request options from provider-specific environment variables."""
    monkeypatch.setenv("REELIO_OPENAI_MODEL", "gpt-5.1")
    monkeypatch.setenv("REELIO_OPENAI_REASONING_EFFORT", "medium")
    monkeypatch.setenv("REELIO_OPENAI_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REELIO_OPENAI_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("REELIO_OPENAI_MAX_RETRIES", "3")

    settings = _without_dotenv(OpenAIConfig, api_key="openai-key")

    assert settings.model == "gpt-5.1"
    assert settings.reasoning_effort == "medium"
    assert settings.request_timeout_seconds == 12.5
    assert settings.max_output_tokens == 4_096
    assert settings.max_retries == 3


def test_deepseek_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse DeepSeek request options from provider-specific environment variables."""
    monkeypatch.setenv("REELIO_DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REELIO_DEEPSEEK_TEMPERATURE", "0.1")
    monkeypatch.setenv("REELIO_DEEPSEEK_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("REELIO_DEEPSEEK_MAX_RETRIES", "3")

    settings = _without_dotenv(DeepSeekConfig, api_key="deepseek-key")

    assert settings.request_timeout_seconds == 12.5
    assert settings.temperature == 0.1
    assert settings.max_output_tokens == 4_096
    assert settings.max_retries == 3


def test_interpretation_material_limit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse Interpretation Material limits from environment variables."""
    monkeypatch.setenv("REELIO_MAX_SOURCE_TITLE_CHARS", "100")
    monkeypatch.setenv("REELIO_MAX_DESCRIPTION_CHARS", "1000")
    monkeypatch.setenv("REELIO_MAX_TRANSCRIPT_LANGUAGE_CHARS", "20")
    monkeypatch.setenv("REELIO_MAX_TRANSCRIPT_CHARS", "50000")

    settings = _without_dotenv(InterpretationConfig)

    assert settings.max_source_title_chars == 100
    assert settings.max_description_chars == 1_000
    assert settings.max_transcript_language_chars == 20
    assert settings.max_transcript_chars == 50_000


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject log levels outside the supported literal values."""
    monkeypatch.setenv("REELIO_LOG_LEVEL", "CHATTY")

    with pytest.raises(ValidationError):
        _without_dotenv(AppConfig)


def test_cache_configuration_defaults_to_disabled_without_redis_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore absent Redis settings while shared caching is disabled."""
    for variable in (
        "REELIO_CACHE_ENABLED",
        "REELIO_CACHE_REDIS_URL",
        "REELIO_CACHE_KEY_SECRET",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = _without_dotenv(CacheConfig)

    assert settings.enabled is False
    assert settings.redis_url is None
    assert settings.key_secret is None
    assert settings.namespace == "reelio:local"


@pytest.mark.parametrize(
    ("redis_url", "key_secret"),
    [
        (None, "cache-key"),
        ("", "cache-key"),
        ("redis://localhost", None),
        ("redis://localhost", "   "),
    ],
)
def test_enabled_cache_requires_nonblank_endpoint_and_key_secret(
    redis_url: str | None,
    key_secret: str | None,
) -> None:
    """Reject missing or blank enabled-cache credentials."""
    with pytest.raises(ValidationError, match="must not be blank"):
        _without_dotenv(
            CacheConfig,
            enabled=True,
            redis_url=redis_url,
            key_secret=key_secret,
        )


@pytest.mark.parametrize(
    "redis_url",
    [
        "https://cache.example",
        "redis://",
        "rediss://cache.example#fragment",
        "rediss://cache.example:0",
        "rediss://cache.example:99999",
    ],
)
def test_enabled_cache_rejects_invalid_redis_endpoint_syntax(redis_url: str) -> None:
    """Allow only absolute Redis URLs with valid host, port, and no fragment."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            CacheConfig,
            enabled=True,
            redis_url=redis_url,
            key_secret="cache-key",
        )


def test_enabled_cache_allows_local_plaintext_redis_endpoints() -> None:
    """Permit loopback and Compose service plaintext endpoints in local development."""
    loopback_settings = _without_dotenv(
        CacheConfig,
        enabled=True,
        environment=Environment.LOCAL,
        redis_url="redis://:password@[::1]:6379/0",
        key_secret="cache-key",
    )
    compose_settings = _without_dotenv(
        CacheConfig,
        enabled=True,
        environment=Environment.LOCAL,
        redis_url="redis://redis:6379/0",
        key_secret="cache-key",
    )

    assert loopback_settings.namespace == "reelio:local"
    assert compose_settings.namespace == "reelio:local"


@pytest.mark.parametrize(
    ("environment", "redis_url"),
    [
        (Environment.LOCAL, "redis://cache.example:6379"),
        (Environment.STAGING, "redis://localhost:6379"),
        (Environment.PRODUCTION, "redis://127.0.0.1:6379"),
        (Environment.STAGING, "rediss://cache.example?ssl_cert_reqs=none"),
        (Environment.PRODUCTION, "rediss://cache.example?ssl_check_hostname=false"),
    ],
)
def test_enabled_cache_requires_secure_transport_outside_local_loopback(
    environment: Environment,
    redis_url: str,
) -> None:
    """Reject remote plaintext and TLS query options that weaken secure defaults."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            CacheConfig,
            enabled=True,
            environment=environment,
            redis_url=redis_url,
            key_secret="cache-key",
        )


def test_enabled_cache_accepts_secure_managed_endpoint_options() -> None:
    """Retain secure TLS defaults while allowing managed endpoint CA configuration."""
    settings = _without_dotenv(
        CacheConfig,
        enabled=True,
        environment=Environment.PRODUCTION,
        redis_url=("rediss://:password@cache.example:6380/0?ssl_ca_path=%2Fetc%2Fssl%2Fcerts"),
        key_secret="cache-key",
    )

    assert settings.namespace == "reelio:production"


def test_coordination_requires_nonblank_database_url_and_hmac_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing or blank coordination credentials at startup."""
    monkeypatch.delenv("REELIO_COORDINATION_DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        _without_dotenv(CoordinationConfig, hmac_secret="coordination-secret")

    with pytest.raises(ValidationError):
        _without_dotenv(
            CoordinationConfig,
            database_url="postgresql+asyncpg://reelio:password@localhost/reelio",
            hmac_secret="   ",
        )


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://reelio:password@localhost/reelio",
        "postgresql+psycopg://reelio:password@localhost/reelio",
        "mysql+asyncmy://reelio:password@localhost/reelio",
        "postgresql+asyncpg:///reelio",
        "postgresql+asyncpg://reelio:password@localhost",
    ],
)
def test_coordination_rejects_non_asyncpg_or_incomplete_database_urls(
    database_url: str,
) -> None:
    """Require a complete asynchronous PostgreSQL coordination endpoint."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            CoordinationConfig,
            database_url=database_url,
            hmac_secret="coordination-secret",
        )


@pytest.mark.parametrize("retention_seconds", [0, -1, 86_401])
def test_coordination_rejects_retention_outside_one_day(retention_seconds: int) -> None:
    """Reject a durable coordination retention period outside one day."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            CoordinationConfig,
            database_url="postgresql+asyncpg://reelio:password@localhost/reelio",
            hmac_secret="coordination-secret",
            retention_seconds=retention_seconds,
        )


def test_coordination_uses_the_documented_retention_default() -> None:
    """Retain durable coordination for one day unless configured otherwise."""
    settings = _without_dotenv(
        CoordinationConfig,
        database_url="postgresql+asyncpg://reelio:password@localhost/reelio",
        hmac_secret="coordination-secret",
    )

    assert settings.retention_seconds == 86_400


@pytest.mark.parametrize(
    "origin",
    [
        "textify:8182",
        "ftp://textify:8182",
        "http://user:password@textify:8182",
        "http://textify:8182/api/v1",
        "http://textify:8182?debug=true",
        "http://textify:8182#fragment",
        "http://textify:0",
        "https://textify:99999",
    ],
)
def test_textify_rejects_non_origin_endpoints(origin: str) -> None:
    """Accept only credential-free HTTP origins for Textify requests."""
    with pytest.raises(ValidationError):
        _without_dotenv(TextifyConfig, origin=origin)


@pytest.mark.parametrize("request_timeout_seconds", [0, -1])
def test_textify_rejects_nonpositive_attempt_timeouts(
    request_timeout_seconds: float,
) -> None:
    """Require a positive per-request Textify timeout."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            TextifyConfig,
            request_timeout_seconds=request_timeout_seconds,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "admission_capacity_retry_budget_seconds",
        "transport_retry_budget_seconds",
    ],
)
@pytest.mark.parametrize("budget_seconds", [0, -1, float("inf"), "not-a-number"])
def test_textify_rejects_nonpositive_or_nonfinite_retry_budgets(
    field_name: str,
    budget_seconds: object,
) -> None:
    """Require finite positive Textify retry budgets."""
    with pytest.raises(ValidationError):
        _without_dotenv(TextifyConfig, **{field_name: budget_seconds})


def test_textify_uses_documented_origin_timeout_and_retry_budget_defaults() -> None:
    """Use the private Textify HTTP endpoint with bounded retry defaults."""
    settings = _without_dotenv(TextifyConfig)

    assert settings.origin == "http://textify:8182"
    assert settings.request_timeout_seconds == 5.0
    assert settings.admission_capacity_retry_budget_seconds == 10.0
    assert settings.transport_retry_budget_seconds == 30.0


@pytest.mark.parametrize("result_retention_seconds", [0, -1])
def test_textify_rejects_nonpositive_result_retention(
    result_retention_seconds: int,
) -> None:
    """Require a positive exact retention duration for successful Textify results."""
    with pytest.raises(ValidationError):
        _without_dotenv(
            TextifyConfig,
            result_retention_seconds=result_retention_seconds,
        )


def test_textify_uses_thirty_day_result_retention_by_default() -> None:
    """Retain successful Textify results for thirty days unless configured otherwise."""
    settings = _without_dotenv(TextifyConfig)

    assert settings.result_retention_seconds == 2_592_000
