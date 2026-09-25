"""Operator-only command-line entry point for provider cache purges."""

import argparse
import asyncio
import logging
from collections.abc import Sequence

from pydantic import ValidationError
from redis.exceptions import RedisError

from reelio.cache.config import CacheConfig
from reelio.cache.purge import (
    ProviderPurgeError,
    PurgeProvider,
    RedisProviderPurger,
    create_provider_purger,
)
from reelio.config import Environment
from reelio.logging import configure_logging

logger = logging.getLogger(__name__)


class _PurgeArguments(argparse.Namespace):
    """Contain validated explicit operator input for one purge invocation."""

    provider: str
    environment: str


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fail-closed provider cache purge from explicit CLI arguments.

    Args:
        argv: Optional argument sequence excluding the executable name.

    Returns:
        Zero after every selected category completes, or one for a sanitized
        configuration or Redis failure.

    Raises:
        SystemExit: If required command-line arguments are absent or invalid.
    """
    arguments = _argument_parser().parse_args(argv, namespace=_PurgeArguments())
    provider = PurgeProvider(arguments.provider)
    environment = Environment(arguments.environment)
    configure_logging("INFO")

    try:
        settings = CacheConfig(enabled=True, environment=environment)
        asyncio.run(_run_purge(settings, provider))
    except ValidationError:
        _log_configuration_failure(provider, environment)
        return 1
    except ProviderPurgeError as error:
        _log_purge_failure(
            error.provider,
            error.environment,
            error.namespace_category.value,
            error.deleted_count,
        )
        return 1
    except (OSError, RedisError, TimeoutError, ValueError):
        _log_purge_failure(provider, environment, "client", 0)
        return 1
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Create the parser that gates settings and Redis construction."""
    parser = argparse.ArgumentParser(
        description="Purge one provider scope from the selected Redis cache environment."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[provider.value for provider in PurgeProvider],
        help="Provider cache scope to purge.",
    )
    parser.add_argument(
        "--environment",
        required=True,
        choices=[environment.value for environment in Environment],
        help="Deployment environment whose cache namespace is purged.",
    )
    return parser


async def _run_purge(settings: CacheConfig, provider: PurgeProvider) -> None:
    """Create, execute, and close one provider purger in a single event loop."""
    purger: RedisProviderPurger | None = None
    try:
        purger = create_provider_purger(settings)
        await purger.purge(provider)
    finally:
        if purger is not None:
            await purger.aclose()


def _log_configuration_failure(provider: PurgeProvider, environment: Environment) -> None:
    """Emit a fixed configuration error without validation details."""
    logger.error(
        "Provider cache purge configuration is invalid. Set cache credentials and retry.",
        extra={"provider": provider.value, "environment": environment.value},
    )


def _log_purge_failure(
    provider: PurgeProvider,
    environment: Environment,
    namespace_category: str,
    deleted_count: int,
) -> None:
    """Emit a fixed Redis failure without endpoint, exception, or key material."""
    logger.error(
        "Provider cache purge did not complete. Check Redis availability and retry.",
        extra={
            "provider": provider.value,
            "environment": environment.value,
            "namespace_category": namespace_category,
            "deleted_count": deleted_count,
        },
    )
