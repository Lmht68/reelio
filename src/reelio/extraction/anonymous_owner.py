"""Cookie-backed anonymous ownership for Source Extraction jobs."""

import base64
import binascii
import hmac
import secrets
from dataclasses import dataclass

from fastapi.responses import Response
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reelio.config import Environment

ANONYMOUS_OWNER_COOKIE_NAME = "reelio_anonymous_owner"
_COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
_TOKEN_BYTES = 32


class AnonymousOwnerConfig(BaseSettings):
    """Validate the independently rotatable anonymous-job ownership secret."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_ANONYMOUS_OWNER_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    hmac_secret: SecretStr
    environment: Environment = Field(
        default=Environment.LOCAL,
        validation_alias="REELIO_ENVIRONMENT",
    )

    @field_validator("hmac_secret")
    @classmethod
    def _validate_hmac_secret(cls, hmac_secret: SecretStr) -> SecretStr:
        """Require one nonblank HMAC secret.

        Args:
            hmac_secret: Configured anonymous-owner HMAC secret.

        Returns:
            The unmodified validated secret.

        Raises:
            ValueError: If the secret is blank.
        """
        if not hmac_secret.get_secret_value().strip():
            raise ValueError("anonymous owner HMAC secret must not be blank")
        return hmac_secret


@dataclass(frozen=True, slots=True)
class ResolvedAnonymousOwner:
    """Contain an anonymous owner's storage digest and response-only cookie value."""

    _token: str
    digest: bytes


class AnonymousOwnerManager:
    """Resolve anonymous owner cookies without persisting bearer credentials."""

    def __init__(self, config: AnonymousOwnerConfig) -> None:
        """Initialize owner resolution from validated deployment configuration.

        Args:
            config: Validated anonymous-owner configuration.
        """
        self._secret = config.hmac_secret.get_secret_value().encode()
        self._environment = config.environment

    def resolve_existing(self, cookie_value: str | None) -> ResolvedAnonymousOwner | None:
        """Resolve a valid owner cookie, rejecting missing and malformed values.

        Args:
            cookie_value: Browser-provided owner cookie value.

        Returns:
            The resolved owner, or None when the cookie is absent or invalid.
        """
        if cookie_value is None or not _is_canonical_token(cookie_value):
            return None
        return self._resolved_owner(cookie_value)

    def resolve_or_issue(self, cookie_value: str | None) -> ResolvedAnonymousOwner:
        """Resolve a valid owner cookie or create a new owner token.

        Args:
            cookie_value: Browser-provided owner cookie value.

        Returns:
            A resolved existing or newly generated owner.
        """
        owner = self.resolve_existing(cookie_value)
        if owner is not None:
            return owner
        return self._resolved_owner(secrets.token_urlsafe(_TOKEN_BYTES))

    def set_cookie(self, response: Response, owner: ResolvedAnonymousOwner) -> None:
        """Attach the owner's HttpOnly host-only cookie to one response.

        Args:
            response: Public API response receiving the cookie.
            owner: Resolved owner whose token is returned only in this cookie.
        """
        response.set_cookie(
            key=ANONYMOUS_OWNER_COOKIE_NAME,
            value=owner._token,
            max_age=_COOKIE_MAX_AGE_SECONDS,
            path="/",
            secure=self._environment is not Environment.LOCAL,
            httponly=True,
            samesite="lax",
        )

    def _resolved_owner(self, token: str) -> ResolvedAnonymousOwner:
        """Build one storage-safe owner identity from a canonical bearer token."""
        return ResolvedAnonymousOwner(
            _token=token,
            digest=hmac.digest(self._secret, token.encode("ascii"), "sha256"),
        )


def _is_canonical_token(token: str) -> bool:
    """Return whether a token canonically encodes exactly 32 random bytes."""
    try:
        token.encode("ascii")
        decoded = base64.b64decode(
            token + "=" * (-len(token) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError, binascii.Error):
        return False
    return (
        len(decoded) == _TOKEN_BYTES
        and base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") == token
    )
