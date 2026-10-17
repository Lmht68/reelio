"""Textify HTTP adapter and settings."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reelio.extraction.coordination import (
    CoordinationConfig,
    CoordinationError,
    CoordinationUnavailableError,
    TextifyCoordination,
    TextifyCoordinationRepository,
    create_postgres_textify_coordination,
)
from reelio.extraction.exceptions import (
    TranscriptionError,
    TranscriptionUnavailableError,
)
from reelio.extraction.types import (
    Platform,
    Source,
    Transcript,
    TranscriptMethod,
    normalize_transcript_segments,
)

_TRANSCRIPTION_JOBS_PATH = "/api/v1/transcription-jobs"
_TRANSCRIPTION_FAILURE_MESSAGE = "Transcript is unavailable for this video."
_TRANSCRIPTION_UNAVAILABLE_MESSAGE = "Transcription is temporarily unavailable."


class TextifyConfig(BaseSettings):
    """Validate the private HTTP origin and timeout used for Textify requests."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_TEXTIFY_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        validate_default=True,
    )

    origin: str = "http://textify:8182"
    request_timeout_seconds: float = Field(default=5.0, gt=0)

    @field_validator("origin")
    @classmethod
    def _validate_origin(cls, origin: str) -> str:
        """Require one credential-free HTTP origin with no path or suffix.

        Args:
            origin: Configured Textify endpoint.

        Returns:
            The unmodified valid origin.

        Raises:
            ValueError: If the endpoint is not an absolute HTTP(S) origin.
        """
        endpoint = urlparse(origin)
        if (
            endpoint.scheme not in {"http", "https"}
            or endpoint.hostname is None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.params
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "Textify origin must be an absolute credential-free http or https origin"
            )
        try:
            port = endpoint.port
        except ValueError as exc:
            raise ValueError("Textify origin must contain a valid port") from exc
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError("Textify origin must contain a valid port")
        return origin


class _TextifyModel(BaseModel):
    """Accept only known type-correct Textify fields while ignoring additions."""

    model_config = ConfigDict(extra="ignore", strict=True)


class _TextifyLinks(_TextifyModel):
    """Read the polling bearer capability from a Textify response."""

    self: str


class _QueuedJobResponse(_TextifyModel):
    """Read a queued admission response."""

    status: Literal["queued"]
    links: _TextifyLinks


class _JobStatusResponse(_TextifyModel):
    """Read the common status and capability from any polling response."""

    status: Literal["queued", "processing", "finished"]
    links: _TextifyLinks


class _FinishedJobResponse(_TextifyModel):
    """Read one terminal Textify job outcome."""

    status: Literal["finished"]
    outcome: Literal["succeeded", "failed", "cancelled"]
    links: _TextifyLinks


class _TextifyError(_TextifyModel):
    """Read one Textify error without exposing it to Reelio callers."""

    code: str
    message: str


class _TextifyErrorResponse(_TextifyModel):
    """Read a Textify error envelope."""

    error: _TextifyError


class _TextifySource(_TextifyModel):
    """Read the complete Source projection required by Reelio."""

    platform: str
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


class _TextifyTranscript(_TextifyModel):
    """Read the compact Transcript projection required by Reelio."""

    method: Literal["youtube_captions", "faster_whisper"]
    language: str
    text: str


class _TextifyResult(_TextifyModel):
    """Read a successful Source-and-Transcript result."""

    source: _TextifySource
    transcript: _TextifyTranscript


class _SucceededJobResponse(_TextifyModel):
    """Read a completed successful Textify job."""

    status: Literal["finished"]
    outcome: Literal["succeeded"]
    links: _TextifyLinks
    result: _TextifyResult


class TextifySourceTranscriptProvider:
    """Acquire Reelio Source and Transcript values through durable Textify jobs."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        coordination_repository: TextifyCoordinationRepository,
        sleeper: Callable[[int], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize the provider with owned HTTP and coordination resources.

        Args:
            client: No-redirect Textify HTTP client resolved against the configured origin.
            coordination_repository: Durable admission state owner.
            sleeper: Awaitable delay used to honor active Textify polling instructions.
        """
        self._client = client
        self._coordination_repository = coordination_repository
        self._sleeper = sleeper

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        """Acquire one Reelio Source and Transcript through Textify.

        Args:
            submitted_url: Exact public Source URL supplied by the caller.

        Returns:
            Reelio-owned normalized Source and Transcript values.

        Raises:
            TranscriptionError: If Textify does not follow its successful job contract.
            TranscriptionUnavailableError: If durable coordination is unavailable or stale.
        """
        try:
            coordination = await self._coordination_repository.get_or_create(submitted_url)
        except CoordinationUnavailableError:
            raise _transcription_unavailable() from None
        capability_link = coordination.capability_link
        if capability_link is None:
            admission = await self._submit(submitted_url, str(coordination.idempotency_key))
            capability_link = admission.links.self
            location = admission.response.headers.get("Location")
            if location != capability_link:
                raise _transcription_failure()

            try:
                accepted = await self._coordination_repository.attach_capability(
                    coordination,
                    capability_link,
                )
            except CoordinationUnavailableError:
                raise _transcription_unavailable() from None
            except CoordinationError:
                raise _transcription_failure() from None
            coordination = accepted
            stored_capability_link = coordination.capability_link
            if stored_capability_link is None or stored_capability_link != capability_link:
                raise _transcription_failure()
            capability_link = stored_capability_link
            retry_after = _retry_after(admission.response)
        else:
            retry_after = 0
        while True:
            await self._sleeper(retry_after)
            response = await self._get(capability_link)
            if response.status_code == httpx.codes.NOT_FOUND:
                textify_error = _parse_error_response(response)
                if textify_error.error.code == "job_not_found":
                    await self._retire(coordination)
                    raise _transcription_unavailable()
                raise _transcription_failure()
            job_status = _parse_job_status(response)
            if job_status.links.self != capability_link:
                raise _transcription_failure()
            if job_status.status in {"queued", "processing"}:
                retry_after = _retry_after(response)
                continue

            finished = _parse_finished_job(response)
            if finished.outcome in {"failed", "cancelled"}:
                await self._retire(coordination)
                raise _transcription_failure()
            succeeded = _parse_succeeded_job(response)
            return _to_reelio_result(succeeded.result)

    async def aclose(self) -> None:
        """Close the owned HTTP client and coordination repository."""
        try:
            await self._client.aclose()
        finally:
            await self._coordination_repository.aclose()

    async def _retire(self, coordination: TextifyCoordination) -> None:
        """Retire observed unusable coordination without creating replacement work.

        Args:
            coordination: Accepted coordination snapshot observed as unusable.

        Raises:
            TranscriptionError: If retirement violates coordination invariants.
            TranscriptionUnavailableError: If PostgreSQL cannot retire coordination.
        """
        try:
            await self._coordination_repository.delete(coordination)
        except CoordinationUnavailableError:
            raise _transcription_unavailable() from None
        except CoordinationError:
            raise _transcription_failure() from None

    async def _submit(
        self,
        submitted_url: str,
        idempotency_key: str,
    ) -> "_Admission":
        """Submit one already-persisted Textify job admission request.

        Args:
            submitted_url: Exact public Source URL supplied by the caller.
            idempotency_key: Canonical UUIDv4 committed before admission.

        Returns:
            Validated accepted Textify admission response and its retry instruction.

        Raises:
            TranscriptionError: If Textify does not return the accepted contract.
        """
        try:
            response = await self._client.post(
                _TRANSCRIPTION_JOBS_PATH,
                json={"url": submitted_url, "exclude": ["transcript.segments"]},
                headers={"Idempotency-Key": idempotency_key},
            )
        except httpx.HTTPError:
            raise _transcription_failure() from None
        if response.status_code != httpx.codes.ACCEPTED:
            raise _transcription_failure()
        return _Admission(
            response=response,
            links=_parse_admission(response),
        )

    async def _get(self, capability_link: str) -> httpx.Response:
        """Inspect one stored relative Textify bearer capability.

        Args:
            capability_link: Validated relative polling capability stored by Reelio.

        Returns:
            A successful or stale-capability Textify inspection response.

        Raises:
            TranscriptionError: If inspection cannot return an expected HTTP status.
        """
        try:
            response = await self._client.get(capability_link)
        except httpx.HTTPError:
            raise _transcription_failure() from None
        if response.status_code not in {httpx.codes.OK, httpx.codes.NOT_FOUND}:
            raise _transcription_failure()
        return response


class _Admission:
    """Pair an accepted HTTP response with its parsed Textify capability link."""

    def __init__(self, response: httpx.Response, links: _TextifyLinks) -> None:
        """Initialize one accepted admission response.

        Args:
            response: HTTP acceptance response containing the retry instruction.
            links: Validated JSON links representation.
        """
        self.response = response
        self.links = links


def _parse_admission(response: httpx.Response) -> _TextifyLinks:
    """Parse a strict queued admission body without preserving provider data.

    Args:
        response: Accepted Textify admission response.

    Returns:
        Parsed relative polling capability links.

    Raises:
        TranscriptionError: If the body is absent or malformed.
    """
    try:
        return _QueuedJobResponse.model_validate(response.json()).links
    except (ValidationError, ValueError):
        raise _transcription_failure() from None


def _parse_job_status(response: httpx.Response) -> _JobStatusResponse:
    """Parse common strict status fields without retaining raw provider data.

    Args:
        response: Successful Textify job inspection response.

    Returns:
        Parsed status and polling capability link.

    Raises:
        TranscriptionError: If the body is absent or malformed.
    """
    try:
        return _JobStatusResponse.model_validate(response.json())
    except (ValidationError, ValueError):
        raise _transcription_failure() from None


def _parse_finished_job(response: httpx.Response) -> _FinishedJobResponse:
    """Parse one finished Textify job outcome.

    Args:
        response: Finished Textify job inspection response.

    Returns:
        Parsed terminal job status and outcome.

    Raises:
        TranscriptionError: If the terminal response is malformed.
    """
    try:
        return _FinishedJobResponse.model_validate(response.json())
    except (ValidationError, ValueError):
        raise _transcription_failure() from None


def _parse_error_response(response: httpx.Response) -> _TextifyErrorResponse:
    """Parse one Textify error envelope.

    Args:
        response: Textify error response to validate before branching.

    Returns:
        Parsed Textify error details.

    Raises:
        TranscriptionError: If the error response is malformed.
    """
    try:
        return _TextifyErrorResponse.model_validate(response.json())
    except (ValidationError, ValueError):
        raise _transcription_failure() from None


def _parse_succeeded_job(response: httpx.Response) -> _SucceededJobResponse:
    """Parse a strict successful Textify completion body.

    Args:
        response: Finished Textify job inspection response.

    Returns:
        Parsed successful Source-and-Transcript result.

    Raises:
        TranscriptionError: If the job is not a successful valid completion.
    """
    try:
        return _SucceededJobResponse.model_validate(response.json())
    except (ValidationError, ValueError):
        raise _transcription_failure() from None


def _retry_after(response: httpx.Response) -> int:
    """Read one nonnegative integer Textify retry instruction.

    Args:
        response: Active Textify response carrying the retry header.

    Returns:
        Exact integer retry delay for the injected sleeper.

    Raises:
        TranscriptionError: If the retry header is missing or malformed.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is None or not retry_after.isascii() or not retry_after.isdecimal():
        raise _transcription_failure()
    return int(retry_after)


def _to_reelio_result(result: _TextifyResult) -> tuple[Source, Transcript]:
    """Convert a validated successful Textify result into Reelio domain values.

    Args:
        result: Strict compact Textify Source-and-Transcript response projection.

    Returns:
        Reelio-owned Source and Transcript values.

    Raises:
        TranscriptionError: If required Reelio values are invalid or unnormalized.
    """
    source = result.source
    transcript = result.transcript
    if (
        not source.video_id.strip()
        or not source.url.strip()
        or source.duration_seconds <= 0
        or not transcript.language.strip()
        or transcript.language != transcript.language.strip()
    ):
        raise _transcription_failure()
    normalized_text = normalize_transcript_segments([transcript.text])
    if not normalized_text or normalized_text != transcript.text:
        raise _transcription_failure()
    try:
        platform = Platform(source.platform)
    except ValueError:
        raise _transcription_failure() from None
    method = (
        TranscriptMethod.WHISPER
        if transcript.method == "faster_whisper"
        else TranscriptMethod.YOUTUBE_CAPTIONS
    )
    return (
        Source(
            platform=platform,
            video_id=source.video_id,
            url=source.url,
            title=source.title,
            description=source.description,
            channel=source.channel,
            duration_seconds=source.duration_seconds,
        ),
        Transcript(
            text=transcript.text,
            language=transcript.language,
            method=method,
        ),
    )


def _transcription_failure() -> TranscriptionError:
    """Create the existing safe public Transcript failure without provider details."""
    return TranscriptionError(_TRANSCRIPTION_FAILURE_MESSAGE)


def _transcription_unavailable() -> TranscriptionUnavailableError:
    """Create the stable temporary-unavailability response for coordination loss."""
    return TranscriptionUnavailableError(_TRANSCRIPTION_UNAVAILABLE_MESSAGE)


async def create_textify_source_transcript_provider(
    textify_settings: TextifyConfig,
    coordination_settings: CoordinationConfig,
) -> TextifySourceTranscriptProvider:
    """Construct one verified durable Textify Source-and-Transcript provider.

    Args:
        textify_settings: Validated Textify origin and per-attempt timeout.
        coordination_settings: Validated dedicated coordination database settings.

    Returns:
        A provider owning verified PostgreSQL coordination and no-redirect HTTP resources.
    """
    coordination_repository = await create_postgres_textify_coordination(coordination_settings)
    try:
        client = httpx.AsyncClient(
            base_url=f"{textify_settings.origin.rstrip('/')}/",
            headers={"Accept": "application/json"},
            timeout=textify_settings.request_timeout_seconds,
            follow_redirects=False,
        )
    except BaseException:
        await coordination_repository.aclose()
        raise
    return TextifySourceTranscriptProvider(
        client=client,
        coordination_repository=coordination_repository,
    )
