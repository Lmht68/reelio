"""Textify HTTP adapter and settings."""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NoReturn, cast
from urllib.parse import urlparse

import httpx
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from reelio.cache import AsyncCache, CacheCodecError, CacheEntry
from reelio.cache.interface import JsonObject
from reelio.extraction.coordination import (
    CoordinationConfig,
    CoordinationError,
    CoordinationUnavailableError,
    TextifyCoordination,
    TextifyCoordinationRepository,
    create_postgres_textify_coordination,
)
from reelio.extraction.exceptions import (
    DurationLimitExceededError,
    InvalidSourceError,
    MetadataProviderError,
    PipelineTimeoutError,
    SourceUnavailableError,
    TranscriptionError,
    TranscriptionUnavailableError,
    UnsupportedPlatformError,
)
from reelio.extraction.types import (
    Platform,
    Source,
    Transcript,
    TranscriptMethod,
    normalize_transcript_segments,
)

logger = logging.getLogger(__name__)

_TRANSCRIPTION_JOBS_PATH = "/api/v1/transcription-jobs"
_CAPABILITY_LINK_PATTERN = re.compile(
    r"/api/v1/transcription-jobs/"
    r"(?P<job_id>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-"
    r"[0-9a-f]{12})"
)
_TRANSPORT_RETRY_DELAY_SECONDS = 1.0
_TRANSCRIPTION_FAILURE_MESSAGE = "Transcript is unavailable for this video."
_TRANSCRIPTION_UNAVAILABLE_MESSAGE = "Transcription is temporarily unavailable."
_INVALID_SOURCE_MESSAGE = "Invalid source URL."
_UNSUPPORTED_PLATFORM_MESSAGE = (
    "Only YouTube, Instagram, Facebook, TikTok, and X URLs are supported."
)
_SOURCE_UNAVAILABLE_MESSAGE = "Source is unavailable."
_DURATION_LIMIT_MESSAGE = "Video exceeds the configured duration limit."
_METADATA_FAILURE_MESSAGE = "Unable to retrieve source metadata."
_METADATA_TIMEOUT_MESSAGE = "Source metadata acquisition timed out."
_TRANSCRIPT_TIMEOUT_MESSAGE = "Transcript acquisition timed out."


class TextifyConfig(BaseSettings):
    """Validate private Textify HTTP and bounded retry settings."""

    model_config = SettingsConfigDict(
        env_prefix="REELIO_TEXTIFY_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        validate_default=True,
    )

    origin: str = "http://textify:8182"
    request_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    admission_capacity_retry_budget_seconds: float = Field(
        default=10.0,
        gt=0,
        allow_inf_nan=False,
    )
    transport_retry_budget_seconds: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )
    result_retention_seconds: int = Field(default=2_592_000, ge=1)

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


type _TextifyErrorCode = Literal[
    "invalid_url",
    "unsupported_platform",
    "invalid_request",
    "unsupported_content",
    "video_too_long",
    "invalid_media_duration",
    "unsupported_media",
    "no_usable_transcript",
    "metadata_retrieval_failed",
    "audio_download_failed",
    "transcription_failed",
    "transcription_capacity_exceeded",
    "idempotency_conflict",
    "job_not_found",
    "job_already_finished",
    "job_store_unavailable",
    "queue_timeout",
    "worker_interrupted",
    "metadata_timeout",
    "audio_download_timeout",
    "transcription_timeout",
    "internal_error",
]


class _TextifyModel(BaseModel):
    """Validate required Textify fields while ignoring additive version-one fields."""

    model_config = ConfigDict(extra="ignore")

    @field_validator("id", mode="before", check_fields=False)
    @classmethod
    def _require_uuid_string(cls, value: object) -> object:
        """Reject non-string job identifiers before UUID version validation."""
        if not isinstance(value, str):
            raise ValueError("Textify job identifier must be a string")
        return value

    @field_validator(
        "submitted_at",
        "started_at",
        "finished_at",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def _require_timestamp_string(cls, value: object) -> object:
        """Reject non-string timestamps before ISO 8601 validation."""
        if value is not None and not isinstance(value, str):
            raise ValueError("Textify timestamp must be a string")
        return value

    @field_validator(
        "submitted_at",
        "started_at",
        "finished_at",
        check_fields=False,
    )
    @classmethod
    def _require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware Textify lifecycle timestamps."""
        if value is not None and value.tzinfo is None:
            raise ValueError("Textify timestamp must include a timezone")
        return value


class _ActiveTextifyLinks(_TextifyModel):
    """Read active polling and cancellation capabilities."""

    self: StrictStr
    cancel: StrictStr


class _FinishedTextifyLinks(_TextifyModel):
    """Read a retained terminal polling capability."""

    self: StrictStr


class _QueuedJobResponse(_TextifyModel):
    """Read one queued Textify job."""

    id: UUID4
    status: Literal["queued"]
    submitted_at: datetime
    links: _ActiveTextifyLinks


class _ProcessingJobResponse(_TextifyModel):
    """Read one processing Textify job."""

    id: UUID4
    status: Literal["processing"]
    submitted_at: datetime
    started_at: datetime
    cancellation_requested: StrictBool
    links: _ActiveTextifyLinks


class _TextifyError(_TextifyModel):
    """Read one documented Textify error without exposing its message."""

    code: _TextifyErrorCode
    message: StrictStr = Field(min_length=1)


class _TextifyErrorResponse(_TextifyModel):
    """Read one Textify error envelope."""

    error: _TextifyError


class _TextifySource(_TextifyModel):
    """Read the complete Source projection required by Reelio."""

    platform: StrictStr
    video_id: StrictStr
    url: StrictStr
    title: StrictStr
    description: StrictStr
    channel: StrictStr
    duration_seconds: StrictInt


class _TextifyTranscript(_TextifyModel):
    """Read the compact Transcript projection required by Reelio."""

    method: Literal["youtube_captions", "faster_whisper"]
    language: StrictStr
    text: StrictStr


class _TextifyResult(_TextifyModel):
    """Read a successful Source-and-Transcript result."""

    source: _TextifySource
    transcript: _TextifyTranscript


class _CachedTextifySource(BaseModel):
    """Validate the exact Reelio Source fields retained for one Textify result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    platform: Platform
    video_id: StrictStr
    url: StrictStr
    title: StrictStr
    description: StrictStr
    channel: StrictStr
    duration_seconds: StrictInt = Field(gt=0)

    @field_validator("platform", mode="before")
    @classmethod
    def _validate_platform(cls, value: object) -> Platform:
        """Reject invalid or coercible cached platform values."""
        if not isinstance(value, str):
            raise ValueError("Cached Source platform must be a string")
        try:
            return Platform(value)
        except ValueError as exc:
            raise ValueError("Cached Source platform is unsupported") from exc

    @field_validator("video_id", "url")
    @classmethod
    def _require_nonblank_identity(cls, value: str) -> str:
        """Reject blank cached Source identity fields."""
        if not value.strip():
            raise ValueError("Cached Source identity field must not be blank")
        return value


class _CachedTextifyTranscript(BaseModel):
    """Validate the exact Reelio Transcript fields retained for one Textify result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: StrictStr
    language: StrictStr
    method: TranscriptMethod

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: object) -> TranscriptMethod:
        """Reject invalid, coercible, and text-submission cached methods."""
        if not isinstance(value, str):
            raise ValueError("Cached Transcript method must be a string")
        try:
            method = TranscriptMethod(value)
        except ValueError as exc:
            raise ValueError("Cached Transcript method is unsupported") from exc
        if method is TranscriptMethod.TEXT_SUBMISSION:
            raise ValueError("Cached Textify Transcript cannot be a text submission")
        return method

    @field_validator("language")
    @classmethod
    def _require_normalized_language(cls, value: str) -> str:
        """Reject blank or whitespace-padded cached Transcript languages."""
        if not value.strip() or value != value.strip():
            raise ValueError("Cached Transcript language must be nonblank and normalized")
        return value

    @field_validator("text")
    @classmethod
    def _require_normalized_text(cls, value: str) -> str:
        """Reject blank or non-normalized cached Transcript text."""
        if not value or value != normalize_transcript_segments((value,)):
            raise ValueError("Cached Transcript text must be nonempty and normalized")
        return value


class _CachedTextifyResult(BaseModel):
    """Validate one complete strict cached Textify result projection."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: _CachedTextifySource
    transcript: _CachedTextifyTranscript


class _TextifyResultCacheCodec:
    """Serialize exact normalized Reelio Source-and-Transcript Textify results."""

    version = "textify-result-v1"

    def encode(self, value: tuple[Source, Transcript]) -> JsonObject:
        """Encode one validated Textify result without its request identity.

        Args:
            value: Normalized Reelio Source and Transcript result.

        Returns:
            Strict JSON object suitable for the shared cache envelope.

        Raises:
            CacheCodecError: If the result cannot meet the cache contract.
        """
        if (
            type(value) is not tuple
            or len(value) != 2
            or not isinstance(value[0], Source)
            or not isinstance(value[1], Transcript)
        ):
            raise CacheCodecError("Invalid Textify result cache value")
        try:
            cache_value = _cache_value_from_reelio_result(value[0], value[1])
        except ValidationError as exc:
            raise CacheCodecError("Invalid Textify result cache value") from exc
        return cast(JsonObject, cache_value.model_dump(mode="json"))

    def decode(self, payload: JsonObject) -> tuple[Source, Transcript]:
        """Decode one strict cached Textify result into fresh domain values.

        Args:
            payload: Decoded cache payload for one retained Textify result.

        Returns:
            Fresh normalized Reelio Source and Transcript values.

        Raises:
            CacheCodecError: If the payload violates the result cache contract.
        """
        try:
            cache_value = _CachedTextifyResult.model_validate(payload)
        except ValidationError as exc:
            raise CacheCodecError("Invalid Textify result cache value") from exc
        return _reelio_result_from_cache_value(cache_value)


class _SucceededJobResponse(_TextifyModel):
    """Read a completed successful Textify job."""

    id: UUID4
    status: Literal["finished"]
    outcome: Literal["succeeded"]
    submitted_at: datetime
    started_at: datetime
    finished_at: datetime
    result: _TextifyResult
    links: _FinishedTextifyLinks


class _FailedJobResponse(_TextifyModel):
    """Read a completed failed Textify job."""

    id: UUID4
    status: Literal["finished"]
    outcome: Literal["failed"]
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime
    error: _TextifyError
    links: _FinishedTextifyLinks


class _CancelledJobResponse(_TextifyModel):
    """Read a completed cancelled Textify job."""

    id: UUID4
    status: Literal["finished"]
    outcome: Literal["cancelled"]
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime
    links: _FinishedTextifyLinks


type _TextifyJobResponse = (
    _QueuedJobResponse
    | _ProcessingJobResponse
    | _SucceededJobResponse
    | _FailedJobResponse
    | _CancelledJobResponse
)


class TextifySourceTranscriptProvider:
    """Acquire Reelio Source and Transcript values through durable Textify jobs."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        coordination_repository: TextifyCoordinationRepository,
        settings: TextifyConfig,
        cache: AsyncCache,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize the provider with owned HTTP and coordination resources.

        Args:
            client: No-redirect Textify HTTP client resolved against the configured origin.
            coordination_repository: Durable admission state owner.
            settings: Validated Textify request, retry, and result-retention policy.
            cache: Shared cache borrowed from the application lifespan.
            clock: Monotonic clock used to bound retry sequences.
            sleeper: Awaitable delay used for Textify and transport retry instructions.
        """
        self._client = client
        self._coordination_repository = coordination_repository
        self._settings = settings
        self._cache = cache
        self._clock = clock
        self._sleeper = sleeper

    async def acquire(self, submitted_url: str) -> tuple[Source, Transcript]:
        """Acquire one Reelio Source and Transcript through Textify.

        Args:
            submitted_url: Exact public Source URL supplied by the caller.

        Returns:
            Reelio-owned normalized Source and Transcript values.

        Raises:
            TranscriptionError: If Textify returns invalid or unusable job data.
            TranscriptionUnavailableError: If coordination, admission, or job storage
                is temporarily unavailable.
            PipelineTimeoutError: If Textify communication exhausts its timeout budget.
        """
        successful_coordination: TextifyCoordination | None = None

        async def load_result() -> tuple[Source, Transcript]:
            """Acquire an uncached Textify result and retain its exact coordination."""
            nonlocal successful_coordination
            result, successful_coordination = await self._acquire_uncached(submitted_url)
            return result

        cache_result = await self._cache.get_or_load_with_status(
            _textify_result_entry(
                submitted_url,
                self._settings.result_retention_seconds,
            ),
            load_result,
        )
        if successful_coordination is not None and cache_result.persisted:
            await self._cleanup_successful_coordination(successful_coordination)
        return cache_result.value

    async def _acquire_uncached(
        self,
        submitted_url: str,
    ) -> tuple[tuple[Source, Transcript], TextifyCoordination]:
        """Acquire one result through PostgreSQL-coordinated Textify work.

        Args:
            submitted_url: Exact public Source URL supplied by the caller.

        Returns:
            Normalized result with the exact successful coordination snapshot.

        Raises:
            TranscriptionError: If Textify returns invalid or unusable job data.
            TranscriptionUnavailableError: If coordination, admission, or job storage
                is temporarily unavailable.
            PipelineTimeoutError: If Textify communication exhausts its timeout budget.
        """
        try:
            coordination = await self._coordination_repository.get_or_create(submitted_url)
        except CoordinationUnavailableError:
            raise _transcription_unavailable() from None

        capability_link = coordination.capability_link
        if capability_link is None:
            admission = await self._submit(submitted_url, str(coordination.idempotency_key))
            capability_link = admission.links.self
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
                _invalid_textify_response("coordination_capability_mismatch")
            capability_link = stored_capability_link
            retry_after: float = float(admission.retry_after)
        else:
            retry_after = 0.0

        while True:
            await self._sleeper(retry_after)
            response, inspection = await self._get(capability_link)
            if isinstance(inspection, _TextifyError):
                if (
                    response.status_code == httpx.codes.NOT_FOUND
                    and inspection.code == "job_not_found"
                ):
                    await self._retire(coordination)
                    raise _transcription_unavailable() from None
                if (
                    response.status_code == httpx.codes.SERVICE_UNAVAILABLE
                    and inspection.code == "job_store_unavailable"
                ):
                    raise _transcription_unavailable() from None
                _invalid_textify_response("incompatible_polling_error")

            if inspection.links.self != capability_link:
                _invalid_textify_response("stored_capability_mismatch")
            if isinstance(inspection, (_QueuedJobResponse, _ProcessingJobResponse)):
                retry_after = float(_retry_after(response))
                continue
            if isinstance(inspection, _FailedJobResponse):
                if not _is_terminal_failure_code(inspection.error.code):
                    _invalid_textify_response("incompatible_terminal_error")
                terminal_error = _terminal_exception(inspection.error.code)
                _log_adapter_event("terminal_job_failed", inspection.error.code)
                await self._retire(coordination)
                raise terminal_error from None
            if isinstance(inspection, _CancelledJobResponse):
                await self._retire(coordination)
                raise _transcription_failure() from None
            return _to_reelio_result(inspection.result), coordination

    async def aclose(self) -> None:
        """Close owned HTTP and coordination resources without closing the borrowed cache."""
        try:
            await self._client.aclose()
        finally:
            await self._coordination_repository.aclose()

    async def _cleanup_successful_coordination(
        self,
        coordination: TextifyCoordination,
    ) -> None:
        """Best-effort retire one result now confirmed in the shared cache.

        Args:
            coordination: Exact successful coordination snapshot to remove.
        """
        try:
            deleted = await self._coordination_repository.delete(coordination)
        except CoordinationError:
            _log_adapter_event("successful_coordination_cleanup_incomplete")
            return
        if not deleted:
            _log_adapter_event("successful_coordination_cleanup_incomplete")

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

    async def _request_with_transport_retries(
        self,
        send: Callable[[float], Awaitable[httpx.Response]],
        capacity_deadline: float | None = None,
    ) -> httpx.Response:
        """Send one logical Textify request within bounded transport retries.

        Args:
            send: Sends one request with its capped per-attempt timeout.
            capacity_deadline: Optional admission-capacity deadline that also bounds
                transport retries.

        Returns:
            The first Textify HTTP response received.

        Raises:
            TranscriptionError: If non-timeout communication retries exhaust.
            TranscriptionUnavailableError: If capacity retry time expires first.
            PipelineTimeoutError: If timeout communication retries exhaust.
        """
        transport_deadline = self._clock() + self._settings.transport_retry_budget_seconds
        last_error: httpx.TransportError | None = None
        while True:
            now = self._clock()
            if capacity_deadline is not None and now >= capacity_deadline:
                raise _capacity_exhausted() from None
            deadline = (
                min(transport_deadline, capacity_deadline)
                if capacity_deadline is not None
                else transport_deadline
            )
            remaining_seconds = deadline - now
            if remaining_seconds <= 0:
                if capacity_deadline is not None and deadline == capacity_deadline:
                    raise _capacity_exhausted() from None
                _raise_transport_exhausted(last_error)

            try:
                return await send(min(self._settings.request_timeout_seconds, remaining_seconds))
            except asyncio.CancelledError:
                raise
            except httpx.TransportError as exc:
                last_error = exc

            now = self._clock()
            if capacity_deadline is not None and now >= capacity_deadline:
                raise _capacity_exhausted() from None
            if now >= transport_deadline:
                _raise_transport_exhausted(last_error)

            deadline = (
                min(transport_deadline, capacity_deadline)
                if capacity_deadline is not None
                else transport_deadline
            )
            remaining_seconds = deadline - now
            if remaining_seconds <= 0:
                if capacity_deadline is not None and deadline == capacity_deadline:
                    raise _capacity_exhausted() from None
                _raise_transport_exhausted(last_error)
            await self._sleeper(min(_TRANSPORT_RETRY_DELAY_SECONDS, remaining_seconds))

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
            Validated accepted Textify admission response and retry instruction.

        Raises:
            TranscriptionError: If Textify rejects or violates its admission contract.
            TranscriptionUnavailableError: If capacity or job storage is unavailable.
            PipelineTimeoutError: If communication retries exhaust after timeouts.
        """
        request_body = {
            "url": submitted_url,
            "exclude": ["transcript.segments"],
        }
        headers = {"Idempotency-Key": idempotency_key}

        async def send(timeout: float) -> httpx.Response:
            """Send an unchanged admission attempt with a capped timeout."""
            return await self._client.post(
                _TRANSCRIPTION_JOBS_PATH,
                json=request_body,
                headers=headers,
                timeout=timeout,
            )

        capacity_deadline: float | None = None
        while True:
            if capacity_deadline is not None and self._clock() >= capacity_deadline:
                raise _capacity_exhausted() from None
            response = await self._request_with_transport_retries(
                send,
                capacity_deadline,
            )
            if response.status_code == httpx.codes.ACCEPTED:
                return _parse_admission(response)

            textify_error = _parse_error_response(response)
            if (
                response.status_code == httpx.codes.SERVICE_UNAVAILABLE
                and textify_error.code == "transcription_capacity_exceeded"
            ):
                _log_adapter_event("admission_rejected", textify_error.code)
                retry_after = _retry_after(response)
                now = self._clock()
                if capacity_deadline is None:
                    capacity_deadline = now + self._settings.admission_capacity_retry_budget_seconds
                remaining_seconds = capacity_deadline - now
                if remaining_seconds <= 0 or float(retry_after) > remaining_seconds:
                    raise _capacity_exhausted() from None
                await self._sleeper(float(retry_after))
                continue

            if not _is_admission_error_pair(
                response.status_code,
                textify_error.code,
            ):
                _invalid_textify_response("unexpected_admission_error")
            _log_adapter_event("admission_rejected", textify_error.code)
            raise _admission_exception(textify_error.code) from None

    async def _get(
        self,
        capability_link: str,
    ) -> tuple[httpx.Response, _TextifyJobResponse | _TextifyError]:
        """Inspect one stored relative Textify bearer capability.

        Args:
            capability_link: Validated relative polling capability stored by Reelio.

        Returns:
            The received response and its validated lifecycle or error projection.

        Raises:
            TranscriptionError: If inspection violates Textify's documented contract.
            PipelineTimeoutError: If communication retries exhaust after timeouts.
        """

        async def send(timeout: float) -> httpx.Response:
            """Send one polling request with a capped timeout."""
            return await self._client.get(capability_link, timeout=timeout)

        response = await self._request_with_transport_retries(send)
        if response.status_code == httpx.codes.OK:
            return response, _parse_job_response(response)

        textify_error = _parse_error_response(response)
        if not _is_polling_error_pair(response.status_code, textify_error.code):
            _invalid_textify_response("unexpected_polling_error")
        _log_adapter_event("polling_rejected", textify_error.code)
        return response, textify_error


@dataclass(frozen=True, slots=True)
class _Admission:
    """Contain one validated accepted Textify admission."""

    links: _ActiveTextifyLinks
    retry_after: int


def _parse_admission(response: httpx.Response) -> _Admission:
    """Parse one complete queued admission response.

    Args:
        response: Accepted Textify admission response.

    Returns:
        Validated admission links and the initial polling instruction.
    """
    job = _parse_job_response(response)
    if not isinstance(job, _QueuedJobResponse):
        _invalid_textify_response("admission_lifecycle_mismatch")
    if response.headers.get("Location") != job.links.self:
        _invalid_textify_response("admission_location_mismatch")
    return _Admission(links=job.links, retry_after=_retry_after(response))


def _parse_job_response(response: httpx.Response) -> _TextifyJobResponse:
    """Parse exactly one known complete Textify job response variant.

    Args:
        response: Successful Textify job lifecycle response.

    Returns:
        One validated queued, processing, succeeded, failed, or cancelled job.
    """
    try:
        body: object = response.json()
    except ValueError:
        _invalid_textify_response("malformed_job_response")
    if not isinstance(body, dict):
        _invalid_textify_response("job_response_not_object")

    status = body.get("status")
    job: _TextifyJobResponse
    try:
        if status == "queued":
            job = _QueuedJobResponse.model_validate(body)
        elif status == "processing":
            job = _ProcessingJobResponse.model_validate(body)
        elif status != "finished":
            _invalid_textify_response("unknown_job_status")
        else:
            outcome = body.get("outcome")
            if outcome == "succeeded":
                job = _SucceededJobResponse.model_validate(body)
            elif outcome == "failed":
                job = _FailedJobResponse.model_validate(body)
            elif outcome == "cancelled":
                job = _CancelledJobResponse.model_validate(body)
            else:
                _invalid_textify_response("unknown_job_outcome")
    except ValidationError:
        _invalid_textify_response("invalid_job_response")

    _validate_job_links(job)
    return job


def _validate_job_links(job: _TextifyJobResponse) -> None:
    """Validate one job's exact capability links against its public identifier.

    Args:
        job: Parsed lifecycle response with version-one links.
    """
    self_link = job.links.self
    match = _CAPABILITY_LINK_PATTERN.fullmatch(self_link)
    if match is None or match.group("job_id") != str(job.id):
        _invalid_textify_response("unsafe_job_self_link")
    if isinstance(job, (_QueuedJobResponse, _ProcessingJobResponse)):
        if job.links.cancel != f"{self_link}/cancellation":
            _invalid_textify_response("unsafe_job_cancellation_link")


def _parse_error_response(response: httpx.Response) -> _TextifyError:
    """Parse one documented Textify error envelope without retaining its message.

    Args:
        response: Textify error response to validate before branching.

    Returns:
        Validated machine-readable provider error.
    """
    try:
        return _TextifyErrorResponse.model_validate(response.json()).error
    except (ValidationError, ValueError):
        _invalid_textify_response("invalid_error_response")


def _is_admission_error_pair(status_code: int, code: _TextifyErrorCode) -> bool:
    """Return whether a documented admission HTTP status permits one error code."""
    return (
        (status_code == httpx.codes.BAD_REQUEST and code in {"invalid_url", "unsupported_platform"})
        or (status_code == httpx.codes.UNPROCESSABLE_ENTITY and code == "invalid_request")
        or (status_code == httpx.codes.CONFLICT and code == "idempotency_conflict")
        or (
            status_code == httpx.codes.SERVICE_UNAVAILABLE
            and code in {"transcription_capacity_exceeded", "job_store_unavailable"}
        )
        or (status_code == httpx.codes.INTERNAL_SERVER_ERROR and code == "internal_error")
    )


def _is_polling_error_pair(status_code: int, code: _TextifyErrorCode) -> bool:
    """Return whether a documented polling HTTP status permits one error code."""
    return (
        (status_code == httpx.codes.NOT_FOUND and code == "job_not_found")
        or (status_code == httpx.codes.UNPROCESSABLE_ENTITY and code == "invalid_request")
        or (status_code == httpx.codes.SERVICE_UNAVAILABLE and code == "job_store_unavailable")
        or (status_code == httpx.codes.INTERNAL_SERVER_ERROR and code == "internal_error")
    )


def _admission_exception(code: _TextifyErrorCode) -> Exception:
    """Translate one documented admission rejection into a Reelio exception."""
    if code == "invalid_url":
        return InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if code == "unsupported_platform":
        return UnsupportedPlatformError(_UNSUPPORTED_PLATFORM_MESSAGE)
    if code == "job_store_unavailable":
        return _transcription_unavailable()
    return _transcription_failure()


def _is_terminal_failure_code(code: _TextifyErrorCode) -> bool:
    """Return whether one Textify error code is valid in a failed job response."""
    return code not in {
        "invalid_request",
        "idempotency_conflict",
        "job_not_found",
        "job_already_finished",
    }


def _terminal_exception(code: _TextifyErrorCode) -> Exception:
    """Translate one documented terminal Textify failure into Reelio's contract."""
    if code in {"invalid_url", "unsupported_media"}:
        return InvalidSourceError(_INVALID_SOURCE_MESSAGE)
    if code == "unsupported_platform":
        return UnsupportedPlatformError(_UNSUPPORTED_PLATFORM_MESSAGE)
    if code == "unsupported_content":
        return SourceUnavailableError(_SOURCE_UNAVAILABLE_MESSAGE)
    if code == "video_too_long":
        return DurationLimitExceededError(_DURATION_LIMIT_MESSAGE)
    if code in {"invalid_media_duration", "metadata_retrieval_failed"}:
        return MetadataProviderError(_METADATA_FAILURE_MESSAGE)
    if code == "metadata_timeout":
        return PipelineTimeoutError(_METADATA_TIMEOUT_MESSAGE)
    if code in {
        "queue_timeout",
        "audio_download_timeout",
        "transcription_timeout",
    }:
        return PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE)
    if code in {"transcription_capacity_exceeded", "job_store_unavailable"}:
        return _transcription_unavailable()
    return _transcription_failure()


def _retry_after(response: httpx.Response) -> int:
    """Read one nonnegative integer Textify retry instruction.

    Args:
        response: Active or capacity Textify response carrying the retry header.

    Returns:
        Exact integer retry delay for the injected sleeper.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is None or not retry_after.isascii() or not retry_after.isdecimal():
        _invalid_textify_response("invalid_retry_after")
    return int(retry_after)


def _textify_result_entry(
    submitted_url: str,
    retention_seconds: int,
) -> CacheEntry[tuple[Source, Transcript]]:
    """Describe one exact submitted-URL cache operation for a Textify result."""
    return CacheEntry(
        layer="source:textify-result",
        key_version="v1",
        identity={"submitted_url": submitted_url},
        codec=_TextifyResultCacheCodec(),
        ttl_seconds=lambda _: retention_seconds,
        wait_timeout_seconds=1.0,
    )


def _cache_value_from_reelio_result(
    source: Source,
    transcript: Transcript,
) -> _CachedTextifyResult:
    """Validate normalized Reelio values against the durable result cache contract."""
    return _CachedTextifyResult.model_validate(
        {
            "source": {
                "platform": source.platform,
                "video_id": source.video_id,
                "url": source.url,
                "title": source.title,
                "description": source.description,
                "channel": source.channel,
                "duration_seconds": source.duration_seconds,
            },
            "transcript": {
                "method": transcript.method,
                "language": transcript.language,
                "text": transcript.text,
            },
        }
    )


def _reelio_result_from_cache_value(
    cache_value: _CachedTextifyResult,
) -> tuple[Source, Transcript]:
    """Construct fresh Reelio domain values from one validated cache projection."""
    source = cache_value.source
    transcript = cache_value.transcript
    return (
        Source(
            platform=source.platform,
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
            method=transcript.method,
        ),
    )


def _to_reelio_result(result: _TextifyResult) -> tuple[Source, Transcript]:
    """Convert a validated successful Textify result into Reelio domain values.

    Args:
        result: Strict compact Textify Source-and-Transcript response projection.

    Returns:
        Reelio-owned normalized Source and Transcript values.
    """
    source = result.source
    transcript = result.transcript
    transcript_method = (
        TranscriptMethod.WHISPER
        if transcript.method == "faster_whisper"
        else TranscriptMethod.YOUTUBE_CAPTIONS
    )
    try:
        cache_value = _CachedTextifyResult.model_validate(
            {
                "source": {
                    "platform": source.platform,
                    "video_id": source.video_id,
                    "url": source.url,
                    "title": source.title,
                    "description": source.description,
                    "channel": source.channel,
                    "duration_seconds": source.duration_seconds,
                },
                "transcript": {
                    "method": transcript_method,
                    "language": transcript.language,
                    "text": transcript.text,
                },
            }
        )
    except ValidationError:
        _invalid_textify_response("invalid_result_projection")
    return _reelio_result_from_cache_value(cache_value)


def _log_adapter_event(reason: str, code: _TextifyErrorCode | None = None) -> None:
    """Record a controlled Textify adapter event without provider data."""
    extra = {
        "operation": "textify_adapter",
        "stage": "transcription",
        "reason": reason,
    }
    if code is not None:
        extra["code"] = code
    logger.warning("textify_adapter_event", extra=extra)


def _invalid_textify_response(reason: str) -> NoReturn:
    """Log one controlled contract-validation failure and raise a safe error."""
    _log_adapter_event(reason)
    raise _transcription_failure() from None


def _capacity_exhausted() -> TranscriptionUnavailableError:
    """Create safe temporary unavailability after capacity retry expiration."""
    _log_adapter_event("admission_capacity_retry_exhausted")
    return _transcription_unavailable()


def _raise_transport_exhausted(error: httpx.TransportError | None) -> NoReturn:
    """Raise the public transport failure contract without exception chaining."""
    if isinstance(error, httpx.TimeoutException):
        _log_adapter_event("transport_timeout_retry_exhausted")
        raise PipelineTimeoutError(_TRANSCRIPT_TIMEOUT_MESSAGE) from None
    _log_adapter_event("transport_retry_exhausted")
    raise _transcription_failure() from None


def _transcription_failure() -> TranscriptionError:
    """Create the existing safe public Transcript failure without provider details."""
    return TranscriptionError(_TRANSCRIPTION_FAILURE_MESSAGE)


def _transcription_unavailable() -> TranscriptionUnavailableError:
    """Create the stable temporary-unavailability response."""
    return TranscriptionUnavailableError(_TRANSCRIPTION_UNAVAILABLE_MESSAGE)


async def create_textify_source_transcript_provider(
    textify_settings: TextifyConfig,
    coordination_settings: CoordinationConfig,
    cache: AsyncCache,
) -> TextifySourceTranscriptProvider:
    """Construct one verified durable Textify Source-and-Transcript provider.

    Args:
        textify_settings: Validated Textify origin, timeout, retry, and retention policy.
        coordination_settings: Validated dedicated coordination database settings.
        cache: Shared cache borrowed from the application lifespan.

    Returns:
        Provider owning verified PostgreSQL coordination and no-redirect HTTP resources.
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
        settings=textify_settings,
        cache=cache,
    )
