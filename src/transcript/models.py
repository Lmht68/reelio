from enum import StrEnum

from pydantic import BaseModel, Field


class Platform(StrEnum):
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    UNKNOWN = "unknown"


class TranscriptSegment(BaseModel):
    """A single timed segment of a transcript."""

    text: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    speaker: str | None = Field(default=None, description="Speaker label, if available")


class TranscriptResult(BaseModel):
    """The complete transcript result returned to callers."""

    full_text: str = Field(description="Full concatenated transcript text")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Time-segmented transcript segments",
    )
    language: str | None = Field(default=None, description="Detected language code, e.g. 'en'")
    platform: Platform = Field(description="The video platform this transcript came from")
    source_url: str = Field(description="The original video URL")
