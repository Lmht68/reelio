import asyncio
import logging
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from faster_whisper import WhisperModel

from src.transcript.cleaner import clean_transcript
from src.transcript.exceptions import (
    TranscriptDownloadError,
    TranscriptTranscriptionError,
)
from src.transcript.models import Transcript
from src.transcript.providers.base import TranscriptProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperDefaults:
    beam_size: int = 1
    vad_filter: bool = True
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    initial_prompt: str = (
        "This transcript may mention movie titles, TV shows, directors, "
        "actors, song titles, albums, artists, bands, books, and authors. "
        "Transcribe proper names accurately."
    )


WHISPER_DEFAULTS = WhisperDefaults()


class WhisperProvider(TranscriptProvider):
    """Downloads video audio via yt-dlp and transcribes with faster-whisper.

    This is the fallback/generic provider used for Facebook, Instagram, TikTok,
    and any other non-YouTube platform.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        temp_dir: str | None = None,
        max_concurrent: int = 2,
        max_duration_seconds: int = 600,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._temp_dir = temp_dir or tempfile.gettempdir()
        self._model: WhisperModel | None = None
        self._max_concurrent = max_concurrent
        self._max_duration_seconds = max_duration_seconds
        self._model_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _load_model(self) -> WhisperModel:
        """Lazy-load the Whisper model on first use (thread-safe)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info(
                        "Loading faster-whisper model: size=%s, device=%s, compute_type=%s",
                        self._model_size,
                        self._device,
                        self._compute_type,
                    )
                    self._model = WhisperModel(
                        self._model_size,
                        device=self._device,
                        compute_type=self._compute_type,
                    )
        return self._model

    async def extract(self, url: str) -> Transcript:
        audio_path: Path | None = None

        async with self._semaphore:
            try:
                audio_path = await self._download_audio(url)
                transcript = await self._transcribe(audio_path)
                cleaned_text = clean_transcript(transcript.full_text)
                return transcript.model_copy(update={"full_text": cleaned_text})
            finally:
                if audio_path and audio_path.exists():
                    try:
                        audio_path.unlink()
                        logger.info("Cleaned up temp file: %s", audio_path)
                    except OSError:
                        logger.warning("Failed to clean up temp file: %s", audio_path)

    async def _download_audio(self, url: str) -> Path:
        """Download audio from the video URL using yt-dlp.

        Returns the path to the downloaded audio file.
        Raises TranscriptDownloadError on failure.
        """
        if not shutil.which("ffmpeg"):
            raise TranscriptDownloadError(
                "ffmpeg is required for audio extraction but was not found on PATH"
            )

        file_token = uuid.uuid4().hex
        output_path = Path(self._temp_dir) / f"reelio_audio_{file_token}.mp3"

        def _duration_filter(info: dict[str, Any], *, incomplete: bool = False) -> str | None:
            duration = info.get("duration")
            if duration is not None and duration > self._max_duration_seconds:
                raise yt_dlp.utils.DownloadCancelled(
                    f"Video duration ({duration}s) exceeds the {self._max_duration_seconds}s limit"
                )
            return None

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(Path(self._temp_dir) / f"reelio_audio_{file_token}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 30,
            "match_filter": _duration_filter,
        }

        try:

            def _run_ytdlp() -> Path:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)
                return output_path

            audio_path = await asyncio.to_thread(_run_ytdlp)
        except yt_dlp.utils.YoutubeDLError as exc:
            raise TranscriptDownloadError(f"Failed to download audio from {url}: {exc}") from exc
        except Exception as exc:
            raise TranscriptDownloadError(
                f"Unexpected error downloading audio from {url}: {exc}"
            ) from exc

        if not audio_path.exists():
            raise TranscriptDownloadError(f"yt-dlp produced no audio file for {url}")
        logger.info("Downloaded audio to: %s", audio_path)
        return audio_path

    async def _transcribe(
        self,
        audio_path: Path,
    ) -> Transcript:
        """Run faster-whisper transcription on the audio file.

        Raises TranscriptTranscriptionError on failure.
        """
        try:
            model = self._load_model()

            def _run_transcription() -> tuple[str, str]:
                segments_raw, info = model.transcribe(
                    str(audio_path),
                    beam_size=WHISPER_DEFAULTS.beam_size,
                    temperature=WHISPER_DEFAULTS.temperature,
                    vad_filter=WHISPER_DEFAULTS.vad_filter,
                    condition_on_previous_text=WHISPER_DEFAULTS.condition_on_previous_text,
                    initial_prompt=WHISPER_DEFAULTS.initial_prompt,
                )
                texts: list[str] = []
                for seg in segments_raw:
                    texts.append(seg.text.strip().replace("\n", " "))
                full_text = " ".join(texts)
                language = info.language
                return full_text, language

            full_text, language = await asyncio.to_thread(_run_transcription)

            return Transcript(
                full_text=full_text,
                language=language,
            )

        except Exception as exc:
            raise TranscriptTranscriptionError(
                f"Whisper transcription failed for {audio_path}: {exc}"
            ) from exc
