import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel

from src.transcript.providers.base import TranscriptProvider
from src.transcript.exceptions import (
    TranscriptDownloadError,
    TranscriptTranscriptionError,
)
from src.transcript.factory import detect_platform
from src.transcript.models import Platform, TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


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
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._temp_dir = temp_dir or tempfile.gettempdir()
        self._model: WhisperModel | None = None

        # Validate that ffmpeg is available (required by yt-dlp for audio extraction)
        if not shutil.which("ffmpeg"):
            logger.warning(
                "ffmpeg not found on system PATH. "
                "yt-dlp requires ffmpeg to extract and convert audio. "
                "Install it from https://ffmpeg.org/download.html"
            )

    def _load_model(self) -> WhisperModel:
        """Lazy-load the Whisper model on first use."""
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

    async def extract(self, url: str) -> TranscriptResult:
        platform = detect_platform(url)
        audio_path: Path | None = None

        try:
            audio_path = await self._download_audio(url)
            result = await self._transcribe(audio_path, url, platform)
            return result
        finally:
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                    logger.debug("Cleaned up temp file: %s", audio_path)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", audio_path)

    async def _download_audio(self, url: str) -> Path:
        """Download audio from the video URL using yt-dlp.

        Returns the path to the downloaded audio file.
        Raises TranscriptDownloadError on failure.
        """
        output_template = str(
            Path(self._temp_dir) / "reelio_audio_%(id)s.%(ext)s"
        )

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            def _run_ytdlp() -> Path:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    file_stem = info["id"]
                    return Path(self._temp_dir) / f"reelio_audio_{file_stem}.mp3"

            audio_path = await asyncio.to_thread(_run_ytdlp)
            logger.info("Downloaded audio to: %s", audio_path)
            return audio_path

        except yt_dlp.utils.DownloadError as exc:
            raise TranscriptDownloadError(
                f"Failed to download audio from {url}: {exc}"
            ) from exc

    async def _transcribe(
        self,
        audio_path: Path,
        url: str,
        platform: Platform,
    ) -> TranscriptResult:
        """Run faster-whisper transcription on the audio file.

        Raises TranscriptTranscriptionError on failure.
        """
        try:
            model = self._load_model()

            def _run_transcription() -> tuple[list[TranscriptSegment], str, str]:
                segments_raw, info = model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    vad_filter=True,
                )
                segments = []
                for seg in segments_raw:
                    segments.append(
                        TranscriptSegment(
                            text=seg.text.strip(),
                            start=seg.start,
                            end=seg.end,
                        )
                    )
                full_text = " ".join(seg.text for seg in segments)
                language = info.language
                return segments, full_text, language

            segments, full_text, language = await asyncio.to_thread(
                _run_transcription
            )

            return TranscriptResult(
                full_text=full_text,
                segments=segments,
                language=language,
                platform=platform,
                source_url=url,
            )

        except Exception as exc:
            raise TranscriptTranscriptionError(
                f"Whisper transcription failed for {audio_path}: {exc}"
            ) from exc
