import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yt_dlp

from src.transcript.exceptions import (
    TranscriptDownloadError,
    TranscriptTranscriptionError,
)
from src.transcript.models import Transcript
from src.transcript.providers.whisper import WhisperProvider


class TestWhisperProviderInit:
    def test_default_values(self):
        provider = WhisperProvider()
        assert provider._model_size == "base"
        assert provider._device == "cpu"
        assert provider._compute_type == "int8"
        assert provider._model is None

    def test_custom_values(self):
        provider = WhisperProvider(
            model_size="tiny",
            device="cuda",
            compute_type="float16",
            temp_dir="/tmp/custom",
        )
        assert provider._model_size == "tiny"
        assert provider._device == "cuda"
        assert provider._compute_type == "float16"
        assert provider._temp_dir == "/tmp/custom"


class TestWhisperProviderExtract:
    @pytest.fixture
    def mock_ytdlp(self, mocker):
        """Mock yt-dlp to return audio without real download."""
        mock_ydl = mocker.patch("yt_dlp.YoutubeDL")
        mock_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {"id": "test_video_id"}
        return mock_ydl, mock_instance

    @pytest.fixture
    def mock_whisper(self, mocker):
        """Mock faster-whisper WhisperModel to return synthetic segments."""
        mock_model_cls = mocker.patch(
            "src.transcript.providers.whisper.WhisperModel"
        )
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        seg1 = MagicMock()
        seg1.text = "This is a test "
        seg1.start = 0.0
        seg1.end = 2.0

        seg2 = MagicMock()
        seg2.text = "transcript of a video."
        seg2.start = 2.0
        seg2.end = 4.0

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        return mock_model_cls, mock_model

    @pytest.mark.anyio
    async def test_extract_success(
        self, mock_ytdlp, mock_whisper, sample_instagram_url
    ):
        provider = WhisperProvider()
        result = await provider.extract(sample_instagram_url)

        assert isinstance(result, Transcript)
        assert result.language == "en"
        assert result.full_text == "This is a test transcript of a video."

    @pytest.mark.anyio
    async def test_extract_download_error(self, mocker, sample_instagram_url):
        mocker.patch(
            "yt_dlp.YoutubeDL",
            side_effect=yt_dlp.utils.DownloadError("Network error"),
        )
        provider = WhisperProvider()
        with pytest.raises(TranscriptDownloadError) as exc_info:
            await provider.extract(sample_instagram_url)
        assert "Network error" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_extract_transcription_error(
        self, mock_ytdlp, mocker, sample_instagram_url
    ):
        mocker.patch(
            "src.transcript.providers.whisper.WhisperModel",
            side_effect=RuntimeError("CUDA out of memory"),
        )
        provider = WhisperProvider()
        with pytest.raises(TranscriptTranscriptionError) as exc_info:
            await provider.extract(sample_instagram_url)
        assert "CUDA out of memory" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_temp_file_cleanup_on_error(
        self, mock_ytdlp, mocker, sample_instagram_url
    ):
        """Verify temp audio files are cleaned up even when transcription fails."""
        mocker.patch(
            "src.transcript.providers.whisper.WhisperModel",
            side_effect=RuntimeError("Transcription failed"),
        )

        # Create a real temp file to verify cleanup
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        # Override _download_audio to return our temp file
        mocker.patch.object(
            WhisperProvider, "_download_audio", return_value=temp_path
        )

        provider = WhisperProvider()
        with pytest.raises(TranscriptTranscriptionError):
            await provider.extract(sample_instagram_url)

        # The temp file should have been cleaned up
        assert not temp_path.exists()

    @pytest.mark.anyio
    async def test_temp_file_cleanup_on_success(
        self, mock_ytdlp, mock_whisper, mocker, sample_instagram_url
    ):
        """Verify temp audio files are cleaned up after successful transcription."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        try:
            mocker.patch.object(
                WhisperProvider, "_download_audio", return_value=temp_path
            )

            provider = WhisperProvider()
            await provider.extract(sample_instagram_url)

            # File should have been cleaned up
            assert not temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.anyio
    async def test_extract_facebook_url(
        self, mock_ytdlp, mock_whisper, sample_facebook_url
    ):
        provider = WhisperProvider()
        result = await provider.extract(sample_facebook_url)
        assert result.language == "en"

    @pytest.mark.anyio
    async def test_extract_tiktok_url(
        self, mock_ytdlp, mock_whisper, sample_tiktok_url
    ):
        provider = WhisperProvider()
        result = await provider.extract(sample_tiktok_url)
        assert result.language == "en"

    @pytest.mark.anyio
    async def test_extract_newlines_are_stripped(
        self, mock_ytdlp, mocker, sample_instagram_url
    ):
        """Verify that newline characters are removed from full_text."""
        mock_model_cls = mocker.patch(
            "src.transcript.providers.whisper.WhisperModel"
        )
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        seg = MagicMock()
        seg.text = "Line one\nLine two"
        seg.start = 0.0
        seg.end = 2.0

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([seg], mock_info)

        provider = WhisperProvider()
        result = await provider.extract(sample_instagram_url)
        assert "\n" not in result.full_text


class TestWhisperProviderLoadModel:
    def test_lazy_load_model(self):
        """Model should not be loaded until first use."""
        provider = WhisperProvider()
        assert provider._model is None

    def test_load_model_cached(self, mocker):
        """Model should only be loaded once."""
        mock_model_cls = mocker.patch(
            "src.transcript.providers.whisper.WhisperModel"
        )
        provider = WhisperProvider()
        provider._load_model()
        provider._load_model()
        assert mock_model_cls.call_count == 1
