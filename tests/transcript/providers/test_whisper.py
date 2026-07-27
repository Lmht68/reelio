import asyncio
import concurrent.futures
import os
import tempfile
import time
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
        assert provider._max_concurrent == 2
        assert provider._max_duration_seconds == 600

    def test_custom_values(self):
        provider = WhisperProvider(
            model_size="tiny",
            device="cuda",
            compute_type="float16",
            temp_dir="/tmp/custom",
            max_concurrent=5,
            max_duration_seconds=60,
        )
        assert provider._model_size == "tiny"
        assert provider._device == "cuda"
        assert provider._compute_type == "float16"
        assert provider._temp_dir == "/tmp/custom"
        assert provider._max_concurrent == 5
        assert provider._max_duration_seconds == 60


class TestWhisperProviderExtract:
    @pytest.fixture
    def mock_download(self, mocker):
        """Patch _download_audio to return a real temp file (content irrelevant)."""

        def _fake_download(url: str) -> Path:
            fd, name = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            return Path(name)

        return mocker.patch.object(WhisperProvider, "_download_audio", side_effect=_fake_download)

    @pytest.fixture
    def mock_whisper(self, mocker):
        """Mock faster-whisper WhisperModel to return synthetic segments."""
        mock_model_cls = mocker.patch("src.transcript.providers.whisper.WhisperModel")
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
    async def test_extract_success(self, mock_download, mock_whisper, sample_instagram_url):
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
    async def test_extract_postprocessing_error(self, mocker, sample_instagram_url):
        mocker.patch(
            "yt_dlp.YoutubeDL",
            side_effect=yt_dlp.utils.PostProcessingError("ffmpeg error"),
        )
        provider = WhisperProvider()
        with pytest.raises(TranscriptDownloadError) as exc_info:
            await provider.extract(sample_instagram_url)
        assert "ffmpeg error" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_extract_os_error(self, mocker, sample_instagram_url):
        mocker.patch(
            "yt_dlp.YoutubeDL",
            side_effect=OSError("disk full"),
        )
        provider = WhisperProvider()
        with pytest.raises(TranscriptDownloadError) as exc_info:
            await provider.extract(sample_instagram_url)
        assert "disk full" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_extract_transcription_error(self, mock_download, mocker, sample_instagram_url):
        mocker.patch(
            "src.transcript.providers.whisper.WhisperModel",
            side_effect=RuntimeError("CUDA out of memory"),
        )
        provider = WhisperProvider()
        with pytest.raises(TranscriptTranscriptionError) as exc_info:
            await provider.extract(sample_instagram_url)
        assert "CUDA out of memory" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_temp_file_cleanup_on_error(self, mocker, sample_instagram_url):
        """Verify temp audio files are cleaned up even when transcription fails."""
        mocker.patch(
            "src.transcript.providers.whisper.WhisperModel",
            side_effect=RuntimeError("Transcription failed"),
        )

        # Create a real temp file to verify cleanup
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        # Override _download_audio to return our temp file
        mocker.patch.object(WhisperProvider, "_download_audio", return_value=temp_path)

        provider = WhisperProvider()
        with pytest.raises(TranscriptTranscriptionError):
            await provider.extract(sample_instagram_url)

        # The temp file should have been cleaned up
        assert not temp_path.exists()

    @pytest.mark.anyio
    async def test_temp_file_cleanup_on_success(self, mock_whisper, mocker, sample_instagram_url):
        """Verify temp audio files are cleaned up after successful transcription."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        try:
            mocker.patch.object(WhisperProvider, "_download_audio", return_value=temp_path)

            provider = WhisperProvider()
            await provider.extract(sample_instagram_url)

            # File should have been cleaned up
            assert not temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.anyio
    async def test_extract_facebook_url(self, mock_download, mock_whisper, sample_facebook_url):
        provider = WhisperProvider()
        result = await provider.extract(sample_facebook_url)
        assert result.language == "en"

    @pytest.mark.anyio
    async def test_extract_tiktok_url(self, mock_download, mock_whisper, sample_tiktok_url):
        provider = WhisperProvider()
        result = await provider.extract(sample_tiktok_url)
        assert result.language == "en"

    @pytest.mark.anyio
    async def test_extract_newlines_are_stripped(self, mock_download, mocker, sample_instagram_url):
        """Verify that newline characters are removed from full_text."""
        mock_model_cls = mocker.patch("src.transcript.providers.whisper.WhisperModel")
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

    @pytest.mark.anyio
    async def test_concurrency_bound(self, mock_whisper, mocker, sample_instagram_url):
        """Semaphore with max_concurrent=1 allows only one concurrent extract."""
        peak = 0
        active = 0

        async def _fake_download(url: str) -> Path:
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            fd, name = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            return Path(name)

        mocker.patch.object(WhisperProvider, "_download_audio", side_effect=_fake_download)

        provider = WhisperProvider(max_concurrent=1)
        await asyncio.gather(
            provider.extract(sample_instagram_url),
            provider.extract(sample_instagram_url),
            provider.extract(sample_instagram_url),
        )
        assert peak == 1


class TestWhisperProviderLoadModel:
    def test_lazy_load_model(self):
        """Model should not be loaded until first use."""
        provider = WhisperProvider()
        assert provider._model is None

    def test_load_model_cached(self, mocker):
        """Model should only be loaded once."""
        mock_model_cls = mocker.patch("src.transcript.providers.whisper.WhisperModel")
        provider = WhisperProvider()
        provider._load_model()
        provider._load_model()
        assert mock_model_cls.call_count == 1

    def test_load_model_thread_safe(self, mocker):
        """Double-checked locking prevents multiple model loads under concurrency."""
        def _slow_model(*args: object, **kw: object) -> MagicMock:
            time.sleep(0.05)  # releases the GIL so all 8 threads observe _model is None without the lock
            return MagicMock()

        mock_model_cls = mocker.patch("src.transcript.providers.whisper.WhisperModel")
        mock_model_cls.side_effect = _slow_model

        provider = WhisperProvider()

        def load():
            return provider._load_model()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load) for _ in range(8)]
            results = [f.result() for f in futures]

        # Without the lock, multiple threads see _model is None and all create one
        assert mock_model_cls.call_count == 1
        # All returned objects are identical
        assert all(r is results[0] for r in results)


class TestDownloadAudio:
    """Tests for WhisperProvider._download_audio."""

    def test_outtmpl_is_unique_per_call(self, mocker):
        """Each download call uses a unique file path with proper opts."""
        captured_opts = []

        class _MockYDL:
            def __init__(self, opts):
                captured_opts.append(opts)
                self._opts = opts
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def extract_info(self, url, download=True):
                outtmpl = self._opts["outtmpl"].replace("%(ext)s", "mp3")
                Path(outtmpl).write_bytes(b"x")
                return None

        mocker.patch("yt_dlp.YoutubeDL", side_effect=_MockYDL)

        provider = WhisperProvider()

        async def _run():
            return await provider._download_audio("https://example.com/video1")

        loop = asyncio.new_event_loop()
        try:
            path1 = loop.run_until_complete(_run())
            path2 = loop.run_until_complete(_run())
        finally:
            loop.close()

        assert path1 != path2
        assert len(captured_opts) >= 2
        opts = captured_opts[0]
        assert opts["noplaylist"] is True
        assert opts["socket_timeout"] == 30
        assert callable(opts["match_filter"])

        # Clean up
        for p in [path1, path2]:
            if p.exists():
                p.unlink()

    def test_duration_filter_rejects_long_videos(self, mocker):
        """match_filter raises DownloadCancelled for videos exceeding max_duration."""
        captured_opts = []

        class _MockYDL:
            def __init__(self, opts):
                captured_opts.append(opts)
                self._opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def extract_info(self, url, download=True):
                outtmpl = self._opts["outtmpl"].replace("%(ext)s", "mp3")
                Path(outtmpl).write_bytes(b"x")
                return None

        mocker.patch("yt_dlp.YoutubeDL", side_effect=_MockYDL)

        provider = WhisperProvider(max_duration_seconds=600)

        async def _run():
            return await provider._download_audio("https://example.com/video")

        loop = asyncio.new_event_loop()
        try:
            path = loop.run_until_complete(_run())
            if path.exists():
                path.unlink()
        finally:
            loop.close()

        assert len(captured_opts) == 1
        match_filter = captured_opts[0]["match_filter"]

        # Duration exceeds limit
        with pytest.raises(yt_dlp.utils.DownloadCancelled):
            match_filter({"duration": 99999})

        # Duration under limit -> None
        assert match_filter({"duration": 10}) is None

        # No duration key -> None
        assert match_filter({}) is None
    @pytest.mark.anyio
    async def test_missing_output_file_raises_download_error(self, mocker):
        """If yt-dlp runs but produces no file, TranscriptDownloadError is raised."""
        mock_ydl = mocker.patch("yt_dlp.YoutubeDL")
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {"id": "x"}
        mock_ydl.return_value.__enter__.return_value = mock_instance

        provider = WhisperProvider()
        with pytest.raises(TranscriptDownloadError) as exc_info:
            await provider._download_audio("https://example.com/video")
        assert "no audio file" in str(exc_info.value).lower()
