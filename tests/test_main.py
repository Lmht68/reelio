from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app, get_transcript_service
from src.transcript import (
    Platform,
    Transcript,
    TranscriptDownloadError,
    TranscriptError,
    TranscriptInvalidURLError,
    TranscriptNotFoundError,
    TranscriptResult,
    TranscriptTranscriptionError,
    TranscriptUnsupportedPlatformError,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_service():
    service = MagicMock()
    app.dependency_overrides[get_transcript_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_success(client, mock_service):
    mock_service.extract = AsyncMock(
        return_value=TranscriptResult(
            transcript=Transcript(full_text="hello", language="en"),
            platform=Platform.YOUTUBE,
            source_url="https://youtube.com/watch?v=test",
        )
    )
    response = client.post("/api/transcript", json={"url": "https://youtube.com/watch?v=test"})
    assert response.status_code == 200
    assert response.json()["transcript"]["full_text"] == "hello"


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (TranscriptInvalidURLError("bad url"), 400),
        (TranscriptUnsupportedPlatformError("unknown platform"), 400),
        (TranscriptNotFoundError("no transcript"), 404),
        (TranscriptDownloadError("download failed"), 502),
        (TranscriptTranscriptionError("transcription failed"), 502),
        (TranscriptError("generic error"), 500),
    ],
)
def test_error_mapping(client, mock_service, error, expected_status):
    mock_service.extract = AsyncMock(side_effect=error)
    response = client.post("/api/transcript", json={"url": "https://youtube.com/watch?v=test"})
    assert response.status_code == expected_status
    body = response.json()
    assert body["detail"] == str(error)
    assert body["error_type"] == type(error).__name__
