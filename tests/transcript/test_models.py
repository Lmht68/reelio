import json

from src.transcript.models import Platform, Transcript, TranscriptResult, TranscriptSegment


class TestPlatform:
    def test_platform_values(self):
        assert Platform.YOUTUBE == "youtube"
        assert Platform.INSTAGRAM == "instagram"
        assert Platform.FACEBOOK == "facebook"
        assert Platform.TIKTOK == "tiktok"
        assert Platform.UNKNOWN == "unknown"

    def test_platform_str_serialization(self):
        assert json.dumps(Platform.YOUTUBE) == '"youtube"'


class TestTranscriptSegment:
    def test_create_minimal_segment(self):
        seg = TranscriptSegment(text="Hello", start=0.0, end=2.0)
        assert seg.text == "Hello"
        assert seg.start == 0.0
        assert seg.end == 2.0
        assert seg.speaker is None

    def test_create_segment_with_speaker(self):
        seg = TranscriptSegment(
            text="Hello", start=0.0, end=2.0, speaker="John"
        )
        assert seg.speaker == "John"

    def test_segment_serialization(self):
        seg = TranscriptSegment(text="Hello", start=1.5, end=3.5)
        data = seg.model_dump()
        assert data["text"] == "Hello"
        assert data["start"] == 1.5
        assert data["end"] == 3.5
        assert data["speaker"] is None

    def test_segment_deserialization(self):
        data = {"text": "Hi", "start": 0.0, "end": 1.0}
        seg = TranscriptSegment.model_validate(data)
        assert seg.text == "Hi"


class TestTranscriptResult:
    def test_create_result(self):
        transcript = Transcript(
            full_text="Hello world",
            language="en",
        )
        result = TranscriptResult(
            transcript=transcript,
            platform=Platform.YOUTUBE,
            source_url="https://youtube.com/watch?v=test",
        )
        assert result.transcript.full_text == "Hello world"
        assert result.transcript.language == "en"
        assert result.platform == Platform.YOUTUBE
        assert result.source_url == "https://youtube.com/watch?v=test"

    def test_result_serialization(self):
        transcript = Transcript(
            full_text="Hello",
            language="en",
        )
        result = TranscriptResult(
            transcript=transcript,
            platform=Platform.YOUTUBE,
            source_url="https://youtube.com/watch?v=test",
        )
        data = result.model_dump()
        assert data["transcript"]["full_text"] == "Hello"
        assert data["platform"] == "youtube"

    def test_result_json_serialization(self):
        transcript = Transcript(
            full_text="Test",
            language=None,
        )
        result = TranscriptResult(
            transcript=transcript,
            platform=Platform.UNKNOWN,
            source_url="https://example.com/video",
        )
        json_str = result.model_dump_json()
        data = json.loads(json_str)
        assert data["transcript"]["full_text"] == "Test"
        assert data["transcript"]["language"] is None
        assert data["platform"] == "unknown"
