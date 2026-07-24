# Reelio

An AI-powered API that extracts book, music, and movie recommendations from reels (short videos).

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for package management
- [ffmpeg](https://ffmpeg.org/download.html) (required for audio extraction from non-YouTube platforms)

### Install

```bash
uv sync
```

### Configure

Copy `.env.example` to `.env` and fill in your API keys:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com` | LLM API endpoint |
| `LLM_MODEL` | `deepseek-v4-pro` | Model name |
| `LLM_API_KEY` | (required) | API key |
| `WHISPER_MODEL` | `base` | Whisper model size: tiny / base / small / medium / large-v3 |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` / `float16` / `float32` |
| `TRANSCRIPT_TEMP_DIR` | system temp | Override temp directory for audio downloads |

### Run

```bash
fastapi dev src/main.py
```

## API

### `POST /api/transcript`

Extract a transcript from a video/reel URL.

**Request:**
```json
{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
```

**Response (200):**
```json
{
  "full_text": "Full transcript text...",
  "segments": [
    {"text": "Hello", "start": 0.0, "end": 2.0, "speaker": null}
  ],
  "language": "en",
  "platform": "youtube",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

Supported platforms: YouTube, Instagram, Facebook, TikTok.

## Architecture

```
src/
  config.py                    # Settings via pydantic-settings
  main.py                      # FastAPI application
  transcript/                  # Transcript extraction module
    models.py                  # Platform enum, TranscriptSegment, TranscriptResult
    exceptions.py              # Error hierarchy
    base.py                    # TranscriptProvider abstract base class
    factory.py                 # URL validation and platform detection
    service.py                 # TranscriptService orchestrator
    providers/
      youtube.py               # YouTubeProvider (youtube-transcript-api)
      whisper.py               # WhisperProvider (yt-dlp + faster-whisper)
```

- **YouTube**: Fetches captions directly via `youtube-transcript-api` (no download).
- **All other platforms**: Downloads audio with `yt-dlp`, transcribes with `faster-whisper` (CTranslate2-backed, 4x faster than openai-whisper, built-in VAD).

## Development

```bash
# Run tests
pytest -v

# Lint
ruff check src/ tests/

# Type check
mypy src/
```
