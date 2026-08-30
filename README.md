# Reelio

Reelio is a FastAPI service that extracts movie mentions from public social-media videos, verifies them against TMDB, and returns enriched metadata and links.

## Overview

The extraction pipeline:

1. Validates the submitted URL, identifies its platform, and canonicalizes the source identity.
2. Retrieves source metadata with `yt-dlp` and enforces the configured maximum video duration.
3. Acquires a normalized transcript from YouTube captions when available and falls back to local Faster-Whisper when needed.
4. Uses Faster-Whisper directly for non-YouTube sources.
5. Sends bounded source metadata and transcript material to the selected LLM provider.
6. Validates the structured LLM response, removes duplicate mentions, and preserves first-reference order.
7. Searches TMDB and resolves a mention only when its canonical title or provider alternative title matches together with the release year.
8. Returns one `resolved` or `unresolved` result per interpreted movie mention.

Movie results can include the title, release year, cast, directors, description, poster URL, TMDB and IMDb identifiers and links, and the TMDB score.

Movie Mention interpretation supports two explicitly selected providers:

- OpenAI uses the Responses API, strict Structured Outputs generated from the application response model, and `store=false`.
- DeepSeek uses Chat Completions with JSON-object output and disabled thinking.

Set `REELIO_LLM_PROVIDER` to exactly `openai` or `deepseek`.

Provider selection happens once during application startup, remains fixed for the application lifespan, and never falls back automatically.

## Quick start

### Prerequisites

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- An LLM API key for the selected provider.
- A TMDB v4 read access token.

### Install and configure

```bash
uv sync
cp .env.example .env
```

Edit `.env` before starting the application.

At minimum, set one LLM provider and its matching credential, plus the TMDB credential:

```dotenv
REELIO_LLM_PROVIDER=openai
REELIO_OPENAI_API_KEY=replace-with-your-openai-key
REELIO_TMDB_API_KEY=replace-with-your-tmdb-read-access-token
```

Use `REELIO_LLM_PROVIDER=deepseek` and set `REELIO_DEEPSEEK_API_KEY` instead when selecting DeepSeek.

Only the selected LLM provider configuration is validated.

On a machine without CUDA, set `REELIO_WHISPER_DEVICE=cpu` or `REELIO_WHISPER_DEVICE=auto` instead of the example's CUDA default.

### Run the API

```bash
uv run uvicorn reelio.main:app --reload
```

The server listens on `http://127.0.0.1:8000` by default.

Interactive API documentation is available at `http://127.0.0.1:8000/docs` in local and staging environments.

Documentation routes are disabled when `REELIO_ENVIRONMENT=production`.

## API

### Health check

```http
GET /health
```

Successful response:

```json
{"status": "ok"}
```

### Extract movie mentions

```http
POST /api/extract
Content-Type: application/json
```

Request body:

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

The response contains:

- `source`: The canonical platform, external video ID, URL, title, description, channel, and duration.
- `transcript`: The normalized transcript text, detected language, and acquisition method.
- `results`: One result for every deduplicated Movie Mention in first-reference order.
- `results[].movie_mention`: The canonical movie title and release year interpreted by the LLM.
- `results[].movie`: TMDB-backed enrichment for a resolved mention, or `null` for an unresolved mention.

Equivalent `curl` request:

```bash
curl -X POST http://127.0.0.1:8000/api/extract \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

Extraction failures use a stable response shape:

```json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable message."
  }
}
```

The API uses these error classes:

| HTTP status | Meaning |
| --- | --- |
| `400` | Invalid URL or unsupported platform. |
| `404` | Source is unavailable, private, or not found. |
| `413` | Source duration or interpretation material exceeds its configured limit. |
| `500` | Unexpected internal failure. |
| `502` | Metadata, transcription, LLM, or TMDB provider failure. |
| `504` | External provider timeout. |

## Configuration

All supported settings and their defaults are documented in [`.env.example`](.env.example).

- `REELIO_ENVIRONMENT` accepts `local`, `staging`, or `production`, and defaults to `local`.
- `REELIO_LOG_LEVEL` defaults to `INFO`.
- `REELIO_LLM_PROVIDER` is required and accepts only the exact lowercase values `openai` and `deepseek`.
- OpenAI uses `gpt-5-nano` by default and supports model, reasoning effort, timeout, output-token, and retry overrides.
- DeepSeek uses `deepseek-v4-flash` and `https://api.deepseek.com` by default and supports endpoint, generation, timeout, output-token, and retry overrides.
- `REELIO_MAX_VIDEO_DURATION_SECONDS` defaults to 1,800 seconds.
- Faster-Whisper uses the `large-v3-turbo` model, CUDA, `float16`, and one concurrent transcription by default.
- Interpretation Material limits default to 500 source-title characters, 2,000 description characters, 64 transcript-language characters, and 100,000 transcript characters.
- TMDB uses `https://api.themoviedb.org/3`, the `w500` image endpoint, and a 10-second request timeout by default.

Credentials are loaded from environment variables and are not written to logs.

## Development

Install the development dependency group with `uv sync`, then run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

The project uses strict mypy checking and pytest's automatic asyncio mode.

## Repository layout

```text
src/reelio/
├── main.py                         FastAPI composition root and lifespan
├── config.py                       Application environment settings
├── ops.py                          Health endpoint
└── extraction/
    ├── router.py                   Extraction HTTP endpoint
    ├── schemas.py                  Request and response models
    ├── service.py                  End-to-end extraction orchestration
    ├── types.py                    Extraction domain types
    └── services/
        ├── transcription/          Metadata inspection and transcript acquisition
        ├── interpretation/         OpenAI and DeepSeek Movie Mention providers
        └── enrichment/             TMDB candidate resolution and enrichment
```

The domain vocabulary and current product boundaries are recorded in [`CONTEXT.md`](CONTEXT.md).

Provider-specific architecture decisions are recorded in [`docs/adr/0002-native-llm-provider-adapters.md`](docs/adr/0002-native-llm-provider-adapters.md).
