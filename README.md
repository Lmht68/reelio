# Reelio

Reelio is a FastAPI service that extracts Movie and TV Series Mentions from public social-media videos, verifies them against TMDB, and returns grouped enriched results.

## Overview

The extraction pipeline:

1. Validates the submitted URL, identifies its platform, and canonicalizes the source identity.
2. Retrieves source metadata with `yt-dlp` and enforces the configured maximum video duration.
3. Acquires a normalized transcript from YouTube captions when available and falls back to local Faster-Whisper when needed.
4. Uses Faster-Whisper directly for non-YouTube sources.
5. Sends bounded source metadata and transcript material to the selected LLM provider.
6. Validates the structured LLM response, deduplicates mentions independently per kind, and preserves first-reference order within each kind.
7. Searches TMDB's Movie and TV endpoints and resolves a Mention only when its canonical title or a provider alternative title matches together with its release or first air year.
8. Returns grouped `movies` and `tv_series` result lists, resolving each Mention to enriched metadata or `null` independently within its kind.

Movie results can include the title, release year, cast, directors, description, poster URL, TMDB and IMDb identifiers and links, and the TMDB score.
TV Series results can include the title, first air year, optional final air year, aggregate cast, Creators, description, poster URL, TMDB and IMDb identifiers and links, and the TMDB score.

Screen Work Mention interpretation supports two explicitly selected providers:

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
- Spotify Client Credentials client ID and client secret.

### Install and configure

```bash
uv sync
cp .env.example .env
```

Edit `.env` before starting the application.

At minimum, set one LLM provider and its matching credential, plus the TMDB and
Spotify credentials:

```dotenv
REELIO_LLM_PROVIDER=openai
REELIO_OPENAI_API_KEY=replace-with-your-openai-key
REELIO_TMDB_API_KEY=replace-with-your-tmdb-read-access-token
REELIO_SPOTIFY_CLIENT_ID=replace-with-your-spotify-client-id
REELIO_SPOTIFY_CLIENT_SECRET=replace-with-your-spotify-client-secret
```

Use `REELIO_LLM_PROVIDER=deepseek` and set `REELIO_DEEPSEEK_API_KEY` instead when selecting DeepSeek.
Only the selected LLM provider configuration is validated.
Spotify catalog configuration is always validated at application startup.

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

### Extract movie and TV series mentions

```http
POST /api/extract
Content-Type: application/json
```

Request body:

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "market": "JP"
}
```

The response contains:
- `market`: The effective uppercase ISO 3166-1 alpha-2 Spotify market.
  Omit it to use configured `REELIO_SPOTIFY_DEFAULT_MARKET`, which defaults to `US`.

- `source`: The canonical platform, external video ID, URL, title, description, channel, and duration.
- `transcript`: The normalized transcript text, detected language, and acquisition method.
- `results`: A grouped object with two always-present lists, `movies` and `tv_series`.
  Each list is deduplicated independently and preserves first-reference order within its kind.
  There is no cross-kind ordering.
- `results.movies[].movie_mention`: The canonical Movie title and release year interpreted by the LLM.
- `results.movies[].movie`: TMDB-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
- `results.tv_series[].tv_series_mention`: The canonical TV Series title and first air year interpreted by the LLM.
- `results.tv_series[].tv_series`: TMDB-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
  `first_air_year` is the verified TV First Air Year.
  A `null` `last_air_year` means the final air year is unavailable, not that the TV Series continues.
  `creators` comes only from TMDB's `created_by` list and retains first-provider order after duplicate names are removed.
  `cast` is the first five TMDB aggregate-cast names in provider order, with no role filtering or person deduplication.
  TMDB and IMDb identifiers and links are included when available, along with the TMDB score.
- Any TMDB provider HTTP, timeout, or required-response validation failure fails the complete request rather than returning partial category results.

Compact success example:

```json
{
  "market": "US",
  "source": {
    "platform": "youtube",
    "video_id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "Screen Work review",
    "description": "A review mentioning a Movie and a TV Series.",
    "channel": "Example channel",
    "duration_seconds": 42
  },
  "transcript": {
    "text": "Dune: Part One and The Last of Us are excellent.",
    "language": "en",
    "method": "youtube_captions"
  },
  "results": {
    "movies": [
      {
        "status": "resolved",
        "movie_mention": {"title": "Dune: Part One", "year": 2021},
        "movie": {
          "title": "Dune: Part One",
          "year": 2021,
          "cast": ["Timothée Chalamet"],
          "directors": ["Denis Villeneuve"],
          "description": "Paul Atreides faces his destiny on Arrakis.",
          "poster_url": "https://image.tmdb.org/t/p/w500/dune.jpg",
          "tmdb_id": 438631,
          "tmdb_url": "https://www.themoviedb.org/movie/438631",
          "imdb_id": "tt1160419",
          "imdb_url": "https://www.imdb.com/title/tt1160419/",
          "tmdb_score": 7.8
        }
      }
    ],
    "tv_series": [
      {
        "status": "resolved",
        "tv_series_mention": {"title": "The Last of Us", "year": 2023},
        "tv_series": {
          "title": "The Last of Us",
          "first_air_year": 2023,
          "last_air_year": null,
          "cast": ["Pedro Pascal", "Bella Ramsey"],
          "creators": ["Craig Mazin", "Neil Druckmann"],
          "description": "A smuggler escorts a teenager across a ruined America.",
          "poster_url": "https://image.tmdb.org/t/p/w500/the-last-of-us.jpg",
          "tmdb_id": 100088,
          "tmdb_url": "https://www.themoviedb.org/tv/100088",
          "imdb_id": "tt3581920",
          "imdb_url": "https://www.imdb.com/title/tt3581920/",
          "tmdb_score": 8.6
        }
      },
      {
        "status": "unresolved",
        "tv_series_mention": {"title": "Unknown TV Series", "year": 2024},
        "tv_series": null
      }
    ]
  }
}
```

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
| `502` | Metadata, transcription, LLM, or TMDB provider failure. Any TMDB failure fails the complete request. |
| `504` | External provider timeout. |

## Configuration

All supported settings and their defaults are documented in [`.env.example`](.env.example).

- `REELIO_LOG_LEVEL` defaults to `INFO`.
- `REELIO_LLM_PROVIDER` is required and accepts only the exact lowercase values `openai` and `deepseek`.
- OpenAI uses `gpt-5-nano` by default and supports model, reasoning effort, timeout, output-token, and retry overrides.
- DeepSeek uses `deepseek-v4-flash` and `https://api.deepseek.com` by default and supports endpoint, generation, timeout, output-token, and retry overrides.
- `REELIO_MAX_VIDEO_DURATION_SECONDS` defaults to 1,800 seconds.
- Faster-Whisper uses the `large-v3-turbo` model, CUDA, `float16`, and one concurrent transcription by default.
- Interpretation Material limits default to 500 source-title characters, 2,000 description characters, 64 transcript-language characters, and 100,000 transcript characters.
- TMDB uses `https://api.themoviedb.org/3`, the `w500` image endpoint, and a 10-second request timeout by default.
- Spotify catalog requests use the configured default market, API and token endpoints, request timeout, and safe token-expiry skew.

Credentials are loaded from environment variables and are not written to logs.

Spotify catalog access uses Client Credentials in development mode as a prototype constraint.
Development account ownership, allowlists, and quota restrictions are not production guarantees.
Before production use, review Spotify Developer Policy, platform terms, attribution requirements, quota eligibility, and the approved use case.
Spotify metadata, artwork, identifiers, and URLs are excluded from LLM prompts and model-training flows.

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
        ├── catalog/                Spotify Client Credentials catalog boundary
        ├── enrichment/             TMDB candidate resolution and enrichment
        ├── interpretation/         OpenAI and DeepSeek Screen Work Mention providers
        └── transcription/          Metadata inspection and transcript acquisition
```
