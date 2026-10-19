# Reelio

Reelio is a FastAPI service that finds movies, TV series, tracks, music releases, and books mentioned in public social videos.
Submit a video URL and receive the source details, its transcript, and catalog-enriched results.

## How it works

1. Textify inspects the public Source and acquires its Transcript.
2. Reelio keeps `POST /api/extractions` synchronous and returns Reelio-owned Source, Transcript, result, and error contracts.
3. Reelio coordinates active Textify work in dedicated PostgreSQL and represents exact submitted-URL identities with its environment-scoped HMAC.
4. An optional shared Redis cache stores successful Textify results and Mention interpretation results without becoming an availability dependency.
5. The selected LLM identifies structured media mentions, then Reelio resolves movies and TV series through TMDB, music through Spotify, and books through Open Library.

Each category is returned independently.
An item is `resolved` when catalog metadata is found or `unresolved` when the original mention could not be matched.

## Quick start

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A reachable Textify service
- Dedicated PostgreSQL 17 for Reelio's active-work coordination
- An optional Redis-compatible cache for successful results and Mention interpretation reuse
- An OpenAI or DeepSeek API key
- A TMDB v4 read access token
- Spotify Client Credentials
- An Open Library contact email for book lookups

```bash
uv sync
cp .env.example .env
```

Configure `.env` with the selected LLM provider and catalog credentials.

```dotenv
REELIO_LLM_PROVIDER=openai
REELIO_OPENAI_API_KEY=replace-with-your-key
REELIO_TMDB_API_KEY=replace-with-your-token
REELIO_SPOTIFY_CLIENT_ID=replace-with-your-client-id
REELIO_SPOTIFY_CLIENT_SECRET=replace-with-your-client-secret
REELIO_OPEN_LIBRARY_CONTACT_EMAIL=you@example.com
```

Use `REELIO_LLM_PROVIDER=deepseek` with `REELIO_DEEPSEEK_API_KEY` to select DeepSeek.
See [`.env.example`](.env.example) for the exact Textify origin, admission and transport retry budgets, result and coordination retention, PostgreSQL coordination, HMAC, cache, LLM, and catalog settings.

```bash
uv run uvicorn reelio.main:app --reload
```

The local API is available at `http://127.0.0.1:8000`.
OpenAPI documentation is available at `http://127.0.0.1:8000/docs` outside production.

## API

### Health check

```http
GET /health
```

```json
{"status": "ok"}
```

### Extract mentions

```bash
curl -X POST http://127.0.0.1:8000/api/extractions \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

`POST /api/extractions` accepts a public video `url`.
It also accepts an optional two-letter Spotify `market`, such as `US` or `JP`.
The market changes Spotify lookups only.
Textify backs Source inspection and Transcript acquisition while Reelio preserves the synchronous public response and Reelio-owned errors.

The response contains:

- `source`: Canonical platform, video ID, URL, title, channel, and duration.
- `transcript`: Normalized text, detected language, and acquisition method.
- `statistics`: Mention, resolved, and unresolved counts for each category.
- `results`: Grouped `movies`, `tv_series`, `tracks`, `music_releases`, and `books` lists.

Each result includes the interpreted mention, a `status`, and provider metadata when resolved.

```json
{
  "market": "US",
  "results": {
    "movies": [
      {
        "status": "resolved",
        "movie_mention": {"title": "Dune: Part One", "year": 2021},
        "movie": {
          "title": "Dune: Part One",
          "year": 2021,
          "tmdb_id": 438631
        }
      }
    ],
    "tv_series": [],
    "tracks": [],
    "music_releases": [],
    "books": []
  }
}
```

Unmatched mentions remain in their category with `status: "unresolved"` and `null` provider metadata.
The OpenAPI schema documents every response field.

Failures use a stable response shape:

```json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable message."
  }
}
```

Common failures are invalid or unsupported URLs (`400`), unavailable sources (`404`), duration limits (`413`), provider failures (`502`), and provider timeouts (`504`).
Temporary Source-and-Transcript acquisition or durable coordination unavailability returns `503` with `transcription_unavailable` and `Retry-After: 2`.

### Direct Transcript Submission

`POST /api/internal/extractions` is available only in local and staging environments.
It accepts directly submitted normalized Transcript text and the optional `market`.
Its response has `transcript.method` set to `text_submission`, omits `source`, and otherwise returns the same market, statistics, and grouped results contract.
Direct Transcript Submission bypasses Textify.

## Shared-cache operations

Provision the optional Redis-compatible cache per environment with `maxmemory-policy allkeys-lfu`, `save ""`, and `appendonly no`.

Run provider cleanup only through the CLI:

```bash
uv run reelio-cache-purge \
  --provider <spotify|tmdb|open-library|source> \
  --environment <local|staging|production>
```

A `source` purge removes successful Textify results and Mention interpretation cache entries without touching active PostgreSQL coordination.

## Development

Install the development dependency group with `uv sync`, then run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```
