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

## Compose deployment

The local Compose deployment requires Docker Engine, Docker Compose 2.20 or newer, a configured ignored `.env`, provider credentials, and outbound public-network access.
Captionless transcription also requires the NVIDIA driver and NVIDIA Container Toolkit required by Textify's GPU worker.

Copy the template without modifying the ignored real `.env` in version control:

```shell
cp .env.example .env
```

`REELIO_POSTGRES_PASSWORD` is required and must be URL-safe because Compose interpolates it into Reelio's asyncpg URL.
Compose also requires nonblank values for `REELIO_COORDINATION_HMAC_SECRET`, `REELIO_CACHE_KEY_SECRET`, `REELIO_LLM_PROVIDER`, `REELIO_TMDB_API_KEY`, `REELIO_SPOTIFY_CLIENT_ID`, `REELIO_SPOTIFY_CLIENT_SECRET`, and `REELIO_OPEN_LIBRARY_CONTACT_EMAIL`.
Every real secret in `.env.example` is blank.
Set only the selected provider's key: `REELIO_OPENAI_API_KEY` for `REELIO_LLM_PROVIDER=openai`, or `REELIO_DEEPSEEK_API_KEY` for `REELIO_LLM_PROVIDER=deepseek`.

Compose fixes `REELIO_CACHE_REDIS_URL` to `redis://redis:6379/0` and defaults `REELIO_TEXTIFY_ORIGIN` to `http://textify:8182`.
Source execution outside containers must instead use host-reachable endpoints, such as `http://127.0.0.1:8182` for Textify and a loopback Redis URL when caching is enabled.

Start Textify first from its checkout with the `textify` project name and the GPU profile.
This creates the independently managed `reelio-textify` network that Reelio joins.

```shell
cd /path/to/textify/backend
docker compose --project-name textify --profile gpu up --build -d --wait
```

Then validate and start Reelio:

```shell
cd /path/to/reelio/backend
docker compose --env-file .env config --quiet
docker compose --env-file .env up --build -d --wait
docker compose ps -a
curl --fail http://127.0.0.1:8181/health
```

The API publishes only `127.0.0.1:${REELIO_API_HOST_PORT:-8181}`.
PostgreSQL and Redis have no host ports.
PostgreSQL persists coordination data in the `reelio-postgres-data` named volume.
Redis is disposable, limits cache data to 512 MiB with LFU eviction, and disables RDB and AOF persistence.

Textify's `/health` is strict full-topology readiness.
Reelio startup and `/health` do not call Textify's health endpoint.
When strict Textify readiness is intentionally unnecessary, request Textify's `/openapi.json` only to check its process and cross-network connectivity.

### Live deployment verification

The opt-in live harness requires an isolated running Reelio and Textify stack, fresh exact URLs, and a sufficiently long uncached restart source to create a deterministic active-job window.
Keep the source URLs private.
Configure `REELIO_LIVE_SOURCE_URL`, `REELIO_LIVE_RESTART_SOURCE_URL`, and `REELIO_LIVE_TEXTIFY_PROJECT_DIRECTORY` without echoing them.
`REELIO_LIVE_BASE_URL` and `REELIO_LIVE_TEXTIFY_BASE_URL` default to Reelio and Textify loopback URLs.

```shell
export REELIO_LIVE_SOURCE_URL='https://public-source.example/unique-cache-source'
export REELIO_LIVE_RESTART_SOURCE_URL='https://public-source.example/unique-restart-source'
export REELIO_LIVE_TEXTIFY_PROJECT_DIRECTORY=/path/to/textify/backend
uv run --locked pytest -m live tests/test_live_compose.py
```

The harness checks the deployed topology, one unchanged public extraction contract, exact-URL cache reuse while Textify API is stopped, and PostgreSQL-backed resumption after a Reelio API restart.
If the restart source finishes before the API restart, use a longer fresh captionless source instead of treating a completed cache hit as resumption.

### Shutdown and rollback

Normal shutdown preserves Reelio's PostgreSQL volume:

```shell
docker compose down
```

Do not add `--volumes` during normal shutdown or rollback because it destroys named data volumes.
Stop Textify's worker before its remaining services according to Textify's documented shutdown procedure.

For a clean release-level rollback, select a previously retained `REELIO_IMAGE_TAG` and its matching Compose release, then run:

```shell
docker compose --env-file .env up -d --no-build
```

Preserve `reelio-postgres-data` and the independent Textify stack throughout rollback.
Never enable or reintroduce an in-process transcription fallback.

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
