# Reelio

Reelio is a FastAPI service that extracts Movie, TV Series, Track, and Music Release Mentions from public social-media videos, verifies Screen Works against TMDB and Music against Spotify, and returns grouped enriched results.

## Overview

The extraction pipeline:

1. Validates the submitted URL, identifies its platform, and canonicalizes the source identity.
2. Retrieves source metadata with `yt-dlp` and enforces the configured maximum video duration.
3. Acquires a normalized transcript from YouTube captions when available and falls back to local Faster-Whisper when needed.
4. Uses Faster-Whisper directly for non-YouTube sources.
5. Sends bounded source metadata and transcript material to the selected LLM provider.
6. Validates the structured LLM response, deduplicates Mentions independently per kind, and preserves first-reference order within each kind.
7. Searches TMDB's Movie and TV endpoints and resolves a Screen Work Mention by matching its canonical or provider alternative title against the exact interpreted year first, then the immediately following and preceding provider years only if no exact-year Candidate matches.
8. Makes one bounded Spotify Track search for each Track Mention in the effective market, checks exact artist-eligible titles first, then reuses those Candidates in Spotify order for controlled Equivalent Track Version resolution.
9. Makes one bounded Spotify Album search for each direct Music Release Mention in the effective market, checks exact artist-eligible titles first, then reuses those Candidates in Spotify order for controlled equivalent-edition resolution.
10. Returns grouped `movies`, `tv_series`, `tracks`, and `music_releases` result lists, resolving each Mention to enriched metadata or `null` independently within its kind.

Movie results can include the title, release year, cast, directors, description, poster URL, TMDB and IMDb identifiers and links, and the TMDB score.
TV Series results can include the title, first air year, optional final air year, aggregate cast, Creators, description, poster URL, TMDB and IMDb identifiers and links, and the TMDB score.
Track results can include Spotify's canonical Track title, ordered artist credits, a playable Spotify Track ID and URL, the accepted Candidate's attached Album as `preferred_music_release`, and a Spotify-hosted cover URL.
Music Release results can include Spotify's canonical Album title, ordered artist credits, provider-reported `release_date`, album type, a Spotify Album ID and URL, and a Spotify-hosted cover URL.

Mention interpretation supports two explicitly selected providers:

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

### Extract Movie, TV Series, Track, and Music Release Mentions

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
- `statistics`: A grouped object before `results` with always-present `movies`, `tv_series`, `tracks`, and `music_releases` category objects.
  Each category contains non-negative integer `n_mentions`, `n_resolved`, and `n_unresolved` fields.
  `n_mentions` counts returned items and equals `n_resolved + n_unresolved`.
- `results`: A grouped object with four always-present lists, `movies`, `tv_series`, `tracks`, and `music_releases`.
  Each list is deduplicated independently and preserves first-reference order within its kind.
  There is no cross-kind ordering.
- `results.movies[].movie_mention`: The canonical Movie title and release year interpreted by the LLM.
- `results.movies[].movie`: TMDB-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
- `results.tv_series[].tv_series_mention`: The canonical TV Series title and first air year interpreted by the LLM.
- `results.tv_series[].tv_series`: TMDB-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
  `first_air_year` is the matched TMDB provider year and can differ from the preserved TV Series Mention year by at most one.
  A `null` `last_air_year` means the final air year is unavailable, not that the TV Series continues.
  `creators` comes only from TMDB's `created_by` list and retains first-provider order after duplicate names are removed.
  `cast` is the first five TMDB aggregate-cast names in provider order, with no role filtering or person deduplication.
  TMDB and IMDb identifiers and links are included when available, along with the TMDB score.
- `Screen Work resolution`: Reelio first verifies same-kind direct TMDB Candidates with exact Screen Work Title Normalization and retains its existing alternative-title checks.
  When strict verification leaves a Movie Mention or TV Series Mention unresolved, Reelio first evaluates fuzzy primary and original titles from the first three existing direct-search Candidates for each interpreted, following, and preceding year, then evaluates the first three TMDB multi-search results only when that stage finds no match.
  Fuzzy comparison uses Unicode NFC normalization, whitespace trimming and collapse, Unicode case folding, preserved punctuation and accents, and a score strictly above 90.
  A fallback Candidate must have the same kind and a detailed TMDB release or first-air year within one year of the Mention.
  The first qualifying Candidate in the defined provider order becomes the ordinary resolved result, while people, opposite-kind results, missing detailed years, and larger year variance remain ineligible.
- `results.tracks[].track_mention`: The interpreted Track title, ordered Track artists, and explicit nullable release title and year context.
- `results.tracks[].track`: Spotify-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
  A resolved Track has Spotify's canonical Track title, ordered artist credits, playable Track ID and URL, `preferred_music_release`, and `cover_url`.
  `preferred_music_release` is the Album attached to the accepted Spotify Track Candidate and contains its canonical title, ordered artist credits, provider-reported `release_date`, Spotify Album identity, and Album `spotify_url`.
  The Track `cover_url` equals its Preferred Music Release `cover_url`, which is the first provider-ordered Spotify-hosted Album image URL or `null` when Spotify returns no Album images.
  Each Track Mention causes one Spotify search in the effective market with offset zero and limit three.
  The Track query uses the Mention title and first ordered Artist Credit only.
  A Candidate is artist-eligible when at least one Candidate Artist Credit matches any Mention Artist Credit under Music Identity Normalization, regardless of credit order, count, or unmatched additions.
  Reelio first checks exact Track title equality and, when supplied, exact attached Music Release title equality across every artist-eligible Candidate in Spotify order.
  Only when no exact Candidate exists does it reuse the same bounded, artist-eligible Candidate sequence for a controlled Equivalent Track Version pass without another Spotify search.
  Equivalent Track Versions accept Remaster, Remastered, Remastered Version, four-digit year Remaster forms, Bonus Edition, Bonus Version, Bonus Track, and Bonus Track Version.
  Each designation must occupy a complete trailing segment bounded by parentheses, brackets, a spaced hyphen, or a colon.
  Stacked recognized trailing segments are removed right to left, and the remaining normalized base Track titles must match exactly.
  This comparison is symmetric between a base Track title and an eligible version title, or between two eligible version titles.
  Deluxe, expanded, special, anniversary, and reissue designations remain literal Track identity and do not independently qualify a Track version.
  Track version comparison rejects trailing segments containing live, remix, remixed, remixes, acoustic, instrumental, radio edit, karaoke, or tribute material.
  With explicit release context, the attached Music Release must match exactly or under the complete Music Release Edition grammar; equivalent attached-release matching accepts only `album` and `single` Candidates.
  Without release context, the attached Album does not constrain matching and no Album search occurs.
  An Equivalent Track Version returns the ordinary `resolved` Result with the accepted playable Spotify Track identity and attached Preferred Music Release metadata unchanged, without match-kind, confidence, or version-relationship fields.
  Release year remains nullable Mention context and does not participate in Spotify retrieval or Candidate verification.
  An unresolved Track preserves its original Track Mention when neither an exact nor an eligible equivalent Candidate appears.
- `results.music_releases[].music_release_mention`: The interpreted direct Music Release title, ordered release artists, and explicit nullable release year.
- `results.music_releases[].music_release`: Spotify-backed enrichment for a resolved Mention, or `null` for an unresolved Mention.
  A resolved Music Release is one independently matched market-available Spotify Album with Spotify's canonical Album title, ordered artist credits, provider-reported `release_date`, `album_type`, a Spotify Album ID and URL, and `cover_url`.
  `cover_url` is the first provider-ordered Spotify-hosted Album image URL or `null` when Spotify returns no Album images, and the Album `spotify_url` is the direct artwork link-back.
  A direct Music Release is never synthesized from a Track's Preferred Music Release.
  Each Music Release Mention causes one Spotify Album search in the effective market with offset zero and limit three.
  The Album query uses the Mention title and first ordered Artist Credit only.
  The same shared normalized Artist Credit eligibility applies.
  Reelio first checks exact title equality under Music Identity Normalization across every artist-eligible Candidate in Spotify order, including Compilations.
  Only when no exact Candidate exists does it reuse that same bounded, artist-eligible Candidate sequence for a controlled Equivalent Music Release Edition pass without another Spotify search.
  The same complete Music Release Edition grammar constrains explicit Track release context through the accepted Track Candidate's attached Album without a separate Album search.
  Equivalent title analysis accepts Remaster, Remastered, Remastered Version, four-digit year Remaster forms, Deluxe, Deluxe Edition, Super Deluxe Edition, Expanded Edition, Special Edition, Anniversary Edition, numeric ordinal Anniversary forms, Reissue, Reissued, Bonus Edition, Bonus Version, Bonus Track, and Bonus Track Version.
  Each designation must occupy a complete trailing segment bounded by parentheses, brackets, a spaced hyphen, or a colon.
  Stacked recognized trailing segments are removed right to left, and the remaining normalized base titles must match exactly.
  This comparison is symmetric between a base title and an eligible edition title, or between two eligible edition titles.
  The equivalent-edition fallback accepts only `album` and `single` Candidates, while a `compilation` remains eligible only for exact-title verification.
  It rejects trailing segments containing live, remix, remixed, remixes, acoustic, instrumental, radio edit, karaoke, tribute, or greatest hits material.
  An Equivalent Track Version or Equivalent Music Release Edition returns the ordinary `resolved` Result with accepted Spotify identity and metadata unchanged, without match-kind, confidence, or edition-relationship fields.
  Release year does not participate in Spotify retrieval or Candidate verification.
  The contract makes no worldwide-edition, sibling-release, release-family, inferred-subtype, or earliest-worldwide-date claims.
  An unresolved Music Release preserves its original Mention when neither an exact nor an eligible equivalent Candidate appears.
- Any TMDB or Spotify provider HTTP, timeout, or required-response validation failure fails the complete request rather than returning partial category results.

Compact success example:

```json
{
  "market": "US",
  "source": {
    "platform": "youtube",
    "video_id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "Movie, TV Series, Track, and Music Release review",
    "description": "A review mentioning a Movie, a TV Series, a Track, and a Music Release.",
    "channel": "Example channel",
    "duration_seconds": 42
  },
  "transcript": {
    "text": "Dune: Part One, The Last of Us, One More Time, and Discovery are excellent.",
    "language": "en",
    "method": "youtube_captions"
  },
  "statistics": {
    "movies": {"n_mentions": 1, "n_resolved": 1, "n_unresolved": 0},
    "tv_series": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    "tracks": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1},
    "music_releases": {"n_mentions": 2, "n_resolved": 1, "n_unresolved": 1}
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
    ],
    "tracks": [
      {
        "status": "resolved",
        "track_mention": {
          "track_title": "One More Time",
          "artists": ["Daft Punk"],
          "release_title": "Discovery",
          "release_year": 2001
        },
        "track": {
          "track_title": "One More Time (2011 Remaster)",
          "artists": [
            {
              "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
              "name": "Daft Punk"
            }
          ],
          "spotify_track_id": "0DiWol3AO6WpXZgp0goxAV",
          "spotify_url": "https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV",
          "preferred_music_release": {
            "release_title": "Discovery (Deluxe Edition)",
            "artists": [
              {
                "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
                "name": "Daft Punk"
              }
            ],
            "release_date": "2001-02-26",
            "album_type": "album",
            "spotify_album_id": "2noRn2Aes5aoNVsU6iWThc",
            "spotify_url": "https://open.spotify.com/album/2noRn2Aes5aoNVsU6iWThc",
            "cover_url": "https://i.scdn.co/image/discovery-cover"
          },
          "cover_url": "https://i.scdn.co/image/discovery-cover"
        }
      },
      {
        "status": "unresolved",
        "track_mention": {
          "track_title": "Unknown Track",
          "artists": ["Unknown Artist"],
          "release_title": null,
          "release_year": null
        },
        "track": null
      }
    ],
    "music_releases": [
      {
        "status": "resolved",
        "music_release_mention": {
          "release_title": "Discovery",
          "artists": ["Daft Punk"],
          "release_year": 2001
        },
        "music_release": {
          "release_title": "Discovery (Deluxe Edition)",
          "artists": [
            {
              "spotify_artist_id": "4tZwfgrHOc3mvqYlEYSvVi",
              "name": "Daft Punk"
            }
          ],
          "release_date": "2001-02-26",
          "album_type": "album",
          "spotify_album_id": "2noRn2Aes5aoNVsU6iWThc",
          "spotify_url": "https://open.spotify.com/album/2noRn2Aes5aoNVsU6iWThc",
          "cover_url": "https://i.scdn.co/image/discovery-cover"
        }
      },
      {
        "status": "unresolved",
        "music_release_mention": {
          "release_title": "Unknown Album",
          "artists": ["Unknown Artist"],
          "release_year": null
        },
        "music_release": null
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
| `502` | Metadata, transcription, LLM, TMDB, or Spotify provider failure. Any TMDB or Spotify failure fails the complete request. |
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
        ├── enrichment/             TMDB and Spotify candidate resolution and enrichment
        ├── interpretation/         OpenAI and DeepSeek structured Mention providers
        └── transcription/          Metadata inspection and transcript acquisition
```
