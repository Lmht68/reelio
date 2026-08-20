# 01 - Add Social Platform Sources Implementation Plan

**Issue:** [`../issues/01-add-social-platform-sources.md`](../issues/01-add-social-platform-sources.md)

**Status:** design confirmed

## Goal

Extend the permanent `POST /api/extract` interface so it accepts public Instagram, Facebook, TikTok, and X Sources that resolve to exactly one finite video.

Use `yt-dlp` for Source metadata and native audio acquisition, route every new platform directly through the existing Whisper path, and preserve YouTube caption-first acquisition with Whisper fallback.

Keep the public Source and Transcript schemas unchanged except for adding the four new serialized `Platform` values.

## Scope

### In scope

- Add `instagram`, `facebook`, `tiktok`, and `x` to the supported `Platform` values.
- Validate submitted URLs against explicit application-owned host and path allowlists before provider access.
- Accept the confirmed official direct and short-link forms for each new platform.
- Resolve social Source metadata through the installed `yt-dlp` adapter.
- Validate the processed extractor identity and canonical webpage URL against the platform inferred from the submitted host.
- Require one processed finite video rather than a playlist, feed, profile, live stream, image-only post, text-only post, or multi-entry post.
- Preserve `video_id` and populate it with the stable processed `yt-dlp` content ID for social Sources.
- Preserve URL-derived YouTube identity and its reconstructed canonical watch URL.
- Normalize social metadata into the existing Source fields.
- Apply the configured duration limit before caption, audio, or Whisper work.
- Route Instagram, Facebook, TikTok, and X directly to native audio download and Whisper.
- Preserve YouTube Caption Track selection and Whisper fallback.
- Apply the existing concurrency-1 Whisper queue, cancellation behavior, cleanup, text normalization, and language detection to every platform.
- Use stable platform-neutral Source error messages and the existing error codes and HTTP statuses.
- Split Source inspection and Transcript acquisition implementation into focused modules under the existing transcription package.
- Add deterministic provider-free coverage for every supported platform and rejection class.
- Verify one public Source from each new platform through the real endpoint during implementation.

### Out of scope

- Platform-specific Caption Track or Transcript providers.
- Native captions or subtitles exposed by `yt-dlp`.
- Authentication, cookies, account sessions, proxying, or restriction bypasses.
- Private, login-gated, age-gated, or region-gated Source support.
- Profiles, feeds, playlists, stories, spaces, broadcasts, live streams, image-only posts, text-only posts, or multi-entry posts.
- Uploaded media.
- Platforms other than YouTube, Instagram, Facebook, TikTok, and X.
- Provider-specific Source fields, engagement counts, media indexes, or schema expansion.
- Changes to Whisper model loading, configured model values, worker count, queue limits, retries, or timeouts.
- Changes to later mention interpretation, enrichment, or placeholder results.
- Permanent tests that contact live providers.
- Changes to the historical Phase 1 specification.

## Confirmed design decisions

| ID | Decision |
|---|---|
| D1 | `ExtractionPipeline.run(url) -> PipelineResult` remains the end-to-end pipeline interface. |
| D2 | The existing `src/reelio/extraction/services/transcription/` package remains the home of Source inspection and Transcript acquisition. |
| D3 | `inspection.py` owns Source URL classification, metadata provider adapters, processed-result validation, Source normalization helpers, and duration-related inspection logic. |
| D4 | `acquisition.py` owns Caption Track adapters, native audio acquisition, Whisper adapters, queueing, cancellation, normalization, and temporary-media cleanup helpers. |
| D5 | `service.py` retains the real public `SourceMetadataService` and `TranscriptionService` orchestration classes rather than becoming an import barrel. |
| D6 | `src/reelio/extraction/service.py::ExtractionPipeline` remains the outer glue that calls Source inspection before Transcript acquisition. |
| D7 | `TranscriptionConfig` and `transcription_settings` remain in `services/transcription/config.py` with every existing field, environment variable, and default unchanged. |
| D8 | `Platform` adds `INSTAGRAM`, `FACEBOOK`, `TIKTOK`, and `X`; `TranscriptMethod` remains unchanged. |
| D9 | The Source response retains `platform`, `video_id`, `url`, `title`, `description`, `channel`, and `duration_seconds`. |
| D10 | Submitted URL trust is established before provider access through exact HTTPS host and path rules, credential rejection, port rejection, and deceptive-host rejection. |
| D11 | Direct and short-link hosts are accepted only through the explicit forms in this plan; `yt-dlp` support alone does not expand Reelio's accepted URL surface. |
| D12 | YouTube remains URL-authoritative: Reelio parses the 11-character ID, reconstructs the canonical watch URL, and checks a returned provider ID when present. |
| D13 | Social Sources become provider-authoritative only after input validation: their non-empty processed `yt-dlp` `id` becomes `video_id`. |
| D14 | Social `Source.url` is the processed, validated `webpage_url`, never the submitted short URL. |
| D15 | Both `x.com` and `twitter.com` serialize as platform `x`; a valid processed canonical host is preserved rather than rewritten. |
| D16 | Processed extractor identities are matched exactly against an application-owned per-platform allowlist. |
| D17 | Shortener extractors must resolve through `yt-dlp` to the final expected video extractor before Reelio accepts the result. |
| D18 | Any processed result containing `entries` is rejected, even when the collection currently contains one entry. |
| D19 | A social Source must expose a non-empty video-bearing format set, a non-empty ID, a validated canonical webpage URL, and a finite non-negative duration. |
| D20 | Live or upcoming content is rejected; absent live metadata or `live_status == "not_live"` is acceptable when every other finite-video invariant holds. |
| D21 | YouTube keeps its current required-title behavior; a missing or blank social title normalizes to an empty string. |
| D22 | Description remains `""` when absent, and channel remains `channel`, then `uploader`, then `""`. |
| D23 | Duration remains required, is rounded up to whole seconds, and is checked before Transcript acquisition. |
| D24 | `TranscriptionService.acquire` branches once on `source.platform`; YouTube tries captions first, while every social platform enters Whisper directly. |
| D25 | No registry, policy flag, null Caption Provider, or platform-specific Transcript adapter is introduced. |
| D26 | Typed metadata timeouts map to the existing `PipelineTimeoutError` and HTTP 504 for all five platforms. |
| D27 | Unsupported URL or processed content shapes map to HTTP 400, unavailable or restricted content maps to 404, over-duration content maps to 413, ordinary provider failure maps to 502, and typed provider timeout maps to 504. |
| D28 | Provider exception text, redirect targets from rejected results, signed media URLs, cookies, headers, and local media details never enter public errors. |
| D29 | The old `test_transcription.py` is cleanly split into `test_inspection.py` and `test_acquisition.py`, then removed after every test is migrated. |
| D30 | Permanent tests use deterministic fakes and never require a network, GPU, model cache, account, cookie file, or live Source. |
| D31 | A restricted live Source is recorded but does not satisfy live success verification; each platform needs one HTTP 200 response with a non-empty Whisper Transcript. |
| D32 | Endpoint OpenAPI wording and extended issue evidence are updated, while historical Phase 1 documentation, `CONTEXT.md`, ADRs, dependencies, and environment variables remain unchanged. |

## Domain language

The existing domain language already covers this feature.

A `Source` remains the canonical identity and normalized metadata for one supported finite video.

Source identity remains the pair of `platform` and stable external `video_id`.

A `Transcript` remains normalized plain text regardless of whether Caption Tracks or Whisper produced it.

`Transcript Method` continues to distinguish `youtube_captions` from `whisper`.

`Transcript Unavailable` remains the result when no acquisition method produces a non-empty Transcript for an otherwise valid Source.

Host rules, extractor keys, provider formats, redirects, and short-link tokens are implementation concepts and do not belong in `CONTEXT.md`.

No new bounded context, domain term, or ADR is required.

## Controlling constraints

### Historical Phase 1 compatibility

The Phase 1 specification intentionally records a YouTube-only scope and must remain unchanged.

The extension must preserve all accepted YouTube URL forms, canonical URL construction, stable ID parsing, Caption Track ranking, Whisper fallback, duration behavior, response fields, error codes, and HTTP statuses.

The only deliberate human-readable YouTube error change is replacing YouTube-specific Source wording with the confirmed platform-neutral wording required by the extended issue.

Typed metadata timeout handling defines a previously inconsistent edge path as HTTP 504 without changing ordinary YouTube provider-failure behavior.

### Existing architectural decisions

`src/reelio/main.py` remains the application composition root and lifespan owner.

One Whisper model, semaphore, TranscriptionService, SourceMetadataService, and ExtractionPipeline remain owned by each entered application lifespan.

The router resolves the pipeline from application state and must not construct providers.

Blocking `yt-dlp`, Caption Track, and faster-whisper operations remain outside the event loop.

### Security boundary

A syntactically valid URL is not trusted merely because `yt-dlp` recognizes it.

Reelio first validates scheme, authority, host, port, credentials, path shape, identifier syntax, and query cardinality.

After provider access, Reelio validates extractor identity and canonical webpage URL before returning or downloading the Source.

A short-link redirect is accepted only when the final processed extractor and canonical URL belong to the expected platform.

### Dependency boundary

The installed `yt-dlp`, `youtube-transcript-api`, faster-whisper, requests, FastAPI, and Pydantic dependencies already cover the implementation.

No dependency or lockfile change is required.

## Current implementation state

- `src/reelio/extraction/types.py::Platform` currently contains only `YOUTUBE`.
- `src/reelio/extraction/schemas.py::Source` already uses the domain `Platform` enum and exposes the required unchanged Source fields.
- `src/reelio/extraction/schemas.py::ExtractRequest` currently documents only a YouTube example.
- `src/reelio/extraction/service.py::ExtractionPipeline.run` already orders Source inspection before Transcript acquisition.
- `src/reelio/extraction/services/transcription/service.py` currently contains both Source inspection and Transcript acquisition implementation in one file.
- `_canonicalize_url` currently accepts only YouTube forms and reconstructs a canonical YouTube watch URL.
- `YtDlpMetadataExtractor.extract` already performs metadata-only access through `extract_info(..., download=False)`.
- Shared `yt-dlp` options currently include quiet output, warning suppression, config isolation, and `noplaylist`.
- `SourceMetadataService.inspect` already runs blocking metadata extraction through `asyncio.to_thread`.
- Metadata normalization currently requires a non-empty title, allows a missing description, falls back from channel to uploader, validates an optional provider ID, and requires finite non-negative duration.
- Duration is already enforced before `TranscriptionService.acquire`.
- `YouTubeCaptionProvider` already isolates `youtube-transcript-api` behind Reelio-owned Caption Track protocols.
- `YtDlpAudioDownloader` already downloads native `bestaudio/best` into a private request directory and rejects multi-entry output.
- `FasterWhisperTranscriber` already consumes the preloaded model and returns normalized text metadata.
- `TranscriptionService` already owns caption-first behavior, the concurrency-1 semaphore, cancellation-safe worker completion, temporary-media cleanup, and Whisper error translation.
- `src/reelio/main.py` already creates production dependencies once per application lifespan.
- Current Source error messages and endpoint documentation still mention YouTube.
- Current deterministic tests already cover YouTube validation, metadata, captions, audio, Whisper, cleanup, cancellation, lifecycle, pipeline ordering, and HTTP error mapping.

## Target file layout

```text
src/reelio/extraction/
├── exceptions.py
├── router.py
├── schemas.py
├── service.py
├── types.py
└── services/
    └── transcription/
        ├── __init__.py
        ├── acquisition.py
        ├── config.py
        ├── inspection.py
        └── service.py

tests/
├── extraction/
│   ├── test_acquisition.py
│   ├── test_inspection.py
│   ├── test_router.py
│   └── test_service.py
├── test_config.py
└── test_main.py
```

`services/transcription/service.py` remains the public orchestration module.

`inspection.py` and `acquisition.py` are internal modules with focused implementation responsibilities.

The split is a clean move, not a second implementation layered beside the old one.

## Target runtime flow

```text
POST /api/extract
  -> router resolves the lifespan-owned ExtractionPipeline
  -> ExtractionPipeline.run(submitted_url)
     -> SourceMetadataService.inspect(submitted_url)
        -> validate generic URL syntax and authority
        -> classify the exact submitted host as one expected Platform
        -> validate the platform-specific direct or short-link path
        -> build a minimal provider request URL
        -> call YtDlpMetadataExtractor in one worker thread
        -> classify timeout, unavailable, unsupported, or ordinary provider failure
        -> reject collection, live, and non-video processed shapes
        -> validate exact extractor_key for the expected Platform
        -> validate canonical webpage_url host and video path
        -> preserve YouTube URL identity or take the social provider ID
        -> normalize title, description, channel, and duration
        -> enforce max_video_duration_seconds
        -> return Source
     -> TranscriptionService.acquire(source)
        -> if platform is youtube
           -> list, rank, and fetch Caption Tracks in one worker thread
           -> return the first non-empty Caption Transcript
           -> otherwise continue to Whisper fallback
        -> if platform is instagram, facebook, tiktok, or x
           -> do not call CaptionProvider
           -> continue directly to Whisper
        -> wait for the lifespan-owned concurrency-1 semaphore
        -> create one unique request directory
        -> download native best audio through yt-dlp
        -> validate the completed local path
        -> run the preloaded Whisper model and exhaust its segments
        -> normalize text and validate language
        -> return Transcript(method=whisper)
        -> remove the complete request directory
     -> retain current placeholder Mention results
  -> serialize the unchanged ExtractResponse shape
```

## Internal contracts

### Public orchestration in `service.py`

`SourceMetadataService.inspect(submitted_url: str) -> Source` remains the only pipeline-facing Source inspection interface.

The class continues to accept the synchronous metadata extractor and `TranscriptionConfig` through its constructor.

Its orchestration responsibilities are:

1. Ask `inspection.py` to classify and normalize the submitted URL into an internal expected-platform request.
2. Run the metadata extractor through `asyncio.to_thread`.
3. Translate typed provider outcomes into stable domain errors.
4. Ask `inspection.py` to validate and normalize the processed result.
5. Emit the existing redacted structured Source metadata event.
6. Enforce the configured duration limit.
7. Return the normalized Source.

`TranscriptionService.acquire(source: Source) -> Transcript` remains the only pipeline-facing Transcript acquisition interface.

Its constructor retains the existing Caption Provider, audio downloader, preloaded transcriber, temporary-media root, and shared semaphore dependencies.

Its orchestration responsibilities are:

1. Branch on `source.platform`.
2. Run Caption Track acquisition only for YouTube.
3. Return a usable YouTube Caption Transcript immediately.
4. Route every social Source directly to Whisper.
5. Map final Whisper timeout and ordinary failure to the existing domain exceptions.

The service module must not duplicate URL parsing, metadata normalization, Caption Track ranking, download, transcriber, or cleanup implementation moved to the focused modules.

The focused modules must not import FastAPI, routers, HTTP schemas, or application state.

### Internal submitted-Source representation

`inspection.py` should use one private frozen, slotted data container for the result of pre-provider validation.

The container should carry only facts needed after validation:

- expected `Platform`;
- minimal provider request URL;
- optional URL-derived YouTube video ID.

Social input IDs may be retained only when needed to reconstruct a minimal provider request URL.

They must not become public Source identity or override the processed `yt-dlp` ID.

Do not expose raw parsed URLs, userinfo, fragments, arbitrary query pairs, or redirect targets through the orchestration interface.

### Generic submitted URL validation

Apply these checks before platform classification:

- The value is non-empty and contains no ASCII control character or whitespace.
- Percent encoding is syntactically well formed.
- The scheme is exactly HTTPS after case normalization.
- A hostname is present.
- Username and password are absent.
- An explicit port is absent, including an explicitly written `:443`.
- The authority contains no ambiguous host syntax.
- The path begins with `/` and contains only the expected number of segments for its platform form.
- Identity-bearing path segments use the form-specific safe ASCII pattern.
- Duplicate or conflicting identity query keys are rejected.
- Fragments never contribute to Source identity.

A host that exactly matches no supported host raises `UnsupportedPlatformError`.

A deceptive suffix or prefix involving a supported platform host raises `InvalidSourceError`.

A supported host with an unsupported path raises `InvalidSourceError` without provider access.

The provider request URL should contain only the validated scheme, expected official host, accepted path, and required identity query keys.

Tracking query values and fragments should not be forwarded.

### Accepted submitted URL forms

#### YouTube

Preserve every currently accepted host and form:

- `youtube.com/watch?v={video_id}`;
- `www.youtube.com/watch?v={video_id}`;
- `m.youtube.com/watch?v={video_id}`;
- `music.youtube.com/watch?v={video_id}`;
- `www.youtube.com/shorts/{video_id}`;
- `www.youtube.com/embed/{video_id}`;
- `www.youtube.com/live/{video_id}`;
- `youtu.be/{video_id}`.

Continue stripping playlist, tracking, and fragment data when reconstructing `https://www.youtube.com/watch?v={video_id}`.

Continue rejecting playlists, channels, profiles, feeds, duplicate IDs, conflicting path and query IDs, and malformed 11-character IDs before provider access.

#### Instagram

Accept only `instagram.com` and `www.instagram.com` with these paths:

- `/p/{shortcode}`;
- `/tv/{shortcode}`;
- `/reel/{shortcode}`;
- `/reels/{shortcode}`.

Allow an optional trailing slash.

Require a non-empty shortcode containing only ASCII letters, digits, underscore, or hyphen.

Reject profiles, stories, tags, audio pages, share paths, arbitrary prefixed paths, and extra path segments.

#### Facebook

Accept only `facebook.com`, `www.facebook.com`, `m.facebook.com`, and `fb.watch`.

Accept these direct forms on a Facebook host:

- `/watch?v={id}`;
- `/reel/{id}`;
- `/{owner}/videos/{id}`;
- `/{owner}/videos/{slug}/{id}`;
- `/{owner}/posts/{id}`;
- `/video.php?v={id}`;
- `/video/video.php?v={id}`.

Accept `fb.watch/{token}` as the only Facebook short-link form.

Allow a numeric ID or `pfbid` token where the installed extractor supports it.

Reject photo, profile, group, event, story, watch-party, plugin, ad-library, general redirect, arbitrary subdomain, and onion forms.

A Facebook post path is only a candidate URL shape; processed-result validation must still prove that it resolves to one video.

#### TikTok

Accept these exact forms:

- `www.tiktok.com/@{user}/video/{numeric_id}`;
- `www.tiktok.com/embed/{numeric_id}`;
- `vm.tiktok.com/{token}`;
- `vt.tiktok.com/{token}`;
- `www.tiktok.com/t/{token}`.

Allow one optional trailing slash and strip tracking query values from the provider request URL.

Reject users, collections, sounds, effects, tags, live paths, mobile live-share paths, Douyin, and extra path segments.

#### X

Accept bare, `www`, `m`, and `mobile` variants of `x.com` and `twitter.com`.

Accept `t.co` as the only X short-link host.

Accept these direct forms:

- `/{user}/status/{numeric_id}`;
- `/i/web/status/{numeric_id}`;
- `/statuses/{numeric_id}`;
- an optional `/video/{positive_index}` suffix on a direct status URL.

Accept `t.co/{token}` and require it to resolve to the final Twitter extractor and an accepted X or Twitter status URL.

Reject cards, spaces, broadcasts, events, photo-index URLs, Amplify URLs, onion hosts, and extra path segments.

### Metadata-only `yt-dlp` adapter

Retain a fresh `yt_dlp.YoutubeDL` context per metadata operation.

Keep quiet output, warning suppression, and configuration isolation.

Use `download=False`.

Do not request subtitles, automatic captions, thumbnails, comments, sidecar metadata, cookies, or authentication.

Use metadata options that preserve collection results so Reelio can reject multi-entry social posts rather than letting `noplaylist` silently select one entry.

The already canonicalized YouTube provider URL contains no playlist query, so preserving collection results does not change accepted YouTube watch behavior.

Keep `noplaylist=True` in the audio downloader as defense in depth after inspection has established one Source.

The adapter returns only a processed mapping or raises a private provider signal.

It does not expose a `YoutubeDL` instance, response payload, or provider exception through the public service interface.

### Exact extractor identity validation

For social Sources, require a non-empty string `extractor_key` and exact membership in this table:

| Expected platform | Accepted processed `extractor_key` |
|---|---|
| Instagram | `Instagram` |
| Facebook | `Facebook`, `FacebookReel` |
| TikTok | `TikTok` |
| X | `Twitter` |

Do not accept case-insensitive matches, prefix matches, the generic extractor, shortener extractors, playlist extractors, or unknown aliases.

A short URL is accepted only after `yt-dlp` has processed its redirect and returned the final expected video extractor.

An absent or non-string extractor key is malformed metadata and maps to 502.

A well-formed but unexpected extractor key is a platform mismatch and maps to 400.

YouTube keeps its current URL-authoritative identity path and does not gain a new extractor-key compatibility requirement in this issue.

### Canonical webpage URL validation

Every social processed result must include a non-empty string `webpage_url`.

Parse it through the same generic HTTPS, credential, port, control-character, and malformed-percent checks used for submitted URLs.

Validate it against the expected platform's accepted direct host and video path rules.

Do not accept a short-link host as the final canonical URL.

Do not accept cross-platform canonical URLs even when the submitted short-link host was valid.

Do not follow another redirect during validation.

Preserve the validated canonical URL string returned by `yt-dlp` after removing a fragment.

Preserve either `x.com` or `twitter.com` for X.

Do not rewrite social URLs from IDs because provider canonical URL structures can change independently of stable content IDs.

A missing or malformed canonical URL maps to 502.

A syntactically valid canonical URL for the wrong host or path maps to 400.

### Exactly-one-finite-video validation

Reject any result that contains an `entries` key.

Do not accept a one-entry playlist, lazy entry collection, transparent collection, or collection with missing entries.

Reject `_type` values representing playlists or URL collections.

Reject `is_live is True`.

Accept `live_status` only when absent or equal to `not_live`.

Reject upcoming, live, post-live, or otherwise non-finite live states.

For social Sources, require `formats` to be a non-empty sequence of mappings.

Require at least one format with a non-empty video codec other than `none`.

An audio-only, image-only, text-only, empty-format, or malformed-format result is not a supported Source.

Treat a recognized provider outcome that explicitly contains no video as unsupported content and map it to 400.

Treat a structurally malformed processed payload as metadata provider failure and map it to 502.

Continue requiring duration before audio download.

A duration of zero remains accepted for compatibility when every other finite-video invariant holds.

### Source identity and metadata normalization

For YouTube:

- Keep the parsed URL video ID as `Source.video_id`.
- Keep the reconstructed canonical watch URL as `Source.url`.
- Allow provider `id` to be absent.
- Require a present provider `id` to equal the parsed ID.
- Require a non-empty string title.

For social Sources:

- Require processed `id` to be a non-empty string after whitespace validation.
- Use the processed `id` exactly as `Source.video_id`.
- Do not substitute a submitted URL token for the processed ID.
- Do not require an input-path ID to equal the processed ID.
- Use the validated processed `webpage_url` as `Source.url`.
- Normalize a missing, null, or blank title to `""`.

For every platform:

- Preserve a non-empty title exactly except for existing whitespace validation.
- Normalize a missing description to `""`.
- Preserve a string description without truncation.
- Select non-empty `channel`, then non-empty `uploader`, then `""`.
- Reject non-string values for title, description, channel, or uploader when the field is present and not null.
- Require duration to be a finite non-negative real number that is not a boolean.
- Round duration upward with `math.ceil`.
- Reject missing, null, string, negative, infinite, or NaN duration as metadata provider failure.

### Duration enforcement

Construct the normalized Source before checking the configured limit so the structured inspection log remains consistent with current behavior.

Compare `duration_seconds` to `max_video_duration_seconds`.

Allow equality.

Raise the existing `DurationLimitExceededError` when the rounded duration is greater than the limit.

Do not call Caption Tracks, the audio downloader, the Whisper transcriber, or later pipeline stages after a duration rejection.

### Source inspection error translation

Use this final outcome table:

| Inspection outcome | Domain error | HTTP | Public message |
|---|---|---:|---|
| Malformed URL, supported host with unsupported path, unsupported processed content shape, or cross-platform redirect. | `InvalidSourceError` | 400 | `Invalid source URL.` |
| Valid HTTPS URL on an unsupported host. | `UnsupportedPlatformError` | 400 | `Only YouTube, Instagram, Facebook, TikTok, and X URLs are supported.` |
| Private, deleted, login-gated, age-gated, region-gated, or otherwise unavailable provider content. | `SourceUnavailableError` | 404 | `Source is unavailable.` |
| Rounded duration exceeds the configured maximum. | `DurationLimitExceededError` | 413 | Keep the existing configured-limit message. |
| Typed metadata-provider timeout. | `PipelineTimeoutError` | 504 | `Source metadata acquisition timed out.` |
| Unknown provider failure, non-mapping payload, absent required provider identity, missing canonical URL, malformed format payload, or invalid required metadata. | `MetadataProviderError` | 502 | `Unable to retrieve source metadata.` |

Inspect nested typed timeout causes before matching unavailable markers.

Keep unavailable matching within the direct `yt-dlp` exception boundary.

Extend markers only for stable inaccessible-content outcomes needed by the four providers.

Do not classify timeouts by human-readable message.

Do not expose the provider message in the raised domain exception.

Do not catch unrelated programming errors outside the provider boundary.

### Transcript routing

Use one explicit branch in `TranscriptionService.acquire`:

```text
if source.platform is Platform.YOUTUBE:
    attempt existing Caption Track acquisition
    return a usable Caption Transcript
    otherwise continue to Whisper
else:
    continue directly to Whisper
```

The non-YouTube branch must not call `CaptionProvider.list_tracks`, construct a YouTube API object, pass the social ID to `youtube-transcript-api`, or treat the deliberate bypass as a Caption Track failure.

Every social success returns `TranscriptMethod.WHISPER`.

Keep Caption Track acquisition outside the Whisper semaphore.

Keep the complete social and fallback Whisper operation inside the shared semaphore.

Do not add a platform routing registry until more than two Transcript acquisition behaviors exist.

### Acquisition implementation move

Move these existing contracts and implementations into `acquisition.py` without changing their observable behavior:

- `CaptionTrack`;
- `CaptionProvider`;
- the YouTube Caption Track adapter;
- `YouTubeCaptionProvider`;
- `AudioDownloader`;
- `YtDlpAudioDownloader`;
- `WhisperResult`;
- `WhisperTranscriber`;
- `FasterWhisperTranscriber`;
- `load_whisper_transcriber`;
- Caption Track ranking and normalization helpers;
- native worker timeout and ordinary-failure signals;
- audio path validation;
- cancellation-safe worker completion;
- temporary request-directory lifecycle.

Keep existing YouTube Caption Track ranking buckets, provider-order tie behavior, no-translation behavior, and formatting-disabled fetches unchanged.

Keep native best-audio format, output-template containment, prepared-path validation, and no-postprocessor behavior unchanged.

Keep the model preload, CUDA preflight, segment exhaustion, detected language, and one-pass normalization unchanged.

### Temporary media, semaphore, and cancellation

Every Whisper operation continues to use one unique `TemporaryDirectory` under `REELIO_TEMP_MEDIA_DIR`.

The directory remains the cleanup boundary for completed audio, partial files, and unexpected `yt-dlp` side files.

Acquire the lifespan-owned semaphore before creating a request directory or native worker.

A request cancelled while queued creates no media.

A request cancelled after native work begins waits for worker completion and cleanup while retaining the semaphore, then propagates cancellation.

Release the semaphore only after cleanup has been attempted.

Do not add a queue timeout, queue limit, retry, background job, or detached worker.

### Logging

Keep the successful Source event name `source metadata normalized`.

Keep these structured fields:

- `stage`;
- sanitized `submitted_url`;
- serialized `platform`;
- `video_id`;
- validated `canonical_url`;
- redacted title, description, and channel values;
- title, description, and channel lengths;
- rounded duration.

Continue redacting sensitive query values and fragments from submitted URL logs.

Do not log provider response bodies, rejected redirect destinations, cookies, headers, signed media URLs, account identifiers beyond normalized public Source fields, or exception strings containing URLs.

Keep successful caption and Whisper logging unchanged.

A social success must emit only the Whisper acquisition success event, never a Caption Track success or failure event.

## File-by-file implementation

### 1. `src/reelio/extraction/types.py`

- Add `INSTAGRAM = "instagram"`, `FACEBOOK = "facebook"`, `TIKTOK = "tiktok"`, and `X = "x"` to `Platform`.
- Keep `YOUTUBE = "youtube"` unchanged.
- Keep `Source`, `Transcript`, `TranscriptMethod`, and every response-related field unchanged.
- Update the Source URL attribute documentation so it no longer says the URL is always reconstructed.
- Do not add aliases such as `TWITTER`.

### 2. `src/reelio/extraction/services/transcription/inspection.py`

- Create the focused Source inspection implementation module.
- Move the metadata extractor protocol and `YtDlpMetadataExtractor` from the old service module.
- Add the private submitted-Source data container.
- Move and generalize generic URL safety checks.
- Keep the existing YouTube parser behavior intact behind the YouTube branch.
- Add the exact Instagram, Facebook, TikTok, and X classifiers from this plan.
- Add application-owned host, direct-path, short-link, and extractor-key tables.
- Add processed canonical URL validation.
- Add collection, live-state, and video-format validation.
- Split metadata-only options from audio-download options so metadata inspection can observe and reject collections.
- Add platform-aware Source identity and title normalization while retaining shared description, channel, and duration rules.
- Add narrowly scoped provider error classification helpers.
- Keep every helper private unless `service.py` genuinely needs it as an internal seam.
- Include complete type hints and Google-style public docstrings.

### 3. `src/reelio/extraction/services/transcription/acquisition.py`

- Create the focused Transcript acquisition implementation module.
- Move Caption Track protocols, YouTube adapters, audio downloader, Whisper adapters, model loader, ranking, normalization, timeout signals, cleanup, and cancellation helpers from the old service module.
- Preserve behavior during the move before adding platform routing in `service.py`.
- Retain `noplaylist=True` for audio download.
- Keep `YtDlpAudioDownloader` generic over the normalized `Source` and use `source.url` for every platform.
- Keep all blocking provider and model work off the event loop.
- Keep complete type hints and Google-style public docstrings.

### 4. `src/reelio/extraction/services/transcription/service.py`

- Reduce the module to the two real public orchestration classes and their direct orchestration logic.
- Retain `SourceMetadataService` constructor and `inspect` interface.
- Retain `TranscriptionService` constructor and `acquire` interface.
- Coordinate focused inspection helpers and adapters without re-exporting them.
- Add the explicit YouTube versus direct-Whisper platform branch.
- Map inspection and acquisition private signals to the stable domain exceptions.
- Keep structured success logging at the orchestration point where normalized domain values are available.
- Delete moved duplicate implementation after every caller has migrated.
- Do not retain deprecated imports, aliases, or compatibility wrappers.

### 5. `src/reelio/extraction/services/transcription/config.py`

- Keep `TranscriptionConfig`, `transcription_settings`, field names, environment prefix, dotenv behavior, types, and defaults unchanged.
- Keep `max_video_duration_seconds` available to `SourceMetadataService`.
- Do not add allowlist, extractor-key, timeout, retry, format, cookie, proxy, or authentication settings.

### 6. `src/reelio/main.py`

- Import `YtDlpMetadataExtractor` from `inspection.py`.
- Import `YouTubeCaptionProvider`, `YtDlpAudioDownloader`, and `load_whisper_transcriber` from `acquisition.py`.
- Continue importing `SourceMetadataService` and `TranscriptionService` from `service.py`.
- Keep production constructor arguments and lifespan ownership unchanged.
- Keep one shared `asyncio.Semaphore(1)`.
- Do not change model loading, application state, router registration, docs gating, or exception handler registration.

### 7. `src/reelio/extraction/service.py`

- Keep the `Pipeline` interface and `ExtractionPipeline.run` ordering unchanged.
- Keep `_SourceMetadataInspector` and `_TranscriptAcquirer` as the structural seams used by tests.
- Keep the placeholder result unchanged.
- Change imports only if module moves require them.
- Do not add platform branching at the pipeline layer.

### 8. `src/reelio/extraction/schemas.py`

- Keep the Source and Transcript field definitions unchanged.
- Rely on the domain `Platform` enum for the expanded serialized values.
- Expand `ExtractRequest.url` OpenAPI examples to include representative YouTube, Instagram, Facebook, TikTok, and X direct forms.
- Do not add provider-specific request fields or discriminated unions.

### 9. `src/reelio/extraction/router.py`

- Change the endpoint summary from YouTube-specific wording to supported Source wording.
- Describe the five supported platforms and the normalized Source plus Transcript result.
- Keep the route, request body, response model, dependency injection, and status unchanged.
- Keep 400, 404, 413, 500, 502, and 504 response entries.
- Use Source and provider wording where video or YouTube wording is unnecessarily narrow.
- Preserve response conversion without adding platform branches.

### 10. `src/reelio/extraction/exceptions.py`

- Keep every exception class, stable code, status code, and handler unchanged.
- Do not add provider-specific public exceptions.
- Pass the confirmed platform-neutral messages from the inspection orchestration layer.

### 11. `tests/extraction/test_inspection.py`

- Create the focused deterministic Source inspection suite.
- Move every current URL, metadata, adapter, duration, provider-error, and Source logging test from `test_transcription.py`.
- Preserve current YouTube cases before adding social cases.
- Add one table-driven accepted direct-form matrix for all platforms.
- Add separate short-link cases for Facebook, TikTok, and X.
- Assert the exact minimal URL passed to the metadata fake.
- Add malformed scheme, credentials, custom port, malformed percent encoding, extra segments, duplicate query identity, deceptive subdomain, and unsupported-host cases for each platform family.
- Assert every pre-provider rejection leaves the extractor fake with zero calls.
- Add exact extractor-key success and mismatch cases per platform.
- Add canonical URL host, path, credential, port, short-host, and cross-platform mismatch cases.
- Add processed ID authority cases for each social platform.
- Prove X and Twitter inputs both serialize as `Platform.X` while preserving the validated processed canonical host.
- Add `entries`, playlist `_type`, live state, empty formats, audio-only formats, image-only results, malformed formats, missing ID, missing canonical URL, and missing duration cases.
- Prove social missing title becomes `""` while YouTube missing title remains 502.
- Preserve description and channel/uploader fallback tests.
- Prove the duration limit applies to every platform and is enforced after one metadata call.
- Add typed metadata timeout tests that produce `PipelineTimeoutError` without message matching.
- Extend unavailable cases with deterministic Instagram, Facebook, TikTok, and X provider errors.
- Assert public domain messages never include provider details, submitted secret query values, or rejected redirect targets.
- Keep `YtDlpMetadataExtractor` option tests deterministic through a fake `YoutubeDL` context.
- Assert metadata options expose collection results and do not request media, captions, cookies, or authentication.

### 12. `tests/extraction/test_acquisition.py`

- Create the focused deterministic Transcript acquisition suite.
- Move every Caption Track, audio, Whisper, logging, semaphore, cleanup, cancellation, and model adapter test from `test_transcription.py`.
- Preserve all current YouTube caption-first and fallback assertions.
- Parameterize Instagram, Facebook, TikTok, and X Sources through the same direct-Whisper contract.
- Use a Caption Provider spy that fails the test if called for any social Source.
- Assert the audio downloader receives the exact normalized Source for each platform.
- Assert every social success returns non-empty normalized text, detected language, and `TranscriptMethod.WHISPER`.
- Preserve ordinary Whisper failure, timeout, empty text, malformed language, and zero-segment mappings.
- Preserve native best-audio, prepared-path containment, multi-entry audio defense, cleanup, queueing, and cancellation tests.
- Keep deterministic synchronization primitives and avoid sleeps for concurrency ordering.
- Assert a social success emits the Whisper success event and no Caption Track event.

### 13. `tests/extraction/test_transcription.py`

- Delete this file only after all existing tests have moved to the two focused suites.
- Do not leave duplicated tests or compatibility imports behind.

### 14. `tests/extraction/test_service.py`

- Preserve Source-before-Transcript ordering and error propagation tests.
- Add or parameterize one social Source with a Whisper Transcript to prove the pipeline is platform-agnostic.
- Assert an inspection rejection prevents Transcript acquisition.
- Do not duplicate URL parsing, provider validation, or Transcript routing tests at this layer.

### 15. `tests/extraction/test_router.py`

- Keep dependency overrides and `ASGITransport` endpoint testing.
- Add one deterministic HTTP 200 serialization case for each new platform.
- Assert the exact Source schema remains unchanged and social Transcript method is `whisper`.
- Preserve YouTube caption and Whisper cases.
- Parameterize exact 400, 404, 413, 502, and 504 response bodies with the platform-neutral messages.
- Assert OpenAPI documents all supported platform values and uses platform-neutral endpoint wording.
- Do not contact `yt-dlp`, social platforms, YouTube, or a real Whisper model.

### 16. `tests/test_main.py`

- Update imports for moved production adapters and model loader.
- Keep model lifecycle, CUDA preflight, pipeline state, and factory call-count assertions unchanged.
- Assert production composition still creates one SourceMetadataService, one TranscriptionService, and one semaphore per lifespan if current tests expose this behavior.
- Do not add platform logic to lifespan tests.

### 17. `tests/test_config.py`

- Keep every existing `TranscriptionConfig` environment and default test unchanged.
- Update imports only if required by the implementation move.
- Add no new configuration cases because the feature adds no setting.

### 18. `docs/phase-01-extended/issues/01-add-social-platform-sources.md`

- Leave acceptance boxes open while implementation is in progress.
- After deterministic checks pass, mark only criteria supported by those checks.
- Append controlled-live evidence for each platform after the real endpoint smoke.
- Record provider restrictions with platform, date, installed `yt-dlp` version, public error code, and non-sensitive summary.
- Do not count a restricted response as the required successful live case.
- Move the issue to `issues/closed/` only after every criterion has proof.

### 19. `docs/phase-01-extended/plans/01-add-social-platform-sources.md`

- Keep this plan synchronized with confirmed behavior if implementation evidence forces a material correction.
- Move it to `plans/closed/` with the issue only after every acceptance criterion passes.
- Do not rewrite historical Phase 1 plans.

## Deterministic test matrix

| Contract | Primary test layer | Required observation |
|---|---|---|
| Existing YouTube URL forms remain accepted. | Inspection. | The same URL-derived ID and canonical watch URL reach the metadata fake. |
| Social direct forms are allowlisted. | Inspection. | Each accepted form reaches the metadata fake once with a minimal provider URL. |
| Official short links are bounded. | Inspection. | Only `fb.watch`, TikTok short hosts, and `t.co` reach the provider, then final identity validation runs. |
| Unsafe URLs stop before provider access. | Inspection. | Credentials, ports, malformed encoding, deceptive hosts, and unsupported paths produce 400 with zero fake calls. |
| Unsupported hosts remain distinct. | Inspection and HTTP. | A valid unsupported host returns `unsupported_platform` and the exact neutral message. |
| Cross-platform redirects are rejected. | Inspection. | A valid short host followed by the wrong extractor or canonical host returns 400. |
| Extractor identity is exact. | Inspection. | Only the confirmed `extractor_key` values succeed. |
| Social identity comes from `yt-dlp`. | Inspection. | `Source.video_id` equals processed `id`, not the submitted path or short token. |
| X aliases normalize correctly. | Inspection and HTTP. | X and Twitter inputs return `platform: "x"` and preserve a valid returned canonical host. |
| Collections are rejected. | Inspection. | Any `entries` key or playlist type returns 400 before acquisition. |
| Live Sources are rejected. | Inspection. | Live and upcoming states return 400 before acquisition. |
| Non-video Sources are rejected. | Inspection. | Empty, malformed, or audio-only format sets return 400 or 502 according to shape validity. |
| Metadata fallback rules are platform-aware. | Inspection. | Social missing title becomes empty; YouTube missing title fails; description and channel fallbacks remain stable. |
| Duration applies everywhere. | Inspection and pipeline. | Each platform allows equality, rejects over-limit with 413, and never calls Transcript acquisition after rejection. |
| Restricted content is unavailable. | Inspection and HTTP. | Known private, deleted, login, age, and region errors map to stable 404 without provider details. |
| Metadata timeouts are terminal. | Inspection and HTTP. | A nested typed timeout maps to stable 504 for every platform. |
| Social Sources bypass captions. | Acquisition. | Caption Provider call count is zero and Whisper is called once. |
| YouTube remains caption-first. | Acquisition. | Caption success returns immediately; caption absence or failure falls back to Whisper. |
| Social Transcripts use Whisper. | Acquisition and HTTP. | Text and language are non-empty and method is `whisper`. |
| Whisper failure semantics remain stable. | Acquisition and HTTP. | Ordinary final failure is 502; terminal timeout is 504; provider details are absent. |
| Whisper remains concurrency-1. | Acquisition. | The second social or fallback request does not start download before the first worker releases. |
| Cancellation preserves cleanup. | Acquisition. | Queued cancellation creates nothing; active cancellation retains the semaphore through cleanup. |
| Native audio remains contained. | Acquisition. | The adapter requests `bestaudio/best` and accepts only one regular file in the private directory. |
| Endpoint schema remains stable. | Router. | Responses contain no provider-specific fields and expose the five Platform enum values. |
| Application lifecycle remains stable. | Main. | One model and pipeline graph are created per entered lifespan. |

## Implementation sequence

1. Move existing Source inspection tests into `test_inspection.py` without changing their assertions.
2. Move existing Caption Track and Whisper tests into `test_acquisition.py` without changing their assertions.
3. Remove `test_transcription.py` after the moved suites collect and preserve every prior test.
4. Add failing deterministic tests for the new Platform enum values and unchanged response schema.
5. Add failing submitted-URL allowlist and pre-provider rejection matrices for every platform.
6. Add failing processed extractor, canonical URL, single-video, live-state, format, identity, metadata fallback, timeout, and duration tests.
7. Move Source inspection implementation into `inspection.py` and keep `SourceMetadataService` orchestration in `service.py`.
8. Implement generic URL safety validation and the platform-specific submitted-form classifiers.
9. Split metadata options from audio options so metadata extraction cannot hide collections.
10. Implement processed extractor, canonical URL, single-video, format, ID, and metadata normalization validation.
11. Add platform-neutral Source messages and metadata timeout translation.
12. Run the focused inspection suite and correct only evidenced contract failures.
13. Move Transcript acquisition implementation into `acquisition.py` and retain `TranscriptionService` orchestration in `service.py`.
14. Add the explicit YouTube caption-first versus social direct-Whisper branch.
15. Run the focused acquisition suite and preserve every existing cleanup, queueing, and cancellation invariant.
16. Update production imports and composition in `main.py` without changing lifecycle ownership.
17. Update pipeline, router, OpenAPI, configuration, and lifecycle tests for moved imports and new platform behavior.
18. Update endpoint examples and platform-neutral documentation wording.
19. Run formatting, lint, strict typing, focused deterministic tests, and the complete deterministic suite.
20. Exercise one public, audible, below-limit Source from each social platform through the real endpoint.
21. Record successful live evidence and any restrictions in the issue.
22. Close and move the issue and plan only after all deterministic and live criteria have proof.

## Verification commands

Run formatting and safe lint fixes once after implementation:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
```

Run formatting, lint, and strict type checks:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src tests
```

Run the focused deterministic checks:

```bash
uv run pytest \
  tests/extraction/test_inspection.py \
  tests/extraction/test_acquisition.py \
  tests/extraction/test_service.py \
  tests/extraction/test_router.py \
  tests/test_main.py \
  tests/test_config.py
```

Run the complete deterministic suite after focused checks pass:

```bash
uv run pytest
```

The deterministic suite must not require provider access, a GPU, a Whisper model cache, FFmpeg, cookies, or account credentials.

## Controlled live endpoint verification

### Preconditions

Use the repository's installed and locked `yt-dlp` version.

Run the actual FastAPI application with a usable configured Whisper device, compute type, and model.

Use the normal lifespan-owned production pipeline rather than dependency overrides.

Use no cookies, account authorization, proxy, VPN bypass, or provider-specific workaround.

Use a dedicated temporary-media root that can be inspected after each request.

Choose one currently public, unauthenticated, finite Source per platform that:

- is below the configured duration limit;
- contains clearly audible speech;
- resolves to exactly one video;
- is not live;
- is not a profile, feed, playlist, or multi-entry post.

Do not store the chosen external URLs in permanent automated tests.

### Per-platform endpoint check

Submit each selected URL to `POST /api/extract`.

Validate the response through the real `ExtractResponse` schema.

Each platform must satisfy all of these conditions:

- HTTP status is 200.
- `source.platform` is the expected serialized platform.
- `source.video_id` is non-empty.
- `source.url` is the validated canonical direct webpage URL and is not the submitted short URL.
- `source.duration_seconds` is at or below the configured limit.
- `transcript.text` is non-empty and recognizably matches audible speech.
- `transcript.language` is non-empty and plausible.
- `transcript.method` is `whisper`.
- No provider-specific Source or Transcript field appears.
- The logged audio path no longer exists after the response.
- The temporary-media root contains no completed request child.
- The application remains healthy for a subsequent request without reloading the model.

Run at least one accepted official short-link form through the controlled endpoint verification when a stable public example is available.

The permanent acceptance threshold remains one successful endpoint response per platform, regardless of whether the successful case used a direct or short URL.

### Restriction evidence

When a public candidate returns a provider restriction:

- confirm Reelio returns the expected stable 404 body;
- record the platform, UTC date, installed `yt-dlp` version, returned error code, and a non-sensitive restriction summary in the issue;
- do not add cookies, credentials, proxying, or bypass logic;
- choose another public candidate;
- keep the live success criterion open until one candidate returns the required HTTP 200 result.

Do not retain live URLs as permanent test fixtures.

## Acceptance traceability

| Issue acceptance criterion | Implementation evidence | Verification proof |
|---|---|---|
| The Platform and API schema expose all five values. | `Platform` enum expansion with unchanged Pydantic Source field. | Enum and endpoint serialization tests for every value. |
| Explicit host allowlists reject unsafe and unsupported URLs before provider access. | Generic authority validation plus exact per-platform classifiers in `inspection.py`. | Accepted and rejected URL matrices with zero-call assertions. |
| Short redirects, extractor identities, and canonical URLs match the expected platform. | Exact `extractor_key` table and final canonical URL validator. | Short-link success and cross-platform mismatch tests. |
| X and Twitter normalize to platform `x`. | Both host families classify as `Platform.X`. | Input alias and canonical-host preservation tests. |
| Social Sources retain the existing fields and stable processed ID. | Platform-aware Source normalization. | Exact domain and HTTP Source assertions per platform. |
| Optional metadata uses confirmed fallbacks while duration remains required. | Platform-aware title rule plus shared description, channel, and duration normalization. | Missing, blank, malformed, and fallback metadata matrix. |
| Exactly one finite video proceeds. | Entry, type, live-state, format, and duration invariants. | Playlist, multi-entry, live, image, text, audio-only, and valid-video tests. |
| Restricted Sources map to 404 without bypasses. | Narrow unavailable classification and unchanged public exception. | Deterministic provider errors plus controlled restriction observations. |
| Duration applies before acquisition. | SourceMetadataService limit check before pipeline progression. | Per-platform over-limit and no-acquisition assertions. |
| Social Sources bypass captions and return Whisper. | Platform branch in `TranscriptionService.acquire`. | Caption spy zero calls, Whisper call, and endpoint method assertions. |
| Native non-YouTube captions remain unused. | No subtitle options and no social Caption Provider branch. | Adapter option and routing tests. |
| Existing Whisper lifecycle and cleanup apply to every platform. | Shared existing Whisper path and lifespan-owned dependencies. | Social queueing, timeout, normalization, language, cleanup, and lifecycle tests. |
| Stable 502 and 504 semantics remain provider-neutral. | Existing exception classes with neutral messages and typed timeout detection. | Exact service and endpoint error-body tests. |
| Existing YouTube behavior remains compatible. | Preserved parser and unchanged caption-first branch. | Complete migrated YouTube regression suite. |
| Endpoint docs are platform-neutral. | Router summary, description, responses, and request examples. | OpenAPI assertions. |
| Permanent tests are deterministic and cover all platforms. | Fake metadata, caption, audio, and Whisper adapters. | Focused and complete suites pass without network access. |
| One public Source per social platform is exercised. | Actual lifespan-owned endpoint and installed provider. | Recorded HTTP 200 evidence for Instagram, Facebook, TikTok, and X. |

## Risks and controls

### `yt-dlp` extractor identity drift

A future `yt-dlp` release may rename an extractor key or change a processed canonical URL.

Exact matching intentionally fails closed.

Update the application allowlist only after source inspection, deterministic contract updates, and controlled live verification.

### Short links can redirect outside the expected platform

Official shorteners can carry arbitrary destinations or warning pages.

Validate the submitted short host before access, then require the final processed extractor and canonical URL to match the expected platform.

Never trust the redirect only because the shortener host is official.

### Collection selection can hide unsupported posts

`noplaylist` can collapse provider collections and undermine the exactly-one-video requirement.

Preserve collection results during metadata inspection and reject any `entries` result.

Retain `noplaylist=True` during audio download only after inspection has established one Source.

### Provider metadata differs by platform

Social extractors do not guarantee identical optional fields.

Keep the Source schema shallow, normalize only confirmed shared fields, and reject missing required identity or duration.

Do not leak raw metadata or add provider-specific branches to the response.

### X post identity can differ from media identity

A status ID and the processed media ID are not interchangeable.

Use the processed `yt-dlp` ID as `video_id` and the validated canonical status URL as `url`.

Do not force equality with the submitted status ID.

### Facebook post paths may resolve to non-video content

A syntactically accepted post URL does not prove video media exists.

Require one processed video-bearing format set after extraction.

Map a recognized non-video result to unsupported Source rather than attempting audio acquisition.

### Unavailable classification relies partly on third-party errors

`yt-dlp` does not provide one stable typed exception for every provider restriction.

Keep message markers narrow, normalized, and contained at the adapter seam.

Typed timeout detection takes precedence, and unknown failures remain 502 rather than being guessed as 404.

### Neutral error wording changes existing YouTube message text

The extended endpoint can no longer truthfully return YouTube-specific Source messages.

Keep machine-readable codes and HTTP statuses unchanged and assert the new exact neutral wording at the HTTP interface.

### Live Sources are unstable verification inputs

Public posts can be deleted, restricted, or changed independently of Reelio.

Use controlled one-time endpoint checks, record the date and provider version, and keep live URLs out of the permanent suite.

### Social Whisper requests consume the shared single slot

Every social request enters Whisper, so social traffic can increase queue occupancy.

Preserve the accepted concurrency-1 invariant rather than adding workers, timeouts, or queue behavior in this issue.

### Provider URLs and media paths are untrusted data

Validate canonical URLs before storing them and validate downloaded paths before stat, logging, or model access.

Keep all media within the private request directory and delete the directory after success, failure, or completed cancellation.

## Documentation decision

No `CONTEXT.md` change is needed because Source, Transcript, Transcript Method, and Transcript Unavailable already describe the feature.

No ADR is needed because the implementation extends existing provider adapters and routing within the Extraction bounded context.

No dependency, lockfile, environment example, or historical Phase 1 documentation change is needed.

The endpoint OpenAPI description and the extended issue provide the required permanent documentation.