# 05 - Whisper Transcript Fallback Implementation Plan

**Issue:** [`../issues/05-whisper-transcript-fallback.md`](../issues/05-whisper-transcript-fallback.md)

**Status:** design confirmed

## Goal

Extend the permanent `TranscriptionService.acquire(source) -> Transcript` boundary so it returns a caption Transcript when possible and otherwise acquires a real Transcript through an audio-only yt-dlp download and the preloaded faster-whisper model.

The fallback must return normalized text, the detected language, and `whisper` as the Transcript Method without changing the public response schema.

The application must preload one model per FastAPI application lifespan, reject an explicitly configured CUDA device when no CUDA device is available, serialize the complete Whisper fallback behind one shared semaphore, and remove request-scoped media on every completed path.

## Scope

### In scope

- Fall back from every ordinary caption failure, unusable Caption Track set, empty caption result, and caption provider timeout.
- Download one Source's native best-audio representation through yt-dlp.
- Transcribe the downloaded audio through faster-whisper.
- Normalize Whisper segment text into the existing plain-text Transcript contract.
- Use faster-whisper's detected language for the Transcript language.
- Return `TranscriptMethod.WHISPER` after successful fallback.
- Preload the configured model during the FastAPI application lifespan.
- Fail startup explicitly when `cuda` is configured and CTranslate2 detects no CUDA devices.
- Let faster-whisper use its standard local-cache or startup-download behavior for model artifacts.
- Own one model, semaphore, transcription service, and extraction pipeline per application lifespan.
- Queue the complete Whisper fallback, including audio download and cleanup, behind a concurrency-1 semaphore.
- Preserve the semaphore and cleanup invariants when an active request is cancelled.
- Use one unique request directory under `REELIO_TEMP_MEDIA_DIR` and delete the complete directory on success and failure.
- Map ordinary dual acquisition failure to Transcript Unavailable through the existing HTTP 502 contract.
- Map only a terminal timeout in the Whisper path to the existing HTTP 504 contract.
- Emit the required structured DEBUG data without logging media bytes or provider payloads.
- Restore `src/reelio/main.py` as the sole application dependency composition root.

### Out of scope

- Configured timeouts, queue timeouts, queue limits, retries, or circuit breakers.
- Background jobs, distributed queues, progress reporting, or cancellation of an already-running native worker thread.
- More than one concurrent Whisper operation per application.
- Multi-GPU scheduling, model sharding, batching, or more than one faster-whisper worker.
- Audio transcoding, FFmpeg post-processing, resampling outside faster-whisper, or a fixed output extension.
- Caption translation or changes to the Caption Track ranking policy.
- Transcript timestamps, word confidence, language probability, persistence, or cross-request caching.
- Uploaded media and platforms other than public YouTube Sources.
- Changes to the public Transcript, response, error body, or Transcript Method schemas.
- Custom model cache configuration or a cache-only deployment mode.
- Changes to later entity extraction, enrichment, or placeholder Mention results.

## Confirmed design decisions

| ID | Decision |
|---|---|
| D1 | `TranscriptionService.acquire(source) -> Transcript` remains the only pipeline-facing Transcript acquisition boundary. |
| D2 | A usable Caption Track returns immediately and never downloads audio or enters the Whisper semaphore. |
| D3 | No usable Caption Track, an ordinary caption provider failure, and a caption provider timeout all attempt Whisper. |
| D4 | The issue 04 rule that a caption timeout immediately becomes HTTP 504 is superseded for this fallback-capable service. |
| D5 | The final attempted acquisition method determines the public failure after fallback starts. |
| D6 | A successful Whisper fallback returns HTTP 200 even when caption acquisition timed out first. |
| D7 | An ordinary Whisper failure or empty Whisper text after any caption failure becomes Transcript Unavailable through `TranscriptionError`, code `transcription_failed`, and HTTP 502. |
| D8 | A terminal timeout in the Whisper download path becomes `PipelineTimeoutError`, code `pipeline_timeout`, and HTTP 504. |
| D9 | One faster-whisper model is loaded for each entered FastAPI application lifespan and is never constructed by a request path. |
| D10 | `src/reelio/main.py` owns model loading and production pipeline construction, consistent with ADR-0003 and the application foundation plan. |
| D11 | The lifespan-owned pipeline is stored on `application.state`, and the router resolves it from the current request instead of a module-level singleton. |
| D12 | A direct `cuda` configuration performs an explicit CTranslate2 device-count check before model construction. |
| D13 | `cpu` and `auto` skip the CUDA-only preflight and let model construction validate the configured device and compute-type combination. |
| D14 | A missing model artifact may be downloaded through faster-whisper's standard Hugging Face cache behavior during startup. |
| D15 | Any model download or model construction failure aborts startup rather than producing a partially initialized application. |
| D16 | The semaphore is created with the lifespan-owned `TranscriptionService` and is therefore shared by every request to that application. |
| D17 | The semaphore covers native best-audio download, faster-whisper decoding and inference, Transcript normalization, success logging, and request-directory cleanup. |
| D18 | A request cancelled while waiting for the semaphore creates no temporary media and leaves the queue normally. |
| D19 | A request cancelled after its worker starts waits for that worker and its cleanup to finish before the semaphore is released, then propagates cancellation. |
| D20 | yt-dlp keeps the provider's native best-audio format and does not invoke an audio post-processor. |
| D21 | Every fallback uses a unique directory below `REELIO_TEMP_MEDIA_DIR`, so deleting the directory removes final audio, partial downloads, and unexpected yt-dlp side files together. |
| D22 | faster-whisper's segment iterator is exhausted inside the blocking worker while the audio file, semaphore, and model are all valid. |
| D23 | Whisper text uses the same whitespace-only normalization semantics as Caption Track text. |
| D24 | Whisper language comes from `TranscriptionInfo.language` and must be a non-empty string. |
| D25 | The successful event remains `transcript acquired`; a Whisper event adds `audio_path` and `audio_size_bytes` while retaining complete text, language, method, and segment count. |
| D26 | No new runtime dependency, configuration field, public exception, public type, glossary term, or ADR is required. |

## Domain language

`CONTEXT.md` already defines the needed concepts precisely.

A `Transcript` remains normalized plain text regardless of whether Caption Tracks or speech transcription supplied it.

`Transcript Method` already includes `youtube_captions` and `whisper`.

`Transcript Unavailable` remains the outcome only when no acquisition method produces a non-empty Transcript for an otherwise valid Source.

A caption acquisition failure is therefore an internal fallback signal after issue 05, not Transcript Unavailable by itself.

Whisper model, audio downloader, temporary directory, timeout, and semaphore are implementation concepts and do not belong in the domain glossary.

No `CONTEXT.md` edit is needed.

No new ADR is needed because this plan follows accepted ADR-0002 and ADR-0003, and the remaining adapter and lifecycle choices are local and reversible.

## Controlling constraints

### ADR-0002

`docs/adr/0002-phase-01-spec-deviations.md` prohibits application-configured timeouts and accepts library defaults.

The implementation must not add timeout settings, wrap the fallback in `asyncio.wait_for`, or add a semaphore wait timeout.

A hung yt-dlp or faster-whisper call may therefore hold the semaphore indefinitely, and later fallback requests may queue indefinitely.

The global duration limit remains enforced immediately after Source metadata extraction and before Caption Track access or audio download.

### ADR-0003

`docs/adr/0003-domain-bounded-context-layout.md` makes `src/reelio/main.py` the application composition root and lifespan owner.

The router may depend on FastAPI request state, but the extraction orchestration and transcription service must not import router or API schema modules.

Transcription internals remain under `src/reelio/extraction/services/transcription/` and may use shared domain result types from `src/reelio/extraction/types.py`.

Module-level settings singletons and import-time settings validation remain unchanged.

## Current implementation state

- `src/reelio/extraction/types.py` already defines `TranscriptMethod.WHISPER`, so no public type change is needed.
- `src/reelio/extraction/services/transcription/config.py` already defines all required environment-backed fields and defaults.
- `pyproject.toml` already declares `faster-whisper>=1.2.1` and `yt-dlp>=2026.7.4` as direct runtime dependencies.
- `uv.lock` already contains the installed dependency graph.
- `YtDlpMetadataExtractor.extract` performs metadata-only access with `download=False`.
- `SourceMetadataService.inspect` runs yt-dlp metadata access in `asyncio.to_thread`, normalizes the Source, and enforces the global duration limit.
- `YouTubeCaptionProvider` isolates the synchronous caption library behind Reelio-owned protocols.
- `TranscriptionService.acquire` currently runs complete Caption Track acquisition in one worker and turns ordinary caption failure into `TranscriptionError` before a fallback can run.
- `_acquire_transcript` currently raises an internal timeout signal that `TranscriptionService.acquire` immediately maps to `PipelineTimeoutError`.
- `ExtractionPipeline.run` already awaits Source inspection before Transcript acquisition.
- `src/reelio/extraction/service.py` currently constructs default production services at module import.
- `src/reelio/extraction/router.py` currently owns a module-level `ExtractionPipeline` singleton.
- `src/reelio/main.py` has an empty lifespan and does not construct the extraction pipeline.
- Existing HTTP schemas and exception handlers already serialize `whisper`, HTTP 502, and HTTP 504 correctly.
- Existing tests cover Caption Track selection, caption normalization, caption provider failures, caption timeouts, pipeline propagation, and endpoint error mapping.
- Existing tests do not run the real FastAPI lifespan and must not load the real model on test hosts.

## Target runtime flow

```text
Process import
  -> validate module-level settings
  -> create FastAPI application without loading a model

FastAPI lifespan startup
  -> if configured device is cuda, query CTranslate2 CUDA device count
  -> fail explicitly when the count is zero
  -> load or download the configured faster-whisper model in a worker thread
  -> build one FasterWhisperTranscriber around that model
  -> build one TranscriptionService and its concurrency-1 semaphore
  -> build one ExtractionPipeline
  -> store the pipeline on application.state

POST /api/extract
  -> router resolves the lifespan-owned Pipeline from request.app.state
  -> ExtractionPipeline.run(url)
     -> SourceMetadataService.inspect(url)
        -> validate and canonicalize the Source URL
        -> retrieve metadata in a worker thread
        -> enforce the configured global duration limit
     -> TranscriptionService.acquire(source)
        -> acquire and rank Caption Tracks in a worker thread
        -> return the first non-empty caption Transcript when available
        -> otherwise enter the shared Whisper semaphore
        -> create one unique request directory under temp_media_dir
        -> download native best audio into that directory
        -> validate the returned audio path and record its size
        -> run faster-whisper and exhaust every segment in the same worker
        -> normalize segment text and validate detected language
        -> emit the successful Whisper DEBUG event
        -> return Transcript(text, language, whisper)
        -> remove the complete request directory
     -> preserve the current placeholder Mention results
  -> serialize the unchanged ExtractResponse contract

FastAPI lifespan shutdown
  -> remove the pipeline from application.state
  -> release references to the pipeline, service, transcriber, semaphore, and model
```

## Internal contracts

### Application pipeline factory

Keep the production factory in `src/reelio/main.py` so all application dependencies have one construction site.

The factory should be asynchronous because model construction is blocking and must run through `asyncio.to_thread`.

A private callable factory seam may be supplied to `create_app` for lifecycle tests.

The default factory must perform these steps exactly once per entered lifespan:

1. Load one faster-whisper transcriber from `transcription_settings` in a worker thread.
2. Construct `SourceMetadataService` with `YtDlpMetadataExtractor` and `transcription_settings`.
3. Construct `TranscriptionService` with `YouTubeCaptionProvider`, `YtDlpAudioDownloader`, the preloaded transcriber, `transcription_settings.temp_media_dir`, and one new semaphore.
4. Construct and return `ExtractionPipeline` with the two required services.

The application object may exist before startup, but `POST /api/extract` is available only while the lifespan-owned pipeline is initialized.

### `AudioDownloader`

Add a Reelio-owned synchronous protocol in `src/reelio/extraction/services/transcription/service.py`:

```python
class AudioDownloader(Protocol):
    def download(self, source: Source, destination: Path) -> Path: ...
```

The protocol accepts the normalized Source and a private, already-created request directory.

It returns the exact completed audio path and does not own deletion of the directory.

It must not expose yt-dlp metadata dictionaries or exceptions to `TranscriptionService`.

### `YtDlpAudioDownloader`

Add a production adapter using a fresh `yt_dlp.YoutubeDL` context per download.

Start from the safe existing yt-dlp options and add only the download-specific values:

- `format` is `bestaudio/best`.
- `outtmpl` is a fixed basename inside the unique request directory with yt-dlp's selected extension placeholder.
- `noplaylist`, `ignoreconfig`, quiet output, and warning suppression remain enabled.
- `download=True` is passed for the canonical Source URL.
- No post-processor, subtitle, thumbnail, metadata sidecar, or conversion option is enabled.

Use yt-dlp's prepared filename for the returned single-video metadata because no post-processing may rename the file.

Validate that the result is a mapping, the prepared path exists as a regular file, and the resolved path remains directly inside the request directory.

Treat missing, multiple, out-of-directory, or non-file results as an ordinary Whisper provider failure.

Translate a `DownloadError` whose preserved `exc_info` or nested network cause is a timeout into a private Whisper timeout signal.

Translate other expected yt-dlp download failures into a private ordinary Whisper provider signal.

Do not classify timeouts by matching human-readable exception messages.

Do not catch unrelated programming failures outside the yt-dlp boundary.

### `WhisperResult`

Add a private frozen, slotted data container for the normalized internal result:

```python
@dataclass(frozen=True, slots=True)
class WhisperResult:
    text: str
    language: str
    segment_count: int
```

This result is internal to the transcription capability and must not move into `extraction/types.py`.

### `WhisperTranscriber`

Add a Reelio-owned synchronous protocol:

```python
class WhisperTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> WhisperResult: ...
```

The protocol represents an already-loaded model and contains no async or FastAPI concern.

Its production implementation owns the faster-whisper model reference but does not construct the model per call.

### `FasterWhisperTranscriber`

Construct the adapter once during application startup around one `faster_whisper.WhisperModel`.

Call `model.transcribe(str(audio_path))` without forcing a language so faster-whisper performs detection.

The faster-whisper call returns a segment iterator and `TranscriptionInfo`.

Consume the complete iterator before returning because decoding and inference continue during iteration.

Normalize each segment's `text` in one pass without retaining audio samples, media bytes, segment objects, timestamps, or probabilities.

Count the original yielded segments for the DEBUG contract.

Validate that `TranscriptionInfo.language` is a non-empty string.

An empty normalized text, invalid language, malformed segment, decoding error, or model inference error is an ordinary Whisper provider failure.

Catch third-party runtime and media-decoding errors only around the direct faster-whisper boundary.

### Model loader

Add one synchronous loader in the transcription service module and call it through `asyncio.to_thread` from the application composition root.

When `settings.whisper_device == "cuda"`, call `ctranslate2.get_cuda_device_count()` before constructing the model.

Raise `RuntimeError("REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available.")` when the count is zero.

Do not perform this explicit preflight for `cpu` or `auto`.

Construct `WhisperModel` with exactly these environment-backed values:

- `model_size_or_path=settings.whisper_model`
- `device=settings.whisper_device`
- `compute_type=settings.whisper_compute_type`

Do not pass `local_files_only`, a custom download root, multiple device indices, multiple workers, or retry settings.

Let the constructor use the normal local cache and download the model when it is absent.

Let any model download, incompatible compute type, corrupted cache, or model construction error abort startup with its chained cause.

### `TranscriptionService`

Make production dependencies required constructor arguments rather than retaining import-time default service objects.

The service should own:

- one `CaptionProvider`;
- one `AudioDownloader`;
- one preloaded `WhisperTranscriber`;
- one configured temporary-media root path;
- one `asyncio.Semaphore(1)`.

`acquire` first runs the complete Caption Track operation through one `asyncio.to_thread` call, preserving the existing same-worker caption session invariant.

A non-empty caption Transcript returns immediately.

A `None` caption result, internal caption provider failure, or internal caption timeout enters the Whisper path.

The service must not raise `TranscriptionError` until the fallback has also failed ordinarily.

The fallback enters the semaphore before creating its worker or request directory.

The complete request-directory lifecycle, audio download, file stat, transcriber call, normalization, logging, and deletion should run inside one synchronous helper invoked by one `asyncio.to_thread` call.

The service maps a private terminal Whisper timeout to `PipelineTimeoutError("Transcript acquisition timed out.")`.

The service maps an ordinary Whisper failure or empty result to `TranscriptionError("Transcript is unavailable for this video.")`.

The service returns a domain `Transcript` with `TranscriptMethod.WHISPER` after a valid Whisper result.

## Caption fallback and error precedence

Use this final-outcome matrix:

| Caption outcome | Whisper outcome | Public outcome |
|---|---|---|
| Non-empty Caption Transcript. | Not attempted. | Return the caption Transcript with `youtube_captions`. |
| Missing, failed, empty, or timed-out captions. | Non-empty Whisper result. | Return the Whisper Transcript with `whisper`. |
| Missing, failed, empty, or timed-out captions. | Ordinary download, decode, inference, payload, or empty-text failure. | Raise Transcript Unavailable through HTTP 502. |
| Missing, failed, empty, or timed-out captions. | Terminal download timeout. | Raise Pipeline Timeout through HTTP 504. |

A caption timeout still stops lower-ranked Caption Track traversal immediately.

The changed behavior is that it now starts Whisper instead of immediately becoming HTTP 504.

Once fallback starts, do not preserve the caption exception as the final public error.

Do not include caption provider details, yt-dlp exception text, local media paths, model internals, or video IDs in the public error message.

## Temporary-media lifecycle

Create `REELIO_TEMP_MEDIA_DIR` with `parents=True` and `exist_ok=True` immediately before the first fallback needs it.

Create one random request directory beneath that root with `tempfile.TemporaryDirectory`.

Use a stable prefix that aids local diagnosis without embedding the submitted URL or credentials.

The request directory is the cleanup boundary, not the final audio file alone.

Keep all yt-dlp output templates inside that directory.

Read file size through `Path.stat()` only after the completed path passes containment and regular-file validation.

Exit the request-directory context after success, download failure, transcription failure, malformed output, logging failure, or cancellation completion.

The configured root remains for reuse, while every request child and all of its contents are removed.

Do not manually unlink only the expected final extension because yt-dlp may create partial or auxiliary files before failing.

A filesystem failure that prevents directory creation or cleanup is an application failure and must not be silently converted into a provider Transcript Unavailable result.

## Semaphore and cancellation behavior

Caption acquisition remains outside the Whisper semaphore.

Acquire the semaphore before starting any fallback download.

Create a task for the `asyncio.to_thread` worker and shield that worker from outer request cancellation.

If the request is cancelled while waiting for the semaphore, propagate cancellation without starting the worker.

If the request is cancelled after the worker starts, retrieve the worker's completion while still holding the semaphore.

Release the semaphore only after the worker has returned or raised and the request-directory context has attempted cleanup.

Then propagate the original cancellation.

This order prevents a cancelled request's native thread from continuing model inference concurrently with the next queued request.

Do not create an untracked background task, release the semaphore early, or assume cancelling the asyncio future stops the native worker thread.

The semaphore has no timeout or queue limit, consistent with ADR-0002.

## Transcript normalization

Reuse one normalization policy for Caption Track strings and Whisper segment strings.

Split each segment on Unicode whitespace, append only non-empty tokens, preserve token and segment order, and join tokens with one ASCII space.

Do not alter punctuation, casing, Unicode characters, or word order.

Do not include timestamps or leading and trailing whitespace.

Normalize the segment iterator in one pass so faster-whisper inference is exhausted once and no duplicate segment-text collection is required.

Keep the caption DEBUG segment count equal to the original provider segment count.

Keep the Whisper DEBUG segment count equal to the number of segment objects yielded by faster-whisper before whitespace normalization.

Treat zero segments and all-whitespace segments as ordinary Whisper failure.

## Logging

Keep the existing successful Caption Track event unchanged:

- Event name: `transcript acquired`.
- `stage`: `transcription`.
- `transcript_text`: complete normalized text.
- `language`: original Caption Track language code.
- `method`: `youtube_captions`.
- `segment_count`: original Caption Track segment count.

Emit the successful Whisper event before the request-directory context removes the audio:

- Event name: `transcript acquired`.
- `stage`: `transcription`.
- `transcript_text`: complete normalized text.
- `language`: detected faster-whisper language.
- `method`: `whisper`.
- `segment_count`: yielded faster-whisper segment count.
- `audio_path`: the validated local file path.
- `audio_size_bytes`: the completed file size in bytes.

A fallback transition may emit an INFO stage event without Caption Track payloads, provider exception text, or media data.

Do not log media bytes, provider response bodies, cookies, headers, signed URLs, exception messages containing URLs, audio content, model cache credentials, or authorization data.

Do not emit a successful `transcript acquired` event for an empty or failed Whisper result.

## File-by-file implementation

### 1. `src/reelio/main.py`

- Import `asyncio` and the production extraction and transcription dependency constructors.
- Add the asynchronous production pipeline factory described above.
- Give `create_app` a private injectable pipeline-factory seam with the production factory as the default.
- Keep `app = create_app()` free of model loading at import time.
- During lifespan startup, await the factory, then assign the returned pipeline to `application.state.extraction_pipeline`.
- During lifespan shutdown, remove the state reference so the pipeline, transcriber, semaphore, and model can be released.
- Keep logging configuration, documentation gating, router registration, and exception handlers unchanged.
- Do not catch and downgrade startup model failures.
- Preserve complete type hints and Google-style public docstrings.

### 2. `src/reelio/extraction/router.py`

- Remove the module-level `_extraction_pipeline` singleton.
- Change `get_pipeline` to accept `fastapi.Request` and return the lifespan-owned pipeline from `request.app.state.extraction_pipeline`.
- Keep `get_pipeline` as the FastAPI dependency override seam used by endpoint tests.
- Keep endpoint serialization and response documentation unchanged.
- Do not construct providers, services, models, or semaphores in the router.

### 3. `src/reelio/extraction/service.py`

- Remove `_DEFAULT_SOURCE_METADATA_SERVICE` and `_DEFAULT_TRANSCRIPTION_SERVICE`.
- Make `ExtractionPipeline` require its source metadata service and transcription service dependencies.
- Keep `run` ordered as Source inspection, then Transcript acquisition, then result assembly.
- Keep the placeholder Mention result branches unchanged.
- Let `TranscriptionError` and `PipelineTimeoutError` propagate unchanged.
- Migrate every constructor call to pass explicit dependencies.

### 4. `src/reelio/extraction/services/transcription/service.py`

- Expand the module responsibility from Source and Caption Track acquisition to complete Transcript acquisition.
- Add the `AudioDownloader` and `WhisperTranscriber` protocols.
- Add the private `WhisperResult` data container and private ordinary-failure and timeout signals.
- Add `YtDlpAudioDownloader` with native best-audio options and strict returned-path validation.
- Add `FasterWhisperTranscriber` around an injected preloaded model.
- Add the synchronous model loader with explicit CUDA availability validation.
- Refactor normalization to consume general segment-text iterables once while retaining segment counts.
- Preserve the current six-bucket Caption Track ranking and caption adapter behavior.
- Change caption timeout handling so it starts Whisper after stopping ranked caption traversal.
- Extend `TranscriptionService` with explicit audio, model, temp-root, and semaphore dependencies.
- Add the synchronous request-directory helper that owns download, transcription, logging, and cleanup.
- Add cancellation-safe semaphore handling around the worker.
- Keep blocking yt-dlp, media decoding, model inference, and generator consumption off the event loop.
- Keep external exception handling narrow and prevent provider details from reaching domain exceptions.

### 5. `src/reelio/extraction/services/transcription/config.py`

- Keep every existing field name, environment variable, type, and default unchanged.
- Do not add timeout, queue, worker-count, model-cache, audio-format, or retry settings.
- Continue using the module-level `transcription_settings` singleton.

### 6. `src/reelio/extraction/types.py`

- Keep `Transcript`, `TranscriptMethod`, and `TranscriptMethod.WHISPER` unchanged.
- Do not add audio paths, byte sizes, segment counts, model metadata, or confidence fields to the domain Transcript.

### 7. `src/reelio/extraction/exceptions.py`

- Keep `TranscriptionError`, `PipelineTimeoutError`, public codes, status codes, and response bodies unchanged.
- Do not add provider-specific public exceptions.

### 8. `tests/extraction/test_transcription.py`

Extend the existing provider fakes with explicit audio downloader and preloaded transcriber fakes.

Add deterministic tests for these observable contracts:

- Caption success never calls the audio downloader or Whisper transcriber.
- No Caption Tracks downloads audio and returns normalized Whisper text, detected language, and method `whisper`.
- An ordinary caption listing failure falls back to Whisper.
- A failed or empty ranked Caption Track set falls back to Whisper.
- A caption listing timeout falls back to Whisper success rather than returning 504.
- A Caption Track fetch timeout stops lower-ranked caption traversal and falls back to Whisper.
- A caption timeout followed by ordinary Whisper failure returns `TranscriptionError`.
- A caption timeout followed by terminal Whisper-path timeout returns `PipelineTimeoutError`.
- An ordinary caption failure followed by terminal Whisper-path timeout returns `PipelineTimeoutError`.
- Zero Whisper segments and all-whitespace Whisper segments return `TranscriptionError`.
- Malformed Whisper language or segment text returns the stable `TranscriptionError` without provider details.
- Unicode whitespace normalizes without changing punctuation, casing, Unicode text, or segment order.
- The production faster-whisper adapter exhausts the segment generator exactly once.
- The production adapter uses `TranscriptionInfo.language` and counts yielded segments.
- yt-dlp receives the canonical Source URL, `download=True`, native best-audio format, and an output template inside the request directory.
- The yt-dlp adapter returns only an existing regular file inside the request directory.
- A missing file, malformed metadata, or out-of-directory path becomes an ordinary fallback failure.
- A typed nested yt-dlp timeout remains distinguishable from an ordinary `DownloadError`.
- Successful fallback removes the request directory and audio file.
- A downloader that writes a partial file and raises still leaves no request directory.
- A transcriber failure after a completed download still leaves no request directory.
- A successful Whisper DEBUG record contains exact path, byte size, complete text, language, method, and segment count.
- Failed or empty Whisper attempts never emit a successful acquisition event.
- Two concurrent fallback acquisitions share one semaphore, the second does not start its download while the first is blocked, and both succeed in order after release.
- Cancellation while queued starts no download and creates no request directory.
- Cancellation during active transcription does not release the semaphore or remove media prematurely, but cleanup completes before cancellation propagates.

Use `threading.Event` or another deterministic synchronization primitive for worker-thread tests.

Do not use sleeps to infer queue ordering.

Use `tmp_path` as the configured media root and assert observable filesystem state rather than private cleanup helper calls.

### 9. `tests/test_main.py`

- Add lifecycle tests through `create_app`'s injected pipeline factory and FastAPI's explicit lifespan context.
- Assert the factory is called once per entered application lifespan.
- Assert the pipeline exists on application state only while the lifespan is active.
- Assert two requests during one lifespan do not invoke the model loader again.
- Patch CTranslate2 device count to zero and assert the exact explicit CUDA startup error.
- Assert the model constructor is not called after failed CUDA preflight.
- Assert `cpu` and `auto` skip the explicit CUDA rejection and pass the configured model, device, and compute type to the loader.
- Assert a model constructor or model download failure aborts lifespan entry.
- Use a fake model or pipeline factory in every deterministic lifecycle test.
- Never download or initialize the real model in the normal test suite.

### 10. `tests/extraction/test_service.py`

- Keep the existing Source-before-Transcript orchestration tests.
- Add or update a method-agnostic pipeline case proving a `whisper` Transcript passes through unchanged.
- Keep failure propagation assertions for `TranscriptionError` and `PipelineTimeoutError`.
- Migrate every `ExtractionPipeline` construction to explicit dependencies.
- Do not duplicate provider, semaphore, temporary-media, or model-lifecycle tests at this orchestration layer.

### 11. `tests/extraction/test_router.py`

- Keep `get_pipeline` dependency overrides for HTTP boundary tests.
- Replace the obsolete no-captions-immediately-502 case with a dual-failure pipeline whose caption and Whisper fakes both fail ordinarily.
- Assert HTTP 502, code `transcription_failed`, and the stable Transcript Unavailable message.
- Add one endpoint success case whose deterministic pipeline returns a real `whisper` Transcript.
- Run two concurrent HTTP requests against one injected pipeline and shared transcription service, hold the first fake Whisper operation, then assert both responses are HTTP 200 and neither contention path returns 5xx.
- Keep the existing parameterized error mapping and OpenAPI response tests unchanged.

### 12. `.env.example`, `pyproject.toml`, and `uv.lock`

- Keep the existing Whisper environment variable names and documented defaults.
- Do not add a model-cache or audio-format variable.
- Do not add a dependency because faster-whisper, yt-dlp, requests, and their required runtime packages are already direct or locked dependencies.
- Change the lockfile only if an implementation command legitimately changes dependency metadata.
- Never edit `uv.lock` by hand.

### 13. `docs/phase-01/issues/05-whisper-transcript-fallback.md`

After implementation and verification succeed:

- Mark an acceptance item complete only when its deterministic or controlled-live proof has passed.
- Move the issue to `docs/phase-01/issues/closed/` only after the real caption-less video smoke test succeeds.
- Move this plan to `docs/phase-01/plans/closed/` in the same issue-closing change.
- Do not close the issue based only on fake-provider tests.

## Deterministic test matrix

| Contract | Primary test layer | Required observation |
|---|---|---|
| Caption success bypasses Whisper. | Transcription service. | The Caption Transcript is returned and fallback fakes have zero calls. |
| Caption absence falls back. | Transcription service. | The returned Transcript has fake model text, detected language, and method `whisper`. |
| Caption timeout falls back. | Transcription service. | Lower-ranked captions stop, Whisper runs, and success is returned. |
| Ordinary dual failure is Transcript Unavailable. | Transcription service and HTTP. | The service raises `TranscriptionError`, and the endpoint returns the stable HTTP 502 body. |
| Terminal Whisper timeout is Pipeline Timeout. | Transcription service and HTTP mapping. | The service raises `PipelineTimeoutError`, and the endpoint contract remains HTTP 504. |
| Model loads once. | Application lifespan. | One loader call occurs for multiple requests inside one entered lifespan. |
| Missing configured CUDA fails startup. | Application lifespan. | Lifespan entry raises the exact explicit error before model construction. |
| CPU and auto remain configurable. | Model loader. | CUDA preflight does not reject them, and constructor arguments match settings. |
| Whisper runs concurrency-1. | Transcription service and concurrent HTTP. | The second fallback waits without starting download, then both calls return successfully. |
| Active cancellation preserves invariants. | Transcription service. | The first worker retains the semaphore through cleanup, and the next worker cannot overlap. |
| Native best audio is used. | yt-dlp adapter. | Options request `bestaudio/best` and contain no post-processor. |
| Output path is controlled. | yt-dlp adapter. | Only an existing regular file directly under the request directory is accepted. |
| Success cleans media. | Filesystem side effect. | The request child directory does not exist after return. |
| Download failure cleans media. | Filesystem side effect. | Partial files and their request directory do not exist after the mapped error. |
| Transcription failure cleans media. | Filesystem side effect. | Completed audio and its request directory do not exist after the mapped error. |
| Empty model output fails. | Transcription service. | Empty normalized text raises the stable `TranscriptionError`. |
| DEBUG payload is complete. | Structured log capture. | Path, bytes, complete text, language, method, and segment count match the fake operation. |
| Public schema is unchanged. | HTTP response validation. | `ExtractResponse` accepts the result and exposes only text, language, and method for Transcript. |

## Implementation sequence

1. Add deterministic audio downloader, model, and synchronization fakes to `tests/extraction/test_transcription.py`.
2. Add failing tests for fallback success, failure precedence, empty output, cleanup, DEBUG fields, semaphore queueing, and cancellation.
3. Add the audio downloader and faster-whisper adapter protocols and production implementations.
4. Add one-pass Whisper normalization and the request-directory worker.
5. Extend `TranscriptionService` with fallback, semaphore, error mapping, and cancellation-safe worker ownership.
6. Update the former immediate caption-timeout tests to the confirmed fallback behavior.
7. Add application lifespan tests with a fake pipeline factory, CUDA device checks, and model-load counting.
8. Move production dependency construction into `src/reelio/main.py` and store the lifespan-owned pipeline on application state.
9. Remove default production services from `extraction/service.py` and the module-level pipeline from `extraction/router.py`.
10. Migrate every pipeline and transcription service constructor call as a clean cutover.
11. Update pipeline and endpoint tests for `whisper`, dual failure, and concurrent HTTP success.
12. Run formatting, lint fixes, strict typing, focused tests, and the full deterministic suite.
13. Run startup failure and real CUDA Whisper smoke scenarios against the actual FastAPI application.
14. Close and move the issue and plan only after every acceptance criterion has proof.

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
uv run pytest tests/extraction/test_transcription.py tests/test_main.py tests/extraction/test_service.py tests/extraction/test_router.py
```

Run the complete deterministic suite after focused checks pass:

```bash
uv run pytest
```

The deterministic suite must use fake models and fake provider boundaries and must not require network access, a GPU, a model cache, or FFmpeg.

## Controlled startup smoke tests

### CUDA unavailable

Run the actual application in an environment with `REELIO_WHISPER_DEVICE=cuda` and no visible CUDA device.

Confirm startup stops before the server accepts traffic and reports exactly:

```text
REELIO_WHISPER_DEVICE is 'cuda', but no CUDA device is available.
```

Confirm no model download or model construction begins after the failed preflight.

### CUDA model preload

Run the actual application on a CUDA-capable host with the default model and compute type.

Allow the first startup to populate faster-whisper's normal model cache when necessary.

Confirm startup completes only after the model is usable.

Send more than one extraction request during the same process lifetime and confirm startup model loading does not recur per request.

## Controlled live fallback smoke test

Select a currently public, short YouTube Source below the configured duration limit that has no usable Caption Track and contains clearly audible speech.

Do not store a permanent external video ID in the deterministic suite because provider availability and caption state can change.

Run the actual FastAPI application on a CUDA-capable host with DEBUG logging and a dedicated temporary-media root.

Call `POST /api/extract` with the selected Source and validate the response through `ExtractResponse`.

The observed response must satisfy all of these conditions:

- HTTP status is 200.
- Source metadata matches the selected YouTube identity.
- Transcript text is non-empty and recognizably matches the spoken audio.
- Transcript language is non-empty and plausible for the audio.
- Transcript method is `whisper`.
- The successful DEBUG event contains the completed audio path, non-zero byte size, complete returned Transcript text, language, method, and segment count.
- The logged request audio path no longer exists after the response.
- The configured media root contains no request child left by the completed fallback.
- The application process remains healthy for a second request without reloading the model.

Run a controlled concurrent smoke only when two short caption-less Sources are available.

Start the second request while the first fallback is active, then confirm both complete successfully and neither returns 5xx because of semaphore contention.

## Acceptance traceability

| Issue acceptance criterion | Implementation evidence | Verification proof |
|---|---|---|
| A caption-less video yields a real Transcript with method `whisper`. | Caption failure enters the native-audio downloader and preloaded faster-whisper adapter inside `TranscriptionService`. | Deterministic fallback response tests plus the controlled live caption-less video smoke. |
| Startup fails fast when CUDA is configured but unavailable. | Model loader checks `ctranslate2.get_cuda_device_count()` before constructing the model. | Exact lifecycle test plus actual no-CUDA startup smoke. |
| The model loads once via lifespan and never per request. | `main.py` creates the transcriber and pipeline once per entered application lifespan and stores the pipeline on app state. | Loader call-count lifecycle test plus repeated-request startup observation. |
| A second concurrent Whisper request queues and succeeds without a contention 5xx. | One lifespan-owned semaphore covers the complete fallback and releases only after worker cleanup. | Deterministic worker gate test, concurrent endpoint test, and optional controlled concurrent smoke. |
| Temporary audio is removed after success and download or transcription failure. | One `TemporaryDirectory` owns all files for the complete synchronous fallback helper. | `tmp_path` tests for success, partial download failure, model failure, and active cancellation. |
| Failure of both captions and Whisper maps to 502 Transcript Unavailable. | Ordinary fallback failure maps to the existing stable `TranscriptionError`. | Service failure-precedence matrix and endpoint exact-body test. |
| DEBUG logs audio path and size plus transcript text and language. | The successful Whisper `transcript acquired` event records all confirmed structured fields before cleanup. | Exact `caplog` field assertions and controlled live log observation. |

## Risks and controls

### Indefinite queue starvation

ADR-0002 accepts that a hung yt-dlp or model call can retain the only semaphore slot indefinitely.

Do not disguise this accepted risk with an undocumented timeout.

### Cancellation does not stop native work

`asyncio.to_thread` cannot stop an already-running native download or model call.

Shield and finish that worker while retaining the semaphore so cancellation cannot violate concurrency or cleanup guarantees.

### Startup may depend on network and model cache state

The confirmed standard faster-whisper behavior may download a large model during startup.

Startup must remain incomplete until loading succeeds, and deployment should warm the standard cache when deterministic startup latency is required.

Do not add cache-only behavior or retry policy in this issue.

### Default CUDA configuration breaks non-GPU startup by design

The default is intentionally `cuda` with `float16`.

Local and test environments without a GPU must use fake lifecycle factories for deterministic tests or explicitly configure a compatible CPU device and compute type for manual runs.

Do not silently fall back from configured CUDA to CPU.

### Provider paths are untrusted data

yt-dlp returns paths derived from provider metadata and output templates.

Resolve and validate the returned file beneath the private request directory before stat, model access, or logging.

### Temp cleanup can fail at the filesystem boundary

A context manager guarantees cleanup attempts but cannot override host permission or filesystem faults.

Do not swallow cleanup exceptions or report success when the request directory could not be removed.

### Model inference is partly lazy

faster-whisper returns a segment generator, and model work continues while it is consumed.

Exhaust the generator inside the worker, semaphore, and temporary-directory contexts.

### GPU memory is application-lifetime state

Multiple FastAPI app instances in one process each own a model by confirmed design.

Production should create one application instance per process unless it intentionally budgets GPU memory for more.

## Documentation decision

No glossary update is needed because the existing Transcript, Transcript Method, and Transcript Unavailable definitions already describe the confirmed behavior.

No ADR is needed because the plan implements existing architectural decisions and introduces no hard-to-reverse or surprising cross-context trade-off.
