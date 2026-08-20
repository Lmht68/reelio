# 01 - Add Social Platform Sources

**What to build:** Extend `POST /api/extract` to accept public Instagram, Facebook, TikTok, and X Sources that resolve through `yt-dlp` to exactly one finite video. Normalize each Source into the existing response schema, acquire its Transcript directly through the existing Whisper path without calling `youtube-transcript-api`, and preserve all existing YouTube behavior.

**Status:** ready-for-human

- [x] `Platform` and the API schema expose `instagram`, `facebook`, `tiktok`, and `x` in addition to `youtube`
- [x] Explicit host allowlists accept each platform's supported official video and short-link URL forms while rejecting malformed URLs, credentials, custom ports, deceptive subdomains, and unsupported hosts before provider access
- [x] Redirected short URLs, `yt-dlp` extractor identities, and canonical webpage URLs are validated against the expected platform before a Source is returned
- [x] Both `x.com` and `twitter.com` Sources normalize to platform `x`
- [x] Instagram, Facebook, TikTok, and X Sources return the existing Source fields, retain the `video_id` field with the stable `yt-dlp` content ID, and use the validated canonical webpage URL
- [x] Missing title, description, or channel metadata uses the existing normalized fallback rules, while missing duration remains a metadata acquisition error
- [x] Public Sources resolving to exactly one finite video proceed, while profiles, feeds, playlists, live streams, image-only or text-only posts, and multi-video posts are rejected
- [x] Private, deleted, login-gated, age-gated, and region-gated Sources map to the existing 404 unavailable response without adding authentication, cookies, proxying, or restriction bypasses
- [x] The configured duration limit applies to every platform and rejects an over-limit Source with 413 before audio download or Whisper transcription
- [ ] Instagram, Facebook, TikTok, and X bypass `youtube-transcript-api`, use `yt-dlp` audio download, and return a non-empty Transcript with method `whisper`
- [x] Native non-YouTube captions and subtitles are not acquired through platform APIs or `yt-dlp`
- [x] Existing Whisper model loading, concurrency-1 queueing, timeout handling, text normalization, language detection, and temporary media cleanup apply to every new platform
- [x] Metadata, download, or Whisper failure that leaves no non-empty Transcript maps to 502 `TranscriptUnavailable`, provider timeout maps to 504, and provider details do not leak
- [x] Existing YouTube URL normalization, caption-first acquisition, Whisper fallback, limits, errors, and response behavior remain unchanged
- [x] Endpoint documentation and stable error messages use platform-neutral wording and include the newly supported Source platforms
- [x] Deterministic automated tests cover normal video URLs, accepted official short-link forms, Source normalization, direct Whisper routing, rejection paths, limits, and errors for all four platforms without live provider calls
- [ ] Implementation verification exercises one public Source from Instagram, Facebook, TikTok, and X through the real endpoint with the installed `yt-dlp` version, and any provider restriction encountered is recorded without adding a permanent live-network test

## Implementation evidence

The implementation keeps `ExtractionPipeline.run(url) -> PipelineResult`, splits Source inspection into `inspection.py`, splits Transcript acquisition into `acquisition.py`, and keeps the two pipeline-facing orchestration classes in `service.py`.

The deterministic suite passed with 234 tests through `uv run pytest -q`.

Strict type checking passed through `uv run mypy src tests`.

Ruff checks and formatting passed through `uv run ruff check src tests` and `uv run ruff format --check src tests`.

The installed provider version used for live checks was `yt-dlp 2026.07.04`.

The live checks used the real endpoint, public URLs, no cookies, no authentication, no proxying, and no restriction bypass.

### Successful live checks

Facebook `https://www.facebook.com/NASA/videos/nasa-astronaut-don-pettit-turns-the-camera-on-science/696829239355289/` returned HTTP 200 with platform `facebook`, a non-empty Whisper Transcript, and `Transcript.method == "whisper"`.

TikTok `https://www.tiktok.com/@patroxofficial/video/6742501081818877190` returned HTTP 200 with platform `tiktok`, a non-empty Whisper Transcript, and `Transcript.method == "whisper"`.

X `https://x.com/historyinmemes/status/1790637656616943991` returned HTTP 200 with platform `x`, the processed media ID, a non-empty Whisper Transcript, and `Transcript.method == "whisper"`.

### Unmet live check

Instagram `https://www.instagram.com/reel/DcPNilkArHt/` returned HTTP 502 with the stable `metadata_provider_failed` response because the installed `yt-dlp` extractor returned a video result with no finite duration.

The Instagram response is not counted as the required successful live verification.

### Provider restriction evidence

TikTok `https://www.tiktok.com/@denidil6/video/7065799023130643713` returned a provider `DownloadError` stating that the IP address was blocked from accessing the post, so it was not counted as a successful live verification.

The implementation does not add a permanent live-network test for any provider.
