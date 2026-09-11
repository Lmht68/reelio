"""Resolve interpreted Music Mentions through the Spotify catalog boundary."""

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.types import AlbumCandidate, TrackCandidate
from reelio.extraction.types import (
    ArtistCredit,
    EnrichedMusicRelease,
    EnrichedTrack,
    MusicMentions,
    MusicReleaseMention,
    MusicReleaseResult,
    MusicResults,
    ResultStatus,
    TrackMention,
    TrackResult,
    normalize_music_identity,
    normalize_music_title_identity,
)

_CANDIDATE_LIMIT = 3
_CANDIDATE_ARTIST_ALIAS_DELIMITER = " / "
_FUZZY_TITLE_SCORE_THRESHOLD = 80.0
_MEDLEY_DESIGNATION = "medley"

_TRACK_AND_RELEASE_EDITION_DESIGNATIONS = frozenset(
    {
        "remaster",
        "remastered",
        "remastered version",
        "bonus edition",
        "bonus version",
        "bonus track",
        "bonus track version",
    }
)
_MUSIC_RELEASE_ONLY_EDITION_DESIGNATIONS = frozenset(
    {
        "deluxe",
        "deluxe edition",
        "super deluxe edition",
        "expanded edition",
        "special edition",
        "anniversary edition",
        "reissue",
        "reissued",
        "extended",
        "extended version",
        "extended edition",
    }
)
_MUSIC_RELEASE_EDITION_DESIGNATIONS = (
    _TRACK_AND_RELEASE_EDITION_DESIGNATIONS | _MUSIC_RELEASE_ONLY_EDITION_DESIGNATIONS
)
_YEAR_REMASTER_DESIGNATION_PATTERN = re.compile(
    r"(?:[0-9]{4} remaster|remastered [0-9]{4}|[0-9]{4} remastered version)"
)
_ORDINAL_ANNIVERSARY_DESIGNATION_PATTERN = re.compile(
    r"[0-9]+(?:st|nd|rd|th) anniversary(?: edition)?"
)
_BLOCKED_TRACK_VERSION_MATERIAL_PATTERN = re.compile(
    r"\b(?:live|remix|remixed|remixes|acoustic|instrumental|karaoke|tribute)\b"
    r"|\bradio(?: |-)edit\b"
)
_BLOCKED_MUSIC_RELEASE_EDITION_MATERIAL_PATTERN = re.compile(
    _BLOCKED_TRACK_VERSION_MATERIAL_PATTERN.pattern + r"|\bgreatest(?: |-)hits?\b"
)
_TRAILING_TITLE_SEGMENT_PATTERNS = (
    re.compile(r"^(?P<base>.+)\((?P<segment>[^()]+)\)$"),
    re.compile(r"^(?P<base>.+)\[(?P<segment>[^\[\]]+)\]$"),
    re.compile(r"^(?P<base>.+) - (?P<segment>.+)$"),
    re.compile(r"^(?P<base>.+):\s*(?P<segment>.+)$"),
)


@dataclass(frozen=True, slots=True)
class _EditionTitleIdentity:
    """Hold a normalized edition-comparison title and its designation state."""

    base_title: str
    designation_removed: bool
    medley_designation_removed: bool


class _MusicCatalog(Protocol):
    """Define the Spotify search capabilities needed for music enrichment."""

    async def search_tracks(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[TrackCandidate, ...]:
        """Return provider-ordered playable Track candidates for one market."""
        ...

    async def search_albums(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[AlbumCandidate, ...]:
        """Return provider-ordered Album candidates for one market."""
        ...


class SpotifyMusicResolver:
    """Resolve grouped Music Mentions against a lifespan-owned Spotify catalog.

    Args:
        catalog: Lifespan-owned catalog that translates Spotify transport failures.
    """

    def __init__(self, catalog: _MusicCatalog) -> None:
        """Initialize resolution with a shared catalog boundary.

        Args:
            catalog: Lifespan-owned Spotify catalog used without taking ownership.
        """
        self._catalog = catalog

    async def resolve(
        self,
        music_mentions: MusicMentions,
        market: SpotifyMarket,
    ) -> MusicResults:
        """Resolve grouped Music Mentions concurrently while preserving per-kind order.

        Args:
            music_mentions: Canonical Music Mentions grouped by resolvable kind.
            market: Effective Spotify market for all catalog searches.

        Returns:
            MusicResults: Resolved or unresolved Results grouped by kind.
        """
        track_candidate_groups, music_release_candidate_groups = await asyncio.gather(
            asyncio.gather(
                *(
                    self._catalog.search_tracks(_build_track_query(track_mention), market)
                    for track_mention in music_mentions.tracks
                )
            ),
            asyncio.gather(
                *(
                    self._catalog.search_albums(_build_album_query(music_release_mention), market)
                    for music_release_mention in music_mentions.music_releases
                )
            ),
        )
        track_results = [
            _resolve_track_mention(track_mention, candidates)
            for track_mention, candidates in zip(
                music_mentions.tracks,
                track_candidate_groups,
                strict=True,
            )
        ]
        music_release_results = [
            _resolve_music_release_mention(music_release_mention, candidates)
            for music_release_mention, candidates in zip(
                music_mentions.music_releases,
                music_release_candidate_groups,
                strict=True,
            )
        ]
        return MusicResults(
            tracks=_drop_duplicate_playable_tracks(track_results),
            music_releases=_drop_duplicate_music_releases(music_release_results),
        )


def _build_track_query(track_mention: TrackMention) -> str:
    """Build one unescaped Spotify field-filter query for a Track Mention."""
    return f"track:{track_mention.track_title} artist:{track_mention.artists[0]}"


def _build_album_query(music_release_mention: MusicReleaseMention) -> str:
    """Build one unescaped Spotify field-filter query for a Music Release Mention."""
    return f"album:{music_release_mention.release_title} artist:{music_release_mention.artists[0]}"


def _to_enriched_music_release(
    album_candidate: AlbumCandidate,
) -> EnrichedMusicRelease:
    """Translate one accepted Spotify Album Candidate into a Music Release."""
    cover_url = album_candidate.images[0].url if album_candidate.images else None
    return EnrichedMusicRelease(
        release_title=album_candidate.title,
        artists=list(album_candidate.artists),
        release_date=album_candidate.release_date,
        album_type=album_candidate.album_type,
        spotify_album_id=album_candidate.spotify_album_id,
        spotify_url=album_candidate.spotify_url,
        cover_url=cover_url,
    )


def _resolve_track_mention(
    track_mention: TrackMention,
    candidates: tuple[TrackCandidate, ...],
) -> TrackResult:
    """Resolve one Track Mention from its provider-ordered bounded candidates."""
    bounded_candidates = candidates[:_CANDIDATE_LIMIT]
    mention_artist_identities = {
        normalize_music_identity(artist) for artist in track_mention.artists
    }
    artist_eligible_candidates = tuple(
        candidate
        for candidate in bounded_candidates
        if _has_artist_credit_match(mention_artist_identities, candidate.artists)
    )
    candidate = next(
        (
            candidate
            for candidate in artist_eligible_candidates
            if _has_exact_track_titles(track_mention, candidate)
        ),
        None,
    )
    if candidate is None:
        mention_track_title_identity = normalize_music_title_identity(track_mention.track_title)
        mention_has_medley_designation = _has_trailing_medley_designation(
            mention_track_title_identity
        )
        mention_track_edition_identity = _track_edition_identity(mention_track_title_identity)
        mention_release_title_identity: str | None = None
        mention_release_edition_identity: _EditionTitleIdentity | None = None
        if track_mention.release_title is not None:
            mention_release_title_identity = normalize_music_title_identity(
                track_mention.release_title
            )
            mention_release_edition_identity = _music_release_edition_identity(
                mention_release_title_identity
            )
        candidate = next(
            (
                candidate
                for candidate in artist_eligible_candidates
                if _has_equivalent_track_titles(
                    mention_track_edition_identity,
                    mention_release_title_identity,
                    mention_release_edition_identity,
                    candidate,
                )
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (
                    candidate
                    for candidate in artist_eligible_candidates
                    if _has_fuzzy_track_titles(
                        mention_track_title_identity,
                        mention_release_title_identity,
                        mention_has_medley_designation,
                        candidate,
                    )
                ),
                None,
            )
    if candidate is None:
        return TrackResult(
            status=ResultStatus.UNRESOLVED,
            track_mention=track_mention,
            track=None,
        )
    preferred_music_release = _to_enriched_music_release(candidate.album)

    return TrackResult(
        status=ResultStatus.RESOLVED,
        track_mention=track_mention,
        track=EnrichedTrack(
            track_title=candidate.title,
            artists=list(candidate.artists),
            spotify_track_id=candidate.spotify_track_id,
            spotify_url=candidate.spotify_url,
            preferred_music_release=preferred_music_release,
            cover_url=preferred_music_release.cover_url,
        ),
    )


def _resolve_music_release_mention(
    music_release_mention: MusicReleaseMention,
    candidates: tuple[AlbumCandidate, ...],
) -> MusicReleaseResult:
    """Resolve one Music Release Mention from its bounded Album candidates."""
    mention_title_identity = normalize_music_title_identity(music_release_mention.release_title)
    bounded_candidates = candidates[:_CANDIDATE_LIMIT]
    mention_artist_identities = {
        normalize_music_identity(artist) for artist in music_release_mention.artists
    }
    artist_eligible_candidates = tuple(
        (candidate, normalize_music_title_identity(candidate.title))
        for candidate in bounded_candidates
        if _has_artist_credit_match(mention_artist_identities, candidate.artists)
    )
    candidate = next(
        (
            candidate
            for candidate, candidate_title_identity in artist_eligible_candidates
            if candidate_title_identity == mention_title_identity
        ),
        None,
    )
    if candidate is None:
        mention_edition_identity = _music_release_edition_identity(mention_title_identity)
        candidate = next(
            (
                candidate
                for candidate, candidate_title_identity in artist_eligible_candidates
                if candidate.album_type in ("album", "single")
                and _has_equivalent_music_release_title(
                    mention_edition_identity,
                    candidate_title_identity,
                )
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (
                    candidate
                    for candidate, candidate_title_identity in artist_eligible_candidates
                    if _has_fuzzy_title_match(
                        mention_title_identity,
                        candidate_title_identity,
                    )
                ),
                None,
            )
    if candidate is None:
        return MusicReleaseResult(
            status=ResultStatus.UNRESOLVED,
            music_release_mention=music_release_mention,
            music_release=None,
        )
    return MusicReleaseResult(
        status=ResultStatus.RESOLVED,
        music_release_mention=music_release_mention,
        music_release=_to_enriched_music_release(candidate),
    )


def _has_exact_track_titles(track_mention: TrackMention, candidate: TrackCandidate) -> bool:
    """Return whether required Track and attached Music Release titles match exactly."""
    if normalize_music_title_identity(track_mention.track_title) != normalize_music_title_identity(
        candidate.title
    ):
        return False
    return track_mention.release_title is None or (
        normalize_music_title_identity(track_mention.release_title)
        == normalize_music_title_identity(candidate.album.title)
    )


def _has_fuzzy_title_match(
    normalized_mention_title: str,
    normalized_candidate_title: str,
) -> bool:
    """Return whether two non-empty normalized titles exceed the fuzzy threshold."""
    return bool(
        normalized_mention_title
        and normalized_candidate_title
        and fuzz.ratio(normalized_mention_title, normalized_candidate_title)
        > _FUZZY_TITLE_SCORE_THRESHOLD
    )


def _has_fuzzy_track_titles(
    mention_track_title_identity: str,
    mention_release_title_identity: str | None,
    mention_has_medley_designation: bool,
    candidate: TrackCandidate,
) -> bool:
    """Return whether required normalized Track and release titles are fuzzy matches."""
    candidate_track_title_identity = normalize_music_title_identity(candidate.title)
    if mention_has_medley_designation or _has_trailing_medley_designation(
        candidate_track_title_identity
    ):
        return _has_fuzzy_medley_track_titles(
            mention_track_title_identity,
            candidate_track_title_identity,
        ) and (
            mention_release_title_identity is None
            or _has_fuzzy_title_match(
                mention_release_title_identity,
                normalize_music_title_identity(candidate.album.title),
            )
        )
    if not _has_fuzzy_title_match(
        mention_track_title_identity,
        candidate_track_title_identity,
    ):
        return False
    return mention_release_title_identity is None or _has_fuzzy_title_match(
        mention_release_title_identity,
        normalize_music_title_identity(candidate.album.title),
    )


def _is_composite_track_base_title(base_title: str) -> bool:
    """Return whether a normalized Track base title names multiple works."""
    title_components = base_title.split("/")
    return len(title_components) >= 2 and all(title_components)


def _has_fuzzy_medley_track_titles(
    mention_track_title_identity: str,
    candidate_track_title_identity: str,
) -> bool:
    """Return whether composite Medley titles have matching fuzzy components."""
    mention_edition_identity = _track_edition_identity(mention_track_title_identity)
    candidate_edition_identity = _track_edition_identity(candidate_track_title_identity)
    if (
        mention_edition_identity is None
        or candidate_edition_identity is None
        or not candidate_edition_identity.medley_designation_removed
        or not _is_composite_track_base_title(mention_edition_identity.base_title)
        or not _is_composite_track_base_title(candidate_edition_identity.base_title)
    ):
        return False
    mention_components = mention_edition_identity.base_title.split("/")
    candidate_components = candidate_edition_identity.base_title.split("/")
    return len(mention_components) == len(candidate_components) and all(
        _has_fuzzy_title_match(mention_component, candidate_component)
        for mention_component, candidate_component in zip(
            mention_components,
            candidate_components,
            strict=True,
        )
    )


def _has_equivalent_track_titles(
    mention_track_edition_identity: _EditionTitleIdentity | None,
    mention_release_title_identity: str | None,
    mention_release_edition_identity: _EditionTitleIdentity | None,
    candidate: TrackCandidate,
) -> bool:
    """Return whether titles differ by controlled Track or release editions."""
    if mention_track_edition_identity is None:
        return False
    candidate_track_edition_identity = _track_edition_identity(
        normalize_music_title_identity(candidate.title)
    )
    if (
        candidate_track_edition_identity is None
        or mention_track_edition_identity.base_title != candidate_track_edition_identity.base_title
    ):
        return False
    if (
        candidate_track_edition_identity.medley_designation_removed
        and not _is_composite_track_base_title(candidate_track_edition_identity.base_title)
    ):
        return False
    if (
        mention_track_edition_identity.medley_designation_removed
        and not candidate_track_edition_identity.medley_designation_removed
    ):
        return False
    if mention_release_title_identity is None:
        return (
            mention_track_edition_identity.designation_removed
            or candidate_track_edition_identity.designation_removed
        )
    candidate_release_title_identity = normalize_music_title_identity(candidate.album.title)
    if mention_release_title_identity == candidate_release_title_identity:
        return (
            mention_track_edition_identity.designation_removed
            or candidate_track_edition_identity.designation_removed
        )
    return candidate.album.album_type in (
        "album",
        "single",
    ) and _has_equivalent_music_release_title(
        mention_release_edition_identity,
        candidate_release_title_identity,
    )


def _split_trailing_title_segment(normalized_title: str) -> tuple[str, str] | None:
    """Return the rightmost supported trailing title segment."""
    longest_match: tuple[str, str] | None = None
    for pattern in _TRAILING_TITLE_SEGMENT_PATTERNS:
        match = pattern.fullmatch(normalized_title)
        if match is None:
            continue
        base_title = match.group("base").strip()
        segment_title = match.group("segment").strip()
        if (
            base_title
            and segment_title
            and (longest_match is None or len(base_title) > len(longest_match[0]))
        ):
            longest_match = (base_title, segment_title)
    return longest_match


def _split_edition_designation_components(segment_title: str) -> tuple[str, ...]:
    """Split one normalized designation segment into slash-separated components."""
    return tuple(segment_title.split("/"))


def _has_trailing_medley_designation(normalized_title: str) -> bool:
    """Return whether a supported trailing segment contains a Medley component."""
    scan_title = normalized_title
    while (trailing_segment := _split_trailing_title_segment(scan_title)) is not None:
        base_title, segment_title = trailing_segment
        if _MEDLEY_DESIGNATION in _split_edition_designation_components(segment_title):
            return True
        scan_title = base_title
    return False


def _is_track_edition_designation_component(segment_title: str) -> bool:
    """Return whether one normalized component names a supported Track version."""
    return (
        segment_title == _MEDLEY_DESIGNATION
        or segment_title in _TRACK_AND_RELEASE_EDITION_DESIGNATIONS
        or _YEAR_REMASTER_DESIGNATION_PATTERN.fullmatch(segment_title) is not None
    )


def _is_music_release_edition_designation_component(segment_title: str) -> bool:
    """Return whether one normalized component names a supported release edition."""
    return (
        segment_title in _MUSIC_RELEASE_EDITION_DESIGNATIONS
        or _YEAR_REMASTER_DESIGNATION_PATTERN.fullmatch(segment_title) is not None
        or _ORDINAL_ANNIVERSARY_DESIGNATION_PATTERN.fullmatch(segment_title) is not None
    )


def _edition_title_identity(
    normalized_title: str,
    is_edition_designation_component: Callable[[str], bool],
    blocked_material_pattern: re.Pattern[str],
) -> _EditionTitleIdentity | None:
    """Return a comparison identity or None when trailing material is excluded."""
    comparison_base = normalized_title
    scan_title = normalized_title
    designation_removed = False
    medley_designation_removed = False
    stripping_designations = True
    while (trailing_segment := _split_trailing_title_segment(scan_title)) is not None:
        base_title, segment_title = trailing_segment
        designation_components = _split_edition_designation_components(segment_title)
        if any(
            blocked_material_pattern.search(component) is not None
            for component in designation_components
        ):
            return None
        if stripping_designations and all(
            is_edition_designation_component(component) for component in designation_components
        ):
            comparison_base = base_title
            designation_removed = True
            medley_designation_removed = (
                medley_designation_removed or _MEDLEY_DESIGNATION in designation_components
            )
        else:
            stripping_designations = False
        scan_title = base_title
    return _EditionTitleIdentity(
        base_title=comparison_base,
        designation_removed=designation_removed,
        medley_designation_removed=medley_designation_removed,
    )


def _track_edition_identity(normalized_title: str) -> _EditionTitleIdentity | None:
    """Return a comparison identity for a Track title."""
    return _edition_title_identity(
        normalized_title,
        _is_track_edition_designation_component,
        _BLOCKED_TRACK_VERSION_MATERIAL_PATTERN,
    )


def _music_release_edition_identity(
    normalized_title: str,
) -> _EditionTitleIdentity | None:
    """Return a comparison identity for a Music Release title."""
    return _edition_title_identity(
        normalized_title,
        _is_music_release_edition_designation_component,
        _BLOCKED_MUSIC_RELEASE_EDITION_MATERIAL_PATTERN,
    )


def _has_equivalent_music_release_title(
    mention_edition_identity: _EditionTitleIdentity | None,
    candidate_title_identity: str,
) -> bool:
    """Return whether titles differ only by controlled Music Release editions."""
    if mention_edition_identity is None:
        return False
    candidate_edition_identity = _music_release_edition_identity(candidate_title_identity)
    return (
        candidate_edition_identity is not None
        and mention_edition_identity.base_title == candidate_edition_identity.base_title
        and (
            mention_edition_identity.designation_removed
            or candidate_edition_identity.designation_removed
        )
    )


def _has_artist_credit_match(
    mention_artist_identities: set[str],
    candidate_artists: Sequence[ArtistCredit],
) -> bool:
    """Return whether a Candidate credit exactly matches a Mention credit or alias."""
    for candidate_artist in candidate_artists:
        candidate_name = candidate_artist.name
        if normalize_music_identity(candidate_name) in mention_artist_identities:
            return True
        if _CANDIDATE_ARTIST_ALIAS_DELIMITER not in candidate_name:
            continue
        if any(
            normalize_music_identity(component) in mention_artist_identities
            for component in candidate_name.split(_CANDIDATE_ARTIST_ALIAS_DELIMITER)
        ):
            return True
    return False


def _drop_duplicate_playable_tracks(results: list[TrackResult]) -> list[TrackResult]:
    """Drop later resolved results with a previously returned playable Track ID."""
    returned_track_ids: set[str] = set()
    deduplicated_results: list[TrackResult] = []
    for result in results:
        if result.track is None:
            deduplicated_results.append(result)
            continue
        if result.track.spotify_track_id in returned_track_ids:
            continue
        returned_track_ids.add(result.track.spotify_track_id)
        deduplicated_results.append(result)
    return deduplicated_results


def _drop_duplicate_music_releases(
    results: list[MusicReleaseResult],
) -> list[MusicReleaseResult]:
    """Drop later resolved results with a previously returned Album ID."""
    returned_album_ids: set[str] = set()
    deduplicated_results: list[MusicReleaseResult] = []
    for result in results:
        if result.music_release is None:
            deduplicated_results.append(result)
            continue
        if result.music_release.spotify_album_id in returned_album_ids:
            continue
        returned_album_ids.add(result.music_release.spotify_album_id)
        deduplicated_results.append(result)
    return deduplicated_results
