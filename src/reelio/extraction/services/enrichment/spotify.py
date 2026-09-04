"""Resolve interpreted Track Mentions through the Spotify catalog boundary."""

import asyncio
from difflib import SequenceMatcher
from typing import Protocol

from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.types import TrackCandidate
from reelio.extraction.types import (
    EnrichedTrack,
    ResultStatus,
    TrackMention,
    TrackResult,
    normalize_music_identity,
)

_FUZZY_MATCH_THRESHOLD = 0.90
_CANDIDATE_LIMIT = 3


class _TrackCatalog(Protocol):
    """Define the Spotify Track search capability needed for enrichment."""

    async def search_tracks(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[TrackCandidate, ...]:
        """Return provider-ordered playable Track candidates for one market."""
        ...


class SpotifyTrackResolver:
    """Resolve Track Mentions against a lifespan-owned Spotify catalog.

    Args:
        catalog: Lifespan-owned catalog that translates Spotify transport failures.
    """

    def __init__(self, catalog: _TrackCatalog) -> None:
        """Initialize resolution with a shared catalog boundary.

        Args:
            catalog: Lifespan-owned Spotify catalog used without taking ownership.
        """
        self._catalog = catalog

    async def resolve(
        self,
        track_mentions: list[TrackMention],
        market: SpotifyMarket,
    ) -> list[TrackResult]:
        """Resolve Track Mentions concurrently while preserving input order.

        Args:
            track_mentions: Ordered canonical Track Mentions to resolve.
            market: Effective Spotify market for all catalog searches.

        Returns:
            list[TrackResult]: Resolved or unresolved results in first-reference order.
        """
        if not track_mentions:
            return []

        candidate_groups = await asyncio.gather(
            *(
                self._catalog.search_tracks(_build_track_query(track_mention), market)
                for track_mention in track_mentions
            )
        )
        results = [
            _resolve_track_mention(track_mention, candidates)
            for track_mention, candidates in zip(
                track_mentions,
                candidate_groups,
                strict=True,
            )
        ]
        return _drop_duplicate_playable_tracks(results)


def _build_track_query(track_mention: TrackMention) -> str:
    """Build one unescaped Spotify field-filter query for a Track Mention."""
    query_terms = [
        f"track:{track_mention.track_title}",
        *(f"artist:{artist}" for artist in track_mention.artists),
    ]
    if track_mention.release_title is not None:
        query_terms.append(f"album:{track_mention.release_title}")
    if track_mention.release_year is not None:
        query_terms.append(f"year:{track_mention.release_year}")
    return " ".join(query_terms)


def _resolve_track_mention(
    track_mention: TrackMention,
    candidates: tuple[TrackCandidate, ...],
) -> TrackResult:
    """Resolve one Track Mention from its provider-ordered bounded candidates."""
    bounded_candidates = candidates[:_CANDIDATE_LIMIT]
    candidate = next(
        (
            candidate
            for candidate in bounded_candidates
            if _is_exact_match(track_mention, candidate)
        ),
        None,
    )
    if candidate is None:
        candidate = next(
            (
                candidate
                for candidate in bounded_candidates
                if _is_fuzzy_match(track_mention, candidate)
            ),
            None,
        )
    if candidate is None:
        return TrackResult(
            status=ResultStatus.UNRESOLVED,
            track_mention=track_mention,
            track=None,
        )
    return TrackResult(
        status=ResultStatus.RESOLVED,
        track_mention=track_mention,
        track=EnrichedTrack(
            track_title=candidate.title,
            artists=list(candidate.artists),
            spotify_track_id=candidate.spotify_track_id,
            spotify_url=candidate.spotify_url,
        ),
    )


def _is_exact_match(track_mention: TrackMention, candidate: TrackCandidate) -> bool:
    """Return whether every supplied Track identity field matches exactly."""
    if normalize_music_identity(track_mention.track_title) != normalize_music_identity(
        candidate.title
    ):
        return False
    if not _has_matching_artist_sequence(track_mention, candidate):
        return False
    if track_mention.release_title is not None and (
        normalize_music_identity(track_mention.release_title)
        != normalize_music_identity(candidate.album.title)
    ):
        return False
    return track_mention.release_year is None or track_mention.release_year == int(
        candidate.album.release_date[:4]
    )


def _is_fuzzy_match(track_mention: TrackMention, candidate: TrackCandidate) -> bool:
    """Return whether every applicable textual Track field clears the fuzzy threshold."""
    if len(track_mention.artists) != len(candidate.artists):
        return False
    if not _has_fuzzy_text_match(track_mention.track_title, candidate.title):
        return False
    if any(
        not _has_fuzzy_text_match(mention_artist, candidate_artist.name)
        for mention_artist, candidate_artist in zip(
            track_mention.artists,
            candidate.artists,
            strict=True,
        )
    ):
        return False
    return track_mention.release_title is None or _has_fuzzy_text_match(
        track_mention.release_title,
        candidate.album.title,
    )


def _has_matching_artist_sequence(
    track_mention: TrackMention,
    candidate: TrackCandidate,
) -> bool:
    """Return whether artist credits have equal length and positional identities."""
    if len(track_mention.artists) != len(candidate.artists):
        return False
    return all(
        normalize_music_identity(mention_artist) == normalize_music_identity(candidate_artist.name)
        for mention_artist, candidate_artist in zip(
            track_mention.artists,
            candidate.artists,
            strict=True,
        )
    )


def _has_fuzzy_text_match(left: str, right: str) -> bool:
    """Return whether case-insensitive normalized text meets the inclusive threshold."""
    return (
        SequenceMatcher(
            a=normalize_music_identity(left),
            b=normalize_music_identity(right),
            autojunk=False,
        ).ratio()
        >= _FUZZY_MATCH_THRESHOLD
    )


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
