"""Resolve interpreted Music Mentions through the Spotify catalog boundary."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

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
)

_CANDIDATE_LIMIT = 3


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
        if _has_shared_artist_credit(mention_artist_identities, candidate.artists)
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
    bounded_candidates = candidates[:_CANDIDATE_LIMIT]
    mention_artist_identities = {
        normalize_music_identity(artist) for artist in music_release_mention.artists
    }
    artist_eligible_candidates = tuple(
        candidate
        for candidate in bounded_candidates
        if _has_shared_artist_credit(mention_artist_identities, candidate.artists)
    )
    candidate = next(
        (
            candidate
            for candidate in artist_eligible_candidates
            if _has_exact_music_release_title(music_release_mention, candidate)
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
    if normalize_music_identity(track_mention.track_title) != normalize_music_identity(
        candidate.title
    ):
        return False
    return track_mention.release_title is None or (
        normalize_music_identity(track_mention.release_title)
        == normalize_music_identity(candidate.album.title)
    )


def _has_exact_music_release_title(
    music_release_mention: MusicReleaseMention,
    candidate: AlbumCandidate,
) -> bool:
    """Return whether a Music Release title matches exactly."""
    return normalize_music_identity(
        music_release_mention.release_title
    ) == normalize_music_identity(candidate.title)


def _has_shared_artist_credit(
    mention_artist_identities: set[str],
    candidate_artists: Sequence[ArtistCredit],
) -> bool:
    """Return whether a Candidate shares a normalized Artist Credit with a Mention."""
    return any(
        normalize_music_identity(candidate_artist.name) in mention_artist_identities
        for candidate_artist in candidate_artists
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
