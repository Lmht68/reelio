"""Spotify Track resolver contract tests."""

from collections import deque
from collections.abc import Sequence

import pytest

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.types import (
    AlbumCandidate,
    ImageCandidate,
    TrackCandidate,
)
from reelio.extraction.services.enrichment.spotify import SpotifyMusicResolver
from reelio.extraction.types import (
    ArtistCredit,
    MusicMentions,
    MusicReleaseMention,
    ResultStatus,
    TrackMention,
    TrackResult,
)

_MARKET = SpotifyMarket("JP")


class _FakeTrackCatalog:
    """Return deterministic provider-ordered Track Candidates."""

    def __init__(
        self,
        candidate_groups: Sequence[tuple[TrackCandidate, ...]] = (),
        album_candidate_groups: Sequence[tuple[AlbumCandidate, ...]] = (),
        error: CatalogProviderError | PipelineTimeoutError | None = None,
    ) -> None:
        """Configure candidate groups or one operational error.

        Args:
            candidate_groups: Track search results returned in call order.
            album_candidate_groups: Album search results returned in call order.
            error: Typed catalog failure raised on every search when supplied.
        """
        self._candidate_groups = deque(candidate_groups)
        self._album_candidate_groups = deque(album_candidate_groups)
        self._error = error
        self.calls: list[tuple[str, SpotifyMarket]] = []
        self.album_calls: list[tuple[str, SpotifyMarket]] = []

    async def search_tracks(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[TrackCandidate, ...]:
        """Record a search and return its configured provider result."""
        self.calls.append((query, market))
        if self._error is not None:
            raise self._error
        if not self._candidate_groups:
            return ()
        return self._candidate_groups.popleft()

    async def search_albums(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[AlbumCandidate, ...]:
        """Record a search and return its configured provider result."""
        self.album_calls.append((query, market))
        if self._error is not None:
            raise self._error
        if not self._album_candidate_groups:
            return ()
        return self._album_candidate_groups.popleft()


def _mention(
    track_title: str = "One More Time",
    artists: Sequence[str] = ("Daft Punk",),
    release_title: str | None = None,
    release_year: int | None = None,
) -> TrackMention:
    """Create one canonical Track Mention for resolver tests."""
    return TrackMention(
        track_title=track_title,
        artists=list(artists),
        release_title=release_title,
        release_year=release_year,
    )


def _album_candidate(
    title: str = "Discovery",
    release_date: str = "2001-02-26",
    images: tuple[ImageCandidate, ...] = (),
) -> AlbumCandidate:
    """Create one attached Spotify Album Candidate for resolver tests."""
    return AlbumCandidate(
        spotify_album_id="album-1",
        spotify_url="https://open.spotify.com/album/album-1",
        title=title,
        artists=(ArtistCredit(spotify_artist_id="album-artist", name="Album Artist"),),
        release_date=release_date,
        album_type="album",
        images=images,
    )


def _candidate(
    spotify_track_id: str = "track-1",
    title: str = "One More Time",
    artists: Sequence[str] = ("Daft Punk",),
    album: AlbumCandidate | None = None,
) -> TrackCandidate:
    """Create one playable Spotify Track Candidate for resolver tests."""
    return TrackCandidate(
        spotify_track_id=spotify_track_id,
        spotify_url=f"https://open.spotify.com/track/{spotify_track_id}",
        title=title,
        artists=tuple(
            ArtistCredit(spotify_artist_id=f"artist-{index}", name=artist)
            for index, artist in enumerate(artists)
        ),
        album=_album_candidate() if album is None else album,
    )


async def _resolve_tracks(
    resolver: SpotifyMusicResolver,
    track_mentions: list[TrackMention],
) -> list[TrackResult]:
    """Resolve Track Mentions through the grouped Music Resolver interface."""
    music_results = await resolver.resolve(
        MusicMentions(tracks=track_mentions, music_releases=[]),
        _MARKET,
    )
    return music_results.tracks


async def test_resolver_resolves_grouped_music_mentions() -> None:
    """Resolve Tracks and Music Releases from one grouped Music Mention input."""
    track_mention = _mention()
    music_release_mention = MusicReleaseMention(
        release_title="Discovery",
        artists=["Daft Punk"],
        release_year=2001,
    )
    album_candidate = AlbumCandidate(
        spotify_album_id="album-identity",
        spotify_url="https://open.spotify.com/album/album-identity",
        title="Discovery",
        artists=(ArtistCredit(spotify_artist_id="artist-0", name="Daft Punk"),),
        release_date="2001-02-26",
        album_type="album",
        images=(),
    )
    catalog = _FakeTrackCatalog(
        candidate_groups=((_candidate(),),),
        album_candidate_groups=((album_candidate,),),
    )

    results = await SpotifyMusicResolver(catalog).resolve(
        MusicMentions(
            tracks=[track_mention],
            music_releases=[music_release_mention],
        ),
        _MARKET,
    )

    assert catalog.calls == [
        ("track:One More Time artist:Daft Punk", _MARKET),
    ]
    assert catalog.album_calls == [
        ("album:Discovery artist:Daft Punk year:2001", _MARKET),
    ]
    assert results.tracks[0].track is not None
    assert results.tracks[0].track.spotify_track_id == "track-1"
    assert results.music_releases[0].music_release is not None
    assert results.music_releases[0].music_release.spotify_album_id == "album-identity"


async def test_resolver_builds_field_scoped_query_and_forwards_effective_market() -> None:
    """Pass an unescaped ordered field query and effective market to Spotify."""
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    title="One More Time (Radio Edit)",
                    artists=("Daft Punk", "Romanthony"),
                ),
            ),
        )
    )
    resolver = SpotifyMusicResolver(catalog)
    mention = _mention(
        track_title="One More Time (Radio Edit)",
        artists=("Daft Punk", "Romanthony"),
        release_title="Discovery",
        release_year=2001,
    )

    results = await _resolve_tracks(resolver, [mention])

    assert catalog.calls == [
        (
            "track:One More Time (Radio Edit) artist:Daft Punk artist:Romanthony "
            "album:Discovery year:2001",
            _MARKET,
        )
    ]
    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_ignores_album_context_when_the_mention_omits_it() -> None:
    """Resolve an exact Track despite unrelated attached Album metadata."""
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    album=_album_candidate(
                        title="Unrelated Release",
                        release_date="2025-01-01",
                    )
                ),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [_mention()])

    assert results[0].status is ResultStatus.RESOLVED
    assert results[0].track is not None
    assert results[0].track.preferred_music_release.release_title == "Unrelated Release"
    assert catalog.album_calls == []


async def test_resolver_inspects_only_the_first_three_candidates() -> None:
    """Leave a fourth exact candidate outside the resolver's hard search bound."""
    candidates = (
        _candidate("track-1", title="Wrong One"),
        _candidate("track-2", title="Wrong Two"),
        _candidate("track-3", title="Wrong Three"),
        _candidate("track-4"),
    )

    results = await _resolve_tracks(
        SpotifyMusicResolver(_FakeTrackCatalog((candidates,))), [_mention()]
    )

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].track_mention == _mention()
    assert results[0].track is None


async def test_resolver_searches_all_exact_candidates_before_fuzzy_candidates() -> None:
    """Choose a later exact Candidate over an earlier fuzzy Candidate."""
    mention = _mention(track_title="Midnight Echo")
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate("fuzzy", title="Midnight Echos"),
                _candidate("exact", title="Midnight Echo"),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].track is not None
    assert results[0].track.spotify_track_id == "exact"


@pytest.mark.parametrize(
    "candidate_artists",
    [
        ("Second Artist", "First Artist"),
        ("First Artist",),
    ],
    ids=["reversed-order", "different-length"],
)
async def test_fuzzy_matching_requires_same_length_positional_artist_sequence(
    candidate_artists: Sequence[str],
) -> None:
    """Reject fuzzy candidates whose artist credits differ by order or length."""
    mention = _mention(
        track_title="abcdefghij",
        artists=("First Artist", "Second Artist"),
    )
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    title="abcdefghiX",
                    artists=candidate_artists,
                ),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.UNRESOLVED


async def test_resolver_applies_optional_album_title_and_year_to_exact_matches() -> None:
    """Require every supplied Album context field before accepting an exact Track."""
    mention = _mention(release_title="Discovery", release_year=2001)
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    "wrong-year",
                    album=_album_candidate(release_date="2000-01-01"),
                ),
                _candidate(
                    "exact",
                    album=_album_candidate(release_date="2001-02-26"),
                ),
                _candidate("wrong-album", album=_album_candidate(title="Homework")),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].track is not None
    assert results[0].track.spotify_track_id == "exact"


async def test_fuzzy_matching_accepts_the_inclusive_ninety_percent_threshold() -> None:
    """Resolve a Candidate whose title similarity is exactly 0.90."""
    mention = _mention(track_title="abcdefghij", artists=("Queen",))
    catalog = _FakeTrackCatalog(((_candidate(title="abcdefghiX", artists=("Queenz",)),),))

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.RESOLVED


async def test_fuzzy_matching_ignores_release_year() -> None:
    """Resolve textual fuzzy matches even when their Album release years differ."""
    mention = _mention(
        track_title="abcdefghij",
        artists=("Queen",),
        release_title="A Night at the Opera",
        release_year=1975,
    )
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    title="abcdefghiX",
                    artists=("Queen",),
                    album=_album_candidate(
                        title="A Night at the Opera",
                        release_date="1976-01-01",
                    ),
                ),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_uses_provider_corrected_track_and_attached_album_values() -> None:
    """Return Spotify's display values, playable Track, and attached Album."""
    mention = _mention(
        track_title="bohemian rhapsody",
        artists=("queen",),
        release_title="a night at the opera",
        release_year=1975,
    )
    attached_album = AlbumCandidate(
        spotify_album_id="album-identity",
        spotify_url="https://open.spotify.com/album/album-identity",
        title="A Night at the Opera",
        artists=(
            ArtistCredit(spotify_artist_id="album-artist-0", name="Queen"),
            ArtistCredit(spotify_artist_id="album-artist-1", name="Opera Singers"),
        ),
        release_date="1975",
        album_type="album",
        images=(
            ImageCandidate(
                url="https://i.scdn.co/image/primary",
                width=640,
                height=640,
            ),
            ImageCandidate(
                url="https://i.scdn.co/image/secondary",
                width=300,
                height=300,
            ),
        ),
    )
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate(
                    spotify_track_id="playable-id",
                    title="Bohemian Rhapsody",
                    artists=("Queen",),
                    album=attached_album,
                ),
            ),
        )
    )

    music_results = await SpotifyMusicResolver(catalog).resolve(
        MusicMentions(tracks=[mention], music_releases=[]),
        _MARKET,
    )

    resolved_track = music_results.tracks[0].track
    assert resolved_track is not None
    assert resolved_track.track_title == "Bohemian Rhapsody"
    assert resolved_track.artists == [ArtistCredit(spotify_artist_id="artist-0", name="Queen")]
    assert resolved_track.spotify_track_id == "playable-id"
    assert resolved_track.spotify_url == "https://open.spotify.com/track/playable-id"
    assert resolved_track.preferred_music_release.release_title == "A Night at the Opera"
    assert resolved_track.preferred_music_release.artists == [
        ArtistCredit(spotify_artist_id="album-artist-0", name="Queen"),
        ArtistCredit(spotify_artist_id="album-artist-1", name="Opera Singers"),
    ]
    assert resolved_track.preferred_music_release.release_date == "1975"
    assert resolved_track.preferred_music_release.album_type == "album"
    assert resolved_track.preferred_music_release.spotify_album_id == "album-identity"
    assert (
        resolved_track.preferred_music_release.spotify_url
        == "https://open.spotify.com/album/album-identity"
    )
    assert resolved_track.preferred_music_release.cover_url == "https://i.scdn.co/image/primary"
    assert resolved_track.cover_url == resolved_track.preferred_music_release.cover_url
    assert music_results.music_releases == []
    assert catalog.album_calls == []


async def test_resolver_keeps_track_resolved_without_attached_album_images() -> None:
    """Expose null Track and Preferred Music Release covers without changing status."""
    catalog = _FakeTrackCatalog(((_candidate(album=_album_candidate(images=())),),))

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [_mention()])

    resolved_track = results[0].track
    assert results[0].status is ResultStatus.RESOLVED
    assert resolved_track is not None
    assert resolved_track.preferred_music_release.cover_url is None
    assert resolved_track.cover_url is None
    assert catalog.album_calls == []


async def test_resolver_returns_unresolved_result_without_candidates() -> None:
    """Keep the original Mention when Spotify returns no eligible Track."""
    mention = _mention()

    results = await _resolve_tracks(SpotifyMusicResolver(_FakeTrackCatalog()), [mention])

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].track_mention is mention
    assert results[0].track is None


@pytest.mark.parametrize(
    "catalog_error",
    [
        CatalogProviderError("catalog unavailable"),
        PipelineTimeoutError("catalog timed out"),
    ],
)
async def test_resolver_propagates_operational_catalog_failures(
    catalog_error: CatalogProviderError | PipelineTimeoutError,
) -> None:
    """Leave typed catalog failures available to enforce atomic aggregation."""
    resolver = SpotifyMusicResolver(_FakeTrackCatalog(error=catalog_error))

    with pytest.raises(type(catalog_error)) as error:
        await _resolve_tracks(resolver, [_mention()])

    assert error.value is catalog_error


async def test_resolver_keeps_the_first_provider_ordered_fuzzy_candidate() -> None:
    """Use provider ordering when multiple fuzzy Candidates are eligible."""
    mention = _mention(track_title="abcdefghij", artists=("Queen",))
    catalog = _FakeTrackCatalog(
        (
            (
                _candidate("first", title="abcdefghiX", artists=("Queen",)),
                _candidate("second", title="abcdefghiY", artists=("Queen",)),
            ),
        )
    )

    results = await _resolve_tracks(SpotifyMusicResolver(catalog), [mention])

    assert results[0].track is not None
    assert results[0].track.spotify_track_id == "first"


async def test_resolver_drops_later_duplicate_playable_ids_but_keeps_unresolved() -> None:
    """Keep first resolved playable IDs and every unresolved Track Mention."""
    first_mention = _mention(track_title="First Song")
    duplicate_mention = _mention(track_title="Second Song")
    unresolved_mention = _mention(track_title="Missing Song")
    catalog = _FakeTrackCatalog(
        (
            (_candidate("shared-id", title="First Song"),),
            (_candidate("shared-id", title="Second Song"),),
            (),
        )
    )

    results = await _resolve_tracks(
        SpotifyMusicResolver(catalog), [first_mention, duplicate_mention, unresolved_mention]
    )

    assert [result.track_mention for result in results] == [
        first_mention,
        unresolved_mention,
    ]
    assert results[0].status is ResultStatus.RESOLVED
    assert results[1].status is ResultStatus.UNRESOLVED
