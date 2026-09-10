"""Spotify Music Release resolver contract tests."""

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
    AlbumType,
    ArtistCredit,
    MusicMentions,
    MusicReleaseMention,
    MusicReleaseResult,
    ResultStatus,
)

_MARKET = SpotifyMarket("JP")


class _FakeAlbumCatalog:
    """Return deterministic provider-ordered Album Candidates."""

    def __init__(
        self,
        candidate_groups: Sequence[tuple[AlbumCandidate, ...]] = (),
        error: CatalogProviderError | PipelineTimeoutError | None = None,
    ) -> None:
        """Configure candidate groups or one operational error.

        Args:
            candidate_groups: Search results returned in call order.
            error: Typed catalog failure raised on every search when supplied.
        """
        self._candidate_groups = deque(candidate_groups)
        self._error = error
        self.calls: list[tuple[str, SpotifyMarket]] = []

    async def search_albums(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[AlbumCandidate, ...]:
        """Record a search and return its configured provider result."""
        self.calls.append((query, market))
        if self._error is not None:
            raise self._error
        if not self._candidate_groups:
            return ()
        return self._candidate_groups.popleft()

    async def search_tracks(
        self,
        query: str,
        market: SpotifyMarket,
    ) -> tuple[TrackCandidate, ...]:
        """Fail if a Music Release-only test resolves a Track."""
        raise AssertionError(f"unexpected track search: {query} in {market}")


def _mention(
    release_title: str = "Discovery",
    artists: Sequence[str] = ("Daft Punk",),
    release_year: int | None = 2001,
) -> MusicReleaseMention:
    """Create one canonical Music Release Mention for resolver tests."""
    return MusicReleaseMention(
        release_title=release_title,
        artists=list(artists),
        release_year=release_year,
    )


def _candidate(
    spotify_album_id: str = "album-1",
    title: str = "Discovery",
    artists: Sequence[str] = ("Daft Punk",),
    release_date: str = "2001-02-26",
    album_type: AlbumType = "album",
    images: tuple[ImageCandidate, ...] = (),
) -> AlbumCandidate:
    """Create one Spotify Album Candidate for resolver tests."""
    return AlbumCandidate(
        spotify_album_id=spotify_album_id,
        spotify_url=f"https://open.spotify.com/album/{spotify_album_id}",
        title=title,
        artists=tuple(
            ArtistCredit(spotify_artist_id=f"artist-{index}", name=artist)
            for index, artist in enumerate(artists)
        ),
        release_date=release_date,
        album_type=album_type,
        images=images,
    )


async def _resolve_music_releases(
    resolver: SpotifyMusicResolver,
    music_release_mentions: list[MusicReleaseMention],
) -> list[MusicReleaseResult]:
    """Resolve Music Release Mentions through the grouped Music Resolver interface."""
    music_results = await resolver.resolve(
        MusicMentions(tracks=[], music_releases=music_release_mentions),
        _MARKET,
    )
    return music_results.music_releases


async def test_resolver_queries_release_title_and_first_artist_only() -> None:
    """Pass the Music Release title and first Artist Credit to Spotify."""
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    title="Discovery",
                    artists=("Daft Punk", "Romanthony"),
                ),
            ),
        )
    )
    resolver = SpotifyMusicResolver(catalog)
    mention = _mention(artists=("Daft Punk", "Romanthony"))

    results = await _resolve_music_releases(resolver, [mention])

    assert catalog.calls == [("album:Discovery artist:Daft Punk", _MARKET)]
    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_excludes_nullable_mention_year_from_the_query() -> None:
    """Use the same query for a null Mention year."""
    catalog = _FakeAlbumCatalog(((_candidate(),),))

    results = await _resolve_music_releases(
        SpotifyMusicResolver(catalog), [_mention(release_year=None)]
    )

    assert catalog.calls == [("album:Discovery artist:Daft Punk", _MARKET)]
    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_skips_catalog_calls_for_empty_input() -> None:
    """Return no results without issuing one Album search for empty mentions."""
    catalog = _FakeAlbumCatalog()

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [])

    assert results == []
    assert catalog.calls == []


async def test_resolver_inspects_only_the_first_three_candidates() -> None:
    """Leave a fourth exact candidate outside the resolver's hard search bound."""
    candidates = (
        _candidate("album-1", title="Wrong One"),
        _candidate("album-2", title="Wrong Two"),
        _candidate("album-3", title="Wrong Three"),
        _candidate("album-4"),
    )

    results = await _resolve_music_releases(
        SpotifyMusicResolver(_FakeAlbumCatalog((candidates,))), [_mention()]
    )

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].music_release_mention == _mention()
    assert results[0].music_release is None


async def test_resolver_keeps_the_first_provider_ordered_exact_candidate() -> None:
    """Use provider order when multiple eligible Candidates have exact titles."""
    mention = _mention(release_title="Midnight Echo")
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate("first", title="Midnight Echo"),
                _candidate("second", title="Midnight Echo"),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.spotify_album_id == "first"


async def test_resolver_ignores_mention_year_when_matching_exact_title() -> None:
    """Resolve an exact shared-credit Candidate despite a different provider year."""
    mention = _mention(release_year=2001)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    spotify_album_id="different-year",
                    release_date="2025-01-01",
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.spotify_album_id == "different-year"


@pytest.mark.parametrize(
    "candidate_artists",
    [
        ("Second Artist", "First Artist"),
        ("First Artist",),
        ("Unmatched Artist", " first artist ", "Second Artist"),
    ],
    ids=["reordered", "shorter", "longer-with-normalized-shared-credit"],
)
async def test_resolver_accepts_any_shared_normalized_artist_credit(
    candidate_artists: Sequence[str],
) -> None:
    """Resolve despite reordered, shorter, or additional Candidate Artist Credits."""
    mention = _mention(artists=("First Artist", "Second Artist"))
    catalog = _FakeAlbumCatalog(((_candidate(artists=candidate_artists),),))

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_rejects_exact_title_without_shared_artist_credit() -> None:
    """Leave an exact-title Candidate unresolved without a shared Artist Credit."""
    catalog = _FakeAlbumCatalog(((_candidate(artists=("Other Artist",)),),))

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [_mention()])

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].music_release is None


@pytest.mark.parametrize("candidate_title", ["Discoveries"], ids=["exact-threshold"])
async def test_resolver_rejects_music_release_title_at_fuzzy_threshold(
    candidate_title: str,
) -> None:
    """Leave a shared-credit Music Release unresolved at the strict fuzzy boundary."""
    catalog = _FakeAlbumCatalog(((_candidate(title=candidate_title),),))

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [_mention()])

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].music_release is None


async def test_resolver_accepts_an_exactly_named_compilation() -> None:
    """Resolve a directly named Compilation without edition-specific filtering."""
    catalog = _FakeAlbumCatalog(((_candidate(album_type="compilation"),),))

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [_mention()])

    assert results[0].status is ResultStatus.RESOLVED
    assert results[0].music_release is not None
    assert results[0].music_release.album_type == "compilation"


async def test_resolver_uses_provider_corrected_release_and_artist_values() -> None:
    """Return Spotify's display strings and Album identity after an exact match."""
    mention = _mention(release_title="discovery", artists=("daft punk",), release_year=None)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    spotify_album_id="album-identity",
                    title="Discovery",
                    artists=("Daft Punk",),
                    release_date="2001-02-26",
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
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.release_title == "Discovery"
    assert results[0].music_release.artists == [
        ArtistCredit(spotify_artist_id="artist-0", name="Daft Punk")
    ]
    assert results[0].music_release.release_date == "2001-02-26"
    assert results[0].music_release.album_type == "album"
    assert results[0].music_release.spotify_album_id == "album-identity"
    assert results[0].music_release.spotify_url == "https://open.spotify.com/album/album-identity"
    assert results[0].music_release.cover_url == "https://i.scdn.co/image/primary"


async def test_resolver_keeps_music_releases_resolved_without_album_images() -> None:
    """Expose a null cover without changing a direct Music Release status."""
    catalog = _FakeAlbumCatalog(((_candidate(images=()),),))

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [_mention()])

    resolved_music_release = results[0].music_release
    assert results[0].status is ResultStatus.RESOLVED
    assert resolved_music_release is not None
    assert resolved_music_release.cover_url is None


async def test_resolver_returns_unresolved_result_without_candidates() -> None:
    """Keep the original Mention when Spotify returns no eligible Album."""
    mention = _mention()

    results = await _resolve_music_releases(SpotifyMusicResolver(_FakeAlbumCatalog()), [mention])

    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].music_release_mention is mention
    assert results[0].music_release is None


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
    resolver = SpotifyMusicResolver(_FakeAlbumCatalog(error=catalog_error))

    with pytest.raises(type(catalog_error)) as error:
        await _resolve_music_releases(resolver, [_mention()])

    assert error.value is catalog_error


async def test_resolver_drops_later_duplicate_album_ids_but_keeps_unresolved() -> None:
    """Keep first resolved Album IDs and every unresolved Music Release Mention."""
    first_mention = _mention(release_title="First Album")
    duplicate_mention = _mention(release_title="Second Album")
    unresolved_mention = _mention(release_title="Missing Album")
    catalog = _FakeAlbumCatalog(
        (
            (_candidate("shared-id", title="First Album"),),
            (_candidate("shared-id", title="Second Album"),),
            (),
        )
    )

    results = await _resolve_music_releases(
        SpotifyMusicResolver(catalog), [first_mention, duplicate_mention, unresolved_mention]
    )

    assert [result.music_release_mention for result in results] == [
        first_mention,
        unresolved_mention,
    ]
    assert results[0].status is ResultStatus.RESOLVED
    assert results[1].status is ResultStatus.UNRESOLVED
