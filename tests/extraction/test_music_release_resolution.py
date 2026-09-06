"""Spotify Music Release resolver contract tests."""

from collections import deque
from collections.abc import Sequence
from typing import Literal

import pytest

from reelio.extraction.exceptions import CatalogProviderError, PipelineTimeoutError
from reelio.extraction.market import SpotifyMarket
from reelio.extraction.services.catalog.types import AlbumCandidate, TrackCandidate
from reelio.extraction.services.enrichment.spotify import SpotifyMusicResolver
from reelio.extraction.types import (
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
    release_date_precision: Literal["year", "month", "day"] = "day",
    album_type: Literal["album", "single", "compilation"] = "album",
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
        release_date_precision=release_date_precision,
        album_type=album_type,
        images=(),
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


async def test_resolver_builds_field_scoped_query_and_forwards_effective_market() -> None:
    """Pass an unescaped ordered album and artist query and effective market to Spotify."""
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

    assert catalog.calls == [
        ("album:Discovery artist:Daft Punk artist:Romanthony year:2001", _MARKET)
    ]
    assert results[0].status is ResultStatus.RESOLVED


async def test_resolver_omits_the_year_term_when_the_mention_has_none() -> None:
    """Keep null-year Mentions eligible and exclude year from the album query."""
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


async def test_resolver_searches_all_exact_candidates_before_fuzzy_candidates() -> None:
    """Choose a later exact Candidate over an earlier fuzzy Candidate."""
    mention = _mention(release_title="Midnight Echo")
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate("fuzzy", title="Midnight Echos"),
                _candidate("exact", title="Midnight Echo"),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.spotify_album_id == "exact"


async def test_exact_matching_ignores_a_missing_release_year() -> None:
    """Resolve an exact Candidate when the Mention supplies no year."""
    mention = _mention(release_year=None)
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


async def test_exact_matching_requires_the_supplied_release_year() -> None:
    """Compare a supplied year only to the first four characters of the release date."""
    mention = _mention(release_year=2001)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate("wrong-year", release_date="2000-01-01"),
                _candidate("exact", release_date="2001-02-26"),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.spotify_album_id == "exact"


@pytest.mark.parametrize(
    "candidate_artists",
    [
        ("Second Artist", "First Artist"),
        ("First Artist",),
    ],
    ids=["reversed-order", "different-length"],
)
async def test_exact_matching_requires_same_length_positional_artist_credits(
    candidate_artists: Sequence[str],
) -> None:
    """Reject exact candidates whose artist credits differ by order or length."""
    mention = _mention(artists=("First Artist", "Second Artist"))
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    artists=candidate_artists,
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.UNRESOLVED


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
        release_title="abcdefghij",
        artists=("First Artist", "Second Artist"),
        release_year=None,
    )
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    title="abcdefghiX",
                    artists=candidate_artists,
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.UNRESOLVED


async def test_fuzzy_matching_accepts_the_inclusive_ninety_percent_threshold() -> None:
    """Resolve a Candidate whose title similarity is exactly 0.90."""
    mention = _mention(release_title="abcdefghij", artists=("Queen",), release_year=None)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    title="abcdefghiX",
                    artists=("Queenz",),
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.RESOLVED


async def test_fuzzy_matching_ignores_release_year() -> None:
    """Resolve textual fuzzy matches even when their release years differ."""
    mention = _mention(release_title="abcdefghij", artists=("Queen",), release_year=1975)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate(
                    title="abcdefghiX",
                    artists=("Queen",),
                    release_date="2001-02-26",
                ),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].status is ResultStatus.RESOLVED


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
                    release_date_precision="day",
                    album_type="album",
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
    assert results[0].music_release.release_date_precision == "day"
    assert results[0].music_release.album_type == "album"
    assert results[0].music_release.spotify_album_id == "album-identity"
    assert results[0].music_release.spotify_url == "https://open.spotify.com/album/album-identity"


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


async def test_resolver_keeps_the_first_provider_ordered_fuzzy_candidate() -> None:
    """Use provider ordering when multiple fuzzy Candidates are eligible."""
    mention = _mention(release_title="abcdefghij", artists=("Queen",), release_year=None)
    catalog = _FakeAlbumCatalog(
        (
            (
                _candidate("first", title="abcdefghiX", artists=("Queen",)),
                _candidate("second", title="abcdefghiY", artists=("Queen",)),
            ),
        )
    )

    results = await _resolve_music_releases(SpotifyMusicResolver(catalog), [mention])

    assert results[0].music_release is not None
    assert results[0].music_release.spotify_album_id == "first"


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
