"""Application-owned candidates returned by catalog provider boundaries."""

from dataclasses import dataclass

from reelio.extraction.types import AlbumType, ArtistCredit


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    """Contain provider-hosted artwork metadata without downloading the image.

    Attributes:
        url: Provider-hosted image URL.
        width: Image width in pixels when Spotify supplies it.
        height: Image height in pixels when Spotify supplies it.
    """

    url: str
    width: int | None
    height: int | None


@dataclass(frozen=True, slots=True)
class AlbumCandidate:
    """Contain one Spotify Album candidate in an effective market.

    Attributes:
        spotify_album_id: Returned Spotify Album identifier.
        spotify_url: Returned direct Spotify Album URL.
        title: Provider-authoritative album title.
        artists: Ordered provider Artist Credits.
        release_date: Spotify-reported release date without transformation.
        album_type: Spotify album classification.
        images: Provider-ordered hosted image candidates.
    """

    spotify_album_id: str
    spotify_url: str
    title: str
    artists: tuple[ArtistCredit, ...]
    release_date: str
    album_type: AlbumType
    images: tuple[ImageCandidate, ...]


@dataclass(frozen=True, slots=True)
class TrackCandidate:
    """Contain one playable Spotify Track candidate in an effective market.

    Attributes:
        spotify_track_id: Returned playable Spotify Track identifier.
        spotify_url: Returned direct Spotify Track URL.
        title: Provider-authoritative track title.
        artists: Ordered provider Artist Credits.
        album: Simplified Spotify Album attached to the returned Track.
    """

    spotify_track_id: str
    spotify_url: str
    title: str
    artists: tuple[ArtistCredit, ...]
    album: AlbumCandidate
