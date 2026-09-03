"""Effective Spotify market value-object tests."""

import pytest

from reelio.extraction.market import SpotifyMarket


def test_spotify_market_retains_a_valid_iso_alpha_two_code() -> None:
    """Represent one validated uppercase market code without normalization."""
    assert SpotifyMarket("JP") == "JP"


@pytest.mark.parametrize("value", ["jp", "JPN", "J1", " J"])
def test_spotify_market_rejects_invalid_syntax(value: str) -> None:
    """Reject invalid market syntax at construction before it reaches providers."""
    with pytest.raises(ValueError, match="uppercase ISO 3166-1 alpha-2"):
        SpotifyMarket(value)
