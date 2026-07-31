import pytest

from src.metadata.matching import (
    escape_search_filter,
    extract_first_year,
    matches_exact,
    normalize_for_cache,
    normalize_for_match,
    parse_year,
)

# ── normalize_for_match ──────────────────────────────────────────────────────

MATCH_NORM_CASES = [
    # (input, expected)
    ("Spider-Man", "spider man"),
    ("Spider - Man", "spider man"),
    ("Beyoncé", "beyonce"),
    ("Beyonce\u0301", "beyonce"),  # decomposed accent
    ("  Beyoncé  ", "beyonce"),
    ("Straße", "strasse"),
    ("STRASSE", "strasse"),
    ("Hello\t\n  World", "hello world"),
    ("Punctuation!!!...and,,,more", "punctuation and more"),
    # emoji/symbols preserved
    ("Star ⭐ Wars", "star ⭐ wars"),
    ("© 2024", "© 2024"),  # © is So (symbol), not P (punctuation) - preserved
    # empty/whitespace
    ("", ""),
    ("   ", ""),
    ("\t\n", ""),
    # composed vs decomposed
    ("café", "cafe"),
]


@pytest.mark.parametrize(("value", "expected"), MATCH_NORM_CASES)
def test_normalize_for_match(value, expected):
    assert normalize_for_match(value) == expected


# ── normalize_for_cache ──────────────────────────────────────────────────────

CACHE_NORM_CASES = [
    # (input, expected)
    ("Hello  World", "hello world"),
    ("\t\n  foo  bar  ", "foo bar"),
    ("Straße", "strasse"),  # NFKC transforms ß → ss
    ("Café", "café"),
    # punctuation preserved
    ("Hello, World!", "hello, world!"),
    ("key:value", "key:value"),
    # None → ""
    (None, ""),
    # blank input
    ("", ""),
    ("   ", ""),
    # NFKC equivalents
    ("\u2160", "i"),  # ROMAN NUMERAL ONE → I then casefold → i
]


@pytest.mark.parametrize(("value", "expected"), CACHE_NORM_CASES)
def test_normalize_for_cache(value, expected):
    assert normalize_for_cache(value) == expected


# ── matches_exact ────────────────────────────────────────────────────────────

MATCH_EXACT_CASES = [
    # single candidate, exact match
    ("Dune", ["Dune"], True),
    # single candidate, case/accent mismatch → normalized match
    ("dune", ["Dune"], True),
    # single candidate, punctuation difference
    ("Spider-Man", ["Spider Man"], True),
    # multiple candidates, second matches
    ("xyz", ["abc", "def", "xyz"], True),
    # empty candidate iterable
    ("Dune", [], False),
    # empty value
    ("", ["Dune"], False),
    ("   ", ["Dune"], False),
    # no match
    ("Stranger", ["Dune", "Foundation"], False),
    # prefix/substring rejection
    ("Dune", ["Dune Messiah"], False),
    ("Dune Messiah", ["Dune"], False),
    # accent equivalence
    ("Beyonce", ["Beyoncé"], True),
]


@pytest.mark.parametrize(("value", "candidates", "expected"), MATCH_EXACT_CASES)
def test_matches_exact(value, candidates, expected):
    assert matches_exact(value, candidates) == expected


# ── parse_year ───────────────────────────────────────────────────────────────

PARSE_YEAR_CASES = [
    # (input, expected)
    ("1986", 1986),
    ("  1986  ", 1986),
    ("0000", 0),
    ("9999", 9999),
    (None, None),
    ("", None),
    ("   ", None),
    ("123", None),  # 3 digits
    ("12345", None),  # 5 digits
    ("1986a", None),  # trailing char
    ("a1986", None),  # leading char
    ("-1986", None),  # sign
    ("19.86", None),  # decimal
    ("nineteen eighty six", None),
    ("\u0966\u0967\u0968\u0969", None),  # non-ASCII digits
]


@pytest.mark.parametrize(("value", "expected"), PARSE_YEAR_CASES)
def test_parse_year(value, expected):
    assert parse_year(value) == expected


# ── extract_first_year ───────────────────────────────────────────────────────

EXTRACT_YEAR_CASES = [
    # (input, expected)
    ("2024", 2024),
    ("2024-06", 2024),
    ("2024-06-15", 2024),
    ("  2024-06-15  ", 2024),
    ("  2024  ", 2024),
    (None, None),
    ("", None),
    ("   ", None),
    ("123", None),  # too short
    ("abc2024", None),  # leading non-numeric
    ("2024/06/15", None),  # slash separator
    ("2024-", 2024),  # YYYY- is valid (dash separator)
    ("\u0968\u0966\u0968\u096a-06", None),  # non-ASCII digits
]


@pytest.mark.parametrize(("value", "expected"), EXTRACT_YEAR_CASES)
def test_extract_first_year(value, expected):
    assert extract_first_year(value) == expected


# ── escape_search_filter ─────────────────────────────────────────────────────

ESCAPE_CASES = [
    # (input, expected)
    ("", ""),
    ("hello world", "hello world"),
    ('say "hello"', 'say \\"hello\\"'),
    ('""', '\\"\\"'),
    ("path\\to\\file", "path\\\\to\\\\file"),
    ("\\\\", "\\\\\\\\"),  # each \ becomes \\
    ('"quoted" and \\backslash', '\\"quoted\\" and \\\\backslash'),
    ('\\then"quote', '\\\\then\\"quote'),
]


@pytest.mark.parametrize(("value", "expected"), ESCAPE_CASES)
def test_escape_search_filter(value, expected):
    assert escape_search_filter(value) == expected
