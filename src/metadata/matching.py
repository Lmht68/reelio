import unicodedata
from collections.abc import Iterable


def normalize_for_match(value: str) -> str:
    """Normalize a string for provider matching: NFKD, strip combining marks, casefold,
    replace punctuation with spaces, collapse whitespace."""
    if not value or not value.strip():
        return ""
    nfkd = unicodedata.normalize("NFKD", value)
    no_marks = "".join(ch for ch in nfkd if not unicodedata.category(ch).startswith("M"))
    casefolded = no_marks.casefold()
    punctuation_replaced = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in casefolded
    )
    collapsed = " ".join(punctuation_replaced.split())
    return collapsed


def normalize_for_cache(value: str | None) -> str:
    """Normalize for cache key identity: NFKC, collapse whitespace, casefold."""
    if value is None:
        return ""
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    return normalized


def matches_exact(value: str, candidates: Iterable[str]) -> bool:
    """Return True iff `value` normalizes to something non-empty that exactly matches
    one of the normalized `candidates`."""
    needle = normalize_for_match(value)
    if not needle:
        return False
    for candidate in candidates:
        if normalize_for_match(candidate) == needle:
            return True
    return False


def parse_year(value: str | None) -> int | None:
    """Parse exactly four ASCII digits after outer whitespace stripping, or return None."""
    if value is None:
        return None
    stripped = value.strip()
    if len(stripped) != 4:
        return None
    if not stripped.isascii() or not stripped.isdigit():
        return None
    return int(stripped)


def extract_first_year(value: str | None) -> int | None:
    """Extract a year from a YYYY, YYYY-MM, or YYYY-MM-DD string prefix, or return None."""
    if value is None:
        return None
    stripped = value.strip()
    if len(stripped) < 4:
        return None
    prefix = stripped[:4]
    if not prefix.isascii() or not prefix.isdigit():
        return None
    if len(stripped) == 4:
        return int(prefix)
    if stripped[4] == "-":
        return int(prefix)
    return None


def escape_search_filter(value: str) -> str:
    """Escape backslashes and double quotes for a search filter string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
