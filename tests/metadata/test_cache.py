from dataclasses import FrozenInstanceError

import pytest
from pydantic import AnyUrl

from src.entities.models import EntityType
from src.metadata.cache import CacheLookup, InMemoryTTLCache
from src.metadata.models import BookMetadata

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_book(provider_id: str, title: str) -> BookMetadata:
    return BookMetadata(
        provider_id=provider_id,
        title=title,
        url=AnyUrl(f"https://books.google.com/books?id={provider_id}"),
    )


def _fake_clock(start: float = 0.0) -> list[float]:
    """Return a mutable clock container so tests can advance time."""
    return [start]


def _make_timer(clock: list[float]):
    def timer() -> float:
        return clock[0]

    return timer


# ── basic operations ─────────────────────────────────────────────────────────


def test_miss_returns_false():
    cache = InMemoryTTLCache(60, 10)
    key = ("google_books", EntityType.BOOK, "dune", "")
    result = cache.get(key)
    assert result.hit is False
    assert result.value is None


def test_positive_hit():
    cache = InMemoryTTLCache(60, 10)
    key = ("google_books", EntityType.BOOK, "dune", "")
    book = _make_book("abc", "Dune")
    cache.set(key, book)
    result = cache.get(key)
    assert result.hit is True
    assert result.value is book


def test_negative_hit():
    cache = InMemoryTTLCache(60, 10)
    key = ("google_books", EntityType.BOOK, "unknown", "")
    cache.set(key, None)
    result = cache.get(key)
    assert result.hit is True
    assert result.value is None


# ── TTL behaviour ────────────────────────────────────────────────────────────


def test_hit_before_expiry():
    clock = _fake_clock(0.0)
    cache = InMemoryTTLCache(60, 10, timer=_make_timer(clock))
    key = ("google_books", EntityType.BOOK, "dune", "")
    book = _make_book("abc", "Dune")
    cache.set(key, book)
    # advance to just before expiry
    clock[0] = 59.9
    result = cache.get(key)
    assert result.hit is True
    assert result.value is book


def test_miss_at_exact_expiry():
    clock = _fake_clock(0.0)
    cache = InMemoryTTLCache(60, 10, timer=_make_timer(clock))
    key = ("google_books", EntityType.BOOK, "dune", "")
    cache.set(key, _make_book("abc", "Dune"))
    clock[0] = 60.0
    result = cache.get(key)
    assert result.hit is False


def test_persistent_miss_after_expiry():
    clock = _fake_clock(0.0)
    cache = InMemoryTTLCache(60, 10, timer=_make_timer(clock))
    key = ("google_books", EntityType.BOOK, "dune", "")
    cache.set(key, _make_book("abc", "Dune"))
    clock[0] = 120.0
    result = cache.get(key)
    assert result.hit is False


# ── LRU eviction ─────────────────────────────────────────────────────────────


def test_lru_eviction():
    cache = InMemoryTTLCache(60, 2)
    key_a = ("google_books", EntityType.BOOK, "a", "")
    key_b = ("google_books", EntityType.BOOK, "b", "")
    key_c = ("google_books", EntityType.BOOK, "c", "")
    book_a = _make_book("a", "A")
    book_b = _make_book("b", "B")
    book_c = _make_book("c", "C")

    cache.set(key_a, book_a)
    cache.set(key_b, book_b)

    # access A so B becomes LRU
    result_a = cache.get(key_a)
    assert result_a.hit is True

    # insert C, evicting B (LRU)
    cache.set(key_c, book_c)

    assert cache.get(key_a).hit is True  # A still present (was recently accessed)
    assert cache.get(key_b).hit is False  # B evicted
    assert cache.get(key_c).hit is True  # C present


# ── colon-bearing keys remain distinct ───────────────────────────────────────


def test_colon_keys_distinct():
    cache = InMemoryTTLCache(60, 10)
    key1 = ("provider", EntityType.BOOK, "name:part", "context")
    key2 = ("provider", EntityType.BOOK, "name", "part:context")
    book1 = _make_book("1", "Book One")
    book2 = _make_book("2", "Book Two")

    cache.set(key1, book1)
    cache.set(key2, book2)

    r1 = cache.get(key1)
    r2 = cache.get(key2)
    assert r1.hit is True
    assert r2.hit is True
    assert r1.value is book1
    assert r2.value is book2


# ── cache instance isolation ─────────────────────────────────────────────────


def test_cache_instance_isolation():
    cache1 = InMemoryTTLCache(60, 10)
    cache2 = InMemoryTTLCache(60, 10)
    key = ("google_books", EntityType.BOOK, "dune", "")
    book = _make_book("abc", "Dune")

    cache1.set(key, book)
    assert cache1.get(key).hit is True
    assert cache2.get(key).hit is False


# ── CacheLookup immutability ─────────────────────────────────────────────────


def test_cache_lookup_immutable():
    lookup = CacheLookup(hit=True, value=None)
    with pytest.raises(FrozenInstanceError):
        lookup.hit = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        lookup.value = _make_book("x", "X")  # type: ignore[misc]


# ── constructor validations ──────────────────────────────────────────────────

CONSTRUCTOR_FAILURE_CASES = [
    # (ttl_seconds, max_entries)
    (0, 10),
    (-1, 10),
    (float("nan"), 10),
    (float("inf"), 10),
    (60, 0),
    (60, -1),
]


@pytest.mark.parametrize(("ttl_seconds", "max_entries"), CONSTRUCTOR_FAILURE_CASES)
def test_constructor_rejects_invalid_params(ttl_seconds, max_entries):
    with pytest.raises(ValueError):
        InMemoryTTLCache(ttl_seconds, max_entries)
