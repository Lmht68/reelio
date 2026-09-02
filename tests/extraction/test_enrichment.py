"""TMDB candidate resolution and enrichment contract tests."""

import asyncio
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from reelio.extraction.exceptions import EnrichmentError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import TMDBConfig
from reelio.extraction.services.enrichment.tmdb import (
    TMDBScreenWorkResolver,
    create_tmdb_screen_work_resolver,
)
from reelio.extraction.types import (
    MovieMention,
    ResultStatus,
    ScreenWorkMentions,
    TVSeriesMention,
)


def _settings(**values: object) -> TMDBConfig:
    settings_type = cast(Callable[..., TMDBConfig], TMDBConfig)
    return settings_type(_env_file=None, api_key="test-tmdb-key", **values)


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.themoviedb.org/3/",
        transport=handler,
    )


def _mentions(
    movies: list[MovieMention] | None = None,
    tv_series: list[TVSeriesMention] | None = None,
) -> ScreenWorkMentions:
    return ScreenWorkMentions(
        movies=[] if movies is None else movies,
        tv_series=[] if tv_series is None else tv_series,
    )


async def test_resolver_selects_first_title_and_year_match_and_enriches() -> None:
    """Return TMDB Movie identity while retaining the original Movie Mention."""
    requested_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/3/search/movie":
            assert request.url.params["query"] == "Amélie"
            assert request.url.params["year"] == "2001"
            assert request.url.params["page"] == "1"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "A Different Film",
                            "original_title": "A Different Film",
                            "release_date": "2001-01-01",
                        },
                        {
                            "id": 2,
                            "title": "Le Fabuleux Destin d'Amélie Poulain",
                            "original_title": "  Ame\u0301lie  ",
                            "release_date": "2001-04-25",
                        },
                    ],
                },
            )
        if request.url.path == "/3/movie/1":
            assert request.url.params["append_to_response"] == "credits,alternative_titles"
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "title": "A Different Film",
                    "release_date": "2001-01-01",
                    "alternative_titles": {"titles": []},
                },
            )

        assert request.url.path == "/3/movie/2"
        assert request.url.params["append_to_response"] == "credits"
        return httpx.Response(
            200,
            json={
                "id": 2,
                "title": "Le Fabuleux Destin d'Amélie Poulain",
                "release_date": "2000-04-25",
                "overview": "A Parisian woman quietly improves the lives around her.",
                "poster_path": "/amelie.jpg",
                "imdb_id": "tt0211915",
                "vote_average": 7.9,
                "credits": {
                    "cast": [
                        {"name": "Audrey Tautou"},
                        {"name": "Mathieu Kassovitz"},
                        {"name": "Rufus"},
                        {"name": "Lorella Cravotta"},
                        {"name": "Serge Merlin"},
                        {"name": "Jamel Debbouze"},
                    ],
                    "crew": [
                        {"name": "Jean-Pierre Jeunet", "job": "Director"},
                        {"name": "Jean-Pierre Jeunet", "job": "Director"},
                        {
                            "name": "Bruno Delbonnel",
                            "job": "Director of Photography",
                        },
                    ],
                },
            },
        )

    transport = httpx.MockTransport(handle)
    client = _client(transport)
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500/")
    movie_mention = MovieMention(title="Amélie", year=2001)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    assert requested_paths == ["/3/search/movie", "/3/movie/1", "/3/movie/2"]
    assert len(results.movies) == 1
    assert results.tv_series == []
    result = results.movies[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie_mention.title == "Amélie"
    assert result.movie_mention.year == 2001
    assert result.movie is not None
    assert result.movie.title == "Le Fabuleux Destin d'Amélie Poulain"
    assert result.movie.year == 2000
    assert result.movie.cast == [
        "Audrey Tautou",
        "Mathieu Kassovitz",
        "Rufus",
        "Lorella Cravotta",
        "Serge Merlin",
    ]
    assert result.movie.directors == ["Jean-Pierre Jeunet"]
    assert result.movie.description == "A Parisian woman quietly improves the lives around her."
    assert result.movie.poster_url == "https://image.tmdb.org/t/p/w500/amelie.jpg"
    assert result.movie.tmdb_id == 2
    assert result.movie.tmdb_url == "https://www.themoviedb.org/movie/2"
    assert result.movie.imdb_id == "tt0211915"
    assert result.movie.imdb_url == "https://www.imdb.com/title/tt0211915/"
    assert result.movie.tmdb_score == 7.9
    await resolver.aclose()
    assert client.is_closed is True


async def test_resolver_matches_provider_alternative_title() -> None:
    """Resolve a Movie Mention through a provider alternative title."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Le Fabuleux Destin d'Amélie Poulain",
                            "original_title": "Le Fabuleux Destin d'Amélie Poulain",
                            "release_date": "2001-04-25",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/movie/1"
        assert request.url.params["append_to_response"] == "credits,alternative_titles"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Le Fabuleux Destin d'Amélie Poulain",
                "release_date": "2001-04-25",
                "alternative_titles": {"titles": [{"title": "Amélie"}]},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(_mentions(movies=[MovieMention(title="Amélie", year=2001)]))

    assert results.movies[0].status is ResultStatus.RESOLVED
    assert results.movies[0].movie is not None
    assert results.movies[0].movie.tmdb_id == 1
    await resolver.aclose()


async def test_resolver_searches_first_page_only_before_returning_unresolved() -> None:
    """Inspect only the first Movie candidate page before returning unresolved."""
    requested_pages: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/3/search/movie"
        page = request.url.params["page"]
        requested_pages.append(page)
        assert page == "1"
        return httpx.Response(
            200,
            json={
                "results": [{"id": 1, "release_date": "1999-01-01"}],
                "total_pages": 2,
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Missing Film", year=2000)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    assert requested_pages == ["1"]
    assert results.movies[0].status is ResultStatus.UNRESOLVED
    assert results.movies[0].movie_mention is movie_mention
    assert results.movies[0].movie is None
    await resolver.aclose()


async def test_resolver_resolves_tv_primary_original_and_alternative_titles() -> None:
    """Resolve TV Series through primary, original, and alternative provider titles."""
    detail_requests: dict[str, str] = {}

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            assert request.url.params["include_adult"] == "true"
            assert request.url.params["language"] == "en-US"
            query = request.url.params["query"]
            assert (
                request.url.params["first_air_date_year"]
                == {
                    "Primary": "2001",
                    "Original": "2002",
                    "Alternative": "2003",
                }[query]
            )
            assert request.url.params["page"] == "1"
            return httpx.Response(
                200,
                json={
                    "results": {
                        "Primary": [
                            {
                                "id": 1,
                                "name": "Primary",
                                "original_name": "Provider Primary",
                                "first_air_date": "2001-01-01",
                            }
                        ],
                        "Original": [
                            {
                                "id": 2,
                                "name": "Translated",
                                "original_name": "Original",
                                "first_air_date": "2002-01-01",
                            }
                        ],
                        "Alternative": [
                            {
                                "id": 3,
                                "name": "Provider Name",
                                "original_name": "Provider Original",
                                "first_air_date": "2003-01-01",
                            }
                        ],
                    }[query]
                },
            )

        detail_requests[request.url.path] = request.url.params["append_to_response"]
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "name": "Provider Name",
                "aggregate_credits": {"cast": []},
                "external_ids": {},
                "alternative_titles": (
                    {"titles": [{"title": "Alternative"}]} if identifier == 3 else {"results": []}
                ),
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    mentions = [
        TVSeriesMention(title="Primary", year=2001),
        TVSeriesMention(title="Original", year=2002),
        TVSeriesMention(title="Alternative", year=2003),
    ]

    results = await resolver.resolve(_mentions(tv_series=mentions))

    assert [result.status for result in results.tv_series] == [
        ResultStatus.RESOLVED,
        ResultStatus.RESOLVED,
        ResultStatus.RESOLVED,
    ]
    assert [result.tv_series_mention for result in results.tv_series] == mentions
    assert [result.tv_series.tmdb_id for result in results.tv_series if result.tv_series] == [
        1,
        2,
        3,
    ]
    assert detail_requests == {
        "/3/tv/1": "aggregate_credits,external_ids",
        "/3/tv/2": "aggregate_credits,external_ids",
        "/3/tv/3": "aggregate_credits,alternative_titles,external_ids",
    }
    assert all(path.startswith("/3/tv/") for path in detail_requests)
    await resolver.aclose()


async def test_resolver_limits_tv_resolution_to_first_page_and_three_candidates() -> None:
    """Request no TV search page beyond one and inspect at most three candidates."""
    requested_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/3/search/tv":
            assert request.url.params["page"] == "1"
            return httpx.Response(
                200,
                json={
                    "total_pages": 2,
                    "results": [
                        {
                            "id": identifier,
                            "name": "No Match",
                            "first_air_date": "2020-01-01",
                        }
                        for identifier in range(1, 5)
                    ],
                },
            )
        assert request.url.path in {"/3/tv/1", "/3/tv/2", "/3/tv/3"}
        assert request.url.params["append_to_response"] == (
            "aggregate_credits,alternative_titles,external_ids"
        )
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "name": "No Match",
                "aggregate_credits": {},
                "external_ids": {},
                "alternative_titles": {"results": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(tv_series=[TVSeriesMention(title="Target", year=2020)])
    )

    assert requested_paths == ["/3/search/tv", "/3/tv/1", "/3/tv/2", "/3/tv/3"]
    assert results.tv_series[0].status is ResultStatus.UNRESOLVED
    assert results.tv_series[0].tv_series is None
    await resolver.aclose()


async def test_resolver_returns_first_matching_tv_candidate() -> None:
    """Select the first TV candidate that matches the canonical title and year."""
    detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Target",
                            "first_air_date": "2015-01-01",
                        },
                        {
                            "id": 2,
                            "name": "Target",
                            "first_air_date": "2015-01-01",
                        },
                    ]
                },
            )
        detail_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": 1,
                "name": "Target",
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(tv_series=[TVSeriesMention(title="Target", year=2015)])
    )

    assert detail_paths == ["/3/tv/1"]
    assert results.tv_series[0].tv_series is not None
    assert results.tv_series[0].tv_series.tmdb_id == 1
    await resolver.aclose()


async def test_resolver_retains_unresolved_tv_mentions_after_mismatches_or_absence() -> None:
    """Keep original TV Mentions when TMDB has no title-and-year match."""
    detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": {
                        "Wrong Year": [
                            {
                                "id": 1,
                                "name": "Wrong Year",
                                "first_air_date": "1999-01-01",
                            }
                        ],
                        "Wrong Title": [
                            {
                                "id": 2,
                                "name": "Provider Title",
                                "first_air_date": "2000-01-01",
                            }
                        ],
                        "Missing": [],
                    }[request.url.params["query"]]
                },
            )
        detail_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": 2,
                "name": "Provider Title",
                "aggregate_credits": {},
                "external_ids": {},
                "alternative_titles": {"results": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    mentions = [
        TVSeriesMention(title="Wrong Year", year=2000),
        TVSeriesMention(title="Wrong Title", year=2000),
        TVSeriesMention(title="Missing", year=2000),
    ]

    results = await resolver.resolve(_mentions(tv_series=mentions))

    assert detail_paths == ["/3/tv/2"]
    assert [result.status for result in results.tv_series] == [
        ResultStatus.UNRESOLVED,
        ResultStatus.UNRESOLVED,
        ResultStatus.UNRESOLVED,
    ]
    assert [result.tv_series_mention for result in results.tv_series] == mentions
    assert all(result.tv_series is None for result in results.tv_series)
    await resolver.aclose()


async def test_resolver_enriches_canonical_tv_identity_and_provider_ordered_metadata() -> None:
    """Use interpreted identity while retaining required provider metadata semantics."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 55,
                            "name": "Localized Provider Title",
                            "original_name": "  Canonical\tTitle ",
                            "first_air_date": "2019-01-01",
                        }
                    ]
                },
            )
        assert request.url.path == "/3/tv/55"
        assert request.url.params["append_to_response"] == "aggregate_credits,external_ids"
        return httpx.Response(
            200,
            json={
                "id": 55,
                "name": "Localized Provider Title",
                "status": "Ended",
                "last_air_date": "2022-07-01",
                "created_by": [
                    {"name": "Creator One"},
                    {"name": "Creator Two"},
                    {"name": "Creator One"},
                ],
                "overview": "Provider synopsis.",
                "poster_path": "/series.jpg",
                "vote_average": 8.2,
                "aggregate_credits": {
                    "cast": [
                        {"name": "Lead"},
                        {"name": "Lead"},
                        {"name": ""},
                        {"name": "Fourth"},
                        {"name": "Fifth"},
                        {"name": "Sixth"},
                    ]
                },
                "external_ids": {"imdb_id": " tt1234567 "},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500/")
    mention = TVSeriesMention(title="Canonical Title", year=2019)

    results = await resolver.resolve(_mentions(tv_series=[mention]))

    resolved = results.tv_series[0]
    assert resolved.status is ResultStatus.RESOLVED
    assert resolved.tv_series_mention is mention
    assert resolved.tv_series is not None
    assert resolved.tv_series.title == "Canonical Title"
    assert resolved.tv_series.first_air_year == 2019
    assert resolved.tv_series.last_air_year == 2022
    assert resolved.tv_series.cast == ["Lead", "Lead", "", "Fourth", "Fifth"]
    assert resolved.tv_series.creators == ["Creator One", "Creator Two"]
    assert resolved.tv_series.description == "Provider synopsis."
    assert resolved.tv_series.poster_url == "https://image.tmdb.org/t/p/w500/series.jpg"
    assert resolved.tv_series.tmdb_id == 55
    assert resolved.tv_series.tmdb_url == "https://www.themoviedb.org/tv/55"
    assert resolved.tv_series.imdb_id == "tt1234567"
    assert resolved.tv_series.imdb_url == "https://www.imdb.com/title/tt1234567/"
    assert resolved.tv_series.tmdb_score == 8.2
    await resolver.aclose()


@pytest.mark.parametrize(
    ("status", "last_air_date", "expected_year"),
    [
        ("Ended", "2025-02-01", 2025),
        ("Canceled", "2024-12-01", 2024),
        ("Returning Series", "2023-01-01", None),
        ("Ended", None, None),
        ("Ended", "", None),
        ("Canceled", "unknown", None),
    ],
)
async def test_resolver_maps_tv_last_air_year_only_for_complete_series(
    status: str,
    last_air_date: str | None,
    expected_year: int | None,
) -> None:
    """Expose final air years only for exact completed TMDB statuses and valid dates."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Status Test",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": 1,
                "name": "Status Test",
                "status": status,
                "last_air_date": last_air_date,
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(tv_series=[TVSeriesMention(title="Status Test", year=2020)])
    )

    assert results.tv_series[0].tv_series is not None
    assert results.tv_series[0].tv_series.last_air_year == expected_year
    await resolver.aclose()


@pytest.mark.parametrize(
    ("aggregate_credits", "expected_cast"),
    [
        ({"cast": [{"name": "One"}, {"name": "Two"}]}, ["One", "Two"]),
        ({}, []),
    ],
)
async def test_resolver_handles_short_or_absent_tv_aggregate_cast(
    aggregate_credits: dict[str, object],
    expected_cast: list[str],
) -> None:
    """Keep fewer-than-five and absent aggregate cast payloads valid and ordered."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Cast Test",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": 1,
                "name": "Cast Test",
                "aggregate_credits": aggregate_credits,
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(tv_series=[TVSeriesMention(title="Cast Test", year=2020)])
    )

    assert results.tv_series[0].tv_series is not None
    assert results.tv_series[0].tv_series.cast == expected_cast
    await resolver.aclose()


async def test_resolver_keeps_optional_tv_metadata_nullable() -> None:
    """Map missing optional TV provider metadata to required nullable fields."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Optional Metadata",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": 1,
                "name": "Optional Metadata",
                "status": "Returning Series",
                "last_air_date": "2023-01-01",
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(tv_series=[TVSeriesMention(title="Optional Metadata", year=2020)])
    )

    assert results.tv_series[0].tv_series is not None
    assert results.tv_series[0].tv_series.last_air_year is None
    assert results.tv_series[0].tv_series.poster_url is None
    assert results.tv_series[0].tv_series.imdb_id is None
    assert results.tv_series[0].tv_series.imdb_url is None
    assert results.tv_series[0].tv_series.creators == []
    await resolver.aclose()


async def test_resolver_preserves_per_kind_order_despite_out_of_order_completion() -> None:
    """Preserve independent mention order while Movie and TV requests overlap."""
    detail_completion_order: list[str] = []
    detail_delays = {
        "/3/movie/1": 0.04,
        "/3/movie/2": 0.03,
        "/3/tv/3": 0.02,
        "/3/tv/4": 0.01,
    }

    async def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/3/search/movie":
            identifier = int(request.url.params["query"].removesuffix(" Movie"))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": identifier,
                            "title": request.url.params["query"],
                            "first_air_date": "2020-01-01",
                            "release_date": "2020-01-01",
                        }
                    ]
                },
            )
        if path == "/3/search/tv":
            identifier = int(request.url.params["query"].removesuffix(" TV"))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": identifier,
                            "name": request.url.params["query"],
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )

        await asyncio.sleep(detail_delays[path])
        detail_completion_order.append(path)
        identifier = int(path.rsplit("/", maxsplit=1)[-1])
        if path.startswith("/3/movie/"):
            return httpx.Response(
                200,
                json={
                    "id": identifier,
                    "title": f"{identifier} Movie",
                    "release_date": "2020-01-01",
                },
            )
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "name": f"{identifier} TV",
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mentions = [
        MovieMention(title="1 Movie", year=2020),
        MovieMention(title="2 Movie", year=2020),
    ]
    tv_mentions = [
        TVSeriesMention(title="3 TV", year=2020),
        TVSeriesMention(title="4 TV", year=2020),
    ]

    results = await resolver.resolve(_mentions(movie_mentions, tv_mentions))

    assert detail_completion_order == ["/3/tv/4", "/3/tv/3", "/3/movie/2", "/3/movie/1"]
    assert [result.movie_mention for result in results.movies] == movie_mentions
    assert [result.tv_series_mention for result in results.tv_series] == tv_mentions
    await resolver.aclose()


async def test_resolver_maps_tmdb_http_failures_to_enrichment_error() -> None:
    """Expose TMDB HTTP failures through the extraction enrichment policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(_mentions(movies=[MovieMention(title="Dune: Part One", year=2021)]))

    await resolver.aclose()


async def test_resolver_maps_tmdb_timeouts_to_pipeline_timeout() -> None:
    """Expose TMDB timeouts through the shared pipeline timeout policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(PipelineTimeoutError, match="TMDB candidate resolution timed out"):
        await resolver.resolve(_mentions(movies=[MovieMention(title="Dune: Part One", year=2021)]))

    await resolver.aclose()


async def test_resolver_maps_invalid_tv_json_to_enrichment_error() -> None:
    """Map invalid JSON from a TV endpoint to the whole-request error policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{")

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(
            _mentions(tv_series=[TVSeriesMention(title="Invalid JSON", year=2020)])
        )

    await resolver.aclose()


@pytest.mark.parametrize("missing_field", ["aggregate_credits", "external_ids"])
async def test_resolver_rejects_missing_required_tv_appended_responses(
    missing_field: str,
) -> None:
    """Fail the complete grouped resolution when a requested TV append is absent."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Required Append",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )
        payload: dict[str, object] = {"id": 1, "name": "Required Append"}
        if missing_field != "aggregate_credits":
            payload["aggregate_credits"] = {}
        if missing_field != "external_ids":
            payload["external_ids"] = {}
        return httpx.Response(200, json=payload)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(
            _mentions(tv_series=[TVSeriesMention(title="Required Append", year=2020)])
        )

    await resolver.aclose()


async def test_resolver_aborts_grouped_resolution_without_partial_results() -> None:
    """Raise a TV provider failure instead of returning a partial grouped result."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Movie Success",
                            "release_date": "2020-01-01",
                        }
                    ]
                },
            )
        if request.url.path == "/3/movie/1":
            await asyncio.sleep(0.01)
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "title": "Movie Success",
                    "release_date": "2020-01-01",
                },
            )
        assert request.url.path == "/3/search/tv"
        return httpx.Response(503, request=request)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(
            _mentions(
                movies=[MovieMention(title="Movie Success", year=2020)],
                tv_series=[TVSeriesMention(title="TV Failure", year=2020)],
            )
        )

    await resolver.aclose()


async def test_factory_builds_closable_tmdb_resolver() -> None:
    """Build the production resolver from validated TMDB settings."""
    resolver = create_tmdb_screen_work_resolver(_settings())

    assert isinstance(resolver, TMDBScreenWorkResolver)

    await resolver.aclose()
