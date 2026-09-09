"""TMDB candidate resolution contract tests."""

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


@pytest.mark.parametrize(
    ("provider_year", "expected_search_years"),
    [
        (2000 - 1, ["2000", "2001", "1999"]),
        (2000 + 1, ["2000", "2001"]),
    ],
)
async def test_resolver_searches_adjacent_years_after_exact_year_has_no_match(
    provider_year: int,
    expected_search_years: list[str],
) -> None:
    """Resolve a Movie whose provider release year differs by one."""
    requested_search_years: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            search_year = request.url.params["year"]
            requested_search_years.append(search_year)
            if int(search_year) != provider_year:
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target",
                            "release_date": f"{provider_year}-01-01",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/movie/1"
        assert request.url.params["append_to_response"] == "credits"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Target",
                "release_date": f"{provider_year}-01-01",
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Target", year=2000)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    assert requested_search_years == expected_search_years
    result = results.movies[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is not None
    assert result.movie.year == provider_year
    await resolver.aclose()


async def test_resolver_rejects_movie_details_release_year_more_than_one_year_away() -> None:
    """Leave a Movie unresolved when its provider release year differs by two."""
    requested_search_years: list[str] = []
    requested_detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            search_year = request.url.params["year"]
            requested_search_years.append(search_year)
            if search_year != "1999":
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target",
                            "release_date": "1999-01-01",
                        }
                    ]
                },
            )

        requested_detail_paths.append(request.url.path)
        assert request.url.path == "/3/movie/1"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Target",
                "release_date": "1998-01-01",
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Target", year=2000)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    assert requested_search_years == ["2000", "2001", "1999"]
    assert requested_detail_paths == ["/3/movie/1"]
    result = results.movies[0]
    assert result.status is ResultStatus.UNRESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is None
    await resolver.aclose()


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


async def test_resolver_searches_only_first_page_per_year_before_returning_unresolved() -> None:
    """Inspect only page one for each tolerated Movie release year."""
    requested_searches: list[tuple[str, str]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/3/search/movie"
        search_year = request.url.params["year"]
        page = request.url.params["page"]
        requested_searches.append((search_year, page))
        assert page == "1"
        return httpx.Response(
            200,
            json={
                "results": [{"id": 1, "release_date": "1998-01-01"}],
                "total_pages": 2,
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Missing Film", year=2000)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    assert requested_searches == [("2000", "1"), ("2001", "1"), ("1999", "1")]
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


@pytest.mark.parametrize(
    ("provider_year", "expected_search_years"),
    [
        (2000 - 1, ["2000", "2001", "1999"]),
        (2000 + 1, ["2000", "2001"]),
    ],
)
async def test_resolver_searches_tv_adjacent_years_after_exact_year_has_no_match(
    provider_year: int,
    expected_search_years: list[str],
) -> None:
    """Resolve a TV Series whose provider first air year differs by one."""
    requested_search_years: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            search_year = request.url.params["first_air_date_year"]
            requested_search_years.append(search_year)
            if int(search_year) != provider_year:
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Target",
                            "first_air_date": f"{provider_year}-01-01",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/tv/1"
        assert request.url.params["append_to_response"] == "aggregate_credits,external_ids"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "name": "Target",
                "aggregate_credits": {"cast": []},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    tv_series_mention = TVSeriesMention(title="Target", year=2000)

    results = await resolver.resolve(_mentions(tv_series=[tv_series_mention]))

    assert requested_search_years == expected_search_years
    result = results.tv_series[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.tv_series_mention is tv_series_mention
    assert result.tv_series_mention.year == 2000
    assert result.tv_series is not None
    assert result.tv_series.first_air_year == provider_year
    await resolver.aclose()


async def test_resolver_limits_tv_resolution_to_first_page_and_three_candidates() -> None:
    """Inspect only page one and the first three candidates for each tolerated TV year."""
    requested_searches: list[tuple[str, str]] = []
    requested_detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            search_year = int(request.url.params["first_air_date_year"])
            page = request.url.params["page"]
            requested_searches.append((str(search_year), page))
            assert page == "1"
            return httpx.Response(
                200,
                json={
                    "total_pages": 2,
                    "results": [
                        {
                            "id": search_year * 10 + identifier,
                            "name": "No Match",
                            "first_air_date": f"{search_year}-01-01",
                        }
                        for identifier in range(1, 5)
                    ],
                },
            )

        requested_detail_paths.append(request.url.path)
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

    assert requested_searches == [("2020", "1"), ("2021", "1"), ("2019", "1")]
    assert requested_detail_paths == [
        f"/3/tv/{search_year * 10 + identifier}"
        for search_year in (2020, 2021, 2019)
        for identifier in range(1, 4)
    ]
    assert results.tv_series[0].status is ResultStatus.UNRESOLVED
    assert results.tv_series[0].tv_series is None
    await resolver.aclose()


async def test_resolver_returns_first_matching_tv_candidate() -> None:
    """Select the first TV candidate that matches the canonical title and year."""
    requested_search_years: list[str] = []
    detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            requested_search_years.append(request.url.params["first_air_date_year"])
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

    assert requested_search_years == ["2015"]
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
                                "first_air_date": "1998-01-01",
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

    assert detail_paths == ["/3/tv/2", "/3/tv/1"]
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
    assert resolved.tv_series is not None
    assert resolved.tv_series.title == "Localized Provider Title"
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


async def test_resolver_waits_for_every_strict_resolution_before_fuzzy_fallback() -> None:
    """Start fuzzy detail loads only after every strict resolution has completed."""
    strict_detail_started = asyncio.Event()
    release_strict_detail = asyncio.Event()
    fuzzy_detail_requested = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target Movi",
                            "release_date": "",
                        }
                    ]
                },
            )
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 2,
                            "name": "Exact Series",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )
        if request.url.path == "/3/movie/1":
            fuzzy_detail_requested.set()
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "title": "Target Movi",
                    "release_date": "2020-01-01",
                },
            )

        assert request.url.path == "/3/tv/2"
        strict_detail_started.set()
        await release_strict_detail.wait()
        return httpx.Response(
            200,
            json={
                "id": 2,
                "name": "Exact Series",
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    fuzzy_movie_mention = MovieMention(title="Target Movie", year=2020)
    exact_tv_mention = TVSeriesMention(title="Exact Series", year=2020)

    resolve_task = asyncio.create_task(
        resolver.resolve(
            _mentions(
                movies=[fuzzy_movie_mention],
                tv_series=[exact_tv_mention],
            )
        )
    )
    await strict_detail_started.wait()
    await asyncio.sleep(0)
    assert fuzzy_detail_requested.is_set() is False

    release_strict_detail.set()
    results = await resolve_task

    assert results.movies[0].movie_mention is fuzzy_movie_mention
    assert results.movies[0].movie is not None
    assert results.movies[0].movie.tmdb_id == 1
    assert results.tv_series[0].tv_series_mention is exact_tv_mention
    assert results.tv_series[0].tv_series is not None
    assert results.tv_series[0].tv_series.tmdb_id == 2
    await resolver.aclose()


@pytest.mark.parametrize("match_source", ["primary", "original", "alternative"])
async def test_resolver_keeps_strict_exact_matches_ahead_of_fuzzy_candidates(
    match_source: str,
) -> None:
    """Choose a later exact candidate instead of an earlier fuzzy candidate."""
    requested_search_years: list[str] = []
    detail_appends: dict[str, str] = {}

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            requested_search_years.append(request.url.params["year"])
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target Movi",
                            "release_date": "2020-01-01",
                        },
                        {
                            "id": 2,
                            "title": (
                                "Target Movie" if match_source == "primary" else "Provider Title"
                            ),
                            "original_title": (
                                "Target Movie"
                                if match_source == "original"
                                else "Provider Original"
                            ),
                            "release_date": "2020-01-01",
                        },
                    ]
                },
            )

        detail_appends[request.url.path] = request.url.params["append_to_response"]
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "title": "Target Movie",
                "release_date": "2020-01-01",
                "alternative_titles": {
                    "titles": (
                        [{"title": "Target Movie"}]
                        if identifier == 2 and match_source == "alternative"
                        else []
                    )
                },
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(movies=[MovieMention(title="Target Movie", year=2020)])
    )

    assert requested_search_years == ["2020"]
    assert detail_appends == {
        "/3/movie/1": "credits,alternative_titles",
        "/3/movie/2": (
            "credits,alternative_titles" if match_source == "alternative" else "credits"
        ),
    }
    assert results.movies[0].status is ResultStatus.RESOLVED
    assert results.movies[0].movie is not None
    assert results.movies[0].movie.tmdb_id == 2
    await resolver.aclose()


async def test_resolver_fuzzy_matches_movie_primary_title_with_unicode_normalization() -> None:
    """Resolve a Movie through a primary title with case and Unicode variation."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "  ame\u0301lie  ",
                            "original_title": "Provider Original",
                            "release_date": "2020-01-01",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/movie/1"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Amélie",
                "release_date": "2020-01-01",
                "alternative_titles": {"titles": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Amélie", year=2020)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    result = results.movies[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is not None
    assert result.movie.tmdb_id == 1
    await resolver.aclose()


async def test_resolver_fuzzy_matches_tv_original_title_with_collapsed_whitespace() -> None:
    """Resolve a TV Series from its original title and detailed first air year."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 2,
                            "name": "Provider Name",
                            "original_name": "  ame\u0301lie\t",
                            "first_air_date": "2020-01-01",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/tv/2"
        return httpx.Response(
            200,
            json={
                "id": 2,
                "name": "Provider Name",
                "first_air_date": "2019-01-01",
                "aggregate_credits": {},
                "external_ids": {},
                "alternative_titles": {"titles": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    tv_series_mention = TVSeriesMention(title="AMÉLIE", year=2020)

    results = await resolver.resolve(_mentions(tv_series=[tv_series_mention]))

    result = results.tv_series[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.tv_series_mention is tv_series_mention
    assert result.tv_series is not None
    assert result.tv_series.tmdb_id == 2
    assert result.tv_series.first_air_year == 2019
    await resolver.aclose()


@pytest.mark.parametrize(
    ("mention_title", "provider_title", "expected_status"),
    [
        ("Test Movie", "Testing Movif", ResultStatus.UNRESOLVED),
        ("Test Movie", "Test Movi", ResultStatus.RESOLVED),
        ("Amélie", "Amelie", ResultStatus.RESOLVED),
        ("A/B/C", "A-B-C", ResultStatus.UNRESOLVED),
    ],
)
async def test_resolver_fuzzy_title_threshold_preserves_accents_and_punctuation(
    mention_title: str,
    provider_title: str,
    expected_status: ResultStatus,
) -> None:
    """Use only scores above ninety without removing accents or punctuation."""
    detail_requests: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": provider_title,
                            "release_date": "",
                        }
                    ]
                },
            )

        detail_requests.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": provider_title,
                "release_date": "2020-01-01",
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title=mention_title, year=2020)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    result = results.movies[0]
    assert result.status is expected_status
    assert result.movie_mention is movie_mention
    assert (result.movie is not None) is (expected_status is ResultStatus.RESOLVED)
    assert detail_requests == (["/3/movie/1"] if expected_status is ResultStatus.RESOLVED else [])
    await resolver.aclose()


async def test_resolver_fuzzy_resolution_ignores_alternative_titles() -> None:
    """Leave a Mention unresolved when only an alternative title is near."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Unrelated Primary",
                            "original_title": "Unrelated Original",
                            "release_date": "2020-01-01",
                        }
                    ]
                },
            )

        assert request.url.path == "/3/movie/1"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Unrelated Primary",
                "release_date": "2020-01-01",
                "alternative_titles": {"titles": [{"title": "Target Movi"}]},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Target Movie", year=2020)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    result = results.movies[0]
    assert result.status is ResultStatus.UNRESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is None
    await resolver.aclose()


async def test_resolver_fuzzy_uses_provider_order_within_three_candidate_bounds() -> None:
    """Choose the first qualifying retained Movie Candidate without more searches."""
    requested_search_years: list[str] = []
    requested_detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            search_year = int(request.url.params["year"])
            requested_search_years.append(str(search_year))
            if search_year == 2020:
                results = [
                    {
                        "id": 100 + identifier,
                        "title": "No Match",
                        "release_date": "",
                    }
                    for identifier in range(3)
                ]
                results.append(
                    {
                        "id": 104,
                        "title": "Target Movi",
                        "release_date": "2020-01-01",
                    }
                )
            elif search_year == 2021:
                results = [
                    {
                        "id": 201,
                        "title": "No Match",
                        "release_date": "",
                    },
                    {
                        "id": 202,
                        "title": "Target Movi",
                        "release_date": "2021-01-01",
                    },
                    {
                        "id": 203,
                        "title": "target movie",
                        "release_date": "2021-01-01",
                    },
                    {
                        "id": 204,
                        "title": "No Match",
                        "release_date": "",
                    },
                ]
            else:
                results = [
                    {
                        "id": 300 + identifier,
                        "title": "No Match",
                        "release_date": "",
                    }
                    for identifier in range(4)
                ]
            return httpx.Response(200, json={"results": results, "total_pages": 2})

        requested_detail_paths.append(request.url.path)
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "title": "Target Movie",
                "release_date": "2021-01-01",
                "alternative_titles": {"titles": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(movies=[MovieMention(title="Target Movie", year=2020)])
    )

    assert requested_search_years == ["2020", "2021", "2019"]
    assert requested_detail_paths == ["/3/movie/202", "/3/movie/203"]
    assert "/3/movie/104" not in requested_detail_paths
    assert results.movies[0].movie is not None
    assert results.movies[0].movie.tmdb_id == 202
    await resolver.aclose()


async def test_resolver_fuzzy_reuses_details_loaded_by_strict_verification() -> None:
    """Reuse a strict verification response instead of fetching fuzzy details twice."""
    detail_requests: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target Movi",
                            "release_date": "2020-01-01",
                        }
                    ]
                },
            )

        detail_requests.append(request.url.path)
        assert request.url.params["append_to_response"] == "credits,alternative_titles"
        return httpx.Response(
            200,
            json={
                "id": 1,
                "title": "Target Movi",
                "release_date": "2020-01-01",
                "alternative_titles": {"titles": []},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve(
        _mentions(movies=[MovieMention(title="Target Movie", year=2020)])
    )

    assert detail_requests == ["/3/movie/1"]
    assert results.movies[0].movie is not None
    assert results.movies[0].movie.tmdb_id == 1
    await resolver.aclose()


@pytest.mark.parametrize(
    "first_detail_release_date",
    [None, "", "not-a-date", "2018-01-01"],
)
async def test_resolver_fuzzy_skips_movies_without_acceptable_detail_release_year(
    first_detail_release_date: object,
) -> None:
    """Continue after a missing, malformed, or distant Movie release date."""
    requested_detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            results = (
                [
                    {
                        "id": 1,
                        "title": "Target Movi",
                        "release_date": "",
                    },
                    {
                        "id": 2,
                        "title": "Target Movix",
                        "release_date": "",
                    },
                ]
                if request.url.params["year"] == "2020"
                else []
            )
            return httpx.Response(200, json={"results": results})

        requested_detail_paths.append(request.url.path)
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "title": "Target Movie",
                "release_date": (first_detail_release_date if identifier == 1 else "2021-01-01"),
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Target Movie", year=2020)

    results = await resolver.resolve(_mentions(movies=[movie_mention]))

    result = results.movies[0]
    assert requested_detail_paths == ["/3/movie/1", "/3/movie/2"]
    assert result.status is ResultStatus.RESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is not None
    assert result.movie.tmdb_id == 2
    assert result.movie.year == 2021
    await resolver.aclose()


@pytest.mark.parametrize(
    "first_detail_first_air_date",
    [None, "", "not-a-date", "2018-01-01"],
)
async def test_resolver_fuzzy_skips_tv_without_acceptable_detail_first_air_year(
    first_detail_first_air_date: object,
) -> None:
    """Continue after a missing, malformed, or distant TV first air date."""
    requested_detail_paths: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            results = (
                [
                    {
                        "id": 1,
                        "name": "Target Serie",
                        "first_air_date": "",
                    },
                    {
                        "id": 2,
                        "name": "Target Seriez",
                        "first_air_date": "",
                    },
                ]
                if request.url.params["first_air_date_year"] == "2020"
                else []
            )
            return httpx.Response(200, json={"results": results})

        requested_detail_paths.append(request.url.path)
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        return httpx.Response(
            200,
            json={
                "id": identifier,
                "name": "Target Series",
                "first_air_date": (
                    first_detail_first_air_date if identifier == 1 else "2021-01-01"
                ),
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    tv_series_mention = TVSeriesMention(title="Target Series", year=2020)

    results = await resolver.resolve(_mentions(tv_series=[tv_series_mention]))

    result = results.tv_series[0]
    assert requested_detail_paths == ["/3/tv/1", "/3/tv/2"]
    assert result.status is ResultStatus.RESOLVED
    assert result.tv_series_mention is tv_series_mention
    assert result.tv_series is not None
    assert result.tv_series.tmdb_id == 2
    assert result.tv_series.first_air_year == 2021
    await resolver.aclose()


@pytest.mark.parametrize("missing_field", ["id", "title"])
async def test_resolver_rejects_missing_required_movie_fallback_details(
    missing_field: str,
) -> None:
    """Fail the grouped request when fuzzy Movie details lack identity fields."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target Movi",
                            "release_date": "",
                        }
                    ]
                },
            )

        payload: dict[str, object] = {
            "id": 1,
            "title": "Target Movie",
            "release_date": "2020-01-01",
        }
        payload.pop(missing_field)
        return httpx.Response(200, json=payload)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(_mentions(movies=[MovieMention(title="Target Movie", year=2020)]))

    await resolver.aclose()


@pytest.mark.parametrize("missing_field", ["aggregate_credits", "external_ids"])
async def test_resolver_rejects_missing_required_tv_fallback_details(
    missing_field: str,
) -> None:
    """Fail the grouped request when fuzzy TV details lack required appends."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "name": "Target Serie",
                            "first_air_date": "",
                        }
                    ]
                },
            )

        payload: dict[str, object] = {
            "id": 1,
            "name": "Target Series",
            "first_air_date": "2020-01-01",
            "aggregate_credits": {},
            "external_ids": {},
        }
        payload.pop(missing_field)
        return httpx.Response(200, json=payload)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve(
            _mentions(tv_series=[TVSeriesMention(title="Target Series", year=2020)])
        )

    await resolver.aclose()


async def test_resolver_preserves_fallback_result_order_after_reverse_detail_completion() -> None:
    """Keep Movie and TV result order when fuzzy details complete in reverse."""
    detail_completion_order: list[str] = []
    detail_delays = {
        "/3/movie/1": 0.04,
        "/3/movie/2": 0.03,
        "/3/tv/3": 0.02,
        "/3/tv/4": 0.01,
    }

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            identifier = {"One Movie": 1, "Two Movie": 2}[request.url.params["query"]]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": identifier,
                            "title": f"{request.url.params['query']}e",
                            "release_date": "",
                        }
                    ]
                },
            )
        if request.url.path == "/3/search/tv":
            identifier = {"Three TV": 3, "Four TV": 4}[request.url.params["query"]]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": identifier,
                            "name": f"{request.url.params['query']}e",
                            "first_air_date": "",
                        }
                    ]
                },
            )

        await asyncio.sleep(detail_delays[request.url.path])
        detail_completion_order.append(request.url.path)
        identifier = int(request.url.path.rsplit("/", maxsplit=1)[-1])
        if request.url.path.startswith("/3/movie/"):
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
                "first_air_date": "2020-01-01",
                "aggregate_credits": {},
                "external_ids": {},
            },
        )

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mentions = [
        MovieMention(title="One Movie", year=2020),
        MovieMention(title="Two Movie", year=2020),
    ]
    tv_series_mentions = [
        TVSeriesMention(title="Three TV", year=2020),
        TVSeriesMention(title="Four TV", year=2020),
    ]

    results = await resolver.resolve(_mentions(movie_mentions, tv_series_mentions))

    assert detail_completion_order == ["/3/tv/4", "/3/tv/3", "/3/movie/2", "/3/movie/1"]
    assert [result.movie_mention for result in results.movies] == movie_mentions
    assert [result.tv_series_mention for result in results.tv_series] == tv_series_mentions
    assert [result.movie.tmdb_id for result in results.movies if result.movie] == [1, 2]
    assert [result.tv_series.tmdb_id for result in results.tv_series if result.tv_series] == [3, 4]
    await resolver.aclose()


@pytest.mark.parametrize(
    ("failure", "expected_exception"),
    [
        ("http", EnrichmentError),
        ("timeout", PipelineTimeoutError),
        ("invalid_json", EnrichmentError),
    ],
)
async def test_resolver_maps_fallback_detail_failures(
    failure: str,
    expected_exception: type[Exception],
) -> None:
    """Map every fuzzy Movie detail failure through the extraction policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Target Movi",
                            "release_date": "",
                        }
                    ]
                },
            )
        if failure == "http":
            return httpx.Response(503, request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, content=b"{")

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBScreenWorkResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(expected_exception, match="TMDB candidate resolution"):
        await resolver.resolve(_mentions(movies=[MovieMention(title="Target Movie", year=2020)]))

    await resolver.aclose()


async def test_resolver_aborts_grouped_fallback_resolution_without_partial_results() -> None:
    """Raise a fallback provider failure after a sibling fallback has succeeded."""
    successful_movie_detail_loaded = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/search/movie":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1,
                            "title": "Movie Succes",
                            "release_date": "",
                        }
                    ]
                },
            )
        if request.url.path == "/3/search/tv":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 2,
                            "name": "TV Failur",
                            "first_air_date": "",
                        }
                    ]
                },
            )
        if request.url.path == "/3/movie/1":
            successful_movie_detail_loaded.set()
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "title": "Movie Succes",
                    "release_date": "2020-01-01",
                },
            )

        assert request.url.path == "/3/tv/2"
        await successful_movie_detail_loaded.wait()
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
