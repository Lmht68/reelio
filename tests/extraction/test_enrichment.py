"""TMDB candidate resolution and enrichment contract tests."""

from collections.abc import Callable
from typing import cast

import httpx
import pytest

from reelio.extraction.exceptions import EnrichmentError, PipelineTimeoutError
from reelio.extraction.services.enrichment.config import TMDBConfig
from reelio.extraction.services.enrichment.tmdb import (
    TMDBMovieResolver,
    create_tmdb_movie_resolver,
)
from reelio.extraction.types import MovieMention, ResultStatus


def _settings(**values: object) -> TMDBConfig:
    settings_type = cast(Callable[..., TMDBConfig], TMDBConfig)
    return settings_type(_env_file=None, api_key="test-tmdb-key", **values)


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.themoviedb.org/3/",
        transport=handler,
    )


async def test_resolver_selects_first_title_and_year_match_and_enriches() -> None:
    """Select the first title-and-year match from the first search page."""
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
                            "original_title": "Amélie",
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
                "release_date": "2001-04-25",
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
                        {"name": "Bruno Delbonnel", "job": "Director of Photography"},
                    ],
                },
            },
        )

    transport = httpx.MockTransport(handle)
    client = _client(transport)
    resolver = TMDBMovieResolver(client, "https://image.tmdb.org/t/p/w500/")
    movie_mention = MovieMention(title="Amélie", year=2001)

    results = await resolver.resolve([movie_mention])

    assert requested_paths == ["/3/search/movie", "/3/movie/1", "/3/movie/2"]
    assert len(results) == 1
    result = results[0]
    assert result.status is ResultStatus.RESOLVED
    assert result.movie_mention is movie_mention
    assert result.movie is not None
    assert result.movie.title == "Amélie"
    assert result.movie.year == 2001
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
    resolver = TMDBMovieResolver(client, "https://image.tmdb.org/t/p/w500")

    results = await resolver.resolve([MovieMention(title="Amélie", year=2001)])

    assert results[0].status is ResultStatus.RESOLVED
    assert results[0].movie is not None
    assert results[0].movie.tmdb_id == 1
    await resolver.aclose()


async def test_resolver_searches_first_page_only_before_returning_unresolved() -> None:
    """Inspect only the first candidate page before returning an unresolved result."""
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
    resolver = TMDBMovieResolver(client, "https://image.tmdb.org/t/p/w500")
    movie_mention = MovieMention(title="Missing Film", year=2000)

    results = await resolver.resolve([movie_mention])

    assert requested_pages == ["1"]
    assert results[0].status is ResultStatus.UNRESOLVED
    assert results[0].movie_mention is movie_mention
    assert results[0].movie is None
    await resolver.aclose()


async def test_resolver_maps_tmdb_http_failures_to_enrichment_error() -> None:
    """Expose TMDB HTTP failures through the extraction enrichment policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBMovieResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(EnrichmentError, match="TMDB candidate resolution"):
        await resolver.resolve([MovieMention(title="Dune: Part One", year=2021)])

    await resolver.aclose()


async def test_resolver_maps_tmdb_timeouts_to_pipeline_timeout() -> None:
    """Expose TMDB timeouts through the shared pipeline timeout policy."""

    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(httpx.MockTransport(handle))
    resolver = TMDBMovieResolver(client, "https://image.tmdb.org/t/p/w500")

    with pytest.raises(PipelineTimeoutError, match="TMDB candidate resolution timed out"):
        await resolver.resolve([MovieMention(title="Dune: Part One", year=2021)])

    await resolver.aclose()


async def test_factory_builds_closable_tmdb_resolver() -> None:
    """Build the production resolver from validated TMDB settings."""
    resolver = create_tmdb_movie_resolver(_settings())

    assert isinstance(resolver, TMDBMovieResolver)

    await resolver.aclose()
