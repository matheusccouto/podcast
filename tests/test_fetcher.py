"""Tests for ExaFetcher web content extraction."""

import json

import httpx
import pytest
from pydantic import SecretStr

from podcast.fetcher import ExaFetcher
from podcast.settings import Settings

_GATED_REUTERS_URL = (
    "https://www.reuters.com/business/finance/"
    "anthropics-historic-debut-put-ais-promise-peril-wall-st-test-2026-11-19/"
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"results": [{"text": "A" * 250}]}, "A" * 250),
        ({"results": [], "statuses": [{"status": "error"}]}, None),
        ({"results": [{"text": "too short"}]}, None),
    ],
)
async def test_exa_fetcher_extracts_or_returns_none(
    payload: dict[str, object],
    expected: str | None,
) -> None:
    """ExaFetcher returns extracted text or None on crawl errors/short bodies."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code=200, json=payload),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = ExaFetcher(api_key=SecretStr("test-key"), client=client)
        assert await fetcher.fetch_content("https://a.test") == expected


async def test_exa_fetcher_searches_open_coverage_when_url_is_unreadable() -> None:
    """When `/contents` fails and `title` is given, ExaFetcher queries `/search`."""
    article = "B" * 250
    captured_search: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/contents":
            return httpx.Response(200, json={"results": []})
        captured_search.update(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"text": article}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        fetcher = ExaFetcher(api_key=SecretStr("test-key"), client=client)
        result = await fetcher.fetch_content(
            "https://www.reuters.com/story",
            title="Anthropic's Nasdaq debut",
        )

    assert result == article
    assert captured_search["query"] == "Anthropic's Nasdaq debut"
    assert captured_search["excludeDomains"] == ["reuters.com"]


async def test_exa_fetcher_raises_loudly_on_server_error() -> None:
    """Upstream 5xx responses fail loudly instead of returning None."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code=500, text="upstream error"),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = ExaFetcher(api_key=SecretStr("test-key"), client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await fetcher.fetch_content("https://a.test")


@pytest.mark.live
async def test_exa_fetcher_retrieves_content_for_gated_reuters_story(
    live_settings: Settings,
) -> None:
    """ExaFetcher retrieves open coverage when the original Reuters URL is gated."""
    async with httpx.AsyncClient() as client:
        fetcher = ExaFetcher(api_key=live_settings.exa_api_key, client=client)
        content = await fetcher.fetch_content(
            _GATED_REUTERS_URL,
            title="Anthropic's Nasdaq debut",
        )

    assert content is not None
    assert len(content) >= 200
