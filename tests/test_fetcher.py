"""Tests for JinaFetcher web content extraction."""

import httpx
import pytest

from podcast.fetcher import JinaFetcher


@pytest.fixture
def live_fetcher() -> JinaFetcher:
    """Provide a default live JinaFetcher instance."""
    return JinaFetcher()


async def test_fetch_content_returns_markdown_for_readable_url(
    live_fetcher: JinaFetcher,
) -> None:
    """Fetching a readable public page through Jina Reader returns >= 200 chars."""
    content = await live_fetcher.fetch_content("https://docs.python.org/3/")

    assert content is not None
    assert len(content) >= 200


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (200, "too short"),
        (404, "missing"),
    ],
)
async def test_fetch_content_returns_none_for_unreadable_responses(
    status_code: int,
    body: str,
) -> None:
    """Short bodies and expected 4xx responses return None instead of failing."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code=status_code, text=body),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        assert await JinaFetcher(client=client).fetch_content("https://a.test") is None


async def test_fetch_content_raises_loudly_on_server_error() -> None:
    """Upstream 5xx responses fail loudly instead of being treated as unreadable."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code=500, text="upstream error"),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await JinaFetcher(client=client).fetch_content("https://a.test")
