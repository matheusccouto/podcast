"""Tests for HackerNewsSource Story and comment collection."""

import httpx
import pytest

from podcast.models import Story
from podcast.sources import HackerNewsSource


@pytest.fixture
def live_source() -> HackerNewsSource:
    """Provide a default live HackerNewsSource instance."""
    return HackerNewsSource()


async def test_hackernews_parses_stories_and_filters_dead_items() -> None:
    """Offline Firebase payloads parse into Stories and skip dead/non-story items."""
    payloads: dict[str, object] = {
        "/v0/topstories.json": [101, 102, 103],
        "/v0/item/101.json": {
            "id": 101,
            "type": "story",
            "title": "Zero-Copy Ring Buffers",
            "url": "https://example.com/ringbuf",
            "score": 420,
            "descendants": 88,
        },
        "/v0/item/102.json": {
            "id": 102,
            "type": "story",
            "title": "Dead Story",
            "dead": True,
        },
        "/v0/item/103.json": {
            "id": 103,
            "type": "job",
            "title": "Hiring Engineers",
        },
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads[request.url.path])

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        stories = await HackerNewsSource(client=client).fetch_stories(limit=5)

    assert len(stories) == 1
    assert stories[0] == Story(
        id="101",
        source="hackernews",
        title="Zero-Copy Ring Buffers",
        url="https://example.com/ringbuf",
        discussion_url="https://news.ycombinator.com/item?id=101",
        score=420,
        comment_count=88,
    )


async def test_hackernews_strips_html_from_comments() -> None:
    """Offline comment payloads strip HTML tags and unescape HTML entities."""
    payloads: dict[str, object] = {
        "/v0/item/101.json": {"id": 101, "kids": [201, 202]},
        "/v0/item/201.json": {
            "id": 201,
            "text": "Latency &amp; throughput.<p>Second <i>line</i>.",
        },
        "/v0/item/202.json": {"id": 202, "deleted": True, "text": "gone"},
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads[request.url.path])

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        comments = await HackerNewsSource(client=client).fetch_comments("101", limit=5)

    assert comments == ["Latency & throughput.\nSecond line."]


@pytest.mark.live
async def test_hackernews_fetch_stories_returns_live_stories(
    live_source: HackerNewsSource,
) -> None:
    """Fetching stories returns top Hacker News Story metadata cleanly."""
    stories = await live_source.fetch_stories(limit=3)

    assert len(stories) == 3
    for story in stories:
        assert isinstance(story, Story)
        assert story.source == "hackernews"
        assert story.id
        assert story.title
        assert (
            story.discussion_url == f"https://news.ycombinator.com/item?id={story.id}"
        )
        assert story.score >= 0
        assert story.comment_count >= 0


@pytest.mark.live
async def test_hackernews_fetch_comments_returns_plain_text_comments(
    live_source: HackerNewsSource,
) -> None:
    """Fetching comments returns non-empty HTML-stripped top-level comments."""
    stories = await live_source.fetch_stories(limit=15)
    commented_story = next(story for story in stories if story.comment_count >= 3)

    comments = await live_source.fetch_comments(commented_story.id, limit=3)

    assert 1 <= len(comments) <= 3
    assert all(isinstance(comment, str) and comment.strip() for comment in comments)
    assert all("<p>" not in comment for comment in comments)


async def test_hackernews_raises_loudly_on_http_error() -> None:
    """Upstream API HTTP errors raise instead of silently returning empty lists."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code=503, text="upstream down"),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        source = HackerNewsSource(client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await source.fetch_stories(limit=3)
