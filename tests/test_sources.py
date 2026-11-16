"""Tests for HackerNewsSource Story and comment collection."""

import httpx
import pytest

from podcast.models import Story
from podcast.sources import HackerNewsSource


@pytest.fixture
def live_source() -> HackerNewsSource:
    """Provide a default live HackerNewsSource instance."""
    return HackerNewsSource()


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
