"""Upstream news Sources and the structural Source protocol."""

from typing import Protocol, runtime_checkable

from podcast.models import Story
from podcast.sources.hackernews import HackerNewsSource


@runtime_checkable
class Source(Protocol):
    """Structural interface implemented by upstream news Sources."""

    async def fetch_stories(self, limit: int = 30) -> list[Story]:
        """Return candidate Stories from this Source."""
        ...

    async def fetch_comments(self, story_id: str, limit: int = 15) -> list[str]:
        """Return plain-text discussion comments for a Story."""
        ...


__all__ = ["HackerNewsSource", "Source", "Story"]
