"""Hacker News upstream Source using daily front pages and the Firebase JSON API."""

import asyncio
import re
from contextlib import nullcontext
from datetime import date
from html import unescape
from typing import Any

import httpx

from podcast.models import Story

_DEFAULT_BASE_URL = "https://hacker-news.firebaseio.com/v0"
_DEFAULT_FRONT_URL = "https://news.ycombinator.com/front"
_REQUEST_TIMEOUT_SECONDS = 15.0
_MAX_CONCURRENCY = 10
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FRONT_ITEM_ID_RE = re.compile(r'<tr[^>]*class="athing[^"]*"[^>]*id="(\d+)"')


class HackerNewsSource:
    """Collector for Hacker News daily front-page Stories and comments."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        front_url: str = _DEFAULT_FRONT_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure API/front-page URLs and optional shared HTTP client."""
        self.base_url = base_url.rstrip("/")
        self.front_url = front_url.rstrip("/")
        self.client = client

    def _client_ctx(self) -> nullcontext[httpx.AsyncClient] | httpx.AsyncClient:
        """Return the injected shared client or a short-lived AsyncClient."""
        if self.client is not None:
            return nullcontext(self.client)
        return httpx.AsyncClient()

    async def _fetch_items(
        self,
        client: httpx.AsyncClient,
        item_ids: list[int],
    ) -> list[dict[str, Any] | None]:
        """Fetch multiple Hacker News items concurrently under bounded concurrency."""
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _get(item_id: int) -> dict[str, Any] | None:
            async with semaphore:
                response = await client.get(
                    f"{self.base_url}/item/{item_id}.json",
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload: dict[str, Any] | None = response.json()
                return payload

        return await asyncio.gather(*(_get(item_id) for item_id in item_ids))

    async def fetch_stories(self, day: date, limit: int = 30) -> list[Story]:
        """Return Hacker News front-page Stories for `day` without comments."""
        if limit <= 0:
            return []

        async with self._client_ctx() as active_client:
            response = await active_client.get(
                self.front_url,
                params={"day": day.isoformat()},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            candidate_ids = [
                int(match) for match in _FRONT_ITEM_ID_RE.findall(response.text)
            ][: limit * 2]
            entries = await self._fetch_items(active_client, candidate_ids)

        stories: list[Story] = []
        for entry in entries:
            if not entry or entry.get("dead") or entry.get("deleted"):
                continue
            if entry.get("type") != "story" or not entry.get("title"):
                continue
            story_id = str(entry["id"])
            stories.append(
                Story(
                    id=story_id,
                    source="hackernews",
                    title=str(entry["title"]),
                    url=entry.get("url"),
                    discussion_url=f"https://news.ycombinator.com/item?id={story_id}",
                    score=int(entry.get("score") or 0),
                    comment_count=int(entry.get("descendants") or 0),
                ),
            )
            if len(stories) >= limit:
                break
        return stories

    async def fetch_comments(self, story_id: str, limit: int = 15) -> list[str]:
        """Return plain-text top-level comments for a Hacker News Story."""
        if limit <= 0:
            return []

        async with self._client_ctx() as active_client:
            entries = await self._fetch_items(active_client, [int(story_id)])
            story_entry = entries[0]
            if not story_entry:
                return []
            kid_ids = [int(kid) for kid in story_entry.get("kids", [])][: limit * 2]
            comment_entries = await self._fetch_items(active_client, kid_ids)

        comments: list[str] = []
        for entry in comment_entries:
            if not entry or entry.get("dead") or entry.get("deleted"):
                continue
            raw_text = entry.get("text")
            if not isinstance(raw_text, str):
                continue
            cleaned = unescape(
                _HTML_TAG_RE.sub("", raw_text.replace("<p>", "\n")),
            ).strip()
            if not cleaned:
                continue
            comments.append(cleaned)
            if len(comments) >= limit:
                break
        return comments
