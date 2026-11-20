"""Jev System One Curator ranking and hydrating daily candidate Stories."""

import logging
from contextlib import nullcontext
from datetime import date
from importlib import resources
from typing import Any

import httpx
from pydantic import SecretStr

from podcast.fetcher import ContentFetcher
from podcast.models import MAX_CONTENT_CHARS, CuratedStory, Segment, Story
from podcast.sources import HackerNewsSource, Source

logger = logging.getLogger(__name__)

_SYSTEM_ONE_URL = "https://openrouter.ai/api/v1/systemone"
_REQUEST_TIMEOUT_SECONDS = 60.0
_TASTE_PROFILE = (
    resources.files(__package__).joinpath("instructions.md").read_text("utf-8").strip()
)
_SCORE_LEVELS = [
    "Skip: Show HN, hobby/retro, niche kernel/compiler internals, or minor crypto",
    "Low: Minor hardware/systems or niche security story",
    "Medium: Solid software engineering, AI dev workflow, or security/privacy story",
    "High: Major tech/business/society/policy impact or notable AI news",
    (
        "Headliner (#1): Biggest front-page news of the day — flagship AI launch, "
        "major tech/business milestone, or huge public shift"
    ),
]


class Curator:
    """Ranks daily candidate Stories via OpenRouter Jev and hydrates readable picks."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        fetcher: ContentFetcher,
        model: str = "jev-1.14",
        source: Source | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure Jev credentials, model, fetcher, and optional source/client."""
        self.api_key = api_key
        self.fetcher = fetcher
        self.model = model
        self.source = source
        self.client = client

    async def _rank_stories(
        self,
        *,
        day: date,
        stories: list[Story],
        client: httpx.AsyncClient,
    ) -> list[Story]:
        """Rank candidate Stories in a single OpenRouter System One Jev call."""
        state = {
            "news_date": day.isoformat(),
            "taste_profile": _TASTE_PROFILE,
            "candidates": [
                {
                    "id": story.id,
                    "title": story.title,
                    "url": story.url,
                    "score": story.score,
                    "comments": story.comment_count,
                }
                for story in stories
            ],
        }
        questions = {
            story.id: {
                "type": "score",
                "instructions": (
                    f"Rate story {story.id} ('{story.title}') against the Taste "
                    "Profile and HN engagement (score + comments) relative to the "
                    "other candidates."
                ),
                "criteria": _SCORE_LEVELS,
            }
            for story in stories
        }
        response = await client.post(
            _SYSTEM_ONE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "state": state, "questions": questions},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        answers: dict[str, dict[str, Any]] = payload.get("answers") or {}
        story_by_id = {story.id: story for story in stories}
        scored = [
            (story_by_id[sid], float(answer["score"]))
            for sid, answer in answers.items()
            if sid in story_by_id
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [story for story, _ in scored]

    async def curate(
        self,
        day: date,
        *,
        limit: int = 5,
    ) -> tuple[list[Segment], list[CuratedStory]]:
        """Rank `day`'s Stories with Jev and hydrate up to `limit` readable picks."""
        ctx = (
            nullcontext(self.client) if self.client is not None else httpx.AsyncClient()
        )
        async with ctx as active_client:
            active_source = self.source or HackerNewsSource(client=active_client)
            candidates = await active_source.fetch_stories(day, limit=30)
            if not candidates:
                msg = f"No candidate Stories returned for {day.isoformat()}."
                raise RuntimeError(msg)

            ranked = await self._rank_stories(
                day=day,
                stories=candidates,
                client=active_client,
            )

            segments: list[Segment] = []
            curated: list[CuratedStory] = []
            for story in ranked:
                if story.url:
                    raw_content = await self.fetcher.fetch_content(
                        story.url,
                        title=story.title,
                    )
                    if raw_content is None:
                        logger.info(
                            "Skipping unreadable Story %s: %s",
                            story.id,
                            story.url,
                        )
                        continue
                    content = raw_content[:MAX_CONTENT_CHARS]
                else:
                    content = story.title

                comments = await active_source.fetch_comments(story.id)
                segments.append(
                    Segment(story_id=story.id, title=story.title, url=story.url),
                )
                curated.append(
                    CuratedStory(
                        title=story.title,
                        url=story.url,
                        content=content,
                        comments=comments,
                    ),
                )
                if len(curated) >= limit:
                    break

        if not curated:
            msg = f"No readable Stories found for {day.isoformat()}."
            raise RuntimeError(msg)
        return segments, curated
