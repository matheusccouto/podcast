"""Curator subagent defined directly at module level with LangChain runtime context."""

import json
import operator
from dataclasses import dataclass
from datetime import date
from importlib import resources
from typing import Annotated, Any, NotRequired, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from langgraph.types import Command

from podcast.agents.model import chat_model
from podcast.fetcher import ContentFetcher
from podcast.models import (
    MAX_CONTENT_CHARS,
    CuratedStory,
    ResearchBrief,
    Segment,
    Story,
)
from podcast.sources import Source

_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    resources.files(__package__).joinpath("instructions.md").read_text("utf-8"),
    template_format="jinja2",
)


@dataclass(frozen=True, slots=True)
class CuratorContext:
    """Per-invocation runtime dependencies injected into the Curator subagent."""

    episode_date: date
    source: Source
    fetcher: ContentFetcher


class CuratorState(AgentState[ResearchBrief]):
    """LangGraph agent state accumulating Stories, bodies, comments, and output."""

    stories: NotRequired[Annotated[dict[str, Story], operator.or_]]
    contents: NotRequired[Annotated[dict[str, str], operator.or_]]
    comments: NotRequired[Annotated[dict[str, list[str]], operator.or_]]
    unreadable: NotRequired[Annotated[set[str], operator.or_]]
    segments: NotRequired[list[Segment]]
    curated_stories: NotRequired[list[CuratedStory]]


@dynamic_prompt
def _curator_prompt(request: ModelRequest[CuratorContext]) -> str:
    """Render Jinja2 curation instructions using the run's episode_date."""
    return _SYSTEM_TEMPLATE.format(
        episode_date=request.runtime.context.episode_date.isoformat(),
    )


@after_agent(state_schema=CuratorState)
def _finalize_curation(
    state: CuratorState,
    runtime: Runtime[CuratorContext],
) -> dict[str, Any] | None:
    """Validate ResearchBrief grounding and materialize CuratedStories in state."""
    del runtime
    brief = state.get("structured_response")
    if brief is None:
        return None
    segments, curated_stories = brief.build_curated_stories(
        stories=state.get("stories", {}),
        contents=state.get("contents", {}),
        comments=state.get("comments", {}),
        unreadable=state.get("unreadable", set()),
    )
    return {"segments": segments, "curated_stories": curated_stories}


@tool
async def list_stories(
    runtime: ToolRuntime[CuratorContext, CuratorState],
    limit: int = 30,
) -> Command[Any]:
    """List top candidate tech news Stories with id, title, url, score."""
    stories = await runtime.context.source.fetch_stories(limit)
    payload = [
        {
            "id": story.id,
            "title": story.title,
            "url": story.url,
            "score": story.score,
            "comment_count": story.comment_count,
        }
        for story in stories
    ]
    return Command(
        update={
            "stories": {story.id: story for story in stories},
            "messages": [
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
    )


@tool
async def fetch_story_content(
    story_id: str,
    runtime: ToolRuntime[CuratorContext, CuratorState],
) -> Command[Any] | str:
    """Extract readable Markdown content for a candidate story_id."""
    story = runtime.state.get("stories", {}).get(story_id)
    if story is None:
        return f"Unknown story_id '{story_id}'. Call list_stories first."
    if not story.url:
        text = f"No external URL (self-post title: {story.title})."
        return Command(
            update={
                "contents": {story_id: story.title},
                "messages": [
                    ToolMessage(
                        content=text,
                        tool_call_id=runtime.tool_call_id,
                    ),
                ],
            },
        )
    content = await runtime.context.fetcher.fetch_content(story.url)
    if content is None:
        return Command(
            update={
                "unreadable": {story_id},
                "messages": [
                    ToolMessage(
                        content="UNREADABLE",
                        tool_call_id=runtime.tool_call_id,
                    ),
                ],
            },
        )
    clipped = content[:MAX_CONTENT_CHARS]
    return Command(
        update={
            "contents": {story_id: clipped},
            "messages": [
                ToolMessage(
                    content=clipped,
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
    )


@tool
async def fetch_comments(
    story_id: str,
    runtime: ToolRuntime[CuratorContext, CuratorState],
    limit: int = 15,
) -> Command[Any]:
    """Fetch top community comments for a shortlisted story_id."""
    comments = await runtime.context.source.fetch_comments(story_id, limit)
    return Command(
        update={
            "comments": {story_id: comments},
            "messages": [
                ToolMessage(
                    content=json.dumps(comments),
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
    )


curator = create_agent(
    model=chat_model,
    tools=[list_stories, fetch_story_content, fetch_comments],
    middleware=[_curator_prompt, cast("Any", _finalize_curation)],
    state_schema=cast("type[AgentState[ResearchBrief]]", CuratorState),
    context_schema=CuratorContext,
    response_format=ToolStrategy(ResearchBrief),
    name="curator",
)
