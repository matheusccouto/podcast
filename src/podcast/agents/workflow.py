"""Module-level LangGraph workflow connecting Curator and Writer subagents."""

import logging
from dataclasses import dataclass
from datetime import date

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from podcast.agents.curator import CuratorContext, curator
from podcast.agents.writer import WriterContext, format_writer_prompt, writer
from podcast.fetcher import ContentFetcher
from podcast.models import (
    CuratedStory,
    Dialogue,
    Persona,
    ResearchBrief,
    Script,
    Segment,
)
from podcast.sources import Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PodcastContext:
    """Per-invocation runtime context passed into `podcast_graph.ainvoke`."""

    personas: tuple[Persona, Persona]
    source: Source
    fetcher: ContentFetcher
    language: str = "pt-BR"


class PodcastState(BaseModel):
    """Shared LangGraph workflow state flowing from curation to scriptwriting."""

    episode_date: date
    title: str = ""
    summary: str = ""
    segments: list[Segment] = Field(default_factory=list)
    curated_stories: list[CuratedStory] = Field(default_factory=list)
    script: Script | None = None


async def _curate_node(
    state: PodcastState,
    runtime: Runtime[PodcastContext],
) -> dict[str, object]:
    """Invoke the Curator subagent with runtime Source/Fetcher context."""
    ctx = runtime.context
    result = await curator.ainvoke(
        {"messages": [{"role": "user", "content": "Curate today's tech Stories."}]},
        context=CuratorContext(
            episode_date=state.episode_date,
            source=ctx.source,
            fetcher=ctx.fetcher,
        ),
        config={"recursion_limit": 30},
    )
    brief: ResearchBrief = result["structured_response"]
    curated_stories: list[CuratedStory] = result["curated_stories"]
    logger.info(
        "Curated %d stories for %s: %s",
        len(curated_stories),
        state.episode_date.isoformat(),
        brief.title,
    )
    return {
        "title": brief.title,
        "summary": brief.summary,
        "segments": result["segments"],
        "curated_stories": curated_stories,
    }


async def _write_node(
    state: PodcastState,
    runtime: Runtime[PodcastContext],
) -> dict[str, object]:
    """Invoke the Writer subagent and assemble the finalized Script."""
    ctx = runtime.context
    prompt = format_writer_prompt(state.episode_date, state.curated_stories)
    result = await writer.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=WriterContext(personas=ctx.personas, language=ctx.language),
    )
    dialogue: Dialogue = result["structured_response"]
    logger.info(
        "Wrote Script (%d turns, %d spoken words)",
        len(dialogue.turns),
        dialogue.spoken_word_count,
    )
    return {
        "script": Script(
            published_at=state.episode_date,
            title=state.title,
            summary=state.summary,
            language=ctx.language,
            segments=state.segments,
            turns=dialogue.turns,
        ),
    }


podcast_graph = (
    StateGraph(PodcastState, context_schema=PodcastContext)
    .add_node("curate", _curate_node)
    .add_node("write_script", _write_node)
    .add_edge(START, "curate")
    .add_edge("curate", "write_script")
    .add_edge("write_script", END)
    .compile()
)
