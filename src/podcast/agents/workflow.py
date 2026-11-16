"""Module-level LangGraph workflow connecting Curator and Writer subagents."""

from datetime import date
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import get_runtime
from pydantic import BaseModel, ConfigDict, Field

from podcast.agents.curator import CuratorContext, curator
from podcast.agents.writer import WriterContext, format_writer_prompt, writer
from podcast.fetcher import JinaFetcher
from podcast.models import CuratedStory, Persona, Script, Segment
from podcast.sources import HackerNewsSource

if TYPE_CHECKING:
    from podcast.models import Dialogue, ResearchBrief


class PodcastContext(BaseModel):
    """Per-invocation runtime context passed into `podcast_graph.ainvoke`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    personas: tuple[Persona, Persona]
    language: str = "pt-BR"
    source: HackerNewsSource = Field(default_factory=HackerNewsSource)
    fetcher: JinaFetcher = Field(default_factory=JinaFetcher)


class PodcastState(BaseModel):
    """Shared LangGraph workflow state flowing from curation to scriptwriting."""

    episode_date: date
    title: str = ""
    summary: str = ""
    segments: list[Segment] = Field(default_factory=list)
    curated_stories: list[CuratedStory] = Field(default_factory=list)
    script: Script | None = None


async def _curate_node(state: PodcastState) -> dict[str, object]:
    """Invoke the Curator subagent with runtime Source/Fetcher context."""
    ctx = get_runtime(PodcastContext).context
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
    return {
        "title": brief.title,
        "summary": brief.summary,
        "segments": result["segments"],
        "curated_stories": result["curated_stories"],
    }


async def _write_node(state: PodcastState) -> dict[str, object]:
    """Invoke the Writer subagent and assemble the finalized Script."""
    ctx = get_runtime(PodcastContext).context
    prompt = format_writer_prompt(state.episode_date, state.curated_stories)
    result = await writer.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=WriterContext(personas=ctx.personas, language=ctx.language),
    )
    dialogue: Dialogue = result["structured_response"]
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
