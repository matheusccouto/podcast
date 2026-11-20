"""Editorial workflow connecting Jev Curator and Writer subagent."""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, SecretStr

from podcast.agents.curator import Curator
from podcast.agents.writer import WriterContext, format_writer_prompt, writer
from podcast.fetcher import ContentFetcher
from podcast.models import CuratedStory, Dialogue, Persona, Script, Segment
from podcast.sources import Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PodcastContext:
    """Per-invocation runtime dependencies for the editorial workflow."""

    personas: tuple[Persona, Persona]
    openrouter_api_key: SecretStr
    source: Source
    fetcher: ContentFetcher
    curator_model: str = "jev-1.14"
    language: str = "pt-BR"
    client: httpx.AsyncClient | None = None


class PodcastState(BaseModel):
    """Shared LangGraph workflow state flowing from curation to scriptwriting."""

    episode_date: date
    segments: list[Segment] = Field(default_factory=list)
    curated_stories: list[CuratedStory] = Field(default_factory=list)
    script: Script | None = None


async def _curate_node(
    state: PodcastState,
    runtime: Runtime[PodcastContext],
) -> dict[str, object]:
    """Rank and hydrate the previous day's Stories via Jev Curator."""
    ctx = runtime.context
    news_day = state.episode_date - timedelta(days=1)
    curator = Curator(
        api_key=ctx.openrouter_api_key,
        model=ctx.curator_model,
        source=ctx.source,
        fetcher=ctx.fetcher,
        client=ctx.client,
    )
    segments, curated_stories = await curator.curate(news_day)
    logger.info(
        "Curated %d stories for %s (lead: %s)",
        len(curated_stories),
        news_day.isoformat(),
        segments[0].title,
    )
    return {
        "segments": segments,
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
        "Wrote Script '%s' (%d turns, %d spoken words)",
        dialogue.title,
        len(dialogue.turns),
        dialogue.spoken_word_count,
    )
    return {
        "script": Script(
            published_at=state.episode_date,
            title=dialogue.title,
            summary=dialogue.summary,
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
