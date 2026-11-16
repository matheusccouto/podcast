"""Click CLI entrypoint for generating and publishing today's Episode."""

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

from podcast.agents import PodcastContext, PodcastState, podcast_graph
from podcast.feed import Feed
from podcast.settings import Settings
from podcast.tts import OpenRouterTTS

if TYPE_CHECKING:
    from podcast.models import Episode, Script


async def run_daily(
    settings: Settings | None = None,
    *,
    today: date | None = None,
) -> Episode:
    """Execute the daily curation, scriptwriting, TTS, and publishing flow."""
    resolved = settings or Settings()
    target_date = today or datetime.now(UTC).date()
    result = await podcast_graph.ainvoke(
        PodcastState(episode_date=target_date),
        context=PodcastContext(
            personas=resolved.personas,
            language=resolved.podcast_language,
        ),
        config=resolved.llm_config,
    )
    script: Script = result["script"]

    tts = OpenRouterTTS(
        api_key=resolved.openrouter_api_key,
        model=resolved.tts_model,
    )
    audio = await tts.synthesize(script.turns, title=script.title)
    return Feed.from_settings(resolved).publish(script, audio)


@click.command()
@click.version_option(package_name="podcast")
@click.option(
    "--date",
    "episode_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Target Episode date (YYYY-MM-DD, defaults to today UTC).",
)
@click.option(
    "--llm-model",
    default=None,
    help="Override the OpenRouter LLM model ID.",
)
@click.option(
    "--tts-model",
    default=None,
    help="Override the OpenRouter TTS model ID.",
)
@click.option(
    "--language",
    default=None,
    help="Override target spoken Episode language (e.g. pt-BR).",
)
@click.option(
    "--dist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory for published RSS feed and Episode files.",
)
def main(
    *,
    episode_date: datetime | None,
    llm_model: str | None,
    tts_model: str | None,
    language: str | None,
    dist_dir: Path | None,
) -> None:
    """Generate and publish a daily tech podcast Episode."""
    settings = Settings()
    updates: dict[str, object] = {}
    if llm_model is not None:
        updates["llm_model"] = llm_model
    if tts_model is not None:
        updates["tts_model"] = tts_model
    if language is not None:
        updates["podcast_language"] = language
    if dist_dir is not None:
        updates["podcast_dist_dir"] = dist_dir
    if updates:
        settings = settings.model_copy(update=updates)

    target_date = episode_date.date() if episode_date is not None else None
    episode = asyncio.run(run_daily(settings=settings, today=target_date))
    click.echo(
        f"Published {episode.published_at.isoformat()} "
        f"({episode.duration_seconds}s, {episode.byte_length} bytes): "
        f"{episode.title}",
    )


if __name__ == "__main__":
    main()
