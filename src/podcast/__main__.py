"""CLI entrypoint for generating and publishing today's Episode."""

import asyncio
import logging
import sys
from datetime import UTC, date, datetime

import httpx
from pydantic import Field
from pydantic_settings import CliApp

from podcast.agents import PodcastContext, PodcastState, llm_config, podcast_graph
from podcast.feed import Feed
from podcast.fetcher import JinaFetcher
from podcast.models import Episode, Script, validate_persona_voices
from podcast.settings import Settings
from podcast.sources import HackerNewsSource
from podcast.tts import OpenRouterTTS


async def run_daily(
    settings: Settings | None = None,
    *,
    today: date | None = None,
) -> Episode:
    """Execute the daily curation, scriptwriting, TTS, and publishing flow."""
    resolved = settings or Settings()
    validate_persona_voices(resolved.personas)
    target_date = today or datetime.now(UTC).date()

    async with httpx.AsyncClient() as client:
        result = await podcast_graph.ainvoke(
            PodcastState(episode_date=target_date),
            context=PodcastContext(
                personas=resolved.personas,
                language=resolved.podcast_language,
                source=HackerNewsSource(client=client),
                fetcher=JinaFetcher(client=client),
            ),
            config=llm_config(
                model=resolved.llm_model,
                api_key=resolved.openrouter_api_key,
            ),
        )
        script: Script = result["script"]

        tts = OpenRouterTTS(
            api_key=resolved.openrouter_api_key,
            model=resolved.tts_model,
            client=client,
        )
        audio = await tts.synthesize(script.turns, title=script.title)

    return Feed.from_settings(resolved).publish(script, audio)


class PodcastCLI(Settings):
    """Generate and publish a daily tech podcast Episode."""

    episode_date: date | None = Field(
        default=None,
        description="Target Episode date (YYYY-MM-DD, defaults to today UTC).",
    )

    def cli_cmd(self) -> None:
        """Run the daily Episode generation workflow and report the published output."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        episode = asyncio.run(run_daily(settings=self, today=self.episode_date))
        sys.stdout.write(
            f"Published {episode.published_at.isoformat()} "
            f"({episode.duration_seconds}s, {episode.byte_length} bytes): "
            f"{episode.title}\n",
        )


def main() -> None:
    """Run the CLI entrypoint."""
    CliApp.run(PodcastCLI)


if __name__ == "__main__":
    main()
