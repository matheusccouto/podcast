"""RSS 2.0 Feed publishing for Episode audio and metadata."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time
from email.utils import format_datetime
from importlib import resources
from pathlib import Path

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict

from podcast.models import Episode, Script, SynthesizedAudio
from podcast.settings import Settings

logger = logging.getLogger(__name__)

_PUBLISH_TIME_UTC = time(hour=9, tzinfo=UTC)
_RSS_TEMPLATE = Environment(autoescape=True, keep_trailing_newline=True).from_string(
    resources.files(__package__).joinpath("feed.xml.j2").read_text("utf-8"),
)


@dataclass(frozen=True, slots=True)
class _FeedItem:
    """View model pairing an Episode with its formatted RSS guid and pubDate."""

    episode: Episode
    guid: str
    pub_date: str


class Feed(BaseModel):
    """RSS 2.0 Feed metadata and local distribution publisher."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    title: str
    description: str
    language: str
    author: str
    base_url: str
    releases_base_url: str
    dist_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> Feed:
        """Construct a Feed publisher from runtime application Settings."""
        return cls(
            title=settings.podcast_title,
            description=settings.podcast_description,
            language=settings.podcast_language,
            author=settings.podcast_author,
            base_url=settings.podcast_base_url,
            releases_base_url=settings.podcast_releases_base_url,
            dist_dir=settings.podcast_dist_dir,
        )

    def publish(self, script: Script, audio: SynthesizedAudio) -> Episode:
        """Write Episode audio/metadata and regenerate newest-first feed.xml."""
        slug = script.published_at.isoformat()
        episodes_dir = self.dist_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        (episodes_dir / f"{slug}.mp3").write_bytes(audio.mp3_bytes)

        releases_base = self.releases_base_url.rstrip("/")
        episode = Episode(
            published_at=script.published_at,
            title=script.title,
            summary=script.summary,
            audio_url=f"{releases_base}/{slug}.mp3",
            byte_length=len(audio.mp3_bytes),
            duration_seconds=audio.duration_seconds,
        )
        (episodes_dir / f"{slug}.json").write_text(
            episode.model_dump_json(indent=2),
            encoding="utf-8",
        )

        episodes = [
            Episode.model_validate_json(path.read_text(encoding="utf-8"))
            for path in episodes_dir.glob("*.json")
        ]
        episodes.sort(key=lambda entry: entry.published_at, reverse=True)

        feed_path = self.dist_dir / "feed.xml"
        feed_path.write_text(self._render_rss(episodes), encoding="utf-8")
        logger.info("Published %s (%d episodes in %s)", slug, len(episodes), feed_path)
        return episode

    def _render_rss(self, episodes: list[Episode]) -> str:
        """Render an iTunes-compatible RSS 2.0 XML document ordered newest-first."""
        base_url = self.base_url.rstrip("/")
        items = [
            _FeedItem(
                episode=episode,
                guid=f"{base_url}/episodes/{episode.published_at.isoformat()}",
                pub_date=format_datetime(
                    datetime.combine(episode.published_at, _PUBLISH_TIME_UTC),
                    usegmt=True,
                ),
            )
            for episode in episodes
        ]
        return _RSS_TEMPLATE.render(
            feed=self,
            feed_url=f"{base_url}/feed.xml",
            items=items,
        )
