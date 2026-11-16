"""RSS 2.0 Feed publishing for Episode audio and metadata."""

from datetime import UTC, datetime, time
from email.utils import format_datetime
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from pydantic import BaseModel

from podcast.models import Episode

if TYPE_CHECKING:
    from podcast.models import Script, SynthesizedAudio
    from podcast.settings import Settings


class Feed(BaseModel):
    """RSS 2.0 Feed metadata and local distribution publisher."""

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

        (self.dist_dir / "feed.xml").write_text(
            self._render_rss(episodes),
            encoding="utf-8",
        )
        return episode

    def _render_rss(self, episodes: list[Episode]) -> str:
        """Render an iTunes-compatible RSS 2.0 XML document ordered newest-first."""
        base_url = self.base_url.rstrip("/")
        feed_url = f"{base_url}/feed.xml"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                '<rss version="2.0" '
                'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" '
                'xmlns:atom="http://www.w3.org/2005/Atom">'
            ),
            "  <channel>",
            f"    <title>{escape(self.title)}</title>",
            f"    <link>{escape(self.base_url)}</link>",
            f"    <description>{escape(self.description)}</description>",
            f"    <language>{escape(self.language)}</language>",
            f"    <itunes:author>{escape(self.author)}</itunes:author>",
            "    <itunes:explicit>false</itunes:explicit>",
            (
                f'    <atom:link href="{escape(feed_url)}" '
                'rel="self" type="application/rss+xml" />'
            ),
        ]
        for episode in episodes:
            guid = f"{base_url}/episodes/{episode.published_at.isoformat()}"
            pub_dt = datetime.combine(episode.published_at, time(hour=9, tzinfo=UTC))
            pub_date = format_datetime(pub_dt, usegmt=True)
            lines.extend(
                [
                    "    <item>",
                    f"      <title>{escape(episode.title)}</title>",
                    f'      <guid isPermaLink="false">{escape(guid)}</guid>',
                    f"      <pubDate>{pub_date}</pubDate>",
                    f"      <description>{escape(episode.summary)}</description>",
                    (
                        f'      <enclosure url="{escape(episode.audio_url)}" '
                        f'length="{episode.byte_length}" type="audio/mpeg" />'
                    ),
                    (
                        "      <itunes:duration>"
                        f"{episode.duration_seconds}"
                        "</itunes:duration>"
                    ),
                    "    </item>",
                ],
            )
        lines.extend(["  </channel>", "</rss>", ""])
        return "\n".join(lines)
