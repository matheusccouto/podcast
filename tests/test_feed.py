"""Tests for RSS 2.0 Feed publishing."""

from datetime import date
from typing import TYPE_CHECKING

import pytest

from podcast.feed import Feed
from podcast.models import Script, Segment, SynthesizedAudio

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from podcast.models import SpeechTurn


@pytest.fixture
def feed(tmp_path: Path) -> Feed:
    """Create a Feed publisher writing artifacts into a temporary dist directory."""
    return Feed(
        title="HN Daily BR",
        description="Resumo diário de tecnologia.",
        language="pt-BR",
        author="Matheus Couto",
        base_url="https://example.github.io/podcast",
        releases_base_url="https://github.com/u/r/releases/download/latest",
        dist_dir=tmp_path / "dist",
    )


@pytest.fixture
def make_script(sample_turns: list[SpeechTurn]) -> Callable[[date, str], Script]:
    """Return a factory creating minimal valid two-turn Scripts for test dates."""

    def _build(published_at: date, title: str) -> Script:
        """Build a two-turn Script with one verified lead Segment."""
        return Script(
            published_at=published_at,
            title=title,
            summary="Resumo tecnico do dia.",
            language="pt-BR",
            segments=[
                Segment(
                    story_id="123",
                    title=title,
                    url="https://example.com/story",
                    is_lead=True,
                    key_facts=["Fact one verified in Story."],
                ),
            ],
            turns=sample_turns,
        )

    return _build


def test_feed_publish_writes_artifacts_and_orders_rss_newest_first(
    feed: Feed,
    make_script: Callable[[date, str], Script],
    tmp_path: Path,
) -> None:
    """Feed.publish saves Episode artifacts and regenerates feed.xml newest-first."""
    dist_dir = tmp_path / "dist"
    ep1 = feed.publish(
        make_script(date(2026, 11, 14), "Dia Um"),
        SynthesizedAudio(mp3_bytes=b"ID3day1audio", duration_seconds=310),
    )
    ep2 = feed.publish(
        make_script(date(2026, 11, 15), "Dia Dois"),
        SynthesizedAudio(mp3_bytes=b"ID3day2audio", duration_seconds=420),
    )

    assert (dist_dir / "episodes" / "2026-11-15.mp3").read_bytes() == b"ID3day2audio"
    assert (dist_dir / "episodes" / "2026-11-15.json").exists()
    assert (
        ep1.audio_url
        == "https://github.com/u/r/releases/download/latest/2026-11-14.mp3"
    )
    assert ep2.duration_seconds == 420
    assert ep2.byte_length == len(b"ID3day2audio")

    rss_xml = (dist_dir / "feed.xml").read_text(encoding="utf-8")
    assert "<title>HN Daily BR</title>" in rss_xml
    assert "<itunes:author>Matheus Couto</itunes:author>" in rss_xml
    assert 'type="audio/mpeg"' in rss_xml
    assert rss_xml.index("Dia Dois") < rss_xml.index("Dia Um")
