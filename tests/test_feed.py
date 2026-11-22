"""Tests for RSS 2.0 Feed publishing."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from pydantic import SecretStr

from podcast.feed import Feed
from podcast.models import Script, Segment, SpeechTurn, SynthesizedAudio
from podcast.settings import Settings

_NAMESPACES = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


@pytest.fixture
def feed(tmp_path: Path) -> Feed:
    """Create a Feed publisher writing artifacts into a temporary dist directory."""
    return Feed(
        title="Daily Diff em Português",
        description="Resumo diário de tecnologia.",
        language="pt-BR",
        author="Matheus Couto",
        owner_email="owner@example.com",
        category="Technology",
        subcategory="Tech News",
        cover_path="cover.jpg",
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
        make_script(date(2026, 11, 15), "Dia Dois & Kernel"),
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
    assert "<title>Daily Diff em Português</title>" in rss_xml
    assert "<title>Dia Dois &amp; Kernel</title>" in rss_xml
    assert "<itunes:author>Matheus Couto</itunes:author>" in rss_xml
    assert 'type="audio/mpeg"' in rss_xml
    assert rss_xml.index("Dia Dois &amp; Kernel") < rss_xml.index("Dia Um")


def test_feed_from_settings_uses_daily_diff_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed.from_settings maps the rebranded identity and directory metadata."""
    for name in (
        "PODCAST_TITLE",
        "PODCAST_LANGUAGE",
        "PODCAST_CATEGORY",
        "PODCAST_SUBCATEGORY",
        "PODCAST_COVER_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        exa_api_key=SecretStr("test-exa-key"),
        podcast_owner_email="owner@example.com",
        _env_file=None,
    )

    feed = Feed.from_settings(settings)

    assert feed.title == "Daily Diff em Português"
    assert feed.language == "pt-BR"
    assert feed.owner_email == "owner@example.com"
    assert feed.category == "Technology"
    assert feed.subcategory == "Tech News"
    assert feed.cover_url == "http://localhost:8000/cover.jpg"


def test_feed_renders_required_itunes_metadata_and_summary_only(
    feed: Feed,
    make_script: Callable[[date, str], Script],
    tmp_path: Path,
) -> None:
    """Rendered RSS follows the iTunes channel and Episode description contract."""
    feed.publish(
        make_script(date(2026, 11, 15), "Dia Dois & Kernel"),
        SynthesizedAudio(mp3_bytes=b"ID3day2audio", duration_seconds=420),
    )
    rss_xml = (tmp_path / "dist" / "feed.xml").read_text(encoding="utf-8")
    # The XML is generated locally by Feed.publish, not untrusted input.
    root = ET.fromstring(rss_xml)  # noqa: S314
    channel = root.find("channel")
    assert channel is not None

    owner_path = "itunes:owner"
    assert channel.findtext(f"{owner_path}/itunes:name", namespaces=_NAMESPACES) == (
        "Matheus Couto"
    )
    assert (
        channel.findtext(
            f"{owner_path}/itunes:email",
            namespaces=_NAMESPACES,
        )
        == "owner@example.com"
    )

    category = channel.find("itunes:category", namespaces=_NAMESPACES)
    assert category is not None
    assert category.attrib == {"text": "Technology"}
    subcategory = category.find("itunes:category", namespaces=_NAMESPACES)
    assert subcategory is not None
    assert subcategory.attrib == {"text": "Tech News"}

    assert channel.findtext("itunes:type", namespaces=_NAMESPACES) == "episodic"
    image = channel.find("itunes:image", namespaces=_NAMESPACES)
    assert image is not None
    assert image.attrib == {
        "href": "https://example.github.io/podcast/cover.jpg",
    }

    descriptions = [item.findtext("description") for item in channel.findall("item")]
    assert descriptions == ["Resumo tecnico do dia."]
