"""Tests for Pydantic validations, NDCG@k metric, and podcast_graph."""

import json
from datetime import date
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from evals.conftest import NDCGAtKMetric
from podcast.agents import PodcastContext, PodcastState, llm_config, podcast_graph
from podcast.feed import Feed
from podcast.fetcher import ContentFetcher
from podcast.models import Script, Story, SynthesizedAudio
from podcast.settings import Settings
from podcast.sources import Source


class InMemorySource:
    """In-memory Source returning controlled test Stories and comments."""

    def __init__(
        self,
        stories: list[Story],
        comments: dict[str, list[str]],
    ) -> None:
        """Store static test Stories and comment lists."""
        self._stories = stories
        self._comments = comments

    async def fetch_stories(self, day: date, limit: int = 30) -> list[Story]:
        """Return up to `limit` configured test Stories for `day`."""
        del day
        return self._stories[:limit]

    async def fetch_comments(self, story_id: str, limit: int = 15) -> list[str]:
        """Return up to `limit` configured comments for `story_id`."""
        return self._comments.get(story_id, [])[:limit]


class InMemoryFetcher:
    """In-memory ContentFetcher returning controlled readable or unreadable bodies."""

    def __init__(self, contents: dict[str, str | None]) -> None:
        """Store static URL-to-content mappings."""
        self._contents = contents

    async def fetch_content(
        self,
        url: str,
        *,
        title: str | None = None,
    ) -> str | None:
        """Return preconfigured Markdown body or None for `url`."""
        del title
        return self._contents.get(url)


@pytest.fixture
def sample_source() -> Source:
    """Provide an in-memory Source with two readable and one bad Story."""
    stories = [
        Story(
            id="9002",
            source="hackernews",
            title="Zero-Copy Ring Buffers in Rust Async Kernels",
            url="https://example.com/rust-ringbuf",
            discussion_url="https://news.ycombinator.com/item?id=9002",
            score=640,
            comment_count=145,
        ),
        Story(
            id="9003",
            source="hackernews",
            title="Unreadable Paywalled Benchmark Story",
            url="https://example.com/paywall",
            discussion_url="https://news.ycombinator.com/item?id=9003",
            score=410,
            comment_count=90,
        ),
        Story(
            id="9004",
            source="hackernews",
            title="SQLite Introduces Built-In Vector Quantization Index",
            url="https://example.com/sqlite-vec",
            discussion_url="https://news.ycombinator.com/item?id=9004",
            score=520,
            comment_count=112,
        ),
    ]
    comments = {
        "9002": [
            "We measured similar tail-latency drops moving away from epoll.",
            "Caveat: registered buffers require careful kernel ulimit tuning.",
        ],
        "9004": [
            "Great for local desktop tools where running pgvector is overkill.",
        ],
    }
    return InMemorySource(stories, comments)


@pytest.fixture
def sample_fetcher() -> ContentFetcher:
    """Provide an in-memory ContentFetcher matching sample_source URLs."""
    contents: dict[str, str | None] = {
        "https://example.com/rust-ringbuf": (
            "A detailed engineering story explaining how io_uring and lock-free "
            "ring buffers in Rust cut tail latency by 42 percent in high-load "
            "network proxies without unsafe heap allocations per packet. " * 3
        ),
        "https://example.com/paywall": None,
        "https://example.com/sqlite-vec": (
            "SQLite engineers published benchmarks showing quantized int8 vector "
            "indexes inside single-file embedded databases reach 12ms p99 search "
            "on one million embeddings with zero external daemon processes. " * 3
        ),
    }
    return InMemoryFetcher(contents)


@pytest.fixture
def graph_settings(tmp_path: Path, live_settings: Settings) -> Settings:
    """Clone live_settings with isolated tmp_path dist dir and distinct test voices."""
    return live_settings.model_copy(
        update={
            "podcast_dist_dir": tmp_path / "dist",
            "persona_a_voice_id": "voice-a",
            "persona_b_voice_id": "voice-b",
        },
    )


@pytest.mark.parametrize(
    ("predicted", "expected_score"),
    [
        (["a", "b", "c", "d", "e"], 1.0),
        (["b", "c", "d", "e", "x"], 0.4677),
        (["x", "y", "z"], 0.0),
    ],
)
def test_ndcg_at_k_scores_exponential_ranking(
    predicted: list[str],
    expected_score: float,
) -> None:
    """NDCGAtKMetric computes exponential NDCG@5 and penalizes missing #1 lead."""
    metric = NDCGAtKMetric(k=5, threshold=0.5)
    case = LLMTestCase(
        input="2026-11-19",
        actual_output=json.dumps(predicted),
        expected_output=json.dumps(["a", "b", "c", "d", "e"]),
    )
    score = metric.measure(case)
    assert score == pytest.approx(expected_score, abs=1e-3)


@pytest.mark.live
async def test_podcast_graph_curates_writes_and_publishes(
    graph_settings: Settings,
    sample_source: Source,
    sample_fetcher: ContentFetcher,
) -> None:
    """LangGraph workflow curates readable Stories, writes turns, and publishes."""
    result = await podcast_graph.ainvoke(
        PodcastState(episode_date=date(2026, 11, 15)),
        context=PodcastContext(
            personas=graph_settings.personas,
            openrouter_api_key=graph_settings.openrouter_api_key,
            curator_model=graph_settings.curator_model,
            language=graph_settings.podcast_language,
            source=sample_source,
            fetcher=sample_fetcher,
        ),
        config=llm_config(
            model=graph_settings.llm_model,
            api_key=graph_settings.openrouter_api_key,
        ),
    )
    script: Script = result["script"]

    assert script.published_at == date(2026, 11, 15)
    assert script.language == "pt-BR"
    assert script.segments
    segment_ids = {segment.story_id for segment in script.segments}
    assert "9003" not in segment_ids
    assert segment_ids & {"9002", "9004"}

    valid_voices = {persona.voice_id for persona in graph_settings.personas}
    assert len(script.turns) >= 4
    assert {turn.voice for turn in script.turns} == valid_voices

    episode = Feed.from_settings(graph_settings).publish(
        script,
        SynthesizedAudio(mp3_bytes=b"ID3testbytes", duration_seconds=360),
        cover_jpeg=b"\xff\xd8\xff\xe0live-cover",
    )
    assert episode.published_at == date(2026, 11, 15)
    assert (graph_settings.podcast_dist_dir / "feed.xml").exists()
