"""Tests for OpenRouter per-Episode cover generation."""

import base64
import json
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from podcast.cover import OpenRouterCover, headline, render_prompt
from podcast.feed import Feed
from podcast.models import Script, Segment, SpeechTurn, SynthesizedAudio

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
)
_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def jpeg_size(jpeg_bytes: bytes) -> tuple[int, int]:
    """Return `(width, height)` parsed from a baseline/extended JPEG SOF marker."""
    index = 2
    while index < len(jpeg_bytes):
        if jpeg_bytes[index] != 0xFF:
            msg = "Malformed JPEG segment"
            raise ValueError(msg)
        marker = jpeg_bytes[index + 1]
        if marker in _SOF_MARKERS:
            height = int.from_bytes(jpeg_bytes[index + 5 : index + 7], "big")
            width = int.from_bytes(jpeg_bytes[index + 7 : index + 9], "big")
            return width, height
        length = int.from_bytes(jpeg_bytes[index + 2 : index + 4], "big")
        index += 2 + length
    msg = "JPEG holds no SOF marker"
    raise ValueError(msg)


@pytest.fixture
def master_cover(tmp_path: Path) -> Path:
    """Write reusable bytes standing in for the staged master cover asset."""
    path = tmp_path / "static" / "cover.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xd8\xff\xe0master-cover-bytes")
    return path


@pytest.fixture
def master_reference(master_cover: Path) -> str:
    """Return the JPEG data URL the generator sends as its only input reference."""
    encoded = base64.b64encode(master_cover.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@pytest.fixture
def make_script(sample_turns: list[SpeechTurn]) -> Callable[[date, str], Script]:
    """Return a factory building Scripts whose lead Segment drives the cover prompt."""

    def _build(published_at: date, title: str) -> Script:
        """Build a minimal valid Script with one lead Segment and two turns."""
        return Script(
            published_at=published_at,
            title=title,
            summary="Resumo tecnico do dia.",
            language="pt-BR",
            segments=[
                Segment(
                    story_id="123",
                    title="Kernel Linux muda escalonador",
                    url="https://example.com/story",
                    key_facts=["Escalonador EEVDF vira padrao no kernel 6.20."],
                ),
            ],
            turns=sample_turns,
        )

    return _build


def test_cover_prompt_uses_short_headline_and_lead_story(
    make_script: Callable[[date, str], Script],
) -> None:
    """The prompt renders a 2-4 word headline, the lead Story, and its facts."""
    script = make_script(date(2026, 11, 24), "Edicao de hoje")

    assert headline(script) == "KERNEL LINUX MUDA ESCALONADOR"

    prompt = render_prompt(script)
    assert '"KERNEL LINUX MUDA ESCALONADOR"' in prompt
    assert "Kernel Linux muda escalonador" in prompt
    assert "EEVDF" in prompt
    assert "microphone" not in prompt.lower()


async def test_generate_posts_single_image_request_and_returns_1400_jpeg(
    master_cover: Path,
    master_reference: str,
    make_script: Callable[[date, str], Script],
) -> None:
    """One OpenRouter call carries the master reference and yields a 1400px JPEG."""
    requests: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        """Capture the generation request and return an inline base64 PNG."""
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "created": 1_760_000_000,
                "data": [
                    {"b64_json": base64.b64encode(_ONE_PIXEL_PNG).decode("ascii")}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        cover = OpenRouterCover(
            api_key=SecretStr("test-key"),
            master_cover_path=master_cover,
            client=client,
        )
        jpeg = await cover.generate(make_script(date(2026, 11, 24), "Edicao de hoje"))

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/v1/images"
    assert request.headers["authorization"] == "Bearer test-key"

    payload = json.loads(request.content)
    assert payload["model"] == "google/gemini-nano-banana-2.1"
    assert payload["aspect_ratio"] == "1:1"
    assert payload["resolution"] == "2K"
    assert payload["response_format"] == "b64_json"
    assert payload["input_references"] == [
        {"type": "image_url", "image_url": {"url": master_reference}},
    ]
    assert '"KERNEL LINUX MUDA ESCALONADOR"' in payload["prompt"]

    assert jpeg is not None
    assert jpeg[:3] == b"\xff\xd8\xff"
    assert jpeg_size(jpeg) == (1400, 1400)


async def test_generate_logs_and_returns_none_when_image_model_is_unavailable(
    master_cover: Path,
    make_script: Callable[[date, str], Script],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed OpenRouter call publishes the Episode without a cover, loudly logged."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(503, json={"error": "model overloaded"}),
    )
    caplog.set_level(logging.WARNING, logger="podcast.cover")

    async with httpx.AsyncClient(transport=transport) as client:
        cover = OpenRouterCover(
            api_key=SecretStr("test-key"),
            master_cover_path=master_cover,
            client=client,
        )
        jpeg = await cover.generate(make_script(date(2026, 11, 24), "Edicao de hoje"))

    assert jpeg is None
    assert "publishing Episode without cover" in caplog.text


async def test_generate_raises_when_response_holds_no_image(
    master_cover: Path,
    make_script: Callable[[date, str], Script],
) -> None:
    """A 200 response without image data breaks the contract loudly."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"created": 1, "data": []}),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        cover = OpenRouterCover(
            api_key=SecretStr("test-key"),
            master_cover_path=master_cover,
            client=client,
        )
        with pytest.raises(ValueError, match="data"):
            await cover.generate(make_script(date(2026, 11, 24), "Edicao de hoje"))


def test_episode_without_cover_renders_no_item_itunes_image(
    tmp_path: Path,
    make_script: Callable[[date, str], Script],
) -> None:
    """Feed.publish omits `<itunes:image>` for Episodes published without artwork."""
    feed = Feed(
        title="Daily Diff",
        description="Descricao",
        language="pt-BR",
        author="Matheus Couto",
        base_url="https://example.github.io/podcast",
        releases_base_url="https://example.com/episodes",
        dist_dir=tmp_path / "dist",
    )
    episode = feed.publish(
        make_script(date(2026, 11, 24), "Edicao de hoje"),
        SynthesizedAudio(mp3_bytes=b"ID3audio", duration_seconds=300),
    )

    assert episode.cover_url is None
    assert not (tmp_path / "dist" / "episodes" / "2026-11-24.jpg").exists()
    rss_xml = (tmp_path / "dist" / "feed.xml").read_text(encoding="utf-8")
    assert "itunes:image" not in rss_xml
