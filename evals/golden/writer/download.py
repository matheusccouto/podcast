"""Download cherry-picked multi-source NPR Indicator golden Fish Audio cases."""

import asyncio
import re
import sys
from collections import Counter
from datetime import date
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from podcast.fetcher import ExaFetcher
from podcast.models import SpeechTurn
from podcast.settings import Settings

_NPR_FEED_URL = "https://feeds.npr.org/510325/podcast.xml"
_GOLDEN_DIR = Path(__file__).parent
_MAX_DURATION_SECONDS = 600
_MIN_HOST_TURNS = 3
_MIN_SOURCES = 2
_MIN_SOURCE_CHARS = 400
_TARGET_FIXTURES = 3
_HTTP_NOT_FOUND = 404
_SKIP_DOMAINS = (
    "npr.org",
    "lnk.to",
    "linktr.ee",
    "shopnpr.org",
    "planetmoneybook.com",
    "dropelectric.bandcamp.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "bsky.app",
    "threads.net",
)

_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL)
_STORYTEXT_RE = re.compile(
    r'<div id="storytext"[^>]*>(.*?)</div>\s*<!--',
    re.DOTALL,
)
_HREF_RE = re.compile(r'href="(https?://[^"]+)"')
_TAG_RE = re.compile(r"<[^>]+>")
_PAREN_NOISE_RE = re.compile(r"\([^)]*\)")
_SPEAKER_LINE_RE = re.compile(r"^([A-Z][A-Z\s.'-]{1,30}):\s+(.+)$")


class GoldenStory(BaseModel):
    """Curated input Story with extracted readable content inline."""

    title: str
    url: str | None = None
    content: str


class GoldenCase(BaseModel):
    """Golden input/output contract matching Stage 2 scriptwriting -> Fish Audio TTS."""

    date: date
    stories: list[GoldenStory]
    expected_output: list[SpeechTurn]


def _extract_tag(block: str, tag: str) -> str | None:
    """Extract and unescape plain text from an XML RSS tag block."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    if not match:
        return None
    return unescape(_TAG_RE.sub("", match.group(1))).strip()


def _parse_duration_seconds(raw: str | None) -> int | None:
    """Parse iTunes HH:MM:SS or integer second strings into total seconds."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.isdigit():
        return int(cleaned)
    parts = cleaned.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def _normalize_lines(html: str) -> list[tuple[str, str]]:
    """Extract normalized (speaker, text) pairs from an NPR transcript HTML page."""
    marker = html.find('class="transcript storytext"')
    chunk = html[marker:] if marker != -1 else html
    raw_lines: list[tuple[str, str]] = []

    for paragraph in re.split(r"<p[^>]*>", chunk):
        text = unescape(_TAG_RE.sub("", paragraph)).strip()
        if not text or text.startswith("Copyright"):
            continue
        match = _SPEAKER_LINE_RE.match(text)
        if not match:
            continue
        speaker = match.group(1).strip()
        if speaker in {"ANNOUNCER", "SPEAKER"}:
            continue
        body = _PAREN_NOISE_RE.sub("", match.group(2)).strip()
        if body:
            raw_lines.append((speaker, body))

    surname_map = {s.split()[-1]: s for s, _ in raw_lines if " " in s}
    return [(surname_map.get(speaker, speaker), body) for speaker, body in raw_lines]


def _extract_fish_turns(html: str) -> list[SpeechTurn]:
    """Normalize NPR transcript lines into validated two-voice SpeechTurn objects."""
    normalized = _normalize_lines(html)
    counts = Counter(speaker for speaker, _ in normalized)
    primary = [s for s, c in counts.most_common(2) if c >= _MIN_HOST_TURNS]
    if len(primary) < 2:
        return []

    voice_for = {primary[0]: "persona_a", primary[1]: "persona_b"}
    turns: list[SpeechTurn] = []
    for speaker, body in normalized:
        voice = voice_for.get(speaker, "persona_a")
        spoken_text = body if speaker in voice_for else f"{speaker.title()}: {body}"

        if turns and turns[-1].voice == voice:
            turns[-1] = SpeechTurn(voice=voice, text=f"{turns[-1].text} {spoken_text}")
            continue
        try:
            turns.append(SpeechTurn(voice=voice, text=spoken_text))
        except ValidationError:
            continue

    return turns if len({t.voice for t in turns}) == 2 else []


def _infer_story_title(content: str, url: str, index: int) -> str:
    """Derive a concise Story title from Markdown headings or URL host."""
    for raw_line in content.splitlines()[:12]:
        line = raw_line.strip()
        if line.startswith("#"):
            cleaned = line.lstrip("# ").strip()
            if len(cleaned) >= 8:
                return cleaned[:120]
    host = urlparse(url).hostname or f"source-{index}"
    return f"Source {index} ({host.removeprefix('www.')})"


async def _fetch_external_urls(
    client: httpx.AsyncClient,
    link: str,
) -> list[str]:
    """Extract deduplicated non-promo external citation URLs from an NPR page."""
    ep_resp = await client.get(link)
    if ep_resp.status_code == _HTTP_NOT_FOUND:
        return []
    ep_resp.raise_for_status()
    story_match = _STORYTEXT_RE.search(ep_resp.text)
    if not story_match:
        return []
    return list(
        dict.fromkeys(
            url
            for url in _HREF_RE.findall(story_match.group(1))
            if (host := (urlparse(url).hostname or "").lower())
            and not any(host == d or host.endswith(f".{d}") for d in _SKIP_DOMAINS)
        ),
    )


async def _build_case(
    client: httpx.AsyncClient,
    item_xml: str,
) -> tuple[str, GoldenCase] | None:
    """Build a multi-source GoldenCase from an eligible NPR RSS item."""
    pub_raw = _extract_tag(item_xml, "pubDate")
    dur_raw = _extract_tag(item_xml, "itunes:duration")
    link = _extract_tag(item_xml, "link")
    if not (pub_raw and dur_raw and link):
        return None

    published = parsedate_to_datetime(pub_raw)
    duration = _parse_duration_seconds(dur_raw)
    path_parts = [part for part in urlparse(link).path.split("/") if part]
    if (
        published.weekday() > 3
        or duration is None
        or duration > _MAX_DURATION_SECONDS
        or len(path_parts) < 2
    ):
        return None

    episode_id = path_parts[-2]
    external_urls = await _fetch_external_urls(client, link)
    tr_resp = await client.get(f"https://www.npr.org/transcripts/{episode_id}")
    if len(external_urls) < _MIN_SOURCES or tr_resp.status_code == _HTTP_NOT_FOUND:
        return None
    tr_resp.raise_for_status()

    turns = _extract_fish_turns(tr_resp.text)
    if not turns:
        return None

    urls = external_urls[:4]
    fetcher = ExaFetcher(api_key=Settings().exa_api_key, client=client)
    fetched = await asyncio.gather(
        *(fetcher.fetch_content(url) for url in urls),
        return_exceptions=True,
    )
    stories = [
        GoldenStory(
            title=_infer_story_title(content, url, idx),
            url=url,
            content=content,
        )
        for idx, (url, content) in enumerate(zip(urls, fetched, strict=True), start=1)
        if isinstance(content, str) and len(content.strip()) >= _MIN_SOURCE_CHARS
    ]
    if len(stories) < _MIN_SOURCES:
        return None

    return episode_id, GoldenCase(
        date=published.date(),
        stories=stories,
        expected_output=turns,
    )


async def download_golden(*, limit: int = _TARGET_FIXTURES) -> list[Path]:
    """Download multi-source golden cases from NPR The Indicator into evals/golden."""
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for stale in _GOLDEN_DIR.glob("*.json"):
        stale.unlink()

    saved: list[Path] = []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        rss = await client.get(_NPR_FEED_URL)
        rss.raise_for_status()

        for item_xml in _ITEM_RE.findall(rss.text)[:150]:
            result = await _build_case(client, item_xml)
            if result is None:
                continue
            episode_id, case = result
            target = _GOLDEN_DIR / f"{episode_id}.json"
            target.write_text(case.model_dump_json(indent=2), encoding="utf-8")
            saved.append(target)
            sys.stdout.write(
                f"Saved multi-source golden ({len(case.stories)} sources, "
                f"{len(case.expected_output)} turns): {target.name}\n",
            )
            if len(saved) >= limit:
                break

    return saved


def main() -> None:
    """CLI entrypoint to populate evals/golden/ with multi-source GoldenCases."""
    paths = asyncio.run(download_golden())
    if not paths:
        msg = "Failed to download any valid multi-source NPR golden cases."
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
