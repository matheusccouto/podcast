"""OpenRouter Fish Audio S2.1 Pro multi-speaker TTS synthesis and MP3 encoding."""

import asyncio
import re
import shutil
from contextlib import nullcontext
from typing import TYPE_CHECKING

import httpx

from podcast.models import SynthesizedAudio

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr

    from podcast.models import SpeechTurn

_RATE_RE = re.compile(r"rate=(\d+)")
_CHANNELS_RE = re.compile(r"channels=(\d+)")


class OpenRouterTTS:
    """Client that synthesizes Fish Audio SpeechTurns and encodes MP3 via ffmpeg."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = "fish-audio/s2.1-pro-free:free",
        base_url: str = "https://openrouter.ai/api/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Store OpenRouter credentials, TTS model identifier, and optional client."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client

    async def synthesize(
        self,
        turns: Sequence[SpeechTurn],
        *,
        title: str,
    ) -> SynthesizedAudio:
        """Synthesize SpeechTurns in one OpenRouter call and encode to MP3."""
        if not turns:
            msg = "Cannot synthesize an empty SpeechTurn list."
            raise ValueError(msg)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            msg = "ffmpeg binary not found in PATH"
            raise FileNotFoundError(msg)

        payload: dict[str, object] = {
            "model": self.model,
            "input": [turn.model_dump() for turn in turns],
            "response_format": "pcm",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        ctx = (
            nullcontext(self.client)
            if self.client is not None
            else httpx.AsyncClient(timeout=180.0)
        )
        async with ctx as active_client:
            response = await active_client.post(
                f"{self.base_url}/audio/speech",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            pcm_bytes = response.content
            content_type = response.headers.get("content-type", "")

        rate_match = _RATE_RE.search(content_type)
        channels_match = _CHANNELS_RE.search(content_type)
        if rate_match is None or channels_match is None:
            msg = (
                "Missing PCM rate or channels in TTS Content-Type header: "
                f"{content_type!r}"
            )
            raise ValueError(msg)

        sample_rate = int(rate_match.group(1))
        channels = int(channels_match.group(1))
        bytes_per_second = sample_rate * channels * 2
        duration_seconds = round(len(pcm_bytes) / bytes_per_second)
        mp3_bytes = await self._encode_pcm_to_mp3(
            pcm_bytes,
            ffmpeg_bin=ffmpeg_bin,
            sample_rate=sample_rate,
            channels=channels,
            title=title,
        )
        return SynthesizedAudio(mp3_bytes=mp3_bytes, duration_seconds=duration_seconds)

    async def _encode_pcm_to_mp3(
        self,
        pcm_bytes: bytes,
        *,
        ffmpeg_bin: str,
        sample_rate: int,
        channels: int,
        title: str,
    ) -> bytes:
        """Normalize raw s16le PCM audio with ffmpeg loudnorm and encode ID3 MP3."""
        process = await asyncio.create_subprocess_exec(
            ffmpeg_bin,
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-af",
            "loudnorm",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-write_id3v2",
            "1",
            "-id3v2_version",
            "3",
            "-metadata",
            f"title={title}",
            "-f",
            "mp3",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(input=pcm_bytes)
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            msg = f"ffmpeg failed with exit code {process.returncode}: {detail}"
            raise RuntimeError(msg)
        return stdout
