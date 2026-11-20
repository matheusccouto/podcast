"""OpenRouter Fish Audio S2.1 Pro multi-speaker TTS synthesis and MP3 encoding."""

import logging
import re
import shutil
from collections.abc import Sequence
from contextlib import nullcontext

import httpx
from ffmpeg.asyncio import FFmpeg
from pydantic import SecretStr

from podcast.models import SpeechTurn, SynthesizedAudio

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 180.0
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

        pcm_bytes, sample_rate, channels = await self._request_pcm(turns)
        duration_seconds = round(len(pcm_bytes) / (sample_rate * channels * 2))
        mp3_bytes = await self._encode_mp3(
            ffmpeg_bin=ffmpeg_bin,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            title=title,
        )
        logger.info(
            "Synthesized %ds MP3 (%d bytes) via %s",
            duration_seconds,
            len(mp3_bytes),
            self.model,
        )
        return SynthesizedAudio(mp3_bytes=mp3_bytes, duration_seconds=duration_seconds)

    async def _request_pcm(
        self,
        turns: Sequence[SpeechTurn],
    ) -> tuple[bytes, int, int]:
        """Request raw PCM bytes and parse sample rate/channels from Content-Type."""
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
            nullcontext(self.client) if self.client is not None else httpx.AsyncClient()
        )
        async with ctx as active_client:
            response = await active_client.post(
                f"{self.base_url}/audio/speech",
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
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
        if sample_rate <= 0 or channels <= 0 or not pcm_bytes:
            msg = (
                "Invalid empty PCM audio or audio format parameters: "
                f"rate={sample_rate}, channels={channels}, bytes={len(pcm_bytes)}"
            )
            raise ValueError(msg)
        return pcm_bytes, sample_rate, channels

    @staticmethod
    async def _encode_mp3(
        *,
        ffmpeg_bin: str,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        title: str,
    ) -> bytes:
        """Encode s16le PCM bytes into loudnorm-normalized ID3v2.3 MP3 via ffmpeg."""
        ffmpeg = (
            FFmpeg(executable=ffmpeg_bin)
            .option("y")
            .input("pipe:0", f="s16le", ar=sample_rate, ac=channels)
            .output(
                "pipe:1",
                {"codec:a": "libmp3lame", "b:a": "128k"},
                af="loudnorm",
                write_id3v2=1,
                id3v2_version=3,
                metadata=f"title={title}",
                f="mp3",
            )
        )
        return await ffmpeg.execute(pcm_bytes)
