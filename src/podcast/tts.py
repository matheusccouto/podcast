"""Multi-speaker TTS synthesis via OpenRouter and MP3 encoding."""

import asyncio
import json
import logging
import math
import re
import shutil
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import httpx
from pydantic import SecretStr

from podcast.models import SpeechTurn, SynthesizedAudio

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 180.0
_BATCH_PAUSE_SECONDS = 0.15
_DEFAULT_MAX_BATCH_CHARS = 2_400
LOUDNESS_TARGET_LUFS = -16.0
LOUDNESS_TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE_LU = 11.0
_LOUDNORM_BASE_FILTER = (
    f"loudnorm=I={LOUDNESS_TARGET_LUFS}:"
    f"TP={LOUDNESS_TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE_LU}"
)
_LOUDNORM_MEASURE_FILTER = f"{_LOUDNORM_BASE_FILTER}:print_format=json"
_LOUDNORM_JSON_RE = re.compile(r'\{[^{}]*"input_i"[^{}]*\}')
_RATE_RE = re.compile(r"rate=(\d+)")
_CHANNELS_RE = re.compile(r"channels=(\d+)")


async def _run_ffmpeg(
    ffmpeg_bin: str,
    arguments: Sequence[str],
    *,
    stdin_bytes: bytes,
) -> tuple[bytes, str]:
    """Run ffmpeg once, feeding stdin_bytes and returning stdout and stderr."""
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        *arguments,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate(stdin_bytes)
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if process.returncode != 0:
        msg = f"ffmpeg failed with exit code {process.returncode}: {stderr[-500:]}"
        raise RuntimeError(msg)
    return stdout_bytes, stderr


@dataclass(frozen=True)
class _LoudnessMeasurement:
    """First-pass loudnorm statistics that drive the linear second pass."""

    input_i: float
    input_lra: float
    input_tp: float
    input_thresh: float
    target_offset: float

    @classmethod
    def from_ffmpeg_stderr(cls, stderr: str) -> _LoudnessMeasurement:
        """Parse the loudnorm JSON block that ffmpeg prints on stderr."""
        match = _LOUDNORM_JSON_RE.search(stderr)
        if match is None:
            msg = (
                f"loudnorm measurement JSON missing from ffmpeg stderr: {stderr[-500:]}"
            )
            raise ValueError(msg)
        payload = json.loads(match.group(0))
        values = {
            "input_i": float(payload["input_i"]),
            "input_lra": float(payload["input_lra"]),
            "input_tp": float(payload["input_tp"]),
            "input_thresh": float(payload["input_thresh"]),
            "target_offset": float(payload["target_offset"]),
        }
        non_finite = [
            name for name, value in values.items() if not math.isfinite(value)
        ]
        if non_finite:
            msg = (
                "loudnorm measured non-finite values "
                f"({', '.join(non_finite)}); the PCM is likely digital silence"
            )
            raise ValueError(msg)
        return cls(**values)

    def linear_filter(self) -> str:
        """Return the loudnorm filter string for the linear second pass."""
        return (
            f"{_LOUDNORM_BASE_FILTER}"
            f":measured_I={self.input_i}"
            f":measured_LRA={self.input_lra}"
            f":measured_TP={self.input_tp}"
            f":measured_thresh={self.input_thresh}"
            f":offset={self.target_offset}"
            ":linear=true"
        )


class OpenRouterTTS:
    """Synthesizes multi-speaker SpeechTurns via OpenRouter and encodes MP3."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = "google/gemini-3.8-flash-tts",
        base_url: str = "https://openrouter.ai/api/v1",
        max_batch_chars: int = _DEFAULT_MAX_BATCH_CHARS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Store OpenRouter credentials, TTS model identifier, and optional client."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_batch_chars = max_batch_chars
        self.client = client

    def _batch_turns(self, turns: Sequence[SpeechTurn]) -> list[list[SpeechTurn]]:
        """Group consecutive turns into multi-turn dialogue batches under char limit."""
        batches: list[list[SpeechTurn]] = []
        current: list[SpeechTurn] = []
        current_chars = 0

        for turn in turns:
            turn_chars = len(turn.text) + len(turn.instructions or "")
            if current and current_chars + turn_chars > self.max_batch_chars:
                batches.append(current)
                current = [turn]
                current_chars = turn_chars
            else:
                current.append(turn)
                current_chars += turn_chars

        if current:
            batches.append(current)
        return batches

    async def synthesize(
        self,
        turns: Sequence[SpeechTurn],
        *,
        title: str,
    ) -> SynthesizedAudio:
        """Synthesize multi-turn dialogue batches, join PCM, and encode MP3."""
        if not turns:
            msg = "Cannot synthesize an empty SpeechTurn list."
            raise ValueError(msg)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            msg = "ffmpeg binary not found in PATH"
            raise FileNotFoundError(msg)

        batches = self._batch_turns(turns)
        pcm_chunks: list[bytes] = []
        audio_format: tuple[int, int] | None = None
        ctx = (
            nullcontext(self.client) if self.client is not None else httpx.AsyncClient()
        )
        async with ctx as active_client:
            for batch in batches:
                chunk, sample_rate, channels = await self._request_batch_pcm(
                    active_client,
                    batch,
                )
                if audio_format is None:
                    audio_format = (sample_rate, channels)
                elif audio_format != (sample_rate, channels):
                    msg = (
                        f"Inconsistent PCM format across batches: expected "
                        f"{audio_format}, got {(sample_rate, channels)}"
                    )
                    raise ValueError(msg)
                pcm_chunks.append(chunk)

        assert audio_format is not None  # noqa: S101
        sample_rate, channels = audio_format
        pause = b"\x00" * (int(sample_rate * _BATCH_PAUSE_SECONDS) * channels * 2)
        pcm_bytes = pause.join(pcm_chunks)
        duration_seconds = round(len(pcm_bytes) / (sample_rate * channels * 2))
        mp3_bytes = await self._encode_mp3(
            ffmpeg_bin=ffmpeg_bin,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            title=title,
        )
        logger.info(
            "Synthesized %d turns in %d batch(es) into %ds MP3 (%d bytes) via %s",
            len(turns),
            len(batches),
            duration_seconds,
            len(mp3_bytes),
            self.model,
        )
        return SynthesizedAudio(mp3_bytes=mp3_bytes, duration_seconds=duration_seconds)

    async def _request_batch_pcm(
        self,
        client: httpx.AsyncClient,
        batch: Sequence[SpeechTurn],
    ) -> tuple[bytes, int, int]:
        """Request raw PCM bytes for a multi-turn dialogue batch."""
        payload: dict[str, object] = {
            "model": self.model,
            "input": [turn.model_dump(exclude_none=True) for turn in batch],
            "response_format": "pcm",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        response = await client.post(
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
        """Encode s16le PCM into -16 LUFS normalized ID3v2.3 MP3 via ffmpeg."""
        measurement = await OpenRouterTTS._measure_loudness(
            ffmpeg_bin=ffmpeg_bin,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
        arguments = [
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-af",
            measurement.linear_filter(),
            "-write_id3v2",
            "1",
            "-id3v2_version",
            "3",
            "-metadata",
            f"title={title}",
            "-f",
            "mp3",
            "pipe:1",
        ]
        mp3_bytes, _ = await _run_ffmpeg(
            ffmpeg_bin,
            arguments,
            stdin_bytes=pcm_bytes,
        )
        return mp3_bytes

    @staticmethod
    async def _measure_loudness(
        *,
        ffmpeg_bin: str,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
    ) -> _LoudnessMeasurement:
        """Run the loudnorm analysis pass and return its measured statistics."""
        arguments = [
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-af",
            _LOUDNORM_MEASURE_FILTER,
            "-f",
            "null",
            "-",
        ]
        _, stderr = await _run_ffmpeg(
            ffmpeg_bin,
            arguments,
            stdin_bytes=pcm_bytes,
        )
        return _LoudnessMeasurement.from_ffmpeg_stderr(stderr)
