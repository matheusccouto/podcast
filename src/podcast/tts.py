"""Multi-speaker TTS synthesis via OpenRouter and MP3 encoding."""

import asyncio
import json
import logging
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import SecretStr

from podcast.models import SpeechTurn, SynthesizedAudio

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 180.0
_BATCH_PAUSE_SECONDS = 0.15
_DEFAULT_MAX_BATCH_CHARS = 2_400
_STINGER_CROSSFADE_SECONDS = 0.5
_S16LE_SAMPLE_BYTES = 2
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

    def __init__(  # noqa: PLR0913
        self,
        *,
        api_key: SecretStr,
        model: str = "google/gemini-3.8-flash-tts",
        base_url: str = "https://openrouter.ai/api/v1",
        max_batch_chars: int = _DEFAULT_MAX_BATCH_CHARS,
        intro_path: Path | None = None,
        outro_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Store credentials, TTS model, optional stinger assets, and client."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_batch_chars = max_batch_chars
        self.intro_path = intro_path
        self.outro_path = outro_path
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
        self._require_stinger_files()

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
        pause = b"\x00" * (
            int(sample_rate * _BATCH_PAUSE_SECONDS) * channels * _S16LE_SAMPLE_BYTES
        )
        pcm_bytes = pause.join(pcm_chunks)
        intro_pcm = await self._decode_stinger_pcm(
            ffmpeg_bin=ffmpeg_bin,
            path=self.intro_path,
            sample_rate=sample_rate,
            channels=channels,
        )
        outro_pcm = await self._decode_stinger_pcm(
            ffmpeg_bin=ffmpeg_bin,
            path=self.outro_path,
            sample_rate=sample_rate,
            channels=channels,
        )
        mixed_pcm = await self._mix_stingers(
            ffmpeg_bin=ffmpeg_bin,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            intro_pcm=intro_pcm,
            outro_pcm=outro_pcm,
        )
        duration_seconds = round(_pcm_duration(mixed_pcm, sample_rate, channels))
        mp3_bytes = await self._encode_mp3(
            ffmpeg_bin=ffmpeg_bin,
            pcm_bytes=mixed_pcm,
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

    def _require_stinger_files(self) -> None:
        """Raise FileNotFoundError when a configured intro or outro asset is absent."""
        for path in (self.intro_path, self.outro_path):
            if path is not None and not path.is_file():
                msg = f"Stinger audio file not found: {path}"
                raise FileNotFoundError(msg)

    @staticmethod
    async def _decode_stinger_pcm(
        *,
        ffmpeg_bin: str,
        path: Path | None,
        sample_rate: int,
        channels: int,
    ) -> bytes | None:
        """Decode an optional stinger asset to s16le PCM in the dialogue format."""
        if path is None:
            return None
        ffmpeg = (
            FFmpeg(executable=ffmpeg_bin)
            .option("y")
            .input(str(path))
            .output(
                "pipe:1",
                {
                    "f": "s16le",
                    "codec:a": "pcm_s16le",
                    "ar": sample_rate,
                    "ac": channels,
                },
            )
        )
        pcm_bytes = await ffmpeg.execute()
        if not pcm_bytes:
            msg = f"Stinger audio file decoded to empty PCM: {path}"
            raise ValueError(msg)
        return pcm_bytes

    @staticmethod
    async def _mix_stingers(  # noqa: PLR0913
        *,
        ffmpeg_bin: str,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        intro_pcm: bytes | None,
        outro_pcm: bytes | None,
    ) -> bytes:
        """Crossfade stingers onto the dialogue PCM; dialogue alone passes through."""
        if intro_pcm is None and outro_pcm is None:
            return pcm_bytes

        with tempfile.TemporaryDirectory(prefix="podcast-stingers-") as temp_dir:
            ffmpeg = FFmpeg(executable=ffmpeg_bin).option("y")
            labels: list[str] = []
            if intro_pcm is not None:
                intro_file = Path(temp_dir) / "intro.raw"
                intro_file.write_bytes(intro_pcm)
                ffmpeg.input(str(intro_file), f="s16le", ar=sample_rate, ac=channels)
                labels.append(f"[{len(labels)}:a]")
            ffmpeg.input("pipe:0", f="s16le", ar=sample_rate, ac=channels)
            labels.append(f"[{len(labels)}:a]")
            if outro_pcm is not None:
                outro_file = Path(temp_dir) / "outro.raw"
                outro_file.write_bytes(outro_pcm)
                ffmpeg.input(str(outro_file), f="s16le", ar=sample_rate, ac=channels)
                labels.append(f"[{len(labels)}:a]")

            filters: list[str] = []
            previous = labels[0]
            for position, label in enumerate(labels[1:], start=1):
                crossfaded = f"[cf{position}]"
                filters.append(
                    f"{previous}{label}"
                    f"acrossfade=d={_STINGER_CROSSFADE_SECONDS}{crossfaded}",
                )
                previous = crossfaded

            ffmpeg.option("filter_complex", ";".join(filters)).output(
                "pipe:1",
                {
                    "f": "s16le",
                    "codec:a": "pcm_s16le",
                    "ar": sample_rate,
                    "ac": channels,
                    "map": previous,
                },
            )
            return await ffmpeg.execute(pcm_bytes)

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


def _pcm_duration(pcm_bytes: bytes, sample_rate: int, channels: int) -> float:
    """Return the duration in seconds of s16le PCM bytes."""
    return len(pcm_bytes) / (sample_rate * channels * _S16LE_SAMPLE_BYTES)
