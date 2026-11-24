"""Tests for SpeechTurn validation and OpenRouterTTS encoding."""

import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from podcast.models import SpeechTurn, SynthesizedAudio
from podcast.settings import Settings
from podcast.tts import LOUDNESS_TARGET_LUFS, OpenRouterTTS

_SAMPLE_VOICES = ("Puck", "Aoede")
_LUFS_RE = re.compile(r"I:\s+(-?\d+\.\d+) LUFS")


@pytest.fixture(scope="module")
def sine_pcm_one_second() -> bytes:
    """Generate 1 second of 440Hz mono s16le PCM audio for offline ffmpeg checks."""
    sample_rate = 24_000
    samples = [
        round(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(sample_rate)
    ]
    return struct.pack(f"<{sample_rate}h", *samples)


@pytest.fixture(scope="module")
def quiet_pcm_six_seconds() -> bytes:
    """Generate 6 seconds of quiet 440Hz mono s16le PCM for loudness checks."""
    sample_rate = 24_000
    samples = [
        round(400 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(sample_rate * 6)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _measure_integrated_lufs(mp3_path: Path) -> float:
    """Decode an MP3 through ffmpeg's ebur128 filter and return integrated LUFS."""
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin is not None
    result = subprocess.run(  # noqa: S603
        [
            ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-i",
            str(mp3_path),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = _LUFS_RE.findall(result.stderr)
    assert matches
    return float(matches[-1])


@pytest.fixture
def live_turns(
    live_settings: Settings,
    sample_turns: list[SpeechTurn],
) -> list[SpeechTurn]:
    """Bind configured or fallback OpenRouter TTS voice IDs onto sample_turns."""
    voices = (
        live_settings.persona_a_voice_id or _SAMPLE_VOICES[0],
        live_settings.persona_b_voice_id or _SAMPLE_VOICES[1],
    )
    return [
        turn.model_copy(update={"voice": voice})
        for turn, voice in zip(sample_turns, voices, strict=True)
    ]


def test_speech_turn_accepts_verbatim_text_and_instructions() -> None:
    """SpeechTurn accepts clean spoken text with model-agnostic delivery direction."""
    valid = SpeechTurn(
        voice="Puck",
        text="Agora vamos entender como o kernel processa isso.",
        instructions="Amused, brief chuckle at the start",
    )
    assert valid.voice == "Puck"
    assert valid.instructions == "Amused, brief chuckle at the start"


@pytest.mark.parametrize(
    "bad_text",
    [
        "Sim.",
        "<speak>Ola mundo em XML.</speak>",
        "Isso foi muito rapido (laughs) no servidor.",
        "[chuckle] Agora vamos entender como o kernel processa isso.",
        "[unbalanced bracket inside spoken sentence here.",
    ],
)
def test_speech_turn_rejects_inline_markup_and_short_turns(bad_text: str) -> None:
    """SpeechTurn rejects short utterances, XML, stage parens, and inline brackets."""
    with pytest.raises(ValidationError):
        SpeechTurn(voice="Puck", text=bad_text)


async def test_synthesize_encodes_pcm_to_mp3_via_ffmpeg(
    sample_turns: list[SpeechTurn],
    sine_pcm_one_second: bytes,
) -> None:
    """Multi-turn batch PCM bytes are normalized into MP3 via ffmpeg."""
    captured_payloads: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        """Capture outgoing multi-turn TTS payload and return synthetic 1s PCM bytes."""
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=sine_pcm_one_second,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        tts = OpenRouterTTS(
            api_key=SecretStr("unused"),
            client=client,
        )
        audio = await tts.synthesize(sample_turns, title="Teste de Sintese")

    assert captured_payloads == [
        {
            "model": "google/gemini-3.8-flash-tts",
            "input": [
                {
                    "text": turn.text,
                    "voice": turn.voice,
                    "instructions": turn.instructions,
                }
                for turn in sample_turns
            ],
            "response_format": "pcm",
        },
    ]
    assert audio.mp3_bytes[:3] == b"ID3"
    assert audio.duration_seconds == 1


async def test_synthesize_normalizes_quiet_audio_to_target_loudness(
    sample_turns: list[SpeechTurn],
    quiet_pcm_six_seconds: bytes,
    tmp_path: Path,
) -> None:
    """Quiet PCM is encoded to an MP3 whose integrated loudness hits -16 LUFS."""
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=quiet_pcm_six_seconds,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        ),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(api_key=SecretStr("unused"), client=client)
        audio = await tts.synthesize(sample_turns, title="Teste Loudness")

    mp3_path = tmp_path / "episode.mp3"
    mp3_path.write_bytes(audio.mp3_bytes)

    assert audio.duration_seconds == 6
    assert _measure_integrated_lufs(mp3_path) == pytest.approx(
        LOUDNESS_TARGET_LUFS,
        abs=1.0,
    )


async def test_synthesize_splits_batches_and_forwards_instructions(
    sine_pcm_one_second: bytes,
) -> None:
    """Long dialogues split on turn boundaries and include per-turn instructions."""
    turns = [
        SpeechTurn(
            voice="v1",
            text="Primeiro turno bem longo para forcar divisao em dois lotes.",
            instructions="Speak warmly in pt-BR.",
        ),
        SpeechTurn(
            voice="v2",
            text="Segundo turno tambem longo para ir no segundo lote de audio.",
        ),
    ]
    captured_payloads: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=sine_pcm_one_second,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        tts = OpenRouterTTS(
            api_key=SecretStr("unused"),
            max_batch_chars=90,
            client=client,
        )
        audio = await tts.synthesize(turns, title="Teste Lotes")

    assert captured_payloads == [
        {
            "model": "google/gemini-3.8-flash-tts",
            "input": [
                {
                    "text": turns[0].text,
                    "voice": "v1",
                    "instructions": "Speak warmly in pt-BR.",
                },
            ],
            "response_format": "pcm",
        },
        {
            "model": "google/gemini-3.8-flash-tts",
            "input": [{"text": turns[1].text, "voice": "v2"}],
            "response_format": "pcm",
        },
    ]
    assert audio.duration_seconds == 2


async def test_synthesize_refuses_to_guess_missing_pcm_header_params(
    sample_turns: list[SpeechTurn],
) -> None:
    """Missing rate or channels in PCM Content-Type raises ValueError loudly."""
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=b"\x00\x00",
            headers={"content-type": "audio/pcm"},
        ),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(api_key=SecretStr("unused"), client=client)
        with pytest.raises(ValueError, match="rate or channels"):
            await tts.synthesize(sample_turns, title="Missing PCM Header")


async def test_synthesize_rejects_empty_pcm_audio(
    sample_turns: list[SpeechTurn],
) -> None:
    """Empty PCM response raises ValueError loudly instead of writing a 0s MP3."""
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=b"",
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        ),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(api_key=SecretStr("unused"), client=client)
        with pytest.raises(ValueError, match="empty PCM"):
            await tts.synthesize(sample_turns, title="Empty PCM")


@pytest.mark.live
async def test_synthesize_live_audio_and_encode_mp3(
    live_settings: Settings,
    live_turns: list[SpeechTurn],
) -> None:
    """Live TTS synthesis returns valid normalized MP3 audio."""
    tts = OpenRouterTTS(
        api_key=live_settings.openrouter_api_key,
        model=live_settings.tts_model,
    )

    audio = await tts.synthesize(live_turns, title="Teste Ao Vivo")

    assert isinstance(audio, SynthesizedAudio)
    assert len(audio.mp3_bytes) > 1000
    assert audio.mp3_bytes[:3] == b"ID3"
    assert audio.duration_seconds >= 1
