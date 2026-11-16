"""Tests for Fish Audio SpeechTurn validation and OpenRouterTTS encoding."""

import json
import math
import struct
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from podcast.models import SpeechTurn, SynthesizedAudio
from podcast.tts import OpenRouterTTS

if TYPE_CHECKING:
    from podcast.settings import Settings

_SAMPLE_VOICE_ID = "b347db033a6549378b48d00acb0d06cd"


@pytest.fixture(scope="module")
def sine_pcm_one_second() -> bytes:
    """Generate 1 second of 440Hz mono s16le PCM audio for offline ffmpeg checks."""
    sample_rate = 24_000
    samples = [
        round(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(sample_rate)
    ]
    return struct.pack(f"<{sample_rate}h", *samples)


@pytest.fixture
def live_turns(
    live_settings: Settings,
    sample_turns: list[SpeechTurn],
) -> list[SpeechTurn]:
    """Bind configured or fallback Fish Audio voice IDs onto sample_turns."""
    voices = (
        live_settings.persona_a_voice_id or _SAMPLE_VOICE_ID,
        live_settings.persona_b_voice_id or _SAMPLE_VOICE_ID,
    )
    return [
        turn.model_copy(update={"voice": voice})
        for turn, voice in zip(sample_turns, voices, strict=True)
    ]


def test_speech_turn_accepts_valid_bracket_cue() -> None:
    """SpeechTurn accepts valid Fish Audio [bracket] performance markers."""
    valid = SpeechTurn(
        voice="v1",
        text="[chuckle] Agora vamos entender como o kernel processa isso.",
    )
    assert valid.voice == "v1"


@pytest.mark.parametrize(
    "bad_text",
    [
        "[excited] Sim.",
        "<speak>Ola mundo em XML.</speak>",
        "Isso foi muito rapido (laughs) no servidor.",
        "[] Texto com colchete vazio aqui.",
        "[12345] Texto com numero no colchete aqui.",
        "[unbalanced bracket inside spoken sentence here.",
    ],
)
def test_speech_turn_rejects_invalid_markup_and_short_turns(bad_text: str) -> None:
    """SpeechTurn rejects short utterances, XML, parentheses, and bad brackets."""
    with pytest.raises(ValidationError):
        SpeechTurn(voice="v1", text=bad_text)


async def test_synthesize_encodes_pcm_to_mp3_via_ffmpeg(
    sample_turns: list[SpeechTurn],
    sine_pcm_one_second: bytes,
) -> None:
    """Synthetic PCM bytes returned by HTTP are normalized into MP3 via real ffmpeg."""
    captured_payload: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        """Capture outgoing TTS payload and return synthetic 1s PCM bytes."""
        captured_payload.update(json.loads(request.content))
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

    assert captured_payload["model"] == "fish-audio/s2.1-pro-free:free"
    assert captured_payload["response_format"] == "pcm"
    assert captured_payload["input"] == [turn.model_dump() for turn in sample_turns]
    assert audio.mp3_bytes[:3] == b"ID3"
    assert audio.duration_seconds == 1


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


async def test_synthesize_live_audio_and_encode_mp3(
    live_settings: Settings,
    live_turns: list[SpeechTurn],
) -> None:
    """OpenRouter TTS calls Fish Audio S2.1 Pro Free and encodes MP3 via ffmpeg."""
    tts = OpenRouterTTS(
        api_key=live_settings.openrouter_api_key,
        model=live_settings.tts_model,
    )

    audio = await tts.synthesize(live_turns, title="Teste Ao Vivo")

    assert isinstance(audio, SynthesizedAudio)
    assert len(audio.mp3_bytes) > 1000
    assert audio.mp3_bytes[:3] == b"ID3"
    assert audio.duration_seconds >= 1
