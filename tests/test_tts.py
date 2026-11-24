"""Tests for SpeechTurn validation and OpenRouterTTS encoding."""

import itertools
import json
import math
import re
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from podcast.models import SpeechTurn, SynthesizedAudio
from podcast.settings import Settings
from podcast.tts import OpenRouterTTS, StingerPaths

_SAMPLE_VOICES = ("Puck", "Aoede")
_LOUDNESS_TARGET_LUFS = -16.0
_LUFS_RE = re.compile(r"I:\s+(-?\d+\.\d+) LUFS")
_DECODE_SAMPLE_RATE = 24_000


def _sine_pcm(
    seconds: float,
    frequency: float,
    *,
    amplitude: int = 8000,
    sample_rate: int = 24_000,
) -> bytes:
    """Build mono s16le PCM sine samples for offline audio tests."""
    samples = [
        round(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(int(sample_rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


@pytest.fixture(scope="module")
def sine_pcm_one_second() -> bytes:
    """Generate 1 second of 440Hz mono s16le PCM audio for offline ffmpeg checks."""
    return _sine_pcm(1.0, 440.0)


@pytest.fixture(scope="module")
def quiet_pcm_six_seconds() -> bytes:
    """Generate 6 seconds of quiet 440Hz mono s16le PCM for loudness checks."""
    return _sine_pcm(6.0, 440.0, amplitude=400)


@pytest.fixture(scope="module")
def nonstationary_pcm_six_seconds() -> bytes:
    """Generate 1s loud then 5s quiet 440Hz PCM, the varying intensity speech has."""
    return _sine_pcm(1.0, 440.0, amplitude=30_000) + _sine_pcm(
        5.0, 440.0, amplitude=400
    )


def _run_ffmpeg(
    arguments: list[str],
    *,
    stdin_bytes: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    """Run ffmpeg once for offline audio assertions, failing loudly on error."""
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin is not None
    return subprocess.run(  # noqa: S603
        [ffmpeg_bin, "-hide_banner", "-nostats", *arguments],
        input=stdin_bytes,
        check=True,
        capture_output=True,
    )


def _measure_integrated_lufs(mp3_path: Path) -> float:
    """Decode an MP3 through ffmpeg's ebur128 filter and return integrated LUFS."""
    result = _run_ffmpeg(["-i", str(mp3_path), "-af", "ebur128", "-f", "null", "-"])
    matches = _LUFS_RE.findall(result.stderr.decode("utf-8", errors="replace"))
    assert matches
    return float(matches[-1])


def _write_sine_wav(
    path: Path,
    *,
    seconds: float,
    frequency: float,
    sample_rate: int = 24_000,
) -> None:
    """Write a mono 16-bit sine WAV fixture for stinger crossfade tests."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(_sine_pcm(seconds, frequency, sample_rate=sample_rate))


def _decode_mp3_samples(mp3_bytes: bytes) -> list[int]:
    """Decode MP3 bytes to mono 24 kHz s16le samples for ordering assertions."""
    result = _run_ffmpeg(
        [
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(_DECODE_SAMPLE_RATE),
            "pipe:1",
        ],
        stdin_bytes=mp3_bytes,
    )
    return list(struct.unpack(f"<{len(result.stdout) // 2}h", result.stdout))


def _zero_crossing_frequency(
    samples: list[int],
    *,
    sample_start: float,
    sample_span: float,
    sample_rate: int = 24_000,
) -> float:
    """Estimate the tone frequency of a window by counting zero crossings."""
    start = int(sample_start * sample_rate)
    window = samples[start : start + int(sample_span * sample_rate)]
    crossings = sum(
        1 for left, right in itertools.pairwise(window) if (left < 0) != (right < 0)
    )
    return crossings / (2 * sample_span)


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


async def _synthesize_pcm_mp3(
    turns: list[SpeechTurn],
    pcm_bytes: bytes,
    title: str,
) -> SynthesizedAudio:
    """Run the real synthesize path over mock-returned PCM bytes."""
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=pcm_bytes,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        ),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(api_key=SecretStr("unused"), client=client)
        return await tts.synthesize(turns, title=title)


async def test_synthesize_normalizes_quiet_audio_to_target_loudness(
    sample_turns: list[SpeechTurn],
    quiet_pcm_six_seconds: bytes,
    tmp_path: Path,
) -> None:
    """Quiet PCM is encoded to an MP3 whose integrated loudness hits -16 LUFS."""
    audio = await _synthesize_pcm_mp3(
        sample_turns,
        quiet_pcm_six_seconds,
        "Teste Loudness",
    )

    mp3_path = tmp_path / "episode.mp3"
    mp3_path.write_bytes(audio.mp3_bytes)

    assert audio.duration_seconds == 6
    assert _measure_integrated_lufs(mp3_path) == pytest.approx(
        _LOUDNESS_TARGET_LUFS,
        abs=1.0,
    )


async def test_synthesize_normalizes_nonstationary_audio_to_target_loudness(
    sample_turns: list[SpeechTurn],
    nonstationary_pcm_six_seconds: bytes,
    tmp_path: Path,
) -> None:
    """Loud-then-quiet PCM still lands on -16 LUFS integrated, not -12.7."""
    audio = await _synthesize_pcm_mp3(
        sample_turns,
        nonstationary_pcm_six_seconds,
        "Teste Loudness Variavel",
    )

    mp3_path = tmp_path / "episode.mp3"
    mp3_path.write_bytes(audio.mp3_bytes)

    assert audio.duration_seconds == 6
    assert _measure_integrated_lufs(mp3_path) == pytest.approx(
        _LOUDNESS_TARGET_LUFS,
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


async def test_synthesize_crossfades_intro_speech_and_outro(
    sample_turns: list[SpeechTurn],
    tmp_path: Path,
) -> None:
    """Decoded audio is 7s of crossfaded parts, not the 8s a concat would give."""
    intro_path = tmp_path / "intro.wav"
    outro_path = tmp_path / "outro.wav"
    _write_sine_wav(intro_path, seconds=2.0, frequency=300.0)
    _write_sine_wav(outro_path, seconds=3.0, frequency=700.0)
    speech_pcm = _sine_pcm(3.0, 440.0)
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=speech_pcm,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        ),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(
            api_key=SecretStr("unused"),
            stingers=StingerPaths(intro=intro_path, outro=outro_path),
            client=client,
        )
        audio = await tts.synthesize(sample_turns, title="Teste Stingers")

    assert audio.mp3_bytes[:3] == b"ID3"
    samples = _decode_mp3_samples(audio.mp3_bytes)
    decoded_seconds = len(samples) / _DECODE_SAMPLE_RATE
    # 2s intro + 3s speech + 3s outro is 8s of material; two 0.5s crossfades
    # must bring the published episode down to 7s.
    assert decoded_seconds == pytest.approx(7.0, rel=0.02)

    mp3_path = tmp_path / "episode.mp3"
    mp3_path.write_bytes(audio.mp3_bytes)
    assert _measure_integrated_lufs(mp3_path) == pytest.approx(
        _LOUDNESS_TARGET_LUFS,
        abs=1.0,
    )
    assert _zero_crossing_frequency(
        samples,
        sample_start=0.25,
        sample_span=0.5,
    ) == pytest.approx(300.0, rel=0.05)
    assert _zero_crossing_frequency(
        samples,
        sample_start=3.0,
        sample_span=0.4,
    ) == pytest.approx(440.0, rel=0.05)
    assert _zero_crossing_frequency(
        samples,
        sample_start=6.0,
        sample_span=0.5,
    ) == pytest.approx(700.0, rel=0.05)


async def test_synthesize_fails_loudly_when_stinger_file_is_missing(
    sample_turns: list[SpeechTurn],
    tmp_path: Path,
) -> None:
    """A configured stinger path that does not exist raises FileNotFoundError."""
    tts = OpenRouterTTS(
        api_key=SecretStr("unused"),
        stingers=StingerPaths(intro=tmp_path / "missing-intro.mp3"),
    )

    with pytest.raises(FileNotFoundError, match="Stinger audio file not found"):
        await tts.synthesize(sample_turns, title="Missing Stinger")


async def test_synthesize_crossfades_outro_only_with_explicit_input_labels(
    sample_turns: list[SpeechTurn],
    tmp_path: Path,
) -> None:
    """Outro-only keeps dialogue as input 0 and the outro as input 1, not shifted."""
    outro_path = tmp_path / "outro-only.wav"
    _write_sine_wav(outro_path, seconds=3.0, frequency=700.0)
    speech_pcm = _sine_pcm(3.0, 440.0)
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=speech_pcm,
            headers={"content-type": "audio/pcm; rate=24000; channels=1"},
        ),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        tts = OpenRouterTTS(
            api_key=SecretStr("unused"),
            stingers=StingerPaths(outro=outro_path),
            client=client,
        )
        audio = await tts.synthesize(sample_turns, title="Teste So Outro")

    samples = _decode_mp3_samples(audio.mp3_bytes)
    # 3s speech + 3s outro minus one 0.5s crossfade must land on 5.5s.
    assert len(samples) / _DECODE_SAMPLE_RATE == pytest.approx(5.5, rel=0.02)
    assert _zero_crossing_frequency(
        samples,
        sample_start=0.5,
        sample_span=0.5,
    ) == pytest.approx(440.0, rel=0.05)
    assert _zero_crossing_frequency(
        samples,
        sample_start=4.5,
        sample_span=0.5,
    ) == pytest.approx(700.0, rel=0.05)


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
