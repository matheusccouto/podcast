"""Zero-dependency domain models with self-contained Pydantic validation."""

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CONTENT_CHARS = 40_000
_MIN_TURN_WORDS = 3
_MIN_DIALOGUE_WORDS = 400
_MAX_DIALOGUE_WORDS = 1500
_MIN_DISTINCT_VOICES = 2

_XML_TAG_RE = re.compile(r"<[^>]+>")
_PAREN_DIRECTION_RE = re.compile(
    r"\((?:laughs?|chuckles?|sighs?|gasps?|groans?|pauses?|music|sfx|applause|"
    r"excited|whispering|whispers?|rindo|risos|pausa|suspira|sussurrando|"
    r"musica|música|aplausos|animad[oa]|surpres[oa])\b[^)]*\)",
    re.IGNORECASE,
)


class _DomainModel(BaseModel):
    """Base immutable value object with whitespace-stripped string fields."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


class Persona(_DomainModel):
    """Speaker identity shared by Script generation and audio synthesis."""

    name: str
    role: str
    tone: str
    voice_id: str


def validate_persona_voices(personas: tuple[Persona, Persona]) -> set[str]:
    """Validate and return the distinct, non-empty voice_id set for two Personas."""
    valid_voices = {persona.voice_id for persona in personas}
    if "" in valid_voices or len(valid_voices) != len(personas):
        msg = "Both Personas require distinct, non-empty voice_id values."
        raise ValueError(msg)
    return valid_voices


class Story(_DomainModel):
    """Raw candidate news Story collected from an upstream Source."""

    id: str
    source: str
    title: str
    url: str | None = None
    discussion_url: str
    score: int = 0
    comment_count: int = 0


class Segment(_DomainModel):
    """Curated topic unit within an Episode referencing a verified Story."""

    story_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str | None = None
    key_facts: list[str] = Field(default_factory=list)


class CuratedStory(_DomainModel):
    """Readable Story material handed to the Writer subagent."""

    title: str
    url: str | None = None
    content: str
    comments: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)


class SpeechTurn(_DomainModel):
    """Single spoken TTS turn for one Persona voice."""

    voice: str = Field(
        min_length=1,
        description="Exact Persona voice_id from the prompt (never the display name).",
    )
    text: str
    instructions: str | None = Field(default=None, max_length=160)

    @field_validator("text")
    @classmethod
    def validate_tts_text(cls, value: str) -> str:
        """Keep turn text verbatim speech: reject XML, stage parens, and brackets."""
        if _XML_TAG_RE.search(value):
            msg = "Contains forbidden XML/SSML tags."
            raise ValueError(msg)
        if _PAREN_DIRECTION_RE.search(value):
            msg = "Put delivery cues in `instructions`, not (parenthesis) in `text`."
            raise ValueError(msg)
        if "[" in value or "]" in value:
            msg = "Put delivery cues in `instructions`, not [bracket] tags in `text`."
            raise ValueError(msg)

        if len(value.split()) < _MIN_TURN_WORDS:
            msg = (
                f"Turn must contain at least {_MIN_TURN_WORDS} spoken words "
                "so TTS language detection stays stable."
            )
            raise ValueError(msg)
        return value


class Dialogue(_DomainModel):
    """Validated two-Persona spoken dialogue emitted by the Writer subagent."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    turns: list[SpeechTurn] = Field(min_length=4)

    @property
    def spoken_word_count(self) -> int:
        """Count spoken words across all turns."""
        return sum(len(turn.text.split()) for turn in self.turns)

    @model_validator(mode="after")
    def validate_dialogue(self) -> Dialogue:
        """Verify two-speaker interaction and reasonable spoken word budget."""
        voices = {turn.voice for turn in self.turns}
        if len(voices) < _MIN_DISTINCT_VOICES:
            msg = "Dialogue must alternate between both Persona voices."
            raise ValueError(msg)
        words = self.spoken_word_count
        if words < _MIN_DIALOGUE_WORDS:
            msg = (
                f"Dialogue has {words} spoken words (min {_MIN_DIALOGUE_WORDS}); "
                "expand discussion with deeper technical details from the stories."
            )
            raise ValueError(msg)
        if words > _MAX_DIALOGUE_WORDS:
            msg = (
                f"Dialogue has {words} spoken words (max {_MAX_DIALOGUE_WORDS}); "
                "keep pacing tight."
            )
            raise ValueError(msg)
        return self

    def checked_turns(self, personas: tuple[Persona, Persona]) -> list[SpeechTurn]:
        """Validate every turn voice matches the configured Persona voice_ids."""
        valid_voices = validate_persona_voices(personas)
        unknown_voices = {turn.voice for turn in self.turns} - valid_voices
        if unknown_voices:
            allowed = ", ".join(sorted(valid_voices))
            bad = ", ".join(sorted(unknown_voices))
            msg = f"Unknown voice(s) {bad}; expected only: {allowed}."
            raise ValueError(msg)
        return self.turns


class Script(_DomainModel):
    """Localized podcast Episode Script ready for synthesis and publishing."""

    published_at: date
    title: str
    summary: str
    language: str = "pt-BR"
    segments: list[Segment] = Field(min_length=1)
    turns: list[SpeechTurn] = Field(min_length=2)


class SynthesizedAudio(_DomainModel):
    """Pure value object containing encoded MP3 bytes and duration."""

    mp3_bytes: bytes
    duration_seconds: int


class Episode(_DomainModel):
    """Published podcast audio artifact and feed metadata."""

    published_at: date
    title: str
    summary: str
    audio_url: str
    byte_length: int
    duration_seconds: int
    cover_url: str
