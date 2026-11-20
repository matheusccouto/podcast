"""Zero-dependency domain models with self-contained Pydantic validation."""

import re
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CONTENT_CHARS = 40_000
_MIN_TURN_WORDS = 3
_MAX_TAG_WORDS = 6
_MIN_DIALOGUE_WORDS = 400
_MAX_DIALOGUE_WORDS = 1500
_MIN_DISTINCT_VOICES = 2

_XML_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_TAG_RE = re.compile(r"\[([^\[\]]*)\]")
_VALID_TAG_BODY_RE = re.compile(r"^[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\s,'-]{0,40}$")
_PAREN_DIRECTION_RE = re.compile(
    r"\((?:laughs?|chuckles?|sighs?|gasps?|groans?|pauses?|music|sfx|applause|"
    r"excited|whispering|whispers?|rindo|risos|pausa|suspira|sussurrando|"
    r"musica|música|aplausos|animad[oa]|surpres[oa])\b[^)]*\)",
    re.IGNORECASE,
)


def _spoken_words(text: str) -> list[str]:
    """Return spoken words in a text string after stripping inline [bracket] cues."""
    return _BRACKET_TAG_RE.sub("", text).split()


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
    is_lead: bool = False
    key_facts: list[str] = Field(default_factory=list)


class CuratedStory(_DomainModel):
    """Readable Story material handed to the Writer subagent."""

    title: str
    url: str | None = None
    content: str
    comments: list[str] = Field(default_factory=list)
    is_lead: bool = False
    key_facts: list[str] = Field(default_factory=list)


class ResearchBrief(_DomainModel):
    """Validated curation plan emitted by the Curator subagent."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    segments: list[Segment] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_segments(self) -> ResearchBrief:
        """Ensure segment story_ids are unique and exactly one segment is lead."""
        ids = [segment.story_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            msg = "Duplicate story_id in curated segments."
            raise ValueError(msg)
        lead_count = sum(1 for segment in self.segments if segment.is_lead)
        if lead_count != 1:
            msg = f"Expected exactly 1 lead Segment, got {lead_count}."
            raise ValueError(msg)
        return self

    def build_curated_stories(
        self,
        *,
        stories: Mapping[str, Story],
        contents: Mapping[str, str],
        comments: Mapping[str, list[str]],
        unreadable: AbstractSet[str],
    ) -> tuple[list[Segment], list[CuratedStory]]:
        """Validate grounding against fetched state and build CuratedStories."""
        normalized: list[Segment] = []
        curated: list[CuratedStory] = []
        for segment in self.segments:
            story = stories.get(segment.story_id)
            if story is None:
                msg = f"Curator selected unknown story_id '{segment.story_id}'."
                raise ValueError(msg)
            if segment.story_id in unreadable:
                msg = f"Curator selected unreadable Story '{segment.story_id}'."
                raise ValueError(msg)
            if segment.story_id not in contents:
                msg = (
                    f"Curator selected Story '{segment.story_id}' without calling "
                    "fetch_story_content first."
                )
                raise ValueError(msg)
            normalized.append(segment.model_copy(update={"url": story.url}))
            curated.append(
                CuratedStory(
                    title=segment.title,
                    url=story.url,
                    content=contents[segment.story_id],
                    comments=comments.get(segment.story_id, []),
                    is_lead=segment.is_lead,
                    key_facts=segment.key_facts,
                ),
            )
        return normalized, curated


class SpeechTurn(_DomainModel):
    """Single TTS turn matching OpenRouter Fish Audio SpeechTurn schema."""

    voice: str = Field(min_length=1)
    text: str

    @field_validator("text")
    @classmethod
    def validate_tts_text(cls, value: str) -> str:
        """Validate spoken turn length, XML/paren exclusion, and [bracket] tags."""
        if _XML_TAG_RE.search(value):
            msg = "Contains forbidden XML/SSML tags."
            raise ValueError(msg)
        if _PAREN_DIRECTION_RE.search(value):
            msg = "Use [bracket] cues instead of (parenthesis) stage directions."
            raise ValueError(msg)

        for match in _BRACKET_TAG_RE.finditer(value):
            tag = match.group(1).strip()
            if (
                not tag
                or not _VALID_TAG_BODY_RE.fullmatch(tag)
                or len(tag.split()) > _MAX_TAG_WORDS
            ):
                msg = (
                    f"Invalid Fish Audio bracket tag '[{tag}]'; use concise "
                    "natural-language delivery cues (1-6 words, no digits or symbols)."
                )
                raise ValueError(msg)

        without_tags = _BRACKET_TAG_RE.sub("", value)
        if "[" in without_tags or "]" in without_tags:
            msg = "Contains unbalanced or nested square brackets."
            raise ValueError(msg)

        if len(without_tags.split()) < _MIN_TURN_WORDS:
            msg = (
                f"Turn must contain at least {_MIN_TURN_WORDS} spoken words "
                "outside [bracket] tags so TTS language detection stays stable."
            )
            raise ValueError(msg)
        return value


class Dialogue(_DomainModel):
    """Validated two-Persona spoken dialogue emitted by the Writer subagent."""

    turns: list[SpeechTurn] = Field(min_length=4)

    @property
    def spoken_word_count(self) -> int:
        """Count spoken words across all turns excluding inline [bracket] cues."""
        return sum(len(_spoken_words(turn.text)) for turn in self.turns)

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
