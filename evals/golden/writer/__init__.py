"""Golden reference models for Writer evaluations."""

from datetime import date

from pydantic import BaseModel

from podcast.models import SpeechTurn


class GoldenStory(BaseModel):
    """Curated input Story with extracted readable content inline."""

    title: str
    url: str | None = None
    content: str


class GoldenCase(BaseModel):
    """Golden input/output contract matching Writer Dialogue generation."""

    date: date
    stories: list[GoldenStory]
    expected_output: list[SpeechTurn]


__all__ = ["GoldenCase", "GoldenStory"]
