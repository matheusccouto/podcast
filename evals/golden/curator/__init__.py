"""Golden validation cases for deterministic Curator ranking evaluations."""

from datetime import date

from pydantic import BaseModel, Field


class CuratorGoldenCase(BaseModel):
    """Expected ordered top-k Hacker News story_ids for a single news day."""

    date: date
    ranking: list[str] = Field(default_factory=list)


__all__ = ["CuratorGoldenCase"]
