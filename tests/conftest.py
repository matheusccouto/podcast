"""Shared pytest fixtures for Podcast unit and integration suites."""

import pytest
from pydantic import ValidationError

from podcast.models import SpeechTurn
from podcast.settings import Settings

_LIVE_KEY_FIELDS = frozenset({"openrouter_api_key", "exa_api_key"})


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    """Return validated runtime Settings, skipping only when live API keys are unset."""
    try:
        return Settings(podcast_owner_email="owner@example.com")
    except ValidationError as error:
        missing = {
            detail["loc"][0] for detail in error.errors() if detail["type"] == "missing"
        }
        if not missing or not missing <= _LIVE_KEY_FIELDS:
            raise
        reason = ", ".join(sorted(map(str, missing)))
        pytest.skip(f"live API keys not configured: {reason}")


@pytest.fixture
def sample_turns() -> list[SpeechTurn]:
    """Return a valid two-turn SpeechTurn sample shared across feed and TTS tests."""
    return [
        SpeechTurn(
            voice="voice-a",
            text="Hoje vamos analisar uma novidade importante de sistemas.",
            instructions="Warm, curious, conversational podcast tone",
        ),
        SpeechTurn(
            voice="voice-b",
            text="Essa mudanca traz ganhos reais de desempenho em producao.",
            instructions="Excited, upbeat reaction",
        ),
    ]
