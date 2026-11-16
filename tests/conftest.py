"""Shared pytest fixtures for Podcast unit and integration suites."""

import pytest
from pydantic import ValidationError

from podcast.models import SpeechTurn
from podcast.settings import Settings


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    """Return validated runtime Settings or skip live tests if API key is unset."""
    try:
        return Settings()
    except ValidationError:
        pytest.skip("OPENROUTER_API_KEY not configured")


@pytest.fixture
def sample_turns() -> list[SpeechTurn]:
    """Return a valid two-turn SpeechTurn sample shared across feed and TTS tests."""
    return [
        SpeechTurn(
            voice="voice-a",
            text="Hoje vamos analisar uma novidade importante de sistemas.",
        ),
        SpeechTurn(
            voice="voice-b",
            text=(
                "[excited] Essa mudanca traz ganhos reais de desempenho em producao."
            ),
        ),
    ]
