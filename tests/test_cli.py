"""Tests for the Click CLI entrypoint in podcast.__main__."""

from datetime import date
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

import podcast.__main__ as cli_module
from podcast.models import Episode

if TYPE_CHECKING:
    from pathlib import Path

    from podcast.settings import Settings


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CliRunner for isolated CLI testing."""
    return CliRunner()


def test_cli_help_lists_options(runner: CliRunner) -> None:
    """Running `podcast --help` exits cleanly and documents CLI options."""
    result = runner.invoke(cli_module.main, ["--help"])

    assert result.exit_code == 0
    assert "--date" in result.output
    assert "--llm-model" in result.output
    assert "--tts-model" in result.output
    assert "--language" in result.output
    assert "--dist-dir" in result.output


def test_cli_applies_option_overrides(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI flags override Settings fields and pass parsed date into run_daily."""
    captured: dict[str, object] = {}

    async def _fake_run_daily(
        settings: Settings | None = None,
        *,
        today: date | None = None,
    ) -> Episode:
        """Capture invoked Settings/date and return a minimal Episode."""
        captured["settings"] = settings
        captured["today"] = today
        assert settings is not None
        assert today is not None
        return Episode(
            published_at=today,
            title="Test Episode",
            summary="Summary",
            audio_url="https://example.com/ep.mp3",
            duration_seconds=120,
            byte_length=4096,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "run_daily", _fake_run_daily)
    dist_dir = tmp_path / "out"

    result = runner.invoke(
        cli_module.main,
        [
            "--date",
            "2026-11-15",
            "--llm-model",
            "custom/llm:free",
            "--tts-model",
            "custom/tts:free",
            "--language",
            "en-US",
            "--dist-dir",
            str(dist_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Published 2026-11-15 (120s, 4096 bytes): Test Episode" in result.output
    invoked_settings = captured["settings"]
    assert captured["today"] == date(2026, 11, 15)
    assert getattr(invoked_settings, "llm_model", None) == "custom/llm:free"
    assert getattr(invoked_settings, "tts_model", None) == "custom/tts:free"
    assert getattr(invoked_settings, "podcast_language", None) == "en-US"
    assert getattr(invoked_settings, "podcast_dist_dir", None) == dist_dir
