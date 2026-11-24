"""Tests for the CLI entrypoint in podcast.__main__."""

from datetime import date
from pathlib import Path

import pytest
from pydantic_settings import CliApp

import podcast.__main__ as cli_module
from podcast.models import Episode
from podcast.settings import Settings


def test_cli_help_lists_options_and_hides_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Running `podcast --help` documents CLI options and suppresses secret flags."""
    with pytest.raises(SystemExit) as exc_info:
        CliApp.run(cli_module.PodcastCLI, cli_args=["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--episode-date" in output
    assert "--llm-model" in output
    assert "--tts-model" in output
    assert "--podcast-language" in output
    assert "--podcast-dist-dir" in output
    assert "--openrouter-api-key" not in output
    assert "--exa-api-key" not in output


def test_cli_applies_option_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
            cover_url="https://example.com/ep.jpg",
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(cli_module, "run_daily", _fake_run_daily)
    dist_dir = tmp_path / "out"

    CliApp.run(
        cli_module.PodcastCLI,
        cli_args=[
            "--episode-date",
            "2026-11-15",
            "--llm-model",
            "custom/llm:free",
            "--tts-model",
            "custom/tts:free",
            "--podcast-language",
            "en-US",
            "--podcast-dist-dir",
            str(dist_dir),
        ],
    )

    output = capsys.readouterr().out
    assert "Published 2026-11-15 (120s, 4096 bytes): Test Episode" in output
    invoked_settings = captured["settings"]
    assert captured["today"] == date(2026, 11, 15)
    assert getattr(invoked_settings, "llm_model", None) == "custom/llm:free"
    assert getattr(invoked_settings, "tts_model", None) == "custom/tts:free"
    assert getattr(invoked_settings, "podcast_language", None) == "en-US"
    assert getattr(invoked_settings, "podcast_dist_dir", None) == dist_dir
