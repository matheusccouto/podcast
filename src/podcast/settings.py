"""Application configuration and Persona definitions."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict

from podcast.models import Persona


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables, `.env`, or CLI flags."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_kebab_case=True,
    )

    openrouter_api_key: CliSuppress[SecretStr]
    exa_api_key: CliSuppress[SecretStr]
    curator_model: str = Field(
        default="jev-1.14",
        description="OpenRouter System One model ID for Curator ranking.",
    )
    llm_model: str = Field(
        default="deepseek/deepseek-v4.1-flash",
        description="OpenRouter LLM model ID.",
    )
    tts_model: str = Field(
        default="google/gemini-3.8-flash-tts",
        description="OpenRouter TTS model ID.",
    )
    podcast_language: str = Field(
        default="pt-BR",
        description="Target spoken Episode language (e.g. pt-BR).",
    )
    podcast_title: str = Field(
        default="HN Daily BR",
        description="RSS Feed title.",
    )
    podcast_description: str = Field(
        default="Resumo diário de tecnologia e Hacker News.",
        description="RSS Feed description.",
    )
    podcast_author: str = Field(
        default="Matheus Couto",
        description="RSS iTunes author name.",
    )
    podcast_base_url: str = Field(
        default="http://localhost:8000",
        description="Public base URL hosting feed.xml.",
    )
    podcast_releases_base_url: str = Field(
        default="http://localhost:8000/episodes",
        description="Public base URL hosting Episode MP3 assets.",
    )
    podcast_dist_dir: Path = Field(
        default=Path("dist"),
        description="Output directory for published RSS Feed and Episode files.",
    )
    podcast_intro_path: Path | None = Field(
        default=None,
        description=(
            "Optional static intro stinger audio path "
            "(e.g. assets/intro.mp3). Omit to start Episodes with dialogue."
        ),
    )
    podcast_outro_path: Path | None = Field(
        default=None,
        description=(
            "Optional static outro stinger audio path "
            "(e.g. assets/outro.mp3). Omit to end Episodes with dialogue."
        ),
    )
    persona_a_voice_id: CliSuppress[str] = "Charon"
    persona_b_voice_id: CliSuppress[str] = "Aoede"

    @field_validator("podcast_intro_path", "podcast_outro_path", mode="before")
    @classmethod
    def _blank_path_is_unset(cls, value: object) -> object:
        """Treat a blank env var (e.g. an unset CI variable) as no stinger."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def personas(self) -> tuple[Persona, Persona]:
        """Return the two configured co-speaker Personas."""
        return (
            Persona(
                name="Carlos",
                role=(
                    "Senior systems engineer and pragmatic skeptic who "
                    "interrogates hype, traces incentives, and asks who pays "
                    "the second-order cost"
                ),
                tone="Measured, sharp, dry deadpan humor, curious rather than cynical",
                voice_id=self.persona_a_voice_id,
            ),
            Persona(
                name="Alice",
                role=(
                    "AI and developer-tools builder who spots why a "
                    "breakthrough matters, unpacks complex mechanisms clearly, "
                    "and brings hands-on curiosity"
                ),
                tone=(
                    "Warm, breezy, quick-witted storyteller with contagious "
                    "energy and plain-spoken clarity"
                ),
                voice_id=self.persona_b_voice_id,
            ),
        )
