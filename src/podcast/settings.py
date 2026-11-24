"""Application configuration and Persona definitions."""

from pathlib import Path

from pydantic import Field, SecretStr
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
    cover_model: str = Field(
        default="google/gemini-nano-banana-2.1",
        description="OpenRouter image model ID for per-Episode cover artwork.",
    )
    podcast_master_cover_path: Path = Field(
        default=Path("static/cover.jpg"),
        description="Master cover artwork used as the Episode style reference.",
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
    persona_a_voice_id: CliSuppress[str] = "Charon"
    persona_b_voice_id: CliSuppress[str] = "Aoede"

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
