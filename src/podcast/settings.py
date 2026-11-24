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
    podcast_language: str = Field(
        default="pt-BR",
        description="Target spoken Episode language (e.g. pt-BR).",
    )
    podcast_title: str = Field(
        default="Daily Diff em Português",
        description="RSS Feed title.",
    )
    podcast_description: str = Field(
        default="Resumo diário de tecnologia e Hacker News.",
        description="RSS Feed description.",
    )
    podcast_author: str = Field(
        default="Matheus Couto",
        description="RSS iTunes author and owner name.",
    )
    podcast_owner_email: str = Field(
        min_length=1,
        description="Public RSS iTunes owner contact email.",
    )
    podcast_category: str = Field(
        default="Technology",
        description="RSS iTunes category.",
    )
    podcast_subcategory: str = Field(
        default="Tech News",
        description="RSS iTunes nested subcategory.",
    )
    podcast_cover_path: str = Field(
        default="cover.jpg",
        description="Master Feed cover path relative to the public base URL.",
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
    podcast_static_dir: Path = Field(
        default=Path("static"),
        description="Directory holding static Feed assets staged into dist.",
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
