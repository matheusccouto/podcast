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
    llm_model: str = Field(
        default="baidu/ernie-5.5-flash-preview:free",
        description="OpenRouter LLM model ID.",
    )
    tts_model: str = Field(
        default="fish-audio/s2.1-pro-free:free",
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
    persona_a_voice_id: CliSuppress[str] = ""
    persona_b_voice_id: CliSuppress[str] = ""

    @property
    def personas(self) -> tuple[Persona, Persona]:
        """Return the two configured co-speaker Personas."""
        return (
            Persona(
                name="Rafael",
                role="Senior systems engineer and curious skeptic",
                tone="Clear, analytical, grounded humor",
                voice_id=self.persona_a_voice_id,
            ),
            Persona(
                name="Beatriz",
                role="AI and developer-tools builder",
                tone="Warm, sharp storyteller who unpacks complex ideas",
                voice_id=self.persona_b_voice_id,
            ),
        )
