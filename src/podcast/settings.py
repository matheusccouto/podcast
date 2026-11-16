"""Application configuration and Persona definitions."""

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from podcast.models import Persona

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr
    llm_model: str = "baidu/ernie-5.5-flash-preview:free"
    tts_model: str = "fish-audio/s2.1-pro-free:free"
    podcast_language: str = "pt-BR"
    podcast_title: str = "HN Daily BR"
    podcast_description: str = "Resumo diário de tecnologia e Hacker News."
    podcast_author: str = "Matheus Couto"
    podcast_base_url: str = "http://localhost:8000"
    podcast_releases_base_url: str = "http://localhost:8000/episodes"
    podcast_dist_dir: Path = Path("dist")
    persona_a_voice_id: str = ""
    persona_b_voice_id: str = ""

    @property
    def personas(self) -> tuple[Persona, Persona]:
        """Return the two configured co-speaker Personas."""
        return (
            Persona(
                name="Lucas",
                role="Senior systems engineer and curious skeptic",
                tone="Clear, analytical, grounded humor",
                voice_id=self.persona_a_voice_id,
            ),
            Persona(
                name="Mariana",
                role="AI and developer-tools builder",
                tone="Warm, sharp storyteller who unpacks complex ideas",
                voice_id=self.persona_b_voice_id,
            ),
        )

    @property
    def llm_config(self) -> RunnableConfig:
        """Return configurable LangChain RunnableConfig for runtime LLM binding."""
        return {
            "configurable": {
                "llm_model": self.llm_model,
                "llm_api_key": self.openrouter_api_key.get_secret_value(),
            },
        }
