"""Shared runtime-configurable OpenRouter chat model."""

from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

chat_model = cast(
    "BaseChatModel",
    init_chat_model(
        "deepseek/deepseek-v4.1-flash",
        model_provider="openrouter",
        configurable_fields=("model", "api_key"),
        config_prefix="llm",
        reasoning={"effort": "medium"},
        timeout=120_000,
    ),
)


def llm_config(*, model: str, api_key: SecretStr) -> RunnableConfig:
    """Build a LangChain RunnableConfig binding runtime model and API key."""
    return {
        "configurable": {
            "llm_model": model,
            "llm_api_key": api_key.get_secret_value(),
        },
    }
