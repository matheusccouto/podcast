"""Shared runtime-configurable OpenRouter chat model."""

from typing import TYPE_CHECKING, cast

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

chat_model = cast(
    "BaseChatModel",
    init_chat_model(
        "baidu/ernie-5.5-flash-preview:free",
        model_provider="openrouter",
        configurable_fields=("model", "api_key"),
        config_prefix="llm",
        timeout=120_000,
    ),
)
