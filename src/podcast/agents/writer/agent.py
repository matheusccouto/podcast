"""Writer subagent defined directly at module level with Jinja2 templates."""

from importlib import resources
from typing import TYPE_CHECKING

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, field_validator

from podcast.agents.model import chat_model
from podcast.models import MAX_CONTENT_CHARS, CuratedStory, Dialogue, Persona

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from langgraph.runtime import Runtime

_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    resources.files(__package__).joinpath("instructions.md").read_text("utf-8"),
    template_format="jinja2",
)
_USER_TEMPLATE = PromptTemplate.from_template(
    resources.files(__package__).joinpath("prompt.md").read_text("utf-8"),
    template_format="jinja2",
)


class WriterContext(BaseModel):
    """Per-invocation runtime context injected into the Writer subagent."""

    personas: tuple[Persona, Persona]
    language: str = "pt-BR"

    @field_validator("personas")
    @classmethod
    def validate_distinct_voices(
        cls,
        personas: tuple[Persona, Persona],
    ) -> tuple[Persona, Persona]:
        """Require non-empty, distinct voice_id values for both Personas."""
        valid_voices = {persona.voice_id.strip() for persona in personas}
        if "" in valid_voices or len(valid_voices) != len(personas):
            msg = "Both Personas require distinct, non-empty voice_id values."
            raise ValueError(msg)
        return personas


def format_writer_prompt(
    episode_date: date,
    stories: Sequence[CuratedStory],
) -> str:
    """Render the Writer user prompt from the co-located Jinja2 template."""
    return _USER_TEMPLATE.format(
        episode_date=episode_date.isoformat(),
        stories=stories,
        max_content_chars=MAX_CONTENT_CHARS,
    )


@dynamic_prompt
def _writer_prompt(request: ModelRequest[WriterContext]) -> str:
    """Render Jinja2 scriptwriting instructions for configured Personas & language."""
    ctx = request.runtime.context
    return _SYSTEM_TEMPLATE.format(
        language=ctx.language,
        personas=ctx.personas,
    )


@after_agent
def _validate_voices(
    state: AgentState[Dialogue],
    runtime: Runtime[WriterContext],
) -> None:
    """Verify generated Dialogue turns match configured Persona voice_ids."""
    dialogue = state.get("structured_response")
    if dialogue is not None:
        dialogue.checked_turns(runtime.context.personas)


writer = create_agent(
    model=chat_model,
    tools=[],
    middleware=[_writer_prompt, _validate_voices],
    context_schema=WriterContext,
    response_format=ToolStrategy(Dialogue),
    name="writer",
)
