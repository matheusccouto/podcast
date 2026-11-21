"""DeepEval evaluations for Writer Script podcast craft and naturalness."""

import asyncio
from datetime import date
from typing import cast

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.conftest import load_golden_dataset, serialize_turns
from evals.golden import GoldenStory
from podcast.agents import WriterContext, format_writer_prompt, llm_config, writer
from podcast.models import CuratedStory, Dialogue
from podcast.settings import Settings

_DATASET = load_golden_dataset()
if not _DATASET.goldens:
    pytest.skip(
        "No fixtures in evals/golden/writer/. Run /refresh-writer-goldens first.",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "golden",
    _DATASET.goldens,
    ids=lambda golden: golden.name or "golden",
)
def test_scriptwriter_matches_podcast_quality_bar(
    golden: Golden,
    eval_settings: Settings,
    script_metrics: list[BaseMetric],
) -> None:
    """Evaluate Writer SpeechTurn podcast craft and naturalness via JevEval."""
    metadata = cast("dict[str, object]", golden.additional_metadata or {})
    raw_stories = cast("list[dict[str, object]]", metadata["stories"])
    golden_stories = [GoldenStory.model_validate(item) for item in raw_stories]
    stories = [
        CuratedStory(
            title=story.title,
            url=story.url,
            content=story.content,
        )
        for story in golden_stories
    ]
    prompt = format_writer_prompt(date.fromisoformat(golden.input), stories)
    result = asyncio.run(
        writer.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            context=WriterContext(
                personas=eval_settings.personas,
                language=eval_settings.podcast_language,
            ),
            config=llm_config(
                model=eval_settings.llm_model,
                api_key=eval_settings.openrouter_api_key,
            ),
        ),
    )
    dialogue: Dialogue = result["structured_response"]

    test_case = LLMTestCase(
        name=golden.name,
        input=golden.input,
        actual_output=serialize_turns(dialogue.turns),
    )
    assert_test(test_case, script_metrics)
