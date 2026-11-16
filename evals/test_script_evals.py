"""DeepEval JevEval evaluations comparing Writer Fish Audio JSON output."""

import asyncio
import json
from datetime import date
from typing import TYPE_CHECKING, cast

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.conftest import load_golden_dataset
from evals.download_golden import GoldenStory
from podcast.agents import WriterContext, format_writer_prompt, writer
from podcast.models import CuratedStory, Dialogue

if TYPE_CHECKING:
    from deepeval.dataset import Golden
    from deepeval.metrics import BaseMetric

    from podcast.settings import Settings

_DATASET = load_golden_dataset()
if not _DATASET.goldens:
    pytest.skip(
        "No fixtures in evals/golden/. Run download_golden first.",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "golden",
    _DATASET.goldens,
    ids=lambda golden: golden.name or "golden",
)
def test_scriptwriter_matches_indicator_quality_bar(
    golden: Golden,
    eval_settings: Settings,
    script_metrics: list[BaseMetric],
) -> None:
    """Evaluate Fish Audio SpeechTurn grounding and style similarity via JevEval."""
    metadata = cast("dict[str, object]", golden.additional_metadata or {})
    raw_stories = cast("list[dict[str, object]]", metadata["stories"])
    golden_stories = [GoldenStory.model_validate(item) for item in raw_stories]
    stories = [
        CuratedStory(
            title=story.title,
            url=story.url,
            content=story.content,
            is_lead=(index == 0),
        )
        for index, story in enumerate(golden_stories)
    ]
    prompt = format_writer_prompt(date.fromisoformat(golden.input), stories)
    result = asyncio.run(
        writer.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            context=WriterContext(
                personas=eval_settings.personas,
                language=eval_settings.podcast_language,
            ),
            config=eval_settings.llm_config,
        ),
    )
    dialogue: Dialogue = result["structured_response"]

    test_case = LLMTestCase(
        name=golden.name,
        input=golden.input,
        actual_output=json.dumps(
            [turn.model_dump() for turn in dialogue.turns],
            ensure_ascii=False,
            indent=2,
        ),
        expected_output=golden.expected_output,
        retrieval_context=golden.retrieval_context,
    )
    assert_test(test_case, script_metrics)
