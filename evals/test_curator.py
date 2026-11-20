"""Deterministic DeepEval ranking evaluations for the Jev Curator."""

import asyncio
import json
from datetime import date

import httpx
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.conftest import load_curator_dataset
from podcast.agents import Curator
from podcast.fetcher import ExaFetcher
from podcast.settings import Settings

_CURATOR_DATASET = load_curator_dataset()
if not _CURATOR_DATASET.goldens:
    pytest.skip(
        "No labeled fixtures in evals/golden/curator/. "
        "Populate 'ranking' in evals/golden/curator/*.json first.",
        allow_module_level=True,
    )


async def _run_curate(day: date, settings: Settings) -> list[str]:
    """Run the production Jev Curator + ExaFetcher for `day` and return ranked IDs."""
    async with httpx.AsyncClient() as client:
        curator = Curator(
            api_key=settings.openrouter_api_key,
            model=settings.curator_model,
            fetcher=ExaFetcher(api_key=settings.exa_api_key, client=client),
            client=client,
        )
        segments, _ = await curator.curate(day)
    return [segment.story_id for segment in segments]


@pytest.mark.parametrize(
    "golden",
    _CURATOR_DATASET.goldens,
    ids=lambda golden: golden.name or "golden",
)
def test_curator_ranking_matches_preferences(
    golden: Golden,
    eval_settings: Settings,
    curator_metrics: list[BaseMetric],
) -> None:
    """Evaluate Jev Curator + ExaFetcher selection against ranked goldens."""
    predicted_ids = asyncio.run(
        _run_curate(date.fromisoformat(golden.input), eval_settings),
    )
    test_case = LLMTestCase(
        name=golden.name,
        input=golden.input,
        actual_output=json.dumps(predicted_ids),
        expected_output=golden.expected_output,
    )
    assert_test(test_case, curator_metrics)
