"""Multimodal DeepEval evaluations for Episode cover artwork."""

import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.cover import cover_test_case, load_cover_dataset
from podcast.cover import OpenRouterCover
from podcast.models import Script
from podcast.settings import Settings

_DATASET = load_cover_dataset()
if not _DATASET.goldens:
    pytest.skip(
        "No Script fixtures in evals/golden/cover/.",
        allow_module_level=True,
    )


async def _generate_cover_case(
    golden: Golden,
    settings: Settings,
    workdir: Path,
) -> LLMTestCase:
    """Generate one Episode cover and build its multimodal evaluation case."""
    metadata = cast("dict[str, object]", golden.additional_metadata or {})
    script = Script.model_validate(metadata["script"])
    async with httpx.AsyncClient() as client:
        generator = OpenRouterCover(
            api_key=settings.openrouter_api_key,
            master_cover_path=settings.podcast_master_cover_path,
            model=settings.cover_model,
            client=client,
        )
        cover_jpeg = await generator.generate(script)
    if cover_jpeg is None:
        msg = "Cover generation returned no artwork."
        raise RuntimeError(msg)
    return await cover_test_case(
        script,
        cover_jpeg,
        master_cover_path=settings.podcast_master_cover_path,
        workdir=workdir,
    )


@pytest.mark.parametrize(
    "golden",
    _DATASET.goldens,
    ids=lambda golden: golden.name or "golden",
)
def test_cover_art_meets_editorial_bar(
    golden: Golden,
    eval_settings: Settings,
    cover_metrics: list[BaseMetric],
    tmp_path: Path,
) -> None:
    """Evaluate generated cover art against orthography, thumbnail, and brand."""
    test_case = asyncio.run(_generate_cover_case(golden, eval_settings, tmp_path))
    assert_test(test_case, cover_metrics)
