"""Offline tests for the multimodal Episode cover evaluation suite."""

import asyncio
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from evals.cover import (
    build_cover_metrics,
    cover_test_case,
    load_cover_dataset,
)
from podcast.cover import render_prompt, to_square_jpeg
from podcast.models import Script, Segment, SpeechTurn
from tests.conftest import ONE_PIXEL_PNG, jpeg_size

_METRIC_NAMES = [
    "Cover Orthography (pt-BR)",
    "Cover Thumbnail Legibility (50x50)",
    "Cover Brand Consistency",
]


class _ScriptedJudge(DeepEvalBaseLLM):
    """Vision judge stub that returns a fixed rubric score and records prompts."""

    def __init__(self, score: float) -> None:
        """Store the scripted rubric score."""
        self._score = score
        self.prompts: list[str] = []
        super().__init__(model="scripted-vision")

    def load_model(self, *_: object, **__: object) -> DeepEvalBaseLLM:
        """Return this stub, which is already loaded."""
        return self

    def get_model_name(self, *_: object, **__: object) -> str:
        """Return the scripted model identifier."""
        return "scripted-vision"

    def supports_multimodal(self) -> bool:
        """Declare multimodal input support so GEval accepts image test cases."""
        return True

    def generate(self, *args: object, **kwargs: object) -> str:
        """Record the judge prompt and answer with the scripted rubric score."""
        prompt = str(args[0])
        self.prompts.append(prompt)
        schema = kwargs.get("schema")
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return "ok"
        answer = schema.model_validate(
            {"reason": "scripted reason", "score": self._score}
        )
        return answer.model_dump_json()

    async def a_generate(self, *args: object, **kwargs: object) -> str:
        """Async interface delegating to the synchronous scripted answer."""
        return self.generate(*args, **kwargs)


@pytest.fixture
def script() -> Script:
    """Build a minimal pt-BR Script whose lead Segment drives the cover."""
    return Script(
        published_at=date(2026, 11, 24),
        title="Edição de hoje",
        summary="Resumo técnico do dia.",
        language="pt-BR",
        segments=[
            Segment(
                story_id="123",
                title="Kernel Linux muda escalonador",
                url="https://example.com/story",
                key_facts=["Escalonador EEVDF vira padrão no kernel 6.20."],
            ),
        ],
        turns=[
            SpeechTurn(voice="Charon", text="Hoje vamos falar do kernel Linux."),
            SpeechTurn(voice="Aoede", text="A mudança afeta servidores em produção."),
        ],
    )


async def _cover_case(script: Script, workdir: Path) -> LLMTestCase:
    """Render a real JPEG cover and build its multimodal evaluation case."""
    cover_jpeg = await to_square_jpeg(ONE_PIXEL_PNG, size=1400)
    master = workdir / "master.jpg"
    master.write_bytes(cover_jpeg)
    return await cover_test_case(
        script,
        cover_jpeg,
        master_cover_path=master,
        workdir=workdir,
    )


def test_cover_metrics_score_the_three_editorial_criteria() -> None:
    """The suite exposes orthography, thumbnail, and brand GEval metrics."""
    metrics = [
        cast("GEval", metric) for metric in build_cover_metrics(_ScriptedJudge(8.0))
    ]

    assert len(metrics) == 3
    assert [metric.name for metric in metrics] == _METRIC_NAMES
    assert isinstance(metrics[0], GEval)
    assert metrics[0].evaluation_params == [SingleTurnParams.ACTUAL_OUTPUT]
    assert metrics[2].evaluation_params == [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]


def test_cover_metric_rubrics_name_checkable_completion_criteria() -> None:
    """Every rubric states positive, checkable completion criteria."""
    metrics = [
        cast("GEval", metric) for metric in build_cover_metrics(_ScriptedJudge(8.0))
    ]

    assert "Brazilian Portuguese" in (metrics[0].criteria or "")
    assert "50x50 pixels" in (metrics[1].criteria or "")
    assert "same publication as the master cover" in (metrics[2].criteria or "")
    for metric in metrics:
        assert metric.evaluation_steps
        assert metric.evaluation_steps[-1].startswith("Score 1")


@pytest.mark.parametrize(
    ("metric_index", "metric_name"),
    list(enumerate(_METRIC_NAMES)),
    ids=_METRIC_NAMES,
)
def test_cover_metrics_normalize_judge_score_against_threshold(
    metric_index: int,
    metric_name: str,
    script: Script,
    tmp_path: Path,
) -> None:
    """A rubric score of 8/10 passes the gate and 7/10 fails it, for every rubric."""
    case = asyncio.run(_cover_case(script, tmp_path))
    passing = cast("GEval", build_cover_metrics(_ScriptedJudge(8.0))[metric_index])
    failing = cast("GEval", build_cover_metrics(_ScriptedJudge(7.0))[metric_index])

    passing.measure(case, _show_indicator=False)
    failing.measure(case, _show_indicator=False)

    assert passing.name == metric_name
    assert passing.score == pytest.approx(0.8)
    assert passing.is_successful()
    assert failing.name == metric_name
    assert failing.score == pytest.approx(0.7)
    assert not failing.is_successful()


def test_cover_dataset_loads_golden_scripts_and_their_prompts() -> None:
    """Golden fixtures round-trip into Scripts that keep Story identities distinct."""
    dataset = load_cover_dataset()

    assert len(dataset.goldens) == 3
    titles_by_story_id: dict[str, str] = {}
    for golden in dataset.goldens:
        assert isinstance(golden, Golden)
        metadata = golden.additional_metadata or {}
        script = Script.model_validate(metadata["script"])
        assert script.language == "pt-BR"
        for segment in script.segments:
            titles_by_story_id[segment.story_id] = segment.title

    story_ids_by_title = {title: sid for sid, title in titles_by_story_id.items()}
    assert len(story_ids_by_title) == len(titles_by_story_id)


async def test_cover_case_embeds_full_cover_thumbnail_and_master(
    script: Script,
    tmp_path: Path,
) -> None:
    """The multimodal case carries a 50x50 thumbnail plus the master reference."""
    case = await _cover_case(script, tmp_path)
    actual_output = case.actual_output or ""
    expected_output = case.expected_output or ""

    assert case.multimodal is True
    assert case.input == render_prompt(script)
    assert actual_output.count("[DEEPEVAL:IMAGE:") == 2
    assert expected_output.count("[DEEPEVAL:IMAGE:") == 1
    assert "Full-size Episode cover" in actual_output
    assert "50x50 pixels" in actual_output
    assert case.metadata == {
        "headline": "KERNEL LINUX MUDA ESCALONADOR",
        "lead_story": "Kernel Linux muda escalonador",
    }
    thumbnail = tmp_path / "2026-11-24-50x50.jpg"
    assert jpeg_size(thumbnail.read_bytes()) == (50, 50)
