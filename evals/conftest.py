"""Shared DeepEval fixtures, OpenRouter Jev adapter, and Golden dataset loader."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics.jev_eval import JevEval, Noul, Score
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
)
from deepeval.test_case import SingleTurnParams
from pydantic import ValidationError

from evals.download_golden import GoldenCase
from podcast.models import MAX_CONTENT_CHARS
from podcast.settings import Settings

if TYPE_CHECKING:
    from deepeval.metrics import BaseMetric
    from deepeval.models.system_one.schema import SystemOneQuestion
    from pydantic import SecretStr

_GOLDEN_DIR = Path(__file__).parent / "golden"
_SYSTEM_ONE_URL = "https://openrouter.ai/api/v1/systemone"
_DEFAULT_JEV_MODEL = "jev-1.13"


class OpenRouterJevModel(DeepEvalBaseSystemOneModel):
    """DeepEval System One judge backed by OpenRouter's Jev endpoint."""

    def __init__(
        self,
        api_key: SecretStr,
        model_name: str = _DEFAULT_JEV_MODEL,
    ) -> None:
        """Initialize the OpenRouter Jev adapter."""
        self._api_key = api_key
        self._model_name = model_name
        super().__init__(model=model_name)

    def load_model(self) -> str:
        """Return the configured System One model identifier."""
        return self._model_name

    def get_model_name(self) -> str:
        """Return the configured System One model identifier."""
        return self._model_name

    @staticmethod
    def _serialize_questions(
        questions: dict[str, SystemOneQuestion],
    ) -> dict[str, dict[str, object]]:
        """Translate DeepEval System One questions into OpenRouter API payloads."""
        serialized: dict[str, dict[str, object]] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                entry: dict[str, object] = {
                    "type": "noul",
                    "instructions": question.instructions,
                }
                if question.true is not None or question.false is not None:
                    entry["criteria"] = {
                        "true": question.true,
                        "false": question.false,
                    }
                serialized[key] = entry
            elif isinstance(question, ChoiceQuestion):
                serialized[key] = {
                    "type": "choice",
                    "instructions": question.instructions,
                    "criteria": dict(question.options),
                }
            elif isinstance(question, ScoreQuestion):
                serialized[key] = {
                    "type": "score",
                    "instructions": question.instructions,
                    "criteria": list(question.levels),
                }
            else:
                msg = f"Unsupported System One question type: {type(question)}"
                raise TypeError(msg)
        return serialized

    @staticmethod
    def _parse_response(
        payload: dict[str, object],
    ) -> tuple[SystemOneAnswers, float | None]:
        """Parse OpenRouter System One JSON answers and optional USD cost."""
        answers = SystemOneAnswers()
        raw_answers = cast("dict[str, dict[str, object]]", payload.get("answers", {}))

        for key, item in raw_answers.items():
            qtype = item.get("type")
            if qtype == "noul":
                answers.nouls[key] = NoulAnswer(
                    probability=float(cast("float", item["noul"])),
                )
            elif qtype == "choice":
                probs = cast("dict[str, float]", item["probabilities"])
                answers.choices[key] = ChoiceAnswer(
                    choice=str(item["choice"]),
                    probabilities={str(k): float(v) for k, v in probs.items()},
                    confidence=float(cast("float", item["confidence"])),
                )
            elif qtype == "score":
                probs_int = cast("dict[str, float]", item["probabilities"])
                answers.scores[key] = ScoreAnswer(
                    score=float(cast("float", item["score"])),
                    probabilities={int(k): float(v) for k, v in probs_int.items()},
                    confidence=float(cast("float", item["confidence"])),
                )
            else:
                msg = f"Unsupported System One answer type: {qtype!r}"
                raise TypeError(msg)

        usage = cast("dict[str, object]", payload.get("usage", {}))
        raw_cost = usage.get("cost")
        cost = float(raw_cost) if isinstance(raw_cost, int | float) else None
        return answers, cost

    def _request_payload(
        self,
        state: object,
        questions: dict[str, SystemOneQuestion],
    ) -> tuple[dict[str, str], dict[str, object]]:
        """Build headers and JSON body for OpenRouter System One calls."""
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        body: dict[str, object] = {
            "model": self._model_name,
            "state": state,
            "questions": self._serialize_questions(questions),
        }
        return headers, body

    def decide(
        self,
        state: object,
        questions: dict[str, SystemOneQuestion],
    ) -> tuple[SystemOneAnswers, float | None]:
        """Run a synchronous Jev probability judgement via OpenRouter."""
        headers, body = self._request_payload(state, questions)
        response = httpx.post(_SYSTEM_ONE_URL, headers=headers, json=body, timeout=60.0)
        response.raise_for_status()
        return self._parse_response(response.json())

    async def a_decide(
        self,
        state: object,
        questions: dict[str, SystemOneQuestion],
    ) -> tuple[SystemOneAnswers, float | None]:
        """Run an asynchronous Jev probability judgement via OpenRouter."""
        headers, body = self._request_payload(state, questions)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(_SYSTEM_ONE_URL, headers=headers, json=body)
            response.raise_for_status()
        return self._parse_response(response.json())


def load_golden_dataset() -> EvaluationDataset:
    """Load evals/golden/*.json into a DeepEval EvaluationDataset of Goldens."""
    goldens: list[Golden] = []
    for path in sorted(_GOLDEN_DIR.glob("*.json")):
        case = GoldenCase.model_validate_json(path.read_text(encoding="utf-8"))
        goldens.append(
            Golden(
                name=path.stem,
                input=case.date.isoformat(),
                expected_output=json.dumps(
                    [turn.model_dump() for turn in case.expected_output],
                    ensure_ascii=False,
                    indent=2,
                ),
                retrieval_context=[
                    story.content[:MAX_CONTENT_CHARS] for story in case.stories
                ],
                additional_metadata={
                    "stories": [story.model_dump() for story in case.stories],
                },
            ),
        )
    return EvaluationDataset(goldens=goldens)


@pytest.fixture(scope="session")
def eval_settings() -> Settings:
    """Build Settings with canonical golden voice IDs for scriptwriter evals."""
    try:
        return Settings(
            persona_a_voice_id="persona_a",
            persona_b_voice_id="persona_b",
        )
    except ValidationError:
        pytest.skip("OPENROUTER_API_KEY not configured for evals")


@pytest.fixture(scope="session")
def jev_judge(eval_settings: Settings) -> OpenRouterJevModel:
    """Provide a session-wide OpenRouter-backed Jev System One judge."""
    return OpenRouterJevModel(
        api_key=eval_settings.openrouter_api_key,
        model_name=os.environ.get("JEV_MODEL", _DEFAULT_JEV_MODEL),
    )


@pytest.fixture(scope="session")
def script_metrics(jev_judge: OpenRouterJevModel) -> list[BaseMetric]:
    """Provide reusable DeepEval JevEval grounding and similarity metrics."""
    grounding = JevEval(
        name="Source Grounding",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        questions=[
            Noul(
                "The key numbers and quantitative claims in actual_output are "
                "supported by retrieval_context rather than invented.",
            ),
            Noul(
                "All named entities, mechanisms, and factual assertions in "
                "actual_output align with retrieval_context without major "
                "contradictions or hallucinations.",
            ),
            Noul(
                "actual_output synthesizes substantive details from the sources "
                "in retrieval_context rather than generic filler.",
            ),
        ],
        system_one_model=jev_judge,
        threshold=0.5,
    )
    similarity = JevEval(
        name="Indicator Style & Similarity",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        questions=[
            Noul(
                "Accounting for localization into Brazilian Portuguese, "
                "actual_output covers the core analytical ideas and story angle "
                "of expected_output.",
            ),
            Noul(
                "Both Persona speakers use natural spoken turns with genuine "
                "back-and-forth handoffs and accessible explanations similar "
                "to expected_output.",
            ),
            Score(
                "How closely does actual_output match expected_output in "
                "pacing, explanatory clarity, and conversational craft "
                "(ignoring language translation into pt-BR and inline "
                "[bracket] Fish Audio cues)?",
                levels=[
                    "Flat monologue or robotic turns lacking narrative flow",
                    "Accurate summary but stiffer or slower than reference",
                    "Engaging two-Persona conversation comparable to reference",
                    "Tight, vivid, broadcast-ready dialogue matching reference",
                ],
            ),
        ],
        system_one_model=jev_judge,
        threshold=0.5,
    )
    return [grounding, similarity]
