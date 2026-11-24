"""Shared DeepEval fixtures, OpenRouter Jev adapter, and Golden dataset loader."""

import json
import math
import os
import re
from pathlib import Path
from typing import Annotated, Literal

import httpx
import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric
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
    SystemOneQuestion,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel, Field, SecretStr, ValidationError

from evals.golden import CuratorGoldenCase, GoldenCase
from podcast.models import MAX_CONTENT_CHARS, SpeechTurn
from podcast.settings import Settings

_WRITER_GOLDEN_DIR = Path(__file__).parent / "golden" / "writer"
_CURATOR_GOLDEN_DIR = Path(__file__).parent / "golden" / "curator"
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_SYSTEM_ONE_URL = "https://openrouter.ai/api/v1/systemone"
_DEFAULT_JEV_MODEL = "jev-1.14"
_EVAL_KEY_FIELDS = frozenset({"openrouter_api_key", "exa_api_key"})


class _NoulPayload(BaseModel):
    """OpenRouter System One noul probability answer payload."""

    type: Literal["noul"]
    noul: float


class _ChoicePayload(BaseModel):
    """OpenRouter System One categorical choice answer payload."""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class _ScorePayload(BaseModel):
    """OpenRouter System One ordinal score answer payload."""

    type: Literal["score"]
    score: float
    probabilities: dict[int, float]
    confidence: float


type _AnswerPayload = Annotated[
    _NoulPayload | _ChoicePayload | _ScorePayload,
    Field(discriminator="type"),
]


class _UsagePayload(BaseModel):
    """Optional usage/cost block returned by OpenRouter System One."""

    cost: float | None = None


class _SystemOneResponse(BaseModel):
    """Validated JSON response returned by OpenRouter's System One endpoint."""

    answers: dict[str, _AnswerPayload] = Field(default_factory=dict)
    usage: _UsagePayload = Field(default_factory=_UsagePayload)


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
        parsed = _SystemOneResponse.model_validate(payload)
        answers = SystemOneAnswers()
        for key, item in parsed.answers.items():
            if isinstance(item, _NoulPayload):
                answers.nouls[key] = NoulAnswer(probability=item.noul)
            elif isinstance(item, _ChoicePayload):
                answers.choices[key] = ChoiceAnswer(
                    choice=item.choice,
                    probabilities=item.probabilities,
                    confidence=item.confidence,
                )
            else:
                answers.scores[key] = ScoreAnswer(
                    score=item.score,
                    probabilities=item.probabilities,
                    confidence=item.confidence,
                )
        return answers, parsed.usage.cost

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


class NDCGAtKMetric(BaseMetric):
    """Deterministic exponential NDCG@k metric comparing ranked story_id lists."""

    def __init__(self, *, k: int = 5, threshold: float = 0.5) -> None:
        """Configure ranking cutoff `k` and passing threshold."""
        self.k = k
        self._threshold = threshold
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    @staticmethod
    def _dcg(relevances: list[int]) -> float:
        """Compute exponential discounted cumulative gain for a relevance sequence."""
        return sum(
            (2**rel - 1) / math.log2(rank + 2)
            for rank, rel in enumerate(relevances)
            if rel > 0
        )

    def measure(self, test_case: LLMTestCase) -> float:
        """Score predicted top-k story_ids against the full expected ranking."""
        expected: list[str] = json.loads(test_case.expected_output or "[]")
        predicted_raw: list[str] = json.loads(test_case.actual_output or "[]")
        predicted = list(dict.fromkeys(predicted_raw))[: self.k]

        if not expected:
            self.score = 0.0
            self.success = False
            self.reason = "Expected ranking is empty."
            return self.score

        grade_for = {
            story_id: len(expected) - idx for idx, story_id in enumerate(expected)
        }
        dcg = self._dcg([grade_for.get(story_id, 0) for story_id in predicted])
        ideal_grades = list(range(len(expected), 0, -1))[: self.k]
        idcg = self._dcg(ideal_grades)
        score = dcg / idcg if idcg > 0 else 0.0
        self.score = score
        self.success = score >= self._threshold
        self.reason = (
            f"NDCG@{self.k}={score:.4f} (threshold={self._threshold}); "
            f"predicted={predicted}, expected={expected}"
        )
        return score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async interface delegating to deterministic synchronous calculation."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Return whether the last measured NDCG@k met the threshold."""
        return bool(self.success)

    @property
    def __name__(self) -> str:  # type: ignore[override]
        """Display name used in DeepEval reports."""
        return f"NDCG@{self.k}"


def serialize_turns(turns: list[SpeechTurn]) -> str:
    """Serialize SpeechTurn items into the canonical JSON payload used by evals."""
    return json.dumps(
        [turn.model_dump(exclude_none=True) for turn in turns],
        ensure_ascii=False,
        indent=2,
    )


def load_golden_dataset() -> EvaluationDataset:
    """Load evals/golden/writer/*.json into a DeepEval EvaluationDataset."""
    goldens: list[Golden] = []
    for path in sorted(_WRITER_GOLDEN_DIR.glob("*.json")):
        case = GoldenCase.model_validate_json(path.read_text(encoding="utf-8"))
        goldens.append(
            Golden(
                name=path.stem,
                input=case.date.isoformat(),
                expected_output=serialize_turns(case.expected_output),
                retrieval_context=[
                    story.content[:MAX_CONTENT_CHARS] for story in case.stories
                ],
                additional_metadata={
                    "stories": [story.model_dump() for story in case.stories],
                },
            ),
        )
    return EvaluationDataset(goldens=goldens)


def load_curator_dataset() -> EvaluationDataset:
    """Load labeled evals/golden/curator/*.json files into an EvaluationDataset."""
    goldens: list[Golden] = []
    for path in sorted(_CURATOR_GOLDEN_DIR.glob("*.json")):
        raw = _LINE_COMMENT_RE.sub("", path.read_text(encoding="utf-8"))
        case = CuratorGoldenCase.model_validate_json(raw)
        if not case.ranking:
            continue
        goldens.append(
            Golden(
                name=path.stem,
                input=case.date.isoformat(),
                expected_output=json.dumps(case.ranking),
            ),
        )
    return EvaluationDataset(goldens=goldens)


@pytest.fixture(scope="session")
def curator_metrics() -> list[BaseMetric]:
    """Provide deterministic NDCG@5 metric for Curator ranking evaluations."""
    return [NDCGAtKMetric(k=5, threshold=0.5)]


@pytest.fixture(scope="session")
def eval_settings() -> Settings:
    """Build Settings for evals, skipping only when live API keys are unset."""
    try:
        return Settings(
            podcast_owner_email="owner@example.com",
            persona_a_voice_id="persona_a",
            persona_b_voice_id="persona_b",
        )
    except ValidationError as error:
        missing = {
            detail["loc"][0] for detail in error.errors() if detail["type"] == "missing"
        }
        if not missing or not missing <= _EVAL_KEY_FIELDS:
            raise
        reason = ", ".join(sorted(map(str, missing)))
        pytest.skip(f"live API keys not configured for evals: {reason}")


@pytest.fixture(scope="session")
def jev_judge(eval_settings: Settings) -> OpenRouterJevModel:
    """Provide a session-wide OpenRouter-backed Jev System One judge."""
    return OpenRouterJevModel(
        api_key=eval_settings.openrouter_api_key,
        model_name=os.environ.get("JEV_MODEL", _DEFAULT_JEV_MODEL),
    )


def build_script_metrics(jev_judge: OpenRouterJevModel) -> list[BaseMetric]:
    """Build reference-free JevEval metrics for podcast craft and localization."""
    podcast_craft = JevEval(
        name="Podcast Craft & Conversational Dynamics",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        questions=[
            Noul(
                "actual_output reads like a genuinely interactive two-Persona "
                "podcast conversation—with varied turn lengths, crisp short "
                "reactions, playful pushback, genuine follow-up questions, and "
                "callbacks to what the other Persona just said—rather than two "
                "Personas merely taking turns reciting sequential summary "
                "paragraphs, essay prose, or AI-generated filler.",
            ),
            Noul(
                "actual_output opens like a produced podcast Episode—greeting "
                "or hooking the listener, introducing the Personas by name, "
                "and framing what the Episode is about before diving into the "
                "reporting—then connects stories via organic thematic bridges "
                "(rather than mechanical signposting like 'moving on to topic "
                "two' or 'in conclusion') and lands on a natural close.",
            ),
            Noul(
                "The turn-level vocal delivery `instructions` in actual_output "
                "give specific, varied, moment-level audio direction (such as "
                "laughs, pauses, emphasis on key words, or emotional shifts, "
                "with neutral lines omitted), rather than missing, generic, or "
                "boilerplate tone labels repeated across turns (like "
                "'informative and professional tone' or 'enthusiastic tone').",
            ),
            Score(
                "Regardless of language or topic, how authentically does "
                "actual_output read as a broadcast-ready, conversational "
                "podcast Script?",
                levels=[
                    "Robotic, stiff, or unmistakably AI-generated summary",
                    "Informative alternating summaries, noticeably scripted or flat",
                    "Natural, engaging two-Persona conversation with solid flow",
                    "Broadcast-ready podcast craft with vivid chemistry and delivery",
                ],
            ),
        ],
        system_one_model=jev_judge,
        threshold=0.75,
    )
    editorial_polish = JevEval(
        name="Editorial Polish & Localization",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        questions=[
            Noul(
                "actual_output reads as an original, standalone editorial "
                "podcast with zero meta-references to online aggregator "
                "threads, comment sections, or forums (such as 'na thread', "
                "'in the thread', 'nos comentários', or 'on Hacker News').",
            ),
            Noul(
                "Whenever actual_output mentions countries, elections, laws, "
                "regulators, or institutions outside the locale of its spoken "
                "language, it identifies their country or context clearly for "
                "local listeners instead of assuming foreign context "
                "implicitly (for example, saying 'nos Estados Unidos' or "
                "'eleições americanas' in Portuguese rather than bare "
                "'as eleições').",
            ),
            Noul(
                "actual_output uses natural, idiomatic spoken phrasing native "
                "to its language rather than stiff word-for-word translations "
                "or awkward foreign-language calques.",
            ),
        ],
        system_one_model=jev_judge,
        threshold=0.75,
    )
    return [podcast_craft, editorial_polish]


@pytest.fixture(scope="session")
def script_metrics(jev_judge: OpenRouterJevModel) -> list[BaseMetric]:
    """Provide reusable DeepEval JevEval podcast-craft metrics."""
    return build_script_metrics(jev_judge)
